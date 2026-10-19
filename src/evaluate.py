"""
The final exam: score every method on the TEST year (the newest data, which
training never saw) and print one comparison table.

    python src/evaluate.py --region CAL

Methods compared:
    Copy yesterday  - predict that the next 24 hours repeat the last 24 hours
    Copy last week - predict that the next 24 hours repeat the same hours 1 week ago
    LSTM, Transformer - the trained models

Run this only when you are done building the model. If you change the model
because of what the test shows, the test stops being a fair exam.
"""

import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import FORECAST_HORIZON, make_datasets
from features import FEATURES_DIR
from train import CHECKPOINT_DIR, build_model


def load_trained(name: str, region: str, device) -> torch.nn.Module:
    model = build_model(name).to(device)
    state = torch.load(CHECKPOINT_DIR / f"{region}_{name}.pt", map_location=device)
    model.load_state_dict(state)
    return model.eval()  # eval mode: dropout off


def predict_everything(test_ds, models: dict, device) -> tuple[dict, np.ndarray]:
    """Run every method over every test window. Returns (predictions, real values)."""
    loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    preds = {"Copy yesterday": [], "Copy last week": [], **{n: [] for n in models}}
    actual = []

    with torch.no_grad():
        for X, y in loader:
            actual.append(y)
            load = X[:, :, 0]  # column 0 = scaled demand for each of the 168 hours
            # the window ends at "now" (hour 167). Next 24 hours = hours 168..191.
            preds["Copy yesterday"].append(load[:, -FORECAST_HORIZON:])  # hours 144..167
            preds["Copy last week"].append(load[:, :FORECAST_HORIZON])   # hours 0..23, exactly 168 earlier
            for name, model in models.items():
                preds[name].append(model(X.to(device)).cpu())

    return {k: torch.cat(v).numpy() for k, v in preds.items()}, torch.cat(actual).numpy()


def evaluate(region: str) -> None:
    device = torch.device("cpu")  # the trusted reference (and what the CI runner uses)
    scaler =json.loads((FEATURES_DIR / "scalers.json").read_text())[region]
    mean, std = scaler["mean"], scaler["std"]

    _, _, test_ds = make_datasets(region)
    print(f"{region}: scoring on {len(test_ds):,} test windows (device: {device})\n")

    models = {"LSTM": load_trained("lstm", region, device),
              "Transformer": load_trained("transformer", region, device)}
    preds, actual = predict_everything(test_ds, models, device)

    # everything so far is in scaled units; undo the scaling to get real MW
    to_mw = lambda a: a * std + mean
    actual_mw = to_mw(actual)
    avg_demand = actual_mw.mean()

    results = {}
    for name, p in preds.items():
        mae = float(np.abs(to_mw(p) - actual_mw).mean())  # average mistake, in MW
        results[name] = (mae, mae / avg_demand * 100)      # and as a % of average demand

    print(f"{'Method':<16}{'MAE (MW)':>10}{'% of demand':>14}")
    print("-" * 40)
    for name, (mae, wape) in results.items():
        print(f"{name:<16}{mae:>10,.0f}{wape:>13.2f}%")

    best = min(results, key=lambda k: results[k][0])
    best_naive = min(results["Copy yesterday"][0], results["Copy last week"][0])
    gain = (best_naive - results[best][0]) / best_naive * 100
    print(f"\nbest: {best}  |  {gain:.1f}% lower error than the best simple baseline")
    if best in ("Copy yesterday", "Copy last week"):
        print("WARNING: no trained model beat a simple baseline.")


def main():
    parser = argparse.ArgumentParser(description="Score all methods on the test year.")
    parser.add_argument("--region", choices=["CAL", "TEX", "PJM", "MISO"], default="CAL")
    evaluate(parser.parse_args().region)


if __name__ == "__main__":
    main()
