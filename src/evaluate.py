"""
Ablation table: run Naive (copy last 24h), LSTM, and Transformer on a region's
untouched test set and compare MAE (MW) and MAPE (%).

    python src/evaluate.py --country CAL

The test set is touched only here. The scaler is recomputed from train rows
only, and predictions are converted back to real MW before scoring.
"""

import argparse
from pathlib import Path

import torch
import numpy as np
import pandas as pd

from dataset import load_datasets, FEATURE_COLS, TRAIN_RATIO, SEQ_LEN, FORECAST_HORIZON
from model import LSTMBaseline, TemporalTransformer

RAW_DIR        = Path(__file__).parent.parent / "data" / "raw"
CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"

LOAD_IDX = FEATURE_COLS.index("load_mw")


def get_scaler(region: str) -> tuple[float, float]:
    """
    Recompute (train_mean, train_std) of load_mw from raw, matching features.py
    (sort, first 70% of rows, mean/std over those only). We recompute rather
    than read df.attrs because parquet often drops attrs.
    """
    df = pd.read_parquet(RAW_DIR / f"{region}.parquet")
    df = df.sort_values("timestamp").reset_index(drop=True)

    train_size = int(len(df) * TRAIN_RATIO)
    train_mean = df["load_mw"].iloc[:train_size].mean()
    train_std  = df["load_mw"].iloc[:train_size].std()
    return float(train_mean), float(train_std)


def denormalize(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Undo z-score: real_MW = normalized * std + mean."""
    return x * std + mean


def mae_mw(preds: np.ndarray, actuals: np.ndarray) -> float:
    """Mean Absolute Error in MW."""
    return float(np.mean(np.abs(preds - actuals)))


def mape(preds: np.ndarray, actuals: np.ndarray) -> float:
    """Mean Absolute Percentage Error (%)."""
    return float(np.mean(np.abs(preds - actuals) / actuals) * 100)


def gather_test_arrays(test_ds) -> tuple[np.ndarray, np.ndarray]:
    """
    Stack the whole test set into X_all (N, 168, 13) and y_all (N, 24) once, so
    every method scores on the exact same windows.
    """
    X_list, y_list = [], []
    for i in range(len(test_ds)):
        X, y = test_ds[i]
        X_list.append(X.numpy())
        y_list.append(y.numpy())
    return np.stack(X_list), np.stack(y_list)


def predict_naive(X_all: np.ndarray) -> np.ndarray:
    """Naive forecast: next 24h = last 24h of the input window. Still normalized."""
    return X_all[:, SEQ_LEN - FORECAST_HORIZON:, LOAD_IDX]


def predict_model(model, X_all: np.ndarray, device) -> np.ndarray:
    """Run a trained model over all test windows in chunks. Returns (N, 24)."""
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(X_all), 256):
            chunk = X_all[start:start + 256]
            X = torch.tensor(chunk, dtype=torch.float32).to(device)
            preds.append(model(X).cpu().numpy())
    return np.concatenate(preds, axis=0)


def load_checkpoint(model_name: str, region: str, device) -> torch.nn.Module:
    """Build the right architecture and load its saved weights."""
    if model_name == "lstm":
        model = LSTMBaseline(input_dim=len(FEATURE_COLS))
    else:
        model = TemporalTransformer(input_dim=len(FEATURE_COLS))

    ckpt = CHECKPOINT_DIR / f"{region}_{model_name}.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model.to(device)


def evaluate(region: str) -> None:
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Device: {device}")
    print(f"Evaluating on TEST set: {region}\n")

    _, _, test_ds = load_datasets(region)
    print(f"Test windows: {len(test_ds):,}")

    mean, std = get_scaler(region)
    print(f"Scaler: mean={mean:,.0f} MW, std={std:,.0f} MW\n")

    X_all, y_all = gather_test_arrays(test_ds)
    actuals_mw = denormalize(y_all, mean, std)

    results = {}

    naive_mw = denormalize(predict_naive(X_all), mean, std)
    results["Naive"] = (mae_mw(naive_mw, actuals_mw), mape(naive_mw, actuals_mw))

    lstm = load_checkpoint("lstm", region, device)
    lstm_mw = denormalize(predict_model(lstm, X_all, device), mean, std)
    results["LSTM"] = (mae_mw(lstm_mw, actuals_mw), mape(lstm_mw, actuals_mw))

    tf = load_checkpoint("transformer", region, device)
    tf_mw = denormalize(predict_model(tf, X_all, device), mean, std)
    results["Transformer"] = (mae_mw(tf_mw, actuals_mw), mape(tf_mw, actuals_mw))

    print(f"{'='*46}")
    print(f"  ABLATION - {region} (test set)")
    print(f"{'='*46}")
    print(f"  {'Method':<14}{'MAE (MW)':>12}{'MAPE':>10}")
    print(f"  {'-'*40}")
    for name, (m, p) in results.items():
        print(f"  {name:<14}{m:>12,.0f}{p:>9.2f}%")
    print(f"{'='*46}")

    best = min(results, key=lambda k: results[k][0])
    naive_mae = results["Naive"][0]
    lift = (naive_mae - results[best][0]) / naive_mae * 100
    print(f"\n  Best: {best}  |  {lift:.1f}% better than Naive baseline")
    if best == "Naive":
        print("  WARNING: models failed to beat copy-yesterday. Something is wrong.")


def main():
    parser = argparse.ArgumentParser(description="Ablation: Naive vs LSTM vs Transformer.")
    parser.add_argument("--country", default="CAL", help="Region: CAL, TEX, PJM, MISO.")
    args = parser.parse_args()
    evaluate(args.country)


if __name__ == "__main__":
    main()
