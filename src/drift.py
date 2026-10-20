"""
Is the champion still making good forecasts? If not, it should be retrained.

    python src/drift.py --region CAL

How: download the last 60 days of real demand, make the champion forecast each
day as if it were back then, and compare with the real answers (known by now).

Some seasons are just harder for every forecaster, so we don't judge the champion's
error alone. We judge it RELATIVE to the simplest forecast, "copy yesterday":

    ratio = champion error / copy-yesterday error

At test time the champion had some ratio (say 0.75). If the ratio now is clearly
worse than that, or the champion no longer beats copy-yesterday, it has gone stale.

Exit code tells the pipeline what happened:
    0  = fine, keep the champion
    10 = drift: retrain
    anything else (e.g. 1) = something broke. That is an error, NOT drift.
"""

import argparse
import os

import mlflow
import torch
from mlflow.tracking import MlflowClient
from torch.utils.data import DataLoader

from dataset import FEATURE_COLS, WindowDataset, copy_yesterday_mae
from features import DEMAND_COLS, add_calendar_features, add_lag_features
from ingestion import fetch_region
from mlflow_setup import setup_mlflow
from registry import get_champion, load_checkpoint
from validation import clean_frame

EXIT_OK, EXIT_DRIFT = 0, 10

RECENT_DAYS = 60       # 7 days are used up as history, the rest is scored
MIN_WINDOWS = 24 * 14  # need at least 2 weeks of forecasts to judge fairly
RATIO_LIMIT = 1.25     # drift if the ratio is 25% worse than it was at test time


def recent_windows(region: str, scaler: dict) -> WindowDataset:
    """Last 60 days of real demand, cleaned and turned into model inputs (champion's scaling)."""
    df = fetch_region(region, os.environ["EIA_API_KEY"], days=RECENT_DAYS)
    df, _ = clean_frame(df, region)
    df = add_lag_features(add_calendar_features(df, region)).dropna()
    for col in DEMAND_COLS:
        df[col] = (df[col] - scaler["mean"]) / scaler["std"]
    return WindowDataset(torch.tensor(df[FEATURE_COLS].values, dtype=torch.float32))


def check_drift(region: str) -> bool:
    setup_mlflow()
    champion = get_champion(MlflowClient(), f"voltcast-{region}")
    if champion is None:
        print(f"{region}: no champion yet -> train one")
        return True

    tags = champion.tags
    if "naive_test_mae_mw" not in tags:
        raise SystemExit(f"{region}: champion v{champion.version} has no copy-yesterday baseline saved. "
                         "Run registry.py again to register a fresh one.")
    model_type = tags["model_type"]
    scaler = {"mean": float(tags["scaler_mean"]), "std": float(tags["scaler_std"])}
    # how the champion compared with copy-yesterday on the test year
    test_ratio = float(tags["test_mae_mw"]) / float(tags["naive_test_mae_mw"])

    folder = mlflow.artifacts.download_artifacts(champion.source)
    model = load_checkpoint(model_type, f"{folder}/{region}_{model_type}.pt").eval()

    ds = recent_windows(region, scaler)
    if len(ds) < MIN_WINDOWS:
        raise SystemExit(f"{region}: only {len(ds)} recent forecasts to check, need {MIN_WINDOWS}")

    model_error, count = 0.0, 0
    with torch.no_grad():
        for X, y in DataLoader(ds, batch_size=256, shuffle=False):
            model_error += (model(X) - y).abs().sum().item()
            count += y.numel()
    recent_mae = model_error / count * scaler["std"]
    yesterday_mae = copy_yesterday_mae(ds) * scaler["std"]

    recent_ratio = recent_mae / yesterday_mae
    worse_than_yesterday = recent_ratio >= 1.0
    drifted = recent_ratio > test_ratio * RATIO_LIMIT or worse_than_yesterday

    print(f"\n{region}: champion v{champion.version} ({model_type}), checked on {len(ds):,} recent forecasts")
    print(f"  champion error:   {recent_mae:>7,.0f} MW")
    print(f"  copy yesterday:   {yesterday_mae:>7,.0f} MW")
    print(f"  ratio now: {recent_ratio:.2f} | at test time: {test_ratio:.2f} | limit: {test_ratio * RATIO_LIMIT:.2f} "
          f"(lower is better, 1.00 = no better than copying yesterday)")
    print(f"  VERDICT: {'DRIFT -> retrain' if drifted else 'fine -> keep the champion'}")
    return drifted


def main():
    parser = argparse.ArgumentParser(description="Check if the champion has gone stale.")
    parser.add_argument("--region", choices=["CAL", "TEX", "PJM", "MISO"], default="CAL")
    raise SystemExit(EXIT_DRIFT if check_drift(parser.parse_args().region) else EXIT_OK)


if __name__ == "__main__":
    main()
