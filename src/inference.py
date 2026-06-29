"""
Generate a real 24-hour-ahead forecast (the actual future, not test scoring)
using the current champion model, and save it for the dashboard.

    python src/inference.py --country CAL

Looks up the champion by alias, feeds it the latest 168 real hours, predicts the
next 24 in one shot (no recursive feeding), denormalizes to MW, and saves.
"""

import argparse
import glob
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

from evaluate import get_scaler, denormalize
from dataset import FEATURE_COLS, SEQ_LEN, FORECAST_HORIZON
from model import LSTMBaseline, TemporalTransformer
from mlflow_setup import setup_mlflow
from storage import save_forecast

FEATURES_DIR = Path(__file__).parent.parent / "data" / "features"


def load_champion(region: str, device) -> tuple[torch.nn.Module, dict]:
    """
    Load whichever model holds the 'champion' alias for this region (by alias,
    never a hardcoded filename). Returns (model, meta) where meta carries the
    version + metrics the dashboard shows.
    """
    client = MlflowClient()
    registered_name = f"voltcast-{region}"

    champion = client.get_model_version_by_alias(registered_name, "champion")
    model_type = champion.tags["model_type"]
    meta = {
        "version":     int(champion.version),
        "model_type":  model_type,
        "test_mae_mw": float(champion.tags.get("test_mae_mw", 0)),
        "test_wape":   float(champion.tags.get("test_wape", 0)),
    }
    print(f"  Champion: {registered_name} v{champion.version} ({model_type})")

    # champion.source points at the stored 'model' artifact folder
    local_dir = mlflow.artifacts.download_artifacts(champion.source)
    ckpt_file = glob.glob(f"{local_dir}/*.pt")[0]

    if model_type == "lstm":
        model = LSTMBaseline(input_dim=len(FEATURE_COLS))
    else:
        model = TemporalTransformer(input_dim=len(FEATURE_COLS))

    model.load_state_dict(torch.load(ckpt_file, map_location=device))
    return model.to(device), meta


def forecast(region: str) -> None:
    setup_mlflow()
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Device: {device}")
    print(f"Forecasting next 24h: {region}\n")

    model, champ_meta = load_champion(region, device)
    model.eval()

    df = pd.read_parquet(FEATURES_DIR / f"{region}.parquet")
    df = df.sort_values("timestamp").reset_index(drop=True)

    # latest 168 rows = the input window (already normalized)
    window = df.iloc[-SEQ_LEN:]
    X = window[FEATURE_COLS].values
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 168, 13)

    with torch.no_grad():
        preds_norm = model(X_tensor).cpu().numpy().squeeze()  # (24,)

    mean, std = get_scaler(region)
    preds_mw = denormalize(preds_norm, mean, std)

    # 24 future hourly timestamps after the last known hour. pd.Timestamp avoids
    # a numpy timedelta-unit deprecation warning.
    # EIA periods are UTC but stored naive. Mark them UTC-aware so the isoformat
    # strings carry +00:00 — otherwise the browser parses them as its own local
    # time and the chart shifts by the viewer's offset.
    last_ts = pd.Timestamp(window["timestamp"].iloc[-1])
    if last_ts.tzinfo is None:
        last_ts = last_ts.tz_localize("UTC")
    future_ts = pd.date_range(
        start=last_ts + pd.Timedelta(hours=1),
        periods=FORECAST_HORIZON,
        freq="h",
    )

    out = pd.DataFrame({
        "timestamp":         future_ts,
        "predicted_load_mw": preds_mw,
    })

    # rich payload so one fetch gives the dashboard everything it needs
    payload = {
        "region":       region,
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "champion":     champ_meta,
        "forecast": [
            {"timestamp": ts.isoformat(), "predicted_load_mw": float(mw)}
            for ts, mw in zip(future_ts, preds_mw)
        ],
    }

    # writes parquet (internal) + JSON (frontend), to S3 if configured else local
    location = save_forecast(region, out, payload=payload)

    print(f"\n  Forecast window: {future_ts[0]} -> {future_ts[-1]}")
    print(f"  Predicted load:  {preds_mw.min():,.0f} - {preds_mw.max():,.0f} MW")
    print(f"  Saved -> {location}")
    print("\n  Next 24 hours:")
    for ts, mw in zip(future_ts, preds_mw):
        print(f"    {ts:%Y-%m-%d %H:%M}  {mw:>8,.0f} MW")


def main():
    parser = argparse.ArgumentParser(description="Generate 24h forecast from champion model.")
    parser.add_argument("--country", default="CAL", help="Region: CAL, TEX, PJM, MISO.")
    args = parser.parse_args()
    forecast(args.country)


if __name__ == "__main__":
    main()
