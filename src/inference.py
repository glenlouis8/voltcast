"""
Make the real 24-hour forecast with the champion model and save it for the dashboard.

    python src/inference.py --region CAL

Steps: find the champion -> take the latest 168 hours -> convert to the champion's
own scaling -> predict the next 24 hours in one shot -> turn back into MW -> save.
"""

import argparse
import json

import mlflow
import pandas as pd
import torch
from mlflow.tracking import MlflowClient

from dataset import FEATURE_COLS, FORECAST_HORIZON, SEQ_LEN, rescale_demand
from features import FEATURES_DIR
from mlflow_setup import setup_mlflow
from registry import get_champion, load_checkpoint
from storage import save_forecast

STALE_HOURS = 12  # warn if the newest data is older than this


def load_champion(region: str):
    """Returns (model, info). info holds what the dashboard shows and the model's own scaler."""
    setup_mlflow()
    champion = get_champion(MlflowClient(), f"voltcast-{region}")
    if champion is None:
        raise SystemExit(f"no champion for {region} yet. Train and run registry.py first.")

    tags = champion.tags
    model_type = tags["model_type"]
    folder = mlflow.artifacts.download_artifacts(champion.source)
    model = load_checkpoint(model_type, f"{folder}/{region}_{model_type}.pt").eval()

    info = {
        "version": int(champion.version),
        "model_type": model_type,
        "test_mae_mw": float(tags["test_mae_mw"]),
        "test_wape": float(tags["test_wape"]),
    }
    scaler = {"mean": float(tags["scaler_mean"]), "std": float(tags["scaler_std"])}
    print(f"champion: voltcast-{region} v{info['version']} ({model_type})")
    return model, info, scaler


def forecast(region: str) -> None:
    model, info, champ_scaler = load_champion(region)
    data_scaler = json.loads((FEATURES_DIR / "scalers.json").read_text())[region]

    df = pd.read_parquet(FEATURES_DIR / f"{region}.parquet").sort_values("timestamp")
    window = df.iloc[-SEQ_LEN:]  # the latest 168 hours = the model's input
    if len(window) < SEQ_LEN:
        raise SystemExit(f"need {SEQ_LEN} hours of data, only have {len(window)}")

    # our timestamps are UTC but stored without a label; add the label so the
    # dashboard doesn't read them as the viewer's local time
    last_hour = window["timestamp"].iloc[-1].tz_localize("UTC")
    age_hours = (pd.Timestamp.now(tz="UTC") - last_hour).total_seconds() / 3600
    print(f"newest data: {last_hour:%Y-%m-%d %H:%M} UTC ({age_hours:.0f}h old)")
    if age_hours > STALE_HOURS:
        print(f"WARNING: data is more than {STALE_HOURS}h old, so the forecast starts in the past")

    X = torch.tensor(window[FEATURE_COLS].values, dtype=torch.float32).unsqueeze(0)  # (1, 168, 10)
    X = rescale_demand(X, data_scaler, champ_scaler)  # the scaling the champion was trained with

    with torch.no_grad():
        scaled = model(X)[0]  # 24 numbers, in the champion's scaling
    predicted_mw = (scaled * champ_scaler["std"] + champ_scaler["mean"]).numpy()

    # 25 hours starting at the last known hour, then drop that first one: the 24 hours after it
    hours = pd.date_range(last_hour, periods=FORECAST_HORIZON + 1, freq="h")[1:]
    table = pd.DataFrame({"timestamp": hours, "predicted_load_mw": predicted_mw})

    # same shape the dashboard already reads
    payload = {
        "region": region,
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "champion": info,
        "forecast": [
            {"timestamp": t.isoformat(), "predicted_load_mw": float(mw)}
            for t, mw in zip(hours, predicted_mw)
        ],
    }
    location = save_forecast(region, table, payload=payload)

    print(f"\nforecast {hours[0]:%Y-%m-%d %H:%M} -> {hours[-1]:%Y-%m-%d %H:%M} UTC")
    print(f"range: {predicted_mw.min():,.0f} to {predicted_mw.max():,.0f} MW")
    print(f"saved -> {location}")


def main():
    parser = argparse.ArgumentParser(description="24-hour forecast from the champion model.")
    parser.add_argument("--region", choices=["CAL", "TEX", "PJM", "MISO"], default="CAL")
    forecast(parser.parse_args().region)


if __name__ == "__main__":
    main()
