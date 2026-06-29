"""
src/inference.py

Generate a REAL 24-hour-ahead forecast — the actual future, not test scoring.

Run:
    python src/inference.py --country CAL

What it does:
    1. Ask the registry for the CHAMPION model of this region (by alias,
       not a hardcoded filename). Find out which architecture it is.
    2. Download the champion's weights, rebuild the model, load them.
    3. Take the LATEST 168 real hours from the feature matrix.
    4. Predict the next 24 hours (one shot — no recursive feeding).
    5. Denormalize predictions back to real MW.
    6. Build the 24 future timestamps.
    7. Save to data/forecasts/<region>_forecast.parquet.

This file is what the API and dashboard will serve.
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

FEATURES_DIR  = Path(__file__).parent.parent / "data" / "features"


# ── load the champion model from the registry ─────────────────────────────────

def load_champion(region: str, device) -> tuple[torch.nn.Module, dict]:
    """
    Fetch whichever model currently holds the 'champion' alias for this region.

    Steps:
        - Look up the champion version by alias (not by filename).
        - Read its 'model_type' tag to know which architecture to rebuild.
        - Download the saved .pt artifact and load the weights.

    Returns (model, meta) where meta = {version, model_type, test_mae_mw} so the
    frontend payload can show which model served and how good it is.

    This is the whole point of the registry: inference never hardcodes
    "transformer". It serves whoever is champion right now.
    """
    client = MlflowClient()
    registered_name = f"voltcast-{region}"

    # Get the version tagged 'champion'.
    champion = client.get_model_version_by_alias(registered_name, "champion")
    model_type = champion.tags["model_type"]          # "transformer" or "lstm"
    meta = {
        "version":     int(champion.version),
        "model_type":  model_type,
        "test_mae_mw": float(champion.tags.get("test_mae_mw", 0)),
        "test_wape":   float(champion.tags.get("test_wape", 0)),
    }
    print(f"  Champion: {registered_name} v{champion.version} ({model_type})")

    # Download the artifact folder for this version to a local cache path.
    # champion.source points at the stored 'model' artifact folder.
    local_dir = mlflow.artifacts.download_artifacts(champion.source)
    # The .pt file lives inside that folder.
    ckpt_file = glob.glob(f"{local_dir}/*.pt")[0]

    # Rebuild the matching architecture, then load the weights into it.
    if model_type == "lstm":
        model = LSTMBaseline(input_dim=len(FEATURE_COLS))
    else:
        model = TemporalTransformer(input_dim=len(FEATURE_COLS))

    model.load_state_dict(torch.load(ckpt_file, map_location=device))
    return model.to(device), meta


# ── main forecast routine ─────────────────────────────────────────────────────

def forecast(region: str) -> None:
    setup_mlflow()
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Device: {device}")
    print(f"Forecasting next 24h: {region}\n")

    model, champ_meta = load_champion(region, device)
    model.eval()

    # ── load the feature matrix, grab the LATEST 168 hours ──
    df = pd.read_parquet(FEATURES_DIR / f"{region}.parquet")
    df = df.sort_values("timestamp").reset_index(drop=True)

    # The most recent 168 rows = the input window the model needs.
    window = df.iloc[-SEQ_LEN:]                       # last 168 rows
    X = window[FEATURE_COLS].values                  # (168, 13), already normalized

    # Model expects a batch dimension: (1, 168, 13).
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(device)

    # ── predict ──
    with torch.no_grad():
        preds_norm = model(X_tensor)                 # (1, 24)
    preds_norm = preds_norm.cpu().numpy().squeeze()  # (24,)

    # ── denormalize back to real MW ──
    mean, std = get_scaler(region)
    preds_mw = denormalize(preds_norm, mean, std)    # (24,) real megawatts

    # ── build the 24 future timestamps ──
    # Last known real hour + 1h, +2h, ... +24h.
    # Wrap in pd.Timestamp so arithmetic is unambiguous (avoids a numpy
    # timedelta-unit deprecation warning when the column dtype is datetime64).
    last_ts = pd.Timestamp(window["timestamp"].iloc[-1])
    future_ts = pd.date_range(
        start=last_ts + pd.Timedelta(hours=1),
        periods=FORECAST_HORIZON,
        freq="h",
    )

    # ── assemble and save ──
    out = pd.DataFrame({
        "timestamp":        future_ts,
        "predicted_load_mw": preds_mw,
    })

    # Build the rich frontend payload: champion meta + when generated + rows.
    # One fetch gives the Vercel app everything it needs to render.
    payload = {
        "region":       region,
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "champion":     champ_meta,
        "forecast": [
            {"timestamp": ts.isoformat(), "predicted_load_mw": float(mw)}
            for ts, mw in zip(future_ts, preds_mw)
        ],
    }

    # save_forecast writes parquet (internal) + the rich JSON (frontend),
    # to S3 if configured, else local disk.
    location = save_forecast(region, out, payload=payload)

    print(f"\n  Forecast window: {future_ts[0]} → {future_ts[-1]}")
    print(f"  Predicted load:  {preds_mw.min():,.0f} – {preds_mw.max():,.0f} MW")
    print(f"  Saved → {location}")
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
