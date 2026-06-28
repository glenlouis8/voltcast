"""
api/main.py

FastAPI service that serves the forecasts.

Start it:
    uvicorn api.main:app --reload
Then open the auto-generated test page:
    http://localhost:8000/docs

Endpoints:
    GET /forecast?country=CAL&hours=24
        → the next N hours of predicted load (reads the parquet
          that inference.py saved).
    GET /health
        → which champion model serves each region, and its test MAE.

Design note: /forecast reads the pre-computed parquet rather than
running the model on every request. The forecast is refreshed when
inference.py runs (daily, by a scheduled job later). Serving a file
is fast and keeps the API simple.
"""

import sys
from pathlib import Path

import pandas as pd
from mlflow.tracking import MlflowClient
from fastapi import FastAPI, HTTPException

# src/ holds shared helpers (mlflow_setup). Add it to the import path.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from mlflow_setup import setup_mlflow

# api/ is one level under the project root, so go up one parent (not two).
FORECASTS_DIR = Path(__file__).parent.parent / "data" / "forecasts"

REGIONS = ["CAL", "TEX", "PJM", "MISO"]

# Create the app object. Uvicorn looks for this `app` variable.
app = FastAPI(
    title="VoltCast",
    description="24-hour electricity demand forecasts for US grid regions.",
    version="1.0",
)

# Point MLflow at the same place the rest of the project uses (DagsHub or
# local), so /health can read the champion registry.
setup_mlflow()


# ── GET /forecast ─────────────────────────────────────────────────────────────

@app.get("/forecast")
def get_forecast(country: str = "CAL", hours: int = 24):
    """
    Return the next `hours` of predicted load for `country`.

    country: one of CAL, TEX, PJM, MISO.
    hours:   how many of the 24 forecast hours to return (1–24).

    FastAPI reads `country` and `hours` straight from the URL query string,
    e.g. /forecast?country=CAL&hours=12. The type hints (str, int) make
    FastAPI validate and convert them automatically.
    """
    # Reject unknown regions early with a clear 400 error.
    if country not in REGIONS:
        raise HTTPException(status_code=400, detail=f"Unknown region '{country}'. Use {REGIONS}.")

    path = FORECASTS_DIR / f"{country}_forecast.parquet"
    if not path.exists():
        # 404 = the forecast hasn't been generated yet (run inference.py).
        raise HTTPException(status_code=404, detail=f"No forecast for {country}. Run inference.py first.")

    df = pd.read_parquet(path)
    df = df.sort_values("timestamp").head(hours)

    # Return as a JSON-friendly structure: a list of {timestamp, mw} rows.
    return {
        "country": country,
        "generated_hours": len(df),
        "forecast": [
            {"timestamp": ts.isoformat(), "predicted_load_mw": round(float(mw), 1)}
            for ts, mw in zip(df["timestamp"], df["predicted_load_mw"])
        ],
    }


# ── GET /health ───────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """
    Report the serving champion for each region and its test MAE.

    Reads the MLflow registry: for each region, look up the model version
    tagged 'champion', read its stored test_mae_mw tag. If a region has no
    champion yet, mark it accordingly instead of crashing.
    """
    client = MlflowClient()
    status = {}

    for region in REGIONS:
        name = f"voltcast-{region}"
        try:
            champ = client.get_model_version_by_alias(name, "champion")
            status[region] = {
                "champion_version": champ.version,
                "model_type":       champ.tags.get("model_type", "unknown"),
                "test_mae_mw":      round(float(champ.tags.get("test_mae_mw", 0)), 1),
            }
        except Exception:
            status[region] = {"champion_version": None, "note": "no champion yet"}

    return {"status": "ok", "regions": status}
