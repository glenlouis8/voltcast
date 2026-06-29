"""
dashboard/app.py

Streamlit dashboard for VoltCast — the project's visual face.

Run:
    streamlit run dashboard/app.py
Then open http://localhost:8501

Shows, per region:
    1. Champion model card — which model serves, its test MAE (from MLflow registry)
    2. 24-hour forecast — line chart + table (from the forecast parquet)

Streamlit model: the script runs top-to-bottom on every interaction.
Each st.something() draws a widget. No HTML, no JavaScript — pure Python.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# src/ holds shared helpers. Add it to the path so we can import setup_mlflow.
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from mlflow_setup import setup_mlflow
from storage import load_forecast as load_forecast_store

from mlflow.tracking import MlflowClient

REGIONS = ["CAL", "TEX", "PJM", "MISO"]


# ── data loaders (cached so they don't re-run on every click) ─────────────────

@st.cache_resource
def get_client() -> MlflowClient:
    """Connect to MLflow (DagsHub or local) once and reuse the client."""
    setup_mlflow()
    return MlflowClient()


@st.cache_data(ttl=300)
def load_champion(region: str) -> dict | None:
    """
    Read the champion model info for a region from the registry.
    Returns None if no champion exists yet (so the UI can say so).
    Cached for 5 minutes (ttl=300) to avoid hammering the registry.
    """
    client = get_client()
    try:
        champ = client.get_model_version_by_alias(f"voltcast-{region}", "champion")
        return {
            "version":    champ.version,
            "model_type": champ.tags.get("model_type", "unknown"),
            "test_mae":   float(champ.tags.get("test_mae_mw", 0)),
        }
    except Exception:
        return None


@st.cache_data(ttl=300)
def load_forecast(region: str) -> pd.DataFrame | None:
    """Read the 24h forecast (from S3 if configured, else local). None if absent."""
    df = load_forecast_store(region)
    if df is None:
        return None
    return df.sort_values("timestamp")


# ── page layout ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="VoltCast", page_icon="⚡", layout="wide")

st.title("⚡ VoltCast — US Electricity Demand Forecasting")
st.caption("24-hour-ahead load forecasts for US grid regions, powered by a from-scratch Transformer.")

# Sidebar region picker.
region = st.sidebar.selectbox("Region", REGIONS)
st.sidebar.markdown(
    "**Regions**\n\n"
    "- CAL — California\n"
    "- TEX — Texas (ERCOT)\n"
    "- PJM — Mid-Atlantic\n"
    "- MISO — Midwest"
)

# ── champion card ──
st.subheader(f"Champion model — {region}")
champ = load_champion(region)
if champ is None:
    st.warning(f"No champion registered for {region} yet. Run: `python src/registry.py --country {region}`")
else:
    c1, c2, c3 = st.columns(3)
    c1.metric("Model", champ["model_type"].title())
    c2.metric("Version", f"v{champ['version']}")
    c3.metric("Test MAE", f"{champ['test_mae']:,.0f} MW")

st.divider()

# ── forecast ──
st.subheader(f"Next 24 hours — {region}")
fc = load_forecast(region)
if fc is None:
    st.info(f"No forecast yet for {region}. Run: `python src/inference.py --country {region}`")
else:
    # Line chart: x = time, y = predicted MW.
    chart_df = fc.rename(columns={"predicted_load_mw": "Predicted load (MW)"}).set_index("timestamp")
    st.line_chart(chart_df["Predicted load (MW)"])

    # Quick stats + raw table.
    s1, s2, s3 = st.columns(3)
    s1.metric("Peak", f"{fc['predicted_load_mw'].max():,.0f} MW")
    s2.metric("Trough", f"{fc['predicted_load_mw'].min():,.0f} MW")
    s3.metric("Window start", f"{fc['timestamp'].min():%b %d %H:%M} UTC")

    with st.expander("Show hourly values"):
        st.dataframe(
            fc.assign(timestamp=fc["timestamp"].dt.strftime("%Y-%m-%d %H:%M"))
              .rename(columns={"predicted_load_mw": "predicted MW"}),
            hide_index=True,
            use_container_width=True,
        )
