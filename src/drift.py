"""
Detect data drift: has recent load data moved away from what the champion
trained on? If so, the model is stale and should be retrained.

    python src/drift.py --country CAL

Compares a reference (the champion's training slice) against current data (a
fresh ~30-day EIA pull) with Evidently's K-S test per column. Exits 1 if the
drifted share crosses the threshold, so CI can branch on it.
"""

import argparse
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from evidently import Report, Dataset, DataDefinition
from evidently.presets import DataDriftPreset

from ingestion import fetch_region
from storage import load_reference

load_dotenv()

RAW_DIR     = Path(__file__).parent.parent / "data" / "raw"
REPORTS_DIR = Path(__file__).parent.parent / "reports"

TRAIN_RATIO = 0.70           # must match training split
CURRENT_HOURS = 720          # ~30 days counts as "current"
MONITOR_COLS = ["load_mw", "rolling_mean_24"]
DRIFT_SHARE_THRESHOLD = 0.5  # half or more of monitored columns drift -> drift


def add_rolling(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["rolling_mean_24"] = df["load_mw"].rolling(24, min_periods=1).mean()
    return df


def build_frames(region: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (reference, current). Reference = the champion's pinned training
    snapshot (so it's the exact distribution the live model learned from);
    current = a fresh EIA pull of the last ~30 days. Comparing fresh data to the
    stored tail of the same file would be comparing data to itself.
    """
    reference = load_reference(region)
    if reference is not None:
        reference = reference[MONITOR_COLS].copy()
        print(f"  Reference: loaded champion snapshot ({len(reference):,} rows)")
    else:
        # no snapshot yet — rebuild the train slice from local raw
        stored = add_rolling(pd.read_parquet(RAW_DIR / f"{region}.parquet"))
        train_end = int(len(stored) * TRAIN_RATIO)
        reference = stored.iloc[:train_end][MONITOR_COLS].copy()
        print(f"  Reference: no snapshot, rebuilt from local raw ({len(reference):,} rows)")

    api_key = os.environ["EIA_API_KEY"]
    # start before the window so the rolling mean has warmup rows
    start = (datetime.now(timezone.utc) - timedelta(hours=CURRENT_HOURS + 48)).strftime("%Y-%m-%dT%H")
    print(f"  Pulling fresh EIA data for {region} since {start}...")
    fresh = add_rolling(fetch_region(region, api_key, start=start))
    current = fresh.iloc[-CURRENT_HOURS:][MONITOR_COLS].copy()

    return reference, current


def check_drift(region: str, save_html: bool = True) -> bool:
    """Run Evidently drift detection. Returns True if drift detected."""
    reference, current = build_frames(region)

    data_def = DataDefinition(numerical_columns=MONITOR_COLS)
    ref_ds = Dataset.from_pandas(reference, data_definition=data_def)
    cur_ds = Dataset.from_pandas(current,   data_definition=data_def)

    report = Report([DataDriftPreset()])
    result = report.run(current_data=cur_ds, reference_data=ref_ds)
    d = result.dict()

    # column -> K-S p-value
    per_column = {}
    for metric in d["metrics"]:
        if metric["metric_name"].startswith("ValueDrift"):
            col = metric["config"].get("column", "?")
            per_column[col] = metric["value"]

    # Verdict from the same p-values we print (a column drifts if p < 0.05), so
    # the verdict can't disagree with the per-column numbers. Evidently's own
    # DriftedColumnsCount uses a different method and can contradict these.
    n_drifted   = sum(1 for p in per_column.values() if p < 0.05)
    drift_share = n_drifted / len(per_column) if per_column else 0.0
    drifted     = drift_share >= DRIFT_SHARE_THRESHOLD

    print(f"\n{'='*46}")
    print(f"  DRIFT CHECK - {region}")
    print(f"{'='*46}")
    print(f"  Reference rows: {len(reference):,}  |  Current rows: {len(current):,}")
    print(f"  Per-column K-S p-value (p < 0.05 = drifted):")
    for col, pval in per_column.items():
        flag = "DRIFT" if pval < 0.05 else "ok"
        print(f"    {col:<18} p={pval:.4g}  [{flag}]")
    print(f"  Drifted share: {drift_share:.0%}  (threshold {DRIFT_SHARE_THRESHOLD:.0%})")
    print(f"  VERDICT: {'DRIFT DETECTED -> retrain' if drifted else 'no drift -> champion OK'}")
    print(f"{'='*46}")

    if save_html:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        html_path = REPORTS_DIR / f"{region}_drift.html"
        result.save_html(str(html_path))
        print(f"  Report saved -> {html_path}")

    return drifted


def main():
    parser = argparse.ArgumentParser(description="Data drift detection.")
    parser.add_argument("--country", default="CAL", help="Region: CAL, TEX, PJM, MISO.")
    args = parser.parse_args()

    drifted = check_drift(args.country)
    raise SystemExit(1 if drifted else 0)  # exit 1 = drift, for CI branching


if __name__ == "__main__":
    main()
