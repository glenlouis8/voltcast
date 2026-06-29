"""
src/drift.py

Detect data drift: has recent electricity-load data drifted away from the
data the model trained on? If yes → the model is stale → trigger a retrain.

Run:
    python src/drift.py --country CAL

How it works:
    REFERENCE = the data the model learned from (training slice).
    CURRENT   = the most recent stretch of data (e.g. last 30 days).
    Evidently runs a Kolmogorov-Smirnov test per column and reports
    which columns drifted and by how much.

    Verdict: if the SHARE of drifted columns >= DRIFT_SHARE_THRESHOLD,
    we call it drift and return True.

We drift-check on RAW megawatts (interpretable), plus a rolling-mean
column, so "DriftedColumnsCount" has more than one signal to weigh.

Note: offline, REFERENCE and CURRENT both come from the stored raw parquet
(train slice vs latest slice). In the cloud workflow, CURRENT will be a
fresh EIA pull — same code, fresher data.
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

load_dotenv()  # so EIA_API_KEY is available

# ── constants ────────────────────────────────────────────────────────────────

RAW_DIR     = Path(__file__).parent.parent / "data" / "raw"
REPORTS_DIR = Path(__file__).parent.parent / "reports"

# Must match the training split so REFERENCE = what the model actually saw.
TRAIN_RATIO = 0.70

# How many recent hours count as "current". 720 = ~30 days.
CURRENT_HOURS = 720

# Columns we monitor for drift.
MONITOR_COLS = ["load_mw", "rolling_mean_24"]

# If this fraction (or more) of monitored columns drift → declare drift.
# 0.5 = "half or more of the signals moved".
DRIFT_SHARE_THRESHOLD = 0.5


# ── build reference + current frames ──────────────────────────────────────────

def add_rolling(df: pd.DataFrame) -> pd.DataFrame:
    """Add the rolling_mean_24 signal so we monitor 2 columns, not just raw load."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["rolling_mean_24"] = df["load_mw"].rolling(24, min_periods=1).mean()
    return df


def build_frames(region: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (reference_df, current_df), each with the MONITOR_COLS.

    reference = the training slice from STORED data — the model's "normal".
    current   = a FRESH pull from EIA of the last ~30 days — what's happening now.

    This is the point of drift: compare the model's old world (reference)
    against genuinely new data (current). If current was just the tail of the
    same stored file, we'd be comparing data to itself — meaningless.
    """
    # ── reference: the champion's pinned training snapshot ──
    # Prefer the snapshot saved to storage (S3/local) when the champion was
    # crowned — that's the exact distribution the live model learned from.
    # Fall back to rebuilding from local raw if no snapshot exists yet.
    reference = load_reference(region)
    if reference is not None:
        reference = reference[MONITOR_COLS].copy()
        print(f"  Reference: loaded champion snapshot ({len(reference):,} rows)")
    else:
        stored = pd.read_parquet(RAW_DIR / f"{region}.parquet")
        stored = add_rolling(stored)
        train_end = int(len(stored) * TRAIN_RATIO)
        reference = stored.iloc[:train_end][MONITOR_COLS].copy()
        print(f"  Reference: no snapshot, rebuilt from local raw ({len(reference):,} rows)")

    # ── current: fresh pull from EIA ──
    api_key = os.environ["EIA_API_KEY"]
    # Start a bit before CURRENT_HOURS ago so the rolling mean has warmup rows.
    start = (datetime.now(timezone.utc) - timedelta(hours=CURRENT_HOURS + 48)).strftime("%Y-%m-%dT%H")
    print(f"  Pulling fresh EIA data for {region} since {start}...")
    fresh = fetch_region(region, api_key, start=start)
    fresh = add_rolling(fresh)
    current = fresh.iloc[-CURRENT_HOURS:][MONITOR_COLS].copy()

    return reference, current


# ── run the drift report ──────────────────────────────────────────────────────

def check_drift(region: str, save_html: bool = True) -> bool:
    """
    Run Evidently drift detection for a region. Returns True if drift detected.

    Also saves an HTML visual report to reports/<region>_drift.html
    (great for the dashboard and as portfolio proof).
    """
    reference, current = build_frames(region)

    # Tell Evidently these columns are numbers (so it picks numeric drift tests).
    data_def = DataDefinition(numerical_columns=MONITOR_COLS)
    ref_ds = Dataset.from_pandas(reference, data_definition=data_def)
    cur_ds = Dataset.from_pandas(current,   data_definition=data_def)

    # DataDriftPreset = a bundle that drift-tests every column + counts how many drifted.
    report = Report([DataDriftPreset()])
    result = report.run(current_data=cur_ds, reference_data=ref_ds)

    # ── pull the numbers out of the result ──
    d = result.dict()

    per_column = {}          # column → K-S p-value
    for metric in d["metrics"]:
        if metric["metric_name"].startswith("ValueDrift"):
            # metric_name looks like: ValueDrift(column=load_mw,method=K-S p_value)
            col = metric["config"].get("column", "?")
            per_column[col] = metric["value"]

    # ── verdict ──
    # Compute the drifted share from the SAME K-S p-values we print, so the
    # verdict and the per-column numbers can never disagree. A column drifted
    # if its p-value < 0.05. (Evidently's own DriftedColumnsCount uses a
    # different internal method and can contradict these p-values, so we ignore it.)
    n_drifted   = sum(1 for p in per_column.values() if p < 0.05)
    drift_share = n_drifted / len(per_column) if per_column else 0.0
    drifted     = drift_share >= DRIFT_SHARE_THRESHOLD

    print(f"\n{'='*46}")
    print(f"  DRIFT CHECK — {region}")
    print(f"{'='*46}")
    print(f"  Reference rows: {len(reference):,}  |  Current rows: {len(current):,}")
    print(f"  Per-column K-S p-value (p < 0.05 = drifted):")
    for col, pval in per_column.items():
        flag = "DRIFT" if pval < 0.05 else "ok"
        print(f"    {col:<18} p={pval:.4g}  [{flag}]")
    print(f"  Drifted share: {drift_share:.0%}  (threshold {DRIFT_SHARE_THRESHOLD:.0%})")
    print(f"  VERDICT: {'DRIFT DETECTED → retrain' if drifted else 'no drift → champion OK'}")
    print(f"{'='*46}")

    # ── save HTML report ──
    if save_html:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        html_path = REPORTS_DIR / f"{region}_drift.html"
        result.save_html(str(html_path))
        print(f"  Report saved → {html_path}")

    return drifted


def main():
    parser = argparse.ArgumentParser(description="Data drift detection.")
    parser.add_argument("--country", default="CAL", help="Region: CAL, TEX, PJM, MISO.")
    args = parser.parse_args()

    drifted = check_drift(args.country)
    # Exit code 1 if drift — lets a CI job branch on it (if drift, run retrain).
    raise SystemExit(1 if drifted else 0)


if __name__ == "__main__":
    main()
