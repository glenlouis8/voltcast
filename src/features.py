"""
Turn clean demand data into model inputs, one file per region:
data/features/<REGION>.parquet

    python src/features.py

Features: calendar (hour, day of week, month, weekend), lags (demand 24 hours
and 168 hours ago), and z-score scaling of the demand columns. The scaling
average/spread are saved to data/features/scalers.json so they can be reversed.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
FEATURES_DIR = Path(__file__).parent.parent / "data" / "features"

REGIONS = ["CAL", "TEX", "PJM", "MISO"]

# 5 years split by time: first 60% (3 years) train, next 20% (1 year) validation,
# last 20% (1 year) test. Only the train share is used here, to fit the scaling.
TRAIN_RATIO = 0.60
DEMAND_COLS = ["load_mw", "lag_24", "lag_168"]  # columns scaled together

# Each grid lives in a different time zone. People use power by the local clock
# (6pm is evening rush everywhere), so calendar features use local time.
TIMEZONES = {
    "CAL": "America/Los_Angeles",
    "TEX": "America/Chicago",
    "PJM": "America/New_York",
    "MISO": "America/Chicago",
}


def add_calendar_features(df: pd.DataFrame, region: str) -> pd.DataFrame:
    # our timestamps are UTC; convert to the region's local clock
    local = df["timestamp"].dt.tz_localize("UTC").dt.tz_convert(TIMEZONES[region])

    hour = local.dt.hour               # 0..23
    day_of_week = local.dt.dayofweek   # Monday = 0 ... Sunday = 6
    month = local.dt.month             # 1..12

    # sin/cos puts a repeating number on a circle, so hour 23 and hour 0 are
    # neighbors instead of far apart. Each needs both sin and cos to be unique.
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    df["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    df["is_weekend"] = (day_of_week >= 5).astype(int)  # Saturday or Sunday
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    # shift(n) slides the column down n rows, so each row gets the demand from n
    # hours earlier. This works because validation made it exactly 1 row per hour.
    df["lag_24"] = df["load_mw"].shift(24)    # same hour yesterday
    df["lag_168"] = df["load_mw"].shift(168)  # same hour last week
    return df


def scale_demand(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    z-score: scaled = (value - average) / spread. The average and spread come from
    the TRAINING rows only (first 60%). Using all rows would let the test period
    leak into the scaling.
    """
    train_end = int(len(df) * TRAIN_RATIO)
    mean = df["load_mw"].iloc[:train_end].mean()
    std = df["load_mw"].iloc[:train_end].std()

    # all three columns are demand in MW, so they share one average and spread
    for col in DEMAND_COLS:
        df[col] = (df[col] - mean) / std

    return df, {"mean": float(mean), "std": float(std), "train_rows": train_end}


def build_features(region: str) -> tuple[pd.DataFrame, dict]:
    df = pd.read_parquet(RAW_DIR / f"{region}.parquet").sort_values("timestamp")
    df = df.reset_index(drop=True)
    df = add_calendar_features(df, region)
    df = add_lag_features(df)

    # the first 168 rows have no "last week" yet (lag_168 is empty), so drop them
    df = df.dropna().reset_index(drop=True)
    return scale_demand(df)


def main():
    parser = argparse.ArgumentParser(description="Build model inputs from the clean raw data.")
    parser.add_argument("--region", choices=REGIONS, help="only this region (default: all 4)")
    args = parser.parse_args()

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    scalers_file = FEATURES_DIR / "scalers.json"
    # keep the scalers of regions we are NOT rebuilding right now
    scalers = json.loads(scalers_file.read_text()) if scalers_file.exists() else {}
    for region in [args.region] if args.region else REGIONS:
        df, scalers[region] = build_features(region)
        df.to_parquet(FEATURES_DIR / f"{region}.parquet", index=False)
        s = scalers[region]
        print(f"{region}: {df.shape[0]:,} rows x {df.shape[1]} columns | "
              f"average {s['mean']:,.0f} MW, spread {s['std']:,.0f} MW")

    (FEATURES_DIR / "scalers.json").write_text(json.dumps(scalers, indent=2))
    print("\nColumns:", list(df.columns))
    print(f"Scalers saved -> {FEATURES_DIR / 'scalers.json'}")


if __name__ == "__main__":
    main()
