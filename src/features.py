"""
Turn cleaned raw parquet (timestamp + load_mw) into a feature matrix, one per
region at data/features/<REGION>.parquet.

    python src/features.py

Adds Fourier time encodings, a weekend flag, lag features, and rolling stats,
then z-score normalizes. The scaler is fit on the training slice only (first
70% by time) to avoid leaking val/test info.
"""

import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR      = Path(__file__).parent.parent / "data" / "raw"
FEATURES_DIR = Path(__file__).parent.parent / "data" / "features"

REGIONS = ["CAL", "TEX", "PJM", "MISO"]

# chronological split — never shuffle a time series
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15  # test = remaining 15%


def add_fourier_encodings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sin/cos encodings for hour, day-of-week, month. We need both sin and cos
    because sin alone is ambiguous (e.g. sin maps hour 3 and hour 9 to the same
    value); together they're unique (x, y) coordinates on the cycle.
    """
    hour = df["timestamp"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    dow = df["timestamp"].dt.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    month = df["timestamp"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    return df


def add_weekend_flag(df: pd.DataFrame) -> pd.DataFrame:
    # demand drops on weekends (offices/factories closed) — a signal the
    # Fourier terms don't capture cleanly. dayofweek >= 5 is Sat/Sun.
    df["is_weekend"] = (df["timestamp"].dt.dayofweek >= 5).astype(int)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    # lag_1 = momentum, lag_24 = same hour yesterday, lag_168 = same hour last
    # week (the strongest signal). First 168 rows get NaN; dropped later.
    df["lag_1"]   = df["load_mw"].shift(1)
    df["lag_24"]  = df["load_mw"].shift(24)
    df["lag_168"] = df["load_mw"].shift(168)
    return df


def add_rolling_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling mean/std over the last 24h. shift(1) first so the window only sees
    the past — including the current row would leak the value we're predicting.
    """
    past_load = df["load_mw"].shift(1)
    df["rolling_mean_24"] = past_load.rolling(24, min_periods=1).mean()
    df["rolling_std_24"]  = past_load.rolling(24, min_periods=1).std().fillna(0)
    return df


def normalize_load(df: pd.DataFrame, train_size: int) -> tuple[pd.DataFrame, float, float]:
    """
    Z-score load_mw and the MW-unit features using TRAIN statistics only. Using
    the full dataset would let val/test influence the scaler (data leakage).
    Returns the df plus train mean/std (saved for inference).
    """
    train_mean = df["load_mw"].iloc[:train_size].mean()
    train_std  = df["load_mw"].iloc[:train_size].std()

    mw_cols = ["load_mw", "lag_1", "lag_24", "lag_168", "rolling_mean_24", "rolling_std_24"]
    for col in mw_cols:
        df[col] = (df[col] - train_mean) / train_std

    return df, train_mean, train_std


def build_features(region: str) -> None:
    print(f"\n{'='*50}")
    print(f"Building features: {region}")
    print(f"{'='*50}")

    df = pd.read_parquet(RAW_DIR / f"{region}.parquet")
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"  Loaded {len(df):,} rows")

    df = add_fourier_encodings(df)
    df = add_weekend_flag(df)
    df = add_lag_features(df)
    df = add_rolling_stats(df)

    train_size = int(len(df) * TRAIN_RATIO)
    val_size   = int(len(df) * VAL_RATIO)
    print(f"  Split -> train: {train_size:,} | val: {val_size:,} | test: {len(df)-train_size-val_size:,}")

    df, mean, std = normalize_load(df, train_size)
    print(f"  Normalization -> mean={mean:,.1f} MW, std={std:,.1f} MW")

    # drop lag-warmup NaNs (first ~168 rows have no lag_168)
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    print(f"  Dropped {before - len(df)} NaN rows (expected ~168 from lag warmup)")

    # stash scaler + split sizes so inference can reuse them
    df.attrs["train_mean"] = mean
    df.attrs["train_std"]  = std
    df.attrs["train_size"] = train_size
    df.attrs["val_size"]   = val_size

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FEATURES_DIR / f"{region}.parquet"
    df.to_parquet(out_path, index=False)

    print(f"  Columns: {list(df.columns)}")
    print(f"  Final shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"  Saved -> {out_path}")


def main():
    for region in REGIONS:
        build_features(region)
    print("\nFeature engineering complete.")


if __name__ == "__main__":
    main()
