"""
src/features.py

Takes cleaned raw parquet (timestamp + load_mw) and builds a rich feature matrix.
Saves one parquet per region to data/features/<REGION>.parquet

Run:
    python src/features.py

What this file does, in order:
    1. Add Fourier encodings (hour, day-of-week, month)
    2. Add is_weekend flag
    3. Add lag features (load 1h, 24h, 168h ago)
    4. Add rolling stats (mean and std of last 24h)
    5. Z-score normalize load_mw (fit on train only — no leakage)
    6. Drop rows that have NaN from lag/rolling computation
    7. Save to data/features/<REGION>.parquet

Important rule: normalization scaler is fit on training data only.
Train = first 70% of rows by time. Never touch val/test to compute mean/std.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── constants ────────────────────────────────────────────────────────────────

RAW_DIR      = Path(__file__).parent.parent / "data" / "raw"
FEATURES_DIR = Path(__file__).parent.parent / "data" / "features"

REGIONS = ["CAL", "TEX", "PJM", "MISO"]

# Train/val/test split ratios.
# We use the first 70% of hours for training, next 15% for validation, last 15% for test.
# These must be chronological — never shuffle a time series.
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# TEST_RATIO  = 0.15  (implied — whatever's left)


# ── feature functions ─────────────────────────────────────────────────────────

def add_fourier_encodings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add sin/cos encodings for hour, day-of-week, and month.

    Why sin AND cos? Because sin alone is ambiguous:
        sin(hour=6)  = 1.0
        sin(hour=18) = -1.0  ... these are different, good.
        BUT sin(hour=3) = sin(hour=9) = 0.5 ... same value, different hours!
    Adding cos breaks that ambiguity. Together, (sin, cos) uniquely identifies
    every position on the cycle — like (x, y) coordinates on a clock face.

    2π = one full circle. Dividing by the cycle length (24, 7, 12) spaces
    the values evenly around the circle.
    """
    # Hour of day: cycles every 24 hours
    hour = df["timestamp"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    # Day of week: Monday=0, Sunday=6, cycles every 7 days
    dow = df["timestamp"].dt.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    # Month: cycles every 12 months
    month = df["timestamp"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    return df


def add_weekend_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a binary column: 1 if Saturday or Sunday, 0 otherwise.

    Electricity demand drops on weekends — factories and offices are closed.
    This single bit gives the model an important signal it can't derive
    from the Fourier encodings alone (weekends aren't evenly spaced in a
    simple cycle relative to load patterns).

    dayofweek: Monday=0, Tuesday=1, ..., Saturday=5, Sunday=6
    So >= 5 means weekend.
    """
    df["is_weekend"] = (df["timestamp"].dt.dayofweek >= 5).astype(int)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag features: what was load_mw N hours ago?

    .shift(N) shifts the column down by N rows, so each row now sees
    the value that was N rows (= N hours) before it.

    Example with shift(1):
        row 0: load=100  → lag_1=NaN   (no previous row)
        row 1: load=110  → lag_1=100
        row 2: load=105  → lag_1=110

    Why these specific lags?
    - lag_1:   what just happened (momentum)
    - lag_24:  same hour yesterday (daily pattern)
    - lag_168: same hour last week (weekly pattern — strongest signal)

    The first 168 rows will have NaN in lag_168. We drop those later.
    """
    df["lag_1"]   = df["load_mw"].shift(1)
    df["lag_24"]  = df["load_mw"].shift(24)
    df["lag_168"] = df["load_mw"].shift(168)
    return df


def add_rolling_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling mean and std over the last 24 hours.

    rolling(24) creates a window of 24 rows. For each row, it looks
    at that row and the 23 rows before it.

    .mean() = average load over the last 24 hours
        → tells model: is current usage above or below recent average?

    .std() = standard deviation over the last 24 hours
        → tells model: is load stable or swinging wildly right now?

    min_periods=1 means: compute even if fewer than 24 rows are available
    (handles the start of the series). Without it, first 23 rows = NaN.

    shift(1) is critical: we shift BEFORE computing stats.
    Without shift(1), the rolling window would include the current row —
    meaning the model gets to "see" the current load_mw as a feature,
    which is the value we're trying to predict. That's cheating (data leakage).
    We only want to look at the PAST, not the present.
    """
    past_load = df["load_mw"].shift(1)
    df["rolling_mean_24"] = past_load.rolling(24, min_periods=1).mean()
    df["rolling_std_24"]  = past_load.rolling(24, min_periods=1).std().fillna(0)
    return df


def normalize_load(df: pd.DataFrame, train_size: int) -> tuple[pd.DataFrame, float, float]:
    """
    Z-score normalize load_mw and all lag features using train statistics only.

    Z-score formula: (x - mean) / std
    Result: mean=0, std=1. Neural networks train much better on small numbers.

    CRITICAL: we compute mean and std from training rows ONLY (first train_size rows).
    If we used the full dataset, val/test data would influence the scaler —
    that's data leakage. The model would indirectly "know" future values.

    We also normalize the lag features (lag_1, lag_24, lag_168, rolling stats)
    using the same mean/std, because they are all in MW units — same scale.

    Returns the modified DataFrame plus mean and std (we save these for inference).
    """
    # Compute mean and std from training rows only
    train_mean = df["load_mw"].iloc[:train_size].mean()
    train_std  = df["load_mw"].iloc[:train_size].std()

    # Columns that are in MW units — normalize all with same mean/std
    mw_cols = ["load_mw", "lag_1", "lag_24", "lag_168", "rolling_mean_24", "rolling_std_24"]

    for col in mw_cols:
        df[col] = (df[col] - train_mean) / train_std

    return df, train_mean, train_std


# ── main ─────────────────────────────────────────────────────────────────────

def build_features(region: str) -> None:
    """Full feature engineering pipeline for one region."""

    print(f"\n{'='*50}")
    print(f"Building features: {region}")
    print(f"{'='*50}")

    # Load cleaned raw data
    df = pd.read_parquet(RAW_DIR / f"{region}.parquet")
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"  Loaded {len(df):,} rows")

    # Step 1–4: add all features
    df = add_fourier_encodings(df)
    df = add_weekend_flag(df)
    df = add_lag_features(df)
    df = add_rolling_stats(df)

    # Step 5: compute train size (70% by row count)
    train_size = int(len(df) * TRAIN_RATIO)
    val_size   = int(len(df) * VAL_RATIO)

    print(f"  Split → train: {train_size:,} | val: {val_size:,} | test: {len(df)-train_size-val_size:,}")

    # Step 6: normalize (fit on train only)
    df, mean, std = normalize_load(df, train_size)
    print(f"  Normalization → mean={mean:,.1f} MW, std={std:,.1f} MW")

    # Step 7: drop rows with NaN
    # The first 168 rows will have NaN in lag_168 because there's no data
    # 168 hours before the start of the series. We can't use those rows.
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    print(f"  Dropped {before - len(df)} NaN rows (expected ~168 from lag warmup)")

    # Save the scaler stats alongside the features so inference can use them later
    df.attrs["train_mean"] = mean
    df.attrs["train_std"]  = std
    df.attrs["train_size"] = train_size
    df.attrs["val_size"]   = val_size

    # Save to data/features/
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FEATURES_DIR / f"{region}.parquet"
    df.to_parquet(out_path, index=False)

    print(f"  Columns: {list(df.columns)}")
    print(f"  Final shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Saved → {out_path}")


def main():
    for region in REGIONS:
        build_features(region)
    print("\nFeature engineering complete.")


if __name__ == "__main__":
    main()
