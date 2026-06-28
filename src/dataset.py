"""
src/dataset.py

PyTorch Dataset and DataLoader for the sliding window time series.

A Dataset is a Python class that wraps your data and lets PyTorch:
    1. Know how many examples exist (__len__)
    2. Fetch any single example by index (__getitem__)

The sliding window:
    Input:  168 consecutive hours of all 13 features
    Target: the next 24 hours of load_mw only

We build three separate datasets per region: train, val, test.
Each uses a different slice of rows — chronological, never shuffled.

Run:
    python src/dataset.py   (runs a quick sanity check)
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# ── constants ────────────────────────────────────────────────────────────────

FEATURES_DIR = Path(__file__).parent.parent / "data" / "features"

# How many past hours the model sees as input
SEQ_LEN = 168          # 1 week

# How many future hours the model predicts
FORECAST_HORIZON = 24  # 1 day

# Feature columns fed to the model (everything except timestamp)
FEATURE_COLS = [
    "load_mw",
    "hour_sin", "hour_cos",
    "dow_sin", "dow_cos",
    "month_sin", "month_cos",
    "is_weekend",
    "lag_1", "lag_24", "lag_168",
    "rolling_mean_24", "rolling_std_24",
]

# Train/val/test split — must match features.py exactly
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15


# ── dataset class ─────────────────────────────────────────────────────────────

class EnergyDataset(Dataset):
    """
    Sliding window dataset for electricity demand forecasting.

    Takes a 2D array of shape (num_rows, num_features).
    Each call to __getitem__(i) returns:
        X: rows i to i+SEQ_LEN-1         → shape (168, 13)  ← model input
        y: load_mw at rows i+SEQ_LEN to i+SEQ_LEN+HORIZON-1 → shape (24,) ← target

    Example:
        i=0  → X=rows 0–167,   y=rows 168–191
        i=1  → X=rows 1–168,   y=rows 169–192
        i=2  → X=rows 2–169,   y=rows 170–193
    """

    def __init__(self, data: np.ndarray, load_col_idx: int):
        """
        Args:
            data:         2D numpy array, shape (num_rows, num_features)
            load_col_idx: which column index is load_mw (the target column)
        """
        self.data         = data
        self.load_col_idx = load_col_idx

        # Number of valid windows we can make.
        # Each window needs SEQ_LEN rows for input + FORECAST_HORIZON rows for target.
        # So the last valid start index is: len(data) - SEQ_LEN - FORECAST_HORIZON
        self.num_samples = len(data) - SEQ_LEN - FORECAST_HORIZON

    def __len__(self) -> int:
        # PyTorch calls this to know how many examples exist.
        return self.num_samples

    def __getitem__(self, i: int):
        # Input: SEQ_LEN rows of ALL features starting at index i
        # Shape: (168, 13)
        X = self.data[i : i + SEQ_LEN]

        # Target: next FORECAST_HORIZON values of load_mw only
        # Shape: (24,)
        y = self.data[i + SEQ_LEN : i + SEQ_LEN + FORECAST_HORIZON, self.load_col_idx]

        # Convert numpy arrays to PyTorch tensors.
        # A tensor is PyTorch's version of a numpy array — same idea,
        # but supports GPU and automatic gradient computation.
        # float32 = 32-bit float, standard for neural network training.
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# ── helper: build train/val/test datasets for one region ─────────────────────

def load_datasets(region: str) -> tuple[EnergyDataset, EnergyDataset, EnergyDataset]:
    """
    Load feature parquet for a region and return (train, val, test) datasets.

    Splits are chronological — we slice by row index, not randomly.
    Train = first 70%, Val = next 15%, Test = last 15%.
    """
    df = pd.read_parquet(FEATURES_DIR / f"{region}.parquet")
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Convert to numpy for fast indexing inside __getitem__
    # We drop timestamp — it's not a number the model can use
    data = df[FEATURE_COLS].values  # shape: (num_rows, 13)

    # Which column is load_mw? We need this to build the target (y)
    load_col_idx = FEATURE_COLS.index("load_mw")

    # Compute split boundaries
    n = len(data)
    train_end = int(n * TRAIN_RATIO)
    val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))

    # Slice the data into three chronological sections.
    # Important: we do NOT overlap these slices.
    # Each split is a separate EnergyDataset with its own sliding windows.
    train_data = data[:train_end]
    val_data   = data[train_end:val_end]
    test_data  = data[val_end:]

    train_ds = EnergyDataset(train_data, load_col_idx)
    val_ds   = EnergyDataset(val_data,   load_col_idx)
    test_ds  = EnergyDataset(test_data,  load_col_idx)

    return train_ds, val_ds, test_ds


def load_dataloaders(
    region: str,
    batch_size: int = 64,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader, test_loader) for a region.

    A DataLoader wraps a Dataset and:
        - Batches examples together (64 at a time)
        - Shuffles training data each epoch (NOT val/test)
        - Handles multi-sample collation automatically

    Why shuffle train but not val/test?
        Shuffling training prevents the model from memorizing the order
        of examples. Val/test are never shuffled because we want consistent
        evaluation — same order every time.

    Why batch_size=64?
        Each forward pass processes 64 windows at once. More efficient
        than one window at a time. 64 is a standard default that fits
        in memory comfortably.
    """
    train_ds, val_ds, test_ds = load_datasets(region)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# ── sanity check ──────────────────────────────────────────────────────────────

def main():
    """Quick check: load CAL, print shapes of one batch."""
    region = "CAL"
    print(f"Loading datasets for {region}...")

    train_ds, val_ds, test_ds = load_datasets(region)

    print(f"  Train windows: {len(train_ds):,}")
    print(f"  Val windows:   {len(val_ds):,}")
    print(f"  Test windows:  {len(test_ds):,}")

    # Fetch one example manually to verify shapes
    X, y = train_ds[0]
    print(f"\nSingle example:")
    print(f"  X shape: {X.shape}  ← (seq_len=168, num_features=13)")
    print(f"  y shape: {y.shape}  ← (forecast_horizon=24)")
    print(f"  X[0] (first hour, all features): {X[0]}")
    print(f"  y (24h target load_mw normalized): {y}")

    # Load one batch via DataLoader
    train_loader, _, _ = load_dataloaders(region)
    X_batch, y_batch = next(iter(train_loader))
    print(f"\nOne batch from DataLoader:")
    print(f"  X_batch shape: {X_batch.shape}  ← (batch=64, seq=168, features=13)")
    print(f"  y_batch shape: {y_batch.shape}  ← (batch=64, horizon=24)")


if __name__ == "__main__":
    main()
