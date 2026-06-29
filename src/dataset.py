"""
PyTorch Dataset/DataLoader for the sliding-window time series. Builds three
chronological datasets per region (train/val/test), each producing windows of
168 input hours -> 24 target hours.

    python src/dataset.py   # quick sanity check
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

FEATURES_DIR = Path(__file__).parent.parent / "data" / "features"

SEQ_LEN = 168          # input: 1 week of hours
FORECAST_HORIZON = 24  # output: next 1 day

# everything except timestamp goes to the model
FEATURE_COLS = [
    "load_mw",
    "hour_sin", "hour_cos",
    "dow_sin", "dow_cos",
    "month_sin", "month_cos",
    "is_weekend",
    "lag_1", "lag_24", "lag_168",
    "rolling_mean_24", "rolling_std_24",
]

# must match features.py
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15


class EnergyDataset(Dataset):
    """
    Sliding-window dataset. Window i is:
        X = rows i .. i+167                     -> (168, 13)
        y = load_mw at rows i+168 .. i+191      -> (24,)
    """

    def __init__(self, data: np.ndarray, load_col_idx: int):
        self.data = data
        self.load_col_idx = load_col_idx
        # need SEQ_LEN input rows + FORECAST_HORIZON target rows per window
        self.num_samples = len(data) - SEQ_LEN - FORECAST_HORIZON

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, i: int):
        X = self.data[i : i + SEQ_LEN]
        y = self.data[i + SEQ_LEN : i + SEQ_LEN + FORECAST_HORIZON, self.load_col_idx]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def load_datasets(region: str) -> tuple[EnergyDataset, EnergyDataset, EnergyDataset]:
    """Load a region's features and split chronologically into train/val/test."""
    df = pd.read_parquet(FEATURES_DIR / f"{region}.parquet")
    df = df.sort_values("timestamp").reset_index(drop=True)

    data = df[FEATURE_COLS].values
    load_col_idx = FEATURE_COLS.index("load_mw")

    n = len(data)
    train_end = int(n * TRAIN_RATIO)
    val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))

    train_ds = EnergyDataset(data[:train_end], load_col_idx)
    val_ds   = EnergyDataset(data[train_end:val_end], load_col_idx)
    test_ds  = EnergyDataset(data[val_end:], load_col_idx)

    return train_ds, val_ds, test_ds


def load_dataloaders(
    region: str,
    batch_size: int = 64,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Batched loaders for a region. Train is shuffled so the model doesn't learn
    example order; val/test are not, for consistent evaluation.
    """
    train_ds, val_ds, test_ds = load_datasets(region)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def main():
    region = "CAL"
    print(f"Loading datasets for {region}...")

    train_ds, val_ds, test_ds = load_datasets(region)
    print(f"  Train windows: {len(train_ds):,}")
    print(f"  Val windows:   {len(val_ds):,}")
    print(f"  Test windows:  {len(test_ds):,}")

    X, y = train_ds[0]
    print(f"\nSingle example:  X {X.shape} (168, 13)  y {y.shape} (24,)")

    train_loader, _, _ = load_dataloaders(region)
    X_batch, y_batch = next(iter(train_loader))
    print(f"One batch:  X {X_batch.shape} (64, 168, 13)  y {y_batch.shape} (64, 24)")


if __name__ == "__main__":
    main()
