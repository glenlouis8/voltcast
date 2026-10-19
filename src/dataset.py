"""
Split each region's feature file by time into train / validation / test and cut
each part into sliding windows: 168 hours in -> next 24 hours out.

    python src/dataset.py   # quick check: sizes, date ranges, one batch
"""

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from features import TRAIN_RATIO  # same train share the scaler used, so they can't disagree

FEATURES_DIR = Path(__file__).parent.parent / "data" / "features"

SEQ_LEN = 168          # model input: the last 168 hours (1 week)
FORECAST_HORIZON = 24  # model output: the next 24 hours
VAL_RATIO = 0.20       # test gets the rest (20%)

# the 10 columns the model sees (timestamp is not one of them).
# load_mw MUST stay first: the target is read from column 0.
FEATURE_COLS = [
    "load_mw",
    "hour_sin", "hour_cos",
    "dow_sin", "dow_cos",
    "month_sin", "month_cos",
    "is_weekend",
    "lag_24", "lag_168",
]


class WindowDataset(Dataset):
    """
    Window i:
        X = rows i .. i+167             -> (168, 10)  what the model sees
        y = load_mw of rows i+168 .. i+191 -> (24,)   what it must predict
    """

    def __init__(self, data: torch.Tensor):
        self.data = data
        # a window needs 168 input rows + 24 target rows, and can start at any row
        # that leaves room for both
        self.num_windows = len(data) - SEQ_LEN - FORECAST_HORIZON + 1

    def __len__(self) -> int:
        return self.num_windows

    def __getitem__(self, i: int):
        X = self.data[i : i + SEQ_LEN]
        y = self.data[i + SEQ_LEN : i + SEQ_LEN + FORECAST_HORIZON, 0]  # column 0 = load_mw
        return X, y


def split_by_time(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Oldest 60% = train, next 20% = validation, newest 20% = test. Never shuffled."""
    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def make_datasets(region: str) -> tuple[WindowDataset, WindowDataset, WindowDataset]:
    df = pd.read_parquet(FEATURES_DIR / f"{region}.parquet")
    df = df.sort_values("timestamp").reset_index(drop=True)

    parts = split_by_time(df)
    # windows are cut inside each part, so no window mixes train and test rows
    return tuple(
        WindowDataset(torch.tensor(p[FEATURE_COLS].values, dtype=torch.float32))
        for p in parts
    )


def make_dataloaders(region: str, batch_size: int = 64):
    """
    Batches of 64 windows. Only the training windows are shuffled (the ORDER OF
    WINDOWS, each window itself stays in time order). Validation/test stay in order.
    """
    train_ds, val_ds, test_ds = make_datasets(region)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )


def main():
    region = "CAL"
    df = pd.read_parquet(FEATURES_DIR / f"{region}.parquet").sort_values("timestamp")
    parts = split_by_time(df.reset_index(drop=True))
    train_ds, val_ds, test_ds = make_datasets(region)

    print(f"{region}: {len(df):,} rows (UTC timestamps)")
    for name, part, ds in zip(("train", "validation", "test"), parts, (train_ds, val_ds, test_ds)):
        print(f"  {name:<10} {len(part):>6,} rows  {part.timestamp.min():%Y-%m-%d} -> "
              f"{part.timestamp.max():%Y-%m-%d}  |  {len(ds):>6,} windows")

    train_loader, _, _ = make_dataloaders(region)
    X, y = next(iter(train_loader))
    print(f"\none batch: X {tuple(X.shape)} (64 windows, 168 hours, 10 features)  "
          f"y {tuple(y.shape)} (64 windows, 24 hours)")


if __name__ == "__main__":
    main()
