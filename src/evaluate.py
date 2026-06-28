"""
src/evaluate.py

The ablation table — the interview centerpiece.

Runs THREE methods on the untouched TEST set for one region and compares:
    1. Naive    — "next 24h = copy last 24h". Zero ML. The floor.
    2. LSTM     — loads checkpoints/<region>_lstm.pt
    3. Transformer — loads checkpoints/<region>_transformer.pt

Reports MAE (in real MW) and MAPE (% error) for each.

Run:
    python src/evaluate.py --country CAL

Rules honored:
    - Test set is touched ONLY here, never during training.
    - Scaler (mean/std) is recomputed from TRAIN rows only — no leakage.
    - Predictions are converted back to real MW before scoring, so the
      numbers are human-readable megawatts, not z-scores.
"""

import argparse
from pathlib import Path

import torch
import numpy as np
import pandas as pd

from dataset import load_datasets, FEATURE_COLS, TRAIN_RATIO, SEQ_LEN, FORECAST_HORIZON
from model import LSTMBaseline, TemporalTransformer

# ── constants ────────────────────────────────────────────────────────────────

RAW_DIR        = Path(__file__).parent.parent / "data" / "raw"
CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"

# Which column in FEATURE_COLS is load_mw? Needed for the Naive baseline
# and to know which channel of the input is the load signal.
LOAD_IDX = FEATURE_COLS.index("load_mw")


# ── scaler: recompute exactly like features.py ────────────────────────────────

def get_scaler(region: str) -> tuple[float, float]:
    """
    Recompute (train_mean, train_std) of load_mw from the raw parquet,
    using the SAME logic features.py used:
        - sort by timestamp
        - train_size = first 70% of rows
        - mean/std over load_mw of those train rows only

    Why recompute instead of reading saved values?
        features.py stored them in df.attrs, but parquet often drops attrs.
        Recomputing from raw is guaranteed consistent and leakage-safe
        (we only ever look at training rows).
    """
    df = pd.read_parquet(RAW_DIR / f"{region}.parquet")
    df = df.sort_values("timestamp").reset_index(drop=True)

    train_size = int(len(df) * TRAIN_RATIO)
    train_mean = df["load_mw"].iloc[:train_size].mean()
    train_std  = df["load_mw"].iloc[:train_size].std()
    return float(train_mean), float(train_std)


def denormalize(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    """
    Undo z-score: real_MW = normalized * std + mean.
    Turns model outputs (small numbers) back into megawatts.
    """
    return x * std + mean


# ── metrics ───────────────────────────────────────────────────────────────────

def mae_mw(preds: np.ndarray, actuals: np.ndarray) -> float:
    """Mean Absolute Error in MW. Average of |pred - actual|."""
    return float(np.mean(np.abs(preds - actuals)))


def mape(preds: np.ndarray, actuals: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error.
    Average of |pred - actual| / actual, as a percent.
    Tells you typical % miss — easy to explain in interviews.
    """
    return float(np.mean(np.abs(preds - actuals) / actuals) * 100)


# ── collect every test window into arrays ─────────────────────────────────────

def gather_test_arrays(test_ds) -> tuple[np.ndarray, np.ndarray]:
    """
    Stack the whole test set into two numpy arrays:
        X_all: (num_windows, 168, 13)  — all inputs
        y_all: (num_windows, 24)       — all targets (normalized load)

    We build these once and reuse for all three methods, so every
    method scores on the exact same windows (fair comparison).
    """
    X_list, y_list = [], []
    for i in range(len(test_ds)):
        X, y = test_ds[i]            # tensors
        X_list.append(X.numpy())
        y_list.append(y.numpy())
    X_all = np.stack(X_list)         # (N, 168, 13)
    y_all = np.stack(y_list)         # (N, 24)
    return X_all, y_all


# ── method 1: Naive baseline ──────────────────────────────────────────────────

def predict_naive(X_all: np.ndarray) -> np.ndarray:
    """
    Naive forecast: next 24 hours = the LAST 24 hours of the input window.
    "Tomorrow looks like today." No model, no training.

    X_all shape: (N, 168, 13). We grab the load_mw channel (LOAD_IDX)
    of the final 24 timesteps: rows 144..167.

    Returns (N, 24) — still normalized (we denormalize later, same as models).
    """
    return X_all[:, SEQ_LEN - FORECAST_HORIZON:, LOAD_IDX]   # (N, 24)


# ── methods 2 & 3: neural models ──────────────────────────────────────────────

def predict_model(model, X_all: np.ndarray, device) -> np.ndarray:
    """
    Run a trained model over all test windows, return predictions (N, 24).

    eval() + no_grad() — pure inference, no weight changes, no gradient memory.
    We process in chunks of 256 so we don't blow up memory on the GPU.
    """
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(X_all), 256):
            chunk = X_all[start:start + 256]                       # (<=256, 168, 13)
            X = torch.tensor(chunk, dtype=torch.float32).to(device)
            out = model(X)                                         # (<=256, 24)
            preds.append(out.cpu().numpy())                       # back to CPU/numpy
    return np.concatenate(preds, axis=0)                          # (N, 24)


def load_checkpoint(model_name: str, region: str, device) -> torch.nn.Module:
    """Build the right architecture and load its saved weights."""
    if model_name == "lstm":
        model = LSTMBaseline(input_dim=len(FEATURE_COLS))
    else:
        model = TemporalTransformer(input_dim=len(FEATURE_COLS))

    ckpt = CHECKPOINT_DIR / f"{region}_{model_name}.pt"
    # map_location=device loads weights straight onto MPS/CPU.
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model.to(device)


# ── main ──────────────────────────────────────────────────────────────────────

def evaluate(region: str) -> None:
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Device: {device}")
    print(f"Evaluating on TEST set: {region}\n")

    # Test set — last 15% of data, untouched until this moment.
    _, _, test_ds = load_datasets(region)
    print(f"Test windows: {len(test_ds):,}")

    # Scaler to convert predictions back to real MW.
    mean, std = get_scaler(region)
    print(f"Scaler: mean={mean:,.0f} MW, std={std:,.0f} MW\n")

    # Stack all test windows once.
    X_all, y_all = gather_test_arrays(test_ds)

    # Ground truth in real MW (same for every method).
    actuals_mw = denormalize(y_all, mean, std)

    # ── run all three methods ──
    results = {}

    # 1. Naive
    naive_norm = predict_naive(X_all)
    naive_mw   = denormalize(naive_norm, mean, std)
    results["Naive"] = (mae_mw(naive_mw, actuals_mw), mape(naive_mw, actuals_mw))

    # 2. LSTM
    lstm = load_checkpoint("lstm", region, device)
    lstm_mw = denormalize(predict_model(lstm, X_all, device), mean, std)
    results["LSTM"] = (mae_mw(lstm_mw, actuals_mw), mape(lstm_mw, actuals_mw))

    # 3. Transformer
    tf = load_checkpoint("transformer", region, device)
    tf_mw = denormalize(predict_model(tf, X_all, device), mean, std)
    results["Transformer"] = (mae_mw(tf_mw, actuals_mw), mape(tf_mw, actuals_mw))

    # ── ablation table ──
    print(f"{'='*46}")
    print(f"  ABLATION — {region} (test set)")
    print(f"{'='*46}")
    print(f"  {'Method':<14}{'MAE (MW)':>12}{'MAPE':>10}")
    print(f"  {'-'*40}")
    for name, (m, p) in results.items():
        print(f"  {name:<14}{m:>12,.0f}{p:>9.2f}%")
    print(f"{'='*46}")

    # ── verdict ──
    best = min(results, key=lambda k: results[k][0])
    naive_mae = results["Naive"][0]
    best_mae  = results[best][0]
    lift = (naive_mae - best_mae) / naive_mae * 100
    print(f"\n  Best: {best}  |  {lift:.1f}% better than Naive baseline")
    if best == "Naive":
        print("  WARNING: models failed to beat copy-yesterday. Something is wrong.")


def main():
    parser = argparse.ArgumentParser(description="Ablation: Naive vs LSTM vs Transformer.")
    parser.add_argument("--country", default="CAL", help="Region: CAL, TEX, PJM, MISO.")
    args = parser.parse_args()
    evaluate(args.country)


if __name__ == "__main__":
    main()
