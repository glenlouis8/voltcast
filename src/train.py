"""
Train one model on one region with a hand-written training loop.

    python src/train.py --model transformer --region CAL
    python src/train.py --model lstm --region CAL

Each round (epoch): the model sees all training windows, then we check it on the
validation year. We keep the best version and stop early if validation stops
improving. The test year is never touched here.
"""

import argparse
import contextlib
import json
from pathlib import Path

import mlflow
import torch
import torch.nn as nn

from dataset import FEATURE_COLS, make_dataloaders
from features import FEATURES_DIR
from mlflow_setup import setup_mlflow
from model import LSTMBaseline, TemporalTransformer

CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"

MAX_EPOCHS = 40       # at most 40 rounds over the training data
PATIENCE = 8          # stop if validation hasn't improved for 8 rounds in a row
LEARNING_RATE = 1e-3  # how big each nudge to the model's numbers is
WEIGHT_DECAY = 1e-4   # a small pull that stops the numbers growing too large
GRAD_CLIP = 1.0       # cap on how big one nudge can be (prevents blow-ups)
BATCH_SIZE = 64
SEED = 42


def get_device() -> torch.device:
    # "mps" is the GPU in Apple chips; otherwise use the normal CPU
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def build_model(name: str) -> nn.Module:
    if name == "lstm":
        return LSTMBaseline(input_dim=len(FEATURE_COLS))
    if name == "transformer":
        return TemporalTransformer(input_dim=len(FEATURE_COLS))
    raise ValueError(f"unknown model: {name}")


def run_epoch(model, loader, loss_fn, device, optimizer=None) -> float:
    """
    One pass over a loader. With an optimizer = training (the model learns).
    Without = checking (nothing changes). Returns the average error (MAE).
    """
    training = optimizer is not None
    model.train(training)  # training mode turns dropout on, checking turns it off

    total_error, total_count = 0.0, 0
    with torch.set_grad_enabled(training):  # no learning needed while checking
        for X, y in loader:
            X, y = X.to(device), y.to(device)

            guess = model(X)             # 1. the model guesses the next 24 hours
            loss = loss_fn(guess, y)     # 2. how far off was it? (MAE)

            if training:
                optimizer.zero_grad()    # 3. forget the previous nudge
                loss.backward()          # 4. work out which way to nudge each number
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()         # 5. nudge the numbers

            total_error += loss.item() * len(X)  # weight by batch size (last batch is smaller)
            total_count += len(X)

    return total_error / total_count


def train(model_name: str, region: str, max_epochs: int, use_mlflow: bool) -> None:
    torch.manual_seed(SEED)  # same starting numbers every run, so runs are comparable
    device = get_device()
    print(f"device: {device} | training {model_name} on {region}")

    # the error is in scaled units; multiply by the spread to read it in MW
    spread = json.loads((FEATURES_DIR / "scalers.json").read_text())[region]["std"]

    train_loader, val_loader, _ = make_dataloaders(region, BATCH_SIZE)  # test loader unused

    model = build_model(model_name).to(device)
    loss_fn = nn.L1Loss()  # MAE: the average size of the mistakes
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # learning rate shrinks smoothly over the planned rounds: big steps early, tiny at the end
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    checkpoint = CHECKPOINT_DIR / f"{region}_{model_name}.pt"

    if use_mlflow:
        setup_mlflow()
        mlflow.set_experiment(f"voltcast-{region}")
        run = mlflow.start_run(run_name=f"{model_name}-{region}")
        mlflow.log_params({
            "model": model_name, "region": region, "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE, "max_epochs": max_epochs,
            "patience": PATIENCE, "grad_clip": GRAD_CLIP, "num_features": len(FEATURE_COLS),
            "num_params": sum(p.numel() for p in model.parameters()),
        })
    else:
        run = contextlib.nullcontext()

    best_val, rounds_without_gain = float("inf"), 0
    with run:
        for epoch in range(1, max_epochs + 1):
            train_mae = run_epoch(model, train_loader, loss_fn, device, optimizer)
            val_mae = run_epoch(model, val_loader, loss_fn, device)
            scheduler.step()

            print(f"epoch {epoch:2d} | train MAE {train_mae:.4f} | val MAE {val_mae:.4f} "
                  f"(= {val_mae * spread:,.0f} MW)")
            if use_mlflow:
                mlflow.log_metric("train_mae", train_mae, step=epoch)
                mlflow.log_metric("val_mae", val_mae, step=epoch)
                mlflow.log_metric("val_mae_mw", val_mae * spread, step=epoch)

            if val_mae < best_val:  # new best: save it
                best_val, rounds_without_gain = val_mae, 0
                torch.save(model.state_dict(), checkpoint)
            else:
                rounds_without_gain += 1
                if rounds_without_gain >= PATIENCE:
                    print(f"stopping early: no improvement for {PATIENCE} rounds")
                    break

        if use_mlflow:
            mlflow.log_metric("best_val_mae", best_val)
            mlflow.log_metric("best_val_mae_mw", best_val * spread)
            mlflow.log_artifact(str(checkpoint))

    print(f"\nbest validation MAE: {best_val:.4f} = {best_val * spread:,.0f} MW")
    print(f"best model saved -> {checkpoint}")


def main():
    parser = argparse.ArgumentParser(description="Train a forecasting model.")
    parser.add_argument("--model", choices=["lstm", "transformer"], required=True)
    parser.add_argument("--region", choices=["CAL", "TEX", "PJM", "MISO"], default="CAL")
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS, help="max rounds (default 40)")
    parser.add_argument("--no-mlflow", action="store_true", help="skip experiment tracking")
    args = parser.parse_args()
    train(args.model, args.region, args.epochs, use_mlflow=not args.no_mlflow)


if __name__ == "__main__":
    main()
