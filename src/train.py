"""
Hand-written training loop (no Trainer abstractions) for one model on one
region, with MLflow tracking and early stopping.

    python src/train.py --model transformer --country CAL
    python src/train.py --model lstm --country CAL

Loads train/val loaders, trains up to MAX_EPOCHS, stops early when val MAE
plateaus, and saves the best checkpoint to checkpoints/<region>_<model>.pt.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import mlflow

from dataset import load_dataloaders, FEATURE_COLS
from model import LSTMBaseline, TemporalTransformer
from mlflow_setup import setup_mlflow

CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"

# hyperparameters (see CLAUDE.md training config table)
MAX_EPOCHS    = 100
PATIENCE      = 10        # stop if val MAE doesn't improve for this many epochs
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
GRAD_CLIP     = 1.0       # cap gradient norm to prevent exploding gradients
BATCH_SIZE    = 64


def get_device() -> torch.device:
    # MPS = Apple GPU on M-series Macs. No CUDA branch (NVIDIA-only).
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(model_name: str, input_dim: int) -> nn.Module:
    if model_name == "lstm":
        return LSTMBaseline(input_dim=input_dim)
    if model_name == "transformer":
        return TemporalTransformer(input_dim=input_dim)
    raise ValueError(f"Unknown model: {model_name}. Use 'lstm' or 'transformer'.")


def train_one_epoch(model, loader, loss_fn, optimizer, device) -> float:
    """One full pass over the training data. Returns mean MAE over batches."""
    model.train()  # dropout on
    total_loss = 0.0
    n_batches = 0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        preds = model(X)
        loss = loss_fn(preds, y)

        optimizer.zero_grad()  # gradients accumulate by default; clear each batch
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def validate(model, loader, loss_fn, device) -> float:
    """Measure val MAE with weights frozen. Returns mean MAE over batches."""
    model.eval()  # dropout off
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)
            loss = loss_fn(model(X), y)
            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches


def train(model_name: str, region: str) -> None:
    device = get_device()
    print(f"Device: {device}")
    print(f"Training {model_name} on {region}\n")

    setup_mlflow()
    mlflow.set_experiment(f"voltcast-{region}")

    # train + val only; test stays untouched until evaluate.py
    train_loader, val_loader, _ = load_dataloaders(region, batch_size=BATCH_SIZE)

    model = build_model(model_name, input_dim=len(FEATURE_COLS)).to(device)
    loss_fn = nn.L1Loss()  # MAE, matches the eval metric
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    # cosine decay: big LR steps early, tiny steps late
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    best_val_mae = float("inf")
    epochs_no_improve = 0
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / f"{region}_{model_name}.pt"

    with mlflow.start_run(run_name=f"{model_name}-{region}"):
        mlflow.log_params({
            "model":         model_name,
            "region":        region,
            "learning_rate": LEARNING_RATE,
            "weight_decay":  WEIGHT_DECAY,
            "batch_size":    BATCH_SIZE,
            "max_epochs":    MAX_EPOCHS,
            "patience":      PATIENCE,
            "grad_clip":     GRAD_CLIP,
            "num_features":  len(FEATURE_COLS),
            "num_params":    sum(p.numel() for p in model.parameters()),
        })

        for epoch in range(1, MAX_EPOCHS + 1):
            train_mae = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
            val_mae = validate(model, val_loader, loss_fn, device)
            scheduler.step()

            # both MAEs are in normalized (z-score) units; converted to MW in evaluate.py
            print(f"Epoch {epoch:3d} | train MAE {train_mae:.4f} | val MAE {val_mae:.4f}")
            mlflow.log_metric("train_mae", train_mae, step=epoch)
            mlflow.log_metric("val_mae", val_mae, step=epoch)

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                epochs_no_improve = 0
                torch.save(model.state_dict(), ckpt_path)
                print(f"          new best, saved -> {ckpt_path.name}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= PATIENCE:
                    print(f"\nEarly stop: val MAE flat for {PATIENCE} epochs.")
                    break

        mlflow.log_metric("best_val_mae", best_val_mae)
        mlflow.log_artifact(str(ckpt_path))

    print(f"\nDone. Best val MAE: {best_val_mae:.4f} (normalized)")
    print(f"Best weights saved at: {ckpt_path}")


def main():
    parser = argparse.ArgumentParser(description="Train a forecasting model.")
    parser.add_argument(
        "--model",
        choices=["lstm", "transformer"],
        required=True,
        help="Which architecture to train.",
    )
    parser.add_argument(
        "--country",
        default="CAL",
        help="Region code: CAL, TEX, PJM, or MISO.",
    )
    args = parser.parse_args()
    train(args.model, args.country)


if __name__ == "__main__":
    main()
