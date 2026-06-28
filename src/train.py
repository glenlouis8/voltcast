"""
src/train.py

Trains ONE model (lstm or transformer) on ONE region.
This is the hand-written training loop — no Trainer abstractions.

Run:
    python src/train.py --model transformer --country CAL
    python src/train.py --model lstm --country CAL

What happens, in order:
    1. Pick device (MPS on Mac M4)
    2. Load train/val dataloaders for the region
    3. Build the chosen model, move it to device
    4. Loop epochs:
         - train phase: weights update
         - val phase:   weights frozen, measure MAE
         - early stop if val MAE stops improving
    5. Save the best checkpoint to checkpoints/<region>_<model>.pt

This v1 has NO MLflow yet. We add experiment tracking after we
see the model actually learn (MAE drop epoch over epoch).
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import mlflow  # experiment tracking — logs params, metrics, artifacts

from dataset import load_dataloaders, FEATURE_COLS
from model import LSTMBaseline, TemporalTransformer
from mlflow_setup import setup_mlflow  # picks DagsHub cloud or local sqlite

# ── constants ────────────────────────────────────────────────────────────────

CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"

# Training hyperparameters (from CLAUDE.md training config table)
MAX_EPOCHS      = 100      # upper limit; early stopping usually ends sooner
PATIENCE        = 10       # stop if val MAE no improve for this many epochs
LEARNING_RATE   = 1e-3     # how big each weight nudge is
WEIGHT_DECAY    = 1e-4     # regularization — keeps weights from growing huge
GRAD_CLIP       = 1.0      # cap gradient size — prevents exploding gradients
BATCH_SIZE      = 64


# ── device pick ───────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """
    Pick the fastest available compute device.

    MPS = Metal Performance Shaders = Apple GPU on M-series Macs.
    Your M4 has this. It runs matrix math far faster than CPU.
    No CUDA branch — that is NVIDIA-only, you don't have it.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── model factory ─────────────────────────────────────────────────────────────

def build_model(model_name: str, input_dim: int) -> nn.Module:
    """
    Return a fresh model instance based on the --model flag.

    input_dim = number of features per timestep (13).
    Both models defined in src/model.py.
    """
    if model_name == "lstm":
        return LSTMBaseline(input_dim=input_dim)
    if model_name == "transformer":
        return TemporalTransformer(input_dim=input_dim)
    raise ValueError(f"Unknown model: {model_name}. Use 'lstm' or 'transformer'.")


# ── one training epoch ────────────────────────────────────────────────────────

def train_one_epoch(model, loader, loss_fn, optimizer, device) -> float:
    """
    Run ONE full pass over the training data. Weights update here.

    Returns the average training loss (MAE) over all batches.
    """
    # .train() puts model in training mode.
    # This turns dropout ON (randomly zero 10% of neurons) — only wanted while training.
    model.train()

    total_loss = 0.0
    n_batches  = 0

    # Each loop = one batch of 64 windows.
    # X shape: (64, 168, 13)   y shape: (64, 24)
    for X, y in loader:
        # Move this batch to the GPU (MPS). Model is already there.
        # Both must be on same device or PyTorch errors.
        X = X.to(device)
        y = y.to(device)

        # ── Step 1: FORWARD — model makes predictions ──
        preds = model(X)              # shape: (64, 24)

        # ── Step 2: LOSS — how wrong? ──
        # L1Loss = MAE = mean(|preds - y|). One number.
        loss = loss_fn(preds, y)

        # ── Step 3: ZERO GRAD — wipe old gradients ──
        # PyTorch accumulates gradients by default. Clear them each batch
        # or this batch's gradient piles onto the last one (wrong).
        optimizer.zero_grad()

        # ── Step 4: BACKWARD — compute gradients ──
        # Fills every weight's .grad with "which way to nudge to lower loss".
        loss.backward()

        # ── Step 5: CLIP — cap gradient size ──
        # If gradients get huge, weights jump wildly and training breaks.
        # This scales them down so their total norm <= GRAD_CLIP.
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        # ── Step 6: STEP — optimizer nudges every weight ──
        optimizer.step()

        # .item() pulls the loss number out of the tensor (off GPU, into Python float).
        total_loss += loss.item()
        n_batches  += 1

    return total_loss / n_batches


# ── one validation epoch ──────────────────────────────────────────────────────

def validate(model, loader, loss_fn, device) -> float:
    """
    Measure MAE on validation data. NO weight updates here.

    Returns average val MAE over all batches.
    """
    # .eval() puts model in evaluation mode — turns dropout OFF.
    # We want the full model, no random neuron dropping, for a clean measurement.
    model.eval()

    total_loss = 0.0
    n_batches  = 0

    # torch.no_grad() tells PyTorch: don't track gradients here.
    # We're only measuring, not learning. Saves memory and time.
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)

            preds = model(X)
            loss  = loss_fn(preds, y)

            total_loss += loss.item()
            n_batches  += 1

    return total_loss / n_batches


# ── main training routine ─────────────────────────────────────────────────────

def train(model_name: str, region: str) -> None:
    device = get_device()
    print(f"Device: {device}")
    print(f"Training {model_name} on {region}\n")

    # ── MLflow setup ──
    # Picks DagsHub cloud if creds are in .env, else local sqlite.
    setup_mlflow()
    # Group all runs for this region under one named experiment.
    # Opening mlflow ui later, you'll see "voltcast-CAL" with every run inside.
    mlflow.set_experiment(f"voltcast-{region}")

    # Load data. We only need train + val here. Test is untouched until evaluate.py.
    train_loader, val_loader, _ = load_dataloaders(region, batch_size=BATCH_SIZE)

    # Build model, move all its weights to the device.
    model = build_model(model_name, input_dim=len(FEATURE_COLS)).to(device)

    # Loss = MAE. Matches our primary eval metric.
    loss_fn = nn.L1Loss()

    # AdamW = optimizer. Reads gradients, updates weights. weight_decay = regularization.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # Scheduler slowly lowers learning rate in a cosine curve over MAX_EPOCHS.
    # Big steps early (learn fast), tiny steps late (settle precisely).
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    # ── early stopping state ──
    best_val_mae   = float("inf")  # best score so far; start at infinity so first epoch wins
    epochs_no_improve = 0          # counter — how many epochs since last improvement
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / f"{region}_{model_name}.pt"

    # start_run opens one tracked run. Everything logged inside the `with`
    # block belongs to this run. run_name shows up in the MLflow UI.
    with mlflow.start_run(run_name=f"{model_name}-{region}"):
        # ── log params (the settings — logged once, never change) ──
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
            # count trainable weights — nice to compare model sizes in the UI
            "num_params":    sum(p.numel() for p in model.parameters()),
        })

        # ── epoch loop ──
        for epoch in range(1, MAX_EPOCHS + 1):
            train_mae = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
            val_mae   = validate(model, val_loader, loss_fn, device)

            # Step the scheduler once per epoch — lowers the learning rate a notch.
            scheduler.step()

            # Note: train_mae and val_mae are in NORMALIZED units (z-score), not raw MW.
            # We convert back to MW later in evaluate.py. For now, lower = better is all we need.
            print(f"Epoch {epoch:3d} | train MAE {train_mae:.4f} | val MAE {val_mae:.4f}")

            # ── log metrics (numbers that change over time) ──
            # step=epoch makes MLflow plot these as a curve across epochs.
            mlflow.log_metric("train_mae", train_mae, step=epoch)
            mlflow.log_metric("val_mae",   val_mae,   step=epoch)

            # ── early stopping check ──
            if val_mae < best_val_mae:
                # New best. Save the model weights and reset the patience counter.
                best_val_mae = val_mae
                epochs_no_improve = 0
                torch.save(model.state_dict(), ckpt_path)
                print(f"          ↳ new best, saved → {ckpt_path.name}")
            else:
                # No improvement this epoch. Tick the counter.
                epochs_no_improve += 1
                if epochs_no_improve >= PATIENCE:
                    print(f"\nEarly stop: val MAE no improve for {PATIENCE} epochs.")
                    break

        # ── log the final summary + the best checkpoint file ──
        # best_val_mae as a single metric = easy to sort runs by in the UI.
        mlflow.log_metric("best_val_mae", best_val_mae)
        # log_artifact uploads the .pt file into this run's storage.
        mlflow.log_artifact(str(ckpt_path))

    print(f"\nDone. Best val MAE: {best_val_mae:.4f} (normalized)")
    print(f"Best weights saved at: {ckpt_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    # argparse reads command-line flags: --model and --country.
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
