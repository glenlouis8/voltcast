"""
src/registry.py

Champion / Challenger promotion using the MLflow Model Registry.

Run:
    python src/registry.py --country CAL

What it does:
    1. Evaluate BOTH trained models (lstm, transformer) on the TEST set.
    2. Challenger = whichever has the lower test MAE (in MW).
    3. Register that model as a new version under name "voltcast-<region>".
    4. Promote logic:
         - No champion yet      → challenger becomes champion.
         - Champion exists       → challenger must beat it by >1% to take over.
                                   Otherwise it stays "challenger".

Aliases (movable labels on a version):
    champion   = the model production/inference should serve
    challenger = newest contender, not yet proven better

We decide on TEST MAE (the untouched, honest grade), not val.
"""

import argparse
from pathlib import Path

import torch
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient

# Reuse the evaluation machinery we already wrote — no duplication.
from evaluate import (
    get_scaler, denormalize, mae_mw,
    gather_test_arrays, predict_model, load_checkpoint,
)
from dataset import load_datasets
from mlflow_setup import setup_mlflow  # same destination as train.py

CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"

# How much better (fraction) a challenger must be to dethrone the champion.
PROMOTION_THRESHOLD = 0.01   # 1%


# ── evaluate one model on the test set → test MAE in MW ───────────────────────

def test_mae_for(model_name: str, region: str, X_all, actuals_mw, mean, std, device) -> float:
    """
    Load a checkpoint, predict on all test windows, return MAE in real MW.

    X_all, actuals_mw, mean, std are passed in (computed once) so both
    models score on the exact same test windows — a fair comparison.
    """
    model = load_checkpoint(model_name, region, device)
    preds_mw = denormalize(predict_model(model, X_all, device), mean, std)
    return mae_mw(preds_mw, actuals_mw)


# ── main promotion routine ────────────────────────────────────────────────────

def run_registry(region: str) -> None:
    setup_mlflow()
    client = MlflowClient()

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Device: {device}")
    print(f"Champion/Challenger check: {region}\n")

    # ── 1. Build the test set once ──
    _, _, test_ds = load_datasets(region)
    mean, std = get_scaler(region)
    X_all, y_all = gather_test_arrays(test_ds)
    actuals_mw = denormalize(y_all, mean, std)

    # ── 2. Score both models, pick the challenger (lower MAE wins) ──
    scores = {
        "lstm":        test_mae_for("lstm",        region, X_all, actuals_mw, mean, std, device),
        "transformer": test_mae_for("transformer", region, X_all, actuals_mw, mean, std, device),
    }
    for name, mae in scores.items():
        print(f"  {name:<12} test MAE: {mae:,.0f} MW")

    challenger_name = min(scores, key=scores.get)   # key with smallest MAE
    challenger_mae  = scores[challenger_name]
    print(f"\n  Challenger: {challenger_name} ({challenger_mae:,.0f} MW)")

    # ── 3. Register the challenger as a new model version ──
    registered_name = f"voltcast-{region}"

    # Make sure the registered model NAME exists (create once, ignore if already there).
    try:
        client.create_registered_model(registered_name)
        print(f"  Created registered model: {registered_name}")
    except Exception:
        pass  # already exists — fine

    # Log the challenger's checkpoint as an artifact inside a fresh run, then
    # register THAT artifact as a new version. A version must point at a stored
    # artifact (model_uri), so we log it here first.
    ckpt_path = CHECKPOINT_DIR / f"{region}_{challenger_name}.pt"
    with mlflow.start_run(run_name=f"registry-{region}") as run:
        mlflow.log_metric("test_mae_mw", challenger_mae)
        mlflow.log_param("challenger_model", challenger_name)
        mlflow.log_artifact(str(ckpt_path), artifact_path="model")

        # Point the version directly at the stored artifact folder.
        # We use client.create_model_version (not mlflow.register_model) because
        # newer MLflow's register_model expects a "logged model" entity that a
        # plain log_artifact doesn't create. create_model_version takes a raw
        # artifact source path, which is exactly what we have.
        source = f"{run.info.artifact_uri}/model"
        version = client.create_model_version(
            name=registered_name,
            source=source,
            run_id=run.info.run_id,
        )
        print(f"  Registered as version {version.version}")

    # Store the test MAE on the version itself (as a tag) so we can compare
    # champion vs challenger later without re-evaluating.
    client.set_model_version_tag(registered_name, version.version, "test_mae_mw", str(challenger_mae))
    client.set_model_version_tag(registered_name, version.version, "model_type", challenger_name)

    # ── 4. Promotion decision ──
    # Try to read the current champion alias. If none exists, this throws.
    try:
        champion = client.get_model_version_by_alias(registered_name, "champion")
        champion_mae = float(champion.tags["test_mae_mw"])
        print(f"\n  Current champion: v{champion.version} ({champion_mae:,.0f} MW)")

        # Improvement = how much lower the challenger's MAE is, as a fraction.
        improvement = (champion_mae - challenger_mae) / champion_mae
        print(f"  Challenger is {improvement*100:+.2f}% vs champion")

        if improvement > PROMOTION_THRESHOLD:
            client.set_registered_model_alias(registered_name, "champion", version.version)
            print(f"  PROMOTED → v{version.version} is the new champion "
                  f"(beat by >{PROMOTION_THRESHOLD*100:.0f}%)")
        else:
            client.set_registered_model_alias(registered_name, "challenger", version.version)
            print(f"  Not enough improvement. v{version.version} stays challenger; "
                  f"champion unchanged.")

    except Exception:
        # No champion yet — first model in. Crown it.
        client.set_registered_model_alias(registered_name, "champion", version.version)
        print(f"\n  No existing champion. v{version.version} crowned as first champion.")

    print("\nRegistry update complete.")


def main():
    parser = argparse.ArgumentParser(description="Champion/Challenger promotion.")
    parser.add_argument("--country", default="CAL", help="Region: CAL, TEX, PJM, MISO.")
    args = parser.parse_args()
    run_registry(args.country)


if __name__ == "__main__":
    main()
