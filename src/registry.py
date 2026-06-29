"""
Champion/Challenger promotion via the MLflow Model Registry.

    python src/registry.py --country CAL

Scores both trained models on the untouched test set, registers the lower-MAE
one as a new version of "voltcast-<region>", and promotes it to the "champion"
alias only if it beats the current champion by >1% (first model in wins by
default). Aliases: champion = served by inference, challenger = unproven contender.
"""

import argparse
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

from evaluate import (
    get_scaler, denormalize, mae_mw,
    gather_test_arrays, predict_model, load_checkpoint,
)
from dataset import load_datasets, TRAIN_RATIO
from mlflow_setup import setup_mlflow
from storage import save_reference

CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"
RAW_DIR        = Path(__file__).parent.parent / "data" / "raw"

# columns drift.py monitors — the reference snapshot must carry these
REFERENCE_COLS = ["load_mw", "rolling_mean_24"]

# how much lower a challenger's MAE must be to dethrone the champion
PROMOTION_THRESHOLD = 0.01   # 1%


def build_reference(region: str) -> pd.DataFrame:
    """
    Reference dataset = the training slice the champion learned from, with the
    columns drift.py monitors. Drift compares fresh data against this.
    """
    df = pd.read_parquet(RAW_DIR / f"{region}.parquet")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["rolling_mean_24"] = df["load_mw"].rolling(24, min_periods=1).mean()

    train_end = int(len(df) * TRAIN_RATIO)  # train slice only, never val/test
    return df.iloc[:train_end][REFERENCE_COLS].copy()


def test_mae_for(model_name: str, region: str, X_all, actuals_mw, mean, std, device) -> float:
    """
    Load a checkpoint and return its test MAE in MW. X_all/actuals_mw/mean/std
    are passed in (computed once) so both models score on identical windows.
    """
    model = load_checkpoint(model_name, region, device)
    preds_mw = denormalize(predict_model(model, X_all, device), mean, std)
    return mae_mw(preds_mw, actuals_mw)


def run_registry(region: str) -> None:
    setup_mlflow()
    client = MlflowClient()

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Device: {device}")
    print(f"Champion/Challenger check: {region}\n")

    # build the test set once, share across both models
    _, _, test_ds = load_datasets(region)
    mean, std = get_scaler(region)
    X_all, y_all = gather_test_arrays(test_ds)
    actuals_mw = denormalize(y_all, mean, std)

    scores = {
        "lstm":        test_mae_for("lstm",        region, X_all, actuals_mw, mean, std, device),
        "transformer": test_mae_for("transformer", region, X_all, actuals_mw, mean, std, device),
    }
    for name, mae in scores.items():
        print(f"  {name:<12} test MAE: {mae:,.0f} MW")

    challenger_name = min(scores, key=scores.get)
    challenger_mae  = scores[challenger_name]

    # WAPE = MAE / mean(test load) as a %. Same data top and bottom, so it's an
    # honest error rate that compares fairly across differently-sized regions.
    challenger_wape = challenger_mae / actuals_mw.mean() * 100
    print(f"\n  Challenger: {challenger_name} ({challenger_mae:,.0f} MW | WAPE {challenger_wape:.2f}%)")

    registered_name = f"voltcast-{region}"

    try:
        client.create_registered_model(registered_name)
        print(f"  Created registered model: {registered_name}")
    except Exception:
        pass  # already exists

    # A version must point at a stored artifact, so log the checkpoint first,
    # then register that artifact folder as a new version.
    ckpt_path = CHECKPOINT_DIR / f"{region}_{challenger_name}.pt"
    with mlflow.start_run(run_name=f"registry-{region}") as run:
        mlflow.log_metric("test_mae_mw", challenger_mae)
        mlflow.log_metric("test_wape", challenger_wape)
        mlflow.log_param("challenger_model", challenger_name)
        mlflow.log_artifact(str(ckpt_path), artifact_path="model")

        # create_model_version (not register_model): newer MLflow's register_model
        # expects a "logged model" entity that plain log_artifact doesn't create;
        # create_model_version takes the raw artifact source path we have.
        source = f"{run.info.artifact_uri}/model"
        version = client.create_model_version(
            name=registered_name,
            source=source,
            run_id=run.info.run_id,
        )
        print(f"  Registered as version {version.version}")

    # tag the version so champion vs challenger compares without re-evaluating
    client.set_model_version_tag(registered_name, version.version, "test_mae_mw", str(challenger_mae))
    client.set_model_version_tag(registered_name, version.version, "test_wape", str(challenger_wape))
    client.set_model_version_tag(registered_name, version.version, "model_type", challenger_name)

    # promotion. if the champion changes we refresh the drift reference snapshot
    # so it matches whoever is champion.
    became_champion = False
    try:
        champion = client.get_model_version_by_alias(registered_name, "champion")
        champion_mae = float(champion.tags["test_mae_mw"])
        print(f"\n  Current champion: v{champion.version} ({champion_mae:,.0f} MW)")

        improvement = (champion_mae - challenger_mae) / champion_mae
        print(f"  Challenger is {improvement*100:+.2f}% vs champion")

        if improvement > PROMOTION_THRESHOLD:
            client.set_registered_model_alias(registered_name, "champion", version.version)
            became_champion = True
            print(f"  PROMOTED -> v{version.version} is the new champion "
                  f"(beat by >{PROMOTION_THRESHOLD*100:.0f}%)")
        else:
            client.set_registered_model_alias(registered_name, "challenger", version.version)
            print(f"  Not enough improvement. v{version.version} stays challenger.")

    except Exception:
        # no champion yet — first model in, crown it
        client.set_registered_model_alias(registered_name, "champion", version.version)
        became_champion = True
        print(f"\n  No existing champion. v{version.version} crowned as first champion.")

    if became_champion:
        ref = build_reference(region)
        loc = save_reference(region, ref)
        print(f"  Reference snapshot saved -> {loc}")

    print("\nRegistry update complete.")


def main():
    parser = argparse.ArgumentParser(description="Champion/Challenger promotion.")
    parser.add_argument("--country", default="CAL", help="Region: CAL, TEX, PJM, MISO.")
    args = parser.parse_args()
    run_registry(args.country)


if __name__ == "__main__":
    main()
