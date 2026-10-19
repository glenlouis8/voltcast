"""
Pick the best trained model, file it in the MLflow Model Registry, and decide if
it becomes the "champion" (the model the daily forecast uses).

    python src/registry.py --region CAL
    python src/registry.py --region CAL --dry-run   # score and decide, but register nothing

How the decision works:
  1. Score both trained models on the VALIDATION year. The better one is the
     "challenger". (The test year is never used to decide anything.)
  2. No champion yet -> the challenger becomes champion.
  3. Otherwise score the current champion on the SAME validation year, and
     promote the challenger only if it is more than 1% better.
"""

import argparse
import json

import mlflow
import torch
from mlflow.tracking import MlflowClient
from torch.utils.data import DataLoader

from dataset import make_datasets, rescale_demand
from features import FEATURES_DIR
from mlflow_setup import setup_mlflow
from train import CHECKPOINT_DIR, build_model

PROMOTION_THRESHOLD = 0.01  # challenger must be more than 1% better than the champion
DEVICE = torch.device("cpu")  # the trusted reference, same as the CI runner


def score_mw(model, dataset, model_scaler: dict, data_scaler: dict) -> tuple[float, float]:
    """
    Average mistake of `model` on `dataset`, in MW. Returns (MAE in MW, average demand in MW).

    The features file is scaled with `data_scaler` (this week's average/spread),
    but a model must see numbers scaled the way IT was trained, `model_scaler`.
    So we convert: scaled -> real MW -> the model's own scaling. For a freshly
    trained model the two scalers are the same and this changes nothing.
    """
    model.eval()
    error_sum, count, demand_sum = 0.0, 0, 0.0
    with torch.no_grad():
        for X, y in DataLoader(dataset, batch_size=256, shuffle=False):
            X = rescale_demand(X, data_scaler, model_scaler)

            pred_mw = model(X) * model_scaler["std"] + model_scaler["mean"]
            actual_mw = y * data_scaler["std"] + data_scaler["mean"]

            error_sum += (pred_mw - actual_mw).abs().sum().item()
            demand_sum += actual_mw.sum().item()
            count += y.numel()
    return error_sum / count, demand_sum / count


def load_checkpoint(model_type: str, path) -> torch.nn.Module:
    model = build_model(model_type).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    return model


def find_registered_model(client: MlflowClient, name: str):
    """The registered model, or None if it doesn't exist. A search returns an empty
    list for 'not found', so we never have to guess what an error means."""
    found = client.search_registered_models(filter_string=f"name='{name}'")
    return found[0] if found else None


def get_champion(client: MlflowClient, name: str):
    """The current champion version, or None if there isn't one yet."""
    model = find_registered_model(client, name)
    if model is None or "champion" not in model.aliases:
        return None
    return client.get_model_version(name, model.aliases["champion"])


def ensure_registered_model(client: MlflowClient, name: str) -> None:
    if find_registered_model(client, name) is None:
        client.create_registered_model(name)


def run_registry(region: str, dry_run: bool) -> None:
    data_scaler = json.loads((FEATURES_DIR / "scalers.json").read_text())[region]
    _, val_ds, test_ds = make_datasets(region)

    # 1. score every trained model on the VALIDATION year and pick the best
    val_scores = {}
    for model_type in ("lstm", "transformer"):
        path = CHECKPOINT_DIR / f"{region}_{model_type}.pt"
        if path.exists():  # the weekly retrain only trains the transformer
            val_scores[model_type] = score_mw(load_checkpoint(model_type, path), val_ds, data_scaler, data_scaler)[0]
    if not val_scores:
        raise SystemExit(f"no trained models found in {CHECKPOINT_DIR} for {region}")

    for name, mae in val_scores.items():
        print(f"  {name:<12} validation MAE: {mae:,.0f} MW")
    challenger_type = min(val_scores, key=val_scores.get)
    challenger_val = val_scores[challenger_type]
    challenger_path = CHECKPOINT_DIR / f"{region}_{challenger_type}.pt"
    print(f"\nchallenger: {challenger_type} ({challenger_val:,.0f} MW on validation)")

    # the test year is only used to REPORT how good the challenger is, never to decide
    test_mae, test_demand = score_mw(load_checkpoint(challenger_type, challenger_path), test_ds, data_scaler, data_scaler)
    test_wape = test_mae / test_demand * 100
    print(f"  report card on test year: {test_mae:,.0f} MW ({test_wape:.2f}% of demand)")

    # 2. compare with the current champion, on the same validation year
    setup_mlflow()
    client = MlflowClient()
    registered_name = f"voltcast-{region}"
    champion = get_champion(client, registered_name)

    if champion is None:
        promote = True
        print("\nno champion yet -> the challenger becomes champion")
    else:
        champ_scaler = {"mean": float(champion.tags["scaler_mean"]), "std": float(champion.tags["scaler_std"])}
        local = mlflow.artifacts.download_artifacts(champion.source)
        champ_model = load_checkpoint(champion.tags["model_type"], f"{local}/{region}_{champion.tags['model_type']}.pt")
        champ_val = score_mw(champ_model, val_ds, champ_scaler, data_scaler)[0]
        improvement = (champ_val - challenger_val) / champ_val
        promote = improvement > PROMOTION_THRESHOLD
        print(f"\ncurrent champion v{champion.version} ({champion.tags['model_type']}): {champ_val:,.0f} MW on this validation year")
        print(f"challenger is {improvement * 100:+.2f}% vs champion (needs more than {PROMOTION_THRESHOLD * 100:.0f}%)")
        print("-> PROMOTE" if promote else "-> keep the champion")

    if dry_run:
        print("\n(dry run: nothing was registered)")
        return

    # 3. file the challenger in the registry, with its scaler, so inference can use it later
    ensure_registered_model(client, registered_name)
    mlflow.set_experiment(f"voltcast-{region}")
    with mlflow.start_run(run_name=f"registry-{region}") as run:
        mlflow.log_metrics({"val_mae_mw": challenger_val, "test_mae_mw": test_mae, "test_wape": test_wape})
        mlflow.log_param("model_type", challenger_type)
        mlflow.log_artifact(str(challenger_path), artifact_path="model")
        mlflow.log_dict(data_scaler, "model/scaler.json")
        version = client.create_model_version(
            name=registered_name, source=f"{run.info.artifact_uri}/model", run_id=run.info.run_id
        )

    for key, value in {
        "model_type": challenger_type, "val_mae_mw": challenger_val, "test_mae_mw": test_mae,
        "test_wape": test_wape, "scaler_mean": data_scaler["mean"], "scaler_std": data_scaler["std"],
    }.items():
        client.set_model_version_tag(registered_name, version.version, key, str(value))

    alias = "champion" if promote else "challenger"
    client.set_registered_model_alias(registered_name, alias, version.version)
    print(f"\nregistered v{version.version} as {alias}")


def main():
    parser = argparse.ArgumentParser(description="Champion/challenger promotion.")
    parser.add_argument("--region", choices=["CAL", "TEX", "PJM", "MISO"], default="CAL")
    parser.add_argument("--dry-run", action="store_true", help="score and decide, register nothing")
    args = parser.parse_args()
    run_registry(args.region, args.dry_run)


if __name__ == "__main__":
    main()
