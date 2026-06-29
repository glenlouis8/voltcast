"""
src/backfill_wape.py

One-time backfill. Existing champions were registered before we tagged WAPE,
so their model versions carry test_mae_mw but not test_wape. This script adds
the missing tag to each region's current champion.

WAPE = MAE / mean(actual load), both on the SAME test set. We already stored
MAE as a tag, so we only need the test set's mean load — no model prediction
needed here. Fast.

After running, re-run inference (python src/inference.py) so the new tag flows
into the S3 JSON the dashboard reads.

    python src/backfill_wape.py
"""

from mlflow.tracking import MlflowClient

from mlflow_setup import setup_mlflow
from dataset import load_datasets
from evaluate import get_scaler, gather_test_arrays, denormalize

REGIONS = ["CAL", "TEX", "PJM", "MISO"]


def backfill(region: str, client: MlflowClient) -> None:
    registered_name = f"voltcast-{region}"

    # Find the current champion version for this region.
    try:
        champion = client.get_model_version_by_alias(registered_name, "champion")
    except Exception:
        print(f"  {region}: no champion — skip")
        return

    if "test_wape" in champion.tags:
        print(f"  {region}: already has test_wape ({champion.tags['test_wape']}) — skip")
        return

    mae = float(champion.tags["test_mae_mw"])

    # Mean actual load on the test set (same data MAE was measured on).
    _, _, test_ds = load_datasets(region)
    mean, std = get_scaler(region)
    _, y_all = gather_test_arrays(test_ds)
    actuals_mw = denormalize(y_all, mean, std)
    mean_load = float(actuals_mw.mean())

    wape = mae / mean_load * 100
    client.set_model_version_tag(registered_name, champion.version, "test_wape", str(wape))
    print(f"  {region}: v{champion.version} MAE {mae:,.0f} / mean {mean_load:,.0f} → WAPE {wape:.2f}% ✓")


def main() -> None:
    setup_mlflow()
    client = MlflowClient()
    print("Backfilling test_wape on current champions:\n")
    for region in REGIONS:
        backfill(region, client)


if __name__ == "__main__":
    main()
