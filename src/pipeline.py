"""
Runs the other scripts in the right order, so one command does a whole job.

    python src/pipeline.py build      # rebuild everything from scratch (manual)
    python src/pipeline.py forecast   # daily: fresh data -> 24-hour forecast, no training
    python src/pipeline.py retrain    # weekly: retrain a region only if it needs it

Each script runs as its own process, so if one step fails the pipeline stops
there instead of carrying bad results into the next step.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

REGIONS = ["CAL", "TEX", "PJM", "MISO"]
SRC = Path(__file__).parent  # scripts are called by full path, so the working folder doesn't matter

DRIFT_EXIT_CODE = 10          # drift.py exits with this when the champion has gone stale
MAX_CHAMPION_AGE_DAYS = 30    # retrain a champion this old even if it still looks fine
FORECAST_DAYS = 21            # a daily forecast only needs the last 3 weeks of data


def run(script: str, *args: str, ok_codes: tuple = (0,)) -> int:
    """Run `python src/<script> <args>`. Stops the whole pipeline unless the exit code is in ok_codes."""
    print(f"\n{'-' * 60}\n> {script} {' '.join(args)}\n{'-' * 60}", flush=True)
    code = subprocess.run([sys.executable, str(SRC / script), *args]).returncode
    if code not in ok_codes:
        raise SystemExit(f"step failed: {script} {' '.join(args)} (exit code {code})")
    return code


def prep_data(region: str | None = None, days: int | None = None) -> None:
    """Download -> clean -> build model inputs. Optionally one region and/or only the last `days` days."""
    args = (["--region", region] if region else [])
    ingest_args = args + (["--days", str(days)] if days else [])
    run("ingestion.py", *ingest_args)
    run("validation.py", *args)
    run("features.py", *args)


def mode_build(regions: list[str]) -> None:
    """From scratch: per region, download all data, train both models, pick the champion, forecast."""
    for region in regions:
        prep_data(region)
        run("train.py", "--model", "transformer", "--region", region)
        run("train.py", "--model", "lstm", "--region", region)
        run("registry.py", "--region", region)
        run("inference.py", "--region", region)


def mode_forecast() -> None:
    """Daily: no training. Champions are already saved with their own scalers."""
    prep_data(days=FORECAST_DAYS)
    for region in REGIONS:
        run("inference.py", "--region", region)


def champion_age_days(region: str) -> float | None:
    """Days since this region's champion was registered, or None if there is none."""
    from mlflow.tracking import MlflowClient  # imported here so build/forecast don't need it up front

    from mlflow_setup import setup_mlflow
    from registry import get_champion

    setup_mlflow()
    champion = get_champion(MlflowClient(), f"voltcast-{region}")
    if champion is None:
        return None
    return (time.time() - champion.creation_timestamp / 1000) / 86400  # timestamp is in milliseconds


def mode_retrain(regions: list[str]) -> None:
    """
    Weekly check. Retrain a region if its champion has drifted, is older than 30 days,
    or doesn't exist. Only the Transformer is retrained: it has been the better model
    in the comparison, and the LSTM only exists for that comparison (build mode).
    """
    retrained = []
    for region in regions:
        # exit 0 = fine, exit 10 = drift. Any other exit code stops the pipeline (a real error).
        drifted = run("drift.py", "--region", region, ok_codes=(0, DRIFT_EXIT_CODE)) == DRIFT_EXIT_CODE

        age = champion_age_days(region)
        too_old = age is None or age > MAX_CHAMPION_AGE_DAYS

        if not (drifted or too_old):
            print(f"\n{region}: no drift, champion is {age:.0f} days old -> keep it")
            continue

        reason = "drift" if drifted else ("no champion yet" if age is None else f"champion is {age:.0f} days old")
        print(f"\n{region}: RETRAIN ({reason})")
        prep_data(region)  # full history, just this region
        run("train.py", "--model", "transformer", "--region", region)
        run("registry.py", "--region", region)
        run("inference.py", "--region", region)
        retrained.append(region)

    print(f"\nretrained: {', '.join(retrained)}" if retrained else "\nnothing needed retraining")


def main():
    parser = argparse.ArgumentParser(description="Voltcast pipeline.")
    parser.add_argument("mode", choices=["build", "forecast", "retrain"])
    parser.add_argument("--region", choices=REGIONS,
                        help="build/retrain only: just this region (the workflows run one job per region)")
    args = parser.parse_args()

    print(f"\n{'=' * 60}\n  VOLTCAST PIPELINE: {args.mode}\n{'=' * 60}")
    regions = [args.region] if args.region else REGIONS
    if args.mode == "retrain":
        mode_retrain(regions)
    elif args.mode == "build":
        mode_build(regions)
    else:
        mode_forecast()
    print(f"\n{'=' * 60}\n  DONE: {args.mode}\n{'=' * 60}")


if __name__ == "__main__":
    main()
