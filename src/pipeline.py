"""
Orchestrator that runs the existing scripts in order, so one command does the
work of ten (locally or in a GitHub Action).

    python src/pipeline.py build      # full rebuild of all regions
    python src/pipeline.py forecast   # refresh data -> predict next 24h
    python src/pipeline.py retrain    # drift/staleness check -> retrain if needed

Scripts run as subprocesses (not imports) so each keeps its own runnable main()
and a failed step halts the pipeline instead of poisoning later steps.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

REGIONS = ["CAL", "TEX", "PJM", "MISO"]

# Retrain a champion older than this even if no drift fires — the K-S test can
# miss slow, gradual shifts, so a hard age cap stops the model serving forever
# on ancient training data.
MAX_CHAMPION_AGE_DAYS = 30

# call sibling scripts by absolute path so cwd doesn't matter
SRC = Path(__file__).parent


def run(script: str, *args: str, allow_fail: bool = False) -> int:
    """
    Run `python src/<script> <args>` in this file's venv (via sys.executable).
    Non-zero exit raises unless allow_fail — used for drift.py, whose exit 1
    means "drift detected" (a signal, not an error).
    """
    cmd = [sys.executable, str(SRC / script), *args]
    pretty = " ".join([script, *args])
    print(f"\n{'-'*60}\n> {pretty}\n{'-'*60}")

    result = subprocess.run(cmd)
    if result.returncode != 0 and not allow_fail:
        raise SystemExit(f"step failed: {pretty} (exit {result.returncode})")
    return result.returncode


def prep_data() -> None:
    """Pull fresh data, validate, build features. Each loops all regions itself."""
    run("ingestion.py")
    run("validation.py")
    run("features.py")


def mode_build() -> None:
    """Full rebuild: train both models, crown a champion, forecast — per region."""
    prep_data()
    for r in REGIONS:
        run("train.py", "--model", "transformer", "--country", r)
        run("train.py", "--model", "lstm",        "--country", r)
        run("registry.py", "--country", r)   # picks best, crowns champion, saves drift reference
        run("inference.py", "--country", r)


def mode_forecast() -> None:
    """Daily: refresh data, then forecast next 24h per region. No training."""
    prep_data()
    for r in REGIONS:
        run("inference.py", "--country", r)


def champion_age_days(region: str) -> float | None:
    """
    Days since this region's champion was registered, or None if there is no
    champion. mlflow imported lazily so the forecast path doesn't pay for it.
    """
    from mlflow.tracking import MlflowClient
    from mlflow_setup import setup_mlflow

    setup_mlflow()
    client = MlflowClient()
    try:
        champ = client.get_model_version_by_alias(f"voltcast-{region}", "champion")
    except Exception:
        return None

    # creation_timestamp is epoch ms; time.time() is epoch seconds
    return (time.time() - champ.creation_timestamp / 1000) / 86400


def mode_retrain() -> None:
    """
    Weekly watchdog. Retrain a region if it drifted OR its champion is older than
    MAX_CHAMPION_AGE_DAYS (or there's no champion). drift.py pulls its own recent
    window, so we only prep_data() for regions that actually need retraining.
    """
    retrained = []
    for r in REGIONS:
        drifted = run("drift.py", "--country", r, allow_fail=True) != 0  # exit 1 = drift

        age = champion_age_days(r)
        if age is None:
            stale, why_age = True, "no champion yet"
        else:
            stale = age > MAX_CHAMPION_AGE_DAYS
            why_age = f"champion {age:.0f}d old (cap {MAX_CHAMPION_AGE_DAYS}d)"

        if not (drifted or stale):
            print(f"  {r}: no drift, {why_age} -> champion kept.")
            continue

        reason = "drift" if drifted else why_age
        print(f"  {r}: RETRAIN ({reason}).")
        prep_data()
        run("train.py", "--model", "transformer", "--country", r)
        run("train.py", "--model", "lstm",        "--country", r)
        run("registry.py", "--country", r)
        run("inference.py", "--country", r)
        retrained.append(r)

    print(f"\nRetrained: {', '.join(retrained)}" if retrained
          else "\nNo region drifted or stale. Nothing retrained.")


def main():
    parser = argparse.ArgumentParser(description="VoltCast pipeline orchestrator.")
    parser.add_argument(
        "mode",
        choices=["build", "forecast", "retrain"],
        help="build = full rebuild | forecast = daily 24h | retrain = weekly drift-gated",
    )
    args = parser.parse_args()

    print(f"\n{'='*60}\n  VOLTCAST PIPELINE - mode: {args.mode}\n{'='*60}")
    {"build": mode_build, "forecast": mode_forecast, "retrain": mode_retrain}[args.mode]()
    print(f"\n{'='*60}\n  PIPELINE DONE - {args.mode}\n{'='*60}")


if __name__ == "__main__":
    main()
