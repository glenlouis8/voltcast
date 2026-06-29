"""
src/pipeline.py

One orchestrator that runs the existing scripts in the right order.
Nothing new is computed here — this just CALLS the scripts you already wrote,
in sequence, so you (or a GitHub Action) run one command instead of ten.

Three modes:

    python src/pipeline.py build      # first-time / full rebuild of all regions
    python src/pipeline.py forecast   # daily: refresh data → predict next 24h
    python src/pipeline.py retrain    # weekly: drift check → retrain only if drifted

Why subprocess instead of importing functions?
    Each script has its own `main()` and is independently runnable + testable.
    Calling them as subprocesses keeps that contract and exactly mirrors what
    the cloud (GitHub Actions) will do. If a step fails, the pipeline stops
    loudly instead of silently continuing on broken data.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# Same 4 regions everywhere.
REGIONS = ["CAL", "TEX", "PJM", "MISO"]

# Staleness fallback: even if NO drift fires, retrain a champion older than this.
# WHY: drift tests can miss slow, gradual shifts the K-S test never flags. A hard
# age cap guarantees the model never serves forever on ancient training data.
MAX_CHAMPION_AGE_DAYS = 30

# Folder this file lives in (src/). We call sibling scripts by absolute path so
# the pipeline works no matter what directory you launch it from.
SRC = Path(__file__).parent


def run(script: str, *args: str, allow_fail: bool = False) -> int:
    """
    Run one script as a subprocess: `python src/<script> <args>`.

    sys.executable = the exact Python running THIS file, so the subprocess uses
    the same virtualenv (same torch, same mlflow). No env surprises.

    allow_fail=False (default): a non-zero exit raises → pipeline halts. We never
        want features.py to run on data that validation.py rejected.
    allow_fail=True: return the exit code instead of raising. Used for drift.py,
        whose exit code 1 means "drift detected" — a signal, not an error.
    """
    cmd = [sys.executable, str(SRC / script), *args]
    pretty = " ".join([script, *args])
    print(f"\n{'─'*60}\n▶ {pretty}\n{'─'*60}")

    result = subprocess.run(cmd)
    if result.returncode != 0 and not allow_fail:
        # Stop the whole pipeline. Better to fail here than poison later steps.
        raise SystemExit(f"✗ step failed: {pretty} (exit {result.returncode})")
    return result.returncode


# ── shared data-prep stage (used by every mode) ───────────────────────────────

def prep_data() -> None:
    """
    Steps 1→3: pull fresh data, validate it, build feature matrices.
    All three loop every region internally, so no per-region calls needed.

        ingestion.py  → data/raw/<region>.parquet      (fresh EIA pull)
        validation.py → reject bad rows (Pandera contract)
        features.py   → data/features/<region>.parquet (model-ready)
    """
    run("ingestion.py")
    run("validation.py")
    run("features.py")


# ── mode: build (full rebuild of every region) ────────────────────────────────

def mode_build() -> None:
    """
    First-time setup or full rebuild. For each region:
        train transformer + lstm → registry (crown champion) → inference.
    Use this after wiping DagsHub models, or when you change the architecture.
    """
    prep_data()
    for r in REGIONS:
        run("train.py", "--model", "transformer", "--country", r)
        run("train.py", "--model", "lstm",        "--country", r)
        run("registry.py", "--country", r)   # picks best, crowns champion, saves drift reference
        run("inference.py", "--country", r)  # first forecast from the new champion


# ── mode: forecast (daily) ────────────────────────────────────────────────────

def mode_forecast() -> None:
    """
    Daily robot. Refresh data, then predict next 24h per region using whoever
    is champion. No training. Output → forecasts (S3 if configured, else local).
    """
    prep_data()
    for r in REGIONS:
        run("inference.py", "--country", r)


# ── champion age (staleness fallback) ─────────────────────────────────────────

def champion_age_days(region: str) -> float | None:
    """
    How many days since this region's champion was registered.

    Reads the champion version's creation_timestamp from the DagsHub/MLflow
    registry (epoch milliseconds), converts to days-ago. Returns None if there
    is no champion yet (so the caller can treat "no champion" as "must build").

    Imported lazily inside the function so the rest of the pipeline (subprocess
    calls) doesn't pay the mlflow import cost unless retrain actually runs.
    """
    from mlflow.tracking import MlflowClient
    from mlflow_setup import setup_mlflow

    setup_mlflow()
    client = MlflowClient()
    try:
        champ = client.get_model_version_by_alias(f"voltcast-{region}", "champion")
    except Exception:
        return None  # no champion registered

    # creation_timestamp is epoch milliseconds. time.time() is epoch seconds.
    age_seconds = time.time() - (champ.creation_timestamp / 1000)
    return age_seconds / 86400  # seconds → days


# ── mode: retrain (weekly, drift-gated) ───────────────────────────────────────

def mode_retrain() -> None:
    """
    Weekly watchdog. Retrain a region if EITHER signal fires:

        1. DRIFT      — drift.py self-pulls a recent window, compares vs the
                        champion's training snapshot. Exit 1 = drifted.
        2. STALENESS  — champion older than MAX_CHAMPION_AGE_DAYS (catches slow
                        shifts drift's K-S test misses). No champion = also retrain.

    No drift AND fresh enough → skip. Saves compute.

    drift.py fetches its OWN small current window, so we don't prep_data() before
    the checks — only after, and only for regions that actually need retraining.
    """
    retrained = []
    for r in REGIONS:
        # ── reason 1: drift ──
        drifted = run("drift.py", "--country", r, allow_fail=True) != 0  # exit 1 = drift

        # ── reason 2: staleness ──
        age = champion_age_days(r)
        if age is None:
            stale, why_age = True, "no champion yet"
        else:
            stale = age > MAX_CHAMPION_AGE_DAYS
            why_age = f"champion {age:.0f}d old (cap {MAX_CHAMPION_AGE_DAYS}d)"

        if not (drifted or stale):
            print(f"  {r}: no drift, {why_age} → champion kept.")
            continue

        reason = "drift" if drifted else why_age
        print(f"  {r}: RETRAIN ({reason}).")
        prep_data()
        run("train.py", "--model", "transformer", "--country", r)
        run("train.py", "--model", "lstm",        "--country", r)
        run("registry.py", "--country", r)
        run("inference.py", "--country", r)
        retrained.append(r)

    if retrained:
        print(f"\nRetrained: {', '.join(retrained)}")
    else:
        print("\nNo region drifted or stale. Nothing retrained.")


def main():
    parser = argparse.ArgumentParser(description="VoltCast pipeline orchestrator.")
    parser.add_argument(
        "mode",
        choices=["build", "forecast", "retrain"],
        help="build = full rebuild | forecast = daily 24h | retrain = weekly drift-gated",
    )
    args = parser.parse_args()

    print(f"\n{'='*60}\n  VOLTCAST PIPELINE — mode: {args.mode}\n{'='*60}")
    {"build": mode_build, "forecast": mode_forecast, "retrain": mode_retrain}[args.mode]()
    print(f"\n{'='*60}\n  PIPELINE DONE — {args.mode}\n{'='*60}")


if __name__ == "__main__":
    main()
