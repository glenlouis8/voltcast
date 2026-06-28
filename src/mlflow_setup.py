"""
src/mlflow_setup.py

One place that decides WHERE MLflow logs to. Every other file calls
setup_mlflow() instead of hardcoding a tracking URI.

Two modes, chosen automatically from environment variables:

    1. DagsHub (cloud)  — if DAGSHUB_REPO_OWNER, DAGSHUB_REPO_NAME, and
                          DAGSHUB_TOKEN are all set (in .env or CI secrets).
                          Models + metrics go to your DagsHub MLflow server.

    2. Local sqlite     — fallback if those are missing. Lets you keep
                          working offline with no cloud account.

Why env-driven? Local dev needs no internet; cloud CI just sets the secrets.
Same code path, no edits to switch.

DagsHub auth uses standard MLflow env vars:
    MLFLOW_TRACKING_USERNAME = your DagsHub username
    MLFLOW_TRACKING_PASSWORD = your DagsHub token
"""

import os

import mlflow
from dotenv import load_dotenv

# Load .env so DAGSHUB_* and the token are available as os.getenv(...).
load_dotenv()


def setup_mlflow() -> str:
    """
    Configure MLflow tracking + registry destination. Returns the URI used.
    Call this once at the start of any script that logs to MLflow.
    """
    owner = os.getenv("DAGSHUB_REPO_OWNER")
    repo  = os.getenv("DAGSHUB_REPO_NAME")
    token = os.getenv("DAGSHUB_TOKEN")

    # All three present → go to DagsHub cloud.
    if owner and repo and token:
        uri = f"https://dagshub.com/{owner}/{repo}.mlflow"
        # MLflow reads these for HTTP basic auth against DagsHub.
        os.environ["MLFLOW_TRACKING_USERNAME"] = owner
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
        mlflow.set_tracking_uri(uri)
        print(f"[mlflow] DagsHub cloud → {uri}")
        return uri

    # Otherwise → local sqlite file (offline dev).
    uri = "sqlite:///mlflow.db"
    mlflow.set_tracking_uri(uri)
    print(f"[mlflow] local → {uri}")
    return uri
