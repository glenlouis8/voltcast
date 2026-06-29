"""
Decides where MLflow logs: DagsHub cloud when DAGSHUB_REPO_OWNER/REPO_NAME/TOKEN
are set (.env or CI secrets), else a local sqlite file for offline dev. Every
script calls setup_mlflow() instead of hardcoding a tracking URI.
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
