"""
src/storage.py

One place that decides WHERE forecast files live: S3 if configured, local
folder if not. Every file that reads/writes forecasts calls these helpers
instead of touching disk or S3 directly.

Two modes, chosen from environment variables:

    1. S3 (cloud)  — if S3_BUCKET + AWS creds are set (.env or CI secrets).
                     Forecasts go to s3://<bucket>/forecasts/<region>.parquet.
                     CI writes them; the dashboard/API read them. Survives the
                     ephemeral GitHub Actions runner being wiped.

    2. Local       — fallback if S3 isn't configured. Writes to data/forecasts/.
                     Lets you work offline with no AWS account.

Why a fallback? Local dev needs no cloud. CI just sets the secrets. Same code.
"""

import io
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

LOCAL_ROOT = Path(__file__).parent.parent / "data"

# S3 prefixes (folders) inside the bucket for each kind of file.
FORECAST_PREFIX  = "forecasts"
REFERENCE_PREFIX = "reference"


def _bucket() -> str | None:
    """Return the S3 bucket name if S3 is configured, else None (local mode)."""
    bucket = os.getenv("S3_BUCKET")
    # Need both a bucket AND credentials to use S3. Missing either → local.
    if bucket and os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"):
        return bucket
    return None


def _s3_client():
    """Build a boto3 S3 client. Imported lazily so local mode needs no boto3 call."""
    import boto3
    return boto3.client("s3")  # reads AWS_* env vars automatically


# ── generic save/load (S3 or local), shared by forecast + reference ───────────

def _save(prefix: str, region: str, df: pd.DataFrame) -> str:
    """Save df to S3 if configured, else local data/<prefix>/. Returns location."""
    bucket = _bucket()
    filename = f"{region}.parquet"

    if bucket:
        # Serialize to parquet bytes in memory, then upload (no temp file).
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        buf.seek(0)
        key = f"{prefix}/{filename}"
        _s3_client().put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
        return f"s3://{bucket}/{key}"

    # Local fallback.
    local_dir = LOCAL_ROOT / prefix
    local_dir.mkdir(parents=True, exist_ok=True)
    path = local_dir / filename
    df.to_parquet(path, index=False)
    return str(path)


def _save_json(prefix: str, region: str, df: pd.DataFrame) -> str:
    """
    Save df as JSON (S3 or local) so a browser frontend (Vercel) can fetch it
    directly — browsers can't read parquet. orient="records" → a plain list of
    row objects; date_format="iso" → timestamps as readable ISO strings.
    """
    bucket = _bucket()
    filename = f"{region}.json"
    body = df.to_json(orient="records", date_format="iso")

    if bucket:
        key = f"{prefix}/{filename}"
        _s3_client().put_object(
            Bucket=bucket,
            Key=key,
            Body=body.encode("utf-8"),
            ContentType="application/json",
        )
        return f"s3://{bucket}/{key}"

    local_dir = LOCAL_ROOT / prefix
    local_dir.mkdir(parents=True, exist_ok=True)
    path = local_dir / filename
    path.write_text(body)
    return str(path)


def _load(prefix: str, region: str) -> pd.DataFrame | None:
    """Load df from S3 if configured, else local. None if it doesn't exist."""
    bucket = _bucket()
    filename = f"{region}.parquet"

    if bucket:
        try:
            obj = _s3_client().get_object(Bucket=bucket, Key=f"{prefix}/{filename}")
            return pd.read_parquet(io.BytesIO(obj["Body"].read()))
        except Exception:
            return None

    path = LOCAL_ROOT / prefix / filename
    if not path.exists():
        return None
    return pd.read_parquet(path)


# ── forecasts ─────────────────────────────────────────────────────────────────

def save_forecast(region: str, df: pd.DataFrame) -> str:
    """
    Save a region's 24h forecast in BOTH formats:
        - parquet → for internal reuse (dashboard, re-loading in Python)
        - JSON    → for the Vercel frontend to fetch directly from S3
    Returns the parquet location (the canonical one).
    """
    location = _save(FORECAST_PREFIX, region, df)
    _save_json(FORECAST_PREFIX, region, df)   # browser-readable copy
    return location


def load_forecast(region: str) -> pd.DataFrame | None:
    """Load a region's 24h forecast (S3 or local). None if absent."""
    return _load(FORECAST_PREFIX, region)


# ── drift reference snapshots ─────────────────────────────────────────────────

def save_reference(region: str, df: pd.DataFrame) -> str:
    """
    Save the champion's reference dataset — a snapshot of the training data
    the current champion learned from. Drift compares fresh data against this.
    Written when a model is crowned champion.
    """
    return _save(REFERENCE_PREFIX, region, df)


def load_reference(region: str) -> pd.DataFrame | None:
    """Load the champion's reference snapshot (S3 or local). None if absent."""
    return _load(REFERENCE_PREFIX, region)
