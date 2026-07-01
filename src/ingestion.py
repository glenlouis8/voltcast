"""
Pull hourly electricity demand from the EIA API for 4 US grid regions and save
one parquet per region to data/raw/<REGION>.parquet.

    python src/ingestion.py

Needs EIA_API_KEY in the environment (or .env).
"""

import os
import time
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

load_dotenv()

# EIA "respondent" codes for the regions we forecast.
REGIONS = ["CAL", "TEX", "PJM", "MISO"]

# Rolling training window: always the last N years, not a fixed anchor.
# Demand drifts (EVs, solar, AC habits, climate) so old patterns go stale, and a
# fixed start would make the dataset grow forever. Rolling = constant size, always
# recent. 5 years (~43,800 hours) spans several seasonal cycles.
TRAIN_WINDOW_YEARS = 5
START_DATE = (
    datetime.now(timezone.utc) - timedelta(days=365 * TRAIN_WINDOW_YEARS)
).strftime("%Y-%m-%dT%H")

PAGE_SIZE = 5000  # EIA returns at most 5000 rows per call, so we paginate
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"

MAX_RETRIES = 5  # EIA occasionally times out or 502s under load; retry before giving up


def fetch_region(region: str, api_key: str, start: str = START_DATE) -> pd.DataFrame:
    """
    Fetch hourly demand for one region from `start` onward, paginating until the
    API returns no more rows. Drift checks pass a recent `start` to grab only
    recent data. Returns columns: timestamp, load_mw.
    """
    url = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
    all_rows = []
    offset = 0

    print(f"  Fetching {region}...")

    while True:
        params = {
            "api_key": api_key,
            "frequency": "hourly",
            "data[0]": "value",
            "facets[type][]": "D",        # D = Demand
            "facets[respondent][]": region,
            "start": start,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "length": PAGE_SIZE,
            "offset": offset,
        }

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                rows = response.json()["response"]["data"]
                break
            except (requests.exceptions.RequestException, KeyError) as e:
                if attempt == MAX_RETRIES:
                    raise
                wait = 2 ** attempt  # 2s, 4s, 8s, 16s
                print(f"    {region}: request failed ({e}), retry {attempt}/{MAX_RETRIES} in {wait}s...")
                time.sleep(wait)

        if not rows:
            break

        all_rows.extend(rows)
        offset += PAGE_SIZE
        print(f"    {region}: fetched {len(all_rows):,} rows so far...")
        time.sleep(0.25)  # be polite to EIA's servers

    print(f"  {region}: done. Total rows = {len(all_rows):,}")

    df = pd.DataFrame(all_rows)
    df["period"] = pd.to_datetime(df["period"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df[["period", "value"]].copy()
    df = df.rename(columns={"period": "timestamp", "value": "load_mw"})
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def main():
    api_key = os.getenv("EIA_API_KEY")
    if not api_key:
        raise ValueError("EIA_API_KEY not set. Run: export EIA_API_KEY=your_key_here")

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for region in REGIONS:
        print(f"\nRegion: {region}")
        df = fetch_region(region, api_key)

        out_path = RAW_DIR / f"{region}.parquet"
        df.to_parquet(out_path, index=False)

        print(f"  Saved -> {out_path}")
        print(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
        print(f"  Date range: {df['timestamp'].min()} -> {df['timestamp'].max()}")
        print(f"  Load range: {df['load_mw'].min():,.0f} MW - {df['load_mw'].max():,.0f} MW")

    print("\nIngestion complete.")


if __name__ == "__main__":
    main()
