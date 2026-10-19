"""
Download hourly electricity demand from the EIA website for 4 grid regions and
save one file per region: data/raw/<REGION>.parquet

    python src/ingestion.py

Needs EIA_API_KEY in your .env file.
"""

import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()  # reads .env so os.getenv can see EIA_API_KEY

REGIONS = ["CAL", "TEX", "PJM", "MISO"]
YEARS_OF_DATA = 5  # 3 years train + 1 year validation + 1 year test
PAGE_SIZE = 5000  # EIA gives at most 5000 rows per request
MAX_TRIES = 5     # tries per request before giving up
URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"


def fetch_region(region: str, api_key: str) -> pd.DataFrame:
    """Download all hourly demand rows for one region. Returns timestamp + load_mw."""
    start = datetime.now(timezone.utc) - timedelta(days=365 * YEARS_OF_DATA)
    all_rows = []
    offset = 0  # which row to start from; moves forward one page at a time

    while True:
        params = {
            "api_key": api_key,
            "frequency": "hourly",
            "data[0]": "value",
            "facets[type][]": "D",           # D = demand
            "facets[respondent][]": region,  # which grid
            "start": start.strftime("%Y-%m-%dT%H"),
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "length": PAGE_SIZE,
            "offset": offset,
        }
        # EIA is slow/flaky, so on failure wait a bit and try the same page again
        for attempt in range(1, MAX_TRIES + 1):
            try:
                response = requests.get(URL, params=params, timeout=120)
                response.raise_for_status()  # turns HTTP errors (502 etc.) into exceptions
                rows = response.json()["response"]["data"]
                break  # success, leave the retry loop
            except (requests.exceptions.RequestException, KeyError) as e:
                if attempt == MAX_TRIES:
                    raise  # out of tries, give up with the real error
                wait = 5 * attempt  # 5s, 10s, 15s, 20s
                print(f"  {region}: failed ({type(e).__name__}), retry {attempt}/{MAX_TRIES - 1} in {wait}s")
                time.sleep(wait)

        all_rows.extend(rows)
        if len(rows) < PAGE_SIZE:  # a short page means this was the last one
            break
        offset += PAGE_SIZE

    if not all_rows:
        raise ValueError(f"{region}: EIA returned no rows")

    df = pd.DataFrame(all_rows)
    df["timestamp"] = pd.to_datetime(df["period"])
    df["load_mw"] = pd.to_numeric(df["value"], errors="coerce")
    df = df[["timestamp", "load_mw"]].sort_values("timestamp")
    return df.reset_index(drop=True)


def main():
    api_key = os.getenv("EIA_API_KEY")
    if not api_key:
        raise ValueError("EIA_API_KEY is not set. Put it in your .env file.")

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for region in REGIONS:
        print(f"Downloading {region}...")
        df = fetch_region(region, api_key)
        df.to_parquet(RAW_DIR / f"{region}.parquet", index=False)
        print(f"  saved {len(df):,} rows ({df['timestamp'].min()} -> {df['timestamp'].max()})")


if __name__ == "__main__":
    main()
