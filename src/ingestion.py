"""
src/ingestion.py

Pulls hourly electricity demand from EIA API for 4 US grid regions.
Saves one parquet file per region to data/raw/<REGION>.parquet

Run:
    python src/ingestion.py

Needs EIA_API_KEY set in your environment:
    export EIA_API_KEY=your_key_here
"""

import os
import time
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# Load variables from .env file into environment automatically.
# After this line, os.getenv("EIA_API_KEY") works.
load_dotenv()

# ── constants ────────────────────────────────────────────────────────────────

# The 4 US grid regions we care about.
# These are EIA's official "respondent" codes.
REGIONS = ["CAL", "TEX", "PJM", "MISO"]

# Pull data starting from this date.
# 2015 gives us ~10 years of hourly data (~87,600 rows per region).
START_DATE = "2015-01-01T00"

# EIA returns max 5000 rows per API call.
# We'll loop (paginate) until we have all rows.
PAGE_SIZE = 5000

# Where to save the raw parquet files.
# Path(__file__) = this file's location (src/ingestion.py)
# .parent = the src/ folder
# .parent.parent = the voltcast/ root folder
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"


# ── helpers ──────────────────────────────────────────────────────────────────

def fetch_region(region: str, api_key: str) -> pd.DataFrame:
    """
    Fetch ALL hourly demand rows for one region from EIA API.
    Handles pagination automatically — keeps calling until no rows left.

    Returns a DataFrame with columns: timestamp, load_mw
    """
    url = "https://api.eia.gov/v2/electricity/rto/region-data/data/"

    all_rows = []   # we'll collect every page of results here
    offset = 0      # offset = how many rows to skip (moves forward each page)

    print(f"  Fetching {region}...")

    while True:
        # Build the query parameters for this API call.
        # Think of this like filling out a form: which region, which dates, how many rows.
        params = {
            "api_key": api_key,
            "frequency": "hourly",
            "data[0]": "value",           # "value" = the MW number we want
            "facets[type][]": "D",        # D = Demand (not generation or interchange)
            "facets[respondent][]": region,
            "start": START_DATE,
            "sort[0][column]": "period",  # sort by time, oldest first
            "sort[0][direction]": "asc",
            "length": PAGE_SIZE,
            "offset": offset,             # skip this many rows (pagination)
        }

        # Make the HTTP GET request to the API.
        # Like typing a URL in a browser, but in Python.
        response = requests.get(url, params=params, timeout=30)

        # If EIA sends back an error (e.g. bad API key, server down),
        # raise_for_status() turns it into a Python exception so we know immediately.
        response.raise_for_status()

        # response.json() converts the raw text response into a Python dict.
        data = response.json()

        # Dig into the nested response structure to get the list of rows.
        rows = data["response"]["data"]

        # If EIA returns 0 rows, we've fetched everything — stop looping.
        if not rows:
            break

        all_rows.extend(rows)  # add this page's rows to our master list
        offset += PAGE_SIZE    # move the starting point forward for next call

        print(f"    {region}: fetched {len(all_rows):,} rows so far...")

        # Be polite — wait 0.25 seconds between calls so we don't hammer EIA's servers.
        time.sleep(0.25)

    print(f"  {region}: done. Total rows = {len(all_rows):,}")

    # Convert list of dicts → DataFrame (a table with named columns)
    df = pd.DataFrame(all_rows)

    # "period" comes back as a string like "2015-01-01T00".
    # pd.to_datetime() converts it to a proper datetime object Python understands.
    df["period"] = pd.to_datetime(df["period"])

    # "value" comes back as a string — convert to float so we can do math.
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # Keep only the two columns we need. Drop everything else (respondent, type, units).
    df = df[["period", "value"]].copy()

    # Rename columns to cleaner names.
    df = df.rename(columns={"period": "timestamp", "value": "load_mw"})

    # Sort by time just to be safe (should already be sorted, but good habit).
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    # Read the API key from your environment variables.
    # Never hardcode secrets in code — that's how they end up on GitHub.
    api_key = os.getenv("EIA_API_KEY")
    if not api_key:
        raise ValueError(
            "EIA_API_KEY not set. Run: export EIA_API_KEY=your_key_here"
        )

    # Create the data/raw/ folder if it doesn't already exist.
    # parents=True means also create parent folders if needed.
    # exist_ok=True means don't crash if the folder already exists.
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for region in REGIONS:
        print(f"\nRegion: {region}")

        df = fetch_region(region, api_key)

        # Save as parquet. Much smaller and faster than CSV.
        # index=False means don't save the row numbers (0, 1, 2...) as a column.
        out_path = RAW_DIR / f"{region}.parquet"
        df.to_parquet(out_path, index=False)

        print(f"  Saved → {out_path}")
        print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"  Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
        print(f"  Load range: {df['load_mw'].min():,.0f} MW – {df['load_mw'].max():,.0f} MW")

    print("\nIngestion complete.")


if __name__ == "__main__":
    # This block only runs when you call this file directly:
    #     python src/ingestion.py
    # It does NOT run if another file imports this module.
    # Standard Python pattern.
    main()
