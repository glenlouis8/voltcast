"""
src/validation.py

Inspects and cleans raw parquet files saved by ingestion.py.
Removes rows that are impossible or corrupt before feature engineering.

Run:
    python src/validation.py

What it checks per region:
    1. Null values in load_mw
    2. Zero or negative load_mw (physically impossible)
    3. Impossibly large values (corrupt integers like 2^31-1)
    4. Gaps in timestamp > 2 hours (missing data)

Saves cleaned files back to data/raw/<REGION>.parquet (overwrites).
Prints a report so you can see exactly what was removed and why.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── constants ────────────────────────────────────────────────────────────────

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"

REGIONS = ["CAL", "TEX", "PJM", "MISO"]

# Physical maximum MW per region — based on known grid capacity.
# Anything above this is a corrupt data point, not a real heatwave.
# CAL peak ever recorded: ~63,000 MW (July 2006)
# TEX peak ever recorded: ~85,000 MW (Feb 2023, Winter Storm Elliott)
# PJM peak ever recorded: ~165,000 MW
# MISO peak ever recorded: ~125,000 MW
# We add 20% buffer above known peaks just to be safe.
MAX_LOAD_MW = {
    "CAL":  80_000,
    "TEX": 105_000,
    "PJM": 200_000,
    "MISO": 155_000,
}


# ── helpers ──────────────────────────────────────────────────────────────────

def validate_region(region: str) -> pd.DataFrame:
    """
    Load raw parquet for one region, run all checks, return cleaned DataFrame.
    Prints a detailed report of every issue found.
    """
    path = RAW_DIR / f"{region}.parquet"
    df = pd.read_parquet(path)
    original_len = len(df)

    print(f"\n{'='*50}")
    print(f"Region: {region}  |  Rows before cleaning: {original_len:,}")
    print(f"{'='*50}")

    # ── Check 1: Null values ─────────────────────────────────────────────────
    # A null means EIA had no reading for that hour.
    null_mask = df["load_mw"].isna()
    n_nulls = null_mask.sum()
    if n_nulls > 0:
        print(f"  [REMOVE] Null values: {n_nulls} rows")
    else:
        print(f"  [OK] No null values")

    # ── Check 2: Zero or negative load ───────────────────────────────────────
    # Electricity demand is always positive. Zero = missing, negative = error.
    zero_mask = df["load_mw"] <= 0
    n_zeros = zero_mask.sum()
    if n_zeros > 0:
        print(f"  [REMOVE] Zero or negative load_mw: {n_zeros} rows")
        print(f"           Values: {df.loc[zero_mask, 'load_mw'].values[:5]}")
    else:
        print(f"  [OK] No zero/negative values")

    # ── Check 3: Impossibly large values ─────────────────────────────────────
    # 2^31 - 1 = 2,147,483,647 — this is a database sentinel for "no data".
    # Also catches any value above the known physical max for this region.
    max_allowed = MAX_LOAD_MW[region]
    spike_mask = df["load_mw"] > max_allowed
    n_spikes = spike_mask.sum()
    if n_spikes > 0:
        print(f"  [REMOVE] Values above {max_allowed:,} MW (max allowed): {n_spikes} rows")
        print(f"           Worst values:")
        print(df.loc[spike_mask, ["timestamp", "load_mw"]].sort_values("load_mw", ascending=False).head(3).to_string(index=False))
    else:
        print(f"  [OK] No impossible spike values (max allowed: {max_allowed:,} MW)")

    # ── Remove all bad rows ───────────────────────────────────────────────────
    # Combine all bad masks with OR — remove a row if ANY check fails.
    bad_mask = null_mask | zero_mask | spike_mask
    df_clean = df[~bad_mask].copy()  # ~ means "NOT" — keep rows where bad_mask is False

    # ── Check 4: Gaps in timestamps ───────────────────────────────────────────
    # After removing bad rows, check if we have missing hours.
    # .diff() computes difference between consecutive timestamps.
    # A normal gap = 1 hour. Anything > 2 hours means data is missing.
    df_clean = df_clean.sort_values("timestamp").reset_index(drop=True)
    time_diffs = df_clean["timestamp"].diff().dropna()
    big_gaps = time_diffs[time_diffs > pd.Timedelta(hours=2)]

    if len(big_gaps) > 0:
        print(f"  [WARN]  Timestamp gaps > 2 hours: {len(big_gaps)} gaps found")
        print(f"          (These are logged but NOT removed — model handles missing hours)")
        for idx in big_gaps.index[:3]:  # show first 3 gaps
            t_before = df_clean["timestamp"].iloc[idx - 1]
            t_after  = df_clean["timestamp"].iloc[idx]
            print(f"          {t_before} → {t_after}  ({time_diffs.iloc[idx]})")
    else:
        print(f"  [OK] No timestamp gaps > 2 hours")

    # ── Summary ───────────────────────────────────────────────────────────────
    removed = original_len - len(df_clean)
    pct = removed / original_len * 100
    print(f"\n  Rows removed: {removed:,} ({pct:.2f}%)")
    print(f"  Rows kept:    {len(df_clean):,}")
    print(f"  Load range:   {df_clean['load_mw'].min():,.0f} – {df_clean['load_mw'].max():,.0f} MW")
    print(f"  Date range:   {df_clean['timestamp'].min()} → {df_clean['timestamp'].max()}")

    return df_clean


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("Running validation on all regions...")

    for region in REGIONS:
        df_clean = validate_region(region)

        # Overwrite the raw parquet with the cleaned version.
        out_path = RAW_DIR / f"{region}.parquet"
        df_clean.to_parquet(out_path, index=False)
        print(f"  Saved cleaned data → {out_path}")

    print("\nValidation complete. All files cleaned.")


if __name__ == "__main__":
    main()
