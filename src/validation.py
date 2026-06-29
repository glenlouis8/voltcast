"""
Clean the raw parquet from ingestion.py before feature engineering: drop nulls,
non-positive load, and corrupt spikes; warn on timestamp gaps. Overwrites
data/raw/<REGION>.parquet and prints a report.

    python src/validation.py

Also exposes a Pandera schema (build_schema / validate) used as a data contract
wherever fresh data enters the pipeline.
"""

import pandas as pd
import numpy as np
import pandera.pandas as pa
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"

REGIONS = ["CAL", "TEX", "PJM", "MISO"]

# Physical ceiling per grid (known record peak + ~20% buffer). Anything above is
# corrupt (e.g. the 2^31-1 "no data" sentinel), not a real heatwave.
MAX_LOAD_MW = {
    "CAL":  80_000,
    "TEX": 105_000,
    "PJM": 200_000,
    "MISO": 155_000,
}


def build_schema(region: str) -> pa.DataFrameSchema:
    """
    Validation schema for a region. Region-specific because the max-load ceiling
    differs per grid. Requires non-null timestamp and positive load below the
    ceiling.
    """
    return pa.DataFrameSchema(
        {
            "timestamp": pa.Column("datetime64[ns]", nullable=False),
            "load_mw": pa.Column(
                float,
                checks=[
                    pa.Check.greater_than(0),
                    pa.Check.less_than(MAX_LOAD_MW[region]),
                ],
                nullable=False,
            ),
        },
        strict=False,   # allow extra columns (rolling_mean_24, etc.)
        coerce=True,    # int load -> float instead of rejecting
    )


def validate(df: pd.DataFrame, region: str) -> pd.DataFrame:
    """
    Gate fresh data: returns df if it passes, raises SchemaError otherwise.
    lazy=True collects all failures, not just the first.
    """
    return build_schema(region).validate(df, lazy=True)


def validate_region(region: str) -> pd.DataFrame:
    """Load a region's raw parquet, run all checks, return the cleaned df."""
    path = RAW_DIR / f"{region}.parquet"
    df = pd.read_parquet(path)
    original_len = len(df)

    print(f"\n{'='*50}")
    print(f"Region: {region}  |  Rows before cleaning: {original_len:,}")
    print(f"{'='*50}")

    null_mask = df["load_mw"].isna()
    n_nulls = null_mask.sum()
    print(f"  [REMOVE] Null values: {n_nulls} rows" if n_nulls else "  [OK] No null values")

    # demand is always positive; zero = missing, negative = error
    zero_mask = df["load_mw"] <= 0
    n_zeros = zero_mask.sum()
    if n_zeros:
        print(f"  [REMOVE] Zero or negative load_mw: {n_zeros} rows")
        print(f"           Values: {df.loc[zero_mask, 'load_mw'].values[:5]}")
    else:
        print(f"  [OK] No zero/negative values")

    max_allowed = MAX_LOAD_MW[region]
    spike_mask = df["load_mw"] > max_allowed
    n_spikes = spike_mask.sum()
    if n_spikes:
        print(f"  [REMOVE] Values above {max_allowed:,} MW: {n_spikes} rows")
        print(df.loc[spike_mask, ["timestamp", "load_mw"]]
              .sort_values("load_mw", ascending=False).head(3).to_string(index=False))
    else:
        print(f"  [OK] No impossible spikes (max allowed: {max_allowed:,} MW)")

    bad_mask = null_mask | zero_mask | spike_mask
    df_clean = df[~bad_mask].copy()

    # gaps: consecutive hours should differ by 1h; >2h means missing data.
    # use total_seconds() to dodge a numpy timedelta-unit deprecation warning.
    df_clean = df_clean.sort_values("timestamp").reset_index(drop=True)
    gap_hours = df_clean["timestamp"].diff().dt.total_seconds() / 3600
    big_gaps = gap_hours[gap_hours > 2]

    if len(big_gaps):
        print(f"  [WARN]  Timestamp gaps > 2 hours: {len(big_gaps)} (logged, not removed)")
        for idx in big_gaps.index[:3]:
            t_before = df_clean["timestamp"].iloc[idx - 1]
            t_after  = df_clean["timestamp"].iloc[idx]
            print(f"          {t_before} -> {t_after}  ({gap_hours.iloc[idx]:.0f}h gap)")
    else:
        print(f"  [OK] No timestamp gaps > 2 hours")

    removed = original_len - len(df_clean)
    print(f"\n  Rows removed: {removed:,} ({removed / original_len * 100:.2f}%)")
    print(f"  Rows kept:    {len(df_clean):,}")
    print(f"  Load range:   {df_clean['load_mw'].min():,.0f} - {df_clean['load_mw'].max():,.0f} MW")
    print(f"  Date range:   {df_clean['timestamp'].min()} -> {df_clean['timestamp'].max()}")

    # cleaned data must satisfy the contract; if not, cleaning missed something
    validate(df_clean, region)
    print(f"  [PASS] Pandera contract satisfied")

    return df_clean


def main():
    print("Running validation on all regions...")

    for region in REGIONS:
        df_clean = validate_region(region)
        out_path = RAW_DIR / f"{region}.parquet"
        df_clean.to_parquet(out_path, index=False)
        print(f"  Saved cleaned data -> {out_path}")

    print("\nValidation complete. All files cleaned.")


if __name__ == "__main__":
    main()
