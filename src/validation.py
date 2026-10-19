"""
Clean the raw demand files so every hour has exactly one good value.

    python src/validation.py

Reads data/raw/<REGION>.parquet and overwrites it with the cleaned version.

Why: later steps count rows to find "same hour last week" (168 rows back). A
missing hour or a fake value would make that counting wrong.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pandera.pandas as pa

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"

REGIONS = ["CAL", "TEX", "PJM", "MISO"]

# Highest demand each grid could physically reach. Anything above is a bad value.
MAX_LOAD_MW = {"CAL": 80_000, "TEX": 105_000, "PJM": 200_000, "MISO": 155_000}

GLITCH_RATIO = 0.25   # a value this far (25%) from its neighbors is a glitch
NEIGHBORS = 3         # compare with the 3 hours before and the 3 hours after
MAX_LINE_GAP = 3      # holes up to this many hours: fill with a straight line
HOURS_PER_WEEK = 168  # longer holes: copy the same hour from one week earlier


def mark_bad_values(s: pd.Series, region: str) -> tuple[pd.Series, int, int]:
    """Blank out impossible values and glitches. Returns (series, n_impossible, n_glitch)."""
    # 1. impossible: zero, negative, or above what the grid can ever reach
    impossible = (s <= 0) | (s > MAX_LOAD_MW[region])
    s = s.mask(impossible)

    # 2. glitch: compare each value with the median of its 6 neighbors
    #    (3 hours before + 3 hours after). A normal hour is close to that median.
    neighbors = pd.concat(
        [s.shift(k) for k in range(-NEIGHBORS, NEIGHBORS + 1) if k != 0], axis=1
    )
    median = neighbors.median(axis=1)  # ignores empty neighbors
    glitch = ((s - median).abs() / median) > GLITCH_RATIO
    s = s.mask(glitch)

    return s, int(impossible.sum()), int(glitch.sum())


def fill_gaps(s: pd.Series) -> tuple[pd.Series, int, int, int]:
    """
    Fill empty hours. Returns (series, n_line, n_last_week, n_fallback).
    Short holes -> straight line. Long holes -> same hour one week earlier.
    """
    empty = s.isna()

    # length of each run of empty hours (so we know if a hole is short or long)
    run_id = (empty != empty.shift()).cumsum()
    run_len = empty.groupby(run_id).transform("sum")
    short = empty & (run_len <= MAX_LINE_GAP)
    long = empty & (run_len > MAX_LINE_GAP)

    # short holes: straight line between the two neighbors
    line = s.interpolate(method="linear", limit_area="inside")
    s = s.where(~short, line)

    # long holes: copy the value from 168 hours (1 week) earlier
    last_week = s.shift(HOURS_PER_WEEK)
    s = s.where(~long, last_week)

    # safety net: if last week was also empty, use a straight line
    n_fallback = int(s.isna().sum())
    if n_fallback:
        s = s.interpolate(method="linear", limit_area="inside")

    return s, int(short.sum()), int(long.sum()), n_fallback


def build_schema(region: str) -> pa.DataFrameSchema:
    """The final check: what a clean file must look like."""
    return pa.DataFrameSchema(
        {
            "timestamp": pa.Column(
                "datetime64[ns]",
                nullable=False,
                unique=True,
                # every row is exactly 1 hour after the one before it
                checks=pa.Check(
                    lambda t: (t.diff().dropna().dt.total_seconds() == 3600).all(),
                    error="timestamps must be exactly 1 hour apart (no gaps)",
                ),
            ),
            "load_mw": pa.Column(
                float,
                nullable=False,
                checks=[
                    pa.Check.greater_than(0),
                    pa.Check.less_than_or_equal_to(MAX_LOAD_MW[region]),
                ],
            ),
        },
        coerce=True,
    )


def clean_region(region: str) -> pd.DataFrame:
    df = pd.read_parquet(RAW_DIR / f"{region}.parquet").sort_values("timestamp")
    rows_in = len(df)
    df = df.drop_duplicates("timestamp", keep="last")

    # put the data on a full hourly calendar; hours EIA skipped become empty
    calendar = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="h")
    s = df.set_index("timestamp")["load_mw"].astype(float).reindex(calendar)
    n_skipped = len(calendar) - len(df)
    n_empty = int(s.isna().sum()) - n_skipped  # rows EIA sent without a number

    s, n_impossible, n_glitch = mark_bad_values(s, region)

    # drop empty hours at the very start/end (nothing to fill them from)
    s = s.loc[s.first_valid_index():s.last_valid_index()]

    s, n_line, n_week, n_fallback = fill_gaps(s)

    out = s.rename("load_mw").rename_axis("timestamp").reset_index()
    build_schema(region).validate(out)  # raises an error if anything is still wrong

    print(f"{region}: {rows_in:,} rows in -> {len(out):,} rows out")
    print(f"  hours EIA skipped: {n_skipped} | empty values: {n_empty}")
    print(f"  impossible values: {n_impossible} | glitches: {n_glitch}")
    print(f"  filled with straight line: {n_line} | with last week: {n_week} | fallback: {n_fallback}")
    return out


def main():
    for region in REGIONS:
        clean = clean_region(region)
        clean.to_parquet(RAW_DIR / f"{region}.parquet", index=False)
    print("\nValidation done. Files overwritten in data/raw/.")


if __name__ == "__main__":
    main()
