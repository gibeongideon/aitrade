"""
csv_loader.py — Phase 1, Step 1

Loads the raw EURUSD_M15 CSV file, detects its format, normalises column names,
drops bad rows, and saves a clean Parquet file for all downstream scripts.

Accepted formats
----------------
Format A  (MT5 History Center export — two date/time columns, angle-bracket headers):
    <DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<TICKVOL>,<VOL>,<SPREAD>
    2023.01.02,00:00,1.07043,1.07059,1.06996,1.07018,234,0,0

Format B  (single datetime column, with or without a header row):
    Date,Open,High,Low,Close,Volume
    2023-01-02 00:00:00,1.07043,1.07059,1.06996,1.07018,234

    or (no header — raw data only):
    2021-09-24 20:15,1.17194,1.17217,1.17194,1.17209,106

Usage
-----
    python -m src.data.csv_loader
    python -m src.data.csv_loader --csv data/raw/EURUSD_M15.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml


# ── Internal column schema ────────────────────────────────────────────────────
INTERNAL_COLS = ["time", "open", "high", "low", "close", "volume"]


def _load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _detect_format(csv_path: Path) -> str:
    """
    Return 'A' for MT5 two-column date/time export, 'B' for single datetime.
    Peeks at the first raw line only.
    """
    with open(csv_path, "r") as f:
        first_line = f.readline().strip()

    # Format A: MT5 History Center — header starts with <DATE>
    if first_line.startswith("<DATE>") or first_line.startswith("<date>"):
        return "A"

    # Try parsing the first field as a date.  Format B has either a header
    # row ("Date,...") or raw data ("2021-09-24 20:15,...").
    # If the first token contains a dot separator like "2023.01.02", it is
    # Format A data without headers — treat as A.
    first_token = first_line.split(",")[0].strip()
    if "." in first_token and len(first_token) == 10:
        # e.g. "2023.01.02"
        return "A_NOHEADER"

    return "B"


def _read_format_a(csv_path: Path, has_header: bool) -> pd.DataFrame:
    """Load MT5 two-column date/time format."""
    if has_header:
        df = pd.read_csv(csv_path, skipinitialspace=True)
        # Strip angle brackets from column names
        df.columns = [c.strip("<>").lower() for c in df.columns]
        date_col, time_col = "date", "time"
        open_col, high_col = "open", "high"
        low_col, close_col = "low", "close"
        vol_col = "tickvol"
    else:
        df = pd.read_csv(csv_path, header=None,
                         names=["date", "time", "open", "high", "low",
                                "close", "tickvol", "vol", "spread"])
        date_col, time_col = "date", "time"
        open_col, high_col = "open", "high"
        low_col, close_col = "low", "close"
        vol_col = "tickvol"

    # Combine date + time into a single datetime string then parse
    combined = df[date_col].astype(str) + " " + df[time_col].astype(str)
    df["time"] = pd.to_datetime(combined, format="%Y.%m.%d %H:%M")

    df = df.rename(columns={
        open_col:  "open",
        high_col:  "high",
        low_col:   "low",
        close_col: "close",
        vol_col:   "volume",
    })
    return df[INTERNAL_COLS].copy()


def _read_format_b(csv_path: Path) -> pd.DataFrame:
    """
    Load single-datetime-column format.
    Auto-detects whether a header row is present by checking if the first
    token is parseable as a date.
    """
    with open(csv_path, "r") as f:
        first_line = f.readline().strip()

    first_token = first_line.split(",")[0].strip()

    # If the first token looks like a datetime value (not a word), no header.
    try:
        pd.to_datetime(first_token)
        has_header = False
    except Exception:
        has_header = True

    if has_header:
        df = pd.read_csv(csv_path, skipinitialspace=True)
        df.columns = [c.strip().lower() for c in df.columns]
        # Normalise common column name variations
        rename_map = {
            "date": "time", "datetime": "time",
            "vol":  "volume", "tick_volume": "volume", "tickvol": "volume",
        }
        df = df.rename(columns=rename_map)
    else:
        df = pd.read_csv(csv_path, header=None,
                         names=["time", "open", "high", "low", "close", "volume"])

    df["time"] = pd.to_datetime(df["time"])
    return df[INTERNAL_COLS].copy()


def load_csv(csv_path: str | Path) -> pd.DataFrame:
    """
    Public entry point. Detects format, loads, normalises, cleans, and returns
    a DataFrame with columns: time, open, high, low, close, volume.
    Sorted ascending by time, deduplicated, bad rows removed.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}")
        sys.exit(1)

    fmt = _detect_format(csv_path)
    print(f"[csv_loader] Detected format: {fmt}")

    if fmt == "A":
        df = _read_format_a(csv_path, has_header=True)
    elif fmt == "A_NOHEADER":
        df = _read_format_a(csv_path, has_header=False)
    else:
        df = _read_format_b(csv_path)

    raw_count = len(df)

    # ── Sort by time ──────────────────────────────────────────────────────────
    df = df.sort_values("time").reset_index(drop=True)

    # ── Deduplicate timestamps ────────────────────────────────────────────────
    before_dedup = len(df)
    df = df.drop_duplicates(subset="time").reset_index(drop=True)
    dupes_removed = before_dedup - len(df)

    # ── Drop bad rows ─────────────────────────────────────────────────────────
    # Zero volume = weekend / holiday / broker outage row
    zero_vol = (df["volume"] == 0).sum()
    df = df[df["volume"] > 0].reset_index(drop=True)

    # Drop rows with any NaN or zero price
    price_cols = ["open", "high", "low", "close"]
    bad_price = df[price_cols].isnull().any(axis=1) | (df[price_cols] == 0).any(axis=1)
    bad_price_count = bad_price.sum()
    df = df[~bad_price].reset_index(drop=True)

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"[csv_loader] Raw rows loaded   : {raw_count:,}")
    if dupes_removed:
        print(f"[csv_loader] Duplicate rows    : {dupes_removed:,} removed")
    print(f"[csv_loader] Zero-volume rows  : {zero_vol:,} removed")
    if bad_price_count:
        print(f"[csv_loader] Bad-price rows    : {bad_price_count:,} removed")
    print(f"[csv_loader] Clean rows        : {len(df):,}")
    print(f"[csv_loader] Date range        : {df['time'].iloc[0]}  →  {df['time'].iloc[-1]}")

    # Report largest gap
    time_diffs = df["time"].diff().dropna()
    max_gap = time_diffs.max()
    max_gap_idx = time_diffs.idxmax()
    if max_gap.total_seconds() > 3600:
        gap_start = df["time"].iloc[max_gap_idx - 1]
        gap_end   = df["time"].iloc[max_gap_idx]
        print(f"[csv_loader] Largest gap       : {max_gap} "
              f"({gap_start.date()} → {gap_end.date()})")

    return df


def main():
    parser = argparse.ArgumentParser(description="Load and clean raw Forex CSV")
    parser.add_argument("--csv",    default=None, help="Path to input CSV file")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--out",    default=None, help="Output Parquet path")
    args = parser.parse_args()

    cfg = _load_config(args.config)

    csv_path = Path(args.csv or cfg["data"]["csv_path"])
    out_path = Path(args.out or "data/raw/EURUSD_M15.parquet")

    df = load_csv(csv_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[csv_loader] Saved → {out_path}  ({len(df):,} rows)")


if __name__ == "__main__":
    main()
