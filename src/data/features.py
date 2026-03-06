"""
features.py — Phase 1, Step 2

Reads the clean OHLCV Parquet produced by csv_loader.py and computes all
derived features needed for tokenisation and labelling.

New columns added
-----------------
    ret         — candle return: (close - open) / open
    body_ratio  — body size as fraction of total range: abs(c-o)/(h-l)
    upper_wick  — upper shadow: high - max(open, close)
    lower_wick  — lower shadow: min(open, close) - low
    MA_16       — 16-period simple moving average of close
    MA_32       — 32-period simple moving average of close
    MA_64       — 64-period simple moving average of close
    ATR_14      — 14-period Average True Range
    vol_ratio   — volume / 20-period rolling mean of volume
    ma_cross    — +1 bull cross, -1 bear cross, 0 no cross (MA_16 vs MA_64)

Usage
-----
    python -m src.data.features
    python -m src.data.features --in data/raw/EURUSD_M15.parquet
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def _load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def compute_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Accept a clean OHLCV DataFrame and return a new DataFrame with all
    derived feature columns appended.  Input DataFrame is not modified.
    """
    df = df.copy()

    ma_periods  = cfg["features"]["ma_periods"]   # [16, 32, 64]
    atr_period  = cfg["features"]["atr_period"]   # 14
    vol_lookback = cfg["features"]["vol_lookback"] # 20

    # ── 1. Candle return ──────────────────────────────────────────────────────
    # Percentage change from open to close within the candle.
    df["ret"] = (df["close"] - df["open"]) / df["open"]

    # ── 2. Body ratio ─────────────────────────────────────────────────────────
    # What fraction of the candle's full range is the body?
    # Handles zero-range candles (doji with high==low) safely.
    candle_range = df["high"] - df["low"]
    body_size    = (df["close"] - df["open"]).abs()
    df["body_ratio"] = np.where(candle_range > 0, body_size / candle_range, 0.0)

    # ── 3. Wicks ──────────────────────────────────────────────────────────────
    top_of_body    = df[["open", "close"]].max(axis=1)
    bottom_of_body = df[["open", "close"]].min(axis=1)
    df["upper_wick"] = df["high"] - top_of_body
    df["lower_wick"] = bottom_of_body - df["low"]

    # ── 4. Moving averages ────────────────────────────────────────────────────
    for p in ma_periods:
        df[f"MA_{p}"] = df["close"].rolling(window=p, min_periods=p).mean()

    # ── 5. Average True Range (ATR) ───────────────────────────────────────────
    # True Range = max of three measures to capture gap moves.
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)

    df["ATR_14"] = tr.rolling(window=atr_period, min_periods=atr_period).mean()

    # ── 6. Volume ratio ───────────────────────────────────────────────────────
    # Current volume relative to recent rolling average.
    vol_ma = df["volume"].rolling(window=vol_lookback, min_periods=vol_lookback).mean()
    df["vol_ratio"] = df["volume"] / vol_ma

    # ── 7. MA crossover ───────────────────────────────────────────────────────
    # Detect when MA_16 crosses MA_64.
    # +1 = bullish cross (16 moved from below to above 64)
    # -1 = bearish cross (16 moved from above to below 64)
    #  0 = no cross this candle
    short_ma = df["MA_16"]
    long_ma  = df["MA_64"]

    prev_short = short_ma.shift(1)
    prev_long  = long_ma.shift(1)

    bull_cross = (prev_short <  prev_long) & (short_ma >= long_ma)
    bear_cross = (prev_short >  prev_long) & (short_ma <= long_ma)

    df["ma_cross"] = 0
    df.loc[bull_cross, "ma_cross"] = 1
    df.loc[bear_cross, "ma_cross"] = -1

    # ── 8. Drop warm-up NaN rows ──────────────────────────────────────────────
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    dropped = before - len(df)

    print(f"[features] Rows before NaN drop : {before:,}")
    print(f"[features] Warm-up rows dropped : {dropped:,}  (expected ~{max(ma_periods)})")
    print(f"[features] Clean feature rows   : {len(df):,}")
    print(f"[features] Columns              : {list(df.columns)}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Compute derived features")
    parser.add_argument("--in",     dest="infile",  default=None)
    parser.add_argument("--out",    dest="outfile", default=None)
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    cfg = _load_config(args.config)

    in_path  = Path(args.infile  or "data/raw/EURUSD_M15.parquet")
    out_path = Path(args.outfile or "data/raw/EURUSD_M15_features.parquet")

    if not in_path.exists():
        print(f"[ERROR] Input not found: {in_path}")
        print("        Run csv_loader.py first.")
        raise SystemExit(1)

    print(f"[features] Loading {in_path}")
    df = pd.read_parquet(in_path)
    print(f"[features] Loaded {len(df):,} rows")

    df = compute_features(df, cfg)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[features] Saved → {out_path}")


if __name__ == "__main__":
    main()
