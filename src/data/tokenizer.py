"""
tokenizer.py — Phase 2, Step 1

Converts the labeled feature DataFrame into discrete market-language token IDs.
Each candle is mapped to exactly 5 token IDs:
    ret_token, body_token, wick_token, vol_token, trend_token

Token ID table
--------------
  0  = [PAD]
  1  = [CLS]
  2  = RET_D3         (strong downward move, bottom 5%)
  3  = RET_D2
  4  = RET_D1
  5  = RET_FLAT
  6  = RET_U1
  7  = RET_U2
  8  = RET_U3         (strong upward move, top 5%)
  9  = BODY_S         (small body — indecision)
 10  = BODY_M
 11  = BODY_L         (large body — conviction)
 12  = WICK_NONE
 13  = WICK_TOP       (upper wick significant — bearish rejection)
 14  = WICK_BOTTOM    (lower wick significant — bullish rejection)
 15  = WICK_BOTH      (wicks on both sides — indecision)
 16  = VOL_LOW
 17  = VOL_NORMAL
 18  = VOL_HIGH
 19  = VOL_SPIKE      (top 10% of volume)
 20  = TREND_UP       (MA_16 > MA_32 > MA_64)
 21  = TREND_DOWN     (MA_16 < MA_32 < MA_64)
 22  = TREND_MIX      (MAs not aligned)
 23  = TREND_CROSS_UP (MA_16 just crossed above MA_64)
 24  = TREND_CROSS_DOWN

Thresholds are computed from the training split ONLY, then saved to
models/tokenizer_thresholds.json for reuse at inference time.

Usage
-----
    python -m src.data.tokenizer
    python -m src.data.tokenizer --in data/raw/EURUSD_M15_labeled.parquet
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ── Token name map for frequency reporting ────────────────────────────────────
TOKEN_NAMES = {
    2: "RET_D3",         3: "RET_D2",     4: "RET_D1",    5: "RET_FLAT",
    6: "RET_U1",         7: "RET_U2",     8: "RET_U3",
    9: "BODY_S",        10: "BODY_M",    11: "BODY_L",
   12: "WICK_NONE",     13: "WICK_TOP",  14: "WICK_BOTTOM", 15: "WICK_BOTH",
   16: "VOL_LOW",       17: "VOL_NORMAL", 18: "VOL_HIGH",  19: "VOL_SPIKE",
   20: "TREND_UP",      21: "TREND_DOWN", 22: "TREND_MIX",
   23: "TREND_CROSS_UP", 24: "TREND_CROSS_DOWN",
}

TOKEN_COL = {
    **{tid: "ret_token"   for tid in range(2, 9)},
    **{tid: "body_token"  for tid in range(9, 12)},
    **{tid: "wick_token"  for tid in range(12, 16)},
    **{tid: "vol_token"   for tid in range(16, 20)},
    **{tid: "trend_token" for tid in range(20, 25)},
}


def _load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── Threshold computation ─────────────────────────────────────────────────────

def compute_thresholds(train_df: pd.DataFrame, wick_min_atr_fraction: float) -> dict:
    """
    Compute quantile-based thresholds from training rows only.
    Returns a dict suitable for JSON serialisation and inference reuse.
    """
    ret  = train_df["ret"].to_numpy()
    body = train_df["body_ratio"].to_numpy()
    vol  = train_df["vol_ratio"].to_numpy()

    return {
        "ret_boundaries":        np.percentile(ret,  [5, 20, 40, 60, 80, 95]).tolist(),
        "body_boundaries":       np.percentile(body, [33, 67]).tolist(),
        "vol_boundaries":        np.percentile(vol,  [25, 75, 90]).tolist(),
        "wick_min_atr_fraction": wick_min_atr_fraction,
    }


# ── Per-feature tokenisers ─────────────────────────────────────────────────────

def _tokenize_ret(ret: np.ndarray, boundaries: list) -> np.ndarray:
    b = boundaries
    tokens = np.full(len(ret), 5, dtype=np.int16)   # default: RET_FLAT
    tokens[ret < b[0]] = 2
    tokens[(ret >= b[0]) & (ret < b[1])] = 3
    tokens[(ret >= b[1]) & (ret < b[2])] = 4
    # b[2]..b[3] → stays 5 (FLAT)
    tokens[(ret >= b[3]) & (ret < b[4])] = 6
    tokens[(ret >= b[4]) & (ret < b[5])] = 7
    tokens[ret >= b[5]] = 8
    return tokens


def _tokenize_body(body: np.ndarray, boundaries: list) -> np.ndarray:
    b = boundaries
    tokens = np.where(body < b[0], 9, np.where(body < b[1], 10, 11))
    return tokens.astype(np.int16)


def _tokenize_wick(upper: np.ndarray, lower: np.ndarray,
                   atr: np.ndarray, frac: float) -> np.ndarray:
    wick_min   = frac * atr
    upper_sig  = upper > wick_min
    lower_sig  = lower > wick_min
    tokens = np.full(len(upper), 12, dtype=np.int16)   # WICK_NONE
    tokens[ upper_sig & ~lower_sig] = 13   # WICK_TOP
    tokens[~upper_sig &  lower_sig] = 14   # WICK_BOTTOM
    tokens[ upper_sig &  lower_sig] = 15   # WICK_BOTH
    return tokens


def _tokenize_vol(vol: np.ndarray, boundaries: list) -> np.ndarray:
    b = boundaries
    tokens = np.full(len(vol), 17, dtype=np.int16)   # VOL_NORMAL
    tokens[vol < b[0]] = 16
    tokens[(vol >= b[1]) & (vol < b[2])] = 18
    tokens[vol >= b[2]] = 19
    return tokens


def _tokenize_trend(ma16: np.ndarray, ma32: np.ndarray,
                    ma64: np.ndarray, ma_cross: np.ndarray) -> np.ndarray:
    tokens = np.full(len(ma16), 22, dtype=np.int16)   # TREND_MIX
    tokens[(ma16 > ma32) & (ma32 > ma64)] = 20   # TREND_UP
    tokens[(ma16 < ma32) & (ma32 < ma64)] = 21   # TREND_DOWN
    # Crossover takes priority — overwrite alignment
    tokens[ma_cross ==  1] = 23   # TREND_CROSS_UP
    tokens[ma_cross == -1] = 24   # TREND_CROSS_DOWN
    return tokens


# ── Main tokenisation entry point ─────────────────────────────────────────────

def apply_tokenization(df: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """
    Vectorised tokenisation of all rows using pre-computed thresholds.
    Adds five new columns: ret_token, body_token, wick_token, vol_token, trend_token.
    Does NOT modify the input DataFrame.
    """
    df = df.copy()
    t  = thresholds

    df["ret_token"]   = _tokenize_ret(
        df["ret"].to_numpy(), t["ret_boundaries"])
    df["body_token"]  = _tokenize_body(
        df["body_ratio"].to_numpy(), t["body_boundaries"])
    df["wick_token"]  = _tokenize_wick(
        df["upper_wick"].to_numpy(), df["lower_wick"].to_numpy(),
        df["ATR_14"].to_numpy(), t["wick_min_atr_fraction"])
    df["vol_token"]   = _tokenize_vol(
        df["vol_ratio"].to_numpy(), t["vol_boundaries"])
    df["trend_token"] = _tokenize_trend(
        df["MA_16"].to_numpy(), df["MA_32"].to_numpy(),
        df["MA_64"].to_numpy(), df["ma_cross"].to_numpy())

    return df


def print_token_frequencies(df: pd.DataFrame) -> None:
    """Print per-token frequency table across the full tokenised DataFrame."""
    n = len(df)
    print(f"\n[tokenizer] Token Frequency Check — {n:,} rows (should all be > 1%):")
    print(f"  {'Token':<22} {'Count':>8}  {'%':>6}  Status")
    print(f"  {'-'*46}")

    ok = True
    for tid, name in TOKEN_NAMES.items():
        col  = TOKEN_COL[tid]
        cnt  = (df[col] == tid).sum()
        pct  = 100 * cnt / n
        flag = "OK" if pct >= 1.0 else "WARN — too rare"
        if pct < 1.0:
            ok = False
        print(f"  {name:<22} {cnt:>8,}  {pct:>5.1f}%  {flag}")

    if ok:
        print("\n[tokenizer] All tokens above 1% threshold — OK")
    else:
        print("\n[tokenizer] WARNING: some tokens are below 1%. Review thresholds.")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Tokenise labeled Forex features")
    parser.add_argument("--in",     dest="infile",  default=None)
    parser.add_argument("--out",    dest="outfile", default=None)
    parser.add_argument("--thresh", dest="thresh",  default=None,
                        help="Output path for tokenizer_thresholds.json")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    cfg = _load_config(args.config)

    in_path     = Path(args.infile  or "data/raw/EURUSD_M15_labeled.parquet")
    out_path    = Path(args.outfile or "data/raw/EURUSD_M15_tokenized.parquet")
    thresh_path = Path(args.thresh  or "models/tokenizer_thresholds.json")

    if not in_path.exists():
        print(f"[ERROR] Input not found: {in_path}")
        print("        Run labeler.py first.")
        raise SystemExit(1)

    print(f"[tokenizer] Loading {in_path}")
    df = pd.read_parquet(in_path)
    n  = len(df)
    print(f"[tokenizer] Loaded {n:,} rows")

    # ── Time-based split (no shuffle — prevents look-ahead bias) ──────────────
    train_ratio = cfg["data"]["train_ratio"]
    val_ratio   = cfg["data"]["val_ratio"]

    train_end = int(n * train_ratio)
    val_end   = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end]

    print(f"[tokenizer] Split sizes — train: {train_end:,}  "
          f"val: {val_end - train_end:,}  test: {n - val_end:,}")

    # ── Compute thresholds on training data only ──────────────────────────────
    wick_frac = cfg["tokenizer"]["wick_min_atr_fraction"]
    thresholds = compute_thresholds(train_df, wick_frac)

    print(f"[tokenizer] RET  boundaries : {[f'{v:.5f}' for v in thresholds['ret_boundaries']]}")
    print(f"[tokenizer] BODY boundaries : {[f'{v:.4f}' for v in thresholds['body_boundaries']]}")
    print(f"[tokenizer] VOL  boundaries : {[f'{v:.4f}' for v in thresholds['vol_boundaries']]}")
    print(f"[tokenizer] WICK min frac   : {wick_frac}")

    # ── Save thresholds ───────────────────────────────────────────────────────
    thresh_path.parent.mkdir(parents=True, exist_ok=True)
    with open(thresh_path, "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"[tokenizer] Thresholds saved → {thresh_path}")

    # ── Tokenise full dataset using training thresholds ───────────────────────
    df = apply_tokenization(df, thresholds)

    print_token_frequencies(df)

    # ── Save tokenized parquet ────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"\n[tokenizer] Saved → {out_path}  ({len(df):,} rows)")
    print()
    print("[tokenizer] NEXT: Run sequences.py to build train/val/test .pt files")


if __name__ == "__main__":
    main()
