"""
labeler.py — Phase 1, Step 3

Generates triple-barrier labels for every candle in the feature DataFrame.

For each candle at time T, a simulated trade is entered at close[T].
Two price barriers are placed at:
    upper = close[T] + k * ATR_14[T]    ← take-profit → label BUY  (2)
    lower = close[T] - k * ATR_14[T]    ← stop-loss   → label SELL (0)

The script looks forward at most H candles (the time barrier).
Whichever barrier is hit first determines the label.
If neither is hit within H candles, the label is HOLD (1).

Label encoding
--------------
    0 = SELL
    1 = HOLD
    2 = BUY

Usage
-----
    python -m src.data.labeler
    python -m src.data.labeler --k 1.0 --horizon 4
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def _load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def generate_labels(df: pd.DataFrame, k: float, horizon: int) -> pd.DataFrame:
    """
    Add a 'label' column to df using the triple-barrier method.

    Parameters
    ----------
    df      : feature DataFrame containing close, high, low, ATR_14 columns
    k       : ATR multiplier — barrier distance = k * ATR_14
    horizon : max candles to look ahead before assigning HOLD

    Returns
    -------
    DataFrame with 'label' column added.
    The last `horizon` rows are dropped (no future data available for them).
    """
    n = len(df)

    closes  = df["close"].to_numpy()
    highs   = df["high"].to_numpy()
    lows    = df["low"].to_numpy()
    atrs    = df["ATR_14"].to_numpy()

    labels = np.full(n, 1, dtype=np.int8)   # default: HOLD

    # Only label rows that have at least `horizon` future candles
    labelable_end = n - horizon

    for i in range(labelable_end):
        entry      = closes[i]
        upper      = entry + k * atrs[i]
        lower      = entry - k * atrs[i]

        label_set = False
        for j in range(i + 1, i + horizon + 1):
            if highs[j] >= upper:
                labels[i] = 2   # BUY
                label_set = True
                break
            if lows[j] <= lower:
                labels[i] = 0   # SELL
                label_set = True
                break
        # If neither barrier hit: label stays HOLD (1)

    df = df.copy()
    df["label"] = labels

    # Drop the last `horizon` rows — they have no reliable future data
    df = df.iloc[:labelable_end].reset_index(drop=True)

    return df


def print_distribution(df: pd.DataFrame, k: float) -> None:
    counts = df["label"].value_counts().sort_index()
    total  = len(df)

    label_names = {0: "SELL", 1: "HOLD", 2: "BUY"}
    print("\n[labeler] Label Distribution:")
    print(f"          k={k}  |  horizon={len(df)} rows labeled")
    print(f"          {'Label':<8} {'Count':>8}  {'%':>6}")
    print(f"          {'-'*28}")
    for code, name in label_names.items():
        cnt = counts.get(code, 0)
        pct = 100 * cnt / total
        print(f"          {name:<8} {cnt:>8,}  {pct:>5.1f}%")
    print(f"          {'Total':<8} {total:>8,}  100.0%")

    directional_pct = 100 * (counts.get(0, 0) + counts.get(2, 0)) / total
    status = "GOOD" if 40 <= directional_pct <= 60 else "ADJUST k"
    print(f"\n          BUY + SELL combined: {directional_pct:.1f}%  ← {status}"
          f"  (target: 40–60%)")

    if directional_pct < 40:
        print(f"          → k={k} is too large. Try decreasing k (e.g. k=0.50).")
    elif directional_pct > 60:
        print(f"          → k={k} is too small. Try increasing k (e.g. k=1.00).\n")
    else:
        print()


def main():
    parser = argparse.ArgumentParser(description="Generate triple-barrier labels")
    parser.add_argument("--in",      dest="infile",  default=None)
    parser.add_argument("--out",     dest="outfile", default=None)
    parser.add_argument("--config",  default="config/config.yaml")
    parser.add_argument("--k",       type=float, default=None,
                        help="Override ATR multiplier from config")
    parser.add_argument("--horizon", type=int,   default=None,
                        help="Override time horizon from config")
    args = parser.parse_args()

    cfg = _load_config(args.config)

    k       = args.k       or cfg["labeling"]["k"]
    horizon = args.horizon or cfg["labeling"]["horizon"]

    in_path  = Path(args.infile  or "data/raw/EURUSD_M15_features.parquet")
    out_path = Path(args.outfile or "data/raw/EURUSD_M15_labeled.parquet")

    if not in_path.exists():
        print(f"[ERROR] Input not found: {in_path}")
        print("        Run features.py first.")
        raise SystemExit(1)

    print(f"[labeler] Loading {in_path}")
    df = pd.read_parquet(in_path)
    print(f"[labeler] Loaded {len(df):,} rows")
    print(f"[labeler] Parameters: k={k}, horizon={horizon} candles")

    df = generate_labels(df, k=k, horizon=horizon)

    print_distribution(df, k)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[labeler] Saved → {out_path}  ({len(df):,} labeled rows)")
    print()
    print("[labeler] NEXT: Run sanity check on a few BUY and SELL rows,")
    print("          then proceed to Phase 2: tokenizer.py")


if __name__ == "__main__":
    main()
