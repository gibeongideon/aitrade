"""
labeler.py — Phase 1, Step 3

Generates NON-OVERLAPPING triple-barrier labels using event filtering.

Why non-overlapping?
--------------------
The naive approach labels EVERY candle by looking H candles ahead:
    T1 → checks T2, T3, T4
    T2 → checks T3, T4, T5   ← T3, T4 reused
    T3 → checks T4, T5, T6   ← T4, T5 reused

This means the model can indirectly "see" future candles through shared
label targets — a form of data leakage. In live trading you cannot open
a new trade while the previous one is still running.

The correct approach (López de Prado, "Advances in Financial Machine Learning"):
    T1 → trade exits at T3 (barrier hit)
    Next trade allowed → T4   ← first candle after exit
    T4 → trade exits at T7
    Next trade → T8
    ...

Each labeled candle uses a completely independent set of future candles.

Label encoding
--------------
    -1 = not an event (candle is skipped — used as context only)
     0 = SELL (lower barrier hit first)
     1 = HOLD (time barrier — neither hit within H candles)
     2 = BUY  (upper barrier hit first)

Output columns added
--------------------
    label    : -1 for non-events, 0/1/2 for events
    exit_idx : row index of exit candle (-1 for non-events)
               Used by sequences.py for purging at split boundaries.

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
    Add 'label' and 'exit_idx' columns using non-overlapping triple-barrier.

    For each event at candle i:
        upper = close[i] + k * ATR_14[i]
        lower = close[i] - k * ATR_14[i]
        Look forward: whichever barrier is hit first within H candles → label
        Next event starts at exit_candle + 1 (no overlap)

    Non-event candles get label = -1 and exit_idx = -1.

    Parameters
    ----------
    df      : feature DataFrame with close, high, low, ATR_14 columns
    k       : ATR multiplier for barrier distance
    horizon : max candles to look ahead (time barrier)

    Returns
    -------
    DataFrame with 'label' and 'exit_idx' columns added (all rows kept).
    """
    n = len(df)

    closes = df["close"].to_numpy()
    highs  = df["high"].to_numpy()
    lows   = df["low"].to_numpy()
    atrs   = df["ATR_14"].to_numpy()

    labels   = np.full(n, -1, dtype=np.int8)   # -1 = not an event
    exit_idx = np.full(n, -1, dtype=np.int32)

    i = 0
    event_count = 0

    while i <= n - horizon - 1:
        entry = closes[i]
        upper = entry + k * atrs[i]
        lower = entry - k * atrs[i]

        label_val = 1           # HOLD (time barrier default)
        exit_j    = i + horizon # default exit: time barrier

        for j in range(i + 1, i + horizon + 1):
            if highs[j] >= upper:
                label_val = 2   # BUY
                exit_j    = j
                break
            if lows[j] <= lower:
                label_val = 0   # SELL
                exit_j    = j
                break

        labels[i]   = label_val
        exit_idx[i] = exit_j
        event_count += 1

        # ── KEY: next event starts AFTER this trade exits ─────────────────
        i = exit_j + 1

    df = df.copy()
    df["label"]    = labels
    df["exit_idx"] = exit_idx

    return df


def print_distribution(df: pd.DataFrame, k: float) -> None:
    events = df[df["label"] >= 0]
    total  = len(events)

    if total == 0:
        print("[labeler] No events found — check k and horizon values.")
        return

    counts = events["label"].value_counts().sort_index()
    label_names = {0: "SELL", 1: "HOLD", 2: "BUY"}

    print("\n[labeler] Label Distribution (non-overlapping events only):")
    print(f"          k={k}  |  {total:,} events from {len(df):,} total candles")
    print(f"          Event density: {100*total/len(df):.1f}% of candles are events")
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
        print(f"          → k={k} is too large. Barriers too far — most expire HOLD.")
        print(f"          → Try decreasing k (e.g. k=0.80).")
    elif directional_pct > 60:
        print(f"          → k={k} is too small. Barriers too close.")
        print(f"          → Try increasing k (e.g. k=1.50).")
    else:
        print()


def main():
    parser = argparse.ArgumentParser(description="Generate non-overlapping triple-barrier labels")
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
    print(f"[labeler] Mode: NON-OVERLAPPING events (no label reuse of future candles)")

    df = generate_labels(df, k=k, horizon=horizon)

    print_distribution(df, k)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    event_count = (df["label"] >= 0).sum()
    print(f"[labeler] Saved → {out_path}")
    print(f"          Total rows (all candles): {len(df):,}")
    print(f"          Event rows (labeled):     {event_count:,}")
    print(f"          Non-event rows (context): {len(df) - event_count:,}")
    print()
    print("[labeler] NEXT: Run tokenizer.py, then sequences.py")
    print("          sequences.py will only build sequences at event rows.")


if __name__ == "__main__":
    main()
