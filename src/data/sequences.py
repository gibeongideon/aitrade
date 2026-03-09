"""
sequences.py — Phase 2, Step 2

Slides a 64-candle window over the tokenized DataFrame and builds
flat integer sequences ready for the transformer encoder.

Each sequence has shape [max_seq_len=335] and is structured as:
    Position 0     : [CLS] token (ID = 1)
    Positions 1–320: 64 candles × 5 tokens (oldest candle first)
    Positions 321–334: [PAD] tokens (ID = 0) — 14 padding tokens

The label is the integer label of the LAST candle in the window (the
candle being predicted).

Split assignment (no cross-contamination):
    A sequence whose last candle is at row i is assigned to:
        train : i <  train_end
        val   : train_end + 63 <= i < val_end   (skip 63 warm-up sequences)
        test  : val_end   + 63 <= i              (same)
    Sequences that span a split boundary are excluded entirely.

Output
------
    data/processed/train_sequences.pt   — TensorDataset (input_ids, labels)
    data/processed/val_sequences.pt
    data/processed/test_sequences.pt

Usage
-----
    python -m src.data.sequences
    python -m src.data.sequences --in data/raw/EURUSD_M15_tokenized.parquet
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
import yaml


CLS_ID  = 1
PAD_ID  = 0
TOKENS_PER_CANDLE = 5
TOKEN_COLS = ["ret_token", "body_token", "wick_token", "vol_token", "trend_token"]


def _load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_sequences(token_array: np.ndarray, labels: np.ndarray,
                    context: int, max_seq_len: int,
                    start: int, end: int) -> tuple:
    """
    Build sequences for rows in [start, end) where start >= context-1
    so that every window fits within the array.

    Parameters
    ----------
    token_array : shape (N, 5) — integer token IDs per candle
    labels      : shape (N,)   — integer labels per candle
    context     : number of candles per sequence (64)
    max_seq_len : total token length including CLS and padding (335)
    start       : first eligible last-candle index (inclusive)
    end         : one past the last eligible last-candle index

    Returns
    -------
    (input_ids_tensor, labels_tensor) both CPU tensors
    """
    content_len = context * TOKENS_PER_CANDLE   # 320
    pad_len     = max_seq_len - 1 - content_len  # 14

    n_samples = end - start
    seq_array = np.zeros((n_samples, max_seq_len), dtype=np.int32)
    lbl_array = np.empty(n_samples, dtype=np.int64)

    for k, i in enumerate(range(start, end)):
        # Rows i-(context-1) through i
        window = token_array[i - context + 1 : i + 1]   # shape (64, 5)
        flat   = window.ravel()                          # shape (320,)

        seq_array[k, 0]              = CLS_ID
        seq_array[k, 1: 1 + content_len] = flat
        # Remaining positions already 0 (PAD) from np.zeros
        lbl_array[k] = labels[i]

    input_ids = torch.tensor(seq_array, dtype=torch.long)
    lbls      = torch.tensor(lbl_array, dtype=torch.long)
    return input_ids, lbls


def _label_dist(labels_tensor: torch.Tensor) -> str:
    total = len(labels_tensor)
    counts = {v: (labels_tensor == v).sum().item() for v in [0, 1, 2]}
    return (f"SELL {100*counts[0]/total:.1f}%  "
            f"HOLD {100*counts[1]/total:.1f}%  "
            f"BUY {100*counts[2]/total:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Build sequence .pt files")
    parser.add_argument("--in",     dest="infile",  default=None)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    cfg = _load_config(args.config)

    in_path = Path(args.infile or "data/raw/EURUSD_M15_tokenized.parquet")
    out_dir = Path(args.outdir or "data/processed")

    if not in_path.exists():
        print(f"[ERROR] Input not found: {in_path}")
        print("        Run tokenizer.py first.")
        raise SystemExit(1)

    context     = cfg["data"]["context_window"]         # 64
    max_seq_len = cfg["model"]["max_seq_len"]            # 335
    train_ratio = cfg["data"]["train_ratio"]             # 0.70
    val_ratio   = cfg["data"]["val_ratio"]               # 0.15

    print(f"[sequences] Loading {in_path}")
    df = pd.read_parquet(in_path)
    n  = len(df)
    print(f"[sequences] Loaded {n:,} rows")

    # Verify token columns are present
    missing = [c for c in TOKEN_COLS if c not in df.columns]
    if missing:
        print(f"[ERROR] Missing token columns: {missing}")
        print("        Run tokenizer.py first.")
        raise SystemExit(1)

    # ── Extract arrays ────────────────────────────────────────────────────────
    token_array = df[TOKEN_COLS].to_numpy(dtype=np.int32)  # (N, 5)
    labels      = df["label"].to_numpy(dtype=np.int64)     # (N,)

    # ── Row-level split indices ───────────────────────────────────────────────
    train_end = int(n * train_ratio)
    val_end   = int(n * (train_ratio + val_ratio))

    # Sequences need context-1 prior rows, so valid last-candle indices are:
    #   train : [context-1,  train_end)
    #   val   : [train_end + context-1,  val_end)     ← skip context-1 warm-up
    #   test  : [val_end   + context-1,  n)

    warmup = context - 1   # 63

    splits = {
        "train": (warmup,              train_end),
        "val":   (train_end + warmup,  val_end),
        "test":  (val_end   + warmup,  n),
    }

    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, (start, end) in splits.items():
        n_samples = max(0, end - start)
        print(f"\n[sequences] Building {split_name} — rows {start}..{end-1} "
              f"({n_samples:,} sequences)")

        if n_samples <= 0:
            print(f"[sequences] WARNING: no samples for {split_name} split — skipping")
            continue

        input_ids, lbls = build_sequences(
            token_array, labels, context, max_seq_len, start, end)

        dataset   = TensorDataset(input_ids, lbls)
        out_path  = out_dir / f"{split_name}_sequences.pt"
        torch.save(dataset, out_path)

        print(f"[sequences] {split_name}: {n_samples:,} samples  |  "
              f"{_label_dist(lbls)}")
        print(f"[sequences] Saved → {out_path}")

        # Quick sanity check on first sequence
        if split_name == "train":
            ids0 = input_ids[0]
            assert ids0[0].item() == CLS_ID, "First token must be CLS"
            assert all(ids0[1:1 + context * TOKENS_PER_CANDLE] >= 2), \
                "Content tokens must be >= 2"
            assert all(ids0[1 + context * TOKENS_PER_CANDLE:] == PAD_ID), \
                "Padding tokens must be 0"
            assert ids0.shape[0] == max_seq_len, \
                f"Sequence length must be {max_seq_len}"
            print(f"[sequences] Sanity check passed — "
                  f"seq shape {list(ids0.shape)}, "
                  f"CLS={ids0[0].item()}, "
                  f"first content={ids0[1:6].tolist()}, "
                  f"padding[0]={ids0[321].item()}")

    print()
    print("[sequences] NEXT: Run trainer.py to train the transformer model")


if __name__ == "__main__":
    main()
