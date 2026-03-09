"""
sequences.py — Phase 2, Step 2

Builds input sequences from non-overlapping labeled events only.

Key difference from naive sliding-window approach
-------------------------------------------------
The labeler marks only non-overlapping event rows with label 0/1/2.
All other rows have label = -1 (context candles, not prediction targets).

sequences.py builds one sequence PER EVENT ROW — not per candle.
This ensures the training set contains no shared future candles between
any two samples, eliminating label leakage.

Split assignment with purging
------------------------------
An event at row i with exit at row exit_j belongs to:
    train  : i >= context-1  AND  exit_j < train_end
    val    : i >= train_end  AND  exit_j < val_end
    test   : i >= val_end    AND  i < n

Events where exit_j crosses a split boundary are purged (excluded) to
prevent training labels from using val/test candles as future data.

Sequence structure (identical to v1)
-------------------------------------
    Position 0     : [CLS] token (ID = 1)
    Positions 1–320: 64 candles × 5 tokens (oldest first)
    Positions 321–334: [PAD] (ID = 0) — 14 padding tokens
    Total length   : 335 tokens

Output
------
    data/processed/train_sequences.pt
    data/processed/val_sequences.pt
    data/processed/test_sequences.pt

Each .pt file is a TensorDataset of (input_ids [335], label [scalar]).

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


CLS_ID            = 1
PAD_ID            = 0
TOKENS_PER_CANDLE = 5
TOKEN_COLS        = ["ret_token", "body_token", "wick_token", "vol_token", "trend_token"]


def _load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_sequences(token_array: np.ndarray,
                    event_indices: np.ndarray,
                    labels: np.ndarray,
                    context: int,
                    max_seq_len: int) -> tuple:
    """
    Build sequences for a set of event row indices.

    Parameters
    ----------
    token_array   : shape (N, 5) — token IDs for every candle
    event_indices : row indices of event candles to build sequences for
    labels        : shape (N,) — label per candle (-1 for non-events)
    context       : candles per sequence (64)
    max_seq_len   : total sequence length including CLS and PAD (335)

    Returns
    -------
    (input_ids_tensor [n_events, 335], labels_tensor [n_events])
    """
    content_len = context * TOKENS_PER_CANDLE   # 320

    n = len(event_indices)
    seq_array = np.zeros((n, max_seq_len), dtype=np.int32)
    lbl_array = np.empty(n, dtype=np.int64)

    for k, i in enumerate(event_indices):
        window = token_array[i - context + 1 : i + 1]   # (64, 5)
        flat   = window.ravel()                          # (320,)
        seq_array[k, 0]               = CLS_ID
        seq_array[k, 1: 1+content_len] = flat
        lbl_array[k] = labels[i]

    input_ids = torch.tensor(seq_array, dtype=torch.long)
    lbls      = torch.tensor(lbl_array, dtype=torch.long)
    return input_ids, lbls


def _label_dist(labels_tensor: torch.Tensor) -> str:
    total = len(labels_tensor)
    if total == 0:
        return "empty"
    counts = {v: (labels_tensor == v).sum().item() for v in [0, 1, 2]}
    return (f"SELL {100*counts[0]/total:.1f}%  "
            f"HOLD {100*counts[1]/total:.1f}%  "
            f"BUY {100*counts[2]/total:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Build sequence .pt files from events")
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

    context     = cfg["data"]["context_window"]       # 64
    max_seq_len = cfg["model"]["max_seq_len"]          # 335
    train_ratio = cfg["data"]["train_ratio"]           # 0.70
    val_ratio   = cfg["data"]["val_ratio"]             # 0.15

    print(f"[sequences] Loading {in_path}")
    df = pd.read_parquet(in_path)
    n  = len(df)
    print(f"[sequences] Loaded {n:,} rows")

    # Verify required columns
    for col in TOKEN_COLS + ["label", "exit_idx"]:
        if col not in df.columns:
            print(f"[ERROR] Missing column: '{col}'")
            print("        Re-run labeler.py and tokenizer.py.")
            raise SystemExit(1)

    token_array = df[TOKEN_COLS].to_numpy(dtype=np.int32)  # (N, 5)
    labels      = df["label"].to_numpy(dtype=np.int64)     # (N,) — -1 for non-events
    exit_idxs   = df["exit_idx"].to_numpy(dtype=np.int32)  # (N,) — exit candle index

    # ── Split boundaries ──────────────────────────────────────────────────────
    train_end = int(n * train_ratio)
    val_end   = int(n * (train_ratio + val_ratio))
    warmup    = context - 1   # 63 — need 63 prior candles for context window

    total_events = (labels >= 0).sum()
    print(f"[sequences] Total events (non-overlapping): {total_events:,}")
    print(f"[sequences] Split boundaries — train_end={train_end:,}  val_end={val_end:,}")

    # ── Per-split event selection with purging ────────────────────────────────
    # An event at row i is valid for a split if:
    #   1. i >= warmup (enough context candles before it)
    #   2. Its label is a real event (>= 0)
    #   3. Its exit_idx does not cross the split boundary (purging)

    def select_events(start_i, end_i, exit_boundary):
        """
        Return event row indices where:
            start_i <= i < end_i           (within split row range)
            i >= warmup                    (enough context)
            labels[i] >= 0                 (is a labeled event)
            exit_idxs[i] < exit_boundary   (purging: exit must stay in split)
        """
        mask = (
            (np.arange(n) >= max(warmup, start_i)) &
            (np.arange(n) < end_i) &
            (labels >= 0) &
            (exit_idxs < exit_boundary)
        )
        return np.where(mask)[0]

    split_defs = {
        "train": select_events(0,         train_end, train_end),
        "val":   select_events(train_end, val_end,   val_end),
        "test":  select_events(val_end,   n,         n),
    }

    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, event_idx in split_defs.items():
        n_events = len(event_idx)
        print(f"\n[sequences] {split_name}: {n_events:,} events")

        if n_events == 0:
            print(f"[sequences] WARNING: no events for {split_name} — skipping")
            continue

        input_ids, lbls = build_sequences(
            token_array, event_idx, labels, context, max_seq_len)

        dataset  = TensorDataset(input_ids, lbls)
        out_path = out_dir / f"{split_name}_sequences.pt"
        torch.save(dataset, out_path)

        print(f"[sequences] {split_name}: {n_events:,} sequences  |  {_label_dist(lbls)}")
        print(f"[sequences] Saved → {out_path}")

        # Sanity check on first sequence of train
        if split_name == "train":
            ids0 = input_ids[0]
            assert ids0[0].item() == CLS_ID, "First token must be CLS"
            content = ids0[1 : 1 + context * TOKENS_PER_CANDLE]
            assert content.min().item() >= 2, "Content tokens must be >= 2"
            padding = ids0[1 + context * TOKENS_PER_CANDLE:]
            assert padding.max().item() == PAD_ID, "Padding must be 0"
            assert ids0.shape[0] == max_seq_len, f"Seq length must be {max_seq_len}"
            print(f"[sequences] Sanity check passed — "
                  f"shape={list(ids0.shape)}, "
                  f"CLS={ids0[0].item()}, "
                  f"first_content={ids0[1:6].tolist()}, "
                  f"pad_start={ids0[321].item()}")

    print()
    print("[sequences] Non-overlapping sequences built. No label reuse between samples.")
    print("[sequences] NEXT: Run trainer.py to train the transformer model")


if __name__ == "__main__":
    main()
