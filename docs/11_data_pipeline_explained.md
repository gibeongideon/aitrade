# 11 — Data Pipeline: How the Data Is Processed

This document explains every step from the raw CSV file to the final `.pt` files
the model trains on — including what the data looks like at each stage.

---

## Overview: 5 Steps, 2 Phases

```
Raw CSV
  │
  ▼  Step 1 — csv_loader.py
Clean OHLCV Parquet  (100,000 rows)
  │
  ▼  Step 2 — features.py
Feature-enriched Parquet  (99,937 rows after warmup drop)
  │
  ▼  Step 3 — labeler.py
Labeled Parquet  (same rows, labels added)
  │
  ▼  Step 4 — tokenizer.py
Tokenized Parquet  (each candle → 5 integer token IDs)
  │
  ▼  Step 5 — sequences.py
train_sequences.pt  (20,608 sequences)
val_sequences.pt    (4,401 sequences)
test_sequences.pt   (4,354 sequences)
```

---

## Step 1 — csv_loader.py: Raw CSV → Clean OHLCV

**Script:** [src/data/csv_loader.py](../src/data/csv_loader.py)
**Input:** `data/raw/EURUSD_M15.csv`
**Output:** `data/raw/EURUSD_M15.parquet`

### What the raw CSV looks like

The source file is EURUSD 15-minute candles from forexsb.com.
Format B was detected (single datetime column, no header row):

```
2021-09-24 20:15,1.17194,1.17217,1.17194,1.17209,106
2021-09-24 20:30,1.17208,1.17215,1.17197,1.17199,87
2021-09-24 20:45,1.17201,1.17210,1.17185,1.17190,123
...
```

### What cleaning does

| Operation | Result |
|-----------|--------|
| Sort by time ascending | Ensures chronological order |
| Deduplicate timestamps | 0 removed in this file |
| Drop zero-volume rows | 0 removed (provider pre-stripped weekends) |
| Drop NaN / zero-price rows | 0 removed |
| **Final clean rows** | **100,000 rows** |

### What the output looks like

```
time                  open      high      low       close     volume
2021-09-24 20:15:00   1.17194   1.17217   1.17194   1.17209   106
2021-09-24 20:30:00   1.17208   1.17215   1.17197   1.17199   87
...
2025-09-26 23:45:00   1.11043   1.11057   1.11031   1.11049   94
```

- **Date range:** 2021-09-24 → 2025-09-26 (~4 years of M15 data)
- **Largest gap:** 3 days over Christmas/New Year 2023-12-29 → 2024-01-01 (expected)
- **Columns:** 6 — `time, open, high, low, close, volume`

---

## Step 2 — features.py: OHLCV → Feature-Enriched

**Script:** [src/data/features.py](../src/data/features.py)
**Input:** `data/raw/EURUSD_M15.parquet`
**Output:** `data/raw/EURUSD_M15_features.parquet`

### What 10 new columns are added

| Column | Formula | What it captures |
|--------|---------|-----------------|
| `ret` | `(close - open) / open` | Candle direction and size (%) |
| `body_ratio` | `abs(close - open) / (high - low)` | Body vs wicks ratio (0=doji, 1=marubozu) |
| `upper_wick` | `high - max(open, close)` | Bearish rejection above |
| `lower_wick` | `min(open, close) - low` | Bullish rejection below |
| `MA_16` | 16-period SMA of close | Short-term trend |
| `MA_32` | 32-period SMA of close | Medium-term trend |
| `MA_64` | 64-period SMA of close | Long-term trend (≈16 hours) |
| `ATR_14` | 14-period Average True Range | Volatility baseline |
| `vol_ratio` | `volume / rolling_mean(volume, 20)` | Volume vs recent average |
| `ma_cross` | `+1` bull cross, `-1` bear cross, `0` none | MA_16 crossing MA_64 |

### Warm-up rows dropped

The longest indicator is MA_64 (needs 64 rows). Rows without all indicators defined
are dropped. **63 warm-up rows removed → 99,937 remaining.**

### What the output looks like

```
time        open     close    ret        body_ratio  upper_wick  lower_wick  MA_16    MA_32    MA_64    ATR_14   vol_ratio  ma_cross
2021-11-24  1.12345  1.12367  +0.000196  0.74        0.000010    0.000003    1.12310  1.12290  1.12270  0.00043  1.12       0
...
```

Total columns after step 2: **16**

---

## Step 3 — labeler.py: Features → Labels

**Script:** [src/data/labeler.py](../src/data/labeler.py)
**Input:** `data/raw/EURUSD_M15_features.parquet`
**Output:** `data/raw/EURUSD_M15_labeled.parquet`

### Method: Non-Overlapping Triple-Barrier Labeling

This is the López de Prado method (from *Advances in Financial Machine Learning*).

**Why not label every candle?**
If you label candle T1 by looking at T2–T4, and also label T2 by looking at T3–T5,
then T3 and T4 appear in both labels. The model can learn to predict T2's label
by "memorising" T1's future — data leakage. In live trading you cannot open two
overlapping trades.

**How it works:**

```
For candle i (entry):
    upper = close[i] + 1.20 × ATR_14[i]   ← take-profit barrier
    lower = close[i] − 1.20 × ATR_14[i]   ← stop-loss barrier

    Look forward up to 3 candles:
        If high[j] >= upper  →  label = 2 (BUY)
        If low[j]  <= lower  →  label = 0 (SELL)
        Neither hit within 3 candles  →  label = 1 (HOLD)

    Next event starts at exit_candle + 1  ← no overlap
```

**Parameters used (from config.yaml):**
- `k = 1.20` (ATR multiplier for barriers — tuned from default 0.75)
- `horizon = 3` (max 3 candles = 45 minutes ahead)

**Why k=1.20?** The default k=0.75 produced 80% directional events on this
dataset because EURUSD M15 ATR is large enough that tight barriers were hit often.
k=1.20 produces a balanced 54% directional distribution.

### Label encoding

| Value | Name | Meaning |
|-------|------|---------|
| `-1` | non-event | Candle skipped — used for context only, not a prediction target |
| `0` | SELL | Lower barrier hit first within 3 candles |
| `1` | HOLD | Neither barrier hit — time expired |
| `2` | BUY | Upper barrier hit first within 3 candles |

### 2 columns added

| Column | Type | Description |
|--------|------|-------------|
| `label` | int8 | -1 for non-events, 0/1/2 for events |
| `exit_idx` | int32 | Row index of exit candle (-1 for non-events) |

### Results

| Metric | Value |
|--------|-------|
| Total candles | 99,937 |
| Event candles (labeled) | 29,382 (29.4% of all candles) |
| Non-event candles (context) | 70,555 (70.6%) |
| BUY events | 7,996 (27.2% of events) |
| HOLD events | 13,560 (46.2%) |
| SELL events | 7,826 (26.6%) |
| BUY + SELL combined | 53.8% ✅ (target 40–60%) |

**Event density ~29%** means on average every 3–4 candles is a labeled trade event,
with the candles between events serving as context but not training targets.

---

## Step 4 — tokenizer.py: Features → Token IDs

**Script:** [src/data/tokenizer.py](../src/data/tokenizer.py)
**Input:** `data/raw/EURUSD_M15_labeled.parquet`
**Output:** `data/raw/EURUSD_M15_tokenized.parquet` + `models/tokenizer_thresholds.json`

### Why tokenize?

Instead of feeding raw floats to the transformer, each candle is converted into
5 integer token IDs — a "market sentence" of 5 words. This makes the model's
vocabulary finite and explicit, similar to how NLP tokenizers work.

### Token vocabulary (25 IDs total)

| ID | Token | Group | Meaning |
|----|-------|-------|---------|
| 0 | [PAD] | Special | Padding |
| 1 | [CLS] | Special | Sequence-start |
| 2 | RET_D3 | Return | Strong down move (bottom 5% of returns) |
| 3 | RET_D2 | Return | Moderate down |
| 4 | RET_D1 | Return | Slight down |
| 5 | RET_FLAT | Return | Near-zero move (40–60th percentile) |
| 6 | RET_U1 | Return | Slight up |
| 7 | RET_U2 | Return | Moderate up |
| 8 | RET_U3 | Return | Strong up move (top 5%) |
| 9 | BODY_S | Body | Small body — doji-like (< 33rd pct) |
| 10 | BODY_M | Body | Medium body |
| 11 | BODY_L | Body | Large body — marubozu-like (> 67th pct) |
| 12 | WICK_NONE | Wick | No significant wicks |
| 13 | WICK_TOP | Wick | Upper wick only — bearish rejection |
| 14 | WICK_BOTTOM | Wick | Lower wick only — bullish rejection |
| 15 | WICK_BOTH | Wick | Wicks both sides — indecision |
| 16 | VOL_LOW | Volume | Below average (< 25th pct) |
| 17 | VOL_NORMAL | Volume | Average volume (25–75th pct) |
| 18 | VOL_HIGH | Volume | Above average (75–90th pct) |
| 19 | VOL_SPIKE | Volume | Extreme volume event (> 90th pct) |
| 20 | TREND_UP | Trend | MA_16 > MA_32 > MA_64 (aligned bullish) |
| 21 | TREND_DOWN | Trend | MA_16 < MA_32 < MA_64 (aligned bearish) |
| 22 | TREND_MIX | Trend | MAs not aligned (ranging / transitional) |
| 23 | TREND_CROSS_UP | Trend | MA_16 just crossed above MA_64 |
| 24 | TREND_CROSS_DOWN | Trend | MA_16 just crossed below MA_64 |

### Thresholds computed from training rows only

Thresholds are computed from the first 70% of rows (train split) and then applied
to the full dataset. This prevents val/test data from influencing the discretization.

| Feature | Boundaries (from training data) |
|---------|--------------------------------|
| RET | -0.00076, -0.00028, -0.00007, +0.00007, +0.00028, +0.00075 |
| BODY | 0.3214, 0.6200 |
| VOL | 0.5831, 1.3944, 1.9541 |
| WICK min | 0.10 × ATR_14 (rule-based) |

### Example: one candle tokenized

```
Candle:
  ret        = +0.00032  → above 0.00028 threshold → RET_U2  (ID=7)
  body_ratio = 0.71      → above 0.62 threshold    → BODY_L  (ID=11)
  upper_wick = 0.00002   → < 0.10 × ATR            → not significant
  lower_wick = 0.00001   → < 0.10 × ATR            → not significant
                                                    → WICK_NONE (ID=12)
  vol_ratio  = 1.65      → between 1.39 and 1.95   → VOL_HIGH (ID=18)
  MA_16 > MA_32 > MA_64                             → TREND_UP  (ID=20)

Token sequence for this candle: [7, 11, 12, 18, 20]
```

### Token frequency in the full dataset (99,937 rows)

| Group | Token | % |
|-------|-------|---|
| RET | D3 / D2 / D1 / FLAT / U1 / U2 / U3 | 4.8 / 15.3 / 20.0 / 19.7 / 20.1 / 15.2 / 4.9 |
| BODY | S / M / L | 33.1 / 34.1 / 32.7 |
| WICK | NONE / TOP / BOTTOM / BOTH | 8.5 / 22.5 / 22.2 / 46.7 |
| VOL | LOW / NORMAL / HIGH / SPIKE | 25.0 / 50.4 / 14.9 / 9.8 |
| TREND | UP / DOWN / MIX / X_UP / X_DOWN | 33.9 / 33.5 / 30.7 / 1.0 / 1.0 |

All 23 content tokens appear > 1% of the time — no dead tokens.

---

## Step 5 — sequences.py: Tokens → .pt Files

**Script:** [src/data/sequences.py](../src/data/sequences.py)
**Input:** `data/raw/EURUSD_M15_tokenized.parquet`
**Output:** `data/processed/train_sequences.pt`, `val_sequences.pt`, `test_sequences.pt`

### One sequence per event (not per candle)

Only the 29,382 event rows (label 0/1/2) become sequences.
The 70,555 non-event rows are used as context candles inside those sequences
but do not generate their own sequences.

### Split assignment (time-based, no shuffle)

Splits are assigned by time to prevent any future data leaking into earlier splits:

```
Total rows: 99,937

train  : rows 0      – 69,955  (70%)  → rows up to ~2024-09
val    : rows 69,956 – 84,944  (15%)  → rows ~2024-09 to ~2025-03
test   : rows 84,945 – 99,936  (15%)  → rows ~2025-03 to 2025-09
```

### Boundary purging (prevents leakage at split edges)

An event at row `i` with exit at row `exit_j` is only included in a split if
`exit_j` also stays within that split. Events whose trade crosses a boundary are
**purged** (excluded).

Example: an event in the last few rows of the train split whose trade exits in
the first few rows of val — that event is removed from train to prevent training
labels from "seeing" validation candles.

### Sequence structure (335 tokens)

Each sequence is a fixed-length integer tensor:

```
Position 0           : [CLS] token  (ID = 1)
Positions 1 – 320    : 64 candles × 5 tokens = 320 content tokens (oldest → newest)
Positions 321 – 334  : 14 × [PAD] token (ID = 0)
─────────────────────────────────────────────────────
Total length         : 335 tokens
```

The 14 PAD tokens exist because `max_seq_len = 335` was set in config to allow
the architecture to support longer sequences in the future without rewriting the
positional embeddings. (64 × 5 + 1 = 321 → padded to 335.)

### What a sequence looks like (raw integer tensor)

```
Sequence for event at 2023-06-15 14:00 (BUY label):

[  1,                         ← [CLS]
   5, 9,12,17,22,             ← candle 1 (oldest): RET_FLAT BODY_S WICK_NONE VOL_NORMAL TREND_MIX
   6,10,14,17,22,             ← candle 2
   7,11,12,18,20,             ← candle 3
   ...                        ← 61 more candles
   7,11,12,18,20,             ← candle 64 (newest, the event candle)
   0, 0, 0, 0, 0, 0, 0,      ← [PAD] × 14
   0, 0, 0, 0, 0, 0, 0  ]

label = 2  (BUY)
```

### Final dataset sizes

| Split | Sequences | SELL | HOLD | BUY |
|-------|-----------|------|------|-----|
| Train | 20,608 | 26.6% | 46.0% | 27.4% |
| Val   | 4,401  | 26.5% | 46.6% | 26.9% |
| Test  | 4,354  | 26.9% | 46.4% | 26.7% |
| **Total** | **29,363** | | | |

Label distribution is consistent across all three splits — no split imbalance.

### What each .pt file is

Each file is a `torch.utils.data.TensorDataset` with two tensors:

```python
# Load example
import torch
dataset = torch.load("data/processed/train_sequences.pt")

input_ids, labels = dataset.tensors

input_ids.shape  # → torch.Size([20608, 335])   dtype=torch.int64
labels.shape     # → torch.Size([20608])          dtype=torch.int64
labels.unique()  # → tensor([0, 1, 2])  — SELL / HOLD / BUY
```

---

## Leakage Prevention Summary

Three independent layers prevent data leakage:

| Layer | What it does |
|-------|-------------|
| **Non-overlapping events** | No two training samples share a future candle as a label target |
| **Time-based split** | Val/test always come after train in time — no future data in training |
| **Boundary purging** | Events whose trade exits cross a split boundary are excluded entirely |
| **Thresholds from train only** | Tokenizer quantiles computed on training rows, applied to all |

---

## How to Reproduce

From the project root:

```bash
python3 -m src.data.csv_loader
python3 -m src.data.features
python3 -m src.data.labeler
python3 -m src.data.tokenizer
python3 -m src.data.sequences
```

All parameters are controlled via [config/config.yaml](../config/config.yaml).
No arguments needed — each script reads the previous step's output automatically.
