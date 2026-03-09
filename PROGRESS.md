# AITrade — Build Progress

Last updated: 2026-03-06

---
data
https://forexsb.com/historical-forex-data

## Phase 1: Data Pipeline  ✅ COMPLETE

**Goal:** Load CSV → engineer features → generate labels → verify quality

| Step | Script | Status | Output |
|------|--------|--------|--------|
| 1.1 | `src/data/csv_loader.py` | ✅ Done | `data/raw/EURUSD_M15.parquet` |
| 1.2 | `src/data/features.py` | ✅ Done | `data/raw/EURUSD_M15_features.parquet` |
| 1.3 | `src/data/labeler.py` | ✅ Done | `data/raw/EURUSD_M15_labeled.parquet` |

### Phase 1 Results

| Metric | Value |
|--------|-------|
| Raw CSV rows | 100,000 |
| After cleaning | 100,000 (no zero-volume rows in this file) |
| After feature warm-up drop | 99,937 |
| Final labeled rows | 99,934 |
| Date range | 2021-09-24 → 2025-09-26 (~4 years) |
| Label k value | 1.20 (tuned — was 0.75 default) |
| Label horizon | 3 candles (45 min) |
| BUY  | 25,718 (25.7%) |
| HOLD | 48,838 (48.9%) |
| SELL | 25,378 (25.4%) |
| BUY + SELL combined | 51.1% ✅ (target: 40–60%) |

### Phase 1 Notes

- CSV format detected as **Format B** (single datetime column, no header)
- k was adjusted from the documented default of 0.75 to **1.20** because this dataset
  has higher ATR candles — at k=0.75 the barriers were too close (80% directional).
  k=1.20 produces a balanced 51% directional distribution.
- No zero-volume rows were present in the source CSV (weekend rows already stripped
  by the data provider)
- Largest gap in data: 3 days (2023-12-29 → 2024-01-01, Christmas/New Year)
- `config/config.yaml` updated with the correct k=1.20

### Phase 1 Pending

- [ ] Manual sanity check: pick 5 BUY rows and verify upper barrier was reached
      within next 3 candles. Use a notebook or quick script.

---

## Phase 2: Tokenisation and Sequences  ✅ COMPLETE

**Goal:** Convert labeled data into model-ready integer sequences

| Step | Script | Status | Output |
|------|--------|--------|--------|
| 2.1 | `src/data/tokenizer.py` | ✅ Done | `data/raw/EURUSD_M15_tokenized.parquet` + `models/tokenizer_thresholds.json` |
| 2.2 | `src/data/sequences.py` | ✅ Done | `data/processed/train_sequences.pt`, `val_sequences.pt`, `test_sequences.pt` |

### Phase 2 Results

| Metric | Value |
|--------|-------|
| Input rows | 99,934 |
| Train split | 69,953 rows → 69,890 sequences |
| Val split | 14,990 rows → 14,927 sequences |
| Test split | 14,991 rows → 14,928 sequences |
| Sequence length | 335 tokens ([CLS] + 64×5 content + 14 [PAD]) |
| Vocab size used | 25 (IDs 0–24) |

### Phase 2 Token Frequencies (full dataset)

| Group | Token | % |
|-------|-------|---|
| RET | D3/D2/D1/FLAT/U1/U2/U3 | 4.8 / 15.3 / 20.0 / 19.7 / 20.1 / 15.2 / 4.9 |
| BODY | S/M/L | 33.1 / 34.1 / 32.7 |
| WICK | NONE/TOP/BOTTOM/BOTH | 8.5 / 22.5 / 22.2 / 46.7 |
| VOL | LOW/NORMAL/HIGH/SPIKE | 25.0 / 50.4 / 14.9 / 9.8 |
| TREND | UP/DOWN/MIX/X_UP/X_DOWN | 33.9 / 33.5 / 30.7 / 1.0 / 1.0 |

All 23 content tokens above 1% threshold. ✅

### Phase 2 Thresholds (from training rows only)

| Feature | Boundaries |
|---------|-----------|
| RET | -0.00076, -0.00028, -0.00007, +0.00007, +0.00028, +0.00075 |
| BODY | 0.3214, 0.6200 |
| VOL | 0.5831, 1.3944, 1.9541 |
| WICK min | 0.10 × ATR_14 (rule-based, no threshold) |

### Phase 2 Label Distribution per Split

| Split | SELL | HOLD | BUY |
|-------|------|------|-----|
| Train | 25.2% | 49.0% | 25.8% |
| Val   | 24.8% | 49.1% | 26.1% |
| Test  | 26.4% | 48.3% | 25.3% |

---

## Phase 3: Model Training  ⏳ TODO

**Goal:** Train the custom transformer encoder and calibrate probabilities

| Step | Script | Status | Output |
|------|--------|--------|--------|
| 3.1 | `src/model/transformer.py` | ⏳ Not started | — |
| 3.2 | `src/model/trainer.py` | ⏳ Not started | `models/transformer_best.pt` |
| 3.3 | `src/model/calibrator.py` | ⏳ Not started | `models/calibration.json` |

---

## Phase 4: Evaluation and Backtesting  ⏳ TODO

**Goal:** Validate on unseen test data before any live use

| Step | Script | Status | Output |
|------|--------|--------|--------|
| 4.1 | `src/model/evaluate.py` | ⏳ Not started | `logs/test_evaluation.txt` |
| 4.2 | `src/trading/backtester.py` | ⏳ Not started | Backtest report |

---

## Phase 5: Live Inference (MT5)  ⏳ TODO

**Goal:** Run model every 15 minutes against live MT5 candles

| Step | Script | Status | Output |
|------|--------|--------|--------|
| 5.1 | `src/inference/predictor.py` | ⏳ Not started | `logs/inference_log.csv` |

> **Requires:** Windows machine with MetaTrader 5 installed and running

---

## Project Structure

```
aitrade/
├── config/
│   └── config.yaml              ✅ created
├── data/
│   ├── raw/
│   │   ├── EURUSD_M15.csv       ✅ source data (moved from root)
│   │   ├── EURUSD_M15.parquet   ✅ Phase 1.1 output
│   │   ├── EURUSD_M15_features.parquet  ✅ Phase 1.2 output
│   │   └── EURUSD_M15_labeled.parquet   ✅ Phase 1.3 output
│   └── processed/               (Phase 2 output goes here)
├── docs/                        ✅ all documentation
├── logs/                        (Phase 3+ output goes here)
├── models/                      (Phase 3+ output goes here)
├── notebooks/
├── src/
│   ├── data/
│   │   ├── csv_loader.py        ✅ Phase 1.1
│   │   ├── features.py          ✅ Phase 1.2
│   │   ├── labeler.py           ✅ Phase 1.3
│   │   ├── tokenizer.py         ⏳ Phase 2.1
│   │   └── sequences.py         ⏳ Phase 2.2
│   ├── model/
│   │   ├── transformer.py       ⏳ Phase 3.1
│   │   ├── trainer.py           ⏳ Phase 3.2
│   │   ├── calibrator.py        ⏳ Phase 3.3
│   │   └── evaluate.py          ⏳ Phase 4.1
│   ├── trading/
│   │   ├── engine.py            ⏳ Phase 5
│   │   └── backtester.py        ⏳ Phase 4.2
│   └── inference/
│       └── predictor.py         ⏳ Phase 5.1
└── requirements.txt             ✅ created
```

---

## How to Run Phase 1

From the project root (`/home/rock/Desktop/2026_Projects/my/aitrade`):

```bash
python3 -m src.data.csv_loader
python3 -m src.data.features
python3 -m src.data.labeler
```

Each script reads from the previous step's output automatically.
All parameters are controlled via `config/config.yaml`.
