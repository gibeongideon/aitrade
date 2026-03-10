# AITrade — Build Progress

Last updated: 2026-03-10

Data source: <https://forexsb.com/historical-forex-data>

---

## Phase 1: Data Pipeline — ✅ COMPLETE

**Goal:** Load CSV → engineer features → generate labels → verify quality

| Step | Script | Status | Output |
| ---- | ------ | ------ | ------ |
| 1.1 | `src/data/csv_loader.py` | ✅ Done | `data/raw/EURUSD_M15.parquet` |
| 1.2 | `src/data/features.py` | ✅ Done | `data/raw/EURUSD_M15_features.parquet` |
| 1.3 | `src/data/labeler.py` | ✅ Done | `data/raw/EURUSD_M15_labeled.parquet` |

### Phase 1 Results

| Metric | Value |
| ------ | ----- |
| Raw CSV rows | 100,000 |
| After cleaning | 100,000 (no zero-volume rows in this file) |
| After feature warm-up drop | 99,937 |
| Final labeled rows | 99,934 |
| Date range | 2021-09-24 to 2025-09-26 (~4 years) |
| Label k value | 1.20 (tuned — was 0.75 default) |
| Label horizon | 3 candles (45 min) |
| BUY | 7,996 (27.2%) — non-overlapping events |
| HOLD | 13,560 (46.2%) — non-overlapping events |
| SELL | 7,826 (26.6%) — non-overlapping events |
| Total events | 29,382 from 99,937 candles (29.4% density) |
| BUY + SELL combined | 53.8% ✅ (target: 40–60%) |

### Phase 1 Notes

- CSV format detected as **Format B** (single datetime column, no header)
- k was adjusted from the documented default of 0.75 to **1.20** because this dataset
  has higher ATR candles — at k=0.75 the barriers were too close (80% directional).
  k=1.20 produces a balanced 54% directional distribution.
- No zero-volume rows were present in the source CSV (weekend rows already stripped
  by the data provider)
- Largest gap in data: 3 days (2023-12-29 to 2024-01-01, Christmas/New Year)
- `config/config.yaml` updated with the correct k=1.20
- **Labeling uses NON-OVERLAPPING events** (López de Prado method): after labeling
  candle T_i with exit at T_j, the next event starts at T_{j+1}. No future candle
  is shared between two training samples. Event density ~29% of all candles.

### Phase 1 Pending

- [ ] Manual sanity check: pick 5 BUY rows and verify upper barrier was reached
      within next 3 candles. Use a notebook or quick script.

---

## Phase 2: Tokenisation and Sequences — ✅ COMPLETE

**Goal:** Convert labeled data into model-ready integer sequences

| Step | Script | Status | Output |
| ---- | ------ | ------ | ------ |
| 2.1 | `src/data/tokenizer.py` | ✅ Done | `data/raw/EURUSD_M15_tokenized.parquet`, `models/tokenizer_thresholds.json` |
| 2.2 | `src/data/sequences.py` | ✅ Done | `data/processed/train_sequences.pt`, `val_sequences.pt`, `test_sequences.pt` |

### Phase 2 Results

| Metric | Value |
| ------ | ----- |
| Input rows | 99,937 |
| Total non-overlapping events | 29,382 |
| Train sequences | 20,608 (events where exit stays in train split) |
| Val sequences | 4,401 (purged at boundary) |
| Test sequences | 4,354 (purged at boundary) |
| Sequence length | 335 tokens ([CLS] + 64x5 content + 14 [PAD]) |
| Vocab size used | 25 (IDs 0-24) |
| Leakage prevention | Non-overlapping events + boundary purging ✅ |

### Phase 2 Token Frequencies (full dataset)

| Group | Token | % |
| ----- | ----- | - |
| RET | D3/D2/D1/FLAT/U1/U2/U3 | 4.8 / 15.3 / 20.0 / 19.7 / 20.1 / 15.2 / 4.9 |
| BODY | S/M/L | 33.1 / 34.1 / 32.7 |
| WICK | NONE/TOP/BOTTOM/BOTH | 8.5 / 22.5 / 22.2 / 46.7 |
| VOL | LOW/NORMAL/HIGH/SPIKE | 25.0 / 50.4 / 14.9 / 9.8 |
| TREND | UP/DOWN/MIX/X_UP/X_DOWN | 33.9 / 33.5 / 30.7 / 1.0 / 1.0 |

All 23 content tokens above 1% threshold. ✅

### Phase 2 Thresholds (from training rows only)

| Feature | Boundaries |
| ------- | ---------- |
| RET | -0.00076, -0.00028, -0.00007, +0.00007, +0.00028, +0.00075 |
| BODY | 0.3214, 0.6200 |
| VOL | 0.5831, 1.3944, 1.9541 |
| WICK min | 0.10 x ATR_14 (rule-based, no threshold) |

### Phase 2 Label Distribution per Split (non-overlapping events only)

| Split | Sequences | SELL | HOLD | BUY |
| ----- | --------- | ---- | ---- | --- |
| Train | 20,608 | 26.6% | 46.0% | 27.4% |
| Val | 4,401 | 26.5% | 46.6% | 26.9% |
| Test | 4,354 | 26.9% | 46.4% | 26.7% |

---

## Phase 3: Model Training — ✅ COMPLETE

**Goal:** Train the custom transformer encoder and calibrate probabilities

| Step | Script | Status | Output |
| ---- | ------ | ------ | ------ |
| 3.1 | `src/model/transformer.py` | ✅ Done | — |
| 3.2 | `src/model/trainer.py` | ✅ Done (trained on Colab T4) | `models/transformer_best.pt` |
| 3.3 | `src/model/calibrator.py` | ✅ Done | `models/calibration.json` |

### Phase 3 Notes

- Training run on **Google Colab T4 GPU** (~54s/epoch)
- Model: 797,187 parameters (d_model=128, 4 layers, 8 heads)
- Warmup completed at epoch 2 (500 steps ≈ 1.6 epochs)
- Real learning began at **epoch 8** (train acc jumped from 35.6% to 43.2%)
- Calibration: temperature scaling T applied on val set, saved to `models/calibration.json`

---

## Phase 4: Evaluation and Backtesting — 🔄 IN PROGRESS

**Goal:** Validate on unseen test data before any live use

| Step | Script | Status | Output |
| ---- | ------ | ------ | ------ |
| 4.1 | `src/model/evaluate.py` | ✅ Script written — ready to run | `logs/test_evaluation.txt`, `logs/test_evaluation.json` |
| 4.2 | `src/trading/backtester.py` | ✅ Script written + run | `logs/backtest_report.txt`, `logs/backtest_trades.csv`, `logs/backtest_summary.json` |

### Backtest Result (epoch-10 model — not fully trained)

- At threshold 60%: **0 trades** — model never reaches 60% confidence on BUY/SELL
- At threshold 40%: 1,622 trades, win rate **14%**, expectancy **-1.0 pip** — negative edge
- Root cause: **model only trained 10 epochs**, real learning just started at epoch 8
- Action required: retrain on Colab to completion (~40–60 epochs), re-run calibrator + backtester

### How to run evaluate.py

```bash
python3 -m src.model.evaluate
```

Requires: `models/transformer_best.pt`, `models/calibration.json`, `data/processed/test_sequences.pt`

Reports: accuracy, per-class F1, confusion matrix, ECE, high-confidence (>=60%) accuracy and coverage

### How to run backtester.py

```bash
python3 -m src.trading.backtester
# or override threshold:
python3 -m src.trading.backtester --threshold 0.55
```

Requires: `models/transformer_best.pt`, `models/calibration.json`, `data/raw/EURUSD_M15_tokenized.parquet`

Reports: win rate, expectancy, profit factor, max drawdown, calibration check — saves txt + csv + json

---

## Phase 5: Live Inference (MT5) — ⏳ TODO

**Goal:** Run model every 15 minutes against live MT5 candles

| Step | Script | Status | Output |
| ---- | ------ | ------ | ------ |
| 5.1 | `src/inference/predictor.py` | ⏳ Not started | `logs/inference_log.csv` |

> **Requires:** Windows machine with MetaTrader 5 installed and running

---

## Project Structure

```text
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
│   │   ├── tokenizer.py         ✅ Phase 2.1
│   │   └── sequences.py         ✅ Phase 2.2
│   ├── model/
│   │   ├── transformer.py       ✅ Phase 3.1
│   │   ├── trainer.py           ✅ Phase 3.2
│   │   ├── calibrator.py        ✅ Phase 3.3
│   │   └── evaluate.py          ✅ Phase 4.1
│   ├── trading/
│   │   ├── engine.py            ⏳ Phase 5
│   │   └── backtester.py        ⏳ Phase 4.2
│   └── inference/
│       └── predictor.py         ⏳ Phase 5.1
└── requirements.txt             ✅ created
```

---

## How to Run

From the project root:

```bash
# Phase 1
python3 -m src.data.csv_loader
python3 -m src.data.features
python3 -m src.data.labeler

# Phase 2
python3 -m src.data.tokenizer
python3 -m src.data.sequences

# Phase 3 (run on Colab GPU, then copy checkpoint back)
python3 -m src.model.trainer
python3 -m src.model.calibrator

# Phase 4
python3 -m src.model.evaluate
```

Each script reads from the previous step's output automatically.
All parameters are controlled via `config/config.yaml`.
