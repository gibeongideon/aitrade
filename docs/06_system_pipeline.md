# 06 — End-to-End System Pipeline

## Overview

This document describes the complete pipeline from raw Forex data to a trading decision,
including the technology stack, data sources, and exactly what happens at each stage.

There are **two distinct data sources** depending on the context:

| Context | Data Source |
|---------|------------|
| Training | Pre-downloaded CSV files (15-min OHLCV, already on disk) |
| Live Inference | MetaTrader 5 Python API (fetches the latest completed candles) |

---

## Technology Stack

| Layer | Tool | Why |
|-------|------|-----|
| Language | Python 3.10+ (Windows) | MT5 Python API is Windows-only |
| ML Framework | PyTorch 2.x | Full control over custom transformer |
| Data processing | pandas + numpy | Industry standard for tabular data |
| Training data | Pre-downloaded CSV files | Already available, stable, no API dependency |
| Live data (inference) | MetaTrader5 Python package | Direct connection to running MT5 terminal |
| Data storage | Parquet files (processed) | Fast, compressed — better than CSV for large datasets |
| Calibration & metrics | scikit-learn | ECE, class weights, precision/recall |
| Backtesting | Custom Python | Simple, no framework lock-in |
| Config management | YAML config file | All hyperparameters in one place |
| Logging | Python logging + CSV | Simple, readable |

> **OS Requirement:** The MetaTrader 5 Python package (`MetaTrader5`) connects to the locally
> installed MT5 terminal via a COM interface. This only works on **Windows**. Training can be
> done on any OS (no MT5 dependency), but the live inference machine must run Windows with
> MT5 installed and logged in.

---

## MetaTrader 5 Integration

### Package and Version

| Property | Value |
|----------|-------|
| Python package | `MetaTrader5` |
| Install | `pip install MetaTrader5` |
| Minimum MT5 terminal build | Build 2361+ (MetaTrader 5 platform, any recent version) |
| Recommended terminal build | Latest stable from your broker |

The Python package version must match your installed MT5 terminal version.
If they mismatch, `mt5.initialize()` will fail. Upgrade both together.

### MT5 Terminal Requirements

The MT5 terminal must be:
- Installed on the same Windows machine as the Python script
- Open and logged into a broker account during inference
- The traded symbol (e.g., EURUSD) must be visible in the Market Watch panel

### MT5 Timing and Server Time

This is critical to get right.

| Property | Detail |
|----------|--------|
| MT5 server time | Set by the broker — typically UTC+2 (winter) / UTC+3 (summer) |
| Local machine time | May differ from server time — do not use for candle timing |
| How to get server time | `mt5.symbol_info_tick(symbol).time` returns server timestamp |
| Candle close detection | A new M15 bar opens every 15 minutes on server time |

**Always use MT5 server time, not local machine time, when checking if a new candle has closed.**

### Fetching Completed Candles — The Critical Rule

MT5's `copy_rates_from_pos` uses position indexing where:
- Position `0` = the **currently forming** (incomplete) bar
- Position `1` = the **most recently completed** bar
- Position `2` = the bar before that, and so on

```
Position:  0      1      2      3    ...
           |----| |----| |----| |----|
           NOW   DONE  DONE  DONE  (time goes right to left)
           (skip) ← use these 64 →
```

**Always fetch starting at position 1, never position 0.**

Fetching 64 completed candles:

```python
rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_M15, start_pos=1, count=64)
```

This returns the 64 most recently completed 15-minute bars, guaranteed to be closed.

---

## Training Data: CSV Files

Training uses pre-downloaded 15-minute OHLCV CSV files. No API calls needed during training.

### Expected CSV Format

The system expects the MT5 standard history export format:

```
<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<TICKVOL>,<VOL>,<SPREAD>
2023.01.02,00:00,1.07043,1.07059,1.06996,1.07018,234,0,0
2023.01.02,00:15,1.07018,1.07045,1.06987,1.07031,198,0,0
```

Or the alternative single-datetime column format (also accepted):

```
Date,Open,High,Low,Close,Volume
2023-01-02 00:00:00,1.07043,1.07059,1.06996,1.07018,234
```

The CSV loader (`csv_loader.py`) normalises both formats into a standard internal schema:

| Internal Column | Description |
|----------------|-------------|
| `time` | UTC datetime (parsed and timezone-aware) |
| `open` | Candle open price |
| `high` | Candle high price |
| `low` | Candle low price |
| `close` | Candle close price |
| `volume` | Tick volume |

### CSV File Naming Convention

```
data/raw/EURUSD_M15_2022_2024.csv
         ^^^^^^  ^^^  ^^^^^^^^^^
         pair    TF   date range (for reference only)
```

Multiple CSV files for the same pair can be concatenated by `csv_loader.py` and
deduplicated by timestamp before processing.

---

## High-Level Architecture

```
TRAINING PATH
─────────────
Pre-downloaded CSV files (data/raw/)
               |
               v
+----------------------------------+
|  CSV Loader (csv_loader.py)      |
|  - Parse date/time columns       |
|  - Normalise column names        |
|  - Deduplicate and sort by time  |
|  - Save as Parquet               |
+----------------------------------+
               |
               v
+----------------------------------+
|  Feature Engineering             |
|  (features.py)                   |
|  - Returns, body, wick ratios    |
|  - MA_16, MA_32, MA_64           |
|  - ATR(14)                       |
|  - Volume ratio                  |
|  - MA crossover events           |
+----------------------------------+
               |
               v
+----------------------------------+
|  Triple-Barrier Label Generator  |
|  (labeler.py)                    |
|  - k=0.75, H=3 candles           |
|  - Assigns BUY / HOLD / SELL     |
+----------------------------------+
               |
               v
+----------------------------------+
|  Market Language Tokenizer       |
|  (tokenizer.py)                  |
|  - Fit thresholds on train split |
|  - Save thresholds to disk       |
|  - Map all splits to token IDs   |
+----------------------------------+
               |
               v
+----------------------------------+
|  Sequence Builder                |
|  (sequences.py)                  |
|  - 64-candle sliding window      |
|  - Output: (token_ids, label)    |
+----------------------------------+
               |
               v
+----------------------------------+
|  Custom Transformer Encoder      |
|  (trainer.py)                    |
|  - 4 layers, d_model=128         |
|  - AdamW + Weighted CE Loss      |
|  - Early stopping on val loss    |
+----------------------------------+
               |
               v
+----------------------------------+
|  Calibration (calibrator.py)     |
|  - Temperature scaling on val    |
|  - Save T to calibration.json    |
+----------------------------------+


INFERENCE PATH
──────────────
MetaTrader 5 Terminal (running, logged in)
               |
               v
+----------------------------------+
|  MT5 Live Feed (mt5_feed.py)     |
|  - mt5.initialize()              |
|  - copy_rates_from_pos(...,      |
|      start_pos=1, count=64)      |
|  - Returns 64 completed candles  |
|  - mt5.shutdown()                |
+----------------------------------+
               |
               v
+----------------------------------+
|  Feature Engineering             |
|  (same logic as training)        |
+----------------------------------+
               |
               v
+----------------------------------+
|  Tokenizer                       |
|  (uses saved thresholds only —   |
|   never refit on live data)      |
+----------------------------------+
               |
               v
+----------------------------------+
|  Model Forward Pass              |
|  + Calibration (T scaling)       |
|  → [p_sell, p_hold, p_buy]       |
+----------------------------------+
               |
               v
+----------------------------------+
|  Trading Decision Engine         |
|  - Threshold filter (>0.60)      |
|  - Edge-proportional sizing      |
|  - Emit signal or no-action      |
+----------------------------------+
               |
               v
+----------------------------------+
|  Log to inference_log.csv        |
|  (timestamp, probs, signal)      |
+----------------------------------+
```

---

## Training Pipeline — Step by Step

### Step 1: Load CSV Data

Script: `src/data/csv_loader.py`

- Accept one or more CSV file paths as input
- Parse date/time columns (handles both MT5 export formats)
- Normalise column names to internal schema
- Concatenate, deduplicate by timestamp, sort ascending by time
- Drop rows with zero volume (weekend gaps, broker anomalies)
- Save clean data to `data/raw/EURUSD_M15.parquet`

### Step 2: Feature Engineering

Script: `src/data/features.py`

- Input: clean Parquet
- Compute: return, body_ratio, upper_wick, lower_wick, vol_ratio, MA_16, MA_32, MA_64, ATR_14, ma_cross
- Drop first 64 rows (NaN from rolling windows)
- Output: feature Parquet

### Step 3: Label Generation

Script: `src/data/labeler.py`

- Triple-barrier: k=0.75, H=3
- Print label distribution; target BUY+SELL = 40–55%
- Output: feature Parquet with `label` column added

### Step 4: Time-Based Split

Split indices by time position, not by random sampling:

```
Train:      rows[0 : int(N*0.70)]
Validation: rows[int(N*0.70) : int(N*0.85)]
Test:       rows[int(N*0.85) :]
```

### Step 5: Tokenization

Script: `src/data/tokenizer.py`

- Compute quantile thresholds from training rows only
- Apply same thresholds to validation and test rows
- Save `models/tokenizer_thresholds.json`

### Step 6: Sequence Construction

Script: `src/data/sequences.py`

- 64-candle sliding window over all rows
- Each sample: token ID tensor of shape `[335]` + integer label
- Save train/val/test as `.pt` files

### Step 7: Model Training

Script: `src/model/trainer.py`

- Read config.yaml
- Train to early stopping, save best checkpoint

### Step 8: Calibration

Script: `src/model/calibrator.py`

- Fit temperature T on validation set
- Save `models/calibration.json`

### Step 9: Evaluation

Script: `src/model/evaluate.py`

- Run on test set; report accuracy, F1, ECE, simulated PnL

---

## Inference Pipeline — Step by Step

Triggered every 15 minutes, after a new MT5 candle closes.

```
1. Detect new candle close:
   - Compare current MT5 server minute to last recorded minute
   - When minute crosses a 15-min boundary: trigger inference

2. Fetch data from MT5:
   mt5.initialize()
   rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_M15, start_pos=1, count=64)
   mt5.shutdown()
   → rates is a numpy structured array with: time, open, high, low, close, tick_volume

3. Convert to DataFrame, compute features (same functions as training)

4. Load tokenizer_thresholds.json, tokenize the 64 candles

5. Load transformer_best.pt + calibration.json

6. Run forward pass → raw logits → softmax(logits / T) → [p_sell, p_hold, p_buy]

7. Trading engine: apply threshold and edge logic → signal or no-action

8. Log row to inference_log.csv:
   timestamp | p_sell | p_hold | p_buy | signal | edge
```

---

## Saved Artifacts (persisted after training)

| File | Contents |
|------|---------|
| `models/transformer_best.pt` | Model weights |
| `models/calibration.json` | Temperature scalar T |
| `models/tokenizer_thresholds.json` | Quantile thresholds — never recomputed at inference |
| `models/class_weights.json` | Training class weights (for reference) |
| `logs/training_log.csv` | Epoch-by-epoch metrics |
| `logs/test_evaluation.txt` | Final test metrics |
| `logs/inference_log.csv` | All live predictions |

---

## Key Constraints

- `start_pos=1` is mandatory in MT5 data fetch — position 0 is always an incomplete bar
- Use MT5 server time for candle close detection, not local machine time
- Tokenizer thresholds are computed from training data and frozen — never refit on live data
- Time-based splitting is enforced — no random splits, no shuffling across time boundaries
- All inference calls are logged — no silent operation
- Backtesting always includes spread cost per trade (use the actual spread from your broker)
