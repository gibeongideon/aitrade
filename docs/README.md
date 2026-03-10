# AITrade — LLM-Based Forex Prediction System

## Documentation Index

| File | Description |
|------|-------------|
| [01_project_overview.md](01_project_overview.md) | Objective, motivation, and intended outcome |
| [02_data_representation.md](02_data_representation.md) | Raw data fields, feature engineering, and full market token vocabulary |
| [03_model_architecture.md](03_model_architecture.md) | Chosen model (custom small transformer), training config, exact training process |
| [04_prediction_target.md](04_prediction_target.md) | Triple-barrier label design, horizon, class imbalance handling |
| [05_trading_engine.md](05_trading_engine.md) | Signal interpretation, entry threshold, edge-based position sizing |
| [06_system_pipeline.md](06_system_pipeline.md) | Full pipeline with tech stack, CSV training data, MT5 live inference |
| [07_challenges.md](07_challenges.md) | Known research and engineering challenges with mitigations |
| [08_future_improvements.md](08_future_improvements.md) | Phased improvement roadmap |
| [09_glossary.md](09_glossary.md) | Token definitions, system terms, financial and ML terminology |
| [10_implementation_guide.md](10_implementation_guide.md) | **Start here to build.** Project structure, MT5 setup, phase-by-phase plan |
| [11_data_pipeline_explained.md](11_data_pipeline_explained.md) | **Data pipeline deep-dive.** Exact steps, actual numbers, what train/val/test look like |

---

## System in One Paragraph

AITrade converts 15-minute Forex candlestick data into a symbolic market language of ~27 tokens,
then trains a custom 2.5M-parameter transformer encoder from scratch to classify each candle's
next 3-candle window as BUY, HOLD, or SELL using triple-barrier labeling (ATR-adjusted barriers).
**Training uses pre-downloaded CSV files** (15-min OHLCV, already on disk — no API needed).
**Live inference connects to MetaTrader 5** via the MT5 Python package to fetch the 64 most
recently completed candles on demand. The model outputs calibrated probabilities; a trading
engine filters signals above 60% confidence and sizes positions by probability edge.

---

## Key Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Model | Custom Transformer Encoder (~2.5M params) | 27-token vocab — no benefit from text pretraining |
| Training | From scratch, no pretrained weights | New market-specific vocabulary |
| Training data source | Pre-downloaded CSV files (local disk) | Stable, no API dependency during training |
| Live data source | MetaTrader 5 Python API (`start_pos=1`) | Direct access to broker candles; `pos=0` skipped (forming bar) |
| OS for inference | Windows (MT5 Python API is Windows-only) | MT5 terminal runs on Windows |
| Labeling method | Triple-barrier (ATR-adjusted, k=0.75, H=3) | Aligned with actual trading outcomes |
| Prediction horizon | 3 candles (45 minutes) | Single candle too noisy; 3 gives move time to develop |

---

## Data Sources

```
TRAINING                          INFERENCE (live)
────────                          ────────────────
CSV files on disk                 MetaTrader 5 terminal
  EURUSD_M15_raw.csv    →           mt5.copy_rates_from_pos(
  (pre-downloaded,                      "EURUSD",
   already available)                   mt5.TIMEFRAME_M15,
                                        start_pos=1,    ← skip forming bar
                                        count=64
                                    )
```

---

## Example Output

```
Time: 12:45 UTC+2 (MT5 server time)
Pair: EURUSD  |  Horizon: 3 candles (45 min)

SELL = 0.11
HOLD = 0.17
BUY  = 0.72

NEXT = BUY (p=0.72)  →  Entry signal generated
```

---

## Project Status

**Phase:** Pre-implementation — documentation complete, ready to build
**Version:** 0.3 — MT5 + CSV real-world specification

**Build order:** Follow [10_implementation_guide.md](10_implementation_guide.md) phase by phase.
Do not skip Phase 1 (CSV loading + labels) or Phase 4 (backtesting).
