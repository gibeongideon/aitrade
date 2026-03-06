# 09 — Glossary

## System-Specific Terms

| Term | Definition |
|------|-----------|
| **Market Language** | The symbolic token-based representation of OHLCV candlestick data used as input to the LLM |
| **Candle Sentence** | The 5-token description of a single candle: `RET_* BODY_* WICK_* VOL_* TREND_*` |
| **Market Paragraph** | A sequence of 64 candle sentences representing ~16 hours of market context |
| **Edge** | The probability difference between directional classes: `p_buy - p_sell`; represents model conviction |
| **Entry Threshold** | Minimum probability value required for the trading engine to generate a trade signal |
| **Prediction Horizon** | The future time window over which the model predicts directional movement (default: 1 candle = 15 minutes) |
| **Probability Calibration** | Post-hoc adjustment of model output probabilities so that `p=0.7` corresponds to actual ~70% accuracy |

---

## Token Vocabulary Reference

### Price Movement Tokens

| Token | Meaning |
|-------|---------|
| `RET_D3` | Strong downward return (extreme percentile) |
| `RET_D2` | Moderate downward return |
| `RET_D1` | Slight downward return |
| `RET_FLAT` | Near-zero return |
| `RET_U1` | Slight upward return |
| `RET_U2` | Moderate upward return |
| `RET_U3` | Strong upward return (extreme percentile) |

### Candle Body Tokens

| Token | Meaning |
|-------|---------|
| `BODY_S` | Small body relative to range (indecision) |
| `BODY_M` | Medium body |
| `BODY_L` | Large body (conviction) |

### Wick Structure Tokens

| Token | Meaning |
|-------|---------|
| `WICK_NONE` | No significant wicks |
| `WICK_TOP` | Significant upper wick (bearish rejection) |
| `WICK_BOTTOM` | Significant lower wick (bullish rejection) |
| `WICK_BOTH` | Significant wicks on both sides |

### Volume Regime Tokens

| Token | Meaning |
|-------|---------|
| `VOL_LOW` | Below-average volume relative to recent history |
| `VOL_NORMAL` | Average volume |
| `VOL_HIGH` | Above-average volume |
| `VOL_SPIKE` | Extreme volume event |

### Trend Structure Tokens

| Token | Meaning |
|-------|---------|
| `TREND_UP` | Bullish MA alignment: MA_16 > MA_32 > MA_64 |
| `TREND_DOWN` | Bearish MA alignment: MA_16 < MA_32 < MA_64 |
| `TREND_MIX` | MAs not clearly aligned; transitional or ranging |
| `TREND_CROSS_UP` | Short MA just crossed above long MA |
| `TREND_CROSS_DOWN` | Short MA just crossed below long MA |

### Special Sequence Tokens

| Token | Meaning |
|-------|---------|
| `[CLS]` | Classification token; placed at start of sequence; final hidden state used for prediction |
| `[SEP]` | Separator token; placed at end of sequence |

---

## Financial Terms

| Term | Definition |
|------|-----------|
| **OHLCV** | Open, High, Low, Close, Volume — the standard representation of a price candle |
| **ATR** | Average True Range — a measure of market volatility over N periods |
| **MA** | Moving Average — smoothed price over N periods |
| **MA Crossover** | Event when a shorter-period MA crosses a longer-period MA |
| **Pip** | Smallest standard price increment in Forex (e.g., 0.0001 for EUR/USD) |
| **Spread** | Difference between bid and ask price; represents transaction cost |
| **Drawdown** | Decline from a peak portfolio value to a subsequent trough |
| **Sharpe Ratio** | Risk-adjusted return: mean return divided by standard deviation of returns |
| **Win Rate** | Percentage of trades that resulted in profit |
| **Expectancy** | Average expected profit per trade: `(win_rate × avg_win) - (loss_rate × avg_loss)` |
| **Look-ahead Bias** | Accidental use of future data in training; produces false backtest performance |
| **Walk-Forward Validation** | Testing methodology where model is trained only on past data and tested strictly on future data |

---

## Machine Learning Terms

| Term | Definition |
|------|-----------|
| **LLM** | Large Language Model — a transformer-based model pretrained on large text corpora |
| **LoRA** | Low-Rank Adaptation — a parameter-efficient fine-tuning technique that injects small trainable matrices into pretrained model layers |
| **Fine-tuning** | Continuing to train a pretrained model on domain-specific data |
| **PEFT** | Parameter-Efficient Fine-Tuning — methods for adapting large models with minimal trainable parameters |
| **Softmax** | Function that converts a vector of raw scores into a probability distribution summing to 1 |
| **Cross-Entropy Loss** | Standard classification training loss measuring the difference between predicted and true probability distributions |
| **ECE** | Expected Calibration Error — metric measuring how well predicted probabilities match actual outcome frequencies |
| **Temperature Scaling** | Post-training calibration method using a single scalar parameter to adjust model confidence |
| **Triple-Barrier Method** | Labeling technique using upper, lower, and time barriers to define trade outcomes |
| **Regime** | A period of market behavior with distinct statistical properties (e.g., trending, ranging, high-volatility) |
| **Epistemic Uncertainty** | Uncertainty arising from limited model knowledge, as opposed to inherent data noise |
