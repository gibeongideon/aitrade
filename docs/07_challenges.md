# 07 — Known Challenges

## Overview

This document catalogs the primary research, engineering, and operational challenges
expected in this project. Understanding these challenges upfront guides design decisions
and prevents common failure modes.

---

## 1. Label Design

**Severity: Critical**

The quality of the entire system depends on whether the labels accurately reflect
tradeable directional opportunities.

| Risk | Description |
|------|-------------|
| Threshold too low | Labels become noisy; model learns from false signals |
| Threshold too high | BUY/SELL events become too rare; severe class imbalance |
| Fixed threshold | Does not adapt to changing volatility; poor generalization |
| Single-horizon labeling | Misses multi-period setups |

**Mitigation:**
- Use ATR-relative thresholds
- Experiment with triple-barrier labeling
- Validate label quality by checking if BUY labels are followed by actual upward moves in backtesting

---

## 2. Data Quantity and Overfitting

**Severity: High**

Forex markets are noisy. A small dataset combined with a complex model will overfit.

| Risk | Description |
|------|-------------|
| Insufficient history | Model memorizes patterns rather than learning generalizable structure |
| Noisy labels | Model trains on signal + noise; cannot generalize |
| Small vocabulary overlap | Market tokens may appear in unnatural combinations for the LLM |

**Mitigation:**
- Use LoRA to minimize trainable parameters
- Apply aggressive regularization (dropout, weight decay)
- Collect multi-year data across multiple pairs
- Use data augmentation techniques (e.g., slight perturbation of thresholds)

---

## 3. Tokenization Quality

**Severity: High**

The market language representation must meaningfully capture market structure.
Poor tokenization destroys information before the model even sees it.

| Risk | Description |
|------|-------------|
| Too coarse | Tokens lose important numeric distinctions |
| Too fine-grained | Vocabulary too large; data too sparse per token combination |
| Fixed thresholds | Tokenization fails to adapt to different pairs or volatility regimes |
| Loss of continuity | Discretization creates artificial discontinuities near boundaries |

**Mitigation:**
- Use quantile-based thresholds derived from training data
- Validate that tokens have balanced class frequencies
- Consider soft or overlapping bin boundaries for boundary cases
- Test tokenization by visual inspection on known market events

---

## 4. Probability Calibration

**Severity: Medium**

Neural network classifiers output probabilities that are often overconfident or
underconfident. Uncalibrated probabilities make the trading engine unreliable.

| Risk | Description |
|------|-------------|
| Overconfidence | Model says p=0.9 but actual win rate is 0.6; oversized positions |
| Underconfidence | Model says p=0.55 but actual win rate is 0.75; undersized positions |
| Calibration drift | Model calibration degrades over time as markets evolve |

**Mitigation:**
- Apply post-hoc calibration (temperature scaling) on validation set
- Monitor ECE (Expected Calibration Error) as a core metric
- Re-calibrate periodically using recent out-of-sample data

---

## 5. Market Regime Changes

**Severity: High**

Forex markets are non-stationary. Patterns that held in one period may fail in another.
A model trained on historical data may underperform in changed market conditions.

| Risk | Description |
|------|-------------|
| Volatility regime shift | Low-vol model fails in high-vol environment |
| Trend-to-range transition | Model trained on trending markets fails in ranging conditions |
| Macro event structural breaks | Central bank policy changes alter market dynamics |
| Correlation breakdowns | Multi-pair strategies fail when correlations shift |

**Mitigation:**
- Implement walk-forward re-training
- Add regime detection as a pre-filter
- Monitor prediction confidence and win rate in production
- Build in automatic alerts when model performance degrades

---

## 6. Look-Ahead Bias

**Severity: Critical**

Any accidental leakage of future information into training features or labels will
produce unrealistically good backtest results that fail in live trading.

| Risk | Description |
|------|-------------|
| Feature computed from future data | E.g., MA computed using future candles |
| Label computed at wrong timestamp | Label assigned to current candle using non-future data |
| Data leakage via normalization | Normalizing features using statistics from full dataset including future |
| Test set contamination | Random split instead of time-based split |

**Mitigation:**
- All features computed strictly from past candles
- Labels computed from strictly future candles only
- Normalization parameters (quantiles, ATR) computed only on training split
- Mandatory time-based train/val/test splits
- Code review specifically targeting temporal data flow

---

## 7. Market Microstructure Noise

**Severity: Medium**

15-minute candles still contain significant noise. The signal-to-noise ratio in
short-term Forex prediction is inherently low.

| Risk | Description |
|------|-------------|
| Random walk behavior | Many short-term moves are effectively unpredictable |
| Spread costs | Transaction costs reduce realized edge significantly |
| Slippage | At entry and exit, prices may differ from expected levels |

**Mitigation:**
- Include transaction cost modeling in all backtests
- Only act on signals with edge significantly above breakeven cost
- Consider moving to 1H candles for cleaner signal if 15-min results are poor

---

## 8. Pretrained LLM Mismatch

**Severity: Medium**

Pretrained LLMs are trained on natural language, not market data. The market token
vocabulary is entirely new to the model, meaning the pretrained weights provide
structure (attention, positional encoding) but not domain knowledge.

| Risk | Description |
|------|-------------|
| Token embedding cold start | New market tokens have random embeddings initially |
| Attention patterns not aligned | Pretrained attention may not transfer to market sequences |
| Over-reliance on pretraining | Fine-tuning may not be sufficient to overcome domain mismatch |

**Mitigation:**
- Allow market token embeddings to train freely (not frozen)
- Use longer fine-tuning schedules
- Consider training a small transformer from scratch as a comparison baseline
- Evaluate whether pretraining actually helps vs. a scratch-trained model of similar size
