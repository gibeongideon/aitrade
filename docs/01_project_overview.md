# 01 — Project Overview

## Objective

Design a probabilistic forecasting system for Forex markets that predicts the most likely
short-term price action for the next 15-minute period using a language-model approach applied
to structured market data.

The system outputs interpretable directional signals with calibrated confidence scores:

```
NEXT = BUY  (p=0.72)
NEXT = SELL (p=0.68)
NEXT = HOLD (p=0.61)
```

The probability value represents the model's calibrated confidence that a statistically
meaningful directional move will occur within the next 15-minute window.

---

## Core Motivation

Traditional machine learning approaches treat OHLCV (Open/High/Low/Close/Volume) data as
continuous numerical time series fed directly into regression or classification models.

This project takes a fundamentally different approach: it **converts market data into a
symbolic language**, then applies pretrained transformer language models (LLMs) to learn
market structure, temporal patterns, and regime behavior.

### The Core Idea

```
Raw Market Data
      |
      v
Market Language Tokenization
      |
      v
Transformer (LLM) Sequence Model
      |
      v
Probability Distribution over {SELL, HOLD, BUY}
      |
      v
Trading Signal
```

By representing price action as tokens in a structured vocabulary, the system enables
transformers to apply their sequence modeling strengths — attention over long contexts,
pattern recognition, regime awareness — to the domain of financial markets.

---

## What the System Does NOT Do

- Does not predict exact future prices
- Does not generate deterministic binary signals
- Does not rely on fixed technical analysis rules
- Does not require real-time streaming data at inference (works on completed candles)

---

## What the System DOES Do

- Detects high-probability short-term directional opportunities
- Outputs calibrated probability distributions
- Enables risk-adjusted trade selection and position sizing
- Provides interpretable symbolic representations of market state

---

## Target Market and Timeframe

| Property | Value |
|----------|-------|
| Market | Forex (currency pairs) |
| Base candle timeframe | 15 minutes |
| Prediction horizon | Next 15 minutes (1 candle ahead) |
| Context window | Last 64 candles (~16 hours of market context) |

---

## Intended Users

- Quantitative researchers exploring LLM applications in finance
- Algorithmic traders building systematic strategies
- Engineers designing probabilistic decision systems for trading

---

## Intended Outcome

A deployable system that provides real-time probabilistic forecasts of short-term Forex
market direction, enabling algorithmic strategies to:

1. Detect tradeable opportunities above a confidence threshold
2. Avoid entering positions during low-confidence market conditions
3. Manage risk dynamically through probability-proportional position sizing
4. Operate across multiple currency pairs with a shared model architecture
