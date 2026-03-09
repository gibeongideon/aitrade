# 02 — Data Representation

## Raw Data Source

The base dataset consists of 15-minute Forex OHLCV candles with derived technical indicators.

### Base Fields per Candle

| Field | Description |
|-------|-------------|
| `time` | Candle open timestamp (UTC) |
| `mid_o` | Mid-price Open |
| `mid_h` | Mid-price High |
| `mid_l` | Mid-price Low |
| `mid_c` | Mid-price Close |
| `Volume` | Tick volume or traded volume |
| `MA_16` | 16-period simple moving average |
| `MA_32` | 32-period simple moving average |
| `MA_64` | 64-period simple moving average |

### Derived Features (computed from base fields)

| Feature | Description |
|---------|-------------|
| `return` | Percentage or log return: `(mid_c - mid_o) / mid_o` |
| `body_ratio` | Body size relative to total range: `abs(mid_c - mid_o) / (mid_h - mid_l)` |
| `upper_wick` | Upper wick size: `mid_h - max(mid_o, mid_c)` |
| `lower_wick` | Lower wick size: `min(mid_o, mid_c) - mid_l` |
| `vol_ratio` | Volume relative to rolling N-period average |
| `ma_spread_16_64` | Difference between short and long MA (trend proxy) |
| `ma_cross` | Detected crossover event between MAs |

---

## Market Language Tokenization

Each candle is converted from numeric values into a set of discrete symbolic tokens.
This transforms a row of numbers into a short "sentence" describing that candle's
market structure.

### Design Principle

Numeric values are mapped to ordered categorical tokens using quantile-based or
threshold-based discretization. The boundaries for each token category are determined
from the training data distribution (e.g., rolling percentiles or fixed statistical thresholds).

---

## Token Vocabulary

### 1. Price Movement Tokens (`RET_*`)

Represents the candle's return discretized into directional bins.

| Token | Meaning |
|-------|---------|
| `RET_D3` | Strong downward move (e.g., bottom 5% of returns) |
| `RET_D2` | Moderate down move |
| `RET_D1` | Slight down move |
| `RET_FLAT` | Near-zero return |
| `RET_U1` | Slight up move |
| `RET_U2` | Moderate up move |
| `RET_U3` | Strong upward move (e.g., top 5% of returns) |

### 2. Candle Body Tokens (`BODY_*`)

Represents the size of the candle body relative to its total range (high - low).

| Token | Meaning |
|-------|---------|
| `BODY_S` | Small body (indecision, doji-like) |
| `BODY_M` | Medium body |
| `BODY_L` | Large body (conviction move) |

### 3. Wick Structure Tokens (`WICK_*`)

Captures rejection patterns via wick presence and location.

| Token | Meaning |
|-------|---------|
| `WICK_NONE` | No significant wicks |
| `WICK_TOP` | Upper wick dominant (bearish rejection) |
| `WICK_BOTTOM` | Lower wick dominant (bullish rejection) |
| `WICK_BOTH` | Wicks on both sides (indecision/volatility) |

### 4. Volume Regime Tokens (`VOL_*`)

Represents volume relative to recent rolling history.

| Token | Meaning |
|-------|---------|
| `VOL_LOW` | Below-average volume |
| `VOL_NORMAL` | Average volume |
| `VOL_HIGH` | Above-average volume |
| `VOL_SPIKE` | Extreme volume event |

### 5. Trend Structure Tokens (`TREND_*`)

Derived from the relationships between the three moving averages (MA_16, MA_32, MA_64).

| Token | Meaning |
|-------|---------|
| `TREND_UP` | MAs aligned bullishly (MA_16 > MA_32 > MA_64) |
| `TREND_DOWN` | MAs aligned bearishly (MA_16 < MA_32 < MA_64) |
| `TREND_MIX` | MAs not aligned (transitional or ranging) |
| `TREND_CROSS_UP` | Short MA just crossed above long MA (bullish signal) |
| `TREND_CROSS_DOWN` | Short MA just crossed below long MA (bearish signal) |

---

## Candle Sentence Structure

Each candle is represented as an ordered sequence of tokens:

```
RET_* BODY_* WICK_* VOL_* TREND_*
```

### Example Candles

```
RET_U1 BODY_M WICK_TOP    VOL_HIGH   TREND_UP
RET_U2 BODY_L WICK_NONE   VOL_HIGH   TREND_UP
RET_FLAT BODY_S WICK_BOTH VOL_NORMAL TREND_UP
RET_D1 BODY_M WICK_BOTTOM VOL_HIGH   TREND_MIX
RET_D2 BODY_M WICK_NONE   VOL_HIGH   TREND_DOWN
```

---

## Sequence Construction

A window of the last N candles is assembled into a multi-candle sequence (the "market paragraph"):

```
[CLS]
RET_U1 BODY_M WICK_TOP    VOL_HIGH   TREND_UP
RET_U2 BODY_L WICK_NONE   VOL_HIGH   TREND_UP
RET_FLAT BODY_S WICK_BOTH VOL_NORMAL TREND_UP
RET_D1 BODY_M WICK_BOTTOM VOL_HIGH   TREND_MIX
...
[SEP]
```

| Parameter | Value |
|-----------|-------|
| Window size | 64 candles |
| Time coverage | ~16 hours of market context |
| Tokens per candle | 5 |
| Total tokens per sequence | ~320 tokens (plus special tokens) |

---

## Notes on Tokenization Design

- Token thresholds are derived from the training data distribution — they should not be fixed constants
- Quantile-based thresholds are preferred over fixed value thresholds to handle different pairs and market regimes
- The vocabulary is intentionally small (~20 tokens) to keep the representation compact and learnable
- Future versions may add tokens for additional features (e.g., volatility regime, session time, day-of-week)
