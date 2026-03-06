# 04 — Prediction Target & Label Design

## Overview

The model predicts a directional trading opportunity over the next candle horizon, not an
exact future price. Labels are derived programmatically from future price data using the
**Triple-Barrier Method** — the primary labeling strategy for this system.

This design is critical. Poor label design will undermine the entire system regardless of
model quality.

---

## Prediction Horizon

| Property | Value |
|----------|-------|
| Base horizon | 3 candles (45 minutes) |
| Why 3 candles instead of 1 | Single candle is too noisy; 3 candles gives a move time to develop |
| Label computed from | Future candles' highs, lows, and closes relative to current close |

> **Note:** The model still runs every candle (every 15 minutes), but each label is computed
> over a 3-candle window. This is a realistic horizon for short-term directional trading.

---

## Label Classes

| Class | Integer ID | Meaning |
|-------|-----------|---------|
| `SELL` | 0 | Stop-loss direction hit before take-profit (downward move) |
| `HOLD` | 1 | Neither barrier hit within the time horizon |
| `BUY`  | 2 | Take-profit direction hit before stop-loss (upward move) |

---

## Primary Labeling Method: Triple-Barrier

The triple-barrier method treats each trade as if you entered at the current candle's
close. It checks which of three events happens first over the next H candles:

```
                    Upper Barrier (Take-Profit)
current_close + k * ATR  ─────────────────────────── → label = BUY
                                          ^
                    Price path          /
                                      /
current_close  ──────────────────────               → label = HOLD (time ran out)
                                      \
                                        \
                                          v
current_close - k * ATR  ─────────────────────────── → label = SELL
                    Lower Barrier (Stop-Loss)

                    |←──── H candles (time barrier) ────→|
```

### Barrier Configuration (starting values)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `k` | 0.75 | Barrier distance as ATR multiple — tune during experimentation |
| `ATR period` | 14 candles | Standard ATR lookback |
| `H` (time barrier) | 3 candles | Maximum horizon before HOLD is assigned |

### Label Assignment Logic

```
For each labeled candle at time t:
  entry_price = close[t]
  upper = entry_price + k * ATR[t]
  lower = entry_price - k * ATR[t]

  For each future candle t+1, t+2, ... t+H:
    if high[t+i] >= upper:
        label = BUY    (take-profit hit first)
        break
    if low[t+i] <= lower:
        label = SELL   (stop-loss hit first)
        break
  else:
    label = HOLD       (time barrier — neither hit)
```

### Why Triple-Barrier is Better Than Simple Excursion Labels

| Simple Excursion | Triple-Barrier |
|-----------------|----------------|
| Labels based on raw future high/low | Labels based on which barrier is hit first |
| Ignores order of events | Respects time — who got there first matters |
| Not aligned with actual trading | Directly mirrors a real entry + target + stop trade |
| Higher label noise | Cleaner labels, better model training |

---

## Handling Class Imbalance

Markets spend significant time ranging. Expect approximately:

| Class | Typical Frequency |
|-------|-----------------|
| HOLD | 40–60% |
| BUY  | 20–30% |
| SELL | 20–30% |

**Strategy:** Use **weighted cross-entropy loss** during training. Weights are computed
from the inverse class frequency in the training set:

```
w_class = total_samples / (3 * count_of_class)
```

This is automatic and requires no manual oversampling.

If HOLD frequency exceeds 65%, increase `k` (move barriers further out) to produce
fewer but higher-quality directional labels.

---

## Tuning the `k` Parameter

`k` is the most important label design parameter. It controls:

- **Too small (k < 0.5):** Barriers too close → noisy labels → model learns noise
- **Too large (k > 1.5):** Events too rare → class imbalance → model biased to HOLD
- **Target:** BUY + SELL combined frequency of 40–55%

Run a quick scan before training to verify label distribution:

```
k=0.50  →  check BUY+SELL %
k=0.75  →  check BUY+SELL %
k=1.00  →  check BUY+SELL %
k=1.25  →  check BUY+SELL %
```

Pick the `k` value that yields 40–55% directional labels on the training set.

---

## Look-Ahead Bias Prevention

This is non-negotiable:

- Labels for candle at time `t` use ONLY candles `t+1` through `t+H`
- Input features for candle at time `t` use ONLY candles `t-64` through `t`
- ATR at time `t` uses only past candles — never future candles
- Tokenizer thresholds computed only on the training split — never on val/test data
- Train/val/test split is strictly time-ordered — no random shuffling ever

---

## Output Format

At inference time, the model returns three probabilities:

```
SELL = 0.11
HOLD = 0.17
BUY  = 0.72
```

A trading signal is issued only when the highest probability exceeds the entry threshold:

```
NEXT = BUY (p=0.72)
```

If no class exceeds the threshold → no signal (system stays flat or holds existing position).
