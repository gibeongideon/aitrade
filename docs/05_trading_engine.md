# 05 — Trading Decision Engine

## Overview

The trading engine consumes probability outputs from the model and translates them into
structured trading decisions. It is responsible for:

- Filtering low-confidence predictions
- Sizing positions based on probability edge
- Managing risk parameters
- Issuing final trade instructions

The engine is intentionally separate from the model to allow independent tuning of
trading logic without retraining the model.

---

## Signal Interpretation

The model outputs a three-class probability distribution:

```
p_sell  = probability of meaningful downward move
p_hold  = probability of no significant move
p_buy   = probability of meaningful upward move
```

Constraint: `p_sell + p_hold + p_buy = 1.0`

### Directional Signal

```
signal = argmax(p_sell, p_hold, p_buy)
```

If `signal == BUY` and `p_buy > entry_threshold` → consider long entry
If `signal == SELL` and `p_sell > entry_threshold` → consider short entry
Otherwise → no trade (stay flat or maintain existing position)

---

## Entry Threshold

A minimum confidence threshold filters out low-conviction predictions.

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `entry_threshold` | Minimum probability required to enter a trade | 0.60 |

Only signals where the directional probability exceeds `entry_threshold` generate a trade.

This is one of the most important tunable parameters. It controls the
**precision vs. recall tradeoff**:

- Higher threshold → fewer signals, higher expected quality
- Lower threshold → more signals, more noise

The `entry_threshold` must be tuned using the **validation set only**, then locked before
touching the test set or running backtesting.

---

## Probability Edge

The **edge** is a measure of how confident the model is in a direction relative to the opposite:

```
edge = p_buy - p_sell      (for a BUY signal)
edge = p_sell - p_buy      (for a SELL signal)
```

Edge ranges from 0 (no directional conviction) to 1.0 (maximum conviction).

---

## Position Sizing

Position size is proportional to the probability edge, scaled by a maximum position size:

```
position_size = base_size * edge
```

### Example

| Scenario | p_buy | p_sell | edge | Position Size |
|----------|-------|--------|------|---------------|
| Strong signal | 0.72 | 0.11 | 0.61 | 61% of max |
| Moderate signal | 0.62 | 0.20 | 0.42 | 42% of max |
| Weak signal | 0.52 | 0.30 | 0.22 | No trade (below threshold) |

This approach naturally **risk-weights** trades: the model is wrong less often at high edge,
so larger positions at high edge improve the overall risk-adjusted return.

---

## Risk Management Parameters

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `entry_threshold` | Min probability to enter | 0.60 |
| `max_position_size` | Maximum trade size (% of capital) | 2% of capital |
| `stop_loss_atr_mult` | Stop-loss distance in ATR multiples | 1.5 × ATR |
| `take_profit_atr_mult` | Take-profit distance in ATR multiples | 2.0 × ATR |
| `max_open_trades` | Maximum concurrent open positions | 1 (initial) |
| `hold_period` | Maximum candles to hold if target/stop not hit | 4 candles |

---

## Trade Lifecycle

```
1. New candle closes
        |
        v
2. Model inference → [p_sell, p_hold, p_buy]
        |
        v
3. Check: max(p) > entry_threshold?
        |
   Yes  |  No
        |----> No action
        v
4. Compute edge
        |
        v
5. Size position proportional to edge
        |
        v
6. Place order with stop-loss and take-profit
        |
        v
7. Monitor until: stop hit / target hit / max hold period exceeded
        |
        v
8. Close trade, log result
```

---

## Decision Rule Summary

```python
# Pseudocode — not implementation

p_sell, p_hold, p_buy = model.predict(sequence)

if p_buy > entry_threshold:
    edge = p_buy - p_sell
    size = base_size * edge
    place_trade(direction=BUY, size=size, stop=ATR*1.5, target=ATR*2.0)

elif p_sell > entry_threshold:
    edge = p_sell - p_buy
    size = base_size * edge
    place_trade(direction=SELL, size=size, stop=ATR*1.5, target=ATR*2.0)

else:
    pass  # No trade
```

---

## Backtesting Methodology

### Why Backtesting is Non-Negotiable

The model may learn genuine patterns or may overfit to noise. The only way to know
before risking real money is a rigorous backtest on data the model has never seen,
simulated exactly as trading would occur in real time.

### The Key Distinction: Labels vs. Backtesting

There is a subtle but critical difference between how labels are computed and how
the backtester works:

| Aspect | Label Generation (training) | Backtester (evaluation) |
|--------|----------------------------|------------------------|
| Looks at future data? | YES — needed to assign BUY/SELL | NO — simulates real-time execution |
| Entry price | N/A (labels don't trade) | Open of the NEXT candle after signal |
| Stop/target | N/A | ATR computed at signal time, past data only |
| Purpose | Teach the model what happens | Validate that the model is tradeable |

Labels must look ahead to know what happened — that is correct and necessary.
The backtester must **never** look ahead — it simulates what a real trader would experience.

---

### Rule 1: Use Only the Test Set

The backtester runs exclusively on the **held-out test set** — the final 15% of data
by time that was never used during training, validation, or hyperparameter tuning.

```
Full dataset (time-ordered)
|────────────────────────────────────────────────────────|
|    Train (70%)    |   Val (15%)   |   Test (15%)       |
|────────────────────────────────────────────────────────|
                                     ↑
                               Backtester only runs here
```

The entry threshold (`entry_threshold`) and label k-value are tuned on the validation
set and then **frozen** before the test set is touched. If you adjust any parameter
after seeing test results, you have overfit to the test set.

---

### Rule 2: Entry at Next Candle Open, Not Signal Candle Close

A signal is generated when the candle at time `T` closes and the model runs inference.
At that moment, the candle at `T` is already closed — you cannot enter at its close price
in real trading because you didn't have the signal until after it closed.

```
Candle T closes → Model runs → Signal generated
                                      |
                                      v
                          Entry at open of Candle T+1
```

The backtester must use `open[T+1]` as the entry price, not `close[T]`.
Using `close[T]` is a common form of look-ahead bias in backtesting.

---

### Rule 3: Stop and Target from ATR at Signal Time

The ATR used to set stop-loss and take-profit levels must be computed from
candles up to and including candle `T` only — the same data available at signal time.

```
atr_at_signal = ATR computed from candles [T-13 ... T]   ← only past data
stop_price  = entry_price - (atr_at_signal * 1.5)        ← for BUY trade
target_price = entry_price + (atr_at_signal * 2.0)       ← for BUY trade
```

Never use a "global" ATR computed from the full dataset — that leaks future volatility
information into stop/target placement.

---

### Rule 4: Exit Simulation Uses Only Candle OHLC, In Order

After entry at `open[T+1]`, the backtester steps forward candle by candle checking exits:

```
For each candle C from T+1 onward (up to max hold_period):

  For BUY trade:
    if low[C]  <= stop_price  → trade closed at stop_price  (LOSS)
    if high[C] >= target_price → trade closed at target_price (WIN)

  If neither hit after hold_period candles:
    trade closed at close[hold_period last candle] (TIME EXIT)
```

Within a single candle, check the stop first (conservative — gives worst-case assumption).
This avoids the ambiguity of which happened first inside a candle.

---

### Rule 5: Include Spread Cost on Every Trade

Every trade entry deducts the broker spread cost from the PnL:

```
For BUY:   effective_entry = open[T+1] + spread
For SELL:  effective_entry = open[T+1] - spread
```

Use the spread value from your broker (e.g., 1.0–1.5 pips for EUR/USD).
Ignoring spread is a common mistake that makes strategies appear profitable
in backtesting but losing in live trading.

---

### Walk-Forward Validation (Recommended for Production)

A single train/test split shows performance on one time period. Walk-forward
validation tests across multiple periods, giving a more robust estimate of
how the model generalises over time.

```
Window 1:  Train on months  1–18 │ Test on months 19–21
Window 2:  Train on months  1–21 │ Test on months 22–24
Window 3:  Train on months  1–24 │ Test on months 25–27
Window 4:  Train on months  1–27 │ Test on months 28–30
           ↑                       ↑
           expanding train window  3-month test window (never overlaps previous test)
```

Each window trains a fresh model on the expanded training set and tests on the
3-month window immediately following. Test windows never overlap each other.

The backtester collects results across all test windows and reports aggregate metrics.
If performance is consistent across windows, the model generalises. If it degrades
sharply in later windows, the model may be overfitting to earlier market conditions.

---

### Backtesting Metrics to Report

| Metric | Description | Minimum Acceptable |
|--------|-------------|-------------------|
| Total trades | Number of signals above threshold | > 50 |
| Win rate | % of trades that hit target before stop | > 50% |
| Expectancy | Average PnL per trade (in pips or %) | Positive |
| Profit factor | Gross profit / Gross loss | > 1.2 |
| Sharpe ratio | Annualised return / volatility | > 0.5 |
| Max drawdown | Worst peak-to-trough capital decline | < 15% |
| Avg hold time | Average candles held per trade | Informational |
| Calibration check | Does p=0.70 signal win ~70% of the time? | Within 10% |

If any metric falls below minimum acceptable:
- First revisit label design (k value in triple-barrier) — this is the most common cause
- Then check entry threshold (raise it to filter more aggressively)
- Do not modify the model architecture — data quality drives results more than model complexity

---

### Common Backtesting Mistakes

| Mistake | Consequence | Prevention |
|---------|------------|-----------|
| Entering at signal candle close | Unrealistically good fills | Always enter at next candle open |
| Using future ATR for stops | Stops are placed too accurately | Compute ATR only from past candles |
| Tuning on test set | Overfitting to test period | Lock parameters before touching test set |
| Ignoring spread | Strategy appears profitable but isn't | Always deduct spread per trade |
| Random train/test split | Look-ahead bias from future data in train | Always split by time index |
| Single test window | Results may be period-specific | Use walk-forward validation |
| Not checking trade count | 10 trades is not statistically meaningful | Require > 50 trades minimum |
