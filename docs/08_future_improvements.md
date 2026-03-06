# 08 — Future Improvements & Research Directions

## Overview

This document outlines planned and potential improvements beyond the base system.
Organised by research maturity — from near-term engineering improvements to longer-term
research directions.

---

## Near-Term Improvements (Phase 2)

### 1. Triple-Barrier Labeling

Already implemented as the primary method in Phase 1. Phase 2 will tune the parameters
(`k`, `H`) systematically using the validation set and walk-forward results.

---

### 2. Walk-Forward Re-Training

Implement continuous model updating to combat regime drift:

```
Window 1:  Train on months  1–18  │  Test on months 19–21
Window 2:  Train on months  1–21  │  Test on months 22–24
Window 3:  Train on months  1–24  │  Test on months 25–27
...
```

Each window trains a fresh model on the expanded set. Tokenizer thresholds are
recomputed for each window's training data. Results are aggregated across windows
to assess generalisation over time.

---

### 3. Multi-Timeframe Context

Include higher timeframe signals as additional input tokens prepended to the sequence:

| Timeframe | Role |
|-----------|------|
| 15-min | Primary prediction target |
| 1-Hour | Intermediate trend context |
| 4-Hour | Macro trend filter |
| Daily | Regime identification |

Encoded with the same token vocabulary. Special separator tokens `[TF1H]`, `[TF4H]`
mark boundaries between timeframes in the sequence.

---

### 4. Cross-Pair Information

Include correlated currency pair data in the input:

- EUR/USD, GBP/USD, USD/JPY are highly correlated
- Market-wide risk events affect multiple pairs simultaneously
- Cross-pair tokens are interleaved or prepended with separator tokens

---

### 5. Volatility Conditioning

Explicitly condition predictions on the current volatility regime:

- Add a `VOL_REGIME` token describing ATR relative to its historical range
- Train separate classification heads for different volatility regimes
- Or use a gating mechanism weighting predictions by detected regime

---

## Medium-Term Research Directions (Phase 3)

### 6. Regime Detection Module

A dedicated regime classifier running in parallel with the directional model:

| Regime | Description |
|--------|-------------|
| TREND_BULL | Sustained upward price structure |
| TREND_BEAR | Sustained downward price structure |
| RANGE | Horizontal price oscillation |
| VOLATILE | High ATR, erratic moves |
| BREAKOUT | Transition from range to trend |

Trading engine applies regime-specific rules:
- TREND → hold positions longer, larger size
- RANGE → tighter targets, mean-reversion logic
- VOLATILE → reduce size or stay flat

---

### 7. Richer Token Vocabulary

Expand the market language with additional token categories:

| New Token Category | Examples |
|-------------------|---------|
| Session tokens | `SESSION_ASIA`, `SESSION_LONDON`, `SESSION_NY`, `SESSION_OVERLAP` |
| Day-of-week | `DOW_MON`, `DOW_FRI` |
| Volatility regime | `VREGIME_LOW`, `VREGIME_HIGH` |
| Support/resistance | `NEAR_HIGH`, `NEAR_LOW`, `MID_RANGE` |
| Momentum | `MOM_ACCEL`, `MOM_DECEL`, `MOM_FLAT` |
| Candle pattern | `DOJI`, `ENGULF_BULL`, `ENGULF_BEAR`, `HAMMER` |

---

### 8. Ensemble Methods

| Approach | Description |
|----------|------------|
| Model ensemble | Average predictions from models trained on different windows |
| Timeframe ensemble | Combine 15-min, 1H, and 4H model predictions |
| Stacked ensemble | Meta-model trained on base model predictions |

---

## Long-Term Research Directions (Phase 4+)

### 9. Reinforcement Learning Integration

#### Why RL After Supervised Learning?

The supervised model is trained to maximise **classification accuracy** — to correctly
predict BUY, HOLD, or SELL labels. But classification accuracy is not what we ultimately
care about. We care about **cumulative trading profit**.

These goals are related but not identical:
- A model with 60% directional accuracy can lose money if it is right on small moves
  and wrong on large moves
- A model with 55% accuracy can be highly profitable if it captures large wins and
  cuts small losses

Reinforcement Learning directly optimises the objective we actually care about:
cumulative risk-adjusted return. The RL agent learns a **trading policy** — not just
which direction to predict, but when to enter, how much to trade, and when to exit —
all in service of maximising long-term PnL.

---

#### How It Works: The RL Trading Setup

```
┌────────────────────────────────────────────────────────────────┐
│                     RL Training Loop                           │
│                                                                │
│   Market Environment (historical candles, replayed in order)   │
│          │                                      ↑              │
│          │  State (market sequence +            │              │
│          │  current position + account)         │ Reward       │
│          ↓                                      │ (PnL step)   │
│        Agent (Policy Network)                   │              │
│          │                                      │              │
│          │  Action (enter/exit/hold/size)  ──────┘              │
│          ↓                                                     │
│        Execute action in simulated market                      │
│        → observe new state → collect reward                    │
└────────────────────────────────────────────────────────────────┘
```

---

#### Components

**Agent**

The agent is a neural network that maps the current state to an action.
It learns by trial and error across thousands of simulated trading episodes.

**Environment**

A simulation of the Forex market using historical candles, replayed in time order.
The environment enforces realistic constraints:
- Spread deducted on every entry
- Slippage modelled on fills
- Maximum position size enforced
- No look-ahead — only past candles visible at each step

**State**

The state the agent observes at each step:

```
State = [
    market_language_sequence (64 candles, tokenized),
    supervised_model_probabilities [p_sell, p_hold, p_buy],
    current_position (-1=short, 0=flat, +1=long),
    unrealised_pnl (float),
    candles_held (int)
]
```

The supervised model's probability output is included as part of the state — the RL
agent learns *when to act on the supervised signal* and when to ignore it.

**Action Space**

Discrete actions (simpler to train, recommended for Phase 4):

| Action | Meaning |
|--------|---------|
| `FLAT` | Close any open position, stay flat |
| `LONG` | Open or maintain long position (full size) |
| `SHORT` | Open or maintain short position (full size) |
| `LONG_HALF` | Open or maintain long at half size |
| `SHORT_HALF` | Open or maintain short at half size |

**Reward Function**

The reward signal is what the agent optimises. This is the most important design choice.

Recommended reward (step-by-step):

```
reward = realised_pnl_this_step
       - spread_cost_if_traded_this_step
       - holding_penalty (small negative reward for being in a trade past hold_period)
```

Optional: use Sharpe-ratio-based reward to penalise volatility in returns, not just losses.

Do NOT use a reward based on classification accuracy — that defeats the purpose of RL.

---

#### Training Algorithm: PPO (Proximal Policy Optimisation)

PPO is the recommended starting algorithm for this use case:

| Property | Value |
|----------|-------|
| Algorithm | PPO (Proximal Policy Optimisation) |
| Why PPO | Stable training, well-understood, good default for discrete action spaces |
| Policy network | Small MLP over the state vector |
| Training data | Historical candles used as environment (same training split as supervised model) |
| Episodes | Each episode = one month of 15-min candles replayed in order |
| Convergence | Typically 500–2000 episodes |

Alternative: SAC (Soft Actor-Critic) if moving to continuous position sizing in Phase 5.

---

#### The Natural Bridge: Supervised → RL

The supervised model is not discarded when moving to RL. Instead it becomes the
foundation of the RL state, and can warm-start the RL policy:

```
Phase 1 (Supervised):
  Train transformer → produces calibrated [p_sell, p_hold, p_buy]
  Backtest confirms directional edge exists

Phase 4 (RL):
  Freeze the transformer (its weights are not updated)
  Add RL policy network on top
  RL agent learns WHEN and HOW MUCH to trade based on the transformer's output
  RL agent optimises cumulative PnL rather than classification accuracy
```

This means Phase 1 work is fully preserved. RL adds a trading policy layer on top
of the existing probabilistic model.

---

#### Key Challenges for RL in Trading

| Challenge | Description |
|-----------|------------|
| Non-stationarity | Market dynamics change; a policy trained on 2022 may fail in 2025 |
| Overfitting to history | Agent may memorise specific historical episodes |
| Reward sparsity | Profits only materialise at trade close — many steps with zero reward |
| Transaction costs | Frequent trading destroys edge — agent must learn patience |
| Evaluation | Standard RL metrics (cumulative reward) must be accompanied by financial metrics |

**Mitigation:** Use walk-forward RL training (train on expanding windows, test on next
unseen window) — the same approach used for the supervised model.

---

#### Expected Outcome of RL Phase

If successful, the RL system will:
- Improve on the supervised baseline's Sharpe ratio
- Naturally learn to be more patient (fewer, higher-quality trades)
- Adapt position sizing to current market conditions dynamically
- Avoid trading in periods the supervised model signals but market conditions are poor

---

### 10. Hybrid Model Architecture

Combine transformers with classical technical components:

```
Market Language Sequence
        |
        v
Transformer Encoder (pattern recognition)
        +
Classical Indicators computed in parallel
(RSI, MACD, Bollinger Band width)
        |
        v
Fusion Layer (concatenate + linear)
        |
        v
Probability Output
```

Classical indicators provide interpretable, well-tested signals. The transformer captures
non-linear temporal context. The fusion layer learns which to trust in which conditions.

---

### 11. Foundation Model for Markets

Long-term vision: train a large-scale market language model on:

- Multiple asset classes (Forex, Equities, Futures, Crypto)
- Multiple timeframes simultaneously
- Decades of historical data
- Market event text (news headlines, central bank statements)

This would serve as a general-purpose pretrained model for financial sequence prediction —
fine-tunable for any pair, timeframe, or prediction task.

---

### 12. Uncertainty Quantification

Model epistemic uncertainty (uncertainty about the model itself):

| Method | Description |
|--------|------------|
| Monte Carlo Dropout | Multiple stochastic forward passes, measure variance |
| Deep Ensembles | Variance across ensemble member predictions |
| Conformal Prediction | Statistically guaranteed prediction intervals |

High uncertainty → reduce position size or skip trade entirely.

---

## Improvement Priority Matrix

| Improvement | Impact | Complexity | Phase |
|-------------|--------|------------|-------|
| Walk-forward re-training | High | Medium | 2 |
| Multi-timeframe context | High | Medium | 2 |
| Richer token vocabulary | Medium | Low | 2 |
| Regime detection | High | High | 3 |
| Cross-pair information | Medium | Medium | 3 |
| Ensemble methods | Medium | Medium | 3 |
| Reinforcement learning | Very High | Very High | 4 |
| Hybrid architecture | Medium | Medium | 3 |
| Uncertainty quantification | High | Medium | 3 |
| Foundation model | Very High | Extreme | 5+ |
