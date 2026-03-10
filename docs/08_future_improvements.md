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
| Chronos/TimesFM fine-tuning | High | Low | 2 |
| Expand data (pairs + history) | Very High | Low | 2 |
| Asymmetric / trend-filtered labels | High | Medium | 2 |
| Meta-labeling | High | High | 3 |
| Model ensemble (5 seeds) | Medium | Low | 2 |

---

## Practical Improvements — Lessons from Phase 3/4 Build

These are grounded in observed results from training and backtesting the first
working model. Added as a reference for the next training iteration.

---

### 13. Expand Data Before Increasing Model Size

The first model (4 years, 100K candles, 1 pair) achieved 51.6% test accuracy —
marginally above the HOLD-always baseline. Before increasing model complexity,
expanding the data is the higher-leverage action.

| Data Expansion | Expected Impact |
| -------------- | --------------- |
| Add GBPUSD, USDJPY, AUDUSD | 3–4x more samples, cross-market generalisation |
| Extend to 10+ years | Covers 2008 crisis, 2020 COVID, 2022 inflation — multiple regimes |
| Add H1 candles as context | Trend direction filter significantly reduces counter-trend errors |

**Rule:** Double the data before doubling the model parameters.

---

### 14. Multi-Timeframe Input Architecture (Priority 2)

Feeding only M15 data means the model cannot distinguish a BUY signal in a
strong downtrend from a BUY signal in a strong uptrend. H1 trend context is
the single most impactful architecture change.

```text
H1 sequence  (last 16 candles = 16 hours):
[CLS_H1] + 16 candles x 5 tokens = 81 tokens

M15 sequence (last 64 candles = 16 hours):
[CLS_M15] + 64 candles x 5 tokens = 321 tokens

Fusion: concatenate [CLS_H1, CLS_M15] embeddings → classification head
```

Existing tokenizer, labeler, and sequences pipeline requires no changes —
only the model input layer and a new H1 data loader need modification.

---

### 15. Trend-Filtered Labels

The current labeler fires events in all market conditions. A large fraction of
HOLD labels are counter-trend trades that always lose. Filtering by H1 trend
before labeling would improve the signal quality without changing the model.

```text
Only label a BUY event at M15 candle i if:
    MA_16[H1] > MA_64[H1] at time i   (H1 trend is bullish)

Only label a SELL event at M15 candle i if:
    MA_16[H1] < MA_64[H1] at time i   (H1 trend is bearish)

Otherwise → label = -1 (skip as non-event, used for context only)
```

This reduces total events but significantly improves their directional reliability.

---

### 16. Model Ensemble — 5 Seeds (Easy Win)

The cheapest accuracy improvement per hour of effort. Train 5 models with
different random seeds on the same data, average their probability outputs.

```python
probs = mean([model_1(x), model_2(x), model_3(x), model_4(x), model_5(x)])
```

Typical gain: +2–4% accuracy, meaningfully fewer catastrophic errors, and
more reliable high-confidence predictions. All 5 models share the same
calibration temperature T (refit on ensemble output).

Cost: 5× Colab training time (~5 hours on T4 for 60 epochs × 5 runs).

---

### 17. Chronos / TimesFM — Alternative to Custom Tokenization

If the custom transformer approach plateaus after proper training, these
foundation models offer a pretrained alternative that skips tokenization.

| Model | Source | Approach |
| ----- | ------ | -------- |
| **Chronos** | Amazon (2024, open source) | Trained on 100K+ real time series, probabilistic forecasting |
| **TimesFM** | Google (2024) | Similar, patch-based numerical forecasting |

**How to adapt to classification:**

```text
1. Feed last 64 close prices (normalised) to Chronos
2. Get forecast distribution for next 3 steps
3. Compare forecast return to +/- ATR barrier thresholds
4. Map: forecast > upper barrier → BUY, forecast < lower → SELL, else → HOLD
```

**Limitation:** These models forecast numerical values — they do not natively
consume volume, wick structure, or MA trend tokens. They work on close prices
only unless you engineer additional channels.

**Recommendation:** Try Chronos only if the custom model fails to reach positive
backtest expectancy after full training with expanded data.

---

### 18. Walk-Forward Validation Before Live Trading (Non-Negotiable)

A single test-set backtest confirms performance on one 7-month period
(Jul 2025 – Mar 2026). This is insufficient to confirm a genuine edge.

Walk-forward across 6+ windows is required before risking capital:

```text
Window 1: train 2021-2022    | test Q1 2023
Window 2: train 2021-Q1 2023 | test Q2 2023
Window 3: train 2021-Q2 2023 | test Q3 2023
Window 4: train 2021-Q3 2023 | test Q4 2023
Window 5: train 2021-Q4 2023 | test Q1 2024
Window 6: train 2021-Q1 2024 | test Q2 2024
```

If positive expectancy appears in 5+ of 6 windows, the edge is likely real.
If it appears in 2 of 6, it is noise.

---

### 19. SELL Class Fix — Class Weight Tuning

The first trained model showed near-zero SELL recall (1.5%) with 68.4% BUY recall.
This means the model collapsed to predicting HOLD and BUY only. Root cause: the
CrossEntropyLoss class weights were insufficient to overcome the BUY bias in the
gradient signal.

**Fix options (in order of effort):**

| Option | How | Expected outcome |
| ------ | --- | ---------------- |
| Increase SELL weight | Set `class_weights[0] = 2.0` in trainer.py | Forces model to attend to SELL examples |
| Oversample SELL events | Duplicate SELL sequences in train_sequences.pt | Balances gradient signal without weight tuning |
| Focal loss | Replace CrossEntropyLoss with focal loss (gamma=2) | Down-weights easy HOLD examples, forces learning of hard SELL/BUY |
| Binary approach | Train two separate models: SELL-vs-not, BUY-vs-not | Sidesteps 3-class imbalance entirely |

The simplest fix: in `trainer.py`, manually override the computed class weights
before training to give SELL equal or higher weight than BUY.
