# 03 — Model Architecture

## Decision: Custom Small Transformer (Trained from Scratch)

After considering all options, the chosen approach is to **train a small transformer encoder
from scratch** rather than fine-tuning a pretrained LLM.

### Why not a pretrained LLM (GPT-2, DistilBERT, etc.)?

The market token vocabulary is entirely custom (~27 tokens). A pretrained LLM trained on
English text has no domain knowledge that transfers to market tokens. Using one would mean:

- Expanding the tokenizer with new embeddings (random init anyway)
- Carrying 82M–3.8B parameters of dead weight irrelevant to this task
- Higher memory, longer training, more complexity, no clear benefit in Phase 1

The only real benefit of a pretrained transformer is its **attention architecture** — and we
get that by building a small transformer from scratch.

### Why custom small transformer?

| Property | Value |
|----------|-------|
| Vocabulary | ~27 tokens (fully known, no surprises) |
| Sequence length | ~330 tokens (controlled, short) |
| Training data | ~30,000–50,000 labeled sequences per pair |
| Task | 3-class classification (not generation) |
| Conclusion | A ~2-4M parameter encoder is exactly right for this problem |

A small custom encoder trains in **hours on CPU**, **minutes on GPU**, is fully transparent,
and avoids unnecessary dependencies.

> **Pretrained LLM with LoRA is reserved for Phase 3** when multi-pair, multi-timeframe
> data significantly increases training data volume and complexity.

---

## Model Specification

### Architecture: Transformer Encoder

The model is an encoder-only transformer (similar in structure to BERT, not GPT).
Encoder-only is correct here because we are doing classification over a full sequence,
not generating tokens.

| Parameter | Value | Notes |
|-----------|-------|-------|
| Architecture | Transformer Encoder | Encoder-only, not generative |
| Layers | 4 | Sufficient for ~330-token sequences |
| Attention heads | 8 | 8 heads × 16 dim each = d_model |
| d_model | 128 | Hidden dimension |
| d_ff (feedforward) | 512 | 4× d_model, standard ratio |
| Dropout | 0.1 | Applied in attention and feedforward |
| Max sequence length | 335 | 64 candles × 5 tokens + special tokens |
| Vocabulary size | 27 | 25 market tokens + [CLS] + [PAD] |
| Total parameters | ~2.5M | Trainable from scratch |

### Positional Encoding

Standard sinusoidal positional encoding. Because candles have a natural time order,
positional encoding is essential to preserve temporal relationships.

### Classification Head

```
[CLS] token hidden state (dim=128)
            |
            v
    LayerNorm(128)
            |
            v
    Linear(128 → 64)
            |
            v
        ReLU
            |
            v
    Dropout(0.1)
            |
            v
    Linear(64 → 3)
            |
            v
        Softmax
            |
            v
  [p_sell, p_hold, p_buy]
```

The `[CLS]` token's hidden state after all encoder layers is used as the sequence
representation for classification — the same pattern used in BERT.

---

## Input Format

Each input sequence represents 64 consecutive 15-minute candles:

```
[CLS] RET_U1 BODY_M WICK_TOP VOL_HIGH TREND_UP RET_U2 BODY_L WICK_NONE VOL_HIGH TREND_UP ... [PAD]
```

| Property | Value |
|----------|-------|
| Tokens per candle | 5 |
| Candles per sequence | 64 |
| Core token count | 320 |
| With [CLS] and padding | up to 335 |
| Token encoding | Integer IDs (0–26) |

---

## Training Configuration

### Dataset

| Property | Value |
|----------|-------|
| Primary pair | EUR/USD |
| Candle timeframe | 15 minutes |
| History | 2 years minimum (~34,800 candles) |
| Sequences after windowing | ~34,736 (candles minus context window) |
| Usable labeled sequences | ~25,000–30,000 after removing HOLD-heavy noise |
| Split | 70% train / 15% validation / 15% test (time-based, no shuffle) |

### Optimizer and Schedule

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 0.01 |
| LR scheduler | Cosine annealing with linear warmup (500 steps) |
| Warmup steps | 500 |
| Batch size | 64 |
| Epochs | Up to 100 (early stopping applies) |
| Early stopping | Patience = 10 epochs on validation loss |
| Gradient clipping | max_norm = 1.0 |

### Loss Function

Weighted cross-entropy to counteract class imbalance:

```
Loss = CrossEntropyLoss(predictions, labels, weight=[w_sell, w_hold, w_buy])
```

Class weights are computed from training label frequencies:

```
w_class = total_samples / (num_classes * count_of_class)
```

---

## Training Process — Step by Step

This is exactly how a training run works:

```
1. Load tokenized sequences and labels from disk (prepared offline)

2. Split into train / val / test by time index

3. Initialize model weights randomly (Xavier uniform for linear layers)

4. For each epoch:
   a. Shuffle training sequences (within train split only)
   b. For each batch of 64 sequences:
      - Forward pass: sequence → encoder → [CLS] state → classification head → logits
      - Compute weighted cross-entropy loss vs. true label
      - Backward pass: compute gradients
      - Clip gradients (max_norm=1.0)
      - AdamW step: update weights
   c. After epoch: run full validation set forward pass (no gradient)
   d. Record validation loss and accuracy
   e. If val loss improved: save model checkpoint
   f. If no improvement for 10 epochs: stop training

5. Load best checkpoint (lowest validation loss)

6. Run calibration on validation set (temperature scaling)

7. Evaluate on test set — report accuracy, F1, ECE, simulated PnL

8. Save: model weights + calibration temperature + tokenizer thresholds
```

**Typical training time:**

| Hardware | Expected Duration |
|----------|-----------------|
| Modern CPU (8-core) | 4–8 hours |
| Entry GPU (RTX 3060, 8GB) | 20–40 minutes |
| Google Colab T4 (free) | 30–60 minutes |

---

## Probability Calibration

After training, the raw softmax outputs are calibrated using **Temperature Scaling** —
the simplest effective method:

```
calibrated_prob = softmax(logits / T)
```

Where `T` is a single scalar learned by minimizing cross-entropy on the validation set
with model weights frozen.

- `T > 1` → softer probabilities (less confident)
- `T < 1` → sharper probabilities (more confident)
- `T = 1` → no change (uncalibrated)

Typical result: `T` ends up in the range 1.2–2.0 for neural classifiers.

The calibration temperature `T` is saved alongside model weights and applied at every
inference call.

---

## Model Upgrade Path (Phase 3)

Once the custom transformer baseline is validated, the upgrade path is:

```
Phase 1: Custom small transformer (~2.5M params) — trained from scratch
Phase 2: Larger custom transformer (~10M params) or add multi-timeframe input
Phase 3: DistilBERT or similar pretrained encoder + LoRA, with larger dataset
         (multiple pairs, 5+ years, multi-timeframe)
```

Upgrading is low-risk because the data pipeline, tokenizer, and trading engine
remain identical — only the model module changes.
