"""
transformer.py — Phase 3, Step 1

Custom Transformer Encoder for market-language classification.

Architecture
------------
    Embedding (vocab_size → d_model)
        ↓
    Sinusoidal Positional Encoding (fixed, not learned)
        ↓
    Dropout
        ↓
    4 × TransformerEncoderLayer (pre-norm, 8 heads, d_ff=512)
        ↓
    LayerNorm
        ↓
    CLS token representation [position 0]
        ↓
    Dropout → Linear(d_model, 3)  →  logits [SELL, HOLD, BUY]

Design decisions
----------------
- Pre-norm (norm_first=True): more stable gradient flow, especially helpful
  on smaller datasets where gradients can be noisy
- Sinusoidal PE (fixed): deterministic position signal, no extra parameters,
  proven to generalise well without learned embeddings
- CLS token at position 0: aggregates global sequence representation for
  classification, matching BERT-style encoder usage
- Padding mask: attention ignores PAD tokens (ID=0) — positions 321-334
- No pretrained weights: our 25-token market vocabulary has zero overlap
  with natural language; pretrained embeddings provide no benefit

Parameter count (approximate)
------------------------------
    Embedding:    27 × 128         =    3,456
    4 × Encoder layer:
        Attention (Q,K,V,Out):    4 × 66,048  = 264,192
        FFN (128→512→128):        4 × 132,224 = 528,896
        LayerNorm × 2:            4 × 256     =   1,024
    Final LayerNorm:                             256
    Classifier:                  128 × 3 + 3 =   387
    Total:                                   ~798,211  (~0.8M params)

Usage
-----
    from src.model.transformer import MarketTransformer, load_model
    model = MarketTransformer.from_config(cfg)
    logits = model(input_ids)   # input_ids: (B, 335) int64
"""

import math
import yaml
from pathlib import Path

import torch
import torch.nn as nn


# ── Sinusoidal positional encoding ────────────────────────────────────────────

def _build_sinusoidal_pe(max_len: int, d_model: int) -> torch.Tensor:
    """
    Returns a fixed sinusoidal positional encoding tensor of shape (1, max_len, d_model).
    Registered as a buffer — saved with the model but not a trainable parameter.
    """
    pe       = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)   # (1, max_len, d_model)


# ── Model ──────────────────────────────────────────────────────────────────────

class MarketTransformer(nn.Module):
    """
    Transformer encoder that maps a market-language token sequence to
    3-class logits: [SELL (0), HOLD (1), BUY (2)].

    Input
    -----
    input_ids : LongTensor of shape (batch, seq_len)
                Token IDs in range [0, vocab_size).
                ID=0 is PAD and is masked out of attention.
                ID=1 is CLS at position 0 — its output is used for classification.

    Output
    ------
    logits : FloatTensor of shape (batch, 3)
             Raw (uncalibrated) class scores. Apply softmax for probabilities.
             Temperature scaling is applied separately at inference time.
    """

    def __init__(self,
                 vocab_size:  int,
                 d_model:     int,
                 num_heads:   int,
                 num_layers:  int,
                 d_ff:        int,
                 dropout:     float,
                 max_seq_len: int,
                 num_classes: int = 3):
        super().__init__()

        # Token embedding (padding_idx=0 → its embedding stays zero-initialised
        # and receives no gradient, consistent with the attention mask)
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)

        # Fixed sinusoidal positional encoding
        self.register_buffer('pe', _build_sinusoidal_pe(max_seq_len, d_model))

        self.input_dropout = nn.Dropout(dropout)

        # Transformer encoder — pre-norm for training stability
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = num_heads,
            dim_feedforward = d_ff,
            dropout         = dropout,
            activation      = 'gelu',
            batch_first     = True,
            norm_first      = True,   # pre-norm (more stable than post-norm)
        )
        encoder_norm = nn.LayerNorm(d_model)
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers = num_layers,
            norm       = encoder_norm,
        )

        # Classification head — applied to CLS token (position 0)
        self.cls_dropout  = nn.Dropout(dropout)
        self.classifier   = nn.Linear(d_model, num_classes)

        # Weight initialisation
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.embedding.weight[0])   # keep PAD embedding at zero
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        input_ids : (B, seq_len) — integer token IDs

        Returns
        -------
        logits : (B, 3)
        """
        # Padding mask: True → ignore that position in attention
        pad_mask = (input_ids == 0)                     # (B, seq_len)

        # Embed + positional encoding
        x = self.embedding(input_ids)                   # (B, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]              # broadcast PE
        x = self.input_dropout(x)                       # (B, seq_len, d_model)

        # Transformer encoder
        x = self.transformer(x, src_key_padding_mask=pad_mask)  # (B, seq_len, d_model)

        # CLS token (position 0) → classification
        cls = x[:, 0, :]                                # (B, d_model)
        logits = self.classifier(self.cls_dropout(cls)) # (B, 3)

        return logits

    @classmethod
    def from_config(cls, cfg: dict) -> "MarketTransformer":
        m = cfg["model"]
        return cls(
            vocab_size  = m["vocab_size"],
            d_model     = m["d_model"],
            num_heads   = m["num_heads"],
            num_layers  = m["num_layers"],
            d_ff        = m["d_ff"],
            dropout     = m["dropout"],
            max_seq_len = m["max_seq_len"],
        )

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def save_checkpoint(model: MarketTransformer, cfg: dict,
                    val_loss: float, epoch: int, path: str | Path) -> None:
    """Save model weights + config + metadata to a .pt file."""
    torch.save({
        "model_state":  model.state_dict(),
        "model_config": cfg["model"],
        "val_loss":     val_loss,
        "epoch":        epoch,
    }, path)


def load_checkpoint(path: str | Path,
                    cfg: dict,
                    device: torch.device) -> MarketTransformer:
    """Load a saved checkpoint and return the model in eval mode."""
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    model = MarketTransformer.from_config(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


# ── CLI: print model summary ──────────────────────────────────────────────────

def main():
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model = MarketTransformer.from_config(cfg)
    n_params = model.count_parameters()

    print(f"\n[transformer] MarketTransformer summary")
    print(f"  vocab_size  : {cfg['model']['vocab_size']}")
    print(f"  d_model     : {cfg['model']['d_model']}")
    print(f"  num_heads   : {cfg['model']['num_heads']}")
    print(f"  num_layers  : {cfg['model']['num_layers']}")
    print(f"  d_ff        : {cfg['model']['d_ff']}")
    print(f"  dropout     : {cfg['model']['dropout']}")
    print(f"  max_seq_len : {cfg['model']['max_seq_len']}")
    print(f"  Parameters  : {n_params:,}")
    print()

    # Test forward pass
    dummy = torch.randint(0, 25, (2, 335))
    dummy[:, 0]    = 1   # CLS
    dummy[:, 321:] = 0   # PAD
    logits = model(dummy)
    print(f"  Test forward — input {list(dummy.shape)} → logits {list(logits.shape)}  ✅")
    print()


if __name__ == "__main__":
    main()
