"""
calibrator.py — Phase 3, Step 3

Probability calibration via temperature scaling.

Why calibrate?
--------------
A trained neural network is often overconfident — it outputs p=0.92 for BUY
but the actual win rate at that confidence level is only 0.60.

Temperature scaling fixes this with a single scalar T:
    calibrated_prob = softmax(logits / T)

    T > 1  →  softer (less confident) probabilities
    T = 1  →  no change (raw model output)
    T < 1  →  sharper (more confident) probabilities

T is found by minimising NLL loss on the validation set — it cannot see
the test set, which is reserved for final evaluation.

Output
------
    models/calibration.json   {"temperature": T, "val_nll_before": x, "val_nll_after": y}

At inference, apply:
    probs = softmax(logits / T)

Usage
-----
    python -m src.model.calibrator
    python -m src.model.calibrator --config config/config.yaml
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from scipy.optimize import minimize_scalar
import yaml

from src.model.transformer import load_checkpoint


def _load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _get_val_logits(model: torch.nn.Module,
                    val_dataset,
                    device: torch.device,
                    batch_size: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run the model on the full val set and collect raw logits + labels.
    Uses large batches (no gradient) for speed.
    """
    from torch.utils.data import DataLoader
    loader = DataLoader(val_dataset, batch_size=batch_size,
                        shuffle=False, num_workers=0)

    all_logits = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for input_ids, labels in loader:
            input_ids = input_ids.to(device)
            logits    = model(input_ids).cpu()
            all_logits.append(logits)
            all_labels.append(labels)

    return torch.cat(all_logits), torch.cat(all_labels)


def _nll(T: float, logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Negative log-likelihood loss at temperature T."""
    scaled = logits / T
    return F.cross_entropy(scaled, labels).item()


def _expected_calibration_error(logits: torch.Tensor,
                                 labels: torch.Tensor,
                                 T: float,
                                 n_bins: int = 15) -> float:
    """
    Expected Calibration Error (ECE): measures how well predicted confidence
    matches actual accuracy across confidence bins.
    ECE < 0.10 is generally considered acceptable.
    """
    probs      = F.softmax(logits / T, dim=1)
    confidences, preds = probs.max(dim=1)
    correct    = preds.eq(labels)

    ece = 0.0
    n   = len(labels)

    for i in range(n_bins):
        lo   = i / n_bins
        hi   = (i + 1) / n_bins
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_conf = confidences[mask].mean().item()
        bin_acc  = correct[mask].float().mean().item()
        bin_size = mask.sum().item()
        ece     += (bin_size / n) * abs(bin_acc - bin_conf)

    return ece


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == labels).float().mean().item()


def main():
    parser = argparse.ArgumentParser(description="Temperature calibration")
    parser.add_argument("--config",       default="config/config.yaml")
    parser.add_argument("--checkpoint",   default="models/transformer_best.pt")
    parser.add_argument("--val-pt",       default="data/processed/val_sequences.pt")
    parser.add_argument("--out",          default="models/calibration.json")
    args = parser.parse_args()

    cfg = _load_config(args.config)

    # ── Checks ───────────────────────────────────────────────────────────────
    for p in [args.checkpoint, args.val_pt]:
        if not Path(p).exists():
            print(f"[ERROR] Not found: {p}")
            raise SystemExit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[calibrator] Device : {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"[calibrator] Loading checkpoint: {args.checkpoint}")
    model = load_checkpoint(args.checkpoint, cfg, device)

    # ── Load val set ──────────────────────────────────────────────────────────
    print(f"[calibrator] Loading val set   : {args.val_pt}")
    val_ds = torch.load(args.val_pt, map_location="cpu", weights_only=False)
    print(f"[calibrator] Val sequences     : {len(val_ds):,}")

    # ── Collect logits ────────────────────────────────────────────────────────
    print("[calibrator] Running forward pass on val set...")
    logits, labels = _get_val_logits(model, val_ds, device)

    # ── Pre-calibration stats ─────────────────────────────────────────────────
    nll_before = _nll(1.0, logits, labels)
    ece_before = _expected_calibration_error(logits, labels, T=1.0)
    acc        = _accuracy(logits, labels)

    print(f"\n[calibrator] Before calibration (T=1.0):")
    print(f"  Val accuracy : {acc:.1%}")
    print(f"  Val NLL      : {nll_before:.4f}")
    print(f"  Val ECE      : {ece_before:.4f}")

    # ── Find optimal T ────────────────────────────────────────────────────────
    print("\n[calibrator] Optimising temperature T on val set...")
    result = minimize_scalar(
        _nll,
        bounds  = (0.1, 10.0),
        method  = "bounded",
        args    = (logits, labels),
        options = {"xatol": 1e-5, "maxiter": 500},
    )
    T = float(result.x)

    # ── Post-calibration stats ────────────────────────────────────────────────
    nll_after = _nll(T, logits, labels)
    ece_after = _expected_calibration_error(logits, labels, T=T)

    print(f"\n[calibrator] After calibration (T={T:.4f}):")
    print(f"  Val NLL  : {nll_before:.4f} → {nll_after:.4f}")
    print(f"  Val ECE  : {ece_before:.4f} → {ece_after:.4f}", end="")
    if ece_after < 0.10:
        print("  ← GOOD (< 0.10)")
    else:
        print("  ← WARN (> 0.10, recalibration may be needed)")

    print(f"\n[calibrator] Temperature T = {T:.4f}")
    if T > 1.5:
        print("  → T > 1.5: model was overconfident — calibration significantly softens probabilities")
    elif T < 0.7:
        print("  → T < 0.7: model was underconfident — unusual, check training")
    else:
        print("  → T near 1.0: model was reasonably calibrated already")

    # ── Save ──────────────────────────────────────────────────────────────────
    result_dict = {
        "temperature":    round(T, 6),
        "val_nll_before": round(nll_before, 6),
        "val_nll_after":  round(nll_after, 6),
        "val_ece_before": round(ece_before, 6),
        "val_ece_after":  round(ece_after, 6),
        "val_accuracy":   round(acc, 6),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result_dict, f, indent=2)

    print(f"\n[calibrator] Saved → {args.out}")
    print()
    print("[calibrator] At inference, apply:  probs = softmax(logits / T)")
    print("[calibrator] NEXT: Run evaluate.py on the test set")


if __name__ == "__main__":
    main()
