"""
evaluate.py — Phase 4, Step 1

Final evaluation of the trained + calibrated model on the held-out test set.
The test set is NEVER touched during training or calibration — this is the
one honest measurement of how the model generalises to completely unseen data.

Metrics reported
----------------
    Accuracy          — overall % correct
    Per-class F1      — F1 score for SELL / HOLD / BUY separately
    Macro F1          — unweighted average F1 across 3 classes
    Confusion matrix  — rows=actual, cols=predicted
    ECE               — Expected Calibration Error (confidence vs accuracy)
    High-confidence   — accuracy when model confidence >= 60% threshold

Output
------
    logs/test_evaluation.txt   — full human-readable report
    logs/test_evaluation.json  — machine-readable summary for downstream use

Usage
-----
    python -m src.model.evaluate
    python -m src.model.evaluate --config config/config.yaml
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml


# ── Label names ───────────────────────────────────────────────────────────────
LABEL_NAMES = {0: "SELL", 1: "HOLD", 2: "BUY"}
N_CLASSES   = 3


def _load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Inference helpers ─────────────────────────────────────────────────────────

def _collect_logits(model, dataset, device, batch_size=256):
    """Run model on full dataset, return (logits, labels) as CPU tensors."""
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=0)
    all_logits, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for input_ids, labels in loader:
            logits = model(input_ids.to(device)).cpu()
            all_logits.append(logits)
            all_labels.append(labels)
    return torch.cat(all_logits), torch.cat(all_labels)


# ── Metric helpers ────────────────────────────────────────────────────────────

def _accuracy(preds, labels):
    return (preds == labels).float().mean().item()


def _per_class_f1(preds, labels):
    """
    Compute precision, recall, F1 for each class.
    Returns a dict: {class_id: {"precision": p, "recall": r, "f1": f1, "support": n}}
    """
    results = {}
    for c in range(N_CLASSES):
        tp = ((preds == c) & (labels == c)).sum().item()
        fp = ((preds == c) & (labels != c)).sum().item()
        fn = ((preds != c) & (labels == c)).sum().item()
        support = (labels == c).sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        results[c] = {
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
            "f1":        round(f1,        4),
            "support":   support,
        }
    return results


def _confusion_matrix(preds, labels):
    """Returns a 3×3 confusion matrix as nested list [actual][predicted]."""
    cm = [[0] * N_CLASSES for _ in range(N_CLASSES)]
    for actual, pred in zip(labels.tolist(), preds.tolist()):
        cm[actual][pred] += 1
    return cm


def _ece(probs, labels, n_bins=15):
    """Expected Calibration Error across confidence bins."""
    confidences, preds = probs.max(dim=1)
    correct = preds.eq(labels)
    ece = 0.0
    n   = len(labels)
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_conf = confidences[mask].mean().item()
        bin_acc  = correct[mask].float().mean().item()
        ece += (mask.sum().item() / n) * abs(bin_acc - bin_conf)
    return ece


def _high_confidence_stats(probs, labels, threshold=0.60):
    """Accuracy and coverage when model confidence >= threshold."""
    confidences, preds = probs.max(dim=1)
    mask    = confidences >= threshold
    covered = mask.sum().item()
    total   = len(labels)
    if covered == 0:
        return 0.0, 0.0, {}
    coverage = covered / total
    acc      = (preds[mask] == labels[mask]).float().mean().item()

    # Per-class breakdown among high-conf predictions
    hc_preds  = preds[mask]
    hc_labels = labels[mask]
    per_class = {}
    for c in range(N_CLASSES):
        c_mask = hc_labels == c
        if c_mask.sum() == 0:
            per_class[c] = {"acc": 0.0, "n": 0}
        else:
            per_class[c] = {
                "acc": (hc_preds[c_mask] == c).float().mean().item(),
                "n":   c_mask.sum().item(),
            }
    return acc, coverage, per_class


# ── Report formatting ─────────────────────────────────────────────────────────

def _format_report(acc, f1_dict, cm, ece_val, hc_acc, hc_cov, hc_per_class,
                   threshold, n_test, T):
    macro_f1 = sum(v["f1"] for v in f1_dict.values()) / N_CLASSES
    lines = []

    lines.append("=" * 60)
    lines.append("  AITrade — Test Set Evaluation Report")
    lines.append("=" * 60)
    lines.append(f"  Test sequences   : {n_test:,}")
    lines.append(f"  Temperature (T)  : {T:.4f}")
    lines.append("")

    # ── Overall accuracy ──────────────────────────────────────────────────────
    lines.append("── Overall Accuracy ─────────────────────────────────────")
    lines.append(f"  {acc:.1%}  (random baseline: 33.3%,  HOLD-always: ~46.6%)")
    lines.append("")

    # ── Per-class metrics ─────────────────────────────────────────────────────
    lines.append("── Per-Class Metrics ────────────────────────────────────")
    lines.append(f"  {'Class':<6}  {'Precision':>9}  {'Recall':>7}  {'F1':>6}  {'Support':>8}")
    lines.append(f"  {'-'*46}")
    for c, name in LABEL_NAMES.items():
        m = f1_dict[c]
        lines.append(f"  {name:<6}  {m['precision']:>9.1%}  {m['recall']:>7.1%}"
                     f"  {m['f1']:>6.4f}  {m['support']:>8,}")
    lines.append(f"  {'Macro':<6}  {'':>9}  {'':>7}  {macro_f1:>6.4f}")
    lines.append("")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    lines.append("── Confusion Matrix (rows=Actual, cols=Predicted) ───────")
    lines.append(f"           {'SELL':>7}  {'HOLD':>7}  {'BUY':>7}")
    for actual, row in enumerate(cm):
        name = LABEL_NAMES[actual]
        lines.append(f"  {name:<6}  {row[0]:>7,}  {row[1]:>7,}  {row[2]:>7,}")
    lines.append("")

    # ── Calibration ───────────────────────────────────────────────────────────
    lines.append("── Probability Calibration ──────────────────────────────")
    ece_status = "GOOD (< 0.10)" if ece_val < 0.10 else "WARN (> 0.10)"
    lines.append(f"  ECE  : {ece_val:.4f}  ← {ece_status}")
    lines.append("")

    # ── High-confidence filter ─────────────────────────────────────────────────
    lines.append(f"── High-Confidence Predictions (threshold ≥ {threshold:.0%}) ─────")
    lines.append(f"  Coverage : {hc_cov:.1%} of test samples pass the threshold")
    lines.append(f"  Accuracy : {hc_acc:.1%}  (on high-confidence samples only)")
    lines.append("")
    lines.append(f"  {'Class':<6}  {'Acc (HC)':>9}  {'N (HC)':>8}")
    lines.append(f"  {'-'*28}")
    for c, name in LABEL_NAMES.items():
        info = hc_per_class.get(c, {"acc": 0.0, "n": 0})
        lines.append(f"  {name:<6}  {info['acc']:>9.1%}  {info['n']:>8,}")
    lines.append("")

    # ── Interpretation ────────────────────────────────────────────────────────
    lines.append("── Interpretation ───────────────────────────────────────")
    if acc >= 0.55:
        lines.append("  Overall accuracy > 55% — model is learning real patterns ✓")
    elif acc >= 0.466:
        lines.append("  Accuracy above HOLD-always baseline — marginal signal present")
    else:
        lines.append("  Accuracy below HOLD baseline — model may be collapsed to one class")

    if macro_f1 >= 0.50:
        lines.append("  Macro F1 >= 0.50 — decent multi-class performance ✓")
    elif macro_f1 >= 0.40:
        lines.append("  Macro F1 0.40–0.50 — model is useful but has room to improve")
    else:
        lines.append("  Macro F1 < 0.40 — model struggles to distinguish all 3 classes")

    if hc_acc >= 0.60 and hc_cov >= 0.10:
        lines.append(f"  High-conf accuracy {hc_acc:.1%} at {hc_cov:.1%} coverage"
                     f" — usable trading signal ✓")
    elif hc_cov < 0.05:
        lines.append("  Very low coverage at 60% threshold — model is underconfident")

    lines.append("")
    lines.append("=" * 60)
    lines.append("  NEXT: Run backtester.py to simulate trading performance")
    lines.append("=" * 60)

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model on test set")
    parser.add_argument("--config",      default="config/config.yaml")
    parser.add_argument("--checkpoint",  default="models/transformer_best.pt")
    parser.add_argument("--calibration", default="models/calibration.json")
    parser.add_argument("--test-pt",     default="data/processed/test_sequences.pt")
    parser.add_argument("--out-txt",     default="logs/test_evaluation.txt")
    parser.add_argument("--out-json",    default="logs/test_evaluation.json")
    args = parser.parse_args()

    cfg = _load_config(args.config)

    # ── Dependency check ──────────────────────────────────────────────────────
    for p in [args.checkpoint, args.calibration, args.test_pt]:
        if not Path(p).exists():
            print(f"[ERROR] Not found: {p}")
            raise SystemExit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[evaluate] Device : {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    from src.model.transformer import load_checkpoint
    print(f"[evaluate] Loading checkpoint  : {args.checkpoint}")
    model = load_checkpoint(args.checkpoint, cfg, device)

    # ── Load temperature ──────────────────────────────────────────────────────
    with open(args.calibration) as f:
        cal = json.load(f)
    T = cal["temperature"]
    print(f"[evaluate] Temperature (T)     : {T:.4f}")

    # ── Load test set ─────────────────────────────────────────────────────────
    print(f"[evaluate] Loading test set    : {args.test_pt}")
    test_ds = torch.load(args.test_pt, map_location="cpu", weights_only=False)
    n_test  = len(test_ds)
    print(f"[evaluate] Test sequences      : {n_test:,}")

    # ── Forward pass ──────────────────────────────────────────────────────────
    print("[evaluate] Running forward pass...")
    logits, labels = _collect_logits(model, test_ds, device)

    # ── Apply temperature and compute probabilities ───────────────────────────
    probs = F.softmax(logits / T, dim=1)
    preds = probs.argmax(dim=1)

    # ── Compute all metrics ───────────────────────────────────────────────────
    threshold = cfg["trading"]["entry_threshold"]   # 0.60

    acc            = _accuracy(preds, labels)
    f1_dict        = _per_class_f1(preds, labels)
    cm             = _confusion_matrix(preds, labels)
    ece_val        = _ece(probs, labels)
    hc_acc, hc_cov, hc_per_class = _high_confidence_stats(probs, labels, threshold)

    # ── Print report ──────────────────────────────────────────────────────────
    report = _format_report(acc, f1_dict, cm, ece_val, hc_acc, hc_cov,
                            hc_per_class, threshold, n_test, T)
    print()
    print(report)

    # ── Save text report ──────────────────────────────────────────────────────
    Path(args.out_txt).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_txt, "w") as f:
        f.write(report + "\n")
    print(f"[evaluate] Report saved → {args.out_txt}")

    # ── Save JSON summary ─────────────────────────────────────────────────────
    macro_f1 = sum(v["f1"] for v in f1_dict.values()) / N_CLASSES
    summary  = {
        "n_test":        n_test,
        "temperature":   T,
        "accuracy":      round(acc,     4),
        "macro_f1":      round(macro_f1, 4),
        "ece":           round(ece_val, 4),
        "hc_accuracy":   round(hc_acc,  4),
        "hc_coverage":   round(hc_cov,  4),
        "hc_threshold":  threshold,
        "per_class":     {LABEL_NAMES[c]: v for c, v in f1_dict.items()},
        "confusion_matrix": cm,
    }
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[evaluate] JSON saved  → {args.out_json}")


if __name__ == "__main__":
    main()
