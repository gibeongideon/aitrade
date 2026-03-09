"""
trainer.py — Phase 3, Step 2

Trains the MarketTransformer on non-overlapping labeled sequences.

Training setup
--------------
    Loss      : Weighted CrossEntropyLoss (weights = N / (3 × count_per_class))
                Compensates for mild class imbalance (27% SELL / 46% HOLD / 27% BUY)
    Optimizer : AdamW (lr=1e-4, weight_decay=0.01)
    Schedule  : Linear warmup for warmup_steps, then cosine decay to 0
    Gradient  : Clip max norm = 1.0
    Stopping  : Early stop if val loss does not improve for patience epochs
    Checkpoint: Save best model (lowest val loss) → models/transformer_best.pt
    Log       : Epoch metrics → logs/training_log.csv

Usage
-----
    python -m src.model.trainer
    python -m src.model.trainer --config config/config.yaml
"""

import argparse
import csv
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from src.model.transformer import MarketTransformer, save_checkpoint


def _load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _lr_lambda(current_step: int, warmup_steps: int, total_steps: int) -> float:
    """Linear warmup then cosine decay."""
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


def _compute_class_weights(dataset) -> torch.Tensor:
    """
    Inverse-frequency class weights.
    weight[c] = total_samples / (num_classes × count[c])
    Higher weight → model penalised more for mistakes on that class.
    """
    labels = dataset.tensors[1]   # shape (N,)
    n      = len(labels)
    counts = torch.bincount(labels, minlength=3).float()
    weights = n / (3.0 * counts)
    return weights


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def _run_epoch(model, loader, criterion, optimizer, scheduler,
               device, grad_clip, train: bool) -> tuple[float, float]:
    """Run one full epoch. Returns (avg_loss, avg_accuracy)."""
    model.train(train)
    total_loss = 0.0
    total_acc  = 0.0
    n_batches  = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for input_ids, labels in loader:
            input_ids = input_ids.to(device)
            labels    = labels.to(device)

            logits = model(input_ids)
            loss   = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                scheduler.step()

            total_loss += loss.item()
            total_acc  += _accuracy(logits.detach(), labels)
            n_batches  += 1

    return total_loss / n_batches, total_acc / n_batches


def main():
    parser = argparse.ArgumentParser(description="Train MarketTransformer")
    parser.add_argument("--config",    default="config/config.yaml")
    parser.add_argument("--train-pt",  default="data/processed/train_sequences.pt")
    parser.add_argument("--val-pt",    default="data/processed/val_sequences.pt")
    parser.add_argument("--out-model", default="models/transformer_best.pt")
    parser.add_argument("--out-log",   default="logs/training_log.csv")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    t   = cfg["training"]
    m   = cfg["model"]

    # ── Device ───────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[trainer] Device : {device}")

    # ── Load datasets ─────────────────────────────────────────────────────────
    for p in [args.train_pt, args.val_pt]:
        if not Path(p).exists():
            print(f"[ERROR] Not found: {p}  — run sequences.py first.")
            raise SystemExit(1)

    print(f"[trainer] Loading {args.train_pt}")
    train_ds = torch.load(args.train_pt, map_location="cpu", weights_only=False)
    print(f"[trainer] Loading {args.val_pt}")
    val_ds   = torch.load(args.val_pt,   map_location="cpu", weights_only=False)

    n_train = len(train_ds)
    n_val   = len(val_ds)
    print(f"[trainer] Train : {n_train:,} sequences")
    print(f"[trainer] Val   : {n_val:,} sequences")

    # ── Data loaders ──────────────────────────────────────────────────────────
    batch_size = t["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=False)

    steps_per_epoch = math.ceil(n_train / batch_size)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = MarketTransformer.from_config(cfg).to(device)
    n_params = model.count_parameters()
    print(f"[trainer] Parameters: {n_params:,}")

    # ── Loss with class weights ───────────────────────────────────────────────
    class_weights = _compute_class_weights(train_ds).to(device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)
    print(f"[trainer] Class weights — SELL:{class_weights[0]:.3f}  "
          f"HOLD:{class_weights[1]:.3f}  BUY:{class_weights[2]:.3f}")

    # ── Optimizer + LR schedule ───────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = t["learning_rate"],
        weight_decay = t["weight_decay"],
        betas        = (0.9, 0.999),
    )

    warmup_steps = t["warmup_steps"]
    max_epochs   = t["max_epochs"]
    total_steps  = max_epochs * steps_per_epoch

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _lr_lambda(step, warmup_steps, total_steps)
    )

    patience    = t["early_stopping_patience"]
    grad_clip   = t["grad_clip"]

    print(f"[trainer] Steps/epoch  : {steps_per_epoch}")
    print(f"[trainer] Warmup steps : {warmup_steps}  (~{warmup_steps/steps_per_epoch:.1f} epochs)")
    print(f"[trainer] Max epochs   : {max_epochs}")
    print(f"[trainer] Early stop   : patience={patience}\n")

    # ── Output paths ──────────────────────────────────────────────────────────
    Path(args.out_model).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_log).parent.mkdir(parents=True, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss   = float("inf")
    patience_count  = 0
    training_start  = time.time()

    header = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "epoch_sec"]

    with open(args.out_log, "w", newline="") as log_f:
        writer = csv.DictWriter(log_f, fieldnames=header)
        writer.writeheader()

        print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>9}  "
              f"{'Val Loss':>8}  {'Val Acc':>7}  {'LR':>9}  {'Time':>6}  Note", flush=True)
        print("-" * 80, flush=True)

        for epoch in range(1, max_epochs + 1):
            t_start = time.time()

            train_loss, train_acc = _run_epoch(
                model, train_loader, criterion, optimizer, scheduler,
                device, grad_clip, train=True)

            val_loss, val_acc = _run_epoch(
                model, val_loader, criterion, optimizer, scheduler,
                device, grad_clip, train=False)

            epoch_sec = time.time() - t_start
            current_lr = scheduler.get_last_lr()[0]

            # ── Check improvement ─────────────────────────────────────────────
            note = ""
            if val_loss < best_val_loss:
                best_val_loss  = val_loss
                patience_count = 0
                save_checkpoint(model, cfg, val_loss, epoch, args.out_model)
                note = "✓ best"
            else:
                patience_count += 1
                note = f"({patience_count}/{patience})"

            # ── Log ───────────────────────────────────────────────────────────
            writer.writerow({
                "epoch":      epoch,
                "train_loss": f"{train_loss:.4f}",
                "train_acc":  f"{train_acc:.4f}",
                "val_loss":   f"{val_loss:.4f}",
                "val_acc":    f"{val_acc:.4f}",
                "lr":         f"{current_lr:.2e}",
                "epoch_sec":  f"{epoch_sec:.1f}",
            })
            log_f.flush()

            print(f"{epoch:>6}  {train_loss:>10.4f}  {train_acc:>8.1%}  "
                  f"{val_loss:>8.4f}  {val_acc:>6.1%}  {current_lr:>9.2e}  "
                  f"{epoch_sec:>5.0f}s  {note}", flush=True)

            if patience_count >= patience:
                print(f"\n[trainer] Early stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs)")
                break

    total_time = time.time() - training_start
    print(f"\n[trainer] Training complete — {total_time/60:.1f} min total")
    print(f"[trainer] Best val loss : {best_val_loss:.4f}")
    print(f"[trainer] Checkpoint    → {args.out_model}")
    print(f"[trainer] Training log  → {args.out_log}")
    print()
    print("[trainer] NEXT: Run calibrator.py to find temperature T on val set")


if __name__ == "__main__":
    main()
