"""
backtester.py — Phase 4, Step 2

Simulates trading on the held-out test set using the trained + calibrated model.

Design rules (from docs/05_trading_engine.md)
---------------------------------------------
Rule 1 — Test set only: backtester never touches train or val data.
Rule 2 — Entry at next candle OPEN (not signal candle close): the signal is
          generated after candle T closes, so the earliest entry is open[T+1].
Rule 3 — Stop and target from ATR at signal time only (past candles only).
Rule 4 — Check stop FIRST within each candle (conservative / worst-case).
Rule 5 — Spread deducted on every trade entry.

Trade lifecycle
---------------
  Signal candle T closes  →  model inference  →  p_buy / p_sell > threshold?
    YES  →  entry at open[T+1] ± spread
            stop   = entry ± ATR_14[T] × atr_stop_mult
            target = entry ∓ ATR_14[T] × atr_target_mult
            step forward candle by candle up to hold_period:
              check stop first → LOSS at stop_price
              then check target → WIN at target_price
            if neither hit → TIME EXIT at close of last candle

PnL is tracked in pips (1 pip = 0.0001 for EURUSD).

Output
------
    logs/backtest_report.txt   — full human-readable report
    logs/backtest_trades.csv   — one row per trade (for plotting / analysis)
    logs/backtest_summary.json — machine-readable summary

Usage
-----
    python -m src.trading.backtester
    python -m src.trading.backtester --threshold 0.65
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml


# ── Constants ─────────────────────────────────────────────────────────────────
TOKEN_COLS        = ["ret_token", "body_token", "wick_token", "vol_token", "trend_token"]
TOKENS_PER_CANDLE = 5
CLS_ID            = 1
PAD_ID            = 0
PIP_SIZE          = 0.0001   # EURUSD pip value


def _load_config(path="config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Sequence builder (mirrors sequences.py logic) ─────────────────────────────

def _build_sequence(token_array: np.ndarray, i: int,
                    context: int, max_seq_len: int) -> torch.Tensor:
    """Build a single 335-token sequence for event at row i."""
    window = token_array[i - context + 1: i + 1]   # (64, 5)
    flat   = window.ravel()                          # (320,)
    seq    = np.zeros(max_seq_len, dtype=np.int32)
    seq[0] = CLS_ID
    seq[1: 1 + context * TOKENS_PER_CANDLE] = flat
    return torch.tensor(seq, dtype=torch.long).unsqueeze(0)  # (1, 335)


# ── Trade simulator ────────────────────────────────────────────────────────────

def _simulate_trade(direction: str, entry_price: float,
                    stop: float, target: float,
                    ohlc: pd.DataFrame, start_j: int,
                    hold_period: int) -> tuple[str, float, int]:
    """
    Step forward candle by candle from start_j, checking exits.

    Parameters
    ----------
    direction   : "BUY" or "SELL"
    entry_price : price including spread
    stop        : stop-loss level
    target      : take-profit level
    ohlc        : DataFrame with columns high, low, close
    start_j     : first candle to check (= entry candle index)
    hold_period : max candles to hold

    Returns
    -------
    (result, exit_price, candles_held)
    result in {"WIN", "LOSS", "TIME"}
    """
    n = len(ohlc)

    for step in range(hold_period):
        j = start_j + step
        if j >= n:
            break

        h = ohlc["high"].iloc[j]
        l = ohlc["low"].iloc[j]

        if direction == "BUY":
            if l <= stop:                          # stop hit first (conservative)
                return "LOSS", stop, step + 1
            if h >= target:
                return "WIN", target, step + 1
        else:  # SELL
            if h >= stop:
                return "LOSS", stop, step + 1
            if l <= target:
                return "WIN", target, step + 1

    # Time barrier — exit at close of last candle checked
    last_j = min(start_j + hold_period - 1, n - 1)
    return "TIME", ohlc["close"].iloc[last_j], hold_period


# ── Metrics ────────────────────────────────────────────────────────────────────

def _pnl_pips(direction: str, entry: float, exit_price: float) -> float:
    if direction == "BUY":
        return (exit_price - entry) / PIP_SIZE
    return (entry - exit_price) / PIP_SIZE


def _max_drawdown(equity_curve: list[float]) -> float:
    """Max peak-to-trough decline in pips."""
    peak = equity_curve[0]
    max_dd = 0.0
    for v in equity_curve:
        if v > peak:
            peak = v
        dd = peak - v
        if dd > max_dd:
            max_dd = dd
    return max_dd


def _profit_factor(pnls: list[float]) -> float:
    gross_win  = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    return gross_win / gross_loss if gross_loss > 0 else float("inf")


def _sharpe(pnls: list[float]) -> float:
    """Per-trade Sharpe (mean/std). Multiply by sqrt(n_annual) for annualised."""
    if len(pnls) < 2:
        return 0.0
    arr = np.array(pnls)
    std = arr.std()
    return (arr.mean() / std) if std > 0 else 0.0


def _calibration_check(trades: list[dict], n_bins: int = 5) -> list[dict]:
    """
    Check if stated confidence matches actual win rate across confidence bins.
    Returns list of bin dicts: {lo, hi, n, win_rate, avg_conf}.
    """
    bins = []
    for b in range(n_bins):
        lo = 0.60 + b * 0.08
        hi = lo + 0.08
        bucket = [t for t in trades if lo <= t["confidence"] < hi]
        if not bucket:
            continue
        win_rate = sum(1 for t in bucket if t["result"] == "WIN") / len(bucket)
        avg_conf = sum(t["confidence"] for t in bucket) / len(bucket)
        bins.append({"lo": lo, "hi": hi, "n": len(bucket),
                     "win_rate": win_rate, "avg_conf": avg_conf})
    return bins


# ── Report ─────────────────────────────────────────────────────────────────────

def _format_report(trades: list[dict], all_signals: int,
                   threshold: float, cfg: dict) -> str:
    lines = []
    lines.append("=" * 62)
    lines.append("  AITrade — Backtest Report")
    lines.append("=" * 62)
    lines.append(f"  Entry threshold : {threshold:.0%}")
    lines.append(f"  Stop mult       : {cfg['trading']['atr_stop_mult']} x ATR")
    lines.append(f"  Target mult     : {cfg['trading']['atr_target_mult']} x ATR")
    lines.append(f"  Spread          : {cfg['trading']['spread_pips']} pip")
    lines.append(f"  Hold period     : {cfg['labeling']['horizon']} candles")
    lines.append("")

    if not trades:
        lines.append("  No trades generated at this threshold.")
        return "\n".join(lines)

    pnls       = [t["pnl_pips"] for t in trades]
    n          = len(trades)
    wins       = sum(1 for t in trades if t["result"] == "WIN")
    losses     = sum(1 for t in trades if t["result"] == "LOSS")
    time_exits = sum(1 for t in trades if t["result"] == "TIME")
    total_pnl  = sum(pnls)
    equity     = [0.0]
    for p in pnls:
        equity.append(equity[-1] + p)

    buy_trades  = [t for t in trades if t["direction"] == "BUY"]
    sell_trades = [t for t in trades if t["direction"] == "SELL"]

    # ── Summary ───────────────────────────────────────────────────────────────
    lines.append("── Signal Summary ───────────────────────────────────────")
    lines.append(f"  Total signals checked  : {all_signals:,}")
    lines.append(f"  Signals above threshold: {n:,}  "
                 f"({100*n/all_signals:.1f}% coverage)")
    lines.append(f"  BUY trades  : {len(buy_trades):,}")
    lines.append(f"  SELL trades : {len(sell_trades):,}")
    lines.append("")

    # ── Trade Outcomes ────────────────────────────────────────────────────────
    lines.append("── Trade Outcomes ───────────────────────────────────────")
    lines.append(f"  WIN      : {wins:,}  ({100*wins/n:.1f}%)")
    lines.append(f"  LOSS     : {losses:,}  ({100*losses/n:.1f}%)")
    lines.append(f"  TIME EXIT: {time_exits:,}  ({100*time_exits/n:.1f}%)")
    lines.append("")

    # ── PnL Metrics ───────────────────────────────────────────────────────────
    lines.append("── PnL Metrics (in pips) ────────────────────────────────")
    lines.append(f"  Total PnL      : {total_pnl:+.1f} pips")
    lines.append(f"  Avg per trade  : {total_pnl/n:+.2f} pips  (expectancy)")
    lines.append(f"  Avg WIN        : {np.mean([t['pnl_pips'] for t in trades if t['result']=='WIN']):+.1f} pips" if wins else "  Avg WIN        : n/a")
    lines.append(f"  Avg LOSS       : {np.mean([t['pnl_pips'] for t in trades if t['result']=='LOSS']):+.1f} pips" if losses else "  Avg LOSS       : n/a")
    lines.append(f"  Profit factor  : {_profit_factor(pnls):.2f}  (target > 1.20)")
    lines.append(f"  Per-trade Sharpe: {_sharpe(pnls):.3f}")
    lines.append(f"  Max drawdown   : -{_max_drawdown(equity):.1f} pips")
    lines.append(f"  Avg hold (candles): {np.mean([t['candles_held'] for t in trades]):.1f}")
    lines.append("")

    # ── Per-direction breakdown ────────────────────────────────────────────────
    for label, group in [("BUY", buy_trades), ("SELL", sell_trades)]:
        if not group:
            continue
        g_pnl  = [t["pnl_pips"] for t in group]
        g_wins = sum(1 for t in group if t["result"] == "WIN")
        lines.append(f"── {label} trades ({len(group)}) ──────────────────────────────────")
        lines.append(f"  Win rate   : {100*g_wins/len(group):.1f}%")
        lines.append(f"  Total PnL  : {sum(g_pnl):+.1f} pips")
        lines.append(f"  Avg trade  : {np.mean(g_pnl):+.2f} pips")
        lines.append(f"  Profit fac : {_profit_factor(g_pnl):.2f}")
        lines.append("")

    # ── Calibration check ─────────────────────────────────────────────────────
    lines.append("── Confidence vs Win-Rate Calibration ───────────────────")
    lines.append(f"  {'Conf range':>12}  {'N':>5}  {'Avg conf':>9}  {'Win rate':>9}")
    lines.append(f"  {'-'*42}")
    for b in _calibration_check(trades):
        lines.append(f"  {b['lo']:.0%} – {b['hi']:.0%}    "
                     f"{b['n']:>5}  {b['avg_conf']:>9.1%}  {b['win_rate']:>9.1%}")
    lines.append("")

    # ── Interpretation ────────────────────────────────────────────────────────
    lines.append("── Interpretation ───────────────────────────────────────")
    win_rate = wins / n
    pf       = _profit_factor(pnls)
    exp      = total_pnl / n

    if n < 50:
        lines.append(f"  WARNING: only {n} trades — not statistically meaningful (need > 50)")
    if win_rate >= 0.55:
        lines.append(f"  Win rate {win_rate:.1%} >= 55% — strong signal ✓")
    elif win_rate >= 0.50:
        lines.append(f"  Win rate {win_rate:.1%} — marginal edge, monitor closely")
    else:
        lines.append(f"  Win rate {win_rate:.1%} < 50% — negative edge, do not trade live")

    if pf >= 1.5:
        lines.append(f"  Profit factor {pf:.2f} — excellent ✓")
    elif pf >= 1.2:
        lines.append(f"  Profit factor {pf:.2f} — acceptable ✓")
    elif pf > 1.0:
        lines.append(f"  Profit factor {pf:.2f} — marginal, spread or slippage could erase edge")
    else:
        lines.append(f"  Profit factor {pf:.2f} — strategy is losing, do not trade live")

    if exp > 0:
        lines.append(f"  Expectancy +{exp:.2f} pips/trade — positive edge ✓")
    else:
        lines.append(f"  Expectancy {exp:.2f} pips/trade — negative expected value")

    lines.append("")
    lines.append("=" * 62)
    lines.append("  NEXT: If metrics pass — move to Phase 5 (live MT5 inference)")
    lines.append("        If metrics fail — revisit labeling k value or threshold")
    lines.append("=" * 62)
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Backtest on test set")
    parser.add_argument("--config",      default="config/config.yaml")
    parser.add_argument("--checkpoint",  default="models/transformer_best.pt")
    parser.add_argument("--calibration", default="models/calibration.json")
    parser.add_argument("--data",        default="data/raw/EURUSD_M15_tokenized.parquet")
    parser.add_argument("--threshold",   type=float, default=None,
                        help="Override entry_threshold from config")
    parser.add_argument("--out-txt",     default="logs/backtest_report.txt")
    parser.add_argument("--out-csv",     default="logs/backtest_trades.csv")
    parser.add_argument("--out-json",    default="logs/backtest_summary.json")
    args = parser.parse_args()

    cfg = _load_config(args.config)

    for p in [args.checkpoint, args.calibration, args.data]:
        if not Path(p).exists():
            print(f"[ERROR] Not found: {p}")
            raise SystemExit(1)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    threshold = args.threshold or cfg["trading"]["entry_threshold"]
    spread    = cfg["trading"]["spread_pips"] * PIP_SIZE
    stop_mult = cfg["trading"]["atr_stop_mult"]
    tgt_mult  = cfg["trading"]["atr_target_mult"]
    context   = cfg["data"]["context_window"]       # 64
    max_seq   = cfg["model"]["max_seq_len"]          # 335
    hold_per  = cfg["labeling"]["horizon"]           # 3
    train_r   = cfg["data"]["train_ratio"]           # 0.70
    val_r     = cfg["data"]["val_ratio"]             # 0.15

    print(f"\n[backtester] Device     : {device}")
    print(f"[backtester] Threshold  : {threshold:.0%}")
    print(f"[backtester] Spread     : {cfg['trading']['spread_pips']} pip")
    print(f"[backtester] Stop/Target: {stop_mult}x / {tgt_mult}x ATR")
    print(f"[backtester] Hold period: {hold_per} candles")

    # ── Load model + temperature ───────────────────────────────────────────────
    from src.model.transformer import load_checkpoint
    model = load_checkpoint(args.checkpoint, cfg, device)
    model.eval()

    with open(args.calibration) as f:
        T = json.load(f)["temperature"]
    print(f"[backtester] Temperature: {T:.4f}")

    # ── Load tokenized data ────────────────────────────────────────────────────
    print(f"[backtester] Loading    : {args.data}")
    df = pd.read_parquet(args.data)
    n  = len(df)
    print(f"[backtester] Total rows : {n:,}")

    # ── Reproduce test split (same logic as sequences.py) ─────────────────────
    train_end = int(n * train_r)
    val_end   = int(n * (train_r + val_r))
    warmup    = context - 1   # 63

    labels   = df["label"].to_numpy(dtype=np.int64)
    exit_idx = df["exit_idx"].to_numpy(dtype=np.int32)

    test_mask = (
        (np.arange(n) >= max(warmup, val_end)) &
        (np.arange(n) < n) &
        (labels >= 0) &
        (exit_idx < n)
    )
    test_events = np.where(test_mask)[0]

    print(f"[backtester] Test events: {len(test_events):,}")
    print(f"[backtester] Test period: {df['time'].iloc[val_end].date()} "
          f"→ {df['time'].iloc[-1].date()}")
    print(f"[backtester] Running simulation...\n")

    # ── Token array for sequence building ─────────────────────────────────────
    token_array = df[TOKEN_COLS].to_numpy(dtype=np.int32)   # (N, 5)

    # ── Simulate each test event ───────────────────────────────────────────────
    trades        = []
    skipped_conf  = 0
    skipped_edge  = 0   # no next candle
    all_signals   = len(test_events)

    for i in test_events:
        # ── Build sequence and run inference ──────────────────────────────────
        seq    = _build_sequence(token_array, i, context, max_seq).to(device)
        with torch.no_grad():
            logits = model(seq)
        probs = F.softmax(logits / T, dim=1).squeeze()

        p_sell = probs[0].item()
        p_buy  = probs[2].item()

        direction  = "BUY" if p_buy >= p_sell else "SELL"
        confidence = p_buy if direction == "BUY" else p_sell

        if confidence < threshold:
            skipped_conf += 1
            continue

        # ── Entry at next candle open ──────────────────────────────────────────
        entry_j = i + 1
        if entry_j >= n:
            skipped_edge += 1
            continue

        entry_raw = df["open"].iloc[entry_j]
        atr       = df["ATR_14"].iloc[i]

        if direction == "BUY":
            entry  = entry_raw + spread
            stop   = entry - stop_mult * atr
            target = entry + tgt_mult  * atr
        else:
            entry  = entry_raw - spread
            stop   = entry + stop_mult * atr
            target = entry - tgt_mult  * atr

        # ── Simulate trade ─────────────────────────────────────────────────────
        result, exit_price, candles_held = _simulate_trade(
            direction, entry, stop, target, df, entry_j, hold_per)

        pnl = _pnl_pips(direction, entry, exit_price)

        trades.append({
            "time":         str(df["time"].iloc[i]),
            "direction":    direction,
            "confidence":   round(confidence, 4),
            "entry_price":  round(entry, 5),
            "stop":         round(stop, 5),
            "target":       round(target, 5),
            "exit_price":   round(exit_price, 5),
            "atr":          round(atr, 5),
            "result":       result,
            "pnl_pips":     round(pnl, 2),
            "candles_held": candles_held,
        })

    # ── Stats summary ──────────────────────────────────────────────────────────
    n_trades  = len(trades)
    pnls      = [t["pnl_pips"] for t in trades]
    total_pnl = sum(pnls)
    wins      = sum(1 for t in trades if t["result"] == "WIN")

    print(f"  Signals above threshold  : {n_trades:,} / {all_signals:,}  "
          f"({100*n_trades/all_signals:.1f}%)")
    if n_trades:
        print(f"  Win rate                 : {100*wins/n_trades:.1f}%")
        print(f"  Total PnL                : {total_pnl:+.1f} pips")
        print(f"  Expectancy               : {total_pnl/n_trades:+.2f} pips/trade")
        print(f"  Profit factor            : {_profit_factor(pnls):.2f}")

    # ── Save trades CSV ────────────────────────────────────────────────────────
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    if trades:
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=trades[0].keys())
            writer.writeheader()
            writer.writerows(trades)
        print(f"\n[backtester] Trades CSV → {args.out_csv}  ({n_trades} rows)")

    # ── Save full report ───────────────────────────────────────────────────────
    report = _format_report(trades, all_signals, threshold, cfg)
    print()
    print(report)

    with open(args.out_txt, "w") as f:
        f.write(report + "\n")
    print(f"[backtester] Report txt → {args.out_txt}")

    # ── Save JSON summary ──────────────────────────────────────────────────────
    summary = {
        "threshold":      threshold,
        "total_signals":  all_signals,
        "total_trades":   n_trades,
        "coverage":       round(n_trades / all_signals, 4) if all_signals else 0,
        "win_rate":       round(wins / n_trades, 4) if n_trades else 0,
        "total_pnl_pips": round(total_pnl, 2),
        "expectancy":     round(total_pnl / n_trades, 4) if n_trades else 0,
        "profit_factor":  round(_profit_factor(pnls), 4) if pnls else 0,
        "per_trade_sharpe": round(_sharpe(pnls), 4) if pnls else 0,
        "max_drawdown_pips": round(_max_drawdown([0.0] + list(np.cumsum(pnls))), 2) if pnls else 0,
        "buy_trades":     sum(1 for t in trades if t["direction"] == "BUY"),
        "sell_trades":    sum(1 for t in trades if t["direction"] == "SELL"),
        "time_exits":     sum(1 for t in trades if t["result"] == "TIME"),
    }
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[backtester] JSON       → {args.out_json}")


if __name__ == "__main__":
    main()
