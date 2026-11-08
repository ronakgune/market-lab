#!/usr/bin/env python3
"""One-command historical backtest for the trend stream (Stream 1).

Runs the existing TSMOM and vt_trend signals over provenanced daily bars
(market_lab.daily_data — Alpaca SIP, never synthetic) through the repo's
sizing backtest engine (next-open fills, 5bps slippage + $1 commission).

Discipline wired in, not optional:
- A lookahead pre-flight (validation.check_lookahead) runs on each signal
  before any numbers are produced; a leak aborts the run.
- Every configuration actually run is recorded in the program-wide
  TrialLedger (data/market-lab/trials.jsonl) BEFORE results are printed,
  so the trial count behind any future "winner" is honest.
- Two evaluations per configuration: full period, and a later-half split
  (evaluation starts 50% through the series) as a crude regime check.
  Both are separate ledger trials, because both are separate looks.

Usage:
    python scripts/run_trend_backtest.py [--symbols SPY QQQ] [--cash 25000]
        [--strategies vt_trend vt_trend_vix]

Strategies: ``tsmom`` and ``vt_trend`` are the base signals.
``vt_trend_vix`` is vt_trend wrapped in the BreakFree-harvested
vol-regime overlay (market_lab.vol_regime: VIXY-ratio veto + regime
sizing, dissected 2026-11-07; see docs/negative-results.md).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from market_lab.backtest import run_signal_backtest_with_sizing
from market_lab.config import PRICE_DIR
from market_lab.daily_data import load_daily_bars
from market_lab.signals import generate_tsmom_signal, generate_vt_trend_signal
from market_lab.validation import TrialLedger, check_lookahead, signal_series
from market_lab.vol_regime import COOLDOWN_BARS, REGIME_LOOKBACK, VETO_RATIO, with_vix_overlay

STRATEGIES = {
    "tsmom": generate_tsmom_signal,
    "vt_trend": generate_vt_trend_signal,
}
# Overlay strategies wrap a base signal; the vol series is VIXY daily
# bars from the same provenanced cache as every other input.
OVERLAY_STRATEGIES = {
    "vt_trend_vix": "vt_trend",
}
VOL_SYMBOL = "VIXY"
LEDGER_PATH = PRICE_DIR.parent / "trials.jsonl"
PREFLIGHT_BARS = 300  # causality is length-independent; 300 bars keeps it exhaustive + fast


def cagr(total_return: float, trading_days: int) -> float:
    if trading_days <= 0 or total_return <= -1.0:
        return 0.0
    years = trading_days / 252.0
    return (1.0 + total_return) ** (1.0 / years) - 1.0


def run_one(symbol: str, strategy_name: str, signal_fn, bars, meta, cash: float,
            evaluation_start_index: int | None, ledger: TrialLedger,
            extra_params: dict | None = None, notes: str | None = None) -> dict:
    # Refuse degenerate runs outright: the engine returns all-zero results
    # for them, and recording those would plant phantom trials in the
    # program-wide ledger (a fake "Sharpe 0.0" measurement that never ran,
    # a window too short to contain a single return interval, or an eval
    # start the engine cannot honor as given).
    if len(bars) < 122:  # engine minimum: min_history (120) + 2
        raise ValueError(f"need at least 122 bars to run, got {len(bars)}")
    if not math.isfinite(cash) or cash <= 0:
        raise ValueError(f"initial cash must be positive and finite, got {cash}")
    if evaluation_start_index is not None and not 120 <= evaluation_start_index <= len(bars) - 2:
        raise ValueError(
            f"evaluation_start_index must be in [120, {len(bars) - 2}], got {evaluation_start_index}"
        )
    result = run_signal_backtest_with_sizing(
        symbol, bars, signal_fn, min_history=120, initial_cash=cash,
        evaluation_start_index=evaluation_start_index,
    )
    eval_days = len(bars) - (evaluation_start_index or 120)
    row = {
        "symbol": symbol,
        "strategy": strategy_name,
        "eval": "full" if evaluation_start_index is None else "later-half",
        "total_return": result.total_return,
        "cagr": cagr(result.total_return, eval_days),
        "benchmark_return": result.benchmark_return,
        "benchmark_cagr": cagr(result.benchmark_return, eval_days),
        "sharpe": result.sharpe,
        "max_drawdown": result.max_drawdown,
        "trades": result.trades,
        "final_equity": result.final_equity,
    }
    ledger.record(
        stream="trend",
        strategy=strategy_name,
        params={
            "signal": signal_fn.__name__,
            "min_history": 120,
            "evaluation_start_index": evaluation_start_index,
            "initial_cash": cash,
            "execution": "next-open, 5bps slippage + $1 commission",
            **(extra_params or {}),
        },
        dataset={
            "symbol": symbol,
            "bars": len(bars),
            "first": bars[0].date.isoformat(),
            "last": bars[-1].date.isoformat(),
            "source": meta.get("source", "unknown"),
            "feed": meta.get("feed", "unknown"),
            "adjustment": meta.get("adjustment", "unknown"),
        },
        metrics={
            "total_return": row["total_return"],
            "cagr": row["cagr"],
            "benchmark_return": row["benchmark_return"],
            "sharpe": row["sharpe"],
            "max_drawdown": row["max_drawdown"],
            "trades": float(row["trades"]),
        },
        notes=notes or f"trend stream first measurement; eval={row['eval']}",
    )
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="*", default=["SPY", "QQQ"])
    parser.add_argument("--strategies", nargs="*", default=list(STRATEGIES))
    parser.add_argument("--cash", type=float, default=25_000.0)
    parser.add_argument("--ledger", type=str, default=str(LEDGER_PATH))
    args = parser.parse_args()

    ledger = TrialLedger(args.ledger)
    rows: list[dict] = []
    vix_bars = None
    if any(name in OVERLAY_STRATEGIES for name in args.strategies):
        vix_bars, _vix_meta = load_daily_bars(VOL_SYMBOL)
    for symbol in args.symbols:
        bars, meta = load_daily_bars(symbol)
        for name in args.strategies:
            extra_params = None
            notes = None
            if name in OVERLAY_STRATEGIES:
                fn = with_vix_overlay(STRATEGIES[OVERLAY_STRATEGIES[name]], vix_bars)
                extra_params = {
                    "vix_overlay": {
                        "vol_symbol": VOL_SYMBOL,
                        "lookback": REGIME_LOOKBACK,
                        "veto_ratio": VETO_RATIO,
                        "cooldown_bars": COOLDOWN_BARS,
                        "source": "BreakFree harvest 2026-11-07 (market_lab.vol_regime)",
                    }
                }
                notes = "vt_trend + VIX-regime overlay (BreakFree harvest); eval={eval}"
            else:
                fn = STRATEGIES[name]
            preflight = check_lookahead(signal_series(fn, symbol), bars[:PREFLIGHT_BARS])
            if not preflight.passed:
                print(f"{symbol} {name}: LOOKAHEAD PRE-FLIGHT FAILED — run aborted")
                for v in preflight.violations[:3]:
                    print(f"  cutoff={v.cutoff} index={v.index} full={v.full_value} truncated={v.truncated_value}")
                return 1
            for eval_start in (None, len(bars) // 2):
                rows.append(run_one(
                    symbol, name, fn, bars, meta, args.cash, eval_start, ledger,
                    extra_params=extra_params,
                    notes=notes.format(eval="full" if eval_start is None else "later-half") if notes else None,
                ))

    print(f"{'symbol':6s} {'strategy':9s} {'eval':10s} {'CAGR':>7s} {'bench':>7s} {'Sharpe':>7s} {'maxDD':>7s} {'trades':>6s}")
    for r in rows:
        print(f"{r['symbol']:6s} {r['strategy']:9s} {r['eval']:10s} "
              f"{r['cagr'] * 100:6.2f}% {r['benchmark_cagr'] * 100:6.2f}% "
              f"{r['sharpe']:7.2f} {r['max_drawdown'] * 100:6.1f}% {r['trades']:6d}")
    print(f"\n{len(rows)} trials recorded in {args.ledger} (total ledger trials: {ledger.count()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
