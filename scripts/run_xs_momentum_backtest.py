#!/usr/bin/env python3
"""One-command historical backtest for the cross-sectional momentum stream (Stream 2).

Runs dual momentum (12-1 formation, monthly rebalance, top-3, SPY absolute
gate) over the full-history single-name universe on provenanced daily bars
(market_lab.daily_data — Alpaca SIP, never synthetic), with the repo's
standard execution costs (5bps + $1) and Daniel-Moskowitz vol scaling —
the configuration the strategy research set for this sleeve.

Discipline wired in, not optional:
- A point-in-time pre-flight proves dual_momentum_targets at sampled
  decision indices is invariant to truncating every series at that index;
  any difference aborts the run (the selection would be using the future).
- Every configuration actually run is recorded in the program-wide
  TrialLedger (data/market-lab/trials.jsonl) BEFORE results are printed.
- Two evaluations per configuration: full period and a later-half split,
  recorded as separate ledger trials.
- Universe honesty: only names with full 2016-2026 history are included.
  COIN (listed 2021) and the crypto ETFs are excluded from this window —
  a name cannot be ranked before it exists. The universe is still a
  survivor-selected watchlist; results carry that caveat, always.

Usage:
    python scripts/run_xs_momentum_backtest.py [--cash 25000]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from market_lab.config import PRICE_DIR
from market_lab.daily_data import load_daily_bars
from market_lab.portfolio_construction import (
    _align_on_common_dates,
    dual_momentum_targets,
    run_dual_momentum_backtest,
)
from market_lab.validation import TrialLedger

UNIVERSE = ["AAPL", "AMD", "AMZN", "AVGO", "GOOGL", "META", "MSFT", "MSTR", "NVDA", "SMCI", "TSLA"]
FORMATION_DAYS = 252
SKIP_DAYS = 21
TOP_N = 3
REBALANCE_EVERY = 21
TARGET_VOL = 0.15
LEDGER_PATH = PRICE_DIR.parent / "trials.jsonl"


def cagr(total_return: float, trading_days: int) -> float:
    if trading_days <= 0 or total_return <= -1.0:
        return 0.0
    years = trading_days / 252.0
    return (1.0 + total_return) ** (1.0 / years) - 1.0


def preflight_point_in_time(bars_by_symbol, spy_bars, aligned_len: int) -> list[str]:
    """Targets at a decision index must be identical whether the caller
    hands over the full series or only the bars up to that index."""
    start = FORMATION_DAYS + SKIP_DAYS
    sample = list(range(start, aligned_len - 1, REBALANCE_EVERY * 13))
    if sample[-1] != aligned_len - 2:
        sample.append(aligned_len - 2)
    failures = []
    for idx in sample:
        full = dual_momentum_targets(
            bars_by_symbol, formation_days=FORMATION_DAYS, skip_days=SKIP_DAYS,
            top_n=TOP_N, max_weight=1.0, decision_index=idx, spy_bars=spy_bars,
        )
        truncated = dual_momentum_targets(
            {s: b[: idx + 1] for s, b in bars_by_symbol.items()},
            formation_days=FORMATION_DAYS, skip_days=SKIP_DAYS,
            top_n=TOP_N, max_weight=1.0, decision_index=None,
            spy_bars=spy_bars[: idx + 1],
        )
        key = lambda ts: [(t.symbol, round(t.target_weight, 12)) for t in ts]
        if key(full) != key(truncated):
            failures.append(f"index {idx}: full={key(full)} truncated={key(truncated)}")
    return failures


def run_one(bars_by_symbol, spy_bars, spy_meta, aligned_len: int, cash: float,
            start_index: int | None, ledger: TrialLedger) -> dict:
    if aligned_len < FORMATION_DAYS + SKIP_DAYS + 2:
        raise ValueError(f"need at least {FORMATION_DAYS + SKIP_DAYS + 2} aligned bars, got {aligned_len}")
    if not math.isfinite(cash) or cash <= 0:
        raise ValueError(f"initial cash must be positive and finite, got {cash}")
    start = start_index if start_index is not None else FORMATION_DAYS + SKIP_DAYS
    if not FORMATION_DAYS + SKIP_DAYS <= start <= aligned_len - 2:
        raise ValueError(f"start_index must be in [{FORMATION_DAYS + SKIP_DAYS}, {aligned_len - 2}], got {start}")
    result = run_dual_momentum_backtest(
        bars_by_symbol,
        formation_days=FORMATION_DAYS,
        skip_days=SKIP_DAYS,
        top_n=TOP_N,
        rebalance_every=REBALANCE_EVERY,
        start_index=start_index,
        initial_cash=cash,
        max_weight=1.0,
        spy_bars=spy_bars,
        target_vol=TARGET_VOL,
    )
    eval_days = aligned_len - 1 - start
    row = {
        "eval": "full" if start_index is None else "later-half",
        "total_return": result.total_return,
        "cagr": cagr(result.total_return, eval_days),
        "benchmark_return": result.benchmark_return,
        "benchmark_cagr": cagr(result.benchmark_return, eval_days),
        "sharpe": result.sharpe,
        "max_drawdown": result.max_drawdown,
        "rebalances": result.rebalances,
        "costs_paid": result.costs_paid,
        "final_equity": result.final_equity,
    }
    ledger.record(
        stream="xs_momentum",
        strategy="dual_momentum_vol_scaled",
        params={
            "formation_days": FORMATION_DAYS,
            "skip_days": SKIP_DAYS,
            "top_n": TOP_N,
            "rebalance_every": REBALANCE_EVERY,
            "max_weight": 1.0,
            "target_vol": TARGET_VOL,
            "start_index": start_index,
            "initial_cash": cash,
            "execution": "next-open, 5bps slippage + $1 commission",
            "universe": UNIVERSE,
        },
        dataset={
            "symbol": "XS_UNIVERSE_11",
            "bars": aligned_len,
            "first": spy_bars[0].date.isoformat(),
            "last": spy_bars[-1].date.isoformat(),
            "source": spy_meta.get("source", "unknown"),
            "feed": spy_meta.get("feed", "unknown"),
            "adjustment": spy_meta.get("adjustment", "unknown"),
        },
        metrics={
            "total_return": row["total_return"],
            "cagr": row["cagr"],
            "benchmark_return": row["benchmark_return"],
            "sharpe": row["sharpe"],
            "max_drawdown": row["max_drawdown"],
            "rebalances": float(row["rebalances"]),
            "costs_paid": row["costs_paid"],
        },
        notes=f"xs momentum stream first measurement; eval={row['eval']}; survivor-selected watchlist universe",
    )
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cash", type=float, default=25_000.0)
    parser.add_argument("--ledger", type=str, default=str(LEDGER_PATH))
    args = parser.parse_args()

    bars_by_symbol = {}
    for symbol in UNIVERSE:
        bars, _meta = load_daily_bars(symbol)
        bars_by_symbol[symbol] = bars
    spy_bars, spy_meta = load_daily_bars("SPY")
    for symbol, bars in bars_by_symbol.items():
        if bars[0].date != spy_bars[0].date or bars[-1].date != spy_bars[-1].date:
            print(f"{symbol}: history window does not match SPY ({bars[0].date}..{bars[-1].date}) — refusing to run a mixed-window universe")
            return 1
    aligned_len = min(len(b) for b in _align_on_common_dates(bars_by_symbol).values())

    failures = preflight_point_in_time(bars_by_symbol, spy_bars, aligned_len)
    if failures:
        print("POINT-IN-TIME PRE-FLIGHT FAILED — run aborted")
        for failure in failures[:3]:
            print(f"  {failure}")
        return 1

    ledger = TrialLedger(args.ledger)
    rows = [
        run_one(bars_by_symbol, spy_bars, spy_meta, aligned_len, args.cash, None, ledger),
        run_one(bars_by_symbol, spy_bars, spy_meta, aligned_len, args.cash, aligned_len // 2, ledger),
    ]
    print(f"{'eval':10s} {'CAGR':>7s} {'bench':>7s} {'Sharpe':>7s} {'maxDD':>7s} {'rebal':>6s} {'costs':>8s}")
    for r in rows:
        print(f"{r['eval']:10s} {r['cagr'] * 100:6.2f}% {r['benchmark_cagr'] * 100:6.2f}% "
              f"{r['sharpe']:7.2f} {r['max_drawdown'] * 100:6.1f}% {r['rebalances']:6d} ${r['costs_paid']:7.0f}")
    print(f"\n{len(rows)} trials recorded in {args.ledger} (total ledger trials: {ledger.count()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
