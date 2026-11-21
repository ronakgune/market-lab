#!/usr/bin/env python3
"""Phase-2 evaluation for the quality dip-buy stream (Stream 7).

Runs the pre-registered signal — docs/dip-buy-preregistration.md,
frozen 2026-11-10 — through the repo's weighted backtest harness
under conditions identical to trials #1-8: per name standalone,
$25,000 starting cash, next-open fills, 5 bps slippage + $1
commission, min_history 120. Two evaluations per name (full period
and the runner's later-half midpoint split) are separate ledger
trials, because both are separate looks.

Benchmarks are computed on the same windows with the harness's own
conventions (full: base = close at the min_history bar; later-half:
base = open at the evaluation-start bar): buy-and-hold of the same
name, and buy-and-hold SPY over the same calendar window.

Discipline wired in, not optional:
- A lookahead pre-flight (validation.check_lookahead) runs on the
  quality-gated signal before any numbers are produced; a leak
  aborts the run.
- Every configuration actually run is recorded in the program-wide
  TrialLedger (data/market-lab/trials.jsonl) BEFORE results are
  printed.
- No parameter is adjustable here: the tranche ladder, the gate
  thresholds, and the survival/kill bars are the pre-registration's.

Compute note (semantics-preserving): the registered signal is a
pure function of (symbol, bars) whose per-call cost is dominated by
indicators.sma's statistics.mean over every window. To make a
12-name evaluation tractable, the engine passes use a fast signal
whose sma is computed with math.fsum window sums (<= 1 ulp from
statistics.mean's exact result) and whose rolling volatility uses
sliding sums (volatility is evidence-only in the Signal; no
decision reads it). Before a name's fast results are used, the
fast signal is verified against the ORIGINAL signal on every bar
that could move money or sit near a decision boundary — every
invested bar, every weight transition, and every bar whose
episode_depth / slope / depth_below evidence lies within 1e-6 of a
registered threshold — plus every 97th bar as a sweep. Any
disagreement on (action, target_weight) discards the fast path for
that name and the original signal is used instead. The lookahead
pre-flight always uses the original, unmodified signal.

Usage:
    python scripts/run_dip_buy_backtest.py [--cash 25000]
"""

from __future__ import annotations

import argparse
import math
import sys
from functools import lru_cache
from pathlib import Path
from statistics import mean, median, pstdev

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import market_lab.dip_buy as dip_buy_module
from market_lab.backtest import run_signal_backtest_with_sizing
from market_lab.config import PRICE_DIR
from market_lab.daily_data import load_daily_bars
from market_lab.data import Bar
from market_lab.dip_buy import generate_dip_buy_signal, with_quality_gate
from market_lab.fundamentals import load_company_facts, quality_as_of
from market_lab.indicators import max_drawdown
from market_lab.validation import TrialLedger, check_lookahead, signal_series

UNIVERSE = [
    "AAPL", "AMD", "AMZN", "AVGO", "COIN", "GOOGL",
    "META", "MSFT", "MSTR", "NVDA", "SMCI", "TSLA",
]
MIN_HISTORY = 120
LEDGER_PATH = PRICE_DIR.parent / "trials.jsonl"
PREFLIGHT_BARS = 300  # causality is length-independent; 300 bars keeps it exhaustive + fast
BOUNDARY_EPS = 1e-6   # evidence distance to a registered threshold that forces verification
DEPTH_THRESHOLDS = (0.03, 0.07, 0.12)


def cagr(total_return: float, trading_days: int) -> float:
    if trading_days <= 0 or total_return <= -1.0:
        return 0.0
    years = trading_days / 252.0
    return (1.0 + total_return) ** (1.0 / years) - 1.0


def _curve_stats(curve: list[float]) -> tuple[float, float, float]:
    """(total_return, sharpe, max_drawdown) with the harness's formulas."""
    total = curve[-1] / curve[0] - 1.0 if curve and curve[0] > 0 else 0.0
    rets = [
        curve[i] / curve[i - 1] - 1.0
        for i in range(1, len(curve))
        if curve[i - 1] > 0 and math.isfinite(curve[i]) and math.isfinite(curve[i - 1])
    ]
    ret_std = pstdev(rets) if len(rets) > 2 else 0.0
    sharpe = (mean(rets) / ret_std * math.sqrt(252)) if len(rets) > 2 and ret_std > 0 else 0.0
    return total, sharpe, max_drawdown(curve)


def buy_and_hold_stats(bars: list[Bar], cash: float,
                       evaluation_start_index: int | None) -> dict:
    """Buy-and-hold of `bars` on the harness's window conventions.

    Full window: base is the close at bar MIN_HISTORY, the curve is
    close-valued from that bar. Later-half: base is the open at the
    evaluation-start bar, and the curve carries the extra open-valued
    point there — exactly how backtest._windowed_stats builds the
    strategy's evaluation curve and benchmark.
    """
    if evaluation_start_index is None:
        base = bars[MIN_HISTORY].close
        curve = [cash * b.close / base for b in bars[MIN_HISTORY:]]
        days = len(bars) - MIN_HISTORY
    else:
        esi = evaluation_start_index
        base = bars[esi].open
        curve = [cash] + [cash * b.close / base for b in bars[esi:]]
        days = len(bars) - esi
    total, sharpe, mdd = _curve_stats(curve)
    return {
        "total_return": total,
        "cagr": cagr(total, days),
        "sharpe": sharpe,
        "max_drawdown": mdd,
    }


def spy_buy_and_hold_stats(spy_bars: list[Bar], name_bars: list[Bar], cash: float,
                           evaluation_start_index: int | None) -> dict:
    """Buy-and-hold SPY over the same calendar window as a name's run."""
    by_date = {b.date: b for b in spy_bars}
    if evaluation_start_index is None:
        start_bar = name_bars[MIN_HISTORY]
        base = by_date[start_bar.date].close
        window = [b for b in spy_bars if start_bar.date <= b.date <= name_bars[-1].date]
        curve = [cash * b.close / base for b in window]
        days = len(name_bars) - MIN_HISTORY
    else:
        start_bar = name_bars[evaluation_start_index]
        base = by_date[start_bar.date].open
        window = [b for b in spy_bars if start_bar.date <= b.date <= name_bars[-1].date]
        curve = [cash] + [cash * b.close / base for b in window]
        days = len(name_bars) - evaluation_start_index
    total, sharpe, mdd = _curve_stats(curve)
    return {
        "total_return": total,
        "cagr": cagr(total, days),
        "sharpe": sharpe,
        "max_drawdown": mdd,
    }


def count_entries(weights: list[float | None], first_signal_index: int) -> int:
    """Position entries: desired weight stepping 0 -> positive.

    The harness starts flat and suppresses fills before the window,
    so the weight before `first_signal_index` is 0. Tranche step-ups
    inside an episode are adds, not entries.
    """
    entries = 0
    prev = 0.0
    for i in range(first_signal_index, len(weights)):
        w = weights[i] or 0.0
        if w > 0 and prev == 0.0:
            entries += 1
        prev = w
    return entries


# ---------------------------------------------------------------------------
# Signal construction: original, fast equivalent, memoization
# ---------------------------------------------------------------------------

def make_quality_fn(symbol: str):
    """Memoized point-in-time gate over one preloaded facts payload."""
    facts = load_company_facts(symbol)

    @lru_cache(maxsize=None)
    def quality_fn(sym: str, as_of) -> object:
        return quality_as_of(sym, as_of, facts)

    return quality_fn


def _fast_sma(values: list[float], window: int) -> list[float | None]:
    """indicators.sma with math.fsum window sums (<= 1 ulp difference)."""
    if window <= 0:
        raise ValueError("window must be positive")
    out: list[float | None] = []
    for i in range(len(values)):
        if i + 1 >= window:
            out.append(math.fsum(values[i - window + 1: i + 1]) / window)
        else:
            out.append(None)
    return out


def _fast_rolling_volatility(values: list[float], window: int = 20,
                             annualization: int = 252) -> list[float | None]:
    """indicators.rolling_volatility via sliding sums (evidence only)."""
    rets: list[float | None] = [None]
    for prev, cur in zip(values, values[1:]):
        rets.append(
            None if prev == 0 or not math.isfinite(prev) or not math.isfinite(cur)
            else cur / prev - 1
        )
    out: list[float | None] = []
    run_sum = 0.0
    run_sq = 0.0
    run_n = 0
    for i in range(len(rets)):
        v = rets[i]
        if v is not None:
            run_sum += v
            run_sq += v * v
            run_n += 1
        if i >= window:
            old = rets[i - window]
            if old is not None:
                run_sum -= old
                run_sq -= old * old
                run_n -= 1
        if i + 1 >= window and run_n >= 2:
            m = run_sum / run_n
            var = run_sq / run_n - m * m
            out.append(math.sqrt(var) * math.sqrt(annualization) if var > 0 else 0.0)
        else:
            out.append(None)
    return out


def make_fast_gated_signal(quality_fn):
    """The gated signal with fast indicator bindings (module-level swap).

    dip_buy imported sma/rolling_volatility by name, so rebinding the
    module attributes swaps the implementation for every caller in
    this process. Callers must restore the originals afterwards.
    """
    dip_buy_module.sma = _fast_sma
    dip_buy_module.rolling_volatility = _fast_rolling_volatility
    return with_quality_gate(generate_dip_buy_signal, quality_fn)


def restore_indicators() -> None:
    from market_lab import indicators
    dip_buy_module.sma = indicators.sma
    dip_buy_module.rolling_volatility = indicators.rolling_volatility


def memoize_signal(signal_fn):
    """Cache a pure signal function's verdict per distinct prefix.

    The pre-registration defines the signal as a pure function of
    (symbol, bars), and every pass in this runner (weights series,
    full run, later-half run) asks for the same prefixes of the same
    series. Memoizing on (symbol, prefix length, last bar date)
    collapses those passes to one computation per prefix. The
    lookahead pre-flight deliberately uses the UNmemoized signal, so
    the causality check stays a genuine check.
    """
    cache: dict = {}

    def wrapped(symbol: str, bars: list[Bar]):
        key = (symbol, len(bars), bars[-1].date if bars else None)
        if key not in cache:
            cache[key] = signal_fn(symbol, bars)
        return cache[key]

    wrapped.__name__ = signal_fn.__name__
    return wrapped


def verify_fast_signal(symbol: str, bars: list[Bar], fast_signals: list,
                       original_fn) -> tuple[bool, int]:
    """Re-check the original signal wherever the fast one could differ.

    A bar is verified if the fast signal invests there, changes
    weight there, carries evidence within BOUNDARY_EPS of a
    registered decision threshold (a tranche depth, zero slope, the
    SMA200 line itself), or is every 97th bar. Returns (all_match,
    bars_checked). Comparison is on (action, target_weight) — the
    only fields the harness consumes.
    """
    check: set[int] = set()
    prev_w = 0.0
    for i, sig in enumerate(fast_signals):
        w = sig.target_weight or 0.0
        ev = sig.evidence if isinstance(sig.evidence, dict) else {}
        near = False
        if ev:
            if any(abs(float(ev.get("episode_depth", 1.0)) - t) <= BOUNDARY_EPS
                   for t in DEPTH_THRESHOLDS):
                near = True
            if abs(float(ev.get("slope", 1.0))) <= BOUNDARY_EPS:
                near = True
            if abs(float(ev.get("depth_below", 1.0))) <= BOUNDARY_EPS:
                near = True
        if w > 0 or w != prev_w or near or i % 97 == 0:
            check.add(i)
        prev_w = w
    for i in sorted(check):
        orig = original_fn(symbol, bars[: i + 1])
        fast = fast_signals[i]
        if orig.action != fast.action or (orig.target_weight or 0.0) != (fast.target_weight or 0.0):
            print(f"  {symbol}: fast/original MISMATCH at bar {i} ({bars[i].date}): "
                  f"fast=({fast.action}, {fast.target_weight}) "
                  f"orig=({orig.action}, {orig.target_weight}) — using original signal")
            return False, len(check)
    return True, len(check)


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

def run_one(symbol: str, signal_fn, bars, meta, spy_bars, weights,
            cash: float, evaluation_start_index: int | None,
            ledger: TrialLedger) -> dict:
    if len(bars) < 122:  # engine minimum: min_history (120) + 2
        raise ValueError(f"need at least 122 bars to run, got {len(bars)}")
    if not math.isfinite(cash) or cash <= 0:
        raise ValueError(f"initial cash must be positive and finite, got {cash}")
    if evaluation_start_index is not None and not 120 <= evaluation_start_index <= len(bars) - 2:
        raise ValueError(
            f"evaluation_start_index must be in [120, {len(bars) - 2}], got {evaluation_start_index}"
        )
    result = run_signal_backtest_with_sizing(
        symbol, bars, signal_fn, min_history=MIN_HISTORY, initial_cash=cash,
        evaluation_start_index=evaluation_start_index,
    )
    eval_days = len(bars) - (evaluation_start_index or MIN_HISTORY)
    bench = buy_and_hold_stats(bars, cash, evaluation_start_index)
    spy_bench = spy_buy_and_hold_stats(spy_bars, bars, cash, evaluation_start_index)
    first_signal = MIN_HISTORY if evaluation_start_index is None else evaluation_start_index - 1
    entries = count_entries(weights, first_signal)
    row = {
        "symbol": symbol,
        "eval": "full" if evaluation_start_index is None else "later-half",
        "total_return": result.total_return,
        "cagr": cagr(result.total_return, eval_days),
        "benchmark_return": result.benchmark_return,
        "benchmark_cagr": bench["cagr"],
        "benchmark_sharpe": bench["sharpe"],
        "benchmark_max_drawdown": bench["max_drawdown"],
        "spy_cagr": spy_bench["cagr"],
        "spy_sharpe": spy_bench["sharpe"],
        "spy_max_drawdown": spy_bench["max_drawdown"],
        "sharpe": result.sharpe,
        "max_drawdown": result.max_drawdown,
        "trades": result.trades,
        "entries": entries,
        "final_equity": result.final_equity,
    }
    # The harness's own benchmark return must agree with the curve
    # reconstruction; a disagreement means the window conventions
    # drifted and none of the benchmark stats can be trusted.
    if abs(bench["total_return"] - result.benchmark_return) > 1e-9:
        raise AssertionError(
            f"{symbol} {row['eval']}: reconstructed B&H return {bench['total_return']} "
            f"!= harness benchmark_return {result.benchmark_return}"
        )
    ledger.record(
        stream="dip_buy",
        strategy="dip_buy",
        params={
            "signal": signal_fn.__name__,
            "min_history": MIN_HISTORY,
            "evaluation_start_index": evaluation_start_index,
            "initial_cash": cash,
            "execution": "next-open, 5bps slippage + $1 commission",
            "quality_gate": "EDGAR point-in-time (filed<=as_of): ROE>=10%, FCF>0 latest FY "
                            "and >=3 of last 5 FY, liabilities/assets<=0.85",
            "preregistration": "docs/dip-buy-preregistration.md (2026-11-10)",
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
            "benchmark_cagr": row["benchmark_cagr"],
            "benchmark_sharpe": row["benchmark_sharpe"],
            "benchmark_max_drawdown": row["benchmark_max_drawdown"],
            "sharpe": row["sharpe"],
            "max_drawdown": row["max_drawdown"],
            "trades": float(row["trades"]),
            "entries": float(entries),
        },
        notes=f"Stream 7 phase 2 (pre-registered); eval={row['eval']}",
    )
    return row


def evaluate_bars(rows: list[dict]) -> dict:
    """Frozen survival bars + kill criteria over one window's rows."""
    med = lambda key: median(r[key] for r in rows)
    s_sharpe, b_sharpe = med("sharpe"), med("benchmark_sharpe")
    s_dd, b_dd = med("max_drawdown"), med("benchmark_max_drawdown")
    s_cagr, b_cagr = med("cagr"), med("benchmark_cagr")
    dd_improvement = 1.0 - abs(s_dd) / abs(b_dd) if b_dd != 0 else 0.0
    cagr_ratio = s_cagr / b_cagr if b_cagr > 0 else float("nan")
    falling_knife = sum(
        1 for r in rows
        if r["max_drawdown"] < r["benchmark_max_drawdown"] - 0.10
        and r["cagr"] < r["benchmark_cagr"]
    )
    return {
        "median_sharpe": s_sharpe, "median_benchmark_sharpe": b_sharpe,
        "median_max_drawdown": s_dd, "median_benchmark_max_drawdown": b_dd,
        "dd_improvement": dd_improvement,
        "median_cagr": s_cagr, "median_benchmark_cagr": b_cagr,
        "cagr_ratio": cagr_ratio,
        "median_entries": med("entries"),
        "falling_knife_names": falling_knife,
        "survival_1_sharpe": s_sharpe >= b_sharpe,
        "survival_2_drawdown": dd_improvement >= 0.25,
        "survival_3_cagr": cagr_ratio >= 0.60,
        "kill_1_sharpe": s_sharpe < b_sharpe - 0.10,
        "kill_2_drawdown": dd_improvement < 0.15,
        "kill_3_cagr": cagr_ratio < 0.50,
        "kill_4_inert": med("entries") < 2,
        "kill_5_falling_knife": falling_knife >= math.ceil(len(rows) / 3),
    }


def print_verdict(label: str, ev: dict, n_names: int) -> None:
    print(f"\n== {label} medians across {n_names} names ==")
    print(f"Sharpe      strategy {ev['median_sharpe']:6.2f}  vs B&H {ev['median_benchmark_sharpe']:6.2f}"
          f"  -> survival bar 1: {'PASS' if ev['survival_1_sharpe'] else 'FAIL'}"
          f" | kill 1: {'TRIGGERED' if ev['kill_1_sharpe'] else 'no'}")
    print(f"maxDD       strategy {ev['median_max_drawdown'] * 100:6.1f}% vs B&H {ev['median_benchmark_max_drawdown'] * 100:6.1f}%"
          f"  improvement {ev['dd_improvement'] * 100:5.1f}%"
          f"  -> survival bar 2 (>=25%): {'PASS' if ev['survival_2_drawdown'] else 'FAIL'}"
          f" | kill 2 (<15%): {'TRIGGERED' if ev['kill_2_drawdown'] else 'no'}")
    print(f"CAGR        strategy {ev['median_cagr'] * 100:6.2f}% vs B&H {ev['median_benchmark_cagr'] * 100:6.2f}%"
          f"  ratio {ev['cagr_ratio'] * 100:5.1f}%"
          f"  -> survival bar 3 (>=60%): {'PASS' if ev['survival_3_cagr'] else 'FAIL'}"
          f" | kill 3 (<50%): {'TRIGGERED' if ev['kill_3_cagr'] else 'no'}")
    print(f"Entries     median {ev['median_entries']:.1f} per name"
          f"  -> kill 4 (<2): {'TRIGGERED' if ev['kill_4_inert'] else 'no'}")
    print(f"Falling knife: {ev['falling_knife_names']} of {n_names} names"
          f"  -> kill 5 (>=1/3): {'TRIGGERED' if ev['kill_5_falling_knife'] else 'no'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="*", default=UNIVERSE)
    parser.add_argument("--cash", type=float, default=25_000.0)
    parser.add_argument("--ledger", type=str, default=str(LEDGER_PATH))
    args = parser.parse_args()

    ledger = TrialLedger(args.ledger)
    spy_bars, _ = load_daily_bars("SPY")
    rows: list[dict] = []
    for symbol in args.symbols:
        bars, meta = load_daily_bars(symbol)
        quality_fn = make_quality_fn(symbol)
        original_fn = with_quality_gate(generate_dip_buy_signal, quality_fn)
        preflight = check_lookahead(signal_series(original_fn, symbol), bars[:PREFLIGHT_BARS])
        if not preflight.passed:
            print(f"{symbol}: LOOKAHEAD PRE-FLIGHT FAILED — run aborted")
            for v in preflight.violations[:3]:
                print(f"  cutoff={v.cutoff} index={v.index} full={v.full_value} truncated={v.truncated_value}")
            return 1
        fast_fn = memoize_signal(make_fast_gated_signal(quality_fn))
        fast_signals = [fast_fn(symbol, bars[: i + 1]) for i in range(len(bars))]
        restore_indicators()
        ok, checked = verify_fast_signal(symbol, bars, fast_signals, original_fn)
        engine_fn = fast_fn if ok else memoize_signal(original_fn)
        if ok:
            print(f"{symbol}: fast signal verified against original on {checked} bars")
        weights = [s.target_weight if s.target_weight is not None else 0.0 for s in fast_signals] \
            if ok else [w if w is not None else 0.0 for w in signal_series(engine_fn, symbol)(bars)]
        rows.append(run_one(symbol, engine_fn, bars, meta, spy_bars, weights,
                            args.cash, None, ledger))
        rows.append(run_one(symbol, engine_fn, bars, meta, spy_bars, weights,
                            args.cash, len(bars) // 2, ledger))

    print(f"{'symbol':6s} {'eval':10s} {'CAGR':>7s} {'B&H':>7s} {'SPY':>7s} {'Sharpe':>7s} "
          f"{'B&H Sh':>7s} {'maxDD':>7s} {'B&H DD':>7s} {'trades':>6s} {'entries':>7s}")
    for r in rows:
        print(f"{r['symbol']:6s} {r['eval']:10s} {r['cagr'] * 100:6.2f}% {r['benchmark_cagr'] * 100:6.2f}% "
              f"{r['spy_cagr'] * 100:6.2f}% {r['sharpe']:7.2f} {r['benchmark_sharpe']:7.2f} "
              f"{r['max_drawdown'] * 100:6.1f}% {r['benchmark_max_drawdown'] * 100:6.1f}% "
              f"{r['trades']:6d} {r['entries']:7d}")

    n = len(args.symbols)
    full = evaluate_bars([r for r in rows if r["eval"] == "full"])
    later = evaluate_bars([r for r in rows if r["eval"] == "later-half"])
    print_verdict("FULL WINDOW", full, n)
    print_verdict("LATER-HALF WINDOW", later, n)
    survives = (
        full["survival_1_sharpe"] and full["survival_2_drawdown"] and full["survival_3_cagr"]
        and not any(full[k] for k in ("kill_1_sharpe", "kill_2_drawdown", "kill_3_cagr",
                                      "kill_4_inert", "kill_5_falling_knife"))
        and not (later["kill_1_sharpe"] or later["kill_2_drawdown"])
    )
    print(f"\nPRE-REGISTERED VERDICT: {'SURVIVES' if survives else 'DOES NOT SURVIVE'}")
    print(f"{len(rows)} trials recorded in {args.ledger} (total ledger trials: {ledger.count()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
