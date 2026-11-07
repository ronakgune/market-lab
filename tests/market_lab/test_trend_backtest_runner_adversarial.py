"""Adversarial tests for the trend backtest runner (scripts/run_trend_backtest).

The runner produces the program's first believed numbers, so these tests
attack the MEASUREMENT: benchmark window alignment, CAGR denominators,
ledger integrity, degenerate inputs, and the limits of the lookahead
pre-flight. Offline fixtures only — no real price cache.

Two findings are pinned here as documentation, not failures:
1. The full-run CAGR denominator is len(bars) - 120 while the equity curve
   spans len(bars) - 121 return intervals — off by one day, in the
   conservative direction (CAGR slightly understated). The later-half
   denominator is exact. See test_cagr_denominator_*.
2. The lookahead pre-flight (signal_series + check_lookahead) is blind to
   leaks that reach OUTSIDE the bars argument (closure / global / file):
   truncated reruns feed byte-identical prefixes, so such a signal sees
   the same "future" in both runs. It does catch cross-call statefulness,
   which is its real value: the production signals are pure prefix
   functions, and purity is what makes them causal by construction.
   See test_preflight_*.
"""

from __future__ import annotations

import math
import sys
from dataclasses import replace
from datetime import date, timedelta

import pytest

from market_lab.data import Bar
from market_lab.signals import Signal, generate_tsmom_signal
from market_lab.validation import TrialLedger, check_lookahead, signal_series

import scripts.run_trend_backtest as runner
from scripts.run_trend_backtest import cagr, run_one

META = {"source": "alpaca:sip", "feed": "sip", "adjustment": "all"}
CASH = 25_000.0


def make_bars(n: int) -> list[Bar]:
    """Deterministic bars with open != close (yesterday's close = today's open)."""
    bars: list[Bar] = []
    price = 100.0
    day = date(2018, 1, 2)
    for i in range(n):
        open_ = price
        price *= 1.0 + 0.001 + 0.003 * math.sin(i / 20.0)
        bars.append(Bar(date=day + timedelta(days=i), open=open_,
                        high=max(open_, price) * 1.002, low=min(open_, price) * 0.998,
                        close=price, volume=1_000_000))
    return bars


def _signal(symbol: str, bars: list[Bar], action: str, weight: float) -> Signal:
    return Signal(symbol, action, 1.0, "adversarial fixture", bars[-1].close,
                  None, None, None, None, "fixture", weight, {})


def always_buy(symbol: str, bars: list[Bar]) -> Signal:
    return _signal(symbol, bars, "BUY", 1.0)


def never_trade(symbol: str, bars: list[Bar]) -> Signal:
    return _signal(symbol, bars, "HOLD", 0.0)


def implied_days(total_return: float, cagr_value: float) -> float:
    """Invert cagr(): days = 252 * ln(1+tr) / ln(1+cagr)."""
    return 252.0 * math.log(1.0 + total_return) / math.log(1.0 + cagr_value)


# ---------------------------------------------------------------------------
# Benchmark honesty: are the strategy and benchmark windows the same window?
# ---------------------------------------------------------------------------

def test_benchmark_full_run_base_is_close_of_first_signal_bar(tmp_path):
    bars = make_bars(400)
    row = run_one("SPY", "fixture", always_buy, bars, META, CASH, None,
                  TrialLedger(tmp_path / "t.jsonl"))
    # Exact: the engine bases the full-run benchmark on bars[120].close and
    # the strategy equity curve also starts at that bar's close (all cash).
    assert row["benchmark_return"] == bars[-1].close / bars[120].close - 1.0


def test_benchmark_later_half_base_is_eval_bar_open(tmp_path):
    bars = make_bars(400)
    e = 200
    row = run_one("SPY", "fixture", always_buy, bars, META, CASH, e,
                  TrialLedger(tmp_path / "t.jsonl"))
    # Exact: windowed benchmark is open(eval bar) -> close(last bar), and the
    # strategy's eval equity starts as cash marked at that same open.
    assert row["benchmark_return"] == bars[-1].close / bars[e].open - 1.0


def test_later_half_strategy_starts_flat_and_pays_only_costs(tmp_path):
    bars = make_bars(400)
    e = 200
    ledger = TrialLedger(tmp_path / "t.jsonl")
    flat = run_one("SPY", "fixture", never_trade, bars, META, CASH, e, ledger)
    assert flat["total_return"] == 0.0 and flat["trades"] == 0
    assert flat["final_equity"] == CASH  # warm-up took no positions

    invested = run_one("SPY", "fixture", always_buy, bars, META, CASH, e, ledger)
    drag = invested["benchmark_return"] - invested["total_return"]
    # Same window, same start: the always-in strategy trails the frictionless
    # benchmark only by fill slippage + commissions — a small positive drag,
    # not a head start in either direction.
    assert 0.0 < drag < 0.01


# ---------------------------------------------------------------------------
# CAGR / evaluation accounting
# ---------------------------------------------------------------------------

def test_cagr_guards():
    assert cagr(-1.0, 252) == 0.0    # wiped out: guarded, not NaN
    assert cagr(-1.5, 252) == 0.0    # beyond wiped out (guarded)
    assert cagr(0.5, 0) == 0.0       # no days, no CAGR
    assert cagr(0.5, -10) == 0.0


def test_cagr_denominator_later_half_is_exact(tmp_path):
    bars = make_bars(400)
    e = 200
    row = run_one("SPY", "fixture", always_buy, bars, META, CASH, e,
                  TrialLedger(tmp_path / "t.jsonl"))
    assert row["total_return"] > 0 and row["cagr"] > 0
    # The eval equity curve holds len - e return intervals (the engine marks
    # equity at the eval bar's open AND its close), and run_one divides by
    # exactly len - e. Consistent.
    assert implied_days(row["total_return"], row["cagr"]) == pytest.approx(len(bars) - e, abs=1e-6)


def test_cagr_denominator_full_run_documents_off_by_one(tmp_path):
    bars = make_bars(400)
    row = run_one("SPY", "fixture", always_buy, bars, META, CASH, None,
                  TrialLedger(tmp_path / "t.jsonl"))
    assert row["total_return"] > 0 and row["cagr"] > 0
    # DOCUMENTED: run_one uses len - 120, but the full-run equity curve
    # spans len - 121 return intervals (first point at bar 120 close, last
    # at bar len-1 close). The denominator is one day too LARGE, so the
    # recorded CAGR is slightly UNDERSTATED (conservative). ~0.36% relative
    # at this fixture's scale, ~0.04% at the real 2,513-bar scale. Pinned so
    # any future change to the convention is deliberate, not silent.
    assert implied_days(row["total_return"], row["cagr"]) == pytest.approx(len(bars) - 120, abs=1e-6)


# ---------------------------------------------------------------------------
# Ledger integrity: recorded trials are exactly the completed, identical runs
# ---------------------------------------------------------------------------

def test_failing_signal_records_no_trial_and_completed_trials_survive(tmp_path):
    bars = make_bars(400)
    path = tmp_path / "t.jsonl"
    ledger = TrialLedger(path)

    def exploding(symbol: str, prefix: list[Bar]) -> Signal:
        if len(prefix) >= 250:
            raise RuntimeError("signal died mid-run")
        return _signal(symbol, prefix, "BUY", 0.5)

    with pytest.raises(RuntimeError):
        run_one("SPY", "fixture", exploding, bars, META, CASH, None, ledger)
    assert ledger.count() == 0  # no partial / phantom trial

    # Simulate the real mid-grid death: one completed run recorded, then the
    # process "dies" inside the next — the ledger must hold exactly the one.
    run_one("SPY", "fixture", always_buy, bars, META, CASH, None, ledger)
    assert ledger.count() == 1
    with pytest.raises(RuntimeError):
        run_one("SPY", "fixture", exploding, bars, META, CASH, len(bars) // 2, ledger)
    assert ledger.count() == 1
    assert TrialLedger(path).count() == 1  # survives reopen, as recovery needs


def test_ledger_metrics_are_identical_to_the_returned_row(tmp_path):
    bars = make_bars(400)
    ledger = TrialLedger(tmp_path / "t.jsonl")
    row = run_one("SPY", "fixture", always_buy, bars, META, CASH, None, ledger)
    record = ledger.records()[0]
    # No rounding drift between what is shown and what is stored: the exact
    # floats from the result land in the ledger (JSON round-trip is exact).
    assert record.metrics["total_return"] == row["total_return"]
    assert record.metrics["cagr"] == row["cagr"]
    assert record.metrics["benchmark_return"] == row["benchmark_return"]
    assert record.metrics["sharpe"] == row["sharpe"]
    assert record.metrics["max_drawdown"] == row["max_drawdown"]
    assert record.metrics["trades"] == float(row["trades"])


def test_determinism_same_inputs_same_metrics_and_params_hash(tmp_path):
    bars = make_bars(400)

    def deterministic_varying(symbol: str, prefix: list[Bar]) -> Signal:
        # Pure function of the prefix with a varying weight: exercises the
        # sizing engine's buy/sell paths. (Determinism is a property of the
        # run_one/engine/ledger plumbing; the real TSMOM path is covered by
        # the baseline wiring test, and its ~0.1s-per-call cost at this
        # length would make this test take ~50s for no extra evidence.)
        weight = 0.5 + 0.4 * math.sin(len(prefix) / 30.0)
        return _signal(symbol, prefix, "BUY", weight)

    ledger = TrialLedger(tmp_path / "t.jsonl")
    r1 = run_one("SPY", "fixture", deterministic_varying, bars, META, CASH, None, ledger)
    r2 = run_one("SPY", "fixture", deterministic_varying, bars, META, CASH, None, ledger)
    for key in ("total_return", "cagr", "benchmark_return", "sharpe", "max_drawdown", "trades", "final_equity"):
        assert r1[key] == r2[key]
    rec1, rec2 = ledger.records()
    assert rec1.params_hash == rec2.params_hash
    assert rec1.metrics == rec2.metrics


# ---------------------------------------------------------------------------
# Degenerate inputs: refuse loudly, record nothing (fixed in run_one)
# ---------------------------------------------------------------------------

def test_short_bars_are_refused_not_recorded_as_zero(tmp_path):
    ledger = TrialLedger(tmp_path / "t.jsonl")
    for n in (0, 50, 121):  # below the engine's min_history (120) + 2
        with pytest.raises(ValueError):
            run_one("SPY", "fixture", always_buy, make_bars(n), META, CASH, None, ledger)
    assert ledger.count() == 0
    # Boundary control: exactly 122 bars is a real (if tiny) run.
    run_one("SPY", "fixture", always_buy, make_bars(122), META, CASH, None, ledger)
    assert ledger.count() == 1


def test_degenerate_eval_index_is_refused_not_recorded(tmp_path):
    bars = make_bars(400)
    ledger = TrialLedger(tmp_path / "t.jsonl")
    # len-1: window holds a single equity point (strategy 0.0 vs a nonzero
    # benchmark — a phantom comparison). 50 and 0: before min_history, where
    # the engine silently evaluates from bar 120 while run_one would divide
    # CAGR by len - eval_index — a materially wrong denominator.
    for bad in (len(bars) - 1, len(bars), 50, 0, -5):
        with pytest.raises(ValueError):
            run_one("SPY", "fixture", always_buy, bars, META, CASH, bad, ledger)
    assert ledger.count() == 0
    # Boundary control: the last legal start still runs and records.
    run_one("SPY", "fixture", always_buy, bars, META, CASH, len(bars) - 2, ledger)
    assert ledger.count() == 1


def test_nonpositive_or_nonfinite_cash_is_refused_not_recorded(tmp_path):
    bars = make_bars(400)
    ledger = TrialLedger(tmp_path / "t.jsonl")
    for bad_cash in (0.0, -100.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            run_one("SPY", "fixture", always_buy, bars, META, bad_cash, None, ledger)
    assert ledger.count() == 0


def test_constant_bars_zero_volatility_produce_finite_metrics(tmp_path):
    day = date(2020, 1, 2)
    bars = [Bar(date=day + timedelta(days=i), open=100.0, high=100.0, low=100.0,
                close=100.0, volume=1_000_000) for i in range(300)]
    ledger = TrialLedger(tmp_path / "t.jsonl")
    row = run_one("SPY", "fixture", always_buy, bars, META, CASH, None, ledger)
    assert row["trades"] >= 1  # the always-buy fixture really traded
    for key in ("total_return", "cagr", "benchmark_return", "sharpe", "max_drawdown", "final_equity"):
        assert math.isfinite(row[key]), key
    # Zero-volatility benchmark and a cost-only strategy: return is the small
    # negative of slippage + commission, never NaN from a pstdev-0 division.
    assert row["benchmark_return"] == 0.0
    assert -0.01 < row["total_return"] < 0.0
    assert ledger.count() == 1


# ---------------------------------------------------------------------------
# Pre-flight trust: what the lookahead check can and cannot see
# ---------------------------------------------------------------------------

def test_preflight_passes_honest_tsmom():
    bars = make_bars(130)
    report = check_lookahead(signal_series(generate_tsmom_signal, "SPY"), bars)
    assert report.passed


def test_preflight_catches_cross_call_statefulness():
    bars = make_bars(130)
    state = {"calls": 0}

    def stateful(symbol: str, prefix: list[Bar]) -> Signal:
        state["calls"] += 1
        weight = min(1.0, state["calls"] / 50.0)
        return _signal(symbol, prefix, "BUY", weight)

    report = check_lookahead(signal_series(stateful, "SPY"), bars)
    assert not report.passed  # reruns diverge: the check's real teeth


def make_peeking_signal(full_bars: list[Bar], gate: int | None = None, base_fn=None):
    """Leaky wrapper: tomorrow's close (reached via a closure over the full
    dataset) decides today's target weight. Defaults to wrapping the real
    TSMOM signal; a cheap base can be substituted where the leak mechanism,
    not the base, is what is under test (TSMOM calls get expensive on long
    prefixes — ~0.11s per call at 340 bars — making long-history checks
    prohibitively slow without changing what they prove)."""
    if base_fn is None:
        base_fn = generate_tsmom_signal

    def fn(symbol: str, prefix: list[Bar]) -> Signal:
        base = base_fn(symbol, list(prefix))
        nxt = len(prefix)
        if (gate is None or nxt > gate) and nxt < len(full_bars):
            if full_bars[nxt].close > prefix[-1].close:
                return replace(base, action="BUY", target_weight=1.0)
            return replace(base, action="HOLD", target_weight=0.0)
        return base
    return fn


def cheap_causal(symbol: str, prefix: list[Bar]) -> Signal:
    return _signal(symbol, prefix, "BUY", 0.5)


def test_peeking_signal_provably_uses_future_data():
    bars = make_bars(200)
    prefix = bars[:125]
    honest_next = make_peeking_signal(bars)("SPY", prefix).target_weight
    crashed = list(bars)
    b = crashed[125]
    crashed[125] = Bar(b.date, b.open, b.high, b.low, prefix[-1].close * 0.5, b.volume)
    crashed_next = make_peeking_signal(crashed)("SPY", prefix).target_weight
    # Same prefix, only the NEXT bar changed -> weight flips. Future data is
    # deciding today's trade. This is a genuine lookahead leak.
    assert honest_next == 1.0 and crashed_next == 0.0


def test_preflight_is_blind_to_closure_peek_documented_limitation():
    bars = make_bars(340)
    peeking = make_peeking_signal(bars)
    # DOCUMENTED LIMITATION: signal_series feeds the function byte-identical
    # prefixes in the full and truncated runs, and the closure supplies the
    # same future bar in both — so the truncation comparison cannot see a
    # leak that reaches outside the bars argument, at ANY window length.
    # (The runner's 300-bar pre-flight window is therefore not the binding
    # constraint for this leak class; the binding constraint is that the
    # production signals must never touch data outside their argument —
    # a code-review property, backstopped by the statefulness check above.)
    # The TSMOM-wrapped instance runs the exhaustive grid at 130 bars (the
    # scale of the honest/stateful controls). The long-history checks wrap
    # a cheap causal base instead: the evasion mechanism — identical
    # prefixes, closure-supplied future — is base-independent, and TSMOM's
    # per-call cost on long prefixes (~0.11s at 340 bars, ~16s for one full
    # series) would make them take minutes without proving anything more.
    assert check_lookahead(signal_series(peeking, "SPY"), bars[:130]).passed

    peeking_cheap = make_peeking_signal(bars, base_fn=cheap_causal)
    assert check_lookahead(signal_series(peeking_cheap, "SPY"), bars, cutoffs=[200, 339]).passed

    # The length-gated variant — causal for its first 300 bars of input
    # (the runner's pre-flight window), leaking only on longer histories —
    # evades for the same reason, and also evades a full-length check.
    gated_cheap = make_peeking_signal(bars, gate=300, base_fn=cheap_causal)
    assert check_lookahead(signal_series(gated_cheap, "SPY"), bars[:300], cutoffs=[299]).passed
    assert check_lookahead(signal_series(gated_cheap, "SPY"), bars, cutoffs=[339]).passed


def test_main_path_does_not_refuse_closure_peek_documented_limitation(tmp_path, monkeypatch):
    bars = make_bars(400)
    peeking = make_peeking_signal(bars, base_fn=cheap_causal)
    ledger_path = tmp_path / "trials.jsonl"
    monkeypatch.setattr(runner, "load_daily_bars", lambda symbol: (bars, META))
    monkeypatch.setattr(runner, "STRATEGIES", {"tsmom": peeking})
    monkeypatch.setattr(sys, "argv", ["run_trend_backtest.py", "--symbols", "SPY",
                                      "--strategies", "tsmom", "--ledger", str(ledger_path)])
    # End-to-end: pre-flight runs, does not refuse, and both evaluations are
    # recorded — the leaked numbers enter the ledger as if they were clean.
    assert runner.main() == 0
    assert TrialLedger(ledger_path).count() == 2
