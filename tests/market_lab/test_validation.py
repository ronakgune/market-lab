"""Baseline tests for market_lab.validation (hygiene layer).

The fixtures here are the point: an honest indicator must pass, and the two
classic leaks (next-bar peek, centered window) must be caught red-handed.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from market_lab.data import Bar
from market_lab.validation import (
    TrialLedger,
    check_lookahead,
    check_recursive,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    min_track_record_length,
)


def make_bars(n: int, start: float = 100.0) -> list[Bar]:
    bars = []
    price = start
    day = date(2020, 1, 2)
    for i in range(n):
        price *= 1.0 + 0.001 * math.sin(i / 7.0) + (0.002 if i % 11 == 0 else 0.0)
        bars.append(
            Bar(date=day + timedelta(days=i), open=price, high=price * 1.01,
                low=price * 0.99, close=price, volume=1000 + i)
        )
    return bars


# --- series fixtures --------------------------------------------------------

def sma_series(period: int):
    def _fn(bars):
        out = []
        for i in range(len(bars)):
            if i + 1 < period:
                out.append(None)
            else:
                window = [b.close for b in bars[i + 1 - period: i + 1]]
                out.append(sum(window) / period)
        return out
    return _fn


def ema_series(span: int):
    alpha = 2.0 / (span + 1.0)

    def _fn(bars):
        out = []
        value = None
        for b in bars:
            value = b.close if value is None else alpha * b.close + (1 - alpha) * value
            out.append(value)
        return out
    return _fn


def next_bar_peek(bars):
    """LEAK: value at i uses bar i+1's close."""
    out = []
    for i in range(len(bars)):
        out.append(bars[i + 1].close if i + 1 < len(bars) else bars[i].close)
    return out


def centered_ma(period: int):
    """LEAK: centered window reaches into the future."""
    half = period // 2

    def _fn(bars):
        out = []
        for i in range(len(bars)):
            lo = max(0, i - half)
            hi = min(len(bars), i + half + 1)
            window = [b.close for b in bars[lo:hi]]
            out.append(sum(window) / len(window))
        return out
    return _fn


# --- lookahead ---------------------------------------------------------------

def test_honest_sma_passes_lookahead():
    report = check_lookahead(sma_series(10), make_bars(120))
    assert report.passed
    assert report.violations == []
    assert report.cutoffs_checked > 0


def test_honest_ema_passes_lookahead():
    report = check_lookahead(ema_series(20), make_bars(120))
    assert report.passed


def test_next_bar_peek_is_caught():
    report = check_lookahead(next_bar_peek, make_bars(120))
    assert not report.passed
    assert report.violations
    v = report.violations[0]
    # the last index of the truncated run reads the bar just past the cutoff
    assert v.index == v.cutoff - 1


def test_centered_ma_is_caught():
    report = check_lookahead(centered_ma(9), make_bars(120))
    assert not report.passed


def test_wrong_length_series_rejected():
    with pytest.raises(ValueError):
        check_lookahead(lambda bars: [1.0], make_bars(50))


def test_tiny_input_is_trivially_clean():
    report = check_lookahead(sma_series(5), make_bars(1))
    assert report.passed


# --- recursive / warmup -------------------------------------------------------

def test_sma_warmup_equals_period_minus_one():
    bars = make_bars(200)
    report = check_recursive(sma_series(10), bars, tol=1e-9)
    # an SMA forgets everything older than its window: once the cold start is
    # `period - 1` bars behind, values match exactly
    assert report.required_warmup == 9
    assert report.max_deviation > 0  # cold starts really do deviate at first


def test_ema_warmup_is_longer_than_sma_and_finite():
    bars = make_bars(400)
    ema = check_recursive(ema_series(20), bars, tol=1e-4)
    sma = check_recursive(sma_series(20), bars, tol=1e-4)
    assert ema.required_warmup > sma.required_warmup
    assert ema.required_warmup < len(bars) // 2  # it does settle


def test_constant_series_needs_no_warmup():
    report = check_recursive(lambda bars: [1.0 for _ in bars], make_bars(100))
    assert report.required_warmup == 0
    assert report.max_deviation == 0.0


# --- trial ledger --------------------------------------------------------------

def test_ledger_records_counts_and_hashes(tmp_path):
    ledger = TrialLedger(tmp_path / "trials.jsonl")
    r1 = ledger.record(stream="trend", strategy="tsmom",
                       params={"lookback": 60}, dataset={"symbol": "SPY"},
                       metrics={"sharpe": 0.5})
    r2 = ledger.record(stream="trend", strategy="tsmom",
                       params={"lookback": 120}, dataset={"symbol": "SPY"},
                       metrics={"sharpe": 0.4})
    ledger.record(stream="xsmom", strategy="dual_momentum",
                  params={"top_n": 3}, dataset={"universe": "watchlist"},
                  metrics={"sharpe": 0.3})
    assert (r1.trial, r2.trial) == (1, 2)
    assert ledger.count() == 3
    assert ledger.count(stream="trend") == 2
    assert ledger.count(strategy="dual_momentum") == 1
    same = ledger.record(stream="trend", strategy="tsmom",
                         params={"lookback": 60}, dataset={"symbol": "SPY"},
                         metrics={"sharpe": 0.55})
    assert same.params_hash == r1.params_hash  # identical params, identical hash
    assert same.params_hash != r2.params_hash


def test_ledger_survives_reopen(tmp_path):
    path = tmp_path / "trials.jsonl"
    TrialLedger(path).record(stream="s", strategy="x", params={},
                              dataset={}, metrics={"a": 1.0})
    reopened = TrialLedger(path)
    assert reopened.count() == 1
    assert reopened.records()[0].metrics == {"a": 1.0}


def test_ledger_rejects_nonfinite_metrics(tmp_path):
    ledger = TrialLedger(tmp_path / "trials.jsonl")
    with pytest.raises(ValueError):
        ledger.record(stream="s", strategy="x", params={}, dataset={},
                      metrics={"sharpe": float("nan")})
    with pytest.raises(ValueError):
        ledger.record(stream="s", strategy="x", params={}, dataset={},
                      metrics={"sharpe": float("inf")})
    assert ledger.count() == 0  # rejected records never reach the file


def test_ledger_rejects_nan_params(tmp_path):
    ledger = TrialLedger(tmp_path / "trials.jsonl")
    with pytest.raises(ValueError):
        ledger.record(stream="s", strategy="x", params={"x": float("nan")},
                      dataset={}, metrics={"a": 1.0})


# --- deflated sharpe / MinTRL ---------------------------------------------------

def test_expected_max_sharpe_grows_with_trials():
    assert expected_max_sharpe(1) == 0.0
    assert expected_max_sharpe(10) < expected_max_sharpe(100) < expected_max_sharpe(10_000)


def test_deflated_sharpe_punishes_trial_count():
    honest = deflated_sharpe_ratio(0.05, n_trials=1, n_obs=1000)
    mined = deflated_sharpe_ratio(0.05, n_trials=1000, n_obs=1000)
    assert honest > mined
    assert 0.0 <= mined <= 1.0


def test_min_track_record_matches_the_16_year_rule_of_thumb():
    daily_sr_for_annual_0p5 = 0.5 / math.sqrt(252)
    days = min_track_record_length(daily_sr_for_annual_0p5)
    years = days / 252
    # order of magnitude is the point: ~11 years at one-sided 95% with this
    # formula, ~15 at 97.5% — the research report's "~16 years" ballpark for
    # Sharpe 0.5 either way. A decade-plus, not a paper-trading season.
    assert 9.0 < years < 20.0


def test_min_track_record_infinite_when_below_benchmark():
    assert min_track_record_length(0.0) == math.inf
    assert min_track_record_length(0.1, benchmark_sr=0.2) == math.inf
