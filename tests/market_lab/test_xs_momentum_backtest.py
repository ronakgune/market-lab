"""Baseline tests for the Stream 2 fixes to run_dual_momentum_backtest:
real costs on by default, SPY as the benchmark, and vol scaling.

Fixtures are deterministic synthetic series (unit tests of the engine's
mechanics — never strategy evidence).
"""
from datetime import date, timedelta

import pytest

from market_lab.backtest import ExecutionModel
from market_lab.data import Bar
from market_lab.portfolio_construction import run_dual_momentum_backtest

FREE = ExecutionModel(0.0, 0.0)


def bars_from_prices(prices, start=date(2024, 1, 1)):
    return [
        Bar(start + timedelta(days=i), p, p * 1.01, p * 0.99, p, 1_000_000)
        for i, p in enumerate(prices)
    ]


def universe(n=400):
    return {
        # Alphabetically first on purpose: the old code benchmarked to it.
        "AAA": bars_from_prices([100 + 0.05 * i for i in range(n)]),
        "LEADER": bars_from_prices([100 + 0.60 * i for i in range(n)]),
        "MID": bars_from_prices([100 + 0.30 * i for i in range(n)]),
    }


def spy(n=400):
    return bars_from_prices([100 + 0.10 * i for i in range(n)])


KW = dict(formation_days=60, skip_days=10, top_n=2, rebalance_every=21,
          start_index=70, initial_cash=25_000.0, max_weight=1.0)


def test_benchmark_is_spy_not_alphabetically_first():
    bars = universe()
    result = run_dual_momentum_backtest(bars, spy_bars=spy(), execution=FREE, **KW)
    assert result.benchmark_symbol == "SPY"
    start, end = 70, 399
    expected = spy()[end].close / spy()[start].close - 1
    assert result.benchmark_return == pytest.approx(expected)
    aaa_expected = bars["AAA"][end].close / bars["AAA"][start].close - 1
    assert result.benchmark_return != pytest.approx(aaa_expected)


def test_costs_on_by_default_and_reduce_equity():
    bars = universe()
    free = run_dual_momentum_backtest(bars, spy_bars=spy(), execution=FREE, **KW)
    costed = run_dual_momentum_backtest(bars, spy_bars=spy(), **KW)  # default execution
    assert costed.costs_paid > 0
    assert free.costs_paid == 0.0
    assert costed.final_equity < free.final_equity


def test_vol_scaling_shrinks_weights_after_vol_spike():
    # Calm grind, then a violent chop: realized vol explodes, so later
    # snapshots must carry a vol_scale below 1 when target_vol is set.
    calm = [100 + 0.2 * i for i in range(200)]
    chop = [140 + (18 if i % 2 else -18) for i in range(200)]
    bars = {
        "AAA": bars_from_prices([100 + 0.05 * i for i in range(400)]),
        "WILD": bars_from_prices(calm + chop),
        "MID": bars_from_prices([100 + 0.25 * i for i in range(400)]),
    }
    kw = dict(KW, top_n=1)
    scaled = run_dual_momentum_backtest(bars, spy_bars=spy(), execution=FREE,
                                        target_vol=0.10, **kw)
    scales = [s.vol_scale for s in scaled.snapshots]
    assert scales[0] == 1.0  # no return history yet
    assert min(scales) < 0.5  # the chop regime forces exposure down hard
    unscaled = run_dual_momentum_backtest(bars, spy_bars=spy(), execution=FREE, **kw)
    assert all(s.vol_scale == 1.0 for s in unscaled.snapshots)


def test_misaligned_spy_is_refused():
    bars = universe()
    short_spy = bars_from_prices([100 + 0.1 * i for i in range(200)],
                                 start=date(2025, 6, 1))
    with pytest.raises(ValueError, match="do not cover"):
        run_dual_momentum_backtest(bars, spy_bars=short_spy, execution=FREE, **KW)


def test_empty_selection_liquidates_to_cash():
    # Everything trends down: the absolute gate selects nothing, so the
    # portfolio must sit in cash and pay only the initial (zero) trades.
    bars = {
        "AAA": bars_from_prices([300 - 0.4 * i for i in range(400)]),
        "BBB": bars_from_prices([300 - 0.3 * i for i in range(400)]),
    }
    result = run_dual_momentum_backtest(bars, spy_bars=spy(), execution=FREE, **KW)
    assert result.rebalances > 0
    assert all(not s.targets for s in result.snapshots)
    assert result.final_equity == pytest.approx(25_000.0)
