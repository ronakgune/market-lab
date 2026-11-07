"""Baseline tests for scripts/run_xs_momentum_backtest.py (Stream 2 runner):
ledger discipline, point-in-time preflight, degenerate-input refusal.
Synthetic fixtures only — mechanics, not strategy evidence.
"""
import importlib.util
from datetime import date, timedelta
from pathlib import Path

import pytest

from market_lab.data import Bar
from market_lab.validation import TrialLedger

_spec = importlib.util.spec_from_file_location(
    "run_xs_momentum_backtest",
    Path(__file__).resolve().parents[2] / "scripts" / "run_xs_momentum_backtest.py",
)
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)


def bars_from_prices(prices, start=date(2024, 1, 1)):
    return [
        Bar(start + timedelta(days=i), p, p * 1.01, p * 0.99, p, 1_000_000)
        for i, p in enumerate(prices)
    ]


def fixture(n=600):
    bars_by_symbol = {
        "AAA": bars_from_prices([100 + 0.30 * i for i in range(n)]),
        "BBB": bars_from_prices([100 + 0.20 * i for i in range(n)]),
        "CCC": bars_from_prices([100 + 0.10 * i for i in range(n)]),
    }
    spy = bars_from_prices([100 + 0.05 * i for i in range(n)])
    return bars_by_symbol, spy


def test_preflight_passes_for_point_in_time_targets():
    bars_by_symbol, spy = fixture()
    assert runner.preflight_point_in_time(bars_by_symbol, spy, 600) == []


def test_run_one_records_exactly_one_trial_matching_the_row(tmp_path):
    bars_by_symbol, spy = fixture()
    ledger = TrialLedger(tmp_path / "trials.jsonl")
    row = runner.run_one(bars_by_symbol, spy, {"source": "fixture"}, 600, 25_000.0, None, ledger)
    assert ledger.count() == 1
    record = ledger.records()[0]
    assert record.stream == "xs_momentum"
    assert record.metrics["total_return"] == row["total_return"]
    assert record.metrics["sharpe"] == row["sharpe"]
    assert record.params["universe"] == runner.UNIVERSE


def test_run_one_refuses_degenerate_inputs(tmp_path):
    bars_by_symbol, spy = fixture(n=100)
    ledger = TrialLedger(tmp_path / "trials.jsonl")
    with pytest.raises(ValueError):
        runner.run_one(bars_by_symbol, spy, {}, 100, 25_000.0, None, ledger)
    bars_by_symbol, spy = fixture()
    with pytest.raises(ValueError):
        runner.run_one(bars_by_symbol, spy, {}, 600, -5.0, None, ledger)
    with pytest.raises(ValueError):
        runner.run_one(bars_by_symbol, spy, {}, 600, 25_000.0, 10, ledger)
    assert ledger.count() == 0  # refusals record nothing
