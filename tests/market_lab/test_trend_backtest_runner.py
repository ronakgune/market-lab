"""Tests for the trend backtest runner's wiring (offline fixtures only)."""

from __future__ import annotations

import math
from datetime import date, timedelta

from market_lab.data import Bar
from market_lab.validation import TrialLedger
from scripts.run_trend_backtest import cagr, run_one


def make_bars(n: int) -> list[Bar]:
    bars = []
    price = 100.0
    day = date(2018, 1, 2)
    for i in range(n):
        price *= 1.0 + 0.0008 + 0.004 * math.sin(i / 25.0)
        bars.append(Bar(date=day + timedelta(days=i), open=price, high=price * 1.005,
                        low=price * 0.995, close=price, volume=1_000_000))
    return bars


def test_cagr_math():
    assert cagr(0.0, 252) == 0.0
    assert abs(cagr(0.21, 504) - 0.10) < 1e-9  # 21% over 2 years = 10%/yr
    assert cagr(-1.0, 252) == 0.0  # wiped out: guarded, not NaN


def test_run_one_records_a_ledger_trial(tmp_path):
    from market_lab.signals import generate_tsmom_signal

    ledger = TrialLedger(tmp_path / "trials.jsonl")
    bars = make_bars(400)
    row = run_one("SPY", "tsmom", generate_tsmom_signal, bars,
                  {"source": "alpaca:sip", "feed": "sip", "adjustment": "all"},
                  25_000.0, None, ledger)
    assert row["symbol"] == "SPY" and row["eval"] == "full"
    assert ledger.count() == 1
    record = ledger.records()[0]
    assert record.stream == "trend" and record.strategy == "tsmom"
    assert record.dataset["bars"] == 400
    assert record.metrics["trades"] == float(row["trades"])
