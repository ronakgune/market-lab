"""Baseline tests for scripts/market_lab_xs_momentum.py (Stream 2 forward
paper tracker): candidate generation, fill accounting, rebalance cadence,
idempotent re-runs, staleness refusal. Synthetic fixtures only.
"""
import importlib.util
import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from market_lab.broker import OrderCandidate, Portfolio, Position, load_portfolio
from market_lab.data import Bar

_spec = importlib.util.spec_from_file_location(
    "market_lab_xs_momentum",
    Path(__file__).resolve().parents[2] / "scripts" / "market_lab_xs_momentum.py",
)
tracker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tracker)


def bars_from_prices(prices, start=date(2024, 1, 1)):
    return [Bar(start + timedelta(days=i), p, p * 1.01, p * 0.99, p, 1_000_000)
            for i, p in enumerate(prices)]


class FakeTarget:
    def __init__(self, symbol, weight):
        self.symbol = symbol
        self.target_weight = weight


def test_targets_to_candidates_buys_sells_and_dropped_names():
    portfolio = Portfolio(cash=10_000.0, positions={
        "AAA": Position("AAA", 10, 100.0),   # held, still targeted
        "OLD": Position("OLD", 5, 100.0),    # held, dropped -> sell all
    })
    prices = {"AAA": 100.0, "BBB": 50.0, "OLD": 100.0}
    targets = [FakeTarget("AAA", 0.5), FakeTarget("BBB", 0.5)]
    candidates = tracker.targets_to_candidates(targets, 1.0, portfolio, prices, "2026-01-01")
    by_symbol = {c.symbol: c for c in candidates}
    equity = 10_000.0 + 10 * 100.0 + 5 * 100.0
    assert by_symbol["AAA"].side == "BUY"
    assert by_symbol["AAA"].quantity == int(equity * 0.5 / 100.0) - 10
    assert by_symbol["BBB"].side == "BUY"
    assert by_symbol["BBB"].quantity == int(equity * 0.5 / 50.0)
    assert by_symbol["OLD"].side == "SELL"
    assert by_symbol["OLD"].quantity == 5


def test_targets_to_candidates_respects_vol_scale_and_dust():
    portfolio = Portfolio(cash=25_000.0, positions={})
    prices = {"AAA": 100.0}
    candidates = tracker.targets_to_candidates([FakeTarget("AAA", 1.0)], 0.5, portfolio, prices, "2026-01-01")
    assert candidates[0].quantity == int(25_000.0 * 0.5 / 100.0)
    # A delta worth less than MIN_TRADE_NOTIONAL is skipped entirely:
    # 1 share of a $10 stock against a $10,000 book.
    portfolio2 = Portfolio(cash=0.0, positions={"AAA": Position("AAA", 1000, 10.0)})
    assert tracker.targets_to_candidates([FakeTarget("AAA", 0.9995)], 1.0, portfolio2,
                                         {"AAA": 10.0}, "2026-01-01") == []


def test_execute_candidate_accounting(tmp_path):
    state = tmp_path / "state.json"
    ledger = tmp_path / "ledger.jsonl"
    tracker.save_portfolio(Portfolio(cash=10_000.0, positions={}), state)
    buy = OrderCandidate("BUY", "AAA", 10, tracker.STRATEGY, 1.0, "t", "2026-01-01", 100.0)
    decision = tracker.execute_candidate(buy, 100.0, {"AAA": 100.0}, state, ledger, "2026-01-02")
    assert decision.accepted
    assert decision.fill_price == pytest.approx(100.05)  # +5bps
    portfolio = load_portfolio(state)
    assert portfolio.cash == pytest.approx(10_000.0 - 10 * 100.05 - 1.0)
    assert portfolio.positions["AAA"].quantity == 10
    assert portfolio.positions["AAA"].avg_price == pytest.approx(100.05)

    sell_all = OrderCandidate("SELL", "AAA", 10, tracker.STRATEGY, 1.0, "t", "2026-01-02", 100.0)
    decision = tracker.execute_candidate(sell_all, 100.0, {"AAA": 100.0}, state, ledger, "2026-01-03")
    assert decision.accepted
    portfolio = load_portfolio(state)
    assert "AAA" not in portfolio.positions
    assert portfolio.cash == pytest.approx(10_000.0 - 10 * 100.05 - 1.0 + 10 * 99.95 - 1.0)

    oversell = OrderCandidate("SELL", "AAA", 1, tracker.STRATEGY, 1.0, "t", "2026-01-03", 100.0)
    decision = tracker.execute_candidate(oversell, 100.0, {"AAA": 100.0}, state, ledger, "2026-01-04")
    assert not decision.accepted  # no shorting

    too_big = OrderCandidate("BUY", "AAA", 10_000, tracker.STRATEGY, 1.0, "t", "2026-01-03", 100.0)
    decision = tracker.execute_candidate(too_big, 100.0, {"AAA": 100.0}, state, ledger, "2026-01-04")
    assert not decision.accepted  # insufficient cash
    assert len(ledger.read_text().strip().splitlines()) == 4  # every attempt is on record


def test_rebalance_due_counts_spy_bars():
    spy = bars_from_prices([100.0] * 30)
    assert tracker.rebalance_due(spy, None) is True
    last = spy[9].date.isoformat()  # 20 bars after
    assert tracker.rebalance_due(spy, last) is False
    last = spy[8].date.isoformat()  # 21 bars after
    assert tracker.rebalance_due(spy, last) is True


def _patched_loader(holder):
    slopes = {symbol: 0.10 + 0.05 * i for i, symbol in enumerate(tracker.UNIVERSE)}

    def load(symbol):
        n = holder["n"]
        if symbol == "SPY":
            prices = [100 + 0.08 * i for i in range(n)]
        else:
            prices = [100 + slopes[symbol] * i for i in range(n)]
        return bars_from_prices(prices), {"source": "fixture"}

    return load


def _paths(tmp_path):
    return dict(
        state_path=tmp_path / "state.json",
        ledger_path=tmp_path / "ledger.jsonl",
        candidates_path=tmp_path / "candidates.jsonl",
        equity_path=tmp_path / "equity.jsonl",
        rebalance_log_path=tmp_path / "rebalances.jsonl",
        report_dir=tmp_path / "reports",
    )


def test_run_end_to_end_idempotent_then_fills_next_bar(tmp_path, monkeypatch):
    holder = {"n": 600}
    monkeypatch.setattr(tracker, "load_daily_bars", _patched_loader(holder))
    paths = _paths(tmp_path)
    last_day = date(2024, 1, 1) + timedelta(days=599)

    tracker.run(**paths, today=last_day + timedelta(days=1))
    candidates = tracker.load_order_candidates(path=paths["candidates_path"])
    assert candidates, "first run must queue the initial rebalance"
    equity_rows = paths["equity_path"].read_text().strip().splitlines()
    assert len(equity_rows) == 1

    # Same-day re-run: no fill (fills need a LATER bar), no double mark,
    # no duplicate candidates.
    tracker.run(**paths, today=last_day + timedelta(days=1))
    assert tracker.load_order_candidates(path=paths["candidates_path"]) == candidates
    assert len(paths["equity_path"].read_text().strip().splitlines()) == 1
    assert not paths["ledger_path"].exists() or paths["ledger_path"].read_text() == ""

    # Next bar arrives: queued candidates fill at that bar's open.
    holder["n"] = 601
    tracker.run(**paths, today=last_day + timedelta(days=2))
    assert tracker.load_order_candidates(path=paths["candidates_path"]) == []
    decisions = [json.loads(line) for line in paths["ledger_path"].read_text().splitlines()]
    assert decisions and all(d["accepted"] for d in decisions)
    portfolio = load_portfolio(paths["state_path"])
    assert portfolio.positions, "fills must leave the book invested"
    assert portfolio.cash < tracker.XS_MOMENTUM_STARTING_CASH


def test_run_refuses_stale_cache(tmp_path, monkeypatch):
    holder = {"n": 600}
    monkeypatch.setattr(tracker, "load_daily_bars", _patched_loader(holder))
    last_day = date(2024, 1, 1) + timedelta(days=599)
    with pytest.raises(RuntimeError, match="stale"):
        tracker.run(**_paths(tmp_path), today=last_day + timedelta(days=30))
