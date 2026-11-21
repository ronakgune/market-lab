"""Baseline tests for scripts/market_lab_vt_trend_forward.py (Stream 1
vt_trend forward paper tracker): sleeve target construction, fill
accounting, per-bar signal cadence, idempotent re-runs, staleness
refusal. Synthetic fixtures only.
"""
import importlib.util
import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from market_lab.broker import OrderCandidate, Portfolio, Position, load_portfolio
from market_lab.data import Bar

_spec = importlib.util.spec_from_file_location(
    "market_lab_vt_trend_forward",
    Path(__file__).resolve().parents[2] / "scripts" / "market_lab_vt_trend_forward.py",
)
tracker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tracker)


def bars_from_prices(prices, start=date(2024, 1, 1)):
    return [Bar(start + timedelta(days=i), p, p * 1.01, p * 0.99, p, 1_000_000)
            for i, p in enumerate(prices)]


def test_signal_targets_apply_sleeve_split():
    # Smooth uptrend: signal saturates at target_weight 1.0, so the
    # effective book target is exactly the 1/2 sleeve share.
    rising = bars_from_prices([100 + 0.08 * i for i in range(300)])
    signals, targets = tracker.signal_targets({"SPY": rising, "QQQ": rising})
    assert signals["SPY"].action == "BUY"
    assert signals["SPY"].target_weight == pytest.approx(1.0)
    assert targets["SPY"] == pytest.approx(0.5)
    assert targets["QQQ"] == pytest.approx(0.5)

    # Broken trend (close below SMA100): flat, in both representations.
    falling = bars_from_prices([300 - 0.08 * i for i in range(300)])
    signals, targets = tracker.signal_targets({"SPY": falling, "QQQ": falling})
    assert signals["SPY"].target_weight == 0.0
    assert targets["SPY"] == 0.0


def test_targets_to_candidates_buys_sells_and_dust():
    portfolio = Portfolio(cash=25_000.0, positions={})
    prices = {"SPY": 100.0, "QQQ": 50.0}
    candidates = tracker.targets_to_candidates({"SPY": 0.5, "QQQ": 0.5}, portfolio, prices, "2026-01-01")
    by_symbol = {c.symbol: c for c in candidates}
    assert by_symbol["SPY"].side == "BUY"
    assert by_symbol["SPY"].quantity == int(25_000.0 * 0.5 / 100.0)
    assert by_symbol["QQQ"].side == "BUY"
    assert by_symbol["QQQ"].quantity == int(25_000.0 * 0.5 / 50.0)

    # Target 0 for a held name sells the whole position.
    portfolio = Portfolio(cash=0.0, positions={"SPY": Position("SPY", 40, 90.0)})
    candidates = tracker.targets_to_candidates({"SPY": 0.0, "QQQ": 0.0}, portfolio,
                                               {"SPY": 100.0, "QQQ": 50.0}, "2026-01-02")
    assert [(c.symbol, c.side, c.quantity) for c in candidates] == [("SPY", "SELL", 40)]

    # A delta worth less than MIN_TRADE_NOTIONAL is skipped entirely.
    portfolio = Portfolio(cash=0.0, positions={"SPY": Position("SPY", 1000, 10.0)})
    assert tracker.targets_to_candidates({"SPY": 0.9995}, portfolio, {"SPY": 10.0}, "2026-01-03") == []


def test_execute_candidate_accounting(tmp_path):
    state = tmp_path / "state.json"
    ledger = tmp_path / "ledger.jsonl"
    tracker.save_portfolio(Portfolio(cash=10_000.0, positions={}), state)
    buy = OrderCandidate("BUY", "SPY", 10, tracker.STRATEGY, 1.0, "t", "2026-01-01", 100.0)
    decision = tracker.execute_candidate(buy, 100.0, {"SPY": 100.0}, state, ledger, "2026-01-02")
    assert decision.accepted
    assert decision.fill_price == pytest.approx(100.05)  # +5bps
    portfolio = load_portfolio(state)
    assert portfolio.cash == pytest.approx(10_000.0 - 10 * 100.05 - 1.0)
    assert portfolio.positions["SPY"].quantity == 10
    assert portfolio.positions["SPY"].avg_price == pytest.approx(100.05)

    sell_all = OrderCandidate("SELL", "SPY", 10, tracker.STRATEGY, 1.0, "t", "2026-01-02", 100.0)
    decision = tracker.execute_candidate(sell_all, 100.0, {"SPY": 100.0}, state, ledger, "2026-01-03")
    assert decision.accepted
    portfolio = load_portfolio(state)
    assert "SPY" not in portfolio.positions
    assert portfolio.cash == pytest.approx(10_000.0 - 10 * 100.05 - 1.0 + 10 * 99.95 - 1.0)


def test_execute_candidate_rejections(tmp_path):
    state = tmp_path / "state.json"
    ledger = tmp_path / "ledger.jsonl"
    tracker.save_portfolio(Portfolio(cash=500.0, positions={}), state)
    too_big = OrderCandidate("BUY", "SPY", 10, tracker.STRATEGY, 1.0, "t", "2026-01-01", 100.0)
    decision = tracker.execute_candidate(too_big, 100.0, {"SPY": 100.0}, state, ledger, "2026-01-02")
    assert not decision.accepted
    assert "insufficient cash" in decision.reason

    oversell = OrderCandidate("SELL", "SPY", 5, tracker.STRATEGY, 1.0, "t", "2026-01-01", 100.0)
    decision = tracker.execute_candidate(oversell, 100.0, {"SPY": 100.0}, state, ledger, "2026-01-02")
    assert not decision.accepted
    assert "insufficient shares" in decision.reason

    nan_buy = OrderCandidate("BUY", "SPY", 5, tracker.STRATEGY, 1.0, "t", "2026-01-01", 100.0)
    decision = tracker.execute_candidate(nan_buy, float("nan"), {"SPY": 100.0}, state, ledger, "2026-01-02")
    assert not decision.accepted
    # Rejections never touch the saved portfolio.
    portfolio = load_portfolio(state)
    assert portfolio.cash == 500.0
    assert portfolio.positions == {}


def _patched_loader(holder):
    def load(symbol):
        n = holder["n"]
        if symbol == "SPY":
            prices = [100 + 0.08 * i for i in range(n)]
        else:  # QQQ: same trend with +-2% alternation -> real vol, partial exposure
            prices = [(100 + 0.08 * i) * (1.02 if i % 2 else 0.98) for i in range(n)]
        return bars_from_prices(prices), {"source": "fixture"}

    return load


def _paths(tmp_path):
    return dict(
        state_path=tmp_path / "state.json",
        ledger_path=tmp_path / "ledger.jsonl",
        candidates_path=tmp_path / "candidates.jsonl",
        equity_path=tmp_path / "equity.jsonl",
        signal_log_path=tmp_path / "signals.jsonl",
        report_dir=tmp_path / "reports",
    )


def test_run_end_to_end_idempotent_then_fills_next_bar(tmp_path, monkeypatch):
    holder = {"n": 600}
    monkeypatch.setattr(tracker, "load_daily_bars", _patched_loader(holder))
    paths = _paths(tmp_path)
    last_day = date(2024, 1, 1) + timedelta(days=599)

    tracker.run(**paths, today=last_day + timedelta(days=1))
    candidates = tracker.load_order_candidates(path=paths["candidates_path"])
    assert {c.symbol for c in candidates} == {"SPY", "QQQ"}, "first run must queue both sleeves"
    assert len(paths["equity_path"].read_text().strip().splitlines()) == 1
    assert len(paths["signal_log_path"].read_text().strip().splitlines()) == 1
    outcome = json.loads((paths["report_dir"] / "last_run.json").read_text())
    assert outcome["candidates_queued"] == 2
    assert outcome["fills_accepted"] == 0

    # Same-day re-run: no fill (fills need a LATER bar), no double mark,
    # no re-decision (signal log unchanged), no duplicate candidates.
    tracker.run(**paths, today=last_day + timedelta(days=1))
    assert tracker.load_order_candidates(path=paths["candidates_path"]) == candidates
    assert len(paths["equity_path"].read_text().strip().splitlines()) == 1
    assert len(paths["signal_log_path"].read_text().strip().splitlines()) == 1
    assert not paths["ledger_path"].exists() or paths["ledger_path"].read_text() == ""

    # Next bar arrives: queued candidates fill at that bar's open.
    # (The new bar is also evaluated, so a fresh candidate for the new
    # date may be queued — but nothing from the first date may remain.)
    holder["n"] = 601
    tracker.run(**paths, today=last_day + timedelta(days=2))
    new_day = (last_day + timedelta(days=1)).isoformat()
    leftover = tracker.load_order_candidates(path=paths["candidates_path"])
    assert all(c.signal_date == new_day for c in leftover)
    decisions = [json.loads(line) for line in paths["ledger_path"].read_text().splitlines()]
    assert len(decisions) == 2 and all(d["accepted"] for d in decisions)
    assert all(d["execution_date"] == (last_day + timedelta(days=1)).isoformat() for d in decisions)
    portfolio = load_portfolio(paths["state_path"])
    assert set(portfolio.positions) == {"SPY", "QQQ"}
    assert portfolio.cash < tracker.VT_TREND_FORWARD_STARTING_CASH
    assert len(paths["signal_log_path"].read_text().strip().splitlines()) == 2


def test_run_refuses_stale_cache(tmp_path, monkeypatch):
    holder = {"n": 600}
    monkeypatch.setattr(tracker, "load_daily_bars", _patched_loader(holder))
    last_day = date(2024, 1, 1) + timedelta(days=599)
    with pytest.raises(RuntimeError, match="stale"):
        tracker.run(**_paths(tmp_path), today=last_day + timedelta(days=30))
