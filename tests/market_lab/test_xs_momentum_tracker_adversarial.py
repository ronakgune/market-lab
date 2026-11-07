"""Adversarial tests for scripts/market_lab_xs_momentum.py (Stream 2 forward
paper tracker): corrupt auxiliary state, duplicate-candidate double fills,
fill-price integrity, full-cycle accounting identities, cadence integrity,
equity-mark monotonicity, and the execute_candidate gatekeeper.

Synthetic fixtures only — these attack the tracker's state machine and
accounting, never strategy claims.
"""
import importlib.util
import json
import math
from datetime import date, timedelta
from pathlib import Path

import pytest

from market_lab.broker import (
    OrderCandidate,
    Portfolio,
    Position,
    load_order_candidates,
    load_portfolio,
    save_order_candidates,
    save_portfolio,
)
from market_lab.data import Bar

_spec = importlib.util.spec_from_file_location(
    "market_lab_xs_momentum_adv",
    Path(__file__).resolve().parents[2] / "scripts" / "market_lab_xs_momentum.py",
)
tracker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tracker)

START = date(2024, 1, 1)


def bars_from_prices(prices, start=START):
    return [Bar(start + timedelta(days=i), p, p * 1.01, p * 0.99, p, 1_000_000)
            for i, p in enumerate(prices)]


def _paths(tmp_path):
    return dict(
        state_path=tmp_path / "state.json",
        ledger_path=tmp_path / "ledger.jsonl",
        candidates_path=tmp_path / "candidates.jsonl",
        equity_path=tmp_path / "equity.jsonl",
        rebalance_log_path=tmp_path / "rebalances.jsonl",
        report_dir=tmp_path / "reports",
    )


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


def _latest_date(load, n):
    bars, _ = load("SPY")
    return bars[-1].date


def _seed_rebalance_log(path, day):
    path.write_text(json.dumps({"date": day.isoformat(), "targets": {},
                                "vol_scale": 1.0, "candidates_queued": 0}) + "\n")


def _ledger_rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# 1) Torn / corrupt auxiliary state must not kill the run
# ---------------------------------------------------------------------------

def test_equity_history_skips_torn_final_line(tmp_path):
    path = tmp_path / "equity.jsonl"
    path.write_text(
        '{"date": "2026-01-01", "equity": 25000.0}\n'
        '{"date": "2026-01-02", "equity": 25100.0}\n'
        '{"date": "2026-01-03", "equ'  # torn mid-write
    )
    rows = tracker._equity_history(path)
    assert rows == [("2026-01-01", 25000.0), ("2026-01-02", 25100.0)]


def test_equity_history_skips_corrupt_and_nonfinite_rows(tmp_path):
    path = tmp_path / "equity.jsonl"
    path.write_text(
        '{"date": "2026-01-01", "equity": 25000.0}\n'
        'not json at all\n'
        '{"unexpected": "shape"}\n'
        '{"date": "2026-01-02", "equity": "abc"}\n'
        '{"date": "2026-01-03", "equity": NaN}\n'
        '{"date": "2026-01-04", "equity": Infinity}\n'
        '{"date": "2026-01-05", "equity": 25200.0}\n'
    )
    rows = tracker._equity_history(path)
    assert rows == [("2026-01-01", 25000.0), ("2026-01-05", 25200.0)]


def test_last_rebalance_date_skips_corrupt_rows(tmp_path):
    path = tmp_path / "rebalances.jsonl"
    path.write_text(
        '{"date": "2026-01-01"}\n'
        '{"date": "2026-02-01", "targets": {\n'  # torn
        'garbage\n'
    )
    assert tracker._last_rebalance_date(path) == "2026-01-01"
    path.write_text('garbage\n{"nope": 1}\n')
    assert tracker._last_rebalance_date(path) is None


def test_run_survives_corrupt_aux_files(tmp_path, monkeypatch):
    holder = {"n": 600}
    load = _patched_loader(holder)
    monkeypatch.setattr(tracker, "load_daily_bars", load)
    paths = _paths(tmp_path)
    paths["equity_path"].write_text('{"date": "2025-01-01", "equity": 24000.0}\n{"date": "2025-01-02", "eq')
    paths["rebalance_log_path"].write_text('{"date": "2025-01-01"}\nnot json\n')
    last_day = _latest_date(load, 600)
    report = tracker.run(**paths, today=last_day + timedelta(days=1))
    assert "XS Momentum Paper Tracker" in report


# ---------------------------------------------------------------------------
# 2) A candidate fills at most once, ever
# ---------------------------------------------------------------------------

def test_duplicate_candidate_fills_exactly_once_across_runs(tmp_path, monkeypatch):
    holder = {"n": 600}
    load = _patched_loader(holder)
    monkeypatch.setattr(tracker, "load_daily_bars", load)
    paths = _paths(tmp_path)
    save_portfolio(Portfolio(cash=25_000.0, positions={}), paths["state_path"])
    candidate = OrderCandidate("BUY", "AAPL", 10, tracker.STRATEGY, 1.0, "seed", "2025-01-01", 100.0)
    save_order_candidates([candidate, candidate], path=paths["candidates_path"])  # duplicated line
    _seed_rebalance_log(paths["rebalance_log_path"], _latest_date(load, 600))

    tracker.run(**paths, today=_latest_date(load, 600) + timedelta(days=1))
    rows = _ledger_rows(paths["ledger_path"])
    assert len(rows) == 1 and rows[0]["accepted"]
    # The duplicate must be gone — not deferred to a later run.
    assert load_order_candidates(path=paths["candidates_path"]) == []

    holder["n"] = 601
    tracker.run(**paths, today=_latest_date(load, 601) + timedelta(days=1))
    assert len(_ledger_rows(paths["ledger_path"])) == 1  # still exactly one fill, ever
    portfolio = load_portfolio(paths["state_path"])
    assert portfolio.positions["AAPL"].quantity == 10


def test_distinct_candidate_same_symbol_is_deferred_not_dropped(tmp_path, monkeypatch):
    holder = {"n": 600}
    load = _patched_loader(holder)
    monkeypatch.setattr(tracker, "load_daily_bars", load)
    paths = _paths(tmp_path)
    save_portfolio(Portfolio(cash=25_000.0, positions={}), paths["state_path"])
    first = OrderCandidate("BUY", "AAPL", 5, tracker.STRATEGY, 1.0, "seed1", "2025-01-01", 100.0)
    second = OrderCandidate("BUY", "AAPL", 7, tracker.STRATEGY, 1.0, "seed2", "2025-01-02", 100.0)
    save_order_candidates([first, second], path=paths["candidates_path"])
    _seed_rebalance_log(paths["rebalance_log_path"], _latest_date(load, 600))

    tracker.run(**paths, today=_latest_date(load, 600) + timedelta(days=1))
    assert len(_ledger_rows(paths["ledger_path"])) == 1  # one fill per symbol per run
    assert load_order_candidates(path=paths["candidates_path"]) == [second]

    holder["n"] = 601
    tracker.run(**paths, today=_latest_date(load, 601) + timedelta(days=1))
    rows = _ledger_rows(paths["ledger_path"])
    assert len(rows) == 2 and all(r["accepted"] for r in rows)
    assert load_portfolio(paths["state_path"]).positions["AAPL"].quantity == 12


# ---------------------------------------------------------------------------
# 3) Fill-price integrity
# ---------------------------------------------------------------------------

def test_fill_uses_own_symbols_latest_open_not_close(tmp_path, monkeypatch):
    holder = {"n": 601}
    base = _patched_loader(holder)
    special_opens = {"AAPL": 999.0, "AMD": 555.0}

    def load(symbol):
        bars, meta = base(symbol)
        if symbol in special_opens:
            b = bars[-1]
            bars = bars[:-1] + [Bar(b.date, special_opens[symbol], b.high, b.low, b.close, b.volume)]
        return bars, meta

    monkeypatch.setattr(tracker, "load_daily_bars", load)
    paths = _paths(tmp_path)
    save_portfolio(Portfolio(cash=25_000.0, positions={}), paths["state_path"])
    save_order_candidates([
        OrderCandidate("BUY", "AAPL", 5, tracker.STRATEGY, 1.0, "seed", "2025-01-01", 100.0),
        OrderCandidate("BUY", "AMD", 5, tracker.STRATEGY, 1.0, "seed", "2025-01-01", 100.0),
    ], path=paths["candidates_path"])
    _seed_rebalance_log(paths["rebalance_log_path"], _latest_date(load, 601))

    tracker.run(**paths, today=_latest_date(load, 601) + timedelta(days=1))
    rows = {r["symbol"]: r for r in _ledger_rows(paths["ledger_path"])}
    assert rows["AAPL"]["requested_price"] == 999.0
    assert rows["AAPL"]["fill_price"] == pytest.approx(999.0 * 1.0005)
    assert rows["AMD"]["requested_price"] == 555.0
    assert rows["AMD"]["fill_price"] == pytest.approx(555.0 * 1.0005)
    portfolio = load_portfolio(paths["state_path"])
    assert portfolio.positions["AAPL"].avg_price == pytest.approx(999.0 * 1.0005)
    assert portfolio.positions["AMD"].avg_price == pytest.approx(555.0 * 1.0005)


def test_missing_symbol_candidate_stays_queued_and_never_blocks(tmp_path, monkeypatch):
    holder = {"n": 600}
    load = _patched_loader(holder)
    monkeypatch.setattr(tracker, "load_daily_bars", load)
    paths = _paths(tmp_path)
    save_portfolio(Portfolio(cash=25_000.0, positions={}), paths["state_path"])
    ghost = OrderCandidate("BUY", "ZZZ", 5, tracker.STRATEGY, 1.0, "seed", "2025-01-01", 100.0)
    real = OrderCandidate("BUY", "AAPL", 5, tracker.STRATEGY, 1.0, "seed", "2025-01-01", 100.0)
    save_order_candidates([ghost, real], path=paths["candidates_path"])
    _seed_rebalance_log(paths["rebalance_log_path"], _latest_date(load, 600))

    tracker.run(**paths, today=_latest_date(load, 600) + timedelta(days=1))
    rows = _ledger_rows(paths["ledger_path"])
    assert [r["symbol"] for r in rows] == ["AAPL"]  # ghost produced no ledger row
    assert load_order_candidates(path=paths["candidates_path"]) == [ghost]

    holder["n"] = 601
    tracker.run(**paths, today=_latest_date(load, 601) + timedelta(days=1))
    assert len(_ledger_rows(paths["ledger_path"])) == 1  # ghost still never fills
    assert load_order_candidates(path=paths["candidates_path"]) == [ghost]


def test_torn_candidates_file_does_not_crash_or_fill(tmp_path, monkeypatch):
    holder = {"n": 600}
    load = _patched_loader(holder)
    monkeypatch.setattr(tracker, "load_daily_bars", load)
    paths = _paths(tmp_path)
    good = json.dumps({"side": "BUY", "symbol": "AAPL", "quantity": 5,
                       "strategy": tracker.STRATEGY, "confidence": 1.0, "reason": "seed",
                       "signal_date": "2025-01-01", "reference_close": 100.0,
                       "intended_execution": "next_open"})
    paths["candidates_path"].write_text(good + "\n" + '{"side": "BUY", "sym')
    _seed_rebalance_log(paths["rebalance_log_path"], _latest_date(load, 600))
    # The broker loader's whole-file fallback drops the batch rather than
    # crashing; the tracker must survive it and fill nothing.
    report = tracker.run(**paths, today=_latest_date(load, 600) + timedelta(days=1))
    assert "XS Momentum Paper Tracker" in report
    assert _ledger_rows(paths["ledger_path"]) == []


# ---------------------------------------------------------------------------
# 4) Accounting identities across a full queue -> fill cycle (incl. a SELL)
# ---------------------------------------------------------------------------

def test_full_cycle_accounting_identities(tmp_path, monkeypatch):
    holder = {"n": 600}
    load = _patched_loader(holder)
    monkeypatch.setattr(tracker, "load_daily_bars", load)
    paths = _paths(tmp_path)
    # Seed a held name the rebalance will drop (AAPL: slowest riser, not top-3).
    save_portfolio(Portfolio(cash=20_000.0, positions={"AAPL": Position("AAPL", 10, 90.0)}),
                   paths["state_path"])

    tracker.run(**paths, today=_latest_date(load, 600) + timedelta(days=1))
    queued = load_order_candidates(path=paths["candidates_path"])
    assert {c.symbol for c in queued} == {"TSLA", "SMCI", "NVDA", "AAPL"}
    assert next(c for c in queued if c.symbol == "AAPL").side == "SELL"

    holder["n"] = 601
    tracker.run(**paths, today=_latest_date(load, 601) + timedelta(days=1))
    rows = _ledger_rows(paths["ledger_path"])
    assert len(rows) == len(queued) and all(r["accepted"] for r in rows)

    execution = tracker.EXECUTION
    bars = {symbol: load(symbol)[0] for symbol in tracker.UNIVERSE}
    expected_cash = 20_000.0
    net_qty = {"AAPL": 10}
    for row in rows:
        open_px = bars[row["symbol"]][-1].open
        fill = execution.fill_price(row["side"], open_px)
        assert row["requested_price"] == pytest.approx(open_px)
        assert row["fill_price"] == pytest.approx(fill)
        if row["side"] == "BUY":
            expected_cash -= row["quantity"] * fill + execution.commission_per_trade
            net_qty[row["symbol"]] = net_qty.get(row["symbol"], 0) + row["quantity"]
        else:
            expected_cash += row["quantity"] * fill - execution.commission_per_trade
            net_qty[row["symbol"]] -= row["quantity"]

    portfolio = load_portfolio(paths["state_path"])
    assert portfolio.cash == pytest.approx(expected_cash)
    for symbol, qty in net_qty.items():
        if qty == 0:
            assert symbol not in portfolio.positions
        else:
            assert portfolio.positions[symbol].quantity == qty
    # Single-fill names carry the fill as their average price.
    for row in rows:
        if row["side"] == "BUY":
            assert portfolio.positions[row["symbol"]].avg_price == pytest.approx(row["fill_price"])
    # The equity mark for the fill date is exactly cash + qty * close.
    marks = [json.loads(line) for line in paths["equity_path"].read_text().splitlines()]
    last_mark = marks[-1]
    spy_bars = load("SPY")[0]
    assert last_mark["date"] == spy_bars[-1].date.isoformat()
    expected_equity = portfolio.cash + sum(
        pos.quantity * bars[symbol][-1].close for symbol, pos in portfolio.positions.items()
    )
    assert last_mark["equity"] == pytest.approx(expected_equity)


# ---------------------------------------------------------------------------
# 5) Rebalance cadence integrity
# ---------------------------------------------------------------------------

def test_zero_candidate_rebalance_advances_cadence(tmp_path, monkeypatch):
    holder = {"n": 600}

    def declining_load(symbol):
        n = holder["n"]
        if symbol == "SPY":
            prices = [100.0] * n
        else:
            prices = [500.0 * (0.999 ** i) for i in range(n)]
        return bars_from_prices(prices), {"source": "fixture"}

    monkeypatch.setattr(tracker, "load_daily_bars", declining_load)
    paths = _paths(tmp_path)
    last_day = _latest_date(declining_load, 600)

    tracker.run(**paths, today=last_day + timedelta(days=1))
    log_rows = [json.loads(line) for line in paths["rebalance_log_path"].read_text().splitlines()]
    assert len(log_rows) == 1
    assert log_rows[0]["candidates_queued"] == 0 and log_rows[0]["targets"] == {}
    assert load_order_candidates(path=paths["candidates_path"]) == []

    holder["n"] = 601
    report = tracker.run(**paths, today=last_day + timedelta(days=2))
    assert "Rebalance due: no" in report
    log_rows = [json.loads(line) for line in paths["rebalance_log_path"].read_text().splitlines()]
    assert len(log_rows) == 1  # cadence advanced despite zero orders


def test_deleted_rebalance_log_redecides_from_current_book(tmp_path, monkeypatch):
    holder = {"n": 600}
    load = _patched_loader(holder)
    monkeypatch.setattr(tracker, "load_daily_bars", load)
    paths = _paths(tmp_path)

    tracker.run(**paths, today=_latest_date(load, 600) + timedelta(days=1))
    holder["n"] = 601
    tracker.run(**paths, today=_latest_date(load, 601) + timedelta(days=1))
    ledger_after_fills = paths["ledger_path"].read_text()
    book_after_fills = load_portfolio(paths["state_path"])
    assert book_after_fills.positions

    paths["rebalance_log_path"].unlink()  # cadence memory lost
    tracker.run(**paths, today=_latest_date(load, 601) + timedelta(days=1))
    # The re-decision diffs against the CURRENT book (already on target):
    # it must queue nothing and resurrect none of the filled candidates.
    assert load_order_candidates(path=paths["candidates_path"]) == []
    assert paths["rebalance_log_path"].exists()  # cadence re-logged

    holder["n"] = 602
    tracker.run(**paths, today=_latest_date(load, 602) + timedelta(days=1))
    assert paths["ledger_path"].read_text() == ledger_after_fills  # no new fills, ever
    book = load_portfolio(paths["state_path"])
    assert book.cash == book_after_fills.cash
    assert {s: p.quantity for s, p in book.positions.items()} == \
           {s: p.quantity for s, p in book_after_fills.positions.items()}


def test_rebalance_due_counts_trading_bars_not_calendar_days():
    days = []
    d = date(2024, 1, 1)  # a Monday
    while len(days) < 40:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    spy = [Bar(day, 100.0, 101.0, 99.0, 100.0, 1000) for day in days]
    # 20 trading bars after bar 19 (28 calendar days later): NOT due.
    assert tracker.rebalance_due(spy, spy[19].date.isoformat()) is False
    # 21 trading bars after bar 18: due.
    assert tracker.rebalance_due(spy, spy[18].date.isoformat()) is True


# ---------------------------------------------------------------------------
# 6) Equity marks: once per bar, strictly increasing, never NaN
# ---------------------------------------------------------------------------

def test_equity_marks_strictly_increasing_across_reruns(tmp_path, monkeypatch):
    holder = {"n": 600}
    load = _patched_loader(holder)
    monkeypatch.setattr(tracker, "load_daily_bars", load)
    paths = _paths(tmp_path)
    for n in (600, 600, 601, 601, 602):
        holder["n"] = n
        tracker.run(**paths, today=_latest_date(load, n) + timedelta(days=1))
    dates = [json.loads(line)["date"] for line in paths["equity_path"].read_text().splitlines()]
    assert len(dates) == 3
    assert dates == sorted(dates) and len(set(dates)) == 3


def test_out_of_order_equity_mark_never_appended(tmp_path, monkeypatch):
    holder = {"n": 600}
    load = _patched_loader(holder)
    monkeypatch.setattr(tracker, "load_daily_bars", load)
    paths = _paths(tmp_path)
    paths["equity_path"].write_text('{"date": "2999-01-01", "equity": 25000.0}\n')
    tracker.run(**paths, today=_latest_date(load, 600) + timedelta(days=1))
    lines = paths["equity_path"].read_text().strip().splitlines()
    assert len(lines) == 1  # the earlier real date was NOT appended after the future one


def test_nan_close_refused_before_any_mark(tmp_path, monkeypatch):
    holder = {"n": 600}
    base = _patched_loader(holder)

    def load(symbol):
        bars, meta = base(symbol)
        if symbol == "AAPL":
            b = bars[-1]
            bars = bars[:-1] + [Bar(b.date, b.open, b.high, b.low, float("nan"), b.volume)]
        return bars, meta

    monkeypatch.setattr(tracker, "load_daily_bars", load)
    paths = _paths(tmp_path)
    save_portfolio(Portfolio(cash=20_000.0, positions={"AAPL": Position("AAPL", 10, 90.0)}),
                   paths["state_path"])
    with pytest.raises(RuntimeError):
        tracker.run(**paths, today=_latest_date(load, 600) + timedelta(days=1))
    assert not paths["equity_path"].exists() or "NaN" not in paths["equity_path"].read_text()


# ---------------------------------------------------------------------------
# 7) execute_candidate gatekeeper: non-finite / fractional / rejected orders
# ---------------------------------------------------------------------------

def _gatekeeper_setup(tmp_path):
    state = tmp_path / "state.json"
    ledger = tmp_path / "ledger.jsonl"
    save_portfolio(Portfolio(cash=10_000.0, positions={"AAA": Position("AAA", 10, 50.0)}), state)
    return state, ledger, state.read_bytes()


def test_execute_candidate_rejects_nan_open(tmp_path):
    state, ledger, before = _gatekeeper_setup(tmp_path)
    for side in ("BUY", "SELL"):
        candidate = OrderCandidate(side, "AAA", 5, tracker.STRATEGY, 1.0, "t", "2026-01-01", 100.0)
        decision = tracker.execute_candidate(candidate, float("nan"), {"AAA": 100.0},
                                             state, ledger, "2026-01-02")
        assert not decision.accepted
    assert state.read_bytes() == before
    rows = _ledger_rows(ledger)
    assert len(rows) == 2 and all(not r["accepted"] for r in rows)


def test_execute_candidate_rejects_inf_open(tmp_path):
    state, ledger, before = _gatekeeper_setup(tmp_path)
    for side in ("BUY", "SELL"):
        candidate = OrderCandidate(side, "AAA", 5, tracker.STRATEGY, 1.0, "t", "2026-01-01", 100.0)
        decision = tracker.execute_candidate(candidate, float("inf"), {"AAA": 100.0},
                                             state, ledger, "2026-01-02")
        assert not decision.accepted
    assert state.read_bytes() == before
    assert all(not r["accepted"] for r in _ledger_rows(ledger))


def test_execute_candidate_rejects_nonfinite_quantity(tmp_path):
    state, ledger, before = _gatekeeper_setup(tmp_path)
    for bad_qty in (float("nan"), float("inf")):
        for side in ("BUY", "SELL"):
            candidate = OrderCandidate(side, "AAA", bad_qty, tracker.STRATEGY, 1.0, "t", "2026-01-01", 100.0)
            decision = tracker.execute_candidate(candidate, 100.0, {"AAA": 100.0},
                                                 state, ledger, "2026-01-02")
            assert not decision.accepted
    assert state.read_bytes() == before
    assert all(not r["accepted"] for r in _ledger_rows(ledger))


def test_execute_candidate_rejects_fractional_quantity(tmp_path):
    state, ledger, before = _gatekeeper_setup(tmp_path)
    candidate = OrderCandidate("BUY", "AAA", 10.5, tracker.STRATEGY, 1.0, "t", "2026-01-01", 100.0)
    decision = tracker.execute_candidate(candidate, 100.0, {"AAA": 100.0}, state, ledger, "2026-01-02")
    assert not decision.accepted  # whole shares only
    assert state.read_bytes() == before


def test_rejections_leave_state_byte_identical(tmp_path):
    state, ledger, before = _gatekeeper_setup(tmp_path)
    oversell = OrderCandidate("SELL", "AAA", 11, tracker.STRATEGY, 1.0, "t", "2026-01-01", 100.0)
    assert not tracker.execute_candidate(oversell, 100.0, {"AAA": 100.0}, state, ledger, "2026-01-02").accepted
    over_cash = OrderCandidate("BUY", "AAA", 10_000, tracker.STRATEGY, 1.0, "t", "2026-01-01", 100.0)
    assert not tracker.execute_candidate(over_cash, 100.0, {"AAA": 100.0}, state, ledger, "2026-01-02").accepted
    zero_open = OrderCandidate("BUY", "AAA", 5, tracker.STRATEGY, 1.0, "t", "2026-01-01", 100.0)
    assert not tracker.execute_candidate(zero_open, 0.0, {"AAA": 100.0}, state, ledger, "2026-01-02").accepted
    assert state.read_bytes() == before


# ---------------------------------------------------------------------------
# 8) Corrupt marks in equity history must not break vol scaling
# ---------------------------------------------------------------------------

def test_run_with_zero_negative_nan_history_marks(tmp_path, monkeypatch):
    holder = {"n": 600}
    load = _patched_loader(holder)
    monkeypatch.setattr(tracker, "load_daily_bars", load)
    paths = _paths(tmp_path)
    lines = []
    for i in range(25):
        day = (date(2025, 6, 1) + timedelta(days=i)).isoformat()
        equity = 25_000.0 + (i % 5) * 10.0
        if i == 3:
            equity = 0.0
        elif i == 7:
            equity = -100.0
        elif i == 11:
            lines.append(f'{{"date": "{day}", "equity": NaN}}')
            continue
        lines.append(json.dumps({"date": day, "equity": equity}))
    paths["equity_path"].write_text("\n".join(lines) + "\n")

    report = tracker.run(**paths, today=_latest_date(load, 600) + timedelta(days=1))
    assert "XS Momentum Paper Tracker" in report
    log_rows = [json.loads(line) for line in paths["rebalance_log_path"].read_text().splitlines()]
    assert len(log_rows) == 1
    scale = log_rows[0]["vol_scale"]
    assert math.isfinite(scale) and 0.0 <= scale <= 1.0
