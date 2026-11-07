"""Baseline tests for market_lab/catalyst_study.py (Stream 3 falsification):
the +0.5%/-0.5% race semantics, gap-through, deadline, costs, aggregation,
and the pre-registered kill rule.
"""
from datetime import date, datetime, time, timedelta, timezone

import pytest

from market_lab.catalyst_study import (
    COST_R,
    aggregate,
    find_gap_events,
    kill_rule_verdict,
    simulate_trade,
)
from market_lab.data import Bar
from market_lab.intraday.bars import MinuteBar

OPEN_ET = datetime(2026, 2, 2, 9, 30, tzinfo=timezone.utc) - timedelta(hours=5)  # 9:30 ET (EST)


def _bar(minute_offset: int, o: float, h: float, l: float, c: float) -> MinuteBar:
    ts = (datetime(2026, 2, 2, 14, 30, tzinfo=timezone.utc) + timedelta(minutes=minute_offset))
    return MinuteBar(symbol="XYZ", timestamp=ts, open=o, high=h, low=l, close=c,
                     volume=1000, source="test")


def _flat(minutes: int, price: float = 100.0) -> list[MinuteBar]:
    return [_bar(i, price, price + 0.01, price - 0.01, price) for i in range(minutes)]


def test_target_first_wins_one_r_gross():
    bars = _flat(10)
    bars[3] = _bar(3, 100.0, 100.6, 99.99, 100.4)  # high crosses +0.5% target, low stays above stop
    trade = simulate_trade(bars, +1)
    assert trade.exit_reason == "target"
    assert trade.gross_r == pytest.approx(1.0)
    assert trade.net_r == pytest.approx(1.0 - COST_R)


def test_stop_checked_before_target_within_a_bar():
    bars = _flat(10)
    bars[2] = _bar(2, 100.0, 100.6, 99.4, 100.0)  # one bar spans BOTH: stop wins (conservative)
    trade = simulate_trade(bars, +1)
    assert trade.exit_reason == "stop"
    assert trade.gross_r == pytest.approx(-1.0)


def test_gap_through_stop_exits_at_the_open_worse_than_one_r():
    bars = _flat(10)
    bars[4] = _bar(4, 99.2, 99.3, 99.1, 99.2)  # opens 0.8% below entry: beyond the -0.5% stop
    trade = simulate_trade(bars, +1)
    assert trade.exit_reason == "gap_through"
    assert trade.gross_r == pytest.approx(-1.6)


def test_deadline_exit_at_last_bar_close():
    bars = _flat(151)
    bars[150] = _bar(150, 100.0, 100.05, 99.95, 100.2)
    trade = simulate_trade(bars, +1)
    assert trade.exit_reason == "deadline"
    assert abs(trade.gross_r - 0.4) < 1e-9


def test_short_direction_mirrors():
    bars = _flat(10)
    bars[3] = _bar(3, 100.0, 100.01, 99.4, 99.6)  # down 0.6%: short target hit
    trade = simulate_trade(bars, -1)
    assert trade.exit_reason == "target" and trade.gross_r == pytest.approx(1.0)
    bars2 = _flat(10)
    bars2[3] = _bar(3, 100.0, 100.6, 99.99, 100.2)  # up: short stop hit
    trade2 = simulate_trade(bars2, -1)
    assert trade2.exit_reason == "stop" and trade2.gross_r == pytest.approx(-1.0)


def test_too_few_bars_returns_none():
    assert simulate_trade(_flat(3), +1) is None
    assert simulate_trade([], +1) is None


def test_find_gap_events_threshold_and_direction():
    days = [date(2026, 1, 5) + timedelta(days=i) for i in range(4)]
    bars = [
        Bar(date=days[0], open=100, high=100, low=100, close=100, volume=1),
        Bar(date=days[1], open=106, high=106, low=106, close=106, volume=1),  # +6%
        Bar(date=days[2], open=102, high=102, low=102, close=102, volume=1),  # -3.8% no
        Bar(date=days[3], open=96, high=96, low=96, close=96, volume=1),      # -5.9%
    ]
    events = find_gap_events({"XYZ": bars}, start_year=2026)
    assert [(e.direction, e.day) for e in events] == [(1, days[1]), (-1, days[3])]


def test_aggregate_and_kill_rule():
    from market_lab.catalyst_study import TradeResult
    d1, d2 = date(2026, 1, 5), date(2026, 1, 6)
    trades = [
        TradeResult("A", d1, 1, 1.0, 0.72, "target", 0.005),
        TradeResult("A", d1, 1, -1.0, -1.28, "stop", -0.005),
        TradeResult("A", d2, 1, 1.0, 0.72, "target", 0.005),
    ]
    stats = aggregate(trades)
    assert stats.trades == 3 and stats.win_rate == 2 / 3
    assert abs(stats.profit_factor - (0.72 * 2) / 1.28) < 1e-9
    survives, reason = kill_rule_verdict(stats)
    assert survives is False and "Sharpe" in reason  # tiny sample cannot clear Sharpe 1.0

    from market_lab.catalyst_study import StudyStats
    strong = StudyStats(trades=400, win_rate=0.6, profit_factor=1.5, mean_net_r=0.2,
                        sharpe_daily_r=1.4, worst_trade_frac=-0.011, total_net_r=80.0)
    survives, _ = kill_rule_verdict(strong)
    assert survives is True
    blowup = StudyStats(trades=400, win_rate=0.6, profit_factor=1.5, mean_net_r=0.2,
                        sharpe_daily_r=1.4, worst_trade_frac=-0.062, total_net_r=80.0)
    survives, reason = kill_rule_verdict(blowup)
    assert survives is False and "-5%" in reason
