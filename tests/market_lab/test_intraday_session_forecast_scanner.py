"""Tests for the day-trader-integration port: session resampling, the
Time Machine forecast, and the scanner (universe / daily setups / ORB).

All offline, all synthetic: sessions are built bar-by-bar from a seeded
generator so expectations are deterministic. See also the independent
adversarial suite in test_intraday_forecast_scanner_adversarial.py.
"""
from __future__ import annotations

import math
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from market_lab.intraday.bars import MinuteBar
from market_lab.intraday.forecast import (
    TradePlan, history_replay, simulate_forecast, verdict_for,
)
from market_lab.intraday.scanner import (
    build_universe, classify_symbol_row, detect_orb, find_daily_setups,
    orb_candidate, rsi_wilder, scan_daily,
)
from market_lab.intraday.session import (
    SessionBar, full_sessions, group_sessions, is_full_session,
    resample_to_session_bars, slots_per_session,
)

ET = ZoneInfo("America/New_York")
DAY = date(2026, 2, 3)  # a Tuesday


def _minute_bars(day: date, base: float = 100.0, drift: float = 0.0,
                 minutes: int = 390) -> list[MinuteBar]:
    out = []
    start = datetime.combine(day, time(9, 30), tzinfo=ET)
    price = base
    for i in range(minutes):
        ts = (start + timedelta(minutes=i)).astimezone(timezone.utc)
        price += drift
        out.append(MinuteBar("ACME", ts, price, price + 0.05, price - 0.05,
                             price + 0.01, 1000, "synthetic"))
    return out


def _session(day: date, closes: list[float]) -> list[SessionBar]:
    start = datetime.combine(day, time(9, 30), tzinfo=ET)
    return [SessionBar("ACME", day, i, start + timedelta(minutes=5 * i),
                       c, c + 0.1, c - 0.1, c, 5000, 5)
            for i, c in enumerate(closes)]


def _flat_sessions(n: int, start_day: date = date(2026, 1, 5)) -> dict:
    sessions = {}
    day = start_day
    made = 0
    while made < n:
        if day.weekday() < 5:
            sessions[day] = _session(day, [100.0 + 0.01 * i for i in range(78)])
            made += 1
        day += timedelta(days=1)
    return sessions


# ---------------------------------------------------------------- session

def test_slots_per_session():
    assert slots_per_session(5) == 78
    assert slots_per_session(1) == 390
    with pytest.raises(ValueError):
        slots_per_session(7)


def test_resample_full_day():
    bars = resample_to_session_bars(_minute_bars(DAY))
    assert len(bars) == 78
    assert is_full_session(bars)
    assert bars[0].slot == 0 and bars[-1].slot == 77
    assert bars[0].volume == 5000  # 5 minutes x 1000


def test_resample_drops_premarket_and_partial_buckets():
    bars = _minute_bars(DAY, minutes=390)
    pre = MinuteBar("ACME", datetime.combine(DAY, time(9, 0), tzinfo=ET).astimezone(timezone.utc),
                    99, 99.1, 98.9, 99, 10, "synthetic")
    out = resample_to_session_bars([pre] + bars[:-2])  # last bucket short 2 minutes
    assert len(out) == 77
    assert not is_full_session(out)


def test_resample_mixed_symbols_rejected():
    a = _minute_bars(DAY, minutes=5)
    b = [MinuteBar("OTHER", a[0].timestamp, 1, 1.1, 0.9, 1, 1, "synthetic")]
    with pytest.raises(ValueError):
        resample_to_session_bars(a + b)


def test_full_sessions_excludes_trade_day_and_after():
    sessions = _flat_sessions(25)
    days = sorted(sessions)
    cut = days[20]
    kept = full_sessions(sessions, before=cut)
    assert max(kept) < cut and len(kept) == 20


# ---------------------------------------------------------------- forecast

def _plan(**over) -> TradePlan:
    fields = dict(symbol="ACME", trade_date=date(2026, 2, 20), entry=101.0,
                  stop=100.0, target=102.0, entry_slot=10)
    fields.update(over)
    return TradePlan(**fields)


def test_plan_validation():
    with pytest.raises(ValueError):
        _plan(stop=102.0)  # stop above entry for a long
    with pytest.raises(ValueError):
        _plan(entry_slot=78)
    short = TradePlan("ACME", date(2026, 2, 20), 100.0, 101.0, 99.0, 10, side="short")
    assert short.risk == 1.0 and short.target_r == 1.0


def test_forecast_reproducible_and_probabilities_sum():
    sessions = _flat_sessions(30)
    r1 = simulate_forecast(_plan(), sessions, replays=500, seed=7)
    r2 = simulate_forecast(_plan(), sessions, replays=500, seed=7)
    assert r1 == r2
    assert math.isclose(r1.p_target + r1.p_stop + r1.p_neither, 1.0)
    assert r1.pool_sessions >= 20
    assert 0.0 <= r1.p_green <= 1.0


def test_forecast_refuses_pool_containing_trade_day():
    sessions = _flat_sessions(30)
    sessions[date(2026, 2, 20)] = _session(date(2026, 2, 20), [100.0] * 78)
    # full_sessions guard strips the trade day; pool still valid, but a
    # pool that only has the trade day must refuse, not leak.
    only_today = {date(2026, 2, 20): sessions[date(2026, 2, 20)]}
    with pytest.raises(ValueError):
        simulate_forecast(_plan(), only_today, replays=10)


def test_forecast_refuses_too_few_sessions():
    with pytest.raises(ValueError):
        simulate_forecast(_plan(), _flat_sessions(5), replays=10)


def test_forecast_costs_reduce_avg_r():
    sessions = _flat_sessions(30)
    free = simulate_forecast(_plan(), sessions, replays=300, seed=3, cost_bps=0)
    costly = simulate_forecast(_plan(), sessions, replays=300, seed=3, cost_bps=50)
    assert costly.avg_r < free.avg_r


def test_verdict_thresholds():
    assert verdict_for(0.30, 0.30, 1.0) == "Close to a coin flip"
    assert verdict_for(0.20, 0.40, 2.0) == "Close to break-even"  # share 1/3 == breakeven at 2R
    assert "target" in verdict_for(0.60, 0.10, 1.0)


def test_history_replay_counts():
    # Every session ramps up: ORB should trigger and hit target often.
    sessions = {}
    day = date(2026, 1, 5)
    made = 0
    while made < 10:
        if day.weekday() < 5:
            closes = [100.0] * 3 + [100.0 + 0.3 * (i - 2) for i in range(3, 78)]
            sessions[day] = _session(day, closes)
            made += 1
        day += timedelta(days=1)
    h = history_replay(sessions)
    assert h.sessions == 10 and h.triggered == 10
    assert h.target + h.stop + h.close == h.triggered


# ---------------------------------------------------------------- scanner

_NASDAQ = """Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares
AAPL|Apple Inc. - Common Stock|Q|N|N|100|N|N
ZZTEST|NASDAQ TEST STOCK|Q|Y|N|100|N|N
SPYY|Some ETF Trust|Q|N|N|100|Y|N
BRK/B|Berkshire Hathaway Inc.|Q|N|N|100|N|N
ACMEW|Acme Corp - Warrants|Q|N|N|100|N|N
File Creation Time: 0203202609:30|||||||
"""
_OTHER = """ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol
XYZ|XYZ Corp - Notes due 2030|N|XYZ|N|100|N|XYZ
PLAIN|Plain Holdings Inc.|N|PLAIN|N|100|N|PLAIN
File Creation Time: 0203202609:30|||||||
"""


def test_universe_build():
    kept, counts, created = build_universe(
        {"nasdaqlisted.txt": _NASDAQ, "otherlisted.txt": _OTHER})
    syms = [e.symbol for e in kept]
    assert "AAPL" in syms and "PLAIN" in syms
    assert "ZZTEST" not in syms and "SPYY" not in syms
    assert "ACMEW" not in syms and "XYZ" not in syms  # warrants / notes out
    assert counts["rows"] == 7
    assert created == "2026-02-03 09:30"


def test_classify_preferred_out():
    row = {"Symbol": "PREF", "Security Name": "Bank - Preferred Shares",
           "Test Issue": "N", "ETF": "N", "NextShares": "N"}
    assert classify_symbol_row(row) == "preferred"


def test_rsi_extremes():
    down = [100.0 - i for i in range(30)]
    assert rsi_wilder(down) == pytest.approx(0.0)
    up = [100.0 + i for i in range(30)]
    assert rsi_wilder(up) == pytest.approx(100.0)
    assert rsi_wilder([1.0] * 10) is None


def test_find_daily_setups_breakout():
    closes = [100.0] * 25 + [101.0, 102.0, 103.0, 104.0, 110.0]
    tags, nums = find_daily_setups(closes)
    assert "breakout_20d" in tags
    with pytest.raises(ValueError):
        find_daily_setups([100.0] * 10)


def test_scan_daily_filters_price_and_length():
    good = {"GOOD": [100.0] * 25 + [101.0, 102.0, 103.0, 104.0, 110.0],
            "CHEAP": [2.0] * 30, "SHORT": [100.0] * 5}
    hits = scan_daily(good)
    assert [h.symbol for h in hits] == ["GOOD"]


def test_orb_detect_and_candidate():
    closes = [100.0, 100.2, 100.1] + [100.3] * 10 + [101.5] + [101.4] * 5
    bars = _session(DAY, closes)
    hit = detect_orb(bars)
    assert hit is not None and hit.entry_slot == 13
    assert hit.or_high == pytest.approx(100.3)
    assert orb_candidate(bars)
    flat = _session(DAY, [100.0] * 20)
    # Stage 1 is a *necessary* condition only (>= highest range close), so a
    # perfectly flat tape passes stage 1 and must be rejected by stage 2 —
    # that separation is the whole point of the two-stage design.
    assert orb_candidate(flat)
    assert detect_orb(flat) is None
