"""Adversarial tests for market_lab/catalyst_study.py (Stream 3).

Mandate: try to break the MEASUREMENT behind the live falsification
(741 trades; holdout PF 0.30, Sharpe -10.87, FALSIFIED). The study's
doctrine is that every simplification FAVORS the strategy, so a kill is
decisive. These tests pin the session slicing, race semantics, event
integrity, aggregation and kill-rule boundaries — and regress three
holes this pass found and fixed in catalyst_study.py:

1. Favorable gap-through was capped at +1R (and a favorable gap open
   followed by a dip to the stop scored -1R). A resting limit order
   fills at the open print. Fixed: a bar opening beyond the target
   exits at its open. Quantified on the live store before fixing:
   5 of 741 trades change, ALL pre-2023; the holdout is byte-identical
   (PF 0.295, Sharpe -10.87) — the falsification is not an artifact.
2. Cross-day bleed: _session_bars filters by clock time only, so bars
   from the NEXT morning at 9:30-12:00 ET extended a trade overnight.
   Fixed: simulate_trade restricts to the first bar's ET date.
3. find_gap_events paired bars in list order: an unsorted caller got
   silent LOOKAHEAD events (a day paired against a future close).
   Fixed: bars are sorted by date before pairing.

Everything else below pins behavior that was already correct.
"""
from datetime import date, datetime, timedelta, timezone
from statistics import mean, pstdev
from math import sqrt
from zoneinfo import ZoneInfo

import pytest

from market_lab.catalyst_study import (
    COST_FRAC,
    COST_R,
    DEADLINE_MINUTES,
    STOP_FRAC,
    TARGET_FRAC,
    StudyStats,
    TradeResult,
    _session_bars,
    aggregate,
    find_gap_events,
    kill_rule_verdict,
    simulate_trade,
)
from market_lab.data import Bar
from market_lab.intraday.bars import MinuteBar

ET = ZoneInfo("America/New_York")
EST_DAY = date(2026, 2, 2)    # Monday, EST (UTC-5)
EDT_DAY = date(2026, 7, 6)    # Monday, EDT (UTC-4)


def et_bar(day, hh, mm, o, h, l, c, symbol="XYZ"):
    ts = datetime(day.year, day.month, day.day, hh, mm, tzinfo=ET).astimezone(timezone.utc)
    return MinuteBar(symbol, ts, o, h, l, c, 1000, "test")


def utc_bar(day, hh, mm, o, h, l, c, symbol="XYZ"):
    ts = datetime(day.year, day.month, day.day, hh, mm, tzinfo=timezone.utc)
    return MinuteBar(symbol, ts, o, h, l, c, 1000, "test")


def et_flat(day, minutes, price=100.0):
    """`minutes` consecutive 1-minute bars from 9:30 ET, none touching
    the +/-0.5% levels around `price`."""
    start = datetime(day.year, day.month, day.day, 9, 30, tzinfo=ET)
    bars = []
    for i in range(minutes):
        ts = (start + timedelta(minutes=i)).astimezone(timezone.utc)
        bars.append(MinuteBar("XYZ", ts, price, price + 0.01, price - 0.01, price, 1000, "test"))
    return bars


def _at(bars, i, o, h, l, c):
    old = bars[i]
    bars[i] = MinuteBar(old.symbol, old.timestamp, o, h, l, c, old.volume, old.source)
    return bars


# ---------------------------------------------------------------------------
# 1. Session slicing
# ---------------------------------------------------------------------------

def test_premarket_bars_never_trade_and_entry_is_the_0930_open():
    bars = [
        et_bar(EST_DAY, 4, 0, 80.0, 80.1, 50.0, 80.0),   # premarket: would smash any stop
        et_bar(EST_DAY, 8, 59, 90.0, 90.1, 89.9, 90.0),  # premarket: wrong entry if leaked
    ]
    session = et_flat(EST_DAY, 10)
    _at(session, 3, 100.0, 100.6, 99.99, 100.4)  # target touch, priced off a 100.0 entry
    trade = simulate_trade(bars + session, +1)
    assert trade.exit_reason == "target"
    assert trade.gross_r == pytest.approx(1.0)  # entry was 100.0 (9:30), not 90 or 80
    assert trade.day == EST_DAY


def test_bars_after_1200_cannot_rescue_a_losing_trade():
    bars = et_flat(EST_DAY, 151)
    _at(bars, 150, 100.0, 100.05, 99.95, 99.7)  # 12:00 bar closes -0.3%
    bars.append(et_bar(EST_DAY, 12, 1, 100.0, 101.0, 99.9, 100.9))   # target bait
    bars.append(et_bar(EST_DAY, 12, 30, 100.0, 101.0, 99.9, 100.9))  # target bait
    trade = simulate_trade(bars, +1)
    assert trade.exit_reason == "deadline"
    assert trade.gross_r == pytest.approx(-0.6)


def test_1200_bar_is_the_deadline_bar_and_1201_is_out():
    # (a) deadline exit reads the 12:00 bar's close
    bars = et_flat(EST_DAY, 151)
    _at(bars, 150, 100.0, 100.05, 99.95, 100.3)
    trade = simulate_trade(bars, +1)
    assert trade.exit_reason == "deadline" and trade.gross_r == pytest.approx(0.6)
    # (b) a target touch INSIDE the 12:00 bar still counts
    bars = et_flat(EST_DAY, 151)
    _at(bars, 150, 100.0, 100.6, 99.99, 100.4)
    trade = simulate_trade(bars, +1)
    assert trade.exit_reason == "target" and trade.gross_r == pytest.approx(1.0)
    # (c) the same touch at 12:01 does not count
    bars = et_flat(EST_DAY, 151)
    bars.append(et_bar(EST_DAY, 12, 1, 100.0, 100.6, 99.99, 100.4))
    trade = simulate_trade(bars, +1)
    assert trade.exit_reason == "deadline" and trade.gross_r == pytest.approx(0.0)


def test_session_slice_est_vs_edt_utc_grid():
    grid = [(12, 30), (13, 30), (14, 30), (16, 30), (17, 0), (17, 1)]
    est = _session_bars([utc_bar(EST_DAY, h, m, 100, 100, 100, 100) for h, m in grid])
    # EST = UTC-5: 9:30 ET = 14:30 UTC, 12:00 ET = 17:00 UTC (inclusive)
    assert [(b.timestamp.hour, b.timestamp.minute) for b in est] == [(14, 30), (16, 30), (17, 0)]
    edt = _session_bars([utc_bar(EDT_DAY, h, m, 100, 100, 100, 100) for h, m in grid])
    # EDT = UTC-4: 9:30 ET = 13:30 UTC, 12:00 ET = 16:00 UTC — 16:30/17:00 UTC are past it
    assert [(b.timestamp.hour, b.timestamp.minute) for b in edt] == [(13, 30), (14, 30)]


def test_session_slice_dst_transition_weeks():
    grid = [(12, 30), (13, 30), (14, 30), (17, 0), (17, 1)]

    def kept(day):
        bars = _session_bars([utc_bar(day, h, m, 100, 100, 100, 100) for h, m in grid])
        return [(b.timestamp.hour, b.timestamp.minute) for b in bars]

    # Spring forward 2026-03-08: Fri Mar 6 is EST, Mon Mar 9 is EDT.
    assert kept(date(2026, 3, 6)) == [(14, 30), (17, 0)]
    assert kept(date(2026, 3, 9)) == [(13, 30), (14, 30)]
    # Fall back 2026-11-01: Fri Oct 30 is EDT, Mon Nov 2 is EST.
    assert kept(date(2026, 10, 30)) == [(13, 30), (14, 30)]
    assert kept(date(2026, 11, 2)) == [(14, 30), (17, 0)]


def test_halt_deadline_is_wall_clock_not_bar_count():
    bars = [et_bar(EST_DAY, 9, 30 + i, 100.0, 100.01, 99.99, 100.0) for i in range(5)]
    bars += [
        et_bar(EST_DAY, 11, 59, 100.0, 100.01, 99.99, 100.0),
        et_bar(EST_DAY, 12, 0, 100.0, 100.05, 99.95, 99.8),
        et_bar(EST_DAY, 12, 1, 100.0, 200.0, 99.99, 150.0),   # target bait after deadline
        et_bar(EST_DAY, 12, 30, 100.0, 200.0, 99.99, 150.0),
    ]
    # A bar-COUNT deadline (150 bars) would trade into the 12:01 bait bar.
    assert len(_session_bars(bars)) == 7
    trade = simulate_trade(bars, +1)
    assert trade.exit_reason == "deadline"
    assert trade.gross_r == pytest.approx(-0.4)  # 12:00 close 99.8


def test_halt_past_the_deadline_exits_at_the_last_pre_halt_close():
    bars = [et_bar(EST_DAY, 9, 30 + i, 100.0, 100.01, 99.99, 100.0) for i in range(5)]
    bars.append(et_bar(EST_DAY, 12, 30, 100.0, 200.0, 99.99, 150.0))  # resumes after 12:00
    trade = simulate_trade(bars, +1)
    assert trade.exit_reason == "deadline"
    assert trade.gross_r == pytest.approx(0.0)  # 9:34 close; the sim does not wait for 12:30


def test_cross_day_bars_cannot_extend_a_trade():
    day1 = et_flat(EST_DAY, 151)
    _at(day1, 150, 100.0, 100.05, 99.95, 99.7)  # day-1 deadline close: -0.6R
    day2 = et_flat(date(2026, 2, 3), 10)
    _at(day2, 0, 100.0, 100.6, 99.99, 100.4)    # next morning's bar touches the target
    trade = simulate_trade(day1 + day2, +1)
    assert trade.day == EST_DAY
    assert trade.exit_reason == "deadline"
    assert trade.gross_r == pytest.approx(-0.6)


def test_entry_is_first_available_session_bar_when_0930_is_missing():
    start = datetime(EST_DAY.year, EST_DAY.month, EST_DAY.day, 10, 15, tzinfo=ET)
    bars = []
    for i in range(10):
        ts = (start + timedelta(minutes=i)).astimezone(timezone.utc)
        bars.append(MinuteBar("XYZ", ts, 100.0, 100.01, 99.99, 100.0, 1000, "test"))
    _at(bars, 3, 100.0, 100.6, 99.99, 100.4)
    trade = simulate_trade(bars, +1)
    assert trade.exit_reason == "target" and trade.gross_r == pytest.approx(1.0)


def test_truncated_morning_exits_at_last_available_close():
    bars = et_flat(EST_DAY, 31)  # data ends at 10:00 ET (e.g. feed truncation)
    _at(bars, 30, 100.0, 100.05, 99.95, 100.2)
    trade = simulate_trade(bars, +1)
    assert trade.exit_reason == "deadline"
    assert trade.gross_r == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# 2. Race semantics
# ---------------------------------------------------------------------------

def test_target_touched_in_the_entry_bar_itself():
    bars = et_flat(EST_DAY, 10)
    _at(bars, 0, 100.0, 100.6, 99.99, 100.4)
    trade = simulate_trade(bars, +1)
    assert trade.exit_reason == "target" and trade.gross_r == pytest.approx(1.0)


def test_entry_bar_dip_through_stop_is_a_stop_even_if_price_recovers():
    bars = et_flat(EST_DAY, 10)
    _at(bars, 0, 100.0, 100.1, 99.4, 100.4)   # minute 1 dips through the stop, then recovers
    _at(bars, 5, 100.0, 100.6, 99.99, 100.4)  # later target touch must not matter
    trade = simulate_trade(bars, +1)
    assert trade.exit_reason == "stop" and trade.gross_r == pytest.approx(-1.0)
    bars = et_flat(EST_DAY, 10)
    _at(bars, 0, 100.0, 100.6, 99.9, 99.6)    # short mirror: entry bar spikes through the stop
    _at(bars, 5, 100.0, 100.01, 99.4, 99.6)
    trade = simulate_trade(bars, -1)
    assert trade.exit_reason == "stop" and trade.gross_r == pytest.approx(-1.0)


def test_first_touch_wins_across_bars_in_both_orders():
    bars = et_flat(EST_DAY, 10)
    _at(bars, 2, 100.0, 100.6, 99.99, 100.4)  # target-only bar first
    _at(bars, 3, 100.0, 100.01, 99.4, 99.6)   # stop-only bar second
    assert simulate_trade(bars, +1).exit_reason == "target"
    bars = et_flat(EST_DAY, 10)
    _at(bars, 2, 100.0, 100.01, 99.4, 99.6)   # stop-only bar first
    _at(bars, 3, 100.0, 100.6, 99.99, 100.4)  # target-only bar second
    assert simulate_trade(bars, +1).exit_reason == "stop"


def test_favorable_gap_through_fills_at_the_open_long():
    bars = et_flat(EST_DAY, 10)
    _at(bars, 5, 101.0, 101.3, 100.8, 101.1)  # opens 1.0% over entry; target is +0.5%
    trade = simulate_trade(bars, +1)
    assert trade.exit_reason == "target"
    assert trade.gross_r == pytest.approx(2.0)  # the open print, NOT capped at +1R


def test_favorable_gap_open_then_dip_through_stop_still_fills_at_the_open():
    bars = et_flat(EST_DAY, 10)
    _at(bars, 5, 101.0, 101.2, 99.4, 100.0)  # opens over target, then dips through the stop
    trade = simulate_trade(bars, +1)
    # A resting limit order filled at the 101.0 open; the later dip is not our trade.
    assert trade.exit_reason == "target"
    assert trade.gross_r == pytest.approx(2.0)


def test_favorable_gap_through_fills_at_the_open_short():
    bars = et_flat(EST_DAY, 10)
    _at(bars, 5, 99.0, 99.2, 98.8, 99.1)  # opens 1.0% under entry; short target is -0.5%
    trade = simulate_trade(bars, -1)
    assert trade.exit_reason == "target"
    assert trade.gross_r == pytest.approx(2.0)


def test_short_adverse_gap_through_exits_at_the_open():
    bars = et_flat(EST_DAY, 10)
    _at(bars, 4, 100.8, 100.9, 100.7, 100.8)  # opens 0.8% over entry; short stop is +0.5%
    trade = simulate_trade(bars, -1)
    assert trade.exit_reason == "gap_through"
    assert trade.gross_r == pytest.approx(-1.6)


def test_short_deadline_exit_sign_is_negative_when_price_rises():
    bars = et_flat(EST_DAY, 151)
    _at(bars, 150, 100.0, 100.05, 99.95, 100.3)
    trade = simulate_trade(bars, -1)
    assert trade.exit_reason == "deadline"
    assert trade.gross_r == pytest.approx(-0.6)


def test_open_exactly_at_the_stop_is_a_gap_through_at_the_stop_price():
    bars = et_flat(EST_DAY, 10)
    _at(bars, 3, 99.5, 99.6, 99.4, 99.5)
    trade = simulate_trade(bars, +1)
    assert trade.exit_reason == "gap_through"
    assert trade.gross_r == pytest.approx(-1.0)


def test_stop_exit_worst_loss_frac_is_gross_excluding_costs():
    bars = et_flat(EST_DAY, 10)
    _at(bars, 2, 100.0, 100.01, 99.4, 99.6)
    trade = simulate_trade(bars, +1)
    assert trade.net_r == pytest.approx(-1.0 - COST_R)
    # The kill rule's -5% gate reads this field: it is the GROSS frac (-0.5%),
    # so the gate tolerates ~0.14% of cost drag on top — lenient, i.e. it
    # favors the strategy, consistent with the study doctrine.
    assert trade.worst_loss_frac == pytest.approx(-STOP_FRAC)
    assert trade.worst_loss_frac > -COST_FRAC - STOP_FRAC


# ---------------------------------------------------------------------------
# 3. Event integrity (find_gap_events)
# ---------------------------------------------------------------------------

def _daily(rows):
    return [Bar(date=d, open=o, high=max(o, c), low=min(o, c), close=c, volume=1)
            for d, o, c in rows]


def test_split_is_not_an_event_on_adjusted_bars_but_is_on_raw_bars():
    d = [date(2026, 1, 5) + timedelta(days=i) for i in range(2)]
    # Adjustment.ALL restates the pre-split close onto the post-split basis:
    adjusted = _daily([(d[0], 100.0, 100.0), (d[1], 100.4, 100.4)])
    assert find_gap_events({"XYZ": adjusted}, start_year=2026) == []
    # The same 2:1 split in RAW prices is a phantom -49.8% "gap": the finder
    # cannot tell the difference — correctness rides entirely on the data
    # layer delivering adjusted bars (daily_data uses Adjustment.ALL).
    raw = _daily([(d[0], 200.0, 200.0), (d[1], 100.4, 100.4)])
    events = find_gap_events({"XYZ": raw}, start_year=2026)
    assert len(events) == 1 and events[0].direction == -1
    assert events[0].gap == pytest.approx(100.4 / 200.0 - 1.0)


def test_nonpositive_prior_close_never_produces_an_event():
    d = [date(2026, 1, 5) + timedelta(days=i) for i in range(2)]
    for bad_close in (0.0, -5.0):
        bars = _daily([(d[0], 100.0, bad_close), (d[1], 130.0, 130.0)])
        assert find_gap_events({"XYZ": bars}, start_year=2026) == []


def test_gap_threshold_is_inclusive_and_just_below_is_excluded():
    d = [date(2026, 1, 5) + timedelta(days=i) for i in range(4)]
    bars = _daily([
        (d[0], 100.0, 100.0),
        (d[1], 105.0, 105.0),  # exactly +5% -> event
        (d[2], 104.9, 104.9),  # -0.095% -> no
        (d[3], 99.66, 99.66),  # 99.66/104.9 - 1 = -4.995% -> just below, no
    ])
    events = find_gap_events({"XYZ": bars}, start_year=2026)
    assert [(e.day, e.direction) for e in events] == [(d[1], 1)]
    down = _daily([(d[0], 100.0, 100.0), (d[1], 95.0, 95.0)])  # exactly -5%
    events = find_gap_events({"XYZ": down}, start_year=2026)
    assert [(e.day, e.direction) for e in events] == [(d[1], -1)]


def test_event_may_use_a_prior_year_close():
    bars = _daily([(date(2025, 12, 31), 100.0, 100.0), (date(2026, 1, 2), 106.0, 106.0)])
    events = find_gap_events({"XYZ": bars}, start_year=2026)
    assert len(events) == 1 and events[0].day == date(2026, 1, 2)


def test_unsorted_input_is_sorted_before_pairing():
    d = [date(2026, 1, 5) + timedelta(days=i) for i in range(3)]
    ordered = _daily([(d[0], 100.0, 100.0), (d[1], 106.0, 106.0), (d[2], 107.0, 107.0)])
    shuffled = [ordered[2], ordered[0], ordered[1]]
    # Before the fix, the shuffled list paired d[0] against d[2]'s close —
    # a lookahead phantom. Now the result is identical to the sorted input.
    assert find_gap_events({"XYZ": shuffled}, start_year=2026) == \
        find_gap_events({"XYZ": ordered}, start_year=2026)


def test_duplicate_dates_pair_adjacently_pin():
    # PIN (corrupt-input behavior): the finder does not deduplicate dates,
    # so two bars stamped the same day pair with each other and manufacture
    # an event. Real daily data has unique dates; this pins that a future
    # dedupe would be a deliberate change, not a silent one.
    d1, d2 = date(2026, 1, 5), date(2026, 1, 6)
    bars = _daily([(d1, 100.0, 100.0), (d1, 120.0, 120.0), (d2, 121.0, 121.0)])
    events = find_gap_events({"XYZ": bars}, start_year=2026)
    assert [(e.day, e.direction) for e in events] == [(d1, 1)]


# ---------------------------------------------------------------------------
# 4. Aggregation honesty
# ---------------------------------------------------------------------------

def _trade(day, net_r, gross_r=None, worst=None, symbol="A"):
    return TradeResult(symbol, day, 1,
                       gross_r if gross_r is not None else net_r,
                       net_r, "target" if net_r > 0 else "stop",
                       worst if worst is not None else (0.005 if net_r > 0 else -0.005))


def test_profit_factor_is_infinite_when_there_are_no_losses():
    trades = [_trade(date(2026, 1, 5), 0.72), _trade(date(2026, 1, 6), 0.72),
              _trade(date(2026, 1, 7), 0.72)]
    stats = aggregate(trades)
    assert stats.profit_factor == float("inf")  # the runner maps this to a 1e9 sentinel
    assert stats.win_rate == 1.0


def test_sharpe_is_zero_with_only_two_trading_days():
    # Even two hugely profitable days cannot manufacture a Sharpe: the
    # estimator requires > 2 daily observations. Conservative by design.
    trades = [_trade(date(2026, 1, 5), 5.0), _trade(date(2026, 1, 6), 5.0)]
    assert aggregate(trades).sharpe_daily_r == 0.0


def test_sharpe_is_zero_when_the_daily_series_is_constant():
    trades = [_trade(date(2026, 1, 5) + timedelta(days=i), 1.0) for i in range(3)]
    stats = aggregate(trades)
    assert stats.mean_net_r == 1.0
    assert stats.sharpe_daily_r == 0.0  # pstdev == 0 -> 0.0, not inf


def test_same_day_trades_are_summed_before_the_sharpe():
    d = [date(2026, 1, 5) + timedelta(days=i) for i in range(3)]
    trades = [_trade(d[0], 2.0, symbol="A"), _trade(d[0], 2.0, symbol="B"),
              _trade(d[1], 2.0), _trade(d[2], 2.0)]
    daily = [4.0, 2.0, 2.0]  # day 1 sums its two symbols
    expected = mean(daily) / pstdev(daily) * sqrt(252)
    stats = aggregate(trades)
    assert stats.sharpe_daily_r == pytest.approx(expected)
    assert stats.sharpe_daily_r > 0  # a per-trade series [2,2,2,2] would give 0.0


def test_win_rate_and_profit_factor_use_net_r_not_gross_r():
    trades = [
        _trade(date(2026, 1, 5), net_r=-0.18, gross_r=0.10),  # gross winner, net loser
        _trade(date(2026, 1, 6), net_r=0.72, gross_r=1.0),
    ]
    stats = aggregate(trades)
    assert stats.win_rate == 0.5
    assert stats.profit_factor == pytest.approx(0.72 / 0.18)


def test_aggregate_worst_is_the_min_of_the_gross_field_regardless_of_net():
    trades = [
        _trade(date(2026, 1, 5), net_r=-9.9, worst=-0.049),
        _trade(date(2026, 1, 6), net_r=0.72, worst=0.005),
    ]
    assert aggregate(trades).worst_trade_frac == pytest.approx(-0.049)


def test_aggregate_of_no_trades_is_all_zeros():
    stats = aggregate([])
    assert stats == StudyStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# 5. Cost math
# ---------------------------------------------------------------------------

def test_cost_constants_from_first_principles():
    assert TARGET_FRAC == STOP_FRAC == 0.005
    assert DEADLINE_MINUTES == 150
    # 2 fills x (5bps slippage + $1 commission on a $5,000 position)
    assert COST_FRAC == pytest.approx(2 * (5 / 10_000) + 2 * (1 / 5_000))
    assert COST_FRAC == pytest.approx(0.0014)
    assert COST_R == pytest.approx(COST_FRAC / STOP_FRAC)
    assert COST_R == pytest.approx(0.28)


def test_costs_are_symmetric_for_longs_and_shorts():
    long_bars = et_flat(EST_DAY, 10)
    _at(long_bars, 3, 100.0, 100.6, 99.99, 100.4)
    short_bars = et_flat(EST_DAY, 10)
    _at(short_bars, 3, 100.0, 100.01, 99.4, 99.6)
    long_t = simulate_trade(long_bars, +1)
    short_t = simulate_trade(short_bars, -1)
    # Borrow-free shorts: identical cost drag — a documented favor to the claim.
    assert long_t.net_r == pytest.approx(1.0 - COST_R)
    assert short_t.net_r == pytest.approx(1.0 - COST_R)
    assert long_t.net_r == pytest.approx(short_t.net_r)


# ---------------------------------------------------------------------------
# 6. Kill rule boundaries
# ---------------------------------------------------------------------------

def _stats(sharpe=1.4, pf=1.5, worst=-0.011):
    return StudyStats(trades=400, win_rate=0.6, profit_factor=pf, mean_net_r=0.2,
                      sharpe_daily_r=sharpe, worst_trade_frac=worst, total_net_r=80.0)


def test_kill_rule_exact_thresholds_pass():
    # "Sharpe >= 1.0" and "PF >= 1.3" are inclusive: exactly at the bar survives.
    survives, _ = kill_rule_verdict(_stats(sharpe=1.0, pf=1.3, worst=-0.049))
    assert survives is True


def test_kill_rule_just_below_each_threshold_fails():
    survives, reason = kill_rule_verdict(_stats(sharpe=0.999))
    assert survives is False and "Sharpe" in reason
    survives, reason = kill_rule_verdict(_stats(pf=1.299))
    assert survives is False and "profit factor" in reason
    survives, reason = kill_rule_verdict(_stats(worst=-0.0501))
    assert survives is False and "-5%" in reason


def test_kill_rule_worst_trade_exactly_minus_5pct_fails_pin():
    # PIN: the gate is `worst <= -0.05`, so a loss of EXACTLY -5.0% fails,
    # although the pre-registered wording is "worse than -5%". The code is
    # one tick stricter than the words at the exact boundary. (No live
    # trade sits anywhere near the boundary: holdout worst was -0.500%.)
    survives, reason = kill_rule_verdict(_stats(worst=-0.05))
    assert survives is False and "-5%" in reason


def test_kill_rule_empty_holdout_is_falsified_never_vacuous():
    survives, reason = kill_rule_verdict(aggregate([]))
    assert survives is False
    assert "Sharpe" in reason and "profit factor" in reason


def test_kill_rule_reports_all_failed_gates_together():
    survives, reason = kill_rule_verdict(_stats(sharpe=-2.0, pf=0.4, worst=-0.09))
    assert survives is False
    assert "Sharpe" in reason and "profit factor" in reason and "-5%" in reason


def test_kill_rule_survives_with_infinite_profit_factor():
    trades = [_trade(date(2026, 1, 5) + timedelta(days=i), 0.72) for i in range(3)]
    stats = aggregate(trades)
    assert stats.profit_factor == float("inf")
    # inf clears the PF gate; with a real Sharpe and a clean worst trade the
    # strategy would survive — the PF=inf path is not a hidden veto.
    survives, _ = kill_rule_verdict(_stats(pf=stats.profit_factor))
    assert survives is True
