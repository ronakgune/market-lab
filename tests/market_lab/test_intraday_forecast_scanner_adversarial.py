"""Adversarial suite for the day-trader-integration port.

Independent of test_intraday_session_forecast_scanner.py: written by
trying to break session.py / forecast.py / scanner.py, not by re-checking
their happy paths.

Threats covered:
  * point-in-time leakage (trade-day / future / partial sessions in a pool)
  * non-determinism, probabilities that don't sum, p_green inconsistencies
  * short-side mirror logic, plan validation, min-session boundaries
  * DST, session boundaries, duplicates, sub-minute timestamps, gaps
  * universe ticker/name edge cases, daily-setup false positives
  * ORB on missing range bars, and a hard safety rule: none of the three
    modules may import network / broker / order machinery.

Three tests here are regression tests for genuine bugs found by this
suite (fixed in the modules, see the test docstrings):
  1. a duplicated minute masking a missing minute in a 5-minute bucket
  2. a microsecond timestamp accepted by the 5-minute resampler
  3. opening range silently built from the wrong slots when a range
     bar was missing
"""
from __future__ import annotations

import ast
import math
import random
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from market_lab.intraday.bars import MinuteBar
from market_lab.intraday.forecast import (
    TradePlan, forecast_from_session_bars, history_replay, simulate_forecast,
    stop_for_rule, verdict_for,
)
from market_lab.intraday.scanner import (
    build_universe, classify_symbol_row, detect_orb, find_daily_setups,
    opening_range, orb_candidate, rsi_wilder, scan_daily,
)
from market_lab.intraday.session import (
    SessionBar, full_sessions, group_sessions, is_full_session,
    resample_to_session_bars, slots_per_session,
)

ET = ZoneInfo("America/New_York")
DAY = date(2026, 2, 3)          # a Tuesday
TRADE = date(2026, 3, 20)       # a Friday


# ------------------------------------------------------------------ helpers

def _minute_bars(day: date, minutes: int = 390, base: float = 100.0,
                 drift: float = 0.01) -> list[MinuteBar]:
    start = datetime.combine(day, time(9, 30), tzinfo=ET)
    out = []
    for i in range(minutes):
        ts = (start + timedelta(minutes=i)).astimezone(timezone.utc)
        p = base + drift * i
        out.append(MinuteBar("ACME", ts, p, p + 0.05, p - 0.05, p, 1000,
                             "synthetic"))
    return out


def _minute_at(day: date, hh: int, mm: int, ss: int = 0,
               price: float = 100.0) -> MinuteBar:
    ts = datetime.combine(day, time(hh, mm, ss), tzinfo=ET).astimezone(
        timezone.utc)
    return MinuteBar("ACME", ts, price, price + 0.05, price - 0.05, price,
                     1000, "synthetic")


def _session(day: date, closes: list[float],
             slots: list[int] | None = None) -> list[SessionBar]:
    start = datetime.combine(day, time(9, 30), tzinfo=ET)
    slots = slots if slots is not None else list(range(len(closes)))
    return [SessionBar("ACME", day, s, start + timedelta(minutes=5 * s),
                       c, c + 0.1, c - 0.1, c, 5000, 5)
            for s, c in zip(slots, closes)]


def _varied_sessions(n: int, drift: float = 0.0, vol: float = 0.004,
                     start_day: date = date(2026, 1, 5)) -> dict:
    """Full sessions whose bars actually vary per day/slot (a flat pool
    would make every seed give the same forecast and hide RNG bugs)."""
    sessions = {}
    day = start_day
    made = 0
    while made < n:
        if day.weekday() < 5:
            rng = random.Random(1000 + made)
            price = 100.0
            closes = []
            for _ in range(78):
                price *= (1 + drift + rng.uniform(-vol, vol))
                closes.append(price)
            sessions[day] = _session(day, closes)
            made += 1
        day += timedelta(days=1)
    return sessions


def _plan(**over) -> TradePlan:
    fields = dict(symbol="ACME", trade_date=TRADE, entry=100.0, stop=99.0,
                  target=101.0, entry_slot=10)
    fields.update(over)
    return TradePlan(**fields)


def _short_plan(**over) -> TradePlan:
    fields = dict(symbol="ACME", trade_date=TRADE, entry=100.0, stop=101.0,
                  target=99.0, entry_slot=10, side="short")
    fields.update(over)
    return TradePlan(**fields)


# ------------------------------------------------------------------ session: DST

def test_dst_spring_forward_week_first_session():
    # Mon 2026-03-09: ET sprang forward Sun Mar 8, offset is now UTC-4.
    bars = resample_to_session_bars(_minute_bars(date(2026, 3, 9)))
    assert len(bars) == 78 and is_full_session(bars)
    assert bars[0].start_et.utcoffset() == timedelta(hours=-4)
    assert bars[0].start_et.astimezone(timezone.utc) == datetime(
        2026, 3, 9, 13, 30, tzinfo=timezone.utc)


def test_dst_fall_back_week_first_session():
    # Mon 2026-11-02: ET fell back Sun Nov 1, offset is now UTC-5.
    bars = resample_to_session_bars(_minute_bars(date(2026, 11, 2)))
    assert len(bars) == 78 and is_full_session(bars)
    assert bars[0].start_et.utcoffset() == timedelta(hours=-5)
    assert bars[0].start_et.astimezone(timezone.utc) == datetime(
        2026, 11, 2, 14, 30, tzinfo=timezone.utc)


# ------------------------------------------------------------------ session: boundaries & ordering

def test_session_boundary_bars_0930_in_1600_out():
    bars = [_minute_at(DAY, 9, 29)]
    bars += [_minute_at(DAY, 9, 30 + i) for i in range(5)]  # 09:30-09:34
    bars += [_minute_at(DAY, 15, 55 + i) for i in range(5)]  # 15:55-15:59
    bars += [_minute_at(DAY, 16, 0)]
    out = resample_to_session_bars(bars)
    assert [b.slot for b in out] == [0, 77]


def test_resample_out_of_order_input_identical_output():
    ordered = resample_to_session_bars(_minute_bars(DAY))
    shuffled = resample_to_session_bars(list(reversed(_minute_bars(DAY))))
    assert shuffled == ordered


def test_resample_empty_input():
    assert resample_to_session_bars([]) == []


def test_bar_minutes_15_full_session_and_divisor_validation():
    bars = resample_to_session_bars(_minute_bars(DAY), bar_minutes=15)
    assert len(bars) == 26
    assert is_full_session(bars, bar_minutes=15)
    assert slots_per_session(15) == 26 and slots_per_session(13) == 30
    for bad in (0, -5, 4, 7, 11):
        with pytest.raises(ValueError):
            slots_per_session(bad)


def test_resample_1min_mode_full_session():
    bars = resample_to_session_bars(_minute_bars(DAY), bar_minutes=1)
    assert len(bars) == 390
    assert is_full_session(bars, bar_minutes=1)
    assert bars[0].slot == 0 and bars[-1].slot == 389


# ------------------------------------------------------------------ session: duplicates & sub-minute

def test_duplicate_minute_1min_mode_drops_that_slot():
    bars = _minute_bars(DAY)
    out = resample_to_session_bars(bars + [bars[100]], bar_minutes=1)
    assert len(out) == 389
    assert 100 not in [b.slot for b in out]
    assert not is_full_session(out, bar_minutes=1)


def test_duplicate_minute_5min_mode_drops_whole_bucket():
    bars = _minute_bars(DAY)
    out = resample_to_session_bars(bars + [bars[10]])
    assert len(out) == 77
    assert 2 not in [b.slot for b in out]  # minute 10 lives in bucket 2


def test_duplicate_minute_must_not_mask_a_missing_minute():
    """REGRESSION (bug found by this suite): replacing minute 4 with a
    second copy of minute 0 left bucket 0 with 5 members, so the bucket
    was emitted and the session reported full although minute 4 never
    existed. A bucket is complete only if its minutes are exactly the
    slot's minutes."""
    bars = _minute_bars(DAY)
    bars[4] = bars[0]
    out = resample_to_session_bars(bars)
    assert len(out) == 77
    assert 0 not in [b.slot for b in out]
    assert not is_full_session(out)


def test_microsecond_timestamp_dropped_in_both_modes():
    """REGRESSION (bug found by this suite): the 5-minute path checked
    only ``second``, so a bar at 09:30:00.500 was aggregated as if it
    were the 09:30 bar, while the 1-minute path dropped it."""
    bars = _minute_bars(DAY)
    bars[0] = MinuteBar("ACME", bars[0].timestamp + timedelta(microseconds=500),
                        bars[0].open, bars[0].high, bars[0].low, bars[0].close,
                        1000, "synthetic")
    out5 = resample_to_session_bars(bars)
    assert len(out5) == 77 and 0 not in [b.slot for b in out5]
    out1 = resample_to_session_bars(bars, bar_minutes=1)
    assert len(out1) == 389


def test_bar_with_seconds_dropped():
    bars = _minute_bars(DAY)
    bars[7] = MinuteBar("ACME", bars[7].timestamp + timedelta(seconds=30),
                        bars[7].open, bars[7].high, bars[7].low, bars[7].close,
                        1000, "synthetic")
    out = resample_to_session_bars(bars)
    assert len(out) == 77 and 1 not in [b.slot for b in out]  # minute 7 -> bucket 1


def test_incomplete_middle_bucket_dropped_others_kept():
    bars = _minute_bars(DAY)
    del bars[50]  # one minute out of bucket 10
    out = resample_to_session_bars(bars)
    assert len(out) == 77
    assert 10 not in [b.slot for b in out]
    assert not is_full_session(out)


# ------------------------------------------------------------------ session: fullness

def test_duplicate_slot_session_is_not_full():
    bars = _session(DAY, [100.0] * 78)
    bars[-1] = SessionBar("ACME", DAY, 0, bars[0].start_et, 100.0, 100.1,
                          99.9, 100.0, 5000, 5)  # slot 0 twice, slot 77 gone
    assert len(bars) == 78
    assert not is_full_session(bars)
    assert full_sessions(group_sessions(bars)) == {}


def test_full_sessions_excludes_gappy_and_is_strictly_before():
    good = _session(date(2026, 2, 2), [100.0] * 78)
    gappy = _session(date(2026, 2, 3), [100.0] * 77)  # one slot short
    later = _session(date(2026, 2, 4), [100.0] * 78)
    grouped = group_sessions(good + gappy + later)
    kept = full_sessions(grouped, before=date(2026, 2, 4))
    assert sorted(kept) == [date(2026, 2, 2)]  # gappy out, boundary day out


# ------------------------------------------------------------------ forecast: point-in-time leakage

def test_forecast_ignores_trade_day_and_future_sessions():
    base = _varied_sessions(30)
    polluted = dict(base)
    polluted[TRADE] = _session(TRADE, [10.0] * 78)            # wild trade day
    polluted[TRADE + timedelta(days=3)] = _session(
        TRADE + timedelta(days=3), [500.0] * 78)              # wild future
    r_clean = simulate_forecast(_plan(), base, replays=400, seed=11)
    r_polluted = simulate_forecast(_plan(), polluted, replays=400, seed=11)
    assert r_polluted == r_clean
    assert r_polluted.pool_sessions == 30


def test_forecast_ignores_partial_and_gappy_sessions():
    base = _varied_sessions(30)
    polluted = dict(base)
    polluted[date(2026, 2, 7)] = _session(date(2026, 2, 7), [999.0] * 40)
    polluted[date(2026, 2, 8)] = _session(date(2026, 2, 8), [0.5] * 77)
    r_clean = simulate_forecast(_plan(), base, replays=400, seed=11)
    r_polluted = simulate_forecast(_plan(), polluted, replays=400, seed=11)
    assert r_polluted == r_clean
    assert r_polluted.pool_sessions == 30


def test_history_replay_excludes_partial_sessions():
    sessions = _varied_sessions(25, drift=-0.003)
    sessions[date(2026, 2, 7)] = _session(date(2026, 2, 7), [1.0] * 10)
    h = history_replay(sessions)
    assert h.sessions == 25


# ------------------------------------------------------------------ forecast: determinism & consistency

def test_forecast_same_seed_identical_different_seed_differs():
    sessions = _varied_sessions(30)
    a = simulate_forecast(_plan(), sessions, replays=400, seed=11)
    b = simulate_forecast(_plan(), sessions, replays=400, seed=11)
    c = simulate_forecast(_plan(), sessions, replays=400, seed=12)
    assert a == b
    assert a != c


@pytest.mark.parametrize("seed", [1, 7, 42, 1234])
def test_forecast_probabilities_sum_and_green_consistency(seed):
    r = simulate_forecast(_plan(), _varied_sessions(30), replays=500, seed=seed)
    assert math.isclose(r.p_target + r.p_stop + r.p_neither, 1.0)
    for p in (r.p_target, r.p_stop, r.p_neither, r.p_green):
        assert 0.0 <= p <= 1.0
    # green = target-first OR still-open-but-above-entry at the end, so it
    # must sit between p_target and p_target + p_neither.
    assert r.p_target - 1e-12 <= r.p_green <= r.p_target + r.p_neither + 1e-12
    assert r.p5_close <= r.median_close <= r.p95_close


def test_forecast_from_session_bars_matches_grouped():
    sessions = _varied_sessions(25)
    flat = [b for day in sorted(sessions) for b in sessions[day]]
    direct = simulate_forecast(_plan(), sessions, replays=200, seed=3)
    via = forecast_from_session_bars(_plan(), flat, replays=200, seed=3)
    assert via == direct


# ------------------------------------------------------------------ forecast: short side

def test_short_mirror_swaps_target_and_stop():
    """Same seed => same simulated price paths. Mirroring the plan must
    swap p_target/p_stop exactly, keep p_neither and the close
    distribution, and the two p_greens must sum to 1 (every non-target
    long outcome is favourable to the mirrored short and vice versa)."""
    sessions = _varied_sessions(30, drift=-0.001, vol=0.006)
    rl = simulate_forecast(_plan(), sessions, replays=600, seed=9, until_slot=20)
    rs = simulate_forecast(_short_plan(), sessions, replays=600, seed=9,
                           until_slot=20)
    assert 0.0 < rl.p_neither < 1.0  # interior case, not a degenerate one
    assert rs.p_target == rl.p_stop
    assert rs.p_stop == rl.p_target
    assert rs.p_neither == rl.p_neither
    assert rs.median_close == rl.median_close
    assert math.isclose(rl.p_green + rs.p_green, 1.0)


def test_short_dominates_on_declining_pool_and_long_does_not():
    sessions = _varied_sessions(30, drift=-0.003)
    rl = simulate_forecast(_plan(), sessions, replays=400, seed=5)
    rs = simulate_forecast(_short_plan(), sessions, replays=400, seed=5)
    assert rs.p_target > 0.9
    assert rl.p_stop > 0.9
    assert rs.p_green > rl.p_green


def test_history_replay_short_fires_on_declining_sessions():
    sessions = _varied_sessions(30, drift=-0.003)
    hs = history_replay(sessions, side="short")
    hl = history_replay(sessions, side="long")
    assert hs.sessions == 30 and hs.triggered == 30
    assert hs.target == 30 and hs.stop == 0 and hs.close == 0
    assert hl.triggered == 0 and hl.avg_r is None
    assert hs.target + hs.stop + hs.close == hs.triggered


# ------------------------------------------------------------------ forecast: plan validation

@pytest.mark.parametrize("over", [
    dict(stop=100.0),            # stop == entry (long)
    dict(stop=101.0),            # stop above entry (long)
    dict(target=100.0),          # target == entry (long)
    dict(target=99.0),           # target below entry (long)
    dict(entry=0.0), dict(entry=-5.0), dict(stop=0.0),
    dict(entry=float("nan")), dict(entry=float("inf")),
    dict(target=float("nan")),
    dict(entry_slot=-1), dict(entry_slot=78),
    dict(side="flat"), dict(side="LONG"),
])
def test_plan_validation_rejects_bad_long_plans(over):
    with pytest.raises(ValueError):
        _plan(**over)


@pytest.mark.parametrize("over", [
    dict(stop=99.0),             # stop below entry (short)
    dict(stop=100.0),            # stop == entry (short)
    dict(target=101.0),          # target above entry (short)
    dict(target=100.0),          # target == entry (short)
])
def test_plan_validation_rejects_bad_short_plans(over):
    with pytest.raises(ValueError):
        _short_plan(**over)


def test_plan_risk_and_target_r_both_sides():
    long2 = _plan(stop=98.0, target=104.0)
    assert long2.risk == 2.0 and long2.target_r == 2.0
    short2 = _short_plan(stop=102.0, target=96.0)
    assert short2.risk == 2.0 and short2.target_r == 2.0
    # entry_slot bound follows bar_minutes: slot 389 legal at 1 minute.
    p = TradePlan("ACME", TRADE, 100.0, 99.0, 101.0, 389, bar_minutes=1)
    assert p.entry_slot == 389
    with pytest.raises(ValueError):
        TradePlan("ACME", TRADE, 100.0, 99.0, 101.0, 390, bar_minutes=1)


# ------------------------------------------------------------------ forecast: boundaries

def test_min_sessions_boundary_exactly_20_passes_19_fails():
    ok = simulate_forecast(_plan(), _varied_sessions(20), replays=50, seed=1)
    assert ok.pool_sessions == 20
    with pytest.raises(ValueError):
        simulate_forecast(_plan(), _varied_sessions(19), replays=50, seed=1)
    # the boundary moves with an explicit min_sessions
    ok5 = simulate_forecast(_plan(), _varied_sessions(5), replays=50,
                            seed=1, min_sessions=5)
    assert ok5.pool_sessions == 5
    with pytest.raises(ValueError):
        simulate_forecast(_plan(), _varied_sessions(4), replays=50,
                          seed=1, min_sessions=5)


def test_until_slot_validation_and_bars_left():
    sessions = _varied_sessions(25)
    r = simulate_forecast(_plan(), sessions, replays=100, seed=2, until_slot=15)
    assert r.bars_left == 5
    for bad_until in (10, 9, 0, 78, 200):
        with pytest.raises(ValueError):
            simulate_forecast(_plan(), sessions, replays=10,
                              until_slot=bad_until)


def test_entry_at_last_usable_slot_and_at_close():
    sessions = _varied_sessions(25)
    r = simulate_forecast(_plan(entry_slot=76), sessions, replays=100, seed=2)
    assert r.bars_left == 1
    # entering on the final bar leaves nothing to replay: refuse.
    with pytest.raises(ValueError):
        simulate_forecast(_plan(entry_slot=77), sessions, replays=10)


def test_window_zero_is_deterministic_and_valid():
    sessions = _varied_sessions(25)
    a = simulate_forecast(_plan(), sessions, replays=300, seed=8, window=0)
    b = simulate_forecast(_plan(), sessions, replays=300, seed=8, window=0)
    assert a == b and a.window == 0
    assert math.isclose(a.p_target + a.p_stop + a.p_neither, 1.0)


def test_replays_and_cost_validation():
    sessions = _varied_sessions(25)
    for bad in (0, -10):
        with pytest.raises(ValueError):
            simulate_forecast(_plan(), sessions, replays=bad)
    with pytest.raises(ValueError):
        simulate_forecast(_plan(), sessions, replays=10, cost_bps=-1)


def test_cost_bps_monotonic_exact_shift_and_no_effect_on_odds():
    sessions = _varied_sessions(30)
    r0 = simulate_forecast(_plan(), sessions, replays=300, seed=4, cost_bps=0)
    r25 = simulate_forecast(_plan(), sessions, replays=300, seed=4, cost_bps=25)
    r100 = simulate_forecast(_plan(), sessions, replays=300, seed=4,
                             cost_bps=100)
    # entry 100, risk 1: 25 bps of entry = 0.25R, 100 bps = 1.0R exactly.
    assert r25.avg_r == pytest.approx(r0.avg_r - 0.25, abs=0.0011)
    assert r100.avg_r == pytest.approx(r0.avg_r - 1.0, abs=0.0011)
    assert r0.avg_r > r25.avg_r > r100.avg_r
    # costs change the accounting, never the simulated outcomes
    assert (r25.p_target, r25.p_stop, r25.p_neither, r25.median_close) == \
        (r0.p_target, r0.p_stop, r0.p_neither, r0.median_close)


# ------------------------------------------------------------------ forecast: verdict & stop rules

def test_verdict_edge_cases():
    assert "No replay hit" in verdict_for(0.0, 0.0, 1.0)
    assert verdict_for(0.10, 0.50, 1.0) == "Strongly toward the stop"
    assert verdict_for(0.40, 0.30, 1.0) == "Leans toward the target"
    assert verdict_for(0.60, 0.10, 1.0) == "Strongly toward the target"
    # at 3R, break-even share is 25%: a 20/60 split leans to the target
    assert verdict_for(0.20, 0.40, 3.0) == "Leans toward the target"


def test_stop_for_rule_rounding_and_variants():
    # or_mid of 101.005/99.0 = 100.0025: long floors to the cent, short ceils.
    assert stop_for_rule("or_mid", 101.005, 99.0, 100.0, "long") == 100.0
    assert stop_for_rule("or_mid", 101.005, 99.0, 100.0, "short") == 100.01
    assert stop_for_rule("or_extreme", 101.0, 99.0, 100.0, "long") == 99.0
    assert stop_for_rule("or_extreme", 101.0, 99.0, 100.0, "short") == 101.0
    assert stop_for_rule("pct", 101.0, 99.0, 100.0, "long", 1.0) == 99.0
    assert stop_for_rule("pct", 101.0, 99.0, 100.0, "short", 1.0) == 101.0
    with pytest.raises(ValueError):
        stop_for_rule("nonsense", 101.0, 99.0, 100.0, "long")
    with pytest.raises(ValueError):
        history_replay(_varied_sessions(3), or_bars=0)


# ------------------------------------------------------------------ scanner: universe edges

def _row(sym: str, name: str, **kw) -> dict:
    r = {"Symbol": sym, "Security Name": name, "Test Issue": "N",
         "ETF": "N", "NextShares": "N"}
    r.update(kw)
    return r


def test_universe_class_letter_ticker_kept():
    assert classify_symbol_row(
        _row("BRK.B", "Berkshire Hathaway Inc. - Common Stock")) == "keep"
    assert classify_symbol_row(_row("BF.A", "Brown-Forman Corp")) == "keep"


@pytest.mark.parametrize("sym", ["ABC.U", "ABC.W", "ABC.R"])
def test_universe_unit_warrant_right_suffixes_out(sym):
    assert classify_symbol_row(_row(sym, "Acme Corp - Common Stock")) == \
        "unit / warrant / right suffix"


def test_universe_dollar_preferred_not_a_plain_ticker():
    assert classify_symbol_row(
        _row("ABR$D", "Arbor Realty Trust - Preferred Shares")) == \
        "not a plain ticker"
    assert classify_symbol_row(_row("TOOLONGX", "Acme Corp")) == \
        "not a plain ticker"
    assert classify_symbol_row(_row("", "Acme Corp")) == "not a plain ticker"


def test_universe_depositary_rule_american_global_kept_plain_out():
    assert classify_symbol_row(
        _row("AMDS", "Acme Inc. - American Depositary Shares")) == "keep"
    assert classify_symbol_row(
        _row("GLDS", "Acme Inc. - Global Depositary Shares")) == "keep"
    assert classify_symbol_row(
        _row("PLDS", "Acme Inc. - Depositary Shares")) == "depositary"
    assert classify_symbol_row(
        _row("PLDX", "Acme Inc. - Depositary Shares, each representing "
                     "one share of Common Stock")) == "depositary"


def test_universe_name_and_flag_rules():
    assert classify_symbol_row(
        _row("CPN", "Acme Corp - 7.00% Fixed Rate Reset Cumulative")) == "coupon"
    assert classify_symbol_row(
        _row("WRN", "Acme Corp - Warrants")) == "warrant"
    assert classify_symbol_row(
        _row("UNT", "Acme Acquisition Corp - Units, each consisting of "
                     "one share")) == "unit or right"
    assert classify_symbol_row(
        _row("TST", "Acme Corp - Common Stock", **{"Test Issue": "Y"})) == \
        "test issue"
    assert classify_symbol_row(
        _row("ETF1", "Acme Index Trust", ETF="Y")) == "ETF"
    assert classify_symbol_row(
        _row("NXT", "Acme Corp - Common Stock", NextShares="Y")) == "NextShares"


def test_build_universe_dedupes_across_files():
    nasdaq = (
        "Symbol|Security Name|Test Issue|ETF|NextShares\n"
        "ACME|Acme Corp - Common Stock|N|N|N\n"
        "File Creation Time: 0203202609:30|||\n")
    other = (
        "ACT Symbol|Security Name|Test Issue|ETF|NextShares\n"
        "ACME|Acme Corp - Common Stock|N|N|N\n"
        "File Creation Time: 0203202609:30|||\n")
    kept, counts, _ = build_universe(
        {"nasdaqlisted.txt": nasdaq, "otherlisted.txt": other})
    assert [e.symbol for e in kept] == ["ACME"]
    with pytest.raises(ValueError):
        build_universe({"nasdaqlisted.txt": nasdaq})  # missing file refuses


# ------------------------------------------------------------------ scanner: RSI

def test_rsi_flat_series_no_gains_no_losses():
    # Wilder convention in this module: avg_loss == 0 -> RSI 100, flat
    # included. The important safety property is downstream: a flat tape
    # must never be tagged oversold / oversold_bounce.
    assert rsi_wilder([100.0] * 30) == 100.0
    tags, nums = find_daily_setups([100.0] * 35)
    assert tags == ()
    assert nums["rsi"] == 100.0


def test_rsi_length_boundary():
    assert rsi_wilder([100.0 + i for i in range(14)]) is None
    assert rsi_wilder([100.0 + i for i in range(15)]) == pytest.approx(100.0)


# ------------------------------------------------------------------ scanner: daily setups fire / don't fire

def _flag_series(run_pct_target: float = 0.12, last5=None) -> list[float]:
    closes = [100.0] * 20
    closes += [102, 104, 106, 108, 110, 111, 112, 112, 112,
               100.0 * (1 + run_pct_target)]  # c[-6]: sets the run
    closes += last5 if last5 is not None else [111.8, 112.2, 111.9, 112.1, 112.0]
    return closes


def test_bull_flag_fires_and_controls_do_not():
    tags, _ = find_daily_setups(_flag_series())
    assert "bull_flag" in tags
    # control 1: run under 10% -> no flag
    tags_low, _ = find_daily_setups(_flag_series(run_pct_target=0.04))
    assert "bull_flag" not in tags_low
    # control 2: run is there but the last 5 closes range too wide -> no flag
    wide = _flag_series(last5=[112.0, 105.0, 118.0, 106.0, 112.0])
    tags_wide, nums_wide = find_daily_setups(wide)
    assert nums_wide["range5_pct"] > 5.0
    assert "bull_flag" not in tags_wide


def test_reclaim_ma20_fires_and_flat_control_does_not():
    tags, _ = find_daily_setups([100.0] * 33 + [95.0, 105.0])
    assert "reclaim_ma20" in tags
    tags_flat, _ = find_daily_setups([100.0] * 35)
    assert "reclaim_ma20" not in tags_flat
    # control: last close above the average but no dip below it first
    tags_nodip, _ = find_daily_setups([100.0] * 34 + [105.0])
    assert "reclaim_ma20" not in tags_nodip


def test_oversold_bounce_fires_and_continuing_decline_does_not():
    bounce = [200.0 - 3.0 * i for i in range(34)] + [200.0 - 3.0 * 33 + 1.0]
    tags, nums = find_daily_setups(bounce)
    assert nums["rsi_prev"] < 30
    assert "oversold_bounce" in tags
    cont = [200.0 - 3.0 * i for i in range(35)]
    tags_cont, _ = find_daily_setups(cont)
    assert "oversold" in tags_cont
    assert "oversold_bounce" not in tags_cont
    # control: an uptick that was never oversold is not a bounce
    up = [100.0 + i for i in range(34)] + [140.0]
    assert "oversold_bounce" not in find_daily_setups(up)[0]


# ------------------------------------------------------------------ scanner: scan_daily ordering

def _breakout_series(top: float) -> list[float]:
    base = top - 10.0
    return [base] * 25 + [base + 1, base + 2, base + 3, base + 4, top]


def test_scan_daily_orders_by_dollar_volume_not_symbol():
    series = {"AAA": _breakout_series(110.0), "ZZZ": _breakout_series(60.0)}
    hits = scan_daily(series, volumes={"AAA": 100, "ZZZ": 10_000})
    # ZZZ: 60 * 10_000 = 600k dollars beats AAA: 110 * 100 = 11k.
    assert [h.symbol for h in hits] == ["ZZZ", "AAA"]


def test_scan_daily_tie_breaks_by_symbol_and_no_volumes_is_alphabetical():
    series = {"ZZZ": _breakout_series(55.0), "AAA": _breakout_series(110.0),
              "MMM": _breakout_series(110.0)}
    hits = scan_daily(series, volumes={"AAA": 100, "ZZZ": 200, "MMM": 100})
    # AAA and MMM tie at 11k, ZZZ at 11k too (55*200): pure symbol order.
    assert [h.symbol for h in hits] == ["AAA", "MMM", "ZZZ"]
    hits_novol = scan_daily(series)
    assert [h.symbol for h in hits_novol] == ["AAA", "MMM", "ZZZ"]


# ------------------------------------------------------------------ scanner: ORB

def test_orb_short_side_fires_long_does_not():
    closes = [100.0, 99.9, 100.05, 99.5, 99.4, 99.3]
    bars = _session(DAY, closes)  # or_low = 99.9 - 0.1 = 99.8, or_high = 100.15
    hit = detect_orb(bars, side="short")
    assert hit is not None and hit.entry_slot == 3
    assert hit.or_low == pytest.approx(99.8)
    assert hit.pct_over < 0
    assert detect_orb(bars, side="long") is None
    with pytest.raises(ValueError):
        detect_orb(bars, side="sideways")


def test_orb_missing_range_bars_by_count():
    bars = _session(DAY, [100.0, 100.1])  # only 2 of the 3 range bars exist
    with pytest.raises(ValueError):
        opening_range(bars, or_bars=3)
    assert detect_orb(bars) is None
    assert orb_candidate(bars) is False


def test_orb_missing_range_slot_gap():
    """REGRESSION (bug found by this suite): with slot 1 missing, the
    old code built the 'opening range' from slots 0, 2, 3 — a bar from
    after the range — and fired breakouts against that wrong range.
    A range with a hole in it is no range at all."""
    bars = _session(DAY, [100.0, 100.5, 101.0, 102.0, 102.0, 102.0],
                    slots=[0, 2, 3, 4, 5, 6])
    with pytest.raises(ValueError):
        opening_range(bars, or_bars=3)
    assert detect_orb(bars) is None
    assert orb_candidate(bars) is False


def test_orb_candidate_is_necessary_not_sufficient():
    # a later close EQUAL to the highest range close passes stage 1
    # (>=) but is not a stage-2 breakout (needs > range HIGH).
    closes = [100.0, 100.2, 100.1, 100.2, 100.2, 100.2]
    bars = _session(DAY, closes)
    assert orb_candidate(bars) is True
    assert detect_orb(bars) is None


def test_opening_range_unordered_input():
    bars = _session(DAY, [100.0, 101.0, 99.0, 100.5])
    hi, lo = opening_range(list(reversed(bars)), or_bars=3)
    assert hi == pytest.approx(101.1) and lo == pytest.approx(98.9)


# ------------------------------------------------------------------ safety: no network / broker / order imports

_FORBIDDEN_IMPORT_ROOTS = {
    "urllib", "requests", "socket", "http", "ftplib", "smtplib",
    "alpaca", "alpaca_trade_api", "websocket", "websockets", "aiohttp",
}
_FORBIDDEN_TEXT = ("urllib", "requests", "socket", "TradingClient",
                   "place_order", "submit_order")


def _module_path(name: str) -> Path:
    return Path(__file__).resolve().parents[2] / "market_lab" / "intraday" / name


@pytest.mark.parametrize("fname", ["session.py", "forecast.py", "scanner.py"])
def test_module_imports_no_network_or_broker_machinery(fname):
    src = _module_path(fname).read_text()
    tree = ast.parse(src)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not (imported & _FORBIDDEN_IMPORT_ROOTS), \
        f"{fname} imports forbidden modules: {imported & _FORBIDDEN_IMPORT_ROOTS}"
    for bad in _FORBIDDEN_TEXT:
        assert bad not in src, f"{fname} mentions {bad!r}"
