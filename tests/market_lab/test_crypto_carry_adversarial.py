"""Adversarial tests for market_lab/crypto_carry.py + the monitor script
(Stream 6). Baseline file: test_crypto_carry.py (untouched).

Attack surface: venue field/period semantics, timestamp integrity, store
integrity under poisoned data, trigger gaming via snapshot density and
staleness, and failure isolation in collect(). Fixtures only — no network.
"""
import importlib.util
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from market_lab.crypto_carry import (
    FundingPoint,
    append_points,
    load_points,
    parse_deribit_history,
    parse_kraken_tickers,
    parse_okx_history,
    summarize,
)

_spec = importlib.util.spec_from_file_location(
    "crypto_carry_monitor",
    Path(__file__).resolve().parents[2] / "scripts" / "crypto_carry_monitor.py",
)
monitor = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(monitor)

T0 = datetime(2026, 11, 1, tzinfo=timezone.utc)


def _ms(dt: datetime) -> str:
    return str(int(dt.timestamp() * 1000))


def _series(venue, asset, rate, period_hours, days, start=T0):
    per_day = int(24 / period_hours)
    return [
        FundingPoint(venue, asset, start + timedelta(hours=period_hours * i), rate, period_hours)
        for i in range(days * per_day)
    ]


# ---------------------------------------------------------------------------
# 1. Annualization / field semantics
# ---------------------------------------------------------------------------

def test_deribit_uses_interest_1h_not_interest_8h():
    # The two fields disagree wildly: using interest_8h as the rate (or the
    # 1h rate with an 8h period) misstates the carry by ~8x either way.
    payload = {"result": [
        {"timestamp": int(T0.timestamp() * 1000), "interest_1h": 0.00001, "interest_8h": 0.9},
    ]}
    points = parse_deribit_history(payload, "BTC")
    assert points[0].rate == 0.00001
    assert points[0].period_hours == 1.0
    assert points[0].annualized() == pytest.approx(0.00001 * 24 * 365)


def test_okx_prefers_realized_rate_for_settled_history():
    # funding-rate-history rows are SETTLED periods: fundingRate is the
    # announced/predicted rate, realizedRate is what was actually charged.
    # A carry measurement must use the charged rate when the two disagree.
    payload = {"code": "0", "data": [
        {"fundingRate": "0.0009", "realizedRate": "0.0001",
         "fundingTime": _ms(T0), "instId": "BTC-USDT-SWAP"},
    ]}
    points = parse_okx_history(payload, "BTC")
    assert points[0].rate == pytest.approx(0.0001)


def test_okx_falls_back_to_funding_rate_when_realized_absent_or_empty():
    payload = {"code": "0", "data": [
        {"fundingRate": "0.0002", "fundingTime": _ms(T0)},
        {"fundingRate": "0.0003", "realizedRate": "", "fundingTime": _ms(T0 + timedelta(hours=8))},
    ]}
    points = parse_okx_history(payload, "BTC")
    assert [p.rate for p in points] == [0.0002, 0.0003]


# ---------------------------------------------------------------------------
# 2. Timestamp integrity
# ---------------------------------------------------------------------------

def test_seconds_timestamps_are_not_stored_as_1970():
    # Venue contracts say milliseconds, but a seconds value divided by 1000
    # lands in 1970 and silently poisons the store. Normalize by magnitude.
    okx_payload = {"code": "0", "data": [
        {"fundingRate": "0.0001", "fundingTime": str(int(T0.timestamp()))},
    ]}
    assert parse_okx_history(okx_payload, "BTC")[0].ts == T0
    deribit_payload = {"result": [
        {"timestamp": int(T0.timestamp()), "interest_1h": 0.00001},
    ]}
    assert parse_deribit_history(deribit_payload, "BTC")[0].ts == T0


def test_duplicate_timestamps_within_one_payload_stored_once(tmp_path):
    payload = {"code": "0", "data": [
        {"fundingRate": "0.0001", "fundingTime": _ms(T0)},
        {"fundingRate": "0.0001", "fundingTime": _ms(T0)},  # exact repeat
    ]}
    path = tmp_path / "funding.jsonl"
    assert append_points(path, parse_okx_history(payload, "BTC")) == 1
    assert len(load_points(path)) == 1


def test_out_of_order_points_summarize_correctly():
    points = _series("okx", "BTC", 0.0001, 8.0, days=30)
    shuffled = list(reversed(points))  # venues return newest-first
    summary = summarize(shuffled)[0]
    assert summary.points == 90
    assert summary.mean_annualized == pytest.approx(0.1095)
    assert summary.latest_annualized == pytest.approx(0.1095)


def test_naive_timestamp_is_coerced_to_utc_and_mixes_safely():
    naive = FundingPoint("kraken", "BTC", datetime(2026, 11, 1, 0, 0, 0), 0.00002, 1.0)
    assert naive.ts.tzinfo is not None
    aware = FundingPoint("kraken", "BTC", T0 + timedelta(hours=1), 0.00002, 1.0)
    # Same group, mixed construction paths: must not raise on sort/subtract.
    summary = summarize([aware, naive])[0]
    assert summary.points == 2


# ---------------------------------------------------------------------------
# 3. Store integrity under poisoned data
# ---------------------------------------------------------------------------

def test_nan_rate_from_payload_never_reaches_store(tmp_path):
    # A NaN rate (string form, as a misbehaving venue/proxy could send) must
    # be dropped at parse time — not crash append_points mid-batch and kill
    # the whole monitor run after a partial write.
    payload = {"code": "0", "data": [
        {"fundingRate": "0.0001", "fundingTime": _ms(T0)},
        {"fundingRate": "NaN", "fundingTime": _ms(T0 + timedelta(hours=8))},
        {"fundingRate": "Infinity", "fundingTime": _ms(T0 + timedelta(hours=16))},
    ]}
    points = parse_okx_history(payload, "BTC")
    assert len(points) == 1
    path = tmp_path / "funding.jsonl"
    assert append_points(path, points) == 1
    assert len(load_points(path)) == 1


def test_append_skips_nonfinite_without_partial_crash(tmp_path):
    good1 = FundingPoint("okx", "BTC", T0, 0.0001, 8.0)
    bad = FundingPoint("okx", "BTC", T0 + timedelta(hours=8), math.inf, 8.0)
    good2 = FundingPoint("okx", "BTC", T0 + timedelta(hours=16), 0.0002, 8.0)
    path = tmp_path / "funding.jsonl"
    assert append_points(path, [good1, bad, good2]) == 2
    assert len(load_points(path)) == 2


def test_nan_and_zero_period_lines_in_store_are_skipped_on_load(tmp_path):
    path = tmp_path / "funding.jsonl"
    path.write_text(
        '{"venue": "okx", "asset": "BTC", "ts": "2026-11-01T00:00:00+00:00", "rate": NaN, "period_hours": 8.0}\n'
        '{"venue": "okx", "asset": "BTC", "ts": "2026-11-01T08:00:00+00:00", "rate": 0.0001, "period_hours": 0}\n'
        '{"venue": "okx", "asset": "BTC", "ts": "2026-11-01T16:00:00+00:00", "rate": 0.0001, "period_hours": 8.0}\n'
    )
    points = load_points(path)
    assert len(points) == 1  # only the healthy row survives
    summary = summarize(points)[0]  # and summarizing cannot divide by zero
    assert math.isfinite(summary.mean_annualized)


def test_negative_and_extreme_rates_roundtrip(tmp_path):
    neg = FundingPoint("okx", "BTC", T0, -0.5 / 1095, 8.0)  # -50% annualized
    huge = FundingPoint("okx", "BTC", T0 + timedelta(hours=8), 1.0, 8.0)  # 109500%
    path = tmp_path / "funding.jsonl"
    assert append_points(path, [neg, huge]) == 2
    loaded = load_points(path)
    assert loaded[0].rate == pytest.approx(-0.5 / 1095)
    assert loaded[0].annualized() == pytest.approx(-0.5)
    assert loaded[1].annualized() == pytest.approx(1095.0)


def test_interleaved_appends_dedupe_across_calls(tmp_path):
    a = _series("deribit", "ETH", 0.00001, 1.0, days=2)
    b = _series("deribit", "ETH", 0.00001, 1.0, days=3)  # overlaps a entirely
    path = tmp_path / "funding.jsonl"
    assert append_points(path, a) == 48
    assert append_points(path, b) == 24  # only day 3 is new
    assert len(load_points(path)) == 72


# ---------------------------------------------------------------------------
# 4. Trigger gaming: window boundaries and staleness
# ---------------------------------------------------------------------------

def test_window_boundary_is_inclusive_at_exactly_30_days():
    base = _series("kraken", "BTC", 0.00002, 1.0, days=30)
    latest = base[-1].ts
    on_edge = FundingPoint("kraken", "BTC", latest - timedelta(days=30), 0.00002, 1.0)
    outside = FundingPoint("kraken", "BTC", latest - timedelta(days=30, seconds=1), 0.00002, 1.0)
    summary = summarize(base + [on_edge, outside])[0]
    assert summary.points == len(base) + 1  # edge counts, 1s beyond does not


def test_stale_series_cannot_trigger():
    # A rich series that ENDS long before the freshest data in the store
    # (e.g. the venue has been failing for months while others update)
    # must not present as a live trigger.
    stale = _series("kraken", "BTC", 0.00002, 1.0, days=30,
                    start=datetime(2026, 6, 1, tzinfo=timezone.utc))
    fresh = _series("okx", "BTC", 0.0001, 8.0, days=30, start=T0)
    by_venue = {s.venue: s for s in summarize(stale + fresh)}
    assert by_venue["kraken"].stale is True
    assert by_venue["kraken"].triggered is False
    assert by_venue["okx"].stale is False


def test_stale_flag_relative_not_wallclock():
    # A lone series is its own freshest reference: summarize stays a pure
    # function of the points (no wall clock), so old-but-only data is not
    # marked stale by summarize itself.
    lone = _series("kraken", "BTC", 0.00002, 1.0, days=30,
                   start=datetime(2025, 1, 1, tzinfo=timezone.utc))
    summary = summarize(lone)[0]
    assert summary.stale is False
    assert summary.triggered is True


# ---------------------------------------------------------------------------
# 5. Kraken snapshot semantics
# ---------------------------------------------------------------------------

def test_kraken_burst_snapshots_collapse_to_one_point_per_period(tmp_path):
    # The monitor (or an operator testing it) may run many times within
    # one hour. Each run snapshots the CURRENT Kraken rate; storing every
    # snapshot lets run frequency, not time, weight the 30d mean — 300
    # snapshots in one rich hour dragged a 1.75% series to a false trigger
    # in the pre-fix code. One stored point per (venue, asset, period).
    path = tmp_path / "funding.jsonl"
    base = _series("kraken", "BTC", 0.000002, 1.0, days=30)  # 1.75% annualized
    append_points(path, base)
    burst = [
        FundingPoint("kraken", "BTC", T0 + timedelta(days=29, minutes=m), 0.0002, 1.0)
        for m in range(0, 60, 2)  # 30 snapshots inside a single hour, 175% ann.
    ]
    # That hour already holds the day's regular hourly point, so the whole
    # burst is refused: the rich snapshots never reach the store at all.
    assert append_points(path, burst) == 0
    assert len(load_points(path)) == 720
    summary = summarize(load_points(path))[0]
    assert summary.triggered is False
    assert summary.mean_annualized < 0.08


def test_kraken_missing_rate_skips_only_that_perp_and_zero_rate_is_kept():
    payload = {"result": "success", "tickers": [
        {"symbol": "PF_XBTUSD"},  # no relativeFundingRate: skip, don't crash
        {"symbol": "PF_ETHUSD", "relativeFundingRate": 0},  # zero is real data
        {"symbol": "PF_DOGEUSD", "relativeFundingRate": 0.001},  # unmapped: ignore
    ]}
    points = parse_kraken_tickers(payload, T0)
    assert [(p.asset, p.rate) for p in points] == [("ETH", 0.0)]


def test_kraken_parse_does_not_gate_on_result_field():
    # Layering pin: parse_kraken_tickers is a pure parser; the success
    # check lives in fetch_kraken. Parsing an error-labelled payload with
    # well-formed tickers still yields the points.
    payload = {"result": "error", "tickers": [
        {"symbol": "PF_XBTUSD", "relativeFundingRate": 0.00001},
    ]}
    assert len(parse_kraken_tickers(payload, T0)) == 1


# ---------------------------------------------------------------------------
# 6. collect() failure isolation
# ---------------------------------------------------------------------------

def test_collect_survives_raising_and_malformed_fetchers():
    good = [FundingPoint("okx", "BTC", T0, 0.0001, 8.0)]

    def raiser():
        raise ConnectionError("venue exploded")

    def malformed():
        return None  # not a (points, error) tuple at all

    points, errors = monitor.collect([lambda: (good, None), raiser, malformed])
    assert points == good
    assert len(errors) == 2


# ---------------------------------------------------------------------------
# 7. summarize() edges
# ---------------------------------------------------------------------------

def test_summarize_empty_and_single_point():
    assert summarize([]) == []
    single = [FundingPoint("deribit", "ETH", T0, 0.00001, 1.0)]
    summary = summarize(single)[0]
    assert summary.points == 1
    assert summary.days_covered == 1
    assert summary.latest_annualized == pytest.approx(0.00001 * 24 * 365)
    assert summary.triggered is False  # rich rate, but 1 day is not sustained


def test_summarize_mixed_periods_same_group_annualizes_each_point():
    points = [
        FundingPoint("okx", "BTC", T0, 0.0001, 8.0),                    # 10.95%
        FundingPoint("okx", "BTC", T0 + timedelta(hours=8), 0.00001, 1.0),  # 8.76%
    ]
    summary = summarize(points)[0]
    assert summary.points == 2
    assert summary.mean_annualized == pytest.approx((0.1095 + 0.0876) / 2)


def test_negative_rate_series_summarizes_below_zero():
    points = _series("okx", "BTC", -0.5 / 1095, 8.0, days=30)  # pays -50% ann.
    summary = summarize(points)[0]
    assert summary.mean_annualized == pytest.approx(-0.5)
    assert summary.share_above_trigger == 0.0
    assert summary.triggered is False


def test_store_file_json_is_plain_and_loadable(tmp_path):
    # Every stored line must be strict JSON (no NaN/Infinity literals) so
    # any other consumer can read the store, not just this module.
    path = tmp_path / "funding.jsonl"
    append_points(path, _series("okx", "BTC", 0.0001, 8.0, days=2))
    for line in path.read_text().splitlines():
        row = json.loads(line, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
        assert math.isfinite(row["rate"])
