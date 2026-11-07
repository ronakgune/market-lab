"""Baseline tests for market_lab/crypto_carry.py (Stream 6 monitor):
venue parsers, annualization, store dedupe, trigger math, failure
isolation. Fixture payloads mirror the real venue response shapes.
"""
import importlib.util
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


def test_parse_okx_and_annualization():
    payload = {"code": "0", "data": [
        {"fundingRate": "0.0001", "fundingTime": str(int(T0.timestamp() * 1000)), "instId": "BTC-USDT-SWAP"},
    ]}
    points = parse_okx_history(payload, "BTC")
    assert len(points) == 1
    assert points[0].venue == "okx" and points[0].period_hours == 8.0
    # 0.01% per 8h -> 0.01% * 3 * 365 = 10.95% annualized
    assert points[0].annualized() == pytest.approx(0.1095)


def test_parse_deribit_hourly():
    payload = {"result": [
        {"timestamp": int(T0.timestamp() * 1000), "interest_1h": 0.000011, "interest_8h": 0.000088},
    ]}
    points = parse_deribit_history(payload, "ETH")
    assert points[0].rate == 0.000011 and points[0].period_hours == 1.0
    assert points[0].annualized() == pytest.approx(0.000011 * 24 * 365)


def test_parse_kraken_tickers_picks_perps_only():
    payload = {"result": "success", "tickers": [
        {"symbol": "PF_XBTUSD", "relativeFundingRate": 0.00001},
        {"symbol": "PF_ETHUSD", "relativeFundingRate": -0.00002},
        {"symbol": "FF_BTCUSD_261225", "relativeFundingRate": 0.5},  # dated future: ignored
    ]}
    points = parse_kraken_tickers(payload, T0)
    assert {(p.venue, p.asset) for p in points} == {("kraken", "BTC"), ("kraken", "ETH")}


def test_append_dedupes_and_tolerates_torn_lines(tmp_path):
    path = tmp_path / "funding.jsonl"
    point = FundingPoint("okx", "BTC", T0, 0.0001, 8.0)
    assert append_points(path, [point]) == 1
    assert append_points(path, [point]) == 0  # duplicate never written twice
    with path.open("a") as f:
        f.write('{"venue": "okx", "asset": "BTC", "ts": "2026-11-0')  # torn write
    assert len(load_points(path)) == 1  # torn line skipped, history intact


def _series(venue, asset, rate, period_hours, days, start=T0):
    per_day = int(24 / period_hours)
    return [
        FundingPoint(venue, asset, start + timedelta(hours=period_hours * i), rate, period_hours)
        for i in range(days * per_day)
    ]


def test_trigger_requires_sustained_mean_above_hurdle():
    rich = _series("kraken", "BTC", 0.00002, 1.0, days=30)   # 17.5% annualized, sustained
    summaries = summarize(rich)
    assert summaries[0].triggered is True

    spike = _series("kraken", "BTC", 0.0001, 1.0, days=3)    # huge but only 3 days
    summaries = summarize(spike)
    assert summaries[0].triggered is False
    assert summaries[0].mean_annualized > 0.08  # the mean alone would mislead

    poor = _series("kraken", "BTC", 0.000002, 1.0, days=30)  # 1.75% — below T-bills
    assert summarize(poor)[0].triggered is False


def test_collect_isolates_fetcher_failures():
    good = [FundingPoint("okx", "BTC", T0, 0.0001, 8.0)]
    fetchers = [
        lambda: (good, None),
        lambda: ([], "okx ETH: boom"),
        lambda: ([], "kraken: down"),
    ]
    points, errors = monitor.collect(fetchers)
    assert points == good
    assert len(errors) == 2
