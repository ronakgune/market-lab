"""Baseline tests for the live venue capture adapter (Stream 5):
descriptor construction from venue payloads, byte-exact raw body, and
the chassis gate — a live capture must land in the raw archive and
quarantine, never normalize into an engine-admissible record.
"""
import base64
import json
from pathlib import Path

import pytest

from market_lab.prediction_markets.capture import (
    build_kalshi_descriptor,
    build_polymarket_descriptor,
    capture_kalshi_market,
    capture_polymarket_market,
)
from market_lab.prediction_markets.models import sha256_hex, strict_raw_body
from market_lab.prediction_markets.store import load_records

RETRIEVED = "2026-11-06T19:00:00Z"

KALSHI_MARKET = {
    "ticker": "KXTEST-26NOV07-ABC",
    "title": "Test market?",
    "status": "active",
    "market_type": "binary",
    "yes_sub_title": "Yes outcome",
    "rules_primary": "Resolves YES if the test passes.",
    "rules_secondary": "Otherwise NO.",
    "open_time": "2026-11-01T00:00:00Z",
    "close_time": "2026-11-21T17:30:00Z",
    "expected_expiration_time": "2026-11-21T18:00:00Z",
    "updated_time": "2026-11-06T10:00:00.123456Z",
    "notional_value_dollars": "1.0000",
    "price_ranges": [{"start": "0.0000", "end": "1.0000", "step": "0.0100"}],
    "yes_bid_dollars": "0.4000", "yes_ask_dollars": "0.4400",
    "no_bid_dollars": "0.5600", "no_ask_dollars": "0.6000",
    "last_price_dollars": "0.4200",
    "yes_bid_size_fp": "120.00", "yes_ask_size_fp": "85.00",
    "no_bid_size_fp": "85.00", "no_ask_size_fp": "120.00",
}

POLY_MARKET = {
    "id": "12345",
    "question": "Will the test pass?",
    "description": "This market resolves Yes if the test passes.",
    "outcomes": "[\"Yes\", \"No\"]",
    "outcomePrices": "[\"0.42\", \"0.58\"]",
    "bestBid": 0.41, "bestAsk": 0.43,
    "orderPriceMinTickSize": 0.01,
    "active": True, "closed": False,
    "startDate": "2026-11-01T00:00:00Z",
    "endDate": "2027-01-01T05:00:00Z",
    "resolutionSource": "the test suite",
}


def _fetcher(payload: dict):
    raw = json.dumps(payload).encode()

    def fetch(url: str):
        return raw, "application/json"

    return fetch, raw


def test_kalshi_descriptor_maps_quotes_sizes_and_rules():
    raw = json.dumps({"market": KALSHI_MARKET}).encode()
    d = build_kalshi_descriptor(raw, KALSHI_MARKET, "https://x/markets/T", RETRIEVED)
    assert d["source_class"] == "live_capture" and d["provider_id"] == "kalshi"
    assert d["raw_sha256"] == sha256_hex(raw)
    assert strict_raw_body(d) == raw  # byte-exact round trip
    m = d["market"]
    assert m["status"] == "OPEN" and m["tick_size"] == "0.010000"
    assert m["closes_at_utc"] == "2026-11-21T17:30:00Z"
    yes, no = m["outcomes"]
    assert (yes["best_bid"], yes["best_ask"], yes["bid_size"]) == ("0.400000", "0.440000", "120.000000")
    assert (no["best_bid"], no["best_ask"]) == ("0.560000", "0.600000")
    assert "test passes" in m["rules_text"] and "Otherwise NO" in m["rules_text"]
    assert d["provider_updated_at_utc"] == "2026-11-06T10:00:00Z"  # fractions stripped to schema format


def test_kalshi_unknown_status_is_a_capture_error():
    raw = b"{}"
    bad = {**KALSHI_MARKET, "status": "some_future_status"}
    with pytest.raises(Exception, match="unmapped market status"):
        build_kalshi_descriptor(raw, bad, "https://x", RETRIEVED)


def test_polymarket_descriptor_derives_no_side_as_complement():
    raw = json.dumps(POLY_MARKET).encode()
    d = build_polymarket_descriptor(raw, POLY_MARKET, "https://x/markets/1", RETRIEVED)
    yes, no = d["market"]["outcomes"]
    assert (yes["best_bid"], yes["best_ask"], yes["last"]) == ("0.410000", "0.430000", "0.420000")
    assert (no["best_bid"], no["best_ask"], no["last"]) == ("0.570000", "0.590000", "0.580000")
    assert yes["bid_size"] is None  # Gamma carries no sizes; never fabricated
    assert d["market"]["status"] == "OPEN"


def test_live_capture_quarantines_at_the_gate_but_raw_is_archived(tmp_path: Path):
    fetch, raw = _fetcher({"market": KALSHI_MARKET})
    result = capture_kalshi_market("KXTEST-26NOV07-ABC", fetcher=fetch, root=tmp_path,
                                   retrieved_at=RETRIEVED)
    assert result["source_class"] == "live_capture"
    assert result["quarantined"] is True  # the source-class gate holds
    assert load_records(tmp_path) == []  # nothing became an engine record
    archived = list((tmp_path / "raw").rglob("raw.bin"))
    assert len(archived) == 1 and archived[0].read_bytes() == raw  # byte-exact archive
    errors = list((tmp_path / "quarantine").rglob("error.json"))
    assert len(errors) == 1 and "source_class" in errors[0].read_text()


def test_polymarket_capture_end_to_end_quarantine(tmp_path: Path):
    fetch, _raw = _fetcher(POLY_MARKET)
    result = capture_polymarket_market("12345", fetcher=fetch, root=tmp_path,
                                       retrieved_at=RETRIEVED)
    assert result["quarantined"] is True and result["provider_id"] == "polymarket"


def test_capture_rejects_a_payload_without_a_market(tmp_path: Path):
    fetch, _raw = _fetcher({"unexpected": True})
    with pytest.raises(Exception, match="no market object"):
        capture_kalshi_market("NOPE", fetcher=fetch, root=tmp_path, retrieved_at=RETRIEVED)
