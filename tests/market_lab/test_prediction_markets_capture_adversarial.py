"""Adversarial tests for the live venue capture adapter (Stream 5).

Attacks provenance integrity, venue-mapping traps, identity/format
safety, store interaction, and the chassis gate itself. The gate
holding — a live capture lands byte-exact in raw/ and quarantined,
never a record — is a REQUIREMENT here, not a bug.

No network: every fetcher is injected.
"""
import base64
import json
from pathlib import Path

import pytest

import market_lab.prediction_markets.capture as capture_module
from market_lab.prediction_markets.capture import (
    ADAPTER_VERSION,
    build_kalshi_descriptor,
    build_polymarket_descriptor,
    capture_kalshi_market,
    capture_polymarket_market,
)
from market_lab.prediction_markets.errors import CaptureError
from market_lab.prediction_markets.models import IDENTITY_RE, sha256_hex
from market_lab.prediction_markets.store import import_descriptor, load_records

RETRIEVED = "2026-11-06T19:00:00Z"
KALSHI_URL = "https://api.elections.kalshi.com/trade-api/v2/markets/KXTEST-26NOV07-ABC"

KALSHI_MARKET = {
    "ticker": "KXTEST-26NOV07-ABC",
    "title": "Test market?",
    "status": "active",
    "market_type": "binary",
    "yes_sub_title": "Yes outcome",
    "rules_primary": "Resolves YES if the test passes.",
    "rules_secondary": "Otherwise NO.",
    "settlement_source_name": "the test suite",
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


def _fetcher(payload, content_type="application/json"):
    raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()

    def fetch(url: str):
        return raw, content_type

    return fetch, raw


def _write_descriptor(tmp_path: Path, descriptor: dict, name="descriptor.json") -> Path:
    path = tmp_path / name
    path.write_bytes(json.dumps(descriptor, indent=1).encode())
    return path


def _no_store_state(root: Path) -> bool:
    return not root.exists() or not any(root.rglob("*"))


# ---------------------------------------------------------------------------
# 1. Provenance integrity
# ---------------------------------------------------------------------------

def test_tampered_raw_body_quarantines_and_is_never_archived_as_raw(tmp_path):
    raw = json.dumps({"market": KALSHI_MARKET}).encode()
    descriptor = build_kalshi_descriptor(raw, KALSHI_MARKET, KALSHI_URL, RETRIEVED)
    tampered_body = base64.b64decode(descriptor["raw_body_base64"]) + b" "
    descriptor["raw_body_base64"] = base64.b64encode(tampered_body).decode()
    # raw_sha256 left pointing at the GENUINE body: body and hash now disagree.
    store = tmp_path / "store"
    result = import_descriptor(store, _write_descriptor(tmp_path, descriptor))
    assert result["quarantined"] is True
    assert result["error_code"] == "PM0_SCHEMA"
    assert "raw_sha256" in result["message"]
    # The tampered body must not be archived as the venue's raw response.
    assert not (store / "raw").exists()
    assert load_records(store) == []
    # What IS preserved is the offending descriptor itself, in quarantine.
    inputs = list((store / "quarantine").rglob("input.bin"))
    assert len(inputs) == 1
    assert json.loads(inputs[0].read_bytes())["raw_body_base64"] == descriptor["raw_body_base64"]


def test_tampered_raw_sha256_field_quarantines(tmp_path):
    raw = json.dumps({"market": KALSHI_MARKET}).encode()
    descriptor = build_kalshi_descriptor(raw, KALSHI_MARKET, KALSHI_URL, RETRIEVED)
    descriptor["raw_sha256"] = sha256_hex(b"some other body entirely")
    store = tmp_path / "store"
    result = import_descriptor(store, _write_descriptor(tmp_path, descriptor))
    assert result["quarantined"] is True and "raw_sha256" in result["message"]
    assert not (store / "raw").exists()


def test_content_type_lie_is_recorded_faithfully_not_corrected(tmp_path):
    fetch, raw = _fetcher({"market": KALSHI_MARKET}, content_type="text/html")
    store = tmp_path / "store"
    result = capture_kalshi_market(KALSHI_MARKET["ticker"], fetcher=fetch, root=store,
                                   retrieved_at=RETRIEVED)
    assert result["quarantined"] is True
    archived = list((store / "raw").rglob("descriptor.json"))
    assert len(archived) == 1
    # The venue's (lying) content type is provenance: recorded as sent.
    assert json.loads(archived[0].read_text())["raw_content_type"] == "text/html"
    assert list((store / "raw").rglob("raw.bin"))[0].read_bytes() == raw


def test_kalshi_list_payload_fails_with_capture_error_and_writes_nothing(tmp_path):
    fetch, _ = _fetcher([{"market": KALSHI_MARKET}])  # valid JSON, wrong top-level type
    store = tmp_path / "store"
    with pytest.raises(CaptureError, match="not a JSON object"):
        capture_kalshi_market("KXTEST-26NOV07-ABC", fetcher=fetch, root=store,
                              retrieved_at=RETRIEVED)
    assert _no_store_state(store)


def test_kalshi_market_field_of_wrong_type_fails_and_writes_nothing(tmp_path):
    fetch, _ = _fetcher({"market": ["not", "a", "dict"]})
    store = tmp_path / "store"
    with pytest.raises(CaptureError, match="no market object"):
        capture_kalshi_market("KXTEST-26NOV07-ABC", fetcher=fetch, root=store,
                              retrieved_at=RETRIEVED)
    assert _no_store_state(store)


@pytest.mark.parametrize("venue", ["kalshi", "polymarket"])
def test_truncated_json_fails_before_any_store_write(tmp_path, venue):
    payload = {"market": KALSHI_MARKET} if venue == "kalshi" else POLY_MARKET
    raw = json.dumps(payload).encode()
    fetch, _ = _fetcher(raw[: len(raw) // 2])  # severed mid-document
    store = tmp_path / "store"
    driver = capture_kalshi_market if venue == "kalshi" else capture_polymarket_market
    with pytest.raises(json.JSONDecodeError):
        driver("whatever", fetcher=fetch, root=store, retrieved_at=RETRIEVED)
    assert _no_store_state(store)


def test_polymarket_payload_without_id_fails_and_writes_nothing(tmp_path):
    fetch, _ = _fetcher({"question": "no id here", "outcomePrices": "[\"0.5\", \"0.5\"]"})
    store = tmp_path / "store"
    with pytest.raises(CaptureError, match="no market object"):
        capture_polymarket_market("12345", fetcher=fetch, root=store, retrieved_at=RETRIEVED)
    assert _no_store_state(store)


# ---------------------------------------------------------------------------
# 2. Kalshi mapping traps
# ---------------------------------------------------------------------------

def test_kalshi_zero_book_builds_and_quarantines_on_class_not_prices(tmp_path):
    zeroed = {**KALSHI_MARKET,
              "yes_bid_dollars": "0.0000", "yes_ask_dollars": "0.0000",
              "no_bid_dollars": "0.0000", "no_ask_dollars": "0.0000",
              "last_price_dollars": "0.0000"}
    raw = json.dumps({"market": zeroed}).encode()
    descriptor = build_kalshi_descriptor(raw, zeroed, KALSHI_URL, RETRIEVED)
    yes, no = descriptor["market"]["outcomes"]
    assert (yes["best_bid"], yes["best_ask"], yes["last"]) == ("0.000000", "0.000000", "0.000000")
    assert (no["best_bid"], no["best_ask"]) == ("0.000000", "0.000000")
    store = tmp_path / "store"
    result = import_descriptor(store, _write_descriptor(tmp_path, descriptor))
    assert result["quarantined"] is True and "source_class" in result["message"]


def test_kalshi_crossed_book_is_preserved_faithfully_never_repaired(tmp_path):
    crossed = {**KALSHI_MARKET, "yes_bid_dollars": "0.6000", "yes_ask_dollars": "0.4000"}
    fetch, raw = _fetcher({"market": crossed})
    store = tmp_path / "store"
    result = capture_kalshi_market(crossed["ticker"], fetcher=fetch, root=store,
                                   retrieved_at=RETRIEVED)
    assert result["quarantined"] is True
    # Rejection is the source-class gate, reached before any book sanity check.
    assert "source_class" in result["message"]
    # The glitching venue's data is archived byte-exact, crossed book and all.
    assert list((store / "raw").rglob("raw.bin"))[0].read_bytes() == raw
    archived = json.loads(list((store / "raw").rglob("descriptor.json"))[0].read_text())
    yes = archived["market"]["outcomes"][0]
    assert (yes["best_bid"], yes["best_ask"]) == ("0.600000", "0.400000")


def test_kalshi_subcent_tick_and_prices_round_trip():
    subcent = {**KALSHI_MARKET,
               "price_ranges": [{"start": "0.0000", "end": "1.0000", "step": "0.0010"}],
               "yes_bid_dollars": "0.1230", "yes_ask_dollars": "0.1240"}
    raw = json.dumps({"market": subcent}).encode()
    descriptor = build_kalshi_descriptor(raw, subcent, KALSHI_URL, RETRIEVED)
    assert descriptor["market"]["tick_size"] == "0.001000"
    yes = descriptor["market"]["outcomes"][0]
    assert (yes["best_bid"], yes["best_ask"]) == ("0.123000", "0.124000")


@pytest.mark.parametrize("ranges", [None, [], [{"start": "0.0000", "end": "1.0000", "step": ""}]])
def test_kalshi_missing_or_blank_price_ranges_fall_back_to_cent_tick(ranges):
    market = {**KALSHI_MARKET, "price_ranges": ranges}
    raw = json.dumps({"market": market}).encode()
    descriptor = build_kalshi_descriptor(raw, market, KALSHI_URL, RETRIEVED)
    assert descriptor["market"]["tick_size"] == "0.010000"


def test_kalshi_missing_sizes_build_as_none_and_no_last_is_none():
    market = {k: v for k, v in KALSHI_MARKET.items()
              if not k.endswith("_size_fp")}
    raw = json.dumps({"market": market}).encode()
    descriptor = build_kalshi_descriptor(raw, market, KALSHI_URL, RETRIEVED)
    yes, no = descriptor["market"]["outcomes"]
    assert yes["bid_size"] is None and yes["ask_size"] is None
    assert no["bid_size"] is None and no["ask_size"] is None
    assert yes["last"] == "0.420000" and no["last"] is None  # last is a YES-only field


def test_kalshi_missing_updated_time_rules_version_falls_back_deterministically():
    market = {k: v for k, v in KALSHI_MARKET.items() if k != "updated_time"}
    raw = json.dumps({"market": market}).encode()
    first = build_kalshi_descriptor(raw, market, KALSHI_URL, RETRIEVED)
    second = build_kalshi_descriptor(raw, market, KALSHI_URL, RETRIEVED)
    assert first["provider_updated_at_utc"] is None
    # Fallback is the first eight digits of the retrieval timestamp: deterministic.
    assert first["market"]["rules_version"] == "20261106" == second["market"]["rules_version"]
    assert first["terms_version"].endswith("20261106")


@pytest.mark.parametrize("venue_status, expected", [
    ("initialized", "PENDING"), ("inactive", "HALTED"), ("closed", "CLOSED"),
    ("determined", "RESOLVED"), ("disputed", "DISPUTED"),
    ("finalized", "RESOLVED"), ("amended", "OPEN"),
])
def test_kalshi_status_mapping_variants(venue_status, expected):
    market = {**KALSHI_MARKET, "status": venue_status}
    raw = json.dumps({"market": market}).encode()
    descriptor = build_kalshi_descriptor(raw, market, KALSHI_URL, RETRIEVED)
    assert descriptor["market"]["status"] == expected


# ---------------------------------------------------------------------------
# 3. Polymarket complement math
# ---------------------------------------------------------------------------

def test_polymarket_yes_ask_zero_puts_no_bid_at_payout_and_vice_versa():
    raw = json.dumps({**POLY_MARKET, "bestBid": 1, "bestAsk": 0}).encode()
    descriptor = build_polymarket_descriptor(raw, {**POLY_MARKET, "bestBid": 1, "bestAsk": 0},
                                             "https://x/markets/1", RETRIEVED)
    yes, no = descriptor["market"]["outcomes"]
    assert yes["best_bid"] == "1.000000" and yes["best_ask"] == "0.000000"
    assert no["best_bid"] == "1.000000"  # 1 - yes_ask(0): at the payout boundary, not above it
    assert no["best_ask"] == "0.000000"  # 1 - yes_bid(1)


def test_polymarket_absent_best_bid_ask_propagate_nulls_nothing_invented():
    market = {k: v for k, v in POLY_MARKET.items() if k not in ("bestBid", "bestAsk")}
    raw = json.dumps(market).encode()
    descriptor = build_polymarket_descriptor(raw, market, "https://x/markets/1", RETRIEVED)
    yes, no = descriptor["market"]["outcomes"]
    assert (yes["best_bid"], yes["best_ask"]) == (None, None)
    assert (no["best_bid"], no["best_ask"]) == (None, None)
    assert (yes["last"], no["last"]) == ("0.420000", "0.580000")  # lasts still come from outcomePrices


def test_polymarket_missing_outcome_prices_raises_capture_error():
    market = {k: v for k, v in POLY_MARKET.items() if k != "outcomePrices"}
    raw = json.dumps(market).encode()
    with pytest.raises(CaptureError, match="no outcomePrices"):
        build_polymarket_descriptor(raw, market, "https://x/markets/1", RETRIEVED)


@pytest.mark.parametrize("bad_prices", ["not-json", '["0.5"]', '["0.4", "0.6", "0.2"]',
                                       '["abc", "0.5"]', '"0.42"'])
def test_polymarket_malformed_outcome_prices_raise_capture_error(bad_prices):
    market = {**POLY_MARKET, "outcomePrices": bad_prices}
    raw = json.dumps(market).encode()
    with pytest.raises(CaptureError):
        build_polymarket_descriptor(raw, market, "https://x/markets/1", RETRIEVED)


def test_polymarket_missing_outcome_prices_capture_writes_nothing(tmp_path):
    market = {k: v for k, v in POLY_MARKET.items() if k != "outcomePrices"}
    fetch, _ = _fetcher(market)
    store = tmp_path / "store"
    with pytest.raises(CaptureError, match="no outcomePrices"):
        capture_polymarket_market("12345", fetcher=fetch, root=store, retrieved_at=RETRIEVED)
    assert _no_store_state(store)


@pytest.mark.parametrize("closed, uma, expected", [
    (True, "resolved", "RESOLVED"),
    (True, None, "CLOSED"),
    (False, None, "PENDING"),  # closed False + active False below; see test body
])
def test_polymarket_status_mapping_variants(closed, uma, expected):
    market = {**POLY_MARKET, "closed": closed, "active": not closed and expected == "OPEN"}
    if uma is not None:
        market["umaResolutionStatus"] = uma
    if closed:
        market["active"] = False
    raw = json.dumps(market).encode()
    descriptor = build_polymarket_descriptor(raw, market, "https://x/markets/1", RETRIEVED)
    assert descriptor["market"]["status"] == expected


def test_polymarket_gamma_list_payload_is_accepted(tmp_path):
    fetch, _ = _fetcher([POLY_MARKET])  # Gamma collections return a list
    store = tmp_path / "store"
    result = capture_polymarket_market("12345", fetcher=fetch, root=store,
                                       retrieved_at=RETRIEVED)
    assert result["quarantined"] is True and result["provider_id"] == "polymarket"


# ---------------------------------------------------------------------------
# 4. Identity / format traps
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_ticker", ["KX:FOO", "KX/FOO", "KX FOO", "KX;FOO"])
def test_kalshi_ticker_outside_identity_alphabet_is_rejected(bad_ticker):
    market = {**KALSHI_MARKET, "ticker": bad_ticker}
    raw = json.dumps({"market": market}).encode()
    with pytest.raises(CaptureError, match="not descriptor-identity-safe"):
        build_kalshi_descriptor(raw, market, KALSHI_URL, RETRIEVED)


def test_kalshi_unsafe_ticker_capture_writes_nothing(tmp_path):
    market = {**KALSHI_MARKET, "ticker": "KX:FOO/BAR"}
    fetch, _ = _fetcher({"market": market})
    store = tmp_path / "store"
    with pytest.raises(CaptureError, match="not descriptor-identity-safe"):
        capture_kalshi_market("KX:FOO/BAR", fetcher=fetch, root=store, retrieved_at=RETRIEVED)
    assert _no_store_state(store)


def test_polymarket_unsafe_id_is_rejected():
    market = {**POLY_MARKET, "id": "12:34"}
    raw = json.dumps(market).encode()
    with pytest.raises(CaptureError, match="not descriptor-identity-safe"):
        build_polymarket_descriptor(raw, market, "https://x/markets/1", RETRIEVED)


def test_every_built_outcome_id_and_market_id_is_identity_safe():
    for status in ("active", "closed", "determined"):
        market = {**KALSHI_MARKET, "status": status}
        raw = json.dumps({"market": market}).encode()
        descriptor = build_kalshi_descriptor(raw, market, KALSHI_URL, RETRIEVED)
        assert IDENTITY_RE.match(descriptor["provider_market_id"])
        for outcome in descriptor["market"]["outcomes"]:
            assert IDENTITY_RE.match(outcome["provider_outcome_id"])


def test_kalshi_times_with_microseconds_are_stripped_to_schema_seconds():
    market = {**KALSHI_MARKET,
              "open_time": "2026-11-01T00:00:00.999999Z",
              "close_time": "2026-11-21T17:30:00.123456Z"}
    raw = json.dumps({"market": market}).encode()
    descriptor = build_kalshi_descriptor(raw, market, KALSHI_URL, RETRIEVED)
    assert descriptor["market"]["opens_at_utc"] == "2026-11-01T00:00:00Z"
    assert descriptor["market"]["closes_at_utc"] == "2026-11-21T17:30:00Z"


def test_kalshi_times_with_utc_offset_convert_to_z():
    market = {**KALSHI_MARKET, "close_time": "2026-11-21T17:30:00+05:30"}
    raw = json.dumps({"market": market}).encode()
    descriptor = build_kalshi_descriptor(raw, market, KALSHI_URL, RETRIEVED)
    assert descriptor["market"]["closes_at_utc"] == "2026-11-21T12:00:00Z"


def test_kalshi_naive_time_is_assumed_utc():
    market = {**KALSHI_MARKET, "close_time": "2026-11-21T17:30:00"}
    raw = json.dumps({"market": market}).encode()
    descriptor = build_kalshi_descriptor(raw, market, KALSHI_URL, RETRIEVED)
    assert descriptor["market"]["closes_at_utc"] == "2026-11-21T17:30:00Z"


def test_polymarket_end_date_with_offset_normalizes_to_z():
    market = {**POLY_MARKET, "endDate": "2027-01-01T05:00:00-05:00"}
    raw = json.dumps(market).encode()
    descriptor = build_polymarket_descriptor(raw, market, "https://x/markets/1", RETRIEVED)
    assert descriptor["market"]["closes_at_utc"] == "2027-01-01T10:00:00Z"
    assert descriptor["market"]["scheduled_resolution_at_utc"] == "2027-01-01T10:00:00Z"


# ---------------------------------------------------------------------------
# 5. Store interaction
# ---------------------------------------------------------------------------

def test_recapturing_the_same_market_is_idempotent(tmp_path):
    fetch, _ = _fetcher({"market": KALSHI_MARKET})
    store = tmp_path / "store"
    first = capture_kalshi_market(KALSHI_MARKET["ticker"], fetcher=fetch, root=store,
                                  retrieved_at=RETRIEVED)
    second = capture_kalshi_market(KALSHI_MARKET["ticker"], fetcher=fetch, root=store,
                                   retrieved_at=RETRIEVED)
    assert first["quarantined"] is True and second["quarantined"] is True
    assert first["descriptor_sha256"] == second["descriptor_sha256"]
    assert len(list((store / "raw").rglob("raw.bin"))) == 1
    assert len(list((store / "raw").rglob("descriptor.json"))) == 1
    assert len(list((store / "quarantine").rglob("error.json"))) == 1
    assert load_records(store) == []


def test_conflicting_descriptor_for_same_raw_hash_never_overwrites(tmp_path):
    fetch, raw = _fetcher({"market": KALSHI_MARKET})
    store = tmp_path / "store"
    capture_kalshi_market(KALSHI_MARKET["ticker"], fetcher=fetch, root=store,
                          retrieved_at=RETRIEVED)
    staged_a = json.dumps(
        build_kalshi_descriptor(raw, KALSHI_MARKET, KALSHI_URL, RETRIEVED),
        indent=1).encode()
    # Same raw body (same raw_sha256), different descriptor bytes (later retrieval).
    descriptor_b = build_kalshi_descriptor(raw, KALSHI_MARKET, KALSHI_URL,
                                           "2026-11-06T20:00:00Z")
    assert json.dumps(descriptor_b, indent=1).encode() != staged_a
    result = import_descriptor(store, _write_descriptor(tmp_path, descriptor_b, "b.json"))
    # The conflict surfaces as a quarantine, never an overwrite, never a record.
    assert result["quarantined"] is True and result["error_code"] == "PM0_SCHEMA"
    archived = list((store / "raw").rglob("descriptor.json"))
    assert len(archived) == 1 and archived[0].read_bytes() == staged_a
    assert len(list((store / "quarantine").rglob("error.json"))) == 2
    assert load_records(store) == []


def test_two_different_markets_do_not_collide(tmp_path):
    other = {**KALSHI_MARKET, "ticker": "KXTEST-26NOV08-XYZ",
             "rules_primary": "A different market entirely."}
    store = tmp_path / "store"
    for market in (KALSHI_MARKET, other):
        fetch, _ = _fetcher({"market": market})
        result = capture_kalshi_market(market["ticker"], fetcher=fetch, root=store,
                                       retrieved_at=RETRIEVED)
        assert result["quarantined"] is True
    assert len(list((store / "raw").rglob("raw.bin"))) == 2
    assert len(list((store / "quarantine").rglob("error.json"))) == 2
    assert load_records(store) == []


def test_staging_file_removed_and_exception_propagates_when_import_raises(tmp_path, monkeypatch):
    seen = {}

    def exploding_import(root, input_path):
        seen["path"] = Path(input_path)
        raise RuntimeError("store exploded")

    monkeypatch.setattr(capture_module, "import_descriptor", exploding_import)
    fetch, _ = _fetcher({"market": KALSHI_MARKET})
    with pytest.raises(RuntimeError, match="store exploded"):
        capture_kalshi_market(KALSHI_MARKET["ticker"], fetcher=fetch, root=tmp_path,
                              retrieved_at=RETRIEVED)
    # The finally in _capture must remove the staging file even on a
    # non-PredictionMarketError, and the error must not be swallowed.
    assert not seen["path"].exists()


def test_staging_file_removed_on_successful_import(tmp_path, monkeypatch):
    seen = {}

    def fake_import(root, input_path):
        seen["path"] = Path(input_path)
        assert Path(input_path).exists()
        return {"status": "quarantined", "quarantined": True}

    monkeypatch.setattr(capture_module, "import_descriptor", fake_import)
    fetch, _ = _fetcher({"market": KALSHI_MARKET})
    result = capture_kalshi_market(KALSHI_MARKET["ticker"], fetcher=fetch, root=tmp_path,
                                   retrieved_at=RETRIEVED)
    assert result["quarantined"] is True and result["provider_id"] == "kalshi"
    assert not seen["path"].exists()


# ---------------------------------------------------------------------------
# 6. The gate itself
# ---------------------------------------------------------------------------

def _research_grade_live_descriptor() -> tuple[dict, bytes]:
    raw = json.dumps({"market": KALSHI_MARKET}).encode()
    descriptor = build_kalshi_descriptor(raw, KALSHI_MARKET, KALSHI_URL, RETRIEVED)
    # Hand the capture everything admissibility asks for: fees, rules URL,
    # resolution source (already in the fixture), complete two-sided quotes
    # WITH sizes (already in the fixture). Nothing left to want — except
    # that it is a live capture.
    descriptor["market"].update({
        "fee_schedule_id": "kalshi-fee-schedule",
        "fee_schedule_version": "2026-01",
        "fee_schedule_url": "https://kalshi.com/fees",
        "rules_url": "https://kalshi.com/rules",
        "void_policy": "voided markets settle at fair value",
    })
    return descriptor, raw


def test_research_grade_live_capture_is_kept_out_by_source_class_alone(tmp_path):
    descriptor, _ = _research_grade_live_descriptor()
    assert descriptor["source_class"] == "live_capture"  # the ONLY defect left
    store = tmp_path / "store"
    result = import_descriptor(store, _write_descriptor(tmp_path, descriptor))
    assert result["quarantined"] is True
    assert result["error_code"] == "PM0_SCHEMA"
    assert "source_class" in result["message"]
    assert load_records(store) == []
    assert not (store / "normalized").exists()


def test_flipped_source_class_normalizes_but_paper_synthetic_gate_still_blocks(tmp_path):
    # Parent decision (Nov 6, 2026): the builder now emits the chassis's
    # canonical six-place decimals, so decimal format is no longer an
    # unintended second barrier. With source_class flipped by hand, a
    # research-grade capture NORMALIZES — the two remaining barriers are
    # the deliberate ones: source_class itself, and the paper engine's
    # synthetic-only gate, which must still refuse this record.
    from decimal import Decimal

    from market_lab.prediction_markets import paper
    from market_lab.prediction_markets.errors import PaperNotSyntheticError

    descriptor, _ = _research_grade_live_descriptor()
    descriptor["source_class"] = "frozen_fixture"
    store = tmp_path / "store"
    result = import_descriptor(store, _write_descriptor(tmp_path, descriptor))
    assert result["quarantined"] is False
    assert result["admissibility"] == "RESEARCH_ADMISSIBLE"
    records = load_records(store)
    assert len(records) == 1 and records[0].provider_id == "kalshi"

    paper.init(store, Decimal("1000"), RETRIEVED)
    with pytest.raises(PaperNotSyntheticError):
        paper.buy(store, result["market_key"], "YES", Decimal("1"),
                  Decimal("0.40"), Decimal("0"), RETRIEVED)


def test_builders_only_ever_emit_live_capture_for_the_real_venue():
    for status in ("initialized", "active", "inactive", "closed", "determined", "disputed"):
        market = {**KALSHI_MARKET, "status": status}
        raw = json.dumps({"market": market}).encode()
        d = build_kalshi_descriptor(raw, market, KALSHI_URL, RETRIEVED)
        assert d["source_class"] == "live_capture"
        assert (d["provider_id"], d["legal_entity_id"], d["venue_id"]) == ("kalshi",) * 3
        assert d["capture_adapter_version"] == ADAPTER_VERSION
        assert d["schema_version"] == "mlab-pm-frozen.v1"
    for closed, active in ((False, True), (True, False), (False, False)):
        market = {**POLY_MARKET, "closed": closed, "active": active}
        raw = json.dumps(market).encode()
        d = build_polymarket_descriptor(raw, market, "https://x/markets/1", RETRIEVED)
        assert d["source_class"] == "live_capture"
        assert d["provider_id"] == "polymarket" and d["legal_entity_id"] == "polymarket"
        assert d["venue_id"] == "polymarket-clob"
        assert d["capture_adapter_version"] == ADAPTER_VERSION
