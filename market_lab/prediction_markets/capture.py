"""Live venue capture adapter (Stream 5, step 1).

Freezes a prediction market fetched from a venue's public API (via an
injected fetcher — this package is network-free by design) into the
chassis's descriptor format with truthful provenance
(source_class "live_capture", the venue's real provider/venue ids,
the exact response bytes as the raw body).

The chassis gate is intentional and untouched: normalize_descriptor
accepts only synthetic fixtures, and the paper engine accepts only
those. A live capture therefore lands in the store's raw archive and
is quarantined by the normalizer with the source-class error — the
capture is preserved byte-exact for research (spread/depth history
for the eventual capped maker experiment) while remaining
structurally inadmissible to any engine. Promoting a capture is a
separate, deliberate chassis decision, not something this module does.

Venue notes (checked live Nov 6, 2026):
- Kalshi: one GET returns the complete two-sided quote with sizes,
  the tick (price_ranges step), and the rules text — a complete-quote
  grade capture.
- Polymarket: the Gamma market object carries market-level best
  bid/ask for YES; NO quotes are the arithmetic complements (binary
  book identity). Sizes are not in the Gamma object, so Polymarket
  captures are discovery-grade (no sizes) until a multi-body raw
  format exists for the CLOB book endpoints.
"""

from __future__ import annotations

import base64
import json
import re
import tempfile
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Callable

from market_lab.prediction_markets.config import prediction_data_root
from market_lab.prediction_markets.errors import CaptureError
from market_lab.prediction_markets.models import sha256_hex
from market_lab.prediction_markets.store import import_descriptor

ADAPTER_VERSION = "venue-capture@1"
KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
POLYMARKET_GAMMA = "https://gamma-api.polymarket.com"

# The prediction_markets package is deliberately network-free (its
# boundary tests forbid urllib/socket/http imports): the actual HTTP
# fetch lives in the caller (scripts/capture_prediction_market.py) and
# is injected here as a Fetcher. Everything in this module is pure
# given the fetched bytes.
Fetcher = Callable[[str], tuple[bytes, str]]  # url -> (body, content_type)


def _rfc3339z(value: str | None) -> str | None:
    if not value:
        return None
    text = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _now_z() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _rules_version(stamp: str | None, fallback: str) -> str:
    digits = "".join(ch for ch in (stamp or "") if ch.isdigit())
    return digits[:14] if len(digits) >= 8 else "".join(ch for ch in fallback if ch.isdigit())[:8]


def _canon(value: Decimal) -> str:
    """Canonical decimal string in the frozen format's six-place form
    (models.parse_decimal demands ^-?\\d+\\.\\d{6}$). Compact forms would
    make every archived capture fail a future deliberate promotion on
    format alone — an unintended second barrier behind the source-class
    gate."""
    return format(value.quantize(Decimal("0.000001")), "f")


def _dec_str(value) -> str | None:
    if value is None or value == "":
        return None
    return _canon(Decimal(str(value)))


# The chassis validates provider/market/outcome ids against this exact
# alphabet (models.IDENTITY_RE, plus an explicit ':' ban). A venue id
# outside it can never be represented in a descriptor, so refuse loudly
# instead of emitting ids the chassis would reject on sight.
_IDENTITY_SAFE_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _require_identity_safe(value: str, field: str) -> str:
    if not _IDENTITY_SAFE_RE.match(value) or ":" in value:
        raise CaptureError(f"{field} {value!r} is not descriptor-identity-safe")
    return value


# ---------------------------------------------------------------------------
# Kalshi
# ---------------------------------------------------------------------------

_KALSHI_STATUS = {
    "initialized": "PENDING", "active": "OPEN", "inactive": "HALTED",
    "closed": "CLOSED", "determined": "RESOLVED", "disputed": "DISPUTED",
    "finalized": "RESOLVED", "amended": "OPEN",
}


def build_kalshi_descriptor(raw: bytes, market: dict, url: str, retrieved_at: str,
                            content_type: str = "application/json") -> dict:
    ticker = _require_identity_safe(market["ticker"], "kalshi ticker")
    status = _KALSHI_STATUS.get(market.get("status", ""))
    if status is None:
        raise CaptureError(f"kalshi: unmapped market status {market.get('status')!r}")
    rules_text = market.get("rules_primary") or ""
    if market.get("rules_secondary"):
        rules_text = rules_text + "\n" + market["rules_secondary"]
    rules_version = _rules_version(market.get("updated_time"), retrieved_at)
    tick = _canon(Decimal("0.01"))
    ranges = market.get("price_ranges") or []
    if ranges and ranges[0].get("step"):
        tick = _dec_str(ranges[0]["step"])

    def outcome(label: str, prefix: str) -> dict:
        return {
            "provider_outcome_id": f"{ticker}-{label}",
            "label": label,
            "best_bid": _dec_str(market.get(f"{prefix}_bid_dollars")),
            "best_ask": _dec_str(market.get(f"{prefix}_ask_dollars")),
            "last": _dec_str(market.get("last_price_dollars")) if label == "YES" else None,
            "bid_size": _dec_str(market.get(f"{prefix}_bid_size_fp")),
            "ask_size": _dec_str(market.get(f"{prefix}_ask_size_fp")),
        }

    return {
        "schema_version": "mlab-pm-frozen.v1",
        "source_class": "live_capture",
        "capture_adapter_version": ADAPTER_VERSION,
        "provider_id": "kalshi",
        "legal_entity_id": "kalshi",
        "venue_id": "kalshi",
        "api_surface_id": "trade-api-v2",
        "jurisdiction_eligibility_status": "US_ELIGIBLE",
        "terms_version": f"kalshi-market-rules-{rules_version}",
        "terms_sha256": sha256_hex(rules_text.encode("utf-8")),
        "provider_market_id": ticker,
        "request_method": "GET",
        "request_url": url,
        "retrieved_at_utc": retrieved_at,
        "observed_at_utc": retrieved_at,
        "provider_updated_at_utc": _rfc3339z(market.get("updated_time")),
        "raw_content_type": content_type,
        "raw_sha256": sha256_hex(raw),
        "raw_body_base64": base64.b64encode(raw).decode("ascii"),
        "retention_note": "Live capture from Kalshi public API; raw body preserved byte-exact. "
                          "Not engine-admissible: the chassis normalizer admits only synthetic "
                          "fixtures (source_class gate), so imports quarantine by design.",
        "market": {
            "title": market.get("title") or ticker,
            "description": market.get("yes_sub_title") or None,
            "market_type": "binary",
            "status": status,
            "currency": "USD",
            "payout_unit": _dec_str(market.get("notional_value_dollars")) or _canon(Decimal("1")),
            "tick_size": tick,
            "opens_at_utc": _rfc3339z(market.get("open_time")),
            "closes_at_utc": _rfc3339z(market.get("close_time")),
            "scheduled_resolution_at_utc": _rfc3339z(market.get("expected_expiration_time")),
            "rules_version": rules_version,
            "rules_text": rules_text or None,
            "rules_url": None,
            "resolution_source_description": market.get("settlement_source_name") or None,
            "resolution_source_url": None,
            "void_policy": None,
            "fee_schedule_id": None,
            "fee_schedule_version": None,
            "fee_schedule_url": None,
            "outcomes": [outcome("YES", "yes"), outcome("NO", "no")],
        },
    }


# ---------------------------------------------------------------------------
# Polymarket (Gamma; discovery-grade — no sizes in this payload)
# ---------------------------------------------------------------------------

def build_polymarket_descriptor(raw: bytes, market: dict, url: str, retrieved_at: str,
                                content_type: str = "application/json") -> dict:
    market_id = _require_identity_safe(str(market["id"]), "polymarket market id")
    raw_prices = market.get("outcomePrices")
    if not raw_prices:
        raise CaptureError(
            f"polymarket: market {market_id} has no outcomePrices; refusing to fabricate prices")
    try:
        prices = json.loads(raw_prices)
    except (TypeError, json.JSONDecodeError) as exc:
        raise CaptureError(f"polymarket: market {market_id} outcomePrices is not valid JSON") from exc
    if not isinstance(prices, list) or len(prices) != 2:
        raise CaptureError(
            f"polymarket: market {market_id} outcomePrices must be a two-element list")
    try:
        yes_last = Decimal(str(prices[0]))
        Decimal(str(prices[1]))  # the NO price must at least be a real number
    except InvalidOperation as exc:
        raise CaptureError(f"polymarket: market {market_id} outcomePrices are not decimals") from exc
    yes_bid = market.get("bestBid")
    yes_ask = market.get("bestAsk")
    one = Decimal("1")

    def dec_or_none(value) -> str | None:
        return None if value is None else _canon(Decimal(str(value)))

    yes = {
        "provider_outcome_id": f"{market_id}-YES", "label": "YES",
        "best_bid": dec_or_none(yes_bid), "best_ask": dec_or_none(yes_ask),
        "last": _canon(yes_last), "bid_size": None, "ask_size": None,
    }
    no = {
        "provider_outcome_id": f"{market_id}-NO", "label": "NO",
        "best_bid": None if yes_ask is None else _canon(one - Decimal(str(yes_ask))),
        "best_ask": None if yes_bid is None else _canon(one - Decimal(str(yes_bid))),
        "last": _canon(one - yes_last), "bid_size": None, "ask_size": None,
    }
    rules_text = market.get("description") or ""
    rules_version = _rules_version(market.get("updatedAt") or market.get("startDate"), retrieved_at)
    if market.get("closed"):
        status = "RESOLVED" if market.get("umaResolutionStatus") == "resolved" else "CLOSED"
    elif market.get("active"):
        status = "OPEN"
    else:
        status = "PENDING"
    return {
        "schema_version": "mlab-pm-frozen.v1",
        "source_class": "live_capture",
        "capture_adapter_version": ADAPTER_VERSION,
        "provider_id": "polymarket",
        "legal_entity_id": "polymarket",
        "venue_id": "polymarket-clob",
        "api_surface_id": "gamma-v1",
        "jurisdiction_eligibility_status": "UNVERIFIED",
        "terms_version": f"polymarket-market-rules-{rules_version}",
        "terms_sha256": sha256_hex(rules_text.encode("utf-8")),
        "provider_market_id": market_id,
        "request_method": "GET",
        "request_url": url,
        "retrieved_at_utc": retrieved_at,
        "observed_at_utc": retrieved_at,
        "provider_updated_at_utc": _rfc3339z(market.get("updatedAt")),
        "raw_content_type": content_type,
        "raw_sha256": sha256_hex(raw),
        "raw_body_base64": base64.b64encode(raw).decode("ascii"),
        "retention_note": "Live capture from Polymarket Gamma API; raw body preserved byte-exact. "
                          "Discovery-grade: Gamma carries no outcome sizes. Not engine-admissible "
                          "(source_class gate), so imports quarantine by design.",
        "market": {
            "title": market.get("question") or market_id,
            "description": None,
            "market_type": "binary",
            "status": status,
            "currency": "USD",
            "payout_unit": _canon(Decimal("1")),
            "tick_size": dec_or_none(market.get("orderPriceMinTickSize")) or _canon(Decimal("0.01")),
            "opens_at_utc": _rfc3339z(market.get("startDate")),
            "closes_at_utc": _rfc3339z(market.get("endDate")),
            "scheduled_resolution_at_utc": _rfc3339z(market.get("endDate")),
            "rules_version": rules_version,
            "rules_text": rules_text or None,
            "rules_url": None,
            "resolution_source_description": market.get("resolutionSource") or None,
            "resolution_source_url": market.get("resolutionSource") or None,
            "void_policy": None,
            "fee_schedule_id": None,
            "fee_schedule_version": None,
            "fee_schedule_url": None,
            "outcomes": [yes, no],
        },
    }


# ---------------------------------------------------------------------------
# Capture drivers
# ---------------------------------------------------------------------------

def _capture(descriptor: dict, root: Path | None) -> dict:
    root = root or prediction_data_root()
    with tempfile.NamedTemporaryFile("wb", suffix=".json", delete=False) as staging:
        staging.write(json.dumps(descriptor, indent=1).encode("utf-8"))
        staging_path = Path(staging.name)
    try:
        result = import_descriptor(root, staging_path)
    finally:
        staging_path.unlink(missing_ok=True)
    return {**result, "provider_id": descriptor["provider_id"],
            "provider_market_id": descriptor["provider_market_id"],
            "source_class": descriptor["source_class"]}


def capture_kalshi_market(ticker: str, fetcher: Fetcher,
                          root: Path | None = None, retrieved_at: str | None = None) -> dict:
    url = f"{KALSHI_BASE}/markets/{ticker}"
    raw, content_type = fetcher(url)
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise CaptureError(f"kalshi: response for {ticker} is not a JSON object")
    market = payload.get("market")
    if not isinstance(market, dict):
        raise CaptureError(f"kalshi: no market object in response for {ticker}")
    descriptor = build_kalshi_descriptor(raw, market, url, retrieved_at or _now_z(), content_type)
    return _capture(descriptor, root)


def capture_polymarket_market(market_id: str, fetcher: Fetcher,
                              root: Path | None = None, retrieved_at: str | None = None) -> dict:
    url = f"{POLYMARKET_GAMMA}/markets/{market_id}"
    raw, content_type = fetcher(url)
    payload = json.loads(raw.decode("utf-8"))
    market = payload[0] if isinstance(payload, list) and payload else payload
    if not isinstance(market, dict) or "id" not in market:
        raise CaptureError(f"polymarket: no market object in response for {market_id}")
    descriptor = build_polymarket_descriptor(raw, market, url, retrieved_at or _now_z(), content_type)
    return _capture(descriptor, root)
