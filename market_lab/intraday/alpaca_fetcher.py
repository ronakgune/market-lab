"""Historical 1-minute bars from Alpaca's market-data API.

DATA ONLY. This module imports exclusively from alpaca.data.* (the
historical data client, request models, and enums). It never imports
alpaca.trading / alpaca.broker or any order-capable client; a boundary
test enforces that.

Design notes (against alpaca-py 0.45.0, verified at build time):
- DataFeed and Adjustment enums exist as expected: DataFeed.SIP / .IEX,
  Adjustment.ALL / .SPLIT / .DIVIDEND / .RAW. Both are explicit parameters
  here, never silently defaulted inside the request.
- Pagination is done by THIS module: it follows `next_page_token` until
  the API returns none, so a fetch can never silently truncate at one
  page. (The SDK's own get_stock_bars paginates internally against a
  10,000-row page size; we deliberately page at the raw REST layer via
  the client's `get` so the loop is visible and testable.)
- Extended-hours bars are included: the bars endpoint returns them and
  no regular-trading-hours filtering is applied.

Credentials come from the environment only: ALPACA_API_KEY and
ALPACA_API_SECRET. There are no fallback files and nothing is hardcoded.
Tests inject `page_fetcher` (or a stub `client`) and never need network
or credentials.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Callable, Mapping, Optional

from alpaca.data.enums import Adjustment, DataFeed

from .bars import MinuteBar, valid_minute_bar

log = logging.getLogger(__name__)

# One raw page from GET /v2/stocks/bars:
# {"bars": {"SYM": [{"t","o","h","l","c","v", ...}, ...]}, "next_page_token": str | None}
PageFetcher = Callable[[dict], Mapping]

MAX_PAGE_LIMIT = 10_000  # Alpaca's per-page cap for the bars endpoint.


class MissingCredentialsError(RuntimeError):
    """Raised when no client/page_fetcher is injected and the Alpaca
    credentials are absent from the environment."""


def _credentials() -> tuple[str, str]:
    key = os.environ.get("ALPACA_API_KEY", "").strip()
    secret = os.environ.get("ALPACA_API_SECRET", "").strip()
    if not key or not secret:
        raise MissingCredentialsError(
            "Alpaca credentials not found: set ALPACA_API_KEY and "
            "ALPACA_API_SECRET in the environment (no fallback files are read)."
        )
    return key, secret


def _default_client():
    from alpaca.data.historical import StockHistoricalDataClient

    key, secret = _credentials()
    return StockHistoricalDataClient(key, secret)


def _page_fetcher_for_client(client) -> PageFetcher:
    def fetch(params: dict) -> Mapping:
        return client.get("/stocks/bars", params)

    return fetch


def _parse_ts(raw: str) -> datetime:
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


def _raw_to_bar(symbol: str, raw: Mapping, source: str) -> Optional[MinuteBar]:
    try:
        bar = MinuteBar(
            symbol=symbol,
            timestamp=_parse_ts(raw["t"]),
            open=float(raw["o"]), high=float(raw["h"]),
            low=float(raw["l"]), close=float(raw["c"]),
            volume=int(raw["v"]),
            source=source,
        )
    except (KeyError, TypeError, ValueError):
        return None
    return bar if valid_minute_bar(bar) else None


def fetch_minute_bars(
    symbol: str,
    start: datetime,
    end: datetime,
    *,
    feed: DataFeed = DataFeed.SIP,
    adjustment: Adjustment = Adjustment.ALL,
    client=None,
    page_fetcher: Optional[PageFetcher] = None,
    page_limit: int = MAX_PAGE_LIMIT,
) -> tuple[list[MinuteBar], str]:
    """Fetch all 1-minute bars for one symbol in [start, end).

    Returns (bars, source) in market_lab's provenance style, where source
    is "alpaca:sip" / "alpaca:iex" / ... matching the feed actually used.
    Pages are followed until `next_page_token` is exhausted.
    """
    for name, ts in (("start", start), ("end", end)):
        if ts.tzinfo is None or ts.utcoffset() is None:
            raise ValueError(f"{name} must be tz-aware")
    if page_fetcher is None:
        if client is None:
            client = _default_client()
        page_fetcher = _page_fetcher_for_client(client)

    source = f"alpaca:{feed.value}"
    params: dict = {
        "symbols": symbol,
        "timeframe": "1Min",
        "start": start.isoformat().replace("+00:00", "Z"),
        "end": end.isoformat().replace("+00:00", "Z"),
        "limit": page_limit,
        "feed": feed.value,
        "adjustment": adjustment.value,
        "sort": "asc",
    }
    bars: list[MinuteBar] = []
    while True:
        page = page_fetcher(params)
        raw_bars = (page.get("bars") or {}).get(symbol, [])
        for raw in raw_bars:
            bar = _raw_to_bar(symbol, raw, source)
            if bar is not None:
                bars.append(bar)
        token = page.get("next_page_token")
        if not token:
            break
        params = {**params, "page_token": token}
    bars.sort(key=lambda b: b.timestamp)
    log.debug("fetched %d minute bars for %s from %s", len(bars), symbol, source)
    return bars, source
