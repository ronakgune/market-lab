"""Daily-bar data spine for strategy streams: Alpaca-only, provenance-first.

The legacy ``data.fetch_prices`` path (yfinance -> cache -> synthetic) is
fine for exploratory reports, but it can silently hand a backtest synthetic
random-walk bars. Streams must never measure themselves on invented data,
so this module is deliberately stricter:

- Bars come from Alpaca (SIP feed, fully adjusted) via the same env-only
  credential contract as the intraday fetcher — no fallback files, and NO
  synthetic fallback of any kind. Failure raises; it never invents.
- Every backfill writes a provenance sidecar next to the CSV cache
  (``<SYMBOL>.csv.meta.json``): source, feed, adjustment, fetch time, and
  the covered range. ``load_daily_bars`` refuses caches without one.
- The on-disk CSVs use ``data.price_path`` so existing cache readers keep
  working; the sidecar is what makes a cache admissible for stream use.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo

from alpaca.data.enums import Adjustment, DataFeed

from .data import Bar, load_cached_prices, price_path, save_prices
from .intraday.alpaca_fetcher import MissingCredentialsError

log = logging.getLogger(__name__)

PageFetcher = Callable[[dict], Mapping]
MAX_PAGE_LIMIT = 10_000  # Alpaca's per-page cap for the bars endpoint.
_ET = ZoneInfo("America/New_York")


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


def _valid_daily_bar(bar: Bar) -> bool:
    values = (bar.open, bar.high, bar.low, bar.close)
    return (
        all(math.isfinite(v) and v > 0 for v in values)
        and bar.high >= bar.low
        and bar.volume >= 0
    )


def _raw_to_bar(raw: Mapping) -> Optional[Bar]:
    try:
        ts = datetime.fromisoformat(str(raw["t"]).replace("Z", "+00:00"))
        if ts.tzinfo is None:
            # A timestamp without an offset has no knowable ET date — its
            # conversion would silently depend on the host's local timezone.
            return None
        bar = Bar(
            date=ts.astimezone(_ET).date(),
            open=float(raw["o"]), high=float(raw["h"]),
            low=float(raw["l"]), close=float(raw["c"]),
            volume=int(raw["v"]),
        )
    except (KeyError, TypeError, ValueError):
        return None
    return bar if _valid_daily_bar(bar) else None


def fetch_daily_bars(
    symbol: str,
    start: date,
    end: date,
    *,
    feed: DataFeed = DataFeed.SIP,
    adjustment: Adjustment = Adjustment.ALL,
    client=None,
    page_fetcher: Optional[PageFetcher] = None,
) -> tuple[list[Bar], str]:
    """Fetch all daily bars for one symbol in [start, end] from Alpaca.

    Returns (bars, source) with source in provenance style ("alpaca:sip").
    Raises on missing credentials or transport failure — never falls back.
    ``end`` is inclusive.
    """
    if end < start:
        raise ValueError("end must be >= start")
    if page_fetcher is None:
        if client is None:
            client = _default_client()
        page_fetcher = _page_fetcher_for_client(client)

    source = f"alpaca:{feed.value}"
    params: dict = {
        "symbols": symbol,
        "timeframe": "1Day",
        "start": datetime(start.year, start.month, start.day, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z"),
        "end": datetime(end.year, end.month, end.day, 23, 59, 59, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z"),
        "limit": MAX_PAGE_LIMIT,
        "feed": feed.value,
        "adjustment": adjustment.value,
        "sort": "asc",
    }
    by_date: dict[date, Bar] = {}
    seen_tokens: set = set()
    while True:
        page = page_fetcher(params)
        bars_map = page.get("bars") or {}
        raw_bars = bars_map.get(symbol)
        if raw_bars is None:
            # The venue keys bars by its canonical symbol case; a caller
            # passing "spy" must not silently collect zero bars for "SPY".
            for key, value in bars_map.items():
                if str(key).upper() == symbol.upper():
                    raw_bars = value
                    break
        for raw in (raw_bars or []):
            bar = _raw_to_bar(raw)
            if bar is not None and start <= bar.date <= end:
                by_date[bar.date] = bar
        token = page.get("next_page_token")
        if not token:
            break
        if token in seen_tokens:
            # A repeated token means the pager is stuck or cycling; looping
            # on it would hang the fetch forever. Fail loudly instead.
            raise RuntimeError(
                f"Alpaca pagination token repeated for {symbol}; aborting fetch"
            )
        seen_tokens.add(token)
        params = {**params, "page_token": token}
    bars = [by_date[d] for d in sorted(by_date)]
    log.debug("fetched %d daily bars for %s from %s", len(bars), symbol, source)
    return bars, source


def sidecar_path(symbol: str) -> Path:
    return price_path(symbol).with_suffix(".csv.meta.json")


@dataclass
class BackfillResult:
    symbol: str
    bars: int
    first: Optional[date]
    last: Optional[date]
    source: str = ""
    error: str = ""


def backfill_daily(
    symbols: Sequence[str],
    start: date,
    end: date,
    *,
    feed: DataFeed = DataFeed.SIP,
    adjustment: Adjustment = Adjustment.ALL,
    client=None,
    page_fetcher: Optional[PageFetcher] = None,
) -> list[BackfillResult]:
    """Fetch + cache daily bars for each symbol, with provenance sidecars.

    One symbol's failure is recorded in its result and does not abort the
    rest of the batch — but it is never papered over with substitute data.
    """
    results: list[BackfillResult] = []
    for symbol in symbols:
        try:
            bars, source = fetch_daily_bars(
                symbol, start, end, feed=feed, adjustment=adjustment,
                client=client, page_fetcher=page_fetcher,
            )
            if not bars:
                results.append(BackfillResult(symbol, 0, None, None, source, "no bars returned"))
                continue
            path = save_prices(symbol, bars)
            meta = {
                "symbol": symbol.upper(),
                "source": source,
                "feed": feed.value,
                "adjustment": adjustment.value,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "requested_start": start.isoformat(),
                "requested_end": end.isoformat(),
                "bars": len(bars),
                "first": bars[0].date.isoformat(),
                "last": bars[-1].date.isoformat(),
            }
            sidecar_path(symbol).write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            results.append(BackfillResult(symbol, len(bars), bars[0].date, bars[-1].date, source))
            log.info("backfilled %s: %d bars %s..%s", symbol, len(bars), bars[0].date, bars[-1].date)
        except Exception as exc:  # per-symbol isolation, reported not hidden
            results.append(BackfillResult(symbol, 0, None, None, "", f"{type(exc).__name__}: {exc}"))
    return results


def load_daily_bars(symbol: str) -> tuple[list[Bar], dict]:
    """Load backfilled bars + provenance for a stream run.

    Raises FileNotFoundError when there is no cache or no provenance
    sidecar — a cache of unknown origin is not admissible evidence.
    """
    meta_file = sidecar_path(symbol)
    if not meta_file.exists():
        raise FileNotFoundError(
            f"no provenance sidecar for {symbol} at {meta_file}; run a backfill first"
        )
    bars = load_cached_prices(symbol)
    if not bars:
        raise FileNotFoundError(f"no cached daily bars for {symbol}")
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    _validate_provenance(symbol, bars, meta)
    return bars, meta


def _validate_provenance(symbol: str, bars: list[Bar], meta) -> None:
    """Cross-check a provenance sidecar against the bars it vouches for.

    A sidecar that disagrees with its CSV — wrong symbol, non-Alpaca
    source, wrong bar count or covered range — means the cache was
    tampered with, mixed up between symbols, or left behind by a
    half-finished backfill whose CSV was later rewritten. Refuse it
    loudly rather than let a stream measure itself on mislabeled data.
    """

    def bad(reason: str) -> ValueError:
        return ValueError(f"provenance sidecar for {symbol} failed validation: {reason}")

    if not isinstance(meta, dict):
        raise bad("sidecar is not a JSON object")
    if str(meta.get("symbol", "")).upper() != symbol.upper():
        raise bad(f"sidecar symbol {meta.get('symbol')!r} does not match {symbol!r}")
    if not str(meta.get("source", "")).startswith("alpaca:"):
        raise bad(f"sidecar source {meta.get('source')!r} is not an Alpaca feed")
    if not isinstance(meta.get("bars"), int) or meta.get("bars") != len(bars):
        raise bad(f"sidecar claims {meta.get('bars')!r} bars, cache holds {len(bars)}")
    if meta.get("first") != bars[0].date.isoformat():
        raise bad(f"sidecar first {meta.get('first')!r} != first cached bar {bars[0].date}")
    if meta.get("last") != bars[-1].date.isoformat():
        raise bad(f"sidecar last {meta.get('last')!r} != last cached bar {bars[-1].date}")
