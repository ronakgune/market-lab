"""Point-in-time earnings events from SEC EDGAR 8-K filings.

An earnings event is an 8-K whose reported items include Item 2.02
("Results of Operations and Financial Condition"). The event time is the
filing's EDGAR *acceptance* timestamp — the moment the information became
public — taken from the submissions feed (acceptanceDateTime), which makes
the series point-in-time by construction: no as-known-today calendar dates.

Session classification uses the America/New_York clock:
    accepted < 09:30 ET            -> "premarket"
    09:30 <= accepted < 16:00 ET   -> "intraday"
    accepted >= 16:00 ET           -> "afterhours"

The actionable trading day is the acceptance day for premarket/intraday
events and the NEXT trading day for afterhours events.

LIMITATION: the trading-day calendar is weekday-only. US market holidays
are not modelled, so an event landing on/near a holiday may be assigned
to a holiday instead of the next session. Acceptable for stage 1; a real
exchange calendar is a documented follow-up.

EDGAR access is behind injectable seams (`submissions_fetcher`,
`cik_resolver`) so tests run offline on recorded fixtures. CIK resolution
uses edgartools' Company lookup; the submissions JSON itself is fetched
from data.sec.gov and cached raw under INTRADAY_EDGAR_DIR. EDGAR fair
access requires a declared User-Agent: it comes from the EDGAR_USER_AGENT
environment variable. If unset, a clearly-marked placeholder is used and
a warning is logged — no real person's name or email is hardcoded here.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Callable, Mapping, Optional
from zoneinfo import ZoneInfo

from ..config import INTRADAY_EDGAR_DIR

log = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
_MARKET_OPEN = time(9, 30)
_MARKET_CLOSE = time(16, 0)

_PLACEHOLDER_UA = "market-lab-research/0.1 (REPLACE-ME: set EDGAR_USER_AGENT; contact@example.com)"

SubmissionsFetcher = Callable[[str, int], Mapping]
CikResolver = Callable[[str], int]


@dataclass(frozen=True)
class EarningsEvent:
    ticker: str
    cik: int
    accession_number: str
    accepted_utc: datetime
    accepted_et: datetime
    session: str  # "premarket" | "intraday" | "afterhours"
    actionable_day: date  # ET trading day the event can first be acted on


def classify_session(accepted_et: datetime) -> str:
    t = accepted_et.time()
    if t < _MARKET_OPEN:
        return "premarket"
    if t >= _MARKET_CLOSE:
        return "afterhours"
    return "intraday"


def next_trading_day(day: date) -> date:
    """Next weekday after `day` (holidays not modelled — see module docstring)."""
    candidate = day + timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    return candidate


def _roll_forward_to_trading_day(day: date) -> date:
    while day.weekday() >= 5:
        day += timedelta(days=1)
    return day


def actionable_day_for(accepted_et: datetime, session: str) -> date:
    day = accepted_et.date()
    if session == "afterhours":
        return next_trading_day(day)
    return _roll_forward_to_trading_day(day)


def _parse_acceptance(raw: str) -> datetime:
    dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _has_item_202(items_field: str) -> bool:
    return any(tok.strip() == "2.02" for tok in (items_field or "").split(","))


def events_from_submissions(
    ticker: str,
    cik: int,
    submissions: Mapping,
    start: date,
    end: date,
) -> list[EarningsEvent]:
    """Extract Item 2.02 8-K events from a data.sec.gov submissions payload.

    Pure function over the recorded JSON shape (filings.recent columnar
    arrays); the date range filters on the ET acceptance date, inclusive.
    """
    recent = (submissions.get("filings") or {}).get("recent") or {}
    forms = recent.get("form") or []
    accessions = recent.get("accessionNumber") or []
    accepted = recent.get("acceptanceDateTime") or []
    items = recent.get("items") or []
    events: list[EarningsEvent] = []
    for i, form in enumerate(forms):
        if form != "8-K" or not _has_item_202(items[i] if i < len(items) else ""):
            continue
        accepted_utc = _parse_acceptance(accepted[i])
        accepted_et = accepted_utc.astimezone(ET)
        if not (start <= accepted_et.date() <= end):
            continue
        session = classify_session(accepted_et)
        events.append(EarningsEvent(
            ticker=ticker.upper(),
            cik=cik,
            accession_number=accessions[i],
            accepted_utc=accepted_utc,
            accepted_et=accepted_et,
            session=session,
            actionable_day=actionable_day_for(accepted_et, session),
        ))
    events.sort(key=lambda e: e.accepted_utc)
    return events


def edgar_user_agent() -> str:
    ua = os.environ.get("EDGAR_USER_AGENT", "").strip()
    if ua:
        return ua
    log.warning("EDGAR_USER_AGENT is not set; using a placeholder User-Agent. "
                "Set EDGAR_USER_AGENT to a real declared identity before live EDGAR use.")
    return _PLACEHOLDER_UA


def resolve_cik(ticker: str) -> int:
    """Resolve a ticker to a CIK via edgartools' Company lookup (network)."""
    from edgar import Company, set_identity

    try:
        set_identity(edgar_user_agent())
    except Exception:  # identity is best-effort; the fetcher sets UA itself too
        log.debug("edgartools set_identity failed", exc_info=True)
    return int(Company(ticker).cik)


def fetch_submissions(cik: int, *, http_get: Optional[Callable] = None,
                      cache_dir: Optional[Path] = None, use_cache: bool = True) -> dict:
    """Fetch (and cache raw) the data.sec.gov submissions JSON for a CIK."""
    cache_root = Path(cache_dir) if cache_dir is not None else INTRADAY_EDGAR_DIR
    cache_path = cache_root / f"CIK{cik:010d}.json"
    if use_cache and cache_path.exists():
        return json.loads(cache_path.read_text())
    url = f"https://data.sec.gov/submissions/CIK{cik:010d}.json"
    headers = {"User-Agent": edgar_user_agent(), "Accept": "application/json"}
    if http_get is not None:
        payload = http_get(url, headers)
    else:
        import httpx

        with httpx.Client(timeout=30.0) as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()
            payload = response.json()
    if use_cache:
        cache_root.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload))
        os.replace(tmp, cache_path)
    return payload


def _default_submissions_fetcher(ticker: str, cik: int) -> Mapping:
    return fetch_submissions(cik)


def fetch_earnings_events(
    tickers: list[str],
    start: date,
    end: date,
    *,
    submissions_fetcher: Optional[SubmissionsFetcher] = None,
    cik_resolver: Optional[CikResolver] = None,
) -> list[EarningsEvent]:
    """Earnings events for `tickers` with ET acceptance dates in [start, end]."""
    resolve = cik_resolver or resolve_cik
    fetch = submissions_fetcher or _default_submissions_fetcher
    events: list[EarningsEvent] = []
    for ticker in tickers:
        cik = resolve(ticker)
        submissions = fetch(ticker, cik)
        events.extend(events_from_submissions(ticker, cik, submissions, start, end))
    events.sort(key=lambda e: e.accepted_utc)
    return events
