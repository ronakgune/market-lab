"""Historical option-chain data layer for strategy streams (Stream 4).

Source: the DoltHub community dataset ``post-no-preference/options``,
table ``option_chain``, read over its public SQL-over-HTTP API. This is
a *new* module on purpose: the legacy ``options_data.py`` serves the
yfinance snapshot/screening/paper stack and must not be disturbed.

Why this module exists separately from ``daily_data.py``: options
history has no free licensed source (OPRA data is exchange-owned and
redistribution-barred), so phase 1 of Stream 4 evaluates on this
community dataset with its limits stated, not hidden:

- Coverage is SPOTTY: chains exist only on some days ("chain days").
  A date with no chain is a first-class result (``NoChainData``),
  never a silent empty chain.
- Chains are SAMPLED: ~26 strikes per expiry, not the full listed
  board; there is NO open interest and NO volume column. The source
  column named ``vol`` is implied volatility (a decimal), NOT volume.
- There are NO index options (SPX/SPXW absent); SPY is the vehicle.
- Provenance discipline mirrors ``daily_data.py``: every cached chain
  carries a sidecar (source URL, retrieval time, row counts, sha256
  of canonical content) and a cache entry whose sidecar is missing or
  disagrees with the content is REFUSED. Nothing here ever fabricates
  a chain: transport failure raises; it never substitutes data.
"""

from __future__ import annotations

import hashlib
import json
import logging
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

from .config import OPTIONS_DIR

log = logging.getLogger(__name__)

API_BASE = "https://www.dolthub.com/api/v1alpha1/post-no-preference/options/master"
SOURCE = "dolthub:post-no-preference/options@master"
CACHE_DIR = OPTIONS_DIR / "history"
COLUMNS = (
    "date", "act_symbol", "expiration", "strike", "call_put",
    "bid", "ask", "vol", "delta", "gamma", "rho", "theta", "vega",
)

# A fetcher takes the full query URL and returns the decoded JSON payload.
HttpFetcher = Callable[[str], Mapping]


def _default_fetcher(url: str) -> Mapping:
    # Deliberately NO auth header: DoltHub returns 400 when one is sent.
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=90) as resp:
        return json.loads(resp.read())


def chain_query_url(symbol: str, day: date) -> str:
    sql = (
        "SELECT " + ", ".join(COLUMNS) + " FROM option_chain "
        f"WHERE act_symbol='{symbol.upper()}' AND date='{day.isoformat()}' "
        "ORDER BY expiration, strike, call_put"
    )
    return API_BASE + "?q=" + urllib.parse.quote(sql)


@dataclass(frozen=True)
class OptionRow:
    """One contract quote on one chain day. All greeks/IV are the
    vendor's values as published; ``iv`` is the source's ``vol``
    column (implied volatility as a decimal — NOT volume)."""

    chain_date: date
    symbol: str
    expiration: date
    strike: float
    right: str  # "CALL" or "PUT"
    bid: float
    ask: float
    iv: float
    delta: float
    gamma: float
    rho: float
    theta: float
    vega: float


@dataclass(frozen=True)
class ChainDay:
    symbol: str
    day: date
    rows: tuple[OptionRow, ...]

    @property
    def expirations(self) -> tuple[date, ...]:
        return tuple(sorted({r.expiration for r in self.rows}))

    def for_expiry(self, expiration: date) -> tuple[OptionRow, ...]:
        return tuple(r for r in self.rows if r.expiration == expiration)

    def get(self, expiration: date, strike: float, right: str) -> Optional[OptionRow]:
        for r in self.rows:
            if r.expiration == expiration and r.strike == strike and r.right == right:
                return r
        return None


@dataclass(frozen=True)
class NoChainData:
    """Explicit result for a date the dataset has no chain for.

    Distinct from an error: the fetch succeeded and the dataset
    genuinely holds zero rows for (symbol, date)."""

    symbol: str
    day: date
    reason: str = "dataset returned zero rows for this (symbol, date)"


def _parse_row(raw: Mapping) -> Optional[OptionRow]:
    try:
        right_raw = str(raw["call_put"]).strip().lower()
        if right_raw not in ("call", "put"):
            return None
        row = OptionRow(
            chain_date=date.fromisoformat(str(raw["date"])),
            symbol=str(raw["act_symbol"]).upper(),
            expiration=date.fromisoformat(str(raw["expiration"])),
            strike=float(raw["strike"]),
            right=right_raw.upper(),
            bid=float(raw["bid"]),
            ask=float(raw["ask"]),
            iv=float(raw["vol"]),  # 'vol' is implied volatility, not volume
            delta=float(raw["delta"]),
            gamma=float(raw["gamma"]),
            rho=float(raw["rho"]),
            theta=float(raw["theta"]),
            vega=float(raw["vega"]),
        )
    except (KeyError, TypeError, ValueError):
        return None
    if row.bid < 0 or row.ask < row.bid or row.strike <= 0:
        return None
    return row


def parse_chain_rows(symbol: str, day: date, raw_rows: Sequence[Mapping]) -> list[OptionRow]:
    rows: list[OptionRow] = []
    for raw in raw_rows:
        row = _parse_row(raw)
        if row is None:
            continue
        if row.symbol != symbol.upper() or row.chain_date != day:
            continue
        rows.append(row)
    rows.sort(key=lambda r: (r.expiration, r.strike, r.right))
    return rows


def _canonical(rows: Sequence[OptionRow]) -> str:
    payload = [
        {
            "expiration": r.expiration.isoformat(), "strike": r.strike,
            "right": r.right, "bid": r.bid, "ask": r.ask, "iv": r.iv,
            "delta": r.delta, "gamma": r.gamma, "rho": r.rho,
            "theta": r.theta, "vega": r.vega,
        }
        for r in rows
    ]
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _sha(rows: Sequence[OptionRow]) -> str:
    return hashlib.sha256(_canonical(rows).encode("utf-8")).hexdigest()


def chain_cache_path(symbol: str, day: date) -> Path:
    return CACHE_DIR / symbol.upper() / f"{day.isoformat()}.json"


def _sidecar_path(symbol: str, day: date) -> Path:
    return chain_cache_path(symbol, day).with_suffix(".meta.json")


def _write_cache(symbol: str, day: date, rows: list[OptionRow],
                 rows_returned: int, url: str) -> None:
    path = chain_cache_path(symbol, day)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical(rows) + "\n", encoding="utf-8")
    meta = {
        "symbol": symbol.upper(),
        "date": day.isoformat(),
        "source": SOURCE,
        "source_url": url,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "rows_returned": rows_returned,
        "rows": len(rows),
        "sha256": _sha(rows),
    }
    _sidecar_path(symbol, day).write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _validate_sidecar(symbol: str, day: date, rows: list[OptionRow], meta) -> None:
    def bad(reason: str) -> ValueError:
        return ValueError(
            f"provenance sidecar for {symbol} {day} failed validation: {reason}")

    if not isinstance(meta, dict):
        raise bad("sidecar is not a JSON object")
    if str(meta.get("symbol", "")).upper() != symbol.upper():
        raise bad(f"sidecar symbol {meta.get('symbol')!r} != {symbol!r}")
    if meta.get("date") != day.isoformat():
        raise bad(f"sidecar date {meta.get('date')!r} != {day.isoformat()}")
    if meta.get("source") != SOURCE:
        raise bad(f"sidecar source {meta.get('source')!r} is not {SOURCE!r}")
    if meta.get("rows") != len(rows):
        raise bad(f"sidecar claims {meta.get('rows')!r} rows, cache holds {len(rows)}")
    if meta.get("sha256") != _sha(rows):
        raise bad("sha256 of cached content does not match sidecar")


def load_chain(symbol: str, day: date) -> ChainDay | NoChainData:
    """Load a previously fetched chain from the local cache.

    Raises FileNotFoundError when the date was never fetched, and
    ValueError when a cache entry's provenance sidecar is missing or
    disagrees with the content — a cache of unknown origin is not
    admissible evidence. A provenanced zero-row fetch returns
    ``NoChainData``.
    """
    path = chain_cache_path(symbol, day)
    meta_file = _sidecar_path(symbol, day)
    if not path.exists() and not meta_file.exists():
        raise FileNotFoundError(
            f"no cached chain for {symbol} {day}; fetch it first")
    if not meta_file.exists():
        raise ValueError(
            f"cache entry for {symbol} {day} has no provenance sidecar; refusing it")
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    if not path.exists():
        # A sidecar vouching for a zero-row fetch, with no content file.
        if isinstance(meta, dict) and meta.get("rows") == 0 and meta.get("source") == SOURCE:
            return NoChainData(symbol.upper(), day)
        raise ValueError(
            f"provenance sidecar for {symbol} {day} exists but content file is missing")
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows: list[OptionRow] = []
    for item in raw:
        rows.append(OptionRow(
            chain_date=day, symbol=symbol.upper(),
            expiration=date.fromisoformat(item["expiration"]),
            strike=float(item["strike"]), right=str(item["right"]),
            bid=float(item["bid"]), ask=float(item["ask"]), iv=float(item["iv"]),
            delta=float(item["delta"]), gamma=float(item["gamma"]),
            rho=float(item["rho"]), theta=float(item["theta"]), vega=float(item["vega"]),
        ))
    rows.sort(key=lambda r: (r.expiration, r.strike, r.right))
    _validate_sidecar(symbol, day, rows, meta)
    if not rows:
        return NoChainData(symbol.upper(), day)
    return ChainDay(symbol.upper(), day, tuple(rows))


def fetch_chain(
    symbol: str,
    day: date,
    *,
    fetcher: Optional[HttpFetcher] = None,
    use_cache: bool = True,
) -> ChainDay | NoChainData:
    """Fetch one (symbol, date) chain from DoltHub, with caching.

    Returns ``NoChainData`` when the dataset holds no chain for the
    date (a provenanced zero-row result is cached too, so absence is
    itself evidence and is not re-fetched). Raises on transport or
    server failure — never falls back, never fabricates.
    """
    if use_cache:
        try:
            return load_chain(symbol, day)
        except FileNotFoundError:
            pass
    url = chain_query_url(symbol, day)
    payload = (fetcher or _default_fetcher)(url)
    if payload.get("query_execution_status") != "Success":
        raise RuntimeError(
            f"DoltHub query failed for {symbol} {day}: "
            f"{payload.get('query_execution_message')}")
    raw_rows = payload.get("rows") or []
    rows = parse_chain_rows(symbol, day, raw_rows)
    _write_cache(symbol, day, rows, len(raw_rows), url)
    if not rows:
        log.info("no chain in dataset for %s %s", symbol, day)
        return NoChainData(symbol.upper(), day)
    log.debug("fetched %d chain rows for %s %s", len(rows), symbol, day)
    return ChainDay(symbol.upper(), day, tuple(rows))
