"""SEC EDGAR fundamentals and the Stream 7 quality gate.

The gate is defined in docs/dip-buy-preregistration.md and nowhere
else; this module implements exactly what is registered there:

- ROE (trailing net income / latest equity) >= 10%
- Free cash flow (annual OCF - capex) positive in the latest fiscal
  year and in at least 3 of the last 5 fiscal years
- Total liabilities / total assets <= 0.85

Point-in-time discipline is strict: a query as of date D only uses
facts whose ``filed`` date is <= D (and whose period end is <= D).
A restated value therefore becomes visible only when the restatement
was filed. Any leg that cannot be evaluated fails the gate with the
reason recorded — the gate never passes on absent data.

Raw EDGAR responses are cached verbatim under
``data/market-lab/fundamentals/`` (gitignored with the rest of
``data/``) with a provenance sidecar, mirroring the daily price
cache. There is no synthetic or fallback data: if EDGAR cannot be
reached and no cache exists, the fetch raises.
"""

from __future__ import annotations

import json
import math
import time
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

from .config import FUNDAMENTALS_DIR

# SEC fair access requires a declared User-Agent with a contact;
# without the email-shaped contact, data.sec.gov answers 403.
USER_AGENT = "market-lab palluron18@gmail.com"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
_REQUEST_DELAY_SECONDS = 0.15  # SEC fair access: stay far under 10 req/s

# --- Pre-registered gate thresholds (docs/dip-buy-preregistration.md) ---
ROE_FLOOR = 0.10
LEVERAGE_CAP = 0.85
FCF_WINDOW_YEARS = 5
FCF_MIN_POSITIVE_YEARS = 3
FCF_MIN_YEARS_AVAILABLE = 3

# Symbols in the price cache that are not operating companies. They
# have no company fundamentals to gate on and are excluded from the
# Stream 7 universe with the reason recorded — never silently.
EXCLUDED_NON_COMPANY: dict[str, str] = {
    "SPY": "index ETF (stream benchmark); no company fundamentals",
    "QQQ": "index ETF (stream benchmark); no company fundamentals",
    "DIA": "index ETF; no company fundamentals",
    "IWM": "index ETF; no company fundamentals",
    "VIXY": "volatility futures ETP; not an operating company",
    "IBIT": "spot crypto ETP (trust); no company facts",
    "ETHA": "spot crypto ETP (trust); no company facts",
    "GLD": "gold trust; no company facts",
    "USO": "oil futures pool; no company facts",
    "TLT": "bond ETF; no company fundamentals",
}

# Standard us-gaap concepts, in preference order per quantity.
_NET_INCOME = ("NetIncomeLoss",)
_EQUITY = (
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    "StockholdersEquity",
)
_ASSETS = ("Assets",)
_LIABILITIES = ("Liabilities",)
_OCF = (
    "NetCashProvidedByUsedInOperatingActivities",
    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
)
_CAPEX = ("PaymentsToAcquirePropertyPlantAndEquipment",)

_ANNUAL_MIN_DAYS, _ANNUAL_MAX_DAYS = 330, 400
_QUARTER_MIN_DAYS, _QUARTER_MAX_DAYS = 70, 110


@dataclass(frozen=True)
class QualityReport:
    symbol: str
    as_of: date
    passes: bool
    roe: float | None
    net_income: float | None
    equity: float | None
    latest_fcf: float | None
    fcf_by_year: tuple[tuple[date, float], ...]  # chronological, matched annual pairs only
    fcf_positive_in_window: int
    leverage: float | None
    reasons: tuple[str, ...]  # why the gate failed; empty when it passes


# --------------------------------------------------------------------------
# EDGAR fetching / caching (the only I/O in the module)
# --------------------------------------------------------------------------

def _fetch_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _tickers_cache_path() -> Path:
    return FUNDAMENTALS_DIR / "company_tickers.json"


def _facts_cache_path(symbol: str) -> Path:
    return FUNDAMENTALS_DIR / f"{symbol.upper()}.companyfacts.json"


def _load_ticker_map(*, refresh: bool = False) -> dict[str, str]:
    """ticker -> zero-padded 10-digit CIK, from the SEC's own map."""
    path = _tickers_cache_path()
    if refresh or not path.exists():
        FUNDAMENTALS_DIR.mkdir(parents=True, exist_ok=True)
        payload = _fetch_json(TICKERS_URL)
        path.write_text(json.dumps(payload))
        time.sleep(_REQUEST_DELAY_SECONDS)
    else:
        payload = json.loads(path.read_text())
    rows = payload.values() if isinstance(payload, dict) else payload
    out: dict[str, str] = {}
    for row in rows:
        ticker = str(row.get("ticker", "")).upper()
        cik = row.get("cik_str")
        if ticker and cik is not None:
            out[ticker] = f"{int(cik):010d}"
    return out


def cik_for_symbol(symbol: str, *, refresh: bool = False) -> str | None:
    return _load_ticker_map(refresh=refresh).get(symbol.upper())


def load_company_facts(symbol: str, *, refresh: bool = False) -> dict:
    """Raw EDGAR companyfacts JSON for a symbol, cached verbatim.

    Raises (KeyError/URLError) rather than substituting anything when
    the symbol is unknown to the SEC map or EDGAR is unreachable and
    no cache exists — no fallback data, ever.
    """
    symbol = symbol.upper()
    path = _facts_cache_path(symbol)
    if not refresh and path.exists():
        return json.loads(path.read_text())
    cik = cik_for_symbol(symbol)
    if cik is None:
        raise KeyError(f"no CIK for {symbol} in the SEC ticker map")
    payload = _fetch_json(COMPANYFACTS_URL.format(cik=cik))
    FUNDAMENTALS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    sidecar = {
        "symbol": symbol,
        "cik": cik,
        "source": COMPANYFACTS_URL.format(cik=cik),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    path.with_suffix(".meta.json").write_text(json.dumps(sidecar, indent=2))
    time.sleep(_REQUEST_DELAY_SECONDS)
    return payload


# --------------------------------------------------------------------------
# Point-in-time fact selection (pure)
# --------------------------------------------------------------------------

def _parse_date(value: object) -> date | None:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _concept_entries(facts: dict, concepts: tuple[str, ...]) -> list[dict]:
    """Entries of the first preferred concept that has any, USD units."""
    gaap = (facts.get("facts") or {}).get("us-gaap") or {}
    for concept in concepts:
        node = gaap.get(concept) or {}
        units = node.get("units") or {}
        if "USD" in units:
            return list(units["USD"])
        for values in units.values():
            if values:
                return list(values)
    return []


def _visible(entry: dict, as_of: date) -> bool:
    """Filed on/before as_of, period ended on/before as_of, finite value."""
    filed = _parse_date(entry.get("filed"))
    end = _parse_date(entry.get("end"))
    val = entry.get("val")
    return (
        filed is not None and filed <= as_of
        and end is not None and end <= as_of
        and isinstance(val, (int, float)) and math.isfinite(val)
    )


def _duration_days(entry: dict) -> int | None:
    start = _parse_date(entry.get("start"))
    end = _parse_date(entry.get("end"))
    if start is None or end is None:
        return None
    return (end - start).days


def _latest_instant(facts: dict, concepts: tuple[str, ...], as_of: date) -> float | None:
    """Latest instant value visible as of the date (max end, then max filed)."""
    best: tuple[date, date, float] | None = None
    for entry in _concept_entries(facts, concepts):
        if not _visible(entry, as_of):
            continue
        key = (_parse_date(entry["end"]), _parse_date(entry["filed"]), float(entry["val"]))
        if best is None or key > best:
            best = key
    return best[2] if best else None


def _annual_values(facts: dict, concepts: tuple[str, ...], as_of: date) -> dict[date, float]:
    """Fiscal-year duration values by period end, as known on as_of.

    Deduped by period end keeping the latest filing visible on
    as_of, so a restatement or comparative column supersedes the
    original only from its own filed date onward.
    """
    by_end: dict[date, tuple[date, float]] = {}
    for entry in _concept_entries(facts, concepts):
        if not _visible(entry, as_of):
            continue
        days = _duration_days(entry)
        if days is None or not (_ANNUAL_MIN_DAYS <= days <= _ANNUAL_MAX_DAYS):
            continue
        end = _parse_date(entry["end"])
        filed = _parse_date(entry["filed"])
        if end not in by_end or filed > by_end[end][0]:
            by_end[end] = (filed, float(entry["val"]))
    return {end: val for end, (filed, val) in sorted(by_end.items())}


def _trailing_net_income(facts: dict, as_of: date) -> float | None:
    """Most recent annual net income; falls back to a 4-quarter sum."""
    annual = _annual_values(facts, _NET_INCOME, as_of)
    if annual:
        return annual[max(annual)]
    quarters: dict[date, tuple[date, float]] = {}
    for entry in _concept_entries(facts, _NET_INCOME):
        if not _visible(entry, as_of):
            continue
        days = _duration_days(entry)
        if days is None or not (_QUARTER_MIN_DAYS <= days <= _QUARTER_MAX_DAYS):
            continue
        end = _parse_date(entry["end"])
        filed = _parse_date(entry["filed"])
        if end not in quarters or filed > quarters[end][0]:
            quarters[end] = (filed, float(entry["val"]))
    if len(quarters) < 4:
        return None
    recent = sorted(quarters)[-4:]
    return sum(quarters[end][1] for end in recent)


def _fcf_by_year(facts: dict, as_of: date) -> dict[date, float]:
    """Annual free cash flow (OCF - capex) for years reporting both."""
    ocf = _annual_values(facts, _OCF, as_of)
    capex = _annual_values(facts, _CAPEX, as_of)
    return {end: ocf[end] - capex[end] for end in sorted(ocf) if end in capex}


# --------------------------------------------------------------------------
# The gate (pure over a facts payload)
# --------------------------------------------------------------------------

def evaluate_quality(symbol: str, facts: dict, as_of: date) -> QualityReport:
    symbol = symbol.upper()
    if symbol in EXCLUDED_NON_COMPANY:
        return QualityReport(
            symbol, as_of, False, None, None, None, None, (), 0, None,
            (f"excluded from the quality universe: {EXCLUDED_NON_COMPANY[symbol]}",),
        )

    reasons: list[str] = []

    net_income = _trailing_net_income(facts, as_of)
    equity = _latest_instant(facts, _EQUITY, as_of)
    roe: float | None = None
    if net_income is None:
        reasons.append("no trailing net income fact filed by the as-of date")
    if equity is None:
        reasons.append("no stockholders' equity fact filed by the as-of date")
    elif equity <= 0:
        reasons.append(f"non-positive stockholders' equity ({equity:,.0f}); ROE undefined")
    if net_income is not None and equity is not None and equity > 0:
        roe = net_income / equity
        if roe < ROE_FLOOR:
            reasons.append(f"ROE {roe:.1%} below the {ROE_FLOOR:.0%} floor")

    fcf = _fcf_by_year(facts, as_of)
    fcf_items = tuple(sorted(fcf.items()))
    latest_fcf = fcf_items[-1][1] if fcf_items else None
    window = fcf_items[-FCF_WINDOW_YEARS:]
    positive_in_window = sum(1 for _, value in window if value > 0)
    if len(fcf_items) < FCF_MIN_YEARS_AVAILABLE:
        reasons.append(
            f"only {len(fcf_items)} fiscal years with matched OCF/capex; "
            f"need at least {FCF_MIN_YEARS_AVAILABLE}"
        )
    else:
        if latest_fcf is not None and latest_fcf <= 0:
            reasons.append(f"latest fiscal-year FCF is not positive ({latest_fcf:,.0f})")
        if positive_in_window < FCF_MIN_POSITIVE_YEARS:
            reasons.append(
                f"FCF positive in {positive_in_window} of the last {len(window)} fiscal "
                f"years; need at least {FCF_MIN_POSITIVE_YEARS} of {FCF_WINDOW_YEARS}"
            )

    assets = _latest_instant(facts, _ASSETS, as_of)
    liabilities = _latest_instant(facts, _LIABILITIES, as_of)
    leverage: float | None = None
    if assets is None or liabilities is None:
        reasons.append("no assets/liabilities facts filed by the as-of date")
    elif assets <= 0:
        reasons.append(f"non-positive total assets ({assets:,.0f}); leverage undefined")
    else:
        leverage = liabilities / assets
        if leverage > LEVERAGE_CAP:
            reasons.append(f"leverage {leverage:.2f} above the {LEVERAGE_CAP:.2f} cap")

    return QualityReport(
        symbol=symbol,
        as_of=as_of,
        passes=not reasons,
        roe=roe,
        net_income=net_income,
        equity=equity,
        latest_fcf=latest_fcf,
        fcf_by_year=fcf_items,
        fcf_positive_in_window=positive_in_window,
        leverage=leverage,
        reasons=tuple(reasons),
    )


def quality_as_of(symbol: str, as_of: date, facts: dict | None = None) -> QualityReport:
    """The pre-registered quality gate for a symbol on a date.

    ``facts`` may be supplied (tests, batch evaluation over one
    cached payload); otherwise the symbol's companyfacts are loaded
    from the EDGAR cache / fetched once.
    """
    if facts is None:
        facts = load_company_facts(symbol)
    return evaluate_quality(symbol, facts, as_of)
