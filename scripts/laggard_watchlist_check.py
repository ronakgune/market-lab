#!/usr/bin/env python3
"""Laggard Top Five — evidence watchlist check (PAPER/WATCHLIST, no capital).

Ronak ruled 2026-11-21: INTU, APP, NOW, ADBE, IT (the "laggard top five"
from the Nov 20, 2026 research report) go on an evidence watchlist.
There are NO positions, NO orders, and NO paper trades attached to this
list — it exists to catch the moment a name's thesis breaks or confirms,
per the invalidation triggers / confirming events stored in the state
file. Price moves are recorded as context and NEVER produce a flag by
themselves; if the list ever graduates, entry staging is thesis-triggered,
never price-thirds.

Each run:
- Refreshes provenanced Alpaca SIP daily bars for exactly the five
  symbols (market_lab.daily_data; credentials are env-only — source
  ~/.config/market-lab/alpaca.env first) and records the last close and
  its % distance from the Nov 20, 2026 reference close.
- Scans SEC EDGAR submissions per name for new filings of the watched
  forms (8-K, 10-Q, 10-K, 4) since the stored watermark. The watermark
  advances only on a successful scan. EDGAR access uses the repo's
  declared-contact User-Agent pattern (EDGAR_USER_AGENT env var, via
  market_lab.intraday.events.edgar_user_agent).

Stdout contract (the weekly cron worker keys off exactly this):
- `WATCHLIST_QUIET` — no new filings, no data problems.
- `WATCHLIST_FLAG <TICKER> <reason> — report: <path>` — one line per new
  watched-form filing.
- `WATCHLIST_ERROR <detail>` — the run failed (exit code non-zero);
  a missing-credentials failure names MissingCredentialsError.

Usage:
    set -a && . ~/.config/market-lab/alpaca.env && set +a
    python scripts/laggard_watchlist_check.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from market_lab.config import (
    LAGGARD_WATCHLIST_REPORT_DIR,
    LAGGARD_WATCHLIST_STATE,
    ensure_dirs,
)
from market_lab.daily_data import backfill_daily, load_daily_bars
from market_lab.intraday.alpaca_fetcher import MissingCredentialsError
from market_lab.intraday.events import edgar_user_agent

TICKERS = ["INTU", "APP", "NOW", "ADBE", "IT"]
REFERENCE_DATE = "2026-11-20"
WATCHED_FORMS = {"8-K", "10-Q", "10-K", "4"}
BACKFILL_WINDOW = timedelta(days=365 * 10 + 30)
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"

# Thesis data as ruled 2026-11-21 from the Nov 20, 2026 research report.
# Reference prices are the report's Nov 20, 2026 closes.
THESIS: dict[str, dict] = {
    "INTU": {
        "reference_price": 270.44,
        "invalidation_triggers": [
            "TurboTax paid units fall again in FY27 tax season AND QuickBooks customer growth misses",
            "The FY27 EPS guide (+23-24%) gets cut",
        ],
        "confirming_events": [
            "Earnings print / guide held or raised confirming the intact-franchise read",
            "Any open-market insider buying",
        ],
        "notes": [],
    },
    "APP": {
        "reference_price": 348.29,
        "invalidation_triggers": [
            "Q4 (due ~Feb 2027) misses its $2.365-2.395B revenue guide",
            "Gaming growth falls below ~30%",
            "E-commerce fails its Q1 seasonal test",
            "A platform (Meta/Google/Apple) actually restricts AXON data access",
        ],
        "confirming_events": [
            "Q4 beat / guide held or raised",
            "A first open-market insider buy (thesis-confirming against the standing selling pattern)",
            "Any open-market insider buying",
        ],
        "notes": [
            "Standing insider pattern (context, not a trigger): ~$475M of insider selling over the past year with ZERO open-market buys; a first open-market insider buy is thesis-confirming",
        ],
    },
    "NOW": {
        "reference_price": 87.61,
        "invalidation_triggers": [
            "cRPO decelerates again in Q4/Q1 (a second consecutive guide-down)",
        ],
        "confirming_events": [
            "cRPO re-acceleration",
            "Earnings print / guide held or raised confirming the intact-franchise read",
            "Any open-market insider buying",
        ],
        "notes": [],
    },
    "ADBE": {
        "reference_price": 217.71,
        "invalidation_triggers": [
            "Revenue growth falls below 10%",
            "RPO growth decelerates again",
            "Net-new ARR misses repeat under the new CEO without an AI-monetization offset",
        ],
        "confirming_events": [
            "ARR stabilization",
            "Earnings print / guide held or raised confirming the intact-franchise read",
            "Any open-market insider buying",
        ],
        "notes": ["Dated catalyst: ADBE Q4 print Dec 9, 2026"],
    },
    "IT": {
        "reference_price": 174.49,
        "invalidation_triggers": [
            "Contract-value (CV) growth stalls at ~2% or turns negative for consecutive quarters",
            "Retention slips below ~97%",
        ],
        "confirming_events": [
            "CV re-acceleration",
            "Earnings print / guide held or raised confirming the intact-franchise read",
            "Any open-market insider buying",
        ],
        "notes": [],
    },
}


# ---------------------------------------------------------------- state

def default_state() -> dict:
    """Fresh watchlist state: thesis data in, no watermarks, never checked."""
    return {
        "watchlist": "laggard_top_five",
        "mode": "paper_watchlist_no_capital",
        "reference_date": REFERENCE_DATE,
        "created": None,
        "tickers": {
            ticker: {
                "reference_price": spec["reference_price"],
                "reference_date": REFERENCE_DATE,
                "invalidation_triggers": list(spec["invalidation_triggers"]),
                "confirming_events": list(spec["confirming_events"]),
                "notes": list(spec["notes"]),
                "status": "watching",
                "edgar_watermark": {"filing_date": None, "accessions": []},
                "last_check": None,
                "last_close": None,
                "last_close_date": None,
                "pct_vs_reference": None,
                "reference_date_close": None,
            }
            for ticker, spec in THESIS.items()
        },
    }


def load_state(path: Path) -> Optional[dict]:
    """Return the stored state, or None when no state file exists yet."""
    if not Path(path).exists():
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_state(path: Path, state: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


# ------------------------------------------------------------ credentials

def ensure_credentials() -> None:
    """Alpaca credentials are env-only by design; fail loudly and by name."""
    if not os.environ.get("ALPACA_API_KEY", "").strip() or not os.environ.get(
        "ALPACA_API_SECRET", ""
    ).strip():
        raise MissingCredentialsError(
            "Alpaca credentials not found: set ALPACA_API_KEY and "
            "ALPACA_API_SECRET in the environment (source "
            "~/.config/market-lab/alpaca.env first; no fallback files are read)."
        )


# ----------------------------------------------------------------- EDGAR

def _http_get_json(url: str) -> Mapping:
    request = urllib.request.Request(
        url, headers={"User-Agent": edgar_user_agent(), "Accept": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def resolve_cik_map(http_get: Callable[[str], Mapping] = _http_get_json) -> dict[str, int]:
    """Resolve ticker -> CIK from the SEC company tickers file."""
    payload = http_get(SEC_TICKERS_URL)
    rows = payload.values() if isinstance(payload, Mapping) else payload
    out: dict[str, int] = {}
    for row in rows:
        ticker = str(row.get("ticker", "")).upper()
        if ticker:
            out[ticker] = int(row["cik_str"])
    return out


def fetch_submissions(cik: int, http_get: Callable[[str], Mapping] = _http_get_json) -> Mapping:
    return http_get(SEC_SUBMISSIONS_URL.format(cik=cik))


def _base_form(form: str) -> str:
    return form.strip().upper().removesuffix("/A")


def parse_recent_filings(payload: Mapping) -> list[dict]:
    """Watched-form filings from a submissions payload's `recent` block.

    Returns dicts {form, filing_date, accession} sorted by (date, accession).
    Only the `recent` block is read: it covers years of history for these
    names, and the watermark only ever needs the frontier.
    """
    recent = (payload.get("filings") or {}).get("recent") or {}
    forms = recent.get("form") or []
    dates = recent.get("filingDate") or []
    accessions = recent.get("accessionNumber") or []
    out: list[dict] = []
    for form, filed, accession in zip(forms, dates, accessions):
        if _base_form(str(form)) in WATCHED_FORMS:
            out.append(
                {"form": str(form), "filing_date": str(filed), "accession": str(accession)}
            )
    out.sort(key=lambda f: (f["filing_date"], f["accession"]))
    return out


def filings_since(filings: Sequence[dict], watermark: Optional[dict]) -> list[dict]:
    """Filings strictly newer than the watermark (all of them if unset)."""
    if not watermark or not watermark.get("filing_date"):
        return list(filings)
    wm_date = watermark["filing_date"]
    seen = set(watermark.get("accessions") or [])
    return [
        f
        for f in filings
        if f["filing_date"] > wm_date
        or (f["filing_date"] == wm_date and f["accession"] not in seen)
    ]


def advance_watermark(filings: Sequence[dict], watermark: Optional[dict]) -> dict:
    """Move the watermark to the newest filing date in `filings`."""
    if not filings:
        return dict(watermark) if watermark else {"filing_date": None, "accessions": []}
    latest = max(f["filing_date"] for f in filings)
    accessions = {f["accession"] for f in filings if f["filing_date"] == latest}
    if watermark and watermark.get("filing_date") == latest:
        accessions |= set(watermark.get("accessions") or [])
    return {"filing_date": latest, "accessions": sorted(accessions)}


def flag_lines(ticker: str, new_filings: Sequence[dict], report_path: Path) -> list[str]:
    """One WATCHLIST_FLAG line per new watched-form filing. Filings only —
    price movement is context and never reaches this function."""
    return [
        f"WATCHLIST_FLAG {ticker} new {f['form']} filing {f['accession']} "
        f"filed {f['filing_date']} — report: {report_path}"
        for f in new_filings
    ]


# ------------------------------------------------------------------- run

def run_check(
    state_path: Path = LAGGARD_WATCHLIST_STATE,
    report_dir: Path = LAGGARD_WATCHLIST_REPORT_DIR,
    *,
    today: Optional[date] = None,
    backfill: Callable = backfill_daily,
    load_bars: Callable = load_daily_bars,
    cik_map: Optional[dict[str, int]] = None,
    submissions_fetcher: Callable[[int], Mapping] = fetch_submissions,
) -> tuple[int, list[str]]:
    """Run one watchlist check. Returns (exit_code, stdout_lines)."""
    ensure_credentials()
    today = today or datetime.now(timezone.utc).date()
    run_at = datetime.now(timezone.utc)

    state = load_state(state_path)
    baseline = state is None
    if baseline:
        state = default_state()
        state["created"] = run_at.isoformat()

    report: dict = {
        "run_at": run_at.isoformat(),
        "baseline": baseline,
        "reference_date": REFERENCE_DATE,
        "tickers": {},
        "flags": [],
        "errors": [],
    }

    # --- prices: refresh exactly the five, through the provenanced cache.
    results = backfill(TICKERS, today - BACKFILL_WINDOW, today - timedelta(days=1))
    backfill_errors = {r.symbol: r.error for r in results if r.error}
    for ticker in TICKERS:
        entry = state["tickers"][ticker]
        trep: dict = {"backfill_error": backfill_errors.get(ticker)}
        if ticker in backfill_errors:
            report["errors"].append(f"{ticker} backfill failed: {backfill_errors[ticker]}")
        else:
            try:
                bars, _meta = load_bars(ticker)
                if bars:
                    last = bars[-1]
                    entry["last_close"] = last.close
                    entry["last_close_date"] = last.date.isoformat()
                    entry["pct_vs_reference"] = round(
                        (last.close / entry["reference_price"] - 1.0) * 100.0, 2
                    )
                    ref_bar = next(
                        (b for b in bars if b.date.isoformat() == REFERENCE_DATE), None
                    )
                    entry["reference_date_close"] = ref_bar.close if ref_bar else None
                    trep.update(
                        {
                            "last_close": entry["last_close"],
                            "last_close_date": entry["last_close_date"],
                            "pct_vs_reference": entry["pct_vs_reference"],
                            "reference_date_close": entry["reference_date_close"],
                        }
                    )
            except Exception as exc:  # cache/provenance problem: report, don't invent
                report["errors"].append(f"{ticker} bar load failed: {type(exc).__name__}: {exc}")
                trep["load_error"] = f"{type(exc).__name__}: {exc}"
        report["tickers"][ticker] = trep

    # --- EDGAR: one CIK-map fetch, then a submissions scan per name.
    if cik_map is None:
        try:
            cik_map = resolve_cik_map()
        except Exception as exc:
            report["errors"].append(
                f"EDGAR CIK resolution failed: {type(exc).__name__}: {exc}"
            )
            cik_map = {}
    flags: list[str] = []
    report_path = Path(report_dir) / f"check-{run_at.strftime('%Y%m%dT%H%M%SZ')}.json"
    for ticker in TICKERS:
        entry = state["tickers"][ticker]
        trep = report["tickers"][ticker]
        cik = (cik_map or {}).get(ticker)
        if cik is None:
            report["errors"].append(f"{ticker} EDGAR scan failed: CIK not resolved")
            trep["edgar_error"] = "CIK not resolved"
            continue
        try:
            filings = parse_recent_filings(submissions_fetcher(cik))
        except Exception as exc:
            report["errors"].append(
                f"{ticker} EDGAR scan failed: {type(exc).__name__}: {exc}"
            )
            trep["edgar_error"] = f"{type(exc).__name__}: {exc}"
            continue  # watermark does NOT advance on a failed scan
        watermark = entry.get("edgar_watermark")
        if baseline:
            # First-ever run for this watchlist: everything seen is
            # baseline, never flagged. (Later runs compare against the
            # watermark even when it is still empty — a name with no
            # filings at baseline must still flag its first filing.)
            trep["baseline_filings"] = filings
            trep["new_filings"] = []
        else:
            new = filings_since(filings, watermark)
            trep["new_filings"] = new
            flags.extend(flag_lines(ticker, new, report_path))
        entry["edgar_watermark"] = advance_watermark(filings, watermark)
        entry["last_check"] = today.isoformat()
        trep["watermark"] = entry["edgar_watermark"]

    report["flags"] = flags
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    save_state(state_path, state)

    if report["errors"]:
        return 1, [f"WATCHLIST_ERROR {e}" for e in report["errors"]]
    if flags:
        return 0, flags
    return 0, ["WATCHLIST_QUIET"]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=LAGGARD_WATCHLIST_STATE)
    parser.add_argument("--report-dir", type=Path, default=LAGGARD_WATCHLIST_REPORT_DIR)
    args = parser.parse_args(argv)
    ensure_dirs()
    try:
        code, lines = run_check(args.state, args.report_dir)
    except MissingCredentialsError as exc:
        print(f"WATCHLIST_ERROR MissingCredentialsError: {exc}")
        return 2
    except Exception as exc:  # surface, never swallow
        print(f"WATCHLIST_ERROR {type(exc).__name__}: {exc}")
        return 1
    for line in lines:
        print(line)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
