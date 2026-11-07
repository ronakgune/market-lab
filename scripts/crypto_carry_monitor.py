#!/usr/bin/env python3
"""Crypto carry funding monitor (Stream 6 — monitor only, no trader).

Fetches perpetual funding rates from the venues readable from the US
(OKX + Deribit history, Kraken current snapshots), appends them to the
normalized funding store, and reports the 30-day picture against the
pre-registered trigger: a trader is considered only when 30-day mean
funding exceeds 8% annualized, sustained (>= 20 days covered), on a
US-compliant venue. Offshore readings are context, not a GO signal.

Usage:
    python scripts/crypto_carry_monitor.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from market_lab.config import (
    CRYPTO_CARRY_FUNDING_PATH,
    CRYPTO_CARRY_REPORT_DIR,
    ensure_dirs,
)
from market_lab.crypto_carry import (
    append_points,
    fetch_deribit,
    fetch_kraken,
    fetch_okx,
    load_points,
    summarize,
)

US_VENUES = {"kraken"}  # venues whose trigger could ever authorize a trader


def collect(fetchers=None) -> tuple[list, list[str]]:
    """Run every fetcher; a venue's failure is recorded, never fatal."""
    if fetchers is None:
        fetchers = [
            lambda: fetch_okx("BTC"),
            lambda: fetch_okx("ETH"),
            lambda: fetch_deribit("BTC"),
            lambda: fetch_deribit("ETH"),
            fetch_kraken,
        ]
    points, errors = [], []
    for fetcher in fetchers:
        try:
            got, error = fetcher()
            points.extend(got)
        except Exception as exc:  # a fetcher that RAISES (or returns a
            # malformed result) is still just one venue failing — the
            # built-in fetchers catch their own errors, but collect()'s
            # contract is isolation regardless of the fetcher's manners.
            errors.append(f"fetcher failed: {exc}")
            continue
        if error:
            errors.append(error)
    return points, errors


def main() -> int:
    ensure_dirs()
    points, errors = collect()
    written = append_points(CRYPTO_CARRY_FUNDING_PATH, points)
    stored = load_points(CRYPTO_CARRY_FUNDING_PATH)
    summaries = summarize(stored)

    lines = [
        f"# Crypto Carry Monitor — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M %Z')}",
        "",
        f"Fetched {len(points)} points ({written} new) from OKX / Deribit / Kraken; "
        f"{len(stored)} points stored in total.",
        "",
        "| Venue | Asset | Points (30d) | Days | Mean ann. | Latest ann. | Share >8% | Trigger |",
        "|---|---|---|---|---|---|---|---|",
    ]
    any_us_trigger = False
    for s in summaries:
        mark = "STALE" if s.stale else ("TRIGGERED" if s.triggered else "—")
        if s.triggered and s.venue in US_VENUES:
            any_us_trigger = True
        lines.append(
            f"| {s.venue} | {s.asset} | {s.points} | {s.days_covered} | "
            f"{s.mean_annualized * 100:.2f}% | {s.latest_annualized * 100:.2f}% | "
            f"{s.share_above_trigger * 100:.0f}% | {mark} |"
        )
    lines += ["", "## Verdict", ""]
    if any_us_trigger:
        lines.append("A US-compliant venue (Kraken) shows sustained 30d funding above 8% annualized. "
                     "The carry TRADER design conversation is open — nothing trades automatically.")
    else:
        lines.append("No US-venue trigger. Carry stays monitor-only (hurdle: 30d mean > 8% annualized, "
                     "sustained >= 20 days, on a US-compliant venue; T-bills ~4% are the alternative).")
    if errors:
        lines += ["", "## Fetch errors (non-fatal)", ""] + [f"- {e}" for e in errors]
    lines += ["", "*Research/paper-only. No live broker orders. No investment advice.*", ""]
    report = "\n".join(lines)

    CRYPTO_CARRY_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (CRYPTO_CARRY_REPORT_DIR / f"carry-{stamp}.md").write_text(report)
    (CRYPTO_CARRY_REPORT_DIR / "latest.md").write_text(report)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
