#!/usr/bin/env python3
"""Weekly roll-up across the three forward paper trackers.

Summarizes, for the week ending today:

- XS momentum (Stream 2) and vt_trend forward (Stream 1): equity,
  cash, positions, week change (from each tracker's own marked equity
  history), and any fills in the last 7 days (from its ledger).
- Crypto carry (Stream 6): monitor-only — no book. Reports the latest
  30-day funding picture per venue/asset and whether the US-venue
  trigger (Kraken 30d mean > 8% annualized, sustained) is met.

Every input is read defensively: a tracker with no state yet (or a
missing file) is reported as such instead of failing the whole
roll-up. The summary is always produced — unlike the daily jobs,
the weekly roll-up is itself the message.

Usage:
    python scripts/weekly_rollup.py
"""

from __future__ import annotations

import json
import math
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from market_lab.broker import load_portfolio
from market_lab.config import (
    CRYPTO_CARRY_FUNDING_PATH,
    REPORT_DIR,
    VT_TREND_FORWARD_EQUITY_HISTORY,
    VT_TREND_FORWARD_LEDGER,
    VT_TREND_FORWARD_STARTING_CASH,
    VT_TREND_FORWARD_STATE,
    XS_MOMENTUM_EQUITY_HISTORY,
    XS_MOMENTUM_LEDGER,
    XS_MOMENTUM_STARTING_CASH,
    XS_MOMENTUM_STATE,
    ensure_dirs,
)
from market_lab.crypto_carry import load_points, summarize

US_VENUES = {"kraken"}  # mirror of scripts/crypto_carry_monitor.py


def _equity_rows(path: Path) -> list[tuple[str, float]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            day, equity = row["date"], float(row["equity"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
        if isinstance(day, str) and math.isfinite(equity):
            rows.append((day, equity))
    return sorted(rows)


def _week_change(rows: list[tuple[str, float]], as_of: date) -> tuple[float, float, str] | None:
    """(abs change, pct change, baseline date) of the latest mark vs the
    last mark at least 7 days older; falls back to the earliest mark
    when the history is shorter than a week."""
    if not rows:
        return None
    latest_day, latest = rows[-1]
    cutoff = (as_of - timedelta(days=7)).isoformat()
    baseline = None
    for day, equity in rows[:-1]:
        if day <= cutoff:
            baseline = (day, equity)
    if baseline is None:
        baseline = rows[0]
    change = latest - baseline[1]
    pct = change / baseline[1] if baseline[1] > 0 else 0.0
    return change, pct, baseline[0]


def _recent_fills(ledger_path: Path, as_of: date, days: int = 7) -> list[dict]:
    if not ledger_path.exists():
        return []
    since = (as_of - timedelta(days=days)).isoformat()
    fills = []
    for line in ledger_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        executed = row.get("execution_date") or ""
        if row.get("accepted") and since < executed <= as_of.isoformat():
            fills.append(row)
    return fills


def summarize_book(name: str, state_path: Path, equity_path: Path, ledger_path: Path,
                   starting_cash: float, as_of: date) -> list[str]:
    lines = [f"### {name}", ""]
    if not state_path.exists():
        lines += ["- No state yet — tracker has not run.", ""]
        return lines
    portfolio = load_portfolio(state_path)
    rows = _equity_rows(equity_path)
    # Latest marked equity; before the first mark exists, fall back to
    # cost-basis equity (Portfolio.equity values missing prices at avg).
    equity = rows[-1][1] if rows else portfolio.equity({})
    lines.append(
        f"- Equity: ${equity:,.2f} (cash ${portfolio.cash:,.2f}; "
        f"{equity - starting_cash:+,.2f} vs the ${starting_cash:,.0f} start)"
    )
    if portfolio.positions:
        lines.append("- Positions: " + ", ".join(
            f"{sym} {pos.quantity} @ ${pos.avg_price:.2f}"
            for sym, pos in sorted(portfolio.positions.items())
        ))
    else:
        lines.append("- Positions: none")
    change = _week_change(rows, as_of)
    if change is not None:
        delta, pct, baseline_day = change
        lines.append(f"- Week change: {delta:+,.2f} ({pct:+.2%}) vs {baseline_day}")
    else:
        lines.append("- Week change: no equity marks yet")
    fills = _recent_fills(ledger_path, as_of)
    if fills:
        lines.append(f"- Fills this week: {len(fills)}")
        for f in fills:
            price = f.get("fill_price")
            price_txt = f" @ ${price:,.2f}" if isinstance(price, (int, float)) else ""
            lines.append(
                f"  - {f.get('execution_date')}: {f.get('side')} {f.get('quantity')} "
                f"{f.get('symbol')}{price_txt}"
            )
    else:
        lines.append("- Fills this week: none")
    lines.append("")
    return lines


def summarize_carry(funding_path: Path) -> list[str]:
    lines = ["### Crypto carry (Stream 6 — monitor only)", ""]
    points = load_points(funding_path) if funding_path.exists() else []
    if not points:
        lines += ["- No funding data stored yet.", ""]
        return lines
    summaries = summarize(points)
    any_us_trigger = False
    for s in summaries:
        mark = "STALE" if s.stale else ("TRIGGERED" if s.triggered else "—")
        if s.triggered and s.venue in US_VENUES:
            any_us_trigger = True
        lines.append(
            f"- {s.venue} {s.asset}: 30d mean {s.mean_annualized * 100:.2f}% ann. "
            f"({s.days_covered} days covered) — trigger: {mark}"
        )
    lines.append(
        "- Verdict: " + (
            "US-venue (Kraken) trigger MET — the carry trader design conversation is open."
            if any_us_trigger else
            "No US-venue trigger; carry stays monitor-only."
        )
    )
    lines.append("")
    return lines


def build_rollup(as_of: date | None = None,
                 xs_paths: tuple[Path, Path, Path] | None = None,
                 vt_paths: tuple[Path, Path, Path] | None = None,
                 carry_path: Path | None = None) -> str:
    as_of = as_of or date.today()
    xs_paths = xs_paths or (XS_MOMENTUM_STATE, XS_MOMENTUM_EQUITY_HISTORY, XS_MOMENTUM_LEDGER)
    vt_paths = vt_paths or (VT_TREND_FORWARD_STATE, VT_TREND_FORWARD_EQUITY_HISTORY, VT_TREND_FORWARD_LEDGER)
    carry_path = carry_path or CRYPTO_CARRY_FUNDING_PATH
    lines = [
        f"# Market Lab Weekly Roll-Up — week ending {as_of.isoformat()}",
        "",
    ]
    lines += summarize_book("XS momentum (Stream 2)", *xs_paths, XS_MOMENTUM_STARTING_CASH, as_of)
    lines += summarize_book("vt_trend forward (Stream 1)", *vt_paths, VT_TREND_FORWARD_STARTING_CASH, as_of)
    lines += summarize_carry(carry_path)
    lines += ["*Research/paper-only. No live broker orders. No investment advice.*", ""]
    return "\n".join(lines)


def main() -> int:
    ensure_dirs()
    report = build_rollup()
    stamp = date.today().strftime("%Y%m%d")
    (REPORT_DIR / f"weekly-rollup-{stamp}.md").write_text(report)
    (REPORT_DIR / "weekly-rollup-latest.md").write_text(report)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
