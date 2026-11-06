"""Scanner: universe construction, daily setups, opening-range breakouts.

Adapted from trader-scanner in day-trader-skill (MIT,
Jakeschincariol/day-trader-skill) and re-engineered for Market Lab:

  * All functions are pure and offline. Universe building takes the text
    of Nasdaq Trader's symbol-directory files as input; daily setups take
    close series; ORB detection takes ``SessionBar`` sequences from our
    own store. Fetching, where a caller wants it, lives outside this
    module — no Yahoo endpoints, no network imports here.
  * A scan hit is a chart matching a rule. It is never a recommendation,
    and this module never sizes, prices an order, or places one.

Universe rule (in order, same as the source — it was the good part):
  1. not a test issue, not an ETF, not NextShares
  2. plain ticker: 1-5 letters, or 1-4 letters + one class letter (BRK.B);
     .U/.W/.R suffixes (units/warrants/rights) are out
  3. out on name: warrants; SPAC units/rights phrasing; notes/ETNs/
     debentures; a % coupon; non-American/Global depositary shares
  4. in on name: common stock/shares/units, ordinary shares, shares of
     beneficial interest, ADS/ADR, American/Global depositary, partnership
     or trust units
  5. out if the name still says unit / right / bond / preferred
  6. in: everything else
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Mapping, Sequence

from .session import SessionBar

# ---------------------------------------------------------------- universe

_TICKER = re.compile(r"^[A-Z]{1,5}$|^[A-Z]{1,4}\.[A-Z]$")
_ALWAYS_OUT = [
    ("warrant", re.compile(r"\bwarrants?\b")),
    ("unit or right", re.compile(r"\b(units?|rights?),? (each|that|entitling)\b|(?<!the )\brights? to\b")),
    ("note", re.compile(r"\b(notes?|etns?|debentures?)\b")),
    ("coupon", re.compile(r"%")),
    ("depositary", re.compile(r"(?<!american )(?<!global )\bdepositary shares?\b")),
]
_COMMON = re.compile(
    r"\bcommon (stock|shares?|units?)\b|\bord(inary)? shares?\b|\bshares? of beneficial interest\b"
    r"|\b(american|global) deposit[ao]ry\b|\bads[s]?\b|\badrs?\b"
    r"|\b(limited partnership|partnership|trust) units?\b|\bl\.?p\.? units?\b"
    r"|\bunits? (of|representing)\b.*\b(limited partner|partnership|beneficial interest|limited liability)")
_SOFT_OUT = [
    ("unit", re.compile(r"\bunits?\b")),
    ("right", re.compile(r"\brights?\b")),
    ("bond", re.compile(r"\bbonds?\b")),
    ("preferred", re.compile(r"\bpreferred\b|\bpreference\b")),
]

NASDAQ_FILES = ("nasdaqlisted.txt", "otherlisted.txt")


@dataclass(frozen=True)
class UniverseEntry:
    symbol: str
    name: str


def classify_symbol_row(row: Mapping[str, str]) -> str:
    """One symbol-directory row -> 'keep' or the rule that drops it."""
    sym = (row.get("Symbol") or row.get("ACT Symbol") or "").strip()
    if (row.get("Test Issue") or "").strip() != "N":
        return "test issue"
    if (row.get("ETF") or "").strip() != "N":
        return "ETF"
    if (row.get("NextShares") or "N").strip() != "N":
        return "NextShares"
    if not _TICKER.match(sym):
        return "not a plain ticker"
    if re.search(r"\.[UWR]$", sym):
        return "unit / warrant / right suffix"
    name = " ".join((row.get("Security Name") or "").lower().split())
    for tag, rx in _ALWAYS_OUT:
        if rx.search(name):
            return tag
    if _COMMON.search(name):
        return "keep"
    for tag, rx in _SOFT_OUT:
        if rx.search(name):
            return tag
    return "keep"


def parse_symbol_directory(text: str) -> tuple[list[dict[str, str]], str | None]:
    """Nasdaq Trader pipe file -> (rows, file-creation stamp or None)."""
    lines = [ln for ln in text.replace("\r", "").split("\n") if ln.strip()]
    if not lines or "|" not in lines[0]:
        raise ValueError("not a Nasdaq Trader symbol directory")
    hdr = [h.strip() for h in lines[0].split("|")]
    if "Security Name" not in hdr or not ({"Symbol", "ACT Symbol"} & set(hdr)):
        raise ValueError("not a Nasdaq Trader symbol directory (bad header)")
    rows: list[dict[str, str]] = []
    created: str | None = None
    for ln in lines[1:]:
        if ln.startswith("File Creation Time"):
            m = re.search(r"(\d{2})(\d{2})(\d{4})(\d{2}):(\d{2})", ln)
            if m:
                created = f"{m.group(3)}-{m.group(1)}-{m.group(2)} {m.group(4)}:{m.group(5)}"
            continue
        rows.append(dict(zip(hdr, ln.split("|"))))
    return rows, created


def build_universe(
    texts: Mapping[str, str],
) -> tuple[list[UniverseEntry], dict[str, int], str | None]:
    """{filename: text} for both Nasdaq files -> (kept, counts, created)."""
    kept: dict[str, str] = {}
    counts: dict[str, int] = {}
    created: str | None = None
    for fname in NASDAQ_FILES:
        if fname not in texts:
            raise ValueError(f"missing {fname}")
        rows, c = parse_symbol_directory(texts[fname])
        created = created or c
        counts["rows"] = counts.get("rows", 0) + len(rows)
        for r in rows:
            why = classify_symbol_row(r)
            counts[why] = counts.get(why, 0) + 1
            if why == "keep":
                sym = (r.get("Symbol") or r.get("ACT Symbol")).strip()
                kept[sym] = " ".join((r.get("Security Name") or "").split())
    return [UniverseEntry(s, kept[s]) for s in sorted(kept)], counts, created


# ---------------------------------------------------------------- daily setups

MIN_DAILY_BARS = 30


def rsi_wilder(closes: Sequence[float], n: int = 14) -> float | None:
    """Wilder's RSI-14 on closes; None when there isn't enough history."""
    if len(closes) < n + 1:
        return None
    gains = [max(0.0, closes[i] - closes[i - 1]) for i in range(1, len(closes))]
    losses = [max(0.0, closes[i - 1] - closes[i]) for i in range(1, len(closes))]
    ag = sum(gains[:n]) / n
    al = sum(losses[:n]) / n
    for i in range(n, len(gains)):
        ag = (ag * (n - 1) + gains[i]) / n
        al = (al * (n - 1) + losses[i]) / n
    if al == 0:
        return 100.0
    return 100 - 100 / (1 + ag / al)


@dataclass(frozen=True)
class DailySetup:
    symbol: str
    setups: tuple[str, ...]
    last_close: float
    close_date: date | None
    rsi: float | None
    volume: int | None


def find_daily_setups(closes: Sequence[float]) -> tuple[tuple[str, ...], dict]:
    """Setup tags for one close series (oldest first, >= MIN_DAILY_BARS).

    Rules are on closes only, exactly as the source defines them — a
    closing-high breakout is not an intraday-high breakout, and callers
    must not present it as one.
    """
    if len(closes) < MIN_DAILY_BARS:
        raise ValueError(f"need at least {MIN_DAILY_BARS} closes, got {len(closes)}")
    c = list(closes)
    last = c[-1]
    hi20 = max(c[-21:-1])
    ma20 = sum(c[-20:]) / 20.0
    ma20_prev = sum(c[-21:-1]) / 20.0
    run = c[-6] / c[-16] - 1
    rng5 = (max(c[-5:]) - min(c[-5:])) / last if last else 0.0
    r = rsi_wilder(c)
    r_prev = rsi_wilder(c[:-1])
    tags: list[str] = []
    if last > hi20:
        tags.append("breakout_20d")
    if run > 0.10 and rng5 < 0.05:
        tags.append("bull_flag")
    if c[-2] < ma20_prev and last > ma20:
        tags.append("reclaim_ma20")
    if r is not None and r < 30:
        tags.append("oversold")
    if r_prev is not None and r_prev < 30 and last > c[-2]:
        tags.append("oversold_bounce")
    return tuple(tags), {
        "rsi": r, "rsi_prev": r_prev, "high20": hi20, "ma20": ma20,
        "run_pct": run * 100, "range5_pct": rng5 * 100,
    }


def scan_daily(
    series: Mapping[str, Sequence[float]],
    *,
    min_price: float = 5.0,
    volumes: Mapping[str, int] | None = None,
    close_dates: Mapping[str, date] | None = None,
) -> list[DailySetup]:
    """Scan {symbol: closes} for daily setups. Most dollars traded first
    when volumes are supplied; symbols with too-short or illiquid series
    are skipped (count them at the call site if you need the audit)."""
    hits: list[DailySetup] = []
    for sym in sorted(series):
        closes = series[sym]
        if len(closes) < MIN_DAILY_BARS:
            continue
        if closes[-1] < min_price:
            continue
        vol = (volumes or {}).get(sym)
        tags, nums = find_daily_setups(closes)
        if tags:
            hits.append(DailySetup(
                symbol=sym, setups=tags, last_close=closes[-1],
                close_date=(close_dates or {}).get(sym),
                rsi=round(nums["rsi"], 1) if nums["rsi"] is not None else None,
                volume=vol))
    hits.sort(key=lambda h: (-(h.last_close * (h.volume or 0)), h.symbol))
    return hits


# ---------------------------------------------------------------- ORB

@dataclass(frozen=True)
class OrbHit:
    symbol: str
    day: date
    entry_slot: int
    entry_price: float  # close of the breakout bar
    or_high: float
    or_low: float
    pct_over: float


def opening_range(bars: Sequence[SessionBar], or_bars: int = 3) -> tuple[float, float]:
    """(high, low) of the first ``or_bars`` session bars."""
    if len(bars) < or_bars:
        raise ValueError(f"need at least {or_bars} bars for the opening range")
    rng = sorted(bars, key=lambda b: b.slot)[:or_bars]
    if [b.slot for b in rng] != list(range(or_bars)):
        raise ValueError(
            f"opening range needs slots 0..{or_bars - 1}, "
            f"got {[b.slot for b in rng]}")
    return max(b.high for b in rng), min(b.low for b in rng)


def orb_candidate(bars: Sequence[SessionBar], or_bars: int = 3) -> bool:
    """Stage-1 (closes only): some later bar closed at/above the highest
    CLOSE of the range — a necessary condition for a breakout, so no true
    breakout is lost by this cheap filter."""
    ordered = sorted(bars, key=lambda b: b.slot)
    if len(ordered) <= or_bars:
        return False
    if [b.slot for b in ordered[:or_bars]] != list(range(or_bars)):
        return False  # range bars missing: the filter cannot be trusted
    top_close = max(b.close for b in ordered[:or_bars])
    return any(b.close >= top_close for b in ordered[or_bars:])


def detect_orb(bars: Sequence[SessionBar], or_bars: int = 3,
               side: str = "long") -> OrbHit | None:
    """Stage-2: first completed bar after the range to CLOSE beyond the
    range extreme (above the high for long, below the low for short)."""
    if side not in ("long", "short"):
        raise ValueError("side must be 'long' or 'short'")
    ordered = sorted(bars, key=lambda b: b.slot)
    if len(ordered) <= or_bars:
        return None
    if [b.slot for b in ordered[:or_bars]] != list(range(or_bars)):
        return None  # no trustworthy opening range without its bars
    or_high, or_low = opening_range(ordered, or_bars)
    for b in ordered[or_bars:]:
        if side == "long" and b.close > or_high:
            return OrbHit(b.symbol, b.day, b.slot, round(b.close, 4),
                          round(or_high, 4), round(or_low, 4),
                          round((b.close / or_high - 1) * 100, 4))
        if side == "short" and b.close < or_low:
            return OrbHit(b.symbol, b.day, b.slot, round(b.close, 4),
                          round(or_high, 4), round(or_low, 4),
                          round((b.close / or_low - 1) * 100, 4))
    return None
