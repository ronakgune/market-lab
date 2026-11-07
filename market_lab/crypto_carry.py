"""Crypto perpetual funding-rate monitor (Stream 6 — monitor only, no trader).

The carry premium (long spot / short perp collecting funding) is real but
currently pays less than T-bills. The research decision (Nov 6, 2026):
build the MONITOR now so a funding history exists when conditions change;
a trader is considered only when 30-day net funding exceeds 8% annualized
sustained on a US-compliant venue.

Venues readable from here (checked live, Nov 6 2026):
- OKX public funding history (8h periods) — offshore reference.
- Deribit public funding history (hourly) — offshore reference.
- Kraken Futures public tickers (hourly relative funding) — the closest
  US-facing venue; current snapshots only, history accrues from our runs.
Binance and Bybit public APIs are geo-blocked from the US and are not used.

Points are normalized to (venue, asset, timestamp, per-period rate,
period hours) and stored append-only as JSONL; duplicates by
(venue, asset, timestamp) are never written twice.
"""

from __future__ import annotations

import json
import math
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

TRIGGER_ANNUALIZED = 0.08  # 30-day mean funding above this = carry beats the hurdle
MIN_TRIGGER_DAYS = 20      # ...and it must be sustained, not a spike


@dataclass(frozen=True)
class FundingPoint:
    venue: str
    asset: str  # "BTC" | "ETH"
    ts: datetime  # timezone-aware, UTC
    rate: float  # relative funding per period (e.g. 0.0001 = 0.01% per period)
    period_hours: float

    def __post_init__(self):
        # A naive timestamp (e.g. a caller passing datetime.now() to the
        # Kraken parser) would crash sorting/subtraction the moment it met
        # an aware point in summarize. Interpret naive as UTC, the only
        # timezone this module speaks.
        if self.ts.tzinfo is None or self.ts.tzinfo.utcoffset(self.ts) is None:
            object.__setattr__(self, "ts", self.ts.replace(tzinfo=timezone.utc))

    def annualized(self) -> float:
        return self.rate * (24.0 / self.period_hours) * 365.0

    def key(self) -> tuple[str, str, str]:
        # Identity is one point per (venue, asset, funding period), not per
        # exact instant: Kraken points are snapshots stamped "now", so exact-
        # timestamp dedupe lets run frequency weight the 30d mean (a burst
        # of runs inside one rich hour could manufacture a trigger). Real
        # venue settlements sit on period boundaries, so bucketing loses
        # nothing genuine.
        step = self.period_hours * 3600.0
        if not step > 0 or not math.isfinite(step):
            return (self.venue, self.asset, self.ts.isoformat())
        bucket = math.floor(self.ts.timestamp() / step) * step
        bucket_ts = datetime.fromtimestamp(bucket, tz=timezone.utc)
        return (self.venue, self.asset, bucket_ts.isoformat())


@dataclass(frozen=True)
class CarrySummary:
    venue: str
    asset: str
    points: int
    days_covered: int
    mean_annualized: float
    latest_annualized: float
    share_above_trigger: float
    triggered: bool
    stale: bool = False  # latest point lags the store's freshest by > window


# ---------------------------------------------------------------------------
# Parsers (pure functions over venue payloads — fixture-tested)
# ---------------------------------------------------------------------------

def _ts_from_epoch(value: int) -> datetime:
    """Venue contracts say milliseconds. A seconds value (~1e9) divided by
    1000 lands in 1970 and would silently poison the store, so normalize
    by magnitude: anything below 1e11 is seconds, not milliseconds."""
    seconds = value / 1000.0 if value >= 100_000_000_000 else float(value)
    return datetime.fromtimestamp(seconds, tz=timezone.utc)


def parse_okx_history(payload: dict, asset: str) -> list[FundingPoint]:
    points = []
    for row in payload.get("data", []):
        try:
            # Settled-history rows carry two rates: fundingRate is the
            # announced/predicted rate, realizedRate is what was actually
            # charged. A carry measurement uses the charged rate.
            raw_rate = row.get("realizedRate") or row.get("fundingRate")
            rate = float(raw_rate)
            if not math.isfinite(rate):
                continue  # a poisoned row loses one point, never the run
            ts = _ts_from_epoch(int(row["fundingTime"]))
        except (ValueError, KeyError, TypeError):
            continue
        points.append(FundingPoint("okx", asset, ts, rate, 8.0))
    return points


def parse_deribit_history(payload: dict, asset: str) -> list[FundingPoint]:
    points = []
    for row in payload.get("result", []):
        try:
            # interest_1h is the hourly rate; interest_8h is a different
            # quantity and using it would misstate the carry ~8x.
            rate = float(row["interest_1h"])
            if not math.isfinite(rate):
                continue
            ts = _ts_from_epoch(int(row["timestamp"]))
        except (ValueError, KeyError, TypeError):
            continue
        points.append(FundingPoint("deribit", asset, ts, rate, 1.0))
    return points


_KRAKEN_ASSETS = {"PF_XBTUSD": "BTC", "PF_ETHUSD": "ETH"}


def parse_kraken_tickers(payload: dict, now: datetime) -> list[FundingPoint]:
    points = []
    for ticker in payload.get("tickers", []):
        asset = _KRAKEN_ASSETS.get(ticker.get("symbol", ""))
        if asset and ticker.get("relativeFundingRate") is not None:
            try:
                rate = float(ticker["relativeFundingRate"])
            except (ValueError, TypeError):
                continue
            if not math.isfinite(rate):
                continue
            points.append(FundingPoint("kraken", asset, now, rate, 1.0))
    return points


# ---------------------------------------------------------------------------
# Fetchers (network; failures return an error string, never raise)
# ---------------------------------------------------------------------------

def _get_json(url: str, timeout: float = 20.0) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "market-lab-carry-monitor/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_okx(asset: str) -> tuple[list[FundingPoint], str | None]:
    inst = f"{asset}-USDT-SWAP"
    try:
        payload = _get_json(f"https://www.okx.com/api/v5/public/funding-rate-history?instId={inst}&limit=100")
        if payload.get("code") != "0":
            return [], f"okx {asset}: API code {payload.get('code')}"
        return parse_okx_history(payload, asset), None
    except Exception as exc:  # network/geo/parse — one venue must not kill the run
        return [], f"okx {asset}: {exc}"


def fetch_deribit(asset: str, days: int = 30) -> tuple[list[FundingPoint], str | None]:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    url = (
        "https://www.deribit.com/api/v2/public/get_funding_rate_history"
        f"?instrument_name={asset}-PERPETUAL"
        f"&start_timestamp={int(start.timestamp() * 1000)}"
        f"&end_timestamp={int(end.timestamp() * 1000)}"
    )
    try:
        payload = _get_json(url)
        if "result" not in payload:
            return [], f"deribit {asset}: no result field"
        return parse_deribit_history(payload, asset), None
    except Exception as exc:
        return [], f"deribit {asset}: {exc}"


def fetch_kraken() -> tuple[list[FundingPoint], str | None]:
    try:
        payload = _get_json("https://futures.kraken.com/derivatives/api/v3/tickers")
        if payload.get("result") != "success":
            return [], "kraken: API did not return success"
        return parse_kraken_tickers(payload, datetime.now(timezone.utc)), None
    except Exception as exc:
        return [], f"kraken: {exc}"


# ---------------------------------------------------------------------------
# Store + summary
# ---------------------------------------------------------------------------

def load_points(path: Path) -> list[FundingPoint]:
    if not path.exists():
        return []
    points = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            rate = float(row["rate"])
            period_hours = float(row["period_hours"])
            # A NaN rate (json.loads accepts the literal) or a non-positive
            # period would poison every summary mean / divide by zero.
            if not math.isfinite(rate) or not math.isfinite(period_hours) or period_hours <= 0:
                continue
            points.append(FundingPoint(
                row["venue"], row["asset"],
                datetime.fromisoformat(row["ts"]),
                rate, period_hours,
            ))
        except (ValueError, KeyError, TypeError):
            continue  # operational store: a torn line loses one point, not the run
    return points


def append_points(path: Path, new_points: list[FundingPoint]) -> int:
    """Append only points whose (venue, asset, ts) is not already stored.
    Returns how many were written."""
    # Dedupe against the store AND within this batch (a payload can repeat
    # a timestamp), and never let a non-finite rate reach the writer:
    # json.dumps(allow_nan=False) raising mid-batch would kill the whole
    # monitor run after a partial write.
    seen = {p.key() for p in load_points(path)}
    fresh = []
    for point in new_points:
        if not math.isfinite(point.rate) or point.key() in seen:
            continue
        seen.add(point.key())
        fresh.append(point)
    if not fresh:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for point in sorted(fresh, key=lambda p: (p.venue, p.asset, p.ts)):
            f.write(json.dumps({
                "venue": point.venue, "asset": point.asset, "ts": point.ts.isoformat(),
                "rate": point.rate, "period_hours": point.period_hours,
            }, sort_keys=True, allow_nan=False) + "\n")
    return len(fresh)


def summarize(points: list[FundingPoint], window_days: int = 30,
              trigger: float = TRIGGER_ANNUALIZED) -> list[CarrySummary]:
    groups: dict[tuple[str, str], list[FundingPoint]] = {}
    for point in points:
        groups.setdefault((point.venue, point.asset), []).append(point)
    # Each group's window anchors on its own latest point (a monitor that
    # runs irregularly still measures the last 30 days it HAS) — but a
    # group whose latest point lags the store's freshest data by more
    # than the window is stale (its venue has been failing while others
    # updated) and must never present as a live trigger.
    freshest = max((p.ts for group in groups.values() for p in group), default=None)
    summaries = []
    for (venue, asset), group in sorted(groups.items()):
        group.sort(key=lambda p: p.ts)
        latest_ts = group[-1].ts
        window = [p for p in group if (latest_ts - p.ts) <= timedelta(days=window_days)]
        if not window:
            continue
        annualized = [p.annualized() for p in window]
        days_covered = len({p.ts.date() for p in window})
        mean_ann = sum(annualized) / len(annualized)
        stale = bool(freshest is not None and (freshest - latest_ts) > timedelta(days=window_days))
        summaries.append(CarrySummary(
            venue=venue,
            asset=asset,
            points=len(window),
            days_covered=days_covered,
            mean_annualized=mean_ann,
            latest_annualized=annualized[-1],
            share_above_trigger=sum(1 for a in annualized if a > trigger) / len(annualized),
            triggered=bool(mean_ann > trigger and days_covered >= MIN_TRIGGER_DAYS and not stale),
            stale=stale,
        ))
    return summaries
