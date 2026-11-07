from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

from .config import PRICE_DIR, SYNTHETIC_PRICE_DIR, ensure_dirs

@dataclass(frozen=True)
class Bar:
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int

    def as_row(self) -> dict[str, str]:
        return {
            "date": self.date.isoformat(),
            "open": f"{self.open:.4f}",
            "high": f"{self.high:.4f}",
            "low": f"{self.low:.4f}",
            "close": f"{self.close:.4f}",
            "volume": str(int(self.volume)),
        }

def _finite_price(value: float) -> bool:
    try:
        return math.isfinite(float(value)) and float(value) > 0
    except (TypeError, ValueError):
        return False


def _valid_bar(bar: Bar) -> bool:
    return all(_finite_price(v) for v in (bar.open, bar.high, bar.low, bar.close)) and bar.high >= bar.low and bar.volume >= 0


def price_path(symbol: str) -> Path:
    safe = symbol.upper().replace("/", "_").replace("-", "_")
    return PRICE_DIR / f"{safe}.csv"

def synthetic_price_path(symbol: str) -> Path:
    safe = symbol.upper().replace("/", "_").replace("-", "_")
    return SYNTHETIC_PRICE_DIR / f"{safe}.csv"

def _load_prices_from_path(path: Path) -> list[Bar]:
    if not path.exists():
        return []
    bars: list[Bar] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                bar = Bar(
                    date=datetime.strptime(row["date"], "%Y-%m-%d").date(),
                    open=float(row["open"]), high=float(row["high"]), low=float(row["low"]),
                    close=float(row["close"]), volume=int(float(row["volume"])),
                )
            except (KeyError, TypeError, ValueError):
                continue
            if _valid_bar(bar):
                bars.append(bar)
    return bars

def load_cached_prices(symbol: str) -> list[Bar]:
    return _load_prices_from_path(price_path(symbol))

def load_cached_synthetic_prices(symbol: str) -> list[Bar]:
    return _load_prices_from_path(synthetic_price_path(symbol))

def save_prices(symbol: str, bars: Iterable[Bar]) -> Path:
    return _save_prices_to_path(price_path(symbol), bars)

def save_synthetic_prices(symbol: str, bars: Iterable[Bar]) -> Path:
    return _save_prices_to_path(synthetic_price_path(symbol), bars)

def _save_prices_to_path(path: Path, bars: Iterable[Bar]) -> Path:
    ensure_dirs()
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted({b.date: b for b in bars}.values(), key=lambda b: b.date)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "open", "high", "low", "close", "volume"])
        writer.writeheader()
        for bar in rows:
            writer.writerow(bar.as_row())
    return path

def _synthetic_prices(symbol: str, days: int = 260, end: date | None = None) -> list[Bar]:
    # Deterministic fallback so tests/reports work without network or yfinance.
    end = end or date.today()
    rng = random.Random(sum(ord(c) for c in symbol.upper()))
    base = 50 + (sum(ord(c) for c in symbol.upper()) % 250)
    drift = rng.uniform(-0.0002, 0.0012)
    vol = rng.uniform(0.012, 0.035)
    price = float(base)
    bars: list[Bar] = []
    d = end - timedelta(days=days * 2)
    while len(bars) < days:
        if d.weekday() < 5:
            shock = rng.gauss(drift, vol)
            prev = price
            price = max(2.0, price * math.exp(shock))
            high = max(prev, price) * (1 + abs(rng.gauss(0, vol / 3)))
            low = min(prev, price) * (1 - abs(rng.gauss(0, vol / 3)))
            bars.append(Bar(d, prev, high, max(0.01, low), price, rng.randint(500_000, 80_000_000)))
        d += timedelta(days=1)
    return bars

def fetch_prices(symbol: str, days: int = 260, prefer_network: bool = True, max_cache_age_days: int = 3, require_live: bool = False) -> tuple[list[Bar], str]:
    """Fetch daily bars via yfinance -> cache -> synthetic fallback.

    WARNING: the final fallback is DETERMINISTIC SYNTHETIC data. Callers
    whose numbers will be believed (strategy streams, backtests) must pass
    ``require_live=True`` — synthetic sources then raise instead of
    returning — or use ``market_lab.daily_data`` (Alpaca + provenance).
    """
    ensure_dirs()
    if prefer_network:
        try:
            import yfinance as yf  # type: ignore
            period = f"{max(days + 30, 60)}d"
            frame = yf.download(symbol, period=period, interval="1d", progress=False, auto_adjust=True)
            if frame is not None and not frame.empty:
                bars=[]
                def scalar(value):
                    # yfinance can return scalar values or one-element Series depending on pandas/yf versions.
                    if hasattr(value, "iloc"):
                        return value.iloc[0]
                    return value
                for idx, row in frame.tail(days).iterrows():
                    bar = Bar(
                        date=idx.date(),
                        open=float(scalar(row["Open"])), high=float(scalar(row["High"])), low=float(scalar(row["Low"])),
                        close=float(scalar(row["Close"])), volume=int(scalar(row.get("Volume", 0)) or 0),
                    )
                    if _valid_bar(bar):
                        bars.append(bar)
                if len(bars) >= min(days, 30):
                    save_prices(symbol, bars)
                    return bars[-days:], "yfinance"
        except Exception:
            pass
    cached = load_cached_prices(symbol)
    if len(cached) >= min(days, 30):
        newest = cached[-1].date
        if newest >= date.today() - timedelta(days=max_cache_age_days):
            return cached[-days:], "cache"
    if require_live:
        raise RuntimeError(
            f"no live or real cached data for {symbol}; refusing synthetic fallback (require_live=True)"
        )
    synthetic_cached = load_cached_synthetic_prices(symbol)
    if len(synthetic_cached) >= min(days, 30):
        newest = synthetic_cached[-1].date
        if newest >= date.today() - timedelta(days=max_cache_age_days):
            return synthetic_cached[-days:], "cache_synthetic"
    bars = _synthetic_prices(symbol, days=days)
    save_synthetic_prices(symbol, bars)
    return bars, "synthetic"

def latest_close(bars: list[Bar]) -> float:
    if not bars:
        raise ValueError("No bars available")
    return bars[-1].close


def compute_spy_benchmark(starting_cash: float, start_date: date | None = None, days: int = 260, prefer_network: bool = True) -> dict[str, float | str | None]:
    """Compute what starting_cash invested in SPY buy/hold would be worth today.

    Returns a dict with:
      - benchmark_equity: current dollar value of the buy/hold position
      - benchmark_return: total return fraction (e.g. -0.025 for -2.5%)
      - start_price: SPY close on the benchmark start date
      - current_price: latest SPY close
      - start_date_str: ISO date string of the actual start bar used
      - data_source: where the prices came from
    """
    bars, source = fetch_prices("SPY", days=days, prefer_network=prefer_network)
    if not bars:
        return {
            "benchmark_equity": starting_cash,
            "benchmark_return": 0.0,
            "start_price": None,
            "current_price": None,
            "start_date_str": None,
            "data_source": source,
        }
    if start_date is not None:
        start_bar = next((b for b in bars if b.date >= start_date), bars[-1])
    else:
        start_bar = bars[0]
    start_price = start_bar.close
    current_price = bars[-1].close
    if start_price and start_price > 0:
        benchmark_equity = (current_price / start_price) * starting_cash
        benchmark_return = current_price / start_price - 1
    else:
        benchmark_equity = starting_cash
        benchmark_return = 0.0
    return {
        "benchmark_equity": benchmark_equity,
        "benchmark_return": benchmark_return,
        "start_price": start_price,
        "current_price": current_price,
        "start_date_str": start_bar.date.isoformat(),
        "data_source": source,
    }
