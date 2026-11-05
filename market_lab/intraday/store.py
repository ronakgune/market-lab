"""Parquet store for minute bars, partitioned by symbol and trading day.

Layout under the store root (default INTRADAY_BARS_DIR from config, which
honours MARKET_LAB_DATA_DIR):

    <root>/symbol=<SYM>/date=<YYYY-MM-DD>/bars.parquet
    <root>/symbol=<SYM>/date=<YYYY-MM-DD>/_provenance.json

The partition date is the bar's America/New_York (trading-day) date, so
premarket and after-hours bars land with the session they belong to. Each
partition records its provenance (the set of `source` strings it holds) in
the sidecar JSON; the source is also a column on every row.

Writes de-duplicate on (symbol, timestamp) keeping the FIRST write: rows
already stored win over rows in a later batch, so re-fetching a window never
rewrites history. Queries run through DuckDB over the Arrow tables.

Synthetic bars must never be written here with the default root; callers
pass INTRADAY_SYNTHETIC_BARS_DIR as root instead (see the smoke script),
mirroring the daily layer's PRICE_DIR / SYNTHETIC_PRICE_DIR quarantine.
"""
from __future__ import annotations

import json
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable, Optional
from zoneinfo import ZoneInfo

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from ..config import INTRADAY_BARS_DIR
from .bars import MinuteBar, valid_minute_bar

_ET = ZoneInfo("America/New_York")

_SCHEMA = pa.schema([
    ("symbol", pa.string()),
    ("timestamp", pa.timestamp("us", tz="UTC")),
    ("open", pa.float64()),
    ("high", pa.float64()),
    ("low", pa.float64()),
    ("close", pa.float64()),
    ("volume", pa.int64()),
    ("source", pa.string()),
])


def _root(root: Optional[Path]) -> Path:
    return Path(root) if root is not None else INTRADAY_BARS_DIR


def _trading_day(ts: datetime) -> date:
    return ts.astimezone(_ET).date()


def _partition_dir(root: Path, symbol: str, day: date) -> Path:
    safe = symbol.upper().replace("/", "_").replace("-", "_")
    return root / f"symbol={safe}" / f"date={day.isoformat()}"


def _bars_to_table(bars: list[MinuteBar]) -> pa.Table:
    return pa.table({
        "symbol": [b.symbol for b in bars],
        "timestamp": [b.timestamp for b in bars],
        "open": [b.open for b in bars],
        "high": [b.high for b in bars],
        "low": [b.low for b in bars],
        "close": [b.close for b in bars],
        "volume": [b.volume for b in bars],
        "source": [b.source for b in bars],
    }, schema=_SCHEMA)


def _table_to_bars(table: pa.Table) -> list[MinuteBar]:
    rows = table.to_pylist()
    out = []
    for r in rows:
        ts = r["timestamp"]
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        out.append(MinuteBar(
            symbol=r["symbol"], timestamp=ts.astimezone(timezone.utc),
            open=r["open"], high=r["high"], low=r["low"], close=r["close"],
            volume=int(r["volume"]), source=r["source"],
        ))
    return out


def _write_partition(root: Path, symbol: str, day: date, bars: list[MinuteBar]) -> int:
    """Merge `bars` into one partition; returns the number of NEW rows kept."""
    pdir = _partition_dir(root, symbol, day)
    pdir.mkdir(parents=True, exist_ok=True)
    target = pdir / "bars.parquet"
    existing: list[MinuteBar] = []
    if target.exists():
        existing = _table_to_bars(pq.read_table(target))
    seen = {b.timestamp for b in existing}
    fresh = [b for b in bars if b.timestamp not in seen]
    merged = sorted(existing + fresh, key=lambda b: b.timestamp)
    table = _bars_to_table(merged)
    tmp = pdir / "bars.parquet.tmp"
    pq.write_table(table, tmp)
    os.replace(tmp, target)
    provenance = {
        "symbol": symbol.upper(),
        "date": day.isoformat(),
        "sources": sorted({b.source for b in merged}),
        "rows": len(merged),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    (pdir / "_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return len(fresh)


def write_bars(bars: Iterable[MinuteBar], root: Optional[Path] = None) -> int:
    """Store bars, partitioned by (symbol, ET trading day).

    De-duplicates on (symbol, timestamp) keeping the first write. Returns
    the number of newly stored rows. Raises ValueError on any invalid bar:
    nothing is written silently.
    """
    batch = list(bars)
    invalid = [b for b in batch if not valid_minute_bar(b)]
    if invalid:
        raise ValueError(f"refusing to store {len(invalid)} invalid minute bar(s)")
    store_root = _root(root)
    groups: dict[tuple[str, date], list[MinuteBar]] = {}
    for b in batch:
        groups.setdefault((b.symbol.upper(), _trading_day(b.timestamp)), []).append(b)
    return sum(_write_partition(store_root, sym, day, rows) for (sym, day), rows in groups.items())


def _symbol_table(root: Path, symbol: str) -> Optional[pa.Table]:
    safe = symbol.upper().replace("/", "_").replace("-", "_")
    sym_dir = root / f"symbol={safe}"
    if not sym_dir.exists():
        return None
    files = sorted(sym_dir.glob("date=*/bars.parquet"))
    if not files:
        return None
    # Read partitions individually: a multi-file dataset read dictionary-
    # encodes strings inconsistently across files and fails to merge.
    tables = [pq.read_table(f) for f in files]
    return tables[0] if len(tables) == 1 else pa.concat_tables(tables)


def _query(root: Path, symbol: str, where: str, params: list) -> list[MinuteBar]:
    table = _symbol_table(_root(root), symbol)
    if table is None:
        return []
    con = duckdb.connect(":memory:")
    try:
        con.register("bars", table)
        # duckdb 2.0: execute() returns the connection; results via to_arrow_table().
        result = con.execute(
            f"SELECT symbol, timestamp, open, high, low, close, volume, source "
            f"FROM bars WHERE {where} ORDER BY timestamp",
            params,
        ).to_arrow_table()
    finally:
        con.close()
    return _table_to_bars(result)


def bars_for_window(symbol: str, start_utc: datetime, end_utc: datetime,
                    root: Optional[Path] = None) -> list[MinuteBar]:
    """Bars for `symbol` with start_utc <= timestamp < end_utc, time-ordered."""
    for name, ts in (("start_utc", start_utc), ("end_utc", end_utc)):
        if ts.tzinfo is None or ts.utcoffset() is None:
            raise ValueError(f"{name} must be tz-aware")
    return _query(_root(root), symbol, "timestamp >= ? AND timestamp < ?",
                  [start_utc.astimezone(timezone.utc), end_utc.astimezone(timezone.utc)])


def bars_for_day(symbol: str, day: date, root: Optional[Path] = None) -> list[MinuteBar]:
    """All stored bars for `symbol` in the partition for ET trading day `day`."""
    store_root = _root(root)
    target = _partition_dir(store_root, symbol, day) / "bars.parquet"
    if not target.exists():
        return []
    con = duckdb.connect(":memory:")
    try:
        con.register("bars", pq.read_table(target))
        result = con.execute(
            "SELECT symbol, timestamp, open, high, low, close, volume, source "
            "FROM bars ORDER BY timestamp"
        ).to_arrow_table()
    finally:
        con.close()
    return _table_to_bars(result)


def partition_provenance(symbol: str, day: date, root: Optional[Path] = None) -> Optional[dict]:
    """The provenance sidecar for one partition, or None if not stored."""
    sidecar = _partition_dir(_root(root), symbol, day) / "_provenance.json"
    if not sidecar.exists():
        return None
    return json.loads(sidecar.read_text())
