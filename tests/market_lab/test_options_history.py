"""Stream 4 data layer tests: DoltHub chain parsing, provenance, NoData.

Fixtures are REAL chains probed from the live dataset on 2026-11-17
(SPY 2020-03-18, the COVID crash, and SPY 2026-11-13), embedded under
tests/market_lab/fixtures/options/.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from market_lab import options_history as oh

FIXTURES = Path(__file__).parent / "fixtures" / "options"


def load_fixture(day: str) -> dict:
    return json.loads((FIXTURES / f"spy_chain_{day}.json").read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(oh, "CACHE_DIR", tmp_path / "history")
    yield


def fixture_fetcher(day: str):
    payload = load_fixture(day)

    def fetch(url: str):
        return payload

    return fetch


def test_parse_types_and_vol_is_iv():
    payload = load_fixture("2020-03-18")
    rows = oh.parse_chain_rows("SPY", date(2020, 3, 18), payload["rows"])
    assert len(rows) == 78
    first = rows[0]
    raw = payload["rows"][0]
    # 'vol' in the source is implied volatility, NOT volume.
    assert first.iv == float(raw["vol"])
    assert first.right == "CALL"  # source 'Call' normalized
    assert isinstance(first.strike, float)
    assert isinstance(first.expiration, date)
    assert first.chain_date == date(2020, 3, 18)


def test_parse_skips_malformed_and_foreign_rows():
    payload = load_fixture("2020-03-18")
    raw = list(payload["rows"])
    raw.append({"date": "2020-03-18", "act_symbol": "SPY",
                "expiration": "2020-04-15", "strike": "200.00",
                "call_put": "Put", "bid": "5.00", "ask": "4.00",  # ask < bid
                "vol": "0.5", "delta": "-0.2", "gamma": "0.01",
                "rho": "0.01", "theta": "-0.1", "vega": "0.1"})
    raw.append({"date": "2020-03-19", "act_symbol": "SPY",  # wrong date
                "expiration": "2020-04-15", "strike": "200.00",
                "call_put": "Put", "bid": "4.00", "ask": "5.00",
                "vol": "0.5", "delta": "-0.2", "gamma": "0.01",
                "rho": "0.01", "theta": "-0.1", "vega": "0.1"})
    raw.append({"date": "2020-03-18", "act_symbol": "SPY",  # garbage numerics
                "expiration": "2020-04-15", "strike": "abc",
                "call_put": "Put", "bid": "4.00", "ask": "5.00",
                "vol": "0.5", "delta": "-0.2", "gamma": "0.01",
                "rho": "0.01", "theta": "-0.1", "vega": "0.1"})
    rows = oh.parse_chain_rows("SPY", date(2020, 3, 18), raw)
    assert len(rows) == 78


def test_fetch_cache_roundtrip():
    result = oh.fetch_chain("SPY", date(2020, 3, 18),
                            fetcher=fixture_fetcher("2020-03-18"))
    assert isinstance(result, oh.ChainDay)
    assert len(result.rows) == 78
    assert result.expirations == (date(2020, 4, 1), date(2020, 4, 15), date(2020, 5, 15))
    loaded = oh.load_chain("SPY", date(2020, 3, 18))
    assert isinstance(loaded, oh.ChainDay)
    assert loaded.rows == result.rows


def test_no_data_is_explicit_and_provenanced():
    def empty_fetcher(url: str):
        return {"query_execution_status": "Success", "rows": []}

    result = oh.fetch_chain("SPY", date(2019, 2, 11), fetcher=empty_fetcher)
    assert isinstance(result, oh.NoChainData)
    # Absence was cached with provenance: loading returns NoData, not an error.
    assert isinstance(oh.load_chain("SPY", date(2019, 2, 11)), oh.NoChainData)


def test_unfetched_date_raises():
    with pytest.raises(FileNotFoundError):
        oh.load_chain("SPY", date(2021, 6, 1))


def test_missing_sidecar_refused(tmp_path):
    oh.fetch_chain("SPY", date(2026, 11, 13), fetcher=fixture_fetcher("2026-11-13"))
    sidecar = oh.chain_cache_path("SPY", date(2026, 11, 13)).with_suffix(".meta.json")
    sidecar.unlink()
    with pytest.raises(ValueError, match="no provenance sidecar"):
        oh.load_chain("SPY", date(2026, 11, 13))


def test_tampered_cache_refused():
    oh.fetch_chain("SPY", date(2026, 11, 13), fetcher=fixture_fetcher("2026-11-13"))
    path = oh.chain_cache_path("SPY", date(2026, 11, 13))
    content = json.loads(path.read_text(encoding="utf-8"))
    content[0]["bid"] = content[0]["bid"] + 1.0  # tamper after the fact
    path.write_text(json.dumps(content), encoding="utf-8")
    with pytest.raises(ValueError, match="failed validation"):
        oh.load_chain("SPY", date(2026, 11, 13))


def test_server_error_raises_never_becomes_nodata():
    def err_fetcher(url: str):
        return {"query_execution_status": "Error",
                "query_execution_message": "context deadline exceeded"}

    with pytest.raises(RuntimeError, match="query failed"):
        oh.fetch_chain("SPY", date(2020, 3, 18), fetcher=err_fetcher)
