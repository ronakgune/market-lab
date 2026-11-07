"""Regression tests for the run-1 baseline security audit fixes.

Each test reproduces the audited attack first (as a plain call against the
fixed code, asserting rejection) — before the fix these inputs were
accepted and poisoned state. All offline, all local files in tmp dirs.

Findings covered (~/security-audit-skill/market-lab/run-1/):
  1. NaN bypasses equity broker risk gates
  2. NaN bypasses options paper risk gates
  3. Evidence segment<->snapshot binding + snapshot digest validation
  4. Fetcher header budget + aggregate deadline
  5. thesis_portfolio negative/NaN conviction sizing (follow-up c)
"""
from __future__ import annotations

import io
import json
import math
import tempfile
import time
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from market_lab.broker import (
    Portfolio, Position, evaluate_order, load_order_candidates,
    load_portfolio, place_mock_order, save_portfolio,
)
from market_lab.options_data import OptionContract, OptionGreeks, OptionQuote, _contract_from_record
from market_lab.options_paper import (
    OptionPaperOrder, OptionPaperPortfolio, evaluate_option_paper_order,
    load_option_paper_candidates, load_option_paper_portfolio,
    save_option_paper_portfolio,
)
from market_lab.thesis_portfolio import PortfolioContext, compute_deterministic_sizing
from market_lab.web_evidence import sha256_hex
from market_lab.web_evidence_providers import _http_fetch, _read_http_response
from market_lab.web_evidence_runner import verify_run
from market_lab.web_evidence_store import _snapshot_dir, read_extracted_text

NAN = float("nan")
INF = float("inf")


# ---------------------------------------------------------------- fix 1: equity broker

def test_nan_quantity_sell_rejected_with_zero_holdings():
    p = Portfolio(cash=100_000)
    d = evaluate_order(p, "SELL", "SPY", NAN, 100.0, {"SPY": 100.0})
    assert not d.accepted and "finite" in d.reason
    assert p.cash == 100_000 and not p.positions


def test_nan_and_inf_price_and_quantity_rejected():
    p = Portfolio(cash=100_000)
    for qty, price in [(10, NAN), (10, INF), (INF, 100.0), (NAN, NAN)]:
        d = evaluate_order(p, "BUY", "SPY", qty, price, {"SPY": 100.0})
        assert not d.accepted and "finite" in d.reason
    assert p.cash == 100_000 and not p.positions


def test_nan_reference_price_rejected():
    p = Portfolio(cash=100_000)
    d = evaluate_order(p, "BUY", "SPY", 10, 100.0, {"SPY": NAN})
    assert not d.accepted and "finite" in d.reason


def test_normal_order_still_accepted():
    p = Portfolio(cash=100_000)
    d = evaluate_order(p, "BUY", "SPY", 10, 100.0, {"SPY": 100.0})
    assert d.accepted and p.positions["SPY"].quantity == 10


def test_load_order_candidates_rejects_nan_token_file():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "candidates.jsonl"
        row = {"side": "SELL", "symbol": "SPY", "quantity": "NaN", "strategy": "s",
               "confidence": 0.5, "reason": "r", "signal_date": "2026-01-01",
               "reference_close": 100.0}
        # write the bare NaN token the way the audited file carried it
        line = json.dumps(row).replace('"NaN"', "NaN")
        path.write_text(line + "\n")
        assert load_order_candidates(path) == []


def test_save_portfolio_refuses_nan_state():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "state.json"
        p = Portfolio(cash=NAN)
        with pytest.raises(ValueError):
            save_portfolio(p, path)
        assert not path.exists() or "NaN" not in path.read_text()


def test_load_portfolio_nan_file_falls_back_to_fresh():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "state.json"
        path.write_text('{"cash": NaN, "positions": {}}')
        p = load_portfolio(path)
        assert math.isfinite(p.cash)


def test_place_mock_order_nan_does_not_persist():
    with tempfile.TemporaryDirectory() as td:
        state, ledger = Path(td) / "s.json", Path(td) / "l.jsonl"
        d = place_mock_order("BUY", "SPY", NAN, 100.0, {"SPY": 100.0}, state, ledger)
        assert not d.accepted
        assert not state.exists()  # rejected orders never write state


# ---------------------------------------------------------------- fix 2: options paper

_EXPIRY = (date.today() + timedelta(days=35)).isoformat()


def _contract(**over) -> OptionContract:
    fields = dict(
        underlying="SPY", expiration=_EXPIRY, strike=95.0, option_type="PUT",
        quote=OptionQuote(1.4, 1.6, 1.5, 180, 1400),
        greeks=OptionGreeks(-0.30, 0.03, -0.025, 0.10, 0.31),
    )
    fields.update(over)
    return OptionContract(**fields)


def test_options_nan_price_rejected():
    order = OptionPaperOrder("SELL_TO_OPEN", _contract(), 1, NAN, "csp")
    d = evaluate_option_paper_order(OptionPaperPortfolio(cash=100_000), Portfolio(cash=100_000), order)
    assert not d.accepted and "finite" in d.reason


def test_options_nan_strike_delta_mid_rejected():
    for contract in (
        _contract(strike=NAN),
        _contract(greeks=OptionGreeks(NAN, 0.03, -0.025, 0.10, 0.31)),
        _contract(quote=OptionQuote(1.4, 1.6, NAN, 180, 1400)),
        _contract(quote=OptionQuote(INF, 1.6, 1.5, 180, 1400)),
    ):
        order = OptionPaperOrder("SELL_TO_OPEN", contract, 1, 1.5, "csp")
        d = evaluate_option_paper_order(OptionPaperPortfolio(cash=100_000), Portfolio(cash=100_000), order)
        assert not d.accepted and "finite" in d.reason


def test_options_normal_order_still_accepted():
    order = OptionPaperOrder("SELL_TO_OPEN", _contract(), 1, 1.5, "csp")
    paper = OptionPaperPortfolio(cash=100_000)
    d = evaluate_option_paper_order(paper, Portfolio(cash=100_000), order)
    assert d.accepted and paper.cash > 100_000  # premium credited


def _record(**contract_over) -> dict:
    rec = {
        "underlying": "SPY", "expiration": _EXPIRY, "strike": 95.0,
        "option_type": "PUT",
        "quote": {"bid": 1.4, "ask": 1.6, "mid": 1.5, "volume": 180, "open_interest": 1400},
        "greeks": {"delta": -0.30, "gamma": 0.03, "theta": -0.025, "vega": 0.10,
                   "implied_volatility": 0.31, "degenerate": False},
    }
    rec.update(contract_over)
    return rec


def test_contract_from_record_rejects_nonfinite():
    with pytest.raises(ValueError):
        _contract_from_record(_record(strike=NAN), "2026-01-01")
    bad_quote = _record()
    bad_quote["quote"] = {**bad_quote["quote"], "mid": NAN}
    with pytest.raises(ValueError):
        _contract_from_record(bad_quote, "2026-01-01")
    assert _contract_from_record(_record(), "2026-01-01").strike == 95.0


def test_options_candidates_file_with_nan_is_skipped():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "cands.jsonl"
        cand = {"action": "SELL_TO_OPEN", "contract": _record(strike=NAN),
                "contracts": 1, "strategy": "csp", "reference_price": 1.5,
                "signal_as_of": date.today().isoformat()}
        path.write_text(json.dumps(cand).replace("NaN", "NaN") + "\n")
        assert load_option_paper_candidates(path) == []


def test_save_option_paper_portfolio_refuses_nan():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "opt.json"
        with pytest.raises(ValueError):
            save_option_paper_portfolio(OptionPaperPortfolio(cash=NAN), path)


def test_load_option_paper_portfolio_nan_file_falls_back():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "opt.json"
        path.write_text('{"cash": NaN}')
        p = load_option_paper_portfolio(path)
        assert math.isfinite(p.cash)


# ---------------------------------------------------------------- fix 3: evidence binding

def test_snapshot_dir_rejects_traversal_and_bad_digests(tmp_path):
    for bad in ("sha256:../..", "sha256:" + "../" * 10 + "aa",
                "sha256:abc", "sha256:" + "G" * 64, "sha256:" + "a" * 63,
                "sha256:" + "a" * 65, "sha256:" + "A" * 64):
        with pytest.raises(ValueError):
            _snapshot_dir(tmp_path, bad)
    good = "sha256:" + "ab" * 32
    assert _snapshot_dir(tmp_path, good).name == "ab" * 32
    with pytest.raises(ValueError):
        read_extracted_text(tmp_path, "sha256:../../etc/passwd")


def _write_snapshot(run_dir: Path, digest: str, extracted: str, raw: bytes) -> Path:
    from market_lab.web_evidence_store import ensure_layout
    base = ensure_layout(run_dir)
    sdir = base / "snapshots" / "sha256" / digest[:2] / digest
    sdir.mkdir(parents=True, exist_ok=True)
    (sdir / "raw.bin").write_bytes(raw)
    (sdir / "extracted.txt").write_text(extracted, encoding="utf-8")
    (sdir / "manifest.json").write_text(json.dumps({"raw_sha256": sha256_hex(raw)}))
    return sdir


def _run_bundle(tmp_path: Path, segment_snapshot: str) -> dict:
    digest_a, digest_b = "a" * 64, "b" * 64
    sid_a, sid_b = f"sha256:{digest_a}", f"sha256:{digest_b}"
    _write_snapshot(tmp_path, digest_a, "alpha excerpt line\n", b"raw-a")
    _write_snapshot(tmp_path, digest_b, "beta excerpt line\n", b"raw-b")
    (tmp_path / "web_evidence").mkdir(exist_ok=True)
    excerpt = "beta excerpt line" if segment_snapshot == sid_b else "alpha excerpt line"
    seg = {"segment_id": "seg-1", "snapshot_id": segment_snapshot,
           "locator_type": "text_line_range", "locator": "1-1",
           "verbatim_excerpt_or_value": excerpt,
           "segment_sha256": sha256_hex(excerpt)}
    (tmp_path / "web_evidence" / "segments.jsonl").write_text(json.dumps(seg) + "\n")
    idx = {"snapshot_id": sid_a, "manifest_path": f"web_evidence/snapshots/sha256/aa/{digest_a}/manifest.json"}
    (tmp_path / "web_evidence" / "snapshot_index.jsonl").write_text(json.dumps(idx) + "\n")
    ev = {"snapshot_id": sid_a, "segment_id": "seg-1"}
    (tmp_path / "evidence.jsonl").write_text(json.dumps(ev) + "\n")
    return verify_run(tmp_path, require_snapshots=True)


def test_verify_run_rejects_cross_snapshot_segment(tmp_path):
    checks = _run_bundle(tmp_path, "sha256:" + "b" * 64)
    assert checks["snapshots_complete"] is False


def test_verify_run_accepts_bound_segment(tmp_path):
    checks = _run_bundle(tmp_path, "sha256:" + "a" * 64)
    assert checks["snapshots_complete"] is True


# ---------------------------------------------------------------- fix 4: fetcher bounds

class _FakeSock:
    def __init__(self, payload: bytes):
        self._buf = io.BytesIO(payload)

    def makefile(self, mode):
        return self._buf


def test_read_http_response_header_count_cap():
    lines = b"HTTP/1.1 200 OK\r\n" + b"".join(b"X-H%d: v\r\n" % i for i in range(150)) + b"\r\nbody"
    with pytest.raises(RuntimeError, match="header limit"):
        _read_http_response(_FakeSock(lines), max_bytes=10_000)


def test_read_http_response_header_byte_cap():
    lines = b"HTTP/1.1 200 OK\r\nX-Big: " + b"v" * 70_000 + b"\r\n\r\nbody"
    with pytest.raises(RuntimeError, match="header limit"):
        _read_http_response(_FakeSock(lines), max_bytes=10_000_000)


def test_read_http_response_normal():
    status, body, headers = _read_http_response(
        _FakeSock(b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\nhello"), max_bytes=1000)
    assert status == 200 and body == b"hello" and headers["content-type"] == "text/plain"


def test_http_fetch_aggregate_deadline_enforced():
    calls = {"n": 0}

    def slow_fetch(url, *, timeout_seconds, max_bytes, deadline=None):
        calls["n"] += 1
        time.sleep(0.05)
        return 301, b"", {"location": "https://example.com/next"}

    with patch("market_lab.web_evidence_providers.classify_url_safety",
               lambda u: (True, u, "")), \
         patch("market_lab.web_evidence_providers._single_pinned_fetch", slow_fetch):
        with pytest.raises(RuntimeError, match="deadline|too_many_redirects"):
            _http_fetch("https://example.com/start", timeout_seconds=0, max_bytes=1000, max_redirects=50)
    assert calls["n"] == 0  # deadline=now: refused before the first hop


# ---------------------------------------------------------------- fix 5: thesis sizing

_CTX = PortfolioContext(total_cash=100_000.0, total_paper_exposure=0.0)


def test_thesis_negative_conviction_rejected():
    d = compute_deterministic_sizing(_CTX, 100.0, conviction=-0.5)
    assert d.rejected and d.approved_notional == 0


def test_thesis_nan_and_over_one_conviction_rejected():
    for conv in (NAN, INF, 1.5):
        d = compute_deterministic_sizing(_CTX, 100.0, conviction=conv)
        assert d.rejected and d.approved_notional == 0


def test_thesis_zero_conviction_rejected_as_nonpositive():
    d = compute_deterministic_sizing(_CTX, 100.0, conviction=0.0)
    assert d.rejected


def test_thesis_normal_conviction_unchanged():
    d = compute_deterministic_sizing(_CTX, 100.0, conviction=0.5)
    assert not d.rejected and d.approved_notional > 0


def test_thesis_nan_price_rejected():
    d = compute_deterministic_sizing(_CTX, NAN, conviction=0.5)
    assert d.rejected
