"""Adversarial tests for the run-1 security fixes — trying to BREAK them.

Baseline (test_security_fixes_run1.py) covers the audited spellings:
NaN / Infinity *tokens* in JSON and hand-passed NaN order fields. This
suite attacks the seams the baseline fixes left:

  * 1e999 and friends — valid JSON syntax that float() parses to inf
    WITHOUT tripping json's parse_constant hook, in every loader.
  * Non-finite values in portfolio *state* (hand-built or loaded), not
    just in the order — the gates read state through comparisons too.
  * Quote volume / open_interest, which the options finiteness lists
    initially omitted: inf volume defeats the min-liquidity gate.
  * A sub-cent approved notional in thesis sizing (rounds to 0.00 but
    was returned with rejected=False).

Genuine holes found by this suite were fixed in the modules; each fix
has a regression test below marked [fix]. Paths probed and found safely
rejected downstream are asserted as such.
"""
from __future__ import annotations

import io
import json
import math
import tempfile
import time
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import pytest

from market_lab.broker import (
    OrderCandidate, Portfolio, Position, candidate_to_order_at_open,
    evaluate_order, load_order_candidates, load_portfolio,
    place_mock_order, save_order_candidates, save_portfolio,
)
from market_lab.options_data import (
    OptionChainSnapshot, OptionContract, OptionGreeks, OptionQuote,
    _contract_from_record, load_available_option_chains,
    load_option_chain_snapshot, save_option_chain_snapshot,
)
from market_lab.options_paper import (
    OptionPaperCandidate, OptionPaperOrder, OptionPaperPortfolio,
    evaluate_option_paper_order, execute_option_paper_candidate,
    load_option_paper_candidates, load_option_paper_portfolio,
    save_option_paper_portfolio,
)
from market_lab.thesis_portfolio import PortfolioContext, compute_deterministic_sizing
from market_lab.web_evidence import sha256_hex
from market_lab.web_evidence_providers import (
    _MAX_HEADER_BYTES, _MAX_RESPONSE_HEADERS, _http_fetch,
    _read_http_response, _single_pinned_fetch,
)
from market_lab.web_evidence_runner import verify_run
from market_lab.web_evidence_store import _snapshot_dir, read_extracted_text, read_snapshot_manifest

NAN = float("nan")
INF = float("inf")
_EXPIRY = (date.today() + timedelta(days=35)).isoformat()
_TODAY = date.today().isoformat()


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


# ================================================================ broker: 1e999 in files

def test_load_portfolio_rejects_1e999_cash_quantity_avg_price():
    # [fix] parse_constant does NOT catch 1e999: float("1e999") is inf.
    # Before the loader validation these loaded as inf state.
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for body in (
            '{"cash": 1e999, "positions": {}}',
            '{"cash": -1e999, "positions": {}}',
            '{"cash": 1000, "positions": {"SPY": {"symbol": "SPY", "quantity": 1e999, "avg_price": 50}}}',
            '{"cash": 1000, "positions": {"SPY": {"symbol": "SPY", "quantity": 10, "avg_price": 1e999}}}',
            '{"cash": 1000, "positions": {"SPY": {"symbol": "SPY", "quantity": 10, "avg_price": -1e999}}}',
        ):
            p = load_portfolio(_write(td / "s.json", body))
            assert math.isfinite(p.cash)
            assert all(math.isfinite(pos.quantity) and math.isfinite(pos.avg_price)
                       for pos in p.positions.values())


def test_load_portfolio_accepts_finite_scientific_notation():
    # Control: the loader fix must not reject legitimate JSON numbers.
    with tempfile.TemporaryDirectory() as td:
        path = _write(Path(td) / "s.json",
                      '{"cash": 1e5, "positions": {"SPY": {"symbol": "SPY", "quantity": 10, "avg_price": 5e1}}}')
        p = load_portfolio(path)
        assert p.cash == 100_000.0 and p.positions["SPY"].avg_price == 50.0


def test_load_order_candidates_rejects_1e999_fields():
    # [fix] quantity/confidence/reference_close == inf previously loaded.
    row = ('{{"side": "BUY", "symbol": "SPY", "quantity": {q}, "strategy": "s", '
           '"confidence": {c}, "reason": "r", "signal_date": "2026-01-01", '
           '"reference_close": {r}}}\n')
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for fields in (dict(q="1e999", c="0.5", r="100.0"),
                       dict(q="10", c="1e999", r="100.0"),
                       dict(q="10", c="0.5", r="-1e999")):
            path = _write(td / "c.jsonl", row.format(**fields))
            assert load_order_candidates(path) == []
        # One corrupt line poisons the file, matching the NaN-token semantics.
        good = row.format(q="10", c="0.5", r="100.0")
        bad = row.format(q="1e999", c="0.5", r="100.0")
        assert load_order_candidates(_write(td / "c2.jsonl", good + bad)) == []
        # Control: a clean file still loads.
        cands = load_order_candidates(_write(td / "c3.jsonl", good))
        assert len(cands) == 1 and cands[0].quantity == 10


def test_candidate_quantity_1e999_execution_safely_rejected():
    # Even a hand-built inf-quantity candidate (never touching the loader)
    # dies at evaluate_order's finiteness gate and writes no state.
    cand = OrderCandidate("BUY", "SPY", INF, "s", 0.5, "r", "2026-01-01", 100.0)
    with tempfile.TemporaryDirectory() as td:
        state, ledger = Path(td) / "s.json", Path(td) / "l.jsonl"
        save_portfolio(Portfolio(cash=100_000), state)
        d = candidate_to_order_at_open(cand, 100.0, {"SPY": 100.0}, state, ledger)
        assert not d.accepted and "finite" in d.reason
        assert load_portfolio(state).cash == 100_000


# ================================================================ broker: state-side NaN/inf

def test_evaluate_rejects_nan_cash_portfolio():
    # [fix] Before: BUY accepted, cash became NaN (state poisoned).
    p = Portfolio(cash=NAN)
    d = evaluate_order(p, "BUY", "SPY", 10, 100.0, {"SPY": 100.0})
    assert not d.accepted and "finite" in d.reason
    assert math.isnan(p.cash) and not p.positions  # untouched, not mutated-then-accepted


def test_evaluate_rejects_inf_cash_and_inf_position_quantity():
    # [fix] inf state defeated the cash gate / position checks.
    d = evaluate_order(Portfolio(cash=INF), "BUY", "SPY", 10, 100.0, {"SPY": 100.0})
    assert not d.accepted and "finite" in d.reason
    p = Portfolio(cash=100_000, positions={"SPY": Position("SPY", INF, 50.0)})
    d = evaluate_order(p, "SELL", "SPY", 10, 100.0, {"SPY": 100.0})
    assert not d.accepted and "finite" in d.reason


def test_evaluate_rejects_nan_avg_price_same_and_other_symbol():
    # [fix] NaN avg_price poisoned the weighted average on BUY and the
    # equity calc (via prices.get fallback) for unrelated symbols.
    p = Portfolio(cash=100_000, positions={"SPY": Position("SPY", 10, NAN)})
    d = evaluate_order(p, "BUY", "SPY", 10, 100.0, {"SPY": 100.0})
    assert not d.accepted and "finite" in d.reason
    p2 = Portfolio(cash=100_000, positions={"MSFT": Position("MSFT", 10, NAN)})
    d2 = evaluate_order(p2, "BUY", "SPY", 10, 100.0, {"SPY": 100.0})
    assert not d2.accepted and "finite" in d2.reason


def test_candidate_to_order_at_open_nan_quantity_hand_built():
    cand = OrderCandidate("SELL", "SPY", NAN, "s", 0.5, "r", "2026-01-01", 100.0)
    with tempfile.TemporaryDirectory() as td:
        state, ledger = Path(td) / "s.json", Path(td) / "l.jsonl"
        d = candidate_to_order_at_open(cand, 100.0, {"SPY": 100.0}, state, ledger)
        assert not d.accepted and "finite" in d.reason
        assert not state.exists()  # rejected: no state written


def test_negative_infinity_order_fields_rejected():
    p = Portfolio(cash=100_000)
    for qty, price in [(-INF, 100.0), (10, -INF), (INF, INF)]:
        d = evaluate_order(p, "BUY", "SPY", qty, price, {"SPY": 100.0})
        assert not d.accepted and "finite" in d.reason


def test_place_mock_order_on_1e999_state_no_longer_crashes():
    # [fix] Before: the loader returned inf-quantity state, the SELL was
    # accepted, and save_portfolio(allow_nan=False) raised ValueError
    # out of place_mock_order. Now the corrupt file falls back to a
    # fresh portfolio and the SELL is rejected for insufficient shares.
    with tempfile.TemporaryDirectory() as td:
        state, ledger = Path(td) / "s.json", Path(td) / "l.jsonl"
        state.write_text('{"cash": 1000, "positions": {"SPY": {"symbol": "SPY", "quantity": 1e999, "avg_price": 50}}}')
        d = place_mock_order("SELL", "SPY", 10, 100.0, {"SPY": 100.0}, state, ledger)
        assert not d.accepted


def test_broker_valid_buy_sell_round_trip_unchanged():
    with tempfile.TemporaryDirectory() as td:
        state, ledger = Path(td) / "s.json", Path(td) / "l.jsonl"
        d1 = place_mock_order("BUY", "SPY", 10, 100.0, {"SPY": 100.0}, state, ledger)
        assert d1.accepted
        p = load_portfolio(state)
        assert p.positions["SPY"].quantity == 10 and p.cash == pytest.approx(100_000 - 1001.5)
        d2 = place_mock_order("SELL", "SPY", 10, 101.0, {"SPY": 101.0}, state, ledger)
        assert d2.accepted
        p = load_portfolio(state)
        assert "SPY" not in p.positions and p.cash == pytest.approx(100_006.995)
        assert "NaN" not in state.read_text() and "Infinity" not in state.read_text()


def test_order_candidates_valid_round_trip_unchanged():
    cand = OrderCandidate("BUY", "SPY", 10, "s", 0.5, "r", "2026-01-01", 100.0)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "c.jsonl"
        save_order_candidates([cand], path)
        assert load_order_candidates(path) == [cand]


# ================================================================ options: state-side + volume/OI

def _contract(**over) -> OptionContract:
    fields = dict(
        underlying="SPY", expiration=_EXPIRY, strike=95.0, option_type="PUT",
        quote=OptionQuote(1.4, 1.6, 1.5, 180, 1400),
        greeks=OptionGreeks(-0.30, 0.03, -0.025, 0.10, 0.31),
    )
    fields.update(over)
    return OptionContract(**fields)


def _paper(**over) -> OptionPaperPortfolio:
    fields = dict(cash=100_000.0)
    fields.update(over)
    return OptionPaperPortfolio(**fields)


def test_options_evaluate_rejects_nan_paper_cash_and_reserved():
    # [fix] Before: SELL_TO_OPEN accepted with NaN cash; reserve checks
    # against NaN available_cash never bound, state stayed poisoned.
    for paper in (_paper(cash=NAN), _paper(reserved_cash=NAN),
                  _paper(avg_price={"SPY-X": NAN})):
        d = evaluate_option_paper_order(
            paper, Portfolio(cash=100_000),
            OptionPaperOrder("SELL_TO_OPEN", _contract(), 1, 1.5, "csp"))
        assert not d.accepted and "finite" in d.reason


def test_options_evaluate_rejects_nan_equity_state_premium_gate():
    # [fix] Before: account_equity was NaN, so a $50k premium passed the
    # 2%-of-equity gate that rejects it for a valid $100k portfolio.
    big = OptionPaperOrder("BUY_TO_OPEN", _contract(), 1, 500.0, "csp")
    d = evaluate_option_paper_order(_paper(), Portfolio(cash=NAN), big)
    assert not d.accepted and "finite" in d.reason
    p2 = Portfolio(cash=100_000, positions={"SPY": Position("SPY", 10, NAN)})
    d2 = evaluate_option_paper_order(_paper(), p2, big)
    assert not d2.accepted and "finite" in d2.reason
    # Control: valid equity rejects the same order on the premium gate itself.
    d3 = evaluate_option_paper_order(_paper(), Portfolio(cash=100_000), big)
    assert not d3.accepted and "premium" in d3.reason


def test_options_inf_volume_and_open_interest_rejected():
    # [fix] volume/open_interest were missing from both finiteness lists:
    # inf volume + inf OI sailed through the min-liquidity guardrail
    # (inf < minimum is False) while 0/0 was correctly rejected.
    bad = _contract(quote=OptionQuote(1.4, 1.6, 1.5, INF, INF))
    d = evaluate_option_paper_order(
        _paper(), Portfolio(cash=100_000),
        OptionPaperOrder("SELL_TO_OPEN", bad, 1, 1.5, "csp"))
    assert not d.accepted and "finite" in d.reason
    control = _contract(quote=OptionQuote(1.4, 1.6, 1.5, 0, 0))
    d2 = evaluate_option_paper_order(
        _paper(), Portfolio(cash=100_000),
        OptionPaperOrder("SELL_TO_OPEN", control, 1, 1.5, "csp"))
    assert not d2.accepted and "liquidity" in d2.reason


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


def test_contract_from_record_rejects_1e999_strike_and_volume():
    # strike=float("1e999") is inf: float parsing, no JSON token involved.
    with pytest.raises(ValueError):
        _contract_from_record(_record(strike=float("1e999")), "2026-01-01")
    bad_quote = _record()
    bad_quote["quote"] = {**bad_quote["quote"], "volume": float("1e999")}
    with pytest.raises(ValueError):
        _contract_from_record(bad_quote, "2026-01-01")
    bad_oi = _record()
    bad_oi["quote"] = {**bad_oi["quote"], "open_interest": float("1e999")}
    with pytest.raises(ValueError):
        _contract_from_record(bad_oi, "2026-01-01")


def test_load_option_paper_portfolio_rejects_1e999_without_crashing():
    # [fix] positions=1e999 previously raised OverflowError (uncaught —
    # int(inf)); cash/avg_price/reserved 1e999 loaded as inf silently.
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for body in (
            '{"cash": 1e999}',
            '{"cash": 1000, "positions": {"X": 1e999}}',
            '{"cash": 1000, "avg_price": {"X": 1e999}}',
            '{"cash": 1000, "reserved_cash": 1e999}',
            '{"cash": 1000, "reserved_shares": {"SPY": 1e999}}',
        ):
            p = load_option_paper_portfolio(_write(td / "o.json", body))
            assert math.isfinite(p.cash) and math.isfinite(p.reserved_cash)
            assert all(math.isfinite(v) for v in p.avg_price.values())


def test_load_option_paper_portfolio_valid_file_unchanged():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "o.json"
        save_option_paper_portfolio(
            OptionPaperPortfolio(cash=88_000.0, positions={"SPY-X": -1},
                                 avg_price={"SPY-X": 1.5}, reserved_cash=9_500.0), path)
        p = load_option_paper_portfolio(path)
        assert p.cash == 88_000.0 and p.positions == {"SPY-X": -1}
        assert p.avg_price == {"SPY-X": 1.5} and p.reserved_cash == 9_500.0


def test_option_candidates_file_with_1e999_literals_skipped():
    # Raw-text 1e999 (json.dumps would have written the Infinity token).
    tpl = ('{{"action": "SELL_TO_OPEN", "contract": {contract}, "contracts": 1, '
           '"strategy": "csp", "reference_price": {ref}, "signal_as_of": "{today}"}}\n')
    contract = json.dumps(_record())
    contract_inf_strike = json.dumps(_record(strike=95.0)).replace('"strike": 95.0', '"strike": 1e999')
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for text in (tpl.format(contract=contract, ref="1e999", today=_TODAY),
                     tpl.format(contract=contract_inf_strike, ref="1.5", today=_TODAY)):
            assert load_option_paper_candidates(_write(td / "c.jsonl", text)) == []
        good = tpl.format(contract=contract, ref="1.5", today=_TODAY)
        cands = load_option_paper_candidates(_write(td / "c2.jsonl", good))
        assert len(cands) == 1 and cands[0].reference_price == 1.5


def test_execute_candidate_nan_reference_and_mid_safely_rejected():
    # mid=0 makes execute fall back to reference_price; NaN there becomes
    # the order price and dies at the finiteness gate. NaN mid likewise.
    zero_mid = _contract(quote=OptionQuote(1.4, 1.6, 0.0, 180, 1400))
    cand = OptionPaperCandidate("SELL_TO_OPEN", zero_mid, 1, "csp", NAN, _TODAY)
    d = execute_option_paper_candidate(cand, _paper(), Portfolio(cash=100_000))
    assert not d.accepted and "finite" in d.reason
    cand2 = OptionPaperCandidate("SELL_TO_OPEN", _contract(
        quote=OptionQuote(1.4, 1.6, NAN, 180, 1400)), 1, "csp", 1.5, _TODAY)
    d2 = execute_option_paper_candidate(cand2, _paper(), Portfolio(cash=100_000))
    assert not d2.accepted and "finite" in d2.reason


def test_options_valid_open_close_round_trip_unchanged():
    paper, equity = _paper(), Portfolio(cash=100_000)
    d1 = execute_option_paper_candidate(
        OptionPaperCandidate("SELL_TO_OPEN", _contract(), 1, "csp", 1.5, _TODAY),
        paper, equity)
    assert d1.accepted and paper.positions[d1.contract_id] == -1
    assert paper.reserved_cash == pytest.approx(9_500.0)
    d2 = execute_option_paper_candidate(
        OptionPaperCandidate("BUY_TO_CLOSE", _contract(), 1, "csp", 1.5, _TODAY),
        paper, equity)
    assert d2.accepted and d2.contract_id not in paper.positions
    assert paper.reserved_cash == 0.0 and paper.cash == pytest.approx(100_000.0)


def test_chain_snapshot_valid_round_trip_and_1e999_strike():
    snap = OptionChainSnapshot("SPY", 600.0, _TODAY, "test", [_contract()])
    with tempfile.TemporaryDirectory() as td:
        chain_dir = Path(td)
        save_option_chain_snapshot(snap, chain_dir)
        loaded = load_option_chain_snapshot("SPY", chain_dir)
        assert loaded.underlying == "SPY" and len(loaded.contracts) == 1
        assert loaded.contracts[0].strike == 95.0
        assert len(load_available_option_chains(chain_dir)) == 1
        # A stored strike of 1e999 (inf after float parsing) is rejected
        # at load, and the collection loader skips the file.
        raw = json.loads((chain_dir / "SPY.json").read_text())
        text = json.dumps(raw).replace('"strike": 95.0', '"strike": 1e999')
        (chain_dir / "SPY.json").write_text(text)
        with pytest.raises(ValueError):
            load_option_chain_snapshot("SPY", chain_dir)
        assert load_available_option_chains(chain_dir) == []


# ================================================================ evidence store / verify_run

def test_snapshot_dir_rejects_uppercase_mixed_and_padded_digests():
    base = Path("/tmp/evidence-base")
    for bad in ("sha256:" + "AB" * 32,          # uppercase hex
                "sha256:" + "aB" * 32,          # mixed case
                "sha256:" + "ab" * 32 + "\n",   # trailing newline (fullmatch, not match+$)
                "sha256: " + "ab" * 32,         # leading space
                "sha256:" + "ab" * 16 + "/../../" + "cd" * 20,  # valid hex prefix + traversal
                "sha256:" + "ab" * 32 + "/extra",
                "SHA256:" + "ab" * 32):
        with pytest.raises(ValueError):
            _snapshot_dir(base, bad)


def test_snapshot_dir_valid_digest_confined_and_missing_fails_cleanly(tmp_path):
    good = "sha256:" + "0123456789abcdef" * 4
    resolved = _snapshot_dir(tmp_path, good)
    assert resolved.is_relative_to(tmp_path / "snapshots")
    # Well-formed but nonexistent: a clean FileNotFoundError, no weird crash.
    with pytest.raises(FileNotFoundError):
        read_extracted_text(tmp_path, good)
    with pytest.raises(FileNotFoundError):
        read_snapshot_manifest(tmp_path, good)


def _write_snapshot(run_dir: Path, digest: str, extracted: str, raw: bytes) -> None:
    from market_lab.web_evidence_store import ensure_layout
    base = ensure_layout(run_dir)
    sdir = base / "snapshots" / "sha256" / digest[:2] / digest
    sdir.mkdir(parents=True, exist_ok=True)
    (sdir / "raw.bin").write_bytes(raw)
    (sdir / "extracted.txt").write_text(extracted, encoding="utf-8")
    (sdir / "manifest.json").write_text(json.dumps({"raw_sha256": sha256_hex(raw)}))


def _bundle(tmp_path: Path, *, evidence_snapshot: str, segment_snapshot: str | None,
            write_segments: bool = True) -> dict:
    digest_a, digest_b = "a" * 64, "b" * 64
    sid = {"a": f"sha256:{digest_a}", "b": f"sha256:{digest_b}"}
    _write_snapshot(tmp_path, digest_a, "alpha excerpt line\n", b"raw-a")
    _write_snapshot(tmp_path, digest_b, "beta excerpt line\n", b"raw-b")
    (tmp_path / "web_evidence").mkdir(exist_ok=True)
    if write_segments:
        excerpt = "beta excerpt line" if segment_snapshot == "b" else "alpha excerpt line"
        seg = {"segment_id": "seg-1", "snapshot_id": sid[segment_snapshot],
               "locator_type": "text_line_range", "locator": "1-1",
               "verbatim_excerpt_or_value": excerpt,
               "segment_sha256": sha256_hex(excerpt)}
        (tmp_path / "web_evidence" / "segments.jsonl").write_text(json.dumps(seg) + "\n")
    idx_rows = [
        {"snapshot_id": sid[k], "manifest_path": f"web_evidence/snapshots/sha256/{d[:2]}/{d}/manifest.json"}
        for k, d in (("a", digest_a), ("b", digest_b))]
    (tmp_path / "web_evidence" / "snapshot_index.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in idx_rows))
    ev = {"snapshot_id": sid[evidence_snapshot], "segment_id": "seg-1"}
    (tmp_path / "evidence.jsonl").write_text(json.dumps(ev) + "\n")
    return verify_run(tmp_path, require_snapshots=True)


def test_verify_run_reverse_direction_cross_snapshot_rejected(tmp_path):
    # Baseline covered evidence=A/segment=B. Reverse: evidence row cites
    # snapshot B, the segment's excerpt lives in snapshot A — the binding
    # check compares against THIS row's snapshot, so it must still fail.
    checks = _bundle(tmp_path, evidence_snapshot="b", segment_snapshot="a")
    assert checks["snapshots_complete"] is False


def test_verify_run_reverse_direction_bound_segment_accepted(tmp_path):
    checks = _bundle(tmp_path, evidence_snapshot="b", segment_snapshot="b")
    assert checks["snapshots_complete"] is True


def test_verify_run_missing_segments_file_fails_cleanly(tmp_path):
    checks = _bundle(tmp_path, evidence_snapshot="a", segment_snapshot=None,
                     write_segments=False)
    assert checks["has_evidence"] is True
    assert checks["snapshots_complete"] is False  # no segment to bind -> not complete, no crash


# ================================================================ fetcher: header caps + deadline

class _FakeSock:
    def __init__(self, payload: bytes):
        self._buf = io.BytesIO(payload)

    def makefile(self, mode):
        return self._buf


def _header_line(name: str, total: int) -> bytes:
    prefix = f"{name}: ".encode()
    line = prefix + b"v" * (total - len(prefix) - 2) + b"\r\n"
    assert len(line) == total
    return line


def test_header_count_exactly_at_cap_passes_and_one_over_fails():
    ok = (b"HTTP/1.1 200 OK\r\n"
          + b"".join(b"X-H%d: v\r\n" % i for i in range(_MAX_RESPONSE_HEADERS))
          + b"\r\nbody")
    status, body, headers = _read_http_response(_FakeSock(ok), max_bytes=1000)
    assert status == 200 and len(headers) == _MAX_RESPONSE_HEADERS

    over = (b"HTTP/1.1 200 OK\r\n"
            + b"".join(b"X-H%d: v\r\n" % i for i in range(_MAX_RESPONSE_HEADERS + 1))
            + b"\r\nbody")
    with pytest.raises(RuntimeError, match="header limit"):
        _read_http_response(_FakeSock(over), max_bytes=1000)


def test_header_duplicate_keys_bounded_by_byte_cap_not_count():
    # The count cap counts DISTINCT keys (dict), so repeated keys are
    # bounded by the byte budget instead — both bounds must hold.
    dup_ok = b"HTTP/1.1 200 OK\r\n" + b"X-Dup: v\r\n" * 200 + b"\r\nbody"
    status, _, headers = _read_http_response(_FakeSock(dup_ok), max_bytes=1000)
    assert status == 200 and headers == {"x-dup": "v"}
    dup_huge = b"HTTP/1.1 200 OK\r\n" + b"X-Dup: v\r\n" * 7000 + b"\r\nbody"  # ~70KB of tiny duplicate lines
    with pytest.raises(RuntimeError, match="header limit"):
        _read_http_response(_FakeSock(dup_huge), max_bytes=10_000_000)


def test_header_bytes_exactly_at_cap_passes_and_one_over_fails():
    half = _MAX_HEADER_BYTES // 2
    at = (b"HTTP/1.1 200 OK\r\n" + _header_line("X-A", half)
          + _header_line("X-B", _MAX_HEADER_BYTES - half) + b"\r\nok")
    status, body, _ = _read_http_response(_FakeSock(at), max_bytes=1000)
    assert status == 200 and body == b"ok"

    over = (b"HTTP/1.1 200 OK\r\n" + _header_line("X-A", half)
            + _header_line("X-B", _MAX_HEADER_BYTES - half + 1) + b"\r\nok")
    with pytest.raises(RuntimeError, match="header limit"):
        _read_http_response(_FakeSock(over), max_bytes=1000)


class _FakeRawSock:
    def __init__(self, payload: bytes):
        self._buf = io.BytesIO(payload)
        self.timeout = None
        self.sent = b""

    def settimeout(self, t):
        self.timeout = t

    def sendall(self, data):
        self.sent = data

    def makefile(self, mode):
        return self._buf

    def close(self):
        pass


def test_single_pinned_fetch_deadline_already_expired():
    created = []
    with patch("market_lab.web_evidence_providers._resolve_public",
               lambda host, port: ["203.0.113.10"]), \
         patch("market_lab.web_evidence_providers.socket.create_connection",
               lambda *a, **k: created.append(1)):
        with pytest.raises(RuntimeError, match="aggregate deadline"):
            _single_pinned_fetch("http://example.com/x", timeout_seconds=45,
                                 max_bytes=1000, deadline=time.time() - 1)
    assert created == []  # refused before opening any socket


def test_single_pinned_fetch_effective_timeout_is_remaining_budget():
    captured = {}

    def fake_connect(addr, timeout=None):
        captured["timeout"] = timeout
        return _FakeRawSock(b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\nhi")

    with patch("market_lab.web_evidence_providers._resolve_public",
               lambda host, port: ["203.0.113.10"]), \
         patch("market_lab.web_evidence_providers.socket.create_connection", fake_connect):
        status, body, _ = _single_pinned_fetch(
            "http://example.com/x", timeout_seconds=45, max_bytes=1000,
            deadline=time.time() + 0.5)
    assert status == 200 and body == b"hi"
    assert 0 < captured["timeout"] <= 0.5  # the remaining budget, not the full 45s

    with patch("market_lab.web_evidence_providers._resolve_public",
               lambda host, port: ["203.0.113.10"]), \
         patch("market_lab.web_evidence_providers.socket.create_connection", fake_connect):
        _single_pinned_fetch("http://example.com/x", timeout_seconds=45, max_bytes=1000)
    assert captured["timeout"] == 45  # no deadline: unchanged behavior


def test_http_fetch_deadline_expires_mid_redirect_chain():
    calls = {"n": 0}

    def slow_fetch(url, *, timeout_seconds, max_bytes, deadline=None):
        calls["n"] += 1
        time.sleep(1.2)  # burn past the 1s aggregate budget on hop one
        return 301, b"", {"location": "https://example.com/next"}

    with patch("market_lab.web_evidence_providers.classify_url_safety",
               lambda u: (True, u, "")), \
         patch("market_lab.web_evidence_providers._single_pinned_fetch", slow_fetch):
        with pytest.raises(RuntimeError, match="aggregate deadline"):
            _http_fetch("https://example.com/start", timeout_seconds=1,
                        max_bytes=1000, max_redirects=50)
    assert calls["n"] == 1  # chain aborted at the deadline, not after 50 hops


def test_http_fetch_redirect_chain_within_budget_unchanged():
    def fast_fetch(url, *, timeout_seconds, max_bytes, deadline=None):
        assert deadline is not None  # _http_fetch always threads its deadline now
        if url.endswith("/start"):
            return 301, b"", {"location": "https://example.com/final"}
        return 200, b"done", {}

    with patch("market_lab.web_evidence_providers.classify_url_safety",
               lambda u: (True, u, "")), \
         patch("market_lab.web_evidence_providers._single_pinned_fetch", fast_fetch):
        status, payload, _, final_url, chain = _http_fetch(
            "https://example.com/start", timeout_seconds=10, max_bytes=1000)
    assert status == 200 and payload == b"done" and len(chain) == 2


# ================================================================ thesis sizing

_CTX = PortfolioContext(total_cash=100_000.0, total_paper_exposure=0.0)


def test_thesis_conviction_boundaries_exact():
    d1 = compute_deterministic_sizing(_CTX, 100.0, conviction=1.0)
    assert not d1.rejected and d1.approved_notional == Decimal("4000.00")
    for conv in (0.0, -0.0):
        d = compute_deterministic_sizing(_CTX, 100.0, conviction=conv)
        assert d.rejected and d.approved_notional == 0
    for conv in (1.0000001, -1e-12, float("1e999")):
        d = compute_deterministic_sizing(_CTX, 100.0, conviction=conv)
        assert d.rejected and d.approved_notional == 0


def test_thesis_tiny_conviction_rounding_to_zero_rejected():
    # [fix] conviction=1e-9 sized 0.000004 -> rounds to Decimal("0.00"),
    # previously returned with rejected=False (a zero approval in name
    # only). One cent exactly must still approve.
    d = compute_deterministic_sizing(_CTX, 100.0, conviction=1e-9)
    assert d.rejected and d.approved_notional == 0
    cent = compute_deterministic_sizing(_CTX, 100.0, conviction=2.5e-6)
    assert not cent.rejected and cent.approved_notional == Decimal("0.01")


def test_thesis_caps_forcing_nonpositive_rejected():
    exhausted = PortfolioContext(total_cash=100_000.0, total_paper_exposure=20_000.0)
    d = compute_deterministic_sizing(exhausted, 100.0, conviction=0.001)
    assert d.rejected and d.approved_notional == 0
    over = PortfolioContext(total_cash=100_000.0, total_paper_exposure=25_000.0)
    d2 = compute_deterministic_sizing(over, 100.0, conviction=0.5)
    assert d2.rejected and d2.approved_notional == 0


def test_thesis_nonfinite_price_and_state_rejected():
    # [fix] inf price passed the old `price > 0` gate; NaN/inf portfolio
    # state produced approved notionals of NaN / Infinity, rejected=False.
    for price in (INF, -INF):
        d = compute_deterministic_sizing(_CTX, price, conviction=0.5)
        assert d.rejected and d.approved_notional == 0
    for ctx in (PortfolioContext(total_cash=NAN, total_paper_exposure=0.0),
                PortfolioContext(total_cash=INF, total_paper_exposure=0.0),
                PortfolioContext(total_cash=100_000.0, total_paper_exposure=NAN)):
        d = compute_deterministic_sizing(ctx, 100.0, conviction=0.5)
        assert d.rejected and d.approved_notional == 0
        assert d.approved_notional.is_finite()
