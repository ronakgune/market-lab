"""Stream 4 machinery tests.

Two layers:
- REAL probed chains (fixtures, fetched live 2026-11-17) pin the
  data-forced behavior: expiry-band skips, risk-cap skips on sampled
  strike spacing, delta-band gaps.
- A small hand-built DENSE chain (clearly synthetic, strikes every 2
  points — the grid the real SPY market has but the sampled dataset
  does not) exercises the success paths: construction math, fill
  sides, management triggers, settlement, book accounting.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

from market_lab.data import Bar
from market_lab.options_history import ChainDay, OptionRow, parse_chain_rows
from market_lab import premium_selling as ps

FIXTURES = Path(__file__).parent / "fixtures" / "options"


def real_chain(day: str) -> ChainDay:
    payload = json.loads((FIXTURES / f"spy_chain_{day}.json").read_text(encoding="utf-8"))
    d = date.fromisoformat(day)
    return ChainDay("SPY", d, tuple(parse_chain_rows("SPY", d, payload["rows"])))


# --- Real-chain behavior -------------------------------------------------------

def test_real_chain_2026_selects_registered_expiry():
    chain = real_chain("2026-11-13")
    assert ps.select_expiry(chain, chain.day) == date(2026, 12, 24)  # 41 DTE


def test_real_chain_2026_put_spread_skips_on_risk_cap():
    chain = real_chain("2026-11-13")
    result = ps.build_put_credit_spread(chain, chain.day)
    assert isinstance(result, ps.Skip)
    # Sampled spacing: short 695 / wing 680 = 15pt wide; credit 0.88.
    assert "risk cap allows 0 contracts" in result.reason
    assert "$1,412" in result.reason


def test_real_chain_2026_condor_skips_on_delta_band_gap():
    chain = real_chain("2026-11-13")
    result = ps.build_iron_condor(chain, chain.day)
    assert isinstance(result, ps.Skip)
    assert "no CALL strike" in result.reason


def test_real_chain_2020_crash_day_skips_on_expiry_band():
    chain = real_chain("2020-03-18")  # expiries at 14 / 28 / 58 DTE only
    for builder in (ps.build_put_credit_spread, ps.build_iron_condor):
        result = builder(chain, chain.day)
        assert isinstance(result, ps.Skip)
        assert "no expiry with DTE in [30,45]" in result.reason


def test_real_chain_wing_missing_below_lowest_strike():
    # 2020-04-15 expiry: the only in-band put is the 190 strike, which is
    # also the lowest strike in the sampled chain — no wing exists.
    chain = real_chain("2020-03-18")
    rows = chain.for_expiry(date(2020, 4, 15))
    short = ps._by_delta(rows, "PUT")
    assert short is not None and short.strike == 190.0
    assert ps._next_strike_beyond(rows, "PUT", short.strike, -1) is None


# --- Dense logic chain (synthetic grid, real-shaped quotes) --------------------

DAY0 = date(2026, 1, 5)
EXP = DAY0 + timedelta(days=37)


def row(day, exp, strike, right, bid, ask, delta, iv=0.20):
    return OptionRow(chain_date=day, symbol="SPY", expiration=exp,
                     strike=float(strike), right=right, bid=bid, ask=ask,
                     iv=iv, delta=delta, gamma=0.01, rho=0.01,
                     theta=-0.02, vega=0.05)


def dense_chain(day=DAY0, *, short_put_ask=2.00, long_put_bid=1.05,
                iv=0.20) -> ChainDay:
    rows = [
        row(day, EXP, 90, "PUT", 0.55, 0.65, -0.05, iv),
        row(day, EXP, 92, "PUT", long_put_bid, 1.15, -0.08, iv),
        row(day, EXP, 94, "PUT", 1.90, short_put_ask, -0.13, iv),
        row(day, EXP, 96, "PUT", 2.60, 2.72, -0.22, iv),
        row(day, EXP, 104, "CALL", 2.62, 2.74, 0.22, iv),
        row(day, EXP, 106, "CALL", 1.85, 1.95, 0.13, iv),
        row(day, EXP, 108, "CALL", 1.00, 1.10, 0.08, iv),
        row(day, EXP, 110, "CALL", 0.52, 0.62, 0.05, iv),
    ]
    return ChainDay("SPY", day, tuple(rows))


def test_put_spread_construction_fill_sides_and_sizing():
    s = ps.build_put_credit_spread(dense_chain(), DAY0)
    assert isinstance(s, ps.Structure)
    # Short fills at BID (1.90), long wing at ASK (1.15): credit 0.75.
    assert s.credit_per_share == 1.90 - 1.15
    assert {leg.strike for leg in s.legs} == {94.0, 92.0}
    # Width 2 -> max loss (2 - 0.75) * 100 = $125 -> 2 contracts under $250.
    assert s.max_loss_per_contract == 125.0
    assert s.contracts == 2
    assert s.max_loss_total == 250.0
    assert s.entry_fees == 2 * 2 * ps.COMMISSION_PER_CONTRACT


def test_condor_construction():
    s = ps.build_iron_condor(dense_chain(), DAY0)
    assert isinstance(s, ps.Structure)
    assert s.credit_per_share == (1.90 + 1.85) - (1.15 + 1.10)
    assert len(s.legs) == 4


def test_mtm_signs_and_take_profit():
    book = ps.PaperBook()
    s = ps.build_put_credit_spread(dense_chain(), DAY0)
    pos = book.open(s, DAY0)
    # Next chain day: short put ask collapses, long put bid holds ->
    # close cost/share = 0.60 - 0.30 = 0.30 -> captured > 50% of credit.
    day1 = dense_chain(DAY0 + timedelta(days=1), short_put_ask=0.60,
                       long_put_bid=0.30)
    dec = ps.evaluate_management(pos, day1)
    assert dec.action == "EXIT" and dec.reason == "take_profit"
    trade = book.close(pos, day1.day, dec.reason, dec.close_cost_total)
    assert trade.pnl > 0
    # PnL = credit 150 - entry fees 2.60 - close (0.30*200 + 2.60) = 84.80
    assert abs(trade.pnl - 84.80) < 1e-9


def test_stop_loss_trigger():
    book = ps.PaperBook()
    s = ps.build_put_credit_spread(dense_chain(), DAY0)
    pos = book.open(s, DAY0)
    day1 = dense_chain(DAY0 + timedelta(days=1), short_put_ask=3.40,
                       long_put_bid=0.90)
    dec = ps.evaluate_management(pos, day1)
    assert dec.action == "EXIT" and dec.reason == "stop_loss"


def test_hard_exit_dte_precedes_other_triggers():
    book = ps.PaperBook()
    s = ps.build_put_credit_spread(dense_chain(), DAY0)
    pos = book.open(s, DAY0)
    late_day = EXP - timedelta(days=5)
    day_late = dense_chain(late_day, short_put_ask=1.45, long_put_bid=0.90)
    dec = ps.evaluate_management(pos, day_late)
    assert dec.action == "EXIT" and dec.reason == "hard_exit_dte"


def test_missing_leg_means_no_decision():
    book = ps.PaperBook()
    s = ps.build_put_credit_spread(dense_chain(), DAY0)
    pos = book.open(s, DAY0)
    partial = ChainDay("SPY", DAY0 + timedelta(days=1),
                       tuple(r for r in dense_chain().rows if r.strike != 92.0))
    dec = ps.evaluate_management(pos, partial)
    assert dec.action == "HOLD" and dec.reason == "leg_missing_from_chain_no_decision"


def test_settlement_at_intrinsic():
    book = ps.PaperBook()
    s = ps.build_put_credit_spread(dense_chain(), DAY0)
    pos = book.open(s, DAY0)
    trade = book.settle(pos, underlying_close=91.0)
    # Short 94 put owes 3, long 92 put receives 1: payoff -2/share.
    # PnL = 150 - 2.60 - 400 = -252.60 (max loss 250 + entry fees).
    assert abs(trade.pnl - (-252.60)) < 1e-9
    assert trade.exit_reason == "expired_settled"


def test_aggregate_risk_cap():
    book = ps.PaperBook()
    s = ps.build_put_credit_spread(dense_chain(), DAY0)
    for _ in range(5):  # 5 x $250 = $1,250 = the aggregate cap
        book.open(s, DAY0)
    ok, why = book.can_open(s)
    assert not ok and "aggregate open-risk cap" in why


def test_excursion_breach_uses_daily_bars():
    book = ps.PaperBook()
    s = ps.build_put_credit_spread(dense_chain(), DAY0)
    pos = book.open(s, DAY0)
    bars = [Bar(date=DAY0 + timedelta(days=2), open=99.0, high=100.0,
                low=93.50, close=98.0, volume=1_000_000)]
    breach = ps.excursion_breach(pos, bars)
    assert breach is not None and "below short PUT strike 94" in breach
    calm = [Bar(date=DAY0 + timedelta(days=2), open=99.0, high=100.0,
                low=95.50, close=98.0, volume=1_000_000)]
    assert ps.excursion_breach(pos, calm) is None


def _iv_chain(day: date, iv: float) -> ChainDay:
    exp = day + timedelta(days=37)
    rows = [
        row(day, exp, 94, "PUT", 1.90, 2.00, -0.13, iv),
        row(day, exp, 106, "CALL", 1.85, 1.95, 0.13, iv),
    ]
    return ChainDay("SPY", day, tuple(rows))


def test_iv_gate_percentile_and_history_floor():
    priors = [_iv_chain(DAY0 - timedelta(days=30 - i), 0.10 + 0.01 * i)
              for i in range(25)]  # IV indices 0.10 .. 0.34
    rich = _iv_chain(DAY0, 0.40)
    ok, _ = ps.iv_gate(rich, priors)
    assert ok
    cheap = _iv_chain(DAY0, 0.05)
    ok, why = ps.iv_gate(cheap, priors)
    assert not ok and "percentile" in why
    ok, why = ps.iv_gate(rich, priors[:10])
    assert not ok and "insufficient IV history" in why


def test_iv_gate_no_lookahead_future_chains_ignored():
    priors = [_iv_chain(DAY0 - timedelta(days=30 - i), 0.10 + 0.01 * i)
              for i in range(25)]
    today = _iv_chain(DAY0, 0.20)
    baseline = ps.iv_gate(today, priors)
    poisoned = priors + [_iv_chain(DAY0 + timedelta(days=7), 9.99),
                         _iv_chain(DAY0, 9.99)]  # same-day and future
    assert ps.iv_gate(today, poisoned) == baseline


def test_snapshot_and_drawdown():
    book = ps.PaperBook()
    s = ps.build_put_credit_spread(dense_chain(), DAY0)
    book.open(s, DAY0)
    snap = book.snapshot(dense_chain())
    assert snap.open_positions == 1 and snap.open_risk == 250.0
    assert ps.max_drawdown([25_000.0, 26_000.0, 23_400.0]) == \
        (26_000.0 - 23_400.0) / 26_000.0
