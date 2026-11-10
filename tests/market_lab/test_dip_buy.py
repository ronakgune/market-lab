"""Baseline tests for market_lab/dip_buy.py (Stream 7 price rule).

Price paths are deterministic constructions, not market data: a
linear +0.05/day rise puts the close a known ~4.2% above its SMA200
with a strongly rising slope, so a one-day drop of a chosen size
lands the close in a known tranche band with wide margins from the
3% / 7% / 12% registered thresholds.
"""
from datetime import date, timedelta

from market_lab.data import Bar
from market_lab.dip_buy import generate_dip_buy_signal, with_quality_gate
from market_lab.fundamentals import QualityReport
from market_lab.validation import check_lookahead, signal_series

START = date(2023, 1, 2)


def bars_from_closes(closes, start=START):
    return [Bar(start + timedelta(days=i), c, c, c, c, 1_000_000)
            for i, c in enumerate(closes)]


def rising_base(n=400, start_price=100.0, step=0.05):
    return [start_price + step * i for i in range(n)]


def dip_series(drop: float, hold_days: int = 3):
    """400 rising bars, then a one-day drop of `drop`, held flat."""
    closes = rising_base()
    dipped = closes[-1] * (1 - drop)
    return bars_from_closes(closes + [dipped] * hold_days)


def test_warmup_returns_insufficient_history_hold():
    for n in (0, 100, 259):
        sig = generate_dip_buy_signal("ACME", bars_from_closes(rising_base(n)))
        assert sig.action == "HOLD" and sig.target_weight == 0.0
        assert "insufficient history" in sig.reason


def test_above_the_line_is_a_flat_sell():
    sig = generate_dip_buy_signal("ACME", bars_from_closes(rising_base()))
    assert sig.action == "SELL" and sig.target_weight == 0.0
    assert sig.evidence["tranche"] == 0
    assert sig.evidence["depth_below"] < 0


def test_shallow_dip_below_first_tranche_holds():
    sig = generate_dip_buy_signal("ACME", dip_series(0.05))  # ~1% below
    assert sig.action == "HOLD" and sig.target_weight == 0.0
    assert sig.evidence["tranche"] == 0
    assert 0 < sig.evidence["depth_below"] < 0.03


def test_tranche_ladder_steps_with_depth():
    cases = [(0.09, 1, 1 / 3), (0.13, 2, 2 / 3), (0.18, 3, 1.0)]
    for drop, tranche, weight in cases:
        sig = generate_dip_buy_signal("ACME", dip_series(drop))
        assert sig.action == "BUY", (drop, sig)
        assert sig.evidence["tranche"] == tranche
        assert sig.target_weight == weight
        assert sig.evidence["slope"] > 0
        assert sig.evidence["sma200"] > 0


def test_episode_depth_sticks_as_price_recovers_within_the_dip():
    closes = rising_base()
    deep = closes[-1] * (1 - 0.13)          # tranche-2 depth
    recovered = closes[-1] * (1 - 0.085)    # back to ~4-5% below the line
    sig = generate_dip_buy_signal("ACME", bars_from_closes(closes + [deep] * 2 + [recovered] * 2))
    assert sig.action == "BUY"
    assert sig.target_weight == 2 / 3  # deepest episode depth still rules
    assert sig.evidence["depth_below"] < 0.07 < sig.evidence["episode_depth"]


def test_reclaim_exits_to_flat():
    closes = rising_base()
    dipped = closes[-1] * (1 - 0.09)
    series = bars_from_closes(closes + [dipped] * 2 + [closes[-1]] * 2)
    sig = generate_dip_buy_signal("ACME", series)
    assert sig.action == "SELL" and sig.target_weight == 0.0
    assert sig.evidence["episode_depth"] == 0.0


def test_falling_average_blocks_the_buy():
    closes = [200.0 - 0.15 * i for i in range(400)]  # steep, steady decline
    sig = generate_dip_buy_signal("ACME", bars_from_closes(closes))
    assert sig.evidence["slope"] < 0
    assert sig.evidence["depth_below"] > 0.07  # deep enough for tranche 2...
    assert sig.action == "HOLD" and sig.target_weight == 0.0  # ...but no knife-catching


def test_signal_is_causal_under_truncation():
    closes = rising_base(300)
    dipped = closes[-1] * (1 - 0.13)
    closes = closes + [dipped] * 10 + [closes[-1] * (1 - 0.04)] * 10
    bars = bars_from_closes(closes)
    report = check_lookahead(signal_series(generate_dip_buy_signal, "ACME"),
                             bars, cutoffs=[265, 290, 310, len(bars)])
    assert report.passed, report.violations[:3]


def _report(passes: bool, reasons=()):
    return QualityReport("ACME", START, passes, None, None, None, None, (), 0, None, tuple(reasons))


def test_quality_gate_passes_buy_through_with_evidence():
    gated = with_quality_gate(generate_dip_buy_signal, lambda s, d: _report(True))
    sig = gated("ACME", dip_series(0.09))
    assert sig.action == "BUY" and sig.target_weight == 1 / 3
    assert sig.evidence["quality"] == "pass"


def test_quality_gate_blocks_buy_with_reasons():
    gated = with_quality_gate(
        generate_dip_buy_signal,
        lambda s, d: _report(False, ("ROE 4.0% below the 10% floor",)),
    )
    sig = gated("ACME", dip_series(0.09))
    assert sig.action == "HOLD" and sig.target_weight == 0.0
    assert sig.evidence["quality"] == "fail"
    assert "ROE 4.0%" in sig.reason


def test_quality_gate_never_blocks_an_exit():
    def boom(symbol, as_of):
        raise AssertionError("gate must not be consulted for non-BUY signals")

    gated = with_quality_gate(generate_dip_buy_signal, boom)
    sig = gated("ACME", bars_from_closes(rising_base()))  # SELL (above the line)
    assert sig.action == "SELL"
