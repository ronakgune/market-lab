"""Adversarial tests for the Stream 2 dual-momentum rewrite.

Attack surface: measurement integrity of run_dual_momentum_backtest and the
xs-momentum runner. Every expected value below is hand-computed from the
fixture constants and the execution SPEC (fill = open x (1 +/- 5bps),
$1 commission per trade, buy quantities scaled so notionals + commissions
fit the cash budget) — never copied from the code's own outputs.

Fixtures are deterministic synthetic series: mechanics, not evidence.
"""
import importlib.util
import math
from datetime import date, timedelta
from pathlib import Path
from statistics import pstdev

import pytest

from market_lab.backtest import ExecutionModel
from market_lab.data import Bar
from market_lab.portfolio_construction import (
    _vol_scale,
    dual_momentum_targets,
    run_dual_momentum_backtest,
)
from market_lab.validation import TrialLedger

FREE = ExecutionModel(0.0, 0.0)
DEFAULT_EXEC = ExecutionModel()  # 5bps + $1

_spec = importlib.util.spec_from_file_location(
    "run_xs_momentum_backtest",
    Path(__file__).resolve().parents[2] / "scripts" / "run_xs_momentum_backtest.py",
)
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)

START_DATE = date(2024, 1, 1)


def bars_oc(pairs, start=START_DATE):
    """Bars from (open, close) pairs; high/low bracket the two."""
    return [
        Bar(start + timedelta(days=i), o, max(o, c), min(o, c), c, 1_000_000)
        for i, (o, c) in enumerate(pairs)
    ]


def bars_from_closes(closes, start=START_DATE):
    return bars_oc([(c, c) for c in closes], start=start)


def rising(n, base, step):
    return [base + step * i for i in range(n)]


# ---------------------------------------------------------------------------
# 1. Cost accounting: forced full turnover, hand-computed to the cent.
#
# OLD rises hard into decision 10, then decays; NEW is flat then rises.
# Rebalance 1 (decision 10, fill 11) buys OLD; rebalance 2 (decision 30,
# fill 31) sells all of OLD and buys NEW. Opens == closes throughout.
# ---------------------------------------------------------------------------

def _turnover_fixture(old_open_31=None):
    old_closes = [100 + 10 * i for i in range(11)] + [200 - 2 * (i - 10) for i in range(11, 40)]
    new_closes = [100] * 21 + [100 + 5 * (i - 20) for i in range(21, 40)]
    old = bars_from_closes(old_closes)
    new = bars_from_closes(new_closes)
    if old_open_31 is not None:
        b = old[31]
        old[31] = Bar(b.date, old_open_31, max(old_open_31, b.close), min(old_open_31, b.close), b.close, b.volume)
    return {"OLD": old, "NEW": new}


TURNOVER_KW = dict(formation_days=10, skip_days=0, top_n=1, rebalance_every=20,
                   initial_cash=10_000.0, max_weight=1.0)


def _expected_turnover(old_open_31=158.0):
    """Hand-computed ledger for the turnover fixture under honest accounting."""
    # Rebalance 1: buy OLD at fill bar 11, open 198.
    buy_px1 = 198.0 * 1.0005
    qty1 = (10_000.0 / buy_px1) * ((10_000.0 - 1.0) / 10_000.0)  # scale reserves the $1
    costs1 = qty1 * (buy_px1 - 198.0) + 1.0
    # Rebalance 2: sell OLD at fill bar 31 open, buy NEW at open 155.
    sell_px = old_open_31 * 0.9995
    proceeds = qty1 * sell_px - 1.0
    equity_at_fill = qty1 * old_open_31
    buy_px2 = 155.0 * 1.0005
    room = proceeds - 1.0  # budget minus the buy commission
    qty2 = (equity_at_fill / buy_px2) * (room / equity_at_fill)
    costs2 = qty1 * (old_open_31 - sell_px) + 1.0 + qty2 * (buy_px2 - 155.0) + 1.0
    return qty1, qty2, costs1 + costs2


def test_full_turnover_cost_accounting_matches_hand_computation():
    result = run_dual_momentum_backtest(_turnover_fixture(), **TURNOVER_KW)
    qty1, qty2, total_costs = _expected_turnover()
    assert result.rebalances == 2
    first, second = result.snapshots
    assert [t.symbol for t in first.targets] == ["OLD"]
    assert [t.symbol for t in second.targets] == ["NEW"]
    assert first.fill_prices["OLD"] == pytest.approx(198.0 * 1.0005, rel=1e-12)
    assert first.equity_before == pytest.approx(10_000.0, rel=1e-12)
    # equity_after must be exactly cash (0 after full investment) + qty x close.
    assert first.equity_after == pytest.approx(qty1 * 198.0, rel=1e-9)
    assert second.equity_after == pytest.approx(qty2 * 155.0, rel=1e-9)
    assert result.costs_paid == pytest.approx(total_costs, rel=1e-9)
    assert result.final_equity == pytest.approx(qty2 * 195.0, rel=1e-9)  # NEW close at bar 39


# ---------------------------------------------------------------------------
# 2. Buy budget scaling: commissions must be reserved, not charged on top.
# ---------------------------------------------------------------------------

def test_buy_scaling_reserves_commissions_no_negative_cash():
    # Three names, one rebalance only, prices frozen at the fill close so the
    # final equity is exactly the bought quantities marked flat, plus cash.
    # With $25/trade commissions the scaling shortfall is decisive, not dust.
    closes_a = rising(12, 50, 5) + [105] * 18
    closes_b = rising(12, 200, 2) + [222] * 18
    closes_c = rising(12, 80, 1) + [91] * 18
    bars = {"A": bars_from_closes(closes_a), "B": bars_from_closes(closes_b),
            "C": bars_from_closes(closes_c)}
    exec_model = ExecutionModel(5.0, 25.0)
    result = run_dual_momentum_backtest(
        bars, formation_days=10, skip_days=0, top_n=3, rebalance_every=1000,
        initial_cash=10_000.0, max_weight=1.0, execution=exec_model,
    )
    assert result.rebalances == 1
    opens = {"A": 105.0, "B": 222.0, "C": 91.0}
    scale = (10_000.0 - 3 * 25.0) / 10_000.0
    expected_final = 0.0
    expected_costs = 3 * 25.0
    for sym, open_px in opens.items():
        buy_px = open_px * 1.0005
        qty = (10_000.0 / 3.0) / buy_px * scale
        expected_final += qty * open_px  # closes frozen at the fill close == open
        expected_costs += qty * (buy_px - open_px)
    # Cash after the rebalance is exactly 0 under honest scaling; a scaler
    # that ignores commissions overspends and leaves cash < 0, which shows
    # up here as a final equity HIGHER than the honest value (phantom money
    # was spent) net of the quantity difference — either way it cannot match.
    assert result.final_equity == pytest.approx(expected_final, rel=1e-9)
    assert result.costs_paid == pytest.approx(expected_costs, rel=1e-9)


def test_gap_up_fill_open_scales_buys_down_and_cash_lands_on_zero():
    # OLD gaps up at the fill open (300 vs 158 close): equity_at_fill is
    # marked at the gapped open, so desired buys exceed realizable proceeds.
    result = run_dual_momentum_backtest(_turnover_fixture(old_open_31=300.0), **TURNOVER_KW)
    qty1, qty2, total_costs = _expected_turnover(old_open_31=300.0)
    second = result.snapshots[1]
    unscaled_qty = (qty1 * 300.0) / (155.0 * 1.0005)
    assert qty2 < unscaled_qty  # the gap was NOT spent twice
    assert second.equity_after == pytest.approx(qty2 * 155.0, rel=1e-9)  # cash == 0, not < 0
    assert result.costs_paid == pytest.approx(total_costs, rel=1e-9)
    assert result.final_equity == pytest.approx(qty2 * 195.0, rel=1e-9)


# ---------------------------------------------------------------------------
# 3. Vol scaling: past data only, exact value, window respected.
# ---------------------------------------------------------------------------

def _curve_from_returns(rets, start=1000.0):
    curve = [start]
    for r in rets:
        curve.append(curve[-1] * (1.0 + r))
    return curve


def test_vol_scale_exact_value_from_known_pstdev():
    rets = [0.01, -0.01] * 10  # pstdev exactly 0.01, mean 0
    curve = _curve_from_returns(rets)
    realized = 0.01 * math.sqrt(252)
    assert _vol_scale(curve, realized / 2.0, 60) == pytest.approx(0.5, rel=1e-6)
    assert _vol_scale(curve, realized, 60) == pytest.approx(1.0, rel=1e-6)
    assert _vol_scale(curve, realized * 10.0, 60) == 1.0  # clamped, never leverages


def test_vol_scale_needs_twenty_returns_and_respects_window():
    wild_short = _curve_from_returns([0.05, -0.05] * 9 + [0.05])  # 19 returns
    assert _vol_scale(wild_short, 0.01, 60) == 1.0  # not enough history: no scaling
    # 40 wild returns then 60 calm ones; window 60 must see ONLY the calm tail.
    rets = [0.05, -0.05] * 20 + [0.001, -0.001] * 30
    curve = _curve_from_returns(rets)
    calm_scale = _vol_scale(curve, 0.01, 60)
    assert calm_scale == pytest.approx(0.01 / (0.001 * math.sqrt(252)), rel=1e-6)
    full_rets = [curve[i] / curve[i - 1] - 1.0 for i in range(1, len(curve))]
    full_scale = min(1.0, 0.01 / (pstdev(full_rets) * math.sqrt(252)))
    assert abs(calm_scale - full_scale) > 0.1  # window truly excludes the wild head
    assert _vol_scale(curve, None, 60) == 1.0  # disabled
    flat = _curve_from_returns([0.01] * 30)  # zero variance
    assert _vol_scale(flat, 0.01, 60) == 1.0


def test_vol_scale_and_snapshots_are_causal_end_to_end():
    n = 120
    base = {
        "AAA": bars_from_closes(rising(n, 100, 0.30)),
        "BBB": bars_from_closes(rising(n, 100, 0.20)),
        "CCC": bars_from_closes(rising(n, 100, 0.10)),
    }
    kw = dict(formation_days=10, skip_days=0, top_n=2, rebalance_every=5,
              initial_cash=25_000.0, max_weight=1.0, target_vol=0.05, vol_window=20)
    full = run_dual_momentum_backtest(base, **kw)
    assert full.snapshots[0].vol_scale == 1.0  # first decision: no return history
    # Mutate everything after bar 60 beyond recognition.
    mutated = {}
    for sym, bars in base.items():
        out = list(bars)
        for i in range(61, n):
            price = 10_000.0 if i % 2 else 10.0
            out[i] = Bar(out[i].date, price, price, price, price, 1_000_000)
        mutated[sym] = out
    mut = run_dual_momentum_backtest(mutated, **kw)
    prefix_full = [s for s in full.snapshots if s.fill_index <= 60]
    prefix_mut = [s for s in mut.snapshots if s.fill_index <= 60]
    assert len(prefix_full) >= 8
    assert prefix_mut == prefix_full  # scales, targets, fills: untouched by the future
    assert mut.final_equity != full.final_equity  # the mutation did bite downstream


# ---------------------------------------------------------------------------
# 4. Benchmark integrity.
# ---------------------------------------------------------------------------

BENCH_KW = dict(formation_days=60, skip_days=10, top_n=2, rebalance_every=21,
                start_index=70, initial_cash=25_000.0, max_weight=1.0)


def _bench_universe(n=400):
    return {
        "AAA": bars_from_closes(rising(n, 100, 0.05)),
        "LEADER": bars_from_closes(rising(n, 100, 0.60)),
        "MID": bars_from_closes(rising(n, 100, 0.30)),
    }


def test_benchmark_uses_spy_closes_at_exact_start_and_end_indices():
    spy_closes = [100.0] * 400
    spy_closes[0] = 40.0     # trap: using bar 0 instead of start (70)
    spy_closes[69] = 555.0   # trap: off-by-one before start
    spy_closes[399] = 250.0  # end = min_len - 1
    result = run_dual_momentum_backtest(
        _bench_universe(), spy_bars=bars_from_closes(spy_closes), execution=FREE, **BENCH_KW)
    assert result.benchmark_symbol == "SPY"
    assert result.benchmark_return == pytest.approx(250.0 / 100.0 - 1.0, rel=1e-12)


def test_benchmark_symbol_empty_and_zero_when_no_spy_anywhere():
    bars = {"AAA": bars_from_closes(rising(400, 100, 0.05)),
            "BBB": bars_from_closes(rising(400, 100, 0.30))}
    result = run_dual_momentum_backtest(bars, execution=FREE, **BENCH_KW)
    # The old code silently benchmarked to the alphabetically first symbol.
    assert result.benchmark_symbol == ""
    assert result.benchmark_return == 0.0


def test_spy_missing_a_single_middle_date_is_refused():
    spy_bars = bars_from_closes(rising(400, 100, 0.10))
    del spy_bars[200]  # one missing session in the middle of the calendar
    with pytest.raises(ValueError, match="do not cover"):
        run_dual_momentum_backtest(_bench_universe(), spy_bars=spy_bars,
                                   execution=FREE, **BENCH_KW)


# ---------------------------------------------------------------------------
# 5. Gate wiring: SPY absolute momentum must actually bite.
# ---------------------------------------------------------------------------

def test_spy_gate_empties_every_snapshot_when_spy_outruns_all_names():
    n = 400
    bars = {
        "AAA": bars_from_closes([100 * 1.001 ** i for i in range(n)]),
        "MID": bars_from_closes([100 * 1.002 ** i for i in range(n)]),
        "LEADER": bars_from_closes([100 * 1.004 ** i for i in range(n)]),
    }
    spy = bars_from_closes([100 * 1.010 ** i for i in range(n)])  # 60d SPY mom ~ +82%
    gated = run_dual_momentum_backtest(bars, spy_bars=spy, execution=FREE, **BENCH_KW)
    assert gated.rebalances > 0
    assert all(not s.targets for s in gated.snapshots)
    assert gated.final_equity == 25_000.0  # never traded: cash untouched, exactly
    assert gated.costs_paid == 0.0
    ungated = run_dual_momentum_backtest(bars, execution=FREE, **BENCH_KW)
    assert any(s.targets for s in ungated.snapshots)  # same names DO select without SPY


# ---------------------------------------------------------------------------
# 6. Lookahead: future bars cannot change a decision, at either level.
# ---------------------------------------------------------------------------

def test_targets_ignore_everything_after_the_decision_index():
    n = 120
    bars = {
        "AAA": bars_from_closes(rising(n, 100, 0.30)),
        "BBB": bars_from_closes(rising(n, 100, 0.20)),
        "CCC": bars_from_closes(rising(n, 100, 0.10)),
    }
    spy = bars_from_closes(rising(n, 100, 0.05))
    kw = dict(formation_days=30, skip_days=5, top_n=2, max_weight=1.0,
              decision_index=50, spy_bars=spy)
    honest = dual_momentum_targets(bars, **kw)
    assert honest  # fixture must actually select something

    def poisoned(series, value):
        out = list(series)
        for i in range(51, n):
            out[i] = Bar(out[i].date, value, value, value, value, 1_000_000)
        return out

    for value in (1e6, 0.01):
        mut_bars = {s: poisoned(b, value) for s, b in bars.items()}
        mut_kw = dict(kw, spy_bars=poisoned(spy, value))
        assert dual_momentum_targets(mut_bars, **mut_kw) == honest


def test_engine_snapshots_match_between_full_and_truncated_series():
    n = 120
    bars = {
        "AAA": bars_from_closes(rising(n, 100, 0.30)),
        "BBB": bars_from_closes(rising(n, 100, 0.20)),
        "CCC": bars_from_closes(rising(n, 100, 0.10)),
    }
    kw = dict(formation_days=10, skip_days=0, top_n=2, rebalance_every=5,
              initial_cash=25_000.0, max_weight=1.0, target_vol=0.05, vol_window=20)
    full = run_dual_momentum_backtest(bars, **kw)
    truncated = run_dual_momentum_backtest({s: b[:90] for s, b in bars.items()}, **kw)
    expected = [s for s in full.snapshots if s.decision_index <= 85]
    assert len(truncated.snapshots) >= 10
    assert truncated.snapshots == expected


# ---------------------------------------------------------------------------
# 7. Determinism and ledger fidelity.
# ---------------------------------------------------------------------------

def test_identical_inputs_produce_identical_results():
    bars = _bench_universe()
    spy = bars_from_closes(rising(400, 100, 0.10))
    kw = dict(BENCH_KW, spy_bars=spy, target_vol=0.15)  # default costs ON
    first = run_dual_momentum_backtest(bars, **kw)
    second = run_dual_momentum_backtest(bars, **kw)
    assert first == second  # metrics, costs, and every snapshot field


def test_runner_ledger_rows_match_returned_rows_exactly(tmp_path):
    n = 600
    bars = {
        "AAA": bars_from_closes(rising(n, 100, 0.30)),
        "BBB": bars_from_closes(rising(n, 100, 0.20)),
        "CCC": bars_from_closes(rising(n, 100, 0.10)),
    }
    spy = bars_from_closes(rising(n, 100, 0.05))
    meta = {"source": "fixture", "feed": "sip", "adjustment": "all"}
    ledger = TrialLedger(tmp_path / "trials.jsonl")
    rows = [
        runner.run_one(bars, spy, meta, n, 25_000.0, None, ledger),
        runner.run_one(bars, spy, meta, n, 25_000.0, n // 2, ledger),
    ]
    assert ledger.count() == 2
    for record, row in zip(ledger.records(), rows):
        assert record.stream == "xs_momentum"
        for key in ("total_return", "cagr", "benchmark_return", "sharpe",
                    "max_drawdown", "costs_paid"):
            assert record.metrics[key] == row[key]
        assert record.metrics["rebalances"] == float(row["rebalances"])
        assert record.params["formation_days"] == runner.FORMATION_DAYS
        assert record.params["target_vol"] == runner.TARGET_VOL
    # Same params into a fresh ledger -> identical params_hash (deterministic identity).
    ledger2 = TrialLedger(tmp_path / "trials2.jsonl")
    runner.run_one(bars, spy, meta, n, 25_000.0, None, ledger2)
    assert ledger2.records()[0].params_hash == ledger.records()[0].params_hash
    assert ledger2.records()[0].params == ledger.records()[0].params


# ---------------------------------------------------------------------------
# 8. Edge inputs.
# ---------------------------------------------------------------------------

def test_zero_or_negative_open_on_buy_fill_means_no_trade(tmp_path):
    closes = rising(40, 100, 5)
    for bad_open in (0.0, -10.0):
        bars = bars_from_closes(closes)
        b = bars[11]
        bars[11] = Bar(b.date, bad_open, max(bad_open, b.close), min(bad_open, b.close), b.close, b.volume)
        result = run_dual_momentum_backtest(
            {"UP": bars}, formation_days=10, skip_days=0, top_n=1,
            rebalance_every=1000, initial_cash=10_000.0, max_weight=1.0)
        assert result.rebalances == 1
        assert result.snapshots[0].targets  # selection happened...
        assert result.snapshots[0].fill_prices == {}  # ...but nothing could fill
        assert result.final_equity == 10_000.0
        assert result.costs_paid == 0.0


def _bad_sell_open_fixture(bad_open):
    """RISE is bought at rebalance 1; at rebalance 2 WINNER takes over and
    RISE is due to be sold — but RISE's fill-bar open is bad_open."""
    rise_closes = [100 + 5 * i for i in range(31)] + [250 - 10 * (i - 30) for i in range(31, 45)]
    winner_closes = [100] * 21 + [100 + 8 * (i - 20) for i in range(21, 45)]
    rise = bars_from_closes(rise_closes)
    b = rise[31]
    rise[31] = Bar(b.date, bad_open, max(bad_open, b.close), min(bad_open, b.close), b.close, b.volume)
    return {"RISE": rise, "WINNER": bars_from_closes(winner_closes)}


@pytest.mark.parametrize("bad_open", [0.0, -50.0])
def test_held_position_cannot_be_sold_at_a_nonpositive_open(bad_open):
    result = run_dual_momentum_backtest(
        _bad_sell_open_fixture(bad_open), formation_days=10, skip_days=0,
        top_n=1, rebalance_every=20, initial_cash=10_000.0, max_weight=1.0)
    # Rebalance 1 (fill bar 11, RISE open==close==155) is the only trade that
    # may ever execute: the bad-open sell is refused and the position is
    # carried to the end. No phantom shorts, no negative cash.
    buy_px = 155.0 * 1.0005
    qty = (10_000.0 / buy_px) * ((10_000.0 - 1.0) / 10_000.0)
    expected_costs = qty * (buy_px - 155.0) + 1.0
    assert result.rebalances == 2
    assert result.costs_paid == pytest.approx(expected_costs, rel=1e-9)
    assert result.final_equity == pytest.approx(qty * 110.0, rel=1e-9)  # RISE close at bar 44
    assert result.final_equity > 0


def test_constant_price_universe_is_flat_and_finite():
    bars = {"AAA": bars_from_closes([100.0] * 400), "BBB": bars_from_closes([50.0] * 400)}
    result = run_dual_momentum_backtest(bars, execution=FREE, target_vol=0.15, **BENCH_KW)
    assert result.total_return == 0.0
    assert result.sharpe == 0.0
    assert result.max_drawdown == 0.0
    assert result.final_equity == 25_000.0
    for value in (result.total_return, result.benchmark_return, result.sharpe,
                  result.max_drawdown, result.final_equity, result.costs_paid):
        assert math.isfinite(value)


def test_symbol_ending_early_truncates_the_common_calendar():
    bars = {
        "LONG1": bars_from_closes(rising(400, 100, 0.30)),
        "LONG2": bars_from_closes(rising(400, 100, 0.20)),
        "SHORT": bars_from_closes(rising(300, 100, 0.50)),  # same start, ends at bar 299
    }
    result = run_dual_momentum_backtest(bars, execution=FREE, **BENCH_KW)
    # Aligned length is 300: decisions 70, 91, ..., 280 -> exactly 11 rebalances.
    assert result.rebalances == 11
    assert result.snapshots[-1].decision_index == 280
    assert math.isfinite(result.final_equity) and result.final_equity > 0


def test_tiny_cash_account_never_spends_money_it_does_not_have():
    # $10 account, $1/trade commissions, one rebalance into 3 names, prices
    # frozen after the fill: final equity is a pure accounting identity.
    closes_a = rising(12, 50, 5) + [105] * 18
    closes_b = rising(12, 200, 2) + [222] * 18
    closes_c = rising(12, 80, 1) + [91] * 18
    bars = {"A": bars_from_closes(closes_a), "B": bars_from_closes(closes_b),
            "C": bars_from_closes(closes_c)}
    result = run_dual_momentum_backtest(
        bars, formation_days=10, skip_days=0, top_n=3, rebalance_every=1000,
        initial_cash=10.0, max_weight=1.0)  # default execution: 5bps + $1
    opens = {"A": 105.0, "B": 222.0, "C": 91.0}
    scale = (10.0 - 3.0) / 10.0  # budget minus 3 commissions, over the notional
    expected_final = 0.0
    expected_costs = 3.0
    for open_px in opens.values():
        buy_px = open_px * 1.0005
        qty = (10.0 / 3.0) / buy_px * scale
        expected_final += qty * open_px
        expected_costs += qty * (buy_px - open_px)
    assert result.final_equity == pytest.approx(expected_final, rel=1e-9)
    assert result.costs_paid == pytest.approx(expected_costs, rel=1e-9)
    assert result.final_equity > 0
