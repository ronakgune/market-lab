"""Adversarial tests for market_lab.ensemble (quorum fail-safe).

Attack stance: the module docstring is the contract — a member counts
as valid ONLY if it returns a Signal with finite confidence and
target weight; below quorum the guard must fail safe to flat, never
crash and never trade on a partial ensemble. Behavior that matches
the docstring is pinned as-is (including the buy/sell threshold
asymmetry inherited from generate_ensemble_signal). Behavior that
contradicts it is asserted against the contract and expected to
fail until fixed; see the pass report.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from market_lab.ensemble import default_min_valid, guarded_ensemble
from market_lab.signals import Signal


def sig(action: str, confidence: float, weight: float = 0.0, **kw) -> Signal:
    fields = dict(
        symbol="SPY", action=action, confidence=confidence, reason="r",
        close=100.0, rsi14=None, sma20=None, sma50=None, volatility20=None,
        strategy="m", target_weight=weight, evidence={},
    )
    fields.update(kw)
    return Signal(**fields)


# Distinctive fallback so field provenance is observable.
FALLBACK = Signal(
    symbol="QQQ", action="HOLD", confidence=0.0, reason="fallback",
    close=555.0, rsi14=11.0, sma20=22.0, sma50=33.0, volatility20=0.44,
    strategy="fallback_strategy", target_weight=0.0, evidence={"origin": "fallback"},
)


def _boom(exc: Exception):
    def member():
        raise exc
    return member


# ============================================================ member validity

def test_member_returning_none_counts_as_failed():
    members = {"ghost": lambda: None, "real": lambda: sig("BUY", 0.9, 0.5)}
    out = guarded_ensemble(members, fallback=FALLBACK, min_valid=2)
    assert out.action == "SELL" and out.target_weight == 0.0
    assert "ghost" in out.reason
    assert out.evidence["quorum_failed"] == "ghost"


def test_member_returning_a_plain_string_fails_safe_instead_of_crashing():
    members = {"str": lambda: "BUY", "real": lambda: sig("BUY", 0.9, 0.5)}
    out = guarded_ensemble(members, fallback=FALLBACK, min_valid=2)
    assert out.action == "SELL"  # string member failed; quorum unmet; flat
    assert out.evidence["quorum_failed"] == "str"


def test_member_returning_signal_impostor_fails_safe_instead_of_crashing():
    # HOLE (see report): the docstring says a member is valid only if
    # it "returns a Signal". The validity check never verifies the
    # type — a duck-typed impostor with finite confidence/weight
    # passes validation, reaches combination, and crashes the guard
    # with AttributeError on primary.rsi14. A fail-safe that crashes
    # on bad member output is not a fail-safe.
    impostor = SimpleNamespace(action="BUY", confidence=0.95, target_weight=0.5)
    members = {"fake": lambda: impostor, "real": lambda: sig("BUY", 0.8, 0.5)}
    out = guarded_ensemble(members, fallback=FALLBACK, min_valid=2)
    assert out.action == "SELL"  # impostor counted failed -> quorum unmet -> flat
    assert out.evidence["quorum_failed"] == "fake"


@pytest.mark.parametrize(
    "bad_sig",
    [
        sig("BUY", float("inf"), 0.5),     # +inf confidence
        sig("BUY", float("-inf"), 0.5),    # -inf confidence
        sig("BUY", 0.9, float("nan")),     # NaN target weight
        sig("BUY", 0.9, float("inf")),     # +inf target weight
        sig("BUY", 0.9, float("-inf")),    # -inf target weight
    ],
    ids=["inf-conf", "-inf-conf", "nan-weight", "inf-weight", "-inf-weight"],
)
def test_nonfinite_member_outputs_count_as_failed(bad_sig):
    members = {"bad": lambda: bad_sig, "real": lambda: sig("BUY", 0.9, 0.5)}
    out = guarded_ensemble(members, fallback=FALLBACK, min_valid=2)
    assert out.action == "SELL"
    assert out.evidence["quorum_failed"] == "bad"


def test_base_exceptions_from_members_are_not_swallowed():
    # The guard catches Exception, not BaseException: a member raising
    # SystemExit must still tear the process down, not be papered over
    # as a "failed vote". Pinned as deliberate Python semantics.
    members = {"fatal": _boom(SystemExit(3)), "real": lambda: sig("BUY", 0.9, 0.5)}
    with pytest.raises(SystemExit):
        guarded_ensemble(members, fallback=FALLBACK, min_valid=1)


def test_every_member_is_called_exactly_once_no_short_circuit():
    calls = {"a": 0, "b": 0, "c": 0, "d": 0}

    def counting(name, s):
        def member():
            calls[name] += 1
            return s
        return member

    members = {
        "a": counting("a", sig("BUY", 0.9, 0.5)),
        "b": counting("b", sig("BUY", 0.9, 0.5)),
        "c": counting("c", sig("BUY", 0.9, 0.5)),
        "d": counting("d", sig("BUY", 0.9, 0.5)),
    }
    guarded_ensemble(members, fallback=FALLBACK, min_valid=1)
    assert calls == {"a": 1, "b": 1, "c": 1, "d": 1}


def test_zero_confidence_member_still_counts_toward_quorum():
    members = {"a": lambda: sig("BUY", 0.0, 0.5), "b": lambda: sig("BUY", 0.0, 0.5)}
    out = guarded_ensemble(members, fallback=FALLBACK, min_valid=2)
    assert out.evidence["quorum_valid"] == 2.0  # validity is finiteness, not conviction
    assert out.action == "HOLD"  # ...but zero scores buy nothing


# ================================================================ quorum edges

@pytest.mark.parametrize(
    "n,expected",
    [(1, 1), (2, 2), (3, 3), (4, 4), (5, 4), (6, 4), (7, 4), (8, 4), (9, 4), (10, 4), (11, 4), (12, 4)],
)
def test_default_min_valid_full_table(n, expected):
    assert default_min_valid(n) == expected


def test_default_min_valid_refuses_nonpositive():
    with pytest.raises(ValueError):
        default_min_valid(-3)


def test_default_quorum_behavior_at_five_members():
    def build(n_valid: int):
        members = {}
        for i in range(5):
            name = f"m{i}"
            members[name] = (lambda s: lambda: s)(sig("BUY", 0.9, 0.5)) if i < n_valid else _boom(RuntimeError("x"))
        return members

    met = guarded_ensemble(build(4), fallback=FALLBACK)      # default need = 4
    assert met.action == "BUY"
    unmet = guarded_ensemble(build(3), fallback=FALLBACK)
    assert unmet.action == "SELL" and unmet.evidence["quorum_needed"] == 4.0


def test_min_valid_negative_refused():
    with pytest.raises(ValueError):
        guarded_ensemble({"a": lambda: sig("BUY", 0.5)}, fallback=FALLBACK, min_valid=-1)


def test_min_valid_one_combines_from_a_single_survivor():
    members = {
        "only": lambda: sig("BUY", 0.8, 0.6),
        "dead1": _boom(RuntimeError("x")),
        "dead2": _boom(RuntimeError("y")),
    }
    out = guarded_ensemble(members, fallback=FALLBACK, min_valid=1)
    assert out.action == "BUY" and out.target_weight == pytest.approx(0.6)
    assert out.confidence == pytest.approx(1.0)  # min(1, 0.8/1 + 0.35)


# ======================================================= combination boundaries

def test_buy_threshold_exactly_070_buys():
    out = guarded_ensemble({"a": lambda: sig("BUY", 0.70, 0.5)}, fallback=FALLBACK, min_valid=1)
    assert out.action == "BUY"


def test_buy_just_below_threshold_holds():
    out = guarded_ensemble({"a": lambda: sig("BUY", 0.6999, 0.5)}, fallback=FALLBACK, min_valid=1)
    assert out.action == "HOLD" and out.target_weight == 0.0


def test_sell_threshold_exactly_045_sells():
    out = guarded_ensemble({"a": lambda: sig("SELL", 0.45)}, fallback=FALLBACK, min_valid=1)
    assert out.action == "SELL" and out.target_weight == 0.0


def test_sell_just_below_threshold_holds():
    out = guarded_ensemble({"a": lambda: sig("SELL", 0.4499)}, fallback=FALLBACK, min_valid=1)
    assert out.action == "HOLD"


def test_buy_sell_tie_at_high_scores_goes_to_sell_PINNED():
    # Pinned asymmetry, inherited from generate_ensemble_signal: BUY
    # must be strictly greater than SELL; SELL wins ties (>=).
    members = {"b": lambda: sig("BUY", 0.8, 0.9), "s": lambda: sig("SELL", 0.8)}
    out = guarded_ensemble(members, fallback=FALLBACK, min_valid=2)
    assert out.action == "SELL" and out.target_weight == 0.0


def test_buy_wins_when_strictly_ahead():
    members = {"b": lambda: sig("BUY", 0.71, 0.5), "s": lambda: sig("SELL", 0.70)}
    out = guarded_ensemble(members, fallback=FALLBACK, min_valid=2)
    assert out.action == "BUY"


def test_sell_member_weight_cannot_leak_into_buy_target():
    members = {
        "b1": lambda: sig("BUY", 0.5, 0.3),
        "b2": lambda: sig("BUY", 0.4, 0.9),
        "s1": lambda: sig("SELL", 0.1, 0.99),  # decoy: biggest weight, wrong side
    }
    out = guarded_ensemble(members, fallback=FALLBACK, min_valid=3)
    assert out.action == "BUY" and out.target_weight == pytest.approx(0.9)


def test_combined_confidence_formulas():
    buy = guarded_ensemble(
        {"a": lambda: sig("BUY", 0.9, 0.5), "b": lambda: sig("HOLD", 0.0),
         "c": lambda: sig("HOLD", 0.0), "d": lambda: sig("HOLD", 0.0)},
        fallback=FALLBACK)
    assert buy.action == "BUY"
    assert buy.confidence == pytest.approx(0.9 / 4 + 0.35)  # family = ALL valid members

    sell = guarded_ensemble(
        {"a": lambda: sig("SELL", 0.3), "b": lambda: sig("SELL", 0.3)},
        fallback=FALLBACK, min_valid=2)
    assert sell.action == "SELL"
    assert sell.confidence == pytest.approx(0.6 / 2 + 0.25)

    hold = guarded_ensemble(
        {"a": lambda: sig("BUY", 0.3, 0.5), "b": lambda: sig("SELL", 0.2)},
        fallback=FALLBACK, min_valid=2)
    assert hold.action == "HOLD"
    assert hold.confidence == pytest.approx(0.3 / 2)  # no bonus on HOLD


def test_hold_members_dilute_buy_confidence_denominator():
    members = {
        "b": lambda: sig("BUY", 0.7, 0.5),
        "h1": lambda: sig("HOLD", 0.0), "h2": lambda: sig("HOLD", 0.0), "h3": lambda: sig("HOLD", 0.0),
    }
    out = guarded_ensemble(members, fallback=FALLBACK)  # default quorum 4, all valid
    assert out.action == "BUY"
    assert out.confidence == pytest.approx(0.7 / 4 + 0.35)


# ============================================================ output provenance

def test_combined_signal_field_provenance():
    primary = sig("BUY", 0.9, 0.5, rsi14=42.0, sma20=101.0, sma50=102.0, volatility20=0.03)
    members = {"p": lambda: primary, "q": lambda: sig("BUY", 0.2, 0.4)}
    out = guarded_ensemble(members, fallback=FALLBACK, min_valid=2)
    assert out.symbol == "QQQ" and out.close == 555.0        # identity from fallback
    assert out.rsi14 == 42.0 and out.sma20 == 101.0          # indicators from primary
    assert out.sma50 == 102.0 and out.volatility20 == 0.03
    assert out.strategy == "guarded_ensemble"


def test_fail_safe_signal_field_provenance_and_evidence():
    members = {"a": _boom(RuntimeError("x")), "b": _boom(ValueError("y"))}
    out = guarded_ensemble(members, fallback=FALLBACK, min_valid=1)
    assert out.action == "SELL" and out.target_weight == 0.0
    assert (out.symbol, out.close, out.rsi14, out.sma20) == ("QQQ", 555.0, 11.0, 22.0)
    assert out.strategy == "guarded_ensemble"
    assert out.evidence["origin"] == "fallback"              # fallback evidence survives
    assert out.evidence["quorum_valid"] == 0.0
    assert out.evidence["quorum_failed"] == "a,b"
    assert "buy_score" not in out.evidence                   # scores only on combined path
    assert "a" in out.reason and "b" in out.reason


def test_combined_evidence_carries_scores_and_quorum():
    members = {"a": lambda: sig("BUY", 0.8, 0.5), "b": lambda: sig("SELL", 0.1)}
    out = guarded_ensemble(members, fallback=FALLBACK, min_valid=2)
    assert out.evidence["buy_score"] == pytest.approx(0.8)
    assert out.evidence["sell_score"] == pytest.approx(0.1)
    assert out.evidence["quorum_valid"] == 2.0
    assert out.evidence["quorum_failed"] is None


# ============================================ trusted-output gaps (pinned, see report)

def test_out_of_range_member_weights_propagate_unchecked_PINNED():
    # PINNED + flagged in the report: validity is finiteness only
    # (per the docstring), so a member's target_weight of 2.5 — or
    # even a NEGATIVE weight on a BUY — flows straight into the
    # combined signal. Downstream gates are the only clamp today.
    big = guarded_ensemble(
        {"a": lambda: sig("BUY", 0.9, 2.5), "b": lambda: sig("BUY", 0.1, 0.1)},
        fallback=FALLBACK, min_valid=2)
    assert big.action == "BUY" and big.target_weight == pytest.approx(2.5)

    negative = guarded_ensemble(
        {"a": lambda: sig("BUY", 0.9, -0.5), "b": lambda: sig("HOLD", 0.1)},
        fallback=FALLBACK, min_valid=2)
    assert negative.action == "BUY" and negative.target_weight == pytest.approx(-0.5)


def test_unknown_action_member_is_valid_voteless_and_can_be_primary_PINNED():
    # PINNED + flagged: Signal never validates action strings, and the
    # guard doesn't either. A member voting "FLATTEN" counts toward
    # quorum, contributes to neither score, and — as the highest-
    # confidence member — supplies the combined signal's indicators.
    weird = sig("FLATTEN", 0.95, 0.0, rsi14=42.0)
    members = {"w": lambda: weird, "b": lambda: sig("BUY", 0.8, 0.5)}
    out = guarded_ensemble(members, fallback=FALLBACK, min_valid=2)
    assert out.evidence["quorum_valid"] == 2.0
    assert out.action == "BUY"           # decided by the one real vote
    assert out.rsi14 == 42.0             # ...but indicators come from the non-vote
