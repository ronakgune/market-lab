"""
Thesis-linked paper portfolio: paper-only proposal, sizing/risk gates, exit plans, monitoring plan.

Every approved memo → paper-position candidate goes through deterministic gates:
  - Input eligibility (memo immutable, evidence-addressed, independently reviewed)
  - Committee request validation
  - Portfolio fit check
  - Deterministic sizing with portfolio-level caps
  - Predeclared monitoring + exit plan

All output is paper-only. No live broker path.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional


# ──────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────

class ProposalStatus(Enum):
    DRAFT = "draft"
    COMMITTEE_REQUESTED = "committee_requested"
    PAPER_READY = "paper_ready"
    PAPER_BLOCKED = "paper_blocked"
    REJECTED = "rejected"
    WATCH_ONLY = "watch_only"
    FILLED = "filled"
    CLOSED = "closed"


class PaperAction(Enum):
    REJECT_THESIS_POSITION = "reject_thesis_position"
    PARK_RESEARCH = "park_research"
    APPROVE_THESIS_POSITION = "approve_thesis_position"
    INPUT_BLOCKED = "input_blocked"


# ──────────────────────────────────────────────
# Contracts (frozen dataclasses)
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class MemoRef:
    """Reference to an approved research memo."""
    memo_id: str
    memo_sha256: str
    thesis_summary: str
    security_identity: str
    benchmark: str
    strategy: str
    valuation_range_low: float
    valuation_range_high: float
    catalysts: List[str] = field(default_factory=list)
    invalidations: List[str] = field(default_factory=list)
    principal_risks: List[str] = field(default_factory=list)
    reviewed_at_utc: str = ""


@dataclass(frozen=True)
class PortfolioContext:
    """Portfolio-level state for sizing/risk gates."""
    total_cash: float
    total_paper_exposure: float
    max_position_pct: float = 0.05       # 5% max per position
    max_portfolio_exposure_pct: float = 0.20  # 20% max total
    max_positions: int = 10
    current_position_count: int = 0
    cash_reserve_pct: float = 0.20       # keep 20% cash reserve


@dataclass(frozen=True)
class ThesisPaperProposal:
    """A memo-derived paper-position proposal with sizing and risk gates."""
    proposal_id: str
    memo: MemoRef
    direction: str  # "LONG" — long-only for MVP
    proposed_size: float
    max_size: float
    reason: str
    confidence_at_entry: float
    portfolio_context_snapshot: PortfolioContext
    created_at_utc: str = ""


@dataclass(frozen=True)
class SizingDecision:
    """Deterministic sizing policy output."""
    proposed_notional: Decimal
    max_notional: Decimal
    position_pct: float
    position_cap_pct: float
    portfolio_cap_remaining: float
    approved_notional: Decimal
    overrides_applied: List[str] = field(default_factory=list)
    rejected: bool = False
    reject_reason: str = ""


@dataclass(frozen=True)
class MonitoringPlan:
    """Predeclared position monitoring rules."""
    catalyst_triggers: List[str] = field(default_factory=list)
    invalidation_triggers: List[str] = field(default_factory=list)
    exit_plan: ExitPlan = field(default_factory=lambda: ExitPlan())
    snapshot_frequency: str = "daily"  # daily|weekly|event
    review_frequency_days: int = 30


@dataclass(frozen=True)
class ExitPlan:
    """Predeclared exit rules for a thesis-linked paper position."""
    stop_loss_pct: float = -0.10          # -10% stop
    trailing_stop_pct: float = 0.0        # 0 = disabled
    time_stop_days: int = 0               # 0 = no time stop
    thesis_invalidation_exit: bool = True
    catalyst_target_exit: bool = False
    benchmark_relative_exit: bool = True
    exit_reason: str = ""


@dataclass(frozen=True)
class CommitteePacket:
    """Packet sent to portfolio committee for gate evaluation."""
    proposal: ThesisPaperProposal
    sizing: SizingDecision
    monitoring_plan: MonitoringPlan
    eligibility_checks: Dict[str, bool]
    eligibility_blockers: List[str] = field(default_factory=list)
    committee_action: Optional[PaperAction] = None
    committee_notes: str = ""


@dataclass(frozen=True)
class GateReport:
    """Result of portfolio gate evaluation."""
    passed: bool
    blockers: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    action: Optional[PaperAction] = None


# ──────────────────────────────────────────────
# Deterministic logic
# ──────────────────────────────────────────────

def check_input_eligibility(memo: MemoRef) -> tuple[bool, List[str]]:
    """Check that a memo is eligible for paper-position proposal.

    Returns (passed, blockers).
    """
    blockers: List[str] = []

    if not memo.memo_id:
        blockers.append("memo_id is required")
    if not memo.memo_sha256 or len(memo.memo_sha256) != 64:
        blockers.append("memo_sha256 must be a valid 64-char hex SHA-256")
    if not memo.security_identity:
        blockers.append("security_identity is required")
    if not memo.benchmark:
        blockers.append("benchmark is required")
    if memo.valuation_range_low <= 0 or memo.valuation_range_high <= 0:
        blockers.append("valuation_range must be positive")
    if memo.valuation_range_low > memo.valuation_range_high:
        blockers.append("valuation_range_low must be <= valuation_range_high")
    if not memo.reviewed_at_utc:
        blockers.append("reviewed_at_utc is required (independent review timestamp)")

    return len(blockers) == 0, blockers


def compute_deterministic_sizing(
    portfolio: PortfolioContext,
    price: float,
    conviction: float = 0.5,
) -> SizingDecision:
    """Deterministic size calculation with portfolio-level caps.

    Uses only portfolio state and conviction parameter. NEVER uses model/LLM prose.
    """
    if not (price > 0) or not math.isfinite(price):  # rejects NaN and inf price
        return SizingDecision(
            proposed_notional=Decimal("0"), max_notional=Decimal("0"),
            position_pct=0.0, position_cap_pct=portfolio.max_position_pct,
            portfolio_cap_remaining=0.0, approved_notional=Decimal("0"),
            rejected=True, reject_reason="invalid price",
        )

    # Non-finite portfolio state (NaN/inf cash or exposure) otherwise
    # flows through the caps and comes out as a NaN/Infinity approved
    # notional with rejected=False — the same class fix 5 closed for
    # conviction, from the state side.
    try:
        state_finite = all(math.isfinite(v) for v in (
            portfolio.total_cash, portfolio.total_paper_exposure,
            portfolio.max_position_pct, portfolio.max_portfolio_exposure_pct,
            portfolio.cash_reserve_pct))
    except TypeError:
        state_finite = False
    if not state_finite:
        return SizingDecision(
            proposed_notional=Decimal("0"), max_notional=Decimal("0"),
            position_pct=0.0, position_cap_pct=portfolio.max_position_pct,
            portfolio_cap_remaining=0.0, approved_notional=Decimal("0"),
            rejected=True, reject_reason="portfolio state must be finite",
        )

    # Conviction is documented as a float in [0, 1]; an out-of-range or
    # non-finite value previously produced a negative (or NaN) approved
    # notional with rejected=False. Reject it at the gate.
    if not (0.0 <= conviction <= 1.0):  # also rejects NaN: its comparisons are False
        return SizingDecision(
            proposed_notional=Decimal("0"), max_notional=Decimal("0"),
            position_pct=0.0, position_cap_pct=portfolio.max_position_pct,
            portfolio_cap_remaining=0.0, approved_notional=Decimal("0"),
            rejected=True, reject_reason="conviction outside [0, 1]",
        )

    if portfolio.total_cash <= 0:
        return SizingDecision(
            proposed_notional=Decimal("0"), max_notional=Decimal("0"),
            position_pct=0.0, position_cap_pct=portfolio.max_position_pct,
            portfolio_cap_remaining=0.0, approved_notional=Decimal("0"),
            rejected=True, reject_reason="no cash available",
        )

    if portfolio.current_position_count >= portfolio.max_positions:
        return SizingDecision(
            proposed_notional=Decimal("0"), max_notional=Decimal("0"),
            position_pct=0.0, position_cap_pct=portfolio.max_position_pct,
            portfolio_cap_remaining=0.0, approved_notional=Decimal("0"),
            rejected=True, reject_reason=f"max positions ({portfolio.max_positions}) reached",
        )

    # Position cap: max_position_pct of cash
    max_by_position = portfolio.total_cash * portfolio.max_position_pct

    # Portfolio cap: max_portfolio_exposure_pct minus existing exposure
    max_by_portfolio = (portfolio.total_cash * portfolio.max_portfolio_exposure_pct) - portfolio.total_paper_exposure
    if max_by_portfolio < 0:
        max_by_portfolio = 0.0

    # Cash reserve deduction
    available_cash = portfolio.total_cash * (1.0 - portfolio.cash_reserve_pct)

    # Conviction-weighted sizing (deterministic: conviction is float 0-1)
    conviction_sizing = available_cash * portfolio.max_position_pct * conviction

    proposed = min(conviction_sizing, max_by_position, max_by_portfolio)
    max_allowed = min(max_by_position, max_by_portfolio)

    # Also reject a proposed amount that rounds to zero cents: approving
    # Decimal("0.00") with rejected=False is a non-positive approval in
    # everything but name (a tiny positive conviction does exactly this).
    if proposed <= 0 or round(proposed, 2) <= 0:
        return SizingDecision(
            proposed_notional=Decimal("0"),
            max_notional=Decimal(str(round(max_allowed, 2))),
            position_pct=0.0, position_cap_pct=portfolio.max_position_pct,
            portfolio_cap_remaining=max_by_portfolio,
            approved_notional=Decimal("0"),
            rejected=True, reject_reason="non-positive approved notional",
        )

    overrides: List[str] = []
    if conviction_sizing > max_by_position:
        overrides.append(f"conviction_sizing ({conviction_sizing:.2f}) capped by max_position_pct ({max_by_position:.2f})")
    if conviction_sizing > max_by_portfolio:
        overrides.append(f"conviction_sizing ({conviction_sizing:.2f}) capped by portfolio capacity ({max_by_portfolio:.2f})")

    return SizingDecision(
        proposed_notional=Decimal(str(round(proposed, 2))),
        max_notional=Decimal(str(round(max_allowed, 2))),
        position_pct=(proposed / portfolio.total_cash * 100) if portfolio.total_cash > 0 else 0.0,
        position_cap_pct=portfolio.max_position_pct * 100,
        portfolio_cap_remaining=max_by_portfolio,
        approved_notional=Decimal(str(round(proposed, 2))),
        overrides_applied=overrides,
        rejected=False,
    )


def build_monitoring_plan(memo: MemoRef) -> MonitoringPlan:
    """Build predeclared monitoring plan from memo contents."""
    return MonitoringPlan(
        catalyst_triggers=list(memo.catalysts),
        invalidation_triggers=list(memo.invalidations),
        exit_plan=ExitPlan(
            thesis_invalidation_exit=True,
            benchmark_relative_exit=True,
        ),
        snapshot_frequency="daily",
        review_frequency_days=30,
    )


def evaluate_portfolio_gate(
    proposal: ThesisPaperProposal,
    sizing: SizingDecision,
    plan: MonitoringPlan,
) -> GateReport:
    """Evaluate combined portfolio gate before committee submission.

    Returns GateReport with PASS/blockers.
    """
    blockers: List[str] = []
    warnings: List[str] = []

    if sizing.rejected:
        blockers.append(f"sizing rejected: {sizing.reject_reason}")

    if proposal.proposed_size <= 0:
        blockers.append("proposed_size must be positive")

    if proposal.direction.upper() != "LONG":
        blockers.append(f"direction '{proposal.direction}' not supported (LONG only)")

    if proposal.confidence_at_entry < 0.1:
        warnings.append("confidence_at_entry is very low (< 0.1)")

    if not plan.exit_plan.thesis_invalidation_exit and not plan.exit_plan.benchmark_relative_exit:
        warnings.append("no exit triggers configured (thesis_invalidation and benchmark_relative both disabled)")

    if not plan.catalyst_triggers and not plan.invalidation_triggers:
        warnings.append("no catalyst or invalidation triggers configured")

    return GateReport(
        passed=len(blockers) == 0,
        blockers=blockers,
        warnings=warnings,
        action=PaperAction.APPROVE_THESIS_POSITION if len(blockers) == 0 else PaperAction.REJECT_THESIS_POSITION,
    )


def serialize_proposal(proposal: ThesisPaperProposal) -> str:
    """Canonical JSON serialization."""
    return json.dumps(asdict(proposal), default=str, sort_keys=True, indent=2)


def serialize_sizing(decision: SizingDecision) -> str:
    """Canonical JSON serialization of sizing decision."""
    d = asdict(decision)
    d["proposed_notional"] = str(d["proposed_notional"])
    d["max_notional"] = str(d["max_notional"])
    d["approved_notional"] = str(d["approved_notional"])
    return json.dumps(d, default=str, sort_keys=True, indent=2)