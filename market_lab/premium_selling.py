"""Stream 4 — defined-risk premium selling: strategy machinery.

Implements exactly the rules frozen in
``docs/premium-selling-preregistration.md``. Pure functions over
``options_history`` chain data — no network calls here (fetching is
the data layer's job), no backtest runner here (phase 2's job).

Design invariants, all consequences of the data and the pre-reg:

- Structures are built ONLY from strikes present in that day's
  (sampled) chain. If the registered structure cannot be formed —
  no expiry in the DTE band, no strike in the delta band, no wing
  strike beyond a short, or the $250 risk cap sizes to zero
  contracts — construction returns an explicit ``Skip`` with a
  reason. It never substitutes a near-enough structure.
- Fills are conservative by construction: short legs fill at the
  BID, long legs at the ASK (natural sides; no mid-price fills).
  Closing marks mirror this: a long leg is valued at its bid, a
  short leg costs its ask to close.
- Decisions happen only on chain days and read only that day's
  chain. What happens between chain days is unobservable in this
  data; the excursion check (against SPY daily bars) is the
  registered conservative supplement, not a price.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Optional, Sequence

from .data import Bar
from .options_history import ChainDay, OptionRow

# --- Frozen pre-registration constants (do not tune here) -------------------
BOOK_EQUITY = 25_000.0
MAX_LOSS_PER_POSITION = 250.0        # 1% of the book, per position
AGGREGATE_OPEN_RISK_CAP = 1_250.0    # 5% of the book across all positions
MULTIPLIER = 100.0                   # SPY options: $100 per point
COMMISSION_PER_CONTRACT = 0.65       # per leg, per contract, per side
DTE_MIN, DTE_MAX, DTE_TARGET = 30, 45, 37
DELTA_BAND_LO, DELTA_BAND_HI, DELTA_TARGET = 0.10, 0.16, 0.13
TAKE_PROFIT_FRACTION = 0.50          # close when >=50% of credit captured
STOP_LOSS_MULTIPLE = 2.0             # close when loss >= 2x credit received
HARD_EXIT_DTE = 7                    # exit at first chain day with DTE <= 7
IV_GATE_PERCENTILE = 60.0            # entry only at/above this trailing pctile
IV_TRAILING_CHAIN_DAYS = 60
IV_MIN_PRIOR_OBS = 20
IV_DELTA_WINDOW = (0.10, 0.20)       # |delta| window for the chain IV index


@dataclass(frozen=True)
class Skip:
    reason: str


@dataclass(frozen=True)
class Leg:
    expiration: date
    strike: float
    right: str          # "CALL" / "PUT"
    side: int           # +1 long, -1 short
    entry_price: float  # per-share fill at entry (short=bid, long=ask)


@dataclass(frozen=True)
class Structure:
    kind: str                       # "iron_condor" / "put_credit_spread"
    legs: tuple[Leg, ...]
    credit_per_share: float         # net credit received per share at entry
    max_loss_per_contract: float    # dollars, worst case for 1 contract
    contracts: int
    entry_fees: float               # dollars

    @property
    def max_loss_total(self) -> float:
        return self.max_loss_per_contract * self.contracts

    @property
    def credit_total(self) -> float:
        return self.credit_per_share * MULTIPLIER * self.contracts


def select_expiry(chain: ChainDay, asof: date) -> date | Skip:
    """Expiry nearest the registered target, inside the DTE band only."""
    candidates = []
    for exp in chain.expirations:
        dte = (exp - asof).days
        if DTE_MIN <= dte <= DTE_MAX:
            candidates.append((abs(dte - DTE_TARGET), exp))
    if not candidates:
        return Skip(
            f"no expiry with DTE in [{DTE_MIN},{DTE_MAX}] on {asof} "
            f"(available: {[str(e) for e in chain.expirations]})")
    candidates.sort()
    return candidates[0][1]


def _by_delta(rows: Sequence[OptionRow], right: str) -> Optional[OptionRow]:
    band = [r for r in rows
            if r.right == right and DELTA_BAND_LO <= abs(r.delta) <= DELTA_BAND_HI]
    if not band:
        return None
    return min(band, key=lambda r: abs(abs(r.delta) - DELTA_TARGET))


def _next_strike_beyond(rows: Sequence[OptionRow], right: str,
                        strike: float, direction: int) -> Optional[OptionRow]:
    """Nearest available strike strictly beyond ``strike`` in the
    given direction (-1 below, +1 above) for the same right/expiry."""
    cands = [r for r in rows if r.right == right
             and (r.strike - strike) * direction > 0]
    if not cands:
        return None
    return min(cands, key=lambda r: abs(r.strike - strike))


def _size(credit_per_share: float, width_points: float) -> tuple[int, float]:
    max_loss_per_contract = (width_points - credit_per_share) * MULTIPLIER
    if max_loss_per_contract <= 0:
        return 0, max_loss_per_contract
    contracts = int(MAX_LOSS_PER_POSITION // max_loss_per_contract)
    return contracts, max_loss_per_contract


def _fees(n_legs: int, contracts: int) -> float:
    return n_legs * contracts * COMMISSION_PER_CONTRACT


def build_put_credit_spread(chain: ChainDay, asof: date) -> Structure | Skip:
    exp = select_expiry(chain, asof)
    if isinstance(exp, Skip):
        return exp
    rows = chain.for_expiry(exp)
    short = _by_delta(rows, "PUT")
    if short is None:
        return Skip(f"no PUT strike with |delta| in "
                    f"[{DELTA_BAND_LO},{DELTA_BAND_HI}] in expiry {exp}")
    long_ = _next_strike_beyond(rows, "PUT", short.strike, -1)
    if long_ is None:
        return Skip(f"no available PUT strike below short strike {short.strike} "
                    f"in expiry {exp} to serve as the wing")
    credit = short.bid - long_.ask  # sell at bid, buy at ask — natural sides
    if credit <= 0:
        return Skip(f"non-positive credit {credit:.2f} for put spread in {exp}")
    width = short.strike - long_.strike
    contracts, max_loss = _size(credit, width)
    if contracts < 1:
        return Skip(f"risk cap allows 0 contracts: width {width:.0f}pt gives "
                    f"max loss ${max_loss:,.0f}/contract > ${MAX_LOSS_PER_POSITION:,.0f} cap")
    legs = (
        Leg(exp, short.strike, "PUT", -1, short.bid),
        Leg(exp, long_.strike, "PUT", +1, long_.ask),
    )
    return Structure("put_credit_spread", legs, credit, max_loss,
                     contracts, _fees(len(legs), contracts))


def build_iron_condor(chain: ChainDay, asof: date) -> Structure | Skip:
    exp = select_expiry(chain, asof)
    if isinstance(exp, Skip):
        return exp
    rows = chain.for_expiry(exp)
    short_put = _by_delta(rows, "PUT")
    if short_put is None:
        return Skip(f"no PUT strike with |delta| in "
                    f"[{DELTA_BAND_LO},{DELTA_BAND_HI}] in expiry {exp}")
    short_call = _by_delta(rows, "CALL")
    if short_call is None:
        return Skip(f"no CALL strike with |delta| in "
                    f"[{DELTA_BAND_LO},{DELTA_BAND_HI}] in expiry {exp}")
    long_put = _next_strike_beyond(rows, "PUT", short_put.strike, -1)
    if long_put is None:
        return Skip(f"no available PUT strike below short strike "
                    f"{short_put.strike} in expiry {exp} to serve as the wing")
    long_call = _next_strike_beyond(rows, "CALL", short_call.strike, +1)
    if long_call is None:
        return Skip(f"no available CALL strike above short strike "
                    f"{short_call.strike} in expiry {exp} to serve as the wing")
    credit = (short_put.bid + short_call.bid) - (long_put.ask + long_call.ask)
    if credit <= 0:
        return Skip(f"non-positive credit {credit:.2f} for iron condor in {exp}")
    width = max(short_put.strike - long_put.strike,
                long_call.strike - short_call.strike)
    contracts, max_loss = _size(credit, width)
    if contracts < 1:
        return Skip(f"risk cap allows 0 contracts: widest side {width:.0f}pt gives "
                    f"max loss ${max_loss:,.0f}/contract > ${MAX_LOSS_PER_POSITION:,.0f} cap")
    legs = (
        Leg(exp, short_put.strike, "PUT", -1, short_put.bid),
        Leg(exp, long_put.strike, "PUT", +1, long_put.ask),
        Leg(exp, short_call.strike, "CALL", -1, short_call.bid),
        Leg(exp, long_call.strike, "CALL", +1, long_call.ask),
    )
    return Structure("iron_condor", legs, credit, max_loss,
                     contracts, _fees(len(legs), contracts))


# --- IV gate ------------------------------------------------------------------

def chain_iv_index(chain: ChainDay) -> Optional[float]:
    """Self-contained IV index for one chain day: mean vendor IV over
    the registered target expiry's rows with |delta| in the IV window.
    Uses only that day's chain — no underlying price is needed."""
    exp = select_expiry(chain, chain.day)
    if isinstance(exp, Skip):
        return None
    lo, hi = IV_DELTA_WINDOW
    vals = [r.iv for r in chain.for_expiry(exp) if lo <= abs(r.delta) <= hi]
    if not vals:
        return None
    return sum(vals) / len(vals)


def iv_gate(chain: ChainDay, prior_chains: Sequence[ChainDay]) -> tuple[bool, str]:
    """Entry gate: today's IV index at/above the registered percentile
    of the trailing chain-day indices (strictly prior days only)."""
    today = chain_iv_index(chain)
    if today is None:
        return False, "no IV index computable for today (no target expiry/band rows)"
    history = [v for c in prior_chains
               if c.day < chain.day and (v := chain_iv_index(c)) is not None]
    history = history[-IV_TRAILING_CHAIN_DAYS:]
    if len(history) < IV_MIN_PRIOR_OBS:
        return False, (f"insufficient IV history: {len(history)} prior chain days "
                       f"< {IV_MIN_PRIOR_OBS} required")
    below = sum(1 for v in history if v < today)
    pctile = 100.0 * below / len(history)
    if pctile >= IV_GATE_PERCENTILE:
        return True, f"IV percentile {pctile:.0f} >= {IV_GATE_PERCENTILE:.0f}"
    return False, f"IV percentile {pctile:.0f} < {IV_GATE_PERCENTILE:.0f}"


# --- Positions, marks, management --------------------------------------------

@dataclass
class Position:
    structure: Structure
    entry_date: date

    @property
    def expiration(self) -> date:
        return self.structure.legs[0].expiration

    @property
    def max_loss_total(self) -> float:
        return self.structure.max_loss_total


@dataclass(frozen=True)
class Decision:
    action: str   # "HOLD" / "EXIT"
    reason: str   # "take_profit" / "stop_loss" / "hard_exit_dte" / ...
    close_cost_total: Optional[float] = None  # dollars to close, incl. exit fees


def close_cost_per_share(position: Position, chain: ChainDay) -> Optional[float]:
    """Cost per share to close at this chain's conservative sides:
    buy back shorts at ask, sell longs at bid. None if any leg's
    contract is absent from the chain (no decision possible today)."""
    total = 0.0
    for leg in position.structure.legs:
        row = chain.get(leg.expiration, leg.strike, leg.right)
        if row is None:
            return None
        total += row.ask if leg.side < 0 else -row.bid
    return total


def evaluate_management(position: Position, chain: ChainDay) -> Decision:
    """Apply the registered management rules on one chain day.

    Rule order is registered: hard DTE exit first, then stop, then
    take-profit. A chain day after expiration never reaches here —
    the position is settled, not managed (see ``settle_expired``).
    """
    if chain.day > position.expiration:
        return Decision("EXIT", "past_expiration_settle_separately")
    cost_ps = close_cost_per_share(position, chain)
    if cost_ps is None:
        return Decision("HOLD", "leg_missing_from_chain_no_decision")
    s = position.structure
    fees_exit = _fees(len(s.legs), s.contracts)
    cost_total = cost_ps * MULTIPLIER * s.contracts + fees_exit
    credit = s.credit_total
    dte = (position.expiration - chain.day).days
    if dte <= HARD_EXIT_DTE:
        return Decision("EXIT", "hard_exit_dte", cost_total)
    if cost_total - credit >= STOP_LOSS_MULTIPLE * credit:
        return Decision("EXIT", "stop_loss", cost_total)
    if cost_total <= (1.0 - TAKE_PROFIT_FRACTION) * credit:
        return Decision("EXIT", "take_profit", cost_total)
    return Decision("HOLD", "no_trigger", cost_total)


def settle_expired(position: Position, underlying_close: float) -> float:
    """Realized PnL (dollars) settling an expired position at
    intrinsic value against the underlying's expiration-date close
    (from the provenanced SPY daily bars — the chain data carries no
    underlying price). Includes entry and zero exit fees (expiry
    settlement is not a trade)."""
    payoff_ps = 0.0
    for leg in position.structure.legs:
        if leg.right == "PUT":
            intrinsic = max(leg.strike - underlying_close, 0.0)
        else:
            intrinsic = max(underlying_close - leg.strike, 0.0)
        payoff_ps += leg.side * intrinsic  # long (+1) receives, short (-1) pays
    s = position.structure
    pnl = s.credit_total - s.entry_fees + payoff_ps * MULTIPLIER * s.contracts
    return pnl


def excursion_breach(position: Position, bars: Sequence[Bar]) -> Optional[str]:
    """Registered between-chain-day supplement: if any daily bar since
    entry traded through a short strike (low below a short put strike,
    high above a short call strike), the excursion is recorded. This
    observes that risk was realized intraperiod; it does not invent a
    price — the exit still happens at the next chain day's quotes."""
    for leg in position.structure.legs:
        if leg.side > 0:
            continue
        for bar in bars:
            if bar.date <= position.entry_date or bar.date > position.expiration:
                continue
            if leg.right == "PUT" and bar.low < leg.strike:
                return (f"underlying traded to {bar.low:.2f} below short PUT "
                        f"strike {leg.strike:.0f} on {bar.date}")
            if leg.right == "CALL" and bar.high > leg.strike:
                return (f"underlying traded to {bar.high:.2f} above short CALL "
                        f"strike {leg.strike:.0f} on {bar.date}")
    return None


# --- Book accounting -----------------------------------------------------------

@dataclass(frozen=True)
class ClosedTrade:
    kind: str
    entry_date: date
    exit_date: date
    exit_reason: str
    contracts: int
    credit_total: float
    pnl: float


@dataclass(frozen=True)
class BookSnapshot:
    day: date
    equity: float
    realized_pnl: float
    unrealized_pnl: float
    open_risk: float
    open_positions: int


@dataclass
class PaperBook:
    equity_start: float = BOOK_EQUITY
    positions: list[Position] = field(default_factory=list)
    closed: list[ClosedTrade] = field(default_factory=list)

    @property
    def realized_pnl(self) -> float:
        return sum(t.pnl for t in self.closed)

    @property
    def open_risk(self) -> float:
        return sum(p.max_loss_total for p in self.positions)

    def can_open(self, structure: Structure) -> tuple[bool, str]:
        if self.open_risk + structure.max_loss_total > AGGREGATE_OPEN_RISK_CAP:
            return False, (f"aggregate open-risk cap: {self.open_risk:,.0f} + "
                           f"{structure.max_loss_total:,.0f} > {AGGREGATE_OPEN_RISK_CAP:,.0f}")
        return True, "ok"

    def open(self, structure: Structure, entry_date: date) -> Position:
        ok, why = self.can_open(structure)
        if not ok:
            raise ValueError(why)
        pos = Position(structure, entry_date)
        self.positions.append(pos)
        return pos

    def close(self, position: Position, exit_date: date, reason: str,
              close_cost_total: float) -> ClosedTrade:
        s = position.structure
        pnl = s.credit_total - s.entry_fees - close_cost_total
        trade = ClosedTrade(s.kind, position.entry_date, exit_date, reason,
                            s.contracts, s.credit_total, pnl)
        self.positions.remove(position)
        self.closed.append(trade)
        return trade

    def settle(self, position: Position, underlying_close: float) -> ClosedTrade:
        pnl = settle_expired(position, underlying_close)
        trade = ClosedTrade(position.structure.kind, position.entry_date,
                            position.expiration, "expired_settled",
                            position.structure.contracts,
                            position.structure.credit_total, pnl)
        self.positions.remove(position)
        self.closed.append(trade)
        return trade

    def snapshot(self, chain: ChainDay) -> BookSnapshot:
        """Equity on a chain day. Positions whose legs are missing
        from today's chain are marked at their last knowable state —
        conservatively, at full remaining max loss against equity is
        NOT assumed; instead unrealized is carried at 0 for unmarkable
        positions and the snapshot says so via open_positions count.
        (Marks exist only where the data exists.)"""
        unreal = 0.0
        for pos in self.positions:
            cost_ps = close_cost_per_share(pos, chain)
            if cost_ps is None:
                continue
            s = pos.structure
            fees_exit = _fees(len(s.legs), s.contracts)
            cost_total = cost_ps * MULTIPLIER * s.contracts + fees_exit
            unreal += s.credit_total - s.entry_fees - cost_total
        equity = self.equity_start + self.realized_pnl + unreal
        return BookSnapshot(chain.day, equity, self.realized_pnl, unreal,
                            self.open_risk, len(self.positions))


def max_drawdown(equities: Sequence[float]) -> float:
    """Worst peak-to-trough decline as a positive fraction."""
    peak = -math.inf
    worst = 0.0
    for e in equities:
        peak = max(peak, e)
        if peak > 0:
            worst = max(worst, (peak - e) / peak)
    return worst
