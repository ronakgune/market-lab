from __future__ import annotations

import json
import math
import os
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Literal

from .broker import Portfolio, _portfolio_numbers_finite
from .config import OPTIONS_CHAIN_DIR, OPTIONS_PAPER_CANDIDATES_PATH, OPTIONS_PAPER_LEDGER_PATH, OPTIONS_PAPER_STATE_PATH, OptionsRiskConfig, OPTIONS_RISK, ensure_dirs
from .options_data import OptionContract


@contextmanager
def _portfolio_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("w") as lock_file:
        locked = False
        try:
            try:
                import fcntl  # type: ignore
            except ModuleNotFoundError:
                fcntl = None  # type: ignore[assignment]
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                locked = True
            yield
        finally:
            if locked:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)  # type: ignore[union-attr]
                except Exception:
                    pass


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as tmp:
        tmp.write(text)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)

OptionAction = Literal["BUY_TO_OPEN", "SELL_TO_OPEN", "BUY_TO_CLOSE", "SELL_TO_CLOSE"]


@dataclass(frozen=True)
class OptionPaperCandidate:
    action: OptionAction
    contract: OptionContract
    contracts: int
    strategy: str
    reference_price: float
    signal_as_of: str


@dataclass(frozen=True)
class OptionPaperOrder:
    action: OptionAction
    contract: OptionContract
    contracts: int
    price: float
    strategy: str
    as_of: str | None = None


@dataclass(frozen=True)
class OptionPaperDecision:
    accepted: bool
    action: OptionAction
    contract_id: str
    contracts: int
    price: float
    premium: float
    reason: str
    timestamp: str
    strategy: str


@dataclass
class OptionPaperPortfolio:
    cash: float = 100_000.0
    positions: dict[str, int] = field(default_factory=dict)
    avg_price: dict[str, float] = field(default_factory=dict)
    reserved_cash: float = 0.0
    reserved_shares: dict[str, int] = field(default_factory=dict)

    @property
    def available_cash(self) -> float:
        return self.cash - self.reserved_cash


def _reject(order: OptionPaperOrder, reason: str) -> OptionPaperDecision:
    return OptionPaperDecision(False, order.action, order.contract.contract_id, order.contracts, order.price, 0.0, reason, datetime.now(timezone.utc).isoformat(), order.strategy)


def _accept(order: OptionPaperOrder, premium: float, reason: str) -> OptionPaperDecision:
    return OptionPaperDecision(True, order.action, order.contract.contract_id, order.contracts, order.price, premium, reason, datetime.now(timezone.utc).isoformat(), order.strategy)


def _opening_guardrail_violation(paper: OptionPaperPortfolio, order: OptionPaperOrder, risk: OptionsRiskConfig) -> str | None:
    contract = order.contract
    q = contract.quote
    as_of = order.as_of or date.today().isoformat()
    dte = contract.dte(as_of)
    if not (risk.min_dte <= dte <= risk.max_dte):
        return "option DTE outside configured paper gate"
    if q.bid <= 0 or q.ask < q.bid or q.spread_pct > risk.max_bid_ask_spread_pct or q.open_interest < risk.min_open_interest or q.volume < risk.min_volume:
        return "option liquidity/spread outside configured paper gate"
    if contract.greeks.degenerate:
        return "option greeks are degenerate (low-quality estimate); not eligible for paper gate"
    if order.action == "SELL_TO_OPEN" and contract.option_type == "CALL" and abs(contract.greeks.delta) > risk.max_abs_short_call_delta:
        return "short call delta exceeds configured paper gate"
    if order.action == "SELL_TO_OPEN" and contract.option_type == "PUT" and abs(contract.greeks.delta) > risk.max_abs_short_put_delta:
        return "short put delta exceeds configured paper gate"
    existing_for_underlying = sum(abs(qty) for cid, qty in paper.positions.items() if cid.startswith(f"{contract.underlying.upper()}-") and qty != 0)
    if existing_for_underlying + order.contracts > risk.max_contracts_per_symbol:
        return "contract count exceeds per-symbol gate including existing positions"
    return None


def _reject_json_constant(value: str) -> None:
    """json parse_constant hook: NaN/Infinity tokens in stored JSON are
    corrupt state, not numbers (see broker._reject_json_constant)."""
    raise ValueError(f"non-finite JSON constant in stored data: {value}")


def _order_numbers_finite(order: OptionPaperOrder) -> bool:
    c = order.contract
    values = [order.price, c.strike, c.quote.bid, c.quote.ask, c.quote.mid,
              c.quote.volume, c.quote.open_interest,
              c.greeks.delta, c.greeks.gamma, c.greeks.theta, c.greeks.vega,
              c.greeks.implied_volatility]
    try:
        return all(math.isfinite(v) for v in values)
    except TypeError:
        return False


def _paper_numbers_finite(paper: OptionPaperPortfolio) -> bool:
    """State-side counterpart of _order_numbers_finite: NaN/inf in paper
    state defeats the premium/reserve/assignment gates from the other
    side. The equity portfolio is checked with broker's equivalent."""
    try:
        values = [paper.cash, paper.reserved_cash,
                  *paper.positions.values(), *paper.avg_price.values(),
                  *paper.reserved_shares.values()]
        return all(math.isfinite(v) for v in values)
    except TypeError:
        return False


def evaluate_option_paper_order(paper: OptionPaperPortfolio, equity_portfolio: Portfolio, order: OptionPaperOrder, risk: OptionsRiskConfig = OPTIONS_RISK) -> OptionPaperDecision:
    if not risk.allow_options or not risk.paper_options_enabled or risk.live_options_enabled:
        return _reject(order, "paper options are disabled or live options unexpectedly enabled")
    # Finiteness gate FIRST (audit run-1, finding 2): NaN in price /
    # strike / quote / greeks makes every guardrail comparison False, so
    # premium, reserve and assignment gates would not bind and NaN would
    # be persisted into paper state. Reject before any comparison.
    if not _order_numbers_finite(order):
        return _reject(order, "order, strike, quote and greeks must be finite")
    if not _paper_numbers_finite(paper) or not _portfolio_numbers_finite(equity_portfolio):
        return _reject(order, "portfolio state must be finite")
    if order.contracts <= 0 or order.price <= 0:
        return _reject(order, "contracts and price must be positive")
    if order.contracts > risk.max_contracts_per_symbol:
        return _reject(order, "contract count exceeds per-symbol gate")
    contract = order.contract
    premium = order.price * 100 * order.contracts
    cid = contract.contract_id

    if order.action in ("BUY_TO_OPEN", "SELL_TO_OPEN"):
        existing_same_contract = paper.positions.get(cid, 0)
        if order.action == "BUY_TO_OPEN" and existing_same_contract < 0:
            return _reject(order, "use BUY_TO_CLOSE to reduce an existing short option position")
        if order.action == "SELL_TO_OPEN" and existing_same_contract > 0:
            return _reject(order, "use SELL_TO_CLOSE to reduce an existing long option position")
        violation = _opening_guardrail_violation(paper, order, risk)
        if violation:
            return _reject(order, violation)

    if order.action == "BUY_TO_OPEN":
        account_equity = max(equity_portfolio.equity({contract.underlying: contract.strike}), paper.cash, 1.0)
        if premium > account_equity * risk.max_option_premium_pct:
            return _reject(order, "long option premium exceeds max premium risk gate")
        if premium > paper.available_cash:
            return _reject(order, "insufficient available cash for long option premium")
        paper.cash -= premium
        paper.positions[cid] = paper.positions.get(cid, 0) + order.contracts
        paper.avg_price[cid] = order.price
        return _accept(order, premium, "accepted paper long option; max loss limited to premium")

    if order.action == "SELL_TO_OPEN":
        if contract.option_type == "CALL":
            if risk.allow_naked_calls:
                return _reject(order, "naked calls remain blocked in paper gate")
            owned = equity_portfolio.positions.get(contract.underlying.upper())
            owned_shares = owned.quantity if owned else 0
            already_reserved = paper.reserved_shares.get(contract.underlying.upper(), 0)
            required = 100 * order.contracts
            if owned_shares - already_reserved < required:
                return _reject(order, "covered shares unavailable; naked calls blocked")
            paper.reserved_shares[contract.underlying.upper()] = already_reserved + required
        else:
            reserve = contract.strike * 100 * order.contracts
            if reserve > paper.available_cash:
                return _reject(order, "insufficient available cash for reserved cash-secured put")
            projected_reserved = paper.reserved_cash + reserve
            assignment_base = max(equity_portfolio.equity({contract.underlying: contract.strike}), paper.cash, 1.0)
            if reserve > assignment_base * risk.max_assignment_notional_pct:
                return _reject(order, "single cash-secured put reserve exceeds per-trade assignment gate")
            if projected_reserved > assignment_base * risk.max_total_options_assignment_pct:
                return _reject(order, "total options assignment reserve exceeds portfolio gate")
            paper.reserved_cash = projected_reserved
        paper.cash += premium
        paper.positions[cid] = paper.positions.get(cid, 0) - order.contracts
        paper.avg_price[cid] = order.price
        return _accept(order, premium, "accepted defined-risk paper short option with collateral reserved")

    current = paper.positions.get(cid, 0)
    if order.action == "SELL_TO_CLOSE":
        if current < order.contracts:
            return _reject(order, "cannot sell to close more long contracts than held")
        paper.cash += premium
        paper.positions[cid] = current - order.contracts
        if paper.positions[cid] == 0:
            paper.positions.pop(cid, None)
        return _accept(order, premium, "accepted paper close of long option")

    if order.action == "BUY_TO_CLOSE":
        if current > -order.contracts:
            return _reject(order, "cannot buy to close more short contracts than open")
        if contract.option_type == "CALL":
            collateral_release = 100 * order.contracts
            cash_available_for_close = paper.available_cash
        else:
            collateral_release = contract.strike * 100 * order.contracts
            cash_available_for_close = paper.available_cash + collateral_release
        if premium > cash_available_for_close:
            return _reject(order, "insufficient cash to buy short option closed")
        paper.cash -= premium
        paper.positions[cid] = current + order.contracts
        if contract.option_type == "CALL":
            paper.reserved_shares[contract.underlying.upper()] = max(0, paper.reserved_shares.get(contract.underlying.upper(), 0) - 100 * order.contracts)
        else:
            paper.reserved_cash = max(0.0, paper.reserved_cash - contract.strike * 100 * order.contracts)
        if paper.positions[cid] == 0:
            paper.positions.pop(cid, None)
        return _accept(order, premium, "accepted paper close of short option")

    return _reject(order, "unsupported paper option action")


def save_option_paper_portfolio(portfolio: OptionPaperPortfolio, path: Path = OPTIONS_PAPER_STATE_PATH) -> None:
    ensure_dirs()
    with _portfolio_lock(path):
        _atomic_write_text(path, json.dumps(asdict(portfolio), indent=2, sort_keys=True, allow_nan=False))


def load_option_paper_portfolio(path: Path = OPTIONS_PAPER_STATE_PATH) -> OptionPaperPortfolio:
    if not path.exists():
        return OptionPaperPortfolio()
    try:
        data = json.loads(path.read_text(), parse_constant=_reject_json_constant)
        # Validate BEFORE int()/float() coercion: 1e999 parses to inf
        # without tripping parse_constant, and int(inf) raises
        # OverflowError (not caught below) while float(inf) loads
        # silently and defeats the cash/reserve gates.
        raw_numbers = [data.get("cash", 100_000.0), data.get("reserved_cash", 0.0),
                       *data.get("positions", {}).values(),
                       *data.get("avg_price", {}).values(),
                       *data.get("reserved_shares", {}).values()]
        if not all(math.isfinite(v) for v in raw_numbers):
            raise ValueError("option paper state contains non-finite values")
        return OptionPaperPortfolio(
            cash=float(data.get("cash", 100_000.0)),
            positions={k: int(v) for k, v in data.get("positions", {}).items()},
            avg_price={k: float(v) for k, v in data.get("avg_price", {}).items()},
            reserved_cash=float(data.get("reserved_cash", 0.0)),
            reserved_shares={k: int(v) for k, v in data.get("reserved_shares", {}).items()},
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        return OptionPaperPortfolio()


def append_option_paper_ledger(decision: OptionPaperDecision, path: Path = OPTIONS_PAPER_LEDGER_PATH) -> None:
    ensure_dirs()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        # Append-only audit record of attempts (incl. rejected non-finite
        # orders): must never raise on a rejection; never parsed into a
        # risk gate. Strictness lives in the state/candidate writers.
        f.write(json.dumps(asdict(decision), sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _parse_contract_id(contract_id: str) -> dict[str, str | float]:
    """Parse canonical contract_id from the right so hyphenated underlyings survive."""
    parts = contract_id.split("-")
    if len(parts) < 6:
        underlying = parts[0] if parts else ""
        option_marker = parts[-2] if len(parts) >= 2 else ""
        option_type = "CALL" if option_marker == "C" else "PUT"
        strike = float(parts[-1]) if parts else 0.0
        expiration = "-".join(parts[1:-2]) if len(parts) > 3 else ""
        return {"underlying": underlying, "expiration": expiration, "option_type": option_type, "strike": strike}
    strike = float(parts[-1])
    option_marker = parts[-2]
    option_type = "CALL" if option_marker == "C" else "PUT"
    expiration = "-".join(parts[-5:-2])
    underlying = "-".join(parts[:-5])
    return {"underlying": underlying, "expiration": expiration, "option_type": option_type, "strike": strike}


def save_option_paper_candidates(
    candidates: list[OptionPaperCandidate],
    path: Path = OPTIONS_PAPER_CANDIDATES_PATH,
) -> None:
    ensure_dirs()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(asdict(candidate), sort_keys=True, allow_nan=False) for candidate in candidates]
    _atomic_write_text(path, ("\n".join(lines) + "\n") if lines else "")


def _candidate_from_record(record: dict) -> OptionPaperCandidate:
    from .options_data import _contract_from_record

    contract_data = record.get("contract", {})
    contract = _contract_from_record(contract_data, record.get("signal_as_of", ""))
    reference_price = float(record.get("reference_price", contract.quote.mid))
    if not math.isfinite(reference_price):
        raise ValueError("candidate reference_price must be finite")
    return OptionPaperCandidate(
        action=record["action"],
        contract=contract,
        contracts=int(record.get("contracts", 0)),
        strategy=record.get("strategy", "paper_options"),
        reference_price=reference_price,
        signal_as_of=record.get("signal_as_of", ""),
    )


def load_option_paper_candidates(path: Path = OPTIONS_PAPER_CANDIDATES_PATH) -> list[OptionPaperCandidate]:
    if not path.exists():
        return []
    candidates: list[OptionPaperCandidate] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            candidates.append(_candidate_from_record(json.loads(line, parse_constant=_reject_json_constant)))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return candidates


def execute_option_paper_candidate(
    candidate: OptionPaperCandidate,
    paper: OptionPaperPortfolio,
    equity_portfolio: Portfolio,
    risk: OptionsRiskConfig = OPTIONS_RISK,
) -> OptionPaperDecision:
    order = OptionPaperOrder(
        candidate.action,
        candidate.contract,
        candidate.contracts,
        candidate.contract.quote.mid or candidate.reference_price,
        candidate.strategy,
        candidate.signal_as_of,
    )
    return evaluate_option_paper_order(paper, equity_portfolio, order, risk)


@dataclass
class OptionPositionView:
    """Human-readable enriched view of a single active paper option position."""
    contract_id: str
    underlying: str
    expiration: str
    option_type: str
    strike: float
    contracts: int
    avg_price: float
    side: str
    mark: float
    unrealized_pnl: float


def build_option_positions_view(
    paper: OptionPaperPortfolio,
    chain_dir: Path = OPTIONS_CHAIN_DIR,
) -> list[OptionPositionView]:
    """Build enriched position views for each non-zero paper option position.

    Prices quotes from cached option chains if available; falls back to avg_price as mark.
    """
    from .options_data import load_option_chain_snapshot
    views: list[OptionPositionView] = []
    quotes: dict[str, float] = {}
    for cid, qty in paper.positions.items():
        if qty == 0:
            continue
        parsed = _parse_contract_id(cid)
        underlying = parsed.get("underlying", "")
        if underlying and (cid not in quotes):
            try:
                chain = load_option_chain_snapshot(underlying, chain_dir)
                for contract in chain.contracts:
                    quotes[contract.contract_id] = contract.quote.mid
            except (OSError, FileNotFoundError, json.JSONDecodeError, TypeError, ValueError, KeyError):
                pass
        avg = paper.avg_price.get(cid, 0.0)
        mark = quotes.get(cid, avg)
        unrealized = (mark - avg) * qty * 100
        views.append(
            OptionPositionView(
                contract_id=cid,
                underlying=str(underlying),
                expiration=str(parsed.get("expiration", "")),
                option_type=str(parsed.get("option_type", "")),
                strike=float(parsed.get("strike", 0.0)),
                contracts=abs(qty),
                avg_price=avg,
                side="LONG" if qty > 0 else "SHORT",
                mark=mark,
                unrealized_pnl=round(unrealized, 2),
            )
        )
    return views
