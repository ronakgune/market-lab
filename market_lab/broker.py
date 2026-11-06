from __future__ import annotations

import json
import math
import os
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from .config import LEDGER_PATH, PENDING_CANDIDATES_PATH, STATE_PATH, RiskConfig, RISK, ensure_dirs

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

Side = Literal["BUY", "SELL"]

@dataclass
class Position:
    symbol: str
    quantity: int = 0
    avg_price: float = 0.0

@dataclass
class Portfolio:
    cash: float = RISK.starting_cash
    positions: dict[str, Position] = field(default_factory=dict)

    def market_value(self, prices: dict[str, float]) -> float:
        return sum(pos.quantity * prices.get(sym, pos.avg_price) for sym, pos in self.positions.items())

    def equity(self, prices: dict[str, float]) -> float:
        return self.cash + self.market_value(prices)

@dataclass(frozen=True)
class OrderDecision:
    accepted: bool
    side: Side
    symbol: str
    quantity: int
    requested_price: float
    fill_price: float | None
    reason: str
    timestamp: str
    strategy: str = "unknown"
    signal_date: str | None = None
    execution_date: str | None = None

@dataclass(frozen=True)
class OrderCandidate:
    side: Side
    symbol: str
    quantity: int
    strategy: str
    confidence: float
    reason: str
    signal_date: str
    reference_close: float
    intended_execution: str = "next_open"

def position_market_value(portfolio: Portfolio, symbol: str, prices: dict[str, float]) -> float:
    pos = portfolio.positions.get(symbol.upper())
    if not pos:
        return 0.0
    return pos.quantity * prices.get(pos.symbol, pos.avg_price)

def _reject_json_constant(value: str) -> None:
    """json parse_constant hook: NaN/Infinity tokens in stored JSON are
    corrupt state, not numbers. Raising ValueError routes loaders into
    their existing corrupt-state fallback instead of parsing a float
    that silently defeats every comparison-based risk gate."""
    raise ValueError(f"non-finite JSON constant in stored data: {value}")


def _portfolio_numbers_finite(portfolio: Portfolio) -> bool:
    """True when every number a risk gate reads from portfolio state is
    finite. parse_constant only stops the NaN/Infinity *tokens*; a value
    like 1e999 is valid JSON syntax that float() parses to inf, and a
    hand-built Portfolio can carry NaN directly — both defeat the same
    comparison gates, so state is validated wherever it is consumed."""
    try:
        if not math.isfinite(portfolio.cash):
            return False
        return all(
            math.isfinite(p.quantity) and math.isfinite(p.avg_price)
            for p in portfolio.positions.values()
        )
    except TypeError:
        return False


def load_portfolio(path: Path = STATE_PATH) -> Portfolio:
    if not path.exists():
        return Portfolio()
    try:
        data=json.loads(path.read_text(), parse_constant=_reject_json_constant)
        positions={sym: Position(**p) for sym,p in data.get("positions", {}).items()}
        portfolio = Portfolio(cash=float(data.get("cash", RISK.starting_cash)), positions=positions)
        if not _portfolio_numbers_finite(portfolio):
            raise ValueError("portfolio state contains non-finite values")
        return portfolio
    except (json.JSONDecodeError, TypeError, ValueError):
        return Portfolio()

def save_portfolio(portfolio: Portfolio, path: Path = STATE_PATH) -> None:
    ensure_dirs()
    data={"cash": portfolio.cash, "positions": {k: asdict(v) for k,v in portfolio.positions.items() if v.quantity != 0}}
    _atomic_write_text(path, json.dumps(data, indent=2, sort_keys=True, allow_nan=False))

def append_ledger(decision: OrderDecision, path: Path = LEDGER_PATH) -> None:
    ensure_dirs()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        # The ledger is an append-only audit record of what was ATTEMPTED,
        # including rejected non-finite orders — it must never raise on a
        # rejected decision, and it is never parsed back into a risk gate.
        # State writers (save_portfolio, candidate writers) are the strict
        # allow_nan=False boundary; this log is not.
        f.write(json.dumps(asdict(decision), sort_keys=True) + "\n")

def append_order_candidates(candidates: list[OrderCandidate], path: Path = PENDING_CANDIDATES_PATH) -> None:
    ensure_dirs()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for candidate in candidates:
            f.write(json.dumps(asdict(candidate), sort_keys=True, allow_nan=False) + "\n")

def load_order_candidates(path: Path = PENDING_CANDIDATES_PATH) -> list[OrderCandidate]:
    if not path.exists():
        return []
    candidates=[]
    try:
        with path.open() as f:
            for line in f:
                line=line.strip()
                if line:
                    candidate = OrderCandidate(**json.loads(line, parse_constant=_reject_json_constant))
                    # 1e999-style values parse to inf without tripping
                    # parse_constant; a candidate carrying one is corrupt.
                    try:
                        finite = all(math.isfinite(v) for v in (
                            candidate.quantity, candidate.confidence,
                            candidate.reference_close))
                    except TypeError:
                        finite = False
                    if not finite:
                        raise ValueError("order candidate contains non-finite values")
                    candidates.append(candidate)
    except (json.JSONDecodeError, TypeError, ValueError):
        return []
    return candidates

def save_order_candidates(candidates: list[OrderCandidate], path: Path = PENDING_CANDIDATES_PATH) -> None:
    ensure_dirs()
    if not candidates:
        _atomic_write_text(path, "")
        return
    text = "".join(json.dumps(asdict(candidate), sort_keys=True, allow_nan=False) + "\n" for candidate in candidates)
    _atomic_write_text(path, text)

def candidate_to_order_at_open(candidate: OrderCandidate, next_open: float, prices: dict[str, float], portfolio_path: Path = STATE_PATH, ledger_path: Path = LEDGER_PATH, execution_date: str | None = None) -> OrderDecision:
    return place_mock_order(
        candidate.side,
        candidate.symbol,
        candidate.quantity,
        next_open,
        prices,
        portfolio_path,
        ledger_path,
        strategy=candidate.strategy,
        signal_date=candidate.signal_date,
        execution_date=execution_date,
    )

def evaluate_order(portfolio: Portfolio, side: Side, symbol: str, quantity: int, price: float, prices: dict[str, float], risk: RiskConfig = RISK) -> OrderDecision:
    now=datetime.now(timezone.utc).isoformat()
    symbol=symbol.upper()
    if risk.live_trading_enabled:
        return OrderDecision(False, side, symbol, quantity, price, None, "live trading flag unexpectedly enabled; mock broker refuses", now)
    # Finiteness gate FIRST (audit run-1, finding 1): every comparison
    # below is False for NaN, so a non-finite quantity/price would sail
    # through positivity, notional, cash and position gates and then be
    # persisted into portfolio state. Reject before any comparison.
    try:
        finite = math.isfinite(quantity) and math.isfinite(price)
    except TypeError:
        finite = False
    if not finite:
        return OrderDecision(False, side, symbol, quantity, price, None, "quantity and price must be finite", now)
    # The order is only half the input: NaN/inf in portfolio state (cash,
    # a position's quantity/avg_price) defeats the same gates from the
    # other side and persists poisoned state on acceptance.
    if not _portfolio_numbers_finite(portfolio):
        return OrderDecision(False, side, symbol, quantity, price, None, "portfolio state must be finite", now)
    if symbol in prices:
        try:
            if not math.isfinite(prices[symbol]):
                return OrderDecision(False, side, symbol, quantity, price, None, "reference price must be finite", now)
        except TypeError:
            return OrderDecision(False, side, symbol, quantity, price, None, "reference price must be finite", now)
    if quantity <= 0 or price <= 0:
        return OrderDecision(False, side, symbol, quantity, price, None, "quantity and price must be positive", now)
    if side not in ("BUY", "SELL"):
        return OrderDecision(False, side, symbol, quantity, price, None, "unsupported side", now)
    pos=portfolio.positions.get(symbol, Position(symbol))
    if side == "SELL" and pos.quantity < quantity:
        if risk.allow_short:
            return OrderDecision(False, side, symbol, quantity, price, None, "shorting is unsupported in MVP mock broker", now)
        return OrderDecision(False, side, symbol, quantity, price, None, "sell rejected: no shorting and insufficient shares", now)
    equity=max(portfolio.equity(prices), 1.0)
    gross=quantity * price
    if gross < risk.min_trade_notional:
        return OrderDecision(False, side, symbol, quantity, price, None, "below minimum trade notional", now)
    if gross > risk.max_trade_notional or gross > equity * risk.max_single_order_pct:
        return OrderDecision(False, side, symbol, quantity, price, None, "order exceeds max order risk gate", now)
    slip = price * (risk.slippage_bps / 10_000)
    fill = price + slip if side == "BUY" else price - slip
    if side == "BUY":
        cost=quantity * fill + risk.commission_per_trade
        if cost > portfolio.cash:
            return OrderDecision(False, side, symbol, quantity, price, None, "insufficient cash", now)
        projected_position_value = position_market_value(portfolio, symbol, prices) + quantity * fill
        if projected_position_value > equity * risk.max_position_pct:
            return OrderDecision(False, side, symbol, quantity, price, None, "position would exceed max position size", now)
        new_qty=pos.quantity + quantity
        pos.avg_price=((pos.quantity*pos.avg_price)+(quantity*fill))/new_qty
        pos.quantity=new_qty
        portfolio.cash -= cost
        portfolio.positions[symbol]=pos
    else:
        proceeds=quantity * fill - risk.commission_per_trade
        pos.quantity -= quantity
        portfolio.cash += proceeds
        if pos.quantity == 0:
            portfolio.positions.pop(symbol, None)
        else:
            portfolio.positions[symbol]=pos
    return OrderDecision(True, side, symbol, quantity, price, fill, "accepted by mock broker", now)

def place_mock_order(
    side: Side,
    symbol: str,
    quantity: int,
    price: float,
    prices: dict[str, float],
    portfolio_path: Path = STATE_PATH,
    ledger_path: Path = LEDGER_PATH,
    strategy: str = "unknown",
    signal_date: str | None = None,
    execution_date: str | None = None,
) -> OrderDecision:
    with _portfolio_lock(portfolio_path):
        portfolio=load_portfolio(portfolio_path)
        decision=evaluate_order(portfolio, side, symbol, quantity, price, prices)
        decision = replace(decision, strategy=strategy, signal_date=signal_date, execution_date=execution_date)
        append_ledger(decision, ledger_path)
        if decision.accepted:
            save_portfolio(portfolio, portfolio_path)
        return decision
