from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Literal

from .config import OPTIONS_CHAIN_DIR, ensure_dirs

try:
    import yfinance as yf
except ImportError:  # pragma: no cover - dependency is declared, defensive for minimal environments
    yf = None

OptionType = Literal["CALL", "PUT"]


@dataclass(frozen=True)
class OptionQuote:
    bid: float
    ask: float
    mid: float
    volume: int
    open_interest: int

    @property
    def spread_pct(self) -> float:
        if self.mid <= 0:
            return 1.0
        return (self.ask - self.bid) / self.mid


@dataclass(frozen=True)
class OptionGreeks:
    delta: float
    gamma: float
    theta: float
    vega: float
    implied_volatility: float
    degenerate: bool = False


@dataclass(frozen=True)
class OptionContract:
    underlying: str
    expiration: str
    strike: float
    option_type: OptionType
    quote: OptionQuote
    greeks: OptionGreeks

    @property
    def contract_id(self) -> str:
        cp = "C" if self.option_type == "CALL" else "P"
        return f"{self.underlying.upper()}-{self.expiration}-{cp}-{self.strike:.2f}"

    def dte(self, as_of: str) -> int:
        as_date = datetime.fromisoformat(as_of.replace("Z", "+00:00")).date() if "T" in as_of else date.fromisoformat(as_of)
        return (date.fromisoformat(self.expiration) - as_date).days


@dataclass(frozen=True)
class OptionChainSnapshot:
    underlying: str
    underlying_price: float
    as_of: str
    source: str
    contracts: list[OptionContract]


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _approx_delta(option_type: OptionType, underlying_price: float, strike: float, dte: int, implied_volatility: float) -> tuple[float, bool]:
    if underlying_price <= 0 or strike <= 0 or dte <= 0 or implied_volatility <= 0:
        moneyness = underlying_price / strike if strike else 1.0
        if option_type == "CALL":
            return max(0.05, min(0.95, moneyness - 0.5)), True
        return -max(0.05, min(0.95, 1.5 - moneyness)), True
    t = max(dte / 365.0, 1 / 365)
    sigma = max(implied_volatility, 0.01)
    d1 = (math.log(underlying_price / strike) + 0.5 * sigma * sigma * t) / (sigma * math.sqrt(t))
    call_delta = _norm_cdf(d1)
    return call_delta if option_type == "CALL" else call_delta - 1.0, False


def _float_value(row, name: str, default: float = 0.0) -> float:
    value = row.get(name, default)
    if value is None:
        return default
    try:
        if math.isnan(float(value)):
            return default
    except (TypeError, ValueError):
        return default
    return float(value)


def _int_value(row, name: str, default: int = 0) -> int:
    return int(max(_float_value(row, name, float(default)), 0.0))


def _contracts_from_frame(frame, symbol: str, expiration: str, option_type: OptionType, underlying_price: float, as_of: str) -> list[OptionContract]:
    out: list[OptionContract] = []
    dte = (date.fromisoformat(expiration) - datetime.fromisoformat(as_of.replace("Z", "+00:00")).date()).days
    for row in frame.to_dict("records"):
        strike = _float_value(row, "strike")
        bid = _float_value(row, "bid")
        ask = _float_value(row, "ask")
        last = _float_value(row, "lastPrice")
        mid = (bid + ask) / 2 if bid > 0 and ask >= bid else last
        iv = _float_value(row, "impliedVolatility")
        if strike <= 0 or mid <= 0:
            continue
        delta, is_degenerate = _approx_delta(option_type, underlying_price, strike, dte, iv)
        out.append(
            OptionContract(
                symbol.upper(),
                expiration,
                strike,
                option_type,
                OptionQuote(bid, ask, mid, _int_value(row, "volume"), _int_value(row, "openInterest")),
                OptionGreeks(delta, 0.0, 0.0, 0.0, iv, is_degenerate),
            )
        )
    return out


def fetch_option_chain_snapshot(symbol: str, min_dte: int = 14, max_dte: int = 60) -> OptionChainSnapshot:
    if yf is None:
        raise RuntimeError("yfinance is required to fetch option chains")
    ticker = yf.Ticker(symbol.upper())
    as_of = datetime.now(timezone.utc).isoformat()
    today = datetime.fromisoformat(as_of).date()
    expirations = []
    for exp in getattr(ticker, "options", []) or []:
        try:
            dte = (date.fromisoformat(exp) - today).days
        except ValueError:
            continue
        if min_dte <= dte <= max_dte:
            expirations.append(exp)
    if not expirations:
        raise ValueError(f"no option expirations inside {min_dte}-{max_dte} DTE for {symbol.upper()}")
    expiration = expirations[0]
    fast_info = getattr(ticker, "fast_info", {}) or {}
    underlying_price = float(fast_info.get("last_price") or fast_info.get("lastPrice") or 0.0)
    chain = ticker.option_chain(expiration)
    if underlying_price <= 0:
        strikes = []
        for frame in (chain.calls, chain.puts):
            for row in frame.to_dict("records"):
                strike = _float_value(row, "strike")
                if strike > 0:
                    strikes.append(strike)
        if strikes:
            underlying_price = sorted(strikes)[len(strikes) // 2]
    contracts = []
    contracts.extend(_contracts_from_frame(chain.calls, symbol, expiration, "CALL", underlying_price, as_of))
    contracts.extend(_contracts_from_frame(chain.puts, symbol, expiration, "PUT", underlying_price, as_of))
    if not contracts:
        raise ValueError(f"no usable option contracts fetched for {symbol.upper()} {expiration}")
    return OptionChainSnapshot(symbol.upper(), underlying_price, as_of, "yfinance", contracts)


def _snapshot_path(symbol: str, chain_dir: Path = OPTIONS_CHAIN_DIR) -> Path:
    safe = symbol.upper().replace("/", "_").replace("-", "_")
    return chain_dir / f"{safe}.json"


def save_option_chain_snapshot(snapshot: OptionChainSnapshot, chain_dir: Path = OPTIONS_CHAIN_DIR) -> Path:
    ensure_dirs()
    path = _snapshot_path(snapshot.underlying, chain_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(snapshot), indent=2, sort_keys=True))
    return path


def _legacy_degenerate_from_record(c: dict, as_of: str) -> bool:
    greeks = c.get("greeks", {})
    iv = float(greeks.get("implied_volatility", 0.0) or 0.0)
    try:
        expiry = date.fromisoformat(str(c.get("expiration", ""))[:10])
        as_of_date = date.fromisoformat(str(as_of)[:10])
        dte = (expiry - as_of_date).days
    except ValueError:
        dte = 0
    delta = float(greeks.get("delta", 0.0) or 0.0)
    return iv <= 0 or dte <= 0 or abs(delta) <= 0


def _contract_from_record(c: dict, as_of: str) -> OptionContract:
    greeks_data = dict(c["greeks"])
    if "degenerate" not in greeks_data:
        greeks_data["degenerate"] = _legacy_degenerate_from_record(c, as_of)
    contract = OptionContract(
        underlying=c["underlying"],
        expiration=c["expiration"],
        strike=float(c["strike"]),
        option_type=c["option_type"],
        quote=OptionQuote(**c["quote"]),
        greeks=OptionGreeks(**greeks_data),
    )
    # Stored-record path finiteness (audit run-1, finding 2): unlike the
    # yfinance path (_float_value maps NaN to a default), records consumed
    # by the paper execution step passed float fields through unchecked.
    # Drop — raise on — any contract with a non-finite strike/quote/greek;
    # callers that load collections already skip records raising ValueError.
    numeric = [contract.strike, contract.quote.bid, contract.quote.ask,
               contract.quote.mid, contract.quote.volume,
               contract.quote.open_interest,
               contract.greeks.delta, contract.greeks.gamma,
               contract.greeks.theta, contract.greeks.vega,
               contract.greeks.implied_volatility]
    try:
        finite = all(math.isfinite(v) for v in numeric)
    except TypeError:
        finite = False
    if not finite:
        raise ValueError("option contract record contains non-finite values")
    return contract


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant in stored data: {value}")


def load_option_chain_snapshot(symbol: str, chain_dir: Path = OPTIONS_CHAIN_DIR) -> OptionChainSnapshot:
    path = _snapshot_path(symbol, chain_dir)
    data = json.loads(path.read_text(), parse_constant=_reject_json_constant)
    contracts = [_contract_from_record(c, data.get("as_of", "")) for c in data.get("contracts", [])]

    return OptionChainSnapshot(
        underlying=data["underlying"],
        underlying_price=float(data["underlying_price"]),
        as_of=data["as_of"],
        source=data.get("source", "unknown"),
        contracts=contracts,
    )


def load_available_option_chains(chain_dir: Path = OPTIONS_CHAIN_DIR) -> list[OptionChainSnapshot]:
    if not chain_dir.exists():
        return []
    snapshots: list[OptionChainSnapshot] = []
    for path in sorted(chain_dir.glob("*.json")):
        try:
            snapshots.append(load_option_chain_snapshot(path.stem, chain_dir))
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return snapshots
