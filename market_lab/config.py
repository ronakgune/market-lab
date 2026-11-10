from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("MARKET_LAB_DATA_DIR", ROOT / "data" / "market-lab")).expanduser().resolve()
PRICE_DIR = DATA_DIR / "prices"
SYNTHETIC_PRICE_DIR = DATA_DIR / "synthetic_prices"
REPORT_DIR = DATA_DIR / "reports"
FACTOR_DIR = DATA_DIR / "factors"
FUNDAMENTALS_DIR = DATA_DIR / "fundamentals"
EVIDENCE_DIR = DATA_DIR / "evidence"
OPTIONS_DIR = DATA_DIR / "options"
OPTIONS_CHAIN_DIR = OPTIONS_DIR / "chains"
OPTIONS_PAPER_STATE_PATH = OPTIONS_DIR / "paper_options_state.json"
OPTIONS_PAPER_LEDGER_PATH = OPTIONS_DIR / "paper_options_ledger.jsonl"
OPTIONS_PAPER_CANDIDATES_PATH = OPTIONS_DIR / "paper_options_candidates.jsonl"
LEDGER_PATH = DATA_DIR / "mock_ledger.jsonl"

INTRADAY_DIR = DATA_DIR / "intraday"
INTRADAY_BARS_DIR = INTRADAY_DIR / "bars"
INTRADAY_SYNTHETIC_BARS_DIR = INTRADAY_DIR / "synthetic_bars"
INTRADAY_EDGAR_DIR = INTRADAY_DIR / "edgar_cache"

SOURCE_THESIS_DIR = DATA_DIR / "source_thesis"
SOURCE_THESIS_REPORT_DIR = SOURCE_THESIS_DIR / "reports"
COMPANY_INTELLIGENCE_DIR = DATA_DIR / "company_intelligence"

VT_TREND_DIR = DATA_DIR / "vt_trend"
VT_TREND_STATE = VT_TREND_DIR / "portfolio_state.json"
VT_TREND_LEDGER = VT_TREND_DIR / "ledger.jsonl"
VT_TREND_CANDIDATES = VT_TREND_DIR / "pending_candidates.jsonl"
VT_TREND_REPORT_DIR = VT_TREND_DIR / "reports"
VT_TREND_STARTING_CASH = 25_000.0

TSMOM_DIR = DATA_DIR / "tsmom"
TSMOM_STATE = TSMOM_DIR / "portfolio_state.json"
TSMOM_LEDGER = TSMOM_DIR / "ledger.jsonl"
TSMOM_CANDIDATES = TSMOM_DIR / "pending_candidates.jsonl"
TSMOM_REPORT_DIR = TSMOM_DIR / "reports"
TSMOM_STARTING_CASH = 25_000.0

CRYPTO_CARRY_DIR = DATA_DIR / "crypto_carry"
CRYPTO_CARRY_FUNDING_PATH = CRYPTO_CARRY_DIR / "funding.jsonl"
CRYPTO_CARRY_REPORT_DIR = CRYPTO_CARRY_DIR / "reports"

XS_MOMENTUM_DIR = DATA_DIR / "xs_momentum"
XS_MOMENTUM_STATE = XS_MOMENTUM_DIR / "portfolio_state.json"
XS_MOMENTUM_LEDGER = XS_MOMENTUM_DIR / "ledger.jsonl"
XS_MOMENTUM_CANDIDATES = XS_MOMENTUM_DIR / "pending_candidates.jsonl"
XS_MOMENTUM_EQUITY_HISTORY = XS_MOMENTUM_DIR / "equity_history.jsonl"
XS_MOMENTUM_REBALANCE_LOG = XS_MOMENTUM_DIR / "rebalance_log.jsonl"
XS_MOMENTUM_REPORT_DIR = XS_MOMENTUM_DIR / "reports"
XS_MOMENTUM_STARTING_CASH = 25_000.0

PENDING_CANDIDATES_PATH = DATA_DIR / "pending_order_candidates.jsonl"
STATE_PATH = DATA_DIR / "mock_portfolio_state.json"

DEFAULT_UNIVERSE = [
    # Broad but liquid starting watchlist: indexes, mega-cap tech, volatility proxy, crypto proxies.
    "SPY", "QQQ", "IWM", "DIA", "TLT", "GLD", "USO",
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA",
    "AMD", "AVGO", "SMCI", "COIN", "MSTR", "IBIT", "ETHA",
]

@dataclass(frozen=True)
class RiskConfig:
    starting_cash: float = 100_000.0
    max_position_pct: float = 0.10
    max_single_order_pct: float = 0.05
    min_trade_notional: float = 500.0
    max_trade_notional: float = 5_000.0
    commission_per_trade: float = 1.00
    slippage_bps: float = 5.0
    allow_short: bool = False
    allow_options: bool = True
    allow_margin: bool = False
    live_trading_enabled: bool = False

@dataclass(frozen=True)
class OptionsRiskConfig:
    allow_options: bool = True
    paper_options_enabled: bool = True
    live_options_enabled: bool = False
    allow_margin: bool = False
    allow_naked_calls: bool = False
    min_dte: int = 14
    max_dte: int = 60
    max_chain_age_days: int = 2
    max_bid_ask_spread_pct: float = 0.25
    min_open_interest: int = 50
    min_volume: int = 10
    max_contracts_per_symbol: int = 1
    max_option_premium_pct: float = 0.02
    max_assignment_notional_pct: float = 0.20
    max_total_options_assignment_pct: float = 0.35
    max_abs_short_call_delta: float = 0.45
    max_abs_short_put_delta: float = 0.40
    min_premium_yield_annualized: float = 0.02

RISK = RiskConfig()
OPTIONS_RISK = OptionsRiskConfig()

def ensure_dirs() -> None:
    for path in (
        DATA_DIR,
        PRICE_DIR,
        SYNTHETIC_PRICE_DIR,
        REPORT_DIR,
        FACTOR_DIR,
        EVIDENCE_DIR,
        OPTIONS_DIR,
        OPTIONS_CHAIN_DIR,
        VT_TREND_DIR,
        VT_TREND_REPORT_DIR,
        TSMOM_DIR,
        TSMOM_REPORT_DIR,
        XS_MOMENTUM_DIR,
        XS_MOMENTUM_REPORT_DIR,
        CRYPTO_CARRY_DIR,
        CRYPTO_CARRY_REPORT_DIR,
        SOURCE_THESIS_DIR,
        SOURCE_THESIS_REPORT_DIR,
        COMPANY_INTELLIGENCE_DIR,
        INTRADAY_DIR,
        INTRADAY_BARS_DIR,
        INTRADAY_SYNTHETIC_BARS_DIR,
        INTRADAY_EDGAR_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
