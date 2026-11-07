"""Boundary tests for market_lab.intraday: the package must stay data-only,
the repo's safety gates must be untouched, and no secrets may be committed.
"""
from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

import pytest

from market_lab.config import OPTIONS_RISK, RISK

REPO = Path(__file__).resolve().parents[2]
PACKAGE_DIR = REPO / "market_lab" / "intraday"
NEW_FILES = sorted(PACKAGE_DIR.glob("*.py")) + [
    REPO / "scripts" / "market_lab_intraday_smoke.py",
    Path(__file__),
    REPO / "tests" / "market_lab" / "test_intraday_bars.py",
    REPO / "tests" / "market_lab" / "test_intraday_store.py",
    REPO / "tests" / "market_lab" / "test_intraday_alpaca_fetcher.py",
    REPO / "tests" / "market_lab" / "test_intraday_events.py",
]

FORBIDDEN_MODULES = ("alpaca.trading", "alpaca.broker", "alpaca.trading.client")
FORBIDDEN_TEXT = ("TradingClient", "BrokerClient", "submit_order", "place_order")


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def test_no_order_capable_imports_in_intraday_package():
    for path in PACKAGE_DIR.glob("*.py"):
        for name in _imports(path):
            assert not name.startswith(FORBIDDEN_MODULES), f"{path.name} imports {name}"
            if name.startswith("alpaca"):
                assert name.startswith("alpaca.data"), f"{path.name} imports non-data alpaca module {name}"
        text = path.read_text()
        for needle in FORBIDDEN_TEXT:
            assert needle not in text, f"{path.name} references {needle}"


def test_safety_gate_flags_remain_off():
    assert RISK.live_trading_enabled is False
    assert OPTIONS_RISK.live_options_enabled is False
    assert OPTIONS_RISK.allow_naked_calls is False


def test_safety_gate_files_unmodified():
    protected = ["market_lab/broker.py", "market_lab/options_paper.py", "market_lab/webapp.py"]
    try:
        diff = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *protected],
                              cwd=REPO, capture_output=True, timeout=30)
        status = subprocess.run(["git", "status", "--porcelain", "--", *protected],
                                cwd=REPO, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        pytest.skip("git unavailable")
    assert diff.returncode == 0, "safety-gated files were modified"
    assert status.stdout.strip() == "", "safety-gated files have uncommitted changes"


SECRET_PATTERNS = [
    re.compile(r"AKIA[0-9A-Z]{16}"),                      # AWS access key id
    re.compile(r"ghp_[A-Za-z0-9]{36}"),                   # GitHub personal token
    re.compile(r"sk-[A-Za-z0-9]{32,}"),                   # OpenAI-style key
    re.compile(r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"(?i)(api_key|api_secret|secret_key|access_token|password)\s*=\s*['\"][^'\"]{8,}['\"]"),
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
]
ALLOWED_EMAIL_DOMAINS = ("example.com", "example.org", "example.net")


def test_no_secret_looking_strings_in_new_files():
    for path in NEW_FILES:
        if not path.exists():
            continue
        text = path.read_text()
        for pattern in SECRET_PATTERNS:
            for match in pattern.finditer(text):
                token = match.group(0)
                if "@" in token and token.rsplit("@", 1)[-1].lower() in ALLOWED_EMAIL_DOMAINS:
                    continue  # documented placeholder identity, not a real address
                raise AssertionError(f"secret-looking string in {path.name}: {pattern.pattern!r}")
