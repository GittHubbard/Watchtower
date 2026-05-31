"""YAML config loaders. Keep simple — read once, return dicts."""
from __future__ import annotations
from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"


def load_watchlist(path: Path | None = None) -> dict:
    """Returns {'defaults': {...}, 'tickers': [{symbol, kind, market_cap_b, ...}, ...]}."""
    p = path or CONFIG_DIR / "watchlist.yaml"
    with open(p) as f:
        return yaml.safe_load(f)


def load_thresholds(path: Path | None = None) -> dict:
    """Returns the full thresholds dict — caller indexes into the section they need."""
    p = path or CONFIG_DIR / "thresholds.yaml"
    with open(p) as f:
        return yaml.safe_load(f)


def get_ticker_config(symbol: str, watchlist: dict | None = None) -> dict:
    """Merge defaults with the ticker's row. Raises KeyError if symbol not in watchlist."""
    wl = watchlist or load_watchlist()
    defaults = wl.get("defaults", {})
    for row in wl.get("tickers", []):
        if row["symbol"] == symbol:
            return {**defaults, **row}
    raise KeyError(f"{symbol} not in watchlist")
