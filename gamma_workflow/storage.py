"""Partitioned-parquet storage.

Layout:
    data/features/date=YYYY-MM-DD.parquet   one row per ticker, full feature vector
    data/signals/date=YYYY-MM-DD.parquet    one row per fired signal
    data/rates/dgs3mo.parquet               appended daily, single file (small)
    data/tickets/YYYY-MM-DD_TICKER_PLAYBOOK.md  human-readable trade tickets

Design notes:
- Per-date partitioning (not per-ticker-per-date) keeps file count low while letting
  us load a single day's snapshot cheaply. Forward-return backfills read a window of
  date files, append the realized return column, and rewrite — atomic via temp file.
- All writes are idempotent: writing the same date twice overwrites.
"""
from __future__ import annotations
from pathlib import Path
from datetime import date as date_cls
import logging
import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
FEATURES_DIR = DATA_DIR / "features"
SIGNALS_DIR = DATA_DIR / "signals"
TICKETS_DIR = DATA_DIR / "tickets"
RATES_DIR = DATA_DIR / "rates"


def _ensure_dirs() -> None:
    for d in (FEATURES_DIR, SIGNALS_DIR, TICKETS_DIR, RATES_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _date_path(base: Path, d: date_cls) -> Path:
    return base / f"date={d.isoformat()}.parquet"


def write_features(df: pd.DataFrame, d: date_cls) -> Path:
    """Write the full day's feature frame (one row per ticker). Overwrites."""
    _ensure_dirs()
    if "ticker" not in df.columns:
        raise ValueError("features frame must have a 'ticker' column")
    path = _date_path(FEATURES_DIR, d)
    df.to_parquet(path, index=False)
    return path


def read_features(d: date_cls) -> pd.DataFrame | None:
    path = _date_path(FEATURES_DIR, d)
    return pd.read_parquet(path) if path.exists() else None


def read_features_range(start: date_cls, end: date_cls) -> pd.DataFrame:
    """Concatenate features across a date range. Missing dates skipped."""
    _ensure_dirs()
    frames = []
    for p in sorted(FEATURES_DIR.glob("date=*.parquet")):
        d_str = p.stem.split("=", 1)[1]
        d = date_cls.fromisoformat(d_str)
        if start <= d <= end:
            df = pd.read_parquet(p)
            df["date"] = d
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def write_signals(df: pd.DataFrame, d: date_cls) -> Path:
    """Write all signals fired on date d. Overwrites."""
    _ensure_dirs()
    path = _date_path(SIGNALS_DIR, d)
    df.to_parquet(path, index=False)
    return path


def read_signals(d: date_cls) -> pd.DataFrame | None:
    path = _date_path(SIGNALS_DIR, d)
    return pd.read_parquet(path) if path.exists() else None


def write_ticket(d: date_cls, ticker: str, playbook: str, content: str) -> Path:
    """Persist a markdown trade ticket. Filename: YYYY-MM-DD_TICKER_PLAYBOOK.md."""
    _ensure_dirs()
    path = TICKETS_DIR / f"{d.isoformat()}_{ticker}_{playbook}.md"
    path.write_text(content)
    return path


# ---- Open positions (Stage 8) ----------------------------------------------
# Minimal JSON store the PM maintains. Each position is a dict; see stage8 for
# the expected schema. This is intentionally simple — there's no execution layer,
# so positions are entered/closed by the PM and this file just tracks state.
POSITIONS_PATH = DATA_DIR / "positions.json"


def read_positions() -> list[dict]:
    """Return the list of open positions, or [] if none recorded."""
    import json
    if not POSITIONS_PATH.exists():
        return []
    try:
        return json.loads(POSITIONS_PATH.read_text())
    except Exception:
        return []


def write_positions(positions: list[dict]) -> Path:
    """Overwrite the positions file."""
    import json
    _ensure_dirs()
    POSITIONS_PATH.write_text(json.dumps(positions, indent=2, default=str))
    return POSITIONS_PATH


# ---- Option-chain snapshots (Stage 4 OI-delta needs yesterday's chain) ------
CHAINS_DIR = DATA_DIR / "chains"


def _chain_path(symbol: str, d: date_cls) -> Path:
    return CHAINS_DIR / f"ticker={symbol}" / f"date={d.isoformat()}.parquet"


def write_chain_snapshot(symbol: str, d: date_cls, chain: pd.DataFrame) -> Path:
    """Persist the day's refined chain so tomorrow's run can compute OI deltas.
    Partitioned by ticker then date. Overwrites."""
    CHAINS_DIR.mkdir(parents=True, exist_ok=True)
    path = _chain_path(symbol, d)
    path.parent.mkdir(parents=True, exist_ok=True)
    chain.to_parquet(path, index=False)
    return path


def read_chain_snapshot(symbol: str, asof: date_cls) -> pd.DataFrame | None:
    """Return the most recent chain snapshot strictly before `asof`, or None.

    We look back day-by-day up to 7 calendar days to skip weekends/holidays."""
    from datetime import timedelta
    tdir = CHAINS_DIR / f"ticker={symbol}"
    if not tdir.exists():
        logger.warning("No prior chain snapshot dir for %s; OI-delta will be NaN.", symbol)
        return None
    for back in range(1, 8):
        d = asof - timedelta(days=back)
        p = _chain_path(symbol, d)
        if p.exists():
            return pd.read_parquet(p)
    logger.warning(
        "No prior chain snapshot for %s within 7 days before %s; OI-delta will be NaN.",
        symbol, asof,
    )
    return None
