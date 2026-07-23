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
import math
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
UPWARD_WATCHLIST_DIR   = DATA_DIR / "Upward-Watchlist"
DOWNWARD_WATCHLIST_DIR = DATA_DIR / "Downward-Watchlist"


def write_techsignal_ticket(symbol: str, d: date_cls, content: str) -> Path:
    """Persist a Stage 5.5 technical-signal ticket per ticker.

    Path: data/chains/ticker=XXX/tickets/YYYY-MM-DD_XXX_techsignal.md
    """
    ticket_dir = CHAINS_DIR / f"ticker={symbol}" / "tickets"
    ticket_dir.mkdir(parents=True, exist_ok=True)
    path = ticket_dir / f"{d.isoformat()}_{symbol}_techsignal.md"
    path.write_text(content)
    return path


def write_watchlist_ticket(symbol: str, d: date_cls, content: str, direction: str) -> Path:
    """Persist a tech-signal ticket in the appropriate watchlist directory.

    direction: "upward" or "downward"
    Path: data/{Upward,Downward}-Watchlist/YYYY-MM-DD/YYYY-MM-DD_XXX_techsignal.md
    """
    base = UPWARD_WATCHLIST_DIR if direction == "upward" else DOWNWARD_WATCHLIST_DIR
    day_dir = base / d.isoformat()
    day_dir.mkdir(parents=True, exist_ok=True)
    path = day_dir / f"{d.isoformat()}_{symbol}_techsignal.md"
    path.write_text(content)
    return path


# ---- Tech-signal ticket log (append-only, long-term assessment) -------------
# A single parquet that accumulates every day's tech-signal tickets — the same
# content written to the per-ticker <date>_<ticker>_techsignal.md files — so the
# full history lives in one queryable table that just keeps growing. Each row is
# one ticket: its date/ticker/recommendation/score/watchlist plus the complete
# markdown body. Writes are idempotent per date (re-running a day replaces that
# day's rows rather than duplicating them) and atomic via a temp file.
#
# The path is derived from DATA_DIR at call time (not cached in a constant) so a
# test that monkeypatches DATA_DIR redirects it too, like the other stores.
TECHSIGNAL_LOG_NAME = "techsignal_log.parquet"

TECHSIGNAL_LOG_COLUMNS = [
    "date", "ticker", "recommendation", "score", "watchlist", "content",
]


def techsignal_log_path() -> Path:
    """Location of the growing tech-signal ticket log (``data/techsignal_log.parquet``)."""
    return DATA_DIR / TECHSIGNAL_LOG_NAME


def append_techsignal_log(records: list[dict], d: date_cls) -> Path | None:
    """Append the day's tech-signal ticket rows to the growing parquet log.

    ``records`` is a list of dicts with the keys in :data:`TECHSIGNAL_LOG_COLUMNS`
    (``date`` as an ISO string). Any existing rows for date ``d`` are replaced
    first, so a same-day re-run overwrites rather than duplicates. Returns the log
    path, or None if ``records`` is empty. The write is atomic (temp file + os
    replace) so a crash mid-write can't corrupt the accumulated history.
    """
    if not records:
        return None
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(records, columns=TECHSIGNAL_LOG_COLUMNS)

    path = techsignal_log_path()
    if path.exists():
        existing = pd.read_parquet(path)
        if "date" in existing.columns:
            existing = existing[existing["date"] != d.isoformat()]
        combined = pd.concat([existing, new], ignore_index=True)
    else:
        combined = new

    tmp = path.with_suffix(".parquet.tmp")
    combined.to_parquet(tmp, index=False)
    tmp.replace(path)  # atomic on the same filesystem
    logger.info("Tech-signal log now %d rows (+%d for %s)", len(combined), len(new), d)
    return path


def read_techsignal_log() -> pd.DataFrame:
    """Return the full accumulated tech-signal ticket log (empty frame if none)."""
    path = techsignal_log_path()
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=TECHSIGNAL_LOG_COLUMNS)


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


# ---- Fundamentals evolution cache (Stage 5.5, Feature 3) --------------------
# Append-only, dated snapshots of the fundamental/analyst scalar payloads — but a
# new snapshot is written ONLY when the payload differs from the latest cached one.
# Two independent buckets ("financials" and "analyst") so a change in analyst
# coverage doesn't rewrite the (unchanged) financials, and vice versa. Stored as
# JSON (small flat dicts, like positions.json) and kept forever until the PM
# deletes them. This versioned history is the substrate the next Stage 5.5 build
# uses to narrate how analyst views have evolved.
FUNDAMENTALS_DIR = DATA_DIR / "fundamentals"


def _fund_dir(symbol: str, bucket: str) -> Path:
    return FUNDAMENTALS_DIR / f"ticker={symbol}" / bucket


def read_latest_fundamentals(symbol: str, bucket: str) -> dict | None:
    """Most recent cached payload for (symbol, bucket), or None if none stored."""
    import json
    d = _fund_dir(symbol, bucket)
    if not d.exists():
        return None
    files = sorted(d.glob("date=*.json"))
    if not files:
        return None
    try:
        return json.loads(files[-1].read_text())
    except Exception as e:
        logger.warning("Failed to read fundamentals cache %s/%s: %s", symbol, bucket, e)
        return None


def _payloads_equal(a: dict, b: dict) -> bool:
    """Dict equality with NaN==NaN (so an all-NaN refetch doesn't churn the cache)."""
    if a is None or b is None:
        return False
    if set(a.keys()) != set(b.keys()):
        return False
    for k in a:
        va, vb = a[k], b[k]
        if isinstance(va, float) and isinstance(vb, float) and math.isnan(va) and math.isnan(vb):
            continue
        if va != vb:
            return False
    return True


def append_fundamentals_if_changed(symbol: str, bucket: str, payload: dict, d: date_cls) -> bool:
    """Write a dated snapshot for (symbol, bucket) only if `payload` differs from the
    latest cached one. Returns True if a new snapshot was written, else False."""
    import json
    latest = read_latest_fundamentals(symbol, bucket)
    if _payloads_equal(latest, payload):
        return False
    out_dir = _fund_dir(symbol, bucket)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"date={d.isoformat()}.json"
    path.write_text(json.dumps(payload, indent=2, default=str, sort_keys=True))
    logger.info("Fundamentals cache updated: %s/%s @ %s", symbol, bucket, d)
    return True


# ---- Fundamentals freshness snapshot (TTL gate for the network fetch) -------
# Distinct from the evolution cache above: that one is append-on-change history,
# so its newest file reflects the last time the values *changed* — useless for a
# "have we fetched recently?" check (values that hold steady for a month would
# look stale and trigger a needless refetch). This snapshot records the last
# *fetch* date plus the full computed scalar set, so run_eod can (a) decide
# whether the TTL has lapsed and (b) reuse the scalars on skip days without
# touching the network. One file per ticker, overwritten on each real fetch.

def _fund_snapshot_path(symbol: str) -> Path:
    return FUNDAMENTALS_DIR / f"ticker={symbol}" / "snapshot.json"


def read_fundamentals_snapshot(symbol: str) -> dict | None:
    """Return {"asof": "YYYY-MM-DD", "scalars": {...}} for the last fetch, or None."""
    import json
    p = _fund_snapshot_path(symbol)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception as e:
        logger.warning("Failed to read fundamentals snapshot for %s: %s", symbol, e)
        return None


def write_fundamentals_snapshot(symbol: str, d: date_cls, scalars: dict) -> Path:
    """Record the fetch date + computed scalar set for `symbol`. Overwrites."""
    import json
    p = _fund_snapshot_path(symbol)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"asof": d.isoformat(), "scalars": scalars}
    p.write_text(json.dumps(payload, indent=2, default=str, sort_keys=True))
    return p


# ---- Earnings-date cache (Stage 1, TTL gate for Ticker.calendar) ------------
# Earnings dates move at roughly a quarterly cadence, so pulling Ticker.calendar
# every day for every ticker is wasted network. We persist the last fetched
# earnings date per ticker here; Stage 1 refreshes only after the TTL lapses.
# Tracked in data/ so it survives the Action's ephemeral runner like other caches.
EARNINGS_DIR = DATA_DIR / "earnings"


def read_earnings_cache(symbol: str) -> dict | None:
    """Return {"fetched": "YYYY-MM-DD", "earnings_date": "YYYY-MM-DD"|None}, or None."""
    import json
    p = EARNINGS_DIR / f"{symbol}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception as e:
        logger.warning("Failed to read earnings cache for %s: %s", symbol, e)
        return None


def write_earnings_cache(symbol: str, fetched: date_cls, earnings_date: date_cls | None) -> Path:
    """Record the fetch date and resolved next-earnings date for `symbol`. Overwrites."""
    import json
    EARNINGS_DIR.mkdir(parents=True, exist_ok=True)
    p = EARNINGS_DIR / f"{symbol}.json"
    payload = {
        "fetched": fetched.isoformat(),
        "earnings_date": earnings_date.isoformat() if earnings_date is not None else None,
    }
    p.write_text(json.dumps(payload, indent=2))
    return p
