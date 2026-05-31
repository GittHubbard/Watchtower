"""Risk-free rate from FRED's DGS3MO series (3-month T-bill, secondary market).

Uses the no-API-key CSV endpoint. Persists a single parquet under data/rates/dgs3mo.parquet
with columns: date (date), rate (float, decimal — e.g. 0.0525 for 5.25%).

`get_rate(d)` returns the latest available rate on or before date d. If no cache exists
or the cache is stale, refreshes from FRED.
"""
from __future__ import annotations
from datetime import date as date_cls, datetime, timedelta
from io import StringIO
from pathlib import Path
import logging

import pandas as pd
import requests

from .storage import RATES_DIR, _ensure_dirs

logger = logging.getLogger(__name__)

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3MO"
CACHE_PATH = RATES_DIR / "dgs3mo.parquet"
DEFAULT_RATE = 0.045  # fallback if FRED is unreachable


def _fetch_fred() -> pd.DataFrame:
    """Pull the full DGS3MO series. Returns df with columns ['date', 'rate']."""
    resp = requests.get(FRED_URL, timeout=15)
    resp.raise_for_status()
    raw = pd.read_csv(StringIO(resp.text))
    # FRED uses '.' for missing days (weekends/holidays); coerce and drop.
    raw.columns = [c.strip() for c in raw.columns]
    date_col = raw.columns[0]
    rate_col = raw.columns[1]
    raw[rate_col] = pd.to_numeric(raw[rate_col], errors="coerce")
    raw = raw.dropna(subset=[rate_col])
    out = pd.DataFrame({
        "date": pd.to_datetime(raw[date_col]).dt.date,
        "rate": raw[rate_col].astype(float) / 100.0,  # FRED reports %, store decimal
    })
    return out.sort_values("date").reset_index(drop=True)


def refresh_cache() -> pd.DataFrame:
    """Force a refresh from FRED and persist."""
    _ensure_dirs()
    df = _fetch_fred()
    df.to_parquet(CACHE_PATH, index=False)
    logger.info("Refreshed FRED DGS3MO cache: %d rows, latest %s", len(df), df["date"].iloc[-1])
    return df


def _load_cache() -> pd.DataFrame | None:
    if not CACHE_PATH.exists():
        return None
    return pd.read_parquet(CACHE_PATH)


def get_rate(d: date_cls | None = None, max_cache_age_days: int = 3) -> float:
    """Return the risk-free rate on or before date d (defaults to today).

    Refreshes the cache from FRED if it's missing or its latest row is older than
    max_cache_age_days business days. Falls back to DEFAULT_RATE if FRED is unreachable
    AND no cache exists.
    """
    target = d or date_cls.today()
    cache = _load_cache()

    needs_refresh = (
        cache is None
        or cache.empty
        or (date_cls.today() - cache["date"].max()).days > max_cache_age_days
    )
    if needs_refresh:
        try:
            cache = refresh_cache()
        except Exception as e:
            logger.warning("FRED fetch failed: %s. Using cached or default.", e)
            if cache is None or cache.empty:
                return DEFAULT_RATE

    on_or_before = cache[cache["date"] <= target]
    if on_or_before.empty:
        return DEFAULT_RATE
    return float(on_or_before["rate"].iloc[-1])
