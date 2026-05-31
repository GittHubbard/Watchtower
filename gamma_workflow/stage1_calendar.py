"""Stage 1 — Calendar & event flags.

Deterministic OPEX and VIX expiration are computed in-code.

Earnings come from yfinance's `Ticker.calendar`, which returns a dict containing
"Earnings Date" → list of `datetime.date` objects (Yahoo may give a single date or
a two-date range when timing is uncertain). When the range is two dates we take
the earliest — that's the conservative choice for an "is an event imminent" check.
ETFs and non-reporting symbols just return an empty dict and we treat that as "no
upcoming earnings". Results are cached per (ticker, asof) within a single process
run to avoid hammering Yahoo when many tickers share a workflow invocation.

An optional manual override CSV at `config/earnings.csv` (columns: ticker, date)
takes precedence when present — useful for known-wrong Yahoo dates.

Macro events (FOMC/CPI/NFP) are read from `config/macro_events.csv`
(columns: date, event). If the file is missing, macro flags stay None — non-fatal.

Public API:
    event_flags(symbol, asof_date) -> dict with binary flags and days-to-next-event.
"""
from __future__ import annotations
from datetime import date as date_cls, timedelta
from pathlib import Path
import logging

import pandas as pd

from .config import CONFIG_DIR

logger = logging.getLogger(__name__)


# ---- Deterministic dates ----------------------------------------------------

def monthly_opex(year: int, month: int) -> date_cls:
    """3rd Friday of the month."""
    first = date_cls(year, month, 1)
    # Mon=0, ..., Fri=4
    offset = (4 - first.weekday()) % 7
    first_friday = first + timedelta(days=offset)
    return first_friday + timedelta(days=14)


def vix_expiration(year: int, month: int) -> date_cls:
    """VIX options expire on the Wednesday 30 days prior to the next-month SPX OPEX
    (third Friday of the following month). Standard CBOE rule."""
    next_month_year = year if month < 12 else year + 1
    next_month = month + 1 if month < 12 else 1
    spx_opex_next = monthly_opex(next_month_year, next_month)
    target = spx_opex_next - timedelta(days=30)
    # Roll to nearest Wednesday if target lands on a non-Wednesday.
    # By CBOE rule it always lands on a Wednesday for standard months, but if 30d
    # back hits a holiday, exchange moves it — we leave that edge case for v2.
    while target.weekday() != 2:  # Wed
        target -= timedelta(days=1)
    return target


def days_to_next_opex(d: date_cls) -> int:
    cur = monthly_opex(d.year, d.month)
    if cur >= d:
        return (cur - d).days
    nxt = monthly_opex(d.year + (d.month == 12), (d.month % 12) + 1)
    return (nxt - d).days


def days_to_next_vixp(d: date_cls) -> int:
    cur = vix_expiration(d.year, d.month)
    if cur >= d:
        return (cur - d).days
    nxt = vix_expiration(d.year + (d.month == 12), (d.month % 12) + 1)
    return (nxt - d).days


# ---- Earnings: yfinance primary, CSV override fallback ---------------------

_earnings_cache: dict[str, date_cls | None] = {}


def _load_earnings_override() -> pd.DataFrame:
    """Optional manual override CSV. Used when Yahoo's date is known to be wrong."""
    p = CONFIG_DIR / "earnings.csv"
    if not p.exists():
        return pd.DataFrame(columns=["ticker", "date"])
    df = pd.read_csv(p)
    if df.empty:
        return pd.DataFrame(columns=["ticker", "date"])
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def _fetch_yf_earnings(symbol: str) -> date_cls | None:
    """Return the next earnings date from yfinance, or None.

    yfinance Ticker.calendar returns a dict. The key we care about is
    "Earnings Date" → list[date]. If the list has two dates (Yahoo's
    estimated range) we take the earlier one. Anything unexpected → None.
    """
    if symbol in _earnings_cache:
        return _earnings_cache[symbol]
    try:
        import yfinance as yf  # imported lazily so module loads in network-free envs
        cal = yf.Ticker(symbol).calendar
    except Exception as e:
        logger.warning("yfinance calendar fetch failed for %s: %s", symbol, e)
        _earnings_cache[symbol] = None
        return None

    result: date_cls | None = None
    if isinstance(cal, dict):
        raw = cal.get("Earnings Date")
        if isinstance(raw, (list, tuple)) and raw:
            # Filter to actual dates and pick the earliest.
            dates = [d for d in raw if isinstance(d, date_cls)]
            if dates:
                result = min(dates)
    _earnings_cache[symbol] = result
    return result


def clear_earnings_cache() -> None:
    """Reset the per-process earnings cache (used by tests)."""
    _earnings_cache.clear()


def _load_macro() -> pd.DataFrame:
    p = CONFIG_DIR / "macro_events.csv"
    if not p.exists():
        return pd.DataFrame(columns=["date", "event"])
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def days_to_next_earnings(symbol: str, asof: date_cls) -> int | None:
    """Days until the next earnings release.

    Resolution order:
      1. If config/earnings.csv has a row for this ticker with date >= asof, use it.
      2. Otherwise fall back to yfinance Ticker.calendar.
      3. None if neither source returns a usable future date.
    """
    override = _load_earnings_override()
    if not override.empty:
        future = override[(override["ticker"] == symbol) & (override["date"] >= asof)]
        if not future.empty:
            return (future["date"].min() - asof).days

    yf_date = _fetch_yf_earnings(symbol)
    if yf_date is not None and yf_date >= asof:
        return (yf_date - asof).days
    return None


def days_to_next_macro(asof: date_cls) -> tuple[int | None, str | None]:
    df = _load_macro()
    future = df[df["date"] >= asof].sort_values("date")
    if future.empty:
        return None, None
    row = future.iloc[0]
    return (row["date"] - asof).days, row["event"]


# ---- Public API -------------------------------------------------------------

def event_flags(symbol: str, asof: date_cls) -> dict:
    """Return a dict of all calendar-driven flags for Stage 6 to consume.

    Keys:
        days_to_opex, days_to_vixp           : ints, always populated
        is_opex_day, is_opex_minus_1, is_opex_plus_1, is_vixp_day  : bools
        days_to_earnings                     : int | None
        days_to_macro, next_macro            : int|None, str|None
        within_event_window                  : bool, True if any binary within 5d
    """
    d_opex = days_to_next_opex(asof)
    d_vixp = days_to_next_vixp(asof)
    d_earn = days_to_next_earnings(symbol, asof)
    d_macro, ev_macro = days_to_next_macro(asof)

    binary_horizons = [x for x in (d_earn, d_macro) if x is not None]
    within_event_window = any(0 <= x <= 5 for x in binary_horizons)

    return {
        "days_to_opex": d_opex,
        "days_to_vixp": d_vixp,
        "is_opex_day": d_opex == 0,
        "is_opex_minus_1": d_opex == 1,
        "is_opex_plus_1": d_opex == -1,  # not reachable from days_to_next; placeholder
        "is_vixp_day": d_vixp == 0,
        "days_to_earnings": d_earn,
        "days_to_macro": d_macro,
        "next_macro": ev_macro,
        "within_event_window": within_event_window,
    }
