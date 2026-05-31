"""Stage 5 — Price & volume confirmation.

The old-school sanity check on what the options data is whispering. Reads daily
underlying bars (close, OHLC, volume) from yfinance and computes:

    volume_z              today's volume vs 20-day mean, in std devs
    atr_pct               today's true range, as % of close
    atr_20d_pct           20-day average true range as % of close
    range_expansion       today's TR divided by 20-day ATR (>1 = expansion)
    close_location        where today's close sits in today's range [0=low, 1=high]
    ma20_dist_pct         (close - SMA20) / SMA20
    ma50_dist_pct
    ma200_dist_pct
    vix_close             VIX closing level (only for index ETFs by default)
    vix_5d_change_pct     5-session change in VIX
                          (Note: the original workflow said "5-day change ≥ 15%"
                          for index confirmation; we compute the value here, the
                          threshold lives in thresholds.yaml under confirm.)

For non-index single-name tickers, VIX fields are still populated (it's a market-
wide regime signal that's useful even outside index trades) but Stage 6 only
enforces the VIX confirmation rule when the ticker's `kind == 'index_etf'` per
the watchlist.

Public API:
    compute_confirmation(symbol, asof, history=None) -> dict
    fetch_underlying_history(symbol, end_date, days=260) -> pd.DataFrame
"""
from __future__ import annotations
from datetime import date as date_cls, timedelta
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def fetch_underlying_history(symbol: str, end_date: date_cls, days: int = 260) -> pd.DataFrame:
    """Pull daily OHLCV from yfinance for the lookback window ending on `end_date`.

    Returns a frame with a date-typed index and columns: Open, High, Low, Close,
    Volume. Empty frame on failure.
    """
    try:
        import yfinance as yf
        start = (end_date - timedelta(days=int(days * 1.6))).isoformat()  # buffer for weekends/holidays
        end_inclusive = (end_date + timedelta(days=1)).isoformat()
        df = yf.Ticker(symbol).history(start=start, end=end_inclusive, auto_adjust=False)
    except Exception as e:
        logger.warning("yfinance history fetch failed for %s: %s", symbol, e)
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()
    df.index = pd.to_datetime(df.index.date)
    return df


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Classic true range: max(H-L, |H-prev_close|, |L-prev_close|)."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr


def _compute_features_from_history(hist: pd.DataFrame, asof: date_cls) -> dict:
    """Given a frame with Date index and OHLCV columns, compute the Stage 5
    features as of `asof`. Returns dict; values are NaN when there's not enough
    history."""
    if hist.empty:
        return {k: float("nan") for k in (
            "volume_z", "atr_pct", "atr_20d_pct", "range_expansion",
            "close_location", "ma20_dist_pct", "ma50_dist_pct", "ma200_dist_pct",
        )}

    # Slice up to and including asof
    asof_ts = pd.Timestamp(asof)
    h = hist[hist.index <= asof_ts]
    if h.empty:
        return {k: float("nan") for k in (
            "volume_z", "atr_pct", "atr_20d_pct", "range_expansion",
            "close_location", "ma20_dist_pct", "ma50_dist_pct", "ma200_dist_pct",
        )}

    today = h.iloc[-1]
    close_today = float(today["Close"])
    high_today = float(today["High"])
    low_today = float(today["Low"])
    vol_today = float(today["Volume"])

    # Volume z-score (20-day, excluding today)
    prior = h.iloc[:-1]
    vol_z = float("nan")
    if len(prior) >= 20:
        vol_window = prior["Volume"].iloc[-20:]
        std = float(vol_window.std())
        if std > 0:
            vol_z = (vol_today - float(vol_window.mean())) / std

    # ATR (today's TR and 20-day average)
    tr = _true_range(h["High"], h["Low"], h["Close"])
    tr_today = float(tr.iloc[-1]) if len(tr) > 0 else float("nan")
    atr_20 = float(tr.iloc[-20:].mean()) if len(tr) >= 20 else float("nan")
    atr_pct = tr_today / close_today if close_today > 0 else float("nan")
    atr_20_pct = atr_20 / close_today if (close_today > 0 and np.isfinite(atr_20)) else float("nan")
    range_exp = tr_today / atr_20 if (np.isfinite(atr_20) and atr_20 > 0) else float("nan")

    # Close location in today's range [0=at low, 1=at high]
    rng = high_today - low_today
    close_loc = (close_today - low_today) / rng if rng > 0 else float("nan")

    # Distance from moving averages
    def _ma_dist(n):
        if len(h) < n:
            return float("nan")
        ma = float(h["Close"].iloc[-n:].mean())
        return (close_today - ma) / ma if ma > 0 else float("nan")

    return {
        "volume_z": vol_z,
        "atr_pct": atr_pct,
        "atr_20d_pct": atr_20_pct,
        "range_expansion": range_exp,
        "close_location": close_loc,
        "ma20_dist_pct": _ma_dist(20),
        "ma50_dist_pct": _ma_dist(50),
        "ma200_dist_pct": _ma_dist(200),
    }


def _compute_vix_features(asof: date_cls) -> dict:
    """VIX close and 5-session % change. Pulled from yfinance via the ^VIX ticker."""
    vix_hist = fetch_underlying_history("^VIX", asof, days=20)
    if vix_hist.empty:
        return {"vix_close": float("nan"), "vix_5d_change_pct": float("nan")}

    asof_ts = pd.Timestamp(asof)
    h = vix_hist[vix_hist.index <= asof_ts]
    if h.empty:
        return {"vix_close": float("nan"), "vix_5d_change_pct": float("nan")}

    vix_close = float(h["Close"].iloc[-1])
    if len(h) <= 5:
        return {"vix_close": vix_close, "vix_5d_change_pct": float("nan")}
    vix_5_ago = float(h["Close"].iloc[-6])
    pct = (vix_close - vix_5_ago) / vix_5_ago if vix_5_ago > 0 else float("nan")
    return {"vix_close": vix_close, "vix_5d_change_pct": pct}


def compute_confirmation(
    symbol: str,
    asof: date_cls,
    history: pd.DataFrame | None = None,
) -> dict:
    """End-to-end Stage 5. `history` is optional pre-fetched OHLCV; if omitted,
    fetches from yfinance. VIX is always fetched separately.
    """
    if history is None:
        history = fetch_underlying_history(symbol, asof)
    feats = _compute_features_from_history(history, asof)
    feats.update(_compute_vix_features(asof))
    return feats
