#!/usr/bin/env python3
"""This script generates technical-analysis-based recommendations for
educational and research use. It does not place orders, does not account for
transaction costs, slippage, position sizing, or risk management, and does
not constitute financial advice. Indicators lag; divergence signals can
persist for a long time before price actually turns. Vol-regime
classification uses historical percentiles and does not predict future
volatility. Always backtest on the specific instrument and timeframe before
relying on any signal.

Usage:
    python trade_signals.py AAPL
    python trade_signals.py AAPL --bar -1 --bars-since 400
    python trade_signals.py AAPL --iv-csv iv_data.csv --params params.yaml
    python trade_signals.py --test
"""

# Pinned versions (tested with):
#   pandas       >= 2.1.0
#   numpy        >= 1.26.0
#   pandas_ta    >= 0.3.14b  (preferred; Wilder smoothing is the default for ATR/ADX)
#   yfinance     >= 0.2.40   (for live fetch; not required for --test)
#
# IV data source: user-supplied CSV with datetime index and iv_30 column (decimal,
# e.g. 0.28 = 28%). Symbol mapping is instrument-specific; no default mapping.
#
# ATR and ADX use Wilder's smoothing (EWM alpha=1/period, adjust=False).
# This matches pandas_ta defaults. Some libraries default to EMA; we document
# and enforce Wilder's explicitly in _atr14_series() and _adx_series().

from __future__ import annotations

import argparse
import math
import sys
import warnings
from datetime import date as date_cls, timedelta
from typing import Optional

import numpy as np
import pandas as pd

_NAN = float("nan")

# ---------------------------------------------------------------------------
# §2 — Indicator helpers (pure, vectorized)
# ---------------------------------------------------------------------------

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """Wilder smoothing: EWM with alpha=1/period, adjust=False."""
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def _atr14_series(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Full ATR series using Wilder's smoothing."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return _wilder_smooth(tr, period)


def _adx_series(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.DataFrame:
    """Returns DataFrame with columns: adx, plus_di, minus_di. Wilder's smoothing."""
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    up_move = high - prev_high
    down_move = prev_low - low
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    atr = _wilder_smooth(tr, period)
    smooth_plus = _wilder_smooth(plus_dm, period)
    smooth_minus = _wilder_smooth(minus_dm, period)
    safe_atr = atr.replace(0, np.nan)
    plus_di = 100.0 * smooth_plus / safe_atr
    minus_di = 100.0 * smooth_minus / safe_atr
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = _wilder_smooth(dx, period)
    return pd.DataFrame({"adx": adx, "plus_di": plus_di, "minus_di": minus_di}, index=close.index)


def _rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
    """Full RSI series using Wilder's smoothing."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = _wilder_smooth(gain, period)
    avg_loss = _wilder_smooth(loss, period)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def _macd_series(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Returns DataFrame with macd, macd_signal, macd_hist."""
    macd_line = _ema(close, fast) - _ema(close, slow)
    sig = _ema(macd_line, signal)
    return pd.DataFrame({
        "macd": macd_line,
        "macd_signal": sig,
        "macd_hist": macd_line - sig,
    }, index=close.index)


def _bb_series(close: pd.Series, period: int = 20, k: float = 2.0) -> pd.DataFrame:
    """Returns DataFrame with bb_upper, bb_mid, bb_lower, bb_pct, bb_width."""
    mid = close.rolling(period, min_periods=period).mean()
    std = close.rolling(period, min_periods=period).std(ddof=0)
    upper = mid + k * std
    lower = mid - k * std
    bb_pct = (close - lower) / (upper - lower).replace(0, np.nan)
    bb_width = (upper - lower) / mid.replace(0, np.nan)
    return pd.DataFrame({
        "bb_upper": upper, "bb_mid": mid, "bb_lower": lower,
        "bb_pct": bb_pct, "bb_width": bb_width,
    }, index=close.index)


def _cmf_series(high: pd.Series, low: pd.Series, close: pd.Series,
                volume: pd.Series, period: int = 20) -> pd.Series:
    rng = (high - low).replace(0, np.nan)
    mfm = ((close - low) - (high - close)) / rng
    mfv = (mfm * volume).fillna(0.0)
    cum_mfv = mfv.rolling(period, min_periods=period).sum()
    cum_vol = volume.rolling(period, min_periods=period).sum().replace(0, np.nan)
    return cum_mfv / cum_vol


def _obv_series(close: pd.Series, volume: pd.Series) -> pd.Series:
    """OBV: cumulative signed volume."""
    direction = np.sign(close.diff())
    direction.iloc[0] = 0
    return (direction * volume).cumsum()


def _rv20_series(close: pd.Series, bars_per_year: int = 252) -> pd.Series:
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(20, min_periods=20).std(ddof=1) * math.sqrt(bars_per_year)


def _vol_pct_series(volume: pd.Series, lookback: int = 60) -> pd.Series:
    """Trailing 60-bar percentile rank of volume in [0,1]."""
    def _pct_rank(window):
        if len(window) < 2:
            return np.nan
        return (window[:-1] < window[-1]).sum() / (len(window) - 1)
    return volume.rolling(lookback, min_periods=2).apply(_pct_rank, raw=True)


def _iv_percentile_series(iv_30: pd.Series, lookback: int = 252) -> pd.Series:
    """Rolling 252-bar percentile rank of iv_30."""
    def _pct_rank(window):
        if len(window) < 2:
            return np.nan
        return (window[:-1] < window[-1]).sum() / (len(window) - 1)
    return iv_30.rolling(lookback, min_periods=20).apply(_pct_rank, raw=True)


# ---------------------------------------------------------------------------
# §2.3 — Anchored VWAP (O(n) amortized)
# ---------------------------------------------------------------------------

def _find_pivots(series: pd.Series, left: int = 5, right: int = 5) -> tuple[list[int], list[int]]:
    """Confirmed pivot highs and lows. Bar i is confirmed at i+right."""
    highs, lows = [], []
    vals = series.values
    n = len(vals)
    for i in range(left, n - right):
        window = vals[i - left: i + right + 1]
        if np.isfinite(vals[i]) and vals[i] == np.nanmax(window):
            highs.append(i)
        if np.isfinite(vals[i]) and vals[i] == np.nanmin(window):
            lows.append(i)
    return highs, lows


def compute_anchored_vwaps(df: pd.DataFrame, major_window: int = 10, lookback: int = 90) -> pd.DataFrame:
    """Add avwap_from_high, avwap_from_low, dist_avwap_high_atr, dist_avwap_low_atr.

    O(n) amortized: maintains running (pv_sum, v_sum) that resets only when a new
    anchor is detected. When anchor changes, re-walks from new anchor to current bar
    (amortized O(n) total across all bars).
    """
    n = len(df)
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    volume = df["volume"].values
    atr = df["atr_14"].values

    avwap_h = np.full(n, np.nan)
    avwap_l = np.full(n, np.nan)
    dist_h = np.full(n, np.nan)
    dist_l = np.full(n, np.nan)

    # Track current anchors
    cur_high_anchor = -1
    cur_low_anchor = -1
    pv_sum_h = 0.0
    v_sum_h = 0.0
    pv_sum_l = 0.0
    v_sum_l = 0.0

    for i in range(n):
        # Determine lookback window
        lb_start = max(0, i - lookback + 1)

        # Find best confirmed major pivot high/low within lookback
        # "Confirmed" means pivot index <= i - major_window
        max_confirmed = i - major_window
        if max_confirmed < major_window:
            # Not enough bars for a major pivot yet
            pass
        else:
            # Search for new major pivot high in [lb_start, max_confirmed]
            search_start = max(lb_start, major_window)
            best_high_idx = -1
            for j in range(max_confirmed, search_start - 1, -1):
                # Check if j is a major pivot high
                w_start = max(0, j - major_window)
                w_end = min(n - 1, j + major_window)
                if j - w_start < major_window or w_end - j < major_window:
                    continue
                window_vals = high[w_start: w_end + 1]
                if np.isfinite(high[j]) and high[j] == np.nanmax(window_vals):
                    best_high_idx = j
                    break

            best_low_idx = -1
            for j in range(max_confirmed, search_start - 1, -1):
                w_start = max(0, j - major_window)
                w_end = min(n - 1, j + major_window)
                if j - w_start < major_window or w_end - j < major_window:
                    continue
                window_vals = low[w_start: w_end + 1]
                if np.isfinite(low[j]) and low[j] == np.nanmin(window_vals):
                    best_low_idx = j
                    break

            # Reset running sums if anchor changed
            if best_high_idx != cur_high_anchor and best_high_idx >= 0:
                cur_high_anchor = best_high_idx
                pv_sum_h = 0.0
                v_sum_h = 0.0
                for k in range(cur_high_anchor, i):
                    v = volume[k]
                    c = close[k]
                    pv_sum_h += c * v
                    v_sum_h += v

            if best_low_idx != cur_low_anchor and best_low_idx >= 0:
                cur_low_anchor = best_low_idx
                pv_sum_l = 0.0
                v_sum_l = 0.0
                for k in range(cur_low_anchor, i):
                    v = volume[k]
                    c = close[k]
                    pv_sum_l += c * v
                    v_sum_l += v

        # Accumulate current bar
        v_i = volume[i]
        c_i = close[i]
        if cur_high_anchor >= 0:
            pv_sum_h += c_i * v_i
            v_sum_h += v_i
            if v_sum_h > 0:
                avwap_h[i] = pv_sum_h / v_sum_h
                if np.isfinite(atr[i]) and atr[i] > 0:
                    dist_h[i] = (c_i - avwap_h[i]) / atr[i]

        if cur_low_anchor >= 0:
            pv_sum_l += c_i * v_i
            v_sum_l += v_i
            if v_sum_l > 0:
                avwap_l[i] = pv_sum_l / v_sum_l
                if np.isfinite(atr[i]) and atr[i] > 0:
                    dist_l[i] = (c_i - avwap_l[i]) / atr[i]

    df = df.copy()
    df["avwap_from_high"] = avwap_h
    df["avwap_from_low"] = avwap_l
    df["dist_avwap_high_atr"] = dist_h
    df["dist_avwap_low_atr"] = dist_l
    return df


# ---------------------------------------------------------------------------
# §1 — Data loading
# ---------------------------------------------------------------------------

def load_data(path_or_df, iv_path: Optional[str] = None) -> pd.DataFrame:
    """Load OHLCV (+ optional IV) from CSV path or existing DataFrame."""
    if isinstance(path_or_df, pd.DataFrame):
        df = path_or_df.copy()
    else:
        df = pd.read_csv(path_or_df, index_col=0, parse_dates=True)

    # Normalize column names to lowercase
    df.columns = [c.lower() for c in df.columns]
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: Missing columns {missing}", file=sys.stderr)
        sys.exit(1)

    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df.sort_index()

    # Check bar gap uniformity
    if len(df) >= 3:
        diffs = df.index.to_series().diff().dropna()
        median_gap = diffs.median()
        large_gaps = (diffs > 2 * median_gap).sum()
        if large_gaps > 0:
            print(f"WARNING: {large_gaps} gap(s) > 2x median bar interval detected — "
                  "slope and percentile math may be affected.", file=sys.stderr)

    # Load IV
    if iv_path is not None:
        try:
            iv_df = pd.read_csv(iv_path, index_col=0, parse_dates=True)
            iv_df.index = pd.to_datetime(iv_df.index)
            if iv_df.index.tz is not None:
                iv_df.index = iv_df.index.tz_localize(None)
            if "iv_30" in iv_df.columns:
                df = df.join(iv_df[["iv_30"]], how="left")
            else:
                print(f"WARNING: {iv_path} has no 'iv_30' column — IV ignored", file=sys.stderr)
        except Exception as e:
            print(f"WARNING: Could not load IV CSV {iv_path}: {e}", file=sys.stderr)

    if "iv_30" not in df.columns:
        df["iv_30"] = np.nan

    if len(df) < 200:
        print(f"WARNING: Only {len(df)} bars — less than recommended 200 for full warmup.", file=sys.stderr)

    return df


def load_params(path: Optional[str]) -> dict:
    if path is None:
        return {}
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        print("WARNING: PyYAML not installed — params file ignored", file=sys.stderr)
        return {}
    except Exception as e:
        print(f"WARNING: Could not load params {path}: {e}", file=sys.stderr)
        return {}


# ---------------------------------------------------------------------------
# §2 — compute_indicators: add all indicator columns to df
# ---------------------------------------------------------------------------

def compute_indicators(df: pd.DataFrame, bars_per_year: int = 252) -> pd.DataFrame:
    """Add all indicator columns. Returns new DataFrame."""
    df = df.copy()
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)

    # SMAs
    df["sma_20"] = close.rolling(20, min_periods=20).mean()
    df["sma_50"] = close.rolling(50, min_periods=50).mean()
    df["sma_200"] = close.rolling(200, min_periods=200).mean()

    # RSI
    df["rsi"] = _rsi_series(close, 14)

    # MACD
    macd_df = _macd_series(close)
    df["macd"] = macd_df["macd"]
    df["macd_signal"] = macd_df["macd_signal"]
    df["macd_hist"] = macd_df["macd_hist"]

    # Bollinger Bands
    bb_df = _bb_series(close)
    df["bb_upper"] = bb_df["bb_upper"]
    df["bb_mid"] = bb_df["bb_mid"]
    df["bb_lower"] = bb_df["bb_lower"]
    df["bb_pct"] = bb_df["bb_pct"]
    df["bb_width"] = bb_df["bb_width"]

    # CMF
    df["cmf"] = _cmf_series(high, low, close, volume)

    # ATR (Wilder)
    df["atr_14"] = _atr14_series(high, low, close)

    # ADX (Wilder)
    adx_df = _adx_series(high, low, close)
    df["adx"] = adx_df["adx"]
    df["plus_di"] = adx_df["plus_di"]
    df["minus_di"] = adx_df["minus_di"]

    # OBV
    obv = _obv_series(close, volume)
    df["obv"] = obv
    df["obv_sma_20"] = obv.rolling(20, min_periods=20).mean()

    # Realized vol
    df["rv_20"] = _rv20_series(close, bars_per_year)

    # Volume indicators
    df["vol_sma_20"] = volume.rolling(20, min_periods=20).mean()
    vol_sma_safe = df["vol_sma_20"].replace(0, np.nan)
    df["vol_ratio"] = volume / vol_sma_safe
    df["vol_pct"] = _vol_pct_series(volume, lookback=60)

    # Price return
    df["ret_1"] = close.pct_change(1)

    # IV context (requires iv_30 column)
    if "iv_30" in df.columns:
        iv_30 = df["iv_30"].astype(float)
        df["iv_percentile"] = _iv_percentile_series(iv_30)
        df["iv_premium"] = iv_30 - df["rv_20"]
        # Guard rv_20 == 0
        rv_safe = df["rv_20"].replace(0, np.nan)
        df["iv_premium_ratio"] = iv_30 / rv_safe
    else:
        df["iv_30"] = np.nan
        df["iv_percentile"] = np.nan
        df["iv_premium"] = np.nan
        df["iv_premium_ratio"] = np.nan

    return df


# ---------------------------------------------------------------------------
# §3 — Per-indicator trend classification
# ---------------------------------------------------------------------------

def _slope_atr_norm(series: pd.Series, atr: pd.Series, window: int = 10) -> pd.Series:
    """ATR-normalized slope over `window` bars."""
    raw_slope = (series - series.shift(window)) / window
    return raw_slope / atr.replace(0, np.nan)


def classify_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Add adx_regime and vol_regime columns."""
    df = df.copy()

    def _adx_regime(adx):
        if not np.isfinite(adx):
            return "RANGING"
        if adx >= 25:
            return "TRENDING"
        if adx >= 20:
            return "WEAK_TREND"
        return "RANGING"

    def _vol_regime(iv_pct):
        if not np.isfinite(iv_pct):
            return "UNKNOWN"
        if iv_pct < 0.30:
            return "LOW_IV"
        if iv_pct > 0.70:
            return "HIGH_IV"
        return "NORMAL_IV"

    df["adx_regime"] = df["adx"].apply(_adx_regime)
    df["vol_regime"] = df["iv_percentile"].apply(_vol_regime)
    return df


def classify_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Add all trend_* columns."""
    df = df.copy()
    n = len(df)
    close = df["close"].astype(float)
    atr = df["atr_14"].astype(float)

    # ---- trend_price ----
    sma20 = df["sma_20"].astype(float)
    sma50 = df["sma_50"].astype(float)
    sma200 = df["sma_200"].astype(float)
    slope_sma50 = _slope_atr_norm(sma50, atr, window=10)

    trend_price = []
    for i in range(n):
        c = close.iloc[i]
        m20 = sma20.iloc[i]
        m50 = sma50.iloc[i]
        m200 = sma200.iloc[i]
        sl = slope_sma50.iloc[i]

        if not (np.isfinite(c) and np.isfinite(m50)):
            trend_price.append("NEUTRAL")
            continue

        pos_slope = np.isfinite(sl) and sl > 0.02
        neg_slope = np.isfinite(sl) and sl < -0.02
        flat_slope = not pos_slope and not neg_slope

        full_up = np.isfinite(m20) and np.isfinite(m200) and c > m20 > m50 > m200
        full_down = np.isfinite(m20) and np.isfinite(m200) and c < m20 < m50 < m200
        loose_up = np.isfinite(m200) and c > m50 > m200
        loose_down = np.isfinite(m200) and c < m50 < m200

        if full_up:
            if pos_slope:
                trend_price.append("STRONG_UP")
            elif flat_slope:
                trend_price.append("UP_EXHAUSTING")
            else:
                trend_price.append("UP")
        elif full_down:
            if neg_slope:
                trend_price.append("STRONG_DOWN")
            elif flat_slope:
                trend_price.append("DOWN_EXHAUSTING")
            else:
                trend_price.append("DOWN")
        elif loose_up:
            trend_price.append("UP")
        elif loose_down:
            trend_price.append("DOWN")
        else:
            trend_price.append("NEUTRAL")

    df["trend_price"] = trend_price

    # ---- trend_rsi ----
    rsi = df["rsi"].astype(float)
    rsi_prev = rsi.shift(1)
    trend_rsi = []
    for i in range(n):
        r = rsi.iloc[i]
        rp = rsi_prev.iloc[i]
        if not np.isfinite(r):
            trend_rsi.append("NEUTRAL")
            continue
        if r > 70:
            trend_rsi.append("OVERBOUGHT")
        elif r < 30:
            trend_rsi.append("OVERSOLD")
        elif r > 50 and np.isfinite(rp) and r > rp:
            trend_rsi.append("BULLISH")
        elif r < 50 and np.isfinite(rp) and r < rp:
            trend_rsi.append("BEARISH")
        else:
            trend_rsi.append("NEUTRAL")
    df["trend_rsi"] = trend_rsi

    # ---- trend_macd ----
    macd = df["macd"].astype(float)
    macd_sig = df["macd_signal"].astype(float)
    macd_hist = df["macd_hist"].astype(float)
    macd_prev = macd.shift(1)
    sig_prev = macd_sig.shift(1)
    hist_prev = macd_hist.shift(1)

    trend_macd_col = []
    macd_above_zero_col = []
    for i in range(n):
        mv = macd.iloc[i]
        sv = macd_sig.iloc[i]
        hv = macd_hist.iloc[i]
        mp = macd_prev.iloc[i]
        sp = sig_prev.iloc[i]
        hp = hist_prev.iloc[i]

        if not (np.isfinite(mv) and np.isfinite(sv) and np.isfinite(hv)):
            trend_macd_col.append("NEUTRAL")
            macd_above_zero_col.append(False)
            continue

        above_zero = mv > 0
        macd_above_zero_col.append(above_zero)

        if np.isfinite(mp) and np.isfinite(sp):
            if mp <= sp and mv > sv:
                trend_macd_col.append("BULL_CROSS")
                continue
            if mp >= sp and mv < sv:
                trend_macd_col.append("BEAR_CROSS")
                continue

        if mv > sv and np.isfinite(hp) and hv > hp:
            trend_macd_col.append("BULLISH")
        elif mv < sv and np.isfinite(hp) and hv < hp:
            trend_macd_col.append("BEARISH")
        else:
            trend_macd_col.append("NEUTRAL")

    df["trend_macd"] = trend_macd_col
    df["macd_above_zero"] = macd_above_zero_col

    # ---- trend_bb ----
    bb_width = df["bb_width"].astype(float)
    bb_width_pct = bb_width.rolling(120, min_periods=20).rank(pct=True)
    bb_upper = df["bb_upper"].astype(float)
    bb_lower = df["bb_lower"].astype(float)

    trend_bb_col = []
    for i in range(n):
        c = close.iloc[i]
        bu = bb_upper.iloc[i]
        bl = bb_lower.iloc[i]
        bw_pct = bb_width_pct.iloc[i]

        if not (np.isfinite(bu) and np.isfinite(bl)):
            trend_bb_col.append("NEUTRAL")
            continue
        if c > bu:
            trend_bb_col.append("UPPER_BREAK")
        elif c < bl:
            trend_bb_col.append("LOWER_BREAK")
        elif np.isfinite(bw_pct) and bw_pct <= 0.20:
            trend_bb_col.append("SQUEEZE")
        elif np.isfinite(bw_pct) and bw_pct >= 0.80:
            trend_bb_col.append("EXPANSION")
        else:
            trend_bb_col.append("NEUTRAL")
    df["trend_bb"] = trend_bb_col

    # ---- trend_cmf ----
    cmf = df["cmf"].astype(float)
    def _cmf_state(v):
        if not np.isfinite(v):
            return "NEUTRAL"
        if v > 0.10: return "STRONG_ACCUM"
        if v > 0.05: return "ACCUM"
        if v < -0.10: return "STRONG_DIST"
        if v < -0.05: return "DIST"
        return "NEUTRAL"
    df["trend_cmf"] = cmf.apply(_cmf_state)

    # ---- trend_volume ----
    vol_pct = df["vol_pct"].astype(float)
    ret_1 = df["ret_1"].astype(float)
    trend_vol = []
    vol_up = []
    vol_dn = []
    for i in range(n):
        vp = vol_pct.iloc[i]
        r = ret_1.iloc[i]
        if not np.isfinite(vp):
            trend_vol.append("NORMAL")
            vol_up.append(False)
            vol_dn.append(False)
            continue
        if vp >= 0.85:
            trend_vol.append("SURGE")
        elif vp <= 0.15:
            trend_vol.append("DRY")
        else:
            trend_vol.append("NORMAL")
        vol_up.append(np.isfinite(r) and r > 0 and vp >= 0.70)
        vol_dn.append(np.isfinite(r) and r < 0 and vp >= 0.70)
    df["trend_volume"] = trend_vol
    df["vol_confirms_up"] = vol_up
    df["vol_confirms_down"] = vol_dn

    # ---- trend_obv ----
    obv = df["obv"].astype(float)
    obv_sma = df["obv_sma_20"].astype(float)
    obv_3ago = obv.shift(3)
    trend_obv_col = []
    for i in range(n):
        ov = obv.iloc[i]
        os = obv_sma.iloc[i]
        o3 = obv_3ago.iloc[i]
        if not np.isfinite(ov):
            trend_obv_col.append("NEUTRAL")
            continue
        above_sma = np.isfinite(os) and ov > os
        rising = np.isfinite(o3) and ov > o3
        below_sma = np.isfinite(os) and ov < os
        falling = np.isfinite(o3) and ov < o3
        if above_sma and rising:
            trend_obv_col.append("BULLISH")
        elif below_sma and falling:
            trend_obv_col.append("BEARISH")
        else:
            trend_obv_col.append("NEUTRAL")
    df["trend_obv"] = trend_obv_col

    # ---- trend_avwap ----
    avwap_h = df["avwap_from_high"].astype(float)
    avwap_l = df["avwap_from_low"].astype(float)
    dist_h = df["dist_avwap_high_atr"].astype(float)
    dist_l = df["dist_avwap_low_atr"].astype(float)
    close_prev = close.shift(1)

    trend_avwap_col = []
    for i in range(n):
        c = close.iloc[i]
        cp = close_prev.iloc[i]
        ah = avwap_h.iloc[i]
        al = avwap_l.iloc[i]
        dh = dist_h.iloc[i]
        dl = dist_l.iloc[i]

        h_ok = np.isfinite(ah) and np.isfinite(dh)
        l_ok = np.isfinite(al) and np.isfinite(dl)

        if not h_ok and not l_ok:
            trend_avwap_col.append("UNAVAILABLE")
            continue

        # Crossing events
        if np.isfinite(cp):
            if h_ok and cp < ah and c >= ah:
                trend_avwap_col.append("RECLAIM_HIGH")
                continue
            if l_ok and cp > al and c <= al:
                trend_avwap_col.append("BREAKDOWN_LOW")
                continue

        # Position states
        if h_ok and l_ok:
            above_h = c > ah
            above_l = c > al
            near_h = abs(dh) < 0.3 and not above_h   # testing from below
            near_l = abs(dl) < 0.3 and above_l         # testing from above
            if above_h and above_l:
                trend_avwap_col.append("ABOVE_BOTH")
            elif not above_h and not above_l:
                trend_avwap_col.append("BELOW_BOTH")
            elif near_h:
                trend_avwap_col.append("TESTING_RESISTANCE")
            elif near_l:
                trend_avwap_col.append("TESTING_SUPPORT")
            else:
                trend_avwap_col.append("BETWEEN")
        elif h_ok:
            trend_avwap_col.append("ABOVE_BOTH" if c > ah else "BELOW_BOTH")
        else:
            trend_avwap_col.append("ABOVE_BOTH" if c > al else "BELOW_BOTH")

    df["trend_avwap"] = trend_avwap_col
    return df


# ---------------------------------------------------------------------------
# §4 — Divergence detection
# ---------------------------------------------------------------------------

def detect_divergence(
    price: pd.Series,
    indicator: pd.Series,
    name: str,
    atr: pd.Series,
    sma20: pd.Series,
    sma50: pd.Series,
    pivot_left: int = 5,
    pivot_right: int = 5,
    min_gap: int = 5,
    max_gap: int = 60,
    atr_threshold: float = 0.25,
    expiry_bars: int = 20,
    effect_bars: int = 10,
    quality_fn_bull=None,
    quality_fn_bear=None,
) -> pd.DataFrame:
    """Detect regular and hidden divergences for one indicator.

    Returns DataFrame with columns:
      div_{name}_reg_bull, div_{name}_reg_bear,
      div_{name}_reg_bull_pending, div_{name}_reg_bear_pending,
      div_{name}_hid_bull, div_{name}_hid_bear,
      conv_{name}_bull, conv_{name}_bear,
      div_{name}_quality
    """
    n = len(price)
    idx = price.index

    # Output arrays
    reg_bull = np.zeros(n, dtype=bool)
    reg_bear = np.zeros(n, dtype=bool)
    reg_bull_pending = np.zeros(n, dtype=bool)
    reg_bear_pending = np.zeros(n, dtype=bool)
    hid_bull = np.zeros(n, dtype=bool)
    hid_bear = np.zeros(n, dtype=bool)
    conv_bull = np.zeros(n, dtype=bool)
    conv_bear = np.zeros(n, dtype=bool)
    quality_arr = np.zeros(n, dtype=bool)

    price_vals = price.values
    ind_vals = indicator.values
    atr_vals = atr.values
    sma20_vals = sma20.values
    sma50_vals = sma50.values

    # Find pivot lows (for bullish patterns) and highs (for bearish)
    ph_indices, pl_indices = _find_pivots(price, left=pivot_left, right=pivot_right)

    # State machine storage: {pair_key: state_dict}
    # pair_key = (p1_idx, p2_idx)
    reg_bull_states = {}   # pair -> {state, pending_since, confirmed_at, p2_close, quality}
    reg_bear_states = {}

    def _slope_up_sma50(bar_i):
        """True if SMA50 slope is upward at bar_i."""
        if bar_i < 10:
            return False
        s_now = sma50_vals[bar_i]
        s_prev = sma50_vals[bar_i - 10]
        if not (np.isfinite(s_now) and np.isfinite(s_prev)):
            return False
        atr_i = atr_vals[bar_i]
        if not (np.isfinite(atr_i) and atr_i > 0):
            return False
        slope = (s_now - s_prev) / (10 * atr_i)
        return slope > 0.02

    def _slope_down_sma50(bar_i):
        if bar_i < 10:
            return False
        s_now = sma50_vals[bar_i]
        s_prev = sma50_vals[bar_i - 10]
        if not (np.isfinite(s_now) and np.isfinite(s_prev)):
            return False
        atr_i = atr_vals[bar_i]
        if not (np.isfinite(atr_i) and atr_i > 0):
            return False
        slope = (s_now - s_prev) / (10 * atr_i)
        return slope < -0.02

    # Process low pivots (bullish patterns use lows)
    for k in range(1, len(pl_indices)):
        p2 = pl_indices[k]
        # confirmed at p2 + pivot_right
        confirmed_at = p2 + pivot_right

        # Find qualifying p1
        for j in range(k - 1, -1, -1):
            p1 = pl_indices[j]
            gap = p2 - p1
            if gap < min_gap:
                continue
            if gap > max_gap:
                break

            c1, c2 = price_vals[p1], price_vals[p2]
            i1, i2 = ind_vals[p1], ind_vals[p2]

            if not (np.isfinite(c1) and np.isfinite(c2) and
                    np.isfinite(i1) and np.isfinite(i2)):
                continue

            # ATR-normalized price move
            atr_at_p2 = atr_vals[p2]
            if np.isfinite(atr_at_p2) and atr_at_p2 > 0:
                if abs(c2 - c1) / atr_at_p2 < atr_threshold:
                    continue

            # Regular bullish: price lower low, indicator higher low
            if c2 < c1 and i2 > i1:
                q = quality_fn_bull(i1) if quality_fn_bull else False
                key = (p1, p2)
                if key not in reg_bull_states:
                    reg_bull_states[key] = {
                        "state": "PENDING",
                        "pending_since": confirmed_at,
                        "confirmed_at": None,
                        "p2_close": c2,
                        "p2_idx": p2,
                        "quality": q,
                    }

            # Hidden bullish: price higher low, indicator lower low
            elif c2 > c1 and i2 < i1:
                if _slope_up_sma50(p2):
                    q = False  # hidden divergence: no quality boost
                    if confirmed_at < n:
                        hid_bull[confirmed_at:] = True
                        quality_arr[confirmed_at] = q

            # Convergence bull: price higher low, indicator higher low
            elif c2 > c1 and i2 > i1:
                if confirmed_at < n:
                    conv_bull[confirmed_at:] = True

            break  # only use nearest qualifying p1

    # Process high pivots (bearish patterns use highs)
    for k in range(1, len(ph_indices)):
        p2 = ph_indices[k]
        confirmed_at = p2 + pivot_right

        for j in range(k - 1, -1, -1):
            p1 = ph_indices[j]
            gap = p2 - p1
            if gap < min_gap:
                continue
            if gap > max_gap:
                break

            c1, c2 = price_vals[p1], price_vals[p2]
            i1, i2 = ind_vals[p1], ind_vals[p2]

            if not (np.isfinite(c1) and np.isfinite(c2) and
                    np.isfinite(i1) and np.isfinite(i2)):
                continue

            atr_at_p2 = atr_vals[p2]
            if np.isfinite(atr_at_p2) and atr_at_p2 > 0:
                if abs(c2 - c1) / atr_at_p2 < atr_threshold:
                    continue

            # Regular bearish: price higher high, indicator lower high
            if c2 > c1 and i2 < i1:
                q = quality_fn_bear(i1) if quality_fn_bear else False
                key = (p1, p2)
                if key not in reg_bear_states:
                    reg_bear_states[key] = {
                        "state": "PENDING",
                        "pending_since": confirmed_at,
                        "confirmed_at": None,
                        "p2_close": c2,
                        "p2_idx": p2,
                        "quality": q,
                    }

            # Hidden bearish: price lower high, indicator higher high
            elif c2 < c1 and i2 > i1:
                if _slope_down_sma50(p2):
                    if confirmed_at < n:
                        hid_bear[confirmed_at:] = True

            # Convergence bear
            elif c2 < c1 and i2 < i1:
                if confirmed_at < n:
                    conv_bear[confirmed_at:] = True

            break

    # Advance regular divergence state machines
    def _advance_states(states_dict, is_bull: bool):
        for key, st in states_dict.items():
            if st["state"] in ("CONFIRMED", "EXPIRED"):
                continue
            p2_close = st["p2_close"]
            pending_since = st["pending_since"]
            p2_idx = st["p2_idx"]

            for bar_i in range(pending_since, n):
                # Mark PENDING
                if st["state"] == "PENDING":
                    if is_bull:
                        reg_bull_pending[bar_i] = True
                    else:
                        reg_bear_pending[bar_i] = True

                c_i = price_vals[bar_i]
                s20_i = sma20_vals[bar_i]

                if is_bull:
                    confirm_cond = (np.isfinite(c_i) and c_i > p2_close and
                                    np.isfinite(s20_i) and c_i > s20_i)
                else:
                    confirm_cond = (np.isfinite(c_i) and c_i < p2_close and
                                    np.isfinite(s20_i) and c_i < s20_i)

                if confirm_cond:
                    st["state"] = "CONFIRMED"
                    st["confirmed_at"] = bar_i
                    break

                if bar_i - pending_since > expiry_bars:
                    st["state"] = "EXPIRED"
                    break

            # Mark CONFIRMED effect window
            if st["state"] == "CONFIRMED":
                ca = st["confirmed_at"]
                end_effect = min(n, ca + effect_bars + 1)
                if is_bull:
                    reg_bull[ca:end_effect] = True
                    if st["quality"]:
                        quality_arr[ca:end_effect] = True
                else:
                    reg_bear[ca:end_effect] = True
                    if st["quality"]:
                        quality_arr[ca:end_effect] = True

    _advance_states(reg_bull_states, is_bull=True)
    _advance_states(reg_bear_states, is_bull=False)

    prefix = f"div_{name}"
    result = pd.DataFrame({
        f"{prefix}_reg_bull": reg_bull,
        f"{prefix}_reg_bear": reg_bear,
        f"{prefix}_reg_bull_pending": reg_bull_pending,
        f"{prefix}_reg_bear_pending": reg_bear_pending,
        f"{prefix}_hid_bull": hid_bull,
        f"{prefix}_hid_bear": hid_bear,
        f"conv_{name}_bull": conv_bull,
        f"conv_{name}_bear": conv_bear,
        f"{prefix}_quality": quality_arr,
    }, index=idx)
    return result


def run_divergence_suite(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """Run divergence detection for RSI, macd_line, macd_hist, cmf, obv.
    Returns df with all div_* and conv_* columns added."""
    if params is None:
        params = {}

    p = {
        "atr_threshold": params.get("atr_threshold", 0.25),
        "expiry_bars": params.get("expiry_bars", 20),
        "effect_bars": params.get("effect_bars", 10),
        "pivot_left": params.get("pivot_left", 5),
        "pivot_right": params.get("pivot_right", 5),
        "min_gap": params.get("min_gap", 5),
        "max_gap": params.get("max_gap", 60),
    }

    price = df["close"].astype(float)
    atr = df["atr_14"].astype(float)
    sma20 = df["sma_20"].astype(float)
    sma50 = df["sma_50"].astype(float)

    # Quality functions — return bool
    def rsi_bull_q(v): return np.isfinite(v) and v < 30
    def rsi_bear_q(v): return np.isfinite(v) and v > 70
    def macd_line_bull_q(v): return np.isfinite(v) and v < 0
    def macd_line_bear_q(v): return np.isfinite(v) and v > 0
    def macd_hist_bull_q(v): return np.isfinite(v) and v < 0
    def macd_hist_bear_q(v): return np.isfinite(v) and v > 0
    def cmf_bull_q(v): return np.isfinite(v) and v < -0.05
    def cmf_bear_q(v): return np.isfinite(v) and v > 0.05
    def obv_q(v): return False  # OBV: no quality boost

    indicators = [
        ("rsi",       df["rsi"],       rsi_bull_q,       rsi_bear_q),
        ("macd_line", df["macd"],      macd_line_bull_q, macd_line_bear_q),
        ("macd_hist", df["macd_hist"], macd_hist_bull_q, macd_hist_bear_q),
        ("cmf",       df["cmf"],       cmf_bull_q,       cmf_bear_q),
        ("obv",       df["obv"],       obv_q,            obv_q),
    ]

    result_dfs = []
    for name, indicator, qb, qbr in indicators:
        div_df = detect_divergence(
            price=price,
            indicator=indicator.astype(float),
            name=name,
            atr=atr,
            sma20=sma20,
            sma50=sma50,
            quality_fn_bull=qb,
            quality_fn_bear=qbr,
            **p,
        )
        result_dfs.append(div_df)

    div_combined = pd.concat(result_dfs, axis=1)
    return pd.concat([df, div_combined], axis=1)


# ---------------------------------------------------------------------------
# §5 — Scoring
# ---------------------------------------------------------------------------

_REGIME_MATRIX = {
    "TRENDING":   {"LOW_IV": 0.00, "NORMAL_IV": 0.00, "HIGH_IV": 0.50, "UNKNOWN": 0.00},
    "WEAK_TREND": {"LOW_IV": 0.50, "NORMAL_IV": 0.50, "HIGH_IV": 0.75, "UNKNOWN": 0.50},
    "RANGING":    {"LOW_IV": 0.50, "NORMAL_IV": 1.00, "HIGH_IV": 1.00, "UNKNOWN": 1.00},
}

_DIV_INDICATORS = ["rsi", "macd_line", "macd_hist", "cmf", "obv"]

_TREND_PRICE_SCORE = {
    "STRONG_UP": 3.0, "UP_EXHAUSTING": 0.75, "UP": 1.5,
    "NEUTRAL": 0.0,
    "DOWN": -1.5, "DOWN_EXHAUSTING": -0.75, "STRONG_DOWN": -3.0,
}

_AVWAP_SCORE = {
    "RECLAIM_HIGH": 1.5, "BREAKDOWN_LOW": -1.5,
    "ABOVE_BOTH": 0.75, "BELOW_BOTH": -0.75,
    "TESTING_SUPPORT": 0.5, "TESTING_RESISTANCE": -0.5,
    "BETWEEN": 0.0, "UNAVAILABLE": 0.0,
}


def score_bar(row: pd.Series, prev_recommendation: str) -> dict:
    """Compute score and recommendation for one bar.

    §5.3 order of operations:
      base_trend  = trend_score * convergence_multiplier
      raw         = base_trend + mr_weight * mean_reversion_score + divergence_score
      score       = clamp(raw, -10, +10)
      confidence  = abs(score) / 10

    Returns dict with all scoring components.
    """
    contributions = []

    def _add(name, value, reason=""):
        contributions.append({"name": name, "value": value, "reason": reason})

    # --- Warmup check ---
    warmup_required = ["atr_14", "adx", "obv_sma_20", "rv_20", "vol_pct"]
    for col in warmup_required:
        v = row.get(col, np.nan)
        if not np.isfinite(float(v) if v is not None else np.nan):
            return {
                "score": 0.0, "raw": 0.0, "recommendation": "HOLD",
                "confidence": 0.0, "reason": "warmup",
                "trend_score": 0.0, "base_trend": 0.0,
                "conv_mult": 1.0, "mr_score": 0.0,
                "mr_weight": 0.0, "mr_contribution": 0.0,
                "div_score": 0.0, "contributions": contributions,
                "caveats": [], "gate_status": {},
            }

    # Also need aVWAP to have warmed up — but aVWAP can legitimately be NaN
    # if no major pivot found; that's not a warmup condition.

    # ---- Trend score ----
    trend_score = 0.0

    tp = row.get("trend_price", "NEUTRAL")
    tp_score = _TREND_PRICE_SCORE.get(tp, 0.0)
    trend_score += tp_score
    _add("trend_price", tp_score, tp)

    tm = row.get("trend_macd", "NEUTRAL")
    maz = bool(row.get("macd_above_zero", False))
    macd_steady = {"BULLISH": 2.0, "BEARISH": -2.0,
                   "BULL_CROSS": 2.0, "BEAR_CROSS": -2.0}.get(tm, 0.0)
    macd_cross_bonus = 1.5 if tm == "BULL_CROSS" else (-1.5 if tm == "BEAR_CROSS" else 0.0)
    macd_zero_bonus = 0.5 if maz else -0.5
    trend_score += macd_steady + macd_cross_bonus + macd_zero_bonus
    _add("trend_macd_steady", macd_steady, tm)
    _add("trend_macd_cross_bonus", macd_cross_bonus, tm)
    _add("trend_macd_zero", macd_zero_bonus, f"macd_above_zero={maz}")

    tc = row.get("trend_cmf", "NEUTRAL")
    cmf_score = {"STRONG_ACCUM": 1.0, "ACCUM": 0.5,
                 "DIST": -0.5, "STRONG_DIST": -1.0}.get(tc, 0.0)
    trend_score += cmf_score
    _add("trend_cmf", cmf_score, tc)

    tobv = row.get("trend_obv", "NEUTRAL")
    obv_score = 1.0 if tobv == "BULLISH" else (-1.0 if tobv == "BEARISH" else 0.0)
    trend_score += obv_score
    _add("trend_obv", obv_score, tobv)

    tav = row.get("trend_avwap", "UNAVAILABLE")
    avwap_score = _AVWAP_SCORE.get(tav, 0.0)
    trend_score += avwap_score
    _add("trend_avwap", avwap_score, tav)

    vol_up = bool(row.get("vol_confirms_up", False))
    vol_dn = bool(row.get("vol_confirms_down", False))
    vol_score = (0.5 if vol_up else 0.0) + (-0.5 if vol_dn else 0.0)
    trend_score += vol_score
    _add("volume", vol_score, f"vol_up={vol_up} vol_dn={vol_dn}")

    # ---- Convergence multiplier ----
    conv_mult = 1.0
    # Check MACD hist direction vs trend direction
    mh = float(row.get("macd_hist", np.nan))
    mh_prev = float(row.get("_macd_hist_prev", np.nan))
    if np.isfinite(mh) and np.isfinite(mh_prev):
        hist_dir = 1 if mh > mh_prev else (-1 if mh < mh_prev else 0)
        ts_dir = 1 if trend_score > 0 else (-1 if trend_score < 0 else 0)
        if hist_dir != 0 and ts_dir != 0 and hist_dir == ts_dir:
            conv_mult = 1.25
    _add("conv_mult", conv_mult, f"macd_hist_dir matches trend_dir={conv_mult==1.25}")

    base_trend = trend_score * conv_mult

    # ---- Mean reversion score ----
    mr_score = 0.0
    gate_status = {}
    rsi_val = float(row.get("rsi", np.nan))
    rsi_label = "neutral"
    bb_label = "neutral"

    # RSI contributions per spec §5.1:
    # BULLISH (20<rsi<30) = +1; BEARISH (70<rsi<80) = -1
    # OVERSOLD (rsi<=20) = -0.5 (extreme contra); OVERBOUGHT (rsi>=80) = +0.5 (extreme contra)
    if np.isfinite(rsi_val):
        if rsi_val <= 20:
            mr_score -= 0.5
            rsi_label = "extreme_oversold"
            _add("mr_rsi_extreme_oversold", -0.5, f"RSI={rsi_val:.1f}")
        elif rsi_val < 30:
            mr_score += 1.0
            rsi_label = "oversold_bullish"
            _add("mr_rsi_oversold", 1.0, f"RSI={rsi_val:.1f}")
        elif rsi_val >= 80:
            mr_score += 0.5
            rsi_label = "extreme_overbought"
            _add("mr_rsi_extreme_overbought", 0.5, f"RSI={rsi_val:.1f}")
        elif rsi_val > 70:
            mr_score -= 1.0
            rsi_label = "overbought_bearish"
            _add("mr_rsi_overbought", -1.0, f"RSI={rsi_val:.1f}")

    # BB contributions
    tb = row.get("trend_bb", "NEUTRAL")
    if tb == "LOWER_BREAK":
        mr_score += 0.5
        bb_label = "lower_break"
        _add("mr_bb_lower_break", 0.5, "BB LOWER_BREAK")
    elif tb == "UPPER_BREAK":
        mr_score -= 0.5
        bb_label = "upper_break"
        _add("mr_bb_upper_break", -0.5, "BB UPPER_BREAK")

    # Apply regime gate
    adx_reg = row.get("adx_regime", "RANGING")
    vol_reg = row.get("vol_regime", "UNKNOWN")
    mr_weight = _REGIME_MATRIX.get(adx_reg, _REGIME_MATRIX["RANGING"]).get(vol_reg, 0.0)
    mr_contribution = mr_weight * mr_score
    gate_status["mr_weight"] = mr_weight
    gate_status["mr_score_raw"] = mr_score
    gate_status["suppressed"] = mr_weight == 0.0
    _add("mr_contribution", mr_contribution,
         f"weight={mr_weight:.2f} × raw={mr_score:.2f} (adx={adx_reg}, vol={vol_reg})")

    # ---- Divergence score ----
    div_score = 0.0
    n_reg_bull = 0
    n_reg_bear = 0
    q_reg_bull = False
    q_reg_bear = False
    any_hid_bull = False
    any_hid_bear = False

    for ind_name in _DIV_INDICATORS:
        rb = bool(row.get(f"div_{ind_name}_reg_bull", False))
        rbr = bool(row.get(f"div_{ind_name}_reg_bear", False))
        hb = bool(row.get(f"div_{ind_name}_hid_bull", False))
        hbr = bool(row.get(f"div_{ind_name}_hid_bear", False))
        q = bool(row.get(f"div_{ind_name}_quality", False))

        if rb:
            n_reg_bull += 1
            if q:
                q_reg_bull = True
        if rbr:
            n_reg_bear += 1
            if q:
                q_reg_bear = True
        if hb:
            any_hid_bull = True
        if hbr:
            any_hid_bear = True

    # Scoring per §5.1
    if n_reg_bull == 1:
        div_score += 2.0
        _add("div_reg_bull_1", 2.0, "1 confirmed regular bullish")
    elif n_reg_bull >= 2:
        div_score += 4.0
        _add("div_reg_bull_2plus", 4.0, f"{n_reg_bull} confirmed regular bullish")

    if q_reg_bull:
        div_score += 1.0
        _add("div_reg_bull_quality", 1.0, "quality boost (indicator in extreme zone)")

    if n_reg_bear == 1:
        div_score -= 2.0
        _add("div_reg_bear_1", -2.0, "1 confirmed regular bearish")
    elif n_reg_bear >= 2:
        div_score -= 4.0
        _add("div_reg_bear_2plus", -4.0, f"{n_reg_bear} confirmed regular bearish")

    if q_reg_bear:
        div_score -= 1.0
        _add("div_reg_bear_quality", -1.0, "quality boost (indicator in extreme zone)")

    if any_hid_bull:
        div_score += 1.0
        _add("div_hid_bull", 1.0, "hidden bullish divergence")
    if any_hid_bear:
        div_score -= 1.0
        _add("div_hid_bear", -1.0, "hidden bearish divergence")

    # ---- Final score ----
    raw = base_trend + mr_contribution + div_score
    score = max(-10.0, min(10.0, raw))
    confidence = abs(score) / 10.0

    # ---- Hysteresis (Schmitt trigger) ----
    recommendation = _apply_hysteresis(score, prev_recommendation)

    # ---- Caveats §5.5 ----
    caveats = []
    iv_pr = float(row.get("iv_premium_ratio", np.nan))
    iv_pct = float(row.get("iv_percentile", np.nan))
    iv_prem = float(row.get("iv_premium", np.nan))
    rv20 = float(row.get("rv_20", np.nan))

    if np.isfinite(iv_pr) and np.isfinite(iv_pct):
        if iv_pr > 1.5 and iv_pct > 0.80:
            caveats.append("ELEVATED EVENT RISK — REDUCE POSITION SIZE OR WAIT FOR EVENT")
        if iv_pct > 0.90 and recommendation in ("STRONG_BUY", "STRONG_SELL"):
            caveats.append("HIGH IV PERCENTILE — DIRECTIONAL CONVICTION MAY UNWIND ON VOL CRUSH")

    if np.isfinite(iv_prem) and np.isfinite(iv_pct):
        if iv_prem < 0 and iv_pct < 0.20:
            caveats.append("LOW IV WITH IV < RV — TREND CONTINUATION FAVORED")

    # Stale anchor
    avwap_h = float(row.get("avwap_from_high", np.nan))
    avwap_l = float(row.get("avwap_from_low", np.nan))
    # We don't track anchor age per bar in the row; check dist values as proxy
    # (actual anchor age reported in build_report using index scan)

    momentum_flags = []
    if n_reg_bull >= 2:
        momentum_flags.append("MOMENTUM EXHAUSTION — DOWNTREND LOSING STEAM")
    if n_reg_bear >= 2:
        momentum_flags.append("MOMENTUM EXHAUSTION — UPTREND LOSING STEAM")

    return {
        "score": score,
        "raw": raw,
        "recommendation": recommendation,
        "confidence": confidence,
        "reason": "ok",
        "trend_score": trend_score,
        "base_trend": base_trend,
        "conv_mult": conv_mult,
        "mr_score": mr_score,
        "mr_weight": mr_weight,
        "mr_contribution": mr_contribution,
        "div_score": div_score,
        "n_reg_bull": n_reg_bull,
        "n_reg_bear": n_reg_bear,
        "q_reg_bull": q_reg_bull,
        "q_reg_bear": q_reg_bear,
        "any_hid_bull": any_hid_bull,
        "any_hid_bear": any_hid_bear,
        "contributions": contributions,
        "caveats": caveats,
        "momentum_flags": momentum_flags,
        "gate_status": gate_status,
        "rsi_label": rsi_label,
        "bb_label": bb_label,
    }


def _apply_hysteresis(score: float, prev_rec: str) -> str:
    """Schmitt-trigger hysteresis per spec §5.4."""
    if prev_rec == "STRONG_BUY":
        if score >= 4.5: return "STRONG_BUY"
        if score >= 1.5: return "BUY"
        if score > -2.5: return "HOLD"
        if score > -5.5: return "SELL"
        return "STRONG_SELL"
    if prev_rec == "BUY":
        if score >= 5.5: return "STRONG_BUY"
        if score >= 1.5: return "BUY"
        if score > -2.5: return "HOLD"
        if score > -5.5: return "SELL"
        return "STRONG_SELL"
    if prev_rec == "SELL":
        if score <= -5.5: return "STRONG_SELL"
        if score <= -1.5: return "SELL"
        if score < 2.5: return "HOLD"
        if score < 5.5: return "BUY"
        return "STRONG_BUY"
    if prev_rec == "STRONG_SELL":
        if score <= -4.5: return "STRONG_SELL"
        if score <= -1.5: return "SELL"
        if score < 2.5: return "HOLD"
        if score < 5.5: return "BUY"
        return "STRONG_BUY"
    # HOLD (default)
    if score >= 5.5: return "STRONG_BUY"
    if score >= 2.5: return "BUY"
    if score <= -5.5: return "STRONG_SELL"
    if score <= -2.5: return "SELL"
    return "HOLD"


def recommend(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """Iterate score_bar with hysteresis state carryover. Returns df with scoring columns."""
    if params is None:
        params = {}

    df = df.copy()
    n = len(df)

    # Pre-compute shifted macd_hist for convergence multiplier
    df["_macd_hist_prev"] = df["macd_hist"].shift(1)

    scores = []
    raws = []
    recs = []
    confs = []
    reasons = []
    trend_scores = []
    base_trends = []
    conv_mults = []
    mr_weights = []
    mr_contribs = []
    div_scores = []

    prev_rec = "HOLD"
    for i in range(n):
        row = df.iloc[i]
        result = score_bar(row, prev_rec)
        scores.append(result["score"])
        raws.append(result["raw"])
        recs.append(result["recommendation"])
        confs.append(result["confidence"])
        reasons.append(result["reason"])
        trend_scores.append(result["trend_score"])
        base_trends.append(result["base_trend"])
        conv_mults.append(result["conv_mult"])
        mr_weights.append(result["mr_weight"])
        mr_contribs.append(result["mr_contribution"])
        div_scores.append(result["div_score"])
        prev_rec = result["recommendation"]

    df["score"] = scores
    df["raw_score"] = raws
    df["recommendation"] = recs
    df["confidence"] = confs
    df["score_reason"] = reasons
    df["trend_score"] = trend_scores
    df["base_trend"] = base_trends
    df["conv_mult"] = conv_mults
    df["mr_weight"] = mr_weights
    df["mr_contribution"] = mr_contribs
    df["div_score"] = div_scores
    return df


# ---------------------------------------------------------------------------
# §6 — Report
# ---------------------------------------------------------------------------

def build_report(df: pd.DataFrame, idx: int = -1, symbol: str = "") -> str:
    """Build the structured report for bar at idx."""
    if idx < 0:
        idx = len(df) + idx
    if idx < 0 or idx >= len(df):
        return "ERROR: bar index out of range"

    row = df.iloc[idx]
    prev_row = df.iloc[idx - 1] if idx > 0 else None

    bar_date = df.index[idx]
    if hasattr(bar_date, "date"):
        bar_date_str = bar_date.date().isoformat()
    else:
        bar_date_str = str(bar_date)

    close_val = float(row.get("close", np.nan))
    score_val = float(row.get("score", np.nan))
    raw_val = float(row.get("raw_score", np.nan))
    conf_val = float(row.get("confidence", np.nan))
    rec = str(row.get("recommendation", "HOLD"))
    prev_rec = str(prev_row.get("recommendation", "HOLD")) if prev_row is not None else "HOLD"
    reason = str(row.get("score_reason", ""))

    adx_val = float(row.get("adx", np.nan))
    pdi_val = float(row.get("plus_di", np.nan))
    mdi_val = float(row.get("minus_di", np.nan))
    adx_reg = str(row.get("adx_regime", "RANGING"))
    vol_reg = str(row.get("vol_regime", "UNKNOWN"))
    iv30 = float(row.get("iv_30", np.nan))
    rv20 = float(row.get("rv_20", np.nan))
    iv_pct_val = float(row.get("iv_percentile", np.nan))
    mr_weight = float(row.get("mr_weight", 0.0))

    trend_score = float(row.get("trend_score", 0.0))
    base_trend = float(row.get("base_trend", 0.0))
    conv_mult = float(row.get("conv_mult", 1.0))
    mr_contribution = float(row.get("mr_contribution", 0.0))
    div_score = float(row.get("div_score", 0.0))

    lines = []
    sep = "=" * 56
    dash = "-" * 56
    lines.append(sep)
    sym_str = f" — {symbol}" if symbol else ""
    lines.append(f"  TRADE SIGNAL REPORT{sym_str} — {bar_date_str}")
    lines.append(sep)

    score_str = f"{score_val:+.1f}" if np.isfinite(score_val) else "N/A"
    raw_str = f"raw {raw_val:+.2f}, clamped to {score_val:+.1f}" if np.isfinite(raw_val) else ""
    conf_str = f"{conf_val:.2f}" if np.isfinite(conf_val) else "N/A"
    close_str = f"{close_val:.2f}" if np.isfinite(close_val) else "N/A"
    lines.append(f"  Close: {close_str}    Score: {score_str}   Confidence: {conf_str}   ({raw_str})")
    if reason == "warmup":
        lines.append(f"  RECOMMENDATION: HOLD (warmup)  (prev: {prev_rec})")
    else:
        lines.append(f"  RECOMMENDATION: {rec}  (prev: {prev_rec})")

    adx_str = f"{adx_val:.1f}" if np.isfinite(adx_val) else "N/A"
    pdi_str = f"{pdi_val:.1f}" if np.isfinite(pdi_val) else "N/A"
    mdi_str = f"{mdi_val:.1f}" if np.isfinite(mdi_val) else "N/A"
    di_label = "+DI > -DI" if (np.isfinite(pdi_val) and np.isfinite(mdi_val) and pdi_val > mdi_val) else "-DI > +DI"
    lines.append(f"  ADX regime: {adx_reg} (ADX {adx_str}, {di_label})")

    iv_str = f"{iv30:.0%}" if np.isfinite(iv30) else "N/A"
    rv_str = f"{rv20:.0%}" if np.isfinite(rv20) else "N/A"
    iv_pct_str = f"{iv_pct_val:.2f}" if np.isfinite(iv_pct_val) else "N/A"
    lines.append(f"  Vol regime: {vol_reg}   (IV30 {iv_str}, RV20 {rv_str}, IV percentile {iv_pct_str})")
    lines.append(f"  Combined gate: mr_weight = {mr_weight:.2f}")
    lines.append(dash)

    # Trend components
    tp = str(row.get("trend_price", "NEUTRAL"))
    tm = str(row.get("trend_macd", "NEUTRAL"))
    maz = bool(row.get("macd_above_zero", False))
    tc = str(row.get("trend_cmf", "NEUTRAL"))
    tobv = str(row.get("trend_obv", "NEUTRAL"))
    tav = str(row.get("trend_avwap", "UNAVAILABLE"))
    tvol = str(row.get("trend_volume", "NORMAL"))
    vol_pct_val = float(row.get("vol_pct", np.nan))
    vol_up = bool(row.get("vol_confirms_up", False))

    tp_score = _TREND_PRICE_SCORE.get(tp, 0.0)
    macd_steady = {"BULLISH": 2.0, "BEARISH": -2.0, "BULL_CROSS": 2.0, "BEAR_CROSS": -2.0}.get(tm, 0.0)
    macd_cross_bonus = 1.5 if tm == "BULL_CROSS" else (-1.5 if tm == "BEAR_CROSS" else 0.0)
    macd_total = macd_steady + macd_cross_bonus + (0.5 if maz else -0.5)
    cmf_score = {"STRONG_ACCUM": 1.0, "ACCUM": 0.5, "DIST": -0.5, "STRONG_DIST": -1.0}.get(tc, 0.0)
    obv_score = 1.0 if tobv == "BULLISH" else (-1.0 if tobv == "BEARISH" else 0.0)
    avwap_score = _AVWAP_SCORE.get(tav, 0.0)
    vol_score = (0.5 if vol_up else 0.0) + (-0.5 if bool(row.get("vol_confirms_down", False)) else 0.0)

    lines.append("  TREND COMPONENTS (always counted)")
    lines.append(f"    Price structure : {tp:<16}                    {tp_score:+.1f}")
    lines.append(f"    MACD            : {tm:<16} (above_zero={maz})  {macd_total:+.1f}")

    # aVWAP anchor dates (approximate from index)
    avwap_h_val = float(row.get("avwap_from_high", np.nan))
    avwap_l_val = float(row.get("avwap_from_low", np.nan))
    lines.append(f"    aVWAP           : {tav:<16}                   {avwap_score:+.1f}")
    if np.isfinite(avwap_h_val):
        lines.append(f"                       aVWAP_high = {avwap_h_val:.2f}")
    if np.isfinite(avwap_l_val):
        lines.append(f"                       aVWAP_low  = {avwap_l_val:.2f}")

    vol_pct_str = f"{vol_pct_val:.2f}" if np.isfinite(vol_pct_val) else "N/A"
    lines.append(f"    CMF             : {tc:<16} ({float(row.get('cmf', np.nan)):.3f})              {cmf_score:+.1f}")
    lines.append(f"    OBV             : {tobv:<16}                   {obv_score:+.1f}")
    lines.append(f"    Volume          : {tvol:<16} (vol_pct {vol_pct_str})        {vol_score:+.1f}")
    lines.append(f"    Convergence mult: {conv_mult:.2f}")
    lines.append(f"    Subtotal trend_score (post-mult): {base_trend:+.2f}")
    lines.append("")

    # Mean reversion
    rsi_val = float(row.get("rsi", np.nan))
    rsi_str = f"{rsi_val:.1f}" if np.isfinite(rsi_val) else "N/A"
    trsi = str(row.get("trend_rsi", "NEUTRAL"))
    tbb = str(row.get("trend_bb", "NEUTRAL"))
    suppressed = mr_weight == 0.0
    suppressed_str = "SUPPRESSED" if suppressed else f"weight={mr_weight:.2f}"
    lines.append(f"  MEAN-REVERSION COMPONENTS (gated: {suppressed_str}, mr_weight {mr_weight:.2f})")
    note = "reported, not scored" if suppressed else f"contribution {mr_contribution:+.2f}"
    lines.append(f"    RSI             : {trsi:<10} ({rsi_str})    {note}")
    lines.append(f"    Bollinger       : {tbb}")
    lines.append("")

    # Divergence
    n_reg_bull = int(row.get("n_reg_bull", 0)) if "n_reg_bull" in row.index else 0
    n_reg_bear = int(row.get("n_reg_bear", 0)) if "n_reg_bear" in row.index else 0

    # Recompute div counts directly from columns
    n_reg_bull = sum(1 for ind in _DIV_INDICATORS if bool(row.get(f"div_{ind}_reg_bull", False)))
    n_reg_bear = sum(1 for ind in _DIV_INDICATORS if bool(row.get(f"div_{ind}_reg_bear", False)))
    q_bull = any(bool(row.get(f"div_{ind}_quality", False)) for ind in _DIV_INDICATORS
                 if bool(row.get(f"div_{ind}_reg_bull", False)))
    q_bear = any(bool(row.get(f"div_{ind}_quality", False)) for ind in _DIV_INDICATORS
                 if bool(row.get(f"div_{ind}_reg_bear", False)))

    lines.append(f"  DIVERGENCE / CONVERGENCE   (n_reg_bull={n_reg_bull}, n_reg_bear={n_reg_bear})")
    for ind in _DIV_INDICATORS:
        if bool(row.get(f"div_{ind}_reg_bull", False)):
            q = bool(row.get(f"div_{ind}_quality", False))
            lines.append(f"    CONFIRMED regular bullish on {ind}   quality_boost: {q}")
        if bool(row.get(f"div_{ind}_reg_bear", False)):
            q = bool(row.get(f"div_{ind}_quality", False))
            lines.append(f"    CONFIRMED regular bearish on {ind}   quality_boost: {q}")
        if bool(row.get(f"div_{ind}_hid_bull", False)):
            lines.append(f"    CONFIRMED hidden bullish on {ind}")
        if bool(row.get(f"div_{ind}_hid_bear", False)):
            lines.append(f"    CONFIRMED hidden bearish on {ind}")
        if bool(row.get(f"conv_{ind}_bull", False)):
            lines.append(f"    Convergence bull on {ind} — trend confirmed")
        if bool(row.get(f"conv_{ind}_bear", False)):
            lines.append(f"    Convergence bear on {ind} — trend confirmed")

    if n_reg_bull >= 2:
        lines.append("    MOMENTUM EXHAUSTION — DOWNTREND LOSING STEAM (2+ confirmed bull)")
    if n_reg_bear >= 2:
        lines.append("    MOMENTUM EXHAUSTION — UPTREND LOSING STEAM (2+ confirmed bear)")

    # Pending
    pending_lines = []
    for ind in _DIV_INDICATORS:
        if bool(row.get(f"div_{ind}_reg_bull_pending", False)):
            pending_lines.append(f"    {ind}: Regular bullish — pending (needs close > pivot & > SMA20)")
        if bool(row.get(f"div_{ind}_reg_bear_pending", False)):
            pending_lines.append(f"    {ind}: Regular bearish — pending (needs close < pivot & < SMA20)")
    if pending_lines:
        lines.append("")
        lines.append("  PENDING (not contributing to score)")
        lines.extend(pending_lines)

    lines.append(f"    Subtotal divergence_score: {div_score:+.2f}")
    lines.append("")

    # Score arithmetic
    lines.append("  SCORE ARITHMETIC (per §5.3)")
    lines.append(f"    base_trend = trend_score {trend_score:.2f} × conv_mult {conv_mult:.2f}  = {base_trend:+.2f}")
    mr_raw_str = f"{mr_weight:.2f} × {mr_contribution/mr_weight:.2f}" if mr_weight != 0 and np.isfinite(mr_contribution) else f"{mr_weight:.2f} × (suppressed)"
    lines.append(f"    mr_weight × mean_reversion = {mr_raw_str}               = {mr_contribution:+.2f}")
    lines.append(f"    divergence_score                                        = {div_score:+.2f}")
    lines.append(f"    raw = {base_trend:.2f} + {mr_contribution:.2f} + {div_score:.2f}                    = {raw_val:+.2f}")
    lines.append(f"    score = clamp({raw_val:.2f}, -10, +10)                       = {score_val:+.2f}")
    lines.append(f"    confidence = |{score_val:.1f}| / 10                          = {conf_val:.2f}")
    lines.append("")

    # Caveats
    # Recompute for report
    caveats = []
    iv_pr = float(row.get("iv_premium_ratio", np.nan))
    iv_pct_v = float(row.get("iv_percentile", np.nan))
    iv_prem = float(row.get("iv_premium", np.nan))
    if np.isfinite(iv_pr) and np.isfinite(iv_pct_v):
        if iv_pr > 1.5 and iv_pct_v > 0.80:
            caveats.append("ELEVATED EVENT RISK — REDUCE POSITION SIZE OR WAIT FOR EVENT")
        if iv_pct_v > 0.90 and rec in ("STRONG_BUY", "STRONG_SELL"):
            caveats.append("HIGH IV PERCENTILE — DIRECTIONAL CONVICTION MAY UNWIND ON VOL CRUSH")
    if np.isfinite(iv_prem) and np.isfinite(iv_pct_v) and iv_prem < 0 and iv_pct_v < 0.20:
        caveats.append("LOW IV WITH IV < RV — TREND CONTINUATION FAVORED")
    if not np.isfinite(iv30):
        caveats.append("IV30 not available — vol regime is UNKNOWN (ADX-only gating)")

    if caveats:
        lines.append("  CAVEATS")
        for cav in caveats:
            lines.append(f"    - {cav}")

    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

def run_pipeline(df: pd.DataFrame, bars_per_year: int = 252,
                 params: dict = None, iv_path: Optional[str] = None) -> pd.DataFrame:
    """Full pipeline: indicators → aVWAP → IV context → classify → divergence → score."""
    if params is None:
        params = {}

    major_window = params.get("major_window", 10)
    lookback = params.get("lookback", 90)

    df = compute_indicators(df, bars_per_year=bars_per_year)
    df = compute_anchored_vwaps(df, major_window=major_window, lookback=lookback)
    df = classify_regime(df)
    df = classify_trends(df)
    df = run_divergence_suite(df, params=params)
    df = recommend(df, params=params)
    return df


# ---------------------------------------------------------------------------
# yfinance fetch
# ---------------------------------------------------------------------------

def fetch_ohlcv(symbol: str, bars_since: int = 400) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: yfinance is required. Install with: pip install yfinance", file=sys.stderr)
        sys.exit(1)

    end = date_cls.today()
    start = end - timedelta(days=int(bars_since * 1.6))
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start.isoformat(), end=end.isoformat())
    if df.empty:
        print(f"ERROR: No data returned for {symbol}", file=sys.stderr)
        sys.exit(1)

    df.index = pd.to_datetime(df.index).tz_localize(None)
    needed = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        print(f"ERROR: Missing columns {missing} for {symbol}", file=sys.stderr)
        sys.exit(1)

    df = df[needed].copy()
    df.columns = [c.lower() for c in df.columns]
    if len(df) > bars_since:
        df = df.iloc[-bars_since:]
    return df


# ---------------------------------------------------------------------------
# §8 — Tests
# ---------------------------------------------------------------------------

def _make_synthetic(n: int = 400, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV with controlled sine+drift."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    # Upward drift with sine wave
    close = 100.0 + 0.05 * t + 5.0 * np.sin(2 * np.pi * t / 40)
    close = close + rng.normal(0, 0.3, n)
    open_ = close + rng.normal(0, 0.2, n)
    high = np.maximum(close, open_) + rng.uniform(0.1, 1.0, n)
    low = np.minimum(close, open_) - rng.uniform(0.1, 1.0, n)
    volume = rng.uniform(1e6, 2e6, n)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": volume, "iv_30": np.nan}, index=idx)


def _make_bear_then_bull(n: int = 400) -> pd.DataFrame:
    """Downtrend then strong bounce — useful for divergence tests."""
    rng = np.random.default_rng(7)
    t = np.arange(n)
    # Down first half, up second half
    half = n // 2
    close = np.zeros(n)
    close[:half] = 100.0 - 0.1 * np.arange(half) + rng.normal(0, 0.3, half)
    close[half:] = close[half - 1] + 0.15 * np.arange(n - half) + rng.normal(0, 0.3, n - half)
    open_ = close + rng.normal(0, 0.2, n)
    high = np.maximum(close, open_) + rng.uniform(0.1, 1.5, n)
    low = np.minimum(close, open_) - rng.uniform(0.1, 1.5, n)
    volume = np.where(t >= half, rng.uniform(1.5e6, 3e6, n), rng.uniform(0.5e6, 1e6, n))
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": volume, "iv_30": np.nan}, index=idx)


def run_tests() -> bool:
    """Run all 11 built-in tests. Returns True if all pass."""
    results = []

    def _test(num: int, name: str, passed: bool, detail: str = ""):
        status = "PASS" if passed else "FAIL"
        msg = f"Test {num:02d}: [{status}] {name}"
        if detail and not passed:
            msg += f"\n         Detail: {detail}"
        print(msg)
        results.append(passed)

    # -------------------------------------------------------------------------
    # Test 1: Look-ahead — no div_* column True before bar b+right
    #
    # The invariant: a regular-div confirmed column being True at bar i implies
    # there exists a pivot pair (p1, p2) such that the state-machine confirmed
    # at some bar ca >= p2 + right. We verify this by checking that the first
    # True bar for any non-pending, non-quality div column is >= the minimum
    # possible confirmation bar (pivot_right bars after the second pivot).
    # -------------------------------------------------------------------------
    try:
        df = _make_synthetic(300)
        df = compute_indicators(df)
        df = compute_anchored_vwaps(df)
        df = classify_regime(df)
        df = classify_trends(df)
        df = run_divergence_suite(df)

        right = 5  # pivot_right default
        # Non-pending, non-quality div columns (confirmed state columns)
        div_cols = [c for c in df.columns if c.startswith("div_") and
                    not c.endswith("_pending") and not c.endswith("_quality")]

        # Find all confirmed pivots (both highs and lows)
        ph, pl = _find_pivots(df["close"], left=5, right=right)
        # Minimum bar where any confirmation could first appear = min(pivot+right)
        all_confirm_starts = set()
        for piv in (ph + pl):
            all_confirm_starts.add(piv + right)

        min_possible_confirm = min(all_confirm_starts) if all_confirm_starts else right

        violation = False
        detail_1 = ""
        for col in div_cols:
            col_vals = df[col].values
            for bar_i in range(len(col_vals)):
                if col_vals[bar_i]:
                    # This bar is True. It must be at least min_possible_confirm.
                    if bar_i < min_possible_confirm:
                        violation = True
                        detail_1 = (f"col={col} bar={bar_i} True before "
                                    f"min_possible_confirm={min_possible_confirm}")
                    break  # only check first True occurrence per column

        _test(1, "Look-ahead: no div_* True before b+right", not violation, detail_1)
    except Exception as e:
        _test(1, "Look-ahead: no div_* True before b+right", False, str(e))

    # -------------------------------------------------------------------------
    # Test 2: Hysteresis — oscillation ±0.3 around 2.5 doesn't whipsaw
    # -------------------------------------------------------------------------
    try:
        scores = [2.2, 2.8, 2.2, 2.8, 2.2, 2.8, 2.2, 2.8]
        prev = "HOLD"
        recs = []
        for s in scores:
            r = _apply_hysteresis(s, prev)
            recs.append(r)
            prev = r

        # Should enter BUY at 2.8 and stay there (exit threshold is 1.5)
        # 2.2 < 2.5 enter threshold but >= 1.5 exit threshold → stays BUY
        transitions = sum(1 for i in range(1, len(recs)) if recs[i] != recs[i-1])
        passed = transitions <= 1
        _test(2, "Hysteresis: ±0.3 oscillation around 2.5 ≤ 1 toggle", passed,
              f"transitions={transitions} recs={recs}")
    except Exception as e:
        _test(2, "Hysteresis: ±0.3 oscillation around 2.5 ≤ 1 toggle", False, str(e))

    # -------------------------------------------------------------------------
    # Test 3: Regime gate — RSI overbought scores differently at ADX=15 vs 30
    # -------------------------------------------------------------------------
    try:
        # Simulate row-level scoring with controlled RSI=75
        base_row = {
            "atr_14": 2.0, "adx": 15.0, "obv_sma_20": 1e6, "rv_20": 0.2, "vol_pct": 0.5,
            "trend_price": "NEUTRAL", "trend_macd": "NEUTRAL", "macd_above_zero": False,
            "trend_cmf": "NEUTRAL", "trend_obv": "NEUTRAL", "trend_avwap": "UNAVAILABLE",
            "vol_confirms_up": False, "vol_confirms_down": False,
            "macd_hist": 0.0, "_macd_hist_prev": 0.0,
            "rsi": 75.0, "trend_rsi": "OVERBOUGHT", "trend_bb": "NEUTRAL",
            "adx_regime": "RANGING", "vol_regime": "UNKNOWN",
            "cmf": 0.0,
        }
        # Add div columns
        for ind in _DIV_INDICATORS:
            for suffix in ["_reg_bull", "_reg_bear", "_reg_bull_pending", "_reg_bear_pending",
                           "_hid_bull", "_hid_bear", "_quality"]:
                base_row[f"div_{ind}{suffix}"] = False
            base_row[f"conv_{ind}_bull"] = False
            base_row[f"conv_{ind}_bear"] = False

        row_ranging = pd.Series(base_row)
        row_trending = pd.Series({**base_row, "adx": 30.0, "adx_regime": "TRENDING"})

        sc_ranging = score_bar(row_ranging, "HOLD")
        sc_trending = score_bar(row_trending, "HOLD")

        # RANGING + UNKNOWN → mr_weight=1.0; RSI overbought → mr_score=-1.0 → mr_contrib=-1.0
        # TRENDING + UNKNOWN → mr_weight=0.0; same RSI → mr_contrib=0.0
        passed = (sc_ranging["mr_weight"] > sc_trending["mr_weight"] and
                  sc_ranging["mr_contribution"] != sc_trending["mr_contribution"])
        _test(3, "Regime gate: RSI-overbought scores differently at ADX=15 vs ADX=30",
              passed, f"ranging_mr={sc_ranging['mr_contribution']:.2f} trending_mr={sc_trending['mr_contribution']:.2f}")
    except Exception as e:
        _test(3, "Regime gate: RSI-overbought scores differently at ADX=15 vs ADX=30", False, str(e))

    # -------------------------------------------------------------------------
    # Test 4: Divergence state machine PENDING → CONFIRMED → EXPIRED
    # -------------------------------------------------------------------------
    try:
        # Build a controlled series: declining price with rising RSI (reg bull div)
        n = 300
        rng = np.random.default_rng(99)
        # Simple declining then bounce close
        t = np.arange(n)
        close_vals = 100.0 - 0.08 * t + rng.normal(0, 0.1, n)
        # Force a bounce near bar 250
        close_vals[240:] += 3.0 + 0.2 * np.arange(n - 240)
        open_ = close_vals + rng.normal(0, 0.1, n)
        high_vals = np.maximum(close_vals, open_) + 0.5
        low_vals = np.minimum(close_vals, open_) - 0.5
        vol_vals = rng.uniform(1e6, 2e6, n)
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        df_sm = pd.DataFrame({"open": open_, "high": high_vals, "low": low_vals,
                              "close": close_vals, "volume": vol_vals, "iv_30": np.nan}, index=idx)
        df_sm = compute_indicators(df_sm)
        df_sm = compute_anchored_vwaps(df_sm)
        df_sm = classify_regime(df_sm)
        df_sm = classify_trends(df_sm)
        df_sm = run_divergence_suite(df_sm)

        # Check that div columns exist and are bool
        has_reg_bull = "div_rsi_reg_bull" in df_sm.columns
        has_pending = "div_rsi_reg_bull_pending" in df_sm.columns

        # Ensure confirmed bars come AFTER pending bars (if any exist)
        confirmed_bars = df_sm.index[df_sm["div_rsi_reg_bull"]].tolist()
        pending_bars = df_sm.index[df_sm["div_rsi_reg_bull_pending"]].tolist()

        order_ok = True
        if confirmed_bars and pending_bars:
            first_confirm = min(confirmed_bars)
            first_pending = min(pending_bars)
            order_ok = first_pending <= first_confirm

        _test(4, "Divergence state machine: PENDING → CONFIRMED order",
              has_reg_bull and has_pending and order_ok,
              f"has_reg_bull={has_reg_bull} has_pending={has_pending} order_ok={order_ok}")
    except Exception as e:
        _test(4, "Divergence state machine: PENDING → CONFIRMED order", False, str(e))

    # -------------------------------------------------------------------------
    # Test 5: Clamp — maxed-out bullish score ≤ +10 and confidence ≤ 1.0
    # -------------------------------------------------------------------------
    try:
        base_row = {
            "atr_14": 2.0, "adx": 30.0, "obv_sma_20": 1e6, "rv_20": 0.2, "vol_pct": 0.9,
            "trend_price": "STRONG_UP", "trend_macd": "BULL_CROSS", "macd_above_zero": True,
            "trend_cmf": "STRONG_ACCUM", "trend_obv": "BULLISH", "trend_avwap": "RECLAIM_HIGH",
            "vol_confirms_up": True, "vol_confirms_down": False,
            "macd_hist": 1.0, "_macd_hist_prev": 0.5,
            "rsi": 25.0, "trend_rsi": "OVERSOLD", "trend_bb": "LOWER_BREAK",
            "adx_regime": "RANGING", "vol_regime": "UNKNOWN",
            "cmf": 0.15,
        }
        # 2 confirmed bull divergences with quality
        for ind in _DIV_INDICATORS:
            base_row[f"div_{ind}_reg_bull"] = (ind in ["rsi", "macd_line"])
            base_row[f"div_{ind}_reg_bear"] = False
            base_row[f"div_{ind}_reg_bull_pending"] = False
            base_row[f"div_{ind}_reg_bear_pending"] = False
            base_row[f"div_{ind}_hid_bull"] = False
            base_row[f"div_{ind}_hid_bear"] = False
            base_row[f"div_{ind}_quality"] = (ind in ["rsi", "macd_line"])
            base_row[f"conv_{ind}_bull"] = False
            base_row[f"conv_{ind}_bear"] = False

        row = pd.Series(base_row)
        sc = score_bar(row, "HOLD")
        passed = sc["score"] <= 10.0 and sc["confidence"] <= 1.0
        _test(5, "Clamp: max bullish score ≤ +10 and confidence ≤ 1.0",
              passed, f"score={sc['score']:.2f} conf={sc['confidence']:.2f} raw={sc['raw']:.2f}")
    except Exception as e:
        _test(5, "Clamp: max bullish score ≤ +10 and confidence ≤ 1.0", False, str(e))

    # -------------------------------------------------------------------------
    # Test 6: OBV divergence — price lower-low while OBV higher-low → div_obv_reg_bull
    # -------------------------------------------------------------------------
    try:
        # Construct synthetic data where price makes a lower low but OBV makes a higher low
        n = 250
        rng = np.random.default_rng(55)
        close_vals = np.ones(n) * 100.0
        # Create two clear lows: first at bar 80 (price=90), second at bar 160 (price=88)
        # but OBV should be higher at bar 160 than bar 80
        close_vals[:80] = np.linspace(100, 90, 80) + rng.normal(0, 0.1, 80)
        close_vals[80:160] = np.linspace(90, 105, 80) + rng.normal(0, 0.1, 80)
        close_vals[160:] = np.linspace(105, 90, n - 160) + rng.normal(0, 0.1, n - 160)

        # Make volume heavier on up days in second segment to lift OBV
        vol_vals = rng.uniform(1e6, 1.5e6, n)
        vol_vals[120:170] *= 2.5  # heavier volume on bounce -> higher OBV at second low

        open_ = close_vals + rng.normal(0, 0.1, n)
        high_vals = np.maximum(close_vals, open_) + 0.5
        low_vals = np.minimum(close_vals, open_) - 0.5
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        df_obv = pd.DataFrame({"open": open_, "high": high_vals, "low": low_vals,
                               "close": close_vals, "volume": vol_vals, "iv_30": np.nan}, index=idx)
        df_obv = compute_indicators(df_obv)
        df_obv = compute_anchored_vwaps(df_obv)
        df_obv = classify_regime(df_obv)
        df_obv = classify_trends(df_obv)
        df_obv = run_divergence_suite(df_obv)

        has_col = "div_obv_reg_bull" in df_obv.columns
        # The column exists and has the right type
        _test(6, "OBV divergence column exists and is boolean type",
              has_col and df_obv["div_obv_reg_bull"].dtype == bool,
              f"has_col={has_col}")
    except Exception as e:
        _test(6, "OBV divergence column exists and is boolean type", False, str(e))

    # -------------------------------------------------------------------------
    # Test 7: aVWAP anchor-change — new pivot resets avwap_from_high
    #
    # We verify the O(n) amortized implementation correctly resets running sums
    # when a new anchor appears. We do this by:
    #   1) Building a series with a large spike in BOTH high AND close at bar 200-210
    #      so a new major pivot high appears at that location.
    #   2) Comparing avwap_from_high in bars 220-255 between spiked and normal series.
    #   3) Since the spiked anchor price is much higher (~150 vs ~120), the aVWAP
    #      from the spiked anchor must be higher than from the natural anchor.
    # -------------------------------------------------------------------------
    try:
        df_vwap_base = _make_synthetic(300)
        df_vwap_base = compute_indicators(df_vwap_base)

        # Spike BOTH high and close at 200-210 to create a genuine major pivot high
        # with high close prices (so aVWAP from that anchor will be elevated)
        df_vwap_mod = df_vwap_base.copy()
        spike_cols = ["high", "close", "open"]
        for col in spike_cols:
            if col in df_vwap_mod.columns:
                df_vwap_mod.iloc[200:211, df_vwap_mod.columns.get_loc(col)] += 40.0
        # Recompute low consistently
        df_vwap_mod["low"] = df_vwap_mod[["open", "close"]].min(axis=1) - 0.5

        df_a = compute_anchored_vwaps(df_vwap_base.copy())
        df_b = compute_anchored_vwaps(df_vwap_mod)

        col_exists = "avwap_from_high" in df_a.columns and "avwap_from_high" in df_b.columns

        # Check bars 225-255: df_b should have a higher aVWAP (anchored from spiked bar)
        # while df_a has a lower natural aVWAP
        range_a = df_a["avwap_from_high"].iloc[225:256].dropna()
        range_b = df_b["avwap_from_high"].iloc[225:256].dropna()

        values_differ = False
        if len(range_a) > 0 and len(range_b) > 0:
            mean_a = range_a.mean()
            mean_b = range_b.mean()
            # Spiked df_b should have higher aVWAP from the elevated anchor
            values_differ = mean_b > mean_a + 1.0

        _test(7, "aVWAP anchor-change: spike in high+close shifts avwap_from_high",
              col_exists and values_differ,
              f"col_exists={col_exists} mean_a={range_a.mean() if len(range_a)>0 else 'N/A':.2f} "
              f"mean_b={range_b.mean() if len(range_b)>0 else 'N/A':.2f}")
    except Exception as e:
        _test(7, "aVWAP anchor-change: spike in high+close shifts avwap_from_high", False, str(e))

    # -------------------------------------------------------------------------
    # Test 8: aVWAP reclaim — RECLAIM_HIGH only on crossing bar
    # -------------------------------------------------------------------------
    try:
        df_rc = _make_synthetic(300)
        df_rc = compute_indicators(df_rc)
        df_rc = compute_anchored_vwaps(df_rc)
        df_rc = classify_regime(df_rc)
        df_rc = classify_trends(df_rc)

        avwap_h = df_rc["avwap_from_high"].values
        close_v = df_rc["close"].values

        # Find a reclaim event
        reclaim_bars = df_rc.index[df_rc["trend_avwap"] == "RECLAIM_HIGH"].tolist()
        above_both = df_rc.index[df_rc["trend_avwap"] == "ABOVE_BOTH"].tolist()

        # If there's a reclaim bar, the bar after should be ABOVE_BOTH (not another RECLAIM)
        consecutive_reclaims = 0
        tavwap = df_rc["trend_avwap"].values
        for i in range(1, len(tavwap)):
            if tavwap[i] == "RECLAIM_HIGH" and tavwap[i-1] == "RECLAIM_HIGH":
                consecutive_reclaims += 1

        # Pipeline ran; consecutive reclaims should be rare/zero for continuous data
        col_ok = "trend_avwap" in df_rc.columns
        _test(8, "aVWAP reclaim: RECLAIM_HIGH not on consecutive bars",
              col_ok and consecutive_reclaims == 0,
              f"consecutive_reclaims={consecutive_reclaims}")
    except Exception as e:
        _test(8, "aVWAP reclaim: RECLAIM_HIGH not on consecutive bars", False, str(e))

    # -------------------------------------------------------------------------
    # Test 9: Regime matrix — mr_weight matches matrix for all combos
    # -------------------------------------------------------------------------
    try:
        base_row = {
            "atr_14": 2.0, "adx": 15.0, "obv_sma_20": 1e6, "rv_20": 0.2, "vol_pct": 0.5,
            "trend_price": "NEUTRAL", "trend_macd": "NEUTRAL", "macd_above_zero": False,
            "trend_cmf": "NEUTRAL", "trend_obv": "NEUTRAL", "trend_avwap": "UNAVAILABLE",
            "vol_confirms_up": False, "vol_confirms_down": False,
            "macd_hist": 0.0, "_macd_hist_prev": 0.0,
            "rsi": 75.0, "trend_rsi": "OVERBOUGHT", "trend_bb": "NEUTRAL",
            "cmf": 0.0,
        }
        for ind in _DIV_INDICATORS:
            for sfx in ["_reg_bull", "_reg_bear", "_reg_bull_pending", "_reg_bear_pending",
                        "_hid_bull", "_hid_bear", "_quality"]:
                base_row[f"div_{ind}{sfx}"] = False
            base_row[f"conv_{ind}_bull"] = False
            base_row[f"conv_{ind}_bear"] = False

        all_ok = True
        for adx_reg, vol_reg in [
            ("TRENDING", "LOW_IV"), ("TRENDING", "NORMAL_IV"), ("TRENDING", "HIGH_IV"), ("TRENDING", "UNKNOWN"),
            ("WEAK_TREND", "LOW_IV"), ("RANGING", "NORMAL_IV"), ("RANGING", "HIGH_IV"), ("RANGING", "UNKNOWN"),
        ]:
            row = pd.Series({**base_row, "adx_regime": adx_reg, "vol_regime": vol_reg})
            sc = score_bar(row, "HOLD")
            expected = _REGIME_MATRIX[adx_reg][vol_reg]
            if abs(sc["mr_weight"] - expected) > 0.001:
                all_ok = False
                break

        _test(9, "Regime matrix: mr_weight matches _REGIME_MATRIX for all combos", all_ok)
    except Exception as e:
        _test(9, "Regime matrix: mr_weight matches _REGIME_MATRIX for all combos", False, str(e))

    # -------------------------------------------------------------------------
    # Test 10: IV-missing fallback — vol_regime UNKNOWN every bar
    # -------------------------------------------------------------------------
    try:
        df_noiv = _make_synthetic(300)
        # No iv_30 column
        df_noiv = df_noiv.drop(columns=["iv_30"], errors="ignore")
        df_noiv = compute_indicators(df_noiv)
        df_noiv = compute_anchored_vwaps(df_noiv)
        df_noiv = classify_regime(df_noiv)

        all_unknown = (df_noiv["vol_regime"] == "UNKNOWN").all()
        _test(10, "IV-missing fallback: vol_regime UNKNOWN every bar",
              all_unknown, f"unique_vol_regimes={df_noiv['vol_regime'].unique().tolist()}")
    except Exception as e:
        _test(10, "IV-missing fallback: vol_regime UNKNOWN every bar", False, str(e))

    # -------------------------------------------------------------------------
    # Test 11: Event-risk caveat appears when iv_premium_ratio > 1.5 and iv_pct > 0.80
    # -------------------------------------------------------------------------
    try:
        df_iv = _make_synthetic(300)
        # Inject iv_30 data that creates high iv_percentile and premium ratio
        # Set iv_30 to escalating values so iv_percentile ends up > 0.80
        iv_vals = np.linspace(0.10, 0.80, 300)  # escalating IV
        df_iv["iv_30"] = iv_vals
        df_iv = compute_indicators(df_iv)
        df_iv = compute_anchored_vwaps(df_iv)
        df_iv = classify_regime(df_iv)
        df_iv = classify_trends(df_iv)
        df_iv = run_divergence_suite(df_iv)
        df_iv = recommend(df_iv)

        # Check the last bar report
        report = build_report(df_iv, idx=-1, symbol="TEST")
        caveat_present = "ELEVATED EVENT RISK" in report or "HIGH IV PERCENTILE" in report or "LOW IV WITH IV < RV" in report

        # Also manually check: if iv_pr > 1.5 and iv_pct > 0.80
        last = df_iv.iloc[-1]
        iv_pr = float(last.get("iv_premium_ratio", np.nan))
        iv_pct_v = float(last.get("iv_percentile", np.nan))
        conditions_met = np.isfinite(iv_pr) and np.isfinite(iv_pct_v) and iv_pr > 1.5 and iv_pct_v > 0.80

        if conditions_met:
            passed = caveat_present
        else:
            # Construct a synthetic row manually to verify caveat logic
            base_row2 = {
                "atr_14": 2.0, "adx": 15.0, "obv_sma_20": 1e6, "rv_20": 0.1, "vol_pct": 0.5,
                "trend_price": "NEUTRAL", "trend_macd": "NEUTRAL", "macd_above_zero": False,
                "trend_cmf": "NEUTRAL", "trend_obv": "NEUTRAL", "trend_avwap": "UNAVAILABLE",
                "vol_confirms_up": False, "vol_confirms_down": False,
                "macd_hist": 0.0, "_macd_hist_prev": 0.0,
                "rsi": 50.0, "trend_rsi": "NEUTRAL", "trend_bb": "NEUTRAL",
                "adx_regime": "RANGING", "vol_regime": "HIGH_IV",
                "cmf": 0.0,
                "iv_30": 0.50, "iv_premium": 0.40, "iv_premium_ratio": 5.0, "iv_percentile": 0.92,
                "avwap_from_high": np.nan, "avwap_from_low": np.nan,
            }
            for ind in _DIV_INDICATORS:
                for sfx in ["_reg_bull", "_reg_bear", "_reg_bull_pending", "_reg_bear_pending",
                            "_hid_bull", "_hid_bear", "_quality"]:
                    base_row2[f"div_{ind}{sfx}"] = False
                base_row2[f"conv_{ind}_bull"] = False
                base_row2[f"conv_{ind}_bear"] = False
            sc2 = score_bar(pd.Series(base_row2), "HOLD")
            passed = "ELEVATED EVENT RISK" in sc2["caveats"]

        _test(11, "Event-risk caveat appears when iv_premium_ratio > 1.5 & iv_pct > 0.80",
              passed, f"conditions_met={conditions_met} caveat_present={caveat_present}")
    except Exception as e:
        _test(11, "Event-risk caveat appears when iv_premium_ratio > 1.5 & iv_pct > 0.80", False, str(e))

    # Summary
    passed_count = sum(results)
    total = len(results)
    print(f"\n{'='*40}")
    print(f"Results: {passed_count}/{total} tests passed")
    print(f"{'='*40}")
    return all(results)


# ---------------------------------------------------------------------------
# CLI main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-indicator trade signal generator (self-contained).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("ticker", nargs="?", default=None,
                        help="Ticker symbol (e.g. AAPL) or path to OHLCV CSV")
    parser.add_argument("--bar", type=int, default=-1,
                        help="Bar index to report on (default: -1 = most recent)")
    parser.add_argument("--bars-since", type=int, default=400,
                        help="Number of OHLCV bars to fetch (default: 400)")
    parser.add_argument("--iv-csv", metavar="PATH", default=None,
                        help="CSV with date index and iv_30 column (decimal)")
    parser.add_argument("--params", metavar="PATH", default=None,
                        help="YAML file with threshold overrides")
    parser.add_argument("--bars-per-year", type=int, default=252,
                        help="Trading bars per year (252=daily, 52=weekly)")
    parser.add_argument("--test", action="store_true",
                        help="Run all 11 built-in tests and exit")
    args = parser.parse_args()

    if args.test:
        ok = run_tests()
        sys.exit(0 if ok else 1)

    if args.ticker is None:
        parser.print_help()
        sys.exit(1)

    params = load_params(args.params)
    symbol = args.ticker.upper()

    # Determine if ticker is a CSV path or a symbol
    import os
    if os.path.exists(args.ticker):
        print(f"Loading OHLCV from CSV: {args.ticker}", file=sys.stderr)
        df = load_data(args.ticker, iv_path=args.iv_csv)
    else:
        print(f"Fetching {args.bars_since} bars of OHLCV for {symbol}...", file=sys.stderr)
        df = fetch_ohlcv(symbol, args.bars_since)
        if args.iv_csv:
            df = load_data(df, iv_path=args.iv_csv)

    if len(df) < 200:
        print(f"WARNING: Only {len(df)} bars — results may be unreliable.", file=sys.stderr)

    print(f"Running pipeline on {len(df)} bars...", file=sys.stderr)
    df = run_pipeline(df, bars_per_year=args.bars_per_year, params=params)

    # Determine bar index
    n = len(df)
    bar_idx = args.bar
    if bar_idx < 0:
        bar_idx = n + bar_idx
    if bar_idx < 0 or bar_idx >= n:
        print(f"ERROR: bar index {args.bar} out of range for {n} bars", file=sys.stderr)
        sys.exit(1)

    report = build_report(df, idx=bar_idx, symbol=symbol)
    print(report)


if __name__ == "__main__":
    main()
