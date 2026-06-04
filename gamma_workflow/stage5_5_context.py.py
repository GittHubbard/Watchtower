"""Stage 5.5 — Technical & fundamental context (Features 2 + 3).

Two families of context that sit alongside the gamma/flow signals but, per the PM,
do NOT gate Stage 6 playbooks. They are logged as scalar features (audit trail)
and surfaced as commentary on the ticket for human review.

  Technicals (Feature 2) — computed from the underlying OHLCV history Stage 5
  already fetches; pure pandas, no new dependency:
      macd, macd_signal, macd_hist          (12 / 26 / 9)
      rsi_14                                 (Wilder's 14)
      cmf_20                                 (Chaikin Money Flow, 20)
      bb_upper, bb_mid, bb_lower, bb_width   (Bollinger 20 / 2)

  Fundamentals (Feature 3) — from yfinance via `acquire_fundamentals`:
      forward_pe, eps_surprise_avg_4q, eps_beat_streak, eps_estimate_trend
      net_income_margin_ttm, debt_to_equity, revenue_growth_yoy
      analyst_consensus, analyst_coverage_count, analyst_pct_buy
      analyst_upgrades_3m, analyst_downgrades_3m, analyst_steady_3m

ETF / index tickers (kind == 'index_etf') skip the fundamentals fetch entirely and
return all-NaN fundamental scalars.

Commentary: full narrative generation is implemented in Sections B–H below.
`technical_commentary` and `analyst_commentary` are built from the scalar features
and the storage fundamentals-evolution cache. They are transient (in-memory only,
not stored in parquet).

Public API:
    acquire_fundamentals(symbol, ticker_obj) -> dict | None        (network)
    compute_technicals(history, asof, symbol="", ivx_30d=nan) -> dict (pure)
    compute_fundamentals(raw, ticker_config, asof, symbol="") -> dict (pure)
    compute_stage5_5(history, asof, raw_bundle, ticker_config, symbol="", ivx_30d=nan) -> dict
    split_for_cache(features) -> (analyst_payload, financials_payload)
"""
from __future__ import annotations
from datetime import date as date_cls
import logging
import math

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---- Column groups ----------------------------------------------------------

TECHNICAL_KEYS = (
    "macd", "macd_signal", "macd_hist",
    "rsi_14", "cmf_20",
    "bb_upper", "bb_mid", "bb_lower", "bb_width",
)
ANALYST_KEYS = (
    "analyst_consensus", "analyst_coverage_count", "analyst_pct_buy",
    "analyst_upgrades_3m", "analyst_downgrades_3m", "analyst_steady_3m",
)
FINANCIAL_KEYS = (
    "forward_pe", "eps_surprise_avg_4q", "eps_beat_streak", "eps_estimate_trend",
    "net_income_margin_ttm", "debt_to_equity", "revenue_growth_yoy",
)

_NAN = float("nan")


def _num(v) -> float:
    """Coerce to float, NaN on anything non-numeric/None."""
    if v is None:
        return _NAN
    try:
        f = float(v)
    except (TypeError, ValueError):
        return _NAN
    return f if np.isfinite(f) else _NAN


# ---- Technical indicators (Feature 2) --------------------------------------

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> float:
    """Wilder's RSI over `period`. NaN if < period+1 closes."""
    if len(close) < period + 1:
        return _NAN
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    # Wilder smoothing == EWM with alpha = 1/period.
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    ag, al = float(avg_gain.iloc[-1]), float(avg_loss.iloc[-1])
    if not (np.isfinite(ag) and np.isfinite(al)):
        return _NAN
    if al == 0:
        return 100.0
    rs = ag / al
    return float(100.0 - 100.0 / (1.0 + rs))


def _cmf(high: pd.Series, low: pd.Series, close: pd.Series, vol: pd.Series, period: int = 20) -> float:
    """Chaikin Money Flow over `period`. NaN if < period bars or zero volume."""
    if len(close) < period:
        return _NAN
    rng = (high - low).replace(0, np.nan)
    mfm = ((close - low) - (high - close)) / rng          # money-flow multiplier
    mfv = (mfm * vol).fillna(0.0)                          # zero-range days contribute 0
    denom = float(vol.iloc[-period:].sum())
    if denom == 0 or not np.isfinite(denom):
        return _NAN
    return float(mfv.iloc[-period:].sum() / denom)


def _bollinger(close: pd.Series, period: int = 20, k: float = 2.0) -> tuple[float, float, float, float]:
    """(upper, mid, lower, width) for Bollinger bands. width = (upper-lower)/mid."""
    if len(close) < period:
        return _NAN, _NAN, _NAN, _NAN
    window = close.iloc[-period:]
    mid = float(window.mean())
    std = float(window.std(ddof=0))
    upper, lower = mid + k * std, mid - k * std
    width = (upper - lower) / mid if mid != 0 else _NAN
    return upper, mid, lower, width


# ============================================================================
# Section B — Commentary-only indicators
# ============================================================================

def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> tuple[float, float, float]:
    """(adx, plus_di, minus_di) Wilder's ADX. NaN triple if < 2*period bars."""
    if len(close) < 2 * period:
        return _NAN, _NAN, _NAN
    try:
        prev_high = high.shift(1)
        prev_low = low.shift(1)
        prev_close = close.shift(1)

        # True Range
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)

        # Directional movement
        up_move = high - prev_high
        down_move = prev_low - low
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

        alpha = 1.0 / period
        atr = tr.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
        smooth_plus = plus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
        smooth_minus = minus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

        plus_di = 100.0 * smooth_plus / atr.replace(0, np.nan)
        minus_di = 100.0 * smooth_minus / atr.replace(0, np.nan)

        dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx_series = dx.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

        adx_val = float(adx_series.iloc[-1])
        pdi_val = float(plus_di.iloc[-1])
        mdi_val = float(minus_di.iloc[-1])

        if not (np.isfinite(adx_val) and np.isfinite(pdi_val) and np.isfinite(mdi_val)):
            return _NAN, _NAN, _NAN
        return adx_val, pdi_val, mdi_val
    except Exception:
        return _NAN, _NAN, _NAN


def _obv_sma(close: pd.Series, volume: pd.Series) -> tuple[float, float, float]:
    """(obv_current, obv_sma20, obv_slope_3) — obv_slope = obv[-1] - obv[-4]."""
    try:
        if len(close) < 2:
            return _NAN, _NAN, _NAN
        direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        direction.iloc[0] = 0
        obv = (direction * volume).cumsum()
        obv_cur = float(obv.iloc[-1])
        sma20 = float(obv.iloc[-20:].mean()) if len(obv) >= 20 else _NAN
        slope = float(obv.iloc[-1] - obv.iloc[-4]) if len(obv) >= 4 else _NAN
        return obv_cur, sma20, slope
    except Exception:
        return _NAN, _NAN, _NAN


def _rv_20(close: pd.Series) -> float:
    """20-bar annualized realized vol (decimal). NaN if < 21 bars."""
    if len(close) < 21:
        return _NAN
    try:
        log_ret = np.log(close / close.shift(1)).dropna()
        if len(log_ret) < 20:
            return _NAN
        rv = float(log_ret.iloc[-20:].std(ddof=1)) * math.sqrt(252)
        return rv if np.isfinite(rv) else _NAN
    except Exception:
        return _NAN


def _find_pivots(series: pd.Series, left: int = 5, right: int = 5) -> tuple[list[int], list[int]]:
    """(high_indices, low_indices) — confirmed pivot bar indices."""
    highs, lows = [], []
    n = len(series)
    vals = series.values
    for i in range(left, n - right):
        window = vals[i - left: i + right + 1]
        if vals[i] == max(window):
            highs.append(i)
        if vals[i] == min(window):
            lows.append(i)
    return highs, lows


def _atr14(high: pd.Series, low: pd.Series, close: pd.Series) -> float:
    """14-bar ATR."""
    try:
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1.0 / 14, adjust=False, min_periods=14).mean()
        val = float(atr.iloc[-1])
        return val if np.isfinite(val) else _NAN
    except Exception:
        return _NAN


def _anchored_vwap(
    close: pd.Series, high: pd.Series, low: pd.Series, volume: pd.Series,
    major_window: int = 10, lookback: int = 90,
) -> tuple:
    """Returns (avwap_high, avwap_low, dist_high_atr, dist_low_atr, high_anchor_age, low_anchor_age).

    Finds most recent confirmed major pivot high/low in trailing `lookback` bars.
    dist_*_atr = (close[-1] - avwap) / ATR14  (positive = above aVWAP).
    anchor_age = bars since anchor pivot.
    Returns NaN for any uncomputable quantity.
    """
    nan6 = (_NAN,) * 6
    try:
        n = len(close)
        if n < major_window * 2 + 1:
            return nan6

        # Restrict to lookback window
        start_idx = max(0, n - lookback)
        c_win = close.iloc[start_idx:]
        h_win = high.iloc[start_idx:]
        l_win = low.iloc[start_idx:]

        high_pivots, low_pivots = _find_pivots(h_win, left=major_window, right=major_window)
        # We need at least one confirmed pivot (confirmed means right bars are in)
        # The last right bars of c_win are not confirmed yet — filter those out
        max_confirmed = len(c_win) - major_window - 1

        high_pivots_ok = [p for p in high_pivots if p <= max_confirmed]
        low_pivots_ok = [p for p in low_pivots if p <= max_confirmed]

        atr = _atr14(high, low, close)
        cur_close = float(close.iloc[-1])

        def _vwap_from(pivot_offset_in_window):
            # pivot_offset_in_window is index within c_win
            abs_idx = start_idx + pivot_offset_in_window
            c_seg = close.iloc[abs_idx:]
            h_seg = high.iloc[abs_idx:]
            l_seg = low.iloc[abs_idx:]
            v_seg = volume.iloc[abs_idx:]
            tp = (h_seg + l_seg + c_seg) / 3.0
            cum_vol = v_seg.cumsum()
            cum_tpv = (tp * v_seg).cumsum()
            last_vol = float(cum_vol.iloc[-1])
            if last_vol == 0:
                return _NAN
            return float(cum_tpv.iloc[-1] / last_vol)

        avwap_h, avwap_l = _NAN, _NAN
        dist_h, dist_l = _NAN, _NAN
        age_h, age_l = _NAN, _NAN

        if high_pivots_ok:
            p_idx = high_pivots_ok[-1]
            avwap_h = _vwap_from(p_idx)
            age_h = float(len(c_win) - 1 - p_idx)
            if np.isfinite(avwap_h) and np.isfinite(atr) and atr != 0:
                dist_h = (cur_close - avwap_h) / atr

        if low_pivots_ok:
            p_idx = low_pivots_ok[-1]
            avwap_l = _vwap_from(p_idx)
            age_l = float(len(c_win) - 1 - p_idx)
            if np.isfinite(avwap_l) and np.isfinite(atr) and atr != 0:
                dist_l = (cur_close - avwap_l) / atr

        return avwap_h, avwap_l, dist_h, dist_l, age_h, age_l
    except Exception:
        return nan6


# ============================================================================
# Section C — IV Context
# ============================================================================

def _iv_percentile(symbol: str, asof: date_cls, ivx_30d_pct: float, lookback: int = 252) -> float:
    """Read prior ivx_30d from features store, return percentile rank of today's value.
    Returns NaN if < 20 prior obs.
    """
    if not np.isfinite(ivx_30d_pct):
        return _NAN
    try:
        from .storage import read_features_range  # local import to avoid circular
        from datetime import timedelta
        end = asof
        start = asof - timedelta(days=int(lookback * 1.5))
        hist = read_features_range(start, end)
        if hist.empty or "ivx_30d" not in hist.columns:
            return _NAN
        sym_hist = hist[hist["ticker"] == symbol]["ivx_30d"].dropna()
        if len(sym_hist) < 20:
            return _NAN
        pct = float((sym_hist < ivx_30d_pct).sum()) / len(sym_hist)
        return pct
    except Exception:
        return _NAN


def _iv_context(ivx_30d_pct: float, rv_20_val: float, iv_percentile_val: float) -> tuple[float, float, float]:
    """(iv_30, iv_premium, iv_premium_ratio) — all decimal. iv_30 = ivx_30d_pct/100."""
    iv_30 = ivx_30d_pct / 100.0 if np.isfinite(ivx_30d_pct) else _NAN
    if np.isfinite(iv_30) and np.isfinite(rv_20_val) and rv_20_val > 0:
        iv_premium = iv_30 - rv_20_val
        iv_premium_ratio = iv_30 / rv_20_val
    else:
        iv_premium = _NAN
        iv_premium_ratio = _NAN
    return iv_30, iv_premium, iv_premium_ratio


# ============================================================================
# Section D — Trend classifications
# ============================================================================

def _trend_price(close: pd.Series, atr14: pd.Series | None = None) -> str:
    """7-state price trend per master spec §3.2."""
    try:
        n = len(close)
        if n < 200:
            if n < 50:
                return "NEUTRAL"
            ma50 = float(close.iloc[-50:].mean())
            cur = float(close.iloc[-1])
            return "UP" if cur > ma50 else ("DOWN" if cur < ma50 else "NEUTRAL")
        cur = float(close.iloc[-1])
        ma20  = float(close.iloc[-20:].mean())
        ma50  = float(close.iloc[-50:].mean())
        ma200 = float(close.iloc[-200:].mean())
        # slope_10(sma_50): ATR-normalised slope of SMA50 over 10 bars
        if n >= 60 and atr14 is not None and len(atr14) >= 1 and np.isfinite(float(atr14.iloc[-1])) and float(atr14.iloc[-1]) > 0:
            ma50_now  = float(close.iloc[-50:].mean())
            ma50_prev = float(close.iloc[-60:-10].mean()) if n >= 60 else ma50_now
            slope_norm = (ma50_now - ma50_prev) / (10 * float(atr14.iloc[-1]))
        else:
            slope_norm = 0.0
        pos_slope = slope_norm > 0.02
        neg_slope = slope_norm < -0.02
        full_up   = cur > ma20 > ma50 > ma200
        full_down = cur < ma20 < ma50 < ma200
        loose_up  = cur > ma50 > ma200
        loose_down = cur < ma50 < ma200
        if full_up:
            return "STRONG_UP" if pos_slope else ("UP_EXHAUSTING" if not neg_slope else "UP")
        if full_down:
            return "STRONG_DOWN" if neg_slope else ("DOWN_EXHAUSTING" if not pos_slope else "DOWN")
        if loose_up:
            return "UP"
        if loose_down:
            return "DOWN"
        return "NEUTRAL"
    except Exception:
        return "NEUTRAL"


def _trend_rsi(rsi: float, close: pd.Series, period: int = 14) -> str:
    """BULLISH / BEARISH / NEUTRAL (>50&rising / <50&falling)."""
    try:
        if not np.isfinite(rsi):
            return "NEUTRAL"
        if len(close) < period + 2:
            return "NEUTRAL"
        prev_rsi = _rsi(close.iloc[:-1], period)
        if not np.isfinite(prev_rsi):
            return "NEUTRAL"
        rising = rsi > prev_rsi
        if rsi > 50 and rising:
            return "BULLISH"
        if rsi < 50 and not rising:
            return "BEARISH"
        return "NEUTRAL"
    except Exception:
        return "NEUTRAL"


def _trend_macd(macd: float, signal: float, hist: float,
                macd_prev: float | None = None, signal_prev: float | None = None) -> tuple[str, bool]:
    """Returns (trend_macd, macd_above_zero). trend_macd per spec §3.4."""
    try:
        if not all(np.isfinite(v) for v in [macd, signal, hist]):
            return "NEUTRAL", False
        above_zero = macd > 0
        # Cross detection using prev values
        if macd_prev is not None and signal_prev is not None and np.isfinite(macd_prev) and np.isfinite(signal_prev):
            bull_cross = (macd_prev <= signal_prev) and (macd > signal)
            bear_cross = (macd_prev >= signal_prev) and (macd < signal)
            if bull_cross:
                return "BULL_CROSS", above_zero
            if bear_cross:
                return "BEAR_CROSS", above_zero
        # Steady state
        if macd > signal:
            return "BULLISH", above_zero
        if macd < signal:
            return "BEARISH", above_zero
        return "NEUTRAL", above_zero
    except Exception:
        return "NEUTRAL", False


def _trend_bb(close: pd.Series, period: int = 20, k: float = 2.0) -> tuple[str, float, float, float, float]:
    """Returns (trend, bb_upper, bb_mid, bb_lower, bb_width). Trend per spec §3.5."""
    try:
        n = len(close)
        if n < period:
            return "NEUTRAL", _NAN, _NAN, _NAN, _NAN
        window = close.iloc[-period:]
        mid = float(window.mean())
        std = float(window.std(ddof=0))
        upper = mid + k * std
        lower = mid - k * std
        width = (upper - lower) / mid if mid != 0 else _NAN
        cur = float(close.iloc[-1])
        # SQUEEZE / EXPANSION: bb_width in bottom/top 20% of trailing 120-bar distribution
        squeeze = False
        expansion = False
        if n >= 120 and np.isfinite(width):
            widths = []
            for i in range(max(0, n - 120), n):
                sl = close.iloc[max(0, i - period + 1):i + 1]
                if len(sl) >= period:
                    m = float(sl.mean())
                    s = float(sl.std(ddof=0))
                    w = (m + k*s - (m - k*s)) / m if m != 0 else _NAN
                    if np.isfinite(w):
                        widths.append(w)
            if len(widths) >= 20:
                pct = sum(1 for w in widths if w < width) / len(widths)
                squeeze = pct <= 0.20
                expansion = pct >= 0.80
        if cur > upper:
            return "UPPER_BREAK", upper, mid, lower, width
        if cur < lower:
            return "LOWER_BREAK", upper, mid, lower, width
        if squeeze:
            return "SQUEEZE", upper, mid, lower, width
        if expansion:
            return "EXPANSION", upper, mid, lower, width
        return "NEUTRAL", upper, mid, lower, width
    except Exception:
        return "NEUTRAL", _NAN, _NAN, _NAN, _NAN


def _trend_cmf(cmf: float) -> str:
    """Per spec §3.6."""
    try:
        if not np.isfinite(cmf): return "NEUTRAL"
        if cmf > 0.10:  return "STRONG_ACCUM"
        if cmf > 0.05:  return "ACCUM"
        if cmf < -0.10: return "STRONG_DIST"
        if cmf < -0.05: return "DIST"
        return "NEUTRAL"
    except Exception:
        return "NEUTRAL"


def _trend_volume(volume: pd.Series, close: pd.Series, lookback: int = 60) -> tuple[str, bool, bool]:
    """Returns (trend, vol_confirms_up, vol_confirms_down). Per spec §3.7."""
    try:
        n = len(volume)
        if n < 2:
            return "NORMAL", False, False
        recent_vol = float(volume.iloc[-1])
        window = volume.iloc[-lookback:] if n >= lookback else volume
        vol_pct = float((window < recent_vol).sum()) / max(1, len(window))
        ret_1 = (float(close.iloc[-1]) / float(close.iloc[-2]) - 1) if len(close) >= 2 else 0.0
        trend = "SURGE" if vol_pct >= 0.85 else ("DRY" if vol_pct <= 0.15 else "NORMAL")
        vol_confirms_up   = (ret_1 > 0) and (vol_pct >= 0.70)
        vol_confirms_down = (ret_1 < 0) and (vol_pct >= 0.70)
        return trend, vol_confirms_up, vol_confirms_down
    except Exception:
        return "NORMAL", False, False


def _trend_obv(obv_cur: float, obv_sma20: float, obv_slope: float) -> str:
    """BULLISH / BEARISH / NEUTRAL."""
    try:
        if not np.isfinite(obv_cur):
            return "NEUTRAL"
        sma_ok = np.isfinite(obv_sma20)
        slope_ok = np.isfinite(obv_slope)
        above_sma = sma_ok and obv_cur > obv_sma20
        pos_slope = slope_ok and obv_slope > 0
        below_sma = sma_ok and obv_cur < obv_sma20
        neg_slope = slope_ok and obv_slope < 0
        if above_sma and pos_slope:
            return "BULLISH"
        if below_sma and neg_slope:
            return "BEARISH"
        return "NEUTRAL"
    except Exception:
        return "NEUTRAL"


def _trend_avwap(
    cur_close: float,
    avwap_h: float,
    avwap_l: float,
    dist_h: float,
    dist_l: float,
    prev_close: float | None = None,
) -> str:
    """RECLAIM_HIGH / BREAKDOWN_LOW / ABOVE_BOTH / BELOW_BOTH /
    TESTING_RESISTANCE / TESTING_SUPPORT / BETWEEN / UNAVAILABLE.
    """
    try:
        h_ok = np.isfinite(avwap_h) and np.isfinite(dist_h)
        l_ok = np.isfinite(avwap_l) and np.isfinite(dist_l)

        if not h_ok and not l_ok:
            return "UNAVAILABLE"

        # Crossing events (need prev_close)
        if prev_close is not None and np.isfinite(prev_close):
            if h_ok and prev_close < avwap_h and cur_close >= avwap_h:
                return "RECLAIM_HIGH"
            if l_ok and prev_close > avwap_l and cur_close <= avwap_l:
                return "BREAKDOWN_LOW"

        # Position states
        if h_ok and l_ok:
            above_h = cur_close > avwap_h
            above_l = cur_close > avwap_l
            if above_h and above_l:
                if 0 < dist_h < 0.5:  # within 0.5 ATR above resistance
                    return "TESTING_RESISTANCE"
                return "ABOVE_BOTH"
            if not above_h and not above_l:
                if -0.5 < dist_l < 0:  # within 0.5 ATR below support
                    return "TESTING_SUPPORT"
                return "BELOW_BOTH"
            return "BETWEEN"
        if h_ok:
            if cur_close > avwap_h:
                return "ABOVE_BOTH"
            return "BELOW_BOTH"
        if l_ok:
            if cur_close > avwap_l:
                return "ABOVE_BOTH"
            return "BELOW_BOTH"
        return "UNAVAILABLE"
    except Exception:
        return "UNAVAILABLE"


# ============================================================================
# Section E — Regime
# ============================================================================

_REGIME_MATRIX = {
    "TRENDING":   {"LOW_IV": 0.00, "NORMAL_IV": 0.00, "HIGH_IV": 0.50, "UNKNOWN": 0.00},
    "WEAK_TREND": {"LOW_IV": 0.50, "NORMAL_IV": 0.50, "HIGH_IV": 0.75, "UNKNOWN": 0.50},
    "RANGING":    {"LOW_IV": 0.50, "NORMAL_IV": 1.00, "HIGH_IV": 1.00, "UNKNOWN": 1.00},
}


def _adx_regime(adx: float) -> str:
    """TRENDING (>=25) / WEAK_TREND (>=20) / RANGING."""
    if not np.isfinite(adx):
        return "RANGING"
    if adx >= 25:
        return "TRENDING"
    if adx >= 20:
        return "WEAK_TREND"
    return "RANGING"


def _vol_regime(iv_percentile: float) -> str:
    """LOW_IV (<0.30) / NORMAL_IV / HIGH_IV (>0.70) / UNKNOWN."""
    if not np.isfinite(iv_percentile):
        return "UNKNOWN"
    if iv_percentile < 0.30:
        return "LOW_IV"
    if iv_percentile > 0.70:
        return "HIGH_IV"
    return "NORMAL_IV"


# ============================================================================
# Section F — Divergence
# ============================================================================

def _rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
    """Full RSI series."""
    try:
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100.0 - 100.0 / (1.0 + rs)
        return rsi
    except Exception:
        return pd.Series(_NAN, index=close.index)


def _cmf_series(high: pd.Series, low: pd.Series, close: pd.Series, vol: pd.Series, period: int = 20) -> pd.Series:
    """Full CMF series."""
    try:
        rng = (high - low).replace(0, np.nan)
        mfm = ((close - low) - (high - close)) / rng
        mfv = (mfm * vol).fillna(0.0)
        cum_mfv = mfv.rolling(period, min_periods=period).sum()
        cum_vol = vol.rolling(period, min_periods=period).sum().replace(0, np.nan)
        return cum_mfv / cum_vol
    except Exception:
        return pd.Series(_NAN, index=close.index)


def _macd_hist_series(close: pd.Series) -> pd.Series:
    """Full MACD histogram series: (EMA12 - EMA26) - EMA9."""
    try:
        macd_line = _ema(close, 12) - _ema(close, 26)
        signal = _ema(macd_line, 9)
        return macd_line - signal
    except Exception:
        return pd.Series(_NAN, index=close.index)


def _obv_series(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Full OBV series."""
    try:
        direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        direction.iloc[0] = 0
        return (direction * volume).cumsum()
    except Exception:
        return pd.Series(0.0, index=close.index)


def _macd_line_series(close: pd.Series) -> pd.Series:
    """Full MACD line: EMA12 - EMA26."""
    try:
        return _ema(close, 12) - _ema(close, 26)
    except Exception:
        return pd.Series(_NAN, index=close.index)


def _atr14_series(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Full 14-bar ATR series (Wilder)."""
    try:
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(alpha=1.0 / 14, adjust=False, min_periods=14).mean()
    except Exception:
        return pd.Series(_NAN, index=close.index)


def _detect_divergence_active(
    close: pd.Series,
    indicator: pd.Series,
    div_type: str,   # "reg_bull" | "reg_bear" | "hid_bull" | "hid_bear"
    sma20: pd.Series,
    expiry_bars: int = 20,
    effect_bars: int = 15,
    pivot_left: int = 5,
    pivot_right: int = 5,
    quality_fn=None,
    atr_series: pd.Series | None = None,
    close_series: pd.Series | None = None,
) -> list[dict]:
    """Returns list of currently-active divergence events.
    Supports regular divergences (reg_bull, reg_bear) and hidden divergences
    (hid_bull, hid_bear). Pass atr_series for ATR-normalized price move threshold.
    Pass close_series (same as close, for hidden div SMA50 slope check).
    """
    try:
        close_arr = close.values
        ind_arr = indicator.values
        sma20_arr = sma20.values
        n = len(close_arr)
        if n < pivot_left + pivot_right + 2:
            return []

        is_hidden = div_type in ("hid_bull", "hid_bear")
        is_bull = div_type in ("reg_bull", "hid_bull")

        # Find pivot indices for the relevant type
        if is_bull:
            _, pivot_idxs = _find_pivots(close, left=pivot_left, right=pivot_right)
        else:
            pivot_idxs, _ = _find_pivots(close, left=pivot_left, right=pivot_right)

        # Only confirmed pivots (right bars in)
        max_confirmed_pivot = n - pivot_right - 1
        pivot_idxs = [p for p in pivot_idxs if p <= max_confirmed_pivot]

        if len(pivot_idxs) < 2:
            return []

        results = []
        # Check consecutive pairs
        for i in range(len(pivot_idxs) - 1):
            p1 = pivot_idxs[i]
            p2 = pivot_idxs[i + 1]

            c1, c2 = close_arr[p1], close_arr[p2]
            ind1, ind2 = ind_arr[p1], ind_arr[p2]

            if not (np.isfinite(c1) and np.isfinite(c2) and np.isfinite(ind1) and np.isfinite(ind2)):
                continue

            # ATR-normalised price move must be >= 0.25 (spec §4.3)
            if atr_series is not None:
                atr_at_p2 = float(atr_series.iloc[p2]) if p2 < len(atr_series) else _NAN
                if np.isfinite(atr_at_p2) and atr_at_p2 > 0:
                    price_move_norm = abs(c2 - c1) / atr_at_p2
                    if price_move_norm < 0.25:
                        continue

            # Divergence condition
            if div_type == "reg_bull":
                price_div = c2 < c1          # lower low in price
                ind_div = ind2 > ind1         # higher low in indicator
            elif div_type == "reg_bear":
                price_div = c2 > c1          # higher high in price
                ind_div = ind2 < ind1         # lower high in indicator
            elif div_type == "hid_bull":
                # Hidden bullish: price higher low + indicator lower low
                price_div = c2 > c1
                ind_div = ind2 < ind1
            else:  # hid_bear
                # Hidden bearish: price lower high + indicator higher high
                price_div = c2 < c1
                ind_div = ind2 > ind1

            if not (price_div and ind_div):
                continue

            # Hidden divergence: check SMA50 slope direction
            if is_hidden and close_series is not None:
                cs = close_series
                cs_n = len(cs)
                if cs_n >= 60:
                    ma50_now = float(cs.iloc[-50:].mean())
                    ma50_prev = float(cs.iloc[-60:-10].mean())
                    slope_up = ma50_now > ma50_prev
                    slope_down = ma50_now < ma50_prev
                    if div_type == "hid_bull" and not slope_up:
                        continue
                    if div_type == "hid_bear" and not slope_down:
                        continue

            # Quality check — pass ind value at p1 (per Fix 10)
            if quality_fn is not None:
                quality = quality_fn(ind1)
            else:
                quality = "standard"

            if is_hidden:
                # Hidden divergence: immediate confirmed state (no state machine)
                results.append({
                    "type": div_type,
                    "state": "confirmed",
                    "p1_idx": p1,
                    "p2_idx": p2,
                    "confirmed_idx": p2,
                    "quality": quality,
                    "ind_p1": ind1,
                    "ind_p2": ind2,
                })
                continue

            # Regular divergence: state machine from p2 forward
            state = "pending"
            confirmed_idx = None

            for j in range(p2 + 1, n):
                cj = close_arr[j]
                if div_type == "reg_bull":
                    sma20_j = sma20_arr[j] if j < len(sma20_arr) else _NAN
                    confirmed = (
                        np.isfinite(cj) and np.isfinite(c1) and cj > c1 and
                        np.isfinite(sma20_j) and cj > sma20_j
                    )
                else:
                    sma20_j = sma20_arr[j] if j < len(sma20_arr) else _NAN
                    confirmed = (
                        np.isfinite(cj) and np.isfinite(c1) and cj < c1 and
                        np.isfinite(sma20_j) and cj < sma20_j
                    )
                if confirmed:
                    state = "confirmed"
                    confirmed_idx = j
                    break
                if j - p2 > expiry_bars:
                    state = "expired"
                    break

            # Filter: include only confirmed-within-effect_bars or pending
            if state == "expired":
                continue
            if state == "confirmed":
                bars_since_confirm = (n - 1) - confirmed_idx
                if bars_since_confirm > effect_bars:
                    continue

            results.append({
                "type": div_type,
                "state": state,
                "p1_idx": p1,
                "p2_idx": p2,
                "confirmed_idx": confirmed_idx,
                "quality": quality,
                "ind_p1": ind1,
                "ind_p2": ind2,
            })

        return results
    except Exception:
        return []


def _run_divergence_suite(
    close: pd.Series, high: pd.Series, low: pd.Series, volume: pd.Series, sma20: pd.Series
) -> dict[str, list[dict]]:
    """Run divergence detection for RSI, MACD line, MACD hist, CMF, OBV.
    Both regular and hidden divergences. Returns {name: [events]}.
    """
    try:
        rsi = _rsi_series(close)
        macd_line = _macd_line_series(close)
        macd_h = _macd_hist_series(close)
        cmf = _cmf_series(high, low, close, volume)
        obv = _obv_series(close, volume)
        atr = _atr14_series(high, low, close)

        # Quality functions — receive ind value at p1 (Fix 10)
        def rsi_bull_quality(val_at_p1):
            return "strong" if np.isfinite(val_at_p1) and val_at_p1 < 30 else "standard"

        def rsi_bear_quality(val_at_p1):
            return "strong" if np.isfinite(val_at_p1) and val_at_p1 > 70 else "standard"

        def macd_line_bull_quality(val_at_p1):
            return "strong" if np.isfinite(val_at_p1) and val_at_p1 < 0 else "standard"

        def macd_line_bear_quality(val_at_p1):
            return "strong" if np.isfinite(val_at_p1) and val_at_p1 > 0 else "standard"

        def macd_hist_bull_quality(val_at_p1):
            return "strong" if np.isfinite(val_at_p1) and val_at_p1 < 0 else "standard"

        def macd_hist_bear_quality(val_at_p1):
            return "strong" if np.isfinite(val_at_p1) and val_at_p1 > 0 else "standard"

        def cmf_bull_quality(val_at_p1):
            return "strong" if np.isfinite(val_at_p1) and val_at_p1 < -0.05 else "standard"

        def cmf_bear_quality(val_at_p1):
            return "strong" if np.isfinite(val_at_p1) and val_at_p1 > 0.05 else "standard"

        def obv_quality(_val):
            return "standard"  # unbounded, no natural extreme zone per spec §4.4

        # (name, indicator, div_type, quality_fn, include_hidden)
        # MACD line: no hidden divergence per spec
        # OBV: no quality boost for hidden per spec
        suite = [
            # RSI: reg + hidden
            ("RSI_reg_bull",  rsi,       "reg_bull",  rsi_bull_quality,       True),
            ("RSI_reg_bear",  rsi,       "reg_bear",  rsi_bear_quality,       True),
            ("RSI_hid_bull",  rsi,       "hid_bull",  rsi_bull_quality,       False),
            ("RSI_hid_bear",  rsi,       "hid_bear",  rsi_bear_quality,       False),
            # MACD line: reg only
            ("MACDLine_reg_bull", macd_line, "reg_bull", macd_line_bull_quality, False),
            ("MACDLine_reg_bear", macd_line, "reg_bear", macd_line_bear_quality, False),
            # MACD hist: reg + hidden
            ("MACDHist_reg_bull", macd_h,  "reg_bull", macd_hist_bull_quality, True),
            ("MACDHist_reg_bear", macd_h,  "reg_bear", macd_hist_bear_quality, True),
            ("MACDHist_hid_bull", macd_h,  "hid_bull", macd_hist_bull_quality, False),
            ("MACDHist_hid_bear", macd_h,  "hid_bear", macd_hist_bear_quality, False),
            # CMF: reg + hidden
            ("CMF_reg_bull",  cmf,       "reg_bull",  cmf_bull_quality,       True),
            ("CMF_reg_bear",  cmf,       "reg_bear",  cmf_bear_quality,       True),
            ("CMF_hid_bull",  cmf,       "hid_bull",  cmf_bull_quality,       False),
            ("CMF_hid_bear",  cmf,       "hid_bear",  cmf_bear_quality,       False),
            # OBV: reg + hidden (no quality boost for hidden — obv_quality always "standard")
            ("OBV_reg_bull",  obv,       "reg_bull",  obv_quality,            True),
            ("OBV_reg_bear",  obv,       "reg_bear",  obv_quality,            True),
            ("OBV_hid_bull",  obv,       "hid_bull",  obv_quality,            False),
            ("OBV_hid_bear",  obv,       "hid_bear",  obv_quality,            False),
        ]

        results = {}
        for name, indicator, div_type, q_fn, _inc in suite:
            results[name] = _detect_divergence_active(
                close, indicator, div_type, sma20,
                quality_fn=q_fn,
                atr_series=atr,
                close_series=close,
            )
        return results
    except Exception:
        return {}


# ============================================================================
# Section G — Scoring
# ============================================================================

def _score_technical(
    trend_price: str,
    trend_macd: str,
    macd_above_zero: bool,
    trend_rsi: str,
    trend_bb: str,
    trend_cmf: str,
    vol_confirms_up: bool,
    vol_confirms_down: bool,
    trend_obv: str,
    trend_avwap: str,
    rsi_val: float,
    close_cur: float,
    bb_upper: float,
    bb_lower: float,
    bb_mid: float,
    close_series: pd.Series,
    macd_hist_series: pd.Series,
    adx_regime: str,
    vol_regime: str,
    div_results: dict,
) -> dict:
    """Returns comprehensive scoring dict with all components for commentary."""
    try:
        # --- Trend score (master spec §5) ---
        trend_score = 0.0

        # Price: 7 states
        trend_score += {
            "STRONG_UP": 3.0, "UP": 1.5, "UP_EXHAUSTING": 0.75,
            "NEUTRAL": 0.0,
            "DOWN_EXHAUSTING": -0.75, "DOWN": -1.5, "STRONG_DOWN": -3.0,
        }.get(trend_price, 0.0)

        # MACD: steady state + cross bonus + above-zero bonus
        macd_steady = {
            "BULLISH": 2.0, "BEARISH": -2.0,
            "BULL_CROSS": 2.0, "BEAR_CROSS": -2.0,
            "NEUTRAL": 0.0,
        }.get(trend_macd, 0.0)
        macd_cross_bonus = 1.5 if trend_macd == "BULL_CROSS" else (-1.5 if trend_macd == "BEAR_CROSS" else 0.0)
        trend_score += macd_steady + macd_cross_bonus
        trend_score += 0.5 if macd_above_zero else -0.5

        # CMF: 4 states
        trend_score += {
            "STRONG_ACCUM": 1.0, "ACCUM": 0.5,
            "DIST": -0.5, "STRONG_DIST": -1.0,
            "NEUTRAL": 0.0,
        }.get(trend_cmf, 0.0)

        # OBV
        if trend_obv == "BULLISH":
            trend_score += 1.0
        elif trend_obv == "BEARISH":
            trend_score -= 1.0

        # aVWAP
        avwap_contrib = {
            "RECLAIM_HIGH": 1.5, "BREAKDOWN_LOW": -1.5,
            "ABOVE_BOTH": 0.75, "BELOW_BOTH": -0.75,
            "TESTING_SUPPORT": 0.5, "TESTING_RESISTANCE": -0.5,
        }.get(trend_avwap, 0.0)
        trend_score += avwap_contrib

        # Volume: vol_confirms_up/down flags (±0.5), directionally weighted
        if vol_confirms_up:
            trend_score += 0.5
        if vol_confirms_down:
            trend_score -= 0.5

        # --- Convergence multiplier ---
        conv_mult = 1.0
        try:
            if len(close_series) >= 6 and len(macd_hist_series) >= 6:
                hist_recent = float(macd_hist_series.iloc[-1])
                hist_prev = float(macd_hist_series.iloc[-6])
                macd_dir = 1 if hist_recent > hist_prev else (-1 if hist_recent < hist_prev else 0)
                ts_dir = 1 if trend_score > 0 else (-1 if trend_score < 0 else 0)
                if macd_dir != 0 and ts_dir != 0 and macd_dir == ts_dir:
                    conv_mult = 1.25
        except Exception:
            pass

        base_trend = trend_score * conv_mult

        # --- Mean reversion score (master spec §5.3) ---
        # RSI: BULLISH zone (20-30) = +1; BEARISH zone (70-80) = -1
        #      OVERSOLD (≤20) = -0.5 (extreme, fade the reversion signal); OVERBOUGHT (≥80) = +0.5
        # BB: LOWER_BREAK = +0.5; UPPER_BREAK = -0.5
        mr_score = 0.0
        rsi_label = "neutral"
        bb_label = "neutral"

        if np.isfinite(rsi_val):
            if rsi_val <= 20:
                mr_score -= 0.5
                rsi_label = "extreme-oversold"
            elif rsi_val < 30:
                mr_score += 1.0
                rsi_label = "oversold"
            elif rsi_val >= 80:
                mr_score += 0.5
                rsi_label = "extreme-overbought"
            elif rsi_val > 70:
                mr_score -= 1.0
                rsi_label = "overbought"

        if all(np.isfinite(v) for v in [close_cur, bb_upper, bb_lower]):
            if close_cur > bb_upper:
                mr_score -= 0.5
                bb_label = "upper-break"
            elif close_cur < bb_lower:
                mr_score += 0.5
                bb_label = "lower-break"

        mr_weight = _REGIME_MATRIX.get(adx_regime, _REGIME_MATRIX["RANGING"]).get(
            vol_regime, 0.0
        )
        mr_contribution = mr_weight * mr_score

        # --- Divergence score (master spec §5.4, CAPPED) ---
        # Count DISTINCT indicators (not per-event) with a confirmed divergence so
        # repeated pivot pairs on the same indicator can't run the score away:
        #   regular:  n == 1 -> ±2.0,  n >= 2 -> ±4.0
        #   quality:  +1.0 / -1.0 once if ANY confirmed regular div of that
        #             direction had a "strong" quality boost
        #   hidden:   ±1.0 once if ANY confirmed hidden div of that direction exists
        reg_bull_inds: set[str] = set()
        reg_bear_inds: set[str] = set()
        hid_bull_any = hid_bear_any = False
        q_reg_bull = q_reg_bear = False

        for name, events in div_results.items():
            indicator = name.rsplit("_", 2)[0]   # "RSI_reg_bull" -> "RSI"
            for ev in events:
                if ev.get("state") != "confirmed":
                    continue
                strong = ev.get("quality") == "strong"
                t = ev["type"]
                if t == "reg_bull":
                    reg_bull_inds.add(indicator)
                    q_reg_bull = q_reg_bull or strong
                elif t == "reg_bear":
                    reg_bear_inds.add(indicator)
                    q_reg_bear = q_reg_bear or strong
                elif t == "hid_bull":
                    hid_bull_any = True
                elif t == "hid_bear":
                    hid_bear_any = True

        n_reg_bull = len(reg_bull_inds)
        n_reg_bear = len(reg_bear_inds)

        div_score = 0.0
        div_score += 4.0 if n_reg_bull >= 2 else (2.0 if n_reg_bull == 1 else 0.0)
        div_score -= 4.0 if n_reg_bear >= 2 else (2.0 if n_reg_bear == 1 else 0.0)
        if q_reg_bull:
            div_score += 1.0
        if q_reg_bear:
            div_score -= 1.0
        if hid_bull_any:
            div_score += 1.0
        if hid_bear_any:
            div_score -= 1.0

        # Distinct-indicator confirmed counts (incl. hidden presence) for reporting.
        bull_confirmed = n_reg_bull + (1 if hid_bull_any else 0)
        bear_confirmed = n_reg_bear + (1 if hid_bear_any else 0)

        # --- Final score ---
        raw = base_trend + mr_contribution + div_score
        score = max(-10.0, min(10.0, raw))

        # Recommendation without hysteresis (hysteresis applied in compute_stage5_5)
        if score >= 5.5:
            rec = "STRONG_BUY"
        elif score >= 2.5:
            rec = "BUY"
        elif score <= -5.5:
            rec = "STRONG_SELL"
        elif score <= -2.5:
            rec = "SELL"
        else:
            rec = "HOLD"

        conf = min(1.0, abs(score) / 10.0)

        return {
            "score": score,
            "raw": raw,
            "rec": rec,
            "conf": conf,
            "trend_score": trend_score,
            "conv_mult": conv_mult,
            "base_trend": base_trend,
            "mr_score": mr_score,
            "mr_weight": mr_weight,
            "mr_contribution": mr_contribution,
            "div_score": div_score,
            "bull_confirmed": bull_confirmed,
            "bear_confirmed": bear_confirmed,
            "rsi_label": rsi_label,
            "bb_label": bb_label,
            "avwap_contrib": avwap_contrib,
        }
    except Exception:
        return {
            "score": 0.0, "raw": 0.0, "rec": "HOLD", "conf": 0.0,
            "trend_score": 0.0, "conv_mult": 1.0, "base_trend": 0.0,
            "mr_score": 0.0, "mr_weight": 0.0, "mr_contribution": 0.0,
            "div_score": 0.0, "bull_confirmed": 0, "bear_confirmed": 0,
            "rsi_label": "neutral", "bb_label": "neutral", "avwap_contrib": 0.0,
        }


# ============================================================================
# Section H-0 — Hysteresis (Schmitt trigger)
# ============================================================================

def _apply_hysteresis(score: float, prev_rec: str | None) -> str:
    """Schmitt-trigger hysteresis per master spec §5.5.

    Enter thresholds:  STRONG_BUY ≥5.5,  BUY ≥2.5,  SELL ≤-2.5,  STRONG_SELL ≤-5.5
    Exit  thresholds:  STRONG_BUY <4.5,  BUY <1.5,  SELL >-1.5,   STRONG_SELL >-4.5
    HOLD is the default when no other state applies.
    """
    if prev_rec is None:
        prev_rec = "HOLD"

    if prev_rec == "STRONG_BUY":
        return "STRONG_BUY" if score >= 4.5 else ("BUY" if score >= 1.5 else ("HOLD" if score > -2.5 else ("SELL" if score > -5.5 else "STRONG_SELL")))
    if prev_rec == "BUY":
        return "STRONG_BUY" if score >= 5.5 else ("BUY" if score >= 1.5 else ("HOLD" if score > -2.5 else ("SELL" if score > -5.5 else "STRONG_SELL")))
    if prev_rec == "SELL":
        return "STRONG_SELL" if score <= -5.5 else ("SELL" if score <= -1.5 else ("HOLD" if score < 2.5 else ("BUY" if score < 5.5 else "STRONG_BUY")))
    if prev_rec == "STRONG_SELL":
        return "STRONG_SELL" if score <= -4.5 else ("SELL" if score <= -1.5 else ("HOLD" if score < 2.5 else ("BUY" if score < 5.5 else "STRONG_BUY")))
    # HOLD: use standard enter thresholds
    if score >= 5.5:
        return "STRONG_BUY"
    if score >= 2.5:
        return "BUY"
    if score <= -5.5:
        return "STRONG_SELL"
    if score <= -2.5:
        return "SELL"
    return "HOLD"


# ============================================================================
# Section H — Commentary builders
# ============================================================================

def _build_technical_commentary(
    history_df: pd.DataFrame,
    asof: date_cls,
    symbol: str = "",
    ivx_30d_pct: float = _NAN,
) -> str:
    """Build the full technical commentary string. Returns empty on failure/warmup."""
    try:
        if history_df is None or history_df.empty:
            return ""
        h = history_df[history_df.index <= pd.Timestamp(asof)] if hasattr(history_df.index, 'freq') or isinstance(history_df.index, pd.DatetimeIndex) else history_df

        required_cols = {"Close", "High", "Low", "Volume"}
        if not required_cols.issubset(h.columns):
            return ""
        if len(h) < 26:
            return ""

        close = h["Close"].astype(float)
        high = h["High"].astype(float)
        low = h["Low"].astype(float)
        volume = h["Volume"].astype(float)

        # Core scalars
        macd_line = _ema(close, 12) - _ema(close, 26)
        signal_line = _ema(macd_line, 9)
        macd_val = float(macd_line.iloc[-1])
        signal_val = float(signal_line.iloc[-1])
        hist_val = float((macd_line - signal_line).iloc[-1])
        rsi_val = _rsi(close, 14)
        cmf_val = _cmf(high, low, close, volume, 20)
        close_cur = float(close.iloc[-1])

        # Additional indicators
        adx_val, pdi_val, mdi_val = _adx(high, low, close)
        obv_cur, obv_sma20, obv_slope = _obv_sma(close, volume)
        rv20 = _rv_20(close)
        avwap_h, avwap_l, dist_h, dist_l, age_h, age_l = _anchored_vwap(close, high, low, volume)

        # IV context
        iv_pct = _iv_percentile(symbol, asof, ivx_30d_pct) if symbol else _NAN
        iv30, iv_premium, iv_premium_ratio = _iv_context(ivx_30d_pct, rv20, iv_pct)

        # SMA20 for divergence
        sma20_series = close.rolling(20, min_periods=20).mean()

        # ATR14 series for trend_price slope and divergence ATR filter
        atr14_s = _atr14_series(high, low, close)

        # Trend classifications (new signatures — note unpacking for tuple-returning fns)
        t_price = _trend_price(close, atr14_s)

        macd_line_prev = float(macd_line.iloc[-2]) if len(macd_line) >= 2 else None
        signal_line_prev = float(signal_line.iloc[-2]) if len(signal_line) >= 2 else None
        t_macd, macd_above_zero = _trend_macd(
            macd_val, signal_val, hist_val,
            macd_prev=macd_line_prev, signal_prev=signal_line_prev,
        )

        t_rsi = _trend_rsi(rsi_val, close, 14)
        t_bb, bb_upper, bb_mid, bb_lower, bb_width = _trend_bb(close)
        t_cmf = _trend_cmf(cmf_val)
        t_vol, vol_confirms_up, vol_confirms_down = _trend_volume(volume, close)
        t_obv = _trend_obv(obv_cur, obv_sma20, obv_slope)

        # prev_close for aVWAP crossing
        prev_close = float(close.iloc[-2]) if len(close) >= 2 else None
        t_avwap = _trend_avwap(close_cur, avwap_h, avwap_l, dist_h, dist_l, prev_close)

        # Regime
        adx_reg = _adx_regime(adx_val)
        vol_reg = _vol_regime(iv_pct)

        # Divergence suite
        div_results = _run_divergence_suite(close, high, low, volume, sma20_series)

        # MACD hist series for convergence
        macd_hist_s = macd_line - signal_line

        # Score
        scoring = _score_technical(
            t_price, t_macd, macd_above_zero, t_rsi, t_bb, t_cmf,
            vol_confirms_up, vol_confirms_down,
            t_obv, t_avwap,
            rsi_val, close_cur, bb_upper, bb_lower, bb_mid,
            close, macd_hist_s,
            adx_reg, vol_reg, div_results,
        )

        score = scoring["score"]
        rec = scoring["rec"]
        conf = scoring["conf"]
        base_trend = scoring["base_trend"]
        trend_score = scoring["trend_score"]
        conv_mult = scoring["conv_mult"]
        mr_weight = scoring["mr_weight"]
        mr_contribution = scoring["mr_contribution"]
        div_score = scoring["div_score"]
        raw = scoring["raw"]
        rsi_label = scoring["rsi_label"]
        bb_label = scoring["bb_label"]

        # --- Format commentary ---
        lines = []

        # Header
        lines.append(f"### Technical Signal: {rec} ({score:+.1f} / confidence {conf:.0%})")
        lines.append("")

        # Regime
        adx_str = f"ADX {adx_val:.0f}, +DI {pdi_val:.0f} vs -DI {mdi_val:.0f}" if np.isfinite(adx_val) else "ADX unavailable"
        iv_str = f"{iv30:.1%}" if np.isfinite(iv30) else "N/A"
        rv_str = f"{rv20:.1%}" if np.isfinite(rv20) else "N/A"
        lines.append(f"**Regime**: ADX {adx_reg} ({adx_str})  |  Vol: {vol_reg} (IV30 {iv_str}, RV20 {rv_str})")
        lines.append(f"**Gate**: MR weight = {mr_weight:.2f}")
        lines.append("")

        # Trend
        lines.append(f"**Trend** ({base_trend:+.2f} = {trend_score:+.2f} × {conv_mult:.2f} convergence)")
        lines.append(f"- Price: {t_price}  MACD: {t_macd}  CMF: {t_cmf}")
        lines.append(f"- OBV: {t_obv}  aVWAP: {t_avwap}  Volume: {t_vol}")

        # aVWAP anchor info
        avwap_parts = []
        if np.isfinite(avwap_h) and np.isfinite(age_h):
            avwap_parts.append(f"aVWAP-high {avwap_h:.2f} (anchor {age_h:.0f} bars ago)")
        if np.isfinite(avwap_l) and np.isfinite(age_l):
            avwap_parts.append(f"aVWAP-low {avwap_l:.2f} (anchor {age_l:.0f} bars ago)")
        if avwap_parts:
            lines.append(f"- {', '.join(avwap_parts)}")
        lines.append("")

        # Mean reversion
        lines.append(f"**Mean Reversion** (gate {mr_weight:.2f} → contribution {mr_contribution:+.2f})")
        rsi_str = f"{rsi_val:.0f}" if np.isfinite(rsi_val) else "N/A"
        lines.append(f"- RSI {rsi_str}: {rsi_label}  Bollinger: {bb_label}")
        lines.append("")

        # Divergence
        lines.append(f"**Divergence** ({div_score:+.1f})")
        # Dedup to one line per (state, type, indicator) — the score counts
        # distinct indicators, so listing every pivot pair would misrepresent it.
        # Prefer a "strong" quality label if any event for that key was strong.
        div_best: dict[tuple, str] = {}
        div_order: list[tuple] = []
        for name, events in div_results.items():
            indicator_label = name.rsplit("_", 2)[0]
            for ev in events:
                key = (ev["state"], ev["type"], indicator_label)
                if key not in div_best:
                    div_order.append(key)
                if ev.get("quality") == "strong" or key not in div_best:
                    div_best[key] = ev.get("quality", "standard")
        if div_order:
            for state, dtype, indicator_label in div_order:
                icon = "✓ CONFIRMED" if state == "confirmed" else "⏳ PENDING"
                quality = div_best[(state, dtype, indicator_label)]
                lines.append(f"- {icon} {dtype} on {indicator_label} (quality: {quality})")
        else:
            lines.append("- No active divergences")
        lines.append("")

        # Score breakdown
        lines.append(
            f"**Score**: {base_trend:+.2f} + {mr_contribution:+.2f} + {div_score:+.2f} "
            f"= {raw:+.2f} → **{score:+.2f}**"
        )

        # Caveats
        caveats = []
        if len(close) < 200:
            caveats.append(f"Note: Only {len(close)} bars — price-trend SMA200 unavailable (warmup)")
        if not np.isfinite(adx_val):
            caveats.append("Note: ADX unavailable — insufficient history for Wilder smoothing")
        if not np.isfinite(iv30):
            caveats.append("Note: IV30 not available — using RV-only vol regime")
        for cav in caveats:
            lines.append(cav)

        return "\n".join(lines)

    except Exception as exc:
        logger.debug("technical commentary build failed: %s", exc)
        return ""


def _build_analyst_commentary(scalars: dict, symbol: str, asof: date_cls) -> str:
    """Build analyst view commentary. Returns empty if all analyst scalars are NaN."""
    try:
        n = scalars.get("analyst_coverage_count", _NAN)
        pct_buy = scalars.get("analyst_pct_buy", _NAN)
        mean_cons = scalars.get("analyst_consensus", _NAN)
        up = scalars.get("analyst_upgrades_3m", _NAN)
        down = scalars.get("analyst_downgrades_3m", _NAN)
        steady = scalars.get("analyst_steady_3m", _NAN)

        # All NaN → skip
        if all(not np.isfinite(v) for v in [n, pct_buy, mean_cons, up, down, steady]):
            return ""

        lines = ["### Analyst View", ""]

        n_str = f"{n:.0f}" if np.isfinite(n) else "N/A"
        pct_str = f"{pct_buy:.0%}" if np.isfinite(pct_buy) else "N/A"
        mean_str = f"{mean_cons:.1f}" if np.isfinite(mean_cons) else "N/A"
        lines.append(f"**Coverage**: {n_str} analysts  |  Buy/OW: {pct_str}  |  Consensus mean: {mean_str}")

        up_str = f"{up:.0f}" if np.isfinite(up) else "N/A"
        down_str = f"{down:.0f}" if np.isfinite(down) else "N/A"
        steady_str = f"{steady:.0f}" if np.isfinite(steady) else "N/A"
        lines.append(f"**90d rating changes**: {up_str} upgrades  ·  {down_str} downgrades  ·  {steady_str} maintained")

        # Evolution: read the last cached snapshot from storage
        try:
            from .storage import read_latest_fundamentals
            prev = read_latest_fundamentals(symbol, "analyst") if symbol else None
        except Exception:
            prev = None

        if prev is None:
            lines.append("No prior cache — first reading")
        else:
            prev_n = prev.get("analyst_coverage_count", _NAN)
            prev_pct = prev.get("analyst_pct_buy", _NAN)
            evol_parts = []
            if np.isfinite(n) and np.isfinite(prev_n) and n != prev_n:
                evol_parts.append(f"Coverage {'rising' if n > prev_n else 'falling'} (was {prev_n:.0f}, now {n:.0f})")
            if np.isfinite(pct_buy) and np.isfinite(prev_pct):
                if pct_buy > prev_pct:
                    evol_parts.append(f"Buy% improving ({prev_pct:.0%} → {pct_buy:.0%})")
                elif pct_buy < prev_pct:
                    evol_parts.append(f"Buy% declining ({prev_pct:.0%} → {pct_buy:.0%})")
            if np.isfinite(up) and np.isfinite(down) and up > down:
                evol_parts.append("Net upgrades in the period — bullish sentiment trend")
            elif np.isfinite(up) and np.isfinite(down) and down > up:
                evol_parts.append("Net downgrades in the period — bearish sentiment trend")
            if evol_parts:
                lines.append(", ".join(evol_parts))

        return "\n".join(lines)

    except Exception as exc:
        logger.debug("analyst commentary build failed: %s", exc)
        return ""


# ============================================================================
# Public API — updated signatures
# ============================================================================

def compute_technicals(
    history: pd.DataFrame | None,
    asof: date_cls,
    symbol: str = "",
    ivx_30d: float = _NAN,
) -> dict:
    """MACD / RSI / CMF / Bollinger from OHLCV up to `asof`. All-NaN if no history.

    Also returns `technical_commentary` built from the full indicator suite.
    """
    out = {k: _NAN for k in TECHNICAL_KEYS}
    out["technical_commentary"] = ""

    if history is None or history.empty or "Close" not in history.columns:
        return out
    h = history[history.index <= pd.Timestamp(asof)]
    if h.empty:
        return out

    close = h["Close"].astype(float)
    high = h["High"].astype(float)
    low = h["Low"].astype(float)
    vol = h["Volume"].astype(float)

    if len(close) >= 26:
        macd_line = _ema(close, 12) - _ema(close, 26)
        signal = _ema(macd_line, 9)
        out["macd"] = float(macd_line.iloc[-1])
        out["macd_signal"] = float(signal.iloc[-1])
        out["macd_hist"] = float((macd_line - signal).iloc[-1])

    out["rsi_14"] = _rsi(close, 14)
    out["cmf_20"] = _cmf(high, low, close, vol, 20)
    out["bb_upper"], out["bb_mid"], out["bb_lower"], out["bb_width"] = _bollinger(close, 20, 2.0)

    # Commentary
    out["technical_commentary"] = _build_technical_commentary(h, asof, symbol, ivx_30d)
    return out


# ---- Fundamentals acquisition (Feature 3, network) -------------------------

def acquire_fundamentals(symbol: str, ticker_obj) -> dict | None:
    """Fetch the raw yfinance objects Stage 5.5 needs. Returns a bundle dict, or
    None if `ticker_obj` is missing. Each attribute is fetched defensively so one
    failed endpoint doesn't lose the rest. This is a network boundary — callers
    gate it on `fundamentals.enabled` and skip it for index ETFs.
    """
    if ticker_obj is None:
        return None

    def _safe(attr: str):
        try:
            return getattr(ticker_obj, attr)
        except Exception as e:
            logger.warning("fundamentals fetch %s.%s failed: %s", symbol, attr, e)
            return None

    return {
        "info": _safe("info") or {},
        "earnings_history": _safe("earnings_history"),
        "eps_trend": _safe("eps_trend"),
        "recommendations": _safe("recommendations"),
        "upgrades_downgrades": _safe("upgrades_downgrades"),
        "ttm_income_stmt": _safe("ttm_income_stmt"),
    }


# ---- Fundamentals extraction (Feature 3, pure) -----------------------------

def _earnings_metrics(eh: pd.DataFrame | None) -> tuple[float, float]:
    """(mean surprise % over last 4 reported quarters, consecutive recent beats)."""
    if eh is None or not hasattr(eh, "empty") or eh.empty or "surprisePercent" not in eh.columns:
        return _NAN, _NAN
    df = eh.sort_index()  # ascending by quarter date; most recent last
    sp = df["surprisePercent"].dropna()
    if sp.empty:
        return _NAN, _NAN
    avg4 = float(sp.iloc[-4:].mean())
    streak = 0
    for v in reversed(sp.tolist()):
        if np.isfinite(v) and v > 0:
            streak += 1
        else:
            break
    return avg4, float(streak)


def _eps_estimate_trend(et: pd.DataFrame | None) -> float:
    """Current-quarter EPS estimate now minus 30 days ago (absolute revision)."""
    if et is None or not hasattr(et, "empty") or et.empty:
        return _NAN
    if "current" not in et.columns or "30daysAgo" not in et.columns:
        return _NAN
    row = None
    for label in ("0q", "+0q", "0Q"):
        if label in et.index:
            row = et.loc[label]
            break
    if row is None:
        row = et.iloc[0]  # fall back to the first (nearest) period
    cur, ago = _num(row.get("current")), _num(row.get("30daysAgo"))
    if np.isfinite(cur) and np.isfinite(ago):
        return cur - ago
    return _NAN


def _ttm_net_margin(ttm: pd.DataFrame | None) -> float:
    """TTM net income / TTM total revenue from the ttm_income_stmt frame."""
    if ttm is None or not hasattr(ttm, "empty") or ttm.empty:
        return _NAN
    col = "TTM" if "TTM" in ttm.columns else ttm.columns[0]

    def _row(*names):
        for n in names:
            if n in ttm.index:
                return _num(ttm.loc[n, col])
        return _NAN

    ni = _row("Net Income", "NetIncome", "Net Income Common Stockholders")
    rev = _row("Total Revenue", "TotalRevenue", "Operating Revenue")
    if np.isfinite(ni) and np.isfinite(rev) and rev > 0:
        return ni / rev
    return _NAN


def _pct_buy(rec: pd.DataFrame | None) -> float:
    """Fraction of analysts at Buy/Strong-Buy (a.k.a. overweight) from the latest
    recommendations-summary row."""
    if rec is None or not hasattr(rec, "empty") or rec.empty:
        return _NAN
    row = None
    if "period" in rec.columns and (rec["period"] == "0m").any():
        row = rec[rec["period"] == "0m"].iloc[0]
    else:
        row = rec.iloc[0]
    sb, b = _num(row.get("strongBuy")), _num(row.get("buy"))
    h, s, ss = _num(row.get("hold")), _num(row.get("sell")), _num(row.get("strongSell"))
    total = np.nansum([sb, b, h, s, ss])
    buy = np.nansum([sb, b])
    if total > 0:
        return float(buy / total)
    return _NAN


def _grade_changes(ud: pd.DataFrame | None, asof: date_cls, days: int = 90) -> tuple[float, float, float]:
    """(#upgrades, #downgrades, #maintained) in the trailing `days` window. NaN
    triple when no frame is available; zeros are a real 'none in window' reading."""
    if ud is None or not hasattr(ud, "empty") or ud.empty or "Action" not in ud.columns:
        return _NAN, _NAN, _NAN
    df = ud.copy()
    idx = pd.to_datetime(df.index, errors="coerce")
    try:
        idx = idx.tz_localize(None)
    except (TypeError, AttributeError):
        idx = idx.tz_convert(None) if getattr(idx, "tz", None) is not None else idx
    df = df.assign(_gd=idx)
    cutoff = pd.Timestamp(asof) - pd.Timedelta(days=days)
    recent = df[df["_gd"] >= cutoff]
    actions = recent["Action"].astype(str).str.lower()
    up = int((actions == "up").sum())
    down = int((actions == "down").sum())
    steady = int(actions.isin(["main", "reit"]).sum())
    return float(up), float(down), float(steady)


def compute_fundamentals(
    raw: dict | None,
    ticker_config: dict,
    asof: date_cls,
    symbol: str = "",
) -> dict:
    """Extract the fundamental scalar features from a raw yfinance bundle. Returns
    all-NaN (plus empty commentary) for index ETFs or when `raw` is None."""
    out = {k: _NAN for k in (ANALYST_KEYS + FINANCIAL_KEYS)}
    out["analyst_commentary"] = ""

    if ticker_config.get("kind") == "index_etf" or raw is None:
        return out

    info = raw.get("info") or {}
    out["forward_pe"] = _num(info.get("forwardPE"))
    out["debt_to_equity"] = _num(info.get("debtToEquity"))
    out["revenue_growth_yoy"] = _num(info.get("revenueGrowth"))
    out["analyst_consensus"] = _num(info.get("recommendationMean"))
    out["analyst_coverage_count"] = _num(info.get("numberOfAnalystOpinions"))

    out["eps_surprise_avg_4q"], out["eps_beat_streak"] = _earnings_metrics(raw.get("earnings_history"))
    out["eps_estimate_trend"] = _eps_estimate_trend(raw.get("eps_trend"))
    out["net_income_margin_ttm"] = _ttm_net_margin(raw.get("ttm_income_stmt"))
    out["analyst_pct_buy"] = _pct_buy(raw.get("recommendations"))
    up, down, steady = _grade_changes(raw.get("upgrades_downgrades"), asof, 90)
    out["analyst_upgrades_3m"], out["analyst_downgrades_3m"], out["analyst_steady_3m"] = up, down, steady

    # Analyst commentary
    out["analyst_commentary"] = _build_analyst_commentary(out, symbol, asof)
    return out


# ---- Orchestration ----------------------------------------------------------

def compute_stage5_5(
    history: pd.DataFrame | None,
    asof: date_cls,
    raw_bundle: dict | None,
    ticker_config: dict,
    symbol: str = "",
    ivx_30d: float = _NAN,
    prev_tech_rec: str | None = None,
) -> dict:
    """Full Stage 5.5 feature slice: technical scalars + fundamental scalars +
    commentary strings + hysteresis-gated recommendation.

    `prev_tech_rec` is yesterday's tech_recommendation (str) — pass None on
    first run; the function reads it from storage if not supplied."""
    out = {}
    tech = compute_technicals(history, asof, symbol=symbol, ivx_30d=ivx_30d)
    out.update(tech)
    out.update(compute_fundamentals(raw_bundle, ticker_config, asof, symbol=symbol))

    # Extract raw score and apply hysteresis
    commentary = tech.get("technical_commentary", "")
    tech_score = _NAN
    raw_rec = "HOLD"
    if commentary:
        # Pull score from the already-computed scoring inside the commentary build.
        # Re-compute to extract the structured scoring dict cleanly.
        try:
            if history is not None and not history.empty:
                h = history[history.index <= pd.Timestamp(asof)]
                if not h.empty and len(h) >= 26:
                    close = h["Close"].astype(float)
                    high_s = h["High"].astype(float)
                    low_s  = h["Low"].astype(float)
                    vol_s  = h["Volume"].astype(float)
                    macd_l = _ema(close, 12) - _ema(close, 26)
                    sig_l  = _ema(macd_l, 9)
                    mv = float(macd_l.iloc[-1])
                    sv = float(sig_l.iloc[-1])
                    hv = float((macd_l - sig_l).iloc[-1])
                    rv = _rsi(close, 14)
                    cv = _cmf(high_s, low_s, close, vol_s, 20)
                    close_cur = float(close.iloc[-1])
                    adx_v, _, _ = _adx(high_s, low_s, close)
                    obv_c, obv_s, obv_sl = _obv_sma(close, vol_s)
                    rv20 = _rv_20(close)
                    avwap_h, avwap_l, dist_h, dist_l, _, _ = _anchored_vwap(close, high_s, low_s, vol_s)
                    iv_pct = _iv_percentile(symbol, asof, ivx_30d) if symbol else _NAN
                    _, _, _ = _iv_context(ivx_30d, rv20, iv_pct)
                    sma20_s = close.rolling(20, min_periods=20).mean()
                    atr14_s = _atr14_series(high_s, low_s, close)
                    t_price = _trend_price(close, atr14_s)
                    ml_prev = float(macd_l.iloc[-2]) if len(macd_l) >= 2 else None
                    sl_prev = float(sig_l.iloc[-2]) if len(sig_l) >= 2 else None
                    t_macd, macd_az = _trend_macd(mv, sv, hv, macd_prev=ml_prev, signal_prev=sl_prev)
                    t_rsi = _trend_rsi(rv, close, 14)
                    t_bb, bb_upper, bb_mid, bb_lower, _ = _trend_bb(close)
                    t_cmf = _trend_cmf(cv)
                    t_vol, vcu, vcd = _trend_volume(vol_s, close)
                    t_obv = _trend_obv(obv_c, obv_s, obv_sl)
                    prev_c = float(close.iloc[-2]) if len(close) >= 2 else None
                    t_avwap = _trend_avwap(close_cur, avwap_h, avwap_l, dist_h, dist_l, prev_c)
                    adx_reg = _adx_regime(adx_v)
                    vol_reg = _vol_regime(iv_pct)
                    div_res = _run_divergence_suite(close, high_s, low_s, vol_s, sma20_s)
                    mh_s = macd_l - sig_l
                    sc = _score_technical(
                        t_price, t_macd, macd_az, t_rsi, t_bb, t_cmf,
                        vcu, vcd, t_obv, t_avwap,
                        rv, close_cur, bb_upper, bb_lower, bb_mid,
                        close, mh_s, adx_reg, vol_reg, div_res,
                    )
                    tech_score = sc["score"]
                    raw_rec = sc["rec"]
        except Exception:
            pass

    # Apply hysteresis
    tech_recommendation = _apply_hysteresis(tech_score, prev_tech_rec)

    out["tech_score"] = tech_score
    out["tech_recommendation"] = tech_recommendation
    return out


def split_for_cache(features: dict) -> tuple[dict, dict]:
    """Split a feature row into the (analyst, financials) payloads tracked by the
    storage evolution cache. Each is a flat dict of just that bucket's scalars."""
    analyst = {k: _num(features.get(k)) for k in ANALYST_KEYS}
    financials = {k: _num(features.get(k)) for k in FINANCIAL_KEYS}
    return analyst, financials