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

def _trend_price(close: pd.Series) -> str:
    """UP / DOWN / NEUTRAL based on SMA stacking (needs 200 bars)."""
    try:
        if len(close) < 200:
            return "NEUTRAL"
        sma20 = float(close.iloc[-20:].mean())
        sma50 = float(close.iloc[-50:].mean())
        sma200 = float(close.iloc[-200:].mean())
        cur = float(close.iloc[-1])
        if cur > sma20 > sma50 > sma200:
            return "UP"
        if cur < sma20 < sma50 < sma200:
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


def _trend_macd(macd: float, signal: float, hist: float) -> str:
    """BULLISH / BEARISH / NEUTRAL."""
    try:
        if not (np.isfinite(macd) and np.isfinite(signal) and np.isfinite(hist)):
            return "NEUTRAL"
        if macd > signal and hist > 0:
            return "BULLISH"
        if macd < signal and hist < 0:
            return "BEARISH"
        return "NEUTRAL"
    except Exception:
        return "NEUTRAL"


def _trend_bb(close_cur: float, bb_upper: float, bb_lower: float, bb_mid: float) -> str:
    """UPPER_BREAK / LOWER_BREAK / ABOVE_MID / BELOW_MID / NEUTRAL."""
    try:
        if not all(np.isfinite(v) for v in [close_cur, bb_upper, bb_lower, bb_mid]):
            return "NEUTRAL"
        if close_cur > bb_upper:
            return "UPPER_BREAK"
        if close_cur < bb_lower:
            return "LOWER_BREAK"
        if close_cur > bb_mid:
            return "ABOVE_MID"
        if close_cur < bb_mid:
            return "BELOW_MID"
        return "NEUTRAL"
    except Exception:
        return "NEUTRAL"


def _trend_cmf(cmf: float) -> str:
    """ACCUM (>0.05) / DISTRIB (<-0.05) / NEUTRAL."""
    try:
        if not np.isfinite(cmf):
            return "NEUTRAL"
        if cmf > 0.05:
            return "ACCUM"
        if cmf < -0.05:
            return "DISTRIB"
        return "NEUTRAL"
    except Exception:
        return "NEUTRAL"


def _trend_volume(vol_z: float) -> str:
    """HIGH (z>=2) / LOW (z<=-1) / NORMAL."""
    try:
        if not np.isfinite(vol_z):
            return "NORMAL"
        if vol_z >= 2:
            return "HIGH"
        if vol_z <= -1:
            return "LOW"
        return "NORMAL"
    except Exception:
        return "NORMAL"


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
    """TRENDING (>=25) / WEAK_TREND (>=15) / RANGING."""
    if not np.isfinite(adx):
        return "RANGING"
    if adx >= 25:
        return "TRENDING"
    if adx >= 15:
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


def _detect_divergence_active(
    close: pd.Series,
    indicator: pd.Series,
    div_type: str,   # "reg_bull" | "reg_bear"
    sma20: pd.Series,
    expiry_bars: int = 20,
    effect_bars: int = 15,
    pivot_left: int = 5,
    pivot_right: int = 5,
    quality_fn=None,
) -> list[dict]:
    """Returns list of currently-active divergence events."""
    try:
        close_arr = close.values
        ind_arr = indicator.values
        sma20_arr = sma20.values
        n = len(close_arr)
        if n < pivot_left + pivot_right + 2:
            return []

        # Find pivot indices for the relevant type
        if div_type == "reg_bull":
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

            # Divergence condition
            if div_type == "reg_bull":
                price_div = c2 < c1          # lower low in price
                ind_div = ind2 > ind1         # higher low in indicator
            else:
                price_div = c2 > c1          # higher high in price
                ind_div = ind2 < ind1         # lower high in indicator

            if not (price_div and ind_div):
                continue

            # Quality check
            if quality_fn is not None:
                quality = quality_fn(ind2)
            else:
                quality = "standard"

            # State machine from p2 forward
            state = "pending"
            confirmed_idx = None
            bars_since_p2 = n - 1 - p2

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
    """Run divergence detection for RSI, macd_hist, cmf, obv. Returns {name: [events]}."""
    try:
        rsi = _rsi_series(close)
        macd_h = _macd_hist_series(close)
        cmf = _cmf_series(high, low, close, volume)
        obv = _obv_series(close, volume)

        def rsi_quality(val):
            if np.isfinite(val) and val < 30:
                return "strong"
            return "standard"

        results = {}
        for name, indicator, div_type, q_fn in [
            ("RSI_bull", rsi, "reg_bull", rsi_quality),
            ("RSI_bear", rsi, "reg_bear", None),
            ("MACD_bull", macd_h, "reg_bull", None),
            ("MACD_bear", macd_h, "reg_bear", None),
            ("CMF_bull", cmf, "reg_bull", None),
            ("CMF_bear", cmf, "reg_bear", None),
            ("OBV_bull", obv, "reg_bull", None),
            ("OBV_bear", obv, "reg_bear", None),
        ]:
            results[name] = _detect_divergence_active(
                close, indicator, div_type, sma20, quality_fn=q_fn
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
    trend_rsi: str,
    trend_bb: str,
    trend_cmf: str,
    trend_volume: str,
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
        # --- Trend score ---
        trend_score = 0.0

        if trend_price == "UP":
            trend_score += 1.5
        elif trend_price == "DOWN":
            trend_score -= 1.5

        if trend_macd == "BULLISH":
            trend_score += 2.0
        elif trend_macd == "BEARISH":
            trend_score -= 2.0

        if trend_cmf == "ACCUM":
            trend_score += 0.5
        elif trend_cmf == "DISTRIB":
            trend_score -= 0.5

        if trend_obv == "BULLISH":
            trend_score += 1.0
        elif trend_obv == "BEARISH":
            trend_score -= 1.0

        avwap_contrib = {
            "RECLAIM_HIGH": 1.5, "BREAKDOWN_LOW": -1.5,
            "ABOVE_BOTH": 0.75, "BELOW_BOTH": -0.75,
            "TESTING_SUPPORT": 0.5, "TESTING_RESISTANCE": -0.5,
        }.get(trend_avwap, 0.0)
        trend_score += avwap_contrib

        # Volume: confirming only
        if trend_volume == "HIGH":
            if trend_score > 0:
                trend_score += 0.25
            elif trend_score < 0:
                trend_score -= 0.25

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

        # --- Mean reversion score ---
        mr_score = 0.0
        rsi_label = "neutral"
        bb_label = "neutral"

        if np.isfinite(rsi_val):
            if rsi_val < 30:
                mr_score += 1.5
                rsi_label = "oversold"
            elif rsi_val < 40:
                mr_score += 0.75
                rsi_label = "near-oversold"
            elif rsi_val > 70:
                mr_score -= 1.5
                rsi_label = "overbought"
            elif rsi_val > 60:
                mr_score -= 0.75
                rsi_label = "near-overbought"

        if all(np.isfinite(v) for v in [close_cur, bb_upper, bb_lower]):
            if close_cur > bb_upper:
                mr_score -= 1.0
                bb_label = "upper-break"
            elif close_cur < bb_lower:
                mr_score += 1.0
                bb_label = "lower-break"

        mr_weight = _REGIME_MATRIX.get(adx_regime, _REGIME_MATRIX["RANGING"]).get(
            vol_regime, 0.0
        )
        mr_contribution = mr_weight * mr_score

        # --- Divergence score ---
        div_score = 0.0
        bull_confirmed = 0
        bear_confirmed = 0

        for name, events in div_results.items():
            for ev in events:
                if ev["state"] == "confirmed":
                    if ev["type"] == "reg_bull":
                        div_score += 2.0
                        bull_confirmed += 1
                    else:
                        div_score -= 2.0
                        bear_confirmed += 1

        if bull_confirmed >= 2:
            div_score += 1.0
        if bear_confirmed >= 2:
            div_score -= 1.0

        # --- Final score ---
        raw = base_trend + mr_contribution + div_score
        score = max(-10.0, min(10.0, raw))

        if score >= 7:
            rec = "STRONG_BUY"
        elif score >= 4:
            rec = "BUY"
        elif score <= -7:
            rec = "STRONG_SELL"
        elif score <= -4:
            rec = "SELL"
        else:
            rec = "HOLD"

        # Confidence: scale from 0 to 1 based on abs(score)/10
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
        bb_upper, bb_mid, bb_lower, bb_width = _bollinger(close, 20, 2.0)
        close_cur = float(close.iloc[-1])

        # Volume z-score (20-bar)
        if len(volume) >= 20:
            vol_mean = float(volume.iloc[-20:].mean())
            vol_std = float(volume.iloc[-20:].std(ddof=1))
            vol_z = (float(volume.iloc[-1]) - vol_mean) / vol_std if vol_std > 0 else 0.0
        else:
            vol_z = _NAN

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

        # Trend classifications
        t_price = _trend_price(close)
        t_macd = _trend_macd(macd_val, signal_val, hist_val)
        t_rsi = _trend_rsi(rsi_val, close, 14)
        t_bb = _trend_bb(close_cur, bb_upper, bb_lower, bb_mid)
        t_cmf = _trend_cmf(cmf_val)
        t_vol = _trend_volume(vol_z)
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
            t_price, t_macd, t_rsi, t_bb, t_cmf, t_vol, t_obv, t_avwap,
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
        div_lines_text = []
        for name, events in div_results.items():
            for ev in events:
                indicator_label = name.replace("_bull", "").replace("_bear", "")
                icon = "✓ CONFIRMED" if ev["state"] == "confirmed" else "⏳ PENDING"
                div_lines_text.append(
                    f"- {icon} {ev['type']} on {indicator_label} (quality: {ev['quality']})"
                )
        if div_lines_text:
            lines.extend(div_lines_text)
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
) -> dict:
    """Full Stage 5.5 feature slice: technical scalars + fundamental scalars +
    commentary strings. Pure with respect to the network."""
    out = {}
    out.update(compute_technicals(history, asof, symbol=symbol, ivx_30d=ivx_30d))
    out.update(compute_fundamentals(raw_bundle, ticker_config, asof, symbol=symbol))
    return out


def split_for_cache(features: dict) -> tuple[dict, dict]:
    """Split a feature row into the (analyst, financials) payloads tracked by the
    storage evolution cache. Each is a flat dict of just that bucket's scalars."""
    analyst = {k: _num(features.get(k)) for k in ANALYST_KEYS}
    financials = {k: _num(features.get(k)) for k in FINANCIAL_KEYS}
    return analyst, financials
