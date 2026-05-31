"""Stage 2 — Regime classification.

The single most important module in the workflow. Everything downstream (which
playbook is active, what stops we use, how aggressively we size) hangs off the
regime label. This module also tracks transitions — most of our edge lives in
regime *changes*, not steady-state regimes.

The four regimes:
    deep_positive       — spot well above flip, GEX above ticker's 80th pct
                          (or > 1% of market cap fallback). Dealers heavily long
                          gamma. Mean-reversion / fade extremes works.
    transitional_positive — spot above flip but GEX falling toward the median.
                            Vol about to expand.
    transitional_negative — spot below flip, GEX building negative. Be careful,
                            momentum regime emerging.
    deep_negative       — spot below flip, GEX below 20th pct (or < -1% of mcap).
                          Dealer hedging amplifies moves. Best long-entry zone
                          on washouts, with Stage 5 confirmation.

Threshold selection:
    If the ticker has >= 60 days of historical GEX in the features store, we use
    the empirical 20th/80th percentile of that ticker's own GEX distribution. This
    is the right way — different tickers have wildly different GEX magnitudes.
    If history is insufficient, we fall back to a market-cap-based absolute
    threshold: |GEX| > 1% * market_cap_$ ⇒ deep. The 1% is calibratable.

Transition tracking:
    classify_with_history reads yesterday's regime from the features store and
    returns both today's regime and a "previous→current" string when they differ.

Public API:
    classify(gex_total, spot, flip_level, ticker, history_df=None, mcap_b=None,
             thresholds=None) -> dict
    classify_with_history(...) -> dict augmented with regime_transition
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date as date_cls, timedelta
import logging

import numpy as np
import pandas as pd

from .config import load_thresholds
from .storage import read_features_range

logger = logging.getLogger(__name__)


REGIME_DEEP_POSITIVE = "deep_positive"
REGIME_TRANSITIONAL_POSITIVE = "transitional_positive"
REGIME_TRANSITIONAL_NEGATIVE = "transitional_negative"
REGIME_DEEP_NEGATIVE = "deep_negative"


@dataclass
class RegimeContext:
    """What was used to pick a regime — captured for logging/audit."""
    method: str          # 'percentile' or 'market_cap'
    upper_threshold: float
    lower_threshold: float
    history_days_used: int


def _resolve_thresholds(
    ticker: str,
    history_df: pd.DataFrame | None,
    mcap_b: float | None,
    thresholds: dict,
) -> tuple[float, float, RegimeContext]:
    """Return (upper, lower, context) where upper = "deep positive" boundary
    and lower = "deep negative" boundary, both in raw GEX dollars."""
    reg_cfg = thresholds["regime"]
    min_days = int(reg_cfg["history_days_required"])
    p_high = float(reg_cfg["percentile_deep_positive"])
    p_low = float(reg_cfg["percentile_deep_negative"])

    n_days = 0
    if history_df is not None and not history_df.empty:
        # gex_total column may be NaN on early days — drop them
        gex_series = history_df.loc[history_df["ticker"] == ticker, "gex_total"].dropna()
        n_days = len(gex_series.unique())  # one row per day, but be defensive
        if n_days >= min_days:
            upper = float(np.percentile(gex_series, p_high))
            lower = float(np.percentile(gex_series, p_low))
            return upper, lower, RegimeContext("percentile", upper, lower, n_days)

    # Fallback: market-cap-based absolute threshold
    pct = float(reg_cfg["fallback_pct_of_market_cap"])
    if mcap_b is None or mcap_b <= 0:
        # Last-resort fallback to a small fixed dollar value so we still produce
        # a regime label rather than crashing. PM should set market_cap_b in YAML.
        logger.warning(
            "No market_cap_b for %s and insufficient history (%d days). "
            "Using a $1B absolute threshold.", ticker, n_days,
        )
        mcap_b = 100.0
    abs_thresh = pct * mcap_b * 1e9  # convert $B to $
    return abs_thresh, -abs_thresh, RegimeContext("market_cap", abs_thresh, -abs_thresh, n_days)


def classify(
    gex_total: float,
    spot: float,
    flip_level: float | None,
    ticker: str,
    history_df: pd.DataFrame | None = None,
    mcap_b: float | None = None,
    thresholds: dict | None = None,
) -> dict:
    """Classify a single point-in-time observation.

    Returns:
        {
            "regime": one of the four constants,
            "distance_to_flip_pct": (spot - flip) / spot, or NaN if flip unknown,
            "threshold_method": 'percentile' | 'market_cap',
            "threshold_upper": float,
            "threshold_lower": float,
            "history_days_used": int,
        }
    """
    thresholds = thresholds or load_thresholds()
    upper, lower, ctx = _resolve_thresholds(ticker, history_df, mcap_b, thresholds)

    # Distance to flip in fractional terms
    if flip_level is None or not np.isfinite(flip_level) or spot <= 0:
        dist_pct = float("nan")
        above_flip = gex_total > 0  # fall back to GEX sign when flip is unknown
    else:
        dist_pct = (spot - flip_level) / spot
        above_flip = spot > flip_level

    if above_flip:
        regime = REGIME_DEEP_POSITIVE if gex_total >= upper else REGIME_TRANSITIONAL_POSITIVE
    else:
        regime = REGIME_DEEP_NEGATIVE if gex_total <= lower else REGIME_TRANSITIONAL_NEGATIVE

    return {
        "regime": regime,
        "distance_to_flip_pct": dist_pct,
        "threshold_method": ctx.method,
        "threshold_upper": ctx.upper_threshold,
        "threshold_lower": ctx.lower_threshold,
        "history_days_used": ctx.history_days_used,
    }


def classify_with_history(
    gex_total: float,
    spot: float,
    flip_level: float | None,
    ticker: str,
    asof: date_cls,
    mcap_b: float | None = None,
    thresholds: dict | None = None,
    history_window_days: int = 90,
) -> dict:
    """Same as classify(), but also pulls the ticker's own history from the
    features store to (a) compute percentile thresholds and (b) detect a
    regime transition vs yesterday.

    Adds keys:
        regime_transition : "prev_regime->current_regime" if changed, else None
        prev_regime       : yesterday's regime string, or None
    """
    # Load enough history for percentile calc plus yesterday's row.
    start = asof - timedelta(days=history_window_days + 30)
    hist = read_features_range(start, asof - timedelta(days=1))
    hist_ticker = hist[hist["ticker"] == ticker] if not hist.empty else pd.DataFrame()

    result = classify(
        gex_total=gex_total,
        spot=spot,
        flip_level=flip_level,
        ticker=ticker,
        history_df=hist_ticker[["ticker", "gex_total"]] if not hist_ticker.empty else None,
        mcap_b=mcap_b,
        thresholds=thresholds,
    )

    # Detect yesterday's regime — use the most recent row strictly before asof.
    prev_regime = None
    if not hist_ticker.empty:
        # 'date' column was added by read_features_range
        hist_sorted = hist_ticker.sort_values("date")
        last_row = hist_sorted.iloc[-1]
        if pd.notna(last_row.get("regime")):
            prev_regime = str(last_row["regime"])

    transition = None
    if prev_regime and prev_regime != result["regime"]:
        transition = f"{prev_regime}->{result['regime']}"

    result["prev_regime"] = prev_regime
    result["regime_transition"] = transition
    return result
