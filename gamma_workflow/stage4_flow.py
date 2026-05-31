"""Stage 4 — Flow & positioning analysis.

For each ticker, given the refined options chain (built by engine_shim) plus the
prior day's chain (passed in or pulled from the features store), produce all flow
features the playbooks consume:

    pcr_volume, pcr_oi                    today's put/call ratios
    pcr_volume_z, pcr_oi_z                20-day z-scores of those ratios
    total_volume                          aggregate option volume today
    total_volume_z                        20-day z-score of total volume
                                          ("unusual activity" flag = z >= 2.0)
    skew_25d                              25-delta put IV minus 25-delta call IV
                                          (the slope; both legs ~30 DTE)
    skew_25d_z                            20-day z-score of that slope
    term_structure_slope                  3M IVX minus front-month IVX
                                          (positive => contango / normal)
    new_oi_strike_count                   # strikes today where volume > OI
                                          (new positioning being put on)
    oi_delta_call, oi_delta_put           net OI change calls/puts vs yesterday

The "large prints / sweeps" feature from the original workflow spec is INTENTIONALLY
omitted — per the PM (Q2), yfinance does not expose sweep flags or bid/ask side
classification. Add when a real feed (Polygon/Tradier/ORATS) is wired in.

Note on z-scores: when fewer than 20 days of history exist, z is NaN. Stage 6 must
treat NaN z-scores as "no signal" rather than "below threshold".

Public API:
    compute_flow(symbol, asof, chain, spot, r, prev_chain=None) -> dict
"""
from __future__ import annotations
from datetime import date as date_cls, timedelta
import logging

import numpy as np
import pandas as pd

from .storage import read_features_range

logger = logging.getLogger(__name__)

ZSCORE_WINDOW_DAYS = 20


def _engine():
    """Lazy accessor for the engine_shim module. Importing it at module load
    time would force every test run to require the engine file on disk; we
    only actually need it inside specific functions."""
    from . import engine_shim
    return engine_shim


# ---- Helpers ---------------------------------------------------------------

def _rolling_z(history_series: pd.Series, current: float) -> float:
    """20-day z-score of `current` against the prior history_series.

    Returns NaN if fewer than ZSCORE_WINDOW_DAYS observations or zero std."""
    s = history_series.dropna()
    if len(s) < ZSCORE_WINDOW_DAYS:
        return float("nan")
    window = s.iloc[-ZSCORE_WINDOW_DAYS:]
    std = float(window.std())
    if std == 0 or not np.isfinite(std):
        return float("nan")
    return (current - float(window.mean())) / std


def _pcr(chain: pd.DataFrame, col: str) -> float:
    """Total put `col` divided by total call `col`. NaN if call side is empty/zero."""
    calls = chain.loc[chain["is_call"], col].sum()
    puts = chain.loc[~chain["is_call"], col].sum()
    if calls <= 0:
        return float("nan")
    return float(puts / calls)


def _nearest_delta_iv(chain_side: pd.DataFrame, target_delta: float, spot: float, r: float) -> float:
    """Return the IV of the option in `chain_side` whose computed delta is
    closest to `target_delta`. Filters to options roughly 30 DTE (15-45 day band)
    when available, otherwise uses whatever DTE is available."""
    if chain_side.empty:
        return float("nan")

    # Prefer the 15-45 DTE band; if empty, fall back to the full chain side.
    # T is in years on the chain (set by build_refined_options_chain).
    band = chain_side[(chain_side["T"] >= 15 / 365) & (chain_side["T"] <= 45 / 365)]
    sub = band if not band.empty else chain_side

    # Compute deltas on the fly
    gamma, delta, _vanna = _engine().black_scholes_metrics_vectorized(
        S=spot,
        K=sub["strike"].values,
        T=sub["T"].values,
        r=r,
        sigma=sub["iv"].values,
        is_call=sub["is_call"].values,
    )
    diffs = np.abs(delta - target_delta)
    if not np.isfinite(diffs).any():
        return float("nan")
    idx = int(np.nanargmin(diffs))
    iv = sub["iv"].iloc[idx]
    return float(iv) if pd.notna(iv) else float("nan")


def _skew_25d(chain: pd.DataFrame, spot: float, r: float) -> float:
    """25-delta put IV minus 25-delta call IV. Positive = put skew (fear bid).

    Convention: put deltas are negative, so target = -0.25 for puts.
    """
    calls = chain[chain["is_call"]]
    puts = chain[~chain["is_call"]]
    iv_put_25 = _nearest_delta_iv(puts, target_delta=-0.25, spot=spot, r=r)
    iv_call_25 = _nearest_delta_iv(calls, target_delta=+0.25, spot=spot, r=r)
    if not (np.isfinite(iv_put_25) and np.isfinite(iv_call_25)):
        return float("nan")
    return iv_put_25 - iv_call_25


def _term_structure_slope(ticker_obj, spot: float, long_target_days: int = 90) -> float:
    """3-month IVX minus 7-day IVX. Positive = contango (normal); negative = backwardation.

    Guards against the engine's IVX extrapolation: calculate_iv_index() reads
    ticker_obj.options directly and, when NO expiry reaches `long_target_days`,
    silently falls back to the two longest available expiries and extrapolates —
    returning a finite value that *looks* like a 90d read but isn't. That would feed
    Playbook C's contango gate a fabricated slope. We therefore require a genuine
    long-dated expiry (within a tolerance below the target) before trusting the long
    leg; otherwise return NaN, which Stage 6 treats as "no signal" (never as a pass).
    """
    try:
        from datetime import date as _date, datetime as _dt
        expiries = getattr(ticker_obj, "options", None) or []
        today = _date.today()
        max_dte = -1
        for e in expiries:
            try:
                dte = (_dt.strptime(e, "%Y-%m-%d").date() - today).days
            except (ValueError, TypeError):
                continue
            if dte > max_dte:
                max_dte = dte
        # Need a real expiry near/at the long target — allow a 10-day shortfall so a
        # ~83d longest expiry still counts, but a 50d one (pure extrapolation) does not.
        if max_dte < long_target_days - 10:
            return float("nan")

        ivx = _engine().calculate_iv_index
        ivx_7 = ivx(ticker_obj, spot, target_days=7, option_type="both")
        ivx_90 = ivx(ticker_obj, spot, target_days=long_target_days, option_type="both")
    except Exception as e:
        logger.warning("term structure calc failed: %s", e)
        return float("nan")
    if not (np.isfinite(ivx_7) and np.isfinite(ivx_90)):
        return float("nan")
    return float(ivx_90 - ivx_7)


def _new_oi_strike_count(chain: pd.DataFrame) -> int:
    """Number of strikes where today's volume exceeds today's OI — strong signal
    that new positioning is being put on at that strike (the OI hasn't settled
    yet, but flow today already eclipses the standing book)."""
    if "volume" not in chain.columns:
        return 0
    mask = chain["volume"].fillna(0) > chain["oi"].fillna(0)
    return int(mask.sum())


def _oi_deltas(chain: pd.DataFrame, prev_chain: pd.DataFrame | None) -> tuple[float, float]:
    """Net OI change vs yesterday, calls and puts. NaN if no prior chain."""
    if prev_chain is None or prev_chain.empty:
        return float("nan"), float("nan")

    def _net(today_side: pd.DataFrame, prev_side: pd.DataFrame) -> float:
        # Merge on (strike, expiry_str) and sum the differences.
        today_grp = today_side.groupby(["strike", "expiry_str"])["oi"].sum()
        prev_grp = prev_side.groupby(["strike", "expiry_str"])["oi"].sum()
        joined = today_grp.subtract(prev_grp, fill_value=0)
        return float(joined.sum())

    return (
        _net(chain[chain["is_call"]], prev_chain[prev_chain["is_call"]]),
        _net(chain[~chain["is_call"]], prev_chain[~prev_chain["is_call"]]),
    )


# ---- Public API ------------------------------------------------------------

def compute_flow(
    symbol: str,
    asof: date_cls,
    chain: pd.DataFrame,
    spot: float,
    r: float,
    ticker_obj=None,
    prev_chain: pd.DataFrame | None = None,
) -> dict:
    """End-to-end Stage 4. `chain` is the refined options chain from engine_shim.

    `ticker_obj` is a yfinance Ticker — needed only for the term-structure IVX
    calls. If None, term_structure_slope is NaN.

    z-scores are computed against the prior 30 days of features in storage; if
    fewer than 20 prior days are available, the relevant *_z fields are NaN.
    """
    # --- Today's point estimates -----------------------------------------------
    pcr_v = _pcr(chain, "volume") if "volume" in chain.columns else float("nan")
    pcr_o = _pcr(chain, "oi")
    total_vol = float(chain["volume"].sum()) if "volume" in chain.columns else float("nan")
    skew = _skew_25d(chain, spot, r)
    term_slope = _term_structure_slope(ticker_obj, spot) if ticker_obj is not None else float("nan")
    new_oi_count = _new_oi_strike_count(chain)
    oi_delta_c, oi_delta_p = _oi_deltas(chain, prev_chain)

    # --- z-scores vs prior 30 days of stored features --------------------------
    start = asof - timedelta(days=45)
    hist = read_features_range(start, asof - timedelta(days=1))
    if not hist.empty:
        hist = hist[hist["ticker"] == symbol].sort_values("date")
    else:
        hist = pd.DataFrame()

    def _z(col, current):
        if hist.empty or col not in hist.columns:
            return float("nan")
        return _rolling_z(hist[col], current)

    return {
        "pcr_volume": pcr_v,
        "pcr_volume_z": _z("pcr_volume", pcr_v),
        "pcr_oi": pcr_o,
        "pcr_oi_z": _z("pcr_oi", pcr_o),
        "total_volume": total_vol,
        "total_volume_z": _z("total_volume", total_vol),
        "skew_25d": skew,
        "skew_25d_z": _z("skew_25d", skew),
        "term_structure_slope": term_slope,
        "new_oi_strike_count": new_oi_count,
        "oi_delta_call": oi_delta_c,
        "oi_delta_put": oi_delta_p,
    }
