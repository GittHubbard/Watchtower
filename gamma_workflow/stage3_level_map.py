"""Stage 3 — Key level map.

For each ticker, given a fresh options chain and a GEX profile over a price grid,
produce the complete level map the playbooks consume:

  Primary walls
    call_wall, call_wall_gex      argmax(GEX) above the gamma flip
    put_wall,  put_wall_gex       argmin(GEX) below the gamma flip

  Secondary walls (>= 2% of spot away from the primary, same side of flip)
    call_wall_secondary
    put_wall_secondary

  High-volume level (HVL)
    The strike with the largest combined call+put volume on the most recent
    session. NOT to be confused with a gamma wall — it's a magnetic level
    based on where activity actually happened today.

  Migration vs yesterday
    call_wall_dod_change, put_wall_dod_change   today's level minus yesterday's
    Directional rules from the workflow:
      call_wall up   = bullish structural shift
      put_wall  up   = support rising, bullish
      put_wall  down = support eroding, bearish

  Persistence
    wall_persistence_days_call/put  number of consecutive prior sessions where
    the call/put wall has been within wall_strike_tolerance_pct (default 0.5%
    of spot) of today's level. >= 3 sessions ⇒ high-conviction magnet.

The 2% separation rule for secondary walls is enforced via a price-grid mask:
mark a window of +/- 2% of spot around the primary's grid index as "used", then
take the argmax/argmin of the remaining same-side region.
"""
from __future__ import annotations
from datetime import date as date_cls, timedelta
import logging

import numpy as np
import pandas as pd

from .config import load_thresholds
from .storage import read_features_range

logger = logging.getLogger(__name__)


# --- Primary + secondary walls from a GEX profile -----------------------------

def _wall_indices(
    prices: np.ndarray,
    gex_array: np.ndarray,
    flip_level: float | None,
    spot: float,
    min_separation_pct: float,
) -> dict:
    """Find primary call/put wall grid indices, then the secondary wall on each
    side that is at least min_separation_pct * spot away from the primary.

    Returns dict with keys: call_idx, put_idx, call_sec_idx, put_sec_idx (any
    may be None if no qualifying point exists on that side).
    """
    n = len(prices)
    out = {"call_idx": None, "put_idx": None, "call_sec_idx": None, "put_sec_idx": None}
    if n == 0:
        return out

    # Split by gamma flip when available; otherwise by GEX sign at each grid pt.
    if flip_level is not None and np.isfinite(flip_level):
        above_mask = prices > flip_level
        below_mask = prices < flip_level
    else:
        above_mask = gex_array > 0
        below_mask = gex_array < 0

    # Primary call wall = argmax(GEX) above flip
    if above_mask.any():
        rel_idx = int(np.argmax(gex_array[above_mask]))
        call_idx = int(np.where(above_mask)[0][rel_idx])
        out["call_idx"] = call_idx

        # Secondary: exclude a +/- min_separation_pct * spot window around call_idx
        sep = min_separation_pct * spot
        sec_mask = above_mask & (np.abs(prices - prices[call_idx]) >= sep)
        if sec_mask.any():
            rel = int(np.argmax(gex_array[sec_mask]))
            out["call_sec_idx"] = int(np.where(sec_mask)[0][rel])

    # Primary put wall = argmin(GEX) below flip
    if below_mask.any():
        rel_idx = int(np.argmin(gex_array[below_mask]))
        put_idx = int(np.where(below_mask)[0][rel_idx])
        out["put_idx"] = put_idx

        sep = min_separation_pct * spot
        sec_mask = below_mask & (np.abs(prices - prices[put_idx]) >= sep)
        if sec_mask.any():
            rel = int(np.argmin(gex_array[sec_mask]))
            out["put_sec_idx"] = int(np.where(sec_mask)[0][rel])

    return out


def compute_levels(
    prices: np.ndarray,
    gex_array: np.ndarray,
    spot: float,
    flip_level: float | None,
    options_chain: pd.DataFrame | None = None,
    min_separation_pct: float = 0.02,
) -> dict:
    """Returns a dict of all current-day levels (no history).

    `options_chain` is the dataframe built by engine_shim.build_refined_options_chain
    — used only for HVL. If None, hvl is NaN.

    Keys returned:
        call_wall, call_wall_gex, call_wall_secondary
        put_wall,  put_wall_gex,  put_wall_secondary
        hvl                                  : float | NaN
    """
    idx = _wall_indices(prices, gex_array, flip_level, spot, min_separation_pct)

    def _price(i):  return float(prices[i]) if i is not None else float("nan")
    def _gex(i):    return float(gex_array[i]) if i is not None else float("nan")

    result = {
        "call_wall": _price(idx["call_idx"]),
        "call_wall_gex": _gex(idx["call_idx"]),
        "call_wall_secondary": _price(idx["call_sec_idx"]),
        "put_wall": _price(idx["put_idx"]),
        "put_wall_gex": _gex(idx["put_idx"]),
        "put_wall_secondary": _price(idx["put_sec_idx"]),
        "hvl": float("nan"),
    }

    # HVL: argmax of total volume by strike (calls + puts combined)
    if options_chain is not None and not options_chain.empty and "volume" in options_chain.columns:
        df = options_chain.dropna(subset=["strike", "volume"])
        if not df.empty:
            by_strike = df.groupby("strike")["volume"].sum()
            if by_strike.sum() > 0:
                result["hvl"] = float(by_strike.idxmax())

    return result


# --- Migration deltas and persistence -----------------------------------------

def compute_migration(
    ticker: str,
    asof: date_cls,
    call_wall: float,
    put_wall: float,
    spot: float,
    tolerance_pct: float | None = None,
    persistence_window_days: int = 10,
) -> dict:
    """Compare today's walls to yesterday's, and count how many consecutive
    prior sessions the wall has been within tolerance of today's level.

    Returns:
        call_wall_dod_change, put_wall_dod_change   floats (NaN if no prior day)
        wall_persistence_days_call/put              ints (0 if no persistence)
    """
    if tolerance_pct is None:
        tolerance_pct = float(load_thresholds()["levels"]["wall_strike_tolerance_pct"])

    start = asof - timedelta(days=persistence_window_days + 10)
    hist = read_features_range(start, asof - timedelta(days=1))
    if hist.empty:
        return {
            "call_wall_dod_change": float("nan"),
            "put_wall_dod_change": float("nan"),
            "wall_persistence_days_call": 0,
            "wall_persistence_days_put": 0,
        }
    hist = hist[hist["ticker"] == ticker].sort_values("date")
    if hist.empty:
        return {
            "call_wall_dod_change": float("nan"),
            "put_wall_dod_change": float("nan"),
            "wall_persistence_days_call": 0,
            "wall_persistence_days_put": 0,
        }

    last = hist.iloc[-1]
    dod_call = call_wall - float(last["call_wall"]) if pd.notna(last["call_wall"]) else float("nan")
    dod_put = put_wall - float(last["put_wall"]) if pd.notna(last["put_wall"]) else float("nan")

    # Persistence: walk backward through history and count consecutive days where
    # |prior_wall - today_wall| / spot <= tolerance_pct. Stops at first miss or NaN.
    tol_dollars = tolerance_pct * spot

    def _persistence(series: pd.Series, target: float) -> int:
        count = 0
        for v in series.iloc[::-1]:
            if pd.isna(v):
                break
            if abs(float(v) - target) <= tol_dollars:
                count += 1
            else:
                break
        return count

    call_persist = _persistence(hist["call_wall"], call_wall)
    put_persist = _persistence(hist["put_wall"], put_wall)

    return {
        "call_wall_dod_change": dod_call,
        "put_wall_dod_change": dod_put,
        "wall_persistence_days_call": call_persist,
        "wall_persistence_days_put": put_persist,
    }


# --- One-shot orchestrator -----------------------------------------------------

def build_level_map(
    ticker: str,
    asof: date_cls,
    prices: np.ndarray,
    gex_array: np.ndarray,
    spot: float,
    flip_level: float | None,
    options_chain: pd.DataFrame | None = None,
    thresholds: dict | None = None,
) -> dict:
    """End-to-end Stage 3 output. Merges current levels with migration/persistence."""
    thresholds = thresholds or load_thresholds()
    min_sep = 0.02  # per PM spec, secondary walls require >= 2% separation
    levels = compute_levels(
        prices, gex_array, spot, flip_level,
        options_chain=options_chain, min_separation_pct=min_sep,
    )
    migration = compute_migration(
        ticker, asof, levels["call_wall"], levels["put_wall"], spot,
        tolerance_pct=thresholds["levels"]["wall_strike_tolerance_pct"],
    )
    return {**levels, "flip_level": flip_level, **migration}
