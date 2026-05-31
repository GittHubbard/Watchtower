"""Stage 6 — Entry trigger logic (the playbooks).

Three playbooks, each a function that reads its gates from thresholds.yaml
(option (a) per PM — functions over a declarative DSL; revisit if we exceed ~10
rule sets). Each evaluator takes the current feature row, the ticker's config,
a short slice of the ticker's recent feature history (for the multi-day gates),
and the thresholds dict. It returns a signal dict if ALL gates pass, else None.

Signal dict shape (consumed by Stage 7):
    {
        "playbook_id": "A" | "B" | "C",
        "playbook_name": str,
        "signal_strength": float in [0, 1],
        "key_levels": {"put_wall":.., "call_wall":.., "flip":.., "spot":..},
        "invalidation_level": float,        # where the thesis is wrong (initial stop)
        "suggested_size_pct": float,        # initial tranche size from the playbook
        "gates_passed": {gate_name: value, ...},   # for the ticket / audit
        "instrument": "shares" | "defined_risk_calls",
    }

signal_strength convention: for each continuous "z-score-like" gate where higher
is stronger, the component is clip(value / (2 * threshold), 0, 1) — so a metric
sitting exactly at its threshold contributes 0.5 and one at twice the threshold
contributes 1.0. The overall strength is the mean of the components. Boolean
gates (regime, event windows) are part of *firing*, not of strength.

The playbooks map directly to the workflow spec:
    A — Washout at Put Wall in Negative Gamma   (shares, 50%)
    B — Gamma Flip Reclaim                       (shares, 75%)
    C — Compression Breakout                     (defined-risk calls, 50%)

Public API:
    evaluate_all(features, ticker_config, history_df, thresholds) -> list[signal]
"""
from __future__ import annotations
import logging

import numpy as np
import pandas as pd

from .config import load_thresholds
from .stage2_regime import (
    REGIME_DEEP_NEGATIVE,
    REGIME_TRANSITIONAL_NEGATIVE,
    REGIME_TRANSITIONAL_POSITIVE,
    REGIME_DEEP_POSITIVE,
)

logger = logging.getLogger(__name__)

NEGATIVE_REGIMES = {REGIME_DEEP_NEGATIVE, REGIME_TRANSITIONAL_NEGATIVE}


# ---- helpers ---------------------------------------------------------------

def _f(features: dict, key: str) -> float:
    """Fetch a numeric feature as float, NaN if missing/None."""
    v = features.get(key)
    if v is None:
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _strength(components: list[float]) -> float:
    """Mean of per-gate strength components, each already in [0, 1]. NaN comps
    are dropped; if none remain, strength is 0.5 (fired but no continuous signal)."""
    vals = [c for c in components if np.isfinite(c)]
    if not vals:
        return 0.5
    return float(np.clip(np.mean(vals), 0.0, 1.0))


def _z_component(value: float, threshold: float) -> float:
    """clip(value / (2*threshold), 0, 1). At threshold → 0.5, at 2x → 1.0."""
    if not np.isfinite(value) or threshold == 0:
        return float("nan")
    return float(np.clip(value / (2.0 * threshold), 0.0, 1.0))


def _gex_ma_inflecting_up(history_df: pd.DataFrame, today_gex: float, window: int) -> bool:
    """True if the `window`-session GEX moving average is rising — i.e. the MA
    computed through today is greater than the MA computed through yesterday.

    Requires at least `window + 1` total observations (window prior days + today).
    Conservative: returns False when there isn't enough history.
    """
    if history_df is None or history_df.empty:
        return False
    gex_hist = history_df.sort_values("date")["gex_total"].dropna().tolist()
    series_with_today = gex_hist + [today_gex]
    if len(series_with_today) < window + 1:
        return False
    ma_today = np.mean(series_with_today[-window:])
    ma_prev = np.mean(series_with_today[-window - 1:-1])
    return bool(ma_today > ma_prev)


def _consecutive_regime_count(history_df: pd.DataFrame, today_regime: str, target: str) -> int:
    """Count consecutive sessions (ending today) where regime == target.
    Today counts as 1 if today_regime == target; then walks backward through
    history until the streak breaks."""
    if today_regime != target:
        return 0
    count = 1
    if history_df is None or history_df.empty:
        return count
    regimes = history_df.sort_values("date")["regime"].tolist()
    for r in reversed(regimes):
        if r == target:
            count += 1
        else:
            break
    return count


def _catalyst_within(features: dict, days: int) -> bool:
    """True if earnings or a macro event is within `days` sessions."""
    d_earn = features.get("days_to_earnings")
    d_macro = features.get("days_to_macro")
    horizons = [x for x in (d_earn, d_macro) if x is not None and np.isfinite(x)]
    return any(0 <= x <= days for x in horizons)


# ---- Playbook A — Washout at Put Wall in Negative Gamma --------------------

def evaluate_playbook_A(features: dict, ticker_config: dict, thresholds: dict) -> dict | None:
    cfg = thresholds["playbooks"]["A_washout"]
    flow = thresholds["flow"]
    confirm = thresholds["confirm"]

    regime = features.get("regime")
    spot = _f(features, "spot")
    put_wall = _f(features, "put_wall")
    pcr_z = _f(features, "pcr_volume_z")
    skew_z = _f(features, "skew_25d_z")
    vol_z = _f(features, "volume_z")
    vix_5d = _f(features, "vix_5d_change_pct")
    within_event = bool(features.get("within_event_window", False))

    # --- Gates ---
    if regime != REGIME_DEEP_NEGATIVE:
        return None
    # Proximity to put wall (within X% of spot)
    if not (np.isfinite(spot) and np.isfinite(put_wall) and spot > 0):
        return None
    proximity = abs(spot - put_wall) / spot
    if proximity > cfg["proximity_to_put_wall_pct"]:
        return None
    # Panic / fear / capitulation gates
    if not (np.isfinite(pcr_z) and pcr_z > flow["put_call_ratio_zscore_panic"]):
        return None
    if not (np.isfinite(skew_z) and skew_z > flow["put_skew_zscore_high"]):
        return None
    if not (np.isfinite(vol_z) and vol_z > confirm["volume_zscore_capitulation"]):
        return None
    # VIX confirmation — only enforced for index ETFs (per Stage 5 design note)
    if ticker_config.get("kind") == "index_etf":
        if not (np.isfinite(vix_5d) and vix_5d >= confirm["vix_5day_change_pct"]):
            return None
    # No binary event within the window
    if within_event:
        return None

    strength = _strength([
        _z_component(pcr_z, flow["put_call_ratio_zscore_panic"]),
        _z_component(skew_z, flow["put_skew_zscore_high"]),
        _z_component(vol_z, confirm["volume_zscore_capitulation"]),
    ])

    return {
        "playbook_id": "A",
        "playbook_name": "Washout at Put Wall in Negative Gamma",
        "signal_strength": strength,
        "key_levels": {
            "put_wall": put_wall,
            "call_wall": _f(features, "call_wall"),
            "flip": _f(features, "flip_level"),
            "spot": spot,
        },
        "invalidation_level": put_wall,   # thesis wrong if we lose the put wall
        "suggested_size_pct": cfg["initial_size_pct"],
        "instrument": "shares",
        "gates_passed": {
            "regime": regime,
            "proximity_to_put_wall_pct": proximity,
            "pcr_volume_z": pcr_z,
            "skew_25d_z": skew_z,
            "volume_z": vol_z,
            "vix_5d_change_pct": vix_5d,
        },
    }


# ---- Playbook B — Gamma Flip Reclaim ---------------------------------------

def evaluate_playbook_B(
    features: dict, ticker_config: dict, history_df: pd.DataFrame, thresholds: dict
) -> dict | None:
    cfg = thresholds["playbooks"]["B_gamma_reclaim"]

    regime = features.get("regime")
    prev_regime = features.get("prev_regime")
    vol_z = _f(features, "volume_z")
    call_wall_dod = _f(features, "call_wall_dod_change")
    gex_total = _f(features, "gex_total")

    # --- Gates ---
    # Transition from a negative regime up into transitional positive
    if not (prev_regime in NEGATIVE_REGIMES and regime == REGIME_TRANSITIONAL_POSITIVE):
        return None
    # GEX 5-day MA inflecting up
    if not _gex_ma_inflecting_up(history_df, gex_total, cfg["gex_ma_window"]):
        return None
    # Call wall migrating higher
    if not (np.isfinite(call_wall_dod) and call_wall_dod > 0):
        return None
    # Volume confirmation on the reclaim day
    if not (np.isfinite(vol_z) and vol_z > cfg["volume_zscore_reclaim"]):
        return None

    strength = _strength([
        _z_component(vol_z, cfg["volume_zscore_reclaim"]),
    ])

    flip = _f(features, "flip_level")
    return {
        "playbook_id": "B",
        "playbook_name": "Gamma Flip Reclaim",
        "signal_strength": strength,
        "key_levels": {
            "put_wall": _f(features, "put_wall"),
            "call_wall": _f(features, "call_wall"),
            "flip": flip,
            "spot": _f(features, "spot"),
        },
        "invalidation_level": flip,   # thesis wrong if we fall back below the flip
        "suggested_size_pct": cfg["initial_size_pct"],
        "instrument": "shares",
        "gates_passed": {
            "regime_transition": features.get("regime_transition"),
            "gex_ma_inflecting_up": True,
            "call_wall_dod_change": call_wall_dod,
            "volume_z": vol_z,
        },
    }


# ---- Playbook C — Compression Breakout -------------------------------------

def evaluate_playbook_C(
    features: dict, ticker_config: dict, history_df: pd.DataFrame, thresholds: dict
) -> dict | None:
    cfg = thresholds["playbooks"]["C_compression"]

    regime = features.get("regime")
    term_slope = _f(features, "term_structure_slope")
    call_persist = _f(features, "wall_persistence_days_call")
    call_wall_dod = _f(features, "call_wall_dod_change")

    # --- Gates ---
    # Deep positive gamma for >= N sessions
    streak = _consecutive_regime_count(history_df, regime, REGIME_DEEP_POSITIVE)
    if streak < cfg["min_compression_days"]:
        return None
    # Call wall persistent at the same strike for >= N sessions
    if not (np.isfinite(call_persist) and call_persist >= cfg["min_compression_days"]):
        return None
    # Steep contango
    if not (np.isfinite(term_slope) and term_slope >= cfg["steep_contango_min_slope"]):
        return None
    # Catalyst within window OR call wall finally migrating up
    catalyst = _catalyst_within(features, cfg["catalyst_window_days"])
    wall_up = np.isfinite(call_wall_dod) and call_wall_dod > 0
    if not (catalyst or wall_up):
        return None

    strength = _strength([
        _z_component(term_slope, cfg["steep_contango_min_slope"]),
    ])

    call_wall = _f(features, "call_wall")
    flip = _f(features, "flip_level")
    return {
        "playbook_id": "C",
        "playbook_name": "Compression Breakout",
        "signal_strength": strength,
        "key_levels": {
            "put_wall": _f(features, "put_wall"),
            "call_wall": call_wall,
            "flip": flip,
            "spot": _f(features, "spot"),
        },
        # Defined-risk structure; invalidation at the flip (loss of the compression regime)
        "invalidation_level": flip,
        "suggested_size_pct": cfg["initial_size_pct"],
        "instrument": "defined_risk_calls",
        "gates_passed": {
            "deep_positive_streak": streak,
            "wall_persistence_days_call": call_persist,
            "term_structure_slope": term_slope,
            "catalyst_within_window": catalyst,
            "call_wall_dod_change": call_wall_dod,
        },
    }


# ---- Public API ------------------------------------------------------------

def evaluate_all(
    features: dict,
    ticker_config: dict,
    history_df: pd.DataFrame | None = None,
    thresholds: dict | None = None,
) -> list[dict]:
    """Run all three playbooks. Returns a list of fired signal dicts (possibly
    empty). Multiple playbooks can fire on the same name on the same day."""
    thresholds = thresholds or load_thresholds()
    signals = []

    a = evaluate_playbook_A(features, ticker_config, thresholds)
    if a:
        signals.append(a)
    b = evaluate_playbook_B(features, ticker_config, history_df, thresholds)
    if b:
        signals.append(b)
    c = evaluate_playbook_C(features, ticker_config, history_df, thresholds)
    if c:
        signals.append(c)

    return signals
