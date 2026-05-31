"""Stage 8 — Monitoring & exit logic.

For each open position, given today's fresh feature row, emit recommended
management actions. No auto-execution — these are alerts for the PM.

Position schema (maintained by the PM in data/positions.json):
    {
        "ticker": "SPY",
        "playbook_id": "A",
        "entry_price": 578.0,
        "entry_date": "2026-05-20",
        "stop": 575.0,
        "target_1": 590.0,
        "target_2": 600.0,
        "shares": 1200,
        "size_pct": 0.50,
        "status": "open"
    }

Exit / management rules (from the workflow spec):
    1. Regime flips negative→positive while long  → trail stop up to the gamma flip
       (let it run).
    2. Regime flips positive→negative while long  → exit half.
    3. Spot reaches/exceeds the call wall AND volume z > scale_out_volume_zscore
       → scale out scale_out_fraction (default 50%).
    4. Put wall breaks on a CLOSING basis (close < put wall) → full exit.
    5. Walls migrate → recompute suggested stop/targets so the PM can adjust.

Each action is a dict: {"action": str, "reason": str, ...payload}. A position can
generate multiple actions in one session (e.g. trail stop + recompute targets).

Public API:
    evaluate_position(position, features, thresholds) -> list[action]
    render_alert(ticker, actions) -> str   (markdown)
"""
from __future__ import annotations
import logging

import numpy as np

from .config import load_thresholds
from .stage2_regime import (
    REGIME_DEEP_NEGATIVE,
    REGIME_TRANSITIONAL_NEGATIVE,
    REGIME_TRANSITIONAL_POSITIVE,
    REGIME_DEEP_POSITIVE,
)

logger = logging.getLogger(__name__)

NEGATIVE_REGIMES = {REGIME_DEEP_NEGATIVE, REGIME_TRANSITIONAL_NEGATIVE}
POSITIVE_REGIMES = {REGIME_TRANSITIONAL_POSITIVE, REGIME_DEEP_POSITIVE}


def _f(d: dict, key: str) -> float:
    v = d.get(key)
    if v is None:
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def evaluate_position(position: dict, features: dict, thresholds: dict | None = None) -> list[dict]:
    """Return a list of recommended actions for one open position given today's
    features. Empty list = hold, nothing to do."""
    thresholds = thresholds or load_thresholds()
    mon = thresholds["monitor"]
    actions: list[dict] = []

    if position.get("status") != "open":
        return actions

    spot = _f(features, "spot")
    call_wall = _f(features, "call_wall")
    put_wall = _f(features, "put_wall")
    flip = _f(features, "flip_level")
    vol_z = _f(features, "volume_z")
    transition = features.get("regime_transition")

    prev_regime = features.get("prev_regime")
    regime = features.get("regime")

    # 1 & 2 — regime flip handling
    if prev_regime in NEGATIVE_REGIMES and regime in POSITIVE_REGIMES:
        if np.isfinite(flip):
            actions.append({
                "action": "trail_stop",
                "reason": f"Regime flipped up ({transition}); let it run, trail stop to gamma flip.",
                "new_stop": flip,
            })
    elif prev_regime in POSITIVE_REGIMES and regime in NEGATIVE_REGIMES:
        actions.append({
            "action": "exit_half",
            "reason": f"Regime flipped down ({transition}); reduce exposure by half.",
        })

    # 3 — call wall hit on volume
    if np.isfinite(spot) and np.isfinite(call_wall) and spot >= call_wall:
        if np.isfinite(vol_z) and vol_z > mon["scale_out_volume_zscore"]:
            actions.append({
                "action": "scale_out",
                "reason": f"Spot tagged call wall ({call_wall:.2f}) on high volume "
                          f"(z={vol_z:.2f}); scale out {mon['scale_out_fraction']:.0%}.",
                "fraction": mon["scale_out_fraction"],
            })

    # 4 — put wall break on closing basis
    if np.isfinite(spot) and np.isfinite(put_wall) and spot < put_wall:
        actions.append({
            "action": "full_exit",
            "reason": f"Close ({spot:.2f}) below put wall ({put_wall:.2f}); thesis invalidated.",
        })

    # 5 — wall migration → recompute suggested stop/targets
    # For Playbook A the stop tracks the put wall; for B/C it tracks the flip level.
    # Compare each playbook's reference wall to the recorded stop so the alert fires
    # only when that specific level has moved since entry — not on every bar where
    # two different levels (e.g. flip vs put wall) happen to differ.
    cur_stop = _f(position, "stop")
    ref_wall = put_wall if position.get("playbook_id") == "A" else flip
    if np.isfinite(ref_wall) and np.isfinite(cur_stop) and abs(ref_wall - cur_stop) > 1e-9:
        actions.append({
            "action": "adjust_levels",
            "reason": "Walls migrated since entry; review stop/targets.",
            "suggested_stop": put_wall if position.get("playbook_id") == "A" else flip,
            "suggested_target_1": call_wall,
        })

    return actions


def render_alert(ticker: str, actions: list[dict]) -> str:
    """Render a markdown alert summarizing recommended actions for one position."""
    if not actions:
        return ""
    lines = [f"# Position Alert — {ticker}", "", "> For human review. No order placed.", ""]
    for a in actions:
        lines.append(f"## {a['action'].replace('_', ' ').title()}")
        lines.append(a["reason"])
        for k, v in a.items():
            if k in ("action", "reason"):
                continue
            if isinstance(v, float):
                lines.append(f"- {k}: {v:,.2f}")
            else:
                lines.append(f"- {k}: {v}")
        lines.append("")
    return "\n".join(lines)
