"""Stage 8 — position monitor tests."""
from __future__ import annotations

import numpy as np
import pytest

from gamma_workflow import stage8_monitor as s8
from gamma_workflow import stage2_regime as s2
from gamma_workflow.config import load_thresholds


THRESHOLDS = load_thresholds()


def _position(**overrides):
    pos = {
        "ticker": "SPY",
        "playbook_id": "A",
        "entry_price": 500.0,
        "stop": 495.0,
        "target_1": 520.0,
        "shares": 50_000,
        "status": "open",
    }
    pos.update(overrides)
    return pos


def _features(**overrides):
    feats = {
        "ticker": "SPY",
        "spot": 505.0,
        "call_wall": 520.0,
        "put_wall": 495.0,
        "flip_level": 510.0,
        "volume_z": 0.5,
        "regime": s2.REGIME_DEEP_NEGATIVE,
        "prev_regime": s2.REGIME_DEEP_NEGATIVE,
        "regime_transition": None,
    }
    feats.update(overrides)
    return feats


def test_no_actions_on_quiet_hold():
    # Stop already equals put wall so no adjust; nothing else triggers
    actions = s8.evaluate_position(_position(stop=495.0), _features(), THRESHOLDS)
    action_types = {a["action"] for a in actions}
    assert "full_exit" not in action_types
    assert "scale_out" not in action_types


def test_regime_flip_up_trails_stop():
    feats = _features(
        prev_regime=s2.REGIME_DEEP_NEGATIVE,
        regime=s2.REGIME_TRANSITIONAL_POSITIVE,
        regime_transition="deep_negative->transitional_positive",
        flip_level=508.0,
    )
    actions = s8.evaluate_position(_position(), feats, THRESHOLDS)
    trail = [a for a in actions if a["action"] == "trail_stop"]
    assert len(trail) == 1
    assert trail[0]["new_stop"] == 508.0


def test_regime_flip_down_exits_half():
    feats = _features(
        prev_regime=s2.REGIME_DEEP_POSITIVE,
        regime=s2.REGIME_TRANSITIONAL_NEGATIVE,
        regime_transition="deep_positive->transitional_negative",
    )
    actions = s8.evaluate_position(_position(), feats, THRESHOLDS)
    assert any(a["action"] == "exit_half" for a in actions)


def test_call_wall_hit_on_volume_scales_out():
    feats = _features(spot=521.0, call_wall=520.0, volume_z=2.5)  # z > 2.0
    actions = s8.evaluate_position(_position(), feats, THRESHOLDS)
    scale = [a for a in actions if a["action"] == "scale_out"]
    assert len(scale) == 1
    assert scale[0]["fraction"] == 0.50


def test_call_wall_hit_without_volume_does_not_scale():
    feats = _features(spot=521.0, call_wall=520.0, volume_z=1.0)  # z below threshold
    actions = s8.evaluate_position(_position(), feats, THRESHOLDS)
    assert not any(a["action"] == "scale_out" for a in actions)


def test_put_wall_break_full_exit():
    feats = _features(spot=494.0, put_wall=495.0)  # close below put wall
    actions = s8.evaluate_position(_position(), feats, THRESHOLDS)
    assert any(a["action"] == "full_exit" for a in actions)


def test_wall_migration_triggers_adjust():
    # Position stop is 495 but put wall has moved to 498 → adjust suggested
    feats = _features(put_wall=498.0, spot=505.0)
    actions = s8.evaluate_position(_position(stop=495.0), feats, THRESHOLDS)
    adjust = [a for a in actions if a["action"] == "adjust_levels"]
    assert len(adjust) == 1
    assert adjust[0]["suggested_stop"] == 498.0  # playbook A → put wall


def test_closed_position_yields_no_actions():
    actions = s8.evaluate_position(_position(status="closed"), _features(spot=494.0, put_wall=495.0), THRESHOLDS)
    assert actions == []


def test_render_alert_produces_markdown():
    feats = _features(spot=494.0, put_wall=495.0)
    actions = s8.evaluate_position(_position(), feats, THRESHOLDS)
    md = s8.render_alert("SPY", actions)
    assert "Position Alert" in md
    assert "SPY" in md
    assert "human review" in md.lower()


def test_render_alert_empty_when_no_actions():
    assert s8.render_alert("SPY", []) == ""
