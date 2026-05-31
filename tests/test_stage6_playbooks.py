"""Stage 6 — playbook rule engine tests.

Each playbook is tested with (a) a feature row crafted to pass every gate and
(b) variants that each break a single gate, confirming the playbook does NOT
fire. This is the heart of the system, so coverage is deliberately thorough.
"""
from __future__ import annotations
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from gamma_workflow import stage6_playbooks as s6
from gamma_workflow import stage2_regime as s2
from gamma_workflow.config import load_thresholds


THRESHOLDS = load_thresholds()
INDEX_CFG = {"symbol": "SPY", "kind": "index_etf", "market_cap_b": 500, "risk_per_trade_bps": 50}
NAME_CFG = {"symbol": "NVDA", "kind": "single_name", "market_cap_b": 3000, "risk_per_trade_bps": 50}


# ---- Playbook A — Washout at Put Wall --------------------------------------

def _row_for_A(**overrides):
    """A feature row that passes every Playbook A gate by default."""
    row = {
        "ticker": "SPY",
        "spot": 500.0,
        "put_wall": 500.5,          # within 0.5% of spot
        "call_wall": 520.0,
        "flip_level": 510.0,
        "regime": s2.REGIME_DEEP_NEGATIVE,
        "pcr_volume_z": 2.0,        # > 1.5 panic
        "skew_25d_z": 1.5,          # > 1.0 fear
        "volume_z": 2.0,            # > 1.5 capitulation
        "vix_5d_change_pct": 0.20,  # >= 0.15
        "within_event_window": False,
    }
    row.update(overrides)
    return row


def test_playbook_A_fires_on_full_setup():
    sig = s6.evaluate_playbook_A(_row_for_A(), INDEX_CFG, THRESHOLDS)
    assert sig is not None
    assert sig["playbook_id"] == "A"
    assert sig["instrument"] == "shares"
    assert sig["suggested_size_pct"] == 0.50
    assert sig["invalidation_level"] == 500.5
    assert 0.0 <= sig["signal_strength"] <= 1.0


def test_playbook_A_blocked_by_wrong_regime():
    sig = s6.evaluate_playbook_A(
        _row_for_A(regime=s2.REGIME_TRANSITIONAL_NEGATIVE), INDEX_CFG, THRESHOLDS)
    assert sig is None


def test_playbook_A_blocked_when_far_from_put_wall():
    sig = s6.evaluate_playbook_A(_row_for_A(put_wall=480.0), INDEX_CFG, THRESHOLDS)
    assert sig is None


def test_playbook_A_blocked_by_low_pcr():
    sig = s6.evaluate_playbook_A(_row_for_A(pcr_volume_z=1.0), INDEX_CFG, THRESHOLDS)
    assert sig is None


def test_playbook_A_blocked_by_low_skew():
    sig = s6.evaluate_playbook_A(_row_for_A(skew_25d_z=0.5), INDEX_CFG, THRESHOLDS)
    assert sig is None


def test_playbook_A_blocked_by_low_volume():
    sig = s6.evaluate_playbook_A(_row_for_A(volume_z=1.0), INDEX_CFG, THRESHOLDS)
    assert sig is None


def test_playbook_A_blocked_by_event_window():
    sig = s6.evaluate_playbook_A(_row_for_A(within_event_window=True), INDEX_CFG, THRESHOLDS)
    assert sig is None


def test_playbook_A_vix_only_enforced_for_index():
    # Single-name should fire even with weak VIX move
    sig = s6.evaluate_playbook_A(_row_for_A(vix_5d_change_pct=0.0), NAME_CFG, THRESHOLDS)
    assert sig is not None
    # Index with weak VIX move should NOT fire
    sig_idx = s6.evaluate_playbook_A(_row_for_A(vix_5d_change_pct=0.0), INDEX_CFG, THRESHOLDS)
    assert sig_idx is None


def test_playbook_A_nan_zscore_does_not_fire():
    # NaN z-scores must be treated as "no signal", not "passes"
    sig = s6.evaluate_playbook_A(_row_for_A(pcr_volume_z=float("nan")), INDEX_CFG, THRESHOLDS)
    assert sig is None


# ---- Playbook B — Gamma Flip Reclaim ---------------------------------------

def _row_for_B(**overrides):
    row = {
        "ticker": "SPY",
        "spot": 512.0,
        "put_wall": 500.0,
        "call_wall": 525.0,
        "flip_level": 510.0,
        "regime": s2.REGIME_TRANSITIONAL_POSITIVE,
        "prev_regime": s2.REGIME_DEEP_NEGATIVE,
        "regime_transition": "deep_negative->transitional_positive",
        "gex_total": 2.0e9,
        "call_wall_dod_change": 2.0,    # migrating up
        "volume_z": 1.5,                # > 1.0 reclaim
    }
    row.update(overrides)
    return row


def _history_rising_gex(window=5):
    """History with a rising GEX series so the MA inflects up when today is added."""
    base = date(2026, 5, 27)
    rows = []
    for i in range(window + 2):
        rows.append({
            "ticker": "SPY",
            "date": base - timedelta(days=(window + 2 - i)),
            "gex_total": -5.0e9 + i * 1.0e9,   # steadily rising
            "regime": s2.REGIME_DEEP_NEGATIVE,
        })
    return pd.DataFrame(rows)


def test_playbook_B_fires_on_full_setup():
    hist = _history_rising_gex()
    sig = s6.evaluate_playbook_B(_row_for_B(), INDEX_CFG, hist, THRESHOLDS)
    assert sig is not None
    assert sig["playbook_id"] == "B"
    assert sig["suggested_size_pct"] == 0.75
    assert sig["invalidation_level"] == 510.0  # the flip


def test_playbook_B_blocked_without_transition():
    hist = _history_rising_gex()
    sig = s6.evaluate_playbook_B(
        _row_for_B(prev_regime=s2.REGIME_TRANSITIONAL_POSITIVE), INDEX_CFG, hist, THRESHOLDS)
    assert sig is None


def test_playbook_B_blocked_when_call_wall_not_rising():
    hist = _history_rising_gex()
    sig = s6.evaluate_playbook_B(_row_for_B(call_wall_dod_change=-1.0), INDEX_CFG, hist, THRESHOLDS)
    assert sig is None


def test_playbook_B_blocked_by_low_volume():
    hist = _history_rising_gex()
    sig = s6.evaluate_playbook_B(_row_for_B(volume_z=0.5), INDEX_CFG, hist, THRESHOLDS)
    assert sig is None


def test_playbook_B_blocked_when_gex_ma_not_rising():
    # Falling GEX history → MA not inflecting up
    base = date(2026, 5, 27)
    rows = [{
        "ticker": "SPY",
        "date": base - timedelta(days=7 - i),
        "gex_total": 5.0e9 - i * 1.0e9,   # falling
        "regime": s2.REGIME_DEEP_NEGATIVE,
    } for i in range(7)]
    hist = pd.DataFrame(rows)
    sig = s6.evaluate_playbook_B(_row_for_B(gex_total=-3.0e9), INDEX_CFG, hist, THRESHOLDS)
    assert sig is None


def test_playbook_B_blocked_with_insufficient_history():
    # Only 2 days of history; need window+1 = 6
    base = date(2026, 5, 27)
    hist = pd.DataFrame([
        {"ticker": "SPY", "date": base - timedelta(days=2), "gex_total": -2e9, "regime": s2.REGIME_DEEP_NEGATIVE},
        {"ticker": "SPY", "date": base - timedelta(days=1), "gex_total": -1e9, "regime": s2.REGIME_DEEP_NEGATIVE},
    ])
    sig = s6.evaluate_playbook_B(_row_for_B(), INDEX_CFG, hist, THRESHOLDS)
    assert sig is None


# ---- Playbook C — Compression Breakout -------------------------------------

def _row_for_C(**overrides):
    row = {
        "ticker": "SPY",
        "spot": 515.0,
        "put_wall": 500.0,
        "call_wall": 520.0,
        "flip_level": 505.0,
        "regime": s2.REGIME_DEEP_POSITIVE,
        "wall_persistence_days_call": 6,    # >= 5
        "term_structure_slope": 3.0,        # >= 2.0 steep contango
        "call_wall_dod_change": 0.0,
        "days_to_earnings": 7,              # catalyst within 10
        "days_to_macro": None,
    }
    row.update(overrides)
    return row


def _history_deep_positive(n=6):
    base = date(2026, 5, 27)
    return pd.DataFrame([{
        "ticker": "SPY",
        "date": base - timedelta(days=n - i),
        "gex_total": 8.0e9,
        "regime": s2.REGIME_DEEP_POSITIVE,
    } for i in range(n)])


def test_playbook_C_fires_on_full_setup():
    hist = _history_deep_positive()
    sig = s6.evaluate_playbook_C(_row_for_C(), INDEX_CFG, hist, THRESHOLDS)
    assert sig is not None
    assert sig["playbook_id"] == "C"
    assert sig["instrument"] == "defined_risk_calls"
    assert sig["suggested_size_pct"] == 0.50


def test_playbook_C_blocked_by_short_streak():
    # History all transitional → streak only counts today = 1
    base = date(2026, 5, 27)
    hist = pd.DataFrame([{
        "ticker": "SPY", "date": base - timedelta(days=3 - i),
        "gex_total": 1e9, "regime": s2.REGIME_TRANSITIONAL_POSITIVE,
    } for i in range(3)])
    sig = s6.evaluate_playbook_C(_row_for_C(), INDEX_CFG, hist, THRESHOLDS)
    assert sig is None


def test_playbook_C_blocked_by_short_wall_persistence():
    hist = _history_deep_positive()
    sig = s6.evaluate_playbook_C(_row_for_C(wall_persistence_days_call=2), INDEX_CFG, hist, THRESHOLDS)
    assert sig is None


def test_playbook_C_blocked_by_flat_term_structure():
    hist = _history_deep_positive()
    sig = s6.evaluate_playbook_C(_row_for_C(term_structure_slope=0.5), INDEX_CFG, hist, THRESHOLDS)
    assert sig is None


def test_playbook_C_blocked_without_catalyst_or_wall_migration():
    hist = _history_deep_positive()
    sig = s6.evaluate_playbook_C(
        _row_for_C(days_to_earnings=30, days_to_macro=None, call_wall_dod_change=0.0),
        INDEX_CFG, hist, THRESHOLDS)
    assert sig is None


def test_playbook_C_fires_on_wall_migration_without_catalyst():
    hist = _history_deep_positive()
    sig = s6.evaluate_playbook_C(
        _row_for_C(days_to_earnings=30, days_to_macro=None, call_wall_dod_change=1.5),
        INDEX_CFG, hist, THRESHOLDS)
    assert sig is not None


# ---- evaluate_all -----------------------------------------------------------

def test_evaluate_all_returns_multiple_signals():
    # Engineer a row that fires A (we'll just confirm the dispatcher collects it)
    hist = pd.DataFrame()
    fired = s6.evaluate_all(_row_for_A(), INDEX_CFG, history_df=hist, thresholds=THRESHOLDS)
    assert any(s["playbook_id"] == "A" for s in fired)


def test_evaluate_all_empty_when_nothing_fires():
    flat_row = {
        "ticker": "SPY", "spot": 500.0, "put_wall": 480.0, "call_wall": 520.0,
        "flip_level": 510.0, "regime": s2.REGIME_TRANSITIONAL_POSITIVE,
        "prev_regime": s2.REGIME_TRANSITIONAL_POSITIVE,
        "pcr_volume_z": 0.0, "skew_25d_z": 0.0, "volume_z": 0.0,
        "vix_5d_change_pct": 0.0, "within_event_window": False,
        "term_structure_slope": 0.0, "wall_persistence_days_call": 0,
        "call_wall_dod_change": 0.0, "gex_total": 1e9,
    }
    fired = s6.evaluate_all(flat_row, INDEX_CFG, history_df=pd.DataFrame(), thresholds=THRESHOLDS)
    assert fired == []
