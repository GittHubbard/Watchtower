"""Stage 3 — level map tests. Built on synthetic GEX profiles so we know the
correct answer in advance."""
from __future__ import annotations
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from gamma_workflow import stage3_level_map, storage


@pytest.fixture(autouse=True)
def _isolate_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DATA_DIR", tmp_path)
    monkeypatch.setattr(storage, "FEATURES_DIR", tmp_path / "features")
    monkeypatch.setattr(storage, "SIGNALS_DIR", tmp_path / "signals")
    monkeypatch.setattr(storage, "TICKETS_DIR", tmp_path / "tickets")
    monkeypatch.setattr(storage, "RATES_DIR", tmp_path / "rates")
    yield


def _synthetic_profile(spot=100.0, flip=98.0, n=200):
    """A clean GEX profile with two distinct peaks above flip (call wall + secondary)
    and two distinct troughs below flip (put wall + secondary)."""
    prices = np.linspace(80.0, 120.0, n)
    # Two Gaussian peaks above flip (call side) at 102 and 110
    g_call = 5e9 * np.exp(-((prices - 102) / 1.5) ** 2)
    g_call_sec = 3e9 * np.exp(-((prices - 110) / 1.5) ** 2)
    # Two Gaussian troughs below flip (put side) at 95 and 88
    g_put = -4e9 * np.exp(-((prices - 95) / 1.5) ** 2)
    g_put_sec = -2.5e9 * np.exp(-((prices - 88) / 1.5) ** 2)
    gex = g_call + g_call_sec + g_put + g_put_sec
    return prices, gex, spot, flip


def test_primary_walls_match_peaks():
    prices, gex, spot, flip = _synthetic_profile()
    out = stage3_level_map.compute_levels(prices, gex, spot, flip)
    # Allow 1-step grid resolution slack
    assert abs(out["call_wall"] - 102.0) < 0.5
    assert abs(out["put_wall"] - 95.0) < 0.5
    assert out["call_wall_gex"] > 0
    assert out["put_wall_gex"] < 0


def test_secondary_walls_respect_2pct_separation():
    prices, gex, spot, flip = _synthetic_profile(spot=100.0)
    out = stage3_level_map.compute_levels(prices, gex, spot, flip, min_separation_pct=0.02)
    # Spot=100 → minimum separation = $2. Secondary call at ~110, secondary put at ~88.
    assert abs(out["call_wall_secondary"] - 110.0) < 0.5
    assert abs(out["put_wall_secondary"] - 88.0) < 0.5
    # Both must be at least 2% of spot ($2) away from their primary
    assert abs(out["call_wall_secondary"] - out["call_wall"]) >= 2.0
    assert abs(out["put_wall_secondary"] - out["put_wall"]) >= 2.0


def test_secondary_returns_nan_when_no_room():
    # Tiny price grid that has nowhere to put a secondary >= 2% away
    prices = np.linspace(99.0, 101.0, 50)
    gex = -1e9 * np.exp(-((prices - 99.5) / 0.2) ** 2) + 1e9 * np.exp(-((prices - 100.5) / 0.2) ** 2)
    out = stage3_level_map.compute_levels(prices, gex, spot=100.0, flip_level=100.0, min_separation_pct=0.02)
    # The grid only spans 2% total, so secondaries can't be placed beyond 2% from primary
    assert np.isnan(out["call_wall_secondary"])
    assert np.isnan(out["put_wall_secondary"])


def test_hvl_picks_highest_volume_strike():
    chain = pd.DataFrame({
        "strike": [95.0, 100.0, 105.0, 100.0],  # 100 appears twice (call+put)
        "volume": [200, 1500, 800, 2000],       # 100 sums to 3500 → wins
        "is_call": [True, True, True, False],
    })
    prices, gex, spot, flip = _synthetic_profile()
    out = stage3_level_map.compute_levels(prices, gex, spot, flip, options_chain=chain)
    assert out["hvl"] == 100.0


def test_hvl_nan_when_no_volume():
    chain = pd.DataFrame({"strike": [100.0], "volume": [0], "is_call": [True]})
    prices, gex, spot, flip = _synthetic_profile()
    out = stage3_level_map.compute_levels(prices, gex, spot, flip, options_chain=chain)
    assert np.isnan(out["hvl"])


def test_unknown_flip_uses_gex_sign():
    prices, gex, spot, _ = _synthetic_profile()
    out = stage3_level_map.compute_levels(prices, gex, spot, flip_level=None)
    # Should still find the peaks via GEX-sign masking
    assert abs(out["call_wall"] - 102.0) < 0.5
    assert abs(out["put_wall"] - 95.0) < 0.5


def test_migration_no_history():
    out = stage3_level_map.compute_migration(
        "SPY", date(2026, 5, 27), call_wall=102.0, put_wall=95.0, spot=100.0,
    )
    assert np.isnan(out["call_wall_dod_change"])
    assert out["wall_persistence_days_call"] == 0


def test_migration_and_persistence_with_history():
    from gamma_workflow.stage9_logging import log_features
    asof = date(2026, 5, 27)
    # Seed 4 prior days where call wall stayed at ~102 (within 0.5% of 100 = $0.50)
    for i, d_off in enumerate([4, 3, 2, 1]):
        d = asof - timedelta(days=d_off)
        log_features([{
            "ticker": "SPY",
            "asof": d,
            "call_wall": 102.0 + 0.1 * i,   # tiny drift, all within tolerance
            "put_wall": 95.0,
        }], d)

    out = stage3_level_map.compute_migration(
        "SPY", asof, call_wall=102.4, put_wall=95.0, spot=100.0,
        tolerance_pct=0.005,
    )
    # Yesterday's call wall was 102.3, today is 102.4 → dod = +0.1
    assert abs(out["call_wall_dod_change"] - 0.1) < 1e-6
    # Put wall unchanged
    assert abs(out["put_wall_dod_change"]) < 1e-6
    # 4 prior days all within tolerance ⇒ persistence count = 4
    assert out["wall_persistence_days_call"] == 4
    assert out["wall_persistence_days_put"] == 4


def test_persistence_breaks_on_first_miss():
    from gamma_workflow.stage9_logging import log_features
    asof = date(2026, 5, 27)
    # Day t-3: wall WAY off; t-2 and t-1: wall close to today
    log_features([{"ticker": "SPY", "asof": asof - timedelta(days=3), "call_wall": 90.0, "put_wall": 95.0}], asof - timedelta(days=3))
    log_features([{"ticker": "SPY", "asof": asof - timedelta(days=2), "call_wall": 102.0, "put_wall": 95.0}], asof - timedelta(days=2))
    log_features([{"ticker": "SPY", "asof": asof - timedelta(days=1), "call_wall": 102.0, "put_wall": 95.0}], asof - timedelta(days=1))

    out = stage3_level_map.compute_migration(
        "SPY", asof, call_wall=102.0, put_wall=95.0, spot=100.0, tolerance_pct=0.005,
    )
    # Walking backward from yesterday: 102.0 ✓, 102.0 ✓, 90.0 ✗ → count stops at 2
    assert out["wall_persistence_days_call"] == 2


def test_build_level_map_end_to_end():
    prices, gex, spot, flip = _synthetic_profile()
    out = stage3_level_map.build_level_map(
        "SPY", date(2026, 5, 27), prices, gex, spot, flip,
    )
    # All keys present and primaries match
    expected_keys = {
        "call_wall", "call_wall_gex", "call_wall_secondary",
        "put_wall", "put_wall_gex", "put_wall_secondary",
        "hvl", "flip_level",
        "call_wall_dod_change", "put_wall_dod_change",
        "wall_persistence_days_call", "wall_persistence_days_put",
    }
    assert expected_keys.issubset(out.keys())
    assert out["flip_level"] == flip
