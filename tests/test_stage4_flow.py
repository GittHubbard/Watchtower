"""Stage 4 — flow & positioning tests.

Synthetic option chains let us verify each computation against known inputs.
We don't need a real engine_shim here since the BS calls happen inside Stage 4
on numpy arrays — they'll work on any chain that has the required columns
(strike, T, iv, oi, is_call, expiry_str, and optionally volume).
"""
from __future__ import annotations
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from gamma_workflow import stage4_flow, storage
from gamma_workflow.stage9_logging import log_features


@pytest.fixture(autouse=True)
def _isolate_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DATA_DIR", tmp_path)
    monkeypatch.setattr(storage, "FEATURES_DIR", tmp_path / "features")
    monkeypatch.setattr(storage, "SIGNALS_DIR", tmp_path / "signals")
    monkeypatch.setattr(storage, "TICKETS_DIR", tmp_path / "tickets")
    monkeypatch.setattr(storage, "RATES_DIR", tmp_path / "rates")
    yield


@pytest.fixture
def fake_engine(monkeypatch):
    """Replace stage4_flow._engine() with a stub that returns approximate BS
    deltas, so tests don't require the real engine module on disk."""
    def fake_bs(S, K, T, r, sigma, is_call):
        K = np.asarray(K, dtype=float)
        is_call = np.asarray(is_call, dtype=bool)
        moneyness = (S - K) / S
        call_delta = np.clip(0.5 + moneyness, 0.0, 1.0)
        delta = np.where(is_call, call_delta, call_delta - 1.0)
        return np.zeros_like(K), delta, np.zeros_like(K)

    class _FakeEngine:
        black_scholes_metrics_vectorized = staticmethod(fake_bs)

    monkeypatch.setattr(stage4_flow, "_engine", lambda: _FakeEngine)
    yield


def _synthetic_chain(
    strikes_call=(95, 100, 105, 110),
    strikes_put=(90, 95, 100, 105),
    oi_call=(100, 500, 800, 200),
    oi_put=(300, 1200, 600, 150),
    vol_call=(50, 600, 200, 50),
    vol_put=(100, 1500, 200, 30),
    iv_call=(0.30, 0.28, 0.27, 0.29),
    iv_put=(0.34, 0.32, 0.30, 0.29),
    T_days=30,
):
    """Build a small synthetic options chain with all columns Stage 4 expects."""
    T = T_days / 365.0
    expiry_str = (date.today() + timedelta(days=T_days)).isoformat()
    rows = []
    for k, oi, v, iv in zip(strikes_call, oi_call, vol_call, iv_call):
        rows.append({"strike": k, "oi": oi, "volume": v, "iv": iv, "is_call": True,
                     "T": T, "expiry_str": expiry_str})
    for k, oi, v, iv in zip(strikes_put, oi_put, vol_put, iv_put):
        rows.append({"strike": k, "oi": oi, "volume": v, "iv": iv, "is_call": False,
                     "T": T, "expiry_str": expiry_str})
    return pd.DataFrame(rows)


def test_pcr_volume_and_oi():
    chain = _synthetic_chain()
    pcr_v = stage4_flow._pcr(chain, "volume")
    pcr_o = stage4_flow._pcr(chain, "oi")
    # calls volume = 50+600+200+50 = 900; puts volume = 100+1500+200+30 = 1830
    assert abs(pcr_v - (1830 / 900)) < 1e-9
    # calls oi = 1600; puts oi = 2250
    assert abs(pcr_o - (2250 / 1600)) < 1e-9


def test_pcr_nan_when_no_calls():
    chain = _synthetic_chain(strikes_call=(), oi_call=(), vol_call=(), iv_call=())
    # _pcr with no calls should return NaN, not crash
    pcr_v = stage4_flow._pcr(chain, "volume")
    assert np.isnan(pcr_v)


def test_new_oi_strike_count():
    chain = _synthetic_chain(
        vol_call=(50, 600, 1000, 50),    # call strike 100 (600>500) and 105 (1000>800) flagged
        vol_put=(100, 1500, 700, 30),    # put strike 95 (1500>1200) and 100 (700>600) flagged
    )
    count = stage4_flow._new_oi_strike_count(chain)
    assert count == 4  # 2 calls + 2 puts


def test_oi_delta_with_prior_chain():
    today = _synthetic_chain(oi_call=(200, 700, 900, 300), oi_put=(400, 1300, 700, 250))
    prev = _synthetic_chain(oi_call=(100, 500, 800, 200), oi_put=(300, 1200, 600, 150))
    delta_c, delta_p = stage4_flow._oi_deltas(today, prev)
    # net call delta: (200-100)+(700-500)+(900-800)+(300-200) = 500
    assert delta_c == 500.0
    # net put delta: (400-300)+(1300-1200)+(700-600)+(250-150) = 400
    assert delta_p == 400.0


def test_oi_delta_nan_when_no_prior():
    today = _synthetic_chain()
    dc, dp = stage4_flow._oi_deltas(today, None)
    assert np.isnan(dc) and np.isnan(dp)


def test_skew_25d_is_put_iv_minus_call_iv(fake_engine):
    # Build a chain where puts have HIGHER IV at the 25-delta strike → positive skew
    chain = _synthetic_chain(
        strikes_call=(95, 100, 105, 110, 115, 120),
        strikes_put=(80, 85, 90, 95, 100, 105),
        oi_call=(100, 500, 800, 600, 400, 200),
        oi_put=(200, 400, 800, 1200, 600, 300),
        vol_call=(50,) * 6,
        vol_put=(50,) * 6,
        iv_call=(0.30, 0.28, 0.27, 0.26, 0.27, 0.28),
        iv_put=(0.38, 0.36, 0.34, 0.32, 0.30, 0.29),
    )
    skew = stage4_flow._skew_25d(chain, spot=100.0, r=0.045)
    assert np.isfinite(skew)
    assert skew > 0  # put skew should be positive given our IV curve


def test_rolling_z_insufficient_history():
    s = pd.Series([1.0, 2.0, 3.0])  # only 3 obs, need 20
    z = stage4_flow._rolling_z(s, 5.0)
    assert np.isnan(z)


def test_rolling_z_with_enough_history():
    # 25 values with mean 50, std 10 → current=70 should give z=2
    rng = np.random.default_rng(7)
    s = pd.Series(50 + 10 * rng.standard_normal(25))
    # adjust to enforce exact std
    s = (s - s.mean()) / s.std() * 10 + 50
    z = stage4_flow._rolling_z(s, 70.0)
    assert abs(z - 2.0) < 0.5  # allow some slack from sampling


def test_compute_flow_end_to_end_no_history(fake_engine):
    chain = _synthetic_chain()
    out = stage4_flow.compute_flow(
        symbol="SPY", asof=date(2026, 5, 27), chain=chain, spot=100.0,
        r=0.045, ticker_obj=None, prev_chain=None,
    )
    # Point estimates present
    assert np.isfinite(out["pcr_volume"])
    assert np.isfinite(out["pcr_oi"])
    # Default synthetic chain has 2 strikes where today's volume > today's OI
    assert out["new_oi_strike_count"] == 2
    # z-scores NaN with no history
    assert np.isnan(out["pcr_volume_z"])
    assert np.isnan(out["total_volume_z"])
    # Term structure NaN with no ticker_obj
    assert np.isnan(out["term_structure_slope"])
    # OI deltas NaN with no prev_chain
    assert np.isnan(out["oi_delta_call"])


def test_compute_flow_z_score_with_seeded_history(fake_engine):
    """Seed 21 prior days of pcr_volume values, then verify the z-score lands
    in a sensible range."""
    asof = date(2026, 5, 27)
    rng = np.random.default_rng(42)
    pcr_vals = 1.2 + 0.15 * rng.standard_normal(21)
    for i, val in enumerate(pcr_vals):
        d = asof - timedelta(days=21 - i)
        log_features(
            [{"ticker": "SPY", "asof": d, "pcr_volume": float(val)}], d,
        )

    chain = _synthetic_chain()  # today's pcr_volume ≈ 2.03 — well above the ~1.2 mean
    out = stage4_flow.compute_flow(
        symbol="SPY", asof=asof, chain=chain, spot=100.0, r=0.045,
    )
    assert np.isfinite(out["pcr_volume_z"])
    assert out["pcr_volume_z"] > 2.0  # today is much higher than the seeded mean
