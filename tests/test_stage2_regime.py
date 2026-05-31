"""Stage 2 — regime classifier tests."""
from __future__ import annotations
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from gamma_workflow import stage2_regime, storage


@pytest.fixture(autouse=True)
def _isolate_storage(tmp_path, monkeypatch):
    """Redirect all data dirs to a fresh temp location per test."""
    monkeypatch.setattr(storage, "DATA_DIR", tmp_path)
    monkeypatch.setattr(storage, "FEATURES_DIR", tmp_path / "features")
    monkeypatch.setattr(storage, "SIGNALS_DIR", tmp_path / "signals")
    monkeypatch.setattr(storage, "TICKETS_DIR", tmp_path / "tickets")
    monkeypatch.setattr(storage, "RATES_DIR", tmp_path / "rates")
    yield


def test_market_cap_fallback_deep_positive():
    # No history. SPY mcap_b = 500 → threshold = 0.01 * 500e9 = $5B
    result = stage2_regime.classify(
        gex_total=10e9,           # above $5B
        spot=580.0,
        flip_level=560.0,         # spot above flip
        ticker="SPY",
        mcap_b=500.0,
    )
    assert result["regime"] == stage2_regime.REGIME_DEEP_POSITIVE
    assert result["threshold_method"] == "market_cap"
    assert result["distance_to_flip_pct"] > 0


def test_market_cap_fallback_transitional_positive():
    result = stage2_regime.classify(
        gex_total=1e9,            # below $5B but spot above flip
        spot=580.0,
        flip_level=560.0,
        ticker="SPY",
        mcap_b=500.0,
    )
    assert result["regime"] == stage2_regime.REGIME_TRANSITIONAL_POSITIVE


def test_market_cap_fallback_deep_negative():
    result = stage2_regime.classify(
        gex_total=-10e9,
        spot=540.0,
        flip_level=560.0,         # spot below flip
        ticker="SPY",
        mcap_b=500.0,
    )
    assert result["regime"] == stage2_regime.REGIME_DEEP_NEGATIVE


def test_market_cap_fallback_transitional_negative():
    result = stage2_regime.classify(
        gex_total=-1e9,
        spot=540.0,
        flip_level=560.0,
        ticker="SPY",
        mcap_b=500.0,
    )
    assert result["regime"] == stage2_regime.REGIME_TRANSITIONAL_NEGATIVE


def test_percentile_path_when_enough_history():
    # Build a fake history df with 60 days of GEX in [0, 100], plus today at 95.
    rng = np.random.default_rng(42)
    gex_vals = rng.uniform(0, 100, size=60)
    history = pd.DataFrame({"ticker": ["SPY"] * 60, "gex_total": gex_vals})

    result = stage2_regime.classify(
        gex_total=95.0,           # above the empirical 80th percentile
        spot=580.0,
        flip_level=560.0,
        ticker="SPY",
        history_df=history,
        mcap_b=500.0,             # should be ignored since history is enough
    )
    assert result["threshold_method"] == "percentile"
    assert result["regime"] == stage2_regime.REGIME_DEEP_POSITIVE


def test_percentile_path_insufficient_history_falls_back():
    history = pd.DataFrame({"ticker": ["SPY"] * 10, "gex_total": np.arange(10, dtype=float)})
    result = stage2_regime.classify(
        gex_total=10e9,
        spot=580.0,
        flip_level=560.0,
        ticker="SPY",
        history_df=history,
        mcap_b=500.0,
    )
    assert result["threshold_method"] == "market_cap"


def test_unknown_flip_uses_gex_sign():
    # No flip → above_flip falls back to GEX > 0
    pos = stage2_regime.classify(
        gex_total=10e9, spot=580.0, flip_level=None, ticker="SPY", mcap_b=500.0,
    )
    assert pos["regime"] == stage2_regime.REGIME_DEEP_POSITIVE
    assert np.isnan(pos["distance_to_flip_pct"])

    neg = stage2_regime.classify(
        gex_total=-10e9, spot=540.0, flip_level=None, ticker="SPY", mcap_b=500.0,
    )
    assert neg["regime"] == stage2_regime.REGIME_DEEP_NEGATIVE


def test_classify_with_history_detects_transition():
    # Seed the features store with yesterday's row showing deep_negative regime.
    asof = date(2026, 5, 27)
    yesterday = asof - timedelta(days=1)
    yest_df = pd.DataFrame([{
        "ticker": "SPY",
        "asof": yesterday,
        "spot": 540.0,
        "regime": stage2_regime.REGIME_DEEP_NEGATIVE,
        "gex_total": -8e9,
    }])
    # Pad to satisfy log_features's column normalization
    from gamma_workflow.stage9_logging import log_features
    log_features(yest_df.to_dict(orient="records"), yesterday)

    # Today: spot has reclaimed and we're now transitional positive
    result = stage2_regime.classify_with_history(
        gex_total=1e9,
        spot=580.0,
        flip_level=570.0,         # spot above flip now
        ticker="SPY",
        asof=asof,
        mcap_b=500.0,
    )
    assert result["regime"] == stage2_regime.REGIME_TRANSITIONAL_POSITIVE
    assert result["prev_regime"] == stage2_regime.REGIME_DEEP_NEGATIVE
    assert result["regime_transition"] == "deep_negative->transitional_positive"


def test_classify_with_history_no_transition_when_unchanged():
    asof = date(2026, 5, 27)
    yesterday = asof - timedelta(days=1)
    yest_df = pd.DataFrame([{
        "ticker": "SPY",
        "asof": yesterday,
        "spot": 580.0,
        "regime": stage2_regime.REGIME_DEEP_POSITIVE,
        "gex_total": 8e9,
    }])
    from gamma_workflow.stage9_logging import log_features
    log_features(yest_df.to_dict(orient="records"), yesterday)

    result = stage2_regime.classify_with_history(
        gex_total=8e9, spot=580.0, flip_level=560.0,
        ticker="SPY", asof=asof, mcap_b=500.0,
    )
    assert result["regime"] == stage2_regime.REGIME_DEEP_POSITIVE
    assert result["regime_transition"] is None
