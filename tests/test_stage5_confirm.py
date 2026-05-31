"""Stage 5 — confirmation tests. Synthetic OHLCV avoids network."""
from __future__ import annotations
from datetime import date, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd

from gamma_workflow import stage5_confirm


def _synthetic_ohlcv(end_date: date, n_days: int = 260, base: float = 100.0, vol: float = 1_000_000):
    """Build a deterministic OHLCV frame with mild trend + noise."""
    rng = np.random.default_rng(123)
    dates = pd.bdate_range(end=pd.Timestamp(end_date), periods=n_days)
    close = base + np.cumsum(rng.standard_normal(n_days) * 0.5)
    high = close + np.abs(rng.standard_normal(n_days)) * 0.6
    low = close - np.abs(rng.standard_normal(n_days)) * 0.6
    open_ = close + rng.standard_normal(n_days) * 0.3
    volume = vol + rng.standard_normal(n_days) * (vol * 0.1)
    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )
    return df


def test_compute_features_basic_shape():
    end = date(2026, 5, 27)
    hist = _synthetic_ohlcv(end, n_days=250)
    out = stage5_confirm._compute_features_from_history(hist, end)
    for key in (
        "volume_z", "atr_pct", "atr_20d_pct", "range_expansion", "close_location",
        "ma20_dist_pct", "ma50_dist_pct", "ma200_dist_pct",
    ):
        assert key in out
        assert np.isfinite(out[key]), f"{key} should be finite given 250 days of history"


def test_volume_z_extreme_spike():
    end = date(2026, 5, 27)
    hist = _synthetic_ohlcv(end, n_days=60, vol=1_000_000)
    # Replace today's volume with 5x mean
    hist.iloc[-1, hist.columns.get_loc("Volume")] = 10_000_000
    out = stage5_confirm._compute_features_from_history(hist, end)
    assert out["volume_z"] > 2.0  # clearly anomalous


def test_close_location_top_and_bottom_of_range():
    end = date(2026, 5, 27)
    hist = _synthetic_ohlcv(end, n_days=60)
    # Force close == high → close_location should be 1.0
    today_idx = hist.index[-1]
    hist.loc[today_idx, "Close"] = float(hist.loc[today_idx, "High"])
    out = stage5_confirm._compute_features_from_history(hist, end)
    assert abs(out["close_location"] - 1.0) < 1e-9


def test_ma_distance_directions():
    end = date(2026, 5, 27)
    # Manufacture a clean uptrend so price > all MAs
    n = 60
    dates = pd.bdate_range(end=pd.Timestamp(end), periods=n)
    close = np.linspace(80, 120, n)
    hist = pd.DataFrame({
        "Open": close, "High": close + 0.5, "Low": close - 0.5,
        "Close": close, "Volume": [1_000_000] * n,
    }, index=dates)
    out = stage5_confirm._compute_features_from_history(hist, end)
    assert out["ma20_dist_pct"] > 0
    assert out["ma50_dist_pct"] > 0
    # ma200 unavailable (n<200) → NaN
    assert np.isnan(out["ma200_dist_pct"])


def test_insufficient_history_returns_nans():
    end = date(2026, 5, 27)
    hist = _synthetic_ohlcv(end, n_days=5)  # way too short
    out = stage5_confirm._compute_features_from_history(hist, end)
    # volume_z requires 20 days
    assert np.isnan(out["volume_z"])
    # atr_pct needs at least 2 days
    assert np.isfinite(out["atr_pct"]) or np.isnan(out["atr_pct"])
    # ATR_20d needs 20 days
    assert np.isnan(out["atr_20d_pct"])


def test_empty_history_returns_all_nans():
    end = date(2026, 5, 27)
    out = stage5_confirm._compute_features_from_history(pd.DataFrame(), end)
    for v in out.values():
        assert np.isnan(v)


def test_compute_confirmation_mocks_yfinance():
    """End-to-end: mock fetch_underlying_history and verify the dict shape."""
    end = date(2026, 5, 27)
    hist_spy = _synthetic_ohlcv(end, n_days=250)
    hist_vix = _synthetic_ohlcv(end, n_days=20, base=15.0, vol=1)

    call_log = []

    def _mock_fetch(symbol, end_date, days=260):
        call_log.append(symbol)
        return hist_spy if symbol != "^VIX" else hist_vix

    with patch.object(stage5_confirm, "fetch_underlying_history", side_effect=_mock_fetch):
        out = stage5_confirm.compute_confirmation("SPY", end)

    # Should have fetched both SPY and ^VIX
    assert "SPY" in call_log
    assert "^VIX" in call_log
    assert np.isfinite(out["volume_z"])
    assert np.isfinite(out["vix_close"])
    assert np.isfinite(out["vix_5d_change_pct"])


def test_vix_features_pre_supplied_history():
    end = date(2026, 5, 27)
    # If VIX fetch returns empty, vix fields should be NaN
    with patch.object(stage5_confirm, "fetch_underlying_history", return_value=pd.DataFrame()):
        out = stage5_confirm._compute_vix_features(end)
    assert np.isnan(out["vix_close"])
    assert np.isnan(out["vix_5d_change_pct"])
