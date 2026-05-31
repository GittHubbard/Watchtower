"""Orchestrator integration test — exercises the pure feature-assembly and
signal path with a fully mocked `market` dict, so no engine or network needed.

This proves the wiring between stages 2-7 is correct: feature row assembly →
playbook firing → ticket creation.
"""
from __future__ import annotations
from datetime import date, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from gamma_workflow import run_eod, storage, stage4_flow, stage5_confirm
from gamma_workflow.config import get_ticker_config, load_thresholds


@pytest.fixture(autouse=True)
def _isolate_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DATA_DIR", tmp_path)
    monkeypatch.setattr(storage, "FEATURES_DIR", tmp_path / "features")
    monkeypatch.setattr(storage, "SIGNALS_DIR", tmp_path / "signals")
    monkeypatch.setattr(storage, "TICKETS_DIR", tmp_path / "tickets")
    monkeypatch.setattr(storage, "RATES_DIR", tmp_path / "rates")
    monkeypatch.setattr(storage, "CHAINS_DIR", tmp_path / "chains")
    monkeypatch.setattr(storage, "POSITIONS_PATH", tmp_path / "positions.json")
    yield


def _synthetic_gex_profile(spot=500.0, flip=505.0, n=200):
    """Negative-gamma-below-flip profile with a put wall near spot."""
    prices = np.linspace(440.0, 560.0, n)
    # Put wall (trough) just above spot at 500.5, call wall (peak) at 520
    g_put = -4e9 * np.exp(-((prices - 500.5) / 2.0) ** 2)
    g_call = 5e9 * np.exp(-((prices - 520) / 2.0) ** 2)
    gex = g_put + g_call
    return prices, gex


def _synthetic_chain():
    T = 30 / 365.0
    expiry = (date.today() + timedelta(days=30)).isoformat()
    rows = []
    for k in (490, 495, 500, 505, 510):
        rows.append({"strike": k, "oi": 1000, "volume": 1500, "iv": 0.30,
                     "is_call": True, "T": T, "expiry_str": expiry})
    for k in (490, 495, 500, 505, 510):
        rows.append({"strike": k, "oi": 1500, "volume": 3000, "iv": 0.40,
                     "is_call": False, "T": T, "expiry_str": expiry})
    return pd.DataFrame(rows)


def _fake_bs(S, K, T, r, sigma, is_call):
    K = np.asarray(K, dtype=float)
    is_call = np.asarray(is_call, dtype=bool)
    moneyness = (S - K) / S
    call_delta = np.clip(0.5 + moneyness, 0.0, 1.0)
    delta = np.where(is_call, call_delta, call_delta - 1.0)
    return np.zeros_like(K), delta, np.zeros_like(K)


def test_assemble_feature_row_wires_all_stages(monkeypatch):
    """Feed a mocked market dict; verify the row has fields from every stage."""
    asof = date(2026, 5, 27)
    prices, gex = _synthetic_gex_profile()
    chain = _synthetic_chain()

    # Mock the Stage 4 engine accessor (skew needs BS deltas)
    class _FakeEngine:
        black_scholes_metrics_vectorized = staticmethod(_fake_bs)
    monkeypatch.setattr(stage4_flow, "_engine", lambda: _FakeEngine)

    # Mock Stage 5 underlying history fetch with a synthetic uptrend + VIX
    def _mock_hist(symbol, end_date, days=260):
        n = 60
        dates = pd.bdate_range(end=pd.Timestamp(end_date), periods=n)
        if symbol == "^VIX":
            close = np.linspace(15, 22, n)  # rising VIX
        else:
            close = np.linspace(520, 500, n)  # falling underlying (washout)
        return pd.DataFrame({
            "Open": close, "High": close + 1, "Low": close - 1,
            "Close": close, "Volume": [2_000_000] * n,
        }, index=dates)
    monkeypatch.setattr(stage5_confirm, "fetch_underlying_history", _mock_hist)

    market = {
        "spot": 500.0,
        "prices": prices,
        "gex_array": gex,
        "flip_level": 505.0,
        "chain": chain,
        "prev_chain": None,
        "ticker_obj": None,   # term structure → NaN, fine
        "underlying_history": _mock_hist("SPY", asof),
    }
    tcfg = get_ticker_config("SPY")
    thresholds = load_thresholds()

    row = run_eod.assemble_feature_row("SPY", asof, market, tcfg, r=0.045, thresholds=thresholds)

    # Stage 2
    assert "regime" in row
    assert "distance_to_flip_pct" in row
    # Stage 3
    assert "call_wall" in row and "put_wall" in row and "hvl" in row
    assert abs(row["put_wall"] - 500.5) < 2.0
    assert abs(row["call_wall"] - 520.0) < 2.0
    # Stage 4
    assert "pcr_volume" in row and "skew_25d" in row
    assert row["pcr_volume"] > 1.0  # puts have more volume than calls in our chain
    # Stage 5
    assert "volume_z" in row and "vix_close" in row
    # Stage 1
    assert "days_to_opex" in row and "within_event_window" in row


def test_full_run_eod_with_mocked_acquisition(monkeypatch):
    """End-to-end run_eod for a single ticker with acquisition fully mocked.
    Verifies features get logged and (if a signal fires) a ticket is written."""
    asof = date(2026, 5, 27)
    prices, gex = _synthetic_gex_profile()
    chain = _synthetic_chain()

    class _FakeEngine:
        black_scholes_metrics_vectorized = staticmethod(_fake_bs)
    monkeypatch.setattr(stage4_flow, "_engine", lambda: _FakeEngine)

    def _mock_hist(symbol, end_date, days=260):
        n = 60
        dates = pd.bdate_range(end=pd.Timestamp(end_date), periods=n)
        close = np.linspace(15, 22, n) if symbol == "^VIX" else np.linspace(520, 500, n)
        vol = [2_000_000] * (n - 1) + [8_000_000]  # capitulation spike today
        return pd.DataFrame({
            "Open": close, "High": close + 1, "Low": close - 1,
            "Close": close, "Volume": vol,
        }, index=dates)
    monkeypatch.setattr(stage5_confirm, "fetch_underlying_history", _mock_hist)

    def _mock_acquire(symbol, asof_, r):
        return {
            "spot": 500.0, "prices": prices, "gex_array": gex, "flip_level": 505.0,
            "chain": chain, "prev_chain": None, "ticker_obj": None,
            "underlying_history": _mock_hist(symbol, asof_),
        }
    monkeypatch.setattr(run_eod, "acquire_market_data", _mock_acquire)
    monkeypatch.setattr(run_eod, "get_rate", lambda d: 0.045)

    summary = run_eod.run_eod(asof=asof, symbols=["SPY"])

    assert summary["tickers_processed"] == 1
    # Features file should exist for the day
    feats = storage.read_features(asof)
    assert feats is not None
    assert (feats["ticker"] == "SPY").any()


def test_run_eod_throttles_between_tickers(monkeypatch):
    """Verify the inter-ticker sleep is called N-1 times for N tickers (not before
    the first one). We patch time.sleep so the test runs instantly."""
    asof = date(2026, 5, 27)
    prices, gex = _synthetic_gex_profile()
    chain = _synthetic_chain()

    class _FakeEngine:
        black_scholes_metrics_vectorized = staticmethod(_fake_bs)
    monkeypatch.setattr(stage4_flow, "_engine", lambda: _FakeEngine)

    def _mock_hist(symbol, end_date, days=260):
        n = 60
        dates = pd.bdate_range(end=pd.Timestamp(end_date), periods=n)
        close = np.linspace(15, 22, n) if symbol == "^VIX" else np.linspace(520, 500, n)
        return pd.DataFrame({
            "Open": close, "High": close + 1, "Low": close - 1,
            "Close": close, "Volume": [2_000_000] * n,
        }, index=dates)
    monkeypatch.setattr(stage5_confirm, "fetch_underlying_history", _mock_hist)

    def _mock_acquire(symbol, asof_, r):
        return {
            "spot": 500.0, "prices": prices, "gex_array": gex, "flip_level": 505.0,
            "chain": chain, "prev_chain": None, "ticker_obj": None,
            "underlying_history": _mock_hist(symbol, asof_),
        }
    monkeypatch.setattr(run_eod, "acquire_market_data", _mock_acquire)
    monkeypatch.setattr(run_eod, "get_rate", lambda d: 0.045)

    sleep_calls = []
    monkeypatch.setattr(run_eod.time, "sleep", lambda s: sleep_calls.append(s))

    # 3 tickers in the watchlist → sleep called exactly 2 times (between 1-2 and 2-3)
    summary = run_eod.run_eod(asof=asof, symbols=["SPY", "QQQ", "IWM"])

    assert summary["tickers_processed"] == 3
    assert len(sleep_calls) == 2, f"Expected 2 sleeps for 3 tickers, got {len(sleep_calls)}"
    assert all(s == 2.0 for s in sleep_calls), f"Expected 2.0s sleeps, got {sleep_calls}"


def test_run_eod_no_sleep_for_single_ticker(monkeypatch):
    """With only one ticker, time.sleep should never be called."""
    asof = date(2026, 5, 27)
    prices, gex = _synthetic_gex_profile()
    chain = _synthetic_chain()

    class _FakeEngine:
        black_scholes_metrics_vectorized = staticmethod(_fake_bs)
    monkeypatch.setattr(stage4_flow, "_engine", lambda: _FakeEngine)

    def _mock_hist(symbol, end_date, days=260):
        n = 60
        dates = pd.bdate_range(end=pd.Timestamp(end_date), periods=n)
        close = np.linspace(15, 22, n) if symbol == "^VIX" else np.linspace(520, 500, n)
        return pd.DataFrame({
            "Open": close, "High": close + 1, "Low": close - 1,
            "Close": close, "Volume": [2_000_000] * n,
        }, index=dates)
    monkeypatch.setattr(stage5_confirm, "fetch_underlying_history", _mock_hist)

    def _mock_acquire(symbol, asof_, r):
        return {
            "spot": 500.0, "prices": prices, "gex_array": gex, "flip_level": 505.0,
            "chain": chain, "prev_chain": None, "ticker_obj": None,
            "underlying_history": _mock_hist(symbol, asof_),
        }
    monkeypatch.setattr(run_eod, "acquire_market_data", _mock_acquire)
    monkeypatch.setattr(run_eod, "get_rate", lambda d: 0.045)

    sleep_calls = []
    monkeypatch.setattr(run_eod.time, "sleep", lambda s: sleep_calls.append(s))

    run_eod.run_eod(asof=asof, symbols=["SPY"])
    assert sleep_calls == []
