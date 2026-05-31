"""Smoke tests for the foundation modules. No network calls."""
from __future__ import annotations
from datetime import date, timedelta
from pathlib import Path
import tempfile
import os

import numpy as np
import pandas as pd
import pytest

from gamma_workflow import config, storage, stage1_calendar, stage9_logging


def test_load_watchlist_and_thresholds():
    wl = config.load_watchlist()
    assert "tickers" in wl
    assert any(t["symbol"] == "SPY" for t in wl["tickers"])

    th = config.load_thresholds()
    assert "regime" in th
    assert th["regime"]["history_days_required"] == 60


def test_get_ticker_config_merges_defaults():
    cfg = config.get_ticker_config("SPY")
    assert cfg["symbol"] == "SPY"
    assert cfg["risk_per_trade_bps"] == 50  # from defaults
    assert cfg["kind"] == "index_etf"

    with pytest.raises(KeyError):
        config.get_ticker_config("DOES_NOT_EXIST")


def test_opex_calculation_known_dates():
    # Sanity-check a few known OPEX dates.
    assert stage1_calendar.monthly_opex(2026, 1) == date(2026, 1, 16)  # 3rd Fri
    assert stage1_calendar.monthly_opex(2026, 6) == date(2026, 6, 19)
    assert stage1_calendar.monthly_opex(2024, 12) == date(2024, 12, 20)


def test_vixp_is_a_wednesday():
    for y in (2025, 2026):
        for m in range(1, 13):
            v = stage1_calendar.vix_expiration(y, m)
            assert v.weekday() == 2, f"VIXP not Wednesday: {y}-{m} -> {v}"


def test_days_to_next_opex_non_negative():
    d = date(2026, 5, 27)
    assert stage1_calendar.days_to_next_opex(d) >= 0


def test_event_flags_keys_present():
    flags = stage1_calendar.event_flags("SPY", date(2026, 5, 27))
    for key in ("days_to_opex", "days_to_vixp", "is_opex_day", "within_event_window"):
        assert key in flags


def test_storage_roundtrip(tmp_path, monkeypatch):
    # Redirect DATA_DIR to a temp location
    monkeypatch.setattr(storage, "DATA_DIR", tmp_path)
    monkeypatch.setattr(storage, "FEATURES_DIR", tmp_path / "features")
    monkeypatch.setattr(storage, "SIGNALS_DIR", tmp_path / "signals")
    monkeypatch.setattr(storage, "TICKETS_DIR", tmp_path / "tickets")
    monkeypatch.setattr(storage, "RATES_DIR", tmp_path / "rates")

    d = date(2026, 5, 27)
    df = pd.DataFrame([
        {"ticker": "SPY", "spot": 580.0, "regime": "deep_positive"},
        {"ticker": "QQQ", "spot": 510.0, "regime": "transitional_positive"},
    ])
    storage.write_features(df, d)
    back = storage.read_features(d)
    assert back is not None
    assert len(back) == 2
    assert set(back["ticker"]) == {"SPY", "QQQ"}


def test_log_features_normalizes_columns(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "FEATURES_DIR", tmp_path / "features")
    monkeypatch.setattr(storage, "DATA_DIR", tmp_path)
    monkeypatch.setattr(storage, "SIGNALS_DIR", tmp_path / "signals")
    monkeypatch.setattr(storage, "TICKETS_DIR", tmp_path / "tickets")
    monkeypatch.setattr(storage, "RATES_DIR", tmp_path / "rates")

    d = date(2026, 5, 27)
    # Pass partial rows; missing cols must be NaN, no crash.
    rows = [
        {"ticker": "SPY", "spot": 580.0, "regime": "deep_positive", "gex_total": 1.5e9},
        {"ticker": "QQQ", "spot": 510.0},  # mostly empty
    ]
    stage9_logging.log_features(rows, d)
    back = storage.read_features(d)
    assert back is not None
    assert "fwd_ret_5d" in back.columns
    assert back["fwd_ret_5d"].isna().all()  # always NaN on write
    assert (back["asof"] == d).all()
