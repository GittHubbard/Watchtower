"""Stage 1 — yfinance earnings integration. Mock the network calls."""
from __future__ import annotations
from datetime import date
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from gamma_workflow import stage1_calendar, storage


@pytest.fixture(autouse=True)
def _isolate_earnings_cache(tmp_path, monkeypatch):
    """Redirect the earnings cache to a tmp dir so these tests never write into
    the real data/earnings/. days_to_next_earnings/event_flags persist the fetched
    date via storage.write_earnings_cache (which reads storage.EARNINGS_DIR at call
    time); without this, running the suite overwrites real cached dates with the
    mocked test values. Also clears the in-process memo around each test.
    """
    monkeypatch.setattr(storage, "EARNINGS_DIR", tmp_path / "earnings")
    stage1_calendar.clear_earnings_cache()
    yield
    stage1_calendar.clear_earnings_cache()


def _mk_ticker_with_calendar(cal_dict):
    """Build a mock yfinance.Ticker whose .calendar returns cal_dict."""
    mock_t = MagicMock()
    mock_t.calendar = cal_dict
    return mock_t


def test_yf_earnings_single_date():
    stage1_calendar.clear_earnings_cache()
    cal = {"Earnings Date": [date(2026, 6, 15)]}
    with patch("yfinance.Ticker", return_value=_mk_ticker_with_calendar(cal)):
        d = stage1_calendar._fetch_yf_earnings("AAPL")
        assert d == date(2026, 6, 15)


def test_yf_earnings_two_date_range_picks_earliest():
    stage1_calendar.clear_earnings_cache()
    cal = {"Earnings Date": [date(2026, 7, 30), date(2026, 7, 25)]}
    with patch("yfinance.Ticker", return_value=_mk_ticker_with_calendar(cal)):
        d = stage1_calendar._fetch_yf_earnings("MSFT")
        assert d == date(2026, 7, 25)


def test_yf_earnings_empty_dict():
    stage1_calendar.clear_earnings_cache()
    with patch("yfinance.Ticker", return_value=_mk_ticker_with_calendar({})):
        d = stage1_calendar._fetch_yf_earnings("SPY")
        assert d is None


def test_yf_earnings_network_failure_returns_none():
    stage1_calendar.clear_earnings_cache()
    with patch("yfinance.Ticker", side_effect=Exception("network")):
        d = stage1_calendar._fetch_yf_earnings("AAPL")
        assert d is None


def test_yf_earnings_cached_per_process():
    stage1_calendar.clear_earnings_cache()
    cal = {"Earnings Date": [date(2026, 8, 1)]}
    mock_ticker = MagicMock()
    with patch("yfinance.Ticker", return_value=_mk_ticker_with_calendar(cal)) as mock_ctor:
        stage1_calendar._fetch_yf_earnings("AAPL")
        stage1_calendar._fetch_yf_earnings("AAPL")
        # Should only call yf.Ticker once thanks to the cache
        assert mock_ctor.call_count == 1


def test_csv_override_wins_over_yfinance(tmp_path, monkeypatch):
    stage1_calendar.clear_earnings_cache()
    # Write an override CSV
    override_path = tmp_path / "earnings.csv"
    override_path.write_text("ticker,date\nAAPL,2026-06-01\n")
    monkeypatch.setattr(stage1_calendar, "CONFIG_DIR", tmp_path)

    # yfinance would say something later, but the CSV should win
    cal = {"Earnings Date": [date(2026, 8, 1)]}
    with patch("yfinance.Ticker", return_value=_mk_ticker_with_calendar(cal)):
        days = stage1_calendar.days_to_next_earnings("AAPL", date(2026, 5, 27))
        # CSV says June 1, asof May 27 → 5 days
        assert days == 5


def test_days_to_next_earnings_yf_fallback():
    stage1_calendar.clear_earnings_cache()
    cal = {"Earnings Date": [date(2026, 6, 30)]}
    with patch("yfinance.Ticker", return_value=_mk_ticker_with_calendar(cal)):
        days = stage1_calendar.days_to_next_earnings("MSFT", date(2026, 5, 27))
        assert days == 34


def test_yf_earnings_real_world_dict_shape():
    """Regression test using the exact Ticker.calendar dict shape that yfinance
    returns in production (per real PM-provided sample). Locks in the parsing
    behavior so future refactors can't break it."""
    import datetime as _dt
    stage1_calendar.clear_earnings_cache()
    real_cal = {
        "Dividend Date": _dt.date(2026, 5, 13),
        "Ex-Dividend Date": _dt.date(2026, 5, 10),
        "Earnings Date": [_dt.date(2026, 7, 30)],
        "Earnings High": 1.99,
        "Earnings Low": 1.83,
        "Earnings Average": 1.89541,
        "Revenue High": 112168000000,
        "Revenue Low": 107501000000,
        "Revenue Average": 109015606590,
    }
    with patch("yfinance.Ticker", return_value=_mk_ticker_with_calendar(real_cal)):
        d = stage1_calendar._fetch_yf_earnings("AAPL")
        assert d == _dt.date(2026, 7, 30)
        days = stage1_calendar.days_to_next_earnings("AAPL", _dt.date(2026, 5, 27))
        assert days == 64


def test_event_flags_includes_yf_earnings():
    stage1_calendar.clear_earnings_cache()
    cal = {"Earnings Date": [date(2026, 5, 30)]}
    with patch("yfinance.Ticker", return_value=_mk_ticker_with_calendar(cal)):
        flags = stage1_calendar.event_flags("NVDA", date(2026, 5, 27))
        assert flags["days_to_earnings"] == 3
        assert flags["within_event_window"] is True
