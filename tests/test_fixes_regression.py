"""Regression tests for the bug-fix pass.

Each test exercises a failure path the original 98-test suite did not cover —
notably the round-trip through log_features -> read_features_range, which is
where the dropped-transition (#4) and premature forward-return (#6) bugs lived.
"""
from __future__ import annotations
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from gamma_workflow import storage
from gamma_workflow import stage9_logging as s9
from gamma_workflow import stage1_calendar as cal
from gamma_workflow import stage2_regime as s2
from gamma_workflow import stage4_flow as s4
from gamma_workflow.config import load_thresholds


THRESHOLDS = load_thresholds()


@pytest.fixture
def tmp_features_dir(tmp_path, monkeypatch):
    """Point storage at a temp features dir so the round-trip is isolated."""
    fdir = tmp_path / "features"
    fdir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(storage, "FEATURES_DIR", fdir)
    return fdir


# ---- Fix #4: regime transition columns survive the parquet round-trip -------

def test_regime_transition_persists_round_trip(tmp_features_dir):
    asof = date(2026, 5, 28)
    rows = [{
        "ticker": "SPY",
        "regime": "transitional_positive",
        "prev_regime": "deep_negative",
        "regime_transition": "deep_negative->transitional_positive",
        "gex_total": 1.0,
        "spot": 500.0,
    }]
    s9.log_features(rows, asof)

    back = storage.read_features_range(asof, asof)
    assert not back.empty
    row = back[back["ticker"] == "SPY"].iloc[0]
    # Before the fix these columns were filtered out at the schema gate -> KeyError/NaN.
    assert row["prev_regime"] == "deep_negative"
    assert row["regime_transition"] == "deep_negative->transitional_positive"


def test_dead_columns_removed_from_schema():
    for dead in ("gex_zscore", "ivx_7d", "ivx_30d", "hv_30d"):
        assert dead not in s9.FEATURE_COLUMNS, f"{dead} should have been removed"
    # Live transition columns are present.
    assert "prev_regime" in s9.FEATURE_COLUMNS
    assert "regime_transition" in s9.FEATURE_COLUMNS
    # is_opex_minus_1 is returned by event_flags and must be persisted.
    assert "is_opex_minus_1" in s9.FEATURE_COLUMNS


# ---- Fix #6: forward returns not computed before enough trading days pass ----

def test_forward_return_gate_blocks_premature_20d(tmp_features_dir, monkeypatch):
    """A date only ~15 trading days (21 calendar days) before asof must NOT get
    fwd_ret_20d populated. The discriminating check is whether yfinance is even
    reached: the OLD gate (threshold = max(horizons) = 20 calendar days) lets a
    21-day-old row through; the FIXED gate (~35 calendar days) blocks it. We record
    the call rather than raise, because backfill swallows exceptions internally."""
    feature_date = date(2026, 5, 1)
    asof = feature_date + timedelta(days=21)  # 21 calendar days ~= 15 trading days

    df = pd.DataFrame([{
        "ticker": "SPY", "asof": feature_date, "spot": 500.0,
        "fwd_ret_1d": np.nan, "fwd_ret_5d": np.nan, "fwd_ret_20d": np.nan,
    }])
    storage.write_features(df, feature_date)

    reached = {"yf": False}

    class _Tracker:
        def __init__(self, *_a, **_k):
            reached["yf"] = True
        def history(self, *_a, **_k):
            return pd.DataFrame({"Close": []})

    monkeypatch.setattr(s9.yf, "Ticker", _Tracker)

    updated = s9.backfill_forward_returns(asof, horizons=(20,))
    assert reached["yf"] is False, "fixed gate must skip the date before reaching yfinance"
    assert updated == 0
    back = storage.read_features_range(feature_date, feature_date)
    assert bool(back["fwd_ret_20d"].isna().all())


def test_forward_return_runs_once_enough_days_elapsed(tmp_features_dir, monkeypatch):
    """When enough calendar days have passed, the row is processed and the
    trading-day-correct forward return is written from a synthetic close series."""
    feature_date = date(2026, 4, 1)
    asof = feature_date + timedelta(days=45)  # comfortably > 20 trading days

    df = pd.DataFrame([{
        "ticker": "SPY", "asof": feature_date, "spot": 100.0,
        "fwd_ret_1d": np.nan, "fwd_ret_5d": np.nan, "fwd_ret_20d": np.nan,
    }])
    storage.write_features(df, feature_date)

    # Build 40 business-day closes starting at feature_date, rising 1/day from 100.
    bdays = pd.bdate_range(feature_date, periods=40)
    closes = pd.Series(np.arange(100.0, 140.0), index=bdays)

    class _FakeTicker:
        def __init__(self, *_a, **_k):
            pass
        def history(self, *_a, **_k):
            return pd.DataFrame({"Close": closes})

    monkeypatch.setattr(s9.yf, "Ticker", _FakeTicker)

    updated = s9.backfill_forward_returns(asof, horizons=(20,))
    assert updated == 1
    back = storage.read_features_range(feature_date, feature_date)
    # 20 trading days forward: close index 0 -> index 20 => 120/100 - 1 = 0.20
    assert back["fwd_ret_20d"].iloc[0] == pytest.approx(0.20)


# ---- Fix #7: regime history counts days, not unique GEX values ---------------

def test_regime_history_counts_rows_not_unique_gex():
    min_days = int(THRESHOLDS["regime"]["history_days_required"])
    # Exactly min_days rows but only a handful of DISTINCT gex values.
    gex_vals = ([1.0, 1.0, 2.0, 2.0] * ((min_days // 4) + 1))[:min_days]
    hist = pd.DataFrame({"ticker": ["SPY"] * min_days, "gex_total": gex_vals})

    out = s2.classify(
        gex_total=1.5, spot=500.0, flip_level=490.0, ticker="SPY",
        history_df=hist, mcap_b=500.0, thresholds=THRESHOLDS,
    )
    # With row-counting we have >= min_days -> percentile method.
    # Under the old .unique() count there were only 2 distinct values -> market_cap.
    assert out["threshold_method"] == "percentile"
    assert out["history_days_used"] == min_days


# ---- Fix #2: is_opex_plus_1 flag removed ------------------------------------

def test_is_opex_plus_1_removed():
    flags = cal.event_flags("SPY", date(2026, 5, 28))
    assert "is_opex_plus_1" not in flags
    # The other opex flags remain.
    assert "is_opex_day" in flags
    assert "is_opex_minus_1" in flags


# ---- Term structure: guard against IVX extrapolation when no ~90d expiry -----


class _FakeTickerOpts:
    """Minimal ticker_obj exposing only .options for the term-structure guard."""
    def __init__(self, expiries):
        self.options = expiries


def _expiries_max_dte(max_dte: int):
    base = date.today()
    # A couple of near expiries plus one at max_dte.
    days = [7, 14, 30, max_dte]
    return [(base + timedelta(days=d)).strftime("%Y-%m-%d") for d in days]


def test_term_structure_nan_when_no_long_expiry(monkeypatch):
    """Longest expiry ~50d, target 90d: the engine would extrapolate a fake 90d
    IVX. The guard must return NaN instead so Playbook C sees 'no signal'."""
    # If the guard fails and calls through, this would be used — make it obvious.
    monkeypatch.setattr(
        s4, "_engine",
        lambda: type("E", (), {"calculate_iv_index": staticmethod(lambda *a, **k: 25.0)})(),
    )
    tk = _FakeTickerOpts(_expiries_max_dte(50))
    assert not np.isfinite(s4._term_structure_slope(tk, spot=500.0))


def test_term_structure_computes_when_long_expiry_present(monkeypatch):
    """With a real ~95d expiry present, the slope is computed (ivx_90 - ivx_7)."""

    def _ivx(ticker_obj, spot, target_days, option_type="both"):
        # 7d -> 20, 90d -> 26 => slope +6
        return 20.0 if target_days == 7 else 26.0

    monkeypatch.setattr(
        s4, "_engine",
        lambda: type("E", (), {"calculate_iv_index": staticmethod(_ivx)})(),
    )
    tk = _FakeTickerOpts(_expiries_max_dte(95))
    slope = s4._term_structure_slope(tk, spot=500.0)
    assert slope == pytest.approx(6.0)
