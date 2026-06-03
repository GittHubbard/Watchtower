"""Tests for the four feature additions (branch feature-addition-0601):

  Feature 1 — IV/HV ratio: ivx_30d (Stage 4), hv_30d (Stage 5), ratio + z,
              Playbook A strength booster, Playbook C strength reducer + ticket note.
  Feature 2 — Technicals (Stage 5.5): MACD / RSI / CMF / Bollinger.
  Feature 3 — Fundamentals (Stage 5.5): scalar extraction, ETF all-NaN,
              evolution cache (append-if-changed), ticket commentary.
  Feature 4 — Alpha (Stage 9): fwd/benchmark/alpha across 1/5/20/60/90d,
              benchmark fetched once, per-horizon gating.

All synthetic — no network or engine required.
"""
from __future__ import annotations
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from gamma_workflow import storage
from gamma_workflow import stage9_logging as s9
from gamma_workflow import stage5_confirm as s5
from gamma_workflow import stage4_flow as s4
from gamma_workflow import stage5_5_context as s55
from gamma_workflow import stage6_playbooks as s6
from gamma_workflow import stage7_ticket as s7
from gamma_workflow.config import load_thresholds, get_ticker_config


THRESHOLDS = load_thresholds()


@pytest.fixture
def tmp_features_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "FEATURES_DIR", tmp_path / "features")
    (tmp_path / "features").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(storage, "DATA_DIR", tmp_path)
    monkeypatch.setattr(storage, "FUNDAMENTALS_DIR", tmp_path / "fundamentals")
    return tmp_path


# =============================================================================
# Feature 4 — Alpha vs benchmark
# =============================================================================

def _hist_df(closes: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"Close": closes})


def _rising(start: date, n: int, slope: float, base: float = 100.0) -> pd.Series:
    idx = pd.bdate_range(start, periods=n)
    return pd.Series(base + slope * np.arange(n), index=idx)


class _FakeYF:
    """yf.Ticker stand-in returning per-symbol close series and logging calls."""
    def __init__(self, series_map):
        self.series_map = series_map
        self.calls = []

    def __call__(self, symbol):
        self.calls.append(symbol)
        outer = self

        class _T:
            def history(self, *a, **k):
                s = outer.series_map.get(symbol)
                return _hist_df(s) if s is not None else pd.DataFrame({"Close": []})
        return _T()


def test_alpha_columns_present_in_schema():
    for h in (1, 5, 20, 60, 90):
        assert f"fwd_ret_{h}d" in s9.FEATURE_COLUMNS
        assert f"benchmark_ret_{h}d" in s9.FEATURE_COLUMNS
        assert f"alpha_{h}d" in s9.FEATURE_COLUMNS


def test_backfill_computes_alpha_and_fetches_benchmark_once(tmp_features_dir, monkeypatch):
    feature_date = date(2026, 4, 1)
    asof = feature_date + timedelta(days=45)

    df = pd.DataFrame([
        {"ticker": "NVDA", "asof": feature_date, "spot": 100.0,
         "fwd_ret_5d": np.nan, "benchmark_ret_5d": np.nan, "alpha_5d": np.nan},
        {"ticker": "MSFT", "asof": feature_date, "spot": 100.0,
         "fwd_ret_5d": np.nan, "benchmark_ret_5d": np.nan, "alpha_5d": np.nan},
    ])
    storage.write_features(df, feature_date)

    series = {
        "NVDA": _rising(feature_date, 40, slope=2.0),   # +2/day → 5d = 110/100-1 = 0.10
        "MSFT": _rising(feature_date, 40, slope=1.0),   # +1/day → 5d = 105/100-1 = 0.05
        "SPY":  _rising(feature_date, 40, slope=0.8),   # +0.8/day → 5d = 104/100-1 = 0.04
    }
    fake = _FakeYF(series)
    monkeypatch.setattr(s9.yf, "Ticker", fake)

    updated = s9.backfill_forward_returns(asof, horizons=(5,), benchmark_ticker="SPY")
    assert updated == 2

    back = storage.read_features_range(feature_date, feature_date)
    nvda = back[back["ticker"] == "NVDA"].iloc[0]
    msft = back[back["ticker"] == "MSFT"].iloc[0]
    assert nvda["fwd_ret_5d"] == pytest.approx(0.10)
    assert nvda["benchmark_ret_5d"] == pytest.approx(0.04)
    assert nvda["alpha_5d"] == pytest.approx(0.06)
    assert msft["alpha_5d"] == pytest.approx(0.01)
    # Benchmark fetched exactly once for the whole run, not once per ticker/file.
    assert fake.calls.count("SPY") == 1


def test_backfill_per_horizon_gating(tmp_features_dir, monkeypatch):
    """A 45-day-old row fills 5d (ready) but not 90d (needs ~140 calendar days)."""
    feature_date = date(2026, 4, 1)
    asof = feature_date + timedelta(days=45)
    df = pd.DataFrame([{
        "ticker": "NVDA", "asof": feature_date, "spot": 100.0,
        "fwd_ret_5d": np.nan, "fwd_ret_90d": np.nan,
    }])
    storage.write_features(df, feature_date)

    fake = _FakeYF({"NVDA": _rising(feature_date, 40, 2.0), "SPY": _rising(feature_date, 40, 0.8)})
    monkeypatch.setattr(s9.yf, "Ticker", fake)

    s9.backfill_forward_returns(asof, horizons=(5, 90), benchmark_ticker="SPY")
    back = storage.read_features_range(feature_date, feature_date).iloc[0]
    assert np.isfinite(back["fwd_ret_5d"])
    assert np.isnan(back["fwd_ret_90d"])      # gated out
    assert np.isnan(back["alpha_90d"])


# =============================================================================
# Feature 1 — IV/HV ratio
# =============================================================================

def test_realized_vol_constant_returns_zero():
    # Constant daily growth → log returns constant → std ~0 → hv ~0.
    closes = pd.Series(100.0 * (1.01 ** np.arange(40)))
    hv = s5._realized_vol(closes, window=30)
    assert hv == pytest.approx(0.0, abs=1e-9)


def test_realized_vol_nan_when_short():
    assert np.isnan(s5._realized_vol(pd.Series(np.arange(10.0)), window=30))


def test_compute_flow_iv_hv_ratio(monkeypatch):
    """ivx (percent) / hv (decimal) reconcile to a dimensionless ratio ≈ 1.0."""
    def _bs(S, K, T, r, sigma, is_call):
        K = np.asarray(K, dtype=float)
        return np.zeros_like(K), np.full_like(K, 0.25), np.zeros_like(K)

    monkeypatch.setattr(
        s4, "_engine",
        lambda: type("E", (), {
            "calculate_iv_index": staticmethod(lambda *a, **k: 30.0),
            "black_scholes_metrics_vectorized": staticmethod(_bs),
        })(),
    )
    # Empty history store → z is NaN, ratio still computed.
    chain = pd.DataFrame({
        "strike": [100], "oi": [10], "volume": [10], "iv": [0.3],
        "is_call": [True], "T": [30 / 365], "expiry_str": ["2026-07-01"],
    })
    out = s4.compute_flow("NVDA", date(2026, 5, 27), chain, spot=100.0, r=0.045,
                          ticker_obj=object(), hv_30d=0.30)
    assert out["ivx_30d"] == pytest.approx(30.0)
    # (30/100) / 0.30 = 1.0
    assert out["iv_hv_ratio_30d"] == pytest.approx(1.0)


def _playbook_C_features(**overrides):
    """A feature row that passes all Playbook C gates."""
    feats = {
        "ticker": "NVDA", "spot": 500.0, "regime": s6.REGIME_DEEP_POSITIVE,
        "term_structure_slope": 4.0, "wall_persistence_days_call": 6,
        "call_wall": 520.0, "call_wall_dod_change": 1.0, "flip_level": 490.0,
        "put_wall": 470.0, "days_to_earnings": 5, "days_to_macro": None,
        "iv_hv_ratio_30d": np.nan, "iv_hv_ratio_z": np.nan,
    }
    feats.update(overrides)
    return feats


def _streak_history(n=6, regime=s6.REGIME_DEEP_POSITIVE):
    dates = pd.bdate_range(end=pd.Timestamp(date(2026, 5, 26)), periods=n)
    return pd.DataFrame({"date": dates, "regime": [regime] * n, "gex_total": [1e9] * n})


def test_playbook_C_reducer_bootstrap_ratio():
    """With <20d history (z NaN) a raw ratio above bootstrap halves strength + notes."""
    hist = _streak_history()
    tcfg = {"kind": "single_name"}
    # No reducer (ratio below bootstrap)
    base = s6.evaluate_playbook_C(_playbook_C_features(iv_hv_ratio_30d=1.0), tcfg, hist, THRESHOLDS)
    # Reducer active (ratio above bootstrap 1.3)
    reduced = s6.evaluate_playbook_C(_playbook_C_features(iv_hv_ratio_30d=1.6), tcfg, hist, THRESHOLDS)
    assert base is not None and reduced is not None
    assert reduced["signal_strength"] == pytest.approx(base["signal_strength"] * 0.5)
    assert reduced["gates_passed"]["iv_hv_reducer_applied"] is True
    assert any("IV/HV elevated" in n for n in reduced["notes"])
    assert base["notes"] == []


def test_playbook_C_reducer_zscore_mode():
    """Once z is finite, the z-gate (not the raw ratio) drives the reducer."""
    hist = _streak_history()
    tcfg = {"kind": "single_name"}
    # z above 1.0 → reducer even though raw ratio is modest.
    reduced = s6.evaluate_playbook_C(
        _playbook_C_features(iv_hv_ratio_30d=1.05, iv_hv_ratio_z=1.5), tcfg, hist, THRESHOLDS)
    assert reduced["gates_passed"]["iv_hv_reducer_applied"] is True
    # z below 1.0 → no reducer even if raw ratio would have tripped bootstrap.
    not_reduced = s6.evaluate_playbook_C(
        _playbook_C_features(iv_hv_ratio_30d=1.6, iv_hv_ratio_z=0.2), tcfg, hist, THRESHOLDS)
    assert not_reduced["gates_passed"]["iv_hv_reducer_applied"] is False
    assert not_reduced["notes"] == []


def test_playbook_A_iv_hv_booster_raises_strength():
    """Elevated IV/HV z lifts Playbook A conviction (fear overpriced)."""
    # Gates are strict ">", so values sit clearly above their thresholds.
    base_feats = {
        "regime": s6.REGIME_DEEP_NEGATIVE, "spot": 100.0, "put_wall": 100.0,
        "pcr_volume_z": 2.0, "skew_25d_z": 1.5, "volume_z": 2.0,
        "vix_5d_change_pct": 0.2, "within_event_window": False,
        "call_wall": 110.0, "flip_level": 95.0,
        "iv_hv_ratio_30d": np.nan, "iv_hv_ratio_z": np.nan,
    }
    tcfg = {"kind": "single_name"}
    base = s6.evaluate_playbook_A(base_feats, tcfg, THRESHOLDS)
    boosted = s6.evaluate_playbook_A({**base_feats, "iv_hv_ratio_z": 2.0}, tcfg, THRESHOLDS)
    assert base is not None and boosted is not None
    assert boosted["signal_strength"] > base["signal_strength"]


def test_ticket_renders_iv_hv_note():
    signal = {
        "playbook_id": "C", "playbook_name": "Compression Breakout",
        "signal_strength": 0.4, "suggested_size_pct": 0.5,
        "instrument": "defined_risk_calls",
        "key_levels": {"spot": 500.0, "put_wall": 470.0, "call_wall": 520.0, "flip": 490.0},
        "invalidation_level": 490.0,
        "notes": ["IV/HV elevated (ratio 1.60 > 1.30, <20d history) — options rich vs realized vol. Conviction reduced ×0.5 (paying up for calls)."],
        "gates_passed": {"term_structure_slope": 4.0},
    }
    features = {"ticker": "NVDA", "regime": "deep_positive", "call_wall_secondary": np.nan}
    _, md = s7.build_ticket(signal, features, {"kind": "single_name"}, date(2026, 5, 27), THRESHOLDS)
    assert "## Notes" in md
    assert "IV/HV elevated" in md


# =============================================================================
# Feature 2 — Technical indicators
# =============================================================================

def _ohlcv(n=60, base=100.0, slope=0.5):
    idx = pd.bdate_range(end=pd.Timestamp(date(2026, 5, 27)), periods=n)
    close = base + slope * np.arange(n)
    return pd.DataFrame({
        "Open": close, "High": close + 1.0, "Low": close - 1.0,
        "Close": close, "Volume": [1_000_000.0] * n,
    }, index=idx)


def test_technicals_basic_shape_and_rsi_uptrend():
    out = s55.compute_technicals(_ohlcv(60), date(2026, 5, 27))
    for k in s55.TECHNICAL_KEYS:
        assert k in out and np.isfinite(out[k]), f"{k} should be finite"
    # Pure uptrend → RSI very high, MACD histogram positive, bands ordered.
    assert out["rsi_14"] > 70
    assert out["macd_hist"] > 0
    assert out["bb_upper"] > out["bb_mid"] > out["bb_lower"]


def test_technicals_nan_on_short_history():
    out = s55.compute_technicals(_ohlcv(10), date(2026, 5, 27))
    assert np.isnan(out["macd"])       # needs >=26
    assert np.isnan(out["bb_mid"])     # needs >=20
    assert out["technical_commentary"] == ""   # stub


def test_technicals_empty_history():
    out = s55.compute_technicals(pd.DataFrame(), date(2026, 5, 27))
    for k in s55.TECHNICAL_KEYS:
        assert np.isnan(out[k])


# =============================================================================
# Feature 3 — Fundamentals
# =============================================================================

def _raw_bundle():
    earnings = pd.DataFrame(
        {"epsActual": [1.0, 1.1, 1.2, 1.3], "epsEstimate": [0.9, 1.0, 1.1, 1.2],
         "surprisePercent": [0.11, 0.10, 0.09, 0.08]},
        index=pd.to_datetime(["2025-08-01", "2025-11-01", "2026-02-01", "2026-05-01"]),
    )
    eps_trend = pd.DataFrame(
        {"current": [1.40], "7daysAgo": [1.38], "30daysAgo": [1.30],
         "60daysAgo": [1.25], "90daysAgo": [1.20]},
        index=["0q"],
    )
    recs = pd.DataFrame([
        {"period": "0m", "strongBuy": 10, "buy": 10, "hold": 5, "sell": 0, "strongSell": 0},
    ])
    ud = pd.DataFrame(
        {"Firm": ["A", "B", "C", "D"], "Action": ["up", "up", "down", "main"]},
        index=pd.to_datetime(["2026-05-20", "2026-05-10", "2026-04-15", "2026-03-20"]),
    )
    ttm = pd.DataFrame({"TTM": [200.0, 1000.0]}, index=["Net Income", "Total Revenue"])
    info = {"forwardPE": 25.0, "debtToEquity": 50.0, "revenueGrowth": 0.20,
            "recommendationMean": 1.8, "numberOfAnalystOpinions": 25}
    return {"info": info, "earnings_history": earnings, "eps_trend": eps_trend,
            "recommendations": recs, "upgrades_downgrades": ud, "ttm_income_stmt": ttm}


def test_fundamentals_extraction():
    out = s55.compute_fundamentals(_raw_bundle(), {"kind": "single_name"}, date(2026, 5, 27))
    assert out["forward_pe"] == pytest.approx(25.0)
    assert out["debt_to_equity"] == pytest.approx(50.0)
    assert out["revenue_growth_yoy"] == pytest.approx(0.20)
    assert out["analyst_consensus"] == pytest.approx(1.8)
    assert out["analyst_coverage_count"] == pytest.approx(25)
    # 4 most recent surprises mean
    assert out["eps_surprise_avg_4q"] == pytest.approx(np.mean([0.11, 0.10, 0.09, 0.08]))
    assert out["eps_beat_streak"] == pytest.approx(4.0)         # all positive
    assert out["eps_estimate_trend"] == pytest.approx(1.40 - 1.30)
    assert out["net_income_margin_ttm"] == pytest.approx(0.20)  # 200/1000
    assert out["analyst_pct_buy"] == pytest.approx(20 / 25)     # (10+10)/25
    assert out["analyst_upgrades_3m"] == pytest.approx(2.0)     # within 90d of asof
    assert out["analyst_downgrades_3m"] == pytest.approx(1.0)
    assert out["analyst_steady_3m"] == pytest.approx(1.0)


def test_fundamentals_etf_all_nan():
    out = s55.compute_fundamentals(_raw_bundle(), {"kind": "index_etf"}, date(2026, 5, 27))
    for k in (s55.ANALYST_KEYS + s55.FINANCIAL_KEYS):
        assert np.isnan(out[k])


def test_fundamentals_none_bundle_all_nan():
    out = s55.compute_fundamentals(None, {"kind": "single_name"}, date(2026, 5, 27))
    for k in (s55.ANALYST_KEYS + s55.FINANCIAL_KEYS):
        assert np.isnan(out[k])


def test_fundamentals_cache_append_if_changed(tmp_features_dir):
    payload1 = {"analyst_consensus": 1.8, "analyst_coverage_count": 25.0}
    payload_same = {"analyst_consensus": 1.8, "analyst_coverage_count": 25.0}
    payload2 = {"analyst_consensus": 2.1, "analyst_coverage_count": 26.0}

    assert storage.append_fundamentals_if_changed("NVDA", "analyst", payload1, date(2026, 5, 27)) is True
    # Unchanged → no new snapshot.
    assert storage.append_fundamentals_if_changed("NVDA", "analyst", payload_same, date(2026, 5, 28)) is False
    # Changed → new snapshot.
    assert storage.append_fundamentals_if_changed("NVDA", "analyst", payload2, date(2026, 5, 29)) is True

    latest = storage.read_latest_fundamentals("NVDA", "analyst")
    assert latest["analyst_consensus"] == pytest.approx(2.1)


def test_fundamentals_cache_nan_equality(tmp_features_dir):
    """An all-NaN refetch must not churn the cache (NaN==NaN)."""
    p = {"x": float("nan"), "y": 1.0}
    assert storage.append_fundamentals_if_changed("ZZZ", "financials", p, date(2026, 5, 27)) is True
    assert storage.append_fundamentals_if_changed("ZZZ", "financials", dict(p), date(2026, 5, 28)) is False


def test_ticket_renders_commentary():
    signal = {
        "playbook_id": "A", "playbook_name": "Washout at Put Wall in Negative Gamma",
        "signal_strength": 0.6, "suggested_size_pct": 0.5, "instrument": "shares",
        "key_levels": {"spot": 100.0, "put_wall": 100.0, "call_wall": 110.0, "flip": 95.0},
        "invalidation_level": 98.0, "notes": [], "gates_passed": {"regime": "deep_negative"},
    }
    features = {
        "ticker": "NVDA", "regime": "deep_negative", "call_wall_secondary": np.nan,
        "technical_commentary": "RSI oversold; MACD turning up.",
        "analyst_commentary": "3 upgrades in the last month.",
    }
    _, md = s7.build_ticket(signal, features, {"kind": "single_name"}, date(2026, 5, 27), THRESHOLDS)
    assert "## Commentary" in md
    assert "RSI oversold" in md
    assert "3 upgrades" in md


# =============================================================================
# Integration — Stage 5.5 scalars survive the parquet round-trip
# =============================================================================

def test_stage5_5_scalars_persisted(tmp_features_dir):
    asof = date(2026, 5, 27)
    context = s55.compute_stage5_5(_ohlcv(60), asof, _raw_bundle(), {"kind": "single_name"})
    row = {"ticker": "NVDA", **context}
    s9.log_features([row], asof)
    back = storage.read_features_range(asof, asof)
    r = back[back["ticker"] == "NVDA"].iloc[0]
    assert np.isfinite(r["rsi_14"])
    assert np.isfinite(r["forward_pe"])
    assert np.isfinite(r["analyst_pct_buy"])
    # Commentary stubs are transient — not part of the parquet schema.
    assert "technical_commentary" not in back.columns
