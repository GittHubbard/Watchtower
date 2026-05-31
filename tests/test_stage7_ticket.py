"""Stage 7 — sizing and ticket rendering tests."""
from __future__ import annotations
from datetime import date

import numpy as np
import pytest

from gamma_workflow import stage7_ticket as s7
from gamma_workflow.config import load_thresholds


THRESHOLDS = load_thresholds()
TICKER_CFG = {"symbol": "SPY", "kind": "index_etf", "market_cap_b": 500, "risk_per_trade_bps": 50}


def _signal():
    return {
        "playbook_id": "A",
        "playbook_name": "Washout at Put Wall in Negative Gamma",
        "signal_strength": 0.75,
        "instrument": "shares",
        "suggested_size_pct": 0.50,
        "invalidation_level": 495.0,
        "key_levels": {"spot": 500.0, "put_wall": 495.0, "call_wall": 520.0, "flip": 510.0},
        "gates_passed": {"regime": "deep_negative", "pcr_volume_z": 2.0},
    }


def test_sizing_basic_math():
    feats = {"ticker": "SPY", "call_wall_secondary": 530.0}
    sizing = s7.compute_sizing(_signal(), feats, TICKER_CFG, THRESHOLDS)

    # NAV 100M * 50bps = 500k risk capital
    assert sizing["risk_capital"] == pytest.approx(500_000.0)
    # risk/share = 500 - 495 = 5
    assert sizing["risk_per_share"] == pytest.approx(5.0)
    # base shares = 500k / 5 = 100k; * 50% tranche = 50k
    assert sizing["base_shares"] == pytest.approx(100_000.0)
    assert sizing["shares"] == pytest.approx(50_000.0)


def test_target_1_is_nearest_call_wall_above_entry():
    feats = {"ticker": "SPY", "call_wall_secondary": 530.0}
    sizing = s7.compute_sizing(_signal(), feats, TICKER_CFG, THRESHOLDS)
    # call wall 520 is above entry 500 → target_1 = 520
    assert sizing["target_1"] == pytest.approx(520.0)
    # secondary 530 > target_1 → target_2 = 530
    assert sizing["target_2"] == pytest.approx(530.0)


def test_target_2_falls_back_to_2R_when_no_secondary():
    feats = {"ticker": "SPY", "call_wall_secondary": float("nan")}
    sizing = s7.compute_sizing(_signal(), feats, TICKER_CFG, THRESHOLDS)
    # 2R = entry + 2*risk = 500 + 10 = 510
    assert sizing["target_2"] == pytest.approx(510.0)


def test_invalid_stop_yields_nan_shares():
    sig = _signal()
    sig["invalidation_level"] = 505.0  # stop ABOVE entry → invalid for a long
    feats = {"ticker": "SPY"}
    sizing = s7.compute_sizing(sig, feats, TICKER_CFG, THRESHOLDS)
    assert np.isnan(sizing["shares"])


def test_build_ticket_returns_dict_and_markdown():
    feats = {"ticker": "SPY", "regime": "deep_negative", "call_wall_secondary": 530.0,
             "regime_transition": None}
    ticket, md = s7.build_ticket(_signal(), feats, TICKER_CFG, date(2026, 5, 27), THRESHOLDS)
    assert ticket["ticker"] == "SPY"
    assert ticket["playbook_id"] == "A"
    assert isinstance(md, str)
    assert "Trade Ticket" in md
    assert "SPY" in md
    assert "Washout at Put Wall" in md
    # Should clearly state it's for human review
    assert "human review" in md.lower()


def test_markdown_handles_defined_risk_instrument():
    sig = _signal()
    sig["instrument"] = "defined_risk_calls"
    feats = {"ticker": "SPY", "regime": "deep_positive", "call_wall_secondary": float("nan")}
    _, md = s7.build_ticket(sig, feats, TICKER_CFG, date(2026, 5, 27), THRESHOLDS)
    assert "Defined-risk" in md or "defined-risk" in md.lower()


def test_fmt_handles_nan_and_none():
    assert s7._fmt(float("nan")) == "—"
    assert s7._fmt(None) == "—"
    assert s7._fmt(0.05, pct=True) == "5.00%"
    assert s7._fmt(1_000_000, money=True) == "$1,000,000"
