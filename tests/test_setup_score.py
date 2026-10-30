"""Tests for the setup-oriented Stage 5.5 technical score (no network)."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from gamma_workflow import stage5_5_context as s55
from scripts.build_trade_log import parse_signal_text

NAN = float("nan")


def _score(**kw):
    args = dict(
        close_cur=100.0, sma20=100.0, sma50=100.0, atr20=2.0, rsi_val=50.0,
        bb_upper=110.0, cmf_val=0.0, trend_obv="NEUTRAL", trend_macd="NEUTRAL",
        macd_val=0.0, momentum_trend_score=0.0, momentum_conv_mult=1.0, div_raw=0.0,
    )
    args.update(kw)
    return s55._score_technical(**args)


def test_pullback_scores_as_buy():
    # 2.5 ATR below the 20-day average and RSI 30: a classic pullback.
    sc = _score(close_cur=95.0, rsi_val=30.0)
    assert sc["pull_ma"] == 1.875
    assert sc["pull_rsi"] == 2.0
    assert sc["overext"] == 0.0
    assert sc["rec"] == "BUY"


def test_overextended_stock_is_penalised_to_strong_sell():
    sc = _score(close_cur=130.0, sma20=110.0, atr20=4.0, sma50=100.0,
                rsi_val=82.0, bb_upper=125.0)
    # upper band, RSI ≥ 75, >15% and >25% above the 50-day, >3 ATR above the 20-day
    assert len(sc["penalties"]) == 5
    assert sc["overext"] == -5.0
    assert sc["pull_ma"] == -3.0 and sc["pull_rsi"] == -2.0
    assert sc["score"] == -10.0
    assert sc["rec"] == "STRONG_SELL"


def test_momentum_is_capped_at_one_point():
    assert _score(momentum_trend_score=11.0, momentum_conv_mult=1.25)["momentum"] == 1.0
    assert _score(momentum_trend_score=-20.0)["momentum"] == -1.0
    # A maxed-out uptrend alone can no longer produce a Buy.
    assert _score(momentum_trend_score=11.0, momentum_conv_mult=1.25)["rec"] == "HOLD"


def test_accumulation_only_counts_on_a_pullback():
    pulled = _score(close_cur=98.0, cmf_val=0.2, trend_obv="BULLISH")
    extended = _score(close_cur=102.0, cmf_val=0.2, trend_obv="BULLISH")
    assert pulled["confirm"] == 1.0
    assert extended["confirm"] == 0.0


def test_macd_turn_only_counts_below_zero():
    assert _score(trend_macd="BULL_CROSS", macd_val=-0.5)["confirm"] == 0.5
    assert _score(trend_macd="BULL_CROSS", macd_val=0.5)["confirm"] == 0.0


def test_divergence_capped_at_one_point():
    assert _score(div_raw=6.0)["div_pts"] == 1.0
    assert _score(div_raw=-2.0)["div_pts"] == -0.5


def test_missing_inputs_contribute_nothing():
    sc = _score(close_cur=100.0, sma20=NAN, sma50=NAN, atr20=NAN, rsi_val=NAN,
                bb_upper=NAN, cmf_val=NAN)
    assert sc["score"] == 0.0 and sc["rec"] == "HOLD"


def _series(closes):
    closes = np.asarray(closes, dtype=float)
    idx = pd.bdate_range(end=pd.Timestamp(date(2026, 5, 27)), periods=len(closes))
    return pd.DataFrame({
        "Open": closes, "High": closes * 1.01, "Low": closes * 0.99,
        "Close": closes, "Volume": np.full(len(closes), 1_000_000.0),
    }, index=idx)


def test_steady_rally_is_not_a_buy():
    # The old momentum score rated a relentless rally STRONG_BUY; now it's stretched.
    ev = s55._evaluate_technical(_series(100 * 1.004 ** np.arange(260)), date(2026, 5, 27))
    assert ev["scoring"]["score"] < 2.5


def test_pullback_after_rally_scores_positive():
    up = 100 * 1.004 ** np.arange(240)
    down = up[-1] * 0.99 ** np.arange(1, 21)
    ev = s55._evaluate_technical(_series(np.r_[up, down]), date(2026, 5, 27))
    assert ev["scoring"]["pullback"] > 2.0
    assert ev["scoring"]["score"] > 0


def test_stage5_5_score_matches_commentary_and_parses():
    h = _series(np.r_[100 * 1.004 ** np.arange(240), 260 * 0.99 ** np.arange(1, 21)])
    out = s55.compute_stage5_5(h, date(2026, 5, 27), None, {"kind": "index_etf"}, symbol="")
    text = out["technical_commentary"]
    assert "**Pullback setup**" in text and "**Overextension penalties**" in text
    assert f"→ **{out['tech_score']:+.2f}**" in text

    # The trade-log parser reads the new layout (it is written to the ticket as-is).
    ticket = (f"# Technical Signal: TST  2026-05-27\n\n**Recommendation**: "
              f"{out['tech_recommendation']}  |  **Score**: {out['tech_score']:+.1f}\n\n{text}")
    rec = parse_signal_text(ticket, ticker="TST", date="2026-05-27", watchlist="up")
    assert np.isclose(rec["pullback"], float(text.split("**Pullback setup** (")[1].split(")")[0]))
    assert np.isfinite(rec["momentum"]) and np.isfinite(rec["overextension"])
    assert np.isnan(rec["mr_gate"])
