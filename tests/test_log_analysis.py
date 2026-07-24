"""Tests for parsing the tech-signal parquet log into a feature frame (no network)."""
from __future__ import annotations

import pandas as pd

from scripts.build_trade_log import build_trade_log_from_log

_TICKET = """# Technical Signal: TST  2026-06-04

**Recommendation**: BUY  |  **Score**: +4.2

### Technical Signal: BUY (+4.2 / confidence 42%)

**Regime**: ADX RANGING (ADX 15, +DI 20 vs -DI 18)  |  Vol: HIGH_IV (IV30 30.0%, RV20 25.0%)
**Gate**: MR weight = 1.00

**Trend** (+3.12 = +2.50 × 1.25 convergence)
- Price: NEUTRAL  MACD: BULLISH  CMF: ACCUM
- OBV: BULLISH  aVWAP: ABOVE_BOTH  Volume: NORMAL

**Mean Reversion** (gate 1.00 → contribution +0.00)
- RSI 58: neutral  Bollinger: neutral

**Divergence** (+1.0)
- ✓ CONFIRMED hid_bull on OBV (quality: standard)

**Score**: +3.12 + +0.00 + +1.00 = +4.12 → **+4.12**
"""


def test_build_trade_log_from_log_parses_content_and_maps_watchlist():
    log = pd.DataFrame([
        {"date": "2026-06-04", "ticker": "TST", "recommendation": "BUY",
         "score": 4.2, "watchlist": "upward", "content": _TICKET},
        {"date": "2026-06-04", "ticker": "HLD", "recommendation": "HOLD",
         "score": 0.0, "watchlist": None, "content": _TICKET.replace("TST", "HLD")},
    ])
    frame = build_trade_log_from_log(log)

    assert build_trade_log_from_log.skipped == 0
    assert len(frame) == 2
    assert set(frame["watchlist"]) == {"up", "none"}  # upward->up, None->none

    row = frame[frame["ticker"] == "TST"].iloc[0]
    assert row["recommendation"] == "BUY"
    assert row["macd"] == "BULLISH" and row["cmf"] == "ACCUM"
    assert row["regime"] == "RANGING" and row["vol_regime"] == "HIGH_IV"
    assert row["iv30"] == 30.0 and row["rv20"] == 25.0
    assert row["div_hid_bull"] == 1 and row["div_confirmed"] == 1
    assert abs(row["score_raw"] - 4.12) < 1e-9
    assert row["confidence_pct"] == 42
    assert pd.api.types.is_datetime64_any_dtype(frame["date"])


def test_build_trade_log_from_log_skips_unparseable_rows():
    log = pd.DataFrame([
        {"date": "2026-06-04", "ticker": "TST", "recommendation": "BUY",
         "score": 4.2, "watchlist": "upward", "content": _TICKET},
        {"date": "2026-06-04", "ticker": "BAD", "recommendation": "BUY",
         "score": 1.0, "watchlist": None, "content": "# not a real ticket\n"},
    ])
    frame = build_trade_log_from_log(log)
    assert len(frame) == 1
    assert build_trade_log_from_log.skipped == 1
