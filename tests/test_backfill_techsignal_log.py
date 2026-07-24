"""Tests for the Mode-A tech-signal log backfill (no network)."""
from __future__ import annotations
from datetime import date

import pytest

from gamma_workflow import storage
from scripts import backfill_techsignal_log as bf


def _write_ticket(chains_dir, ticker, d, rec, score):
    tdir = chains_dir / f"ticker={ticker}" / "tickets"
    tdir.mkdir(parents=True, exist_ok=True)
    body = (
        f"# Technical Signal: {ticker}  {d.isoformat()}\n\n"
        f"**Recommendation**: {rec}  |  **Score**: {score:+.1f}\n\n"
        f"**Regime**: ADX RANGING (ADX 15, +DI 20 vs -DI 18)  |  Vol: HIGH_IV (IV30 30.0%, RV20 25.0%)\n"
    )
    (tdir / f"{d.isoformat()}_{ticker}_techsignal.md").write_text(body)
    return body


@pytest.fixture
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DATA_DIR", tmp_path)
    monkeypatch.setattr(storage, "CHAINS_DIR", tmp_path / "chains")
    monkeypatch.setattr(storage, "UPWARD_WATCHLIST_DIR", tmp_path / "Upward-Watchlist")
    monkeypatch.setattr(storage, "DOWNWARD_WATCHLIST_DIR", tmp_path / "Downward-Watchlist")
    return tmp_path


def test_backfill_gathers_filters_and_routes(_isolate):
    tmp = _isolate
    chains = tmp / "chains"
    d0, d1, d2 = date(2026, 6, 3), date(2026, 6, 4), date(2026, 6, 5)

    # d0 is before --since and must be excluded.
    _write_ticket(chains, "SPY", d0, "BUY", 3.0)
    spy_body = _write_ticket(chains, "SPY", d1, "STRONG_BUY", 8.2)
    _write_ticket(chains, "QQQ", d1, "HOLD", 0.0)
    _write_ticket(chains, "IWM", d2, "STRONG_SELL", -7.0)

    # SPY on d1 was routed upward; IWM on d2 routed downward.
    up = tmp / "Upward-Watchlist" / d1.isoformat()
    up.mkdir(parents=True)
    (up / f"{d1.isoformat()}_SPY_techsignal.md").write_text(spy_body)
    down = tmp / "Downward-Watchlist" / d2.isoformat()
    down.mkdir(parents=True)
    (down / f"{d2.isoformat()}_IWM_techsignal.md").write_text("stub")

    summary = bf.backfill(since=d1)
    assert summary["rows"] == 3                       # d0 excluded
    assert summary["date_range"] == (d1.isoformat(), d2.isoformat())
    assert summary["recommendations"] == {"HOLD": 1, "STRONG_BUY": 1, "STRONG_SELL": 1}

    log = storage.read_techsignal_log()
    assert len(log) == 3
    assert set(log["date"]) == {d1.isoformat(), d2.isoformat()}
    # Routing recovered from the watchlist folders.
    assert log[log.ticker == "SPY"]["watchlist"].iloc[0] == "upward"
    assert log[log.ticker == "IWM"]["watchlist"].iloc[0] == "downward"
    assert log[log.ticker == "QQQ"]["watchlist"].isna().all()
    # Full ticket markdown preserved verbatim.
    assert log[log.ticker == "SPY"]["content"].iloc[0] == spy_body


def test_backfill_is_idempotent(_isolate):
    tmp = _isolate
    chains = tmp / "chains"
    d1 = date(2026, 6, 4)
    _write_ticket(chains, "SPY", d1, "BUY", 4.2)
    _write_ticket(chains, "QQQ", d1, "SELL", -3.1)

    bf.backfill(since=d1)
    bf.backfill(since=d1)  # second run must not duplicate
    log = storage.read_techsignal_log()
    assert len(log) == 2
    assert log.duplicated(subset=["date", "ticker"]).sum() == 0


def test_dry_run_writes_nothing(_isolate):
    tmp = _isolate
    _write_ticket(tmp / "chains", "SPY", date(2026, 6, 4), "BUY", 4.2)
    summary = bf.backfill(since=date(2026, 6, 4), dry_run=True)
    assert summary["rows"] == 1 and summary["dry_run"] is True
    assert not storage.techsignal_log_path().exists()
