"""EOD orchestrator — wires stages 1 through 9 into one daily run.

Pipeline per the workflow:
    For each watchlist ticker (after the close):
        1. Acquire market data (engine GEX profile + flip + chain, yfinance bars)
        2. Stage 3 — level map
        3. Stage 2 — regime (with history → percentile thresholds + transition)
        4. Stage 4 — flow & positioning
        5. Stage 5 — price/volume confirmation
        6. Stage 1 — calendar event flags
        -> assemble one feature row
    Stage 9 — log all feature rows for the day
    Stage 6 — run playbooks per ticker (uses the row + recent history)
    Stage 7 — build a markdown ticket for each fired signal
    Stage 8 — evaluate open positions, emit alerts
    Stage 9 — backfill forward returns on prior days

Design: `acquire_market_data` is the ONLY function that touches the engine and
the network. Everything else operates on plain dicts and is unit-testable without
the engine. `run_eod` is the top-level entry point called by the GitHub Action.
"""
from __future__ import annotations
from datetime import date as date_cls
import logging
import time

import numpy as np
import pandas as pd

from . import storage
from .config import load_watchlist, load_thresholds, get_ticker_config
from .rates import get_rate
from . import stage1_calendar as cal
from . import stage2_regime as s2
from . import stage3_level_map as s3
from . import stage4_flow as s4
from . import stage5_confirm as s5
from . import stage6_playbooks as s6
from . import stage7_ticket as s7
from . import stage8_monitor as s8
from .stage9_logging import log_features, backfill_forward_returns

logger = logging.getLogger(__name__)


# ---- Acquisition (engine + network) ----------------------------------------

def acquire_market_data(symbol: str, asof: date_cls, r: float) -> dict | None:
    """Fetch everything the feature assembly needs that requires the engine or
    the network. Returns a dict, or None if the ticker can't be processed.

    Keys returned:
        spot, prices (np.ndarray), gex_array (np.ndarray), flip_level,
        chain (pd.DataFrame), prev_chain (pd.DataFrame|None),
        ticker_obj (yfinance Ticker), underlying_history (pd.DataFrame)

    This is the only function that imports the engine. It is intentionally thin
    so a different data source can be slotted in later without touching the
    stages.
    """
    try:
        import yfinance as yf
        from . import engine_shim as eng
    except Exception as e:
        logger.error("Acquisition imports failed for %s: %s", symbol, e)
        return None

    try:
        ticker_obj = yf.Ticker(symbol)
        spot = float(ticker_obj.fast_info["lastPrice"])

        # max_days=90 (vs the engine's 45d default) is deliberate: the term-structure
        # slope (Stage 4) needs a real ~90d expiry in reach for a true "3M − 7d" read,
        # which feeds Playbook C's contango gate. This also widens the GEX/wall basis to
        # ~1 quarter of dealer positioning, which is intended. Do NOT revert to 45.
        chain = eng.build_refined_options_chain(ticker_obj, max_days=90)
        if chain is None or chain.empty:
            logger.warning("%s: no options chain after filtering; skipping.", symbol)
            return None

        # Price grid for the GEX profile — reuse the engine's bounds logic.
        cwp, pwp, fl = eng.calculate_price_bounds(chain, spot)
        lo = min(spot * 0.85, pwp * 0.95, (fl or spot) * 0.95)
        hi = max(spot * 1.15, cwp * 1.05, (fl or spot) * 1.05)
        prices = np.linspace(lo, hi, 200)
        gex_array = eng.calculate_gex(prices, chain)
        flip_level = eng.find_flip_level(prices, gex_array)

        underlying_history = s5.fetch_underlying_history(symbol, asof)

        prev_chain = storage.read_chain_snapshot(symbol, asof)
        # Persist today's snapshot for tomorrow's OI-delta calc.
        storage.write_chain_snapshot(symbol, asof, chain)

        return {
            "spot": spot,
            "prices": prices,
            "gex_array": gex_array,
            "flip_level": flip_level,
            "chain": chain,
            "prev_chain": prev_chain,
            "ticker_obj": ticker_obj,
            "underlying_history": underlying_history,
        }
    except Exception as e:
        logger.error("Acquisition failed for %s: %s", symbol, e)
        return None


# ---- Pure feature assembly --------------------------------------------------

def assemble_feature_row(
    symbol: str,
    asof: date_cls,
    market: dict,
    ticker_config: dict,
    r: float,
    thresholds: dict,
) -> dict:
    """Run stages 1-5 on acquired market data and return one feature row dict.

    Pure with respect to the engine/network — `market` carries everything fetched
    in acquire_market_data. Stages 2/3/4 still read the feature store for history.
    """
    spot = market["spot"]
    prices = market["prices"]
    gex_array = market["gex_array"]
    flip_level = market["flip_level"]
    chain = market["chain"]

    gex_total = float(np.interp(spot, prices, gex_array))

    # Stage 3 — levels (also gives migration/persistence from history)
    levels = s3.build_level_map(
        symbol, asof, prices, gex_array, spot, flip_level,
        options_chain=chain, thresholds=thresholds,
    )

    # Stage 2 — regime + transition
    regime = s2.classify_with_history(
        gex_total=gex_total, spot=spot, flip_level=flip_level,
        ticker=symbol, asof=asof, mcap_b=ticker_config.get("market_cap_b"),
        thresholds=thresholds,
    )

    # Stage 4 — flow
    flow = s4.compute_flow(
        symbol=symbol, asof=asof, chain=chain, spot=spot, r=r,
        ticker_obj=market.get("ticker_obj"), prev_chain=market.get("prev_chain"),
    )

    # Stage 5 — confirmation
    confirm = s5.compute_confirmation(symbol, asof, history=market.get("underlying_history"))

    # Stage 1 — calendar
    flags = cal.event_flags(symbol, asof)

    row = {
        "ticker": symbol,
        "asof": asof,
        "spot": spot,
        "gex_total": gex_total,
        "regime": regime["regime"],
        "prev_regime": regime.get("prev_regime"),
        "regime_transition": regime.get("regime_transition"),
        "distance_to_flip_pct": regime["distance_to_flip_pct"],
        **levels,        # call_wall, put_wall, secondaries, hvl, migration, persistence, flip_level
        **flow,          # pcr, z-scores, skew, term structure, oi deltas, new_oi_count
        **confirm,       # volume_z, atr, ma distances, vix
        **flags,         # days_to_*, within_event_window, etc.
    }
    return row


# ---- Top-level orchestration ------------------------------------------------

def run_eod(asof: date_cls | None = None, symbols: list[str] | None = None) -> dict:
    """Run the full EOD pipeline. Returns a summary dict.

    asof defaults to today. symbols defaults to the full watchlist.
    """
    asof = asof or date_cls.today()
    watchlist = load_watchlist()
    thresholds = load_thresholds()
    r = get_rate(asof)

    if symbols is None:
        symbols = [t["symbol"] for t in watchlist["tickers"]]

    throttle_sec = float(thresholds.get("acquisition", {}).get("inter_ticker_sleep_sec", 2.0))
    logger.info("EOD run %s — %d tickers, r=%.4f, throttle=%.1fs", asof, len(symbols), r, throttle_sec)

    rows: list[dict] = []
    for i, symbol in enumerate(symbols):
        if i > 0 and throttle_sec > 0:
            time.sleep(throttle_sec)   # be polite to yfinance / Yahoo
        try:
            tcfg = get_ticker_config(symbol, watchlist)
        except KeyError:
            logger.warning("%s not in watchlist; skipping.", symbol)
            continue
        market = acquire_market_data(symbol, asof, r)
        if market is None:
            continue
        try:
            row = assemble_feature_row(symbol, asof, market, tcfg, r, thresholds)
            rows.append(row)
        except Exception as e:
            logger.error("Feature assembly failed for %s: %s", symbol, e)

    # Stage 9 — log all rows for the day (must happen before signal eval so the
    # day's row is part of any subsequent same-day history reads, though signals
    # below use the in-memory rows directly).
    if rows:
        log_features(rows, asof)

    # Stage 6 + 7 — signals and tickets
    signals_out: list[dict] = []
    for row in rows:
        symbol = row["ticker"]
        tcfg = get_ticker_config(symbol, watchlist)
        history_df = _ticker_history(symbol, asof)
        fired = s6.evaluate_all(row, tcfg, history_df=history_df, thresholds=thresholds)
        for sig in fired:
            ticket, md = s7.build_ticket(sig, row, tcfg, asof, thresholds)
            storage.write_ticket(asof, symbol, sig["playbook_id"], md)
            signals_out.append({
                "ticker": symbol,
                "asof": asof,
                "playbook_id": sig["playbook_id"],
                "signal_strength": sig["signal_strength"],
                "entry": ticket["entry"],
                "stop": ticket["stop"],
                "target_1": ticket["target_1"],
                "target_2": ticket["target_2"],
                "shares": ticket["shares"],
            })

    if signals_out:
        storage.write_signals(pd.DataFrame(signals_out), asof)

    # Stage 8 — monitor open positions
    alerts = _run_monitor(rows, asof, thresholds)

    # Stage 9 — forward-return backfill on prior days
    try:
        backfill_forward_returns(asof)
    except Exception as e:
        logger.warning("Forward-return backfill failed: %s", e)

    summary = {
        "asof": asof.isoformat(),
        "tickers_processed": len(rows),
        "signals_fired": len(signals_out),
        "alerts": len(alerts),
    }
    logger.info("EOD summary: %s", summary)
    return summary


def _ticker_history(symbol: str, asof: date_cls, days: int = 30) -> pd.DataFrame:
    from datetime import timedelta
    start = asof - timedelta(days=days)
    hist = storage.read_features_range(start, asof - timedelta(days=1))
    if hist.empty:
        return hist
    return hist[hist["ticker"] == symbol].sort_values("date")


def _run_monitor(rows: list[dict], asof: date_cls, thresholds: dict) -> list[dict]:
    """Evaluate open positions against today's feature rows; write alert files."""
    positions = storage.read_positions()
    if not positions:
        return []
    by_ticker = {row["ticker"]: row for row in rows}
    alerts_out = []
    for pos in positions:
        feats = by_ticker.get(pos.get("ticker"))
        if feats is None:
            continue
        actions = s8.evaluate_position(pos, feats, thresholds)
        if actions:
            md = s8.render_alert(pos["ticker"], actions)
            storage.write_ticket(asof, pos["ticker"], "ALERT", md)
            alerts_out.append({"ticker": pos["ticker"], "actions": actions})
    return alerts_out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_eod()
