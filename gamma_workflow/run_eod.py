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
from . import stage5_5_context as s55
from . import stage6_playbooks as s6
from . import stage7_ticket as s7
from . import stage8_monitor as s8
from .stage9_logging import log_features, backfill_forward_returns

logger = logging.getLogger(__name__)


# ---- Acquisition (engine + network) ----------------------------------------

def _memoize_option_chain(ticker_obj):
    """Cache `ticker_obj.option_chain(expiry)` results on the instance.

    In one EOD pass the same expiry chains are downloaded several times: once by
    build_refined_options_chain (every expiry ≤90d) and again by Stage 4's three
    calculate_iv_index legs (30d / 7d / 90d), each of which re-fetches its
    bracketing expiries. Memoizing per expiry collapses those to a single network
    download per expiry. The cached object is the very same chain, so every
    downstream computed value is identical — this only removes redundant I/O.
    """
    try:
        original = ticker_obj.option_chain
    except Exception:
        return ticker_obj
    cache: dict = {}

    def cached_option_chain(date=None):
        if date not in cache:
            cache[date] = original(date)
        return cache[date]

    try:
        ticker_obj.option_chain = cached_option_chain
    except Exception:
        # Some objects disallow attribute assignment; fall back to uncached.
        pass
    return ticker_obj


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
        ticker_obj = _memoize_option_chain(yf.Ticker(symbol))
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
        # Thread the run's FRED rate through — leaving it off silently used the
        # engine's hardcoded 0.045 default while Stage 4 priced off the live rate.
        # (calculate_price_bounds above only sizes the grid, so its internal
        # default-rate GEX pass is immaterial.)
        gex_array = eng.calculate_gex(prices, chain, r=r)
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
    prev_tech_rec: str | None = None,
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

    # Stage 5 — confirmation (runs before Stage 4 so hv_30d feeds the IV/HV ratio).
    # vix_history is fetched once per run and shared across tickers (VIX is market-wide).
    confirm = s5.compute_confirmation(
        symbol, asof, history=market.get("underlying_history"),
        vix_history=market.get("vix_history"),
    )

    # Stage 4 — flow (hv_30d from Stage 5 lets the IV/HV ratio + z be formed here,
    # off the single history read compute_flow already performs)
    flow = s4.compute_flow(
        symbol=symbol, asof=asof, chain=chain, spot=spot, r=r,
        ticker_obj=market.get("ticker_obj"), prev_chain=market.get("prev_chain"),
        hv_30d=confirm.get("hv_30d", float("nan")),
    )

    # Stage 5.5 — technical + fundamental context (Features 2 & 3). Scalars are
    # logged; the commentary strings ride along in-memory for the ticket.
    context = s55.compute_stage5_5(
        history=market.get("underlying_history"), asof=asof,
        raw_bundle=market.get("fundamentals_raw"), ticker_config=ticker_config,
        symbol=symbol,
        ivx_30d=flow.get("ivx_30d", float("nan")),
        prev_tech_rec=prev_tech_rec,
        cached_fundamentals=market.get("cached_fundamentals"),
    )

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
        **flow,          # pcr, z-scores, skew, term structure, oi deltas, new_oi_count, iv/hv
        **confirm,       # volume_z, atr, ma distances, vix, hv_30d
        **context,       # technicals + fundamentals scalars (+ stub commentary)
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

    # VIX is market-wide and identical for every ticker, so fetch it once here
    # instead of re-pulling it inside each ticker's Stage 5 confirmation.
    vix_history = s5.fetch_underlying_history("^VIX", asof, days=20)

    # Fundamentals network fetch is gated by a freshness TTL: pulling 6 yfinance
    # endpoints per ticker daily is wasteful for data that moves ~quarterly. We
    # refetch only after `refresh_days` have elapsed since the last fetch and reuse
    # the cached scalar set in between.
    fund_cfg = thresholds.get("fundamentals", {})
    fund_enabled = bool(fund_cfg.get("enabled", True))
    fund_refresh_days = int(fund_cfg.get("refresh_days", 7))

    # Transition sets for watchlist routing
    _UPWARD_TRANSITIONS = {
        ("HOLD", "BUY"), ("HOLD", "STRONG_BUY"), ("BUY", "STRONG_BUY"),
    }
    _DOWNWARD_TRANSITIONS = {
        ("BUY", "SELL"), ("STRONG_BUY", "SELL"),
        ("STRONG_BUY", "STRONG_SELL"), ("SELL", "STRONG_SELL"),
    }

    from datetime import timedelta as _td
    _yesterday = asof - _td(days=1)

    rows: list[dict] = []
    techsignal_records: list[dict] = []  # accumulates tickets for the growing parquet log
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
        market["vix_history"] = vix_history

        # Stage 5.5 fundamentals fetch (network) — gated by config and skipped for
        # index ETFs, which have no meaningful fundamentals. Companion to
        # acquire_market_data so it can be turned off independently.
        #
        # Freshness gate: only hit the network when the per-ticker snapshot is
        # missing or older than `fund_refresh_days`. On a skip day we reuse the
        # cached scalar set so the logged row is unchanged. `fund_fetched` records
        # whether we actually fetched, so we know to refresh the snapshot below.
        fundamentals_applicable = fund_enabled and tcfg.get("kind") != "index_etf"
        market["fundamentals_raw"] = None
        market["cached_fundamentals"] = None
        fund_fetched = False
        if fundamentals_applicable:
            snap = storage.read_fundamentals_snapshot(symbol)
            snap_fresh = False
            if snap and snap.get("asof"):
                try:
                    age = (asof - date_cls.fromisoformat(snap["asof"])).days
                    snap_fresh = 0 <= age < fund_refresh_days
                except (ValueError, TypeError):
                    snap_fresh = False
            if snap_fresh:
                market["cached_fundamentals"] = snap.get("scalars")
                logger.info("%s: fundamentals snapshot fresh; skipping network fetch.", symbol)
            else:
                market["fundamentals_raw"] = s55.acquire_fundamentals(symbol, market.get("ticker_obj"))
                fund_fetched = True

        # Read yesterday's tech_recommendation for hysteresis
        prev_tech_rec: str | None = None
        try:
            prev_feats = storage.read_features(_yesterday)
            if prev_feats is not None and not prev_feats.empty and "ticker" in prev_feats.columns:
                ticker_row = prev_feats[prev_feats["ticker"] == symbol]
                if not ticker_row.empty and "tech_recommendation" in ticker_row.columns:
                    val = ticker_row["tech_recommendation"].iloc[0]
                    if isinstance(val, str) and val:
                        prev_tech_rec = val
        except Exception:
            pass

        try:
            row = assemble_feature_row(symbol, asof, market, tcfg, r, thresholds, prev_tech_rec)
            rows.append(row)
            # Persist the fundamentals/analyst evolution history (only on change).
            if fundamentals_applicable:
                try:
                    analyst_payload, fin_payload = s55.split_for_cache(row)
                    storage.append_fundamentals_if_changed(symbol, "analyst", analyst_payload, asof)
                    storage.append_fundamentals_if_changed(symbol, "financials", fin_payload, asof)
                    # Refresh the freshness snapshot only when we actually fetched
                    # this run and got at least one real value — so a failed fetch
                    # (all-NaN) retries tomorrow rather than caching the blank.
                    if fund_fetched:
                        scalars = {**analyst_payload, **fin_payload}
                        if any(isinstance(v, float) and np.isfinite(v) for v in scalars.values()):
                            storage.write_fundamentals_snapshot(symbol, asof, scalars)
                except Exception as e:
                    logger.warning("Fundamentals cache update failed for %s: %s", symbol, e)

            # Write Stage 5.5 tech-signal ticket
            try:
                commentary = row.get("technical_commentary", "")
                tech_rec = row.get("tech_recommendation", "HOLD")
                tech_sc  = row.get("tech_score", float("nan"))
                ticket_header = (
                    f"# Technical Signal: {symbol}  {asof.isoformat()}\n\n"
                    f"**Recommendation**: {tech_rec}  |  **Score**: {tech_sc:+.1f}\n\n"
                )
                ticket_body = ticket_header + (commentary or "(no commentary — insufficient history)")
                storage.write_techsignal_ticket(symbol, asof, ticket_body)

                # Watchlist routing based on recommendation transition
                transition = (str(prev_tech_rec or "HOLD"), str(tech_rec))
                watchlist_dir: str | None = None
                if transition in _UPWARD_TRANSITIONS:
                    watchlist_dir = "upward"
                elif transition in _DOWNWARD_TRANSITIONS:
                    watchlist_dir = "downward"
                if watchlist_dir is not None:
                    storage.write_watchlist_ticket(symbol, asof, ticket_body, watchlist_dir)

                # Accumulate this ticket into the growing long-term parquet log.
                techsignal_records.append({
                    "date": asof.isoformat(),
                    "ticker": symbol,
                    "recommendation": str(tech_rec),
                    "score": float(tech_sc) if tech_sc == tech_sc else float("nan"),
                    "watchlist": watchlist_dir,
                    "content": ticket_body,
                })
            except Exception as e:
                logger.warning("Tech-signal ticket write failed for %s: %s", symbol, e)

        except Exception as e:
            logger.error("Feature assembly failed for %s: %s", symbol, e)

    # Stage 9 — log all rows for the day (must happen before signal eval so the
    # day's row is part of any subsequent same-day history reads, though signals
    # below use the in-memory rows directly).
    if rows:
        log_features(rows, asof)

    # Append every tech-signal ticket written this run to the growing parquet log
    # (data/techsignal_log.parquet) for long-term assessment. Idempotent per date.
    if techsignal_records:
        try:
            storage.append_techsignal_log(techsignal_records, asof)
        except Exception as e:
            logger.warning("Tech-signal log append failed: %s", e)

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

    # Stage 9 — forward-return + benchmark + alpha backfill on prior days
    try:
        benchmark = thresholds.get("backfill", {}).get("benchmark_ticker", "SPY")
        backfill_forward_returns(asof, benchmark_ticker=benchmark)
    except Exception as e:
        logger.warning("Forward-return backfill failed: %s", e)

    summary = {
        "asof": asof.isoformat(),
        "tickers_processed": len(rows),
        "signals_fired": len(signals_out),
        "alerts": len(alerts),
        "techsignals_logged": len(techsignal_records),
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
