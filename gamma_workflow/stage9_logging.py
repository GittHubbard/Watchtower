"""Stage 9 — Logging & backtest substrate.

Built first because the workflow is explicit: "If we can't audit it, we can't trust it."

This module owns two things:
    1. Feature-vector logging (one row per ticker per day) via `log_features`.
    2. Forward-returns backfill — for every prior date in storage, look up actual
       1d/5d/20d forward returns on the underlying and merge them into that day's
       features parquet. Idempotent.

Forward returns are computed from yfinance daily closes. Run nightly via
`backfill_forward_returns()` after the EOD pipeline writes today's features.

Schema for the features parquet (one row per ticker per date):
    ticker, asof, spot,
    regime, prev_regime, regime_transition, gex_total, distance_to_flip_pct, flip_level,
    call_wall, call_wall_gex, put_wall, put_wall_gex,
    call_wall_dod_change, put_wall_dod_change,
    skew_z, pcr,
    volume_z, atr_pct, vix_close, vix_3d_change,
    days_to_earnings, days_to_macro, within_event_window,
    fwd_ret_1d, fwd_ret_5d, fwd_ret_20d           (backfilled later — null on write)

Stages 2–5 will populate their slice of these columns; this module just persists
whatever dict it is handed and reconciles columns into one frame per date.
"""
from __future__ import annotations
from datetime import date as date_cls, timedelta
import logging

import numpy as np
import pandas as pd
import yfinance as yf

from .storage import read_features, write_features

logger = logging.getLogger(__name__)


# Canonical column order. Anything missing from a row gets NaN.
FEATURE_COLUMNS = [
    "ticker", "asof", "spot",
    # Stage 2 — regime
    "regime", "prev_regime", "regime_transition",
    "gex_total", "distance_to_flip_pct", "flip_level",
    # Stage 3 — level map
    "call_wall", "call_wall_gex", "put_wall", "put_wall_gex",
    "call_wall_secondary", "put_wall_secondary",
    "call_wall_dod_change", "put_wall_dod_change",
    "wall_persistence_days_call", "wall_persistence_days_put",
    "hvl",
    # Stage 4 — flow & positioning
    "pcr_volume", "pcr_volume_z", "pcr_oi", "pcr_oi_z",
    "total_volume", "total_volume_z",
    "skew_25d", "skew_25d_z",
    "term_structure_slope",
    "new_oi_strike_count", "oi_delta_call", "oi_delta_put",
    # Stage 4 — IV/HV (Feature 1). ivx_30d in percent, hv_30d decimal, ratio dimensionless.
    "ivx_30d", "iv_hv_ratio_30d", "iv_hv_ratio_z",
    # Stage 5 — confirmation
    "volume_z", "atr_pct", "atr_20d_pct", "range_expansion", "close_location",
    "hv_30d",
    "ma20_dist_pct", "ma50_dist_pct", "ma200_dist_pct",
    "vix_close", "vix_5d_change_pct",
    # Stage 5.5 — technical indicators (Feature 2)
    "macd", "macd_signal", "macd_hist", "rsi_14", "cmf_20",
    "bb_upper", "bb_mid", "bb_lower", "bb_width",
    # Stage 5.5 — technical signal output (Feature 2 summary)
    "tech_recommendation", "tech_score",
    # Stage 5.5 — fundamentals (Feature 3)
    "forward_pe", "eps_surprise_avg_4q", "eps_beat_streak", "eps_estimate_trend",
    "net_income_margin_ttm", "debt_to_equity", "revenue_growth_yoy",
    "analyst_consensus", "analyst_coverage_count", "analyst_pct_buy",
    "analyst_upgrades_3m", "analyst_downgrades_3m", "analyst_steady_3m",
    # Stage 1 — calendar
    "days_to_earnings", "days_to_macro", "next_macro", "within_event_window",
    "days_to_opex", "days_to_vixp", "is_opex_day", "is_opex_minus_1", "is_vixp_day",
    # Stage 9 — backfilled forward returns, benchmark returns, and alpha (Feature 4)
    "fwd_ret_1d", "fwd_ret_5d", "fwd_ret_20d", "fwd_ret_60d", "fwd_ret_90d",
    "benchmark_ret_1d", "benchmark_ret_5d", "benchmark_ret_20d",
    "benchmark_ret_60d", "benchmark_ret_90d",
    "alpha_1d", "alpha_5d", "alpha_20d", "alpha_60d", "alpha_90d",
]

# Forward-return horizons (trading days). Benchmark + alpha are computed on the
# same horizons. Adding/removing a horizon here automatically extends the columns
# above (keep them in sync).
FORWARD_HORIZONS = (1, 5, 20, 60, 90)
DEFAULT_BENCHMARK = "SPY"


def _empty_row() -> dict:
    return {c: np.nan for c in FEATURE_COLUMNS}


def log_features(rows: list[dict], asof: date_cls) -> None:
    """Persist a day's features. `rows` is a list of dicts, one per ticker.

    Missing columns are filled with NaN. Existing file for asof is overwritten —
    callers should pass the full day's rows in one call.
    """
    if not rows:
        logger.warning("log_features called with empty rows for %s", asof)
        return
    normalized = []
    for r in rows:
        row = _empty_row()
        row.update({k: v for k, v in r.items() if k in FEATURE_COLUMNS})
        row["asof"] = asof
        normalized.append(row)
    df = pd.DataFrame(normalized, columns=FEATURE_COLUMNS)
    write_features(df, asof)
    logger.info("Wrote %d feature rows for %s", len(df), asof)


# ---- Forward-returns backfill ----------------------------------------------

def _forward_return(closes: pd.Series, asof: pd.Timestamp, horizon_days: int) -> float:
    """Return on close-to-close basis, N trading days forward. NaN if not enough data."""
    idx = closes.index
    if asof not in idx:
        # Find next available trading day after asof
        future = idx[idx > asof]
        if future.empty:
            return np.nan
        start = future[0]
    else:
        start = asof
    end_pos = idx.get_loc(start) + horizon_days
    if end_pos >= len(idx):
        return np.nan
    p0 = closes.loc[start]
    p1 = closes.iloc[end_pos]
    return float(p1 / p0 - 1.0)


def _min_calendar_days_for(horizon: int) -> int:
    """Calendar days that must elapse before `horizon` TRADING days have certainly
    passed (~5/7 weekday ratio plus a holiday cushion). Filling a horizon earlier
    produces a plausible-but-wrong return computed against too few trading days."""
    return int(horizon * 1.5) + 5


def _fetch_closes(ticker: str, start: date_cls, end: date_cls) -> pd.Series | None:
    """Split/div-adjusted daily closes for [start, end], date-indexed. None on
    empty/failure. auto_adjust=True — raw closes corrupt returns across corporate
    actions."""
    try:
        hist = yf.Ticker(ticker).history(
            start=start.isoformat(), end=end.isoformat(), auto_adjust=True,
        )
    except Exception as e:
        logger.warning("Close fetch failed for %s: %s", ticker, e)
        return None
    if hist is None or hist.empty:
        return None
    closes = hist["Close"]
    closes.index = pd.to_datetime(closes.index.date)  # drop tz/time
    return closes


def backfill_forward_returns(
    asof: date_cls,
    horizons: tuple = FORWARD_HORIZONS,
    benchmark_ticker: str | None = None,
) -> int:
    """For every prior date file in storage, populate fwd_ret_*, benchmark_ret_*,
    and alpha_* where missing. Returns the number of rows updated. Idempotent.

    Per-horizon gating: each horizon is filled only once enough calendar days have
    elapsed for that many trading days to have passed. Short horizons therefore
    backfill promptly even when long horizons (60/90d) are still pending — gating
    on max(horizons) would otherwise delay every column for months.

    alpha_Nd = fwd_ret_Nd − benchmark_ret_Nd, where the benchmark (default SPY) is
    configurable via thresholds `backfill.benchmark_ticker`. The benchmark close
    series is fetched ONCE per run and reused across every ticker and date, rather
    than re-fetched per ticker-file.
    """
    from .storage import FEATURES_DIR  # local to avoid circulars at import time

    benchmark_ticker = benchmark_ticker or DEFAULT_BENCHMARK
    buffer_days = _min_calendar_days_for(max(horizons)) + 10  # weekends + holidays

    # Pass 1 — decide which files have at least one horizon ready, WITHOUT fetching.
    # (Fetching only when something is ready keeps us off the network on no-op runs.)
    ready_files: list[tuple] = []
    for p in sorted(FEATURES_DIR.glob("date=*.parquet")):
        d = date_cls.fromisoformat(p.stem.split("=", 1)[1])
        if d >= asof:
            continue  # cannot have forward returns for today or the future
        elapsed = (asof - d).days
        ready = tuple(h for h in horizons if elapsed >= _min_calendar_days_for(h))
        if ready:
            ready_files.append((p, d, ready))

    if not ready_files:
        logger.info("Forward-return backfill: no dates with elapsed horizons yet.")
        return 0

    # Benchmark series fetched once over the full span any ready file could need.
    earliest = min(d for _, d, _ in ready_files)
    bench_closes = _fetch_closes(
        benchmark_ticker, earliest, asof + timedelta(days=buffer_days),
    )
    if bench_closes is None:
        logger.warning(
            "Benchmark %s closes unavailable; alpha/benchmark_ret will stay NaN.",
            benchmark_ticker,
        )

    updated = 0
    for p, d, ready in ready_files:
        df = pd.read_parquet(p)
        # Ensure every target column exists (older files predate these columns).
        for h in horizons:
            for col in (f"fwd_ret_{h}d", f"benchmark_ret_{h}d", f"alpha_{h}d"):
                if col not in df.columns:
                    df[col] = np.nan

        ready_ret_cols = [f"fwd_ret_{h}d" for h in ready]
        missing_mask = df[ready_ret_cols].isna().any(axis=1)
        if not missing_mask.any():
            continue

        asof_ts = pd.Timestamp(d)
        # Benchmark forward returns for this date — computed once, shared by all tickers.
        bench_fwd = {
            h: (_forward_return(bench_closes, asof_ts, h)
                if bench_closes is not None else float("nan"))
            for h in ready
        }

        for ticker, sub in df[missing_mask].groupby("ticker"):
            closes = _fetch_closes(ticker, d, d + timedelta(days=buffer_days))
            if closes is None:
                continue
            idx_rows = sub.index
            for h in ready:
                fr = _forward_return(closes, asof_ts, h)
                bf = bench_fwd[h]
                df.loc[idx_rows, f"fwd_ret_{h}d"] = fr
                df.loc[idx_rows, f"benchmark_ret_{h}d"] = bf
                df.loc[idx_rows, f"alpha_{h}d"] = (
                    fr - bf if (np.isfinite(fr) and np.isfinite(bf)) else float("nan")
                )
            updated += len(sub)

        df.to_parquet(p, index=False)

    logger.info(
        "Forward-return backfill: updated %d rows (benchmark=%s)", updated, benchmark_ticker,
    )
    return updated
