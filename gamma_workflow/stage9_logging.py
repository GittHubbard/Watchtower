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
    regime, gex_total, gex_zscore, distance_to_flip_pct, flip_level,
    call_wall, call_wall_gex, put_wall, put_wall_gex,
    call_wall_dod_change, put_wall_dod_change,
    ivx_7d, ivx_30d, hv_30d, skew_z, pcr,
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
    "regime", "gex_total", "gex_zscore", "distance_to_flip_pct", "flip_level",
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
    # Engine-derived IV/HV context (populated by orchestrator from engine output)
    "ivx_7d", "ivx_30d", "hv_30d",
    # Stage 5 — confirmation
    "volume_z", "atr_pct", "atr_20d_pct", "range_expansion", "close_location",
    "ma20_dist_pct", "ma50_dist_pct", "ma200_dist_pct",
    "vix_close", "vix_5d_change_pct",
    # Stage 1 — calendar
    "days_to_earnings", "days_to_macro", "next_macro", "within_event_window",
    "days_to_opex", "days_to_vixp", "is_opex_day", "is_vixp_day",
    # Stage 9 — backfilled
    "fwd_ret_1d", "fwd_ret_5d", "fwd_ret_20d",
]


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


def backfill_forward_returns(asof: date_cls, horizons: tuple = (1, 5, 20)) -> int:
    """For every prior date file in storage, populate fwd_ret_* where missing.

    Returns the number of rows updated. Idempotent — only fills NaNs.

    Strategy: load each historical features file, group by ticker, fetch daily
    closes from yfinance covering [date, date + max(horizons) + buffer], compute
    the forward returns, write back.
    """
    from .storage import FEATURES_DIR  # local to avoid circulars at import time

    updated = 0
    buffer_days = max(horizons) + 10  # weekends + holidays

    for p in sorted(FEATURES_DIR.glob("date=*.parquet")):
        d_str = p.stem.split("=", 1)[1]
        d = date_cls.fromisoformat(d_str)
        if d >= asof:
            continue  # cannot have forward returns for today or the future
        if (asof - d).days < max(horizons):
            continue  # not enough calendar days elapsed yet

        df = pd.read_parquet(p)
        missing_mask = df[[f"fwd_ret_{h}d" for h in horizons]].isna().any(axis=1)
        if not missing_mask.any():
            continue

        for ticker, sub in df[missing_mask].groupby("ticker"):
            try:
                hist = yf.Ticker(ticker).history(
                    start=d.isoformat(),
                    end=(d + timedelta(days=buffer_days)).isoformat(),
                    auto_adjust=False,
                )
                if hist.empty:
                    continue
                closes = hist["Close"]
                # Normalize index to date (drop tz/time)
                closes.index = pd.to_datetime(closes.index.date)
                asof_ts = pd.Timestamp(d)
                for h in horizons:
                    col = f"fwd_ret_{h}d"
                    idx_rows = sub.index
                    df.loc[idx_rows, col] = _forward_return(closes, asof_ts, h)
                updated += len(sub)
            except Exception as e:
                logger.warning("Forward-return backfill failed for %s on %s: %s", ticker, d, e)

        df.to_parquet(p, index=False)

    logger.info("Forward-return backfill: updated %d rows", updated)
    return updated
