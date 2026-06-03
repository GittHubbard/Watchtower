#!/usr/bin/env python3
"""Standalone multi-indicator trade signal generator.

Fetches OHLCV from yfinance (no engine required) and runs the Stage 5.5
technical analysis pipeline, printing a full signal report to stdout.

Usage:
    python trade_signals.py AAPL
    python trade_signals.py AAPL --bar -1
    python trade_signals.py AAPL --bars-since 400
    python trade_signals.py AAPL --iv-csv iv_data.csv
    python trade_signals.py AAPL --params params.yaml
"""
from __future__ import annotations

import argparse
import math
import sys
from datetime import date as date_cls, timedelta

import numpy as np
import pandas as pd

# Import private + public functions from the stage5_5_context module
from gamma_workflow.stage5_5_context import (
    _ema,
    _rsi,
    _cmf,
    _bollinger,
    _adx,
    _obv_sma,
    _rv_20,
    _anchored_vwap,
    _iv_context,
    _trend_price,
    _trend_rsi,
    _trend_macd,
    _trend_bb,
    _trend_cmf,
    _trend_volume,
    _trend_obv,
    _trend_avwap,
    _adx_regime,
    _vol_regime,
    _rsi_series,
    _cmf_series,
    _macd_hist_series,
    _run_divergence_suite,
    _score_technical,
    _REGIME_MATRIX,
    _NAN,
)

_NAN = float("nan")


def fetch_ohlcv(symbol: str, bars_since: int = 400) -> pd.DataFrame:
    """Fetch OHLCV from yfinance for the given symbol."""
    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: yfinance is required. Install with: pip install yfinance", file=sys.stderr)
        sys.exit(1)

    end = date_cls.today()
    start = end - timedelta(days=int(bars_since * 1.5))   # extra buffer for weekends
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start.isoformat(), end=end.isoformat())
    if df.empty:
        print(f"ERROR: No data returned for {symbol}", file=sys.stderr)
        sys.exit(1)

    # Normalize column names
    df.index = pd.to_datetime(df.index).tz_localize(None)
    needed = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        print(f"ERROR: Missing columns {missing} for {symbol}", file=sys.stderr)
        sys.exit(1)

    df = df[needed].copy()
    # Trim to requested bars_since (business days approximation)
    if len(df) > bars_since:
        df = df.iloc[-bars_since:]
    return df


def load_iv_csv(path: str | None) -> pd.Series | None:
    """Load optional IV CSV. Expects date index and `iv_30` column (decimal)."""
    if path is None:
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        if "iv_30" in df.columns:
            return df["iv_30"]
        print(f"WARNING: {path} has no 'iv_30' column — IV ignored", file=sys.stderr)
        return None
    except Exception as e:
        print(f"WARNING: Could not load IV CSV {path}: {e}", file=sys.stderr)
        return None


def load_params(path: str | None) -> dict:
    """Load optional YAML parameter overrides."""
    if path is None:
        return {}
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        print("WARNING: PyYAML not installed — params file ignored", file=sys.stderr)
        return {}
    except Exception as e:
        print(f"WARNING: Could not load params {path}: {e}", file=sys.stderr)
        return {}


def _compute_vol_z(volume: pd.Series) -> float:
    """20-bar volume z-score."""
    if len(volume) < 20:
        return _NAN
    vol_mean = float(volume.iloc[-20:].mean())
    vol_std = float(volume.iloc[-20:].std(ddof=1))
    if vol_std == 0:
        return 0.0
    return (float(volume.iloc[-1]) - vol_mean) / vol_std


def analyze_bar(df: pd.DataFrame, bar_idx: int, ivx_30d_pct: float = _NAN) -> dict:
    """Run full technical analysis on df.iloc[:bar_idx+1].

    Returns a dict with all computed values for reporting.
    """
    # Slice: all bars up to and including the target
    h = df.iloc[:bar_idx + 1].copy()

    close = h["Close"].astype(float)
    high = h["High"].astype(float)
    low = h["Low"].astype(float)
    volume = h["Volume"].astype(float)

    result = {
        "n_bars": len(h),
        "bar_date": h.index[-1],
        "close": float(close.iloc[-1]),
    }

    # Core scalars
    if len(close) >= 26:
        macd_line = _ema(close, 12) - _ema(close, 26)
        signal_line = _ema(macd_line, 9)
        result["macd"] = float(macd_line.iloc[-1])
        result["macd_signal"] = float(signal_line.iloc[-1])
        result["macd_hist"] = float((macd_line - signal_line).iloc[-1])
        result["macd_hist_series"] = macd_line - signal_line
    else:
        result["macd"] = _NAN
        result["macd_signal"] = _NAN
        result["macd_hist"] = _NAN
        result["macd_hist_series"] = pd.Series(dtype=float)

    result["rsi_14"] = _rsi(close, 14)
    result["cmf_20"] = _cmf(high, low, close, volume, 20)
    bb_upper, bb_mid, bb_lower, bb_width = _bollinger(close, 20, 2.0)
    result["bb_upper"] = bb_upper
    result["bb_mid"] = bb_mid
    result["bb_lower"] = bb_lower
    result["bb_width"] = bb_width

    # Additional indicators
    adx_val, pdi_val, mdi_val = _adx(high, low, close)
    result["adx"] = adx_val
    result["plus_di"] = pdi_val
    result["minus_di"] = mdi_val

    obv_cur, obv_sma20, obv_slope = _obv_sma(close, volume)
    result["obv_cur"] = obv_cur
    result["obv_sma20"] = obv_sma20
    result["obv_slope"] = obv_slope

    result["rv_20"] = _rv_20(close)

    avwap_h, avwap_l, dist_h, dist_l, age_h, age_l = _anchored_vwap(close, high, low, volume)
    result["avwap_h"] = avwap_h
    result["avwap_l"] = avwap_l
    result["dist_h"] = dist_h
    result["dist_l"] = dist_l
    result["age_h"] = age_h
    result["age_l"] = age_l

    # Volume z
    vol_z = _compute_vol_z(volume)
    result["vol_z"] = vol_z

    # IV context
    iv30, iv_premium, iv_premium_ratio = _iv_context(ivx_30d_pct, result["rv_20"], _NAN)
    result["iv30"] = iv30
    result["iv_premium"] = iv_premium
    result["iv_premium_ratio"] = iv_premium_ratio

    # Trend classifications
    result["trend_price"] = _trend_price(close)
    result["trend_macd"] = _trend_macd(result["macd"], result["macd_signal"], result["macd_hist"])
    result["trend_rsi"] = _trend_rsi(result["rsi_14"], close, 14)
    result["trend_bb"] = _trend_bb(result["close"], bb_upper, bb_lower, bb_mid)
    result["trend_cmf"] = _trend_cmf(result["cmf_20"])
    result["trend_volume"] = _trend_volume(vol_z)
    result["trend_obv"] = _trend_obv(obv_cur, obv_sma20, obv_slope)

    prev_close = float(close.iloc[-2]) if len(close) >= 2 else None
    result["trend_avwap"] = _trend_avwap(
        result["close"], avwap_h, avwap_l, dist_h, dist_l, prev_close
    )

    # Regime
    result["adx_regime"] = _adx_regime(adx_val)
    result["vol_regime"] = _vol_regime(_NAN)   # no historical IV store in standalone

    # Divergence suite
    sma20_series = close.rolling(20, min_periods=20).mean()
    div_results = _run_divergence_suite(close, high, low, volume, sma20_series)
    result["div_results"] = div_results

    # Score
    scoring = _score_technical(
        result["trend_price"], result["trend_macd"], result["trend_rsi"],
        result["trend_bb"], result["trend_cmf"], result["trend_volume"],
        result["trend_obv"], result["trend_avwap"],
        result["rsi_14"], result["close"], bb_upper, bb_lower, bb_mid,
        close, result["macd_hist_series"],
        result["adx_regime"], result["vol_regime"],
        div_results,
    )
    result.update(scoring)

    return result


def render_report(symbol: str, bar_idx: int, r: dict) -> str:
    """Render a human-readable signal report."""
    lines = []

    bar_date = r["bar_date"]
    if hasattr(bar_date, "date"):
        bar_date_str = bar_date.date().isoformat()
    else:
        bar_date_str = str(bar_date)

    lines.append(f"# Trade Signal Report: {symbol}")
    lines.append(f"Bar index: {bar_idx}  |  Date: {bar_date_str}  |  Close: {r['close']:.2f}")
    lines.append(f"History depth: {r['n_bars']} bars")
    lines.append("")

    # --- Main signal ---
    score = r["score"]
    rec = r["rec"]
    conf = r["conf"]
    lines.append(f"### Technical Signal: {rec} ({score:+.1f} / confidence {conf:.0%})")
    lines.append("")

    # Regime
    adx_val = r["adx"]
    pdi_val = r["plus_di"]
    mdi_val = r["minus_di"]
    adx_str = (
        f"ADX {adx_val:.0f}, +DI {pdi_val:.0f} vs -DI {mdi_val:.0f}"
        if math.isfinite(adx_val) else "ADX unavailable"
    )
    iv30 = r["iv30"]
    rv20 = r["rv_20"]
    iv_str = f"{iv30:.1%}" if math.isfinite(iv30) else "N/A"
    rv_str = f"{rv20:.1%}" if math.isfinite(rv20) else "N/A"
    mr_weight = r["mr_weight"]
    adx_reg = r["adx_regime"]
    vol_reg = r["vol_regime"]
    lines.append(
        f"**Regime**: ADX {adx_reg} ({adx_str})  |  Vol: {vol_reg} (IV30 {iv_str}, RV20 {rv_str})"
    )
    lines.append(f"**Gate**: MR weight = {mr_weight:.2f}")
    lines.append("")

    # Trend
    base_trend = r["base_trend"]
    trend_score = r["trend_score"]
    conv_mult = r["conv_mult"]
    lines.append(f"**Trend** ({base_trend:+.2f} = {trend_score:+.2f} × {conv_mult:.2f} convergence)")
    lines.append(
        f"- Price: {r['trend_price']}  MACD: {r['trend_macd']}  CMF: {r['trend_cmf']}"
    )
    lines.append(
        f"- OBV: {r['trend_obv']}  aVWAP: {r['trend_avwap']}  Volume: {r['trend_volume']}"
    )

    avwap_parts = []
    if math.isfinite(r["avwap_h"]) and math.isfinite(r["age_h"]):
        avwap_parts.append(f"aVWAP-high {r['avwap_h']:.2f} (anchor {r['age_h']:.0f} bars ago)")
    if math.isfinite(r["avwap_l"]) and math.isfinite(r["age_l"]):
        avwap_parts.append(f"aVWAP-low {r['avwap_l']:.2f} (anchor {r['age_l']:.0f} bars ago)")
    if avwap_parts:
        lines.append(f"- {', '.join(avwap_parts)}")
    lines.append("")

    # Mean reversion
    mr_contribution = r["mr_contribution"]
    rsi_val = r["rsi_14"]
    rsi_str = f"{rsi_val:.0f}" if math.isfinite(rsi_val) else "N/A"
    rsi_label = r["rsi_label"]
    bb_label = r["bb_label"]
    lines.append(f"**Mean Reversion** (gate {mr_weight:.2f} → contribution {mr_contribution:+.2f})")
    lines.append(f"- RSI {rsi_str}: {rsi_label}  Bollinger: {bb_label}")
    lines.append("")

    # Divergence
    div_score = r["div_score"]
    lines.append(f"**Divergence** ({div_score:+.1f})")
    div_results = r["div_results"]
    div_lines = []
    for name, events in div_results.items():
        for ev in events:
            indicator_label = name.replace("_bull", "").replace("_bear", "")
            icon = "✓ CONFIRMED" if ev["state"] == "confirmed" else "⏳ PENDING"
            div_lines.append(
                f"- {icon} {ev['type']} on {indicator_label} (quality: {ev['quality']})"
            )
    if div_lines:
        lines.extend(div_lines)
    else:
        lines.append("- No active divergences")
    lines.append("")

    # Score breakdown
    raw = r["raw"]
    lines.append(
        f"**Score**: {base_trend:+.2f} + {mr_contribution:+.2f} + {div_score:+.2f} "
        f"= {raw:+.2f} → **{score:+.2f}**"
    )

    # Caveats
    if r["n_bars"] < 200:
        lines.append(f"Note: Only {r['n_bars']} bars — price-trend SMA200 unavailable (warmup)")
    if not math.isfinite(adx_val):
        lines.append("Note: ADX unavailable — insufficient history for Wilder smoothing")
    if not math.isfinite(iv30):
        lines.append("Note: IV30 not available — using RV-only vol regime")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone multi-indicator trade signal generator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("ticker", help="Ticker symbol (e.g. AAPL)")
    parser.add_argument(
        "--bar", type=int, default=-1,
        help="Bar index to report on (default: -1 = most recent)",
    )
    parser.add_argument(
        "--bars-since", type=int, default=400,
        help="Number of OHLCV bars to fetch (default: 400)",
    )
    parser.add_argument(
        "--iv-csv", metavar="PATH", default=None,
        help="CSV with date index and iv_30 column (decimal values)",
    )
    parser.add_argument(
        "--params", metavar="PATH", default=None,
        help="YAML file with threshold overrides",
    )
    args = parser.parse_args()

    symbol = args.ticker.upper()
    bar_idx = args.bar
    bars_since = args.bars_since

    # Fetch OHLCV
    print(f"Fetching {bars_since} bars of OHLCV for {symbol}...", file=sys.stderr)
    df = fetch_ohlcv(symbol, bars_since)

    # Normalize bar_idx
    n = len(df)
    if bar_idx < 0:
        bar_idx = n + bar_idx
    if bar_idx < 0 or bar_idx >= n:
        print(f"ERROR: bar index {args.bar} out of range for {n} bars", file=sys.stderr)
        sys.exit(1)

    # Load optional IV data
    iv_series = load_iv_csv(args.iv_csv)
    ivx_30d_pct = _NAN
    if iv_series is not None:
        bar_date = df.index[bar_idx]
        # Find closest date at or before bar_date
        iv_avail = iv_series[iv_series.index <= bar_date]
        if not iv_avail.empty:
            raw_iv = float(iv_avail.iloc[-1])
            # Convert decimal to percent (e.g. 0.25 → 25.0)
            ivx_30d_pct = raw_iv * 100.0 if raw_iv < 2.0 else raw_iv

    # Load params (reserved for future use)
    _params = load_params(args.params)

    # Analyze
    result = analyze_bar(df, bar_idx, ivx_30d_pct=ivx_30d_pct)

    # Render and print
    report = render_report(symbol, bar_idx, result)
    print(report)


if __name__ == "__main__":
    main()
