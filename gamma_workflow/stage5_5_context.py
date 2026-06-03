"""Stage 5.5 — Technical & fundamental context (Features 2 + 3).

Two families of context that sit alongside the gamma/flow signals but, per the PM,
do NOT gate Stage 6 playbooks. They are logged as scalar features (audit trail)
and surfaced as commentary on the ticket for human review.

  Technicals (Feature 2) — computed from the underlying OHLCV history Stage 5
  already fetches; pure pandas, no new dependency:
      macd, macd_signal, macd_hist          (12 / 26 / 9)
      rsi_14                                 (Wilder's 14)
      cmf_20                                 (Chaikin Money Flow, 20)
      bb_upper, bb_mid, bb_lower, bb_width   (Bollinger 20 / 2)

  Fundamentals (Feature 3) — from yfinance via `acquire_fundamentals`:
      forward_pe, eps_surprise_avg_4q, eps_beat_streak, eps_estimate_trend
      net_income_margin_ttm, debt_to_equity, revenue_growth_yoy
      analyst_consensus, analyst_coverage_count, analyst_pct_buy
      analyst_upgrades_3m, analyst_downgrades_3m, analyst_steady_3m

ETF / index tickers (kind == 'index_etf') skip the fundamentals fetch entirely and
return all-NaN fundamental scalars.

Commentary: this module plumbs the commentary path end-to-end, but the narrative
generation is intentionally a STUB for now — `technical_commentary` and
`analyst_commentary` come back empty until the full Stage 5.5 brief lands. The
scalar features above are fully live and logged today. The convergence/divergence
and analyst-evolution narratives are a separate task that will read the scalar
columns plus the storage fundamentals-evolution cache.

Public API:
    acquire_fundamentals(symbol, ticker_obj) -> dict | None        (network)
    compute_technicals(history, asof) -> dict                      (pure)
    compute_fundamentals(raw_bundle, ticker_config, asof) -> dict  (pure)
    compute_stage5_5(history, asof, raw_bundle, ticker_config) -> dict
    split_for_cache(features) -> (analyst_payload, financials_payload)
"""
from __future__ import annotations
from datetime import date as date_cls
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---- Column groups ----------------------------------------------------------

TECHNICAL_KEYS = (
    "macd", "macd_signal", "macd_hist",
    "rsi_14", "cmf_20",
    "bb_upper", "bb_mid", "bb_lower", "bb_width",
)
ANALYST_KEYS = (
    "analyst_consensus", "analyst_coverage_count", "analyst_pct_buy",
    "analyst_upgrades_3m", "analyst_downgrades_3m", "analyst_steady_3m",
)
FINANCIAL_KEYS = (
    "forward_pe", "eps_surprise_avg_4q", "eps_beat_streak", "eps_estimate_trend",
    "net_income_margin_ttm", "debt_to_equity", "revenue_growth_yoy",
)

_NAN = float("nan")


def _num(v) -> float:
    """Coerce to float, NaN on anything non-numeric/None."""
    if v is None:
        return _NAN
    try:
        f = float(v)
    except (TypeError, ValueError):
        return _NAN
    return f if np.isfinite(f) else _NAN


# ---- Technical indicators (Feature 2) --------------------------------------

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> float:
    """Wilder's RSI over `period`. NaN if < period+1 closes."""
    if len(close) < period + 1:
        return _NAN
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    # Wilder smoothing == EWM with alpha = 1/period.
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    ag, al = float(avg_gain.iloc[-1]), float(avg_loss.iloc[-1])
    if not (np.isfinite(ag) and np.isfinite(al)):
        return _NAN
    if al == 0:
        return 100.0
    rs = ag / al
    return float(100.0 - 100.0 / (1.0 + rs))


def _cmf(high: pd.Series, low: pd.Series, close: pd.Series, vol: pd.Series, period: int = 20) -> float:
    """Chaikin Money Flow over `period`. NaN if < period bars or zero volume."""
    if len(close) < period:
        return _NAN
    rng = (high - low).replace(0, np.nan)
    mfm = ((close - low) - (high - close)) / rng          # money-flow multiplier
    mfv = (mfm * vol).fillna(0.0)                          # zero-range days contribute 0
    denom = float(vol.iloc[-period:].sum())
    if denom == 0 or not np.isfinite(denom):
        return _NAN
    return float(mfv.iloc[-period:].sum() / denom)


def _bollinger(close: pd.Series, period: int = 20, k: float = 2.0) -> tuple[float, float, float, float]:
    """(upper, mid, lower, width) for Bollinger bands. width = (upper-lower)/mid."""
    if len(close) < period:
        return _NAN, _NAN, _NAN, _NAN
    window = close.iloc[-period:]
    mid = float(window.mean())
    std = float(window.std(ddof=0))
    upper, lower = mid + k * std, mid - k * std
    width = (upper - lower) / mid if mid != 0 else _NAN
    return upper, mid, lower, width


def compute_technicals(history: pd.DataFrame | None, asof: date_cls) -> dict:
    """MACD / RSI / CMF / Bollinger from OHLCV up to `asof`. All-NaN if no history.

    Also returns `technical_commentary` (STUB — empty until the full brief lands).
    """
    out = {k: _NAN for k in TECHNICAL_KEYS}
    out["technical_commentary"] = ""

    if history is None or history.empty or "Close" not in history.columns:
        return out
    h = history[history.index <= pd.Timestamp(asof)]
    if h.empty:
        return out

    close = h["Close"].astype(float)
    high = h["High"].astype(float)
    low = h["Low"].astype(float)
    vol = h["Volume"].astype(float)

    if len(close) >= 26:
        macd_line = _ema(close, 12) - _ema(close, 26)
        signal = _ema(macd_line, 9)
        out["macd"] = float(macd_line.iloc[-1])
        out["macd_signal"] = float(signal.iloc[-1])
        out["macd_hist"] = float((macd_line - signal).iloc[-1])

    out["rsi_14"] = _rsi(close, 14)
    out["cmf_20"] = _cmf(high, low, close, vol, 20)
    out["bb_upper"], out["bb_mid"], out["bb_lower"], out["bb_width"] = _bollinger(close, 20, 2.0)
    return out


# ---- Fundamentals acquisition (Feature 3, network) -------------------------

def acquire_fundamentals(symbol: str, ticker_obj) -> dict | None:
    """Fetch the raw yfinance objects Stage 5.5 needs. Returns a bundle dict, or
    None if `ticker_obj` is missing. Each attribute is fetched defensively so one
    failed endpoint doesn't lose the rest. This is a network boundary — callers
    gate it on `fundamentals.enabled` and skip it for index ETFs.
    """
    if ticker_obj is None:
        return None

    def _safe(attr: str):
        try:
            return getattr(ticker_obj, attr)
        except Exception as e:
            logger.warning("fundamentals fetch %s.%s failed: %s", symbol, attr, e)
            return None

    return {
        "info": _safe("info") or {},
        "earnings_history": _safe("earnings_history"),
        "eps_trend": _safe("eps_trend"),
        "recommendations": _safe("recommendations"),
        "upgrades_downgrades": _safe("upgrades_downgrades"),
        "ttm_income_stmt": _safe("ttm_income_stmt"),
    }


# ---- Fundamentals extraction (Feature 3, pure) -----------------------------

def _earnings_metrics(eh: pd.DataFrame | None) -> tuple[float, float]:
    """(mean surprise % over last 4 reported quarters, consecutive recent beats)."""
    if eh is None or not hasattr(eh, "empty") or eh.empty or "surprisePercent" not in eh.columns:
        return _NAN, _NAN
    df = eh.sort_index()  # ascending by quarter date; most recent last
    sp = df["surprisePercent"].dropna()
    if sp.empty:
        return _NAN, _NAN
    avg4 = float(sp.iloc[-4:].mean())
    streak = 0
    for v in reversed(sp.tolist()):
        if np.isfinite(v) and v > 0:
            streak += 1
        else:
            break
    return avg4, float(streak)


def _eps_estimate_trend(et: pd.DataFrame | None) -> float:
    """Current-quarter EPS estimate now minus 30 days ago (absolute revision)."""
    if et is None or not hasattr(et, "empty") or et.empty:
        return _NAN
    if "current" not in et.columns or "30daysAgo" not in et.columns:
        return _NAN
    row = None
    for label in ("0q", "+0q", "0Q"):
        if label in et.index:
            row = et.loc[label]
            break
    if row is None:
        row = et.iloc[0]  # fall back to the first (nearest) period
    cur, ago = _num(row.get("current")), _num(row.get("30daysAgo"))
    if np.isfinite(cur) and np.isfinite(ago):
        return cur - ago
    return _NAN


def _ttm_net_margin(ttm: pd.DataFrame | None) -> float:
    """TTM net income / TTM total revenue from the ttm_income_stmt frame."""
    if ttm is None or not hasattr(ttm, "empty") or ttm.empty:
        return _NAN
    col = "TTM" if "TTM" in ttm.columns else ttm.columns[0]

    def _row(*names):
        for n in names:
            if n in ttm.index:
                return _num(ttm.loc[n, col])
        return _NAN

    ni = _row("Net Income", "NetIncome", "Net Income Common Stockholders")
    rev = _row("Total Revenue", "TotalRevenue", "Operating Revenue")
    if np.isfinite(ni) and np.isfinite(rev) and rev > 0:
        return ni / rev
    return _NAN


def _pct_buy(rec: pd.DataFrame | None) -> float:
    """Fraction of analysts at Buy/Strong-Buy (a.k.a. overweight) from the latest
    recommendations-summary row."""
    if rec is None or not hasattr(rec, "empty") or rec.empty:
        return _NAN
    row = None
    if "period" in rec.columns and (rec["period"] == "0m").any():
        row = rec[rec["period"] == "0m"].iloc[0]
    else:
        row = rec.iloc[0]
    sb, b = _num(row.get("strongBuy")), _num(row.get("buy"))
    h, s, ss = _num(row.get("hold")), _num(row.get("sell")), _num(row.get("strongSell"))
    total = np.nansum([sb, b, h, s, ss])
    buy = np.nansum([sb, b])
    if total > 0:
        return float(buy / total)
    return _NAN


def _grade_changes(ud: pd.DataFrame | None, asof: date_cls, days: int = 90) -> tuple[float, float, float]:
    """(#upgrades, #downgrades, #maintained) in the trailing `days` window. NaN
    triple when no frame is available; zeros are a real 'none in window' reading."""
    if ud is None or not hasattr(ud, "empty") or ud.empty or "Action" not in ud.columns:
        return _NAN, _NAN, _NAN
    df = ud.copy()
    idx = pd.to_datetime(df.index, errors="coerce")
    try:
        idx = idx.tz_localize(None)
    except (TypeError, AttributeError):
        idx = idx.tz_convert(None) if getattr(idx, "tz", None) is not None else idx
    df = df.assign(_gd=idx)
    cutoff = pd.Timestamp(asof) - pd.Timedelta(days=days)
    recent = df[df["_gd"] >= cutoff]
    actions = recent["Action"].astype(str).str.lower()
    up = int((actions == "up").sum())
    down = int((actions == "down").sum())
    steady = int(actions.isin(["main", "reit"]).sum())
    return float(up), float(down), float(steady)


def compute_fundamentals(raw: dict | None, ticker_config: dict, asof: date_cls) -> dict:
    """Extract the fundamental scalar features from a raw yfinance bundle. Returns
    all-NaN (plus empty commentary) for index ETFs or when `raw` is None."""
    out = {k: _NAN for k in (ANALYST_KEYS + FINANCIAL_KEYS)}
    out["analyst_commentary"] = ""

    if ticker_config.get("kind") == "index_etf" or raw is None:
        return out

    info = raw.get("info") or {}
    out["forward_pe"] = _num(info.get("forwardPE"))
    out["debt_to_equity"] = _num(info.get("debtToEquity"))
    out["revenue_growth_yoy"] = _num(info.get("revenueGrowth"))
    out["analyst_consensus"] = _num(info.get("recommendationMean"))
    out["analyst_coverage_count"] = _num(info.get("numberOfAnalystOpinions"))

    out["eps_surprise_avg_4q"], out["eps_beat_streak"] = _earnings_metrics(raw.get("earnings_history"))
    out["eps_estimate_trend"] = _eps_estimate_trend(raw.get("eps_trend"))
    out["net_income_margin_ttm"] = _ttm_net_margin(raw.get("ttm_income_stmt"))
    out["analyst_pct_buy"] = _pct_buy(raw.get("recommendations"))
    up, down, steady = _grade_changes(raw.get("upgrades_downgrades"), asof, 90)
    out["analyst_upgrades_3m"], out["analyst_downgrades_3m"], out["analyst_steady_3m"] = up, down, steady
    return out


# ---- Orchestration ----------------------------------------------------------

def compute_stage5_5(
    history: pd.DataFrame | None,
    asof: date_cls,
    raw_bundle: dict | None,
    ticker_config: dict,
) -> dict:
    """Full Stage 5.5 feature slice: technical scalars + fundamental scalars +
    (stub) commentary strings. Pure with respect to the network."""
    out = {}
    out.update(compute_technicals(history, asof))
    out.update(compute_fundamentals(raw_bundle, ticker_config, asof))
    return out


def split_for_cache(features: dict) -> tuple[dict, dict]:
    """Split a feature row into the (analyst, financials) payloads tracked by the
    storage evolution cache. Each is a flat dict of just that bucket's scalars."""
    analyst = {k: _num(features.get(k)) for k in ANALYST_KEYS}
    financials = {k: _num(features.get(k)) for k in FINANCIAL_KEYS}
    return analyst, financials
