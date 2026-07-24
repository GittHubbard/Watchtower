"""Build a trade log from the Downward and Upward technical-signal watchlists.

Each watchlist lives under ``data/<direction>-Watchlist/<YYYY-MM-DD>/`` and holds
one ``<date>_<TICKER>_techsignal.md`` file per name per day. This module walks
both trees, extracts the recommendation, score and confidence from every signal
file, and returns them as a tidy :class:`pandas.DataFrame` that a downstream step
can use to analyse the quality of the recommendations.

Public API
----------
``build_trade_log(down_dir=..., up_dir=...) -> pandas.DataFrame``
    One row per signal file with columns:

    ==================  ==========================================================
    column              meaning
    ==================  ==========================================================
    ``ticker``          symbol parsed from the filename (e.g. ``ADP``)
    ``date``            signal date (``datetime64[ns]``)
    ``watchlist``       ``"down"`` or ``"up"`` (which watchlist the file came from)
    ``recommendation``  ``BUY`` / ``STRONG_BUY`` / ``SELL`` / ``STRONG_SELL``
    ``score_capped``    displayed score, clamped to the [-10, +10] band
    ``score_raw``       uncapped score before clamping (higher resolution)
    ``confidence_pct``  confidence as an integer percentage (0-100)
    ==================  ==========================================================

    plus every technical rating parsed from the file: the categorical features
    in :data:`CATEGORICAL_FEATURES` (``regime``, ``vol_regime``, ``price_state``,
    ``macd``, ``cmf``, ``obv``, ``avwap``, ``volume``, ``rsi_state``,
    ``bollinger``) and the numeric features in :data:`NUMERIC_FEATURES` (ADX/DI,
    IV30/RV20, trend contribution, RSI, divergence score and counts, etc.).

``analyze_signal_predictiveness(frame=None, horizon=20, ...) -> dict``
    Attaches a fixed-horizon forward alpha (vs ``^GSPC``) to *every* signal —
    irrespective of its recommendation — then reports which technical features
    predict a positive forward alpha, via a univariate mean-alpha/win-rate table
    and a logistic regression of P(alpha > 0) on the features. Answers questions
    like "when MACD is BULLISH and CMF is ACCUM and RSI < 50, is forward alpha
    positive even on SELL-rated names?".

``analyze_performance(frame=None, price_loader=...) -> pandas.DataFrame``
    Runs a long-only backtest over the trade log and returns one row per closed
    trade with its % return and alpha vs the S&P 500. Strategy:

    * Start flat on the first date in the data.
    * Each day, open a position in any ticker rated ``STRONG_BUY`` — optionally
      also requiring ``score_raw >= min_score_raw`` — that we do not already
      hold (re-entry allowed after a prior exit).
    * Close a held position the first day the ticker is rated ``SELL`` or
      ``STRONG_SELL``.
    * Any position still open on the last date is closed at that date's price.

    Each closed trade records ``buy_score_raw`` (the opening signal's uncapped
    score). For every closed trade the stock's % return (adjusted close, buy date
    to sell date) is compared against the S&P 500 (``^GSPC``) over the *same* date
    window; ``alpha_pct`` is the difference. Prices come from Yahoo Finance via
    yfinance and are downloaded once (all traded tickers in one batch, the
    benchmark once) and cached in memory. The fetch is injectable via
    ``price_loader`` so it can be stubbed for testing or pointed elsewhere.

Run as a script (``python scripts/build_trade_log.py``) to print a short summary
and preview of the signal log; no file is written. Add ``--performance`` to run
the backtest (optionally gated with ``--filter macd=BULLISH`` etc.), and/or
``--regress`` to run the signal-predictiveness analysis. Both price-based steps
download from Yahoo Finance.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOWN_DIR = REPO_ROOT / "data" / "Downward-Watchlist"
DEFAULT_UP_DIR = REPO_ROOT / "data" / "Upward-Watchlist"

# 2026-07-22_ADP_techsignal.md -> ("2026-07-22", "ADP")
_FILENAME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_([A-Z0-9.\-]+)_techsignal\.md$")

# **Recommendation**: STRONG_BUY  |  **Score**: +10.0
_HEADER_RE = re.compile(
    r"\*\*Recommendation\*\*:\s*([A-Z_]+)\s*\|\s*\*\*Score\*\*:\s*([+-][\d.]+)"
)

# ### Technical Signal: BUY (+4.1 / confidence 41%)
_CONFIDENCE_RE = re.compile(
    r"^###\s*Technical Signal:\s*[A-Z_]+\s*\(\s*[+-][\d.]+\s*/\s*confidence\s*(\d+)%\s*\)",
    re.MULTILINE,
)

# **Score**: +11.56 + +0.00 + +1.00 = +12.56 → **+10.00**
#   group 1 = raw (uncapped) sum, group 2 = capped (clamped) final
_FINAL_SCORE_RE = re.compile(r"=\s*([+-][\d.]+)\s*→\s*\*\*([+-][\d.]+)\*\*")

# **Regime**: ADX TRENDING (ADX 30, +DI 30 vs -DI 19)
_REGIME_RE = re.compile(
    r"\*\*Regime\*\*:\s*ADX\s+([A-Z_]+)\s*\(ADX\s+(\d+),\s*\+DI\s+(\d+)\s*vs\s*-DI\s+(\d+)\)"
)
# Vol: NORMAL_IV (IV30 37.0%, RV20 36.3%)   (IV30 may be "N/A")
_VOL_RE = re.compile(
    r"Vol:\s*([A-Z_]+)\s*\(IV30\s+(N/A|[\d.]+)%?,\s*RV20\s+(N/A|[\d.]+)%?\)"
)
# **Trend** (-2.81 = -2.25 × 1.25 convergence)   (× is a non-ASCII glyph -> \S)
_TREND_HEAD_RE = re.compile(
    r"\*\*Trend\*\*\s*\(([+-][\d.]+)\s*=\s*([+-][\d.]+)\s*\S\s*([\d.]+)\s*convergence\)"
)
# - Price: NEUTRAL  MACD: BEAR_CROSS  CMF: STRONG_ACCUM
_TREND_L1_RE = re.compile(r"Price:\s*([A-Z_]+)\s+MACD:\s*([A-Z_]+)\s+CMF:\s*([A-Z_]+)")
# - OBV: BEARISH  aVWAP: ABOVE_BOTH  Volume: DRY
_TREND_L2_RE = re.compile(r"OBV:\s*([A-Z_]+)\s+aVWAP:\s*([A-Z_]+)\s+Volume:\s*([A-Z_]+)")
# **Mean Reversion** (gate 0.00 → contribution +0.00)
_MR_HEAD_RE = re.compile(
    r"\*\*Mean Reversion\*\*\s*\(gate\s*([\d.]+)\s*\S\s*contribution\s*([+-][\d.]+)\)"
)
# - RSI 55: neutral  Bollinger: neutral   (RSI state may be hyphenated, e.g. extreme-overbought)
_MR_L1_RE = re.compile(r"RSI\s+(\d+):\s*([a-z-]+)\s+Bollinger:\s*([a-z-]+)")
# **Divergence** (+4.0)
_DIV_HEAD_RE = re.compile(r"\*\*Divergence\*\*\s*\(([+-][\d.]+)\)")
# - ✓ CONFIRMED reg_bull on MACDLine (quality: strong)
_DIV_ITEM_RE = re.compile(r"(CONFIRMED|PENDING)\s+(reg_bull|reg_bear|hid_bull|hid_bear)\s+on\s+")

# Categorical signal features (the "ratings") extracted from every signal file.
CATEGORICAL_FEATURES = [
    "regime", "vol_regime", "price_state", "macd", "cmf", "obv",
    "avwap", "volume", "rsi_state", "bollinger",
]
# Numeric signal features.
NUMERIC_FEATURES = [
    "adx", "plus_di", "minus_di", "iv30", "rv20",
    "trend_contribution", "trend_base", "convergence",
    "mr_gate", "mr_contribution", "rsi", "divergence_score",
    "div_reg_bull", "div_reg_bear", "div_hid_bull", "div_hid_bear",
    "div_confirmed", "div_pending",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES

COLUMNS = [
    "ticker",
    "date",
    "watchlist",
    "recommendation",
    "score_capped",
    "score_raw",
    "confidence_pct",
] + FEATURE_COLUMNS


def parse_signal_text(text: str, *, ticker: str, date: str, watchlist: str, source: str = "") -> dict:
    """Extract the trade-log fields from a techsignal ticket's markdown ``text``.

    ``ticker``/``date``/``watchlist`` are recorded as given (``date`` an ISO
    string). ``source`` is a label used only in error messages. Raises
    ``ValueError`` if any required field cannot be parsed, so malformed tickets
    surface loudly rather than silently yielding NaN. This is the shared core of
    :func:`parse_signal_file` (which reads a ``.md`` file) and
    :func:`build_trade_log_from_log` (which reads the parquet log's ``content``).
    """
    source = source or f"{date}_{ticker}"

    header = _HEADER_RE.search(text)
    if not header:
        raise ValueError(f"Missing Recommendation/Score header in {source}")
    recommendation = header.group(1)
    score_capped = float(header.group(2))

    confidence = _CONFIDENCE_RE.search(text)
    if not confidence:
        raise ValueError(f"Missing confidence line in {source}")
    confidence_pct = int(confidence.group(1))

    final = _FINAL_SCORE_RE.search(text)
    if not final:
        raise ValueError(f"Missing final score breakdown in {source}")
    score_raw = float(final.group(1))

    record = {
        "ticker": ticker,
        "date": date,
        "watchlist": watchlist,
        "recommendation": recommendation,
        "score_capped": score_capped,
        "score_raw": score_raw,
        "confidence_pct": confidence_pct,
    }
    record.update(_parse_features(text, source))
    return record


def parse_signal_file(path: Path, watchlist: str) -> dict:
    """Extract the trade-log fields from a single ``*_techsignal.md`` file.

    ``watchlist`` is the direction label (``"down"`` / ``"up"``) to record for
    this file. Raises ``ValueError`` if the filename or any required field cannot
    be parsed, so malformed files surface loudly rather than silently vanishing.
    """
    name_match = _FILENAME_RE.match(path.name)
    if not name_match:
        raise ValueError(f"Unexpected signal filename: {path.name}")
    date_str, ticker = name_match.group(1), name_match.group(2)
    text = path.read_text(encoding="utf-8")
    return parse_signal_text(text, ticker=ticker, date=date_str, watchlist=watchlist, source=str(path))


def _nan_or_float(token: str) -> float:
    """Convert a numeric token to float, mapping ``"N/A"`` to NaN."""
    return float("nan") if token == "N/A" else float(token)


def _parse_features(text: str, source: str) -> dict:
    """Extract the Regime / Vol / Trend / Mean Reversion / Divergence fields.

    Every field is required and present in all well-formed signal files, so a
    miss raises ``ValueError`` naming the file rather than silently yielding NaN.
    """
    regime = _REGIME_RE.search(text)
    vol = _VOL_RE.search(text)
    trend_h = _TREND_HEAD_RE.search(text)
    trend1 = _TREND_L1_RE.search(text)
    trend2 = _TREND_L2_RE.search(text)
    mr_h = _MR_HEAD_RE.search(text)
    mr1 = _MR_L1_RE.search(text)
    div_h = _DIV_HEAD_RE.search(text)
    for name, match in (
        ("Regime", regime), ("Vol", vol), ("Trend header", trend_h),
        ("Trend line 1", trend1), ("Trend line 2", trend2),
        ("Mean Reversion header", mr_h), ("Mean Reversion line", mr1),
        ("Divergence header", div_h),
    ):
        if not match:
            raise ValueError(f"Missing {name} section in {source}")

    # Divergence detail: count active divergences by type and confirmation state.
    div_counts = {"reg_bull": 0, "reg_bear": 0, "hid_bull": 0, "hid_bear": 0}
    confirmed = pending = 0
    for status, kind in _DIV_ITEM_RE.findall(text):
        div_counts[kind] += 1
        if status == "CONFIRMED":
            confirmed += 1
        else:
            pending += 1

    return {
        "regime": regime.group(1),
        "adx": int(regime.group(2)),
        "plus_di": int(regime.group(3)),
        "minus_di": int(regime.group(4)),
        "vol_regime": vol.group(1),
        "iv30": _nan_or_float(vol.group(2)),
        "rv20": _nan_or_float(vol.group(3)),
        "trend_contribution": float(trend_h.group(1)),
        "trend_base": float(trend_h.group(2)),
        "convergence": float(trend_h.group(3)),
        "price_state": trend1.group(1),
        "macd": trend1.group(2),
        "cmf": trend1.group(3),
        "obv": trend2.group(1),
        "avwap": trend2.group(2),
        "volume": trend2.group(3),
        "mr_gate": float(mr_h.group(1)),
        "mr_contribution": float(mr_h.group(2)),
        "rsi": int(mr1.group(1)),
        "rsi_state": mr1.group(2),
        "bollinger": mr1.group(3),
        "divergence_score": float(div_h.group(1)),
        "div_reg_bull": div_counts["reg_bull"],
        "div_reg_bear": div_counts["reg_bear"],
        "div_hid_bull": div_counts["hid_bull"],
        "div_hid_bear": div_counts["hid_bear"],
        "div_confirmed": confirmed,
        "div_pending": pending,
    }


def _collect(directory: Path, watchlist: str) -> list[dict]:
    if not directory.exists():
        raise FileNotFoundError(f"Watchlist directory not found: {directory}")
    rows = [
        parse_signal_file(path, watchlist)
        for path in sorted(directory.rglob("*_techsignal.md"))
    ]
    return rows


def build_trade_log(
    down_dir: Path = DEFAULT_DOWN_DIR,
    up_dir: Path = DEFAULT_UP_DIR,
) -> pd.DataFrame:
    """Read both watchlists and return the trade log as a DataFrame.

    One row per signal file, sorted by date, then watchlist, then ticker. The
    ``date`` column is parsed to ``datetime64[ns]``; all other columns keep the
    dtypes described in the module docstring.
    """
    rows = _collect(down_dir, "down") + _collect(up_dir, "up")
    frame = pd.DataFrame(rows, columns=COLUMNS)
    frame["date"] = pd.to_datetime(frame["date"], format="%Y-%m-%d")
    frame = frame.sort_values(["date", "watchlist", "ticker"]).reset_index(drop=True)
    return frame


_LOG_WATCHLIST_MAP = {"upward": "up", "downward": "down"}


def build_trade_log_from_log(log: pd.DataFrame) -> pd.DataFrame:
    """Build the same feature frame as :func:`build_trade_log`, but from the
    accumulated tech-signal parquet log (``storage.read_techsignal_log()``).

    The log covers *every* ticker each day — including HOLD-rated names that never
    reached a watchlist — so this is the richer, fuller universe. Each row's
    ``content`` markdown is parsed with :func:`parse_signal_text` into the full
    feature set. The ``watchlist`` column is normalised to ``up`` / ``down`` /
    ``none`` to match :func:`build_trade_log`. Rows whose content can't be parsed
    are dropped and counted in a ``build_trade_log_from_log.skipped`` attribute.
    """
    rows: list[dict] = []
    skipped = 0
    for r in log.itertuples(index=False):
        wl = _LOG_WATCHLIST_MAP.get(r.watchlist, "none") if isinstance(r.watchlist, str) else "none"
        try:
            rows.append(parse_signal_text(
                r.content, ticker=r.ticker, date=str(r.date), watchlist=wl,
            ))
        except ValueError:
            skipped += 1
    frame = pd.DataFrame(rows, columns=COLUMNS)
    frame["date"] = pd.to_datetime(frame["date"], format="%Y-%m-%d")
    frame = frame.sort_values(["date", "watchlist", "ticker"]).reset_index(drop=True)
    build_trade_log_from_log.skipped = skipped  # type: ignore[attr-defined]
    return frame


# --------------------------------------------------------------------------- #
# Performance analysis layer
# --------------------------------------------------------------------------- #

OPEN_RATINGS = frozenset({"STRONG_BUY"})            # ratings that open a position
CLOSE_RATINGS = frozenset({"SELL", "STRONG_SELL"})  # ratings that close a position
BENCHMARK_SYMBOL = "^GSPC"                           # S&P 500 index

TRADE_COLUMNS = [
    "ticker",
    "buy_date",
    "sell_date",
    "holding_days",
    "closed_reason",
    "buy_score_raw",
    "buy_price",
    "sell_price",
    "stock_return_pct",
    "benchmark_return_pct",
    "alpha_pct",
]


def simulate_trades(
    frame: pd.DataFrame,
    *,
    min_score_raw: float | None = None,
    open_filter=None,
    open_ratings=OPEN_RATINGS,
    close_ratings=CLOSE_RATINGS,
    close_filter=None,
    max_hold_days: int | None = None,
) -> pd.DataFrame:
    """Replay a signal-driven long strategy over the trade log (no prices).

    Walks the signal dates in chronological order and returns one row per closed
    trade with columns ``ticker``, ``buy_date``, ``sell_date``, ``buy_score_raw``
    and ``closed_reason``.

    Entry — a flat name opens on day ``d`` when **all** hold:
      * its recommendation is in ``open_ratings`` (pass ``None`` to allow *any*
        rating — e.g. a contrarian strategy that buys SELL-rated washouts),
      * ``score_raw >= min_score_raw`` if a floor is given,
      * ``open_filter(row)`` is True if a filter is given.

    Exit — a held name closes on the first day **any** of these hold:
      * its recommendation is in ``close_ratings`` (``None`` disables rating exits)
        → ``closed_reason="sell_signal"``,
      * ``close_filter(row)`` is True → ``closed_reason="signal_exit"``,
      * it has been held ``max_hold_days`` calendar days or more (a time stop)
        → ``closed_reason="max_hold"``.
    Anything still open on the last date is closed there (``"end_of_data"``).

    ``open_filter`` / ``close_filter`` are callables taking the signal row (a
    namedtuple with every feature column, e.g. ``row.macd``, ``row.rsi``). The
    defaults (``open_ratings={STRONG_BUY}``, ``close_ratings={SELL, STRONG_SELL}``,
    no filters, no time stop) reproduce the original STRONG_BUY strategy exactly.
    Positions are equal-weight and binary; a ticker can be re-entered after exit.
    """
    cols = ["ticker", "buy_date", "sell_date", "buy_score_raw", "closed_reason"]
    if frame.empty:
        return pd.DataFrame(columns=cols)

    dates = sorted(frame["date"].unique())
    last_date = dates[-1]
    by_date = {d: g for d, g in frame.groupby("date")}

    holdings: dict[str, tuple[pd.Timestamp, float]] = {}  # ticker -> (buy_date, buy_score_raw)
    trades: list[dict] = []

    def _close(ticker: str, sell_date, reason: str) -> None:
        buy_date, buy_score_raw = holdings.pop(ticker)
        trades.append({
            "ticker": ticker, "buy_date": buy_date, "sell_date": sell_date,
            "buy_score_raw": buy_score_raw, "closed_reason": reason,
        })

    for d in dates:
        day = by_date[d]
        closed_today: set[str] = set()

        # Time stop first: close anything held at/over the max hold, regardless of
        # whether the name prints a signal today.
        if max_hold_days is not None:
            for ticker in list(holdings):
                if (d - holdings[ticker][0]).days >= max_hold_days:
                    _close(ticker, d, "max_hold")
                    closed_today.add(ticker)

        # Signal-driven exits.
        for row in day.itertuples(index=False):
            if row.ticker not in holdings or row.ticker in closed_today:
                continue
            if close_ratings is not None and row.recommendation in close_ratings:
                _close(row.ticker, d, "sell_signal")
                closed_today.add(row.ticker)
            elif close_filter is not None and close_filter(row):
                _close(row.ticker, d, "signal_exit")
                closed_today.add(row.ticker)

        # Entries — names we do not already hold and did not just exit today.
        for row in day.itertuples(index=False):
            if (
                row.ticker not in holdings
                and row.ticker not in closed_today
                and (open_ratings is None or row.recommendation in open_ratings)
                and (min_score_raw is None or row.score_raw >= min_score_raw)
                and (open_filter is None or open_filter(row))
            ):
                holdings[row.ticker] = (d, row.score_raw)

    # Force-close whatever is still open at the last date in the data.
    for ticker in list(holdings):
        _close(ticker, last_date, "end_of_data")

    result = pd.DataFrame(trades, columns=cols)
    return result.sort_values(["buy_date", "ticker"]).reset_index(drop=True)


def load_prices_yfinance(
    tickers: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[dict[str, pd.Series], pd.Series]:
    """Download adjusted-close price history from Yahoo Finance, once.

    Returns ``(stock_prices, benchmark)`` where ``stock_prices`` maps each ticker
    to its adjusted-close :class:`pandas.Series` and ``benchmark`` is the S&P 500
    (``^GSPC``) adjusted-close series covering ``[start, end]``. All stock tickers
    are pulled in a single batched request and the benchmark in one more; nothing
    is re-downloaded per date. Raises ``RuntimeError`` if Yahoo returns nothing
    (e.g. the egress network policy blocks ``*.yahoo.com``).
    """
    import yfinance as yf  # imported lazily so parsing works without the dep

    # yfinance treats ``end`` as exclusive; pad by a day to include the last date.
    dl_start = pd.Timestamp(start).strftime("%Y-%m-%d")
    dl_end = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    raw = yf.download(
        tickers,
        start=dl_start,
        end=dl_end,
        auto_adjust=True,          # "Close" is split/dividend adjusted
        progress=False,
        group_by="ticker",
        threads=True,
    )
    stock_prices: dict[str, pd.Series] = {}
    if isinstance(raw.columns, pd.MultiIndex):
        available = set(raw.columns.get_level_values(0))
        for ticker in tickers:
            if ticker in available and "Close" in raw[ticker]:
                series = raw[ticker]["Close"].dropna()
                if not series.empty:
                    stock_prices[ticker] = series
    elif not raw.empty and "Close" in raw:  # single-ticker shape
        series = raw["Close"].dropna()
        if not series.empty:
            stock_prices[tickers[0]] = series

    if not stock_prices:
        raise RuntimeError(
            "No stock price data returned from Yahoo Finance. In this session the "
            "egress policy blocks *.yahoo.com (403 CONNECT); run where Yahoo is "
            "reachable, or pass a custom price_loader."
        )

    bench_raw = yf.download(
        BENCHMARK_SYMBOL,
        start=dl_start,
        end=dl_end,
        auto_adjust=True,
        progress=False,
    )
    benchmark = bench_raw["Close"].dropna()
    if isinstance(benchmark, pd.DataFrame):  # single-symbol frames can nest
        benchmark = benchmark.iloc[:, 0]
    if benchmark.empty:
        raise RuntimeError(f"No benchmark price data returned for {BENCHMARK_SYMBOL}.")

    return stock_prices, benchmark


def _price_asof(series: pd.Series, when: pd.Timestamp) -> float:
    """Adjusted close on ``when``, or the last trading day before it (NaN if none)."""
    if series is None or series.empty:
        return float("nan")
    value = series.asof(pd.Timestamp(when))
    return float(value) if pd.notna(value) else float("nan")


def analyze_performance(
    frame: pd.DataFrame | None = None,
    *,
    min_score_raw: float | None = None,
    open_filter=None,
    open_ratings=OPEN_RATINGS,
    close_ratings=CLOSE_RATINGS,
    close_filter=None,
    max_hold_days: int | None = None,
    price_loader=load_prices_yfinance,
) -> pd.DataFrame:
    """Backtest a signal-driven long strategy and return per-trade returns/alpha.

    ``frame`` defaults to :func:`build_trade_log`. ``min_score_raw``,
    ``open_filter``, ``open_ratings``, ``close_ratings``, ``close_filter`` and
    ``max_hold_days`` are forwarded to :func:`simulate_trades` — so beyond the
    default STRONG_BUY strategy you can, for example, buy any oversold washout
    (``open_ratings=None`` + an ``open_filter``) and exit on a feature or a time
    stop. ``price_loader`` is any callable
    ``(tickers, start, end) -> (dict[str, Series], Series)`` returning stock and
    benchmark adjusted-close prices; it defaults to :func:`load_prices_yfinance`
    and is downloaded exactly once. The returned DataFrame has one row per closed
    trade with the columns listed in :data:`TRADE_COLUMNS`, including
    ``buy_score_raw`` (the opening signal's uncapped score), ``stock_return_pct``,
    ``benchmark_return_pct`` and ``alpha_pct`` (= stock return − benchmark return
    over the same buy→sell window).
    """
    if frame is None:
        frame = build_trade_log()

    trades = simulate_trades(
        frame, min_score_raw=min_score_raw, open_filter=open_filter,
        open_ratings=open_ratings, close_ratings=close_ratings,
        close_filter=close_filter, max_hold_days=max_hold_days,
    )
    if trades.empty:
        return pd.DataFrame(columns=TRADE_COLUMNS)

    tickers = sorted(trades["ticker"].unique())
    stock_prices, benchmark = price_loader(tickers, frame["date"].min(), frame["date"].max())

    records: list[dict] = []
    for row in trades.itertuples(index=False):
        prices = stock_prices.get(row.ticker)
        buy_price = _price_asof(prices, row.buy_date)
        sell_price = _price_asof(prices, row.sell_date)
        bench_buy = _price_asof(benchmark, row.buy_date)
        bench_sell = _price_asof(benchmark, row.sell_date)

        stock_ret = (sell_price / buy_price - 1.0) * 100.0 if buy_price else float("nan")
        bench_ret = (bench_sell / bench_buy - 1.0) * 100.0 if bench_buy else float("nan")

        records.append({
            "ticker": row.ticker,
            "buy_date": row.buy_date,
            "sell_date": row.sell_date,
            "holding_days": (row.sell_date - row.buy_date).days,
            "closed_reason": row.closed_reason,
            "buy_score_raw": row.buy_score_raw,
            "buy_price": round(buy_price, 4) if pd.notna(buy_price) else float("nan"),
            "sell_price": round(sell_price, 4) if pd.notna(sell_price) else float("nan"),
            "stock_return_pct": round(stock_ret, 4) if pd.notna(stock_ret) else float("nan"),
            "benchmark_return_pct": round(bench_ret, 4) if pd.notna(bench_ret) else float("nan"),
            "alpha_pct": round(stock_ret - bench_ret, 4) if pd.notna(stock_ret) and pd.notna(bench_ret) else float("nan"),
        })

    return pd.DataFrame(records, columns=TRADE_COLUMNS)


def _print_performance(trades: pd.DataFrame) -> None:
    """Print a compact summary of the backtest trade table."""
    if trades.empty:
        print("\nBacktest: 0 trades (no entries matched).")
        return
    reasons = ", ".join(f"{n} {r}" for r, n in trades["closed_reason"].value_counts().items())
    print(f"\nBacktest: {len(trades)} trades ({reasons})")
    priced = trades.dropna(subset=["alpha_pct"])
    if priced.empty:
        print("No priced trades (price data unavailable).")
        return
    if len(priced) < len(trades):
        print(f"Priced trades: {len(priced)} (dropped {len(trades) - len(priced)} without price data)")
    win_rate = (priced["stock_return_pct"] > 0).mean() * 100.0
    beat_rate = (priced["alpha_pct"] > 0).mean() * 100.0
    print(f"Avg stock return: {priced['stock_return_pct'].mean():+.2f}%  |  "
          f"Avg benchmark:  {priced['benchmark_return_pct'].mean():+.2f}%  |  "
          f"Avg alpha: {priced['alpha_pct'].mean():+.2f}%")
    print(f"Median alpha: {priced['alpha_pct'].median():+.2f}%  |  "
          f"Win rate: {win_rate:.1f}%  |  Beat-benchmark rate: {beat_rate:.1f}%  |  "
          f"Avg hold: {priced['holding_days'].mean():.1f}d")


# --------------------------------------------------------------------------- #
# Signal-predictiveness analysis
#
# Question: which technical signals predict a *positive forward alpha*,
# irrespective of the recommendation on the ticket? We attach a fixed-horizon
# forward alpha to every signal row (not just STRONG_BUY entries), then (a) build
# a univariate table of mean alpha / win-rate per feature value, and (b) fit a
# multivariate logistic regression of P(alpha > 0) on the features.
# --------------------------------------------------------------------------- #

DEFAULT_HORIZON = 20  # trading days from the signal date to the forward outcome


def compute_forward_alpha(
    frame: pd.DataFrame,
    *,
    horizon: int = DEFAULT_HORIZON,
    price_loader=load_prices_yfinance,
) -> pd.DataFrame:
    """Attach a fixed-horizon forward alpha to every signal row in ``frame``.

    For each signal, the forward window runs ``horizon`` *trading days* from the
    signal date (measured on the stock's own price calendar). ``fwd_return_pct``
    is the stock's adjusted-close return over that window, ``benchmark_fwd_pct``
    is the S&P 500 (``^GSPC``) return over the identical calendar window, and
    ``alpha_pct`` is their difference. Rows whose forward window extends past the
    available price history (i.e. the most recent ``horizon`` trading days, which
    have no future data yet) get NaN and are meant to be dropped by callers.

    Prices for *all* tickers plus the benchmark are downloaded once via
    ``price_loader``; the download window is extended past the last signal date so
    the forward returns can be computed.
    """
    frame = frame.copy()
    tickers = sorted(frame["ticker"].unique())
    start = frame["date"].min()
    # Pad the end generously so ``horizon`` trading days of forward prices exist.
    end = frame["date"].max() + pd.Timedelta(days=horizon * 2 + 7)
    stock_prices, benchmark = price_loader(tickers, start, end)

    fwd_return = np.full(len(frame), np.nan)
    bench_fwd = np.full(len(frame), np.nan)
    fwd_dates: list[pd.Timestamp | None] = [None] * len(frame)

    for i, row in enumerate(frame.itertuples(index=False)):
        prices = stock_prices.get(row.ticker)
        if prices is None or prices.empty:
            continue
        idx = prices.index
        pos = idx.searchsorted(pd.Timestamp(row.date), side="right") - 1
        if pos < 0:
            continue  # signal predates this ticker's price history
        fwd_pos = pos + horizon
        if fwd_pos >= len(idx):
            continue  # forward window runs past available (future) data
        entry_date, exit_date = idx[pos], idx[fwd_pos]
        entry_price, exit_price = prices.iloc[pos], prices.iloc[fwd_pos]
        if not entry_price:
            continue
        fwd_return[i] = (exit_price / entry_price - 1.0) * 100.0
        bench_buy = _price_asof(benchmark, entry_date)
        bench_sell = _price_asof(benchmark, exit_date)
        if bench_buy:
            bench_fwd[i] = (bench_sell / bench_buy - 1.0) * 100.0
        fwd_dates[i] = exit_date

    frame["fwd_date"] = fwd_dates
    frame["fwd_return_pct"] = fwd_return
    frame["benchmark_fwd_pct"] = bench_fwd
    frame["alpha_pct"] = fwd_return - bench_fwd
    return frame


def _univariate_alpha(data: pd.DataFrame, min_support: int) -> pd.DataFrame:
    """Mean forward alpha and win-rate for each feature value.

    Categorical features are grouped by level; numeric features are split at their
    median into ``<= median`` / ``> median`` buckets. Only cells with at least
    ``min_support`` observations are kept. Sorted by mean alpha, descending.
    """
    rows: list[dict] = []
    for feature in CATEGORICAL_FEATURES:
        for value, grp in data.groupby(feature):
            if len(grp) >= min_support:
                rows.append({
                    "feature": feature, "value": str(value), "n": len(grp),
                    "mean_alpha_pct": grp["alpha_pct"].mean(),
                    "win_rate_pct": (grp["alpha_pct"] > 0).mean() * 100.0,
                })
    for feature in NUMERIC_FEATURES:
        series = data[feature]
        if series.notna().sum() < 2 * min_support or series.nunique(dropna=True) < 2:
            continue
        median = series.median()
        for label, mask in ((f"<= {median:g}", series <= median), (f"> {median:g}", series > median)):
            grp = data[mask]
            if len(grp) >= min_support:
                rows.append({
                    "feature": feature, "value": label, "n": len(grp),
                    "mean_alpha_pct": grp["alpha_pct"].mean(),
                    "win_rate_pct": (grp["alpha_pct"] > 0).mean() * 100.0,
                })
    table = pd.DataFrame(rows, columns=["feature", "value", "n", "mean_alpha_pct", "win_rate_pct"])
    return table.sort_values("mean_alpha_pct", ascending=False).reset_index(drop=True)


def _design_matrix(data: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Build a logistic-regression design matrix from the signal features.

    Categoricals are one-hot encoded dropping the most common level as the
    reference; numerics are mean-imputed then standardised (z-scored) so their
    coefficients are comparable per one standard deviation. Returns the matrix
    (without an intercept column) and the matching term names.
    """
    columns: list[np.ndarray] = []
    names: list[str] = []
    for feature in CATEGORICAL_FEATURES:
        counts = data[feature].value_counts()
        reference = counts.index[0]
        for level in counts.index[1:]:
            columns.append((data[feature] == level).to_numpy(dtype=float))
            names.append(f"{feature}={level} (vs {reference})")
    for feature in NUMERIC_FEATURES:
        values = data[feature].to_numpy(dtype=float)
        mean = np.nanmean(values)
        values = np.where(np.isnan(values), mean, values)
        std = values.std()
        if std == 0:
            continue  # constant column carries no information
        columns.append((values - values.mean()) / std)
        names.append(f"{feature} (per +1 SD)")
    matrix = np.column_stack(columns) if columns else np.empty((len(data), 0))
    return matrix, names


def _fit_logistic(
    X: np.ndarray,
    y: np.ndarray,
    *,
    ridge: float = 1e-3,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit L2-regularised logistic regression by IRLS (Newton-Raphson).

    A small ``ridge`` keeps the Hessian invertible under collinear or perfectly
    separated features. Returns ``(beta, std_err)`` including the intercept as the
    first element; standard errors come from the inverse ridge-penalised Hessian
    (approximate, but adequate for ranking predictiveness).
    """
    n, p = X.shape
    Xi = np.column_stack([np.ones(n), X])  # prepend intercept
    beta = np.zeros(p + 1)
    penalty = ridge * np.eye(p + 1)
    penalty[0, 0] = 0.0  # never penalise the intercept
    for _ in range(max_iter):
        eta = Xi @ beta
        mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
        W = np.clip(mu * (1.0 - mu), 1e-9, None)
        gradient = Xi.T @ (y - mu) - penalty @ beta
        hessian = Xi.T @ (Xi * W[:, None]) + penalty
        step = np.linalg.solve(hessian, gradient)
        beta += step
        if np.max(np.abs(step)) < tol:
            break
    eta = Xi @ beta
    mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
    W = np.clip(mu * (1.0 - mu), 1e-9, None)
    cov = np.linalg.pinv(Xi.T @ (Xi * W[:, None]) + penalty)
    std_err = np.sqrt(np.clip(np.diag(cov), 0, None))
    return beta, std_err


def _normal_sf(z: float) -> float:
    """Two-sided normal survival tail 2·P(Z > |z|), via the stdlib erfc."""
    return math.erfc(abs(z) / math.sqrt(2.0))


def analyze_signal_predictiveness(
    frame: pd.DataFrame | None = None,
    *,
    horizon: int = DEFAULT_HORIZON,
    min_support: int = 30,
    price_loader=load_prices_yfinance,
) -> dict:
    """Find which signals predict positive forward alpha, ignoring the rating.

    Attaches a ``horizon``-day forward alpha to every signal (via
    :func:`compute_forward_alpha`), then returns a dict with:

    * ``"data"``    – the per-signal frame with ``alpha_pct`` (NaN rows dropped),
    * ``"base_rate"``, ``"n_used"`` – overall win-rate and sample size,
    * ``"univariate"`` – mean alpha / win-rate per feature value
      (:func:`_univariate_alpha`),
    * ``"logit"``   – a logistic regression of P(alpha > 0) on all features,
      one row per term with ``coef``, ``odds_ratio``, ``std_err``, ``z`` and
      ``p_value``, sorted by significance.

    ``price_loader`` is injectable for offline testing.
    """
    if frame is None:
        frame = build_trade_log()

    enriched = compute_forward_alpha(frame, horizon=horizon, price_loader=price_loader)
    data = enriched.dropna(subset=["alpha_pct"]).reset_index(drop=True)
    result: dict = {
        "horizon": horizon,
        "data": data,
        "n_used": len(data),
        "n_total": len(frame),
        "base_rate": (data["alpha_pct"] > 0).mean() * 100.0 if len(data) else float("nan"),
        "univariate": _univariate_alpha(data, min_support) if len(data) else pd.DataFrame(),
        "logit": pd.DataFrame(),
    }
    if len(data) < 50:
        return result  # too few rows for a stable regression

    y = (data["alpha_pct"] > 0).to_numpy(dtype=float)
    X, names = _design_matrix(data)
    beta, std_err = _fit_logistic(X, y)
    terms = ["intercept"] + names
    z_scores = np.divide(beta, std_err, out=np.zeros_like(beta), where=std_err > 0)
    logit = pd.DataFrame({
        "term": terms,
        "coef": beta,
        "odds_ratio": np.exp(beta),
        "std_err": std_err,
        "z": z_scores,
        "p_value": [_normal_sf(z) for z in z_scores],
    })
    logit = logit[logit["term"] != "intercept"].copy()
    logit["abs_z"] = logit["z"].abs()
    result["logit"] = logit.sort_values("abs_z", ascending=False).drop(columns="abs_z").reset_index(drop=True)
    return result


def _print_predictiveness(result: dict, *, top: int = 15) -> None:
    """Print the univariate table and logistic-regression ranking."""
    print(f"\nSignal predictiveness ({result['horizon']}-day forward alpha vs ^GSPC)")
    print(f"Signals used: {result['n_used']} of {result['n_total']} "
          f"(rest lack a full forward window or price data)")
    if not result["n_used"]:
        print("No usable signals — price data unavailable for the forward window.")
        return
    print(f"Base rate (share with positive alpha): {result['base_rate']:.1f}%")

    uni = result["univariate"]
    if not uni.empty:
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(f"\nMost bullish feature values (top {top}, by mean forward alpha):")
            print(uni.head(top).to_string(index=False))
            print(f"\nMost bearish feature values (bottom {top}):")
            print(uni.tail(top).iloc[::-1].to_string(index=False))

    logit = result["logit"]
    if not logit.empty:
        with pd.option_context("display.max_columns", None, "display.width", 200,
                               "display.float_format", lambda v: f"{v:.4f}"):
            print(f"\nLogistic regression — P(alpha>0) drivers, ranked by |z| (top {top}):")
            print(logit.head(top).to_string(index=False))
        print("\nRead: coef>0 (odds_ratio>1) raises the odds of positive forward alpha; "
              "numeric terms are per +1 SD; categoricals are vs the reference level. "
              "A mild L2 penalty is applied, so treat p-values as approximate and "
              "beware collinear signals.")


_FILTER_OPS = ("!=", ">=", "<=", "=", ">", "<")  # longest-match first


def build_open_filter(specs: list[str]):
    """Turn ``col=VAL`` / ``col<N`` style strings into an ``open_filter`` callable.

    Each spec compares a feature column against a value; all specs are ANDed. For
    ``=`` / ``!=`` the value is matched as a string and a comma-separated list is
    treated as a set membership test (e.g. ``cmf=ACCUM,STRONG_ACCUM``). For
    ``<``, ``<=``, ``>``, ``>=`` the value is numeric (e.g. ``rsi<50``). Returns
    None for an empty list. Raises ``ValueError`` on an unparseable spec or an
    unknown column.
    """
    if not specs:
        return None
    valid = set(FEATURE_COLUMNS) | {"score_raw", "score_capped", "confidence_pct", "recommendation"}
    predicates: list = []
    for spec in specs:
        for op in _FILTER_OPS:
            if op in spec:
                col, raw = spec.split(op, 1)
                col, raw = col.strip(), raw.strip()
                break
        else:
            raise ValueError(f"Cannot parse filter '{spec}' (expected e.g. macd=BULLISH or rsi<50)")
        if col not in valid:
            raise ValueError(f"Unknown filter column '{col}'. Valid columns: {sorted(valid)}")
        if op in ("=", "!="):
            allowed = {v.strip() for v in raw.split(",")}
            predicates.append((lambda row, c=col, a=allowed, negate=(op == "!="):
                               (str(getattr(row, c)) in a) != negate))
        else:
            threshold = float(raw)
            comparisons = {
                ">=": lambda x, t: x >= t, "<=": lambda x, t: x <= t,
                ">": lambda x, t: x > t, "<": lambda x, t: x < t,
            }
            predicates.append((lambda row, c=col, t=threshold, cmp=comparisons[op]:
                               pd.notna(getattr(row, c)) and cmp(float(getattr(row, c)), t)))
    return lambda row: all(pred(row) for pred in predicates)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--down-dir", type=Path, default=DEFAULT_DOWN_DIR)
    parser.add_argument("--up-dir", type=Path, default=DEFAULT_UP_DIR)
    parser.add_argument(
        "--rows",
        type=int,
        default=10,
        help="How many preview rows to print (default: 10).",
    )
    parser.add_argument(
        "--performance",
        action="store_true",
        help=(
            "Also run the price-based backtest (downloads prices from Yahoo "
            "Finance via yfinance) and print per-trade returns and alpha."
        ),
    )
    parser.add_argument(
        "--min-score-raw",
        type=float,
        default=None,
        help=(
            "Only open a position when a STRONG_BUY signal's score_raw is at "
            "least this value (default: no floor, every STRONG_BUY qualifies)."
        ),
    )
    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        metavar="COL[op]VAL",
        dest="filters",
        help=(
            "Gate backtest entries on a signal feature; repeatable, ANDed. "
            "Examples: --filter macd=BULLISH --filter cmf=ACCUM,STRONG_ACCUM "
            "--filter rsi<50"
        ),
    )
    parser.add_argument(
        "--regress",
        action="store_true",
        help=(
            "Run the signal-predictiveness analysis: attach a forward alpha to "
            "every signal and report which features predict positive alpha "
            "(univariate table + logistic regression). Downloads prices."
        ),
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_HORIZON,
        help=f"Forward-alpha horizon in trading days for --regress (default: {DEFAULT_HORIZON}).",
    )
    parser.add_argument(
        "--min-support",
        type=int,
        default=30,
        help="Minimum observations per cell in the univariate table (default: 30).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    frame = build_trade_log(args.down_dir, args.up_dir)

    print(f"Trade log: {len(frame)} signals "
          f"({(frame['watchlist'] == 'down').sum()} down, "
          f"{(frame['watchlist'] == 'up').sum()} up)")
    if not frame.empty:
        print(f"Date range: {frame['date'].min().date()} → {frame['date'].max().date()}")
        print("\nBy recommendation × watchlist:")
        print(frame.groupby(["watchlist", "recommendation"]).size().to_string())
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(f"\nPreview (first {args.rows} rows):")
            print(frame.head(args.rows).to_string(index=False))

    if frame.empty:
        return 0

    try:
        open_filter = build_open_filter(args.filters)
    except ValueError as exc:
        print(f"Bad --filter: {exc}", file=sys.stderr)
        return 2

    if args.performance:
        gates = ["STRONG_BUY"]
        if args.min_score_raw is not None:
            gates.append(f"score_raw >= {args.min_score_raw:g}")
        gates.extend(args.filters)
        print(f"\nBacktest entry gate: {' and '.join(gates)}")
        try:
            trades = analyze_performance(
                frame, min_score_raw=args.min_score_raw, open_filter=open_filter
            )
        except Exception as exc:  # noqa: BLE001 - surface network/price issues clearly
            print(f"\nBacktest failed: {exc}", file=sys.stderr)
            return 1
        _print_performance(trades)
        if not trades.empty:
            with pd.option_context("display.max_columns", None, "display.width", 200):
                print(f"\nTrades (first {args.rows} rows):")
                print(trades.head(args.rows).to_string(index=False))

    if args.regress:
        try:
            result = analyze_signal_predictiveness(
                frame, horizon=args.horizon, min_support=args.min_support
            )
        except Exception as exc:  # noqa: BLE001 - surface network/price issues clearly
            print(f"\nRegression failed: {exc}", file=sys.stderr)
            return 1
        _print_predictiveness(result)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
