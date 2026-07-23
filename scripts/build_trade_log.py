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

``analyze_performance(frame=None, price_loader=...) -> pandas.DataFrame``
    Runs a long-only backtest over the trade log and returns one row per closed
    trade with its % return and alpha vs the S&P 500. Strategy:

    * Start flat on the first date in the data.
    * Each day, open a position in any ticker rated ``STRONG_BUY`` that we do
      not already hold (re-entry allowed after a prior exit).
    * Close a held position the first day the ticker is rated ``SELL`` or
      ``STRONG_SELL``.
    * Any position still open on the last date is closed at that date's price.

    For every closed trade the stock's % return (adjusted close, buy date to
    sell date) is compared against the S&P 500 (``^GSPC``) over the *same* date
    window; ``alpha_pct`` is the difference. Prices come from Yahoo Finance via
    yfinance and are downloaded once (all traded tickers in one batch, the
    benchmark once) and cached in memory. The fetch is injectable via
    ``price_loader`` so it can be stubbed for testing or pointed elsewhere.

Run as a script (``python scripts/build_trade_log.py``) to print a short summary
and preview of the signal log; no file is written. Add ``--performance`` to also
run the backtest (this is the step that downloads prices from Yahoo Finance).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

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

COLUMNS = [
    "ticker",
    "date",
    "watchlist",
    "recommendation",
    "score_capped",
    "score_raw",
    "confidence_pct",
]


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

    header = _HEADER_RE.search(text)
    if not header:
        raise ValueError(f"Missing Recommendation/Score header in {path}")
    recommendation = header.group(1)
    score_capped = float(header.group(2))

    confidence = _CONFIDENCE_RE.search(text)
    if not confidence:
        raise ValueError(f"Missing confidence line in {path}")
    confidence_pct = int(confidence.group(1))

    final = _FINAL_SCORE_RE.search(text)
    if not final:
        raise ValueError(f"Missing final score breakdown in {path}")
    score_raw = float(final.group(1))

    return {
        "ticker": ticker,
        "date": date_str,
        "watchlist": watchlist,
        "recommendation": recommendation,
        "score_capped": score_capped,
        "score_raw": score_raw,
        "confidence_pct": confidence_pct,
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
    "buy_price",
    "sell_price",
    "stock_return_pct",
    "benchmark_return_pct",
    "alpha_pct",
]


def simulate_trades(frame: pd.DataFrame) -> pd.DataFrame:
    """Replay the STRONG_BUY / SELL strategy over the trade log (no prices).

    Walks the signal dates in chronological order and returns one row per closed
    trade with columns ``ticker``, ``buy_date``, ``sell_date`` and
    ``closed_reason`` (``"sell_signal"`` or ``"end_of_data"``). Positions are
    equal-weight and binary (held or not); a ticker can be re-entered after it
    has been exited. Prices are intentionally *not* touched here so the trade
    timeline can be tested without any network access.
    """
    if frame.empty:
        return pd.DataFrame(columns=["ticker", "buy_date", "sell_date", "closed_reason"])

    dates = sorted(frame["date"].unique())
    last_date = dates[-1]
    by_date = {d: g for d, g in frame.groupby("date")}

    holdings: dict[str, pd.Timestamp] = {}  # ticker -> buy_date
    trades: list[dict] = []

    for d in dates:
        day = by_date[d]
        closed_today: set[str] = set()

        # Close first: any held name that crossed into SELL / STRONG_SELL today.
        for ticker in day.loc[day["recommendation"].isin(CLOSE_RATINGS), "ticker"]:
            if ticker in holdings:
                trades.append({
                    "ticker": ticker,
                    "buy_date": holdings.pop(ticker),
                    "sell_date": d,
                    "closed_reason": "sell_signal",
                })
                closed_today.add(ticker)

        # Then open: STRONG_BUY names we do not already hold (and did not just
        # exit today, to avoid same-day churn).
        for ticker in day.loc[day["recommendation"].isin(OPEN_RATINGS), "ticker"]:
            if ticker not in holdings and ticker not in closed_today:
                holdings[ticker] = d

    # Force-close whatever is still open at the last date in the data.
    for ticker, buy_date in holdings.items():
        trades.append({
            "ticker": ticker,
            "buy_date": buy_date,
            "sell_date": last_date,
            "closed_reason": "end_of_data",
        })

    result = pd.DataFrame(trades, columns=["ticker", "buy_date", "sell_date", "closed_reason"])
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
    price_loader=load_prices_yfinance,
) -> pd.DataFrame:
    """Backtest the STRONG_BUY strategy and return per-trade returns and alpha.

    ``frame`` defaults to :func:`build_trade_log`. ``price_loader`` is any callable
    ``(tickers, start, end) -> (dict[str, Series], Series)`` returning stock and
    benchmark adjusted-close prices; it defaults to :func:`load_prices_yfinance`
    and is downloaded exactly once. The returned DataFrame has one row per closed
    trade with the columns listed in :data:`TRADE_COLUMNS`, including
    ``stock_return_pct``, ``benchmark_return_pct`` and ``alpha_pct``
    (= stock return − benchmark return over the same buy→sell window).
    """
    if frame is None:
        frame = build_trade_log()

    trades = simulate_trades(frame)
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
            "buy_price": round(buy_price, 4) if pd.notna(buy_price) else float("nan"),
            "sell_price": round(sell_price, 4) if pd.notna(sell_price) else float("nan"),
            "stock_return_pct": round(stock_ret, 4) if pd.notna(stock_ret) else float("nan"),
            "benchmark_return_pct": round(bench_ret, 4) if pd.notna(bench_ret) else float("nan"),
            "alpha_pct": round(stock_ret - bench_ret, 4) if pd.notna(stock_ret) and pd.notna(bench_ret) else float("nan"),
        })

    return pd.DataFrame(records, columns=TRADE_COLUMNS)


def _print_performance(trades: pd.DataFrame) -> None:
    """Print a compact summary of the backtest trade table."""
    print(f"\nBacktest: {len(trades)} trades "
          f"({(trades['closed_reason'] == 'sell_signal').sum()} on sell signal, "
          f"{(trades['closed_reason'] == 'end_of_data').sum()} closed at last date)")
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

    if frame.empty or not args.performance:
        return 0

    try:
        trades = analyze_performance(frame)
    except Exception as exc:  # noqa: BLE001 - surface network/price issues clearly
        print(f"\nBacktest failed: {exc}", file=sys.stderr)
        return 1

    _print_performance(trades)
    if not trades.empty:
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(f"\nTrades (first {args.rows} rows):")
            print(trades.head(args.rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
