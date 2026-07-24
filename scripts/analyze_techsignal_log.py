"""Backtest + signal regression over the full tech-signal parquet log, offline.

Unlike ``build_trade_log`` (watchlist folders only, ~3.6k routed signals), this
runs on the accumulated ``data/techsignal_log.parquet`` — every ticker every day,
including HOLD (~17.5k signals) — and prices everything from the *local* feature
store (``data/features/date=*.parquet`` ``spot`` column, SPY as the benchmark).
No network is needed, so it works where Yahoo is blocked.

Two analyses (both on the fuller universe):
  * ``--backtest`` — the STRONG_BUY / SELL strategy with per-trade alpha.
  * ``--regress``  — which technical signals predict positive forward alpha,
    irrespective of the recommendation (univariate table + logistic regression).

Caveats worth remembering when reading the output:
  * ``spot`` is a close-ish price, not split/dividend-adjusted (minor over weeks).
  * The panel is ~7 weeks / one market regime; results are in-sample, not OOS.
  * Forward windows overlap and the market co-moves, so regression p-values are
    optimistic — read the signs and rankings, not the exact significance.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gamma_workflow import storage  # noqa: E402
from scripts.build_trade_log import (  # noqa: E402
    OPEN_RATINGS,
    CLOSE_RATINGS,
    analyze_performance,
    analyze_signal_predictiveness,
    build_open_filter,
    build_trade_log_from_log,
    _print_performance,
    _print_predictiveness,
)

BENCHMARK_TICKER = "SPY"  # the pipeline's own default benchmark; ^GSPC isn't in the store


def load_spot_panel(features_dir: Path | None = None) -> pd.DataFrame:
    """Build a wide (date × ticker) spot-price panel from the feature store."""
    features_dir = features_dir or storage.FEATURES_DIR
    frames = []
    for path in sorted(features_dir.glob("date=*.parquet")):
        d = re.search(r"date=(\d{4}-\d{2}-\d{2})", path.name).group(1)
        df = pd.read_parquet(path, columns=["ticker", "spot"])
        df["date"] = pd.Timestamp(d)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No feature files under {features_dir}")
    panel = pd.concat(frames, ignore_index=True)
    return panel.pivot_table(index="date", columns="ticker", values="spot").sort_index()


def make_feature_store_loader(panel: pd.DataFrame, benchmark: str = BENCHMARK_TICKER):
    """A ``price_loader`` closure over the local spot panel (ticker + benchmark series)."""
    if benchmark not in panel.columns:
        raise KeyError(f"Benchmark {benchmark!r} not in the feature store panel")
    bench = panel[benchmark].dropna()

    def loader(tickers, start, end):
        prices = {t: panel[t].dropna() for t in tickers if t in panel.columns}
        return prices, bench

    return loader


# The contrarian preset: buy any oversold washout (RSI < 40, irrespective of the
# ticket's recommendation), exit when it mean-reverts (RSI > 55) or after 15 days.
# This is the "fade weakness" mirror of the STRONG_BUY strategy, encoding the edge
# the regression surfaced (oversold / low-momentum names had positive forward alpha).
CONTRARIAN = {
    "open_ratings": None,                       # any rating can open
    "open_filter": build_open_filter(["rsi<40"]),
    "close_ratings": None,                      # no rating-based exit
    "close_filter": build_open_filter(["rsi>55"]),
    "max_hold_days": 15,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backtest", action="store_true", help="Run the backtest.")
    parser.add_argument("--regress", action="store_true", help="Run the signal-predictiveness regression.")
    parser.add_argument("--contrarian", action="store_true",
                        help="Backtest the contrarian preset (buy RSI<40 washouts, exit RSI>55 or 15d) instead of STRONG_BUY.")
    parser.add_argument("--open-any", action="store_true",
                        help="Allow entries on ANY recommendation (not just STRONG_BUY).")
    parser.add_argument("--open-filter", action="append", default=[], metavar="COL[op]VAL",
                        help="Entry gate on a feature; repeatable, ANDed (e.g. --open-filter rsi<40 --open-filter iv30<30).")
    parser.add_argument("--close-filter", action="append", default=[], metavar="COL[op]VAL",
                        help="Exit gate on a feature; repeatable, ANDed (e.g. --close-filter rsi>55).")
    parser.add_argument("--max-hold", type=int, default=None, help="Time-stop: close a position after N calendar days.")
    parser.add_argument("--no-rating-exit", action="store_true", help="Disable the default SELL/STRONG_SELL exit.")
    parser.add_argument("--horizon", type=int, default=10, help="Forward-alpha horizon in trading days (default 10).")
    parser.add_argument("--min-support", type=int, default=100, help="Min observations per univariate cell (default 100).")
    parser.add_argument("--min-score-raw", type=float, default=None, help="Optional score_raw floor for backtest entries.")
    parser.add_argument("--rows", type=int, default=12, help="Preview/top-N rows to print (default 12).")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not (args.backtest or args.regress):
        args.backtest = args.regress = True  # default: run both

    frame = build_trade_log_from_log(storage.read_techsignal_log())
    loader = make_feature_store_loader(load_spot_panel())
    print(f"Full log: {len(frame)} signals, {frame['ticker'].nunique()} tickers, "
          f"{frame['date'].min().date()} → {frame['date'].max().date()}")
    print("By recommendation: "
          + ", ".join(f"{k}={v}" for k, v in frame['recommendation'].value_counts().items()))

    if args.backtest:
        if args.contrarian:
            print("\nStrategy: CONTRARIAN — open any rating with RSI<40, exit RSI>55 or 15-day time stop.")
            strat = CONTRARIAN
        else:
            strat = {
                "open_ratings": None if args.open_any else OPEN_RATINGS,
                "open_filter": build_open_filter(args.open_filter),
                "close_ratings": None if args.no_rating_exit else CLOSE_RATINGS,
                "close_filter": build_open_filter(args.close_filter),
                "max_hold_days": args.max_hold,
            }
            gate = ("any rating" if strat["open_ratings"] is None else "STRONG_BUY") \
                + (f" + {' & '.join(args.open_filter)}" if args.open_filter else "")
            print(f"\nStrategy: open {gate}")
        trades = analyze_performance(frame, min_score_raw=args.min_score_raw, price_loader=loader, **strat)
        _print_performance(trades)

    if args.regress:
        result = analyze_signal_predictiveness(
            frame, horizon=args.horizon, min_support=args.min_support, price_loader=loader
        )
        _print_predictiveness(result, top=args.rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
