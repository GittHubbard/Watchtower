"""Backtest the Stage 5.5 technical score on multi-year daily OHLCV.

Runs the production scorer (``stage5_5_context._evaluate_technical``) on a
trailing window every ``--step`` bars for every ticker, records each score
component, and attaches forward returns and alpha vs SPY. Because every
component is stored, the legacy momentum score is reconstructed alongside the
current setup score so the two can be compared on identical samples.

Input: a directory of ``<TICKER>_10y.parquet`` files (Open/High/Low/Close/Volume,
date index) — e.g. MegaApp's ``scanner/backtest/cache``.

    python research/backtest_setup_score.py --cache /path/to/cache --out samples.parquet
    python research/backtest_setup_score.py --report samples.parquet
"""
from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gamma_workflow import stage5_5_context as s55  # noqa: E402

WINDOW = 290          # ≈ what Stage 5 fetches (260 calendar days × 1.6)
HORIZONS = (5, 20, 60)
BENCH = "SPY"


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(pd.Index(df.index).date)
    return df[["Open", "High", "Low", "Close", "Volume"]].astype(float).dropna(subset=["Close"])


def _ticker_samples(args) -> list[dict]:
    path, step = args
    ticker = path.name.split("_")[0]
    df = _load(path)
    close = df["Close"]
    rows = []
    for end in range(WINDOW, len(df), step):
        h = df.iloc[end - WINDOW:end]
        asof = h.index[-1]
        try:
            ev = s55._evaluate_technical(h, asof.date(), "")
        except Exception:
            continue
        if ev is None:
            continue
        sc = ev["scoring"]
        pen = dict((k, v) for k, v in sc["penalties"])
        row = {
            "ticker": ticker, "date": asof, "close": ev["close_cur"],
            "score": sc["score"], "pullback": sc["pullback"], "pull_ma": sc["pull_ma"],
            "pull_rsi": sc["pull_rsi"], "overext": sc["overext"], "n_penalties": len(pen),
            "confirm": sc["confirm"], "div_pts": sc["div_pts"], "momentum": sc["momentum"],
            "ext_atr": sc["ext_atr"], "ma50_dist": sc["ma50_dist"],
            "base_trend": sc["base_trend"], "div_raw": ev["div"]["raw"],
            "rsi": ev["rsi"], "adx": ev["adx"], "bb_upper": ev["bb_upper"], "bb_lower": ev["bb_lower"],
            "sma200": ev["sma200"], "t_price": ev["t_price"], "t_macd": ev["t_macd"],
        }
        for n in HORIZONS:
            j = end - 1 + n
            row[f"fwd_{n}d"] = float(close.iloc[j] / close.iloc[end - 1] - 1) if j < len(df) else np.nan
        rows.append(row)
    return rows


def legacy_score(s: pd.DataFrame) -> pd.Series:
    """Reconstruct the pre-redesign score from stored components (IV regime
    unknown in history → TRENDING 0.0 / WEAK_TREND 0.5 / RANGING 1.0 MR weight)."""
    rsi = s["rsi"]
    mr = np.select([rsi <= 20, rsi < 30, rsi >= 80, rsi > 70], [-0.5, 1.0, 0.5, -1.0], 0.0)
    mr = mr + np.select([s["close"] > s["bb_upper"], s["close"] < s["bb_lower"]], [-0.5, 0.5], 0.0)
    w = np.select([s["adx"] >= 25, s["adx"] >= 20], [0.0, 0.5], 1.0)
    return (s["base_trend"] + w * mr + s["div_raw"]).clip(-10, 10)


def rate(score: pd.Series) -> pd.Series:
    return pd.Series(np.select(
        [score >= 5.5, score >= 2.5, score <= -5.5, score <= -2.5],
        ["STRONG_BUY", "BUY", "STRONG_SELL", "SELL"], "HOLD"), index=score.index)


def build(cache: Path, out: Path, step: int, procs: int) -> pd.DataFrame:
    paths = sorted(cache.glob("*_10y.parquet"))
    with Pool(procs) as pool:
        rows = [r for chunk in pool.imap_unordered(_ticker_samples, [(p, step) for p in paths]) for r in chunk]
    s = pd.DataFrame(rows)
    bench = _load(cache / f"{BENCH}_10y.parquet")["Close"]
    for n in HORIZONS:
        bf = bench.shift(-n) / bench - 1
        s[f"alpha_{n}d"] = s[f"fwd_{n}d"] - s["date"].map(bf)
    s["legacy_score"] = legacy_score(s)
    s.to_parquet(out, index=False)
    return s


def report(s: pd.DataFrame, horizon: int = 20) -> None:
    y = f"alpha_{horizon}d"
    s = s.dropna(subset=[y]).copy()
    s["year"] = s["date"].dt.year
    # Excess vs the average stock on the same date removes market-wide moves.
    s["excess"] = s[y] - s.groupby("date")[y].transform("mean")

    def ic(col):
        return s.groupby("date").apply(lambda g: g[col].rank().corr(g[y].rank())).mean()

    print(f"\n{horizon}-day horizon, {len(s):,} samples, "
          f"{s['date'].min().date()}..{s['date'].max().date()}")
    for col, label in (("legacy_score", "OLD momentum score"), ("score", "NEW setup score")):
        r = rate(s[col])
        print(f"\n{label}: rank-IC {ic(col):+.3f}")
        t = s.groupby(r)["excess"].agg(["count", "mean", lambda x: (x > 0).mean()])
        t.columns = ["n", "excess_vs_avg_stock", "hit_rate"]
        print(t.reindex(["STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"]).round(4).to_string())
        buys = r.isin(["BUY", "STRONG_BUY"])
        print("  Buy+StrongBuy excess by year:",
              s[buys].groupby("year")["excess"].mean().mul(100).round(2).to_dict())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path)
    ap.add_argument("--out", type=Path, default=Path("samples.parquet"))
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--report", type=Path)
    a = ap.parse_args()
    if a.report:
        smp = pd.read_parquet(a.report)
    else:
        smp = build(a.cache, a.out, a.step, a.procs)
    for hz in HORIZONS:
        report(smp, hz)
