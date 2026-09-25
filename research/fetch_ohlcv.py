"""Download ~6 years of daily OHLCV for every watchlist ticker into one parquet.

Used by the backtest (research/backtest_setup_score.py). Runs in GitHub Actions,
where Yahoo Finance is reachable. Output: research/ohlcv.parquet (long format).
"""
from pathlib import Path

import pandas as pd
import yaml
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research" / "ohlcv.parquet"


def main() -> None:
    wl = yaml.safe_load((ROOT / "config" / "watchlist.yaml").read_text())
    symbols = sorted({t["symbol"] for t in wl["tickers"]})
    frames = []
    for i in range(0, len(symbols), 50):
        chunk = symbols[i:i + 50]
        df = yf.download(chunk, period="6y", auto_adjust=True, group_by="ticker",
                         threads=True, progress=False)
        for s in chunk:
            try:
                sub = df[s].dropna(subset=["Close"])
            except KeyError:
                continue
            if sub.empty:
                continue
            sub = sub[["Open", "High", "Low", "Close", "Volume"]].astype("float32")
            sub.index = pd.to_datetime(sub.index.date)
            frames.append(sub.assign(ticker=s).rename_axis("date").reset_index())
        print(f"downloaded {min(i + 50, len(symbols))}/{len(symbols)}", flush=True)
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(OUT, index=False, compression="zstd")
    print(f"wrote {len(out)} rows, {out.ticker.nunique()} tickers, "
          f"{out.date.min().date()}..{out.date.max().date()}, {OUT.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
