#!/usr/bin/env python3
"""
fetch_market_data.py  —  Hedge Fund Swarm  (yfinance edition)
=============================================================
Fetches every data point the swarm needs for a US stock and writes
them into per-section plain-text files.  All 12 swarm agents read
only the section files their role requires; none make their own API calls.

Vendor: yfinance only (no Alpha Vantage dependency).

Output files (auto-named unless --base-name is provided):
    {TICKER}_{DATE}_manifest.txt            ← index of all section files
    {TICKER}_{DATE}_s01_ohlcv.txt
    {TICKER}_{DATE}_s02_technicals.txt
    {TICKER}_{DATE}_s03_profile.txt
    {TICKER}_{DATE}_s04_valuation.txt
    {TICKER}_{DATE}_s05_balance_sheet.txt
    {TICKER}_{DATE}_s06_income_stmt.txt
    {TICKER}_{DATE}_s07_cash_flow.txt
    {TICKER}_{DATE}_s08_earnings_history.txt
    {TICKER}_{DATE}_s09_analyst_estimates.txt
    {TICKER}_{DATE}_s10_analyst_recs.txt
    {TICKER}_{DATE}_s11_insider_txns.txt
    {TICKER}_{DATE}_s12_holders.txt
    {TICKER}_{DATE}_s13_options.txt
    {TICKER}_{DATE}_s14_company_news.txt
    {TICKER}_{DATE}_s15_macro_news.txt
    {TICKER}_{DATE}_s16_stocktwits.txt
    {TICKER}_{DATE}_s17_reddit.txt

Sections
────────
  1  OHLCV Price History        (180 days daily)
  2  Technical Indicators       (12 indicators via stockstats)
  3  Company Profile & Key Ratios
  4  Valuation Metrics
  5  Balance Sheet              (quarterly + annual)
  6  Income Statement           (quarterly + annual)
  7  Cash Flow Statement        (quarterly + annual)
  8  Earnings History           (actual vs estimate, surprise %)
  9  Analyst Estimates          (EPS & revenue, trend, revisions, growth)
 10  Analyst Recommendations    (consensus + history + upgrades/downgrades)
 11  Insider Transactions
 12  Institutional & Major Holders
 13  Options Snapshot           (IV, put/call ratio, key strikes)
 14  Company News               (past 7 days)
 15  Global Macro News          (past 7 days)
 16  StockTwits Sentiment       (30 most recent messages)
 17  Reddit Posts               (20 searches across 6 subreddits, past 7 days)
                                 High-upvote posts include top 10 comments.

Required packages:
    pip install yfinance requests pandas stockstats python-dateutil

Optional (authenticated Reddit via praw — falls back to public API):
    pip install praw
    Set env vars: REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT

Usage:
    python fetch_market_data.py --ticker NVDA --trade-date 2026-05-22
    python fetch_market_data.py --ticker AAPL --trade-date 2026-05-22 --base-name aapl_may22
    python fetch_market_data.py --ticker TSLA --trade-date 2026-05-22 --comment-threshold 100
"""

import argparse
import os
import sys
import time
import warnings
import shutil
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")  # suppress yfinance noise


# ─────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────

def get_default_trade_date() -> str:
    """
    Returns today's date if it's a weekday (Mon-Fri),
    or the most recent Friday if it's a weekend.
    Format: YYYY-MM-DD
    """
    today = datetime.now()
    weekday = today.weekday()  # 0=Mon, 1=Tue, ..., 5=Sat, 6=Sun
    
    if weekday < 5:  # Monday through Friday
        return today.strftime("%Y-%m-%d")
    else:  # Saturday or Sunday
        # Return last Friday
        days_back = weekday - 4  # 1 for Saturday, 2 for Sunday
        last_friday = today - timedelta(days=days_back)
        return last_friday.strftime("%Y-%m-%d")


def parse_args():
    p = argparse.ArgumentParser(
        description="Hedge Fund Swarm — Market Data Collector (yfinance)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--ticker",            required=True,
                   help="US stock ticker, e.g. NVDA")
    p.add_argument("--trade-date",        required=False, default=None,
                   help="Analysis date YYYY-MM-DD (default: today or last weekday if weekend)")
    p.add_argument("--base-name",         default=None,
                   help="Base name prefix for output files (default: {TICKER}_{DATE})")
    p.add_argument("--lookback",          type=int, default=180,
                   help="Days of OHLCV price history (default 180)")
    p.add_argument("--news-days",         type=int, default=7,
                   help="Days of news to fetch (default 7)")
    p.add_argument("--st-limit",          type=int, default=30,
                   help="Max StockTwits messages (default 30)")
    p.add_argument("--option-exp",        type=int, default=2,
                   help="Number of options expiries to show (default 2)")
    p.add_argument("--comment-threshold", type=int, default=200,
                   help="Fetch top 10 Reddit comments for posts with score >= this (default 200)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────

SEP  = "═" * 72
THIN = "─" * 72

def section_header(n: int, title: str) -> str:
    return f"\n{SEP}\n  SECTION {n:02d}: {title}\n{SEP}\n"

def sub(title: str) -> str:
    return f"\n{THIN}\n  {title}\n{THIN}\n"

def na(val, fmt=None):
    """Return formatted value or N/A."""
    if val is None or (isinstance(val, float) and val != val):
        return "N/A"
    if fmt == "pct":
        return f"{val * 100:.2f}%"
    if fmt == "M":
        return f"${val / 1_000_000:,.1f}M"
    if fmt == "B":
        return f"${val / 1_000_000_000:,.2f}B"
    if fmt == "price":
        return f"${val:,.2f}"
    return str(val)

def kv(label: str, value, width: int = 38, fmt=None) -> str:
    return f"  {label:<{width}}: {na(value, fmt)}"

def unavailable(reason: str = "") -> str:
    return f"  [UNAVAILABLE{': ' + reason if reason else ''}]\n"


# ─────────────────────────────────────────────────────────────
# Section 1 — OHLCV Price History
# ─────────────────────────────────────────────────────────────

def fetch_ohlcv(ticker: str, start: str, end: str) -> str:
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            return unavailable("No price data returned")
        df.index = df.index.strftime("%Y-%m-%d")
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return df.round(4).to_string()
    except Exception as e:
        return unavailable(str(e))


# ─────────────────────────────────────────────────────────────
# Section 2 — Technical Indicators
# ─────────────────────────────────────────────────────────────

INDICATORS = [
    "close_50_sma", "close_200_sma", "close_10_ema",
    "macd", "macds", "macdh",
    "rsi",
    "boll", "boll_ub", "boll_lb",
    "atr", "vwma",
]

def fetch_indicators(ticker: str, trade_date: str, window: int = 60) -> str:
    try:
        import yfinance as yf
        from stockstats import StockDataFrame

        end_dt   = datetime.strptime(trade_date, "%Y-%m-%d")
        start_dt = end_dt - timedelta(days=window + 260)  # extra warm-up days

        df = yf.download(
            ticker,
            start=start_dt.strftime("%Y-%m-%d"),
            end=(end_dt + timedelta(days=1)).strftime("%Y-%m-%d"),
            progress=False, auto_adjust=True,
        )
        if df.empty:
            return unavailable("No price data for indicators")

        df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
        stock = StockDataFrame.retype(df.copy())

        blocks = []
        for ind in INDICATORS:
            try:
                series = stock[ind].tail(window).dropna()
                series.index = series.index.strftime("%Y-%m-%d")
                latest = series.iloc[-1] if len(series) else float("nan")
                blocks.append(
                    f"\n  [{ind.upper()}]  latest = {latest:.4f}\n"
                    + series.round(4).to_string(header=False)
                )
            except Exception as e:
                blocks.append(f"\n  [{ind.upper()}]\n  {unavailable(str(e))}")

        return "\n".join(blocks)
    except Exception as e:
        return unavailable(str(e))


# ─────────────────────────────────────────────────────────────
# Section 3 — Company Profile & Key Ratios
# ─────────────────────────────────────────────────────────────

def fetch_profile(info: dict) -> str:
    lines = [
        sub("Identity"),
        kv("Company Name",          info.get("longName")),
        kv("Ticker",                info.get("symbol")),
        kv("Exchange",              info.get("exchange")),
        kv("Currency",              info.get("currency")),
        kv("Sector",                info.get("sector")),
        kv("Industry",              info.get("industry")),
        kv("Country",               info.get("country")),
        kv("Full-Time Employees",   info.get("fullTimeEmployees")),
        kv("Website",               info.get("website")),
        "",
        sub("Business Summary"),
        "  " + (info.get("longBusinessSummary") or "N/A"),
        "",
        sub("Price & Market Cap"),
        kv("Current Price",         info.get("currentPrice"),    fmt="price"),
        kv("Previous Close",        info.get("previousClose"),   fmt="price"),
        kv("52-Week High",          info.get("fiftyTwoWeekHigh"),fmt="price"),
        kv("52-Week Low",           info.get("fiftyTwoWeekLow"), fmt="price"),
        kv("50-Day Average",        info.get("fiftyDayAverage"), fmt="price"),
        kv("200-Day Average",       info.get("twoHundredDayAverage"), fmt="price"),
        kv("Market Cap",            info.get("marketCap"),       fmt="B"),
        kv("Enterprise Value",      info.get("enterpriseValue"), fmt="B"),
        kv("Shares Outstanding",    info.get("sharesOutstanding")),
        kv("Float Shares",          info.get("floatShares")),
        kv("Shares Short",          info.get("sharesShort")),
        kv("Short % of Float",      info.get("shortPercentOfFloat"), fmt="pct"),
        kv("Short Ratio",           info.get("shortRatio")),
        kv("Beta (5Y Monthly)",     info.get("beta")),
        kv("Average Volume (10d)",  info.get("averageVolume10days")),
        kv("Average Volume (3m)",   info.get("averageVolume")),
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Section 4 — Valuation Metrics
# ─────────────────────────────────────────────────────────────

def fetch_valuation(info: dict) -> str:
    lines = [
        sub("Earnings-Based"),
        kv("Trailing PE",               info.get("trailingPE")),
        kv("Forward PE",                info.get("forwardPE")),
        kv("PEG Ratio (5Y Expected)",   info.get("pegRatio")),
        kv("Trailing EPS",              info.get("trailingEps"),    fmt="price"),
        kv("Forward EPS",               info.get("forwardEps"),     fmt="price"),
        kv("EPS Growth YoY (ttm)",      info.get("earningsGrowth"), fmt="pct"),
        kv("EPS Growth Quarterly",      info.get("earningsQuarterlyGrowth"), fmt="pct"),
        "",
        sub("Revenue-Based"),
        kv("Total Revenue (TTM)",       info.get("totalRevenue"),   fmt="B"),
        kv("Revenue Per Share",         info.get("revenuePerShare"), fmt="price"),
        kv("Revenue Growth YoY",        info.get("revenueGrowth"),  fmt="pct"),
        kv("Price / Sales (TTM)",       info.get("priceToSalesTrailing12Months")),
        kv("EV / Revenue",              info.get("enterpriseToRevenue")),
        "",
        sub("Asset-Based"),
        kv("Price / Book",              info.get("priceToBook")),
        kv("Book Value Per Share",      info.get("bookValue"),      fmt="price"),
        kv("EV / EBITDA",               info.get("enterpriseToEbitda")),
        kv("EBITDA",                    info.get("ebitda"),         fmt="B"),
        kv("Gross Profit (TTM)",        info.get("grossProfits"),   fmt="B"),
        kv("Net Income (TTM)",          info.get("netIncomeToCommon"), fmt="B"),
        "",
        sub("Profitability Margins"),
        kv("Gross Margin",              info.get("grossMargins"),   fmt="pct"),
        kv("Operating Margin",          info.get("operatingMargins"), fmt="pct"),
        kv("Profit Margin (Net)",       info.get("profitMargins"),  fmt="pct"),
        kv("Return on Equity (TTM)",    info.get("returnOnEquity"), fmt="pct"),
        kv("Return on Assets (TTM)",    info.get("returnOnAssets"), fmt="pct"),
        "",
        sub("Liquidity & Leverage"),
        kv("Total Cash",                info.get("totalCash"),      fmt="B"),
        kv("Cash Per Share",            info.get("totalCashPerShare"), fmt="price"),
        kv("Total Debt",                info.get("totalDebt"),      fmt="B"),
        kv("Debt / Equity",             info.get("debtToEquity")),
        kv("Current Ratio",             info.get("currentRatio")),
        kv("Quick Ratio",               info.get("quickRatio")),
        kv("Operating Cash Flow (TTM)", info.get("operatingCashflow"), fmt="B"),
        kv("Free Cash Flow (TTM)",      info.get("freeCashflow"),   fmt="B"),
        "",
        sub("Dividends"),
        kv("Dividend Rate",             info.get("dividendRate"),   fmt="price"),
        kv("Dividend Yield",            info.get("dividendYield"),  fmt="pct"),
        kv("Payout Ratio",              info.get("payoutRatio"),    fmt="pct"),
        kv("Ex-Dividend Date",          info.get("exDividendDate")),
        kv("Last Dividend Value",       info.get("lastDividendValue"), fmt="price"),
        kv("5Y Avg Dividend Yield",     info.get("fiveYearAvgDividendYield"), fmt="pct"),
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Sections 5, 6, 7 — Financial Statements
# ─────────────────────────────────────────────────────────────

def _fmt_df(df, label: str) -> str:
    if df is None or (hasattr(df, "empty") and df.empty):
        return unavailable(f"No {label} available")
    try:
        formatted = df.copy().astype(object)
        for col in formatted.columns:
            for idx in formatted.index:
                v = formatted.at[idx, col]
                try:
                    fv = float(v)
                    if abs(fv) >= 1e9:
                        formatted.at[idx, col] = f"${fv/1e9:,.3f}B"
                    elif abs(fv) >= 1e6:
                        formatted.at[idx, col] = f"${fv/1e6:,.1f}M"
                    else:
                        formatted.at[idx, col] = f"{fv:,.0f}"
                except (TypeError, ValueError):
                    pass
        return formatted.to_string()
    except Exception:
        return df.to_string()

def fetch_financials(ticker_obj) -> dict:
    try:
        return {
            "bs_q":  _fmt_df(ticker_obj.quarterly_balance_sheet, "quarterly balance sheet"),
            "bs_a":  _fmt_df(ticker_obj.balance_sheet,           "annual balance sheet"),
            "inc_q": _fmt_df(ticker_obj.quarterly_income_stmt,   "quarterly income statement"),
            "inc_a": _fmt_df(ticker_obj.income_stmt,             "annual income statement"),
            "cf_q":  _fmt_df(ticker_obj.quarterly_cash_flow,     "quarterly cash flow"),
            "cf_a":  _fmt_df(ticker_obj.cash_flow,               "annual cash flow"),
        }
    except Exception as e:
        msg = unavailable(str(e))
        return {k: msg for k in ["bs_q", "bs_a", "inc_q", "inc_a", "cf_q", "cf_a"]}


# ─────────────────────────────────────────────────────────────
# Section 8 — Earnings History
# ─────────────────────────────────────────────────────────────

def fetch_earnings_history(ticker_obj) -> str:
    try:
        df = ticker_obj.earnings_history
        if df is None or df.empty:
            return unavailable("No earnings history available")

        lines = [
            f"  {'Date':<14} {'Period':<12} {'EPS Estimate':>14} {'EPS Actual':>12} {'Surprise':>10} {'Surprise %':>11}",
            "  " + "-" * 67,
        ]
        for dt, row in df.sort_index(ascending=False).head(12).iterrows():
            date_s   = str(dt)[:10]
            period   = str(row.get("period", "")).strip()
            estimate = row.get("epsEstimate", float("nan"))
            actual   = row.get("epsActual",   float("nan"))
            diff     = row.get("epsDifference", float("nan"))
            surprise = row.get("surprisePercent", float("nan"))

            def _f(v, prefix="$"):
                try: return f"{prefix}{float(v):.3f}" if v == v else "N/A"
                except: return "N/A"

            def _fp(v):
                try: return f"{float(v)*100:.2f}%" if v == v else "N/A"
                except: return "N/A"

            lines.append(
                f"  {date_s:<14} {period:<12} {_f(estimate):>14} "
                f"{_f(actual):>12} {_f(diff,''):>10} {_fp(surprise):>11}"
            )

        # Beat / miss streak
        beat_miss = []
        for _, row in df.sort_index(ascending=False).head(8).iterrows():
            sp = row.get("surprisePercent", float("nan"))
            try:
                beat_miss.append("Beat" if float(sp) > 0 else "Miss")
            except:
                beat_miss.append("?")

        lines.append("")
        lines.append(f"  Last 8 quarters (newest first): {' → '.join(beat_miss)}")

        consecutive = 0
        for bm in beat_miss:
            if bm == "Beat": consecutive += 1
            else: break
        if consecutive > 1:
            lines.append(f"  ★ Beat estimate {consecutive} quarters in a row")

        return "\n".join(lines)
    except Exception as e:
        return unavailable(str(e))


# ─────────────────────────────────────────────────────────────
# Section 9 — Analyst Estimates
# ─────────────────────────────────────────────────────────────

def fetch_analyst_estimates(ticker_obj) -> str:
    blocks = []

    try:
        df = ticker_obj.eps_estimates
        if df is not None and not df.empty:
            blocks.append(sub("EPS Estimates"))
            blocks.append(df.to_string())
    except Exception as e:
        blocks.append(sub("EPS Estimates"))
        blocks.append(unavailable(str(e)))

    try:
        df = ticker_obj.revenue_estimates
        if df is not None and not df.empty:
            blocks.append(sub("Revenue Estimates"))
            blocks.append(df.to_string())
    except Exception as e:
        blocks.append(sub("Revenue Estimates"))
        blocks.append(unavailable(str(e)))

    try:
        df = ticker_obj.eps_trend
        if df is not None and not df.empty:
            blocks.append(sub("EPS Trend (Current vs. 7d / 30d / 60d / 90d ago)"))
            blocks.append(df.to_string())
    except Exception as e:
        blocks.append(sub("EPS Trend"))
        blocks.append(unavailable(str(e)))

    try:
        df = ticker_obj.eps_revisions
        if df is not None and not df.empty:
            blocks.append(sub("EPS Revisions (Upgrades vs. Downgrades)"))
            blocks.append(df.to_string())
    except Exception as e:
        blocks.append(sub("EPS Revisions"))
        blocks.append(unavailable(str(e)))

    try:
        df = ticker_obj.growth_estimates
        if df is not None and not df.empty:
            blocks.append(sub("Growth Estimates (EPS Growth %)"))
            blocks.append(df.to_string())
    except Exception as e:
        blocks.append(sub("Growth Estimates"))
        blocks.append(unavailable(str(e)))

    return "\n".join(blocks) if blocks else unavailable("No analyst estimates available")


# ─────────────────────────────────────────────────────────────
# Section 10 — Analyst Recommendations
# ─────────────────────────────────────────────────────────────

def fetch_recommendations(info: dict, ticker_obj) -> str:
    lines = []

    lines.append(sub("Current Consensus"))
    lines.append(kv("Recommendation",          info.get("recommendationKey", "N/A").upper()))
    lines.append(kv("Mean Rating Score (1–5)", info.get("recommendationMean")))
    lines.append(kv("Number of Analysts",      info.get("numberOfAnalystOpinions")))
    lines.append(kv("Target Price (Mean)",     info.get("targetMeanPrice"),   fmt="price"))
    lines.append(kv("Target Price (Median)",   info.get("targetMedianPrice"), fmt="price"))
    lines.append(kv("Target Price (High)",     info.get("targetHighPrice"),   fmt="price"))
    lines.append(kv("Target Price (Low)",      info.get("targetLowPrice"),    fmt="price"))
    curr = info.get("currentPrice") or info.get("previousClose")
    mean = info.get("targetMeanPrice")
    if curr and mean:
        try:
            upside = (float(mean) / float(curr) - 1) * 100
            lines.append(kv("Implied Upside to Mean Target", f"{upside:+.1f}%"))
        except: pass

    try:
        df = ticker_obj.recommendations
        if df is not None and not df.empty:
            lines.append(sub("Recommendation History (last 12 entries)"))
            recent = df.tail(12).iloc[::-1]
            lines.append(recent.to_string())
    except Exception as e:
        lines.append(sub("Recommendation History"))
        lines.append(unavailable(str(e)))

    try:
        df = ticker_obj.upgrades_downgrades
        if df is not None and not df.empty:
            lines.append(sub("Upgrades / Downgrades (last 20)"))
            lines.append(df.head(20).to_string())
    except Exception as e:
        lines.append(sub("Upgrades / Downgrades"))
        lines.append(unavailable(str(e)))

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Section 11 — Insider Transactions
# ─────────────────────────────────────────────────────────────

def fetch_insider(ticker_obj) -> str:
    try:
        df = ticker_obj.insider_transactions
        if df is None or df.empty:
            return unavailable("No insider transaction data")
        return df.head(30).to_string()
    except Exception as e:
        return unavailable(str(e))


# ─────────────────────────────────────────────────────────────
# Section 12 — Institutional & Major Holders
# ─────────────────────────────────────────────────────────────

def fetch_holders(ticker_obj) -> str:
    blocks = []

    try:
        df = ticker_obj.major_holders
        if df is not None and not df.empty:
            blocks.append(sub("Major Holders (% of shares)"))
            blocks.append(df.to_string(header=False))
    except Exception as e:
        blocks.append(sub("Major Holders"))
        blocks.append(unavailable(str(e)))

    try:
        df = ticker_obj.institutional_holders
        if df is not None and not df.empty:
            blocks.append(sub("Top Institutional Holders"))
            blocks.append(df.head(20).to_string(index=False))
    except Exception as e:
        blocks.append(sub("Top Institutional Holders"))
        blocks.append(unavailable(str(e)))

    try:
        df = ticker_obj.mutualfund_holders
        if df is not None and not df.empty:
            blocks.append(sub("Top Mutual Fund Holders"))
            blocks.append(df.head(15).to_string(index=False))
    except Exception as e:
        blocks.append(sub("Top Mutual Fund Holders"))
        blocks.append(unavailable(str(e)))

    return "\n".join(blocks) if blocks else unavailable("No holder data available")


# ─────────────────────────────────────────────────────────────
# Section 13 — Options Snapshot
# ─────────────────────────────────────────────────────────────

def fetch_options(ticker_obj, current_price: float, num_expiries: int = 2) -> str:
    try:
        expiries = ticker_obj.options
        if not expiries:
            return unavailable("No options data available")

        lines = [f"  Available expiries: {', '.join(expiries[:8])}"]
        lines.append(f"  Current underlying price: ${current_price:,.2f}\n")

        for exp in expiries[:num_expiries]:
            chain = ticker_obj.option_chain(exp)
            calls = chain.calls
            puts  = chain.puts

            if calls.empty and puts.empty:
                continue

            lines.append(sub(f"Expiry: {exp}"))

            call_oi  = calls["openInterest"].sum() if "openInterest" in calls else 0
            put_oi   = puts["openInterest"].sum()  if "openInterest" in puts  else 0
            total_oi = call_oi + put_oi
            pc_ratio = put_oi / call_oi if call_oi > 0 else float("nan")

            lines.append(f"  Total Open Interest: {total_oi:,.0f}")
            if total_oi:
                lines.append(f"    Call OI: {call_oi:,.0f}  ({call_oi/total_oi*100:.1f}%)")
                lines.append(f"    Put OI:  {put_oi:,.0f}  ({put_oi/total_oi*100:.1f}%)")
            try:
                lines.append(f"  Put/Call Ratio (OI): {pc_ratio:.3f}  "
                             f"({'bearish' if pc_ratio > 1.2 else 'bullish' if pc_ratio < 0.7 else 'neutral'})")
            except: pass

            def atm_iv(df_opt, price):
                if df_opt.empty or "impliedVolatility" not in df_opt:
                    return None
                df_opt = df_opt.copy()
                df_opt["dist"] = (df_opt["strike"] - price).abs()
                atm = df_opt.nsmallest(3, "dist")
                return atm["impliedVolatility"].mean()

            call_iv = atm_iv(calls, current_price)
            put_iv  = atm_iv(puts,  current_price)
            if call_iv: lines.append(f"  ATM Call IV:  {call_iv*100:.1f}%")
            if put_iv:  lines.append(f"  ATM Put IV:   {put_iv*100:.1f}%")

            try:
                strikes = sorted(set(calls["strike"].tolist() + puts["strike"].tolist()))
                pain = {}
                for s in strikes:
                    pain[s] = (
                        calls[calls["strike"] < s]["openInterest"].fillna(0).sum()
                        + puts[puts["strike"] > s]["openInterest"].fillna(0).sum()
                    )
                max_pain_strike = max(pain, key=pain.get)
                lines.append(f"  Max Pain Strike: ${max_pain_strike:,.2f}")
            except: pass

            if not calls.empty and "openInterest" in calls:
                top_calls = calls.nlargest(5, "openInterest")[["strike", "openInterest", "impliedVolatility", "lastPrice"]]
                lines.append(f"\n  Top 5 Calls by Open Interest:\n{top_calls.to_string(index=False)}")
            if not puts.empty and "openInterest" in puts:
                top_puts = puts.nlargest(5, "openInterest")[["strike", "openInterest", "impliedVolatility", "lastPrice"]]
                lines.append(f"\n  Top 5 Puts by Open Interest:\n{top_puts.to_string(index=False)}")

        return "\n".join(lines)
    except Exception as e:
        return unavailable(str(e))


# ─────────────────────────────────────────────────────────────
# Section 14 — Company News
# ─────────────────────────────────────────────────────────────

def fetch_company_news(ticker_obj, ticker: str, start: str, end: str) -> str:
    try:
        news = ticker_obj.news or []
        start_ts = int(datetime.strptime(start, "%Y-%m-%d").timestamp())
        end_ts   = int(datetime.strptime(end,   "%Y-%m-%d").timestamp()) + 86400

        lines = []
        for item in news:
            pub = item.get("providerPublishTime", 0)
            if not (start_ts <= pub <= end_ts):
                continue
            title   = item.get("title", "N/A")
            source  = item.get("publisher", "N/A")
            link    = item.get("link", "")
            date_s  = datetime.fromtimestamp(pub).strftime("%Y-%m-%d %H:%M")
            summary = (item.get("summary") or "")[:500]

            lines.append(f"  [{date_s}]  {source}")
            lines.append(f"  Headline: {title}")
            if link:
                lines.append(f"  URL:      {link}")
            if summary:
                lines.append(f"  Summary:  {summary}")
            lines.append("")

        if not lines:
            return f"  No news found for {ticker} between {start} and {end}\n"
        return "\n".join(lines)
    except Exception as e:
        return unavailable(str(e))


# ─────────────────────────────────────────────────────────────
# Section 15 — Global Macro News
# ─────────────────────────────────────────────────────────────

MACRO_PROXIES = [
    ("SPY",       "US Equity Market"),
    ("QQQ",       "Tech/Growth"),
    ("^VIX",      "Volatility / Fear Index"),
    ("^TNX",      "10-Year Treasury Yield"),
    ("GC=F",      "Gold"),
    ("CL=F",      "Crude Oil"),
    ("DX-Y.NYB",  "US Dollar Index"),
]

def fetch_global_news(trade_date: str, days: int = 7) -> str:
    try:
        import yfinance as yf
        end_dt   = datetime.strptime(trade_date, "%Y-%m-%d")
        start_dt = end_dt - timedelta(days=days)
        start_ts = int(start_dt.timestamp())
        end_ts   = int(end_dt.timestamp()) + 86400

        seen  = set()
        lines = []

        for proxy, label in MACRO_PROXIES:
            try:
                news = yf.Ticker(proxy).news or []
                proxy_lines = []
                for item in news:
                    pub   = item.get("providerPublishTime", 0)
                    title = item.get("title", "")
                    if not (start_ts <= pub <= end_ts) or title in seen:
                        continue
                    seen.add(title)
                    source  = item.get("publisher", "N/A")
                    date_s  = datetime.fromtimestamp(pub).strftime("%Y-%m-%d %H:%M")
                    summary = (item.get("summary") or "")[:300]
                    proxy_lines.append(f"  [{date_s}]  {source}  [{label}]")
                    proxy_lines.append(f"  {title}")
                    if summary:
                        proxy_lines.append(f"  → {summary}")
                    proxy_lines.append("")
                if proxy_lines:
                    lines.extend(proxy_lines)
            except Exception:
                continue

        if not lines:
            return f"  No macro news found between {start_dt.strftime('%Y-%m-%d')} and {trade_date}\n"
        return "\n".join(lines)
    except Exception as e:
        return unavailable(str(e))


# ─────────────────────────────────────────────────────────────
# Section 16 — StockTwits
# ─────────────────────────────────────────────────────────────

def fetch_stocktwits(ticker: str, limit: int = 30) -> str:
    try:
        import requests
        clean = ticker.split(".")[0].split("-")[0]
        url   = f"https://api.stocktwits.com/api/2/streams/symbol/{clean}.json?limit={limit}"
        r = requests.get(url, timeout=10, headers={"User-Agent": "HedgeFundSwarm/0.1"})
        if r.status_code != 200:
            return unavailable(f"HTTP {r.status_code}")

        messages = r.json().get("messages", [])
        if not messages:
            return unavailable("No messages returned")

        bullish   = sum(1 for m in messages if (m.get("entities", {}).get("sentiment") or {}).get("basic", "").lower() == "bullish")
        bearish   = sum(1 for m in messages if (m.get("entities", {}).get("sentiment") or {}).get("basic", "").lower() == "bearish")
        total     = len(messages)
        unlabeled = total - bullish - bearish

        lines = [
            f"  Total messages: {total}",
            f"  Bullish: {bullish} ({bullish/total*100:.0f}%)  "
            f"Bearish: {bearish} ({bearish/total*100:.0f}%)  "
            f"Unlabeled: {unlabeled}",
            "",
        ]

        for msg in messages:
            user      = msg.get("user", {}).get("username", "unknown")
            body      = (msg.get("body") or "").replace("\n", " ")[:300]
            created   = (msg.get("created_at") or "")[:10]
            sentiment = (msg.get("entities", {}).get("sentiment") or {}).get("basic", "No label")
            tag       = {"Bullish": "🟢", "Bearish": "🔴"}.get(sentiment, "⬜")
            lines.append(f"  {tag} [{created}] @{user} [{sentiment}]: {body}")

        return "\n".join(lines)
    except Exception as e:
        return unavailable(str(e))


# ─────────────────────────────────────────────────────────────
# Section 17 — Reddit
# ─────────────────────────────────────────────────────────────

# 20 varied search queries across 6 subreddits.
# Each tuple: (subreddit, query_template, sort)
# {ticker} is replaced with the cleaned ticker symbol at runtime.
REDDIT_SEARCH_PLAN = [
    # wallstreetbets — 5 queries
    ("wallstreetbets",   "{ticker}",              "top"),
    ("wallstreetbets",   "${ticker}",             "top"),
    ("wallstreetbets",   "{ticker} stock",        "top"),
    ("wallstreetbets",   "{ticker} options",      "top"),
    ("wallstreetbets",   "{ticker}",              "new"),
    # stocks — 5 queries
    ("stocks",           "{ticker}",              "top"),
    ("stocks",           "${ticker}",             "top"),
    ("stocks",           "{ticker} earnings",     "top"),
    ("stocks",           "{ticker} stock",        "top"),
    ("stocks",           "{ticker}",              "new"),
    # investing — 5 queries
    ("investing",        "{ticker}",              "top"),
    ("investing",        "${ticker}",             "top"),
    ("investing",        "{ticker} stock",        "top"),
    ("investing",        "{ticker} analysis",     "top"),
    ("investing",        "{ticker}",              "new"),
    # options — 2 queries
    ("options",          "{ticker}",              "top"),
    ("options",          "${ticker}",             "top"),
    # StockMarket — 2 queries
    ("StockMarket",      "{ticker}",              "top"),
    ("StockMarket",      "${ticker}",             "top"),
    # SecurityAnalysis — 1 query
    ("SecurityAnalysis", "{ticker}",              "top"),
]

# Subreddit display order in final output
REDDIT_SUB_ORDER = [
    "wallstreetbets", "stocks", "investing",
    "options", "StockMarket", "SecurityAnalysis",
]


def fetch_reddit(ticker: str, comment_threshold: int = 200) -> str:
    """Run 20 Reddit searches. High-upvote posts get top-10 comments appended."""
    clean = ticker.split(".")[0].split("-")[0].upper()

    try:
        import praw
        cid    = os.environ.get("REDDIT_CLIENT_ID", "")
        secret = os.environ.get("REDDIT_CLIENT_SECRET", "")
        agent  = os.environ.get("REDDIT_USER_AGENT", "HedgeFundSwarm/0.1")
        if cid and secret:
            reddit = praw.Reddit(client_id=cid, client_secret=secret, user_agent=agent)
            return _reddit_via_praw(reddit, clean, comment_threshold)
    except ImportError:
        pass

    return _reddit_public(clean, comment_threshold)


def _reddit_via_praw(reddit, clean: str, comment_threshold: int) -> str:
    """Authenticated Reddit path: 20 searches + comment fetches for qualifying posts."""
    seen_ids = {}   # post.id → post object (dedup across searches)
    call_log = []

    print(f"    Running 20 Reddit searches via praw (1s apart)...")
    for i, (sub_name, query_tmpl, sort) in enumerate(REDDIT_SEARCH_PLAN):
        query = query_tmpl.format(ticker=clean)
        if i > 0:
            time.sleep(1)
        try:
            subreddit = reddit.subreddit(sub_name)
            posts = list(subreddit.search(query, time_filter="week", sort=sort, limit=25))
            new_count = 0
            for post in posts:
                if post.id not in seen_ids:
                    seen_ids[post.id] = post
                    new_count += 1
            call_log.append(
                f"  [{i+1:02d}/20] r/{sub_name:<20} q={query!r:<30} sort={sort}: "
                f"{len(posts)} results, {new_count} new"
            )
        except Exception as e:
            call_log.append(f"  [{i+1:02d}/20] r/{sub_name} q={query!r}: ERROR — {e}")

    lines = [
        f"  Search method: praw (authenticated)",
        f"  Searches run: {len(REDDIT_SEARCH_PLAN)} | Unique posts found: {len(seen_ids)}",
        f"  Comment threshold: ≥{comment_threshold} upvotes → top 10 comments fetched",
        "",
        "  Search log:",
    ] + call_log + [""]

    # Group by subreddit
    by_sub = {}
    for post in seen_ids.values():
        sn = post.subreddit.display_name
        by_sub.setdefault(sn, []).append(post)

    for sub_name in REDDIT_SUB_ORDER:
        posts_in_sub = sorted(by_sub.get(sub_name, []), key=lambda p: p.score, reverse=True)
        if not posts_in_sub:
            lines.append(f"\n  r/{sub_name}: No posts found this week.")
            continue
        lines.append(f"\n  r/{sub_name} — {len(posts_in_sub)} unique posts:")
        for post in posts_in_sub:
            date_s = datetime.utcfromtimestamp(post.created_utc).strftime("%Y-%m-%d")
            body   = (post.selftext or "")[:350].replace("\n", " ")
            lines.append(
                f"\n    ↑{post.score:>6}  💬{post.num_comments:>4}  [{date_s}]\n"
                f"    Title: {post.title}\n"
                f"    Body:  {body}\n"
                f"    URL:   https://reddit.com{post.permalink}"
            )
            # Fetch top comments for high-upvote posts
            if post.score >= comment_threshold:
                try:
                    time.sleep(1)
                    post.comments.replace_more(limit=0)
                    top_comments = sorted(
                        [c for c in post.comments.list() if hasattr(c, "score") and hasattr(c, "body")],
                        key=lambda c: c.score,
                        reverse=True
                    )[:10]
                    if top_comments:
                        lines.append(f"    ── Top Comments (post score {post.score:,} ≥ threshold {comment_threshold:,}) ──")
                        for c in top_comments:
                            author = str(c.author) if c.author else "[deleted]"
                            body_c = (c.body or "").replace("\n", " ")[:300]
                            lines.append(f"      ↑{c.score:>5}  @{author}: {body_c}")
                except Exception as e:
                    lines.append(f"    [Comments unavailable: {e}]")

    return "\n".join(lines)


def _reddit_public(clean: str, comment_threshold: int) -> str:
    """Unauthenticated Reddit public JSON API path: 20 searches + comment fetches."""
    try:
        import requests
    except ImportError:
        return unavailable("requests package not installed")

    headers  = {"User-Agent": "HedgeFundSwarm/0.1"}
    seen_ids = {}   # post id str → post data dict
    call_log = []

    print(f"    Running 20 Reddit searches via public API (1s apart)...")
    for i, (sub_name, query_tmpl, sort) in enumerate(REDDIT_SEARCH_PLAN):
        query = query_tmpl.format(ticker=clean)
        if i > 0:
            time.sleep(1)
        url = (
            f"https://www.reddit.com/r/{sub_name}/search.json"
            f"?q={query}&restrict_sr=1&sort={sort}&t=week&limit=25"
        )
        try:
            r = requests.get(url, headers=headers, timeout=12)
            if r.status_code != 200:
                call_log.append(f"  [{i+1:02d}/20] r/{sub_name:<20} q={query!r:<30}: HTTP {r.status_code}")
                continue
            posts = r.json().get("data", {}).get("children", [])
            new_count = 0
            for child in posts:
                p   = child["data"]
                pid = p.get("id", "")
                if pid and pid not in seen_ids:
                    seen_ids[pid] = p
                    new_count += 1
            call_log.append(
                f"  [{i+1:02d}/20] r/{sub_name:<20} q={query!r:<30} sort={sort}: "
                f"{len(posts)} results, {new_count} new"
            )
        except Exception as e:
            call_log.append(f"  [{i+1:02d}/20] r/{sub_name} q={query!r}: ERROR — {e}")

    lines = [
        f"  Search method: public JSON API (no praw credentials set)",
        f"  Searches run: {len(REDDIT_SEARCH_PLAN)} | Unique posts found: {len(seen_ids)}",
        f"  Comment threshold: ≥{comment_threshold} upvotes → top 10 comments fetched",
        "",
        "  Search log:",
    ] + call_log + [""]

    # Group by subreddit
    by_sub = {}
    for p in seen_ids.values():
        sn = p.get("subreddit", "unknown")
        by_sub.setdefault(sn, []).append(p)

    for sub_name in REDDIT_SUB_ORDER:
        posts_in_sub = sorted(
            by_sub.get(sub_name, []),
            key=lambda p: p.get("score", 0),
            reverse=True,
        )
        if not posts_in_sub:
            lines.append(f"\n  r/{sub_name}: No posts found this week.")
            continue
        lines.append(f"\n  r/{sub_name} — {len(posts_in_sub)} unique posts:")
        for p in posts_in_sub:
            date_s = datetime.utcfromtimestamp(p.get("created_utc", 0)).strftime("%Y-%m-%d")
            body   = (p.get("selftext") or "")[:350].replace("\n", " ")
            score  = p.get("score", 0)
            num_c  = p.get("num_comments", 0)
            title  = p.get("title", "N/A")
            plink  = p.get("permalink", "")
            pid    = p.get("id", "")

            lines.append(
                f"\n    ↑{score:>6}  💬{num_c:>4}  [{date_s}]\n"
                f"    Title: {title}\n"
                f"    Body:  {body}\n"
                f"    URL:   https://reddit.com{plink}"
            )
            # Fetch top comments for high-upvote posts
            if score >= comment_threshold and pid:
                try:
                    time.sleep(1)
                    comment_url = (
                        f"https://www.reddit.com/r/{sub_name}/comments/{pid}.json"
                        f"?limit=25&sort=top"
                    )
                    r2 = requests.get(comment_url, headers=headers, timeout=12)
                    if r2.status_code == 200:
                        cdata = r2.json()
                        # cdata[1] contains the comments listing
                        comment_children = cdata[1]["data"]["children"]
                        top_comments = sorted(
                            [
                                c["data"] for c in comment_children
                                if c.get("kind") == "t1"
                                and c["data"].get("body")
                                and c["data"].get("author") != "[deleted]"
                            ],
                            key=lambda c: c.get("score", 0),
                            reverse=True,
                        )[:10]
                        if top_comments:
                            lines.append(f"    ── Top Comments (post score {score:,} ≥ threshold {comment_threshold:,}) ──")
                            for c in top_comments:
                                author = c.get("author", "[deleted]")
                                body_c = (c.get("body") or "").replace("\n", " ")[:300]
                                c_score = c.get("score", 0)
                                lines.append(f"      ↑{c_score:>5}  @{author}: {body_c}")
                    else:
                        lines.append(f"    [Comments unavailable: HTTP {r2.status_code}]")
                except Exception as e:
                    lines.append(f"    [Comments unavailable: {e}]")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Per-section file writer
# ─────────────────────────────────────────────────────────────

def write_section_file(base_name: str, num: int, slug: str, title: str, content: str, folder: str = ""):
    """Write one section to its own file. Returns (filename, size_kb)."""
    filename = f"{base_name}_s{num:02d}_{slug}.txt"
    if folder:
        filepath = os.path.join(folder, filename)
    else:
        filepath = filename
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(section_header(num, title))
        f.write("\n")
        f.write(content)
        f.write("\n")
    size_kb = os.path.getsize(filepath) / 1024
    return filename, size_kb


# ─────────────────────────────────────────────────────────────
# Master assembler
# ─────────────────────────────────────────────────────────────

def build(args, base_name: str, folder: str = "", trade_date: str = "") -> list:
    """
    Fetch all 17 sections and write each to its own file.
    Returns list of (filename, size_kb) tuples.
    """
    import yfinance as yf

    ticker = args.ticker.upper().strip()
    lookback  = args.lookback
    news_days = args.news_days
    st_limit  = args.st_limit
    opt_exp   = args.option_exp
    comment_threshold = args.comment_threshold

    end_dt     = datetime.strptime(trade_date, "%Y-%m-%d")
    start      = (end_dt - timedelta(days=lookback)).strftime("%Y-%m-%d")
    news_start = (end_dt - timedelta(days=news_days)).strftime("%Y-%m-%d")

    print(f"  Loading yfinance Ticker object for {ticker}...")
    t    = yf.Ticker(ticker)
    info = t.info or {}

    current_price = (
        info.get("currentPrice")
        or info.get("regularMarketPrice")
        or info.get("previousClose")
        or 0.0
    )

    written = []   # accumulates (filename, size_kb) as each section completes

    def ws(num, slug, title, content):
        fname, size = write_section_file(base_name, num, slug, title, content, folder)
        written.append((fname, size))
        print(f"    → {fname}  ({size:.1f} KB)")
        return fname

    # ── Section 1 ────────────────────────────────────────────
    print("  [01/17] OHLCV price history...")
    ws(1, "ohlcv",
       f"OHLCV PRICE HISTORY  ({lookback} days: {start} → {trade_date})",
       fetch_ohlcv(ticker, start, trade_date))

    # ── Section 2 ────────────────────────────────────────────
    print("  [02/17] Technical indicators (12 indicators)...")
    ws(2, "technicals",
       "TECHNICAL INDICATORS  (60-day window, 12 series)",
       "  Indicators: " + ", ".join(INDICATORS) + "\n" + fetch_indicators(ticker, trade_date))

    # ── Section 3 ────────────────────────────────────────────
    print("  [03/17] Company profile...")
    ws(3, "profile",
       "COMPANY PROFILE & KEY RATIOS",
       fetch_profile(info))

    # ── Section 4 ────────────────────────────────────────────
    print("  [04/17] Valuation metrics...")
    ws(4, "valuation",
       "VALUATION METRICS",
       fetch_valuation(info))

    # ── Sections 5–7: Financials (fetch once, write three files) ─
    print("  [05/17] Financial statements (balance sheet, P&L, cash flow)...")
    fin = fetch_financials(t)

    ws(5, "balance_sheet",
       "BALANCE SHEET",
       sub("Quarterly (most recent 4 quarters)") + "\n" + fin["bs_q"]
       + "\n" + sub("Annual (most recent 4 years)") + "\n" + fin["bs_a"])

    ws(6, "income_stmt",
       "INCOME STATEMENT",
       sub("Quarterly") + "\n" + fin["inc_q"]
       + "\n" + sub("Annual") + "\n" + fin["inc_a"])

    ws(7, "cash_flow",
       "CASH FLOW STATEMENT",
       sub("Quarterly") + "\n" + fin["cf_q"]
       + "\n" + sub("Annual") + "\n" + fin["cf_a"])

    # ── Section 8 ────────────────────────────────────────────
    print("  [08/17] Earnings history (actual vs estimate)...")
    ws(8, "earnings_history",
       "EARNINGS HISTORY  (actual vs. estimate, last 12 quarters)",
       fetch_earnings_history(t))

    # ── Section 9 ────────────────────────────────────────────
    print("  [09/17] Analyst estimates (EPS, revenue, growth)...")
    ws(9, "analyst_estimates",
       "ANALYST ESTIMATES  (EPS, revenue, growth, revisions)",
       fetch_analyst_estimates(t))

    # ── Section 10 ───────────────────────────────────────────
    print("  [10/17] Analyst recommendations & price targets...")
    ws(10, "analyst_recs",
       "ANALYST RECOMMENDATIONS & PRICE TARGETS",
       fetch_recommendations(info, t))

    # ── Section 11 ───────────────────────────────────────────
    print("  [11/17] Insider transactions...")
    ws(11, "insider_txns",
       "INSIDER TRANSACTIONS  (last 30 entries)",
       fetch_insider(t))

    # ── Section 12 ───────────────────────────────────────────
    print("  [12/17] Institutional & major holders...")
    ws(12, "holders",
       "INSTITUTIONAL & MAJOR HOLDERS",
       fetch_holders(t))

    # ── Section 13 ───────────────────────────────────────────
    print(f"  [13/17] Options snapshot ({opt_exp} expiries)...")
    ws(13, "options",
       f"OPTIONS SNAPSHOT  (next {opt_exp} expiries)",
       fetch_options(t, current_price, num_expiries=opt_exp))

    # ── Section 14 ───────────────────────────────────────────
    print(f"  [14/17] Company news (past {news_days} days)...")
    ws(14, "company_news",
       f"COMPANY NEWS  ({news_days} days: {news_start} → {trade_date})",
       fetch_company_news(t, ticker, news_start, trade_date))

    # ── Section 15 ───────────────────────────────────────────
    print(f"  [15/17] Global macro news (past {news_days} days)...")
    ws(15, "macro_news",
       f"GLOBAL MACRO NEWS  ({news_days} days: {news_start} → {trade_date})",
       fetch_global_news(trade_date, days=news_days))

    # ── Section 16 ───────────────────────────────────────────
    print(f"  [16/17] StockTwits (limit={st_limit})...")
    ws(16, "stocktwits",
       f"STOCKTWITS SENTIMENT  (most recent {st_limit} messages)",
       fetch_stocktwits(ticker, limit=st_limit))

    # ── Section 17 ───────────────────────────────────────────
    print(f"  [17/17] Reddit posts (20 searches, comment threshold ≥{comment_threshold})...")
    ws(17, "reddit",
       "REDDIT POSTS  (20 searches across 6 subreddits — past 7 days)",
       fetch_reddit(ticker, comment_threshold=comment_threshold))

    return written


# ─────────────────────────────────────────────────────────────
# Manifest writer
# ─────────────────────────────────────────────────────────────

# Maps agent role → section numbers it should read
AGENT_SECTION_MAP = {
    "Market Analyst":              [1, 2],
    "Fundamentals Analyst":        [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    "News Analyst":                [14, 15],
    "Sentiment Analyst":           [14, 16, 17],
    "Aggressive Risk Analyst":     [13],
    "Conservative Risk Analyst":   [13],
    "Neutral Risk Analyst":        [13],
}

def write_manifest(base_name: str, ticker: str, trade_date: str,
                   generated: str, written: list, folder: str = "") -> str:
    """Write the manifest file. Returns filename."""
    manifest_file = f"{base_name}_manifest.txt"
    manifest_filepath = os.path.join(folder, manifest_file) if folder else manifest_file
    total_kb = sum(size for _, size in written)

    lines = [
        SEP,
        "  HEDGE FUND SWARM — DATA MANIFEST",
        f"  Ticker: {ticker}  |  Analysis Date: {trade_date}  |  Generated: {generated}",
        SEP,
        "",
        "  ⚠  AGENT INSTRUCTIONS",
        "  " + THIN,
        "  Read only the section files listed under your role below.",
        "  Do NOT call external APIs or fetch additional data yourself.",
        "  Sections marked [UNAVAILABLE] indicate a data gap — note it",
        "  in your report and proceed with what is available.",
        "",
        "  AGENT → SECTION FILE MAPPING",
        "  " + THIN,
    ]

    for agent, sections in AGENT_SECTION_MAP.items():
        file_list = ", ".join(f"s{s:02d}" for s in sections)
        lines.append(f"  {agent:<35} reads: {file_list}")

    lines += [
        "",
        "  ALL SECTION FILES",
        "  " + THIN,
        f"  {'#':<5} {'Filename':<52} {'Size':>8}",
        "  " + "-" * 68,
    ]

    for fname, size in written:
        # Extract section number from filename for the # column
        try:
            num = int(fname.split("_s")[1].split("_")[0])
        except Exception:
            num = 0
        lines.append(f"  {num:02d}    {fname:<52} {size:>6.1f} KB")

    lines += [
        "  " + "-" * 68,
        f"  {'Total:':<57} {total_kb:>6.1f} KB",
        f"  {'Files:':<57} {len(written)}",
        "",
    ]

    with open(manifest_filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return manifest_file


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    ticker     = args.ticker.upper().strip()
    trade_date = args.trade_date or get_default_trade_date()
    base_name  = args.base_name or f"{ticker}_{trade_date}"
    ticker_folder = ticker
    generated  = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    # Create/overwrite ticker folder
    if os.path.exists(ticker_folder):
        shutil.rmtree(ticker_folder)
    os.makedirs(ticker_folder, exist_ok=True)

    print(f"\n{'═'*62}")
    print(f"  Hedge Fund Swarm — Market Data Collector")
    print(f"  Ticker     : {ticker}")
    print(f"  Trade date : {trade_date}")
    print(f"  Base name  : {base_name}  (18 files: manifest + 17 sections)")
    print(f"  Folder     : {ticker_folder}")
    print(f"  Comment ≥  : {args.comment_threshold} upvotes → fetch top 10 comments")
    print(f"{'═'*62}\n")

    # Dependency check
    missing = []
    for pkg in ["yfinance", "stockstats", "requests", "pandas"]:
        try: __import__(pkg)
        except ImportError: missing.append(pkg)
    if missing:
        print(f"✗  Missing packages: {', '.join(missing)}")
        print(f"   Run: pip install {' '.join(missing)}")
        sys.exit(1)

    written = build(args, base_name, ticker_folder, trade_date)

    manifest_file = write_manifest(base_name, ticker, trade_date, generated, written, ticker_folder)
    manifest_path = os.path.join(ticker_folder, manifest_file)
    manifest_kb   = os.path.getsize(manifest_path) / 1024
    total_kb      = sum(size for _, size in written)

    print(f"\n{'═'*62}")
    print(f"  ✓  Data collection complete")
    print(f"  Manifest  : {os.path.join(ticker_folder, manifest_file)}  ({manifest_kb:.1f} KB)")
    print(f"  Sections  : {len(written)} files  ({total_kb:.1f} KB total)")
    print(f"\n  Pass the manifest path to each analyst skill:")
    print(f"    {os.path.join(ticker_folder, manifest_file)}")
    print(f"{'═'*62}\n")


if __name__ == "__main__":
    main()
