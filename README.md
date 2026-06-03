# gamma_workflow

Options-driven entry signal system for a long-only book. Reads dealer GEX, walls, flow,
and price/volume signals end-of-day, classifies regime, fires playbook triggers, and
emits markdown trade tickets for human review.

## Status

**Complete pipeline.** All stages (1 calendar, 2 regime, 3 levels, 4 flow,
5 confirmation, 5.5 technical/fundamental context, 6 playbooks, 7 ticket,
8 monitor, 9 logging) plus the EOD orchestrator are built and tested (131 tests).
The only piece that requires your environment is the engine
(`GeminiRevisionGammaProfile3.py`) — drop it at the repo root and the
orchestrator's data-acquisition layer wires into it.

## Layout

```
gamma_workflow/        # python package
  engine_shim.py       # thin wrapper over GeminiRevisionGammaProfile3.py
  config.py            # YAML loaders
  rates.py             # FRED risk-free rate (DGS3MO), cached daily
  storage.py           # partitioned-parquet read/write + positions + chain snapshots
                       #   + fundamentals evolution cache (append-only, change-only)
  stage1_calendar.py   # OPEX, VIXP (deterministic), earnings (yfinance), macro (CSV)
  stage2_regime.py     # 4-state regime classifier + transition tracking
  stage3_level_map.py  # walls (primary+secondary, 2% sep) + migration + persistence + HVL
  stage4_flow.py       # skew (25d), PCR + z-scores, OI delta, term structure, IV/HV ratio
  stage5_confirm.py    # volume z, ATR, MA distance, VIX 5d change, HV(30d)
  stage5_5_context.py  # Stage 5.5: technicals (MACD/RSI/CMF/Bollinger + ADX/OBV/aVWAP/
                       #   divergence scoring) + fundamentals (analyst/earnings/health);
                       #   generates markdown commentary for the ticket
  stage6_playbooks.py  # rule engine — Playbooks A/B/C
  stage7_ticket.py     # sizing + markdown trade ticket builder (+ IV/HV notes & commentary)
  stage8_monitor.py    # open-position exit/trail/scale logic
  stage9_logging.py    # feature logger + forward-returns/benchmark/alpha backfill
  run_eod.py           # orchestrator: stages 1->9, called by the GitHub Action

trade_signals.py       # standalone CLI: run Stage 5.5 technical analysis on any ticker
                       #   without the engine (yfinance only)

config/
  watchlist.yaml       # tickers + per-ticker overrides (kind: index_etf | single_name)
  thresholds.yaml      # ALL tunable thresholds (regime, flow, playbooks, sizing,
                       #   iv_hv, backfill, fundamentals, acquisition)
  earnings.csv         # optional manual earnings override (yfinance is primary)
  macro_events.csv     # FOMC/CPI/NFP dates (manual)

data/                  # committed parquet + tickets, partitioned by date
  features/date=YYYY-MM-DD.parquet
  signals/date=YYYY-MM-DD.parquet
  chains/ticker=XXX/date=YYYY-MM-DD.parquet   # for OI-delta day-over-day
  fundamentals/ticker=XXX/{analyst,financials}/date=*.json  # change-only evolution cache
  tickets/YYYY-MM-DD_TICKER_PLAYBOOK.md
  positions.json       # PM-maintained open positions (Stage 8 input)
  rates/dgs3mo.parquet

.github/workflows/eod.yml   # scheduled EOD run, weekdays 21:30 UTC
```

## Running locally

```bash
pip install -e .[dev]
# Drop GeminiRevisionGammaProfile3.py at the repo root first.
python -m gamma_workflow.run_eod        # full EOD pipeline
pytest                                  # 131 tests, no network/engine needed

# Standalone technical analysis (no engine required):
python trade_signals.py AAPL
python trade_signals.py SPY --bar -1 --bars-since 400
python trade_signals.py AAPL --iv-csv iv_data.csv --params params.yaml
```

## The three playbooks (Stage 6)

| ID | Name | Regime | Instrument | Tranche |
|----|------|--------|------------|---------|
| A | Washout at Put Wall | deep_negative | shares | 50% |
| B | Gamma Flip Reclaim | negative -> transitional_positive | shares | 75% |
| C | Compression Breakout | deep_positive (>=5 sessions) | defined-risk calls | 50% |

All gates and thresholds live in `config/thresholds.yaml` — tune quarterly, no
code changes.

## Stage 5.5 — Technical & Fundamental Context

Stage 5.5 runs after Stage 5 (confirmation) and before Stage 6 (playbooks). It does
not gate the playbooks; it enriches the ticket with context for human review.

**Scalar features** (logged to parquet, available for future signal research):

| Group | Scalars |
|-------|---------|
| Technicals | `macd`, `macd_signal`, `macd_hist`, `rsi_14`, `cmf_20`, `bb_upper/mid/lower/width` |
| Analyst | `analyst_consensus`, `analyst_coverage_count`, `analyst_pct_buy`, `analyst_upgrades/downgrades/steady_3m` |
| Earnings | `forward_pe`, `eps_surprise_avg_4q`, `eps_beat_streak`, `eps_estimate_trend` |
| Balance sheet | `net_income_margin_ttm`, `debt_to_equity`, `revenue_growth_yoy` |

**Ticket commentary** (in-memory only, rendered in `## Commentary` section):

- *Technical commentary*: ADX regime, vol regime (IV percentile vs realized vol), trend
  classifications across 8 indicators (price, MACD, RSI, CMF, OBV, Bollinger, volume,
  anchored VWAP), divergence detection (RSI/MACD/CMF/OBV with PENDING→CONFIRMED→EXPIRED
  state machine), 2-D regime-gated scoring, and caveats (event risk, vol crush, staleness).
- *Analyst commentary*: coverage count, % Buy/Overweight, 90-day upgrade/downgrade/steady
  counts, and a trend narrative built from the evolution cache.

**Index ETF handling** (SPY, QQQ, IWM, etc.):

- Technical commentary runs normally — all indicators are pure OHLCV.
- Fundamental scalars are all NaN; analyst commentary is empty.
- The fundamentals network fetch is skipped entirely (no yfinance calls for ETF fundamentals).
- All three playbooks remain eligible; Playbook A enforces VIX confirmation only for ETFs.

## IV / HV Ratio (Stage 4 × Stage 5)

`ivx_30d` (30-day IV index, **percent**) is computed by the engine in Stage 4.
`hv_30d` (30-day realized vol, **decimal**) is computed from price history in Stage 5.
The ratio `iv_hv_ratio_30d = (ivx_30d / 100) / hv_30d` is dimensionless (≈ 1 when IV == HV).

- Playbook A: z-score used as a strength *booster*.
- Playbook C: adaptive *reducer* (×0.5) when IV is rich — bootstrap threshold (ratio > 1.3)
  until 20 days of history, then z-score > 1.0 (≈ 85th percentile). Noted on the ticket.

## Forward Returns & Alpha (Stage 9)

Five horizons: 1 / 5 / 20 / 60 / 90 trading days. Backfilled nightly for all prior dates
once sufficient calendar days have elapsed per horizon (independent gating — 1d fills
promptly; 90d waits ~140 calendar days). SPY benchmark (configurable via
`backfill.benchmark_ticker`) is fetched **once** per run and reused across all tickers.

`alpha_Nd = fwd_ret_Nd − benchmark_ret_Nd`

## Standalone Signal Script

`trade_signals.py` (repo root) runs the full Stage 5.5 technical analysis pipeline
without the engine or any options data. Useful for quick analysis of any ticker:

```
python trade_signals.py AAPL
```

Output includes: score (+/- 10 scale), recommendation (STRONG_BUY → STRONG_SELL),
regime classification (ADX × vol), trend component breakdown, divergence events,
mean-reversion gate status, and risk caveats.

## Conventions

- All times are UTC in storage. Display in ET as needed.
- The engine (`GeminiRevisionGammaProfile3.py`) is treated as legacy. Do not refactor
  it; wrap it via `engine_shim`. A clean port is a separate task.
- No auto-execution. Output is markdown tickets + a daily features/signals parquet.
- NaN z-scores mean "not enough history" and are treated as "no signal" by the
  playbooks — never as "passes the gate".
- Index ETFs (kind: index_etf) skip fundamentals fetch; all other pipeline stages
  run identically for ETFs and single names.
