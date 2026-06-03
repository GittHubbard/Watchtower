# gamma_workflow

Options-driven entry signal system for a long-only book. Reads dealer GEX, walls, flow,
and price/volume signals end-of-day, classifies regime, fires playbook triggers, and
emits markdown trade tickets for human review.

## Status

**Complete pipeline.** All stages (1 calendar, 2 regime, 3 levels, 4 flow,
5 confirmation, 5.5 technical/fundamental context, 6 playbooks, 7 ticket,
8 monitor, 9 logging) plus the EOD orchestrator are built and tested (127 tests).
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
  stage1_calendar.py   # OPEX, VIXP (deterministic), earnings (yfinance), macro (CSV)
  stage2_regime.py     # 4-state regime classifier + transition tracking
  stage3_level_map.py  # walls (primary+secondary, 2% sep) + migration + persistence + HVL
  stage4_flow.py       # skew (25d), PCR + z-scores, OI delta, term structure, IV/HV ratio
  stage5_confirm.py    # volume z, ATR, MA distance, VIX 5d change, HV(30d)
  stage5_5_context.py  # technicals (MACD/RSI/CMF/Bollinger) + fundamentals (analyst/earnings/health)
  stage6_playbooks.py  # rule engine — Playbooks A/B/C
  stage7_ticket.py     # sizing + markdown trade ticket builder (+ notes & commentary)
  stage8_monitor.py    # open-position exit/trail/scale logic
  stage9_logging.py    # feature logger + forward-returns/benchmark/alpha backfill
  run_eod.py           # orchestrator: stages 1->9, called by the GitHub Action

config/
  watchlist.yaml       # tickers + per-ticker overrides
  thresholds.yaml      # ALL tunable thresholds (regime, flow, playbooks, sizing)
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

```
pip install -e .[dev]
# Drop GeminiRevisionGammaProfile3.py at the repo root first.
python -m gamma_workflow.run_eod        # full EOD pipeline
pytest                                  # 96 tests, no network/engine needed
```

## The three playbooks (Stage 6)

| ID | Name | Regime | Instrument | Tranche |
|----|------|--------|------------|---------|
| A | Washout at Put Wall | deep_negative | shares | 50% |
| B | Gamma Flip Reclaim | negative -> transitional_positive | shares | 75% |
| C | Compression Breakout | deep_positive (>=5 sessions) | defined-risk calls | 50% |

All gates and thresholds live in `config/thresholds.yaml` — tune quarterly, no
code changes.

## Conventions

- All times are UTC in storage. Display in ET as needed.
- The engine (`GeminiRevisionGammaProfile3.py`) is treated as legacy. Do not refactor
  it; wrap it via `engine_shim`. A clean port is a separate task.
- No auto-execution. Output is markdown tickets + a daily features/signals parquet.
- NaN z-scores mean "not enough history" and are treated as "no signal" by the
  playbooks — never as "passes the gate".
