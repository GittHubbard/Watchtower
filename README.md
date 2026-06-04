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
  stage5_5_context.py  # Stage 5.5: technicals (9 trend classifiers — price 7-state,
                       #   MACD w/ cross detect, RSI, BB w/ squeeze, CMF 4-tier,
                       #   vol_pct, OBV, aVWAP; 5-indicator divergence suite incl.
                       #   hidden; Schmitt-trigger hysteresis) + fundamentals
                       #   (analyst/earnings/health); emits per-ticker tech-signal
                       #   ticket and routes upward/downward watchlist transitions
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
  chains/ticker=XXX/date=YYYY-MM-DD.parquet     # for OI-delta day-over-day
  chains/ticker=XXX/tickets/YYYY-MM-DD_XXX_techsignal.md  # Stage 5.5 per-ticker signal ticket
  Upward-Watchlist/YYYY-MM-DD/                  # tickers transitioning toward bullish
  Downward-Watchlist/YYYY-MM-DD/                # tickers transitioning toward bearish
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
not gate the playbooks; it enriches the ticket with context for human review and
emits a standalone per-ticker tech-signal ticket.

**Scalar features** (logged to parquet, available for future signal research):

| Group | Scalars |
|-------|---------|
| Technicals | `macd`, `macd_signal`, `macd_hist`, `rsi_14`, `cmf_20`, `bb_upper/mid/lower/width` |
| Signal summary | `tech_recommendation` (STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL), `tech_score` (−10..+10) |
| Analyst | `analyst_consensus`, `analyst_coverage_count`, `analyst_pct_buy`, `analyst_upgrades/downgrades/steady_3m` |
| Earnings | `forward_pe`, `eps_surprise_avg_4q`, `eps_beat_streak`, `eps_estimate_trend` |
| Balance sheet | `net_income_margin_ttm`, `debt_to_equity`, `revenue_growth_yoy` |

### Trend classification (9 indicators)

| Indicator | States |
|-----------|--------|
| Price (7-state) | `STRONG_UP` / `UP` / `UP_EXHAUSTING` / `NEUTRAL` / `DOWN_EXHAUSTING` / `DOWN` / `STRONG_DOWN` (ATR-normalized SMA50 slope on full MA stack) |
| MACD | `BULL_CROSS` / `BEAR_CROSS` (detected via prev bar) / `BULLISH` / `BEARISH` / `NEUTRAL` + `macd_above_zero` |
| RSI | `BULLISH` (20–30) / `BEARISH` (70–80) / `OVERSOLD` (≤20) / `OVERBOUGHT` (≥80) / `NEUTRAL` |
| Bollinger | `UPPER_BREAK` / `LOWER_BREAK` / `SQUEEZE` / `EXPANSION` (120-bar bb_width percentile, top/bottom 20%) / `NEUTRAL` |
| CMF (4-tier) | `STRONG_ACCUM` (>0.10) / `ACCUM` (>0.05) / `DIST` / `STRONG_DIST` (<−0.10) / `NEUTRAL` |
| Volume | `SURGE` (vol_pct ≥0.85) / `DRY` (≤0.15) / `NORMAL` + `vol_confirms_up`/`vol_confirms_down` flags (60-bar rolling percentile rank) |
| OBV | `BULLISH` / `BEARISH` / `NEUTRAL` (OBV vs SMA20 + slope) |
| Anchored VWAP | `RECLAIM_HIGH` / `BREAKDOWN_LOW` / `ABOVE_BOTH` / `BELOW_BOTH` / `TESTING_SUPPORT` / `TESTING_RESISTANCE` |
| ADX regime | `TRENDING` (≥25) / `WEAK_TREND` (≥20) / `RANGING` |

### Divergence suite (5 indicators × 2 types)

RSI, MACD line, MACD histogram, CMF, OBV — each scanned for **regular** and **hidden**
divergences (MACD line: regular only). Pivot pairs are filtered by ATR-normalized price
move ≥ 0.25 to suppress noise.

- **Regular divergences** run through a `PENDING → CONFIRMED → EXPIRED` state machine
  (20-bar expiry, 15-bar effect window). Only `CONFIRMED` events score.
- **Hidden divergences** confirm immediately at the right-shoulder; SMA50 slope must
  agree (bullish hidden needs an up-trending SMA50, bearish hidden needs down-trending).
- **Quality tier** (`strong` vs `standard`) is set from the p1 indicator value (e.g. RSI
  bull-div is `strong` if p1 RSI < 30). `strong` adds a +0.5 score bonus.

### Scoring (master spec §5)

```
trend_score   = price + macd (+ cross bonus, ±0.5 above-zero) + cmf + obv + avwap + vol_confirms
base_trend    = trend_score × conv_mult     (1.25× when MACD-hist direction agrees with trend)
mr_score      = RSI (BULLISH ±1, OVERBOUGHT/OVERSOLD ±0.5 contra) + BB (±0.5)
mr_contrib    = mr_weight[adx_regime][vol_regime] × mr_score   (2-D regime gate)
div_score     = Σ confirmed events (reg ±2, hidden ±1, strong +0.5, ≥2 same-side ±1 stacking)
raw           = base_trend + mr_contrib + div_score
score         = clamp(raw, −10, +10)
```

**Recommendation** (Schmitt-trigger hysteresis, `tech_recommendation` column):

| State | Enter | Exit (drop one tier) |
|-------|-------|----------------------|
| `STRONG_BUY` | score ≥ 5.5 | score < 4.5 |
| `BUY` | score ≥ 2.5 | score < 1.5 |
| `HOLD` | — | — |
| `SELL` | score ≤ −2.5 | score > −1.5 |
| `STRONG_SELL` | score ≤ −5.5 | score > −4.5 |

Yesterday's `tech_recommendation` is read from `data/features/` to seed the
hysteresis state.

### Ticket output (Stage 5.5 → disk)

Each ticker gets a markdown tech-signal ticket every run:

- **Always written**: `data/chains/ticker=XXX/tickets/YYYY-MM-DD_XXX_techsignal.md`
  — full commentary (regime, trend breakdown, divergence events, MR gate, score
  arithmetic, caveats).
- **Watchlist routing** when the recommendation transitions vs yesterday:
  - `data/Upward-Watchlist/YYYY-MM-DD/`: `HOLD→BUY`, `HOLD→STRONG_BUY`,
    `BUY→STRONG_BUY`
  - `data/Downward-Watchlist/YYYY-MM-DD/`: `BUY→SELL`, `STRONG_BUY→SELL`,
    `STRONG_BUY→STRONG_SELL`, `SELL→STRONG_SELL`

The Stage 7 playbook ticket also gets a one-line tech signal summary in its
header (`**Tech Signal**: BUY (+3.4)  |  [full analysis in data/chains/...]`).

**In-ticket commentary** (rendered when a playbook fires):

- *Technical commentary*: ADX × vol regime, MR weight, 9-indicator trend breakdown,
  aVWAP anchor ages, RSI/BB MR labels, divergence event list, score arithmetic
  line, caveats (warmup, ADX/IV missing).
- *Analyst commentary*: coverage count, % Buy/Overweight, 90-day rating-change
  counts, and a trend narrative built from the evolution cache.

**Index ETF handling** (SPY, QQQ, IWM, etc.):

- Technical commentary and tech-signal ticket run normally — all indicators are pure OHLCV.
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

`trade_signals.py` (repo root) is a self-contained rebuild of the Stage 5.5
technical pipeline — no `gamma_workflow` imports, no engine, yfinance only.
Useful for quick analysis of any ticker.

```bash
python trade_signals.py AAPL                          # latest bar
python trade_signals.py SPY --bar -1 --bars-since 400 # analyze a specific historical bar
python trade_signals.py AAPL --iv-csv iv_data.csv     # bring your own IV series
python trade_signals.py AAPL --params params.yaml     # override thresholds
python trade_signals.py --test                        # 11 built-in tests, no network
```

Output (`§6` report format) includes: score (±10 scale), recommendation
(STRONG_BUY → STRONG_SELL) after Schmitt-trigger hysteresis, ADX × vol regime
classification, 9-indicator trend component breakdown, divergence event list,
MR gate status, **SCORE ARITHMETIC** section (base_trend + mr_contrib + div_score
= raw → clamped), and risk caveats.

Internals match the in-pipeline `stage5_5_context.py`: 7-state price trend with
ATR-normalized SMA50 slope, BULL_CROSS/BEAR_CROSS detection, BB squeeze/expansion,
4-tier CMF, vol_pct percentile, O(n) anchored VWAP with anchor-reset, 5-indicator
divergence suite (regular state machine + immediate-confirm hidden divs), and the
2-D regime-gated MR contribution.

## Conventions

- All times are UTC in storage. Display in ET as needed.
- The engine (`GeminiRevisionGammaProfile3.py`) is treated as legacy. Do not refactor
  it; wrap it via `engine_shim`. A clean port is a separate task.
- No auto-execution. Output is markdown tickets + a daily features/signals parquet.
- NaN z-scores mean "not enough history" and are treated as "no signal" by the
  playbooks — never as "passes the gate".
- Index ETFs (kind: index_etf) skip fundamentals fetch; all other pipeline stages
  run identically for ETFs and single names.
