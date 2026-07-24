# Handoff: Flip the buy/sell signals toward the contrarian process

**Mode**: Checkpoint (transferring a completed analysis effort + launching a new initiative)
**Created**: 2026-07-24
**Branch**: `claude/trade-log-watchlist-script-6tupum` (PR #9)

## The mission for the next agent
Rework the signal process to be **contrarian**: effectively **flip the buy and sell
signals**. The evidence below shows the pipeline's bullish ratings (STRONG_BUY on
strength/momentum) *lost* alpha over the sample, while buying washed-out/oversold names
*earned* alpha. The next agent should turn that finding into a real process change — not
just a backtest toggle — and decide **where** the flip belongs (signal generation vs.
routing vs. strategy defaults). Treat this as a design task first: propose a plan before
editing production signal code.

## Where we are (what's already done)
Over this session we built, from scratch, an analysis stack on top of the existing
`gamma_workflow` EOD pipeline and validated a contrarian edge **offline** (Yahoo is blocked
in this environment; we priced everything from the local feature store instead). All work
is committed and pushed to PR #9. Nothing is in progress / uncommitted.

## Key results (in-sample, offline, SPY-benchmarked)
Full tech-signal log = 17,523 signals, 516 tickers, 2026-06-04 → 2026-07-22, every ticker
every day incl. HOLD.

| Strategy | Avg alpha | Median | Win rate | Beat-SPY |
|---|---|---|---|---|
| STRONG_BUY → SELL (as designed) | **−0.89%** | −1.04% | 38.4% | 39.7% |
| Contrarian: buy RSI<40 washout, exit RSI>55 or 15d | **+2.49%** | +1.69% | 59.8% | 62.1% |

Regression (P(positive forward alpha) on the technical features, both 5- and 10-day
horizons, consistent): **mean-reversion signature.** Oversold / DOWN_EXHAUSTING / low-RSI /
bollinger lower-break / low-IV names → positive forward alpha; overbought / STRONG_UP /
TRENDING / **high IV30 (strongest numeric predictor, z≈−6.5)** → negative. This is exactly
why the momentum-driven STRONG_BUY rating underperformed.

**Caveats (do not over-trust):** ~7 weeks, in-sample, one market regime; forward windows
overlap and the market co-moves, so regression p-values are optimistic — read signs/rankings,
not exact significance; prices are `spot` (close-ish, not split/div-adjusted); `avwap=UNAVAILABLE`
top univariate cell is a data artifact (tiny n), ignore it. **Validate out-of-sample before trusting.**

## Candidate flip points (the core decision)
The tech signal is produced purely from price history in Stage 5.5; the score's sign is the crux.
- **Signal generation (source of truth):**
  - `gamma_workflow/stage5_5_context.py:950` `_score_technical` — builds `tech_score` from trend/MR/divergence; sets `rec` at lines `1126-1141`.
  - `gamma_workflow/stage5_5_context.py:1170-1194` `_apply_hysteresis` — maps score→recommendation with enter/exit thresholds. Flipping = negate the score feeding this, or swap the STRONG_BUY↔STRONG_SELL / BUY↔SELL labels. Cleanest single lever is usually the score sign.
  - `gamma_workflow/stage5_5_context.py:1752` — where `tech_recommendation` is finalized.
- **Watchlist routing:** `gamma_workflow/run_eod.py:259-265` `_UPWARD_TRANSITIONS` / `_DOWNWARD_TRANSITIONS` (used at `:360`). If the score/label is flipped upstream, routing may need no change; if flipping at routing level instead, swap these sets.
- **Backtest strategy defaults (analysis only, not production):** `scripts/build_trade_log.py:341-342` `OPEN_RATINGS` / `CLOSE_RATINGS`.

Recommend deciding at the **score level** (one place, keeps hysteresis/labels coherent) and
re-running the analysis to confirm the flipped signals now carry positive alpha.

## Files to look at first
- `gamma_workflow/stage5_5_context.py:950-1194` — signal scoring + score→recommendation mapping (the flip target).
- `gamma_workflow/run_eod.py:259-265, 344-363` — recommendation → ticket + watchlist routing, and the parquet-log append.
- `scripts/analyze_techsignal_log.py` — offline backtest+regression driver; `CONTRARIAN` preset at `:81`; feature-store price loader (`load_spot_panel`, `make_feature_store_loader`).
- `scripts/build_trade_log.py` — parser + feature extraction (`parse_signal_text`, `build_trade_log_from_log`), `simulate_trades` (now takes `open_ratings`/`close_ratings`/`close_filter`/`max_hold_days`), `analyze_performance`, `analyze_signal_predictiveness`.
- `scripts/backfill_techsignal_log.py` — how the log was reconstructed from existing tickets (Mode A).
- `gamma_workflow/storage.py` — `append_techsignal_log` / `read_techsignal_log` (the growing `data/techsignal_log.parquet`).
- `data/techsignal_log.parquet` — the built log (17,523 rows, real IV in 97.5%).
- `data/features/date=*.parquet` — per-ticker/day `spot` + `ivx_30d`; the offline price source.

## How to reproduce (offline, no network)
```bash
pip install pandas numpy pyarrow
python scripts/analyze_techsignal_log.py                 # backtest + regression on full log
python scripts/analyze_techsignal_log.py --backtest --contrarian
python scripts/analyze_techsignal_log.py --regress --horizon 10
```
`--open-any / --open-filter COL[op]VAL / --close-filter / --max-hold` compose for custom strategies.

## Repo state at handoff
- **Branch**: `claude/trade-log-watchlist-script-6tupum` (PR #9 open)
- **Last commits**:
  - `0e6b1f4b8` Generalize backtest entry/exit; add contrarian strategy
  - `cabac4867` Analyze the full tech-signal log offline (backtest + regression)
  - `d01c5492c` Backfill tech-signal parquet log from existing tickets (Mode A)
  - `6c6ebd28f` Accumulate tech-signal tickets into a growing parquet log
  - `7a38c1f7e` Add technical features, feature-gated backtest, and signal regression
- **Uncommitted changes**: none. (Note: running the full `pytest tests/` suite writes stale
  dates into `data/earnings/{MSFT,NVDA}.json` due to a pre-existing test-isolation leak in
  `tests/test_stage1_yfinance.py` — `git restore` those if they reappear; don't commit them.)

## Project docs worth reading
- `README.md`, `USER_GUIDE.md` (repo root) — pipeline overview and Stage 1–9 workflow.
- `stage5p5_technicals_guide.md`, `stage_5p5_update.md` (repo root) — how the Stage 5.5 technical score is constructed (directly relevant to the flip).

## Open questions / pending decisions
- **Where to flip**: score-sign negation vs. label swap vs. routing swap. (Recommend score-level.)
- **Regenerate vs. keep history**: after flipping, do we relabel/rebuild the existing tickets and `techsignal_log.parquet`, or only apply going forward?
- **Out-of-sample validation**: the +2.49% is in-sample/one regime — needs a hold-out or later dates before trusting.
- **IV horizon**: real IV only exists within the chain-snapshot window (≥2026-06-02). Phase 2 (a separate `*_Historic` log from yfinance price history, IV=UNKNOWN pre-window) is still pending and would extend the sample for validation.
- **Environment**: Yahoo Finance is blocked here; all price-based analysis used the feature-store `spot` panel. A machine with Yahoo access can use the default `load_prices_yfinance` loader.

## Suggested first prompt for the next session
> Read `.claude/handoffs/2026-07-24-checkpoint-flip-signals-contrarian.md` and the files it
> links to (especially `gamma_workflow/stage5_5_context.py:950-1194`). We want to make the
> signal process contrarian — flip the buy/sell signals. Propose a plan: where exactly to
> flip (score sign vs. labels vs. routing), what it does to the existing tickets and
> `data/techsignal_log.parquet`, and how you'll re-verify the flipped signals now carry
> positive alpha using `scripts/analyze_techsignal_log.py`. Don't change production signal
> code until the plan is agreed.
