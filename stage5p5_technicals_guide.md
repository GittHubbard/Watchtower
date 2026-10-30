# Stage 5.5 Technicals — User Guide

A plain-English guide to what the Stage 5.5 technical signal does, **when its
signals fire**, **what the indicator states mean**, and **how to read convergence
and divergence**. It describes the code as actually built in `stage5p5.py` (which
deviates from the original master spec in a few places — those are flagged).

> **This is not financial advice.** Every signal here is a lagging,
> technical-analysis heuristic for research and human review. Divergences can
> persist for a long time before price turns. Stage 5.5 **does not gate** the core
> EPS-growth screen — it is logged as context on the memo for a human to weigh.

---

## 1. What it is, in one paragraph

For one ticker, Stage 5.5 takes ~14 months of daily OHLCV bars, computes a suite of
technical indicators, classifies each into a categorical **state**, detects
**divergences** between price and momentum, scores everything into a single number
in **−10 … +10**, and maps that number to a recommendation
(**STRONG_SELL → SELL → HOLD → BUY → STRONG_BUY**). It also pulls analyst /
fundamental context (coverage, consensus, rating changes). The output is two
commentary blocks injected into the daily memo, plus the scalar `tech_score` and
`tech_recommendation`.

---

## 2. How to run it

```bash
# Technical + analyst commentary -> injected into memos/<today>.md
python stage5p5_harness.py                 # whole watchlist
python stage5p5_harness.py AAPL MSFT NVDA  # a subset
python stage5p5_harness.py --limit 10      # first 10 of the watchlist

# Visual verification chart -> charts/<SYM>_stage5p5_<today>.png
python stage5p5_plot.py AAPL MSFT NVDA
```

The harness feeds a **300-bar (~14 month) trailing window** (`TRAILING_BARS`) into
the signal. That window is the outer bound on how far back anything can look.

---

## 3. When does a signal "fire"?

Stage 5.5 recomputes everything **on the latest bar of each run**. A "fire" means
one of:

| Fire | Trigger | Lookback / recency |
|---|---|---|
| **Recommendation change** | The final score crosses a bucket threshold (with hysteresis, §8) | Latest bar |
| **MACD cross** | `macd` crosses its signal line this bar | Latest bar |
| **RSI / Bollinger / CMF / aVWAP state change** | The latest bar enters a new categorical state | Latest bar |
| **Divergence confirmed** | A price↔indicator divergence meets its confirmation rule | Must have confirmed within the last **15 bars** (§6.4) |

Everything except divergences is a **point-in-time** read of the most recent bar —
there is no "stale signal from last year" problem for them. **Divergences** are the
one stateful, historical signal, and they are bounded to a **15-bar (~3-week)
relevance window** for both regular and hidden types. Older divergence fires are
shown on the chart for context but no longer affect the score.

---

## 4. The indicators (what each one measures)

| Indicator | Params | Reads on… |
|---|---|---|
| **MACD** | 12 / 26 / 9 | Momentum & trend turns (fast EMA − slow EMA, vs its own signal line) |
| **RSI** | 14 (Wilder) | Overbought / oversold momentum, 0–100 |
| **CMF** (Chaikin Money Flow) | 20 | Buying vs selling pressure weighted by where price closes in its range, −1…+1 |
| **Bollinger Bands** | 20, 2σ | Volatility envelope; breaks and squeezes |
| **ATR** | 14 (Wilder) | Average true range — the volatility unit used to normalize slopes & moves |
| **ADX / +DI / −DI** | 14 (Wilder) | Trend *strength* (ADX) and direction (+DI vs −DI) |
| **OBV** | — | On-balance volume: cumulative volume signed by daily direction |
| **Anchored VWAP** | major pivot 10, lookback 90 | Volume-weighted average price anchored to the last major swing high & low |
| **Realized vol** | 20, annualized | Recent actual volatility (√252) |
| **IV context** | optional | Implied vol vs realized; **not wired in this repo** → always `UNKNOWN` |

> **IV note:** this deployment has no implied-vol feed, so `vol_regime` is always
> `UNKNOWN` and the IV-premium fields are blank. That is expected, not a bug.

---

## 5. Indicator states and what they mean

Each indicator is reduced to a categorical state. These are what you'll see in the
memo's **Trend** line (the momentum readout).

### 5.1 Price trend (`trend_price`) — 7 states
Based on the ladder of price vs SMA 20 / 50 / 200 and the ATR-normalized slope of
SMA50.

| State | Meaning |
|---|---|
| `STRONG_UP` | Full bullish ladder (price > SMA20 > SMA50 > SMA200) **and** SMA50 rising |
| `UP` | Looser uptrend (price > SMA50 > SMA200) |
| `UP_EXHAUSTING` | Full ladder up but SMA50 slope has gone flat — rally losing drive |
| `NEUTRAL` | No clean structure |
| `DOWN_EXHAUSTING` | Full ladder down but SMA50 slope flat — selloff losing drive |
| `DOWN` | Looser downtrend (price < SMA50 < SMA200) |
| `STRONG_DOWN` | Full bearish ladder **and** SMA50 falling |

### 5.2 RSI trend (`trend_rsi`)
`BULLISH` (RSI > 50 and rising), `BEARISH` (RSI < 50 and falling), else `NEUTRAL`.
Mean-reversion scoring (§7) separately reads the raw RSI zones:
oversold (20–30), extreme-oversold (≤20), overbought (70–80), extreme-overbought (≥80).

### 5.3 MACD trend (`trend_macd`)
`BULL_CROSS` / `BEAR_CROSS` (crossed its signal line this bar), `BULLISH` /
`BEARISH` (above / below signal line), else `NEUTRAL`. Also a `macd_above_zero` flag.

### 5.4 Bollinger trend (`trend_bb`)
`UPPER_BREAK` / `LOWER_BREAK` (close outside a band), `SQUEEZE` (band width in the
bottom 20% of its 120-bar range — low volatility, expansion often follows),
`EXPANSION` (top 20%), else `NEUTRAL`.

### 5.5 CMF trend (`trend_cmf`)
`STRONG_ACCUM` (> 0.10) / `ACCUM` (> 0.05) = net buying;
`STRONG_DIST` (< −0.10) / `DIST` (< −0.05) = net selling; else `NEUTRAL`.

### 5.6 Volume (`trend_volume`)
`SURGE` (today's volume in the top 15% of the last 60 bars), `DRY` (bottom 15%),
else `NORMAL`. Plus `vol_confirms_up/down`: a move with volume ≥ 70th percentile.

### 5.7 OBV trend (`trend_obv`)
`BULLISH` (OBV above its 20-bar SMA and rising), `BEARISH` (below and falling),
else `NEUTRAL`.

### 5.8 Anchored-VWAP state (`trend_avwap`)
Where price sits relative to the VWAPs anchored at the last major swing high and low:
`RECLAIM_HIGH` (just crossed back above the high-anchor VWAP — bullish event),
`BREAKDOWN_LOW` (just lost the low-anchor VWAP — bearish event), `ABOVE_BOTH`,
`BELOW_BOTH`, `TESTING_RESISTANCE`, `TESTING_SUPPORT`, `BETWEEN`, `UNAVAILABLE`.

### 5.9 Regime (`adx_regime`, `vol_regime`)
- **ADX regime:** `TRENDING` (ADX ≥ 25), `WEAK_TREND` (20–25), `RANGING` (< 20).
- **Vol regime:** `LOW_IV` / `NORMAL_IV` / `HIGH_IV` / `UNKNOWN` (always `UNKNOWN`
  here — no IV feed).

The two regimes combine into a **gate** that scales the mean-reversion component
(§7). Intuition: mean-reversion bets are trusted in **ranging** markets and
distrusted in strong **trends** (where you don't fade the move).

---

## 6. Convergence and Divergence — the core idea

This is the most important and least intuitive part, so read it slowly.

### 6.1 The one-sentence definition
**Divergence** = price and a momentum indicator **disagree** about direction.
**Convergence** = they **agree**. Disagreement is an early warning of a possible
turn; agreement reinforces the current trend.

Why it works: indicators like RSI/MACD/OBV/CMF measure the *force* behind a move. If
price keeps making new highs but the force behind each push is weaker (indicator
makes lower highs), the trend is running out of fuel — even though price hasn't
turned yet.

### 6.2 The four divergence patterns

Take two consecutive pivots in price, `a` (older) then `b` (newer):

| Pattern | Price does… | Indicator does… | Reading |
|---|---|---|---|
| **Regular Bullish** | Lower low | Higher low | Downtrend losing steam → possible **bottom / reversal up** |
| **Regular Bearish** | Higher high | Lower high | Uptrend losing steam → possible **top / reversal down** |
| **Hidden Bullish** | Higher low | Lower low | Pullback held → **uptrend continuation** |
| **Hidden Bearish** | Lower high | Higher high | Bounce failed → **downtrend continuation** |

- **Regular** divergences are **reversal** signals (price about to change direction).
- **Hidden** divergences are **continuation** signals (the existing trend resumes
  after a pullback). Hidden bull/bear are only counted when the SMA50 slope agrees
  with the trend direction.

Stage 5.5 runs this on five indicators: **RSI, MACD line, MACD histogram, CMF,
OBV**. (OBV often leads CMF, because CMF is bounded ±1 and saturates while OBV is
unbounded.)

### 6.3 Quality boost
A regular divergence is **"strong"** (worth extra points) if, at the first pivot,
the indicator was already in its extreme zone — e.g. RSI < 30 for a bullish
divergence, RSI > 70 for bearish, CMF beyond ±0.05, MACD on the right side of zero.
OBV has no natural extreme zone, so its divergences are never quality-boosted.

### 6.4 The divergence state machine (regular only)
A **regular** divergence isn't trusted the instant it appears — price has to confirm
it:

```
NONE → PENDING → CONFIRMED → (expires)
```

- **PENDING:** the divergence shape exists, but price hasn't followed through. This
  is a *watch*, not a trigger — it does **not** score.
- **CONFIRMED:** for a bullish divergence, price closes back **above the first
  pivot's level AND above SMA20** (symmetric for bearish). Only `CONFIRMED`
  divergences score.
- **Expiry / recency:** if price doesn't confirm within ~20 bars the divergence
  expires. Once confirmed, it stays "live" for **15 bars** (`effect_bars`), then
  ages out. **Hidden** divergences confirm immediately at the pivot but are held to
  the **same 15-bar window** (a recent fix — previously they never expired; see
  `stage_5p5_update.md`).

So the practical rule: **a divergence only matters if it confirmed within the last
~15 trading days.**

### 6.5 Convergence (and how it's actually implemented here)
Textbook convergence = price higher-high **and** indicator higher-high (or both
lower-lows): the trend is confirmed by its momentum, so trend-following is favored.

> **As-built deviation:** `stage5p5.py` does **not** detect pivot-based convergence.
> Instead it uses a proxy — if the MACD-histogram's 5-bar slope agrees with the sign
> of the trend score, it multiplies the trend score by **×1.25**. Treat the
> "convergence" multiplier as "momentum agrees with trend," not as a formal
> price/indicator HH-HH confirmation.

### 6.6 How to interpret them in practice
- A **confirmed regular** divergence against a tiring trend is the strongest
  contrarian tell — especially if it's "strong" quality and shows up on 2+
  indicators at once.
- **Hidden** divergences are your "the trend is still intact, this dip is a
  continuation" reassurance.
- **Divergence is early, not exact.** It can fire repeatedly before price actually
  turns. That's why only *confirmed* ones score, and why they expire.
- **Convergence/agreement** says "don't fight the trend" — it amplifies trend-
  following rather than triggering anything on its own.

---

## 7. How the score is built

> **Redesigned (Sep 2026).** The score used to be dominated by the trend group
> (up to ±13.75 points), which in practice measured momentum — and momentum proved
> inversely predictive of 1-month forward alpha. The score now looks for stocks
> **set up for a leg up**: pulled back, not overextended, with signs of buying.

The final score (−10 … +10) is the sum of four parts:

```
pullback      = clip(−0.75 × (close − SMA20) / ATR20, −3, +3)    # below the 20-day → positive
              + clip((50 − RSI14) / 10, −2, +2)                  # low RSI → positive
overextension = −1 for each of: close above upper Bollinger band, RSI ≥ 75,
                >15% above SMA50, >25% above SMA50, >3 ATR above SMA20   (never positive)
confirmation  = +0.5 CMF > 0.05 and +0.5 OBV bullish   (only while price is below SMA20)
              + 0.5 MACD bull cross while MACD < 0      (momentum turning up from a dip)
              + divergence score / 4, capped at ±1
momentum      = legacy trend score × convergence / 11, capped at ±1
score         = clamp(sum, −10, +10);  confidence = |score| / 10
```

- **Pullback** is the core. It rewards buying weakness and penalises chasing.
- **Overextension** adds explicit penalties for stretched names.
- **Confirmation** rewards evidence that the dip is being bought.
- **Momentum** (the old trend group: price state, MACD, CMF, OBV, aVWAP, volume)
  is still computed and shown on the ticket, but can move the score by ±1 at most.
- The old ADX × IV mean-reversion gate is gone; ADX/IV regime and price vs the
  200-day are shown as context only.

### 7.1 How the redesign tested

`research/backtest_setup_score.py` runs the production scorer every 5 trading
days over ~10 years of daily prices for ~512 watchlist names (2017–2026,
~154k samples on full-universe dates), rebuilding the old score alongside.
"Excess" = return minus the average watchlist stock over the same window.

| | Old score 2017–21 | Old 2022–26 | New score 2017–21 | New 2022–26 |
|---|---|---|---|---|
| Rank-IC vs 60-day excess | −0.021 | −0.002 | **+0.026** | **+0.015** |
| Buy/Strong Buy: mean 60-day excess | −0.26% | +0.05% | **+1.02%** | **+0.37%** |
| Buy/Strong Buy: median 60-day excess | −0.50% | −0.99% | **+0.32%** | −0.28% |

The new score is better than the old one in both halves, but the edge over ten
years is **small** — much smaller than in the Jun–Sep 2026 live log (where new-style
Buys beat the average stock by ~3% a month). Treat the rating as one input, not a
stand-alone trading signal. Caveats: current index members only (survivorship
bias); divergence and aVWAP use the same code as production; IV regime is not
available historically.

## 8. Recommendation buckets and hysteresis

The score maps to a recommendation with a **Schmitt trigger** (hysteresis) so it
doesn't flip on noise:

| Bucket | Enter at | Exit at |
|---|---|---|
| `STRONG_BUY` | score ≥ 5.5 | < 4.5 |
| `BUY` | ≥ 2.5 | < 1.5 |
| `HOLD` | (between) | — |
| `SELL` | ≤ −2.5 | > −1.5 |
| `STRONG_SELL` | ≤ −5.5 | > −4.5 |

Because of the gap between enter and exit thresholds, a recommendation only changes
when the score moves decisively — a value bouncing around 2.5 won't toggle
BUY↔HOLD every day. The previous day's recommendation is persisted in
`state/stage5p5_recs.json` so this carries across runs.

---

## 9. Reading the chart (`stage5p5_plot.py`)

Seven stacked panels share a date axis: **Price** (Close, SMA 20/50/200, Bollinger
fill, anchored-VWAP high/low with ★ anchors), **Volume**, **RSI**, **MACD**,
**CMF**, **OBV**, **ADX/±DI**.

**Marker legend:**

| Marker | Meaning |
|---|---|
| `^` green / `v` red | Divergence fire — bull / bear |
| filled / hollow marker | Regular / hidden divergence |
| solid / dotted leg | Regular / hidden divergence pivot leg |
| Coloured **leg line** between two dots | The two pivots of the most-recent confirmed divergence (drawn on price **and** the indicator, so you can see the opposing slopes) |
| `^` / `v` on the MACD line | MACD bull / bear cross |
| small `^` / `v` on price | Bollinger lower / upper break |
| ★ on price | Anchored-VWAP anchor pivot |
| Gold shaded band (right edge) | The **15-bar live window** — only fires inside it still count toward the score |

The chart intentionally shows the **full history** of divergence fires, but only
those inside the gold band are scored. This is exactly how you "verify the signal
fired for the right reason": find the marker, then look left at the price-vs-
indicator leg to confirm the lower-low/higher-low (or whichever pattern) is real.

---

## 10. Caveats and known deviations

- **Educational only** — lagging indicators; divergences can persist; no costs,
  sizing, or risk management.
- **No IV feed** → `vol_regime` is always `UNKNOWN`.
- **As-built differs from the master spec** in: MACD trend weights, the redesigned
  mean-reversion RSI bands, the MACD-histogram-slope "convergence" proxy (no true
  pivot convergence), `trend_avwap` thresholds, and the absence of the spec's
  vol-context caveats / warmup gate. These are intentional and documented in
  `stage_5p5_update.md`.
- **Forward-looking only** — like the rest of the screener, treat this as one input
  among many, backtested on your own instrument and timeframe before relying on it.

---

### See also
- `stage5p5.py` — the implementation.
- `stage5p5_harness.py` — wiring into the screener / memo.
- `stage5p5_plot.py` — the visual verifier.
- `stage_5p5_update.md` — the bug fixes and deviations log.
- `build_trade_signal_script_master.md` — the original master spec this derives from.
