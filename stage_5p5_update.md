# Stage 5.5 Update — Divergence Score Runaway Fix

**File touched:** `stage5p5.py`
**Date:** 2026-06-04
**Scope:** `_score_technical` (scoring) and `_build_technical_commentary` (display)

---

## 1. Summary

While integrating `stage5p5.py` into this screener (via `stage5p5_harness.py`), the
technical signal pinned **every ticker to `+10.0 STRONG_BUY`**. Root cause was the
divergence component of the score: it added points **per detected divergence event**
with no cap and no recency bound, so a long history accumulated hundreds of stale
pivot pairs and saturated the score at the `+10` clamp.

The fix makes the divergence score follow the master spec's **capped tiers**
(`build_trade_signal_script_master.md` §5.4): count **distinct indicators**, not
events. The companion commentary listing was deduped to match.

---

## 2. The issue

### 2.1 Symptom

Running the harness over `TickerBundle.history_5y` (≈1250 daily bars) produced:

```
AAPL: STRONG_BUY (+10.0)
MSFT: STRONG_BUY (+10.0)
NVDA: STRONG_BUY (+10.0)
```

with memo divergence blocks like `**Divergence** (+63.0)` listing the *same*
`hid_bull on RSI` line dozens of times. Every name railed bullish, so the signal
carried no information.

### 2.2 Mechanism

Two compounding factors:

1. **Per-event summation (the bug).** In `_score_technical`, the old divergence
   loop walked every event returned by `_run_divergence_suite` and added a fixed
   amount for each one:

   - regular confirmed: ±2.0 each
   - hidden confirmed: ±1.0 each
   - "strong" quality: +0.5 each
   - plus a +1.0 / −1.0 "stacking" bonus at ≥2 confirmed

   `_detect_divergence_active` returns **one event per qualifying pivot pair**, and
   hidden divergences are confirmed immediately with **no expiry / effect-bars
   recency filter**. Over a multi-year history a single indicator yields many pivot
   pairs, each adding another ±1.0 — so `div_score` grew to +50…+60 and the final
   `clamp(raw, -10, +10)` saturated.

2. **Unbounded input window (the trigger).** The harness initially fed the full
   `history_5y`, maximizing the number of stale pivot pairs the per-event loop
   could sum.

### 2.3 Spec deviation

The master spec (§5.4) never intended per-event accumulation. It specifies **capped
tiers over distinct indicators**:

| Condition | Value |
|---|---|
| `n_reg_bull == 1` | +2 |
| `n_reg_bull >= 2` | +4 |
| any confirmed regular bullish is "strong" (`Q_reg_bull`) | +1 (once) |
| `n_reg_bear == 1` | −2 |
| `n_reg_bear >= 2` | −4 |
| `Q_reg_bear` | −1 (once) |
| any confirmed hidden bullish | +1 (once) |
| any confirmed hidden bearish | −1 (once) |

where `n_reg_bull` / `n_reg_bear` count **divergences across indicators**
{RSI, MACD line, MACD hist, CMF, OBV} — not the number of pivot pairs.

---

## 3. The fix

### 3.1 `_score_technical` — capped, distinct-indicator scoring

Replaced the per-event accumulation with set-based counting of distinct indicators,
plus once-only quality and hidden contributions:

- Derive the indicator from the suite key (`"RSI_reg_bull"` → `"RSI"`).
- Collect `reg_bull_inds` / `reg_bear_inds` as **sets** (so repeated pivot pairs on
  the same indicator collapse to one).
- `q_reg_bull` / `q_reg_bear` are booleans set if **any** confirmed regular div of
  that direction was `"strong"`.
- `hid_bull_any` / `hid_bear_any` are booleans set if **any** confirmed hidden div
  of that direction exists.

Score assembly now mirrors the spec table exactly:

```python
div_score += 4.0 if n_reg_bull >= 2 else (2.0 if n_reg_bull == 1 else 0.0)
div_score -= 4.0 if n_reg_bear >= 2 else (2.0 if n_reg_bear == 1 else 0.0)
if q_reg_bull:  div_score += 1.0
if q_reg_bear:  div_score -= 1.0
if hid_bull_any: div_score += 1.0
if hid_bear_any: div_score -= 1.0
```

`bull_confirmed` / `bear_confirmed` (returned for reporting) are now distinct-indicator
regular counts plus hidden presence, rather than raw event counts.

### 3.2 `_build_technical_commentary` — deduped divergence listing

The narrative previously printed one bullet **per event**, which is what produced the
dozens of identical lines. It now prints **one line per `(state, type, indicator)`**,
preferring a `"strong"` quality label when any event for that key was strong. This
keeps the displayed list consistent with the now-capped score.

### 3.3 Harness-side input bound (related, in `stage5p5_harness.py`)

Independently, the harness feeds a **300-bar (~14 month) trailing window**
(`TRAILING_BARS`) rather than the full 5 years. This keeps SMA200 (200),
Bollinger-squeeze percentile (120), and the aVWAP/divergence lookbacks warm while
restricting divergence detection to recent context. With the capped score this is no
longer load-bearing, but it remains a sensible default.

---

## 4. Result

Same three tickers, before → after:

| Ticker | Before (per-event) | After (capped) |
|---|---|---|
| AAPL | +10.0 STRONG_BUY | +2.2 HOLD |
| MSFT | +10.0 STRONG_BUY | −2.8 SELL |
| NVDA | +10.0 STRONG_BUY | −7.0 STRONG_SELL |

Example arithmetic (NVDA): confirmed `reg_bear` on 2 indicators (MACDHist, CMF) → −4.0;
`q_reg_bear` strong → −1.0; `hid_bull_any` → +1.0; net divergence **−4.0**, combined
with trend −3.0 → **−7.0 STRONG_SELL**. Scores now use the full −10…+10 range and
reflect direction instead of saturating bullish.

---

## 5. Scope / caveats

- This change makes `stage5p5.py` **intentionally diverge from the vendored "as-is"
  file** for the divergence component only.
- Other documented deviations from the master spec are **unchanged**: MACD trend
  weights, the redesigned mean-reversion RSI bands, the MACD-histogram-slope
  convergence heuristic (in place of true pivot convergence), the `trend_avwap`
  TESTING_* thresholds, and the absent §5.5 vol-context caveats / §5.6 warmup gate.
- No engine files (`screener.py`, `fetcher.py`, `memo.py`, `tracking.py`) were
  modified.

---

# Follow-up — Hidden-Divergence Recency Gate

**File touched:** `stage5p5.py` (`_detect_divergence_active`, `_run_divergence_suite`)

## Issue

Regular divergences are recency-gated: a confirmed regular divergence only counts
if it confirmed within `effect_bars = 15` bars of *now* (`_detect_divergence_active`
`bars_since_confirm > effect_bars` check). **Hidden divergences had no such gate** —
the `is_hidden` branch appended a `"confirmed"` event for every qualifying pivot pair
regardless of age, so a hidden divergence from months ago kept contributing to the
score. The only effective bound was the harness input window (`TRAILING_BARS = 300`).
Net effect: "a signal that fired last year still counts."

## Fix

- In the hidden branch, drop events whose pivot `p2` is older than `effect_bars`
  (`(n - 1) - p2 > effect_bars`) — the same ~15-bar (~3-week) window regular
  divergences already use.
- Made `effect_bars` a parameter of `_run_divergence_suite` (default `15`, so
  scoring/commentary behaviour is unchanged) so the chart can pass a large value to
  retain the full divergence history for context.

## Result

Hidden divergences now expire like regular ones. Example: NVDA moved from −7.0 to
−8.0 once a stale hidden-bull (which had been adding +1) aged out; AAPL +2.2 → +1.2.
`stage5p5_plot.py` requests `effect_bars=10_000` to keep all historical fire markers,
and shades the trailing 15-bar "live window" so it's clear which fires still score.

## Visualization companion (`stage5p5_plot.py`)

Added a read-only chart tool so signals can be eyeballed for *why* they fired. It
reuses the harness window (`TRAILING_BARS`) and the same `stage5p5` series /
divergence functions, then renders a 7-panel chart (Price + SMAs / Bollinger /
aVWAP, Volume, RSI, MACD, CMF, OBV, ADX) with:

- divergence fire markers (`^` bull / `v` bear; hollow = hidden, filled = regular),
- the most-recent confirmed bull/bear divergence pivot **leg** on both the price and
  indicator panels (so the opposing slopes are visible side by side),
- MACD crosses, Bollinger breaks, aVWAP anchors,
- the shaded **15-bar live window** marking which fires still count.

```
python stage5p5_plot.py AAPL MSFT NVDA   # -> charts/<SYM>_stage5p5_<date>.png
```

`matplotlib` was added to `requirements.txt` (used only by this tool, not the
screener). See `stage5p5_technicals_guide.md` for how to read the output.
