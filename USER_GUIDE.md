# User Guide — Gamma Workflow Signal System

**Audience**: New analysts on the desk. Read this once front-to-back before
touching anything. Bookmark it, come back to specific sections as needed.

This guide covers what the system does, why it works, how to read what it
produces, and where to be careful. Specialist terminology is unavoidable in
parts — when it appears, it's explained on the spot.

---

## Table of contents

1. [What this system does, in one paragraph](#1-what-this-system-does-in-one-paragraph)
2. [The market theory you need to understand first](#2-the-market-theory-you-need-to-understand-first)
3. [The three playbooks](#3-the-three-playbooks)
4. [How the pipeline is wired together](#4-how-the-pipeline-is-wired-together)
5. [The configuration knobs you can turn](#5-the-configuration-knobs-you-can-turn)
6. [The output files and how to read them](#6-the-output-files-and-how-to-read-them)
7. [What to do each morning](#7-what-to-do-each-morning)
8. [Caveats and things to watch out for](#8-caveats-and-things-to-watch-out-for)

---

## 1. What this system does, in one paragraph

This is a tool that watches the options market every night after the close, looks
for specific patterns that historically precede good long entries, and writes a
trade ticket if it sees one. It does not place trades — a human (you, eventually)
reads each ticket the next morning and decides whether to act on it. The whole
point of the system is to scan a watchlist faster and more consistently than a
human could, while leaving final judgment to a person.

---

## 2. The market theory you need to understand first

You can skip ahead and use the system without this section. But you'll make
better decisions about which tickets to act on if you understand *why* the
patterns it looks for are interesting in the first place.

### 2.1 Who's on the other side of every option you buy

When you buy a call or a put, somebody is selling it to you. Most of the time
that "somebody" is a market-making firm — Citadel Securities, Susquehanna, a few
others. These firms don't want to take a directional bet on the stock; they want
to collect the spread between bid and ask. So they immediately hedge their
position in the underlying stock to neutralize the direction.

If a market maker sells you a call (a bet that the stock will go up), they're
now short that call. To stay neutral, they buy some shares of the stock. How
many shares? Enough to offset the call's delta — the option's sensitivity to
moves in the stock. This hedging is mechanical and constant.

### 2.2 Why this matters: gamma

Here's the wrinkle. The option's delta isn't constant — it changes as the stock
moves. The *rate of change* of delta is called gamma. If the market maker is
short gamma, their hedge gets worse as the stock moves; they have to keep
buying as it goes up and selling as it goes down, which **amplifies** the
move. If they're long gamma, their hedge gets better as it moves; they buy on
dips and sell on rallies, which **dampens** the move.

When market makers, in aggregate, are heavily long gamma, the market trades
in a tight range — every rally gets sold into, every dip gets bought. When they
are heavily short gamma, moves accelerate — every selloff feeds more selling,
every rally feeds more buying.

This is not theoretical. It's mechanical, it's measurable, and it explains a
huge fraction of intraday and short-horizon price behavior in major US stocks
and index ETFs.

### 2.3 The "gamma flip" — where the regime changes

For any stock, there's a price level called the **gamma flip** (or "zero gamma
level"). Above this level, dealer positioning is net long gamma — moves are
dampened. Below it, dealer positioning is net short gamma — moves are amplified.

This level isn't static. It moves as options expire and new positions get put
on. Tracking where it is and which side of it the stock is on is the single
most important signal this system produces.

### 2.4 Walls

There are also specific price levels where so many options exist (so much "open
interest") that dealer hedging creates strong gravitational effects:

- **Call wall** — a price above the current stock price where many calls are
  sold. As the stock approaches this level, dealers' hedging mechanically pushes
  back against further rallies. Stocks tend to pause or reverse near call walls.
- **Put wall** — a price below the current price where many puts are sold.
  Acts as support; the stock tends to bounce off these levels.

Walls aren't magic. They can break, especially when the underlying news flow is
strong enough to overwhelm the mechanical hedging. But they're statistically
meaningful and a good place to look for entries.

### 2.5 Skew, IV, and term structure — what the options chain tells you about fear

A few more concepts you'll see in tickets:

- **Implied volatility (IV)** is the market's forward-looking estimate of how
  much a stock will move. Higher IV = options are more expensive = the market
  expects bigger moves.
- **Put skew** is the difference in IV between put options and call options of
  similar moneyness. When skew is elevated, puts are unusually expensive relative
  to calls — meaning the market is paying a premium for downside protection.
  High skew = fear is priced in. Counterintuitively, this is often the moment
  *just before* a bottom, not the start of further selling, because everyone
  who wanted protection has already bought it.
- **Term structure** describes how IV varies across expiration dates. When IV
  on short-dated options is much *lower* than long-dated (called "contango"),
  the market expects calm now and bigger moves later. When it's higher on
  short-dated ("backwardation"), the market expects immediate turmoil. Contango
  is the normal state.

### 2.6 The whole picture

Combining all of the above: this system identifies moments when dealer
positioning (gamma sign), price action relative to key levels (walls), and
fear pricing in the options market (skew, IV) line up in ways that have
historically preceded profitable long entries. It's not a magic formula —
it's a way of formalizing the kinds of patterns experienced traders look for
intuitively, so we can scan dozens of names every night without missing one.

---

## 3. The three playbooks

A "playbook" is a named pattern with specific entry conditions. The system has
three. Each is designed for a different market regime.

### 3.1 Playbook A — Washout at Put Wall

**Plain English**: The stock has been selling off hard. Fear is showing up
everywhere in the options chain — heavy put buying, expensive put protection,
elevated VIX. Today the stock dropped to the put wall on capitulation volume.
We buy a starter position on the theory that the people who needed to sell have
mostly sold, and we'll bounce off the put wall.

**Market theory**: This is a classic "sell the rumor, buy the news" setup, but
mechanized. In short-gamma regimes (the "deep negative" condition), every
selloff feeds more selling. But this doesn't go on forever — eventually you
exhaust the panic sellers, the put wall acts as mechanical support from dealer
hedging, and the stock reverses. The challenge is timing it. The gates we
require — VIX surge, put-call ratio spike, fear-pricing in skew, capitulation
volume — are all things that tend to be elevated *at* the bottom, not before it.

**Example scenario**: SPY drops 4% over three days. VIX moves from 16 to 22.
Put volume is 2× the typical day. The 25-delta put is trading at IV 28% versus
calls at 18%. SPY closes at 495 — exactly on the put wall. The system fires
Playbook A: enter long at 495, stop at the put wall (so you exit if it breaks),
target the call wall up at 520.

**Conviction**: Moderate. We size at 50% of full position because washouts can
have second legs — sometimes the put wall breaks before it holds.

**Critical filter**: We do NOT fire this within 5 days of a known macro event
(FOMC, CPI, NFP, earnings) because we don't want to be long into a surprise
announcement that could blow through the put wall.

### 3.2 Playbook B — Gamma Flip Reclaim

**Plain English**: The stock has been in a downtrend, in the "scary" short-gamma
regime where every dip extends further. Today it crossed back above the gamma
flip level on heavy volume. The dealer positioning has flipped from amplifying
moves to dampening them. The regime has changed. We buy with higher conviction
than Playbook A because the change has already been confirmed.

**Market theory**: This is the moment a downtrend mechanically *can't* continue
in the same way it has been. Above the flip, every rally is no longer met with
new selling pressure from dealer hedges. The path of least resistance shifts
from down to up. This pattern has historically been one of the highest-hit-rate
setups in equity markets — when it works, it works clean.

**Example scenario**: NVDA traded in negative gamma for two weeks, drifting
from 480 down to 440. The gamma flip has been sitting at 460. Today it rallies
to 465 on volume 1.8× the 20-day average. The call wall above ticked higher
than yesterday's level. The 5-day average of total gamma exposure has started
turning up. Playbook B fires: buy at 465, stop at the flip level (460), target
the call wall.

**Conviction**: High. We size at 75% of full position because the regime change
is confirmed, not anticipated.

### 3.3 Playbook C — Compression Breakout

**Plain English**: The stock has been pinned in a tight range for weeks. The
options market is screaming "calm" — short-dated options are cheap, long-dated
options are expensive (steep contango), and the same call wall has been holding
above price for many consecutive sessions. There's a known catalyst coming
soon (earnings, Fed meeting). We buy a defined-risk options structure
(typically a call spread, not shares) to position for a breakout, with the
options structure capping our risk.

**Market theory**: Long stretches of compressed volatility almost always
resolve in expansion — the question is when, not if. The options market often
underprices the eventual move. A defined-risk options structure lets us
participate in the upside while keeping the downside bounded if the breakout
doesn't materialize when expected.

**Example scenario**: SPY has been in a 502–508 range for two weeks. Total
gamma exposure has been in the top decile of its 60-day distribution. The
525 call wall hasn't moved. The 3-month implied vol is 4 points above the
7-day implied vol. FOMC is in 7 days. Playbook C fires: enter a defined-risk
bullish options structure with a stop equivalent to the gamma flip.

**Conviction**: Moderate (50% sizing). The structure is the conviction —
defined-risk means we know the worst case going in.

### 3.4 Summary table

| Playbook | When to fire | Instrument | Tranche | Stop |
|---|---|---|---|---|
| A | Capitulation at put wall, short gamma | Shares | 50% | Put wall |
| B | Gamma flip reclaim with confirmation | Shares | 75% | Gamma flip |
| C | Volatility compression with upcoming catalyst | Options spread | 50% | Gamma flip |

---

## 4. How the pipeline is wired together

The system runs once every weekday after the US market close. Here's what
happens, in order:

```
1. Calendar       → flags upcoming events (OPEX, FOMC, earnings)
2. Data fetch     → options chains, daily bars, risk-free rate
3. Regime         → classifies the stock into one of 4 gamma regimes
4. Level map      → finds the walls, flip, and high-volume strike
5. Flow           → measures skew, put-call ratio, OI changes
6. Confirmation   → checks volume, ATR, VIX, distance from MAs
7. Logging        → writes one feature row per ticker
8. Playbooks      → runs A, B, C against each ticker's feature row
9. Tickets        → writes a markdown ticket for each fired signal
10. Monitor       → checks open positions, writes alerts if exits hit
11. Backfill      → calculates forward returns on older days (for tracking)
```

Each step feeds the next. The output of steps 1–6 becomes the "feature row" —
a snapshot of everything we measure about a ticker on a given day. Steps 8
and 10 consume that row to make decisions.

### 4.1 The package structure

The code lives in a folder called `gamma_workflow/`. The files are named so
you can find what you need:

| File | What it does |
|---|---|
| `engine_shim.py` | Wraps the calculation engine (the file that does the dealer-positioning math). Don't modify. |
| `config.py` | Reads the YAML configuration files. |
| `rates.py` | Pulls the current risk-free rate from the Federal Reserve's free data source. |
| `storage.py` | Reads and writes all the data files. |
| `stage1_calendar.py` | Computes OPEX/VIX expiration dates, reads earnings from Yahoo Finance, reads macro events from a CSV. |
| `stage2_regime.py` | Classifies each ticker as deep_positive / transitional_positive / transitional_negative / deep_negative. |
| `stage3_level_map.py` | Finds the call wall, put wall, secondary walls, gamma flip, and high-volume strike. |
| `stage4_flow.py` | Computes put/call ratios, skew, term structure, OI changes, z-scores of all these. |
| `stage5_confirm.py` | Computes volume z-score, ATR, distance from moving averages, VIX. |
| `stage6_playbooks.py` | The three playbook rules. |
| `stage7_ticket.py` | Sizing math + markdown ticket builder. |
| `stage8_monitor.py` | Watches open positions for exit triggers. |
| `stage9_logging.py` | Writes feature rows; backfills forward returns. |
| `run_eod.py` | The orchestrator — ties stages 1–9 together. |

### 4.2 How the outputs of each stage feed into the next

The key thing to understand: every stage produces **a small piece of the feature
row** for each ticker. Stage 9 then concatenates everything into one row,
writes it to disk, and stages 6 and 8 read from that row to decide what to do.

For example: Stage 4 computes `pcr_volume_z` (the put-call ratio's z-score
against its own 20-day history). Playbook A in Stage 6 then asks: "is
`pcr_volume_z > 1.5`?" If yes, that's one of the gates passing toward firing
the signal.

This separation matters because it means every input to every decision is
**logged**. If a trade doesn't work out, you can go back to the feature row
for that day and see exactly what the system was looking at. There's no black
box.

---

## 5. The configuration knobs you can turn

Everything tunable lives in two YAML files in the `config/` folder. You don't
need a programmer to change these — open them in any text editor, save, commit.
The next scheduled run uses the new values.

### 5.1 `watchlist.yaml` — what stocks to scan

```yaml
defaults:
  risk_per_trade_bps: 50

tickers:
  - symbol: SPY
    kind: index_etf
    market_cap_b: 500
  - symbol: NVDA
    kind: single_name
    market_cap_b: 3000
```

- **`symbol`**: the ticker.
- **`kind`**: either `index_etf` or `single_name`. The system enforces a VIX
  confirmation check on index ETFs (a real selloff in SPY should be matched
  by a VIX spike). Single names skip this check.
- **`market_cap_b`**: market cap in billions of dollars. Used to set a
  fallback threshold for what counts as "deep" gamma positioning when the
  system doesn't yet have 60 days of history for that name.
- **`risk_per_trade_bps`**: optional. Override the default risk budget for
  this specific ticker.

To add a new ticker, just append a new entry. To remove one, delete its entry.

### 5.2 `thresholds.yaml` — all the numbers that go into the rules

This is the file you'll tune most often. Every number that the system uses
to make a decision is here. The PM does a quarterly review of this file
based on what's worked and what hasn't.

The most important sections:

**`regime`** — how aggressive to be in calling a stock "deeply" positive or
negative gamma:

```yaml
regime:
  history_days_required: 60      # use this ticker's own history if available
  percentile_deep_positive: 80   # top 20% of its GEX distribution = deep positive
  percentile_deep_negative: 20   # bottom 20% = deep negative
  fallback_pct_of_market_cap: 0.01  # if no history, use 1% of market cap
```

Tightening the percentile gates (e.g., 90/10 instead of 80/20) makes the
system call fewer days "deep" — Playbook C will fire less often.

**`flow`** and **`confirm`** — z-score thresholds:

```yaml
flow:
  put_call_ratio_zscore_panic: 1.5
  put_skew_zscore_high: 1.0
  unusual_activity_zscore: 2.0

confirm:
  volume_zscore_capitulation: 1.5
  vix_5day_change_pct: 0.15
```

These determine how extreme a reading has to be before it counts as a "panic"
or "capitulation." Lowering them = more signals, more false positives.
Raising them = fewer signals, missed opportunities.

**`playbooks`** — per-playbook settings:

```yaml
playbooks:
  A_washout:
    proximity_to_put_wall_pct: 0.005   # within 0.5% of put wall counts as "at"
    initial_size_pct: 0.50             # take half a position
  B_gamma_reclaim:
    volume_zscore_reclaim: 1.0
    initial_size_pct: 0.75
  C_compression:
    min_compression_days: 5            # require N consecutive compressed days
    steep_contango_min_slope: 2.0      # vol-points difference
    catalyst_window_days: 10
    initial_size_pct: 0.50
```

**`sizing`** — how big the positions get:

```yaml
sizing:
  default_risk_per_trade_bps: 50
  nav_usd: 100000000                   # ⚠️ MUST be set to your real book size
```

The `nav_usd` value is critical. It's the fund's total Net Asset Value in
dollars. Every share count on every ticket scales linearly with this number.
If you ever see ticket size that looks wrong by 10× or 100×, the first place
to check is whether `nav_usd` matches the real book.

**`monitor`** — when to scale out or exit:

```yaml
monitor:
  scale_out_volume_zscore: 2.0
  scale_out_fraction: 0.50
```

**`acquisition`** — pipeline plumbing:

```yaml
acquisition:
  inter_ticker_sleep_sec: 2.0
```

If the data source starts rate-limiting (you'll see "Too Many Requests" in the
logs), raise this number to 4 or 5.

### 5.3 `macro_events.csv` — known event dates

A simple CSV with columns `date,event`. Format: `YYYY-MM-DD,EVENT_LABEL`.
Either maintained manually or refreshed by the macro scraper workflow each
Monday morning. This file gates Playbook A: signals are blocked within 5
days of any listed event.

If this file is empty or out of date, Playbook A can fire into a Fed meeting,
which is exactly the kind of trade we don't want to take. Verify it stays
current.

### 5.4 `positions.json` — your open positions

Format:

```json
[
  {
    "ticker": "SPY",
    "playbook_id": "A",
    "entry_price": 500.0,
    "entry_date": "2026-05-28",
    "stop": 495.0,
    "target_1": 520.0,
    "target_2": 530.0,
    "shares": 50000,
    "size_pct": 0.50,
    "status": "open"
  }
]
```

You maintain this file by hand. When you enter a trade, add an entry. When
you close it, set `status` to `closed` (or remove the entry). Stage 8 reads
this file every night and writes alerts when exit conditions are hit.

---

## 6. The output files and how to read them

Everything the system produces lives under `data/` in the repo. Files are
organized by date for easy lookup.

### 6.1 `data/features/date=YYYY-MM-DD.parquet`

One file per trading day. Each file contains one row per ticker. This is the
machine-readable record — the snapshot of what the system saw on that day.

You read it with pandas:

```python
import pandas as pd
df = pd.read_parquet("data/features/date=2026-05-28.parquet")
print(df.columns.tolist())
print(df[df["ticker"] == "SPY"].T)
```

Important columns (there are about 50 in total):

| Column | What it means |
|---|---|
| `ticker` | The stock symbol. |
| `asof` | The date this row is for. |
| `spot` | The closing price. |
| `regime` | One of the four regime labels. |
| `prev_regime` | Yesterday's regime. |
| `regime_transition` | A string like `deep_negative->transitional_positive` when regime changed. |
| `gex_total` | Total dealer gamma exposure at the current price. |
| `flip_level` | The gamma flip price. |
| `distance_to_flip_pct` | How far the stock is from the flip, as a percent. |
| `call_wall`, `put_wall` | The primary walls. |
| `call_wall_secondary`, `put_wall_secondary` | Secondary walls (next-strongest, at least 2% away from the primary). |
| `call_wall_dod_change`, `put_wall_dod_change` | How much each wall moved since yesterday. |
| `wall_persistence_days_call/put` | How many days in a row the wall has been at the same level. |
| `hvl` | High-volume level — strike with the most total option volume today. |
| `pcr_volume`, `pcr_oi` | Today's put-call ratios by volume and open interest. |
| `pcr_volume_z`, `pcr_oi_z` | Those ratios expressed as z-scores against the past 20 days. |
| `total_volume`, `total_volume_z` | Total option volume and its z-score. |
| `skew_25d`, `skew_25d_z` | The 25-delta put skew and its z-score. |
| `term_structure_slope` | 3-month IV minus 7-day IV. |
| `new_oi_strike_count` | Number of strikes where volume exceeded open interest today (new positioning). |
| `oi_delta_call`, `oi_delta_put` | Net change in calls and puts open interest since yesterday. |
| `volume_z` | Underlying stock's volume z-score (20-day). |
| `atr_pct`, `atr_20d_pct` | Today's true range and 20-day average, as a percent of price. |
| `range_expansion` | Today's range divided by the 20-day average. |
| `close_location` | Where the close sat in today's range (0 = at low, 1 = at high). |
| `ma20_dist_pct`, `ma50_dist_pct`, `ma200_dist_pct` | Distance from 20/50/200-day moving averages. |
| `vix_close`, `vix_5d_change_pct` | The VIX close and its 5-session change. |
| `days_to_earnings`, `days_to_macro` | Days until the next earnings release / next macro event. |
| `within_event_window` | True if any binary event is within 5 days. |
| `days_to_opex`, `days_to_vixp` | Days until next monthly options expiration / VIX expiration. |
| `fwd_ret_1d`, `fwd_ret_5d`, `fwd_ret_20d` | Backfilled later: the stock's actual return 1/5/20 days after this date. Used to evaluate which features predict good entries. |

Most of these columns will be `NaN` (empty) for the first 20-30 trading days
after the system goes live, because they require historical data to compute.
This is expected behavior.

### 6.2 `data/signals/date=YYYY-MM-DD.parquet`

One file per day, with one row per fired signal. Many days will have no
signals at all — this file may be missing on those days.

Columns:

| Column | What it means |
|---|---|
| `ticker` | The stock. |
| `asof` | The date. |
| `playbook_id` | `A`, `B`, or `C`. |
| `signal_strength` | A score from 0 to 1; closer to 1 means the gates passed more decisively. |
| `entry` | Suggested entry price (today's close). |
| `stop` | Suggested stop price. |
| `target_1`, `target_2` | Profit targets. |
| `shares` | Suggested position size in shares. |

Use this file to build a daily list of what fired across all tickers in one
query. The tickets (next section) have all the detail; this is the index.

### 6.3 `data/tickets/YYYY-MM-DD_TICKER_PLAYBOOK.md`

**This is the file you read each morning.** One markdown file per fired
signal. It renders directly in the GitHub web UI — just click the file.

Structure of a ticket:

1. **Header** — ticker, date, playbook name, signal strength, and a
   prominent reminder that this is for human review only.
2. **Sizing table** — NAV, risk budget, entry, stop, risk per share,
   suggested shares, and the two profit targets. Use these numbers
   directly when entering the trade (assuming you decide to take it).
3. **Key levels** — spot, put wall, call wall, gamma flip. These are
   the reference levels you'll watch while the trade is on.
4. **Why it fired** — the regime label plus each gate that passed,
   with the values. Read this carefully. If something looks marginal
   (e.g., a z-score barely above its threshold), the signal is less
   strong than the headline suggests.
5. **Management notes** — brief reminders on how to manage the trade
   once on.

How to read it in practice:

- Open the GitHub repo each morning.
- Look at `data/tickets/` — any new files starting with today's date?
- Open each one. Read it top to bottom.
- For each ticket, ask: "do I agree with this?" Common reasons to
  *not* take a fired signal:
  - Earnings is actually scheduled in 3 days and Yahoo got the date wrong.
  - The stock is at a critical technical level the system isn't aware of
    (e.g., a major announcement is rumored, or a 52-week high is just above).
  - You already have a position in a correlated name.
- For tickets you decide to take: enter the trade. Then update
  `positions.json` to reflect the new position so Stage 8 can monitor it.

### 6.4 `data/tickets/YYYY-MM-DD_TICKER_ALERT.md`

When Stage 8 sees that an open position has hit a trigger (regime change,
call wall hit on volume, put wall break), it writes an alert in the same
folder with `_ALERT` in the filename.

These are recommendations to act on existing positions: trail your stop up,
scale out half, exit fully. Read them with the same scrutiny as entry tickets
— the system is suggesting, you're deciding.

### 6.5 `data/chains/` and `data/rates/`

Internal data the pipeline uses to compute features. You shouldn't need to
read these directly. They're committed to the repo so day-over-day
calculations (like OI delta) work correctly.

---

## 7. What to do each morning

Once the system has been running for a few weeks and you're trusted to act
on tickets, your morning routine will look something like this:

1. **Open the repo on GitHub.** Confirm the EOD pipeline ran cleanly
   yesterday (Actions tab, look for green check on the most recent run).
   If it failed, fix it before doing anything else — a failed run means
   you have no data for that day.
2. **Look at `data/tickets/`.** Sort by date or look for today's date prefix.
3. **For each new ticket**: read it. Decide whether to act.
4. **For each ticket you act on**: enter the trade per the suggested entry,
   stop, and size. Then update `positions.json` with the new position.
5. **For each alert on an existing position**: act on the recommendation
   (trail stop, scale out, exit). Update `positions.json` to reflect the
   new state.
6. **Periodically — say, monthly** — open one of the feature parquet files
   and look at the data. Compare what fired signals on days the market
   actually rallied versus days it kept falling. This is how you'll
   eventually contribute to tuning the thresholds.

A few do's and don'ts:

- **Do** read every ticket end-to-end. Don't act on just the headline.
- **Do** check the macro events CSV is up to date. If the macro scraper
  fails for a few weeks and no one notices, Playbook A starts firing into
  Fed meetings.
- **Do** verify the `nav_usd` setting matches reality before sizing any
  trade. The math is unforgiving.
- **Don't** treat a fired signal as an order. The whole system is built
  around the principle that the machine proposes and the human disposes.
- **Don't** chase signals from days you missed. The entry price is stale,
  the setup may have invalidated.

---

## 8. Caveats and things to watch out for

### 8.1 The first month will be sparse

Almost every signal-firing gate depends on a z-score, which needs 20 days of
history. The regime classifier uses percentiles which need 60 days. Until the
system has accumulated this much history, most playbooks won't fire even
on textbook setups. This is correct behavior — we'd rather miss real signals
than fire on insufficient data. Don't tune anything until you have at least
3 months of data.

### 8.2 Yahoo Finance is the data source

The free options data we use comes from Yahoo Finance via the `yfinance`
library. Yahoo's data is good but not professional-grade. Specific known
limitations:

- Open interest is reported with a one-day lag.
- Yahoo doesn't expose option sweep flags or bid/ask side classification,
  so the system can't see "unusually large institutional sweeps."
- Yahoo occasionally has bad ticks. If you see a feature value that looks
  insane (e.g., a put wall 30% below spot when the chart shows nothing of
  the sort), suspect bad data.
- Yahoo's earnings calendar is sometimes wrong by a day or two. The macro
  CSV override exists exactly for this reason.

### 8.3 Macro events scraper can break silently

The Monday macro events scraper pulls from Trading Economics' public website
via HTML scraping. Their site layout changes occasionally. When it does, the
scraper either fails loudly (workflow goes red, you get an email) or — worst
case — succeeds but produces incomplete data. **Spot-check the macro CSV
once a quarter** even when everything looks green.

### 8.4 The system has no view on news

It only sees the options market and price/volume data. It does not know that
the CEO just resigned, that a deal was announced, or that an FDA decision is
imminent. If a stock is in deep negative gamma but the underlying news flow
is structurally bad (think a fraud allegation), the put wall will break
through and Playbook A's stop will trigger immediately.

You are the news filter. Skip any signal where you know something the system
can't.

### 8.5 The market regime affects which playbooks work

In sustained bull markets, Playbook B (gamma flip reclaim) tends to fire most
often and works well. In choppy ranging markets, Playbook A (washouts) works.
In persistent low-vol environments, Playbook C (compression) is the play.
There will be stretches where one playbook dominates and others go quiet —
this is normal.

### 8.6 Stop discipline is non-negotiable

The sizing math assumes you'll actually stop out at the suggested stop level.
If you let trades run past their stop, the risk budget calculation becomes
meaningless and your actual losses can be far larger than the ticket said.
Use the suggested stop. Always.

### 8.7 The system is for entries, not exits

Stage 8 monitors open positions and suggests management actions, but it's
much simpler than the entry logic. It's a useful guide, not a complete
exit strategy. As you gain experience, you'll likely overlay your own exit
discipline on top of (or instead of) Stage 8's suggestions.

### 8.8 No two days look exactly the same

Even a textbook fired signal will look slightly different in its specifics
from any prior signal. Don't develop a mental template of "what a Playbook A
ticket looks like" — every ticket has its own context. Read each one fresh.

---

## Where to ask for help

If a ticket fires and you don't understand why, you can always look at the
feature row for that ticker on that date — every gate it passed is logged
there with the exact value. If something seems off, raise it. We tune this
system based on real observations, not assumptions, and "this ticket looked
weird" from a careful analyst is exactly the kind of feedback that
improves it.

Welcome to the desk.
