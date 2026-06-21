# 📈 Investo — NSE Swing Trading System

A professional-grade Streamlit application for NSE swing trading with intelligent entry signals, verified sector data, and systematic risk management.

---

## 📋 Table of Contents

- [Overview](#overview)
- [App Structure](#app-structure)
- [What's New (20-Jun-2026)](#whats-new-20-jun-2026)
- [Monthly Swing Scanner](#monthly-swing-scanner)
- [SMA Weekly Scanner](#sma-weekly-scanner)
- [Confident Score System](#confident-score-system)
- [Sector Ranking System](#sector-ranking-system)
- [RS vs Own Sector](#rs-vs-own-sector)
- [Higher Timeframe Alignment](#higher-timeframe-alignment)
- [Volatility Squeeze Detection](#volatility-squeeze-detection)
- [ADX Trend Strength](#adx-trend-strength)
- [Nifty Market States](#nifty-market-states)
- [Drawdown-Based Position Sizing](#drawdown-based-position-sizing)
- [Price Action Analysis](#price-action-analysis)
- [F&O Expiry Intelligence](#fo-expiry-intelligence)
- [CSV Upload Features](#csv-upload-features)
- [Diagnostic & Debug Tools](#diagnostic--debug-tools)
- [Portfolio Management](#portfolio-management)
- [Installation](#installation)
- [Configuration](#configuration)
- [Key Concepts](#key-concepts)
- [Changelog](#changelog)

---

## Overview

Investo is built for active NSE swing traders who want a systematic, data-driven approach to finding high-probability trade setups. It combines 25+ trading concepts into one decisive Confident Score per stock, with every sector ETF mapping empirically verified against live Yahoo Finance data rather than assumed.

```
Scanner finds the stock
Price Action + Squeeze + ADX confirm the entry
RS vs Sector + Higher Timeframe confirm the conviction
Confident Score tells you to enter or skip
PSAR tells you when to exit
```

### Tech Stack

```
Frontend:    Streamlit
Data:        yfinance (OHLCV + sector indices), Zerodha Kite API (live prices)
AI:          Anthropic Claude API (stock validation)
Language:    Python 3.10+
Lines:       17,500+
```

---

## App Structure

```
🌅 Dashboard        Pre-market intelligence, Nifty state, VIX
📊 Scanner          Full intraday scan, shortlist, deep analysis
🚀 Early Movers     Gap-up stocks 9:15–9:25 AM
🔓 ORB Scanner      Opening Range Breakout detection
💼 Portfolio        Open positions, P&L, PSAR signals, F&O warnings
🔔 Alert Log        Trade alerts and notifications
📈 SMA Weekly       Daily chart scanner — 3–7 day hold
📅 Monthly Swing    Weekly chart scanner — 3–5 week hold
🧪 Backtest         Strategy validation + isolated diagnostic tools
🏆 Sector Leaders   Folder-based NSE sector CSV ranking (standalone)
```

---

## What's New (20-Jun-2026)

A full day dedicated to auditing and fixing the sector/relative-strength layer that both scanners depend on — verifying every assumption against live data rather than trusting it.

```
1. Sector Foundation Audit
   - 2 rounds of live yfinance ticker verification (~30 candidates)
   - Found FINANCE's ticker (^CNXFIN) was completely DEAD on Yahoo
     — sector ranking had been silently falling back to neutral
     defaults for an unknown period. Replaced with verified
     NIFTY_FIN_SERVICE.NS
   - Upgraded 3 sectors from stock-basket PROXY to real ETF data:
     PSU_BANK, PVT_BANK, CONSUMPTION
   - Empirically REJECTED 5 tempting "catch-all" index mappings
     (e.g. TELECOM→Services Sector index, DEFENCE/LOGISTICS→Infra)
     using correlation testing against real sample stocks
   - Fixed 2 stock misclassification bugs: ONESOURCE (pharma CDMO,
     was wrongly tagged 'IT'), SWIGGY (food delivery, same issue)
   - Fixed 1 dead ticker bug: MAHINDLOG → MAHLOG (Mahindra
     Logistics' actual NSE symbol)

2. RS vs Own Sector — fixed and wired into both scanners
   - Found and fixed a unit-mismatch bug (was comparing a stock's
     raw return against an already-Nifty-relative sector number)
   - Rewrote with correct apples-to-apples math
   - Validated via an isolated diagnostic tool before touching
     any live scanner

3. Higher Timeframe Alignment — fixed and wired into both scanners
   - Found 3 bugs: a resample() call that always threw an
     exception (silent no-op), insufficient data depth for the
     monthly check (~4-5 bars), and a too-long lookback period
     that gave zero discrimination across test stocks (4 of 5
     scored identically, including a known loss trade)
   - Rewrote to fetch the higher timeframe directly via yfinance
     with timeframe-appropriate SMA periods

4. New diagnostic infrastructure
   - Isolated test tools in the Backtest tab for both fixes above
   - Per-scanner debug error expanders (visible even when results
     aren't empty — non-fatal errors no longer hide silently)
   - Monthly Swing "Scan Funnel" — shows exact stock dropout count
     at every gate, to tell apart "genuine bearish market, nothing
     qualifies" from "something's actually broken"

5. CSV-driven configuration (self-service, no code edits needed)
   - Stock universe CSV uploader for both scanners
   - Dynamic F&O stocks list (CSV upload, saved to disk, survives
     restarts) — NSE revises this list periodically; previously
     required a code change each time
```

See [Changelog](#changelog) for the full dated history.

---

## Monthly Swing Scanner

Scans NSE stocks using **weekly candles** for medium-term swing trades held 3–5 weeks.

### Data Source
```
Interval:  Weekly (1wk)
Period:    2 years (signal detection) + direct fetches for
           sector RS and higher-timeframe checks
Entry:     Live daily price (separate fetch)
PSAR:      Weekly candles, step=0.01, max=0.10
```

### Hard Gates (any fail = stock rejected)
| Gate | Condition |
|------|-----------|
| Trend | SMA20 > SMA50 |
| SMA20 slope | > 0 (must be rising) |
| SMA50 slope | > −0.5% |
| Price | > Weekly SMA20 |
| Candles | Not 3 consecutive red weeks |
| Recovery | Close > previous OR close > open |
| Proximity | % above SMA20 ≤ 5% (normal) / ≤ 3% (high ATR) |
| Fibonacci | Retrace 15–78.6% of swing |
| Weekly move | ≤ 8% |
| RSI | ≤ 70 |
| RS vs Nifty | ≥ 0.95 |
| Candle close position | ≥ 25% of weekly range (sellers-won weeks rejected) |
| Fundamentals | D/E ≤ 2.0, EPS > 0, Promoter > 20% |

> A live **Scan Funnel** (Backtest-style diagnostic, shown automatically when results are empty) reports exactly how many stocks survive each gate — see [Diagnostic & Debug Tools](#diagnostic--debug-tools).

### Signal Types
```
Signal 1 — SMA Cross:        SMA20 crossed SMA50 in last 3 weeks
Signal 2 — Pullback Bounce:  Trend ≥ 4 weeks + touched SMA20 + recovering
Signal 3 — 13-Week Breakout: Price > 13-week high with volume ≥ 1.5×
Signal 4 — 52W High Breakout: Price breaking to a new 52-week high
Signal 5 — Squeeze Fired:    Volatility compression resolving with direction
```

### Scoring Factors (additional, on top of base technical score)
```
Beta (Nifty-state aware)    ±15
Sector rank (bonus-only)    -3 to +10
RS vs own sector            -4 to +15
Higher timeframe alignment  -2 to +8   (6/12-month SMA on monthly chart)
Candle close position       0 to +8
Body momentum               -8 to +8
Volatility squeeze          -10 to +30
ADX trend strength          -5 to +13
Volume dry-up on pullback   0 to +15
OBV (accumulation)          ±10
Inside week                 ±8
Volatility ATR%             −15 to +8
```

### Trade Plan
```
Entry:  Live daily price
SL:     max(entry − 2×ATR, SMA20 × 0.97)
T1:     Entry + 1× weekly ATR
T2:     Entry + 2× weekly ATR
T3:     Entry + 3× weekly ATR
```
Position size is automatically adjusted by current Nifty state and personal drawdown — see [Drawdown-Based Position Sizing](#drawdown-based-position-sizing).

### PSAR Trailing SL (after T1 hit)
```
⏳ Before T1:  Use original SL — PSAR not active
✅ T1 hit + above PSAR: HOLD — trail SL to PSAR
🔴 T1 hit + below PSAR: EXIT — lock profit
Check: Every Friday only
```

---

## SMA Weekly Scanner

Scans NSE stocks using **daily candles** for short-term swing trades held 3–7 days.

### Data Source
```
Interval:  Daily (1d)
Period:    1 year + direct fetches for sector RS and
           higher-timeframe (weekly) checks
PSAR:      Daily candles, step=0.02, max=0.20
```

### Hard Gates
| Gate | Condition |
|------|-----------|
| Trend | SMA20 > SMA50 |
| SMA20 slope | > 0 (hard reject if flat/declining) |
| Price | > Daily SMA20 |
| SMA50 slope | > −0.5% |
| Candle close position | ≥ 25% of weekly range (sellers-won weeks rejected) |

### Signal Types
```
Signal 1 — Fresh Cross:     SMA20 crossed SMA50 in last 5 days
Signal 2 — Pullback Bounce: SMA20 > SMA50 ≥ 5 days + pulled to SMA20 + bouncing
Signal 3 — Squeeze Fired:   Volatility compression resolving with direction
```

### Differences from Monthly Swing
| Factor | Monthly Swing | SMA Weekly |
|--------|--------------|------------|
| Candles | Weekly | Daily |
| PSAR step | 0.01 / 0.10 | 0.02 / 0.20 |
| ATR limit | Weekly 6% max | Daily 4% max |
| HTF check | Weekly → Monthly (6/12-mo SMA) | Daily → Weekly (20/50-wk SMA) |
| Fundamentals | D/E, EPS, Promoter | Not checked |
| Targets | 1× / 2× / 3× ATR | 0.5× / 1× / 1.5× ATR |
| Check frequency | Every Friday | Every morning |

---

## Confident Score System

Every stock gets a **Confident Score** (typically 0–160+) combining 8 weighted components plus all active filter bonuses. Eliminates the need for manual re-analysis after the scan.

### Score Components (C1–C8)
| # | Factor | Max Points |
|---|--------|-----------|
| C1 | Technical scanner score (tiered, calibrated to real raw-score range) | 25 |
| C2 | PSAR status (most important single signal) | 20 |
| C3 | Price structure HH + HL | 15 |
| C4 | Entry badge proximity | 15 |
| C5 | Risk:Reward at T2 | 10 |
| C6 | Liquidity grade | 5 |
| C7 | F&O expiry penalty/bonus | ±15 |
| C8 | Sector rank (bonus-only, verified ETF data) | -3 to +10 |

C1 is fed by the running raw `score`, which itself accumulates every filter below (squeeze, ADX, RS vs sector, HTF alignment, candle quality, etc.) — so all newer filters genuinely move the headline number, not just a decorative card line.

### Score Thresholds
```
≥ 130  🔥 CONFIDENT BUY  — Highest conviction
100–129 ✅ STRONG SETUP   — High quality
75–99   👍 GOOD SETUP     — Default visible range
55–74   ⚠️ WEAK SETUP     — Hidden by default, toggle to view
< 55    ❌ SKIP           — Not shown in results
```
Default visibility threshold auto-adjusts with the current [Nifty Market State](#nifty-market-states) — stricter in bearish conditions.

### Entry Badge System
| Badge | Normal ATR (≤5%) | High ATR (>5%) |
|-------|-----------------|----------------|
| 🟢 ENTER NOW | ≤ 2% above SMA20 | ≤ 1% above SMA20 |
| 🟡 ACCEPTABLE | 2–5% above SMA20 | 1–3% above SMA20 |
| ❌ HIDDEN | > 5% | > 3% |

---

## Sector Ranking System

A single, unified ranking (`get_unified_sector_rankings()`) feeds the SMA Weekly, Monthly Swing, and Sector Leaders tabs identically — no more inconsistent rankings between tabs.

### 21 Sectors — 16 Real ETF, 7 Stock-Basket Proxy
Every ETF ticker below was **empirically tested against live Yahoo Finance data** (not assumed) across two verification rounds, including correlation testing for tempting but ultimately rejected "catch-all" mappings.

| Sector | Type | Ticker / Basis |
|--------|------|-----------------|
| BANK | ETF ✅ | `^NSEBANK` |
| IT | ETF ✅ | `^CNXIT` |
| AUTO | ETF ✅ | `^CNXAUTO` |
| PHARMA | ETF ✅ | `^CNXPHARMA` |
| FMCG | ETF ✅ | `^CNXFMCG` |
| METALS | ETF ✅ | `^CNXMETAL` |
| ENERGY | ETF ✅ | `^CNXENERGY` |
| REALTY | ETF ✅ | `^CNXREALTY` |
| INFRA | ETF ✅ | `^CNXINFRA` |
| MEDIA | ETF ✅ | `^CNXMEDIA` |
| FINANCE | ETF ✅ | `NIFTY_FIN_SERVICE.NS` *(fixed — old `^CNXFIN` was dead)* |
| PSU_BANK | ETF ✅ | `^CNXPSUBANK` *(upgraded from proxy)* |
| PVT_BANK | ETF ✅ | `NIFTY_PVT_BANK.NS` *(upgraded from proxy)* |
| CONSUMPTION | ETF ✅ | `^CNXCONSUM` *(upgraded from proxy)* |
| HEALTHCARE | Proxy | Stock-basket average — no working ticker found |
| CHEMICALS | Proxy | Stock-basket average — no working ticker found |
| DEFENCE | Proxy | Stock-basket average — `^CNXINFRA` tested, correlation too weak (0.63 vs 0.82 control) |
| TELECOM | Proxy | Stock-basket average — Services Sector index rejected (only 3.85% telecom weight) |
| TEXTILES | Proxy | Stock-basket average — no working ticker found |
| AGRI | Proxy | Stock-basket average — no working ticker found |
| LOGISTICS | Proxy | Stock-basket average — `^CNXINFRA` tested, correlation too weak (0.58 vs 0.82 control) |

### Sector Scoring — Bonus-Only Philosophy
```
Rank 1-2:  +10 pts   Rank 3-4: +7 pts   Rank 5-6: +3 pts
Rank 7-9:   0 pts    Rank 10+: -3 pts   Bearish sector: -2 extra
```
Deliberately bonus-weighted rather than punitive — a stock in a weak/mediocre-ranked sector is never blocked outright, since individual leaders exist in every sector. [RS vs Own Sector](#rs-vs-own-sector) is the real differentiator for that case.

---

## RS vs Own Sector

Finds stocks that are genuinely outperforming **their own sector**, regardless of where that sector ranks overall — the question this answers is "is this stock a leader within its peer group?" not "is this stock's sector currently fashionable?"

### The Math
```
stock_RS_vs_Nifty  = weighted multi-period stock return − Nifty return
sector_RS_vs_Nifty  = same calculation at the sector level (already cached)

diff = stock_RS_vs_Nifty − sector_RS_vs_Nifty
```
Both sides expressed in the same "vs Nifty" units — genuinely comparable, unlike an earlier version that mistakenly compared a stock's raw return against an already-relative sector number.

### Scoring
| Outperformance vs Sector | Score | Label |
|---|---|---|
| ≥ +8pp | +15 | 🌟 Strong sector leader |
| ≥ +5pp | +10 | 🏆 Sector leader |
| ≥ +2pp | +6 | ✅ Outperforming sector |
| ≥ −3pp | 0 | ➡️ Inline with sector |
| ≥ −8pp | −2 | ⚠️ Slightly behind sector |
| < −8pp | −4 | ❌ Lagging sector |

Zero extra API cost — reuses the sector ranking data already fetched once per scan.

---

## Higher Timeframe Alignment

Checks whether the *next higher* timeframe confirms the current signal — two timeframes agreeing is a meaningfully stronger signal than one alone.

```
SMA Weekly (daily signal)   → checks the WEEKLY chart
Monthly Swing (weekly signal) → checks the MONTHLY chart
```

### Periods (timeframe-appropriate, not generic)
| Scanner | Higher TF | Periods | Why |
|---|---|---|---|
| SMA Weekly | Weekly | 20/50-week SMA | ~4.6mo / ~11.5mo — validated, good differentiation |
| Monthly Swing | Monthly | 6/12-month SMA | A longer 20/50-month lookback measured decade-scale secular drift (true for almost any surviving NSE stock) rather than recent momentum — caught via isolated testing before going live |

### Scoring
```
Above + rising:   +8   ✅ Uptrend confirmed
Above, flat:      +4   ✅ Bullish
Near support:      0   ⚠️ Caution
Below support:    −2   ❌ Bearish
```
Fetches the higher timeframe directly via yfinance (1 extra API call per stock) — an earlier resample-based approach was found to always throw a silent exception on the daily→weekly path.

---

## Volatility Squeeze Detection

TTM-style squeeze — detects when Bollinger Bands compress inside the Keltner Channel, signalling energy building for an explosive move.

### States
```
🔥 FIRED (bullish):   Bands just expanded outward, price above SMA20
🟡 BUILDING:          Still compressed, watch for breakout
⬇️ FIRED (bearish):   Expanded downward — skip
⚠️ HIGH VOLATILITY:   Bands already wide — no compression
```

### Scoring — Fired
| Bars compressed | Score |
|---|---|
| 1-2 | +10 |
| 3-4 | +15 |
| 5-6 | +20 |
| 7-9 | +25 (sweet spot) |
| 10-14 | +22 |
| 15+ | +15 |

### Scoring — Building
```
1-2 bars: +2   3-4 bars: +5   5-6 bars: +8   7+ bars: +12
```
A dedicated **🔥 Squeeze** filter tab is available on both scanners, sorted by fired-status then bar count.

---

## ADX Trend Strength

Measures how strong a trend is — separate from direction (which PSAR and SMA gates already cover).

```
Auto-selects period based on available data:
  ≥60 bars → period 14 (standard)
  ≥40 bars → period 10
  ≥25 bars → period 7
  <25 bars → N/A (insufficient data)
```
Uses correct Wilder RMA smoothing (an earlier formula bug produced values >200, breaking the 0-100 scale entirely).

### Scoring
```
ADX >40: +10   ADX >30: +8   ADX >25: +6   ADX >20: +3
ADX >15:  0    ADX <15: −5 (ranging/choppy)
+DI > -DI and ADX >25: +3 bonus
```

---

## Nifty Market States

Five states (not three) — added LATE_BULL and EARLY_BEAR as transition warnings between the original BULLISH/CAUTION/BEARISH, since the bullish-to-bearish transition is historically the most dangerous period for swing entries.

| State | Position Size | Banner |
|---|---|---|
| 🟢 BULLISH | 100% | None |
| 🟡 LATE_BULL | 75% | "Trend flattening — be selective" |
| ⚠️ CAUTION | 50% | "Selective entry mode" |
| 🟠 EARLY_BEAR | 35% | "Transitioning to bearish — reduce now" |
| 🔴 BEARISH | 25% | "Defensive mode — new entries not recommended" (Monthly Swing only — scan still runs, never hard-blocked) |

---

## Drawdown-Based Position Sizing

Position size automatically shrinks based on **both** current Nifty state and your own running drawdown — protects capital precisely when conditions are worst.

```
Final risk% = base_risk% × nifty_state_multiplier × drawdown_multiplier

Drawdown tiers:
  <3%:  ×1.00 (Normal)
  <7%:  ×0.75 (Caution)
  <12%: ×0.50 (Reduced)
  ≥12%: ×0.25 (Danger)
```
Configurable via a Peak Capital / Current Capital input in the SMA Weekly settings — the app computes your live drawdown % automatically.

---

## Price Action Analysis

Three independent checks run on every stock after indicator gates pass.

### Check 1 — Candle Quality
| Pattern | Score | Signal |
|---------|-------|--------|
| 🔨 Hammer | +15 | Strong buyer rejection — ideal entry |
| 🟢 Bullish Engulfing | +12 | Bulls overwhelmed bears |
| 💚 Strong Bull | +8 | Dominant bullish week |
| 🟡 Mild Bull | +4 | Acceptable entry |
| ➖ Doji | −5 | Indecision — wait |
| 🔴 Bearish Candle | −10 | Sellers in control |
| 🌠 Shooting Star | −15 | Sellers rejected rally |

### Check 2 — Support Proximity
| Level | Score |
|-------|-------|
| At SMA20 (≤ 1%) | +20 |
| Confluence (2+ levels) | +25 |
| At Fibonacci 38.2% | +15 |
| Breakout retest | +15 |
| > 8% above SMA20 | −15 |

### Check 3 — Price Structure
| Structure | Score | Action |
|-----------|-------|--------|
| HH + HL both ✅ | +12 | Uptrend confirmed |
| 3 consecutive green weeks | +10 | Strong momentum |
| Close in top 25% of weekly range | +8 | Weekly strength (also a hard gate — bottom 25% closes are rejected outright) |
| Lower High + Lower Low | −12 | Weakening |
| Structure broken (LH+LL×2) | HARD REJECT | Do not enter |

---

## F&O Expiry Intelligence

NSE F&O stocks get **price-pinned** near expiry as options writers defend positions. The app detects this automatically and adjusts scores.

### Expiry Zones
| Zone | Days to Expiry | Action |
|------|---------------|--------|
| 🟢 SAFE | ≥ 15 days | Enter freely |
| ⚠️ CAUTION | 8–14 days | Prefer non-F&O |
| 🔴 DANGER | 1–7 days | Avoid F&O stocks |
| 🟢 FRESH | 0 days (post expiry) | Best entry window |

### Score Impact
```
F&O + DANGER zone:    −15 pts
F&O + CAUTION zone:   −8 pts
F&O + FRESH zone:     +5 pts
Non-F&O + DANGER:     +10 pts
Non-F&O + CAUTION:    +5 pts
```

### F&O List — Now Self-Service
The F&O stocks list (which NSE revises roughly every 6 months) can now be updated by uploading a CSV in the sidebar (**📋 F&O Stocks List**) instead of requiring a code change. Saved to disk, persists across restarts, with a one-click reset to the built-in default list.

---

## CSV Upload Features

### Stock Universe (both scanners)
```
Universe selector → "📁 Upload My List"
Accepts: CSV or Excel with a 'Symbol' column
         (or symbols in the first column, any common header name)
Auto-cleans: strips/adds '.NS', removes duplicates, handles
             'NSE:' prefixes
```

### F&O Stocks List (sidebar)
```
📋 F&O Stocks List (expander, visible on every page)
Shows: "Active: N symbols · Source: built-in / custom CSV (saved)"
Upload guard against infinite-rerun: tracks filename+size to
avoid reprocessing the same file on every Streamlit rerun
```

---

## Diagnostic & Debug Tools

Built after two separate "scanner returns zero results" incidents — both traced back to silent exceptions inside newly-added filters. These tools exist so that never happens invisibly again.

### Per-Scanner Debug Expanders
```
🔍 Debug: N non-fatal error(s) during scan
  Shown even when results aren't empty — non-fatal filter
  failures (e.g. insufficient price history for one stock)
  default to neutral and don't block the stock, but are now
  visible instead of silently swallowed.
```

### Monthly Swing Scan Funnel
```
📊 Scan Funnel — where did N stocks go?
  Step-by-step countdown through every gate (data fetch →
  SMA trend → candle recovery → proximity → Fibonacci →
  RS vs Nifty → signal pattern → fundamentals → structure →
  final score), with a drop-count at each step.

  Distinguishes "genuinely zero stocks qualify in this
  bearish market" (drops to 0 early, at a trend gate) from
  "something's actually broken" (survives early gates fine,
  then inexplicably drops to 0 at a specific later step).
```

### Isolated Diagnostic Tests (Backtest tab)
```
🔬 Diagnostic: Test RS-vs-Sector Math
🔬 Diagnostic: Test HTF Alignment Math
  Run any new scoring function on a handful of known stocks
  BEFORE wiring it into a live scanner. Both major fixes in
  the 20-Jun-2026 session were validated this way first —
  one bug (HTF's monthly lookback being too long) was caught
  at this stage, never reaching the live Monthly Swing
  scanner at all.
```

---

## Portfolio Management

### Position Card Features
```
Source badge:     📅 Monthly / 📈 Weekly / ⚡ Intraday / 🏷️ Manual
Live P&L:         Real-time unrealised profit/loss
Progress bar:     Visual SL → Current → T2 range
Target boxes:     SL / T1 / T2 / T3 / T4 with HIT detection
PSAR strip:       ⏳ Waiting T1 / ✅ HOLD / 🔴 EXIT
F&O warning:      Expiry proximity alert
Card border:      Red (SL hit) / Green (PSAR bullish) / Blue (waiting)
```

### Manual Add Position
```
Fields: Symbol, Entry, Qty, SL, T1, T2, Trade Type, Date, Notes
Types:  Intraday / SMA Weekly / Monthly Swing / Delivery
Badge:  🏷️ Manual shown on card
```

### Auto-Sell Feature
```
Take Profit %:  Auto square off when gain hits threshold
Stop Loss %:    Auto square off when loss hits threshold
Triggers on:    Next "Refresh P&L" button press
```

---

## Installation

### Requirements
```bash
pip install streamlit yfinance pandas numpy kiteconnect anthropic
```

### Run
```bash
streamlit run intraday_trading_app.py
```

### Credentials Setup
Create these files in `~/Downloads/`:

**kite_creds.json**
```json
{
  "api_key": "your_kite_api_key",
  "access_token": "your_access_token"
}
```

**anthropic_creds.json**
```json
{
  "api_key": "your_anthropic_api_key"
}
```

**investo_intraday_portfolio.json**
```json
[]
```

---

## Configuration

### Monthly Swing Settings
| Setting | Default | Description |
|---------|---------|-------------|
| Universe | Midcap 100 / Upload My List | Stock universe to scan |
| Capital | ₹5,00,000 | Trading capital |
| Risk % | 2% | Max risk per trade (further auto-adjusted by Nifty state + drawdown) |
| Min Score | 65 | Minimum raw scanner score |
| Volatility Penalty | Starts at 6% ATR | Weekly ATR penalty threshold |

### SMA Weekly Settings
| Setting | Default | Description |
|---------|---------|-------------|
| Universe | Midcap 100 / Upload My List | Stock universe to scan |
| Capital | ₹5,00,000 | Trading capital |
| Risk % | 2% | Max risk per trade (further auto-adjusted by Nifty state + drawdown) |
| Min Score | 65 | Minimum raw scanner score |
| Volatility Penalty | Starts at 4% ATR | Daily ATR penalty threshold |
| Peak/Current Capital | — | Drives drawdown-based position sizing |

---

## Key Concepts

### 25+ Trading Concepts Used

| # | Concept | Purpose |
|---|---------|---------|
| 1 | Trend Following | Only trade in direction of trend |
| 2 | MA Crossover | Detect trend change signal |
| 3 | Pullback Trading | Buy near support not breakout |
| 4 | Breakout Trading | 13-week / 52-week high with volume |
| 5 | Fibonacci Retracement | Natural support/resistance zones |
| 6 | Relative Strength (vs Nifty) | Compare to index benchmark |
| 7 | Relative Strength (vs own Sector) | Find leaders within any sector, regardless of sector rank |
| 8 | Higher Timeframe Alignment | Confirm signal on the next-higher timeframe |
| 9 | Volatility Squeeze | Detect energy compression before a breakout |
| 10 | ADX | Trend strength, independent of direction |
| 11 | OBV | Volume confirms price trend |
| 12 | MACD | Momentum confirmation |
| 13 | RSI | Overbought/oversold detection |
| 14 | ATR | Volatility-based SL and targets |
| 15 | Parabolic SAR | Trailing stop loss |
| 16 | Sector Momentum (verified ETF data) | Trade with sector tailwind |
| 17 | Inside Bar | Energy compression signal |
| 18 | HH / HL Structure | Uptrend structure validation |
| 19 | Fundamental Analysis | D/E, EPS, Promoter quality |
| 20 | Price Action | Candle quality at support |
| 21 | Candle Close Position | Reject weeks where sellers won (bottom 25% close) |
| 22 | Volatility Filter | ATR% scoring and display |
| 23 | Liquidity Filter | Ensure exit is always possible |
| 24 | Market State Transitions | 5-state Nifty model, not just bull/bear |
| 25 | Drawdown-Adjusted Sizing | Position size shrinks as personal drawdown grows |

### PSAR Settings Explained
```
Monthly Swing:  step=0.01, max=0.10 (slow — weekly chart)
  → Designed for 3–5 week hold
  → Ignores small weekly dips
  → Only exits on genuine reversal

SMA Weekly:     step=0.02, max=0.20 (fast — daily chart)
  → Designed for 3–7 day hold
  → Matches TradingView default
  → Responsive to daily moves
```

### F&O Expiry Calendar Rule
```
Week 1–2 of month:  Enter any stock freely
Week 3 of month:    Prefer non-F&O stocks
Expiry week:        Only non-F&O stocks
Post expiry:        Best entry window — enter freely
```

### Entry Quality Rule
```
High ATR stock (>5% weekly):
  Only enter within 1% of SMA20
  Beyond 3% → app hides the stock

Normal ATR stock (≤5% weekly):
  Enter within 2% of SMA20
  Beyond 5% → app hides the stock
```

---

## How to Use — Weekend Routine

```
Saturday morning:

1. Open Monthly Swing tab
   → Check Nifty state banner (especially LATE_BULL/EARLY_BEAR)
   → If results are empty, check the 📊 Scan Funnel first
   → Look for 🔥 CONFIDENT BUY / ✅ STRONG SETUP stocks
   → Check entry badge (🟢 ENTER NOW first)
   → Verify PSAR is bullish
   → Check RS vs Sector and HTF lines — both should agree
     with the overall signal direction
   → Note SL and T1 targets

2. Open SMA Weekly tab
   → Same process
   → Pick 1–2 max stocks

3. Portfolio tab
   → Check PSAR status for all positions
   → Update Zerodha SL to PSAR level
   → Note any F&O expiry warnings

4. Sunday evening
   → Re-check SMA Weekly for Monday entry
   → Confirm Nifty trend is bullish

5. Monday morning 9:20 AM
   → Enter positions (not 9:15 — too volatile)
   → Set SL in Zerodha immediately
```

---

## Changelog

### 20-Jun-2026 — Sector & Relative-Strength Audit
```
Fixed:
  - FINANCE sector ticker (^CNXFIN) found completely dead on
    Yahoo — was silently using neutral fallback values
  - RS-vs-sector unit-mismatch bug (raw return vs relative number)
  - HTF alignment: 3 separate bugs (broken resample, insufficient
    monthly data depth, too-long lookback giving zero discrimination)
  - 2 stock sector misclassifications (ONESOURCE, SWIGGY)
  - 1 dead stock ticker (MAHINDLOG → MAHLOG)

Added:
  - RS vs Own Sector — live on both scanners
  - Higher Timeframe Alignment — live on both scanners
  - Isolated diagnostic test tools (Backtest tab)
  - Per-scanner debug error expanders
  - Monthly Swing Scan Funnel diagnostic
  - Stock universe CSV uploader (both scanners)
  - Dynamic, self-service F&O stocks list (sidebar CSV upload)

Upgraded:
  - 3 sectors moved from stock-basket proxy to verified real
    ETF data: PSU_BANK, PVT_BANK, CONSUMPTION
```

### Earlier sessions (v7 baseline, see prior transcripts)
```
- Unified sector ranking across all 3 sector-aware tabs
- Volatility squeeze detection (TTM-style)
- ADX calculation (Wilder RMA, period auto-selection)
- 5-state Nifty market model (added LATE_BULL, EARLY_BEAR)
- Drawdown-based position sizing
- Candle close position + body momentum filters
- Confident score threshold recalibration (130/100/75/55)
```

---

## Support

Built for personal use — NSE trading, India.
Data via yfinance (delayed) or Zerodha Kite API (real-time).

> Scanner finds the stock · You read the chart · You set SL and target