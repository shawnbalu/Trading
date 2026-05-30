# 📈 Investo — NSE Swing Trading System

A professional-grade Streamlit application for NSE swing trading with AI-powered analysis, real-time portfolio tracking, and intelligent entry signals.

---

## 📋 Table of Contents

- [Overview](#overview)
- [App Structure](#app-structure)
- [Monthly Swing Scanner](#monthly-swing-scanner)
- [SMA Weekly Scanner](#sma-weekly-scanner)
- [Confident Score System](#confident-score-system)
- [Price Action Analysis](#price-action-analysis)
- [F&O Expiry Intelligence](#fo-expiry-intelligence)
- [Portfolio Management](#portfolio-management)
- [Installation](#installation)
- [Configuration](#configuration)
- [Key Concepts](#key-concepts)

---

## Overview

Investo is built for active NSE swing traders who want a systematic, data-driven approach to finding high-probability trade setups. It eliminates guesswork by combining 18 trading concepts into one decisive score per stock.

```
Scanner finds the stock
Price Action confirms the entry
Confident Score tells you to enter or skip
PSAR tells you when to exit
```

### Tech Stack

```
Frontend:    Streamlit
Data:        yfinance (OHLCV), Zerodha Kite API (live prices)
AI:          Anthropic Claude API (stock validation)
Language:    Python 3.10+
Lines:       13,000+
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
```

---

## Monthly Swing Scanner

Scans NSE stocks using **weekly candles** for medium-term swing trades held 3–5 weeks.

### Data Source
```
Interval:  Weekly (1wk)
Period:    2 years
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
| Fundamentals | D/E ≤ 2.0, EPS > 0, Promoter > 20% |

### Signal Types
```
Signal 1 — SMA Cross:       SMA20 crossed SMA50 in last 3 weeks
Signal 2 — Pullback Bounce: Trend ≥ 4 weeks + touched SMA20 + recovering
Signal 3 — 13-Week Breakout: Price > 13-week high with volume ≥ 1.5×
```

### Scoring Factors (additional)
```
Nifty alignment        ±10
RS vs Nifty            ±10
OBV (accumulation)     ±10
Sector momentum        ±8
52W proximity          ±10
MACD weekly            ±12
Inside week            ±8
Volatility ATR%        ±8 to −15
```

### Trade Plan
```
Entry:  Live daily price
SL:     max(entry − 2×ATR, SMA20 × 0.97)
T1:     Entry + 1× weekly ATR
T2:     Entry + 2× weekly ATR
T3:     Entry + 3× weekly ATR
```

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
Period:    1 year
PA checks: Resampled daily → weekly
PSAR:      Daily candles, step=0.02, max=0.20
```

### Hard Gates
| Gate | Condition |
|------|-----------|
| Trend | SMA20 > SMA50 |
| SMA20 slope | > 0 (hard reject if flat/declining) |
| Price | > Daily SMA20 |
| SMA50 slope | > −0.5% |

### Signal Types
```
Signal 1 — Fresh Cross:    SMA20 crossed SMA50 in last 5 days
Signal 2 — Pullback Bounce: SMA20 > SMA50 ≥ 5 days + pulled to SMA20 + bouncing
```

### Differences from Monthly Swing
| Factor | Monthly Swing | SMA Weekly |
|--------|--------------|------------|
| Candles | Weekly | Daily |
| PSAR step | 0.01 / 0.10 | 0.02 / 0.20 |
| ATR limit | Weekly 6% max | Daily 4% max |
| Fundamentals | D/E, EPS, Promoter | Not checked |
| Targets | 1× / 2× / 3× ATR | 0.5× / 1× / 1.5× ATR |
| Check frequency | Every Friday | Every morning |

---

## Confident Score System

Every stock gets a **0–100 Confident Score** combining 7 factors. Eliminates the need for manual re-analysis after the scan.

### Score Components
| # | Factor | Max Points |
|---|--------|-----------|
| 1 | Technical scanner score | 30 |
| 2 | PSAR status (most important) | 20 |
| 3 | Price structure HH + HL | 15 |
| 4 | Entry badge proximity | 15 |
| 5 | Risk:Reward at T2 | 10 |
| 6 | Liquidity grade | 5 |
| 7 | F&O expiry penalty/bonus | ±15 |

### Score Thresholds
```
≥ 80  🔥 CONFIDENT BUY  — Enter without further analysis
60–79 ✅ GOOD SETUP     — Quick chart check then enter
40–59 ⚠️ WEAK SETUP     — Skip this week
< 40  ❌ SKIP           — Not shown in results
```

### Entry Badge System
Stocks are shown only if they meet proximity criteria. Results are sorted with `🟢 ENTER NOW` first.

| Badge | Normal ATR (≤5%) | High ATR (>5%) |
|-------|-----------------|----------------|
| 🟢 ENTER NOW | ≤ 2% above SMA20 | ≤ 1% above SMA20 |
| 🟡 ACCEPTABLE | 2–5% above SMA20 | 1–3% above SMA20 |
| ❌ HIDDEN | > 5% | > 3% |

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
| Close in top 25% | +8 | Weekly strength |
| Lower High + Lower Low | −12 | Weakening |
| Structure broken (LH+LL×2) | HARD REJECT | Do not enter |

### PA Signal Labels
```
🔥 STRONG SETUP  PA score ≥ 30  Enter Monday open
✅ GOOD SETUP    PA score ≥ 15  Enter with normal size
⚠️ WEAK SETUP    PA score ≥ 0   Wait for better candle
🔴 RISKY         PA score < 0   Wait for pullback
🔴 AVOID         Structure broken — never enter
```

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
F&O + DANGER zone:    −15 pts  (drops from GOOD → WEAK)
F&O + CAUTION zone:   −8 pts
F&O + FRESH zone:     +5 pts
Non-F&O + DANGER:     +10 pts  (rises to top of list)
Non-F&O + CAUTION:    +5 pts
```

### What You See
```
Scanner header:  "🔴 Expiry in 6 days — Non-F&O stocks shown first"
Result card:     "📌 F&O · 6 days · ⚠️ Price may be pinned"
Portfolio card:  "⚠️ HINDCOPPER — Expiry in 6 days — Consider tightening SL"
Post expiry:     "🟢 Post-Expiry — Best entry window — Fresh cycle"
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
Add any stock not from the scanner:
```
Fields: Symbol, Entry, Qty, SL, T1, T2, Trade Type, Date, Notes
Types:  Intraday / SMA Weekly / Monthly Swing / Delivery
Badge:  🏷️ Manual shown on card
```

### Auto-Sell Feature
Configure automatic exit triggers:
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
| Universe | Midcap 100 | Stock universe to scan |
| Capital | ₹5,00,000 | Trading capital |
| Risk % | 2% | Max risk per trade |
| Min Score | 65 | Minimum scanner score |
| Volatility Penalty | Starts at 6% ATR | Weekly ATR penalty threshold |

### SMA Weekly Settings
| Setting | Default | Description |
|---------|---------|-------------|
| Universe | Midcap 100 | Stock universe to scan |
| Capital | ₹5,00,000 | Trading capital |
| Risk % | 2% | Max risk per trade |
| Min Score | 65 | Minimum scanner score |
| Volatility Penalty | Starts at 4% ATR | Daily ATR penalty threshold |

---

## Key Concepts

### 18 Trading Concepts Used

| # | Concept | Purpose |
|---|---------|---------|
| 1 | Trend Following | Only trade in direction of trend |
| 2 | MA Crossover | Detect trend change signal |
| 3 | Pullback Trading | Buy near support not breakout |
| 4 | Breakout Trading | 13-week high with volume |
| 5 | Fibonacci Retracement | Natural support/resistance zones |
| 6 | Relative Strength | Compare to Nifty benchmark |
| 7 | OBV | Volume confirms price trend |
| 8 | MACD | Momentum confirmation |
| 9 | RSI | Overbought/oversold detection |
| 10 | ATR | Volatility-based SL and targets |
| 11 | Parabolic SAR | Trailing stop loss |
| 12 | Sector Momentum | Trade with sector tailwind |
| 13 | Inside Bar | Energy compression signal |
| 14 | HH / HL Structure | Uptrend structure validation |
| 15 | Fundamental Analysis | D/E, EPS, Promoter quality |
| 16 | Price Action | Candle quality at support |
| 17 | Volatility Filter | ATR% scoring and display |
| 18 | Liquidity Filter | Ensure exit is always possible |

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

## CSV Export

Both scanners export results as dated CSV files.

### Monthly Swing CSV Columns
```
Symbol, Score, Entry, SMA20, SMA50, ATR, RSI, Fib_Retrace, Fib_Zone,
HH, HL, RS_vs_Nifty, OBV_Slope, MACD, Stop_Loss, T1, T2, T3,
Qty, Investment, RR_T1, RR_T2, RR_T3, Liquidity, DE_Ratio,
Promoter_Pct, EPS, Results_Warning, Cap_Tier, Scan_Date
```

### SMA Weekly CSV Columns
```
Symbol, Score, Signal_Type, Entry, SMA20, SMA50, ATR,
RSI, Vol_Ratio, Trend_Days, SMA20_Slope, HH, HL,
Stop_Loss, T1, T2, T3, Qty, PSAR, PSAR_Bullish,
Liquidity, Cap_Tier, Week_Change, Scan_Date
```

---

## How to Use — Weekend Routine

```
Saturday morning:

1. Open Monthly Swing tab
   → Check expiry banner
   → Look for 🔥 CONFIDENT BUY stocks
   → Check entry badge (🟢 ENTER NOW first)
   → Verify PSAR is bullish
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

## Support

Built for personal use — NSE trading, India.
Data via yfinance (delayed) or Zerodha Kite API (real-time).

> Scanner finds the stock · You read the chart · You set SL and target
