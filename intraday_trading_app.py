"""
============================================================
  INTRADAY TRADING SYSTEM — STREAMLIT APP
  Zerodha Kite API · Real-Time 1min Candles
============================================================
INSTALL:
    pip3 install streamlit kiteconnect pandas numpy plotly pytz

RUN:
    streamlit run intraday_trading_app.py

ZERODHA KITE SETUP:
  1. Login at https://kite.trade → Create app → get API Key + Secret
  2. On first run, enter API Key + Secret in the sidebar
  3. Click "Generate Login URL" → login to Zerodha → copy request_token
  4. Paste request_token → Click "Connect" → access_token saved for the day
  5. Token auto-expires at midnight, reconnect next trading day

DATA SOURCE:
  • Primary  : Zerodha Kite API — real-time 1min candles (0 delay)
  • Fallback : yfinance — 5min candles with 15–20 min delay
    (fallback activates automatically if Kite not connected)
============================================================
KEY FEATURES:
  • Real-Time    1min candles via Kite API (no delay)
  • Auto-Refresh every 1 min during market hours
  • Smart Alerts 🔔 BUY / ⚠️ EXIT / 🛑 STOP LOSS triggered live
  • Alert Log    full history of all alerts fired this session
  • Volume Profile  Point of Control + Value Area on chart
  • EMA Ribbon   5/9/21/50 EMA stack for trend clarity
  • Signals:     Tighter thresholds · VWAP anchored intraday
  • Charges:     Intraday brokerage (0.03%) · No STT on buy
  •              STT only on sell side (0.025% intraday)
  • Targets:     Smaller ATR multiples (0.5x, 1x, 1.5x, 2x)
  • Stops:       Tighter (0.5×ATR)
  • LSTM:        Predicts next 3 CANDLES (not days)
  • Session:     9:15 AM – 3:30 PM IST market hours
============================================================
"""

import sys, os, tempfile, warnings, time, json, pathlib
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, date, timedelta
import pytz

os.environ["YFINANCE_CACHE_DIR"] = tempfile.gettempdir()

# ── Global in-memory data cache ───────────────────────────
# Prevents re-fetching same stock within same scan session.
# Key = symbol_interval_timebucket  Value = (df, source)
# Cleared at start of each new scan run.
_DATA_CACHE: dict = {}

# ── Zerodha Kite API ──────────────────────────────────────
try:
    from kiteconnect import KiteConnect
    KITE_AVAILABLE = True
except ImportError:
    KITE_AVAILABLE = False

# ── yfinance fallback ─────────────────────────────────────
try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

if not KITE_AVAILABLE and not YF_AVAILABLE:
    st.error("❌ Install kiteconnect: pip3 install kiteconnect\n"
             "   Or yfinance fallback: pip3 install yfinance"); st.stop()

# ── Kite credentials file (saves API key/secret/token) ───
KITE_CREDS_FILE     = pathlib.Path.home() / "Downloads" / "kite_creds.json"
ANTHROPIC_CREDS_FILE= pathlib.Path.home() / "Downloads" / "anthropic_creds.json"

def load_anthropic_key() -> str:
    """Load Anthropic API key from file or session state."""
    # 1. Check session state first (set via sidebar input)
    if st.session_state.get('anthropic_api_key'):
        return st.session_state['anthropic_api_key']
    # 2. Check local file
    try:
        if ANTHROPIC_CREDS_FILE.exists():
            data = json.loads(ANTHROPIC_CREDS_FILE.read_text(encoding='utf-8'))
            key  = data.get('api_key', '')
            if key:
                st.session_state['anthropic_api_key'] = key
                return key
    except Exception:
        pass
    return ''

def save_anthropic_key(key: str):
    """Save Anthropic API key to file."""
    try:
        ANTHROPIC_CREDS_FILE.write_text(
            json.dumps({'api_key': key}, indent=2, ensure_ascii=False),
            encoding='utf-8')
    except Exception as e:
        st.warning(f"⚠️ Could not save Anthropic key: {e}")

def load_kite_creds():
    try:
        if KITE_CREDS_FILE.exists():
            return json.loads(KITE_CREDS_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}

def save_kite_creds(creds: dict):
    try:
        KITE_CREDS_FILE.write_text(
            json.dumps(creds, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        st.warning(f"⚠️ Could not save Kite creds: {e}")

def get_kite_client():
    """
    Returns an authenticated KiteConnect instance if credentials exist,
    otherwise None. Caches in session_state['kite'].
    """
    if not KITE_AVAILABLE:
        return None
    if 'kite' in st.session_state and st.session_state['kite'] is not None:
        return st.session_state['kite']
    creds = load_kite_creds()
    api_key      = creds.get('api_key', '')
    access_token = creds.get('access_token', '')
    token_date   = creds.get('token_date', '')
    today_str    = datetime.now().strftime('%Y-%m-%d')
    if api_key and access_token and token_date == today_str:
        try:
            kite = KiteConnect(api_key=api_key)
            kite.set_access_token(access_token)
            st.session_state['kite'] = kite
            return kite
        except Exception:
            pass
    st.session_state['kite'] = None
    return None

# ── Kite instrument token cache ───────────────────────────
INSTRUMENT_CACHE_FILE = pathlib.Path.home() / "Downloads" / "kite_instruments.json"

def load_instrument_cache():
    try:
        if INSTRUMENT_CACHE_FILE.exists():
            data = json.loads(INSTRUMENT_CACHE_FILE.read_text(encoding="utf-8"))
            if data.get('date') == datetime.now().strftime('%Y-%m-%d'):
                return data.get('tokens', {})
    except Exception:
        pass
    return {}

def save_instrument_cache(tokens: dict):
    try:
        INSTRUMENT_CACHE_FILE.write_text(
            json.dumps({'date': datetime.now().strftime('%Y-%m-%d'), 'tokens': tokens},
                       indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

def get_instrument_token(kite, symbol_ns: str) -> int | None:
    """
    Convert NSE symbol (e.g. 'RELIANCE.NS') to Kite instrument token.
    Uses a local daily cache to avoid repeated API calls.
    Handles known symbol mismatches between Yahoo Finance and Kite NSE.
    """
    # ── Known Yahoo → Kite symbol differences ────────────
    SYMBOL_MAP = {
        # Reliance group
        'RELINFRA':     'RELINFRA',      # May be suspended/delisted on Kite
        # Adani group
        'ADANIENSOL':   'ADANIENSOL',
        # Tata group
        'TATAINVEST':   'TATAINVST',
        # Bajaj group
        'BAJAJHFL':     'BAJAJHFL',
        # Banking
        'IDFCFIRSTB':   'IDFCFIRSTB',
        # Auto
        'MOTHERSON':    'MOTHERSON',
        'TMPV':         'TITAGARH',      # different listing
        # Energy
        'ATHERENERG':   'ATGL',
        'PREMIERENE':   'PREMIERENE',
        # IT
        'NETWEB':       'NETWEB',
        # Others with known Kite name differences
        'ARE&M':        'AREIM',
        'M&M':          'MM',
        'M&MFIN':       'MMFIN',
        'J&KBANK':      'JKBANK',
        'GVT&D':        'GVTD',
        'HEXT':         'HEXT',
        'ABLBL':        'ABLBL',
        'ABREL':        'ABREL',
        'AIIL':         'AIIL',
        'COHANCE':      'COHANCE',
        'ENRIN':        'ENRIN',
        'IGIL':         'IGIL',
        'INDGN':        'INDGN',
        'IKS':          'IKS',
        'JBMA':         'JBMA',
        'JWL':          'JWL',
        'ONESOURCE':    'ONESOURCE',
        'PTCIL':        'PTCIL',
        'SAILIFE':      'SAILIFE',
        'SAMMAANCAP':   'SAMMAANCAP',
        'SWANCORP':     'SWANCORP',
        'TARIL':        'TARIL',
        'VENTIVE':      'VENTIVE',
        'VMM':          'VMM',
        'ZENTEC':       'ZENTEC',
    }

    raw_symbol = symbol_ns.replace('.NS', '').upper()
    symbol     = SYMBOL_MAP.get(raw_symbol, raw_symbol)

    cache = load_instrument_cache()
    if symbol in cache:
        return cache[symbol]

    # Also check raw symbol in cache (in case it was stored under original name)
    if raw_symbol in cache:
        return cache[raw_symbol]

    try:
        instruments = kite.instruments("NSE")
        tokens = {i['tradingsymbol']: i['instrument_token']
                  for i in instruments if i['exchange'] == 'NSE'}
        save_instrument_cache(tokens)
        # Try mapped symbol first, then raw
        return tokens.get(symbol) or tokens.get(raw_symbol)
    except Exception:
        return None


# ─────────────────────────────────────────────
#  AUTO-REFRESH ENGINE
#  Uses st.empty + time.sleep loop to count down
#  5 min during market hours, paused otherwise
# ─────────────────────────────────────────────

AUTO_REFRESH_SECONDS = 60   # 1 minute (matches 1min candle interval)

def should_auto_refresh():
    """Returns True if it's time for an auto-refresh."""
    if 'last_auto_refresh' not in st.session_state:
        st.session_state['last_auto_refresh'] = time.time()
        return False
    elapsed = time.time() - st.session_state['last_auto_refresh']
    return elapsed >= AUTO_REFRESH_SECONDS

def reset_refresh_timer():
    st.session_state['last_auto_refresh'] = time.time()

def seconds_until_refresh():
    if 'last_auto_refresh' not in st.session_state:
        return AUTO_REFRESH_SECONDS
    elapsed = time.time() - st.session_state['last_auto_refresh']
    return max(0, int(AUTO_REFRESH_SECONDS - elapsed))


# ─────────────────────────────────────────────
#  ALERT ENGINE
#  Evaluates BUY / EXIT / STOP LOSS conditions
#  for every scanned stock and stores in session
# ─────────────────────────────────────────────

ALERT_LOG_KEY = 'alert_log'

def _init_alert_log():
    if ALERT_LOG_KEY not in st.session_state:
        st.session_state[ALERT_LOG_KEY] = []

def _add_alert(symbol, alert_type, message, price, icon='📣'):
    _init_alert_log()
    log = st.session_state[ALERT_LOG_KEY]
    # Avoid duplicate alerts for same symbol+type within 5 min window
    now_str = ist_now().strftime('%H:%M')
    for prev in log:
        if prev['symbol'] == symbol and prev['type'] == alert_type and prev['time'] == now_str:
            return
    log.insert(0, {
        'symbol':  symbol,
        'type':    alert_type,
        'message': message,
        'price':   price,
        'icon':    icon,
        'time':    ist_now().strftime('%H:%M IST'),
        'date':    ist_now().strftime('%d %b %Y'),
    })
    # Keep only last 50 alerts
    st.session_state[ALERT_LOG_KEY] = log[:50]

def evaluate_alerts(result, portfolio):
    """
    Evaluates a scanned stock result against all alert conditions.
    Fires alerts and returns list of active alert dicts for this stock.
    """
    sym        = result['symbol'].replace('.NS', '')
    price      = result['price']
    rsi        = result['rsi']
    vwap_pos   = result['vwap']
    vol_ratio  = result['vol_ratio']
    verdict    = result.get('_verdict', '')
    pick_score = result.get('_pick_score', 0)
    sig_val    = result['signal_val']
    macd       = result['macd']
    latest     = result['latest']
    supertrend = result['supertrend']
    adx        = result['adx']
    live_bull  = result['live_bull']
    live_bear  = result['live_bear']

    alerts = []

    # ── 🔔 BUY ALERT ──────────────────────────────────────
    # Fires when ALL 5 key criteria align perfectly
    if (verdict in ['⭐⭐⭐ STRONG BUY', '⭐⭐ BUY'] and
        vwap_pos == 'ABOVE' and
        vol_ratio >= 2.0 and
        45 <= rsi <= 65 and
        supertrend == 1):
        msg = (f"📥 BUY SIGNAL — {sym} @ ₹{price:,.2f} | "
               f"VWAP Above · Vol {vol_ratio:.1f}× · RSI {rsi:.0f} · "
               f"Supertrend Bull · Score {pick_score}")
        _add_alert(sym, 'BUY', msg, price, '🔔')
        alerts.append({'type': 'BUY', 'msg': msg, 'color': '#16a34a', 'bg': '#f0fdf4', 'icon': '🔔'})

    # ── 🚨 STRONG BUY ALERT (extra loud when score ≥ 80) ──
    if verdict == '⭐⭐⭐ STRONG BUY' and pick_score >= 80 and vwap_pos == 'ABOVE':
        msg = (f"🚨 STRONG BUY — {sym} @ ₹{price:,.2f} | "
               f"Score {pick_score}/100 · Vol {vol_ratio:.1f}× · RSI {rsi:.0f}")
        _add_alert(sym, 'STRONG_BUY', msg, price, '🚨')
        alerts.append({'type': 'STRONG_BUY', 'msg': msg, 'color': '#15803d', 'bg': '#dcfce7', 'icon': '🚨'})

    # ── ⚡ VOLUME SURGE ALERT ──────────────────────────────
    # Tiered alerts — institutional surge gets stronger label
    if vol_ratio >= 3.0:
        direction = "📈 BULL" if price > float(result['prev']['Close']) else "📉 BEAR"
        if vol_ratio >= 15.0:
            _tier = "🏦 INSTITUTIONAL SURGE"
            _icon = '🏦'
        elif vol_ratio >= 8.0:
            _tier = "🔥 MAJOR VOLUME"
            _icon = '🔥'
        elif vol_ratio >= 5.0:
            _tier = "⚡ STRONG SURGE"
            _icon = '⚡'
        else:
            _tier = "⚡ VOLUME SURGE"
            _icon = '⚡'
        msg = (f"{_tier} — {sym} @ ₹{price:,.2f} | "
               f"{vol_ratio:.1f}× avg volume · {direction}")
        _add_alert(sym, 'VOL_SURGE', msg, price, _icon)
        alerts.append({'type': 'VOL_SURGE', 'msg': msg, 'color': '#d97706', 'bg': '#fffbeb', 'icon': _icon})

    # ── ⚠️ VWAP BREAKDOWN ALERT (for open positions) ──────
    open_syms = [p.get('symbol','') for p in portfolio if p.get('status') == 'OPEN']
    if sym in open_syms and vwap_pos == 'BELOW':
        msg = (f"⚠️ VWAP BREAK — {sym} @ ₹{price:,.2f} | "
               f"Price fell below VWAP · Consider exit")
        _add_alert(sym, 'VWAP_BREAK', msg, price, '⚠️')
        alerts.append({'type': 'VWAP_BREAK', 'msg': msg, 'color': '#dc2626', 'bg': '#fff5f5', 'icon': '⚠️'})

    # ── 🔴 RSI OVERBOUGHT ALERT (for open positions) ───────
    if sym in open_syms and rsi > 72:
        msg = (f"🔴 RSI OVERBOUGHT — {sym} @ ₹{price:,.2f} | "
               f"RSI-7 = {rsi:.0f} · Book partial profit now")
        _add_alert(sym, 'RSI_OB', msg, price, '🔴')
        alerts.append({'type': 'RSI_OB', 'msg': msg, 'color': '#dc2626', 'bg': '#fff5f5', 'icon': '🔴'})

    # ── 🛑 STOP LOSS ALERT (for open positions) ────────────
    for p in portfolio:
        if p.get('symbol') == sym and p.get('status') == 'OPEN':
            sl = _f(p.get('stop_loss', 0))
            if sl > 0 and price <= sl:
                msg = (f"🛑 STOP LOSS HIT — {sym} @ ₹{price:,.2f} | "
                       f"SL was ₹{sl:,.2f} · EXIT IMMEDIATELY")
                _add_alert(sym, 'STOP_LOSS', msg, price, '🛑')
                alerts.append({'type': 'STOP_LOSS', 'msg': msg, 'color': '#7f1d1d', 'bg': '#fef2f2', 'icon': '🛑'})

    # ── 🎯 TARGET HIT ALERTS (for open positions) ──────────
    for p in portfolio:
        if p.get('symbol') == sym and p.get('status') == 'OPEN':
            for tkey, tlabel in [('t1','T1 Scalp'), ('t2','T2 Target'), ('t3','T3 Extended'), ('t4','T4 Stretch')]:
                tval = _f(p.get(tkey, 0))
                if tval > 0 and price >= tval:
                    msg = (f"🎯 {tlabel} HIT — {sym} @ ₹{price:,.2f} | "
                           f"Target was ₹{tval:,.2f} · Book {'50%' if tkey=='t1' else '30%' if tkey=='t2' else '20%' if tkey=='t3' else 'rest'} now")
                    _add_alert(sym, f'TARGET_{tkey.upper()}', msg, price, '🎯')
                    alerts.append({'type': f'TARGET_{tkey.upper()}', 'msg': msg,
                                   'color': '#15803d', 'bg': '#f0fdf4', 'icon': '🎯'})
                    break  # Only alert for highest target hit

    # ── 🕒 TIME WARNING (3:00 PM) ──────────────────────────
    if sym in open_syms:
        now_ist = ist_now()
        if now_ist.hour == 15 and now_ist.minute >= 0 and now_ist.minute < 15:
            msg = (f"🕒 TIME WARNING — {sym} | "
                   f"3:00 PM IST — Start exiting positions · 15 min left")
            _add_alert(sym, 'TIME_WARN', msg, price, '🕒')
            alerts.append({'type': 'TIME_WARN', 'msg': msg, 'color': '#92400e', 'bg': '#fffbeb', 'icon': '🕒'})

    return alerts



# ─────────────────────────────────────────────
#  OPENING RANGE BREAKOUT (ORB) ENGINE
#  Catches moves like LTTS +13% at 9:16 AM
#  before standard indicators warm up
# ─────────────────────────────────────────────

def detect_opening_breakout(df, symbol, price, prev_close):
    """
    5 breakout rules checked in first 30 min of market open.
    Returns list of breakout signals found.
    """
    breakouts = []
    sym = symbol.replace('.NS', '')

    try:
        now_ist    = ist_now()
        mkt_start  = now_ist.replace(hour=9, minute=15, second=0, microsecond=0)
        try:
            mins_open = int((now_ist - mkt_start.astimezone(now_ist.tzinfo)).total_seconds() / 60) \
                        if market_open() else 999
        except Exception:
            mins_open = 999

        # Today's candles
        try:
            today_date = pd.Timestamp.now().date()
            today_df   = df[pd.to_datetime(df.index).date == today_date]
            if len(today_df) < 2:
                today_df = df.tail(60)
        except Exception:
            today_df = df.tail(60)

        if len(today_df) < 2:
            return breakouts

        vol_ma         = float(df['Volume_MA'].iloc[-1]) if 'Volume_MA' in df.columns \
                         and not pd.isna(df['Volume_MA'].iloc[-1]) else max(float(df['Volume'].mean()), 1)
        first_candle   = today_df.iloc[0]
        first_vol      = float(first_candle['Volume'])
        first_high     = float(first_candle['High'])
        first_low      = float(first_candle['Low'])
        first_close    = float(first_candle['Close'])
        last_candle    = today_df.iloc[-1]
        last_close     = float(last_candle['Close'])
        last_vol       = float(last_candle['Volume'])
        vwap           = float(df['VWAP'].iloc[-1]) if 'VWAP' in df.columns \
                         and not pd.isna(df['VWAP'].iloc[-1]) else 0

        first_vol_ratio = first_vol / vol_ma
        last_vol_ratio  = last_vol  / vol_ma
        chg_pct         = ((last_close - prev_close) / prev_close * 100) if prev_close > 0 else 0

        # Rule 1 — Opening Volume Explosion (≥3× on first candle, bullish)
        if first_vol_ratio >= 3.0 and first_close > prev_close:
            breakouts.append({
                'type':   'ORB_VOL',
                'icon':   '🚀',
                'title':  'Opening Volume Explosion',
                'msg':    f"🚀 OPENING BREAKOUT — {sym} | First candle {first_vol_ratio:.0f}× vol · ₹{price:,.2f} · {chg_pct:+.1f}%",
                'score':  95,
                'color':  '#7c3aed',
                'bg':     '#f5f3ff',
                'action': 'STRONG BUY — Enter now, do not wait for indicators to confirm',
            })

        # Rule 2 — Gap & Hold (>1.5% gap OR volume confirming, holding above open)
        if chg_pct >= 1.5 and last_close >= first_close * 0.998 and last_vol_ratio >= 1.5:
            breakouts.append({
                'type':   'ORB_GAP',
                'icon':   '📈',
                'title':  'Gap & Hold Breakout',
                'msg':    f"📈 GAP & HOLD — {sym} | +{chg_pct:.1f}% from ₹{prev_close:,.2f} · Holding · Vol {last_vol_ratio:.1f}×",
                'score':  82,
                'color':  '#15803d',
                'bg':     '#f0fdf4',
                'action': 'BUY — Gap holding with volume. Enter on 1min pullback to VWAP.',
            })

        # Rule 3 — VWAP Reclaim (price was below VWAP, crossed above with volume)
        if (vwap > 0 and last_close > vwap and mins_open <= 45 and
                last_vol_ratio >= 1.5 and first_close <= vwap * 1.003):
            breakouts.append({
                'type':   'ORB_VWAP',
                'icon':   '💛',
                'title':  'VWAP Reclaim Breakout',
                'msg':    f"💛 VWAP RECLAIM — {sym} @ ₹{price:,.2f} | Crossed VWAP ₹{vwap:,.2f} in first {mins_open} min · Vol {last_vol_ratio:.1f}×",
                'score':  78,
                'color':  '#d97706',
                'bg':     '#fffbeb',
                'action': 'BUY — Strongest intraday entry. Stop below VWAP.',
            })

        # Rule 4 — ORB High Breakout (price breaks first candle high, within 90 min)
        if (last_close > first_high * 1.001 and last_vol_ratio >= 1.2 and mins_open <= 90):
            orb_move = (last_close - first_high) / first_high * 100
            breakouts.append({
                'type':   'ORB_HIGH',
                'icon':   '🔓',
                'title':  'ORB High Breakout',
                'msg':    f"🔓 ORB BREAK — {sym} @ ₹{price:,.2f} | Above first candle high ₹{first_high:,.2f} · +{orb_move:.1f}% from ORB · Vol {last_vol_ratio:.1f}×",
                'score':  75,
                'color':  '#0369a1',
                'bg':     '#f0f9ff',
                'action': 'BUY — Classic ORB. Stop = first candle low ₹' + f"{first_low:,.2f}",
            })

        # Rule 5 — Momentum Burst (3 consecutive bull candles + volume)
        if len(today_df) >= 3:
            last3       = today_df.tail(3)
            all_bull    = all(float(r['Close']) > float(r['Open']) for _, r in last3.iterrows())
            vr3         = float(last3['Volume'].mean()) / vol_ma
            if all_bull and vr3 >= 1.2 and chg_pct >= 0.5:
                breakouts.append({
                    'type':   'ORB_MOMENTUM',
                    'icon':   '⚡',
                    'title':  'Momentum Burst',
                    'msg':    f"⚡ MOMENTUM — {sym} @ ₹{price:,.2f} | 3 bull candles · Vol {vr3:.1f}× · {chg_pct:+.1f}%",
                    'score':  70,
                    'color':  '#c2410c',
                    'bg':     '#fff7ed',
                    'action': 'WATCH → BUY on next 1-candle pullback',
                })

    except Exception:
        pass

    return breakouts


def run_breakout_screener(selected_stocks, interval, kite, port):
    """
    Fast dedicated breakout scan — no signal scoring, just 5 ORB rules.
    Always uses 5-min data regardless of scanner timeframe setting.
    Reason: 5-min first candle = genuine opening range (5 min of price discovery)
            1-min first candle = just 60 seconds of noise → many fake breakouts
    Reuses cached data so much faster on second run.
    """
    # ORB always uses 5-min — hardcoded, not from scanner sidebar
    _orb_interval = '5minute'
    results = []
    total   = len(selected_stocks)
    _prog   = st.progress(0, text="🚀 Running Breakout Screener...")
    _stat   = st.empty()

    for idx, symbol in enumerate(selected_stocks):
        pct       = int(((idx + 1) / total) * 100)
        sym_clean = symbol.replace('.NS', '')
        _prog.progress(pct, text=f"🚀 {idx+1}/{total} · {sym_clean}")

        try:
            _ck = _cache_key(symbol, _orb_interval)
            if _ck in _DATA_CACHE:
                df, src = _DATA_CACHE[_ck]
            else:
                df, src = fetch_intraday(symbol, _orb_interval, period='1d', kite=kite)
                if df is None:
                    continue

            if 'VWAP' not in df.columns:
                df = calculate_intraday_indicators(df)

            latest     = df.iloc[-1]
            prev       = df.iloc[-2]
            price      = float(latest['Close'])
            prev_close = float(prev['Close'])
            vol_ratio  = float(latest.get('Volume_Ratio', 1.0)) if not pd.isna(latest.get('Volume_Ratio', np.nan)) else 1.0

            bos = detect_opening_breakout(df, symbol, price, prev_close)
            if bos:
                for bo in bos:
                    _add_alert(sym_clean, bo['type'], bo['msg'], price, bo['icon'])
                best = max(bos, key=lambda x: x['score'])
                # Extract first candle data for accurate SL/target calculation
                try:
                    import pytz as _ptz_orb
                    _ist_orb   = _ptz_orb.timezone('Asia/Kolkata')
                    _today_orb = datetime.now(_ist_orb).date()
                    _idx_orb   = pd.to_datetime(df.index)
                    if _idx_orb.tzinfo is None:
                        _idx_orb = _idx_orb.tz_localize('UTC').tz_convert('Asia/Kolkata')
                    else:
                        _idx_orb = _idx_orb.tz_convert('Asia/Kolkata')
                    _today_orb_df = df[_idx_orb.date == _today_orb]
                    _first_orb    = _today_orb_df.iloc[0] if len(_today_orb_df) > 0 else df.iloc[-1]
                    _first_low    = float(_first_orb['Low'])
                    _first_high   = float(_first_orb['High'])
                    _orb_rng      = round(_first_high - _first_low, 2)
                except Exception:
                    _first_low  = round(price * 0.995, 2)
                    _first_high = round(price * 1.005, 2)
                    _orb_rng    = round(price * 0.005, 2)

                results.append({
                    'symbol':    symbol,
                    'sym_clean': sym_clean,
                    'price':     price,
                    'prev_close':prev_close,
                    'chg_pct':   round((price - prev_close) / prev_close * 100, 2),
                    'vol_ratio': round(vol_ratio, 1),
                    'breakouts': bos,
                    'best':      best,
                    'df':        df,
                    'src':       src,
                    'first_low':  _first_low,
                    'first_high': _first_high,
                    'orb_range':  _orb_rng,
                })

            if (idx + 1) % 25 == 0:
                _stat.markdown(
                    f"<div style='font-size:12px;color:#64748b;padding:4px 0'>"
                    f"🚀 {len(results)} breakouts · {idx+1}/{total} scanned</div>",
                    unsafe_allow_html=True)
        except Exception:
            continue

    _prog.empty()
    _stat.empty()
    results.sort(key=lambda x: x['best']['score'], reverse=True)
    return results

# ─────────────────────────────────────────────
#  PORTFOLIO PERSISTENCE (Intraday — daily reset)
# ─────────────────────────────────────────────

PORTFOLIO_FILE   = pathlib.Path.home() / "Downloads" / "investo_intraday_portfolio.json"
SCAN_HISTORY_FILE = pathlib.Path.home() / "Downloads" / "investo_scan_history.csv"
PORTFOLIO_FILE.parent.mkdir(parents=True, exist_ok=True)

IST = pytz.timezone("Asia/Kolkata")

def ist_now():
    return datetime.now(IST)


# ── NSE Holiday Calendar ───────────────────────────────
# Primary source: pandas_market_calendars (accurate, auto-updated)
# Fallback: hardcoded list (used if library not installed)
def get_nse_holidays() -> dict:
    """
    Fetch NSE trading holidays directly from NSE's official API.
    Returns dict: {'YYYY-MM-DD': 'Holiday Name', ...}

    NSE endpoint returns JSON with tradingHolidays list for current year.
    Cached in st.session_state for the day — only fetches once per session.
    Falls back to pandas_market_calendars if NSE API fails.
    Falls back to empty dict if both fail (app still works, just no holiday detection).
    """
    _cache_key = f"nse_holidays_{ist_now().strftime('%Y')}"

    # Return cached result if already fetched this session
    if _cache_key in st.session_state:
        return st.session_state[_cache_key]

    _holidays = {}

    # ── Method 1: NSE official API ────────────────────────
    try:
        import requests
        _year = ist_now().year
        # NSE API — market holidays endpoint
        _headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Referer': 'https://www.nseindia.com/',
        }
        # First hit NSE homepage to get session cookies
        _session = requests.Session()
        _session.get('https://www.nseindia.com', headers=_headers, timeout=10)

        # Now fetch holiday list
        _url = f'https://www.nseindia.com/api/holiday-master?type=trading'
        _resp = _session.get(_url, headers=_headers, timeout=10)

        if _resp.status_code == 200:
            _data = _resp.json()
            # NSE returns: {"CM": [...], "FO": [...], ...}
            # CM = Capital Markets (equity) holidays
            _cm_holidays = _data.get('CM', [])
            for _h in _cm_holidays:
                # Each entry: {"tradingDate": "26-Mar-2026", "weekDay": "Thursday",
                #              "description": "Shri Ram Navami", "Sr_no": "4"}
                _raw_date = _h.get('tradingDate', '')
                _desc     = _h.get('description', 'Holiday')
                try:
                    from datetime import datetime as _dt
                    _parsed = _dt.strptime(_raw_date, '%d-%b-%Y')
                    _date_str = _parsed.strftime('%Y-%m-%d')
                    _holidays[_date_str] = _desc
                except Exception:
                    pass

    except Exception:
        pass

    # ── Method 2: pandas_market_calendars (if NSE API failed) ─
    if not _holidays:
        try:
            import pandas_market_calendars as mcal
            import pandas as pd
            _nse       = mcal.get_calendar('NSE')
            _start     = f'{ist_now().year}-01-01'
            _end       = f'{ist_now().year + 1}-12-31'
            _all_bdays = pd.bdate_range(_start, _end)
            _sched     = _nse.schedule(start_date=_start, end_date=_end)
            _open_days = pd.DatetimeIndex(_sched.index)
            _hol_dates = _all_bdays.difference(_open_days)
            for _d in _hol_dates:
                _holidays[_d.strftime('%Y-%m-%d')] = 'NSE Holiday'
        except Exception:
            pass

    # Cache result for this session
    st.session_state[_cache_key] = _holidays
    return _holidays


def is_nse_holiday(date_str: str = None) -> tuple:
    """
    Returns (is_holiday: bool, holiday_name: str).
    date_str format: 'YYYY-MM-DD'. Defaults to today IST.
    """
    if date_str is None:
        date_str = ist_now().strftime('%Y-%m-%d')
    _holidays = get_nse_holidays()
    _name     = _holidays.get(date_str)
    return (_name is not None), (_name or '')


def market_open() -> bool:
    """True if NSE is currently open — weekday, not a holiday, 9:15–15:30 IST."""
    now = ist_now()
    if now.weekday() >= 5:
        return False
    _is_hol, _ = is_nse_holiday(now.strftime('%Y-%m-%d'))
    if _is_hol:
        return False
    t = now.time()
    from datetime import time as _t
    return _t(9, 15) <= t <= _t(15, 30)



def detect_expiry(now=None):
    """
    Detect if today is an NSE options expiry day.
    Returns dict with expiry type, rules, and trading guidance.
    Nifty weekly  = every Thursday
    Bank Nifty    = every Wednesday
    Monthly       = last Thursday of month
    """
    from calendar import monthcalendar
    if now is None:
        now = ist_now()
    _wd = now.weekday()   # 0=Mon … 6=Sun
    _d  = now.day
    _m  = now.month
    _y  = now.year

    _nifty_exp  = (_wd == 3)   # Thursday
    _bnifty_exp = (_wd == 2)   # Wednesday
    _is_monthly = False

    if _nifty_exp:
        _cal   = monthcalendar(_y, _m)
        _thurs = [w[3] for w in _cal if w[3] != 0]
        _is_monthly = (_d == _thurs[-1])

    if _nifty_exp:
        _exp_type = 'NIFTY_MONTHLY' if _is_monthly else 'NIFTY_WEEKLY'
    elif _bnifty_exp:
        _exp_type = 'BANKNIFTY_WEEKLY'
    else:
        _exp_type = None

    _is_expiry = _exp_type is not None

    if not _is_expiry:
        return {
            'is_expiry': False, 'expiry_type': None,
            'is_monthly': False, 'expiry_label': '',
            'best_entry_time': '9:35 AM – 2:30 PM',
            'exit_time': '3:15 PM',
            'min_candles_confirm': 1,
            'gap_fill_prob': 30,
            'target_multiplier': 1.5,
        }

    _label = {
        'NIFTY_WEEKLY':    '📅 Nifty Weekly Expiry (Thursday)',
        'NIFTY_MONTHLY':   '📅 Nifty MONTHLY Expiry (Last Thursday) — Most Volatile',
        'BANKNIFTY_WEEKLY':'📅 Bank Nifty Weekly Expiry (Wednesday)',
    }.get(_exp_type, '📅 Expiry Day')

    return {
        'is_expiry':           _is_expiry,
        'expiry_type':         _exp_type,
        'is_monthly':          _is_monthly,
        'expiry_label':        _label,
        'best_entry_time':     '10:00 AM – 10:30 AM  or  1:30 PM – 2:30 PM',
        'exit_time':           '2:30 PM',
        'min_candles_confirm': 3,
        'gap_fill_prob':       65,
        'target_multiplier':   0.5,
    }


def load_portfolio() -> list:
    try:
        if PORTFOLIO_FILE.exists():
            return json.loads(PORTFOLIO_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return []

def save_portfolio(port: list) -> None:
    try:
        PORTFOLIO_FILE.write_text(
            json.dumps(port, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
    except Exception as e:
        st.warning(f"⚠️ Could not save portfolio: {e}")


def save_scan_history(results: list, interval: str, nifty_state: str, vix) -> None:
    """
    Auto-save every scan result to CSV for ML training data.
    Appends rows — never overwrites. File: ~/Downloads/investo_scan_history.csv
    Columns: timestamp, symbol, interval, score, verdict, price, change_pct,
             rsi, vwap, vol_ratio, cpr_width, rs_vs_nifty, sector,
             nifty_state, vix, gap_pct, warmup
    """
    try:
        import csv as _csv
        _cols = [
            'timestamp','symbol','interval','score','verdict','price',
            'change_pct','rsi','vwap','vol_ratio','cpr_width',
            'rs_vs_nifty','sector','nifty_state','vix','gap_pct','warmup',
        ]
        _exists = SCAN_HISTORY_FILE.exists()
        with open(SCAN_HISTORY_FILE, 'a', newline='', encoding='utf-8') as _fh:
            _w = _csv.DictWriter(_fh, fieldnames=_cols, extrasaction='ignore')
            if not _exists:
                _w.writeheader()
            _ts = ist_now().strftime('%Y-%m-%d %H:%M')
            for r in results:
                _w.writerow({
                    'timestamp':   _ts,
                    'symbol':      r.get('symbol','').replace('.NS',''),
                    'interval':    interval,
                    'score':       r.get('_pick_score', 0),
                    'verdict':     r.get('_verdict', ''),
                    'price':       round(float(r.get('price', 0)), 2),
                    'change_pct':  round(float(r.get('change_pct', 0)), 2),
                    'rsi':         round(float(r.get('rsi', 0)), 1),
                    'vwap':        r.get('vwap', ''),
                    'vol_ratio':   round(float(r.get('vol_ratio', 0)), 2),
                    'cpr_width':   round(float(r.get('cpr_width', 0)), 3) if r.get('cpr_width') else '',
                    'rs_vs_nifty': round(float(r.get('rs_vs_nifty', 0)), 2) if r.get('rs_vs_nifty') is not None else '',
                    'sector':      r.get('sector', ''),
                    'nifty_state': nifty_state,
                    'vix':         round(float(vix), 2) if vix else '',
                    'gap_pct':     round(float(r.get('gap_pct', 0)), 2),
                    'warmup':      r.get('warmup', ''),
                })
    except Exception:
        pass   # never crash the app for logging


def _f(v, fallback=0.0):
    try:
        return float(v) if v is not None else float(fallback)
    except Exception:
        return float(fallback)

def _safe_get(d, key, fallback=0.0):
    return _f(d.get(key, fallback), fallback)


# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="NSE Intraday Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sanitize stale auto-sell session state values ─────────
# Clamp any cached tp/sl pct values that are below widget minimum
for _k in list(st.session_state.keys()):
    if _k.startswith('autosell_pct_') and isinstance(st.session_state[_k], (int, float)):
        if st.session_state[_k] < 0.1:
            st.session_state[_k] = 0.1

st.session_state['paper_portfolio'] = load_portfolio()

# ─────────────────────────────────────────────
#  CSS — Dark sidebar + clean cards
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
    background: #f0f2f7;
}

/* ════════════════════════════════════════════
   SIDEBAR — Clean dark design matching body
   ════════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: #0f172a !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
    padding-top: 0 !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stSlider > div,
[data-testid="stSidebar"] .stNumberInput > div { background: #1e293b !important; border-color: #334155 !important; }
[data-testid="stSidebar"] label { color: #64748b !important; font-size:11px !important; font-weight:600 !important; letter-spacing:0.8px !important; text-transform:uppercase !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stRadio label { color: #64748b !important; font-size:11px !important; }

/* ── Logo bar ── */
.sb-logo {
    display:flex; align-items:center; gap:12px;
    padding:20px 20px 16px 20px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.sb-logo-icon {
    background: linear-gradient(135deg,#f59e0b,#d97706);
    border-radius:12px; width:38px; height:38px;
    display:flex; align-items:center; justify-content:center; flex-shrink:0;
    box-shadow: 0 4px 12px rgba(245,158,11,0.3);
}
.sb-logo-name { font-size:18px; font-weight:800; color:#ffffff !important; letter-spacing:-0.5px; line-height:1.1; }
.sb-logo-tag  { font-size:9px;  font-weight:700; color:#f59e0b !important; letter-spacing:2px; text-transform:uppercase; }

/* ── Market status strip ── */
.sb-market-strip {
    display:flex; align-items:center; justify-content:space-between;
    padding:10px 20px; background:rgba(255,255,255,0.03);
    border-bottom:1px solid rgba(255,255,255,0.06);
}
.sb-mkt-open   { display:flex;align-items:center;gap:6px;font-size:11px;font-weight:700;color:#34d399 !important; }
.sb-mkt-closed { display:flex;align-items:center;gap:6px;font-size:11px;font-weight:700;color:#f87171 !important; }
.sb-mkt-dot-open   { width:7px;height:7px;border-radius:50%;background:#34d399;animation:pulse-green 2s infinite; }
.sb-mkt-dot-closed { width:7px;height:7px;border-radius:50%;background:#f87171; }
@keyframes pulse-green { 0%,100%{opacity:1} 50%{opacity:0.4} }
.sb-mkt-time { font-size:10px; color:#475569 !important; font-weight:500; }

/* ── Portfolio strip ── */
.sb-port-strip {
    padding:14px 20px;
    border-bottom:1px solid rgba(255,255,255,0.06);
}
.sb-port-label { font-size:10px;font-weight:700;color:#475569 !important;letter-spacing:1px;text-transform:uppercase; }
.sb-port-row   { display:flex;align-items:flex-end;justify-content:space-between;margin-top:4px; }
.sb-port-val   { font-size:20px;font-weight:800;color:#ffffff !important;font-family:'JetBrains Mono',monospace; }
.sb-port-pnl-pos { font-size:13px;font-weight:700;color:#34d399 !important; }
.sb-port-pnl-neg { font-size:13px;font-weight:700;color:#f87171 !important; }

/* ── Nav items ── */
.sb-nav-section {
    padding:16px 20px 6px 20px;
    font-size:9px; font-weight:700; color:#334155 !important;
    letter-spacing:2px; text-transform:uppercase;
}
.sb-nav-item {
    display:flex; align-items:center; gap:12px;
    padding:10px 20px; margin:2px 8px;
    border-radius:10px; cursor:pointer;
    transition:background 0.15s;
}
.sb-nav-item:hover   { background:rgba(255,255,255,0.05); }
.sb-nav-item.active  { background:rgba(245,158,11,0.12); border:1px solid rgba(245,158,11,0.2); }
.sb-nav-icon         { display:flex;align-items:center;flex-shrink:0;opacity:0.5; }
.sb-nav-icon.active  { opacity:1; }
.sb-nav-text         { font-size:13px;font-weight:500;color:#94a3b8 !important;flex:1; }
.sb-nav-text.active  { font-weight:700;color:#f59e0b !important; }
.sb-nav-badge {
    background:#ef4444;color:#ffffff !important;
    font-size:10px;font-weight:800;border-radius:20px;
    padding:1px 7px;min-width:18px;text-align:center;
}

/* ── Config section ── */
.sb-section-divider {
    margin:8px 20px;
    border:none;border-top:1px solid rgba(255,255,255,0.06);
}
.sb-section-label {
    padding:12px 20px 6px;
    font-size:9px;font-weight:700;color:#334155 !important;
    letter-spacing:2px;text-transform:uppercase;
}

/* ── Kite status pill ── */
.sb-kite-connected {
    margin:8px 16px;padding:10px 14px;
    background:rgba(52,211,153,0.1);
    border:1px solid rgba(52,211,153,0.2);
    border-radius:10px;
}
.sb-kite-disconnected {
    margin:8px 16px;padding:10px 14px;
    background:rgba(255,255,255,0.03);
    border:1px solid rgba(255,255,255,0.08);
    border-radius:10px;
}
.sb-kite-label { font-size:11px;font-weight:700; }
.sb-kite-sub   { font-size:10px;margin-top:2px;opacity:0.7; }

/* ── Data source badge ── */
.sb-datasrc {
    margin:0 16px 8px;padding:8px 12px;
    border-radius:8px;font-size:10px;font-weight:700;
}

/* ── Alert item in sidebar ── */
.sb-alert-item {
    margin:0 12px 5px;padding:9px 12px;
    border-radius:9px;
}
.sb-alert-sym  { font-size:12px;font-weight:700; }
.sb-alert-msg  { font-size:10px;opacity:0.65;margin-top:2px;line-height:1.35; }
.sb-alert-time { font-size:9px;opacity:0.4;margin-top:3px; }

/* ── Disclaimer ── */
.sb-disclaimer {
    padding:12px 20px 20px;
    font-size:10px;color:#334155 !important;
    text-align:center;line-height:1.6;
    border-top:1px solid rgba(255,255,255,0.06);
    margin-top:8px;
}

/* ── Hide Streamlit ghost buttons & radio ── */
div[data-testid="stSidebar"] .stRadio { display:none !important; }
div[data-testid="stSidebar"] .stButton > button {
    background:transparent !important;border:none !important;
    color:transparent !important;height:0 !important;padding:0 !important;
    margin:0 !important;position:absolute !important;pointer-events:none !important;
}

/* ════════════════════════════════════════════
   BODY / MAIN AREA
   ════════════════════════════════════════════ */

/* ── Scrollbar ── */
::-webkit-scrollbar { width:6px; }
::-webkit-scrollbar-track { background:#f1f5f9; }
::-webkit-scrollbar-thumb { background:#cbd5e1;border-radius:3px; }

/* ── Topbar ── */
.topbar {
    background:linear-gradient(135deg,#1a2035 0%,#2d3561 100%);
    border-radius:16px;padding:18px 28px;margin-bottom:20px;
    display:flex;align-items:center;justify-content:space-between;
    box-shadow:0 4px 20px rgba(26,32,53,0.15);
}
.topbar-title    { font-size:22px;font-weight:800;color:#ffffff;display:flex;align-items:center;gap:10px; }
.topbar-subtitle { font-size:13px;color:rgba(255,255,255,0.6);margin-top:4px; }
.topbar-right    { display:flex;align-items:center;gap:10px; }
.topbar-badge    { background:rgba(245,158,11,0.2);color:#f59e0b;border:1px solid rgba(245,158,11,0.4);
    border-radius:20px;padding:4px 14px;font-size:12px;font-weight:700; }
.topbar-time        { color:#34d399;font-size:13px;font-weight:700; }
.topbar-time-closed { color:#f87171;font-size:13px;font-weight:700; }
.timeframe-pill {
    background:#f59e0b;color:#1a2035;border-radius:20px;
    padding:3px 12px;font-size:11px;font-weight:800;letter-spacing:0.5px;
}

/* ── Stat cards ── */
.stat-card {
    background:#ffffff;border:1px solid #e8ecf3;border-radius:16px;
    padding:18px 20px;margin-bottom:12px;
    box-shadow:0 1px 4px rgba(0,0,0,0.04);
}
.stat-card-icon { width:40px;height:40px;border-radius:12px;
    display:flex;align-items:center;justify-content:center;margin-bottom:10px; }
.stat-label { font-size:11px;font-weight:700;color:#94a3b8;letter-spacing:1px;text-transform:uppercase;margin-bottom:4px; }
.stat-value { font-size:28px;font-weight:800;color:#1a2035;font-family:'JetBrains Mono',monospace; }
.stat-sub   { font-size:12px;color:#94a3b8;margin-top:2px; }
.stat-green { color:#16a34a !important; }
.stat-amber { color:#d97706 !important; }
.stat-red   { color:#dc2626 !important; }

/* ── Signal cards ── */
.signal-buy  { background:#f0fdf4;border:1px solid #bbf7d0;border-radius:12px;padding:14px 18px; }
.signal-sell { background:#fff5f5;border:1px solid #fecaca;border-radius:12px;padding:14px 18px; }
.signal-none { background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:14px 18px; }

/* ── Metric card ── */
.metric-card { background:#f8fafc;border:1px solid #e8ecf3;border-radius:12px;padding:14px 18px; }
.metric-label { font-size:11px;font-weight:700;color:#94a3b8;letter-spacing:1px;text-transform:uppercase; }
.conf-bar-bg   { background:#e8ecf3;border-radius:4px;height:8px;width:100%;overflow:hidden; }
.conf-bar-fill { height:8px;border-radius:4px;transition:width 0.5s ease; }

/* ── Score badges ── */
.score-badge    { padding:3px 10px;border-radius:20px;font-size:11px;font-weight:700;letter-spacing:0.5px; }
.badge-strong   { background:#dcfce7;color:#15803d; }
.badge-good     { background:#d1fae5;color:#065f46; }
.badge-moderate { background:#fef3c7;color:#92400e; }
.badge-weak     { background:#fee2e2;color:#991b1b; }
.badge-none     { background:#f1f5f9;color:#64748b; }

/* ── Section header ── */
.section-header {
    font-size:13px;font-weight:700;color:#1a2035;
    padding:6px 0;margin:16px 0 10px 0;
    border-bottom:2px solid #e8ecf3;
    display:flex;align-items:center;gap:8px;
}

/* ── Verdict pills ── */
.verdict-strong { color:#15803d;background:#dcfce7;border-radius:8px;padding:3px 10px;font-weight:700;font-size:12px; }
.verdict-buy    { color:#1d4ed8;background:#dbeafe;border-radius:8px;padding:3px 10px;font-weight:700;font-size:12px; }
.verdict-watch  { color:#92400e;background:#fef3c7;border-radius:8px;padding:3px 10px;font-weight:700;font-size:12px; }
.verdict-neutral{ color:#64748b;background:#f1f5f9;border-radius:8px;padding:3px 10px;font-weight:700;font-size:12px; }
.verdict-avoid  { color:#991b1b;background:#fee2e2;border-radius:8px;padding:3px 10px;font-weight:700;font-size:12px; }

/* ── Portfolio card ── */
.port-card { background:#ffffff;border:1.5px solid #e8ecf3;border-radius:16px;
    padding:20px 22px;margin-bottom:14px;box-shadow:0 2px 8px rgba(0,0,0,0.04); }

/* ── Alert banner ── */
@keyframes slideIn { from{opacity:0;transform:translateY(-8px)} to{opacity:1;transform:translateY(0)} }
.alert-banner { border-radius:12px;padding:12px 18px;margin-bottom:8px;
    display:flex;align-items:center;gap:14px;animation:slideIn 0.3s ease; }

/* ── Intraday VWAP pill ── */
.vwap-above { color:#16a34a;background:#f0fdf4;border-radius:6px;padding:2px 8px;font-size:11px;font-weight:700; }
.vwap-below { color:#dc2626;background:#fff5f5;border-radius:6px;padding:2px 8px;font-size:11px;font-weight:700; }

/* ── Intraday target row ── */
.intraday-target { background:#ffffff;border:1px solid #e8ecf3;border-radius:12px;
    padding:14px 18px;margin-bottom:8px;display:flex;align-items:center;justify-content:space-between; }

/* ── Market open/closed (legacy) ── */
.market-open   { background:#064e3b;color:#34d399;border:1px solid #065f46;border-radius:20px;padding:3px 12px;font-size:11px;font-weight:700; }
.market-closed { background:#450a0a;color:#fca5a5;border:1px solid #7f1d1d;border-radius:20px;padding:3px 12px;font-size:11px;font-weight:700; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  F&O STOCK LIST + EXPIRY FUNCTIONS
#  ~180 NSE F&O listed stocks
#  Used to detect expiry pinning risk
#  Near expiry: F&O stocks get penalty
#  Non-F&O stocks get bonus
# ─────────────────────────────────────────────

FNO_STOCKS = {
    # Nifty 50
    'ADANIENT','ADANIPORTS','APOLLOHOSP','ASIANPAINT','AXISBANK',
    'BAJAJ-AUTO','BAJFINANCE','BAJAJFINSV','BPCL','BHARTIARTL',
    'BRITANNIA','CIPLA','COALINDIA','DIVISLAB','DRREDDY',
    'EICHERMOT','GRASIM','HCLTECH','HDFCBANK','HDFCLIFE',
    'HEROMOTOCO','HINDALCO','HINDUNILVR','ICICIBANK','ITC',
    'INDUSINDBK','INFY','JSWSTEEL','KOTAKBANK','LT',
    'LTIM','M&M','MARUTI','NESTLEIND','NTPC',
    'ONGC','POWERGRID','RELIANCE','SBILIFE','SBIN',
    'SUNPHARMA','TCS','TATACONSUM','TATAMOTORS','TATASTEEL',
    'TECHM','TITAN','TRENT','ULTRACEMCO','WIPRO',
    # Nifty Next 50 + popular F&O
    'ABB','ABBOTINDIA','ABCAPITAL','ABFRL','ACC',
    'ADANIENSOL','ADANIGREEN','ADANIPOWER','ALKEM','AMBUJACEM',
    'AUROPHARMA','BANDHANBNK','BANKBARODA','BEL','BERGEPAINT',
    'BIOCON','BOSCHLTD','CANBK','CHOLAFIN','COLPAL',
    'CONCOR','COROMANDEL','CROMPTON','CUMMINSIND','DLF',
    'DABUR','DALBHARAT','DEEPAKNTR','ESCORTS','FEDERALBNK',
    'GAIL','GODREJCP','GODREJPROP','GRANULES','HAVELLS',
    'HAL','HFCL','HINDCOPPER','HINDZINC','IDFCFIRSTB',
    'INDHOTEL','INDIAMART','INDUSTOWER','IRCTC','IRFC',
    'JINDALSTEL','JUBLFOOD','L&TFH','LALPATHLAB','LICHSGFIN',
    'LUPIN','MANAPPURAM','MARICO','MCDOWELL-N','MCX',
    'MFSL','MGL','MOTHERSON','MPHASIS','MRF',
    'MUTHOOTFIN','NAUKRI','NMDC','OBEROIRLTY','OFSS',
    'PAGEIND','PEL','PETRONET','PFC','PIDILITIND',
    'PIIND','PNB','POLYCAB','PVR','RAMCOCEM',
    'RBLBANK','RECLTD','SAIL','SHREECEM','SIEMENS',
    'SRF','SUNPHARMA','SUNTV','SUZLON','TATACOMM',
    'TATAELXSI','TATAINVEST','TATAPOWER','TTML','UBL',
    'UNIONBANK','UPL','VEDL','VOLTAS','WHIRLPOOL',
    'ZEEL','ZOMATO','NYKAA','PAYTM','DMART',
    'POLICYBZR','INDIGO','SPICEJET','IDEALREALTY','LODHA',
    'PRESTIGE','SOBHA','PHOENIXLTD','BRIGADE','MAHLIFE',
    'AARTIIND','ATUL','CASTROLIND','CHAMBLFERT','COFORGE',
    'CYIENT','DIXON','EMAMILTD','FINEORG','GLENMARK',
    'GNFC','GSPL','HAPPSTMNDS','IPCALAB','JKCEMENT',
    'JUBLINGREA','KPITTECH','LAURUSLABS','METROPOLIS','NAM-INDIA',
    'NATCOPHARM','NAVINFLUOR','NLCINDIA','PERSISTENT','PFIZER',
    'RAIN','ROUTE','SEQUENT','SOLARINDS','STAR',
    'SYNGENE','THERMAX','TTKPRESTIG','VGUARD','VBL',
    'ZYDUSLIFE','360ONE','ASTRAL','AARVIIND','CESC',
}

# ─────────────────────────────────────────────────────────────
#  DYNAMIC F&O LIST — overrides FNO_STOCKS if user uploads a CSV
#  Saved to disk so it persists across app restarts.
#  Falls back to the hardcoded FNO_STOCKS above if no upload yet.
# ─────────────────────────────────────────────────────────────
import os as _os_fno
_FNO_CUSTOM_PATH = _os_fno.path.join(
    _os_fno.path.dirname(_os_fno.path.abspath(__file__)),
    'fno_stocks_custom.csv')


def load_custom_fno_list():
    """
    Loads the F&O symbol set to use this session.
    Priority:
      1. Already cached in session_state (avoid re-reading disk)
      2. Saved custom CSV on disk (from a previous upload)
      3. Hardcoded FNO_STOCKS (built-in default)

    Returns: (set_of_symbols, source_label)
    """
    import streamlit as _st_fno
    if 'fno_stocks_active' in _st_fno.session_state:
        return (_st_fno.session_state['fno_stocks_active'],
                _st_fno.session_state.get('fno_stocks_source', 'built-in'))

    if _os_fno.path.exists(_FNO_CUSTOM_PATH):
        try:
            import pandas as _pd_fno
            _df = _pd_fno.read_csv(_FNO_CUSTOM_PATH)
            _cols_lower = {c.lower().strip(): c for c in _df.columns}
            _sym_col = _cols_lower.get('symbol', _df.columns[0])
            _raw = _df[_sym_col].dropna().astype(str).tolist()
            _clean = set()
            for _s in _raw:
                _s = _s.strip().upper().replace('.NS', '').replace('NSE:', '')
                if _s and _s not in ('SYMBOL', 'SYMBOLS'):
                    _clean.add(_s)
            if _clean:
                _st_fno.session_state['fno_stocks_active'] = _clean
                _st_fno.session_state['fno_stocks_source'] = 'custom CSV (saved)'
                return _clean, 'custom CSV (saved)'
        except Exception:
            pass

    _st_fno.session_state['fno_stocks_active'] = FNO_STOCKS
    _st_fno.session_state['fno_stocks_source'] = 'built-in'
    return FNO_STOCKS, 'built-in'


def save_custom_fno_list(uploaded_file):
    """
    Saves an uploaded F&O CSV to disk (persists across restarts)
    and updates the active session list immediately.

    Returns: (success: bool, message: str, count: int)
    """
    import streamlit as _st_fno
    try:
        _symbols, _err = parse_csv_stock_list(uploaded_file)
        if _err:
            return False, _err, 0
        # Strip .NS suffix for storage (FNO_STOCKS format has no suffix)
        _clean = {s.replace('.NS', '') for s in _symbols}

        # Save to disk as simple CSV for persistence
        import pandas as _pd_fno
        _pd_fno.DataFrame({'Symbol': sorted(_clean)}).to_csv(
            _FNO_CUSTOM_PATH, index=False)

        # Update active session immediately
        _st_fno.session_state['fno_stocks_active'] = _clean
        _st_fno.session_state['fno_stocks_source'] = 'custom CSV (saved)'
        return True, f"Saved {len(_clean)} F&O symbols", len(_clean)
    except Exception as _e:
        return False, f"Could not save: {str(_e)[:100]}", 0


def reset_fno_list_to_default():
    """Removes custom F&O list — reverts to built-in FNO_STOCKS."""
    import streamlit as _st_fno
    try:
        if _os_fno.path.exists(_FNO_CUSTOM_PATH):
            _os_fno.remove(_FNO_CUSTOM_PATH)
    except Exception:
        pass
    _st_fno.session_state['fno_stocks_active'] = FNO_STOCKS
    _st_fno.session_state['fno_stocks_source'] = 'built-in'


def get_monthly_expiry(dt=None):
    """
    Returns the last Thursday of the current/next month.
    If today is past this month's expiry, returns next month's.
    """
    import calendar
    from datetime import date, timedelta

    if dt is None:
        dt = ist_now().date()

    # Find last Thursday of current month
    year, month = dt.year, dt.month
    last_day = calendar.monthrange(year, month)[1]
    last_date = date(year, month, last_day)

    # Walk back to Thursday (weekday 3)
    while last_date.weekday() != 3:
        last_date -= timedelta(days=1)

    # If today is past this month's expiry use next month
    if dt > last_date:
        if month == 12:
            year, month = year + 1, 1
        else:
            month += 1
        last_day = calendar.monthrange(year, month)[1]
        last_date = date(year, month, last_day)
        while last_date.weekday() != 3:
            last_date -= timedelta(days=1)

    return last_date


def days_to_expiry():
    """Returns calendar days to next monthly expiry."""
    from datetime import date
    today    = ist_now().date()
    expiry   = get_monthly_expiry(today)
    return (expiry - today).days


def get_expiry_zone(dte=None):
    """
    Classify the expiry proximity zone.

    FRESH   (post expiry 0-3 days) → best entry window
    SAFE    (≥ 15 days)            → enter freely
    CAUTION (8-14 days)            → prefer non-F&O
    DANGER  (1-7 days)             → avoid F&O stocks
    """
    if dte is None:
        dte = days_to_expiry()

    if   dte <= 0:  return 'FRESH'    # post expiry
    elif dte <= 7:  return 'DANGER'   # expiry week
    elif dte <= 14: return 'CAUTION'  # second half
    else:           return 'SAFE'     # first half


def get_fno_info(sym_clean, dte=None):
    """
    Returns F&O status, zone and confident score penalty.
    sym_clean = symbol without .NS suffix

    Returns dict:
        is_fno, expiry_zone, days_to_exp,
        fno_penalty, fno_badge, fno_clr,
        fno_bg, fno_bdr, fno_note
    """
    if dte is None:
        dte = days_to_expiry()

    _fno_set, _ = load_custom_fno_list()
    is_fno = sym_clean.upper() in _fno_set
    zone   = get_expiry_zone(dte)

    # ── Penalty / Bonus ───────────────────────────
    if is_fno:
        if   zone == 'DANGER':  penalty = -15  # expiry week
        elif zone == 'CAUTION': penalty = -8   # second half
        elif zone == 'FRESH':   penalty = +5   # post expiry
        else:                   penalty =  0   # safe zone
    else:
        # Non-F&O gets bonus near expiry (moves freely)
        if   zone == 'DANGER':  penalty = +10
        elif zone == 'CAUTION': penalty = +5
        elif zone == 'FRESH':   penalty = +5
        else:                   penalty =  0

    # ── Badge display ─────────────────────────────
    if is_fno:
        if   zone == 'DANGER':
            badge = f'📌 F&O · {dte}d to expiry'
            clr   = '#dc2626'; bg = '#fef2f2'; bdr = '#fca5a5'
            note  = '⚠️ Expiry week — price may be pinned'
        elif zone == 'CAUTION':
            badge = f'📌 F&O · {dte}d to expiry'
            clr   = '#d97706'; bg = '#fffbeb'; bdr = '#fde68a'
            note  = '⚠️ Second half — reduce size, prefer non-F&O'
        elif zone == 'FRESH':
            badge = f'📌 F&O · Post expiry'
            clr   = '#15803d'; bg = '#f0fdf4'; bdr = '#86efac'
            note  = '✅ Post expiry — fresh cycle, enter freely'
        else:
            badge = f'📌 F&O · {dte}d to expiry'
            clr   = '#1d4ed8'; bg = '#eff6ff'; bdr = '#93c5fd'
            note  = '✅ Safe zone — normal movement expected'
    else:
        if   zone in ('DANGER', 'CAUTION'):
            badge = f'✅ Non-F&O · {dte}d to expiry'
            clr   = '#15803d'; bg = '#f0fdf4'; bdr = '#86efac'
            note  = '✅ Not pinned by expiry — moves freely'
        elif zone == 'FRESH':
            badge = '✅ Non-F&O · Post expiry'
            clr   = '#15803d'; bg = '#f0fdf4'; bdr = '#86efac'
            note  = '✅ Best entry window — fresh cycle'
        else:
            badge = '✅ Non-F&O'
            clr   = '#64748b'; bg = '#f8fafc'; bdr = '#e2e8f0'
            note  = ''

    return {
        'is_fno':      is_fno,
        'expiry_zone': zone,
        'days_to_exp': dte,
        'fno_penalty': penalty,
        'fno_badge':   badge,
        'fno_clr':     clr,
        'fno_bg':      bg,
        'fno_bdr':     bdr,
        'fno_note':    note,
    }


# ─────────────────────────────────────────────
#  STOCK UNIVERSE
# ─────────────────────────────────────────────
# ── Tier 1: Nifty 50 — EXCELLENT liquidity, highest institutional activity ──
# Scanned first — best signals appear within 9 seconds
POPULAR_STOCKS = list([
    "HDFCBANK.NS", "RELIANCE.NS", "ICICIBANK.NS", "TCS.NS", "INFY.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "AXISBANK.NS",
    "SBIN.NS", "LT.NS", "HINDUNILVR.NS", "BAJFINANCE.NS", "ASIANPAINT.NS", "MARUTI.NS", "HCLTECH.NS", "SUNPHARMA.NS",
    "TITAN.NS", "WIPRO.NS", "NTPC.NS", "POWERGRID.NS", "ULTRACEMCO.NS", "TECHM.NS", "BAJAJFINSV.NS", "NESTLEIND.NS",
    "GRASIM.NS", "ADANIPORTS.NS", "COALINDIA.NS", "JSWSTEEL.NS", "TATASTEEL.NS", "ONGC.NS", "BPCL.NS", "HEROMOTOCO.NS",
    "APOLLOHOSP.NS", "CIPLA.NS", "DRREDDY.NS", "DIVISLAB.NS", "EICHERMOT.NS", "SHRIRAMFIN.NS", "TATACONSUM.NS", "ADANIENT.NS",
    "HINDALCO.NS", "INDUSINDBK.NS", "M&M.NS", "HDFCLIFE.NS", "SBICARD.NS", "BRITANNIA.NS", "BAJAJ-AUTO.NS", "TRENT.NS",
    "VEDL.NS", "ADANIPOWER.NS", "ADANIGREEN.NS", "RVNL.NS", "IRFC.NS", "HUDCO.NS", "HAL.NS", "BEL.NS",
    "BHEL.NS", "SAIL.NS", "RECLTD.NS", "PFC.NS", "SJVN.NS", "NHPC.NS", "SUZLON.NS", "YESBANK.NS",
    "IDEA.NS", "PNB.NS", "BANKBARODA.NS", "RBLBANK.NS", "FEDERALBNK.NS", "IDFCFIRSTB.NS", "AUBANK.NS", "BANDHANBNK.NS",
    "PERSISTENT.NS", "LTTS.NS", "COFORGE.NS", "MPHASIS.NS", "OFSS.NS", "KPITTECH.NS", "TATAELXSI.NS", "ATGL.NS",
    "IGL.NS", "MGL.NS", "GAIL.NS", "PETRONET.NS", "TATAPOWER.NS", "TORNTPOWER.NS", "DIXON.NS", "JIOFIN.NS",
    "ABCAPITAL.NS", "MUTHOOTFIN.NS", "NYKAA.NS", "PAYTM.NS", "ZYDUSLIFE.NS", "LUPIN.NS", "AUROPHARMA.NS", "LODHA.NS",
    "DLF.NS", "OBEROIRLTY.NS", "GRAPHITE.NS", "AARTIIND.NS", "PRESTIGE.NS", "BRIGADE.NS", "IRCTC.NS", "HDFCAMC.NS",
    "ASTRAL.NS", "BHARATFORG.NS", "BSE.NS", "CAMS.NS", "CANFINHOME.NS", "CDSL.NS", "CHOLAFIN.NS", "DELHIVERY.NS",
    "ELGIEQUIP.NS", "GRANULES.NS", "GSPL.NS", "INDHOTEL.NS", "INDIAMART.NS", "INDUSTOWER.NS", "JSWENERGY.NS", "JUBLFOOD.NS",
    "LICHSGFIN.NS", "M&MFIN.NS", "MAXHEALTH.NS", "MCX.NS", "MFSL.NS", "NAUKRI.NS", "PAGEIND.NS", "POLICYBZR.NS",
    "SUNDARMFIN.NS", "SUNTV.NS", "TATACOMM.NS", "TATAINVEST.NS", "TBOTEK.NS", "TIINDIA.NS", "UBL.NS", "UNIONBANK.NS",
    "UNITDSPR.NS", "WAAREEENER.NS", "AMBUJACEM.NS", "ACC.NS", "RAMCOCEM.NS", "JKCEMENT.NS", "SHREECEM.NS", "COROMANDEL.NS",
    "PIIND.NS", "UPL.NS", "CHAMBLFERT.NS", "SRF.NS", "TATACHEM.NS", "SIEMENS.NS", "ABB.NS", "HAVELLS.NS",
    "CUMMINSIND.NS", "VOLTAS.NS", "CROMPTON.NS", "TVSMOTOR.NS", "ESCORTS.NS", "ASHOKLEY.NS", "MOTHERSON.NS", "BALKRISIND.NS",
    "APOLLOTYRE.NS", "AFFLE.NS", "HAPPSTMNDS.NS", "PVRINOX.NS", "DCMSHRIRAM.NS", "WELCORP.NS", "JBCHEPHARM.NS", "ANANTRAJ.NS",
    "SOBHA.NS", "GODREJPROP.NS", "APTUS.NS", "ACMESOLAR.NS", "ABFRL.NS", "BIKAJI.NS", "NUVOCO.NS", "RADICO.NS",
    "IPCALAB.NS", "ALKEM.NS", "TORNTPHARM.NS", "GLENMARK.NS", "MANKIND.NS", "KALYANKJIL.NS", "BAJAJHLDNG.NS", "GODREJCP.NS",
    "DABUR.NS", "MARICO.NS", "COLPAL.NS", "360ONE.NS", "3MINDIA.NS", "AADHARHFC.NS", "AAVAS.NS", "ABBOTINDIA.NS",
    "ABLBL.NS", "ABREL.NS", "ABSLAMC.NS", "ACE.NS", "ADANIENSOL.NS", "AEGISLOG.NS", "AEGISVOPAK.NS", "AFCONS.NS",
    "AGARWALEYE.NS", "AIAENG.NS", "AIIL.NS", "AJANTPHARM.NS", "AKUMS.NS", "AKZOINDIA.NS", "ALKYLAMINE.NS", "ALOKINDS.NS",
    "AMBER.NS", "ANANDRATHI.NS", "ANGELONE.NS", "APARINDS.NS", "APLAPOLLO.NS", "APLLTD.NS", "ARE&M.NS", "ASAHIINDIA.NS",
    "ASTERDM.NS", "ASTRAZEN.NS", "ATHERENERG.NS", "ATUL.NS", "AWL.NS", "BAJAJHFL.NS", "BALRAMCHIN.NS", "BANKINDIA.NS",
    "BASF.NS", "BATAINDIA.NS", "BAYERCROP.NS", "BBTC.NS", "BDL.NS", "BEML.NS", "BERGEPAINT.NS", "BHARTIHEXA.NS",
    "BIOCON.NS", "BLS.NS", "BLUEDART.NS", "BLUEJET.NS", "BLUESTARCO.NS", "BOSCHLTD.NS", "BSOFT.NS", "CAMPUS.NS",
    "CANBK.NS", "CAPLIPOINT.NS", "CARBORUNIV.NS", "CASTROLIND.NS", "CCL.NS", "CEATLTD.NS", "CENTRALBK.NS", "CENTURYPLY.NS",
    "CERA.NS", "CESC.NS", "CGCL.NS", "CGPOWER.NS", "CHALET.NS", "CHENNPETRO.NS", "CHOICEIN.NS", "CHOLAHLDNG.NS",
    "CLEAN.NS", "COCHINSHIP.NS", "COHANCE.NS", "CONCOR.NS", "CONCORDBIO.NS", "CRAFTSMAN.NS", "CREDITACC.NS", "CRISIL.NS",
    "CUB.NS", "CYIENT.NS", "DALBHARAT.NS", "DATAPATTNS.NS", "DBREALTY.NS", "DEEPAKFERT.NS", "DEEPAKNTR.NS", "DEVYANI.NS",
    "DMART.NS", "DOMS.NS", "ECLERX.NS", "EIDPARRY.NS", "EIHOTEL.NS", "ELECON.NS", "EMAMILTD.NS", "EMCURE.NS",
    "ENDURANCE.NS", "ENGINERSIN.NS", "ENRIN.NS", "ERIS.NS", "ETERNAL.NS", "EXIDEIND.NS", "FACT.NS", "FINCABLES.NS",
    "FINPIPE.NS", "FIRSTCRY.NS", "FIVESTAR.NS", "FLUOROCHEM.NS", "FORCEMOT.NS", "FORTIS.NS", "FSL.NS", "GESHIP.NS",
    "GICRE.NS", "GILLETTE.NS", "GLAND.NS", "GLAXO.NS", "GMDCLTD.NS", "GMRAIRPORT.NS", "GODFRYPHLP.NS", "GODIGIT.NS",
    "GODREJAGRO.NS", "GODREJIND.NS", "GPIL.NS", "GRAVITA.NS", "GRSE.NS", "GUJGASLTD.NS", "GVT&D.NS", "HBLENGINE.NS",
    "HEG.NS", "HEXT.NS", "HFCL.NS", "HINDCOPPER.NS", "HINDPETRO.NS", "HINDZINC.NS", "HOMEFIRST.NS", "HONASA.NS",
    "HONAUT.NS", "HSCL.NS", "HYUNDAI.NS", "ICICIGI.NS", "ICICIPRULI.NS", "IDBI.NS", "IEX.NS", "IFCI.NS",
    "IGIL.NS", "IIFL.NS", "IKS.NS", "INDGN.NS", "INDIACEM.NS", "INDIANB.NS", "INDIGO.NS", "INOXINDIA.NS",
    "INOXWIND.NS", "INTELLECT.NS", "IOB.NS", "IOC.NS", "IRB.NS", "IRCON.NS", "IREDA.NS", "ITC.NS",
    "ITCHOTELS.NS", "ITI.NS", "J&KBANK.NS", "JBMA.NS", "JINDALSAW.NS", "JINDALSTEL.NS", "JKTYRE.NS", "JMFINANCIL.NS",
    "JPPOWER.NS", "JSL.NS", "JSWCEMENT.NS", "JSWINFRA.NS", "JUBLINGREA.NS", "JUBLPHARMA.NS", "JWL.NS", "JYOTHYLAB.NS",
    "JYOTICNC.NS", "KAJARIACER.NS", "KARURVYSYA.NS", "KAYNES.NS", "KEC.NS", "KEI.NS", "KFINTECH.NS", "KIMS.NS",
    "KIRLOSBROS.NS", "KIRLOSENG.NS", "KPIL.NS", "KPRMILL.NS", "KSB.NS", "LALPATHLAB.NS", "LATENTVIEW.NS", "LAURUSLABS.NS",
    "LEMONTREE.NS", "LICI.NS", "LINDEINDIA.NS", "LLOYDSME.NS", "LTF.NS", "LTFOODS.NS", "LTM.NS", "MAHABANK.NS",
    "MAHSCOOTER.NS", "MAHSEAMLES.NS", "MANAPPURAM.NS", "MANYAVAR.NS", "MAPMYINDIA.NS", "MAZDOCK.NS", "MEDANTA.NS", "METROPOLIS.NS",
    "MINDACORP.NS", "MMTC.NS", "MOTILALOFS.NS", "MRF.NS", "MRPL.NS", "MSUMI.NS", "NAM-INDIA.NS", "NATCOPHARM.NS",
    "NATIONALUM.NS", "NAVA.NS", "NAVINFLUOR.NS", "NBCC.NS", "NCC.NS", "NETWEB.NS", "NEULANDLAB.NS", "NEWGEN.NS",
    "NH.NS", "NIACL.NS", "NIVABUPA.NS", "NLCINDIA.NS", "NMDC.NS", "NSLNISP.NS", "NTPCGREEN.NS", "NUVAMA.NS",
    "OIL.NS", "OLAELEC.NS", "OLECTRA.NS", "ONESOURCE.NS", "PATANJALI.NS", "PCBL.NS", "PFIZER.NS", "PGEL.NS",
    "PGHH.NS", "PHOENIXLTD.NS", "PIDILITIND.NS", "PNBHOUSING.NS", "POLYCAB.NS", "POLYMED.NS", "POONAWALLA.NS", "POWERINDIA.NS",
    "PPLPHARMA.NS", "PRAJIND.NS", "PREMIERENE.NS", "PTCIL.NS", "RAILTEL.NS", "RAINBOW.NS", "RCF.NS", "REDINGTON.NS",
    "RELINFRA.NS", "RHIM.NS", "RITES.NS", "RKFORGE.NS", "RPOWER.NS", "RRKABEL.NS", "SAGILITY.NS", "SAILIFE.NS",
    "SAMMAANCAP.NS", "SAPPHIRE.NS", "SARDAEN.NS", "SAREGAMA.NS", "SBFC.NS", "SBILIFE.NS", "SCHAEFFLER.NS", "SCHNEIDER.NS",
    "SCI.NS", "SHYAMMETL.NS", "SIGNATURE.NS", "SOLARINDS.NS", "SONACOMS.NS", "SONATSOFTW.NS", "STARHEALTH.NS", "SUMICHEM.NS",
    "SUNDRMFAST.NS", "SUPREMEIND.NS", "SWANCORP.NS", "SWIGGY.NS", "SYNGENE.NS", "SYRMA.NS", "TARIL.NS", "TATATECH.NS",
    "TECHNOE.NS", "TEJASNET.NS", "THELEELA.NS", "THERMAX.NS", "TIMKEN.NS", "TITAGARH.NS", "TMPV.NS", "TRIDENT.NS",
    "TRITURBINE.NS", "TRIVENI.NS", "TTML.NS", "UCOBANK.NS", "UNOMINDA.NS", "USHAMART.NS", "UTIAMC.NS", "VBL.NS",
    "VENTIVE.NS", "VGUARD.NS", "VIJAYA.NS", "VMM.NS", "VTL.NS", "WELSPUNLIV.NS", "WHIRLPOOL.NS", "WOCKPHARMA.NS",
    "ZEEL.NS", "ZENSARTECH.NS", "ZENTEC.NS", "ZFCVINDIA.NS",
])

# ── Early Mover stocks — 100 most active gap-up candidates ──
# Used by Early Movers and ORB Scanner "Top 100" universe
# Stocks most likely to gap up on news, results, global cues
EARLY_MOVER_STOCKS = sorted(set([
    # Nifty 50 — all gap candidates
    "HDFCBANK.NS","RELIANCE.NS","ICICIBANK.NS","TCS.NS","INFY.NS",
    "BHARTIARTL.NS","KOTAKBANK.NS","AXISBANK.NS","SBIN.NS","LT.NS",
    "HINDUNILVR.NS","BAJFINANCE.NS","MARUTI.NS","HCLTECH.NS","SUNPHARMA.NS",
    "TITAN.NS","WIPRO.NS","NTPC.NS","POWERGRID.NS","ULTRACEMCO.NS",
    "TECHM.NS","TATAMOTORS.NS","BAJAJFINSV.NS","NESTLEIND.NS","GRASIM.NS",
    "ADANIPORTS.NS","COALINDIA.NS","JSWSTEEL.NS","TATASTEEL.NS","ONGC.NS",
    "BPCL.NS","HEROMOTOCO.NS","APOLLOHOSP.NS","CIPLA.NS","DRREDDY.NS",
    "EICHERMOT.NS","TATACONSUM.NS","ADANIENT.NS","HINDALCO.NS","M&M.NS",
    "BAJAJ-AUTO.NS","TRENT.NS","VEDL.NS","SHRIRAMFIN.NS","INDUSINDBK.NS",
    "HDFCLIFE.NS","BRITANNIA.NS","DIVISLAB.NS","ASIANPAINT.NS","SBICARD.NS",
    # High-vol midcap — frequent gap plays
    "ADANIPOWER.NS","ADANIGREEN.NS","RVNL.NS","IRFC.NS","HUDCO.NS",
    "HAL.NS","BEL.NS","BHEL.NS","SAIL.NS","RECLTD.NS","PFC.NS",
    "SJVN.NS","NHPC.NS","SUZLON.NS","YESBANK.NS","IDEA.NS",
    "PNB.NS","BANKBARODA.NS","RBLBANK.NS","FEDERALBNK.NS",
    "IDFCFIRSTB.NS","AUBANK.NS","BANDHANBNK.NS",
    "PERSISTENT.NS","LTTS.NS","COFORGE.NS","MPHASIS.NS","OFSS.NS",
    "KPITTECH.NS","TATAELXSI.NS","TATAPOWER.NS",
    "ATGL.NS","IGL.NS","MGL.NS","GAIL.NS","PETRONET.NS",
    "DIXON.NS","JIOFIN.NS","ABCAPITAL.NS","MUTHOOTFIN.NS","ZOMATO.NS",
    "GRAPHITE.NS","AARTIIND.NS","LODHA.NS","DLF.NS","ANANTRAJ.NS",
    "APARINDS.NS","ONESOURCE.NS","OFSS.NS",
]))


# These 60 stocks are highest volume, most liquid, move first.
# Scanning them first shows results in ~20s instead of 90s.
PRIORITY_STOCKS = sorted(set([
    # Nifty 50 heavyweights
    "RELIANCE.NS","HDFCBANK.NS","ICICIBANK.NS","INFY.NS","TCS.NS",
    "SBIN.NS","AXISBANK.NS","KOTAKBANK.NS","LT.NS","BAJFINANCE.NS",
    "HCLTECH.NS","WIPRO.NS","TECHM.NS","BHARTIARTL.NS","NTPC.NS",
    "POWERGRID.NS","ONGC.NS","MARUTI.NS","M&M.NS","TATASTEEL.NS",
    "JSWSTEEL.NS","HINDALCO.NS","COALINDIA.NS","BAJAJ-AUTO.NS","TITAN.NS",
    # High beta / first movers
    "ADANIPOWER.NS","ADANIGREEN.NS","TATAPOWER.NS","SUZLON.NS","RVNL.NS",
    "WAAREEENER.NS","IRFC.NS","RECLTD.NS","PFC.NS","HAL.NS",
    "BEL.NS","NHPC.NS","SJVN.NS","IREDA.NS","NTPCGREEN.NS",
    # High volume intraday
    "YESBANK.NS","IDEA.NS","RPOWER.NS","JPPOWER.NS","INDUSINDBK.NS",
    "ICICIPRULI.NS","HDFCLIFE.NS","SBILIFE.NS","BAJAJFINSV.NS","SHRIRAMFIN.NS",
    # Sector leaders that move with market
    "LTTS.NS","PERSISTENT.NS","COFORGE.NS","MPHASIS.NS","KPITTECH.NS",
    "SUNPHARMA.NS","DRREDDY.NS","CIPLA.NS","DLF.NS","GODREJPROP.NS",
]))

# ── Cap Tier Map ────────────────────────────────────────────
# LARGECAP  = Nifty 50 (safest, lower move but reliable)
# MIDCAP    = Nifty Midcap 100 + high-vol midcap (higher move, higher risk)
# SMALLCAP  = below midcap (highest move, highest risk)
# Used for: tier badge on shortlist cards + BEAR day filter
_NIFTY_MIDCAP_100 = {
    'ABCAPITAL','ABFRL','ACC','AFFLE','ALKEM','AMBUJACEM','APOLLOHOSP',
    'AARTIIND','ASTRAL','ATGL','AUBANK','AUROPHARMA','BANDHANBNK',
    'BANKBARODA','BEL','BHARATFORG','BHEL','BRIGADE','BSE','CAMS',
    'CANFINHOME','CDSL','CHOLAFIN','COFORGE','COROMANDEL','CROMPTON',
    'CUMMINSIND','DELHIVERY','DIXON','DLF','ELGIEQUIP','ESCORTS',
    'FEDERALBNK','GLENMARK','GODREJPROP','GRANULES','GSPL','HAL',
    'HAVELLS','HDFCAMC','HUDCO','IDFCFIRSTB','IGL','INDHOTEL',
    'INDIAMART','INDUSTOWER','IRCTC','JKCEMENT','JSWENERGY','JUBLFOOD',
    'KALYANKJIL','KPITTECH','LICHSGFIN','LODHA','LTTS','LUPIN',
    'M&MFIN','MAXHEALTH','MCX','MFSL','MGL','MPHASIS','MUTHOOTFIN',
    'NAUKRI','NHPC','OBEROIRLTY','OFSS','PAGEIND','PERSISTENT',
    'PETRONET','PIIND','POLICYBZR','PRESTIGE','PVRINOX',
    'RBLBANK','RECLTD','RVNL','SAIL','SHRIRAMFIN','SIEMENS',
    'SRF','SUNDARMFIN','SUNTV','TATACOMM','TATAELXSI','TATAINVEST',
    'TBOTEK','TIINDIA','TORNTPHARM','TRENT','TVSMOTOR','UBL',
    'UNIONBANK','UNITDSPR','UPL','VOLTAS','WHIRL','ZOMATO',
    'ADANIGREEN','ADANIPOWER','IRFC','SJVN','SUZLON','WAAREEENER',
}

_NIFTY50_SET = {s.replace('.NS','') for s in POPULAR_STOCKS[:49]}

def get_cap_tier(symbol):
    """Returns LARGECAP / MIDCAP / SMALLCAP for a given symbol."""
    sym = symbol.replace('.NS','')
    if sym in _NIFTY50_SET:
        return 'LARGECAP'
    if sym in _NIFTY_MIDCAP_100:
        return 'MIDCAP'
    return 'SMALLCAP'

CAP_TIER_BADGE = {
    'LARGECAP':  ('🔵', 'Largecap', '#1d4ed8', '#dbeafe'),
    'MIDCAP':    ('🟡', 'Midcap',   '#92400e', '#fef3c7'),
    'SMALLCAP':  ('🟠', 'Smallcap', '#c2410c', '#fff7ed'),
}

# ── Cap-filtered stock lists ───────────────────────────────
# Built from POPULAR_STOCKS using get_cap_tier()
# Used by Scanner, Early Movers, ORB universe selector
LARGECAP_STOCKS = [s for s in POPULAR_STOCKS if get_cap_tier(s.replace('.NS','')) == 'LARGECAP']
MIDCAP_STOCKS   = [s for s in POPULAR_STOCKS if get_cap_tier(s.replace('.NS','')) == 'MIDCAP']
SMALLCAP_STOCKS = [s for s in POPULAR_STOCKS if get_cap_tier(s.replace('.NS','')) == 'SMALLCAP']

# ─────────────────────────────────────────────────────────────
#  SUPERTREND + PSAR HELPER FUNCTIONS
#  Used by both SMA Weekly and Monthly Swing tabs
# ─────────────────────────────────────────────────────────────

def calc_supertrend(df, atr_period=7, multiplier=2.0):
    import pandas as _pd
    df = df.copy()
    df['_hl']  = df['High'] - df['Low']
    df['_hpc'] = (df['High'] - df['Close'].shift(1)).abs()
    df['_lpc'] = (df['Low']  - df['Close'].shift(1)).abs()
    df['_tr']  = df[['_hl','_hpc','_lpc']].max(axis=1)
    df['_atr'] = df['_tr'].rolling(atr_period).mean()
    df['_hl2'] = (df['High'] + df['Low']) / 2
    st = [float('nan')] * len(df)
    trend = [1] * len(df)
    for i in range(atr_period, len(df)):
        _av = df['_atr'].iloc[i]
        if _pd.isna(_av): continue
        ub = df['_hl2'].iloc[i] + multiplier * _av
        lb = df['_hl2'].iloc[i] - multiplier * _av
        if not _pd.isna(st[i-1]):
            if trend[i-1] == -1: ub = min(ub, st[i-1])
            if trend[i-1] == 1:  lb = max(lb, st[i-1])
        cl = df['Close'].iloc[i]
        if trend[i-1] == 1:
            if cl < lb: trend[i]=-1; st[i]=ub
            else:       trend[i]=1;  st[i]=lb
        else:
            if cl > ub: trend[i]=1;  st[i]=lb
            else:       trend[i]=-1; st[i]=ub
    df['ST_line']  = st
    df['ST_trend'] = trend
    df.drop(columns=['_hl','_hpc','_lpc','_tr','_hl2'], inplace=True, errors='ignore')
    return df


# ─────────────────────────────────────────────────────────────
#  CONFIDENT SCORE — 100 point combined score
#  Combines 6 key factors into one decisive number
#  ≥ 80 = 🔥 CONFIDENT BUY  (enter without analysis)
#  60-79 = ✅ GOOD           (check chart once then enter)
#  40-59 = ⚠️ WEAK           (skip this week)
#  < 40  = ❌ NOT SHOWN      (filtered out)
# ─────────────────────────────────────────────────────────────

def calc_confident_score(score, psar_bullish, hh, hl,
                         entry_badge, rr_t2, liq_grade):
    """
    Calculate confident score out of 100.

    COMPONENT 1 — Technical Score     (30 pts max)
    COMPONENT 2 — PSAR Status         (25 pts max)
    COMPONENT 3 — Structure HH+HL     (15 pts max)
    COMPONENT 4 — Entry Badge         (15 pts max)
    COMPONENT 5 — R:R Quality         (10 pts max)
    COMPONENT 6 — Liquidity           ( 5 pts max)
    """
    # Component 1 — Technical score (normalised)
    if   score >= 130: c1 = 30
    elif score >= 120: c1 = 25
    elif score >= 110: c1 = 20
    elif score >= 100: c1 = 15
    else:              c1 = 10

    # Component 2 — PSAR status (most important)
    # PSAR bearish = no entry regardless of other factors
    c2 = 20 if psar_bullish else 0

    # Component 3 — Price structure HH+HL
    if   hh and hl: c3 = 15
    elif hh or hl:  c3 = 8
    else:           c3 = 0

    # Component 4 — Entry badge (proximity to support)
    if   entry_badge == 'ENTER NOW':  c4 = 15
    elif entry_badge == 'ACCEPTABLE': c4 = 8
    else:                             c4 = 0

    # Component 5 — R:R quality
    rr = float(rr_t2) if rr_t2 else 0
    if   rr >= 3.0: c5 = 10
    elif rr >= 2.0: c5 = 8
    elif rr >= 1.5: c5 = 5
    else:           c5 = 0

    # Component 6 — Liquidity
    if   liq_grade == 'EXCELLENT': c6 = 5
    elif liq_grade == 'HIGH':      c6 = 3
    elif liq_grade == 'MEDIUM':    c6 = 1
    else:                          c6 = 0

    total = c1 + c2 + c3 + c4 + c5 + c6

    # Signal label
    if   total >= 130:
        signal = '🔥 CONFIDENT BUY'
        clr    = '#15803d'
        bg     = '#f0fdf4'
        border = '#86efac'
    elif total >= 100:
        signal = '✅ STRONG SETUP'
    elif total >= 75:
        signal = '👍 GOOD SETUP'
        clr    = '#16a34a'
        bg     = '#dcfce7'
        border = '#bbf7d0'
    elif total >= 55:
        signal = '⚠️ WEAK SETUP'
        clr    = '#d97706'
        bg     = '#fffbeb'
        border = '#fde68a'
    else:
        signal = '❌ SKIP'
        clr    = '#dc2626'
        bg     = '#fef2f2'
        border = '#fca5a5'

    return {
        'confident_score':  total,
        'confident_signal': signal,
        'confident_label':  signal,
        'confident_clr':    clr,
        'confident_bg':     bg,
        'confident_bdr':    border,
        'confident_border': border,
        'c1_tech':          c1,
        'c2_psar':          c2,
        'c3_struct':        c3,
        'c3_structure':     c3,
        'c4_entry':         c4,
        'c4_badge':         c4,
        'c5_rr':            c5,
        'c6_liq':           c6,
    }


#  Check 1: Weekly Candle Quality
#  Check 2: Support Proximity
#  Check 3: Price Structure
#
#  Used by Monthly Swing + SMA Weekly scanners
#  Adds high-probability entry confirmation
#  on top of existing indicator-based scoring
# ─────────────────────────────────────────────────────────────

def pa_candle_quality(o, h, l, c, prev_o, prev_h, prev_l, prev_c):
    """
    Check 1 — Analyse the most recent weekly/daily candle.
    Returns (pattern_name, score, emoji, description)

    Patterns detected (in priority order):
    Bullish Hammer, Bullish Engulfing, Strong Bull,
    Doji, Shooting Star, Bearish Engulfing, Bearish candle
    """
    body      = abs(c - o)
    rng       = h - l if h > l else 0.001
    upper_wick= h - max(c, o)
    lower_wick= min(c, o) - l
    body_pct  = body / rng          # body as % of total range
    close_pos = (c - l) / rng       # where close is in range (0=low, 1=high)
    prev_body = abs(prev_c - prev_o)

    # ── Shooting Star / Bearish Pin Bar ──────────────────
    # Long upper wick ≥ 2× body, close near low
    if (upper_wick >= 2.0 * body and
        lower_wick <= 0.3 * body and
        close_pos <= 0.35 and body > 0):
        return ('Shooting Star', -15, '🌠',
                'Sellers rejected rally — bearish reversal signal')

    # ── Bearish Engulfing ─────────────────────────────────
    if (c < o and                          # red candle
        o >= prev_c and c <= prev_o and    # engulfs previous
        prev_c > prev_o and                # previous was green
        body >= prev_body * 0.9):
        return ('Bearish Engulfing', -12, '🔴',
                'Bears overwhelmed bulls — avoid entry this week')

    # ── Doji — Indecision ─────────────────────────────────
    if body_pct < 0.08 and rng > 0:
        return ('Doji', -5, '➖',
                'Market indecision — wait for next candle direction')

    # ── Bearish candle (red, close in bottom half) ────────
    if c < o and close_pos < 0.40:
        return ('Bearish Candle', -10, '🔴',
                'Sellers in control this week — wait for recovery')

    # ── Bullish Hammer ────────────────────────────────────
    # Long lower wick ≥ 2× body, close near high
    if (lower_wick >= 2.0 * body and
        upper_wick <= 0.4 * body and
        close_pos >= 0.60 and
        body > 0):
        return ('Hammer', +15, '🔨',
                'Strong buyer rejection at lows — ideal entry signal')

    # ── Bullish Engulfing ─────────────────────────────────
    if (c > o and                           # green candle
        o <= prev_c and c >= prev_o and     # engulfs previous
        prev_c < prev_o and                 # previous was red
        body >= prev_body * 0.9):
        return ('Bullish Engulfing', +12, '🟢',
                'Bulls overwhelmed bears — strong entry signal')

    # ── Strong Bull Candle ────────────────────────────────
    # Close in top 30%, body ≥ 60% of range
    if (c > o and
        close_pos >= 0.70 and
        body_pct >= 0.55):
        return ('Strong Bull', +8, '💚',
                'Bulls dominated all week — momentum strong')

    # ── Weak Bull (green but not strong) ─────────────────
    if c > o and close_pos >= 0.50:
        return ('Mild Bull', +4, '🟡',
                'Mild bullish — acceptable but not ideal')

    # ── Default neutral ───────────────────────────────────
    return ('Neutral', 0, '⚪', 'No clear directional signal')


def pa_support_proximity(price, sma20, sma50, prev_swing_high,
                         fib_382, fib_500, fib_618, week_high, week_low):
    """
    Check 2 — Is current price near a key support level?
    Returns (level_name, score, pct_from_level, description)

    Support levels checked (in priority):
    1. Weekly SMA20      (primary dynamic support)
    2. Previous swing high turned support
    3. Fibonacci levels  (38.2%, 50%, 61.8%)
    4. Round numbers     (₹500, ₹1000 etc)
    5. SMA50            (secondary support)
    """
    best_score = -20
    best_name  = 'Far from support'
    best_pct   = 99.0
    best_desc  = 'Price extended — wait for pullback'

    def _pct(ref):
        return abs(price - ref) / ref * 100 if ref > 0 else 99

    # ── SMA20 proximity ───────────────────────────────────
    _d20 = _pct(sma20)
    if   _d20 <= 1.0: _s20 = 20; _n20 = 'At SMA20 (≤1%)'
    elif _d20 <= 2.0: _s20 = 16; _n20 = 'Near SMA20 (1-2%)'
    elif _d20 <= 3.5: _s20 = 10; _n20 = 'Close to SMA20 (2-3.5%)'
    elif _d20 <= 5.0: _s20 =  4; _n20 = 'Approaching SMA20 (3.5-5%)'
    elif _d20 <= 8.0: _s20 = -5; _n20 = 'Moderate distance from SMA20'
    else:             _s20 =-15; _n20 = f'Extended {_d20:.1f}% above SMA20'
    if _s20 > best_score:
        best_score = _s20; best_name = _n20
        best_pct   = round(_d20, 1)
        best_desc  = (f'Price is {_d20:.1f}% from weekly SMA20 ₹{sma20:,.0f}')

    # ── Previous swing high (support after breakout) ──────
    if prev_swing_high and prev_swing_high > 0 and price > prev_swing_high * 0.97:
        _dpsh = _pct(prev_swing_high)
        if _dpsh <= 1.5:
            _spsh = 15
            _npsh = f'At prev swing high ₹{prev_swing_high:,.0f} (retest ✅)'
            _dpsh_desc = 'Classic breakout retest — very strong entry'
            if _spsh > best_score:
                best_score = _spsh; best_name = _npsh
                best_pct   = round(_dpsh, 1); best_desc = _dpsh_desc

    # ── Fibonacci levels ──────────────────────────────────
    for _fval, _fname, _fscore in [
        (fib_382, 'Fib 38.2%', 15),
        (fib_500, 'Fib 50.0%', 12),
        (fib_618, 'Fib 61.8%', 10),
    ]:
        if _fval and _fval > 0:
            _df = _pct(_fval)
            if _df <= 1.5 and _fscore > best_score:
                best_score = _fscore
                best_name  = f'At {_fname} ₹{_fval:,.0f}'
                best_pct   = round(_df, 1)
                best_desc  = f'Price at {_fname} retracement — institutional support zone'

    # ── Round number proximity ────────────────────────────
    _round_levels = []
    _base = int(price / 100) * 100
    for _rl in [_base - 100, _base, _base + 100]:
        if _rl > 0: _round_levels.append(_rl)
    for _rl in _round_levels:
        _dr = _pct(_rl)
        if _dr <= 0.8:
            _sr = 10
            if _sr > best_score:
                best_score = _sr
                best_name  = f'At round ₹{_rl:,}'
                best_pct   = round(_dr, 1)
                best_desc  = f'Psychological support at ₹{_rl:,} — institutions buy here'

    # ── SMA50 as secondary support ────────────────────────
    if sma50 and sma50 > 0:
        _d50 = _pct(sma50)
        if _d50 <= 2.0 and 8 > best_score:
            best_score = 8
            best_name  = f'Near SMA50 ₹{sma50:,.0f}'
            best_pct   = round(_d50, 1)
            best_desc  = 'Secondary support — acceptable entry zone'

    # ── Confluence bonus ──────────────────────────────────
    # Two or more support levels within 2% of each other = very strong
    _supports = []
    if sma20 > 0 and _pct(sma20) <= 3: _supports.append('SMA20')
    if fib_382 > 0 and _pct(fib_382) <= 3: _supports.append('Fib38')
    if fib_500 > 0 and _pct(fib_500) <= 3: _supports.append('Fib50')
    if fib_618 > 0 and _pct(fib_618) <= 3: _supports.append('Fib62')
    if prev_swing_high > 0 and _pct(prev_swing_high) <= 3: _supports.append('PrevHigh')
    if len(_supports) >= 2:
        best_score += 10
        best_name   = f'CONFLUENCE: {" + ".join(_supports)}'
        best_desc  += f' — Multiple support levels overlapping (strongest entry)'

    return (best_name, best_score, best_pct, best_desc)


def pa_price_structure(df_weekly):
    """
    Check 3 — Is the price structure bullish (HH + HL)?
    Returns (structure, score, hard_reject, description)

    Analyses last 8 weekly candles:
    - Swing High/Low sequence (HH+HL = uptrend)
    - Last 3 closes trending up
    - Weekly close position (top/bottom of range)
    - Structure break detection (hard reject)
    """
    import pandas as _pd
    import numpy as _np

    try:
        _df = df_weekly.copy()
        if len(_df) < 6:
            return ('Unknown', 0, False, 'Not enough data')

        _closes = _df['Close'].values
        _highs  = _df['High'].values
        _lows   = _df['Low'].values
        _opens  = _df['Open'].values

        # ── Last 3 weekly closes trending ────────────────
        _c1 = float(_closes[-1])   # this week
        _c2 = float(_closes[-2])   # last week
        _c3 = float(_closes[-3])   # 2 weeks ago
        _c4 = float(_closes[-4])   # 3 weeks ago
        _c5 = float(_closes[-5])   # 4 weeks ago

        _consec_up   = _c1 > _c2 > _c3          # 3 consec green weeks
        _consec_down = _c1 < _c2 < _c3          # 3 consec red weeks
        _last_up     = _c1 > _c2                 # last week green
        _last_down   = _c1 < _c2                 # last week red

        # ── Swing High / Low sequence ─────────────────────
        # Compare this 2-week block vs previous 2-week block
        _curr_high  = max(float(_highs[-1]),  float(_highs[-2]))
        _prev_high  = max(float(_highs[-3]),  float(_highs[-4]))
        _older_high = max(float(_highs[-5]),  float(_highs[-6])) if len(_df)>=6 else _prev_high

        _curr_low   = min(float(_lows[-1]),   float(_lows[-2]))
        _prev_low   = min(float(_lows[-3]),   float(_lows[-4]))
        _older_low  = min(float(_lows[-5]),   float(_lows[-6])) if len(_df)>=6 else _prev_low

        _hh = _curr_high > _prev_high          # higher high
        _hl = _curr_low  > _prev_low           # higher low
        _lh = _curr_high < _prev_high          # lower high
        _ll = _curr_low  < _prev_low           # lower low

        # ── Close position within weekly range ────────────
        _wk_rng   = float(_highs[-1]) - float(_lows[-1])
        _close_pos= (float(_closes[-1]) - float(_lows[-1])) / (_wk_rng + 0.001)

        # ── HARD REJECT: confirmed downtrend structure ────
        # Two consecutive Lower Highs AND Lower Lows = sell
        _lh2 = _prev_high < _older_high        # prev also lower high
        _ll2 = _prev_low  < _older_low         # prev also lower low
        _structure_broken = (_lh and _ll and _lh2 and _ll2)

        # ── Scoring ───────────────────────────────────────
        score = 0
        desc_parts = []

        if _structure_broken:
            return ('Structure Broken ❌', -25, True,
                    'Confirmed downtrend — Lower Highs + Lower Lows for 3 weeks. '
                    'Do not enter. Wait for structure to recover.')

        if _hh and _hl:
            score += 12
            desc_parts.append('HH+HL ✅ uptrend structure intact')
        elif _hh:
            score += 6
            desc_parts.append('Higher High ✅ but HL not confirmed')
        elif _hl:
            score += 4
            desc_parts.append('Higher Low ✅ but HH not confirmed')
        elif _lh and _ll:
            score -= 12
            desc_parts.append('LH+LL ⚠️ structure weakening')
        elif _lh:
            score -= 6
            desc_parts.append('Lower High ⚠️ momentum fading')

        if _consec_up:
            score += 10
            desc_parts.append('3 consecutive green weeks 🟢')
        elif _last_up and _c2 < _c3:
            score += 5
            desc_parts.append('Recovering after pullback ✅')
        elif _consec_down:
            score -= 10
            desc_parts.append('3 consecutive red weeks 🔴')
        elif _last_down:
            score -= 4
            desc_parts.append('Last week red ⚠️')

        if _close_pos >= 0.75:
            score += 8
            desc_parts.append('Closed in top 25% of week 💪')
        elif _close_pos >= 0.50:
            score += 3
            desc_parts.append('Closed above midpoint')
        elif _close_pos <= 0.25:
            score -= 8
            desc_parts.append('Closed in bottom 25% of week ⚠️')

        structure = ('Bullish ✅' if score >= 10
                     else 'Neutral ⚠️' if score >= 0
                     else 'Bearish ❌')
        desc = ' · '.join(desc_parts) if desc_parts else 'No clear structure'
        return (structure, score, False, desc)

    except Exception:
        return ('Unknown', 0, False, 'Could not analyse structure')


def run_price_action_analysis(df_weekly, price, sma20, sma50,
                               fib_382, fib_500, fib_618):
    """
    Master function — runs all 3 PA checks and returns combined result.
    Returns dict with all PA data for display and scoring.
    """
    import pandas as _pd

    result = {
        # Check 1
        'candle_pattern':   'Unknown',
        'candle_score':     0,
        'candle_emoji':     '⚪',
        'candle_desc':      '',
        # Check 2
        'support_name':     'Unknown',
        'support_score':    0,
        'support_pct':      0,
        'support_desc':     '',
        # Check 3
        'structure':        'Unknown',
        'structure_score':  0,
        'structure_reject': False,
        'structure_desc':   '',
        # Combined
        'pa_total_score':   0,
        'pa_signal':        '⚪ Unknown',
        'pa_signal_clr':    '#64748b',
        'pa_signal_bg':     '#f8fafc',
    }

    try:
        # ── Check 1: Latest candle ────────────────────────
        if len(df_weekly) >= 2:
            _l  = df_weekly.iloc[-1]
            _p  = df_weekly.iloc[-2]
            _cp, _cs, _ce, _cd = pa_candle_quality(
                float(_l['Open']), float(_l['High']),
                float(_l['Low']),  float(_l['Close']),
                float(_p['Open']), float(_p['High']),
                float(_p['Low']),  float(_p['Close']))
            result['candle_pattern'] = _cp
            result['candle_score']   = _cs
            result['candle_emoji']   = _ce
            result['candle_desc']    = _cd

        # ── Check 2: Support proximity ────────────────────
        # Find previous swing high (last peak before current)
        _prev_swing_h = 0
        try:
            _window  = df_weekly['High'].values[-20:-2]
            _peak_idx= int(_window.argmax())
            _prev_swing_h = float(_window[_peak_idx])
        except Exception:
            pass

        _wk_h = float(df_weekly['High'].iloc[-1])
        _wk_l = float(df_weekly['Low'].iloc[-1])
        _sn, _ss, _sp, _sd = pa_support_proximity(
            price, sma20, sma50, _prev_swing_h,
            fib_382, fib_500, fib_618, _wk_h, _wk_l)
        result['support_name']  = _sn
        result['support_score'] = _ss
        result['support_pct']   = _sp
        result['support_desc']  = _sd

        # ── Check 3: Price structure ──────────────────────
        _st, _ss2, _sr, _sd2 = pa_price_structure(df_weekly)
        result['structure']        = _st
        result['structure_score']  = _ss2
        result['structure_reject'] = _sr
        result['structure_desc']   = _sd2

        # ── Combined PA score ─────────────────────────────
        _pa_total = (result['candle_score'] +
                     result['support_score'] +
                     result['structure_score'])
        result['pa_total_score'] = _pa_total

        # ── PA Signal label ───────────────────────────────
        if result['structure_reject']:
            result['pa_signal']     = '🔴 AVOID — Structure broken'
            result['pa_signal_clr'] = '#dc2626'
            result['pa_signal_bg']  = '#fef2f2'
        elif _pa_total >= 30:
            result['pa_signal']     = '🔥 STRONG SETUP — Enter Monday open'
            result['pa_signal_clr'] = '#15803d'
            result['pa_signal_bg']  = '#f0fdf4'
        elif _pa_total >= 15:
            result['pa_signal']     = '✅ GOOD SETUP — Enter with normal size'
            result['pa_signal_clr'] = '#16a34a'
            result['pa_signal_bg']  = '#dcfce7'
        elif _pa_total >= 0:
            result['pa_signal']     = '⚠️ WEAK SETUP — Wait for better candle'
            result['pa_signal_clr'] = '#d97706'
            result['pa_signal_bg']  = '#fffbeb'
        else:
            result['pa_signal']     = '🔴 RISKY — Wait for pullback to SMA20'
            result['pa_signal_clr'] = '#dc2626'
            result['pa_signal_bg']  = '#fef2f2'

    except Exception:
        pass

    return result


def calc_psar(df, step=0.02, max_af=0.20):
    import pandas as _pd
    n = len(df)
    psar=[float('nan')]*n; bull=[True]*n; af=[step]*n; ep=[float('nan')]*n
    psar[0]=float(df['Low'].iloc[0]); ep[0]=float(df['High'].iloc[0])
    for i in range(1, n):
        pp=psar[i-1]; pe=ep[i-1]; pa=af[i-1]; pb=bull[i-1]
        hi=float(df['High'].iloc[i]); lo=float(df['Low'].iloc[i])
        if _pd.isna(pp): pp=lo
        if pb:
            np2=pp+pa*(pe-pp)
            if i>=2: np2=min(np2,float(df['Low'].iloc[i-1]),float(df['Low'].iloc[i-2]))
            if lo<np2:
                bull[i]=False; psar[i]=pe; ep[i]=lo; af[i]=step
            else:
                bull[i]=True; psar[i]=np2
                if hi>pe: ep[i]=hi; af[i]=min(pa+step,max_af)
                else:     ep[i]=pe; af[i]=pa
        else:
            np2=pp+pa*(pe-pp)
            if i>=2: np2=max(np2,float(df['High'].iloc[i-1]),float(df['High'].iloc[i-2]))
            if hi>np2:
                bull[i]=True; psar[i]=pe; ep[i]=hi; af[i]=step
            else:
                bull[i]=False; psar[i]=np2
                if lo<pe: ep[i]=lo; af[i]=min(pa+step,max_af)
                else:     ep[i]=pe; af[i]=pa
    df['PSAR']=[round(x,2) if not _pd.isna(x) else float('nan') for x in psar]
    df['PSAR_bull']=bull
    return df


def get_st_psar(df, timeframe='daily'):
    """
    Get Supertrend + PSAR signals from OHLC dataframe.
    timeframe: 'daily' or 'weekly'
    Returns dict with st_bullish, st_fresh_flip, st_line, st_score,
                       st_periods, psar, psar_bullish, psar_score
    """
    import pandas as _pd
    ps = 0.01 if timeframe=='weekly' else 0.02
    pm = 0.10 if timeframe=='weekly' else 0.20
    out = {
        'st_bullish':False,'st_fresh_flip':False,
        'st_line':None,'st_score':0,'st_periods':0,
        'psar':None,'psar_bullish':False,'psar_score':0,
    }
    try:
        ds = calc_supertrend(df.copy(), 7, 2.0)
        if len(ds)>=2:
            sn=int(ds['ST_trend'].iloc[-1]); sp=int(ds['ST_trend'].iloc[-2])
            sl=float(ds['ST_line'].iloc[-1])
            out['st_bullish']=sn==1
            out['st_fresh_flip']=sn==1 and sp==-1
            out['st_line']=round(sl,2) if not _pd.isna(sl) else None
            w=0
            for i in range(len(ds)-1,-1,-1):
                if int(ds['ST_trend'].iloc[i])==1: w+=1
                else: break
            out['st_periods']=w
            out['st_score']=(15 if out['st_fresh_flip'] else 8 if out['st_bullish'] else -10)
    except Exception: pass
    try:
        dp = calc_psar(df.copy(), step=ps, max_af=pm)
        if len(dp)>=1:
            pv=float(dp['PSAR'].iloc[-1]); pb=bool(dp['PSAR_bull'].iloc[-1])
            cl=float(dp['Close'].iloc[-1])
            out['psar']=round(pv,2) if not _pd.isna(pv) else None
            out['psar_bullish']=pb and cl>pv
            out['psar_score']=(8 if out['psar_bullish'] else -8)
    except Exception: pass
    return out

# ── Sector map ────────────────────────────────────────────
SECTOR_MAP = {
    # ── Banking (PSU + Private) ───────────────────────────
    'HDFCBANK':'BANKING','ICICIBANK':'BANKING','SBIN':'BANKING','AXISBANK':'BANKING',
    'KOTAKBANK':'BANKING','BANDHANBNK':'BANKING','IDFCFIRSTB':'BANKING','INDUSINDBK':'BANKING',
    'FEDERALBNK':'BANKING','CANBK':'BANKING','PNB':'BANKING','BANKBARODA':'BANKING',
    'AUBANK':'BANKING','RBLBANK':'BANKING','KARURVYSYA':'BANKING','CENTRALBK':'BANKING',
    'INDIANB':'BANKING','MAHABANK':'BANKING','UCOBANK':'BANKING','UNIONBANK':'BANKING',
    'BANKINDIA':'BANKING','IOB':'BANKING','IDBI':'BANKING','J&KBANK':'BANKING',
    'CUB':'BANKING','YESBANK':'BANKING','VIJAYA':'BANKING','IFCI':'BANKING',

    # ── NBFC ──────────────────────────────────────────────
    'BAJFINANCE':'NBFC','BAJAJFINSV':'NBFC','CHOLAFIN':'NBFC','MUTHOOTFIN':'NBFC',
    'IIFL':'NBFC','M&MFIN':'NBFC','APTUS':'NBFC','CREDITACC':'NBFC',
    'MANAPPURAM':'NBFC','SHRIRAMFIN':'NBFC','LICHSGFIN':'NBFC','PNBHOUSING':'NBFC',
    'CANFINHOME':'NBFC','HOMEFIRST':'NBFC','AAVAS':'NBFC','ABCAPITAL':'NBFC',
    'CHOLAHLDNG':'NBFC','BAJAJHFL':'NBFC','SBFC':'NBFC','FIVESTAR':'NBFC',
    'AADHARHFC':'NBFC','POONAWALLA':'NBFC','JIOFIN':'NBFC','JMFINANCIL':'NBFC',
    'SUNDARMFIN':'NBFC','360ONE':'NBFC','ANANDRATHI':'NBFC','MOTILALOFS':'NBFC',
    'NUVAMA':'NBFC','ANGELONE':'NBFC','CHOICEIN':'NBFC','LTF':'NBFC',
    'CGCL':'NBFC',

    # ── Insurance ─────────────────────────────────────────
    'HDFCLIFE':'INSURANCE','SBILIFE':'INSURANCE','ICICIGI':'INSURANCE',
    'ICICIPRULI':'INSURANCE','STARHEALTH':'INSURANCE','NIACL':'INSURANCE',
    'GICRE':'INSURANCE','LICI':'INSURANCE','GODIGIT':'INSURANCE',
    'NIVABUPA':'INSURANCE','POLICYBZR':'INSURANCE','MFSL':'INSURANCE',

    # ── Capital Markets / Exchanges ───────────────────────
    'BSE':'CAPITAL_MARKETS','MCX':'CAPITAL_MARKETS','CDSL':'CAPITAL_MARKETS',
    'CAMS':'CAPITAL_MARKETS','KFINTECH':'CAPITAL_MARKETS','IEX':'CAPITAL_MARKETS',
    'CRISIL':'CAPITAL_MARKETS','HDFCAMC':'CAPITAL_MARKETS','ABSLAMC':'CAPITAL_MARKETS',
    'UTIAMC':'CAPITAL_MARKETS','NAM-INDIA':'CAPITAL_MARKETS',

    # ── IT & Software ─────────────────────────────────────
    'TCS':'IT','INFY':'IT','WIPRO':'IT','HCLTECH':'IT','TECHM':'IT',
    'LTTS':'IT','MPHASIS':'IT','COFORGE':'IT','PERSISTENT':'IT',
    'OFSS':'IT','KPITTECH':'IT','TATAELXSI':'IT','CYIENT':'IT',
    'BSOFT':'IT','SONATSOFTW':'IT','ZENSARTECH':'IT','NEWGEN':'IT',
    'INTELLECT':'IT','LATENTVIEW':'IT','ECLERX':'IT','MAPMYINDIA':'IT',
    'INDIAMART':'IT','NAUKRI':'IT','HAPPSTMNDS':'IT','NETWEB':'IT',
    'SYRMA':'IT','KAYNES':'IT','SAGILITY':'IT','TATATECH':'IT',
    'AFFLE':'IT','FIRSTCRY':'IT','PAYTM':'IT','ETERNAL':'IT',
    'IKS':'IT','REDINGTON':'IT',

    # ── Telecom ───────────────────────────────────────────
    'BHARTIARTL':'TELECOM','IDEA':'TELECOM','TATACOMM':'TELECOM','HFCL':'TELECOM',
    'BHARTIHEXA':'TELECOM','INDUSTOWER':'TELECOM','TEJASNET':'TELECOM','TTML':'TELECOM',

    # ── Auto & Auto Ancillary ─────────────────────────────
    'MARUTI':'AUTO','TATAMOTORS':'AUTO','M&M':'AUTO','BAJAJ-AUTO':'AUTO',
    'HEROMOTOCO':'AUTO','EICHERMOT':'AUTO','TVSMOTOR':'AUTO','ASHOKLEY':'AUTO',
    'ESCORTS':'AUTO','MOTHERSON':'AUTO','BHARATFORG':'AUTO','BOSCHLTD':'AUTO',
    'TIINDIA':'AUTO','ENDURANCE':'AUTO','SONACOMS':'AUTO','APOLLOTYRE':'AUTO',
    'CEATLTD':'AUTO','BALKRISIND':'AUTO','MRF':'AUTO','EXIDEIND':'AUTO',
    'UNOMINDA':'AUTO','MINDACORP':'AUTO','MAHSCOOTER':'AUTO','FORCEMOT':'AUTO',
    'SUNDRMFAST':'AUTO','SCHAEFFLER':'AUTO','TIMKEN':'AUTO','JKTYRE':'AUTO',
    'RKFORGE':'AUTO','CRAFTSMAN':'AUTO','ASAHIINDIA':'AUTO','MSUMI':'AUTO',
    'HYUNDAI':'AUTO','TMPV':'AUTO',

    # ── Pharma & Healthcare ───────────────────────────────
    'SUNPHARMA':'PHARMA','DRREDDY':'PHARMA','CIPLA':'PHARMA','DIVISLAB':'PHARMA',
    'AUROPHARMA':'PHARMA','ALKEM':'PHARMA','LUPIN':'PHARMA','TORNTPHARM':'PHARMA',
    'IPCALAB':'PHARMA','GRANULES':'PHARMA','GLENMARK':'PHARMA','NATCOPHARM':'PHARMA',
    'ABBOTINDIA':'PHARMA','PFIZER':'PHARMA','GLAXO':'PHARMA','LAURUSLABS':'PHARMA',
    'APOLLOHOSP':'PHARMA','MAXHEALTH':'PHARMA','FORTIS':'PHARMA','METROPOLIS':'PHARMA',
    'AJANTPHARM':'PHARMA','BIOCON':'PHARMA','LALPATHLAB':'PHARMA','RAINBOW':'PHARMA',
    'SYNGENE':'PHARMA','ASTERDM':'PHARMA','MEDANTA':'PHARMA','KIMS':'PHARMA',
    'NH':'PHARMA','MANKIND':'PHARMA','JBCHEPHARM':'PHARMA','CAPLIPOINT':'PHARMA',
    'NEULANDLAB':'PHARMA','ERIS':'PHARMA','CONCORDBIO':'PHARMA','EMCURE':'PHARMA',
    'GLAND':'PHARMA','ZYDUSLIFE':'PHARMA','WOCKPHARMA':'PHARMA','PPLPHARMA':'PHARMA',
    'AKUMS':'PHARMA','POLYMED':'PHARMA','AGARWALEYE':'PHARMA',
    'ONESOURCE':'PHARMA',  # FIXED 20-Jun-2026: was wrongly 'IT' — actually a
                            # pharma CDMO (Dr Reddy's semaglutide mfg partner)

    # ── Energy & Oil ──────────────────────────────────────
    'RELIANCE':'ENERGY','ONGC':'ENERGY','BPCL':'ENERGY','IOC':'ENERGY',
    'NTPC':'ENERGY','POWERGRID':'ENERGY','ADANIPOWER':'ENERGY','TATAPOWER':'ENERGY',
    'GAIL':'ENERGY','PETRONET':'ENERGY','GUJGASLTD':'ENERGY','MGL':'ENERGY',
    'IGL':'ENERGY','ATGL':'ENERGY','TORNTPOWER':'ENERGY','CESC':'ENERGY',
    'HINDPETRO':'ENERGY','OIL':'ENERGY','MRPL':'ENERGY','CHENNPETRO':'ENERGY',
    'GSPL':'ENERGY','GSPL':'ENERGY','FACT':'ENERGY','DEEPAKFERT':'ENERGY',
    'CHAMBLFERT':'ENERGY','RCF':'ENERGY','COROMANDEL':'ENERGY',

    # ── Solar / Renewables ────────────────────────────────
    'WAAREEENER':'SOLAR','PREMIERENE':'SOLAR','SUZLON':'SOLAR','ADANIGREEN':'SOLAR',
    'NHPC':'SOLAR','SJVN':'SOLAR','INOXWIND':'SOLAR','NTPCGREEN':'SOLAR',
    'ACMESOLAR':'SOLAR','ADANIENSOL':'SOLAR','ATHERENERG':'SOLAR',
    'OLECTRA':'SOLAR','OLAELEC':'SOLAR','JSWENERGY':'SOLAR',

    # ── Metals & Mining ───────────────────────────────────
    'TATASTEEL':'METALS','JSWSTEEL':'METALS','HINDALCO':'METALS','SAIL':'METALS',
    'VEDL':'METALS','NATIONALUM':'METALS','NMDC':'METALS','COALINDIA':'METALS',
    'HINDCOPPER':'METALS','WELCORP':'METALS','JINDALSAW':'METALS','JINDALSTEL':'METALS',
    'JSL':'METALS','NSLNISP':'METALS','GPIL':'METALS','SHYAMMETL':'METALS',
    'GRAVITA':'METALS','HEG':'METALS','GRAPHITE':'METALS','NAVA':'METALS',
    'GMDCLTD':'METALS','HINDZINC':'METALS','MOIL':'METALS','MMTC':'METALS',

    # ── Capital Goods / Infra / Defence ───────────────────
    'LT':'INFRA','SIEMENS':'INFRA','ABB':'INFRA','BHEL':'INFRA',
    'THERMAX':'INFRA','CUMMINSIND':'INFRA','GRSE':'INFRA','BEL':'INFRA',
    'HAL':'INFRA','COCHINSHIP':'INFRA','RVNL':'INFRA','IRFC':'INFRA',
    'RAILTEL':'INFRA','IRCTC':'INFRA','BDL':'INFRA','BEML':'INFRA',
    'MAZDOCK':'INFRA','DATAPATTNS':'INFRA','KEC':'INFRA','KPIL':'INFRA',
    'NCC':'INFRA','NBCC':'INFRA','IRB':'INFRA','ENGINERSIN':'INFRA',
    'RITES':'INFRA','IRCON':'INFRA','IREDA':'INFRA','HUDCO':'INFRA',
    'RECLTD':'INFRA','PFC':'INFRA','JSWINFRA':'INFRA','AFCONS':'INFRA',
    'TRITURBINE':'INFRA','ELECON':'INFRA','ELGIEQUIP':'INFRA','KIRLOSBROS':'INFRA',
    'KIRLOSENG':'INFRA','KSB':'INFRA','TITAGARH':'INFRA','GMRAIRPORT':'INFRA',
    'AIAENG':'INFRA','ACE':'INFRA','JBMA':'INFRA','POWERINDIA':'INFRA',
    'GVT&D':'INFRA','HBLENGINE':'INFRA','ARE&M':'INFRA','SCI':'INFRA',
    'GESHIP':'INFRA','CONCOR':'INFRA',

    # ── FMCG & Consumer Staples ───────────────────────────
    'HINDUNILVR':'FMCG','ITC':'FMCG','NESTLEIND':'FMCG','BRITANNIA':'FMCG',
    'DABUR':'FMCG','MARICO':'FMCG','TATACONSUM':'FMCG','GODREJCP':'FMCG',
    'COLPAL':'FMCG','EMAMILTD':'FMCG','VBL':'FMCG','RADICO':'FMCG',
    'UNITDSPR':'FMCG','BIKAJI':'FMCG','JYOTHYLAB':'FMCG','PATANJALI':'FMCG',
    'GODFRYPHLP':'FMCG','GILLETTE':'FMCG','PGHH':'FMCG','AWL':'FMCG',
    'HONASA':'FMCG','DOMS':'FMCG','BALRAMCHIN':'FMCG','TRIVENI':'FMCG',

    # ── Real Estate ───────────────────────────────────────
    'DLF':'REALTY','GODREJPROP':'REALTY','PRESTIGE':'REALTY','OBEROIRLTY':'REALTY',
    'BRIGADE':'REALTY','SOBHA':'REALTY','PHOENIXLTD':'REALTY','ANANTRAJ':'REALTY',
    'LODHA':'REALTY','SIGNATURE':'REALTY','DBREALTY':'REALTY','CHALET':'REALTY',
    'VENTIVE':'REALTY','JUBLINGREA':'REALTY',

    # ── Cement ────────────────────────────────────────────
    'ULTRACEMCO':'CEMENT','AMBUJACEM':'CEMENT','ACC':'CEMENT','SHREECEM':'CEMENT',
    'DALBHARAT':'CEMENT','RAMCOCEM':'CEMENT','JKCEMENT':'CEMENT','NUVOCO':'CEMENT',
    'INDIACEM':'CEMENT','JSWCEMENT':'CEMENT',

    # ── Chemicals & Specialty ─────────────────────────────
    'PIDILITIND':'CHEMICALS','ASIANPAINT':'CHEMICALS','BERGEPAINT':'CHEMICALS',
    'AARTIIND':'CHEMICALS','DEEPAKNTR':'CHEMICALS','NAVINFLUOR':'CHEMICALS',
    'CLEAN':'CHEMICALS','TATACHEM':'CHEMICALS','SRF':'CHEMICALS',
    'ATUL':'CHEMICALS','ALKYLAMINE':'CHEMICALS','FLUOROCHEM':'CHEMICALS',
    'PCBL':'CHEMICALS','SUMICHEM':'CHEMICALS','PIIND':'CHEMICALS',
    'DCMSHRIRAM':'CHEMICALS','GODREJAGRO':'CHEMICALS','BASF':'CHEMICALS',
    'BAYERCROP':'CHEMICALS','UPL':'CHEMICALS','CASTROLIND':'CHEMICALS',
    'AKZOINDIA':'CHEMICALS','LINDEINDIA':'CHEMICALS','RHIM':'CHEMICALS',
    'CARBORUNIV':'CHEMICALS','PRAJIND':'CHEMICALS','HSCL':'CHEMICALS',

    # ── Consumer Durables & Electronics ───────────────────
    'HAVELLS':'CONSUMER','VOLTAS':'CONSUMER','WHIRLPOOL':'CONSUMER',
    'CROMPTON':'CONSUMER','DIXON':'CONSUMER','AMBER':'CONSUMER',
    'BATAINDIA':'CONSUMER','VGUARD':'CONSUMER','POLYCAB':'CONSUMER',
    'KEI':'CONSUMER','RRKABEL':'CONSUMER','FINCABLES':'CONSUMER',
    'BLUESTARCO':'CONSUMER','LLOYDSME':'CONSUMER','CGPOWER':'CONSUMER',
    'SOLARINDS':'CONSUMER','USHAMART':'CONSUMER','CERA':'CONSUMER',
    'KAJARIACER':'CONSUMER','CENTURYPLY':'CONSUMER','ASTRAL':'CONSUMER',
    'APLAPOLLO':'CONSUMER','FINPIPE':'CONSUMER','SUPREMEIND':'CONSUMER',
    'TRIDENT':'CONSUMER','WELSPUNLIV':'CONSUMER','GRASIM':'CONSUMER',
    'KPRMILL':'CONSUMER','TECHNOE':'CONSUMER','BLUEJET':'CONSUMER',
    'APARINDS':'CONSUMER',

    # ── Retail & Consumer Services ────────────────────────
    'DMART':'RETAIL','TRENT':'RETAIL','NYKAA':'RETAIL','DEVYANI':'RETAIL',
    'JUBLFOOD':'RETAIL','SAPPHIRE':'RETAIL','CAMPUS':'RETAIL',
    'KALYANKJIL':'RETAIL','MANYAVAR':'RETAIL','TITAN':'RETAIL',
    'PAGEIND':'RETAIL','BBTC':'RETAIL','HONAUT':'RETAIL',
    'THELEELA':'RETAIL','LEMONTREE':'RETAIL',
    'INDHOTEL':'RETAIL','EIHOTEL':'RETAIL',
    'SWIGGY':'RETAIL',  # FIXED 20-Jun-2026: was wrongly 'IT' — food
                         # delivery/quick-commerce is consumer retail, not IT

    # ── Media & Entertainment ─────────────────────────────
    'ZEEL':'MEDIA','SUNTV':'MEDIA','PVRINOX':'MEDIA','SAREGAMA':'MEDIA',
    'NAZARA':'MEDIA','NETWORK18':'MEDIA',

    # ── Logistics & Shipping ──────────────────────────────
    'DELHIVERY':'LOGISTICS','BLUEDART':'LOGISTICS','CONCOR':'LOGISTICS',
    'AEGISLOG':'LOGISTICS','AEGISVOPAK':'LOGISTICS',

    # ── Aviation ─────────────────────────────────────────
    'INDIGO':'AVIATION',

    # ── Miscellaneous / Conglomerate ─────────────────────
    'GODREJIND':'CONGLOMERATE','TATAINVEST':'CONGLOMERATE',
    'BAJAJHLDNG':'CONGLOMERATE','3MINDIA':'CONGLOMERATE',
    'ADANIENT':'CONGLOMERATE','ADANIPORTS':'INFRA',

    # ── Remaining unmapped ────────────────────────────────
    'ABFRL':'CONSUMER',        # Aditya Birla Fashion — retail/apparel
    'ABLBL':'CONSUMER',        # Aditya Birla — consumer
    'ABREL':'CONSUMER',        # Aditya Birla Real Estate
    'AIIL':'INFRA',            # Authbridge / Infra
    'ALOKINDS':'CHEMICALS',    # Alok Industries — textiles/chemicals
    'APLLTD':'PHARMA',         # APL Apollo — steel tubes (INFRA)
    'ASTRAZEN':'PHARMA',       # AstraZeneca Pharma
    'BLS':'INFRA',             # BLS International — services
    'CCL':'FMCG',              # CCL Products — coffee/FMCG
    'COHANCE':'IT',            # Cohance Lifesciences
    'EIDPARRY':'FMCG',         # EID Parry — sugar/FMCG
    'ENRIN':'ENERGY',          # Energy / renewables
    'FSL':'IT',                # Firstsource Solutions — IT/BPO
    'HEXT':'IT',               # Hexaware Technologies — IT
    'IGIL':'INFRA',            # IGIL Infra
    'INDGN':'IT',              # Indegene — IT/healthcare
    'INOXINDIA':'INFRA',       # Inox India — industrial gases
    'ITCHOTELS':'RETAIL',      # ITC Hotels — hospitality
    'ITI':'INFRA',             # ITI Limited — telecom/infra
    'JPPOWER':'ENERGY',        # Jaiprakash Power — energy
    'JUBLPHARMA':'PHARMA',     # Jubilant Pharmova — pharma
    'JWL':'CONSUMER',          # Jupiter Wagons — consumer/infra
    'JYOTICNC':'INFRA',        # Jyoti CNC — capital goods
    'LTFOODS':'FMCG',          # LT Foods — rice/FMCG
    'LTM':'INFRA',             # L&T Metro Rail
    'MAHSEAMLES':'METALS',     # Maharastra Seamless — metals/pipes
    'NLCINDIA':'ENERGY',       # NLC India — energy/coal
    'PGEL':'ENERGY',           # PG Electroplast — energy
    'PTCIL':'ENERGY',          # PTC India — power trading
    'RELINFRA':'INFRA',        # Reliance Infra
    'RPOWER':'ENERGY',         # Reliance Power
    'SAILIFE':'PHARMA',        # Sai Life Sciences — pharma
    'SAMMAANCAP':'NBFC',       # Sammaan Capital — NBFC
    'SARDAEN':'ENERGY',        # Sarda Energy — energy/metals
    'SBICARD':'NBFC',          # SBI Cards — NBFC/payments
    'SCHNEIDER':'INFRA',       # Schneider Electric — capital goods
    'SWANCORP':'CONSUMER',     # Swan Energy — consumer/textiles
    'TARIL':'INFRA',           # TARIL — infra
    'TBOTEK':'IT',             # TBO Tek — travel tech/IT
    'UBL':'FMCG',              # United Breweries — FMCG
    'VMM':'METALS',            # Vishnu Metallics — metals
    'VTL':'INFRA',             # Vardhman Textiles — consumer
    'ZENTEC':'IT',             # Zen Technologies — IT/defence
    'ZFCVINDIA':'AUTO',        # ZF Commercial Vehicle — auto
}

def get_nifty_market_state(kite=None):
    """
    Fetch Nifty 50 + India VIX Index.
    VIX thresholds calibrated for INDIA (not US):
      India VIX is structurally higher than US VIX.
      VIX 16-20 = completely normal for India.
      < 13  -> CALM     (very rare — perfect conditions)
      13-16 -> NORMAL   (best trading conditions)
      16-20 -> ELEVATED (normal India range — trade freely)
      20-25 -> HIGH     (expiry/event day — reduce size 30%)
      25-30 -> EXTREME  (real fear — only strongest signals)
      > 30  -> CRISIS   (COVID/war level — avoid intraday)
    """
    result = {
        'state':      'UNKNOWN',
        'vix':        None,
        'vix_level':  'UNKNOWN',
        'nifty_chg':  0.0,
        'nifty_last': 0.0,
        'nifty_vwap': 0.0,
        'ema_trend':  'UNKNOWN',
    }
    try:
        import pytz as _pytz
        _ist       = _pytz.timezone('Asia/Kolkata')
        _now_ist   = datetime.now(_ist)
        _today_ist = _now_ist.date()

        _slot    = str(_now_ist.minute // 15)
        _now_str = _now_ist.strftime('%Y%m%d_%H') + _slot

        # ── Fetch Nifty 5-min ─────────────────────────────
        _ck = f"^NSEI_5m_{_now_str}"
        if _ck in _DATA_CACHE:
            _ndf, _ = _DATA_CACHE[_ck]
        else:
            for _old in [k for k in list(_DATA_CACHE.keys()) if k.startswith('^NSEI_5m_')]:
                del _DATA_CACHE[_old]
            _ndf = yf.Ticker('^NSEI').history(period='5d', interval='5m', auto_adjust=True)
            if _ndf is None or _ndf.empty:
                return result
            _ndf.columns = [c.split(' ')[0] if ' ' in str(c) else c for c in _ndf.columns]
            _ndf = _ndf[['Open','High','Low','Close','Volume']].dropna()
            if _ndf.index.tzinfo is None:
                _ndf.index = _ndf.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
            else:
                _ndf.index = _ndf.index.tz_convert('Asia/Kolkata')
            _DATA_CACHE[_ck] = (_ndf, 'yfinance')

        # ── Fetch VIX ─────────────────────────────────────
        _vix_val = None
        _vck = f"^VIX_5m_{_now_str}"
        try:
            if _vck in _DATA_CACHE:
                _vdf, _ = _DATA_CACHE[_vck]
            else:
                for _old in [k for k in list(_DATA_CACHE.keys()) if k.startswith('^VIX_5m_')]:
                    del _DATA_CACHE[_old]
                _vdf = yf.Ticker('^INDIAVIX').history(period='5d', interval='5m', auto_adjust=True)
                if _vdf is not None and not _vdf.empty:
                    _vdf.columns = [c.split(' ')[0] if ' ' in str(c) else c for c in _vdf.columns]
                    if _vdf.index.tzinfo is None:
                        _vdf.index = _vdf.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
                    else:
                        _vdf.index = _vdf.index.tz_convert('Asia/Kolkata')
                    _DATA_CACHE[_vck] = (_vdf, 'yfinance')
            if _vdf is not None and not _vdf.empty:
                _vdf_today = _vdf[_vdf.index.date == _today_ist]
                if len(_vdf_today) > 0:
                    _vix_val = float(_vdf_today['Close'].iloc[-1])
                else:
                    _vix_val = float(_vdf['Close'].iloc[-1])
        except Exception:
            pass

        if len(_ndf) < 5:
            return result

        # ── Today's Nifty candles (IST-aware) ─────────────
        _today_df = _ndf[_ndf.index.date == _today_ist]
        if len(_today_df) < 3:
            _all_dates = sorted(set(_ndf.index.date))
            _today_df  = _ndf[_ndf.index.date == _all_dates[-1]] if _all_dates else _ndf.tail(20)

        _last  = float(_today_df['Close'].iloc[-1])
        _open_ = float(_today_df['Open'].iloc[0])
        _chg   = (_last - _open_) / _open_ * 100
        _ema9  = float(_today_df['Close'].ewm(span=9,  adjust=False).mean().iloc[-1])
        _ema21 = float(_today_df['Close'].ewm(span=21, adjust=False).mean().iloc[-1])
        _tp    = (_today_df['High'] + _today_df['Low'] + _today_df['Close']) / 3
        _vwap  = float((_tp * _today_df['Volume']).sum() / (_today_df['Volume'].sum() + 1e-9))

        result['nifty_chg']  = round(_chg, 2)
        result['nifty_last'] = round(_last, 2)
        result['nifty_vwap'] = round(_vwap, 2)
        result['ema_trend']  = 'BULL' if _ema9 > _ema21 else 'BEAR'

        # ── VIX classification — India-calibrated ────────
        # India VIX is structurally higher than US VIX.
        # VIX 16-20 = completely normal for India.
        # VIX 20-25 = slightly elevated (Budget/expiry days).
        # VIX > 30  = true crisis (COVID/war level).
        if _vix_val is not None:
            result['vix'] = round(_vix_val, 2)
            if _vix_val < 13:
                result['vix_level'] = 'CALM'      # very rare — perfect conditions
            elif _vix_val < 16:
                result['vix_level'] = 'NORMAL'    # best conditions
            elif _vix_val < 20:
                result['vix_level'] = 'ELEVATED'  # normal India range — trade freely
            elif _vix_val < 25:
                result['vix_level'] = 'HIGH'      # expiry/event day — reduce size
            elif _vix_val < 30:
                result['vix_level'] = 'EXTREME'   # real fear — only strong signals
            else:
                result['vix_level'] = 'CRISIS'    # COVID/war level — avoid intraday

        # ── Market state: price % is primary ─────────────
        _above_vwap = _last > _vwap
        if   _chg >= 0.5 and _above_vwap:      result['state'] = 'BULL'
        elif _chg <= -0.5 and not _above_vwap:  result['state'] = 'BEAR'
        elif _chg >= 0.5:   result['state'] = 'BULL'
        elif _chg <= -0.5:  result['state'] = 'BEAR'
        elif _chg >= 0.2:   result['state'] = 'BULL'
        elif _chg <= -0.2:  result['state'] = 'BEAR'
        else:               result['state'] = 'SIDEWAYS'

    except Exception:
        pass

    return result


# ─────────────────────────────────────────────────────────────
#  NIFTY SWING STATE — Weekly/Daily SMA-based
#  Different from intraday get_nifty_market_state()
#  Used by Monthly Swing and SMA Weekly scanners
#  Based on SMA20 vs SMA50 (not intraday % change)
# ─────────────────────────────────────────────────────────────

def get_nifty_swing_state(timeframe='weekly'):
    """
    Returns Nifty state for swing trading decisions.

    timeframe='weekly' → for Monthly Swing (weekly candles)
    timeframe='daily'  → for SMA Weekly (daily candles)

    Returns dict:
        state:    BULLISH / CAUTION / BEARISH / UNKNOWN
        close:    last close
        sma20:    SMA20 value
        sma50:    SMA50 value
        slope:    SMA20 5-period slope %
        pct_from_sma20: % distance from SMA20
        vix:      India VIX
        vix_level: CALM/NORMAL/ELEVATED/HIGH/EXTREME/CRISIS
    """
    result = {
        'state':          'UNKNOWN',
        'close':          0.0,
        'sma20':          0.0,
        'sma50':          0.0,
        'slope':          0.0,
        'pct_from_sma20': 0.0,
        'vix':            None,
        'vix_level':      'UNKNOWN',
        'timeframe':      timeframe,
    }
    try:
        interval = '1wk' if timeframe == 'weekly' else '1d'
        period   = '2y'  if timeframe == 'weekly' else '6mo'

        _cache_key = f'nifty_swing_{timeframe}'
        import time as _time_mod
        if _cache_key in _DATA_CACHE:
            _cached, _cached_ts = _DATA_CACHE[_cache_key]
            if _time_mod.time() - _cached_ts < 3600:  # cache 1 hour
                return _cached

        nf  = yf.Ticker('^NSEI')
        df  = nf.history(period=period, interval=interval,
                         auto_adjust=True, actions=False)
        if df is None or len(df) < 25:
            return result

        df.columns = [c.split(' ')[0] if ' ' in str(c) else c for c in df.columns]
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['SMA50'] = df['Close'].rolling(50).mean()
        df = df.dropna()

        close = float(df['Close'].iloc[-1])
        sma20 = float(df['SMA20'].iloc[-1])
        sma50 = float(df['SMA50'].iloc[-1])

        # SMA20 slope (5-period)
        sma20_prev = float(df['SMA20'].iloc[-6]) if len(df) >= 6 else sma20
        slope = round((sma20 - sma20_prev) / sma20_prev * 100, 3) if sma20_prev > 0 else 0

        pct_from_sma20 = round((close - sma20) / sma20 * 100, 2) if sma20 > 0 else 0

        result['close']          = round(close, 2)
        result['sma20']          = round(sma20, 2)
        result['sma50']          = round(sma50, 2)
        result['slope']          = slope
        result['pct_from_sma20'] = pct_from_sma20

        # State classification — 5 states including LATE_BULL transition
        if close > sma20 > sma50 and slope > 0.3:
            result['state'] = 'BULLISH'    # ✅ strong uptrend
        elif close > sma20 > sma50 and slope <= 0.3:
            result['state'] = 'LATE_BULL'  # ⚠️ trend flattening — transition warning
        elif close > sma20 and sma20 > sma50:
            result['state'] = 'LATE_BULL'  # ⚠️ above MAs but slope flat
        elif close > sma20:
            result['state'] = 'CAUTION'    # ⚠️ half size
        elif close > sma50:
            result['state'] = 'CAUTION'    # ⚠️ weak — watch
        else:
            result['state'] = 'BEARISH'    # ❌ no new entries

        # VIX
        try:
            vf = yf.Ticker('^INDIAVIX')
            vdf = vf.history(period='5d', interval='1d',
                             auto_adjust=True, actions=False)
            if vdf is not None and len(vdf) > 0:
                vdf.columns = [c.split(' ')[0] if ' ' in str(c) else c for c in vdf.columns]
                vix = float(vdf['Close'].iloc[-1])
                result['vix'] = round(vix, 2)
                if   vix < 13: result['vix_level'] = 'CALM'
                elif vix < 16: result['vix_level'] = 'NORMAL'
                elif vix < 20: result['vix_level'] = 'ELEVATED'
                elif vix < 25: result['vix_level'] = 'HIGH'
                elif vix < 30: result['vix_level'] = 'EXTREME'
                else:          result['vix_level'] = 'CRISIS'
        except Exception:
            pass

        _DATA_CACHE[_cache_key] = (result, _time_mod.time())

    except Exception:
        pass

    return result


# ─────────────────────────────────────────────────────────────
#  BETA CALCULATION
#  Calculates stock beta vs Nifty using weekly returns
#  Used for dynamic scoring based on market state
# ─────────────────────────────────────────────────────────────

def calc_stock_beta(stock_df, nifty_df, periods=52):
    """
    Calculate beta of stock vs Nifty.
    Uses weekly returns for last 'periods' weeks.

    Beta < 0.7  = DEFENSIVE (low correlation to market)
    Beta 0.7-1.1 = NEUTRAL
    Beta > 1.4  = HIGH BETA (amplifies market moves)
    """
    try:
        stock_ret = stock_df['Close'].pct_change().dropna()
        nifty_ret = nifty_df['Close'].pct_change().dropna()

        # Align on common dates
        aligned = stock_ret.to_frame('stock').join(
                  nifty_ret.to_frame('nifty'), how='inner').dropna()

        if len(aligned) < 20:
            return 1.0  # default to market beta

        # Use last N periods
        aligned = aligned.tail(periods)

        cov = aligned['stock'].cov(aligned['nifty'])
        var = aligned['nifty'].var()
        beta = round(cov / var, 2) if var > 0 else 1.0

        # Cap at reasonable range
        return max(0.1, min(beta, 3.0))

    except Exception:
        return 1.0


def get_beta_score(beta, nifty_swing_state):
    """
    Dynamic beta score based on Nifty state.

    BULLISH market  → reward high beta (more upside)
    BEARISH market  → penalise high beta (protect capital)
    CAUTION market  → neutral to slight defensive preference
    """
    state = nifty_swing_state.get('state', 'UNKNOWN')

    if state == 'BULLISH':
        # Reward aggression — ride the bull
        if   beta >= 1.5: return +10, '🚀 High Beta', '#15803d'
        elif beta >= 1.1: return +5,  '📈 Medium Beta', '#16a34a'
        elif beta >= 0.7: return  0,  '➡️ Neutral Beta', '#64748b'
        else:             return -5,  '🛡️ Low Beta', '#d97706'

    elif state == 'CAUTION':
        # Slight defensive preference
        if   beta >= 1.5: return -5,  '⚠️ High Beta', '#d97706'
        elif beta >= 1.1: return  0,  '➡️ Medium Beta', '#64748b'
        elif beta >= 0.7: return +3,  '✅ Stable Beta', '#16a34a'
        else:             return +5,  '🛡️ Defensive', '#15803d'

    elif state == 'BEARISH':
        # Protect capital — penalise aggression
        if   beta >= 1.5: return -15, '🔴 High Beta (RISK)', '#dc2626'
        elif beta >= 1.2: return -10, '🔴 Sensitive Beta', '#dc2626'
        elif beta >= 1.0: return -5,  '⚠️ Above Market', '#d97706'
        elif beta >= 0.7: return +5,  '✅ Stable Beta', '#16a34a'
        else:             return +10, '🛡️ Defensive', '#15803d'

    else:  # UNKNOWN
        return 0, '❓ Beta', '#64748b'


def get_beta_grade(beta):
    """Return grade label, colour and icon for display."""
    if   beta < 0.5:  return 'DEFENSIVE',  '#15803d', '#f0fdf4', '#86efac', '🛡️'
    elif beta < 0.8:  return 'STABLE',     '#16a34a', '#dcfce7', '#bbf7d0', '🟢'
    elif beta < 1.1:  return 'NEUTRAL',    '#64748b', '#f8fafc', '#e2e8f0', '➡️'
    elif beta < 1.4:  return 'SENSITIVE',  '#d97706', '#fffbeb', '#fde68a', '🟠'
    else:             return 'HIGH BETA',  '#dc2626', '#fef2f2', '#fca5a5', '🔴'


# ─────────────────────────────────────────────────────────────
#  UNIFIED SECTOR RANKING — Single source of truth
#  Used by: Monthly Swing, SMA Weekly, Sector Leaders
#  11 unique sectors, no duplicates
# ─────────────────────────────────────────────────────────────

SECTOR_ETF_UNIFIED = {
    'BANK':        '^NSEBANK',           # Nifty Bank        ✅ verified working (10 bars)
    'IT':          '^CNXIT',             # Nifty IT          ✅ verified working (10 bars)
    'AUTO':        '^CNXAUTO',           # Nifty Auto        ✅ verified working (5 bars)
    'PHARMA':      '^CNXPHARMA',         # Nifty Pharma      ✅ verified working (10 bars)
    'FMCG':        '^CNXFMCG',           # Nifty FMCG        ✅ verified working (5 bars)
    'METALS':      '^CNXMETAL',          # Nifty Metal       ✅ verified working (5 bars)
    'ENERGY':      '^CNXENERGY',         # Nifty Energy      ✅ verified working (5 bars)
    'REALTY':      '^CNXREALTY',         # Nifty Realty      ✅ verified working (5 bars)
    'INFRA':       '^CNXINFRA',          # Nifty Infra       ✅ verified working (5 bars)
    'MEDIA':       '^CNXMEDIA',          # Nifty Media       ✅ verified working (5 bars)
    # FIXED 20-Jun-2026: '^CNXFIN' was DEAD on Yahoo (confirmed via
    # live test — control check failed, 0 bars returned). Sector
    # ranking for FINANCE had been silently falling back to neutral
    # defaults (0.0 RS, bullish=True) for an unknown period before
    # this was caught. Replaced with verified-working alternate.
    'FINANCE':     'NIFTY_FIN_SERVICE.NS',  # Nifty Fin Service ✅ verified working (5 bars) — replaces dead ^CNXFIN
    # UPGRADED 20-Jun-2026: confirmed real index data exists —
    # moved off PROXY (stock-average) onto actual ETF data
    'PSU_BANK':    '^CNXPSUBANK',        # Nifty PSU Bank    ✅ verified working (5 bars)
    'PVT_BANK':    'NIFTY_PVT_BANK.NS',  # Nifty Pvt Bank    ✅ verified working (5 bars)
    'CONSUMPTION': '^CNXCONSUM',         # Nifty Consumption ✅ verified working (5 bars)
    # ── Still PROXY (no working yfinance ticker found after 2 rounds
    #    of testing — 6 candidates each tried for HEALTHCARE, CHEMICALS,
    #    TELECOM, DEFENCE) ─────────────────────────────────────────
    'HEALTHCARE':  'PROXY',       # Hospitals — tried 4 candidates, none worked
    'CHEMICALS':   'PROXY',       # Specialty chemicals — tried 3 candidates, none worked
    'DEFENCE':     'PROXY',       # Defence — tried 3 candidates, none worked
    'TELECOM':     'PROXY',       # Telecom — ^CNXSERVICE rejected (only 3.85% telecom
                                   # weight, 57.7% Financial Services — would mislead)
    'TEXTILES':    'PROXY',       # Textiles — no NSE sectoral index exists
    'AGRI':        'PROXY',       # Agri/Fertilisers — no NSE sectoral index exists
    'LOGISTICS':   'PROXY',       # Logistics/Supply chain — no NSE sectoral index exists
}

# ── Proxy stocks for sectors without yfinance ETF ─────────────
# Average returns of top 5 stocks = sector performance proxy
SECTOR_PROXY_STOCKS = {
    'HEALTHCARE':  ['APOLLOHOSP','MAXHEALTH','FORTIS','KIMS','ASTER'],
    'PSU_BANK':    ['SBIN','PNB','CANBK','BANKBARODA','UNIONBANK'],
    'PVT_BANK':    ['HDFCBANK','ICICIBANK','KOTAKBANK','AXISBANK','INDUSINDBK'],
    'CHEMICALS':   ['PIDILITIND','SRF','AARTIIND','DEEPAKNTR','NAVINFLUOR'],
    'DEFENCE':     ['HAL','BEL','BDL','BEML','COCHINSHIP'],
    'CONSUMPTION': ['DMART','TRENT','TITAN','WHIRLPOOL','VOLTAS'],
    'TELECOM':     ['BHARTIARTL','INDUSTOWER','TATACOMM','HFCL','IDEA'],
    'TEXTILES':    ['PAGEIND','VARDHMAN','TRIDENT','WELSPUNIND','RAYMOND'],
    'AGRI':        ['PIIND','UPL','CHAMBLFERT','GSFC','COROMANDEL'],
    'LOGISTICS':   ['CONCOR','BLUEDART','DELHIVERY','TCI','MAHLOG'],
}

# ─────────────────────────────────────────────────────────────
#  INDUSTRY → SECTOR mapping
#  Maps every SECTOR_MAP industry label → SECTOR_ETF_UNIFIED key
#  This is the SINGLE authoritative mapping used everywhere
#  Covers all 22 industry categories found in SECTOR_MAP
# ─────────────────────────────────────────────────────────────
INDUSTRY_TO_SECTOR = {
    # Direct mappings
    'AUTO':           'AUTO',
    'BANKING':        'BANK',
    'CHEMICALS':      'CHEMICALS',
    'ENERGY':         'ENERGY',
    'FMCG':           'FMCG',
    'INFRA':          'INFRA',
    'IT':             'IT',
    'MEDIA':          'MEDIA',
    'METALS':         'METALS',
    'PHARMA':         'PHARMA',
    'REALTY':         'REALTY',
    'TELECOM':        'TELECOM',
    # Grouped mappings
    'INSURANCE':      'FINANCE',      # Insurance → Financial Services
    'NBFC':           'FINANCE',      # Non-Banking Finance → Finance
    'CAPITAL_MARKETS':'FINANCE',      # Brokers, AMCs → Finance
    'CONSUMER':       'CONSUMPTION',  # Consumer durables/discretionary
    'RETAIL':         'CONSUMPTION',  # Retail → Consumption
    'SOLAR':          'ENERGY',       # Solar/Renewables → Energy
    'CEMENT':         'INFRA',        # Cement → Infrastructure
    'AVIATION':       'INFRA',        # Aviation → Infrastructure
    'CONGLOMERATE':   'INFRA',        # Conglomerates → Infrastructure (mixed)
    'LOGISTICS':      'LOGISTICS',    # Logistics → Logistics proxy
}


def classify_stock_sector(sym):
    """
    SINGLE authoritative function to map stock → sector.
    Used by ALL parts of the app.

    Priority:
    1. SECTOR_MAP → INDUSTRY_TO_SECTOR  (most accurate, covers 500+ stocks)
    2. Hardcoded exceptions               (stocks not in SECTOR_MAP)
    3. UNKNOWN                            (truly unmapped → neutral C8 score)

    Returns: sector string (key in SECTOR_ETF_UNIFIED)
    """
    sym = sym.upper().replace('.NS', '').strip()

    # ── Step 1: SECTOR_MAP lookup (primary, most accurate) ────
    if sym in SECTOR_MAP:
        industry = SECTOR_MAP[sym].upper()
        if industry in INDUSTRY_TO_SECTOR:
            return INDUSTRY_TO_SECTOR[industry]
        # Industry exists in SECTOR_MAP but not in our map
        # Return industry directly if it matches a sector key
        if industry in SECTOR_ETF_UNIFIED:
            return industry
        # Unknown industry — return UNKNOWN for neutral handling
        return 'UNKNOWN'

    # ── Step 2: Hardcoded for stocks not in SECTOR_MAP ────────
    # Only for well-known stocks that are commonly scanned
    _HARDCODED = {
        # Healthcare (hospitals — separate from pharma)
        'APOLLOHOSP': 'HEALTHCARE', 'MAXHEALTH': 'HEALTHCARE',
        'FORTIS':     'HEALTHCARE', 'KIMS':      'HEALTHCARE',
        'ASTER':      'HEALTHCARE', 'NHPC':      'ENERGY',
        # PSU Banks
        'SBIN':       'PSU_BANK',   'PNB':       'PSU_BANK',
        'CANBK':      'PSU_BANK',   'BANKBARODA':'PSU_BANK',
        'UNIONBANK':  'PSU_BANK',   'INDIANB':   'PSU_BANK',
        'BANKINDIA':  'PSU_BANK',   'MAHABANK':  'PSU_BANK',
        'IOB':        'PSU_BANK',   'CENTRALBK': 'PSU_BANK',
        # Private Banks
        'HDFCBANK':   'PVT_BANK',   'ICICIBANK': 'PVT_BANK',
        'KOTAKBANK':  'PVT_BANK',   'AXISBANK':  'PVT_BANK',
        'INDUSINDBK': 'PVT_BANK',   'FEDERALBNK':'PVT_BANK',
        'BANDHANBNK': 'PVT_BANK',   'IDFCFIRSTB':'PVT_BANK',
        'AUBANK':     'PVT_BANK',   'RBLBANK':   'PVT_BANK',
        # Defence
        'HAL':        'DEFENCE',    'BEL':       'DEFENCE',
        'BDL':        'DEFENCE',    'BEML':      'DEFENCE',
        'COCHINSHIP': 'DEFENCE',    'GRSE':      'DEFENCE',
        'MAZDOCK':    'DEFENCE',    'DATAPATTNS':'DEFENCE',
        # Textiles
        'PAGEIND':    'TEXTILES',   'VARDHMAN':  'TEXTILES',
        'TRIDENT':    'TEXTILES',   'WELSPUNIND':'TEXTILES',
        'RAYMOND':    'TEXTILES',   'ARVIND':    'TEXTILES',
        'GRASIM':     'TEXTILES',   'ALOKTEXT':  'TEXTILES',
        # Logistics
        'CONCOR':     'LOGISTICS',  'BLUEDART':  'LOGISTICS',
        'DELHIVERY':  'LOGISTICS',  'TCI':       'LOGISTICS',
        'MAHLOG':     'LOGISTICS',  'ALLCARGO':  'LOGISTICS',
        # Agri/Fertilisers
        'PIIND':      'AGRI',       'UPL':       'AGRI',
        'CHAMBLFERT': 'AGRI',       'GSFC':      'AGRI',
        'COROMANDEL': 'AGRI',       'NFL':       'AGRI',
        'GNFC':       'AGRI',       'DEEPAKFERT':'AGRI',
    }
    if sym in _HARDCODED:
        return _HARDCODED[sym]

    # ── Step 3: Truly unknown → UNKNOWN ───────────────────────
    # Caller handles UNKNOWN → neutral C8 score (0 pts)
    return 'UNKNOWN'

# Cached sector rankings — refreshed once per hour
_SECTOR_RANK_CACHE = {}

def _get_proxy_sector_data(proxy_stocks, period='6mo'):
    """
    Calculate sector ETF-equivalent data using proxy stocks.
    Returns a synthetic DataFrame with averaged Close prices.
    """
    import pandas as _pd3
    _all_closes = []
    for _ps in proxy_stocks[:5]:  # max 5 stocks
        try:
            _pdf = yf.Ticker(_ps+'.NS').history(
                period=period, interval='1d',
                auto_adjust=True, actions=False)
            if _pdf is None or len(_pdf) < 10:
                continue
            _pdf.columns = [c.split(' ')[0] if ' ' in str(c) else c
                            for c in _pdf.columns]
            _pdf = _pdf[['Close']].dropna()
            # Normalise to 100 base for averaging
            if float(_pdf['Close'].iloc[0]) > 0:
                _pdf['Close'] = _pdf['Close'] / float(_pdf['Close'].iloc[0]) * 100
                _all_closes.append(_pdf['Close'])
        except Exception:
            continue

    if not _all_closes:
        return None

    # Align and average
    _combined = _pd3.concat(_all_closes, axis=1).dropna()
    _combined.columns = [f's{i}' for i in range(len(_combined.columns))]
    _synthetic = _pd3.DataFrame()
    _synthetic['Close'] = _combined.mean(axis=1)
    return _synthetic


def parse_csv_stock_list(uploaded_file):
    """
    Parse an uploaded CSV/Excel of NSE symbols into a clean
    list of '.NS' tickers for the scanner.

    Supports common NSE export formats:
      - 'Symbol' column (NSE sector/index lists)
      - 'SYMBOL' column (some broker exports)
      - First column fallback if no header matches

    Returns: (symbols_list, error_message_or_None)
    """
    import pandas as _pd2
    try:
        _fname = uploaded_file.name.lower()
        if _fname.endswith('.xlsx') or _fname.endswith('.xls'):
            _df = _pd2.read_excel(uploaded_file)
        else:
            _df = _pd2.read_csv(uploaded_file)

        if _df is None or len(_df) == 0:
            return [], "File is empty"

        # Find the symbol column — case-insensitive match
        _cols_lower = {c.lower().strip(): c for c in _df.columns}
        _sym_col = None
        for _candidate in ['symbol', 'symbols', 'ticker', 'tickers',
                            'stock', 'scrip', 'nse symbol', 'nse code']:
            if _candidate in _cols_lower:
                _sym_col = _cols_lower[_candidate]
                break

        if _sym_col is None:
            # Fallback — use first column
            _sym_col = _df.columns[0]

        _raw = _df[_sym_col].dropna().astype(str).tolist()

        # Clean symbols — remove .NS suffix if present, uppercase, strip
        _clean = []
        _seen  = set()
        for _s in _raw:
            _s = _s.strip().upper()
            _s = _s.replace('.NS', '').replace('NSE:', '').strip()
            if not _s or _s in _seen:
                continue
            # Skip header-like rows (e.g. "SYMBOL")
            if _s in ('SYMBOL', 'SYMBOLS', 'TICKER', 'NSE CODE'):
                continue
            _seen.add(_s)
            _clean.append(f'{_s}.NS')

        if not _clean:
            return [], "No valid symbols found in file"

        return _clean, None

    except Exception as _e:
        return [], f"Could not read file: {str(_e)[:100]}"


def get_unified_sector_rankings(formula='weekly'):
    """
    Single shared function for sector rankings.
    Used by ALL three tabs — consistent results everywhere.

    formula='weekly':  1M(50%) + 2W(30%) + 1W(20%)  — SMA Weekly
    formula='monthly': 3M(50%) + 1M(30%) + 2W(20%)  — Monthly Swing

    ETF sectors: use yfinance ETF directly
    PROXY sectors: average top stocks as synthetic ETF

    Returns:
        rank_map:   {sector: rank_number}    rank 1 = strongest
        rs_map:     {sector: weighted_rs%}
        status_map: {sector: (bullish, gap%)}
        detail_map: {sector: {r1,r2,r3,weighted}}
        ranked:     [(sector, score), ...] sorted desc
    """
    import time as _t
    global _SECTOR_RANK_CACHE
    _cache_key = f'sector_rankings_{formula}'
    if _cache_key in _SECTOR_RANK_CACHE:
        _cached, _ts = _SECTOR_RANK_CACHE[_cache_key]
        if _t.time() - _ts < 3600:   # 1-hour cache
            return _cached

    # Periods based on formula
    if formula == 'weekly':
        _w1, _w2, _w3 = 0.50, 0.30, 0.20
        _p1, _p2, _p3 = 20, 10, 5
    else:  # monthly
        _w1, _w2, _w3 = 0.50, 0.30, 0.20
        _p1, _p2, _p3 = 60, 20, 10

    # Fetch Nifty base returns (daily)
    _nf_r1 = _nf_r2 = _nf_r3 = 0.0
    try:
        _nf = yf.Ticker('^NSEI').history(
            period='6mo', interval='1d',
            auto_adjust=True, actions=False)
        _nf.columns = [c.split(' ')[0] if ' ' in str(c) else c for c in _nf.columns]
        _nf = _nf[['Close']].dropna()
        if len(_nf) >= _p1:
            _nf_r1 = float((_nf['Close'].iloc[-1]-_nf['Close'].iloc[-_p1])/_nf['Close'].iloc[-_p1]*100)
        if len(_nf) >= _p2:
            _nf_r2 = float((_nf['Close'].iloc[-1]-_nf['Close'].iloc[-_p2])/_nf['Close'].iloc[-_p2]*100)
        if len(_nf) >= _p3:
            _nf_r3 = float((_nf['Close'].iloc[-1]-_nf['Close'].iloc[-_p3])/_nf['Close'].iloc[-_p3]*100)
    except Exception:
        pass

    rank_map   = {}
    rs_map     = {}
    status_map = {}
    detail_map = {}

    for _sec, _etf in SECTOR_ETF_UNIFIED.items():
        try:
            # ── Get price data ────────────────────────────
            if _etf == 'PROXY':
                # Use proxy stocks for this sector
                _proxy_stocks = SECTOR_PROXY_STOCKS.get(_sec, [])
                if not _proxy_stocks:
                    rs_map[_sec]     = 0.0
                    status_map[_sec] = (True, 0.0)
                    detail_map[_sec] = {'r1':0,'r2':0,'r3':0,'weighted':0,'source':'PROXY'}
                    continue
                _sf = _get_proxy_sector_data(_proxy_stocks)
                if _sf is None or len(_sf) < _p3 + 2:
                    rs_map[_sec]     = 0.0
                    status_map[_sec] = (True, 0.0)
                    detail_map[_sec] = {'r1':0,'r2':0,'r3':0,'weighted':0,'source':'PROXY'}
                    continue
                _source = 'PROXY'
            else:
                # Use ETF directly
                _sf = yf.Ticker(_etf).history(
                    period='6mo', interval='1d',
                    auto_adjust=True, actions=False)
                if _sf is None or len(_sf) < _p3 + 2:
                    rs_map[_sec]     = 0.0
                    status_map[_sec] = (True, 0.0)
                    detail_map[_sec] = {'r1':0,'r2':0,'r3':0,'weighted':0,'source':'ETF'}
                    continue
                _sf.columns = [c.split(' ')[0] if ' ' in str(c) else c for c in _sf.columns]
                _sf = _sf[['Close']].dropna()
                _source = 'ETF'

            # ── Multi-period RS vs Nifty ──────────────────
            _e_r1 = float((_sf['Close'].iloc[-1]-_sf['Close'].iloc[-min(_p1,len(_sf)-1)])/_sf['Close'].iloc[-min(_p1,len(_sf)-1)]*100) if len(_sf)>=_p1 else 0
            _e_r2 = float((_sf['Close'].iloc[-1]-_sf['Close'].iloc[-min(_p2,len(_sf)-1)])/_sf['Close'].iloc[-min(_p2,len(_sf)-1)]*100) if len(_sf)>=_p2 else 0
            _e_r3 = float((_sf['Close'].iloc[-1]-_sf['Close'].iloc[-min(_p3,len(_sf)-1)])/_sf['Close'].iloc[-min(_p3,len(_sf)-1)]*100) if len(_sf)>=_p3 else 0

            _rs1 = _e_r1 - _nf_r1
            _rs2 = _e_r2 - _nf_r2
            _rs3 = _e_r3 - _nf_r3
            _weighted = round(_w1*_rs1 + _w2*_rs2 + _w3*_rs3, 2)

            # ── SMA20 vs SMA50 for bullish/bearish ────────
            _sf['SMA20'] = _sf['Close'].rolling(20).mean()
            _sf['SMA50'] = _sf['Close'].rolling(50).mean()
            _sf = _sf.dropna()
            _s20 = float(_sf['SMA20'].iloc[-1]) if len(_sf)>0 else 0
            _s50 = float(_sf['SMA50'].iloc[-1]) if len(_sf)>0 else 0
            _gap = (_s20-_s50)/_s50*100 if _s50>0 else 0

            rs_map[_sec]     = _weighted
            status_map[_sec] = (_s20 > _s50, round(_gap, 2))
            detail_map[_sec] = {
                'r1':      round(_rs1, 1),
                'r2':      round(_rs2, 1),
                'r3':      round(_rs3, 1),
                'weighted':_weighted,
                'source':  _source,
            }
        except Exception:
            rs_map[_sec]     = 0.0
            status_map[_sec] = (True, 0.0)
            detail_map[_sec] = {'r1':0,'r2':0,'r3':0,'weighted':0,'source':'ERROR'}

    # Build rank map (rank 1 = strongest)
    _ranked  = sorted(rs_map.items(), key=lambda x: x[1], reverse=True)
    rank_map = {s: i+1 for i, (s, _) in enumerate(_ranked)}

    _result = {
        'rank_map':   rank_map,
        'rs_map':     rs_map,
        'status_map': status_map,
        'detail_map': detail_map,
        'ranked':     _ranked,
        # ── Added for RS-vs-sector math (zero extra API cost — ──
        # already computed above for the sector ranking itself).
        # Lets get_rs_vs_sector() compute a stock's RS-vs-Nifty
        # using the SAME periods/weights as this sector ranking,
        # for a genuine apples-to-apples comparison.
        'nifty_returns': {'r1': _nf_r1, 'r2': _nf_r2, 'r3': _nf_r3},
        'periods':       (_p1, _p2, _p3),
        'weights':       (_w1, _w2, _w3),
    }
    _SECTOR_RANK_CACHE[_cache_key] = (_result, _t.time())
    return _result


def get_sector_score_for_stock(sym, formula='weekly'):
    """
    Get sector rank + C8 score for a stock.
    Uses classify_stock_sector() — single authoritative mapping.
    Returns: (sector_name, rank, rs, bullish, gap, score, score_label, clr)
    """
    sec = classify_stock_sector(sym)

    # Handle UNKNOWN sector → neutral, no penalty
    if sec == 'UNKNOWN':
        return 'UNKNOWN', 99, 0.0, True, 0.0, 0, '❓ Unknown sector', '#64748b'

    # Get unified rankings
    _rankings = get_unified_sector_rankings(formula)
    rank      = _rankings['rank_map'].get(sec, 10)
    rs        = _rankings['rs_map'].get(sec, 0.0)
    bull, gap = _rankings['status_map'].get(sec, (True, 0.0))

    # C8 score — BONUS ONLY, minimal penalty
    # Sector tells context, not individual stock quality
    # Strong sector = bonus, weak sector = small penalty only
    # Individual stock RS vs sector is the real differentiator
    if   rank <= 2 and bull:  _score = +10; _lbl = f'🥇 #{rank} {sec} (Top sector)'
    elif rank <= 4 and bull:  _score = +7;  _lbl = f'🥈 #{rank} {sec}'
    elif rank <= 6 and bull:  _score = +3;  _lbl = f'🥉 #{rank} {sec}'
    elif rank <= 9 and bull:  _score = 0;   _lbl = f'➡️ #{rank} {sec}'
    else:                     _score = -3;  _lbl = f'⚠️ #{rank} {sec} (Weak sector)'

    # Sector bearish = small additional penalty
    if not bull:
        _score = max(_score - 2, -3)

    _clr = ('#15803d' if _score >= 7 else
            '#16a34a' if _score >= 3 else
            '#64748b' if _score == 0 else
            '#d97706')

    return sec, rank, rs, bull, gap, _score, _lbl, _clr


# ─────────────────────────────────────────────────────────────
#  FILTER 1 — CLOSING POSITION IN CANDLE
#  week_pos = (Close - Low) / (High - Low)
#  > 0.75 = buyers won = +8 pts
#  < 0.25 = sellers won = REJECT
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
#  VOLATILITY SQUEEZE DETECTION (TTM Squeeze Logic)
#  Detects when Bollinger Bands compress inside Keltner Channel
#  = energy building up for an explosive move
#
#  Logic:
#    Bollinger Bands:   SMA20 ± 2.0× StdDev(20)
#    Keltner Channel:   SMA20 ± 1.5× ATR(20)
#    SQUEEZE ON:  BB inside KC  (BB_upper < KC_upper)
#    SQUEEZE OFF: BB outside KC (BB_upper > KC_upper)
#    FIRED:       Was squeezed → now expanding
# ─────────────────────────────────────────────────────────────

def detect_volatility_squeeze(df, bb_mult=2.0, kc_mult=1.5, atr_period=20):
    """
    Detects volatility squeeze using TTM Squeeze logic.
    Works on any timeframe (daily, weekly).

    Returns dict:
      squeeze_on:       bool  — currently compressed
      squeeze_fired:    bool  — just broke out of squeeze
      squeeze_weeks:    int   — how many bars squeeze has been building
      direction:        str   — 'BULLISH' / 'BEARISH' / 'NEUTRAL'
      bb_width_pct:     float — current BB width as % of price
      bb_expanding:     bool  — BB width growing (breakout)
      bb_width_change:  float — % change in BB width (last 3 bars)
      kc_width_pct:     float — KC width as % of price
      momentum:         float — squeeze momentum value
      score:            int   — scoring contribution
      label:            str   — display label
      clr:              str   — display colour
    """
    import numpy as _np

    _result = {
        'squeeze_on':      False,
        'squeeze_fired':   False,
        'squeeze_weeks':   0,
        'direction':       'NEUTRAL',
        'bb_width_pct':    0.0,
        'bb_expanding':    False,
        'bb_width_change': 0.0,
        'kc_width_pct':    0.0,
        'momentum':        0.0,
        'score':           0,
        'label':           '➡️ No squeeze',
        'clr':             '#64748b',
        'ico':             '➡️',
    }

    try:
        if len(df) < atr_period + 5:
            return _result

        df = df.copy()
        close = df['Close'].values.astype(float)
        high  = df['High'].values.astype(float)
        low   = df['Low'].values.astype(float)

        # ── Bollinger Bands ─────────────────────────────
        _sma20 = _np.array([
            close[max(0,i-atr_period):i].mean()
            for i in range(atr_period, len(close)+1)
        ])
        _std20 = _np.array([
            close[max(0,i-atr_period):i].std()
            for i in range(atr_period, len(close)+1)
        ])
        _bb_upper = _sma20 + bb_mult * _std20
        _bb_lower = _sma20 - bb_mult * _std20
        _bb_width = (_bb_upper - _bb_lower) / _sma20 * 100

        # ── Keltner Channel (using True Range ATR) ──────
        _tr = _np.array([
            max(high[i]-low[i],
                abs(high[i]-close[i-1]) if i>0 else 0,
                abs(low[i]-close[i-1])  if i>0 else 0)
            for i in range(len(close))
        ])
        _atr = _np.array([
            _tr[max(0,i-atr_period):i].mean()
            for i in range(atr_period, len(_tr)+1)
        ])
        n = min(len(_sma20), len(_atr))
        _sma20 = _sma20[-n:]
        _bb_upper = _bb_upper[-n:]
        _bb_lower = _bb_lower[-n:]
        _bb_width = _bb_width[-n:]
        _atr = _atr[-n:]

        _kc_upper = _sma20 + kc_mult * _atr
        _kc_lower = _sma20 - kc_mult * _atr
        _kc_width = (_kc_upper - _kc_lower) / _sma20 * 100

        # ── Squeeze detection per bar ───────────────────
        _squeeze_bars = (_bb_upper < _kc_upper) & (_bb_lower > _kc_lower)

        if len(_squeeze_bars) < 2:
            return _result

        # Current state
        _curr_squeeze = bool(_squeeze_bars[-1])
        _prev_squeeze = bool(_squeeze_bars[-2])

        # Squeeze just fired = was ON, now OFF
        _fired = _prev_squeeze and not _curr_squeeze

        # Count consecutive squeeze bars
        _sq_count = 0
        for i in range(len(_squeeze_bars)-1, -1, -1):
            if _squeeze_bars[i]:
                _sq_count += 1
            else:
                break
        # If fired, count how long it was squeezed
        if _fired:
            _sq_count = 0
            for i in range(len(_squeeze_bars)-2, -1, -1):
                if _squeeze_bars[i]:
                    _sq_count += 1
                else:
                    break

        # BB width change (expansion detection)
        _curr_bbw = float(_bb_width[-1])
        _prev_bbw = float(_bb_width[-4]) if len(_bb_width) >= 4 else _curr_bbw
        _bbw_change = (_curr_bbw - _prev_bbw) / _prev_bbw * 100 if _prev_bbw > 0 else 0
        _expanding = _bbw_change > 10  # expanding > 10% = real breakout

        # Direction (price vs SMA20)
        _close_now = float(close[-1])
        _sma20_now = float(_sma20[-1])
        if   _close_now > _sma20_now * 1.005: _direction = 'BULLISH'
        elif _close_now < _sma20_now * 0.995: _direction = 'BEARISH'
        else:                                  _direction = 'NEUTRAL'

        # Momentum (simplified — delta of midline)
        _mid_now  = float((_bb_upper[-1] + _bb_lower[-1]) / 2)
        _mid_prev = float((_bb_upper[-3] + _bb_lower[-3]) / 2) if len(_bb_upper) >= 3 else _mid_now
        _momentum = (_mid_now - _mid_prev) / _mid_prev * 100 if _mid_prev > 0 else 0

        # ── Score calculation ───────────────────────────
        _score = 0
        _label = ''
        _clr   = '#64748b'
        _ico   = '➡️'

        if _fired and _direction == 'BULLISH':
            # FIRED — score based on how long it was building
            _weeks_txt = f'{_sq_count} bar{"s" if _sq_count!=1 else ""}'
            if   _sq_count >= 15: _score = 15   # very long = possibly stuck
            elif _sq_count >= 10: _score = 22   # long — big move expected
            elif _sq_count >= 7:  _score = 25   # sweet spot ✅✅
            elif _sq_count >= 5:  _score = 20   # good ✅
            elif _sq_count >= 3:  _score = 15   # developing
            else:                 _score = 10   # early

            if _sq_count >= 7:
                _label = f'🔥 SQUEEZE FIRED — {_weeks_txt} compressed · STRONG'
            else:
                _label = f'🔥 SQUEEZE FIRED — {_weeks_txt} compressed'
            _clr = '#15803d'
            _ico = '🔥'
            if _expanding:
                _score += 5   # confirmed BB expansion
                _label += ' · BB Expanding'

        elif _fired and _direction == 'BEARISH':
            _score = -10
            _label = f'⬇️ Squeeze fired BEARISH — skip'
            _clr   = '#dc2626'
            _ico   = '⬇️'

        elif _curr_squeeze and _direction == 'BULLISH':
            # BUILDING — score based on compression duration
            if   _sq_count >= 7:  _score = 12; _lbl_sfx = f'{_sq_count} bars · Almost ready 🔜'
            elif _sq_count >= 5:  _score = 8;  _lbl_sfx = f'{_sq_count} bars · Watch closely'
            elif _sq_count >= 3:  _score = 5;  _lbl_sfx = f'{_sq_count} bars · Building'
            else:                 _score = 2;  _lbl_sfx = f'{_sq_count} bar · Early stage'
            _label = f'🟡 Squeeze building {_lbl_sfx}'
            _clr   = '#d97706'
            _ico   = '🟡'

        elif not _curr_squeeze and not _fired:
            # High volatility — bad time to enter
            if _curr_bbw > float(_bb_width[-min(10,len(_bb_width)):].mean()) * 1.3:
                _score = -8
                _label = '⚠️ High volatility — no squeeze'
                _clr   = '#dc2626'
                _ico   = '⚠️'
            else:
                _score = 0
                _label = '➡️ No squeeze detected'
                _clr   = '#64748b'
                _ico   = '➡️'

        _result.update({
            'squeeze_on':      _curr_squeeze,
            'squeeze_fired':   _fired,
            'squeeze_weeks':   _sq_count,
            'direction':       _direction,
            'bb_width_pct':    round(_curr_bbw, 2),
            'bb_expanding':    _expanding,
            'bb_width_change': round(_bbw_change, 1),
            'kc_width_pct':    round(float(_kc_width[-1]), 2),
            'momentum':        round(_momentum, 2),
            'score':           _score,
            'label':           _label,
            'clr':             _clr,
            'ico':             _ico,
        })

    except Exception:
        pass

    return _result



# ─────────────────────────────────────────────────────────────
#  ADX — Average Directional Index
#  Measures TREND STRENGTH (not direction)
#  ADX > 25 = strong trend = higher win rate
#  ADX < 20 = ranging/choppy = avoid
#
#  Also calculates +DI and -DI for direction
#  +DI > -DI = bullish direction
#  -DI > +DI = bearish direction
# ─────────────────────────────────────────────────────────────

def calc_adx(df, period=14):
    """
    Calculate ADX, +DI, -DI using correct Wilder RMA smoothing.
    Returns (adx, plus_di, minus_di) or (None, None, None).

    Auto-reduces period for stocks with limited data:
      len(df) >= 60: period = 14 (standard)
      len(df) >= 40: period = 10 (reduced)
      len(df) >= 25: period = 7  (minimal)
      len(df) <  25: return None
    """
    import numpy as _np
    try:
        # Auto-select period based on available data
        if   len(df) >= 60: period = 14
        elif len(df) >= 40: period = 10
        elif len(df) >= 25: period = 7
        else:               return None, None, None

        _min_bars = period * 3
        if len(df) < _min_bars:
            return None, None, None

        _hi = df['High'].values.astype(float)
        _lo = df['Low'].values.astype(float)
        _cl = df['Close'].values.astype(float)
        _n  = len(_cl)

        # True Range and Directional Movement
        _tr  = _np.zeros(_n)
        _pdm = _np.zeros(_n)
        _mdm = _np.zeros(_n)
        for i in range(1, _n):
            _tr[i]  = max(_hi[i]-_lo[i],
                          abs(_hi[i]-_cl[i-1]),
                          abs(_lo[i]-_cl[i-1]))
            _up   = _hi[i] - _hi[i-1]
            _down = _lo[i-1] - _lo[i]
            _pdm[i] = _up   if (_up > _down and _up > 0)   else 0.0
            _mdm[i] = _down if (_down > _up and _down > 0) else 0.0

        # Wilder RMA (Running Moving Average) — CORRECT formula
        # Seed with SMA of first `period` values, then apply EMA
        def _wilder_rma(arr, p):
            _s = _np.zeros(len(arr))
            if p >= len(arr):
                return _s
            # Seed: simple average of first p non-zero bars
            _seed_vals = arr[1:p+1]
            _s[p] = _seed_vals.mean() if len(_seed_vals) > 0 else 0.0
            for i in range(p+1, len(arr)):
                _s[i] = (_s[i-1] * (p - 1) + arr[i]) / p
            return _s

        _atr_s = _wilder_rma(_tr,  period)
        _pdm_s = _wilder_rma(_pdm, period)
        _mdm_s = _wilder_rma(_mdm, period)

        # +DI and -DI
        _pdi = _np.where(_atr_s > 0, 100.0 * _pdm_s / _atr_s, 0.0)
        _mdi = _np.where(_atr_s > 0, 100.0 * _mdm_s / _atr_s, 0.0)

        # DX
        _denom = _pdi + _mdi
        _dx    = _np.where(_denom > 0,
                           100.0 * _np.abs(_pdi - _mdi) / _denom,
                           0.0)

        # ADX = Wilder RMA of DX
        _adx_arr = _wilder_rma(_dx, period)

        # Get last valid values
        _adx     = float(_adx_arr[-1])
        _plus_di = float(_pdi[-1])
        _minus_di= float(_mdi[-1])

        # Validate
        if not (0 < _adx <= 100):
            return None, None, None
        if not (0 <= _plus_di <= 100):
            return None, None, None

        return round(_adx, 1), round(_plus_di, 1), round(_minus_di, 1)

    except Exception:
        return None, None, None


def get_adx_score(adx, plus_di, minus_di):
    """
    Returns (score, label, colour) based on ADX reading.

    Scoring:
      ADX > 40:    +10 pts  Very strong trend
      ADX 30-40:    +8 pts  Strong trend
      ADX 25-30:    +6 pts  Good trend
      ADX 20-25:    +3 pts  Developing
      ADX 15-20:     0 pts  Weak
      ADX < 15:     -5 pts  Ranging/choppy

    Direction bonus:
      +DI > -DI AND ADX > 25: +3 pts extra
    """
    if adx is None:
        return 0, '➡️ ADX N/A', '#64748b'

    # Base score
    if   adx > 40:  _sc = 10; _str = 'Very Strong'
    elif adx > 30:  _sc = 8;  _str = 'Strong'
    elif adx > 25:  _sc = 6;  _str = 'Good'
    elif adx > 20:  _sc = 3;  _str = 'Developing'
    elif adx > 15:  _sc = 0;  _str = 'Weak'
    else:           _sc = -5; _str = 'Ranging'

    # Direction bonus
    _dir_bonus = 0
    _dir_txt   = ''
    if plus_di is not None and minus_di is not None:
        if plus_di > minus_di and adx > 25:
            _dir_bonus = 3
            _dir_txt   = ' +DI↑'

    _total = _sc + _dir_bonus

    # Label
    _ico  = ('📈' if adx > 25 else '➡️' if adx > 15 else '📉')
    _lbl  = f'{_ico} ADX {adx:.0f} {_str}{_dir_txt} ({"+"+str(_total) if _total>=0 else str(_total)}pts)'

    # Colour
    _clr  = ('#15803d' if _total >= 6 else
             '#16a34a' if _total >= 3 else
             '#d97706' if _total == 0 else
             '#dc2626')

    return _total, _lbl, _clr


def get_candle_close_position(df, bars=1):
    """
    Returns (week_pos, score, label, reject) for the last N bars.
    bars=1  → current candle only
    bars=3  → average of last 3 candles (more reliable)
    """
    try:
        _positions = []
        for i in range(1, bars + 1):
            if i >= len(df): break
            _h = float(df['High'].iloc[-i])
            _l = float(df['Low'].iloc[-i])
            _c = float(df['Close'].iloc[-i])
            if _h == _l: continue
            _positions.append((_c - _l) / (_h - _l))

        if not _positions:
            return 0.5, 0, '➡️ Neutral', False

        week_pos = round(sum(_positions) / len(_positions), 3)

        if   week_pos >= 0.75: return week_pos, +8,  '🟢 Strong close (top 25%)',   False
        elif week_pos >= 0.50: return week_pos, +3,  '🟡 Above mid',                False
        elif week_pos >= 0.25: return week_pos,  0,  '🟠 Below mid',                False
        else:                  return week_pos, -99, '🔴 Weak close (bottom 25%)',   True  # REJECT
    except Exception:
        return 0.5, 0, '➡️ Unknown', False


# ─────────────────────────────────────────────────────────────
#  FILTER 2 — PRIOR CANDLE COMPARISON
#  Compares current candle body vs previous candle body
#  Growing body = conviction building = +5 pts
#  Shrinking body = momentum fading = -5 pts
# ─────────────────────────────────────────────────────────────
def get_candle_body_momentum(df):
    """
    Compares current vs previous candle body size.
    Returns (ratio, score, label)
    """
    try:
        if len(df) < 2:
            return 1.0, 0, '➡️ No data'

        _curr_body = abs(float(df['Close'].iloc[-1]) - float(df['Open'].iloc[-1]))
        _prev_body = abs(float(df['Close'].iloc[-2]) - float(df['Open'].iloc[-2]))

        if _prev_body <= 0:
            return 1.0, 0, '➡️ Neutral'

        ratio = round(_curr_body / _prev_body, 2)

        if   ratio >= 1.5:  return ratio, +8,  f'🔥 Strong ({ratio:.1f}× prev body)'
        elif ratio >= 1.2:  return ratio, +5,  f'✅ Growing ({ratio:.1f}× prev body)'
        elif ratio >= 0.8:  return ratio,  0,  f'➡️ Similar ({ratio:.1f}× prev body)'
        elif ratio >= 0.5:  return ratio, -5,  f'⚠️ Shrinking ({ratio:.1f}× prev body)'
        else:               return ratio, -8,  f'❌ Collapsing ({ratio:.1f}× prev body)'
    except Exception:
        return 1.0, 0, '➡️ Unknown'


# ─────────────────────────────────────────────────────────────
#  FILTER 3 — DRAWDOWN-BASED POSITION SIZING
#  Adjusts risk% based on Nifty state + personal drawdown
#  Protects capital during bad runs automatically
# ─────────────────────────────────────────────────────────────
def get_dynamic_risk_pct(base_risk_pct, nifty_state, drawdown_pct):
    """
    Returns adjusted risk% based on:
      - Nifty swing state (BULLISH/CAUTION/BEARISH)
      - Personal drawdown % (running loss from peak)

    drawdown_pct: positive number = % drawdown
      e.g. 5.0 = currently 5% below peak capital

    Returns: (adj_risk_pct, label, color, reduce_reason)
    """
    # Step 1 — Nifty state multiplier
    _nifty_mult = {
        'BULLISH':    1.00,   # full size
        'LATE_BULL':  0.75,   # reduce — transition warning
        'CAUTION':    0.50,   # half size
        'EARLY_BEAR': 0.35,   # significant reduction
        'BEARISH':    0.25,   # minimal
        'UNKNOWN':    0.75,
    }.get(nifty_state, 0.75)

    # Step 2 — Drawdown multiplier
    if   drawdown_pct < 3:   _dd_mult = 1.00; _dd_label = 'Normal'
    elif drawdown_pct < 7:   _dd_mult = 0.75; _dd_label = 'Caution'
    elif drawdown_pct < 12:  _dd_mult = 0.50; _dd_label = 'Reduced'
    else:                    _dd_mult = 0.25; _dd_label = 'Danger'

    # Combined
    adj = round(base_risk_pct * _nifty_mult * _dd_mult, 2)
    adj = max(0.25, min(adj, base_risk_pct))  # floor 0.25%, ceiling = base

    # Label
    if   adj >= base_risk_pct: _lbl = f'✅ Full size ({adj:.2f}%)'
    elif adj >= base_risk_pct * 0.75: _lbl = f'⚠️ Slightly reduced ({adj:.2f}%)'
    elif adj >= base_risk_pct * 0.50: _lbl = f'🔴 Half size ({adj:.2f}%)'
    else:                             _lbl = f'⛔ Minimum size ({adj:.2f}%)'

    _reasons = []
    if _nifty_mult < 1.0: _reasons.append(f'Nifty {nifty_state}')
    if _dd_mult    < 1.0: _reasons.append(f'Drawdown {drawdown_pct:.1f}% ({_dd_label})')
    _reason = ' + '.join(_reasons) if _reasons else 'No adjustment'

    _clr = '#15803d' if adj >= base_risk_pct else \
           '#d97706' if adj >= base_risk_pct*0.5 else '#dc2626'

    return adj, _lbl, _clr, _reason


def get_drawdown_pct(capital):
    """
    Get current drawdown% from Streamlit session state.
    User inputs peak capital; app tracks running P&L.
    """
    import streamlit as _st2
    peak   = _st2.session_state.get('peak_capital', capital)
    curr   = _st2.session_state.get('current_capital', capital)
    if peak <= 0: return 0.0
    dd = max(0.0, (peak - curr) / peak * 100)
    return round(dd, 2)


# ─────────────────────────────────────────────────────────────
#  RS vs OWN SECTOR
#  Compares stock return vs its sector ETF/proxy over 20 days
#  Stock outperforming sector = genuine leader ✅
#  Stock underperforming sector = laggard ❌
# ─────────────────────────────────────────────────────────────

def get_rs_vs_sector(df, sector_name, rankings):
    """
    RS vs own sector — CORRECTED VERSION (20-Jun-2026).

    FIXED BUG: the previous version compared the stock's RAW
    return directly against rankings['rs_map'][sector], but
    rs_map stores sector RS *already relative to Nifty* (e.g.
    +3% means sector beat Nifty by 3%). Subtracting a raw stock
    return from an already-relative sector number is a unit
    mismatch — it silently produced wrong leader/laggard signals.

    CORRECT METHOD:
      1. Compute the stock's OWN multi-period return using the
         exact same periods/weights as the sector ranking formula
      2. Subtract Nifty's multi-period return (from the cached
         rankings dict — zero extra API call) → stock_RS_vs_Nifty
      3. Compare to sector_RS_vs_Nifty (already in rankings['rs_map'])
      4. diff = stock_RS_vs_Nifty - sector_RS_vs_Nifty
         Both terms now share the same "vs Nifty" units — a true
         apples-to-apples leader/laggard signal.

    df:          stock daily/weekly OHLCV (already downloaded —
                 no extra API call needed)
    sector_name: sector string from classify_stock_sector()
    rankings:    the FULL dict from get_unified_sector_rankings()
                 (not just rs_map — needs nifty_returns/periods/
                 weights too)

    Returns (diff_pct, score, label, clr)
    """
    try:
        if df is None or rankings is None:
            return 0.0, 0, '', '#64748b'

        _rs_map = rankings.get('rs_map', {})
        if sector_name not in _rs_map:
            return 0.0, 0, '', '#64748b'

        _nifty_rets = rankings.get('nifty_returns')
        _periods    = rankings.get('periods')
        _weights    = rankings.get('weights')
        if not _nifty_rets or not _periods or not _weights:
            # Rankings dict is from before this fix / malformed —
            # fail safe rather than guess
            return 0.0, 0, '', '#64748b'

        _p1, _p2, _p3 = _periods
        _w1, _w2, _w3 = _weights
        _nf_r1 = _nifty_rets.get('r1', 0.0)
        _nf_r2 = _nifty_rets.get('r2', 0.0)
        _nf_r3 = _nifty_rets.get('r3', 0.0)

        _cl = df['Close'].dropna()
        if len(_cl) < min(_p3, 5) + 2:
            return 0.0, 0, '', '#64748b'

        def _ret(period):
            _n = min(period, len(_cl) - 1)
            if _n <= 0:
                return 0.0
            return float((_cl.iloc[-1] - _cl.iloc[-_n]) / _cl.iloc[-_n] * 100)

        _s_r1 = _ret(_p1)
        _s_r2 = _ret(_p2)
        _s_r3 = _ret(_p3)

        # Stock's RS vs Nifty — SAME formula as sector ranking
        _stock_rs1 = _s_r1 - _nf_r1
        _stock_rs2 = _s_r2 - _nf_r2
        _stock_rs3 = _s_r3 - _nf_r3
        _stock_rs_vs_nifty = round(_w1*_stock_rs1 + _w2*_stock_rs2 + _w3*_stock_rs3, 2)

        # Sector's RS vs Nifty — already computed, cached, reused
        _sector_rs_vs_nifty = float(_rs_map[sector_name])

        # TRUE apples-to-apples comparison — both in "vs Nifty" units
        _diff = round(_stock_rs_vs_nifty - _sector_rs_vs_nifty, 2)

        # Bonus-weighted scoring — rewards leaders, minimal penalty
        # for laggards (same philosophy as sector-rank scoring)
        if   _diff >= 8.0:  _sc = 15; _lbl = f'🌟 Strong sector leader (+{_diff:.1f}pp vs sector)'
        elif _diff >= 5.0:  _sc = 10; _lbl = f'🏆 Sector leader (+{_diff:.1f}pp vs sector)'
        elif _diff >= 2.0:  _sc = 6;  _lbl = f'✅ Outperforming sector (+{_diff:.1f}pp)'
        elif _diff >= -3.0: _sc = 0;  _lbl = f'➡️ Inline with sector ({_diff:+.1f}pp)'
        elif _diff >= -8.0: _sc = -2; _lbl = f'⚠️ Slightly behind sector ({_diff:.1f}pp)'
        else:               _sc = -4; _lbl = f'❌ Lagging sector ({_diff:.1f}pp)'

        _clr = '#15803d' if _sc > 0 else '#64748b' if _sc == 0 else '#d97706'
        return _diff, _sc, _lbl, _clr

    except Exception:
        return 0.0, 0, '', '#64748b'


def get_htf_alignment(df, current_tf='daily'):
    """
    Higher timeframe alignment using ALREADY FETCHED data.
    Resamples daily df to weekly — NO extra API call.

    df:         stock OHLCV (daily for SW, weekly for MS)
    current_tf: 'daily' → check weekly | 'weekly' → check monthly

    Returns (score, label, clr)
    """
    try:
        if df is None or len(df) < 10:
            return 0, '', '#64748b'

        if current_tf == 'daily':
            # Resample daily → weekly to check higher TF
            _tf_lbl = '1wk'
            _df = df.resample('W', on=df.index.name if df.index.name else None).agg({
                'Close': 'last', 'High': 'max',
                'Low': 'min',   'Open': 'first'
            }).dropna() if hasattr(df.index, 'freq') else df

            # If resample doesn't work, use weekly approximation from daily
            # Group every 5 rows as a week
            _closes = df['Close'].dropna().values
            _weekly = [float(_closes[max(0,i-4):i+1].mean())
                       for i in range(4, len(_closes), 5)]
            if len(_weekly) < 5:
                return 0, '', '#64748b'
        else:
            # For monthly: group every ~21 days
            _closes = df['Close'].dropna().values
            _weekly = [float(_closes[max(0,i-20):i+1].mean())
                       for i in range(20, len(_closes), 21)]
            _tf_lbl = '1mo'
            if len(_weekly) < 3:
                return 0, '', '#64748b'

        import numpy as _np2
        _weekly = _np2.array(_weekly)
        _p20 = min(20, len(_weekly))
        _p50 = min(50, len(_weekly))

        _s20   = float(_np2.mean(_weekly[-_p20:]))
        _s50   = float(_np2.mean(_weekly[-_p50:]))
        _price = float(_weekly[-1])
        _s20p  = float(_np2.mean(_weekly[-min(25,len(_weekly)):-min(5,len(_weekly))])) if len(_weekly) >= 6 else _s20

        _above  = _price > _s20 > _s50
        _rising = _s20 > _s20p

        # Smaller scores — reward alignment, minimal penalty
        if   _above and _rising:
            _sc = 8;  _lbl = f'✅ HTF {_tf_lbl}: Uptrend confirmed'
            _clr = '#15803d'
        elif _above:
            _sc = 4;  _lbl = f'✅ HTF {_tf_lbl}: Bullish'
            _clr = '#16a34a'
        elif _price > _s50:
            _sc = 0;  _lbl = f'⚠️ HTF {_tf_lbl}: Caution'
            _clr = '#d97706'
        else:
            _sc = -2; _lbl = f'❌ HTF {_tf_lbl}: Bearish'
            _clr = '#dc2626'

        return _sc, _lbl, _clr

    except Exception:
        return 0, '', '#64748b'


# ─────────────────────────────────────────────────────────────
#  LATE BULL TRANSITION DETECTION
#  Detects when market is transitioning from BULLISH to BEARISH
#  This is the MOST DANGEROUS period for swing traders
#  = trend still technically up but momentum slowing
# ─────────────────────────────────────────────────────────────

def get_nifty_transition_state(nifty_df):
    """
    Detects 5 market states with finer granularity:

    BULLISH:     SMA20 > SMA50, SMA20 rising strongly (slope > 0.5%)
    LATE_BULL:   SMA20 > SMA50, SMA20 flattening (slope 0-0.5%)
    CAUTION:     SMA20 flattening or just crossed below SMA50
    EARLY_BEAR:  SMA20 < SMA50, declining but not deep
    BEARISH:     SMA20 < SMA50, declining significantly

    Returns: (state, slope_pct, gap_pct, label, clr, bg)
    """
    try:
        _cl = nifty_df['Close'].dropna()
        if len(_cl) < 55:
            return 'UNKNOWN', 0, 0, 'Unknown', '#64748b', '#f8fafc'

        _sma20  = float(_cl.rolling(20).mean().iloc[-1])
        _sma50  = float(_cl.rolling(50).mean().iloc[-1])
        _sma20p = float(_cl.rolling(20).mean().iloc[-6])  # 5 bars ago
        _price  = float(_cl.iloc[-1])

        _slope  = round((_sma20 - _sma20p) / _sma20p * 100, 3) if _sma20p > 0 else 0
        _gap    = round((_sma20 - _sma50) / _sma50 * 100, 2) if _sma50 > 0 else 0
        _above  = _sma20 > _sma50

        if   _above and _slope >= 0.5:
            _st='BULLISH';    _lbl='🟢 BULLISH';    _clr='#15803d'; _bg='#f0fdf4'
        elif _above and _slope >= 0.1:
            _st='LATE_BULL';  _lbl='🟡 LATE BULL — Trend Flattening'; _clr='#d97706'; _bg='#fffbeb'
        elif _above and _slope >= 0:
            _st='CAUTION';    _lbl='⚠️ CAUTION';    _clr='#d97706'; _bg='#fffbeb'
        elif not _above and _gap >= -2:
            _st='EARLY_BEAR'; _lbl='🟠 EARLY BEAR'; _clr='#ea580c'; _bg='#fff7ed'
        else:
            _st='BEARISH';    _lbl='🔴 BEARISH';    _clr='#dc2626'; _bg='#fef2f2'

        return _st, _slope, _gap, _lbl, _clr, _bg

    except Exception:
        return 'UNKNOWN', 0, 0, 'Unknown', '#64748b', '#f8fafc'



#  FILTER A — RS vs Own Sector
#  Measures if stock is outperforming its sector
#  = sector leaders vs sector laggards
#  Only sector leaders tend to sustain moves
# ─────────────────────────────────────────────────────────────
def get_sector_momentum(results_so_far):
    """
    Calculate average % change per sector from scan results collected so far.
    Returns dict: {sector: avg_change_pct}
    """
    _sector_data = {}
    for r in results_so_far:
        _sym    = r.get('symbol','').replace('.NS','')
        _sector = SECTOR_MAP.get(_sym, '')
        if not _sector:
            continue
        _chg = r.get('change_pct', 0.0)
        if _sector not in _sector_data:
            _sector_data[_sector] = []
        _sector_data[_sector].append(_chg)
    return {s: round(sum(v)/len(v), 2) for s, v in _sector_data.items() if v}


def compute_relative_strength(stock_chg_pct, nifty_chg_pct):
    """
    RS = Stock change% - Nifty change%
    Positive = outperforming market (strong stock)
    Negative = underperforming market (weak stock)
    """
    if nifty_chg_pct is None:
        return None
    return round(float(stock_chg_pct) - float(nifty_chg_pct), 2)


def fetch_multi_timeframe(symbol, kite=None):
    """
    Fetch 1min, 5min, 15min data for a symbol and determine
    trend alignment across timeframes.
    Returns: dict with trend for each TF and alignment score
    """
    _tf_map = {
        '1m':  ('1minute',  '1m',  '5d'),
        '5m':  ('5minute',  '5m',  '5d'),
        '15m': ('15minute', '15m', '5d'),
    }
    _results = {}

    for _tf_label, (_kite_iv, _yf_iv, _yf_period) in _tf_map.items():
        try:
            _ck = f"{symbol}_{_tf_label}_{datetime.now().strftime('%Y%m%d_%H')}"
            if _ck in _DATA_CACHE:
                _df, _ = _DATA_CACHE[_ck]
            else:
                # Try Kite first
                _df = None
                if kite is not None:
                    try:
                        _token = get_instrument_token(kite, symbol)
                        if _token:
                            _ist     = pytz.timezone('Asia/Kolkata')
                            _today   = datetime.now(_ist).date()
                            _from_dt = datetime.combine(_today - timedelta(days=5), datetime.min.time())
                            _to_dt   = datetime.now(_ist).replace(tzinfo=None)
                            _records = kite.historical_data(
                                instrument_token=_token, from_date=_from_dt,
                                to_date=_to_dt, interval=_kite_iv,
                                continuous=False, oi=False)
                            if _records:
                                _df = pd.DataFrame(_records)
                                _df.rename(columns={'date':'Datetime','open':'Open','high':'High',
                                                    'low':'Low','close':'Close','volume':'Volume'}, inplace=True)
                                _df.set_index('Datetime', inplace=True)
                                _df.index = pd.to_datetime(_df.index)
                                _df = _df[['Open','High','Low','Close','Volume']].dropna()
                    except Exception:
                        pass

                # yfinance fallback
                if _df is None or len(_df) < 10:
                    _ticker = yf.Ticker(symbol)
                    _df     = _ticker.history(period=_yf_period, interval=_yf_iv, auto_adjust=True)
                    if _df is not None and not _df.empty:
                        _df.columns = [c.split(' ')[0] if ' ' in str(c) else c for c in _df.columns]
                        _df = _df[['Open','High','Low','Close','Volume']].dropna()

                if _df is None or len(_df) < 10:
                    continue
                _DATA_CACHE[_ck] = (_df, 'kite' if kite else 'yfinance')

            # Calculate simple trend for this timeframe
            _ema9  = float(_df['Close'].ewm(span=9,  adjust=False).mean().iloc[-1])
            _ema21 = float(_df['Close'].ewm(span=21, adjust=False).mean().iloc[-1])
            _last  = float(_df['Close'].iloc[-1])
            _tp    = (_df['High'] + _df['Low'] + _df['Close']) / 3
            _vol   = _df['Volume']
            _vwap  = float((_tp * _vol).cumsum().iloc[-1] / (_vol.cumsum().iloc[-1] + 1e-9))
            _rsi_d = _df['Close'].diff()
            _gain  = _rsi_d.clip(lower=0).ewm(com=6, adjust=False).mean()
            _loss  = (-_rsi_d.clip(upper=0)).ewm(com=6, adjust=False).mean()
            _rsi   = float(100 - (100 / (1 + _gain.iloc[-1] / (_loss.iloc[-1] + 1e-9))))

            _trend = 'BULL' if (_ema9 > _ema21 and _last > _vwap and _rsi > 50) \
                     else ('BEAR' if (_ema9 < _ema21 and _last < _vwap and _rsi < 50) \
                     else 'NEUTRAL')
            _results[_tf_label] = {
                'trend': _trend, 'ema9': round(_ema9,2), 'ema21': round(_ema21,2),
                'vwap': round(_vwap,2), 'rsi': round(_rsi,1), 'last': round(_last,2)
            }
        except Exception:
            continue

    # Calculate alignment score
    _trends = [_results[tf]['trend'] for tf in ['1m','5m','15m'] if tf in _results]
    _bull_count = _trends.count('BULL')
    _bear_count = _trends.count('BEAR')

    if _bull_count == 3:
        _alignment = 'STRONG_BULL'
        _score = 20
    elif _bull_count == 2:
        _alignment = 'BULL'
        _score = 10
    elif _bull_count == 1 and _bear_count == 0:
        _alignment = 'WEAK_BULL'
        _score = 4
    elif _bear_count == 3:
        _alignment = 'STRONG_BEAR'
        _score = -20
    elif _bear_count == 2:
        _alignment = 'BEAR'
        _score = -10
    elif _bear_count == 1 and _bull_count == 0:
        _alignment = 'WEAK_BEAR'
        _score = -4
    else:
        _alignment = 'CONFLICTING'
        _score = -5

    _results['alignment'] = _alignment
    _results['mtf_score'] = _score
    return _results




def _cache_key(symbol, interval):
    # 1min/3min data: refresh every 5 minutes (stale data = missed signals)
    # 5min/15min data: refresh every 15 minutes
    # 60min data: refresh every hour
    _now = datetime.now()
    if interval in ('1minute', '3minute'):
        _bucket = _now.strftime('%Y%m%d_%H') + str(_now.minute // 5)
    elif interval in ('5minute', '15minute'):
        _bucket = _now.strftime('%Y%m%d_%H') + str(_now.minute // 15)
    else:
        _bucket = _now.strftime('%Y%m%d_%H')
    return f"{symbol}_{interval}_{_bucket}"


def fetch_intraday(symbol, interval="1minute", period="1d", kite=None):
    """
    Fetch intraday OHLCV — sequential, Kite-first with yfinance fallback.
    Hourly cache prevents redundant re-fetches within same scan session.
    """
    # ── Check cache first ─────────────────────────────────
    _ck = _cache_key(symbol, interval)
    if _ck in _DATA_CACHE:
        return _DATA_CACHE[_ck]

    # ── Kite path (real-time) ─────────────────────────────
    if kite is not None:
        try:
            token = get_instrument_token(kite, symbol)
            if token is not None:
                ist     = pytz.timezone("Asia/Kolkata")
                today   = datetime.now(ist).date()
                from_dt = datetime.combine(today - timedelta(days=5), datetime.min.time())
                to_dt   = datetime.now(ist).replace(tzinfo=None)
                records = kite.historical_data(
                    instrument_token = token,
                    from_date        = from_dt,
                    to_date          = to_dt,
                    interval         = interval,
                    continuous       = False,
                    oi               = False
                )
                if records:
                    df = pd.DataFrame(records)
                    df.rename(columns={
                        'date':'Datetime','open':'Open','high':'High',
                        'low':'Low','close':'Close','volume':'Volume'
                    }, inplace=True)
                    df.set_index('Datetime', inplace=True)
                    if not isinstance(df.index, pd.DatetimeIndex):
                        df.index = pd.to_datetime(df.index)
                    df = df[['Open','High','Low','Close','Volume']].dropna()
                    if len(df) >= 30:
                        _DATA_CACHE[_ck] = (df, 'kite')
                        return df, 'kite'
        except Exception:
            pass  # fall through to yfinance

    # ── yfinance fallback — thread-safe via Ticker ────────
    if not YF_AVAILABLE:
        return None, 'none'

    yf_interval_map = {
        '1minute':   ('1m',  '5d'),
        '3minute':   ('5m',  '5d'),
        '5minute':   ('5m',  '5d'),
        '15minute':  ('15m', '5d'),
        '30minute':  ('30m', '5d'),
        '60minute':  ('1h',  '30d'),
    }
    yf_interval, yf_period = yf_interval_map.get(interval, ('5m', '5d'))

    for attempt in range(3):
        try:
            # ✅ Ticker().history() is thread-safe; yf.download() is NOT
            ticker = yf.Ticker(symbol)
            df     = ticker.history(period=yf_period, interval=yf_interval,
                                    auto_adjust=True, raise_errors=False)
            if df is None or df.empty:
                return None, 'none'
            # Normalize column names
            df.columns = [c.split(' ')[0] if ' ' in str(c) else c for c in df.columns]
            df = df[['Open','High','Low','Close','Volume']].dropna()
            if len(df) < 30:
                return None, 'none'
            _DATA_CACHE[_ck] = (df, 'yfinance')
            return df, 'yfinance'
        except Exception:
            if attempt < 2:
                time.sleep(0.5)
    return None, 'none'



def calculate_intraday_indicators(df):
    """
    1-minute candle tuned indicators:
    • EMA 5/9/21/50 ribbon  (faster response on 1min)
    • RSI-7                 (sensitive, reacts within minutes)
    • MACD 5/13/3           (intraday standard)
    • ATR-7                 (tight stops)
    • VWAP reset per day    (intraday anchor)
    • Stoch 5,3,3           (quick cycles)
    • Volume Profile        (POC + Value Area)
    """
    # ── EMA Ribbon: 5/9/21/50 ────────────────────────────
    df['EMA_5']  = df['Close'].ewm(span=5,  adjust=False).mean()
    df['EMA_9']  = df['Close'].ewm(span=9,  adjust=False).mean()
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    # Aliases for compatibility
    df['EMA_20'] = df['EMA_21']
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['SMA_200']= df['Close'].rolling(min(200, len(df)//2)).mean()

    # ── RSI-7 ─────────────────────────────────────────────
    delta = df['Close'].diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    df['RSI'] = 100 - (100 / (1 + gain.ewm(com=6, adjust=False).mean() /
                               loss.ewm(com=6, adjust=False).mean()))

    # ── MACD 5/13/3 ───────────────────────────────────────
    ema5  = df['Close'].ewm(span=5,  adjust=False).mean()
    ema13 = df['Close'].ewm(span=13, adjust=False).mean()
    df['MACD']        = ema5 - ema13
    df['MACD_Signal'] = df['MACD'].ewm(span=3, adjust=False).mean()
    df['MACD_Hist']   = df['MACD'] - df['MACD_Signal']

    # ── Bollinger Bands (10 period for 1min) ─────────────
    df['BB_Middle'] = df['Close'].rolling(10).mean()
    bb_std          = df['Close'].rolling(10).std()
    df['BB_Upper']  = df['BB_Middle'] + 2 * bb_std
    df['BB_Lower']  = df['BB_Middle'] - 2 * bb_std
    df['BB_Width']  = (df['BB_Upper'] - df['BB_Lower']) / (df['BB_Middle'] + 1e-9)

    # ── ATR-7 ─────────────────────────────────────────────
    hl  = df['High'] - df['Low']
    hc  = (df['High'] - df['Close'].shift()).abs()
    lc  = (df['Low']  - df['Close'].shift()).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(7).mean()

    # ── Stochastic 5,3,3 ─────────────────────────────────
    low5        = df['Low'].rolling(5).min()
    high5       = df['High'].rolling(5).max()
    df['Stoch_K'] = 100 * (df['Close'] - low5) / (high5 - low5 + 1e-9)
    df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()

    # ── ADX-7 ─────────────────────────────────────────────
    plus_dm  = df['High'].diff().clip(lower=0)
    minus_dm = (-df['Low'].diff()).clip(lower=0)
    tr7       = tr.ewm(alpha=1/7, adjust=False).mean()
    plus_di   = 100 * plus_dm.ewm(alpha=1/7, adjust=False).mean() / (tr7 + 1e-9)
    minus_di  = 100 * minus_dm.ewm(alpha=1/7, adjust=False).mean() / (tr7 + 1e-9)
    dx        = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100
    df['ADX']      = dx.ewm(alpha=1/7, adjust=False).mean()
    df['Plus_DI']  = plus_di
    df['Minus_DI'] = minus_di

    # ── Volume MA + Ratio ─────────────────────────────────
    df['Volume_MA']    = df['Volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['Volume'] / (df['Volume_MA'] + 1e-9)

    # ── VWAP — reset each calendar day ───────────────────
    try:
        _dates = pd.to_datetime(df.index).date
    except Exception:
        _dates = np.array([datetime.now().date()] * len(df))
    df['_date'] = _dates
    df['VWAP']  = np.nan
    for day, grp in df.groupby('_date'):
        cum_tp_vol = (grp['Close'] * grp['Volume']).cumsum()
        cum_vol    = grp['Volume'].cumsum()
        df.loc[grp.index, 'VWAP'] = cum_tp_vol / (cum_vol + 1e-9)
    df.drop(columns=['_date'], inplace=True, errors='ignore')

    # ── Pivot levels ──────────────────────────────────────
    df['Pivot'] = (df['High'].shift(1) + df['Low'].shift(1) + df['Close'].shift(1)) / 3
    df['R1']    = 2 * df['Pivot'] - df['Low'].shift(1)
    df['S1']    = 2 * df['Pivot'] - df['High'].shift(1)
    df['R2']    = df['Pivot'] + (df['High'].shift(1) - df['Low'].shift(1))
    df['S2']    = df['Pivot'] - (df['High'].shift(1) - df['Low'].shift(1))

    # ── Previous Day High / Low ───────────────────────────
    # PDH = resistance level — stocks near PDH face selling
    # PDL = support level — stocks near PDL may bounce
    try:
        import pytz as _ptz2
        _ist2   = _ptz2.timezone('Asia/Kolkata')
        _idx2   = pd.to_datetime(df.index)
        if _idx2.tzinfo is None:
            _idx2 = _idx2.tz_localize('UTC').tz_convert('Asia/Kolkata')
        else:
            _idx2 = _idx2.tz_convert('Asia/Kolkata')
        _dates2   = sorted(set(_idx2.date))
        df['PDH'] = np.nan
        df['PDL'] = np.nan
        df['PDC'] = np.nan
        for _i2, _d2 in enumerate(_dates2):
            if _i2 == 0:
                continue
            _prev_d2   = _dates2[_i2 - 1]
            _prev_mask = _idx2.date == _prev_d2
            _curr_mask = _idx2.date == _d2
            if _prev_mask.any() and _curr_mask.any():
                _pdh = float(df.loc[_prev_mask, 'High'].max())
                _pdl = float(df.loc[_prev_mask, 'Low'].min())
                _pdc = float(df.loc[_prev_mask, 'Close'].iloc[-1])
                df.loc[_curr_mask, 'PDH'] = round(_pdh, 2)
                df.loc[_curr_mask, 'PDL'] = round(_pdl, 2)
                df.loc[_curr_mask, 'PDC'] = round(_pdc, 2)
    except Exception:
        df['PDH'] = np.nan
        df['PDL'] = np.nan
        df['PDC'] = np.nan

    # ── CPR — Central Pivot Range ─────────────────────────
    # Uses YESTERDAY's daily H/L/C (not per-candle)
    # Groups candles by date, takes previous day's H/L/C
    try:
        _date_col = pd.to_datetime(df.index).date
        _dates    = sorted(set(_date_col))

        # Pre-build daily OHLC lookup
        _daily = {}
        for _d in _dates:
            _mask = _date_col == _d
            _daily[_d] = {
                'H': float(df.loc[_mask, 'High'].max()),
                'L': float(df.loc[_mask, 'Low'].min()),
                'C': float(df.loc[_mask, 'Close'].iloc[-1]),
            }

        # Assign CPR for each candle using previous day's data
        df['CPR_Pivot'] = np.nan
        df['CPR_TC']    = np.nan
        df['CPR_BC']    = np.nan
        df['CPR_R1']    = np.nan
        df['CPR_S1']    = np.nan
        df['CPR_R2']    = np.nan
        df['CPR_S2']    = np.nan
        df['CPR_Width'] = np.nan

        for _i, _d in enumerate(_dates):
            if _i == 0:
                continue  # no previous day for first day
            _prev        = _dates[_i - 1]
            _prev_data   = _daily[_prev]
            _pH = _prev_data['H']
            _pL = _prev_data['L']
            _pC = _prev_data['C']

            _pivot = (_pH + _pL + _pC) / 3
            _tc    = (_pivot + (_pH + _pL) / 2) / 2      # (Pivot + prev HL mid) / 2
            _bc    = (2 * _pivot - _pH + 2 * _pivot - _pL) / 2  # symmetric to TC
            # Correct CPR formula
            _bc    = _pivot - (_tc - _pivot)
            _r1    = 2 * _pivot - _pL
            _s1    = 2 * _pivot - _pH
            _r2    = _pivot + (_pH - _pL)
            _s2    = _pivot - (_pH - _pL)
            _width = abs(_tc - _bc)
            _width_pct = (_width / _pivot * 100) if _pivot > 0 else 0

            _mask = _date_col == _d
            df.loc[_mask, 'CPR_Pivot'] = round(_pivot, 2)
            df.loc[_mask, 'CPR_TC']    = round(_tc, 2)
            df.loc[_mask, 'CPR_BC']    = round(_bc, 2)
            df.loc[_mask, 'CPR_R1']    = round(_r1, 2)
            df.loc[_mask, 'CPR_S1']    = round(_s1, 2)
            df.loc[_mask, 'CPR_R2']    = round(_r2, 2)
            df.loc[_mask, 'CPR_S2']    = round(_s2, 2)
            df.loc[_mask, 'CPR_Width'] = round(_width_pct, 3)
    except Exception:
        for _col in ['CPR_Pivot','CPR_TC','CPR_BC','CPR_R1','CPR_S1','CPR_R2','CPR_S2','CPR_Width']:
            df[_col] = np.nan

    # ── Supertrend (7, 2) ─────────────────────────────────
    df = calculate_supertrend_intraday(df)

    # ── Volume Profile (last 200 candles) ─────────────────
    df = calculate_volume_profile(df)

    return df


def calculate_volume_profile(df, bins=20):
    """
    Vectorized Volume Profile — pure NumPy, no Python loops.
    ~50× faster than the iterrows version.
    """
    try:
        recent = df.tail(200)
        lo     = float(recent['Low'].min())
        hi     = float(recent['High'].max())
        if hi <= lo:
            df['VP_POC'] = np.nan; df['VP_VAH'] = np.nan; df['VP_VAL'] = np.nan
            return df

        edges = np.linspace(lo, hi, bins + 1)
        mids  = (edges[:-1] + edges[1:]) / 2

        # Vectorized: for each candle compute overlap with every bin at once
        lows    = recent['Low'].values.reshape(-1, 1)      # (N, 1)
        highs   = recent['High'].values.reshape(-1, 1)     # (N, 1)
        volumes = recent['Volume'].values.reshape(-1, 1)   # (N, 1)
        spans   = (highs - lows)                           # (N, 1)

        bin_lo  = edges[:-1].reshape(1, -1)                # (1, bins)
        bin_hi  = edges[1:].reshape(1, -1)                 # (1, bins)

        overlap = np.maximum(0, np.minimum(highs, bin_hi) - np.maximum(lows, bin_lo))  # (N, bins)
        safe_sp = np.where(spans > 0, spans, 1.0)
        vols    = (volumes * overlap / safe_sp).sum(axis=0)  # (bins,)

        poc_idx   = int(np.argmax(vols))
        poc_price = float(mids[poc_idx])

        # Value Area — 70% of total volume
        total_vol  = vols.sum()
        target_vol = total_vol * 0.70
        accum = vols[poc_idx]; lo_idx = poc_idx; hi_idx = poc_idx
        while accum < target_vol and (lo_idx > 0 or hi_idx < bins - 1):
            add_lo = vols[lo_idx - 1] if lo_idx > 0 else 0
            add_hi = vols[hi_idx + 1] if hi_idx < bins - 1 else 0
            if add_lo >= add_hi and lo_idx > 0:
                lo_idx -= 1; accum += vols[lo_idx]
            elif hi_idx < bins - 1:
                hi_idx += 1; accum += vols[hi_idx]
            else:
                lo_idx -= 1; accum += vols[lo_idx]

        # Store only scalar values — not per-row JSON
        df['VP_POC'] = round(poc_price, 2)
        df['VP_VAH'] = round(float(mids[hi_idx]), 2)
        df['VP_VAL'] = round(float(mids[lo_idx]), 2)
        # Store profile as metadata in session (not DataFrame column)
        _vp_key = f"vp_{id(df)}"
        st.session_state[_vp_key] = {
            'mids': mids.tolist(), 'vols': vols.tolist()
        }
    except Exception:
        df['VP_POC'] = np.nan; df['VP_VAH'] = np.nan; df['VP_VAL'] = np.nan
    return df


def calculate_supertrend_intraday(df, period=7, multiplier=2):
    """Vectorized Supertrend — no Python for-loop."""
    atr    = df['ATR'].values
    close  = df['Close'].values
    hl2    = ((df['High'] + df['Low']) / 2).values
    upper  = hl2 + multiplier * atr
    lower  = hl2 - multiplier * atr
    n      = len(df)
    st_arr = np.zeros(n)
    dir_arr= np.zeros(n, dtype=int)

    for i in range(1, n):
        if close[i] > upper[i - 1]:
            dir_arr[i] = 1
        elif close[i] < lower[i - 1]:
            dir_arr[i] = -1
        else:
            dir_arr[i] = dir_arr[i - 1]
        st_arr[i] = lower[i] if dir_arr[i] == 1 else upper[i]

    df['Supertrend']           = st_arr
    df['Supertrend_Direction'] = dir_arr
    return df


# ─────────────────────────────────────────────
#  INTRADAY SIGNAL SCORING  (100 pts)
# ─────────────────────────────────────────────

def score_intraday_signal(row, prev, df_slice):
    """
    Intraday-tuned scoring — VWAP and volume get higher weight,
    SMA200 ignored (not meaningful on 5m candles).
    """
    bull, bear, reasons, bd = 0, 0, [], {}

    # 1. Trend (28 pts)
    eb, es = 0, 0
    if prev['EMA_9'] <= prev['EMA_21'] and row['EMA_9'] > row['EMA_21']:
        eb += 8; reasons.append("EMA 9/21 Golden Cross")
    if row['EMA_9'] > row['EMA_21']:
        eb += 4
    elif row['EMA_9'] < row['EMA_21']:
        es += 4
    bull += eb; bear += es; bd['EMA Trend'] = (eb, es)

    sb, ss = (8, 0) if row['Supertrend_Direction'] == 1 else (0, 8)
    if sb: reasons.append("Supertrend Bullish")
    bull += sb; bear += ss; bd['Supertrend'] = (sb, ss)

    # VWAP — CRITICAL for intraday (8 pts)
    vb, vs = 0, 0
    if not pd.isna(row['VWAP']):
        diff_pct = (row['Close'] - row['VWAP']) / row['VWAP'] * 100
        if row['Close'] > row['VWAP']:
            vb = 8 if diff_pct > 0.3 else 5
            reasons.append(f"Above VWAP +{diff_pct:.1f}%")
        else:
            vs = 8 if diff_pct < -0.3 else 5
    bull += vb; bear += vs; bd['VWAP'] = (vb, vs)

    # HH/HL pattern (4 pts)
    hb, hs = 0, 0
    h = df_slice['High'].values; l = df_slice['Low'].values
    if len(h) >= 3:
        if h[-1]>h[-2]>h[-3] and l[-1]>l[-2]>l[-3]:
            hb = 4; reasons.append("HH+HL Pattern")
        elif h[-1]<h[-2]<h[-3] and l[-1]<l[-2]<l[-3]:
            hs = 4
    bull += hb; bear += hs; bd['Price Structure'] = (hb, hs)

    # 2. Momentum (30 pts)
    rb, rs = 0, 0
    rsi = row['RSI']
    if 45 <= rsi <= 60 and prev['RSI'] < 45:  rb = 10; reasons.append("RSI Recovery")
    elif 50 <= rsi <= 65:                       rb = 6;  reasons.append("RSI Bullish Zone")
    elif rsi < 30:                              rb = 4;  reasons.append("RSI Oversold")
    if rsi > 70:                                rs = 8
    elif prev['RSI'] >= 70 and rsi < 70:       rs = 10
    bull += rb; bear += rs; bd['RSI-7'] = (rb, rs)

    mb, ms = 0, 0
    if prev['MACD'] <= prev['MACD_Signal'] and row['MACD'] > row['MACD_Signal']:
        mb += 8; reasons.append("MACD Crossover")
    if row['MACD'] > 0:   mb += 4; reasons.append("MACD Positive")
    if prev['MACD'] >= prev['MACD_Signal'] and row['MACD'] < row['MACD_Signal']: ms += 8
    if row['MACD'] < 0:   ms += 4
    bull += mb; bear += ms; bd['MACD 5/13'] = (mb, ms)

    stb, sts = 0, 0
    if prev['Stoch_K'] < 20 and row['Stoch_K'] > row['Stoch_D']:
        stb = 8; reasons.append("Stoch Oversold Reversal")
    elif row['Stoch_K'] > row['Stoch_D'] and row['Stoch_K'] < 50:
        stb = 4; reasons.append("Stoch Bullish Cross")
    if row['Stoch_K'] > 80 and row['Stoch_K'] < row['Stoch_D']: sts = 8
    bull += stb; bear += sts; bd['Stochastic 5,3'] = (stb, sts)

    # 3. Volatility (14 pts)
    bbb, bbs = 0, 0
    if prev['Close'] <= prev['BB_Lower'] and row['Close'] > row['BB_Lower']:
        bbb = 8; reasons.append("BB Lower Bounce")
    elif row['Close'] < row['BB_Middle']: bbb = 2
    if row['Close'] >= row['BB_Upper']:   bbs = 8
    if not pd.isna(row['BB_Width']) and row['BB_Width'] < 0.02:
        bbb += 3; reasons.append("BB Squeeze")
    bull += bbb; bear += bbs; bd['Bollinger Bands'] = (bbb, bbs)

    ab, as_ = 0, 0
    if not pd.isna(row['ATR']) and not pd.isna(prev['ATR']):
        if row['ATR'] > prev['ATR']:
            if row['Close'] > prev['Close']: ab = 3; reasons.append("ATR Expanding Bull")
            else:                            as_ = 3
    bull += ab; bear += as_; bd['ATR'] = (ab, as_)

    # 4. Volume — HIGH WEIGHT for intraday (20 pts)
    vob, vos = 0, 0
    vr = row['Volume_Ratio']
    # Weighted volume — institutional surge (8×+) far more significant than retail (2×)
    if vr > 15.0:
        if row['Close'] > prev['Close']:   vob = 20; reasons.append("🏦 Institutional Surge Bull")
        else:                              vos = 20
    elif vr > 8.0:
        if row['Close'] > prev['Close']:   vob = 17; reasons.append("🔥 Major Volume Bull")
        else:                              vos = 17
    elif vr > 5.0:
        if row['Close'] > prev['Close']:   vob = 14; reasons.append("🔥 Surge Volume Bull")
        else:                              vos = 14
    elif vr > 3.0:
        if row['Close'] > prev['Close']:   vob = 11; reasons.append("Strong Volume Bull")
        else:                              vos = 11
    elif vr > 2.0:
        if row['Close'] > prev['Close']:   vob = 8;  reasons.append("High Volume Bull")
        else:                              vos = 8
    elif vr > 1.5:
        if row['Close'] > prev['Close']:   vob = 5;  reasons.append("Above Avg Volume")
        else:                              vos = 5
    elif vr > 1.0:
        if row['Close'] > prev['Close']:   vob = 2
        else:                              vos = 2
    bull += vob; bear += vos; bd['Volume'] = (vob, vos)

    ob, os2 = 0, 0
    if row['Close'] > prev['Close'] and row['Volume'] > prev['Volume']:
        ob = 4; reasons.append("OBV Rising")
    elif row['Close'] < prev['Close'] and row['Volume'] > prev['Volume']:
        os2 = 4
    bull += ob; bear += os2; bd['OBV'] = (ob, os2)

    # 5. ADX (8 pts)
    adb, ads = 0, 0
    if row['ADX'] > 25:
        if row['Plus_DI'] > row['Minus_DI']: adb = 8; reasons.append("ADX Strong Bull")
        else:                                 ads = 8
    elif row['ADX'] > 20:
        if row['Plus_DI'] > row['Minus_DI']: adb = 4
        else:                                 ads = 4
    bull += adb; bear += ads; bd['ADX-7'] = (adb, ads)

    return bull, bear, reasons, bd


def generate_intraday_signals(df):
    df['Signal']    = 0;  df['Signal_Type'] = '';  df['Confidence'] = 0
    df['Bull_Score']= 0;  df['Bear_Score']  = 0;   df['Score_Breakdown'] = ''
    MAX_BULL = 100
    for i in range(1, len(df)):
        row   = df.iloc[i];  prev = df.iloc[i-1]
        slice_ = df.iloc[max(0, i-5):i+1]
        bull, bear, reasons, bd = score_intraday_signal(row, prev, slice_)
        bd_str = ' | '.join([f"{k}:{v[0]}b/{v[1]}s" for k, v in bd.items() if v[0]>0 or v[1]>0])
        df.at[df.index[i], 'Bull_Score']      = bull
        df.at[df.index[i], 'Bear_Score']      = bear
        df.at[df.index[i], 'Score_Breakdown'] = bd_str
        # Intraday threshold: slightly tighter (need ≥30 + ≥12 gap)
        if bull >= 30 and (bull - bear) >= 12:
            df.at[df.index[i], 'Signal']      = 1
            df.at[df.index[i], 'Signal_Type'] = ' | '.join(reasons)
            df.at[df.index[i], 'Confidence']  = min(int((bull / MAX_BULL) * 100), 100)
        elif bear >= 30 and (bear - bull) >= 12:
            df.at[df.index[i], 'Signal']      = -1
            df.at[df.index[i], 'Confidence']  = min(int((bear / MAX_BULL) * 100), 100)
    return df


# ─────────────────────────────────────────────
#  INTRADAY TRADE PLAN  (tight stops, small targets)
# ─────────────────────────────────────────────

INTRADAY_STT_RATE  = 0.00025   # 0.025% on sell only (intraday STT)
INTRADAY_BROK_RATE = 0.0003    # 0.03% capped ₹20

def get_position_size_multiplier(rs_vs_nifty=0.0):
    """
    Returns (multiplier, label, color) based on market conditions.
    Market conditions affect HOW MUCH to trade, not WHETHER to trade.

    BULL + CALM/NORMAL:    100% — full size
    BULL + HIGH VIX:        70% — reduce slightly
    SIDEWAYS:               80% — moderate caution
    BEAR + RS > 1.5%:       50% — half size, strong stock
    BEAR + RS < 1.5%:       25% — quarter size, use shortlist RS gate
    CRISIS VIX:             25% — minimal
    """
    _nifty = st.session_state.get('nifty_market_state', 'UNKNOWN')
    _vix   = st.session_state.get('nifty_context', {}).get('vix_level', 'UNKNOWN')
    _rs    = rs_vs_nifty or 0.0

    if _vix == 'CRISIS':
        return 0.25, '25% size — VIX Crisis', '#dc2626'
    elif _vix == 'EXTREME':
        return 0.35, '35% size — VIX Extreme', '#ea580c'
    elif _nifty == 'BEAR':
        if _rs >= 1.5:
            return 0.50, '50% size — BEAR day (stock outperforming)', '#d97706'
        else:
            return 0.25, '25% size — BEAR day (weak RS)', '#dc2626'
    elif _nifty == 'SIDEWAYS':
        if _vix == 'HIGH':
            return 0.60, '60% size — Sideways + High VIX', '#d97706'
        return 0.80, '80% size — Sideways market', '#d97706'
    elif _nifty == 'BULL':
        if _vix == 'HIGH':
            return 0.70, '70% size — Bull + High VIX', '#d97706'
        return 1.00, '100% size — Full position', '#16a34a'
    return 0.80, '80% size — Unknown conditions', '#64748b'


def get_intraday_trade_plan(df, capital, risk_pct):
    latest = df.iloc[-1]
    entry  = float(latest['Close'])

    # ATR-7 for tighter stop
    atr = None
    for i in range(1, min(8, len(df))):
        v = df['ATR'].iloc[-i]
        if not pd.isna(v) and v > 0:
            atr = float(v); break

    if atr is None:
        return None

    # ── Intraday stop: 0.5× ATR (tighter than swing's 1.5×) ──
    stop_loss = round(entry - 0.5 * atr, 2)
    rps       = entry - stop_loss
    if rps <= 0 or pd.isna(rps):
        rps       = round(entry * 0.005, 2)   # 0.5% fallback
        stop_loss = round(entry - rps, 2)

    # ── Time-based target multiplier ─────────────────────
    # Targets shrink as day progresses — less time = tighter targets
    _now = ist_now()
    _tm  = _now.hour * 60 + _now.minute
    if _tm <= 690:        # 9:15–11:30 AM — best window
        _tmult  = 1.0
        _tlabel = 'Best window (full targets)'
    elif _tm <= 810:      # 11:30 AM–1:30 PM — lunch zone
        _tmult  = 0.7
        _tlabel = 'Lunch zone (reduced 0.7×)'
    elif _tm <= 870:      # 1:30–2:30 PM — second window
        _tmult  = 0.7
        _tlabel = 'Second window (reduced 0.7×)'
    elif _tm <= 900:      # 2:30–3:00 PM — late entry
        _tmult  = 0.5
        _tlabel = 'Late entry (scalp only 0.5×)'
    else:                 # after 3:00 PM
        _tmult  = 0.3
        _tlabel = 'Too late — exit only'

    # ── Targets scaled by time multiplier ────────────────
    t1 = round(entry + rps * 1.0 * _tmult, 2)   # R:R 1:1   — book 50%
    t2 = round(entry + rps * 1.5 * _tmult, 2)   # R:R 1.5:1 — move SL to entry
    t3 = round(entry + rps * 2.0 * _tmult, 2)   # R:R 2:1   — trail SL to T1
    t4 = round(entry + rps * 3.0 * _tmult, 2)   # R:R 3:1   — let it run

    ra  = capital * (risk_pct / 100)
    ps  = max(1, int(ra / rps))
    inv = round(entry * ps, 2)

    # ── Intraday charges (different from delivery) ──
    # Buy side: brokerage + exchange + SEBI + stamp + GST   (NO STT on buy)
    brok_b  = min(20, inv * INTRADAY_BROK_RATE)
    exc_b   = inv * 0.0000297
    sebi_b  = inv * 0.000001
    stamp_b = inv * 0.00003    # 0.003% stamp (intraday)
    gst_b   = (brok_b + exc_b + sebi_b) * 0.18
    total_b = round(brok_b + exc_b + sebi_b + stamp_b + gst_b, 2)
    actual  = round(inv + total_b, 2)

    def sell_ch_intraday(price, qty):
        sv    = price * qty
        brok  = min(20, sv * INTRADAY_BROK_RATE)
        stt   = sv * INTRADAY_STT_RATE    # STT on sell side only (intraday)
        exc   = sv * 0.0000297
        sebi  = sv * 0.000001
        gst   = (brok + exc + sebi) * 0.18
        return sv, round(brok + stt + exc + sebi + gst, 2)

    rows = []
    for label, price in [
        ("T1 — Target (1R)",      t1),
        ("T2 — Extended (1.5R)",  t2),
        ("T3 — Strong (2R)",      t3),
        ("T4 — Stretch (3R)",     t4),
        ("Stop Loss",             stop_loss)
    ]:
        sv, sc  = sell_ch_intraday(price, ps)
        gross   = round((price - entry) * ps, 2)
        net_pl  = round(gross - total_b - sc, 2)
        # No STCG for intraday (taxed as business income/speculative)
        ret_p   = round((net_pl / actual) * 100, 2) if actual > 0 else 0
        rows.append({
            "Scenario": label, "Sell Value": round(sv, 2),
            "Sell Charges": sc, "Gross P&L": gross,
            "Net P&L": net_pl, "Return%": ret_p
        })

    return {
        "entry": entry, "stop_loss": stop_loss,
        "t1": t1, "t2": t2, "t3": t3, "t4": t4,
        "rps": round(rps, 2), "qty": ps,
        "investment": inv, "actual_cost": actual,
        "time_mult":  round(_tmult, 1),
        "time_label": _tlabel,
        "risk_amount": round(ra, 2),
        "buy_charges": {
            "brokerage": round(brok_b, 2), "stt_buy": 0,
            "exchange":  round(exc_b, 2),  "sebi": round(sebi_b, 2),
            "stamp":     round(stamp_b, 2), "gst": round(gst_b, 2),
            "total":     total_b
        },
        "pl_table": rows,
        "atr": round(atr, 2),
        "vwap": float(latest['VWAP']) if not pd.isna(latest.get('VWAP', np.nan)) else None,
        "r1":   float(latest['R1'])   if not pd.isna(latest.get('R1', np.nan)) else None,
        "s1":   float(latest['S1'])   if not pd.isna(latest.get('S1', np.nan)) else None,
    }


# ─────────────────────────────────────────────
#  PICK SCORE (intraday-tuned)
# ─────────────────────────────────────────────

def conf_label(score):
    if score >= 80: return "STRONG",   "badge-strong"
    if score >= 60: return "GOOD",     "badge-good"
    if score >= 40: return "MODERATE", "badge-moderate"
    if score >= 20: return "WEAK",     "badge-weak"
    return "NONE", "badge-none"

def conf_color(score):
    if score >= 80: return "#15803d"
    if score >= 60: return "#16a34a"
    if score >= 40: return "#ca8a04"
    if score >= 20: return "#ea580c"
    return "#94a3b8"


# ─────────────────────────────────────────────
#  MINIMUM CANDLE THRESHOLDS
#  Below these counts indicators are unreliable
# ─────────────────────────────────────────────
MIN_CANDLES_HARD  = 7    # absolute minimum — below this return WARMING UP
MIN_CANDLES_SOFT  = 20   # soft minimum — below this flag as partially ready
MIN_CANDLES_FULL  = 50   # full reliability — all indicators meaningful

# Candles per day by interval — used for warmup status
CANDLES_PER_DAY = {
    '1minute':  375,   # 9:15 AM to 3:30 PM = 375 min
    '3minute':  125,
    '5minute':  75,
    '15minute': 25,
    '30minute': 13,
    '60minute': 7,     # Only 6-7 candles per day on hourly
}

def candle_warmup_status(df, interval='1minute'):
    """
    Returns (status, candles_today, mins_open, pct_ready)
    Interval-aware: 60min needs far fewer candles to be READY
    Uses IST timezone for correct today detection
    """
    try:
        import pytz as _ptz
        _ist       = _ptz.timezone('Asia/Kolkata')
        _today_ist = datetime.now(_ist).date()
        # Convert index to IST before date comparison
        _idx = pd.to_datetime(df.index)
        if _idx.tzinfo is None:
            _idx = _idx.tz_localize('UTC').tz_convert('Asia/Kolkata')
        else:
            _idx = _idx.tz_convert('Asia/Kolkata')
        today_df = df[_idx.date == _today_ist]
        n        = len(today_df)
    except Exception:
        n = len(df)

    total_n     = len(df)
    cpd         = CANDLES_PER_DAY.get(interval, 375)
    mins        = n * (375 // cpd)   # approximate minutes open

    # Scale thresholds by interval
    # 60min: only need 3 candles today = READY (3 hours of data)
    # 1min:  need 20+ candles = READY
    _hard = max(3, MIN_CANDLES_HARD  // max(1, cpd // 75))
    _soft = max(3, MIN_CANDLES_SOFT  // max(1, cpd // 75))
    _full = max(10, MIN_CANDLES_FULL // max(1, cpd // 75))

    if total_n < _hard or n < 1:
        return 'WARMING_UP', n, mins, 0
    elif total_n < _soft or n < 2:
        return 'PARTIAL', n, mins, int((total_n / _full) * 100)
    else:
        return 'READY', n, mins, min(100, int((total_n / _full) * 100))


def compute_intraday_pick_score(r):
    df     = r.get('df')
    warmup, n_today, mins, pct = candle_warmup_status(df, r.get('interval','1minute')) \
                                  if df is not None \
                                  else ('WARMING_UP', 0, 0, 0)

    if warmup == 'WARMING_UP':
        return 0, {}, '\u23f3 WARMING UP'

    scores = {}
    _price = r.get('price', 0)
    _df    = df

    # ── Partial guard ─────────────────────────────────────
    if warmup == 'PARTIAL':
        vol = r['vol_ratio']
        scores['Volume']    = (20 if vol >= 15.0 else
                               17 if vol >= 8.0  else
                               14 if vol >= 5.0  else
                               10 if vol >= 3.0  else
                               8  if vol >= 2.0  else
                               5  if vol >= 1.5  else 0)
        scores['VWAP']      = 12 if r['vwap'] == 'ABOVE' else 0
        liq = r.get('liquidity', {})
        scores['Liquidity'] = 7 if liq.get('grade') in ['EXCELLENT','HIGH'] else 0
        total = max(0, sum(scores.values()))
        verdict = '\u2b50 WATCH (early data)' if total >= 20 else '\u26a0\ufe0f NEUTRAL (warming up)'
        return total, scores, verdict

    # ════════════════════════════════════════════
    # PRIORITY 1 — NIFTY MARKET FILTER
    # ════════════════════════════════════════════
    # Penalty reduced on BEAR days — strong RS stocks
    # can overcome the market headwind. Blanket -20
    # was eliminating genuinely strong defensive stocks.
    _nifty_state = st.session_state.get('nifty_market_state', 'UNKNOWN')
    if _nifty_state == 'BEAR':
        scores['Market_Filter'] = -10   # reduced from -20
    elif _nifty_state == 'SIDEWAYS':
        scores['Market_Filter'] = -8
    elif _nifty_state == 'BULL':
        scores['Market_Filter'] = 8

    # ── Core indicators ───────────────────────────────────
    scores['Signal'] = 20 if r['signal_val'] == 1 else (0 if r['signal_val'] == 0 else -10)
    conf = r['live_conf']
    scores['Conf%']  = 15 if conf >= 130 else (12 if conf >= 100 else (9 if conf >= 75 else (5 if conf >= 55 else 0)))

    # ── Trend signals need minimum 3-candle confirmation ──
    # Prevents single-candle false flips (Brigade pattern)
    _trend_confirmed = False
    _ema_confirmed   = False
    if _df is not None and len(_df) >= 5:
        try:
            # Supertrend: must be bullish for last 3 candles, not just latest
            _st_last3 = _df['Supertrend_Direction'].iloc[-3:].values
            _trend_confirmed = all(_v == 1 for _v in _st_last3 if not pd.isna(_v))
            # EMA: EMA9 must be above EMA21 for last 3 candles
            _ema9_l3  = _df['EMA_9'].iloc[-3:].values
            _ema21_l3 = _df['EMA_21'].iloc[-3:].values
            _ema_confirmed = all(
                _ema9_l3[_i] > _ema21_l3[_i]
                for _i in range(len(_ema9_l3))
                if not (pd.isna(_ema9_l3[_i]) or pd.isna(_ema21_l3[_i]))
            )
        except Exception:
            _trend_confirmed = r['supertrend'] == 1
            _ema_confirmed   = r['ema_trend'] == 'BULL'

    scores['Trend'] = 10 if _trend_confirmed else 0
    scores['EMA']   = 10 if _ema_confirmed   else 0
    rsi = r['rsi']
    scores['RSI']    = 10 if 45<=rsi<=65 else (6 if 35<=rsi<45 else (4 if 65<rsi<=70 else 0))
    adx = r['adx']
    scores['ADX']    = 10 if adx >= 25 else (6 if adx >= 20 else 0)
    scores['VWAP']   = 12 if r['vwap'] == 'ABOVE' else 0
    vol = r['vol_ratio']
    # Weighted volume scoring — institutional surge (8×+) is far more significant
    # Also checks direction: high volume in direction of trend = stronger signal
    _chg = r.get('change_pct', 0.0)
    _vol_dir_match = (_chg > 0)   # price up = bullish volume confirmation
    if vol >= 15.0:
        scores['Volume'] = 22 if _vol_dir_match else 8   # institutional surge
    elif vol >= 8.0:
        scores['Volume'] = 19 if _vol_dir_match else 6   # major volume event
    elif vol >= 5.0:
        scores['Volume'] = 16 if _vol_dir_match else 4   # strong institutional
    elif vol >= 3.0:
        scores['Volume'] = 12 if _vol_dir_match else 3   # solid buying
    elif vol >= 2.0:
        scores['Volume'] = 8  if _vol_dir_match else 2   # above average
    elif vol >= 1.5:
        scores['Volume'] = 5  if _vol_dir_match else 1   # moderate interest
    elif vol >= 1.0:
        scores['Volume'] = 2                              # in line with average
    else:
        scores['Volume'] = -5                             # low volume = no conviction
    bb = r['bb_pos']
    scores['BB']     = 4 if bb == 'LOWER' else (2 if bb == 'MID' else 0)
    gap = r['live_bull'] - r['live_bear']
    scores['Bull/Bear Gap'] = 4 if gap >= 30 else (3 if gap >= 20 else (2 if gap >= 10 else (1 if gap >= 0 else 0)))
    liq = r.get('liquidity', {})
    liq_grade = liq.get('grade', 'LOW')
    scores['Liquidity'] = 10 if liq_grade == 'EXCELLENT' else (7 if liq_grade == 'HIGH' else (4 if liq_grade == 'MEDIUM' else -5))

    # ── CPR scoring ───────────────────────────────────────
    _cpr_tc = r.get('cpr_tc'); _cpr_bc = r.get('cpr_bc'); _cpr_w = r.get('cpr_width')
    if _cpr_tc and _cpr_bc and _price > 0:
        scores['CPR_Position'] = 10 if _price > _cpr_tc else (2 if _price > _cpr_bc else -8)
        if _cpr_w is not None:
            scores['CPR_Width'] = 8 if _cpr_w < 0.3 else (5 if _cpr_w < 0.6 else (0 if _cpr_w < 1.0 else -10))

    # ── Downtrend guard ───────────────────────────────────
    if _df is not None and len(_df) >= 50 and _price > 0:
        try:
            _sma50  = float(_df['SMA_50'].iloc[-1])  if 'SMA_50'  in _df.columns else None
            _sma200 = float(_df['SMA_200'].iloc[-1]) if 'SMA_200' in _df.columns else None
            if _sma50 and _sma200:
                if _price < _sma50 and _price < _sma200:
                    scores['Downtrend_Guard'] = -15
                elif _price < _sma50:
                    scores['Downtrend_Guard'] = -5
        except Exception:
            pass

    # ════════════════════════════════════════════
    # PRIORITY 2 — RISK:REWARD FILTER
    # ════════════════════════════════════════════
    _tp = r.get('trade_plan')
    if _tp and isinstance(_tp, dict):
        _t1 = _f(_tp.get('t1', 0))
        _sl = _f(_tp.get('stop_loss', 0))
        if _t1 > 0 and _sl > 0 and _price > 0:
            _sd = _price - _sl
            _td = _t1 - _price
            if _sd > 0:
                _rr = _td / _sd
                scores['RR_Quality'] = (10 if _rr >= 2.0 else (6 if _rr >= 1.5 else
                                        (2 if _rr >= 1.0 else (-8 if _rr >= 0.5 else -15))))

    # ════════════════════════════════════════════
    # PRIORITY 3 — TIME-OF-DAY CONTEXT
    # ════════════════════════════════════════════
    try:
        _now = ist_now()
        _tm  = _now.hour * 60 + _now.minute
        if 555 <= _tm <= 585:    scores['Time_Context'] = 5    # 9:15-9:45 opening
        elif 585 <= _tm <= 690:  scores['Time_Context'] = 10   # 9:45-11:30 BEST
        elif 690 <= _tm <= 810:  scores['Time_Context'] = -3   # 11:30-1:30 lunch
        elif 810 <= _tm <= 870:  scores['Time_Context'] = 4    # 1:30-2:30 second wind
        elif 870 <= _tm <= 915:  scores['Time_Context'] = -8   # 2:30-3:15 late
        elif _tm > 915:          scores['Time_Context'] = -20  # 3:15+ square off
    except Exception:
        pass

    # ════════════════════════════════════════════
    # PRIORITY 4 — SECTOR MOMENTUM
    # ════════════════════════════════════════════
    _sym_sector = r.get('sector', '')
    _sector_mom = st.session_state.get('sector_momentum', {})
    if _sym_sector and _sym_sector in _sector_mom:
        _sc = _sector_mom[_sym_sector]
        scores['Sector_Momentum'] = (10 if _sc >= 1.5 else (5 if _sc >= 0.5 else
                                     (0 if _sc >= -0.5 else (-8 if _sc >= -1.5 else -15))))

    # ════════════════════════════════════════════
    # PRIORITY 5 — OPENING 15-MIN QUALITY
    # ════════════════════════════════════════════
    if _df is not None and n_today >= 3:
        try:
            _td  = pd.Timestamp.now().date()
            _tc  = _df[pd.to_datetime(_df.index).date == _td].head(15)
            if len(_tc) >= 3:
                _uw  = _tc['High'] - _tc[['Close','Open']].max(axis=1)
                _rng = _tc['High'] - _tc['Low']
                _awr = float((_uw / (_rng + 0.01)).mean())
                _bc  = int(sum(1 for _, c in _tc.iterrows() if float(c['Close']) > float(c['Open'])))
                if _awr < 0.25 and _bc >= 3:
                    scores['ORB_Quality'] = 10
                elif _awr < 0.4 and _bc >= 2:
                    scores['ORB_Quality'] = 5
                elif _awr > 0.6:
                    scores['ORB_Quality'] = -10
                elif _bc == 0:
                    scores['ORB_Quality'] = -8
        except Exception:
            pass

    # ════════════════════════════════════════════
    # PRIORITY 6 — GAP CLASSIFICATION
    # ════════════════════════════════════════════
    _gap_pct = r.get('gap_pct', 0.0) or 0.0
    _g = float(_gap_pct)
    if -0.3 <= _g <= 0.3:    scores['Gap_Quality'] = 6
    elif 0.3 < _g <= 1.0:    scores['Gap_Quality'] = 4
    elif 1.0 < _g <= 2.0:    scores['Gap_Quality'] = 0
    elif _g > 2.0:            scores['Gap_Quality'] = -8
    elif -1.0 <= _g < -0.3:  scores['Gap_Quality'] = -5
    else:                     scores['Gap_Quality'] = -12

    # ════════════════════════════════════════════
    # PREVIOUS DAY HIGH RESISTANCE
    # Stocks near PDH face strong selling pressure
    # ════════════════════════════════════════════
    _pdh = r.get('pdh')
    _pdl = r.get('pdl')
    if _pdh and _pdl and _price > 0:
        _pdh_dist_pct = (_pdh - _price) / _price * 100   # % away from PDH
        _pdl_dist_pct = (_price - _pdl) / _price * 100   # % away from PDL

        if _price > _pdh * 1.002:
            # Broken above PDH = strong breakout signal
            scores['PDH_Level'] = 10
        elif _price >= _pdh * 0.995:
            # Within 0.5% of PDH = at resistance — risky entry
            scores['PDH_Level'] = -8
        elif _price >= _pdh * 0.98:
            # 0.5–2% below PDH = approaching resistance — caution
            scores['PDH_Level'] = -3
        elif _pdl_dist_pct <= 1.0:
            # Near previous day low = support zone — potential bounce
            scores['PDH_Level'] = 3
        else:
            # Healthy distance from PDH — no penalty, no bonus
            scores['PDH_Level'] = 0

    # ════════════════════════════════════════════
    # PRIORITY 7 — CONSECUTIVE RED DAYS
    # ════════════════════════════════════════════
    if _df is not None and len(_df) >= 10:
        try:
            _td2 = pd.Timestamp.now().date()
            _dc  = {}
            for _d in sorted(set(pd.to_datetime(_df.index).date)):
                _m = pd.to_datetime(_df.index).date == _d
                _dc[_d] = float(_df.loc[_m, 'Close'].iloc[-1])
            _sd = sorted(_dc.keys())
            _cr = 0; _cg = 0
            for _i in range(len(_sd)-1, 0, -1):
                _dc2 = _sd[_i]; _dp = _sd[_i-1]
                if _dc2 == _td2: continue
                if _dc[_dc2] < _dc[_dp]:
                    if _cg > 0: break
                    _cr += 1
                else:
                    if _cr > 0: break
                    _cg += 1
                if _cr >= 4 or _cg >= 3: break
            if _cr >= 4:   scores['Consec_Red']   = -15
            elif _cr >= 3: scores['Consec_Red']   = -10
            elif _cr >= 2: scores['Consec_Red']   = -5
            elif _cg >= 3: scores['Consec_Green'] = 6
            elif _cg >= 2: scores['Consec_Green'] = 3
        except Exception:
            pass

    # ════════════════════════════════════════════
    # RELATIVE STRENGTH vs NIFTY
    # Outperforming = real strength, not market ride
    # ════════════════════════════════════════════
    _rs = r.get('rs_vs_nifty')
    if _rs is not None:
        if _rs >= 3.0:
            scores['Rel_Strength'] = 15   # Strongly outperforming — Waaree/Premier pattern
        elif _rs >= 1.5:
            scores['Rel_Strength'] = 10
        elif _rs >= 0.5:
            scores['Rel_Strength'] = 5
        elif _rs >= -0.5:
            scores['Rel_Strength'] = 0    # In line with market
        elif _rs >= -1.5:
            scores['Rel_Strength'] = -8
        else:
            scores['Rel_Strength'] = -15

    # ════════════════════════════════════════════
    # PREVIOUS DAY HIGH/LOW RESISTANCE/SUPPORT
    # PDH = strong resistance → avoid entry near it
    # PDL = support → gives confidence if price holds
    # ════════════════════════════════════════════
    _pdh = r.get('pdh')
    _pdl = r.get('pdl')
    if _pdh and _pdl and _price > 0:
        _dist_pdh_pct = (_pdh - _price) / _price * 100   # positive = price below PDH
        _dist_pdl_pct = (_price - _pdl) / _price * 100   # positive = price above PDL

        # Price approaching PDH (within 0.5%) = heavy resistance = bad entry
        if _dist_pdh_pct < 0:
            scores['PDH_Resistance'] = -12  # Already above PDH = breakout zone
        elif _dist_pdh_pct < 0.3:
            scores['PDH_Resistance'] = -10  # Right at PDH = strong resistance
        elif _dist_pdh_pct < 0.6:
            scores['PDH_Resistance'] = -5   # Close to PDH = caution
        elif _dist_pdh_pct < 1.0:
            scores['PDH_Resistance'] = 0    # Moderate distance
        else:
            scores['PDH_Resistance'] = 5    # Good room to run before resistance

        # Price holding above PDL = support confirmed
        if _dist_pdl_pct > 2.0:
            scores['PDL_Support'] = 4       # Well above PDL = strong base
        elif _dist_pdl_pct > 1.0:
            scores['PDL_Support'] = 2
        elif _dist_pdl_pct > 0:
            scores['PDL_Support'] = 0       # Just above PDL = weak base
        else:
            scores['PDL_Support'] = -8      # Below PDL = broke support = avoid

    # ════════════════════════════════════════════
    # FINAL VERDICT WITH HARD CAPS
    # ════════════════════════════════════════════
    # Architecture: Market conditions affect POSITION SIZE, not score threshold.
    # Only VIX CRISIS/EXTREME get hard caps (genuine market breakdown).
    # BEAR days: RS gate in shortlist handles selection.
    # Position size multiplier computed separately and shown on card.
    total = max(0, sum(scores.values()))

    _vix_level   = st.session_state.get('nifty_context', {}).get('vix_level', 'UNKNOWN')
    _nifty_state = st.session_state.get('nifty_market_state', 'UNKNOWN')

    # Only hard-cap truly dangerous conditions
    if _vix_level == 'CRISIS':          # VIX > 30 — COVID/war level
        total = min(total, 45)          # Near-total block
    elif _vix_level == 'EXTREME':       # VIX 25-30 — serious fear
        total = min(total, 60)          # Block most signals

    # Time-based hard cap — after 3:15 PM never trade
    if scores.get('Time_Context', 0) <= -20:
        total = min(total, 35)

    if total >= 80:   verdict = '\u2b50\u2b50\u2b50 STRONG BUY'
    elif total >= 65: verdict = '\u2b50\u2b50 BUY'
    elif total >= 50: verdict = '\u2b50 WATCH'
    elif total >= 35: verdict = '\u26a0\ufe0f NEUTRAL'
    else:             verdict = '\u274c AVOID'
    return total, scores, verdict


# ─────────────────────────────────────────────
#  LIQUIDITY ENGINE
#  Measures 6 dimensions of intraday liquidity
# ─────────────────────────────────────────────

def compute_liquidity(df, price, capital):
    """
    Computes 6 liquidity metrics critical for intraday trading:

    1. Avg Daily Volume    — total shares traded per day
    2. Avg Daily Turnover  — ₹ value traded per day (ADV × price)
    3. Volume Consistency  — how reliably volume shows up (not just 1 day spikes)
    4. Intraday Liquidity  — today's volume vs avg (how active right now)
    5. Slippage Risk       — ATR as % of price (wide ATR = high slippage)
    6. Position Liquidity  — can you enter/exit your full position easily?

    Returns a dict with all metrics + overall grade: EXCELLENT / HIGH / MEDIUM / LOW
    """
    try:
        # ── Raw data ─────────────────────────────────────
        closes  = df['Close'].dropna().values.astype(float)
        volumes = df['Volume'].dropna().values.astype(float)
        highs   = df['High'].dropna().values.astype(float)
        lows    = df['Low'].dropna().values.astype(float)
        n       = len(volumes)

        if n < 10:
            return _liquidity_unknown()

        # ── 1. Avg Daily Volume (last 20 candles rolling) ──
        # For 1min data, 375 candles = 1 day
        # Use last 20 candles as a proxy for recent avg volume/candle
        recent_vols   = volumes[-20:]
        avg_vol_candle = float(np.mean(recent_vols))
        # Estimate daily volume: avg candle vol × 375 (full session)
        est_daily_vol  = avg_vol_candle * 375
        today_vol      = float(volumes[-1]) if n > 0 else 0

        # ── 2. Avg Daily Turnover (₹) ────────────────────
        avg_turnover = avg_vol_candle * float(price) * 375  # estimated ₹/day

        # ── 3. Volume Consistency ─────────────────────────
        # Coefficient of variation — lower = more consistent
        if len(recent_vols) > 1 and np.mean(recent_vols) > 0:
            cv = float(np.std(recent_vols) / np.mean(recent_vols))
        else:
            cv = 1.0
        consistency_pct = max(0, min(100, int((1 - min(cv, 1)) * 100)))

        # ── 4. Today's Volume vs Average ─────────────────
        vol_ratio = float(df['Volume_Ratio'].iloc[-1]) if 'Volume_Ratio' in df.columns else 1.0

        # ── 5. Slippage Risk (ATR% of price) ─────────────
        atr = float(df['ATR'].iloc[-1]) if 'ATR' in df.columns and not pd.isna(df['ATR'].iloc[-1]) else 0
        atr_pct = (atr / price * 100) if price > 0 else 0
        # Lower ATR% = less slippage
        if atr_pct < 0.2:   slippage = 'VERY LOW';  slip_score = 5
        elif atr_pct < 0.4: slippage = 'LOW';       slip_score = 4
        elif atr_pct < 0.7: slippage = 'MEDIUM';    slip_score = 3
        elif atr_pct < 1.2: slippage = 'HIGH';      slip_score = 2
        else:               slippage = 'VERY HIGH'; slip_score = 1

        # ── 6. Position Liquidity ─────────────────────────
        # Can you fill your full position in 1 candle?
        # Estimate: avg candle volume × price = avg ₹ per candle
        avg_candle_rs = avg_vol_candle * price
        position_size = capital  # your full capital as proxy
        # If avg candle turnover > 3× your position = excellent fill
        if avg_candle_rs <= 0:
            pos_fill_ratio = 0
        else:
            pos_fill_ratio = avg_candle_rs / position_size
        if pos_fill_ratio >= 5:    pos_liquidity = 'EASY';      pos_score = 5
        elif pos_fill_ratio >= 2:  pos_liquidity = 'GOOD';      pos_score = 4
        elif pos_fill_ratio >= 1:  pos_liquidity = 'MODERATE';  pos_score = 3
        elif pos_fill_ratio >= 0.5:pos_liquidity = 'TIGHT';     pos_score = 2
        else:                      pos_liquidity = 'ILLIQUID';  pos_score = 1

        # ── Overall Grade ─────────────────────────────────
        # Score based on daily turnover + slippage + position fill
        if avg_turnover >= 50_00_00_000:   turnover_score = 5  # ₹50 Cr+/day
        elif avg_turnover >= 10_00_00_000: turnover_score = 4  # ₹10 Cr+/day
        elif avg_turnover >= 1_00_00_000:  turnover_score = 3  # ₹1 Cr+/day
        elif avg_turnover >= 10_00_000:    turnover_score = 2  # ₹10L+/day
        else:                              turnover_score = 1  # below ₹10L

        grade_score = turnover_score + slip_score + pos_score
        if grade_score >= 13:   grade = 'EXCELLENT'
        elif grade_score >= 10: grade = 'HIGH'
        elif grade_score >= 7:  grade = 'MEDIUM'
        else:                   grade = 'LOW'

        # ── Tradeable flag ────────────────────────────────
        # Hard rules that disqualify a stock regardless of grade
        tradeable = True
        warnings  = []
        if avg_turnover < 5_00_000:          # < ₹5L daily turnover
            tradeable = False
            warnings.append("⚠️ Very low turnover — avoid")
        if slippage == 'VERY HIGH':
            tradeable = False
            warnings.append("⚠️ Extremely wide spreads — high slippage risk")
        if pos_liquidity == 'ILLIQUID':
            tradeable = False
            warnings.append("⚠️ Position too large for this stock's liquidity")
        if consistency_pct < 30:
            warnings.append("⚠️ Inconsistent volume — price can gap suddenly")

        return {
            'grade':            grade,
            'tradeable':        tradeable,
            'warnings':         warnings,
            'avg_daily_vol':    int(est_daily_vol),
            'avg_turnover':     round(avg_turnover, 0),
            'consistency_pct':  consistency_pct,
            'vol_ratio_now':    round(vol_ratio, 2),
            'atr_pct':          round(atr_pct, 3),
            'slippage':         slippage,
            'pos_liquidity':    pos_liquidity,
            'pos_fill_ratio':   round(pos_fill_ratio, 2),
            'grade_score':      grade_score,
        }
    except Exception:
        return _liquidity_unknown()


def _liquidity_unknown():
    return {
        'grade': 'UNKNOWN', 'tradeable': True, 'warnings': [],
        'avg_daily_vol': 0, 'avg_turnover': 0, 'consistency_pct': 0,
        'vol_ratio_now': 1.0, 'atr_pct': 0, 'slippage': 'UNKNOWN',
        'pos_liquidity': 'UNKNOWN', 'pos_fill_ratio': 0, 'grade_score': 0,
    }


def _fmt_turnover(val):
    """Format turnover in human-readable ₹ format."""
    if val >= 1_00_00_00_000:  return f"₹{val/1_00_00_00_000:.1f}K Cr"
    if val >= 1_00_00_000:     return f"₹{val/1_00_00_000:.1f} Cr"
    if val >= 1_00_000:        return f"₹{val/1_00_000:.1f}L"
    if val >= 1_000:           return f"₹{val/1_000:.1f}K"
    return f"₹{val:.0f}"


# ─────────────────────────────────────────────
#  LSTM — NEXT 3 CANDLES PREDICTION
#  Pure NumPy — no sklearn, no tensorflow
# ─────────────────────────────────────────────

def lstm_predict_next_candles(df, symbol, n_candles=3):
    """
    Predicts next n_candles closing prices using a pure-NumPy LSTM.
    For intraday, n_candles=3 means the next 3 × interval candles.
    """
    try:
        import numpy as _np

        closes = df['Close'].dropna().values.astype(float)
        if len(closes) < 40:
            return {'error': f'Not enough candles ({len(closes)}, need ≥ 40).'}
        closes     = closes[-80:]
        last_price = float(closes[-1])

        c_min = closes.min(); c_max = closes.max()
        c_rng = c_max - c_min if c_max != c_min else 1e-8
        scaled = (closes - c_min) / c_rng

        SEQ = 15; H = 32

        Xs, ys = [], []
        for i in range(SEQ, len(scaled)):
            Xs.append(scaled[i-SEQ:i]); ys.append(scaled[i])
        Xs = _np.array(Xs); ys = _np.array(ys); M = len(Xs)

        _np.random.seed(42)
        def _xavier(r, c): return _np.random.randn(r, c) * _np.sqrt(2.0 / (r + c))

        Wx = _xavier(1, 4*H); Wh = _xavier(H, 4*H)
        b  = _np.zeros(4*H); Wy = _xavier(H, 1); by = _np.zeros(1)

        def _sig(x):
            x = _np.clip(x, -30, 30)
            return _np.where(x>=0, 1/(1+_np.exp(-x)), _np.exp(x)/(1+_np.exp(x)))
        _tanh = lambda x: _np.tanh(_np.clip(x, -30, 30))

        def _fwd(seq):
            h = _np.zeros(H); c = _np.zeros(H)
            for t in range(len(seq)):
                s = float(seq[t]); g = s*Wx[0] + h@Wh + b
                ig=_sig(g[:H]); fg=_sig(g[H:2*H]); gg=_tanh(g[2*H:3*H]); og=_sig(g[3*H:])
                c = fg*c + ig*gg; h = og*_tanh(c)
            return float(_np.dot(h, Wy.flatten()) + by[0]), h

        params = [Wx, Wh, b, Wy, by]
        ms = [_np.zeros_like(p) for p in params]
        vs = [_np.zeros_like(p) for p in params]
        t_s = 0; best_loss = _np.inf; best_snap = None; no_imp = 0

        for ep in range(60):
            ep_loss = 0.0
            for i in _np.random.permutation(M):
                seq = Xs[i]; tgt = ys[i]
                h = _np.zeros(H); c = _np.zeros(H)
                for t in range(SEQ):
                    s = float(seq[t]); g = s*Wx[0] + h@Wh + b
                    ig=_sig(g[:H]); fg=_sig(g[H:2*H]); gg=_tanh(g[2*H:3*H]); og=_sig(g[3*H:])
                    c = fg*c + ig*gg; h = og*_tanh(c)
                y_hat = float(_np.dot(h, Wy.flatten()) + by[0])
                err   = y_hat - tgt; ep_loss += err**2
                dWy = h.reshape(-1,1)*(2*err); dby = _np.array([2*err])
                dh  = Wy.flatten()*(2*err)
                s   = float(seq[-1]); g = s*Wx[0] + h@Wh + b
                ig=_sig(g[:H]); fg=_sig(g[H:2*H]); gg=_tanh(g[2*H:3*H]); og=_sig(g[3*H:])
                tc  = _tanh(c)
                do  = dh*tc*og*(1-og); dc = dh*og*(1-tc**2)
                di  = dc*gg*ig*(1-ig); df_ = dc*(c-ig*gg)*fg*(1-fg); dg_ = dc*ig*(1-gg**2)
                dgts = _np.concatenate([di, df_, dg_, do])
                grads = [(s*dgts).reshape(1,-1), h.reshape(-1,1)@dgts.reshape(1,-1), dgts, dWy, dby]
                t_s += 1
                for p, g_, m, v in zip(params, grads, ms, vs):
                    g_ = _np.clip(g_, -1, 1)
                    m[:] = 0.9*m + 0.1*g_; v[:] = 0.999*v + 0.001*g_**2
                    mh = m/(1-0.9**t_s); vh = v/(1-0.999**t_s)
                    p -= 0.005 * mh / (_np.sqrt(vh) + 1e-8)

            avg = ep_loss / M
            if avg < best_loss - 1e-7:
                best_loss = avg; best_snap = [p.copy() for p in params]; no_imp = 0
            else:
                no_imp += 1
                if no_imp >= 10: break

        if best_snap:
            for p, s in zip(params, best_snap): p[:] = s

        seed = list(scaled[-SEQ:])
        preds_s = []
        for _ in range(n_candles):
            yp, _ = _fwd(_np.array(seed[-SEQ:]))
            yp    = float(_np.clip(yp, 0.0, 1.0))
            preds_s.append(yp); seed.append(yp)

        preds = [round(p * c_rng + c_min, 2) for p in preds_s]
        pcts  = [round((p - last_price) / last_price * 100, 2) for p in preds]
        direction = 'BULLISH' if preds[-1] > last_price else 'BEARISH'

        # Candle timestamps
        try:
            interval_minutes = 5  # default
            if hasattr(df.index, 'freq') and df.index.freq is not None:
                freq_str = str(df.index.freq)
                if '5' in freq_str: interval_minutes = 5
                elif '15' in freq_str: interval_minutes = 15
                elif '60' in freq_str or '1h' in freq_str.lower(): interval_minutes = 60
            last_ts = pd.Timestamp(df.index[-1])
            future_ts = [str((last_ts + pd.Timedelta(minutes=interval_minutes*(k+1))).strftime('%H:%M')) for k in range(n_candles)]
        except Exception:
            future_ts = [f"C+{k+1}" for k in range(n_candles)]

        return {
            'preds': preds, 'pcts': pcts,
            'last_price': round(last_price, 2),
            'direction': direction,
            'future_ts': future_ts,
            'history_prices': [float(x) for x in closes[-20:]],
        }
    except Exception as e:
        import traceback
        return {'error': f'{e}  ({traceback.format_exc().splitlines()[-1]})'}


# ─────────────────────────────────────────────
#  INTRADAY CHART
# ─────────────────────────────────────────────

def build_intraday_chart(df, symbol, interval):
    buys  = df[df['Signal'] == 1]
    sells = df[df['Signal'] == -1]

    # Only show last 200 candles for clarity on 1min
    df_plot = df.tail(200)
    buys    = buys[buys.index >= df_plot.index[0]]
    sells   = sells[sells.index >= df_plot.index[0]]

    fig = make_subplots(
        rows=5, cols=1, shared_xaxes=True,
        row_heights=[0.44, 0.14, 0.14, 0.14, 0.14],
        vertical_spacing=0.02,
        subplot_titles=["", "Volume + VP", "RSI-7", "MACD 5/13", "ADX-7 / DI"]
    )

    # ── Candlestick ──
    fig.add_trace(go.Candlestick(
        x=df_plot.index, open=df_plot['Open'], high=df_plot['High'],
        low=df_plot['Low'], close=df_plot['Close'],
        name="Price", increasing_line_color='#22c55e',
        decreasing_line_color='#ef4444', showlegend=False
    ), row=1, col=1)

    # ── EMA Ribbon: 5/9/21/50 ──
    for col, color, name, width in [
        ('EMA_5',  '#fbbf24', 'EMA 5',  1.2),
        ('EMA_9',  '#f59e0b', 'EMA 9',  1.4),
        ('EMA_21', '#ec4899', 'EMA 21', 1.6),
        ('EMA_50', '#8b5cf6', 'EMA 50', 1.8),
    ]:
        if col in df_plot.columns:
            fig.add_trace(go.Scatter(
                x=df_plot.index, y=df_plot[col],
                line=dict(color=color, width=width, dash='dash'),
                name=name, showlegend=True), row=1, col=1)

    # ── VWAP ──
    if 'VWAP' in df_plot.columns:
        fig.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot['VWAP'],
            line=dict(color='#f59e0b', width=2.5),
            name='VWAP', showlegend=True), row=1, col=1)

    # ── Volume Profile levels ──
    if 'VP_POC' in df_plot.columns and not pd.isna(df_plot['VP_POC'].iloc[-1]):
        poc = float(df_plot['VP_POC'].iloc[-1])
        vah = float(df_plot['VP_VAH'].iloc[-1])
        val = float(df_plot['VP_VAL'].iloc[-1])
        x0  = df_plot.index[0]; x1 = df_plot.index[-1]
        fig.add_shape(type="line", x0=x0, x1=x1, y0=poc, y1=poc,
                      line=dict(color="#f97316", width=2, dash="dash"),
                      row=1, col=1)
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=val, y1=vah,
                      fillcolor="rgba(99,102,241,0.06)",
                      line=dict(color="rgba(99,102,241,0.3)", width=1),
                      row=1, col=1)
        fig.add_annotation(x=x1, y=poc, text=f"POC ₹{poc:,.0f}",
                           showarrow=False, xanchor='right',
                           font=dict(size=10, color='#f97316'),
                           bgcolor='white', bordercolor='#f97316', borderwidth=1)
        fig.add_annotation(x=x1, y=vah, text=f"VAH ₹{vah:,.0f}",
                           showarrow=False, xanchor='right',
                           font=dict(size=9, color='#6366f1'),
                           bgcolor='white', bordercolor='#6366f1', borderwidth=1)
        fig.add_annotation(x=x1, y=val, text=f"VAL ₹{val:,.0f}",
                           showarrow=False, xanchor='right',
                           font=dict(size=9, color='#6366f1'),
                           bgcolor='white', bordercolor='#6366f1', borderwidth=1)

    # ── BB ──
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['BB_Upper'],
        line=dict(color='rgba(150,150,150,0.35)', width=1),
        name='BB Upper', showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['BB_Lower'],
        line=dict(color='rgba(150,150,150,0.35)', width=1),
        fill='tonexty', fillcolor='rgba(150,150,150,0.05)',
        name='BB', showlegend=False), row=1, col=1)

    # ── Supertrend ──
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Supertrend'],
        line=dict(color='#06b6d4', width=1.5), name='Supertrend'), row=1, col=1)

    # ── Pivot R1/S1 ──
    for level, color, name in [('R1','#ef4444','R1'), ('S1','#22c55e','S1')]:
        if level in df_plot.columns:
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot[level],
                line=dict(color=color, width=0.8, dash='dot'),
                name=name, showlegend=True), row=1, col=1)

    # ── CPR — Central Pivot Range ──────────────────────────
    if 'CPR_TC' in df_plot.columns and not pd.isna(df_plot['CPR_TC'].iloc[-1]):
        _x0     = df_plot.index[0]
        _x1     = df_plot.index[-1]
        _cpr_tc = float(df_plot['CPR_TC'].iloc[-1])
        _cpr_bc = float(df_plot['CPR_BC'].iloc[-1])
        _cpr_pv = float(df_plot['CPR_Pivot'].iloc[-1])
        _cpr_r1 = float(df_plot['CPR_R1'].iloc[-1]) if not pd.isna(df_plot['CPR_R1'].iloc[-1]) else None
        _cpr_s1 = float(df_plot['CPR_S1'].iloc[-1]) if not pd.isna(df_plot['CPR_S1'].iloc[-1]) else None
        _cpr_r2 = float(df_plot['CPR_R2'].iloc[-1]) if not pd.isna(df_plot['CPR_R2'].iloc[-1]) else None
        _cpr_s2 = float(df_plot['CPR_S2'].iloc[-1]) if not pd.isna(df_plot['CPR_S2'].iloc[-1]) else None
        _cpr_w  = float(df_plot['CPR_Width'].iloc[-1]) if not pd.isna(df_plot['CPR_Width'].iloc[-1]) else 0

        # CPR band (TC to BC) — shaded zone
        fig.add_shape(type="rect", x0=_x0, x1=_x1, y0=_cpr_bc, y1=_cpr_tc,
                      fillcolor="rgba(251,191,36,0.10)",
                      line=dict(color="rgba(251,191,36,0.4)", width=1),
                      row=1, col=1)

        # TC — Top Central
        fig.add_trace(go.Scatter(
            x=df_plot.index, y=[_cpr_tc] * len(df_plot),
            line=dict(color='#f59e0b', width=1.8, dash='dash'),
            name=f'CPR TC ₹{_cpr_tc:,.1f}', showlegend=True), row=1, col=1)

        # BC — Bottom Central
        fig.add_trace(go.Scatter(
            x=df_plot.index, y=[_cpr_bc] * len(df_plot),
            line=dict(color='#f59e0b', width=1.8, dash='dash'),
            name=f'CPR BC ₹{_cpr_bc:,.1f}', showlegend=True), row=1, col=1)

        # Pivot
        fig.add_trace(go.Scatter(
            x=df_plot.index, y=[_cpr_pv] * len(df_plot),
            line=dict(color='#fbbf24', width=1.2, dash='dot'),
            name=f'Pivot ₹{_cpr_pv:,.1f}', showlegend=True), row=1, col=1)

        # CPR_R1 and CPR_S1
        if _cpr_r1:
            fig.add_trace(go.Scatter(
                x=df_plot.index, y=[_cpr_r1] * len(df_plot),
                line=dict(color='#f87171', width=1.0, dash='dot'),
                name=f'CPR R1 ₹{_cpr_r1:,.1f}', showlegend=True), row=1, col=1)
        if _cpr_s1:
            fig.add_trace(go.Scatter(
                x=df_plot.index, y=[_cpr_s1] * len(df_plot),
                line=dict(color='#86efac', width=1.0, dash='dot'),
                name=f'CPR S1 ₹{_cpr_s1:,.1f}', showlegend=True), row=1, col=1)
        if _cpr_r2:
            fig.add_trace(go.Scatter(
                x=df_plot.index, y=[_cpr_r2] * len(df_plot),
                line=dict(color='#fca5a5', width=0.8, dash='longdash'),
                name=f'CPR R2 ₹{_cpr_r2:,.1f}', showlegend=True), row=1, col=1)
        if _cpr_s2:
            fig.add_trace(go.Scatter(
                x=df_plot.index, y=[_cpr_s2] * len(df_plot),
                line=dict(color='#bbf7d0', width=0.8, dash='longdash'),
                name=f'CPR S2 ₹{_cpr_s2:,.1f}', showlegend=True), row=1, col=1)

        # CPR width annotation top right
        _w_label = "NARROW ⚡" if _cpr_w < 0.4 else ("MODERATE" if _cpr_w < 0.8 else "WIDE ⚠️")
        _w_color = "#16a34a" if _cpr_w < 0.4 else ("#d97706" if _cpr_w < 0.8 else "#dc2626")
        fig.add_annotation(
            x=_x1, y=_cpr_tc,
            text=f"CPR {_w_label} ({_cpr_w:.2f}%)",
            showarrow=False, xanchor='right',
            font=dict(size=10, color=_w_color),
            bgcolor='white', bordercolor=_w_color, borderwidth=1)

    # ── Previous Day High / Low ────────────────────────────
    if 'PDH' in df_plot.columns and not pd.isna(df_plot['PDH'].iloc[-1]):
        _pdh_v = float(df_plot['PDH'].iloc[-1])
        _pdl_v = float(df_plot['PDL'].iloc[-1]) if not pd.isna(df_plot['PDL'].iloc[-1]) else None
        _x0p   = df_plot.index[0]
        _x1p   = df_plot.index[-1]
        _last_p = float(df_plot['Close'].iloc[-1])

        # PDH line — red (resistance)
        _pdh_clr = '#ef4444' if abs(_last_p - _pdh_v) / _pdh_v < 0.005 else '#f97316'
        fig.add_shape(type="line", x0=_x0p, x1=_x1p, y0=_pdh_v, y1=_pdh_v,
                      line=dict(color=_pdh_clr, width=1.5, dash='dashdot'), row=1, col=1)
        fig.add_annotation(x=_x1p, y=_pdh_v,
                           text=f"PDH ₹{_pdh_v:,.1f}",
                           showarrow=False, xanchor='right',
                           font=dict(size=10, color=_pdh_clr),
                           bgcolor='white', bordercolor=_pdh_clr, borderwidth=1)

        # PDL line — green (support)
        if _pdl_v:
            fig.add_shape(type="line", x0=_x0p, x1=_x1p, y0=_pdl_v, y1=_pdl_v,
                          line=dict(color='#22c55e', width=1.2, dash='dashdot'), row=1, col=1)
            fig.add_annotation(x=_x1p, y=_pdl_v,
                               text=f"PDL ₹{_pdl_v:,.1f}",
                               showarrow=False, xanchor='right',
                               font=dict(size=10, color='#22c55e'),
                               bgcolor='white', bordercolor='#22c55e', borderwidth=1)

    # ── Buy/Sell signals ──
    if len(buys):
        fig.add_trace(go.Scatter(x=buys.index, y=buys['Close'], mode='markers',
            marker=dict(symbol='triangle-up', size=12, color='#22c55e',
                        line=dict(color='white', width=1.5)),
            name='BUY'), row=1, col=1)
    if len(sells):
        fig.add_trace(go.Scatter(x=sells.index, y=sells['Close'], mode='markers',
            marker=dict(symbol='triangle-down', size=12, color='#ef4444',
                        line=dict(color='white', width=1.5)),
            name='SELL'), row=1, col=1)

    # ── Volume bars + Volume Profile histogram ──
    vol_colors = ['#22c55e' if c >= o else '#ef4444'
                  for c, o in zip(df_plot['Close'], df_plot['Open'])]
    fig.add_trace(go.Bar(x=df_plot.index, y=df_plot['Volume'],
        marker_color=vol_colors, opacity=0.6,
        name='Volume', showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Volume_MA'],
        line=dict(color='orange', width=1.2),
        name='Vol MA', showlegend=False), row=2, col=1)

    # ── RSI ──
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['RSI'],
        line=dict(color='#a855f7', width=1.5),
        name='RSI-7', showlegend=False), row=3, col=1)
    for lvl, col in [(70,'rgba(239,68,68,0.3)'),(30,'rgba(34,197,94,0.3)'),(50,'rgba(255,255,255,0.1)')]:
        fig.add_hline(y=lvl, line_dash="dash", line_color=col, row=3, col=1)

    # ── MACD ──
    hist_colors = ['#22c55e' if v >= 0 else '#ef4444' for v in df_plot['MACD_Hist']]
    fig.add_trace(go.Bar(x=df_plot.index, y=df_plot['MACD_Hist'],
        marker_color=hist_colors, opacity=0.7,
        name='MACD Hist', showlegend=False), row=4, col=1)
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MACD'],
        line=dict(color='#3b82f6', width=1.5),
        name='MACD', showlegend=False), row=4, col=1)
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MACD_Signal'],
        line=dict(color='#ef4444', width=1.5),
        name='Signal', showlegend=False), row=4, col=1)

    # ── ADX ──
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['ADX'],
        line=dict(color='white', width=1.5),
        name='ADX', showlegend=False), row=5, col=1)
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Plus_DI'],
        line=dict(color='#22c55e', width=1, dash='dash'),
        name='+DI', showlegend=False), row=5, col=1)
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Minus_DI'],
        line=dict(color='#ef4444', width=1, dash='dash'),
        name='-DI', showlegend=False), row=5, col=1)
    fig.add_hline(y=20, line_dash="dot", line_color="rgba(255,255,255,0.3)", row=5, col=1)

    # Data source in title
    src = st.session_state.get('data_source', '')
    fig.update_layout(
        title=dict(
            text=f"<b>{symbol.replace('.NS','')}</b> — {interval} Chart  <span style='font-size:12px;color:#94a3b8'>{src}</span>",
            font=dict(color='#1a2035', size=15)),
        height=820, paper_bgcolor='#ffffff', plot_bgcolor='#fafbfc',
        font=dict(color='#4a5568', family='Outfit'),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='left', x=0,
                    font=dict(size=11), bgcolor='rgba(255,255,255,0.95)',
                    bordercolor='#e8ecf0', borderwidth=1),
        margin=dict(l=50, r=20, t=70, b=20),
    )
    for i in range(1, 6):
        fig.update_xaxes(gridcolor='#e8ecf0', row=i, col=1)
        fig.update_yaxes(gridcolor='#e8ecf0', row=i, col=1)
    return fig

with st.sidebar:
    _mkt_open = market_open()

    # ── Logo ─────────────────────────────────────────────
    st.markdown(f"""
    <div class='sb-logo'>
        <div class='sb-logo-icon'>
            <svg xmlns='http://www.w3.org/2000/svg' width='20' height='20' fill='none'
                 viewBox='0 0 24 24' stroke='#1a2035' stroke-width='2.5'>
                <polyline points='13 2 13 9 19 9'/>
                <polyline points='11 22 11 15 5 15'/>
                <path d='M3 3h7v7H3z'/>
                <path d='M14 14h7v7h-7z'/>
            </svg>
        </div>
        <div>
            <div class='sb-logo-name'>Investo</div>
            <div class='sb-logo-tag'>Intraday</div>
        </div>
    </div>
    <div class='sb-market-strip'>
        <div class='{"sb-mkt-open" if _mkt_open else "sb-mkt-closed"}'>
            <div class='{"sb-mkt-dot-open" if _mkt_open else "sb-mkt-dot-closed"}'></div>
            {"MARKET OPEN" if _mkt_open else "MARKET CLOSED"}
        </div>
        <div class='sb-mkt-time'>{ist_now().strftime('%H:%M IST')}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Portfolio strip ───────────────────────────────────
    _port_all   = load_portfolio()
    _open_pos   = [p for p in _port_all if p.get('status') == 'OPEN']
    _port_count = len(_open_pos)
    _total_inv  = sum(_f(p.get('actual_cost', _f(p.get('investment', 0)))) for p in _open_pos)
    _unreal_val = 0.0
    try:
        for _op in _open_pos:
            _s = _op.get('symbol', '')
            if _s:
                try:
                    _hticker = _s if _s.endswith('.NS') else _s + '.NS'
                    _h  = yf.Ticker(_hticker).history(period='1d', interval='5m')
                    _cp = float(_h['Close'].iloc[-1]) if not _h.empty else _f(_op.get('entry', 0))
                except Exception:
                    _cp = _f(_op.get('entry', 0))
                _unreal_val += (_cp - _f(_op.get('entry', 0))) * int(_f(_op.get('qty', 0)))
    except Exception:
        pass
    _unreal_pct = round(_unreal_val / _total_inv * 100, 2) if _total_inv > 0 else 0.0
    _pnl_cls    = "sb-port-pnl-pos" if _unreal_val >= 0 else "sb-port-pnl-neg"
    _pnl_sign   = "+" if _unreal_val >= 0 else ""

    st.markdown(f"""
    <div class='sb-port-strip'>
        <div class='sb-port-label'>Portfolio · {_port_count} open</div>
        <div class='sb-port-row'>
            <div class='sb-port-val'>{("₹"+f"{_total_inv:,.0f}") if _total_inv else "₹0"}</div>
            <div class='{_pnl_cls}'>{_pnl_sign}{_unreal_pct:.2f}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Nav ───────────────────────────────────────────────
    st.markdown("""<style>
    div[data-testid="stSidebar"] .stRadio { display:none !important; }
    </style>""", unsafe_allow_html=True)

    if 'active_page' not in st.session_state:
        st.session_state['active_page'] = "🌅  Dashboard"

    _NAV = [
        ("🌅  Dashboard",     "Dashboard"),
        ("📊  Scanner",       "Scanner"),
        ("🚀  Early Movers",  "Early Movers"),
        ("🔓  ORB Scanner",   "ORB Scanner"),
        ("💼  Portfolio",     "Portfolio"),
        ("🔔  Alert Log",     "Alert Log"),
        ("📈  SMA Weekly",    "SMA Weekly"),
        ("📅  Monthly Swing", "Monthly Swing"),
        ("🧪  Backtest",      "Backtest"),
        ("🏆  Sector Leaders","Sector Leaders"),
    ]
    # Inject sidebar button styles once — clean single-item nav
    st.markdown("""
    <style>
    /* Remove default Streamlit button styling in sidebar nav */
    section[data-testid="stSidebar"] div[data-testid="stButton"] button {
        background: transparent !important;
        color: #94a3b8 !important;
        border: 1px solid transparent !important;
        border-radius: 10px !important;
        text-align: left !important;
        font-size: 13px !important;
        font-weight: 400 !important;
        padding: 10px 14px !important;
        width: 100% !important;
        transition: all 0.15s !important;
        box-shadow: none !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stButton"] button:hover {
        background: rgba(255,255,255,0.06) !important;
        color: #e2e8f0 !important;
        border-color: rgba(255,255,255,0.1) !important;
    }
    </style>""", unsafe_allow_html=True)
    st.markdown("<div class='sb-nav-section'>Navigation</div>", unsafe_allow_html=True)
    for _pkey, _plabel in _NAV:
        _active  = st.session_state['active_page'] == _pkey
        _al_cnt  = len(st.session_state.get(ALERT_LOG_KEY, []))
        _icon    = _pkey.split()[0]   # emoji from key e.g. "🌅"

        # Build display label with badge count
        _cnt = 0
        if _plabel == "Portfolio":
            _cnt = _port_count
        elif _plabel == "Alert Log":
            _cnt = _al_cnt
        elif _plabel == "Early Movers":
            _cnt = len(st.session_state.get('early_movers', []))
        elif _plabel == "ORB Scanner":
            _cnt = len(st.session_state.get('orb_results', []))

        _badge_str = f"  ({_cnt})" if _cnt > 0 else ""
        _disp      = f"{_icon}  {_plabel}{_badge_str}"

        # Single button — active state via inline style prefix on label
        _disp_styled = f"{'→ ' if _active else '   '}{_disp}"

        if st.button(_disp_styled, key=f"navbtn_{_pkey}",
                     use_container_width=True):
            st.session_state['active_page'] = _pkey
            st.rerun()

    active_page = st.session_state.get('active_page', "🌅  Dashboard")

    # ── Config ────────────────────────────────────────────
    st.markdown("<hr class='sb-section-divider'>", unsafe_allow_html=True)
    st.markdown("<div class='sb-section-label'>⚙ Config</div>", unsafe_allow_html=True)

    interval_label = st.selectbox("📊 Scanner Timeframe",
        ["5min — Standard", "1min — Real-Time", "3min — Fast", "15min — Swing", "60min — Positional"],
        help="Only applies to Main Scanner. Early Movers uses 1-min (hardcoded). ORB uses 5-min (hardcoded).")
    interval_map = {
        "1min — Real-Time":   "1minute",
        "3min — Fast":        "3minute",
        "5min — Standard":    "5minute",
        "15min — Swing":      "15minute",
        "60min — Positional": "60minute",
    }
    interval = interval_map[interval_label]
    period   = "1d"

    capital  = st.number_input("Capital (₹)", min_value=10000, max_value=10000000,
                                value=100000, step=10000, format="%d")
    risk_pct = st.slider("Risk / Trade (%)", min_value=0.5, max_value=3.0, value=1.0, step=0.5)

    st.markdown(f"""
    <div style='display:flex;justify-content:space-between;align-items:center;
                background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.2);
                border-radius:8px;padding:8px 14px;margin:4px 0'>
        <span style='font-size:10px;font-weight:700;color:#64748b;letter-spacing:1px;text-transform:uppercase'>Max Loss</span>
        <span style='font-size:16px;font-weight:800;color:#ef4444;font-family:JetBrains Mono,monospace'>₹{capital * risk_pct / 100:,.0f}</span>
    </div>""", unsafe_allow_html=True)

    # ── Scanner ───────────────────────────────────────────
    st.markdown("<hr class='sb-section-divider'>", unsafe_allow_html=True)
    st.markdown("<div class='sb-section-label'>🔍 Scanner</div>", unsafe_allow_html=True)

    scan_mode = st.radio("Universe",
        ["🔵 Largecap (Nifty 50)",
         "🟡 Midcap (Nifty Midcap 100)",
         "🟠 Smallcap",
         "📊 Nifty 500 (All)",
         "Custom Watchlist"])
    custom_stocks = []
    if scan_mode == "Custom Watchlist":
        custom_stocks = st.multiselect("Stocks", POPULAR_STOCKS,
            default=["RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS"])
        extra = st.text_input("Add symbol", "")
        if extra and extra.upper() not in custom_stocks:
            custom_stocks.append(extra.upper())

    selected_stocks = (
        LARGECAP_STOCKS  if scan_mode == "🔵 Largecap (Nifty 50)"          else
        MIDCAP_STOCKS    if scan_mode == "🟡 Midcap (Nifty Midcap 100)"    else
        SMALLCAP_STOCKS  if scan_mode == "🟠 Smallcap"                     else
        custom_stocks    if scan_mode == "Custom Watchlist"                  else
        POPULAR_STOCKS   # Nifty 500 (All)
    )

    min_verdict = st.select_slider("Min Verdict",
        options=["❌ AVOID","⚠️ NEUTRAL","⭐ WATCH","⭐⭐ BUY","⭐⭐⭐ STRONG BUY"],
        value="⭐ WATCH")


    run_btn = st.button("▶  Scan Now", use_container_width=True, type="primary")

    # ── Anthropic AI API Key ───────────────────────────────
    st.markdown("<hr class='sb-section-divider'>", unsafe_allow_html=True)
    st.markdown("<div class='sb-section-label'>🤖 AI Validation (Claude)</div>",
                unsafe_allow_html=True)

    _ant_key     = load_anthropic_key()
    _ant_masked  = ('sk-ant-...'+_ant_key[-6:]) if len(_ant_key) > 10 else ''
    _ant_status  = '🟢 Connected' if _ant_key else '🔴 Not set'
    st.markdown(
        f"<div style='font-size:11px;color:#64748b;margin-bottom:6px'>"
        f"Status: <b>{_ant_status}</b>"
        f"{f' ({_ant_masked})' if _ant_masked else ''}"
        f"</div>",
        unsafe_allow_html=True)

    with st.expander("⚙️ Set Anthropic API Key", expanded=not bool(_ant_key)):
        _new_ant_key = st.text_input(
            "Anthropic API Key",
            value=_ant_key,
            type="password",
            placeholder="sk-ant-api03-...",
            key="ant_key_input",
            help="Get your key from console.anthropic.com")
        _ant_col1, _ant_col2 = st.columns(2)
        with _ant_col1:
            if st.button("💾 Save Key", key="save_ant_key",
                         use_container_width=True):
                if _new_ant_key.startswith('sk-ant-'):
                    save_anthropic_key(_new_ant_key)
                    st.session_state['anthropic_api_key'] = _new_ant_key
                    st.success("✅ Saved!")
                    st.rerun()
                else:
                    st.error("Key must start with sk-ant-")
        with _ant_col2:
            if st.button("🗑️ Clear", key="clear_ant_key",
                         use_container_width=True):
                st.session_state.pop('anthropic_api_key', None)
                try:
                    ANTHROPIC_CREDS_FILE.unlink(missing_ok=True)
                except Exception:
                    pass
                st.rerun()
        st.markdown(
            "<div style='font-size:10px;color:#94a3b8;margin-top:4px'>"
            "Key saved to ~/Downloads/anthropic_creds.json<br>"
            "Get API key: console.anthropic.com → API Keys"
            "</div>", unsafe_allow_html=True)

    # ── Kite API ──────────────────────────────────────────
    st.markdown("<hr class='sb-section-divider'>", unsafe_allow_html=True)
    st.markdown("<div class='sb-section-label'>🔌 Zerodha Kite</div>", unsafe_allow_html=True)

    kite_client = get_kite_client()
    creds       = load_kite_creds()

    if kite_client is not None:
        st.markdown("""
        <div class='sb-kite-connected'>
            <div class='sb-kite-label' style='color:#34d399'>✅ Connected — Real-Time</div>
            <div class='sb-kite-sub'   style='color:#6ee7b7'>1min live data active</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Disconnect", key="kite_disconnect"):
            creds.pop('access_token', None); creds.pop('token_date', None)
            save_kite_creds(creds); st.session_state.pop('kite', None); st.rerun()
    else:
        if not KITE_AVAILABLE:
            st.markdown("""
            <div class='sb-kite-disconnected'>
                <div class='sb-kite-label' style='color:#f87171'>kiteconnect not installed</div>
                <div class='sb-kite-sub'   style='color:#fca5a5'>pip3 install kiteconnect</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='sb-kite-disconnected'>
                <div class='sb-kite-label' style='color:#94a3b8'>⏳ Not connected</div>
                <div class='sb-kite-sub'   style='color:#64748b'>yfinance fallback active</div>
            </div>""", unsafe_allow_html=True)
            with st.expander("Connect Kite", expanded=False):
                _api_key = st.text_input("API Key", value=creds.get('api_key',''),
                                          type="password", key="kite_api_key")
                _api_sec = st.text_input("API Secret", value=creds.get('api_secret',''),
                                          type="password", key="kite_api_secret")
                if _api_key and _api_sec:
                    if st.button("Generate Login URL", key="kite_gen_url"):
                        try:
                            _kite_tmp  = KiteConnect(api_key=_api_key)
                            _login_url = _kite_tmp.login_url()
                            st.markdown(f"""
                            <div style='background:#1e293b;border-radius:8px;padding:10px 12px;margin-top:8px'>
                                <div style='font-size:11px;color:#93c5fd;line-height:1.7;margin-bottom:8px'>
                                    1. Click link → Login to Zerodha<br>
                                    2. Copy <b>request_token</b> from URL<br>
                                    3. Paste below → Connect
                                </div>
                                <a href='{_login_url}' target='_blank'
                                   style='color:#f59e0b;font-size:12px;font-weight:700'>
                                    🔗 Open Zerodha Login →
                                </a>
                            </div>""", unsafe_allow_html=True)
                            creds['api_key'] = _api_key; creds['api_secret'] = _api_sec
                            save_kite_creds(creds)
                        except Exception as e:
                            st.error(f"Error: {e}")
                    _req_token = st.text_input("Request Token", key="kite_req_token",
                                               placeholder="Paste from redirect URL")
                    if _req_token and st.button("⚡ Connect", key="kite_connect"):
                        try:
                            _kite_conn    = KiteConnect(api_key=_api_key)
                            _sess         = _kite_conn.generate_session(_req_token, api_secret=_api_sec)
                            _access_token = _sess["access_token"]
                            _kite_conn.set_access_token(_access_token)
                            creds.update({'api_key':_api_key,'api_secret':_api_sec,
                                          'access_token':_access_token,
                                          'token_date':datetime.now().strftime('%Y-%m-%d')})
                            save_kite_creds(creds); st.session_state['kite'] = _kite_conn
                            st.success("✅ Connected!"); st.rerun()
                        except Exception as e:
                            st.error(f"Failed: {e}")

    _dsrc     = st.session_state.get('data_source', 'Not scanned yet')
    _dsrc_bg  = "rgba(52,211,153,0.08)"  if "Kite" in _dsrc else "rgba(251,191,36,0.08)"
    _dsrc_bdr = "rgba(52,211,153,0.2)"   if "Kite" in _dsrc else "rgba(251,191,36,0.2)"
    _dsrc_col = "#34d399"                if "Kite" in _dsrc else "#fbbf24"
    st.markdown(f"""
    <div style='margin:6px 0;padding:7px 12px;background:{_dsrc_bg};
                border:1px solid {_dsrc_bdr};border-radius:8px'>
        <div style='font-size:10px;font-weight:700;color:{_dsrc_col}'>{_dsrc}</div>
    </div>""", unsafe_allow_html=True)

    # ── Alerts ───────────────────────────────────────────
    st.markdown("<hr class='sb-section-divider'>", unsafe_allow_html=True)
    _init_alert_log()
    _alerts  = st.session_state.get(ALERT_LOG_KEY, [])
    _al_cnt2 = len(_alerts)
    st.markdown(f"<div class='sb-section-label'>🔔 Alerts {f'({_al_cnt2})' if _al_cnt2 else ''}</div>",
                unsafe_allow_html=True)

    _AL_C = {
        'BUY':         ('#34d399','rgba(52,211,153,0.1)'),
        'STRONG_BUY':  ('#4ade80','rgba(74,222,128,0.1)'),
        'VOL_SURGE':   ('#fbbf24','rgba(251,191,36,0.1)'),
        'VWAP_BREAK':  ('#f87171','rgba(248,113,113,0.1)'),
        'RSI_OB':      ('#f87171','rgba(248,113,113,0.1)'),
        'STOP_LOSS':   ('#fca5a5','rgba(252,165,165,0.1)'),
        'TARGET_T1':   ('#34d399','rgba(52,211,153,0.1)'),
        'TARGET_T2':   ('#34d399','rgba(52,211,153,0.1)'),
        'TARGET_T3':   ('#6ee7b7','rgba(110,231,183,0.1)'),
        'TARGET_T4':   ('#6ee7b7','rgba(110,231,183,0.1)'),
        'TIME_WARN':   ('#fbbf24','rgba(251,191,36,0.1)'),
        'ORB_VOL':     ('#c4b5fd','rgba(196,181,253,0.1)'),
        'ORB_GAP':     ('#34d399','rgba(52,211,153,0.1)'),
        'ORB_VWAP':    ('#fbbf24','rgba(251,191,36,0.1)'),
        'ORB_HIGH':    ('#7dd3fc','rgba(125,211,252,0.1)'),
        'ORB_MOMENTUM':('#fb923c','rgba(251,146,60,0.1)'),
    }
    if _alerts:
        for _al in _alerts[:5]:
            _tc, _bc = _AL_C.get(_al['type'], ('#94a3b8','rgba(148,163,184,0.06)'))
            st.markdown(f"""
            <div style='margin:0 4px 5px;padding:9px 12px;background:{_bc};
                        border-left:3px solid {_tc};border-radius:0 8px 8px 0'>
                <div style='font-size:12px;font-weight:700;color:{_tc}'>
                    {_al.get('icon','📣')} {_al['symbol']}
                    <span style='font-size:9px;opacity:0.5;float:right'>{_al['time']}</span>
                </div>
                <div style='font-size:10px;color:#64748b;margin-top:3px;line-height:1.4'>
                    {_al['message'][:55]}{'…' if len(_al['message'])>55 else ''}
                </div>
            </div>""", unsafe_allow_html=True)
        if st.button("Clear alerts", key="clear_alerts_sidebar"):
            st.session_state[ALERT_LOG_KEY] = []; st.rerun()
    else:
        st.markdown("""<div style='padding:10px 12px;font-size:11px;color:#334155;text-align:center'>
            No alerts · Run scanner first</div>""", unsafe_allow_html=True)

    # ── F&O Stocks List Management ────────────────────────
    # NSE revises F&O list every ~6 months — keep it current
    # without needing code changes each time
    with st.expander("📋 F&O Stocks List"):
        _fno_active_set, _fno_source = load_custom_fno_list()
        st.caption(f"Active: **{len(_fno_active_set)} symbols** · Source: *{_fno_source}*")

        _fno_upload = st.file_uploader(
            "Upload updated F&O list (CSV/Excel)",
            type=['csv', 'xlsx', 'xls'],
            key="fno_list_upload",
            help="Needs a 'Symbol' column — same format as the NSE "
                 "F&O list or any broker's F&O export. "
                 "Saved permanently until you upload a new one.")

        if _fno_upload is not None:
            # Guard against infinite rerun loop:
            # Streamlit keeps the uploaded file across reruns,
            # so without this check, save_custom_fno_list() +
            # st.rerun() would fire forever on every refresh.
            _fno_file_id = f"{_fno_upload.name}_{_fno_upload.size}"
            if st.session_state.get('fno_last_processed_file') != _fno_file_id:
                _fno_ok, _fno_msg, _fno_count = save_custom_fno_list(_fno_upload)
                st.session_state['fno_last_processed_file'] = _fno_file_id
                if _fno_ok:
                    st.success(f"✅ {_fno_msg}")
                    st.rerun()
                else:
                    st.error(f"❌ {_fno_msg}")
            else:
                st.caption("✅ This file is already loaded — upload a different file to update again.")

        if _fno_source == 'custom CSV (saved)':
            if st.button("↩️ Reset to built-in list", key="fno_reset_btn"):
                reset_fno_list_to_default()
                st.rerun()

    st.markdown("""
    <div class='sb-disclaimer'>
        ⚠️ High risk · Educational only<br>Not financial advice
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MAIN CONTENT
# ─────────────────────────────────────────────

if active_page == "🌅  Dashboard":
    _show_dashboard   = True
    _show_scanner     = False
    _show_portfolio   = False
    _show_alertlog    = False
    _show_earlymovers = False
    _show_orb         = False
    _show_smaweekly   = False
    _show_monthlyswing= False
    _show_backtest    = False
    _show_sectorleaders= False
elif active_page == "💼  Portfolio":
    _show_dashboard   = False
    _show_scanner     = False
    _show_portfolio   = True
    _show_alertlog    = False
    _show_earlymovers = False
    _show_orb         = False
    _show_smaweekly   = False
    _show_monthlyswing= False
    _show_backtest    = False
    _show_sectorleaders= False
elif active_page == "🔔  Alert Log":
    _show_dashboard   = False
    _show_scanner     = False
    _show_portfolio   = False
    _show_alertlog    = True
    _show_earlymovers = False
    _show_orb         = False
    _show_smaweekly   = False
    _show_monthlyswing= False
    _show_backtest    = False
    _show_sectorleaders= False
elif active_page == "🚀  Early Movers":
    _show_dashboard   = False
    _show_scanner     = False
    _show_portfolio   = False
    _show_alertlog    = False
    _show_earlymovers = True
    _show_orb         = False
    _show_smaweekly   = False
    _show_monthlyswing= False
    _show_backtest    = False
    _show_sectorleaders= False
elif active_page == "🔓  ORB Scanner":
    _show_dashboard   = False
    _show_scanner     = False
    _show_portfolio   = False
    _show_alertlog    = False
    _show_earlymovers = False
    _show_orb         = True
elif active_page == "📊  Scanner":
    _show_dashboard   = False
    _show_scanner     = True
    _show_portfolio   = False
    _show_alertlog    = False
    _show_earlymovers = False
    _show_orb         = False
    _show_smaweekly   = False
    _show_monthlyswing= False
    _show_backtest    = False
    _show_sectorleaders= False
elif active_page == "📈  SMA Weekly":
    _show_dashboard   = False
    _show_scanner     = False
    _show_portfolio   = False
    _show_alertlog    = False
    _show_earlymovers = False
    _show_orb         = False
    _show_smaweekly   = True
    _show_monthlyswing= False
    _show_backtest    = False
    _show_sectorleaders= False
elif active_page == "📅  Monthly Swing":
    _show_dashboard   = False
    _show_scanner     = False
    _show_portfolio   = False
    _show_alertlog    = False
    _show_earlymovers = False
    _show_orb         = False
    _show_smaweekly   = False
    _show_monthlyswing= True
    _show_backtest    = False
    _show_sectorleaders= False
elif active_page == "🧪  Backtest":
    _show_dashboard    = False
    _show_scanner      = False
    _show_portfolio    = False
    _show_alertlog     = False
    _show_earlymovers  = False
    _show_orb          = False
    _show_smaweekly    = False
    _show_monthlyswing = False
    _show_backtest     = True
    _show_sectorleaders= False
elif active_page == "🏆  Sector Leaders":
    _show_dashboard    = False
    _show_scanner      = False
    _show_portfolio    = False
    _show_alertlog     = False
    _show_earlymovers  = False
    _show_orb          = False
    _show_smaweekly    = False
    _show_monthlyswing = False
    _show_backtest     = False
    _show_sectorleaders= True
else:
    # Default → Dashboard
    _show_dashboard   = True
    _show_scanner     = False
    _show_portfolio   = False
    _show_alertlog    = False
    _show_earlymovers = False
    _show_orb         = False
    _show_smaweekly   = False
    _show_monthlyswing= False
    _show_backtest    = False
    _show_sectorleaders= False

# ─────────────────────────────────────────────
#  DASHBOARD PAGE
#  Pre-market intelligence + live market conditions
#  Opens by default every morning
# ─────────────────────────────────────────────
if _show_dashboard:

    # ── Data source status ────────────────────────────────
    _dash_kite    = get_kite_client()
    _dash_kite_on = _dash_kite is not None
    _dash_src_lbl = 'Kite API — Real-time' if _dash_kite_on else 'yfinance — 15 min delay'
    _dash_src_clr = '#16a34a' if _dash_kite_on else '#d97706'
    _dash_src_bg  = '#dcfce7' if _dash_kite_on else '#fef3c7'
    _dash_src_ico = '🟢' if _dash_kite_on else '🟡'

    st.markdown(f"""
    <div class='topbar'>
        <div>
            <div class='topbar-title'>🌅 Dashboard — Today's Market Intelligence</div>
            <div class='topbar-subtitle'>
                Pre-market conditions · Global cues · Strategy for today ·
                Live market state
            </div>
        </div>
        <div style='display:flex;align-items:center;gap:8px'>
            <div style='background:{_dash_src_bg};border:1px solid {_dash_src_clr}44;
                        border-radius:8px;padding:6px 14px;text-align:center'>
                <div style='font-size:10px;font-weight:700;color:{_dash_src_clr};
                            letter-spacing:1px'>DATA SOURCE</div>
                <div style='font-size:13px;font-weight:700;color:{_dash_src_clr};
                            margin-top:2px'>{_dash_src_ico} {_dash_src_lbl}</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    # ── Fetch all dashboard data ──────────────────────────
    _db_now  = ist_now()
    _db_hour = _db_now.hour
    _db_min  = _db_now.minute
    _db_tm   = _db_hour * 60 + _db_min

    # ── Expiry detection ──────────────────────────────────
    _expiry_info  = detect_expiry(_db_now)
    _exp_type     = _expiry_info['expiry_type']
    _exp_monthly  = _expiry_info['is_monthly']
    _is_expiry_db = _expiry_info['is_expiry']

    # ── Fetch global markets ──────────────────────────────
    @st.cache_data(ttl=900)  # 15 min cache
    def fetch_global_markets():
        _data = {}
        _tickers = {
            'S&P 500':   '^GSPC',
            'Nasdaq':    '^IXIC',
            'Dow Jones': '^DJI',
            'Nikkei':    '^N225',
            'Hang Seng': '^HSI',
            'Crude Oil': 'CL=F',
            'Gold':      'GC=F',
            'USD/INR':   'USDINR=X',
        }
        for _name, _sym in _tickers.items():
            try:
                _t   = yf.Ticker(_sym)
                _h   = _t.history(period='5d', interval='1d')
                if _h is not None and len(_h) >= 2:
                    _last  = float(_h['Close'].iloc[-1])
                    _prev  = float(_h['Close'].iloc[-2])
                    _chg   = round((_last - _prev) / _prev * 100, 2)
                    _data[_name] = {'price': round(_last, 2), 'chg': _chg, 'sym': _sym}
            except Exception:
                pass
        return _data

    # Nifty + VIX — auto-fetch on first load if not in session state
    if 'nifty_context' not in st.session_state or not st.session_state.get('nifty_context'):
        with st.spinner('🔍 Loading market state...'):
            _db_kite_init = get_kite_client()
            _db_ctx_init  = get_nifty_market_state(kite=_db_kite_init)
            st.session_state['nifty_context']      = _db_ctx_init
            st.session_state['nifty_market_state'] = _db_ctx_init['state']
            st.session_state['nifty_ctx_date']     = _db_now.strftime('%Y-%m-%d %H:%M')

    _db_mkt_ctx = st.session_state.get('nifty_context', {})
    _db_vix     = _db_mkt_ctx.get('vix')
    _db_vix_lvl = _db_mkt_ctx.get('vix_level', 'UNKNOWN')
    _db_nifty   = _db_mkt_ctx.get('nifty_chg', 0)
    _db_nstate  = _db_mkt_ctx.get('state', 'UNKNOWN')

    # Sector momentum (from last scan)
    _db_sectors = st.session_state.get('sector_momentum', {})

    # Open positions summary
    _db_port    = load_portfolio()
    _db_open    = [p for p in _db_port if p.get('status') == 'OPEN']
    _db_closed_today = [p for p in _db_port
                        if p.get('status') != 'OPEN'
                        and p.get('exit_date', '').startswith(_db_now.strftime('%d %b %Y'))]
    _db_today_pnl = sum(_f(p.get('net_pl', 0)) for p in _db_closed_today)

    # ─────────────────────────────────────────────────────
    # SECTION 1 — DAY TYPE + TIME CONTEXT
    # ─────────────────────────────────────────────────────
    # Day type banner
    if _is_expiry_db:
        _exp_labels = {
            'NIFTY_MONTHLY':   ('🚨', 'Nifty MONTHLY Expiry', '#450a0a', '#fca5a5', '#dc2626'),
            'NIFTY_WEEKLY':    ('⚠️', 'Nifty Weekly Expiry', '#1c1917', '#fbbf24', '#d97706'),
            'BANKNIFTY_WEEKLY':('⚠️', 'Bank Nifty Weekly Expiry', '#1c1917', '#fbbf24', '#d97706'),
        }
        _eico, _elbl, _ebg, _etc, _ebdr = _exp_labels.get(_exp_type, ('⚠️','Expiry','#1c1917','#fbbf24','#d97706'))
        st.markdown(
            f"<div style='background:{_ebg};border:2px solid {_ebdr};"
            f"border-radius:12px;padding:12px 18px;margin-bottom:12px;"
            f"display:flex;align-items:center;gap:12px'>"
            f"<span style='font-size:22px'>{_eico}</span>"
            f"<div>"
            f"<div style='font-size:15px;font-weight:800;color:{_etc}'>{_elbl} Today</div>"
            f"<div style='font-size:11px;color:{_etc};opacity:0.8;margin-top:2px'>"
            f"Entry rules changed · Best window: 10:00–10:30 AM or 1:30–2:30 PM · "
            f"Exit by 2:30 PM · Banking stocks: avoid</div>"
            f"</div></div>", unsafe_allow_html=True)

    # Time context bar
    _time_windows = [
        (555, 575,  "⏳ 9:15–9:35 AM",   "Warmup — indicators not ready. Use Early Movers only.",          "#d97706","#fffbeb"),
        (575, 690,  "🟢 9:35–11:30 AM",  "BEST WINDOW — all indicators ready, strongest signals.",         "#15803d","#f0fdf4"),
        (690, 810,  "🟡 11:30–1:30 PM",  "Lunch zone — avoid new entries, let positions run.",             "#d97706","#fffbeb"),
        (810, 870,  "🟢 1:30–2:30 PM",   "Second wind — good setups form again.",                          "#15803d","#f0fdf4"),
        (870, 915,  "🔴 2:30–3:15 PM",   "Danger zone — only exit, no new entries.",                       "#dc2626","#fff5f5"),
        (915, 9999, "🚫 After 3:15 PM",  "Square off zone — close all positions.",                         "#7f1d1d","#fef2f2"),
    ]
    _tw_label = "⚪ Market Closed"
    _tw_desc  = "NSE trading hours: 9:15 AM – 3:30 PM IST on weekdays."
    _tw_clr   = "#64748b"; _tw_bg = "#f8fafc"
    if market_open():
        for _ts, _te, _tl, _td, _tc, _tbg in _time_windows:
            if _ts <= _db_tm < _te:
                _tw_label = _tl; _tw_desc = _td; _tw_clr = _tc; _tw_bg = _tbg
                break
    st.markdown(
        f"<div style='background:{_tw_bg};border:1px solid {_tw_clr}33;"
        f"border-radius:10px;padding:10px 18px;margin-bottom:14px;"
        f"display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px'>"
        f"<div style='font-size:14px;font-weight:700;color:{_tw_clr}'>{_tw_label}</div>"
        f"<div style='font-size:12px;color:{_tw_clr};opacity:0.8'>{_tw_desc}</div>"
        f"<div style='font-size:11px;color:{_tw_clr};font-family:var(--font-mono)'>"
        f"{_db_now.strftime('%H:%M IST')}</div>"
        f"</div>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────
    # SECTION 2 — LIVE MARKET STATE
    # ─────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>📊 Live Market State</div>",
                unsafe_allow_html=True)

    _mk1, _mk2, _mk3, _mk4 = st.columns(4)

    # Nifty
    _nc = {'BULL':'#16a34a','SIDEWAYS':'#d97706','BEAR':'#dc2626','UNKNOWN':'#64748b'}.get(_db_nstate,'#64748b')
    with _mk1:
        st.markdown(
            f"<div style='background:{_nc}15;border:1px solid {_nc}44;"
            f"border-radius:10px;padding:12px 14px'>"
            f"<div style='font-size:10px;font-weight:700;color:{_nc};letter-spacing:1px'>NIFTY 50</div>"
            f"<div style='font-size:22px;font-weight:800;color:{_nc};"
            f"font-family:JetBrains Mono;margin:4px 0'>"
            f"{'+' if _db_nifty>=0 else ''}{_db_nifty:.2f}%</div>"
            f"<div style='font-size:11px;color:{_nc}'>{_db_nstate}</div>"
            f"</div>", unsafe_allow_html=True)

    # VIX
    _vc = {'CALM':'#16a34a','NORMAL':'#16a34a','ELEVATED':'#d97706',
           'HIGH':'#ea580c','EXTREME':'#dc2626','CRISIS':'#7f1d1d','UNKNOWN':'#64748b'}.get(_db_vix_lvl,'#64748b')
    with _mk2:
        st.markdown(
            f"<div style='background:{_vc}15;border:1px solid {_vc}44;"
            f"border-radius:10px;padding:12px 14px'>"
            f"<div style='font-size:10px;font-weight:700;color:{_vc};letter-spacing:1px'>INDIA VIX</div>"
            f"<div style='font-size:22px;font-weight:800;color:{_vc};"
            f"font-family:JetBrains Mono;margin:4px 0'>"
            f"{f'{_db_vix:.2f}' if _db_vix else '—'}</div>"
            f"<div style='font-size:11px;color:{_vc}'>{_db_vix_lvl}</div>"
            f"</div>", unsafe_allow_html=True)

    # Position size guidance
    _ps_map = {
        'CALM':    ('100%','#16a34a','Full position size'),
        'NORMAL':  ('100%','#16a34a','Full position size'),
        'ELEVATED':('100%','#16a34a','Full position size'),
        'HIGH':    ('70%', '#d97706','Reduce size by 30%'),
        'EXTREME': ('50%', '#ea580c','Reduce size by 50%'),
        'CRISIS':  ('0%',  '#dc2626','Avoid intraday'),
        'UNKNOWN': ('100%','#64748b','Unknown VIX'),
    }
    _ps_pct, _ps_clr, _ps_lbl = _ps_map.get(_db_vix_lvl, ('100%','#64748b','—'))
    with _mk3:
        st.markdown(
            f"<div style='background:{_ps_clr}15;border:1px solid {_ps_clr}44;"
            f"border-radius:10px;padding:12px 14px'>"
            f"<div style='font-size:10px;font-weight:700;color:{_ps_clr};letter-spacing:1px'>POSITION SIZE</div>"
            f"<div style='font-size:22px;font-weight:800;color:{_ps_clr};"
            f"font-family:JetBrains Mono;margin:4px 0'>{_ps_pct}</div>"
            f"<div style='font-size:11px;color:{_ps_clr}'>{_ps_lbl}</div>"
            f"</div>", unsafe_allow_html=True)

    # Today P&L
    _pnl_clr = '#16a34a' if _db_today_pnl >= 0 else '#dc2626'
    with _mk4:
        st.markdown(
            f"<div style='background:{_pnl_clr}15;border:1px solid {_pnl_clr}44;"
            f"border-radius:10px;padding:12px 14px'>"
            f"<div style='font-size:10px;font-weight:700;color:{_pnl_clr};letter-spacing:1px'>TODAY P&L</div>"
            f"<div style='font-size:22px;font-weight:800;color:{_pnl_clr};"
            f"font-family:JetBrains Mono;margin:4px 0'>"
            f"{'+' if _db_today_pnl>=0 else ''}₹{_db_today_pnl:,.0f}</div>"
            f"<div style='font-size:11px;color:{_pnl_clr}'>"
            f"{len(_db_closed_today)} closed · {len(_db_open)} open</div>"
            f"</div>", unsafe_allow_html=True)

    # Refresh market data button
    _db_ref_col1, _db_ref_col2 = st.columns([3, 1])
    with _db_ref_col2:
        if st.button("🔄 Refresh Market Data", key="db_refresh_mkt",
                     use_container_width=True):
            _db_kite = get_kite_client()
            _db_new  = get_nifty_market_state(kite=_db_kite)
            st.session_state['nifty_context']      = _db_new
            st.session_state['nifty_market_state'] = _db_new['state']
            st.session_state['nifty_ctx_date']     = _db_now.strftime('%Y-%m-%d %H:%M')
            st.rerun()
    with _db_ref_col1:
        _ctx_date = st.session_state.get('nifty_ctx_date', '')
        if _ctx_date:
            st.markdown(
                f"<div style='font-size:11px;color:#94a3b8;padding:10px 0'>"
                f"Last updated: {_ctx_date}</div>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────
    # SECTION 3 — GLOBAL MARKETS
    # ─────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>🌍 Global Markets</div>",
                unsafe_allow_html=True)

    with st.spinner("Fetching global markets..."):
        _global = fetch_global_markets()

    if _global:
        _gm_cols = st.columns(4)
        _gm_order = ['S&P 500','Nasdaq','Nikkei','Hang Seng','Crude Oil','Gold','USD/INR','Dow Jones']
        for _gi, _gname in enumerate([n for n in _gm_order if n in _global]):
            _gd   = _global[_gname]
            _gc   = '#16a34a' if _gd['chg'] >= 0 else '#dc2626'
            _gi2  = '#f0fdf4' if _gd['chg'] >= 0 else '#fff5f5'
            _garr = '▲' if _gd['chg'] >= 0 else '▼'
            with _gm_cols[_gi % 4]:
                st.markdown(
                    f"<div style='background:{_gi2};border:1px solid {_gc}33;"
                    f"border-radius:8px;padding:10px 12px;margin-bottom:8px'>"
                    f"<div style='font-size:10px;font-weight:700;color:#64748b;"
                    f"letter-spacing:1px'>{_gname.upper()}</div>"
                    f"<div style='font-size:16px;font-weight:800;color:{_gc};"
                    f"font-family:JetBrains Mono;margin:3px 0'>"
                    f"{_garr} {'+' if _gd['chg']>=0 else ''}{_gd['chg']:.2f}%</div>"
                    f"<div style='font-size:10px;color:#94a3b8'>{_gd['price']:,.1f}</div>"
                    f"</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            "<div style='font-size:12px;color:#94a3b8;padding:8px 0'>"
            "Global market data unavailable — check internet connection</div>",
            unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────
    # SECTION 4 — TODAY'S STRATEGY
    # ─────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>🎯 Today's Strategy</div>",
                unsafe_allow_html=True)

    # ── NSE Holiday detection ─────────────────────────────
    # Fetched live from NSE API — no hardcoded list
    _today_str   = _db_now.strftime('%Y-%m-%d')
    _is_weekend  = _db_now.weekday() >= 5
    _is_holiday, _holiday_name = is_nse_holiday(_today_str)
    _is_trading  = not _is_weekend and not _is_holiday

    # Show NSE API status + refresh option in sidebar area
    _nse_cache_key = f"nse_holidays_{ist_now().strftime('%Y')}"
    _hol_fetched   = _nse_cache_key in st.session_state
    _hol_count     = len(st.session_state.get(_nse_cache_key, {}))
    _hol_source    = "NSE API" if _hol_count > 5 else ("pandas_market_calendars" if _hol_count > 0 else "Not fetched yet")
    if st.sidebar.button("🔄 Refresh Holidays", key="refresh_nse_holidays",
                         help="Force re-fetch holiday list from NSE API"):
        st.session_state.pop(_nse_cache_key, None)
        st.rerun()

    # Holiday / weekend banner
    if _is_holiday:
        st.markdown(
            f"<div style='background:#1e1b4b;border:2px solid #818cf8;"
            f"border-radius:12px;padding:12px 18px;margin-bottom:12px;"
            f"display:flex;align-items:center;gap:12px'>"
            f"<span style='font-size:22px'>🏖️</span>"
            f"<div>"
            f"<div style='font-size:15px;font-weight:800;color:#c7d2fe'>"
            f"NSE Holiday — {_holiday_name}</div>"
            f"<div style='font-size:11px;color:#a5b4fc;margin-top:2px'>"
            f"Indian stock market is closed today "
            f"({_db_now.strftime('%A, %d %B %Y')}). "
            f"No trading. Use this time to review charts and prepare your watchlist.</div>"
            f"</div></div>", unsafe_allow_html=True)
    elif _is_weekend:
        st.markdown(
            f"<div style='background:#1e1b4b;border:2px solid #818cf8;"
            f"border-radius:12px;padding:12px 18px;margin-bottom:12px;"
            f"display:flex;align-items:center;gap:12px'>"
            f"<span style='font-size:22px'>📅</span>"
            f"<div>"
            f"<div style='font-size:15px;font-weight:800;color:#c7d2fe'>"
            f"{'Saturday' if _db_now.weekday()==5 else 'Sunday'} — Market Closed</div>"
            f"<div style='font-size:11px;color:#a5b4fc;margin-top:2px'>"
            f"NSE opens Monday 9:15 AM. Good time to review last week and plan for next.</div>"
            f"</div></div>", unsafe_allow_html=True)

    # ── Strategy cards ────────────────────────────────────
    def build_strategy_cards():
        cards = []

        # Card builder helper
        def card(icon, title, body, color='#374151', bg='#f8fafc', border='#e2e8f0'):
            return {'icon':icon,'title':title,'body':body,'color':color,'bg':bg,'border':border}

        if not _is_trading:
            cards.append(card('🏖️','Market Closed',
                'No trading today. Review your scan history and update watchlist for next session.',
                '#6366f1','#eef2ff','#c7d2fe'))
            return cards

        # Day type card
        if _is_expiry_db:
            _exp_names = {
                'NIFTY_MONTHLY':   'Nifty MONTHLY Expiry',
                'NIFTY_WEEKLY':    'Nifty Weekly Expiry',
                'BANKNIFTY_WEEKLY':'Bank Nifty Weekly Expiry',
            }
            cards.append(card('⚠️', _exp_names.get(_exp_type,'Expiry Day'),
                'Avoid entries before 10 AM · Entry windows: 10:00–10:30 AM and 1:30–2:30 PM · '
                'Exit ALL by 2:30 PM · Skip banking stocks — options pinning risk',
                '#92400e','#fffbeb','#fde68a'))
        else:
            cards.append(card('✅','Normal Trading Day',
                'Best entry window: 9:35–11:30 AM · Second window: 1:30–2:30 PM · '
                'Exit by 3:00 PM · Square off by 3:15 PM',
                '#15803d','#f0fdf4','#bbf7d0'))

        # VIX card
        _vix_cards = {
            'CALM':    ('📉','VIX Calm — Perfect Conditions',
                        'Trending day likely. Full position size. Momentum stocks work well.',
                        '#15803d','#f0fdf4','#bbf7d0'),
            'NORMAL':  ('📊','VIX Normal — Best Conditions',
                        'Trade full size. Standard rules. All signals valid.',
                        '#15803d','#f0fdf4','#bbf7d0'),
            'ELEVATED':('📊','VIX Elevated — Still Good',
                        'Normal for India. Full position size. Trade freely.',
                        '#15803d','#f0fdf4','#bbf7d0'),
            'HIGH':    ('⚠️','VIX High — Reduce Size',
                        'Trade at 70% position size. Widen SL slightly. Only score ≥ 75 signals.',
                        '#92400e','#fffbeb','#fde68a'),
            'EXTREME': ('🔴','VIX Extreme — Trade Cautiously',
                        'Trade at 50% position size. Only STRONG BUY signals (score ≥ 80). Tight SL.',
                        '#991b1b','#fff5f5','#fecaca'),
            'CRISIS':  ('🚫','VIX Crisis — Avoid Intraday',
                        'Market in extreme fear. Avoid new intraday positions today.',
                        '#7f1d1d','#fef2f2','#fecaca'),
            'UNKNOWN': ('❓','VIX Unknown',
                        'Click Refresh Market Data to fetch current VIX level.',
                        '#64748b','#f8fafc','#e2e8f0'),
        }
        _vc = _vix_cards.get(_db_vix_lvl, _vix_cards['UNKNOWN'])
        cards.append(card(*_vc))

        # Nifty direction card
        _nifty_cards = {
            'BULL':     ('📈','Nifty Bullish',
                         'Favour long entries. Strong RS stocks outperform. Avoid short setups.',
                         '#15803d','#f0fdf4','#bbf7d0'),
            'BEAR':     ('📉','Nifty Bearish',
                         'Trade only stocks with RS > 1.0% vs Nifty — defensive/commodity/sector-specific only. Reduce size 50%.',
                         '#991b1b','#fff5f5','#fecaca'),
            'SIDEWAYS': ('↔️','Nifty Sideways',
                         'Stock-specific moves only. Wait for clear direction before entering.',
                         '#92400e','#fffbeb','#fde68a'),
            'UNKNOWN':  ('❓','Nifty Unknown',
                         'Refresh market data to get current Nifty direction.',
                         '#64748b','#f8fafc','#e2e8f0'),
        }
        _nc2 = _nifty_cards.get(_db_nstate, _nifty_cards['UNKNOWN'])
        cards.append(card(*_nc2))

        # Global cue card
        sp_chg  = _global.get('S&P 500',{}).get('chg',0) if _global else 0
        nas_chg = _global.get('Nasdaq',{}).get('chg',0) if _global else 0
        if abs(sp_chg) >= 1.0 or abs(nas_chg) >= 1.0:
            _gl_lines = []
            if sp_chg >= 1.0:
                _gl_lines.append(f"S&P 500 +{sp_chg:.1f}% — positive for IT and financials")
            elif sp_chg <= -1.0:
                _gl_lines.append(f"S&P 500 {sp_chg:.1f}% — headwind for IT stocks")
            if nas_chg >= 1.5:
                _gl_lines.append(f"Nasdaq +{nas_chg:.1f}% — TCS, INFY, HCLTECH likely to open strong")
            elif nas_chg <= -1.5:
                _gl_lines.append(f"Nasdaq {nas_chg:.1f}% — avoid IT stocks today")
            _gl_pos = sp_chg >= 0 and nas_chg >= 0
            cards.append(card(
                '🌍','Global Cues — ' + ('Positive' if _gl_pos else 'Negative'),
                ' · '.join(_gl_lines),
                '#15803d' if _gl_pos else '#991b1b',
                '#f0fdf4' if _gl_pos else '#fff5f5',
                '#bbf7d0' if _gl_pos else '#fecaca'))

        # Tools card
        if _is_expiry_db:
            _tools = 'ORB Scanner at 10:00 AM → Scanner after 10:30 AM'
        elif _db_nstate == 'BULL' and _db_vix_lvl in ('CALM','NORMAL','ELEVATED'):
            _tools = 'Early Movers at 9:15 AM → ORB at 9:20 AM → Scanner at 9:35 AM'
        else:
            _tools = 'Skip Early Movers → Scanner at 9:35 AM for confirmed signals'
        cards.append(card('📱','Tools for Today', _tools, '#1d4ed8','#eff6ff','#bfdbfe'))

        return cards

    _strat_cards = build_strategy_cards()
    # Render as 2-column grid of cards
    _sc_rows = [_strat_cards[i:i+2] for i in range(0, len(_strat_cards), 2)]
    for _sc_row in _sc_rows:
        _sc_cols = st.columns(len(_sc_row))
        for _sci, _sc in enumerate(_sc_row):
            with _sc_cols[_sci]:
                st.markdown(
                    f"<div style='background:{_sc['bg']};border:1px solid {_sc['border']};"
                    f"border-radius:10px;padding:14px 16px;margin-bottom:8px;height:100%'>"
                    f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:6px'>"
                    f"<span style='font-size:16px'>{_sc['icon']}</span>"
                    f"<span style='font-size:12px;font-weight:700;color:{_sc['color']}'>"
                    f"{_sc['title']}</span></div>"
                    f"<div style='font-size:11px;color:#374151;line-height:1.7'>"
                    f"{_sc['body']}</div>"
                    f"</div>", unsafe_allow_html=True)


        _lines = []

        # Day type
        if _is_expiry_db:
            _exp_names = {
                'NIFTY_MONTHLY':   'Nifty MONTHLY Expiry (most volatile)',
                'NIFTY_WEEKLY':    'Nifty Weekly Expiry',
                'BANKNIFTY_WEEKLY':'Bank Nifty Weekly Expiry',
            }
            _lines.append(f"⚠️ <b>{_exp_names.get(_exp_type,'Expiry Day')}</b> — special rules apply")
            _lines.append("🔴 Avoid entries before 10:00 AM — fake gap moves likely")
            _lines.append("🟢 Best windows: 10:00–10:30 AM or 1:30–2:30 PM only")
            _lines.append("🏦 Avoid banking stocks (HDFCBANK, ICICIBANK, AXISBANK) — options pinning")
            _lines.append("🚪 Exit ALL positions by 2:30 PM — last hour extremely volatile")
        else:
            _lines.append("✅ Normal trading day — standard rules apply")
            _lines.append("🟢 Best entry window: 9:35 AM – 11:30 AM")
            _lines.append("🚪 Start exiting by 3:00 PM · Square off by 3:15 PM")


    # SECTION 5 — SECTOR HEATMAP
    # ─────────────────────────────────────────────────────
    if _db_sectors:
        st.markdown("<div class='section-header'>🗺️ Sector Momentum</div>",
                    unsafe_allow_html=True)
        _sorted_sectors = sorted(_db_sectors.items(), key=lambda x: x[1], reverse=True)
        _hm_cols = st.columns(5)
        for _si, (_sname, _schg) in enumerate(_sorted_sectors[:20]):
            _sc  = ('#16a34a' if _schg >= 1.0 else
                    '#65a30d' if _schg >= 0.3 else
                    '#d97706' if _schg >= -0.3 else
                    '#ea580c' if _schg >= -1.0 else '#dc2626')
            _sbg = ('#f0fdf4' if _schg >= 1.0 else
                    '#f7fee7' if _schg >= 0.3 else
                    '#fffbeb' if _schg >= -0.3 else
                    '#fff7ed' if _schg >= -1.0 else '#fff5f5')
            with _hm_cols[_si % 5]:
                st.markdown(
                    f"<div style='background:{_sbg};border:1px solid {_sc}33;"
                    f"border-radius:8px;padding:8px 10px;margin-bottom:6px;text-align:center'>"
                    f"<div style='font-size:10px;font-weight:700;color:#64748b'>{_sname}</div>"
                    f"<div style='font-size:14px;font-weight:800;color:{_sc};"
                    f"font-family:JetBrains Mono'>"
                    f"{'+' if _schg>=0 else ''}{_schg:.1f}%</div>"
                    f"</div>", unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:11px;color:#94a3b8;margin-top:-4px'>"
            "Based on last scan — run Scanner to update</div>",
            unsafe_allow_html=True)
    else:
        st.markdown(
            "<div style='background:#f8fafc;border:1px solid #e2e8f0;"
            "border-radius:10px;padding:14px 18px;margin-top:8px'>"
            "<div style='font-size:13px;font-weight:600;color:#1a2035;margin-bottom:4px'>"
            "🗺️ Sector Heatmap</div>"
            "<div style='font-size:12px;color:#94a3b8'>"
            "Run a scan first to see sector momentum. "
            "Go to 📊 Scanner → click Scan Now.</div>"
            "</div>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────
    # SECTION 6 — OPEN POSITIONS QUICK VIEW
    # ─────────────────────────────────────────────────────
    if _db_open:
        st.markdown("<div class='section-header'>💼 Open Positions</div>",
                    unsafe_allow_html=True)
        _op_cols = st.columns(min(len(_db_open), 3))
        for _opi, _op in enumerate(_db_open[:3]):
            _op_sym   = _op.get('symbol', '')
            _op_entry = _f(_op.get('entry', 0))
            _op_qty   = int(_f(_op.get('qty', 0)))
            _op_sl    = _f(_op.get('stop_loss', 0))
            # Get cached live price
            _op_live  = st.session_state.get('pf_live_prices', {}).get(_op_sym, _op_entry)
            _op_pl    = (_op_live - _op_entry) * _op_qty
            _op_pct   = (_op_pl / (_op_entry * _op_qty)) * 100 if _op_entry > 0 else 0
            _op_clr   = '#16a34a' if _op_pl >= 0 else '#dc2626'
            _op_sl_hit = _op_sl > 0 and _op_live <= _op_sl
            with _op_cols[_opi]:
                st.markdown(
                    f"<div style='background:{'#fef2f2' if _op_sl_hit else '#ffffff'};"
                    f"border:1.5px solid {'#dc2626' if _op_sl_hit else '#e8ecf3'};"
                    f"border-radius:10px;padding:12px 14px'>"
                    f"<div style='font-size:14px;font-weight:800;color:#1a2035'>{_op_sym}</div>"
                    f"<div style='font-size:11px;color:#64748b;margin-top:2px'>"
                    f"Entry ₹{_op_entry:,.2f} · {_op_qty} shares</div>"
                    f"<div style='font-size:18px;font-weight:800;color:{_op_clr};"
                    f"font-family:JetBrains Mono;margin:6px 0'>"
                    f"{'+' if _op_pl>=0 else ''}₹{_op_pl:,.0f}</div>"
                    f"<div style='font-size:11px;color:{_op_clr}'>"
                    f"{'+' if _op_pct>=0 else ''}{_op_pct:.2f}%</div>"
                    + ("<div style=\"font-size:11px;font-weight:700;color:#dc2626;margin-top:4px\">🛑 SL HIT — Exit Now</div>" if _op_sl_hit else "")
                    + "</div>", unsafe_allow_html=True)
        if len(_db_open) > 3:
            st.markdown(
                f"<div style='font-size:12px;color:#94a3b8;margin-top:4px'>"
                f"+ {len(_db_open)-3} more positions — see 💼 Portfolio</div>",
                unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────
    # SECTION 7 — QUICK NAVIGATION CARDS
    # ─────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>⚡ Quick Actions</div>",
                unsafe_allow_html=True)

    _qa_items = [
        ("📊", "Scanner",       "Scan 498 stocks\nShortlist · Deep analysis",    "📊  Scanner"),
        ("🚀", "Early Movers",  "Gap-up stocks\nFirst 15 minutes",               "🚀  Early Movers"),
        ("🔓", "ORB Scanner",   "Opening range\n9:20 AM – 10:30 AM",             "🔓  ORB Scanner"),
        ("💼", "Portfolio",     "Open positions\nP&L · Square off",              "💼  Portfolio"),
    ]
    _qa_cols = st.columns(4, gap="small")
    for _qi, (_qicon, _qlbl, _qdesc, _qpage) in enumerate(_qa_items):
        with _qa_cols[_qi]:
            # Card HTML — fixed height so all 4 are identical size
            _qa_active = (active_page == _qpage)
            _qa_bg     = "#1a2035"  if _qa_active else "#ffffff"
            _qa_clr    = "#f59e0b"  if _qa_active else "#1a2035"
            _qa_sub    = "rgba(255,255,255,0.6)" if _qa_active else "#64748b"
            _qa_bdr    = "#f59e0b"  if _qa_active else "#e8ecf3"
            st.markdown(
                f"<div style='background:{_qa_bg};border:1.5px solid {_qa_bdr};"
                f"border-radius:12px;padding:16px 12px;text-align:center;"
                f"min-height:100px;display:flex;flex-direction:column;"
                f"align-items:center;justify-content:center;gap:6px'>"
                f"<div style='font-size:26px'>{_qicon}</div>"
                f"<div style='font-size:13px;font-weight:700;color:{_qa_clr}'>{_qlbl}</div>"
                f"<div style='font-size:10px;color:{_qa_sub};line-height:1.5;white-space:pre-line'>{_qdesc}</div>"
                f"</div>", unsafe_allow_html=True)
            # Invisible button overlay — takes full column width
            if st.button(f"Go to {_qlbl}", key=f"db_nav_{_qi}",
                         use_container_width=True):
                st.session_state['active_page'] = _qpage
                st.rerun()




# ─────────────────────────────────────────────
#  SCANNER PAGE
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
#  PARALLEL SCAN WORKER
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────

if _show_scanner:
    # ── GLOBAL ALERT BANNER (top of page) ────────────────
    _init_alert_log()
    _page_alerts = st.session_state.get(ALERT_LOG_KEY, [])
    if _page_alerts:
        _recent = _page_alerts[:5]
        _alert_colors = {
            'BUY':         ('#f0fdf4','#15803d'),
            'STRONG_BUY':  ('#dcfce7','#14532d'),
            'VOL_SURGE':   ('#fffbeb','#92400e'),
            'VWAP_BREAK':  ('#fff5f5','#dc2626'),
            'RSI_OB':      ('#fff5f5','#dc2626'),
            'STOP_LOSS':   ('#fef2f2','#7f1d1d'),
            'TARGET_T1':   ('#f0fdf4','#15803d'),
            'TARGET_T2':   ('#f0fdf4','#15803d'),
            'TARGET_T3':   ('#f0fdf4','#15803d'),
            'TARGET_T4':   ('#f0fdf4','#15803d'),
            'TIME_WARN':   ('#fffbeb','#92400e'),
            'ORB_VOL':     ('#f5f3ff','#7c3aed'),
            'ORB_GAP':     ('#f0fdf4','#15803d'),
            'ORB_VWAP':    ('#fffbeb','#d97706'),
            'ORB_HIGH':    ('#f0f9ff','#0369a1'),
            'ORB_MOMENTUM':('#fff7ed','#c2410c'),
        }
        _icons = {
            'BUY':'🔔','STRONG_BUY':'🚨','VOL_SURGE':'⚡',
            'VWAP_BREAK':'⚠️','RSI_OB':'🔴','STOP_LOSS':'🛑',
            'TARGET_T1':'🎯','TARGET_T2':'🎯','TARGET_T3':'🎯','TARGET_T4':'🎯',
            'TIME_WARN':'🕒',
            'ORB_VOL':'🚀','ORB_GAP':'📈','ORB_VWAP':'💛',
            'ORB_HIGH':'🔓','ORB_MOMENTUM':'⚡',
        }
        with st.expander(f"🔔 **{len(_page_alerts)} Active Alerts** — click to expand", expanded=len(_page_alerts) <= 3):
            for _al in _recent:
                _bg, _tc = _alert_colors.get(_al['type'], ('#f8fafc','#1a2035'))
                _ic = _icons.get(_al['type'], '📣')
                st.markdown(f"""
                <div style='background:{_bg};border:1.5px solid {_tc}33;border-radius:12px;
                            padding:12px 18px;margin-bottom:8px;
                            display:flex;align-items:center;justify-content:space-between;gap:12px'>
                    <div style='display:flex;align-items:center;gap:12px;flex:1'>
                        <span style='font-size:22px'>{_ic}</span>
                        <span style='font-size:13px;font-weight:700;color:{_tc}'>{_al["message"]}</span>
                    </div>
                    <span style='font-size:11px;color:#94a3b8;flex-shrink:0'>{_al["time"]}</span>
                </div>""", unsafe_allow_html=True)
            if len(_page_alerts) > 5:
                st.caption(f"+ {len(_page_alerts)-5} more alerts in sidebar log")

    # ── STATUS BAR ────────────────────────────────────────
    _missing = st.session_state.get('kite_missing_tokens', set())
    _missing_str = f" &nbsp;·&nbsp; <span style='color:#f59e0b'>⚠️ {len(_missing)} via yfinance fallback</span>" if _missing else ""

    _mkt_str  = "Market OPEN 🟢" if market_open() else "Market CLOSED 🔴"
    _scan_dur = st.session_state.get('scan_duration', 0)
    _dur_str2 = f" &nbsp;·&nbsp; ⚡ scanned in {_scan_dur}s" if _scan_dur else ""

    _interval_warn = ""
    if not market_open() and interval == '1minute':
        _interval_warn = " &nbsp;·&nbsp; <span style='color:#ef4444'>⚠️ 1min unavailable after hours — switch to 15min or 1hr</span>"

    # ── Last scanned display ──────────────────────────────
    _secs_since  = int(time.time() - st.session_state.get('last_auto_refresh', time.time()))
    _ago_str     = (f"{_secs_since}s ago" if _secs_since < 60
                    else f"{_secs_since//60}m {_secs_since%60}s ago")
    _refresh_str = (f" &nbsp;·&nbsp; 🕐 Last scanned <b>{_ago_str}</b>"
                    if 'scan_results' in st.session_state else "")

    # ── Nifty + VIX live bar (refreshes independently of scan) ──
    _sb1, _sb2, _sb3 = st.columns([4, 1, 1])
    with _sb1:
        st.markdown(f"""
        <div style='background:#f1f5f9;border:1px solid #e2e8f0;border-radius:10px;
                    padding:8px 18px;margin-bottom:8px;font-size:12px;color:#475569'>
            {_mkt_str} &nbsp;·&nbsp; Last scan: {st.session_state.get('scan_time','—')}{_dur_str2}{_missing_str}{_interval_warn}{_refresh_str}
        </div>""", unsafe_allow_html=True)
    with _sb2:
        _refresh_mkt = st.button("🔄 Market Data", key="refresh_market_ctx",
                                  use_container_width=True,
                                  help="Re-fetch Nifty + VIX without running full scan")
    with _sb3:
        _manual_rescan = st.button("🔁 Rescan", key="manual_rescan",
                                    use_container_width=True,
                                    type="primary",
                                    help="Manually re-run the full scan with fresh data")
        if _manual_rescan:
            _DATA_CACHE.clear()
            st.session_state.pop('scan_results', None)
            st.session_state.pop('scan_raw', None)
            st.session_state.pop('scan_key', None)
            reset_refresh_timer()
            st.rerun()

    # Refresh market context on button click
    if _refresh_mkt:
        _kite_ctx = get_kite_client()
        _new_ctx  = get_nifty_market_state(kite=_kite_ctx)
        st.session_state['nifty_context']      = _new_ctx
        st.session_state['nifty_market_state'] = _new_ctx['state']
        st.session_state['nifty_ctx_date']     = datetime.now().strftime('%Y-%m-%d %H:%M')
        st.rerun()

    # Show VIX/Nifty banner only when context exists AND is from today
    _mkt_ctx_cached = st.session_state.get('nifty_context', {})
    _ctx_date       = st.session_state.get('nifty_ctx_date', '')
    _ctx_today      = _ctx_date.startswith(datetime.now().strftime('%Y-%m-%d')) if _ctx_date else False

    if _mkt_ctx_cached and _ctx_today:
        _nifty_state_bar = _mkt_ctx_cached.get('state', 'UNKNOWN')
        _vix_bar         = _mkt_ctx_cached.get('vix')
        _vix_level_bar   = _mkt_ctx_cached.get('vix_level', 'UNKNOWN')
        _nifty_chg_bar   = _mkt_ctx_cached.get('nifty_chg', 0)

        _nc_bar = {'BULL':'#16a34a','SIDEWAYS':'#d97706','BEAR':'#dc2626','UNKNOWN':'#64748b'}.get(_nifty_state_bar,'#64748b')
        _vc_bar = {
            'CALM':    '#16a34a',
            'NORMAL':  '#16a34a',
            'ELEVATED':'#d97706',
            'HIGH':    '#ea580c',
            'EXTREME': '#dc2626',
            'CRISIS':  '#7f1d1d',
            'UNKNOWN': '#64748b',
        }.get(_vix_level_bar,'#64748b')
        _vix_adv = {
            'CALM':     '✅ VIX < 13 — very calm, ideal trading day',
            'NORMAL':   '✅ VIX 13–16 — best conditions, trade freely',
            'ELEVATED': '✅ VIX 16–20 — normal for India, trade normally',
            'HIGH':     '⚠️ VIX 20–25 — reduce position size 30%',
            'EXTREME':  '⚠️ VIX 25–30 — only strongest signals (score ≥ 75)',
            'CRISIS':   '🚫 VIX > 30 — avoid intraday (COVID/war level)',
            'UNKNOWN':  '',
        }.get(_vix_level_bar, '')

        _bar_cols = st.columns([1, 2, 1])
        with _bar_cols[0]:
            _ni = {'BULL':'📈','SIDEWAYS':'↔️','BEAR':'📉','UNKNOWN':'❓'}.get(_nifty_state_bar,'❓')

            # Get swing state from session (set by last scan) or fetch fresh
            _sw_swing = st.session_state.get('nifty_swing_weekly', {})
            _sw_state = _sw_swing.get('state', 'UNKNOWN')
            _sw_sma20 = _sw_swing.get('sma20', 0)
            _sw_close = _sw_swing.get('close', 0)
            _sw_clr   = {'BULLISH':'#15803d','CAUTION':'#d97706',
                         'BEARISH':'#dc2626','UNKNOWN':'#64748b'}.get(_sw_state,'#64748b')
            _sw_ico   = {'BULLISH':'✅','CAUTION':'⚠️',
                         'BEARISH':'🔴','UNKNOWN':'❓'}.get(_sw_state,'❓')
            # Daily swing state (from SMA Weekly scan)
            _sd_swing = st.session_state.get('nifty_swing_daily', {})
            _sd_state = _sd_swing.get('state', 'UNKNOWN')
            _sd_sma20 = _sd_swing.get('sma20', 0)
            _sd_clr   = {'BULLISH':'#15803d','CAUTION':'#d97706',
                         'BEARISH':'#dc2626','UNKNOWN':'#64748b'}.get(_sd_state,'#64748b')
            _sd_ico   = {'BULLISH':'✅','CAUTION':'⚠️',
                         'BEARISH':'🔴','UNKNOWN':'❓'}.get(_sd_state,'❓')

            st.markdown(
                f"<div style='background:{_nc_bar}22;border:1px solid {_nc_bar}44;"
                f"border-radius:8px;padding:10px 14px;margin-bottom:8px'>"
                f"<div style='font-size:11px;font-weight:700;color:{_nc_bar}'>"
                f"{_ni} Nifty Today: {_nifty_state_bar}</div>"
                f"<div style='font-size:18px;font-weight:800;color:{_nc_bar};"
                f"font-family:JetBrains Mono'>"
                f"{'+' if _nifty_chg_bar>=0 else ''}{_nifty_chg_bar:.2f}%</div>"
                + (f"<div style='margin-top:6px;padding-top:6px;"
                   f"border-top:1px solid {_nc_bar}33;font-size:10px;"
                   f"font-weight:700;color:{_sw_clr}'>"
                   f"{_sw_ico} Monthly Swing: {_sw_state}"
                   + (f" · Weekly SMA20 ₹{_sw_sma20:,.0f}"
                      if _sw_sma20 > 0 else "")
                   + "</div>" if _sw_state != 'UNKNOWN' else "")
                + (f"<div style='margin-top:4px;font-size:10px;"
                   f"font-weight:700;color:{_sd_clr}'>"
                   f"{_sd_ico} SMA Weekly: {_sd_state}"
                   + (f" · Daily SMA20 ₹{_sd_sma20:,.0f}"
                      if _sd_sma20 > 0 else "")
                   + "</div>" if _sd_state != 'UNKNOWN' else "")
                + "</div>", unsafe_allow_html=True)

        with _bar_cols[1]:
            _vix_str2 = f"{_vix_bar:.2f}" if _vix_bar else "—"
            st.markdown(
                f"<div style='background:{_vc_bar}22;border:1px solid {_vc_bar}44;"
                f"border-radius:8px;padding:10px 14px;margin-bottom:8px'>"
                f"<div style='font-size:11px;font-weight:700;color:{_vc_bar}'>"
                f"📊 India VIX — {_vix_level_bar}</div>"
                f"<div style='display:flex;align-items:baseline;gap:10px;margin-top:2px'>"
                f"<div style='font-size:22px;font-weight:800;color:{_vc_bar};"
                f"font-family:JetBrains Mono'>{_vix_str2}</div>"
                f"<div style='font-size:11px;color:{_vc_bar}'>{_vix_adv}</div>"
                f"</div></div>", unsafe_allow_html=True)

        with _bar_cols[2]:
            st.markdown(
                f"<div style='background:#f8fafc;border:1px solid #e2e8f0;"
                f"border-radius:8px;padding:10px 14px;margin-bottom:8px'>"
                f"<div style='font-size:10px;color:#64748b'>🕐 {_ctx_date}</div>"
                f"</div>", unsafe_allow_html=True)

        # Warning banner — only for CRISIS (VIX > 30) or EXTREME on BEAR day
        _show_vix_warn = (
            _vix_level_bar == 'CRISIS' or
            (_vix_level_bar == 'EXTREME' and _nifty_state_bar == 'BEAR')
        )
        if _show_vix_warn:
            _warn_msg = (
                "🚫 India VIX > 30 — True market crisis. Avoid intraday trading today."
                if _vix_level_bar == 'CRISIS'
                else "⚠️ VIX 25–30 + Nifty BEAR — High risk. Only trade score ≥ 75 with strong RS."
            )
            _ext_col1, _ext_col2 = st.columns([5, 1])
            with _ext_col1:
                st.warning(_warn_msg)
            with _ext_col2:
                if st.button("✕ Dismiss", key="dismiss_vix_warning"):
                    st.session_state['nifty_ctx_date'] = ''
                    st.rerun()

    elif _mkt_ctx_cached and not _ctx_today:
        # Context exists but is from a previous day — show stale notice
        st.markdown(
            "<div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;"
            "padding:6px 14px;margin-bottom:8px;font-size:11px;color:#94a3b8'>"
            "📊 Market context from previous session — click <b>🔄 Market Data</b> to refresh"
            "</div>", unsafe_allow_html=True)

    # ── Indicator Warmup Banner ───────────────────────────
    # Show when market has been open less than 20 minutes
    if market_open():
        try:
            _now_ist   = ist_now()
            _mkt_start = _now_ist.replace(hour=9, minute=15, second=0, microsecond=0)
            _mins_since_open = int((_now_ist - _mkt_start.astimezone(_now_ist.tzinfo)).total_seconds() / 60)
        except Exception:
            _mins_since_open = 999

        if _mins_since_open < 7:
            st.markdown(f"""
            <div style='background:#450a0a;border:2px solid #dc2626;border-radius:12px;
                        padding:14px 20px;margin-bottom:12px'>
                <div style='font-size:15px;font-weight:800;color:#fca5a5'>
                    🚫 TOO EARLY — Indicators Not Ready ({_mins_since_open} min since open)
                </div>
                <div style='font-size:13px;color:#fca5a5;margin-top:6px;line-height:1.6'>
                    Market opened {_mins_since_open} minute(s) ago. All technical indicators
                    (RSI, MACD, EMA, Supertrend, ADX) need at least <b>7+ candles</b> to calculate.
                    Scores and verdicts shown now are <b>completely unreliable</b>.<br><br>
                    ✅ <b>Only trust:</b> Volume Ratio and VWAP position right now.<br>
                    ⏰ <b>Wait until 9:22 AM</b> before acting on any scanner results.
                </div>
            </div>""", unsafe_allow_html=True)

        elif _mins_since_open < 20:
            _pct = int((_mins_since_open / 20) * 100)
            st.markdown(f"""
            <div style='background:#451a03;border:1.5px solid #d97706;border-radius:12px;
                        padding:12px 18px;margin-bottom:12px'>
                <div style='display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px'>
                    <div>
                        <div style='font-size:14px;font-weight:800;color:#fbbf24'>
                            ⏳ Indicators Warming Up — {_mins_since_open} min since open
                        </div>
                        <div style='font-size:12px;color:#fde68a;margin-top:4px'>
                            RSI · MACD · EMA · Supertrend are <b>partially reliable</b>.
                            Wait until <b>9:35 AM</b> for full confidence.
                            Right now: trust <b>Volume + VWAP</b> only.
                        </div>
                    </div>
                    <div style='text-align:center'>
                        <div style='font-size:10px;color:#fbbf24;font-weight:700;letter-spacing:1px'>INDICATOR READINESS</div>
                        <div style='font-size:22px;font-weight:800;color:#fbbf24;font-family:JetBrains Mono'>{_pct}%</div>
                    </div>
                </div>
                <div style='background:rgba(0,0,0,0.3);border-radius:4px;height:6px;margin-top:8px'>
                    <div style='background:#f59e0b;height:6px;border-radius:4px;width:{_pct}%;transition:width 0.5s'></div>
                </div>
            </div>""", unsafe_allow_html=True)

        elif _mins_since_open < 50:
            st.markdown(f"""
            <div style='background:#064e3b;border:1px solid #065f46;border-radius:10px;
                        padding:10px 18px;margin-bottom:12px;
                        display:flex;align-items:center;justify-content:space-between'>
                <div style='font-size:13px;font-weight:700;color:#34d399'>
                    ✅ Indicators Ready — {_mins_since_open} min since open
                </div>
                <div style='font-size:11px;color:#6ee7b7'>
                    All signals reliable · Best entry window active
                </div>
            </div>""", unsafe_allow_html=True)
    scan_label    = st.session_state.get('scan_time', '')
    scan_n        = st.session_state.get('scan_total', len(POPULAR_STOCKS))
    scan_duration = st.session_state.get('scan_duration', 0)
    _dur_str      = f" · ⚡ {scan_duration}s" if scan_duration else ""
    _mkt_status   = f"<span class='topbar-time'>● MARKET OPEN</span>" if market_open() else "<span class='topbar-time-closed'>● MARKET CLOSED</span>"
    _sc_kite_on   = get_kite_client() is not None
    _sc_src_lbl   = 'Kite — Live' if _sc_kite_on else 'yfinance — Delayed'
    _sc_src_clr   = '#16a34a' if _sc_kite_on else '#d97706'
    _sc_src_bg    = '#dcfce7' if _sc_kite_on else '#fef3c7'
    st.markdown(f"""
    <div class='topbar'>
        <div>
            <div class='topbar-title'>
                <svg xmlns='http://www.w3.org/2000/svg' width='20' height='20' fill='none' viewBox='0 0 24 24' stroke='white' stroke-width='2'>
                    <polyline points='13 2 13 9 19 9'/><polyline points='11 22 11 15 5 15'/>
                    <path d='M3 3h7v7H3z'/><path d='M14 14h7v7h-7z'/>
                </svg>
                &nbsp; Intraday Scanner
                <span class='timeframe-pill'>{interval_label.split("(")[0].strip()}</span>
            </div>
            <div class='topbar-subtitle'>{scan_n} stocks · {scan_label or ist_now().strftime('%d %b %Y, %H:%M IST')}{_dur_str}</div>
        </div>
        <div class='topbar-right'>
            <span style='background:{_sc_src_bg};color:{_sc_src_clr};font-size:11px;
                         font-weight:700;border-radius:6px;padding:3px 10px;
                         border:1px solid {_sc_src_clr}44'>
                {'🟢' if _sc_kite_on else '🟡'} {_sc_src_lbl}
            </span>
            {_mkt_status}
        </div>
    </div>
    """, unsafe_allow_html=True)

    VERDICT_RANK = {"❌ AVOID":0,"⚠️ NEUTRAL":1,"⭐ WATCH":2,"⭐⭐ BUY":3,"⭐⭐⭐ STRONG BUY":4}
    min_rank     = VERDICT_RANK.get(min_verdict, 2)
    cache_key    = f"{interval}_{period}_{capital}_{risk_pct}_{','.join(sorted(selected_stocks))}"

    # ── Auto-refresh logic ────────────────────────────────
    if run_btn or ('scan_results' not in st.session_state) or (st.session_state.get('scan_key') != cache_key):
        if not selected_stocks:
            st.warning("⚠️ No stocks selected."); st.stop()

        total       = len(selected_stocks)
        raw_results = []
        kite        = get_kite_client()
        port        = load_portfolio()
        scan_start  = time.time()
        _DATA_CACHE.clear()
        for _k in list(st.session_state.keys()):
            if _k.startswith('chart_fig_') or _k.startswith('chart_interval_'):
                del st.session_state[_k]

        # ── Priority-first ordering ───────────────────────
        # Scan high-priority stocks first so results appear
        # in ~20 seconds instead of waiting 90s for all 498.
        # Priority stocks = Nifty 50 + high beta + high volume
        _priority_set  = set(PRIORITY_STOCKS)
        _priority_first = [s for s in selected_stocks if s in _priority_set]
        _rest           = [s for s in selected_stocks if s not in _priority_set]
        _ordered_stocks = _priority_first + _rest
        total           = len(_ordered_stocks)

        # ── Priority 1: Fetch Nifty + VIX market state FIRST ─
        _prog = st.progress(0, text="🔍 Checking Nifty + VIX market state...")
        _mkt_ctx = get_nifty_market_state(kite=kite)
        _nifty_state = _mkt_ctx['state']
        st.session_state['nifty_market_state'] = _nifty_state
        st.session_state['nifty_context']      = _mkt_ctx
        st.session_state['nifty_ctx_date']     = datetime.now().strftime('%Y-%m-%d %H:%M')

        _nifty_colors = {'BULL':'#16a34a','SIDEWAYS':'#d97706','BEAR':'#dc2626','UNKNOWN':'#64748b'}
        _nifty_icons  = {'BULL':'📈','SIDEWAYS':'↔️','BEAR':'📉','UNKNOWN':'❓'}
        _vix_val      = _mkt_ctx.get('vix')
        _vix_level    = _mkt_ctx.get('vix_level','UNKNOWN')
        _vix_colors   = {
            'CALM':    '#16a34a',
            'NORMAL':  '#16a34a',
            'ELEVATED':'#d97706',
            'HIGH':    '#ea580c',
            'EXTREME': '#dc2626',
            'CRISIS':  '#7f1d1d',
            'UNKNOWN': '#64748b',
        }
        _vix_advice   = {
            'CALM':     'VIX < 13 — Very calm, ideal conditions ✅',
            'NORMAL':   'VIX 13–16 — Best conditions, trade freely ✅',
            'ELEVATED': 'VIX 16–20 — Normal for India, trade freely ✅',
            'HIGH':     'VIX 20–25 — Reduce position size 30% ⚠️',
            'EXTREME':  'VIX 25–30 — Only strongest signals (score ≥ 75) ⚠️',
            'CRISIS':   'VIX > 30 — Avoid intraday (COVID/war level) 🚫',
            'UNKNOWN':  'VIX unavailable',
        }
        _nc = _nifty_colors.get(_nifty_state, '#64748b')
        _vc = _vix_colors.get(_vix_level, '#64748b')
        _vix_str = f"₹{_vix_val:.1f}" if _vix_val else "—"

        st.markdown(
            f"<div style='display:flex;gap:10px;margin-bottom:10px;flex-wrap:wrap'>"
            f"<div style='flex:2;background:{_nc}22;border:1px solid {_nc}44;"
            f"border-radius:8px;padding:10px 14px'>"
            f"<div style='font-size:11px;font-weight:700;color:{_nc};letter-spacing:1px'>NIFTY 50</div>"
            f"<div style='font-size:14px;font-weight:800;color:{_nc};margin-top:2px'>"
            f"{_nifty_icons.get(_nifty_state,'❓')} {_nifty_state} &nbsp;·&nbsp; "
            f"{'+' if _mkt_ctx['nifty_chg']>=0 else ''}{_mkt_ctx['nifty_chg']:.2f}% &nbsp;·&nbsp; "
            f"₹{_mkt_ctx['nifty_last']:,.0f}</div></div>"
            f"<div style='flex:3;background:{_vc}22;border:1px solid {_vc}44;"
            f"border-radius:8px;padding:10px 14px'>"
            f"<div style='font-size:11px;font-weight:700;color:{_vc};letter-spacing:1px'>"
            f"INDIA VIX — {_vix_level}</div>"
            f"<div style='font-size:13px;font-weight:700;color:{_vc};margin-top:2px'>"
            f"{_vix_str} &nbsp;·&nbsp; {_vix_advice.get(_vix_level,'')}</div></div>"
            f"</div>",
            unsafe_allow_html=True)

        if _vix_level in ('CRISIS', 'EXTREME'):
            st.session_state['vix_extreme_warned'] = True

        # ── Progress UI ───────────────────────────────────
        _prog    = st.progress(0, text="Starting scan...")
        _stat    = st.empty()
        _sym_ph  = st.empty()

        _priority_done   = False
        _live_ph         = st.empty()   # live results — updates every stock

        for idx, symbol in enumerate(_ordered_stocks):
            pct        = int(((idx + 1) / total) * 100)
            elapsed    = int(time.time() - scan_start)
            eta        = int((elapsed / (idx + 1)) * (total - idx - 1)) if idx > 0 else 0
            eta_str    = f"{eta//60}m {eta%60}s" if eta >= 60 else f"{eta}s"
            sym_clean  = symbol.replace('.NS', '')

            # ── Live results panel — updates after every stock ──
            # Show BUY+ signals immediately as they are found.
            # This replaces the old "one-time flash after priority batch" approach.
            # Now you see results streaming in real time throughout the scan.
            _live_buys = [r for r in raw_results
                          if r.get('_verdict','') in ('⭐⭐⭐ STRONG BUY','⭐⭐ BUY')]
            _live_buys.sort(key=lambda x: x.get('_pick_score',0), reverse=True)

            if _live_buys:
                _is_priority_phase = idx < len(_priority_first)
                _phase_label = (
                    f"⚡ Priority stocks ({idx}/{len(_priority_first)}) · {len(_live_buys)} BUY signals"
                    if _is_priority_phase else
                    f"🔍 Full scan ({idx}/{total}) · {len(_live_buys)} BUY signals"
                )
                _live_html = (
                    f"<div style='background:#0f172a;border:1.5px solid #16a34a44;"
                    f"border-radius:12px;padding:12px 16px;margin-bottom:8px'>"
                    f"<div style='font-size:11px;font-weight:700;color:#34d399;margin-bottom:8px'>"
                    f"{_phase_label}</div>"
                    f"<div style='display:flex;gap:6px;flex-wrap:wrap'>"
                )
                for _lb in _live_buys[:8]:
                    _lb_sym   = _lb['symbol'].replace('.NS','')
                    _lb_chg   = _lb.get('change_pct', 0)
                    _lb_score = _lb.get('_pick_score', 0)
                    _lb_verd  = _lb.get('_verdict','')
                    _lb_vol   = _lb.get('vol_ratio', 0)
                    _lb_cc    = "#34d399" if _lb_chg >= 0 else "#f87171"
                    _lb_vbg   = "#dcfce722" if '⭐⭐⭐' in _lb_verd else "#dbeafe22"
                    _lb_vbc   = "#34d399" if '⭐⭐⭐' in _lb_verd else "#93c5fd"
                    # Signal age badge
                    _lb_age   = _lb.get('sig_age_candles', 0)
                    _lb_fresh = "🟢" if _lb_age <= 2 else ("🟡" if _lb_age <= 5 else "🔴")
                    _live_html += (
                        f"<div style='background:{_lb_vbg};border:1px solid {_lb_vbc}44;"
                        f"border-radius:8px;padding:6px 12px;min-width:110px'>"
                        f"<div style='font-size:13px;font-weight:800;color:#f8fafc'>{_lb_sym}</div>"
                        f"<div style='font-size:10px;margin-top:2px'>"
                        f"<span style='color:{_lb_cc}'>{'+' if _lb_chg>=0 else ''}{_lb_chg:.1f}%</span>"
                        f" &nbsp;·&nbsp; <span style='color:#a78bfa'>Score {_lb_score}</span>"
                        f"</div>"
                        f"<div style='font-size:10px;margin-top:1px;color:#94a3b8'>"
                        f"Vol {_lb_vol:.1f}× &nbsp;·&nbsp; {_lb_fresh} {_lb_age}c ago</div>"
                        f"</div>"
                    )
                _live_html += "</div></div>"
                _live_ph.markdown(_live_html, unsafe_allow_html=True)

            # Mark when priority phase ends
            if not _priority_done and idx >= len(_priority_first):
                _priority_done = True

            _prog.progress(pct, text=f"Scanning {idx+1}/{total} · {sym_clean} · {pct}%")
            _sym_ph.markdown(
                f"<div style='display:flex;justify-content:space-between;align-items:center;"
                f"background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;"
                f"padding:8px 16px;font-size:12px;color:#475569;margin-bottom:4px'>"
                f"<span>🔍 <b>{sym_clean}</b></span>"
                f"<span>✅ {len(raw_results)} found &nbsp;·&nbsp; ⏱ {elapsed}s &nbsp;·&nbsp; ETA {eta_str}</span>"
                f"</div>", unsafe_allow_html=True)

            # ── Fetch data ────────────────────────────────
            try:
                df, src = fetch_intraday(symbol, interval, period, kite=kite)
                if df is None:
                    continue
            except Exception:
                continue

            # ── Indicators + signals ──────────────────────
            try:
                df = calculate_intraday_indicators(df)
                df = generate_intraday_signals(df)
            except Exception:
                continue

            # ── Build result dict ─────────────────────────
            try:
                latest    = df.iloc[-1]
                prev      = df.iloc[-2]
                sig_rows  = df[df['Signal'] != 0]
                last_sig  = sig_rows.iloc[-1] if len(sig_rows) > 0 else None
                last_sig_val = int(last_sig['Signal']) if last_sig is not None else 0
                live_bull = int(latest.get('Bull_Score', 0)) if not pd.isna(latest.get('Bull_Score', 0)) else 0
                live_bear = int(latest.get('Bear_Score', 0)) if not pd.isna(latest.get('Bear_Score', 0)) else 0
                live_conf = min(int((live_bull / 100) * 100), 100)
                price     = float(latest['Close']) if not pd.isna(latest['Close']) else 0.0

                # ── Signal age — candles since BUY signal fired ──
                # 0-2 candles = fresh signal → enter now
                # 3-5 candles = moderate → check if price extended
                # 6+ candles  = stale → likely chasing
                _sig_age_candles = 0
                _sig_price_at_fire = price
                _sig_move_since    = 0.0
                if last_sig is not None and last_sig_val == 1:
                    try:
                        _sig_idx       = df.index.get_loc(last_sig.name)
                        _curr_idx      = len(df) - 1
                        _sig_age_candles = _curr_idx - _sig_idx
                        _sig_price_at_fire = float(last_sig['Close'])
                        if _sig_price_at_fire > 0:
                            _sig_move_since = round(
                                (price - _sig_price_at_fire) / _sig_price_at_fire * 100, 2)
                    except Exception:
                        pass

                # ── Candle warmup status ──────────────────
                _warmup, _n_today, _mins, _pct_ready = candle_warmup_status(df, interval)
                trade_plan = get_intraday_trade_plan(df, capital, risk_pct)
                liquidity  = compute_liquidity(df, price or 1, capital)

                r = {
                    'symbol':      symbol,
                    'data_src':    src,
                    'df':          df,
                    'latest':      latest,
                    'prev':        prev,
                    'last_signal': last_sig,
                    'signal_val':  last_sig_val,
                    'live_bull':   live_bull,
                    'live_bear':   live_bear,
                    'live_conf':   live_conf,
                    'sig_age_candles':   _sig_age_candles,
                    'sig_price_at_fire': _sig_price_at_fire,
                    'sig_move_since':    _sig_move_since,
                    'trade_plan':  trade_plan,
                    'liquidity':   liquidity,
                    'warmup':      _warmup,
                    'n_today':     _n_today,
                    'mins_open':   _mins,
                    'pct_ready':   _pct_ready,
                    'interval':    interval,
                    'price':       price,
                    'change_pct':  float(((latest['Close']/prev['Close'])-1)*100)
                                   if not pd.isna(prev['Close']) and prev['Close'] != 0 else 0.0,
                    'rsi':         float(latest['RSI'])  if not pd.isna(latest['RSI'])  else 50.0,
                    'adx':         float(latest['ADX'])  if not pd.isna(latest['ADX'])  else 0.0,
                    'macd':        float(latest['MACD']) if not pd.isna(latest['MACD']) else 0.0,
                    'supertrend':  int(latest['Supertrend_Direction'])
                                   if not pd.isna(latest['Supertrend_Direction']) else 0,
                    'ema_trend':   'BULL' if (not pd.isna(latest['EMA_9']) and
                                   not pd.isna(latest['EMA_21']) and
                                   latest['EMA_9'] > latest['EMA_21']) else 'BEAR',
                    'vwap':        'ABOVE' if (not pd.isna(latest.get('VWAP', np.nan)) and
                                   latest['Close'] > latest['VWAP']) else 'BELOW',
                    'vol_ratio':   float(latest['Volume_Ratio'])
                                   if not pd.isna(latest['Volume_Ratio']) else 1.0,
                    'bb_pos':      'UPPER' if (not pd.isna(latest['BB_Upper']) and
                                   latest['Close'] >= latest['BB_Upper']) else
                                   ('LOWER' if (not pd.isna(latest['BB_Lower']) and
                                   latest['Close'] <= latest['BB_Lower']) else 'MID'),
                    # ── CPR fields ──
                    'cpr_pivot':  float(latest['CPR_Pivot']) if 'CPR_Pivot' in df.columns and not pd.isna(latest.get('CPR_Pivot', np.nan)) else None,
                    'cpr_tc':     float(latest['CPR_TC'])    if 'CPR_TC'    in df.columns and not pd.isna(latest.get('CPR_TC',    np.nan)) else None,
                    'cpr_bc':     float(latest['CPR_BC'])    if 'CPR_BC'    in df.columns and not pd.isna(latest.get('CPR_BC',    np.nan)) else None,
                    'cpr_r1':     float(latest['CPR_R1'])    if 'CPR_R1'    in df.columns and not pd.isna(latest.get('CPR_R1',    np.nan)) else None,
                    'cpr_s1':     float(latest['CPR_S1'])    if 'CPR_S1'    in df.columns and not pd.isna(latest.get('CPR_S1',    np.nan)) else None,
                    'cpr_r2':     float(latest['CPR_R2'])    if 'CPR_R2'    in df.columns and not pd.isna(latest.get('CPR_R2',    np.nan)) else None,
                    'cpr_s2':     float(latest['CPR_S2'])    if 'CPR_S2'    in df.columns and not pd.isna(latest.get('CPR_S2',    np.nan)) else None,
                    'cpr_width':  float(latest['CPR_Width']) if 'CPR_Width' in df.columns and not pd.isna(latest.get('CPR_Width', np.nan)) else None,
                    # ── Priority 4 & 6 fields ──
                    'sector':    SECTOR_MAP.get(sym_clean, ''),
                    'cap_tier':  get_cap_tier(sym_clean),
                    'gap_pct':   float(((latest['Open'] - prev['Close']) / prev['Close'] * 100))
                                 if not pd.isna(prev['Close']) and prev['Close'] != 0 else 0.0,
                    # Previous day high/low for resistance/support scoring
                    'pdh':       float(prev['High'])  if 'High'  in prev.index and not pd.isna(prev['High'])  else None,
                    'pdl':       float(prev['Low'])   if 'Low'   in prev.index and not pd.isna(prev['Low'])   else None,
                    # ── Relative Strength vs Nifty ──
                    'rs_vs_nifty': compute_relative_strength(
                                     float(((latest['Close']/prev['Close'])-1)*100)
                                     if not pd.isna(prev['Close']) and prev['Close'] != 0 else 0.0,
                                     st.session_state.get('nifty_context', {}).get('nifty_chg', 0.0)
                                  ),
                    # ── Previous Day High / Low ──
                    'pdh': float(latest['PDH']) if 'PDH' in df.columns and not pd.isna(latest.get('PDH', np.nan)) else None,
                    'pdl': float(latest['PDL']) if 'PDL' in df.columns and not pd.isna(latest.get('PDL', np.nan)) else None,
                    'pdc': float(latest['PDC']) if 'PDC' in df.columns and not pd.isna(latest.get('PDC', np.nan)) else None,
                }
                ps, _, vrd   = compute_intraday_pick_score(r)
                r['_pick_score'] = ps
                r['_verdict']    = vrd
                r['_alerts']     = evaluate_alerts(r, port)
                raw_results.append(r)

                # ── Live stat update every 10 stocks ──────
                if len(raw_results) % 10 == 0:
                    _stat.markdown(
                        f"<div style='display:flex;gap:20px;padding:6px 0;font-size:12px;color:#64748b'>"
                        f"<span>📊 <b>{len(raw_results)}</b> valid</span>"
                        f"<span>⭐⭐⭐ <b>{sum(1 for x in raw_results if x.get('_verdict')=='⭐⭐⭐ STRONG BUY')}</b> Strong Buy</span>"
                        f"<span>⭐⭐ <b>{sum(1 for x in raw_results if x.get('_verdict')=='⭐⭐ BUY')}</b> Buy</span>"
                        f"<span>{'⚡ Kite' if src=='kite' else '⏳ yfinance'}</span>"
                        f"</div>", unsafe_allow_html=True)

                # ── Update sector momentum every 50 stocks ─
                if len(raw_results) % 50 == 0 and raw_results:
                    st.session_state['sector_momentum'] = get_sector_momentum(raw_results)
            except Exception:
                continue

        # ── Scan complete ──────────────────────────────────
        _scan_duration = round(time.time() - scan_start, 1)
        _prog.progress(100, text=f"✅ Complete — {total} stocks in {_scan_duration}s · {len(raw_results)} results")
        _sym_ph.empty()
        _live_ph.empty()   # clear live panel — shortlist takes over
        # Final sector momentum update
        st.session_state['sector_momentum'] = get_sector_momentum(raw_results)

        _srcs = [r.get('data_src','') for r in raw_results]
        _kc   = _srcs.count('kite')
        _yc   = _srcs.count('yfinance')
        if _kc > 0 and _yc == 0:
            st.session_state['data_source'] = f"⚡ Kite API · {interval} · Real-Time"
        elif _kc > 0:
            st.session_state['data_source'] = f"⚡ Kite ({_kc}) + ⏳ yfinance ({_yc})"
        else:
            yf_map = {'1minute':'1m','5minute':'5m','15minute':'15m','60minute':'1h'}
            st.session_state['data_source'] = f"⏳ yfinance · {yf_map.get(interval,'5m')} · ~15min delay"

        filtered = [r for r in raw_results if VERDICT_RANK.get(r['_verdict'], 0) >= min_rank]
        filtered.sort(key=lambda x: x['_pick_score'], reverse=True)
        st.session_state['scan_results']  = filtered
        st.session_state['scan_raw']      = raw_results
        st.session_state['scan_key']      = cache_key
        st.session_state['scan_time']     = ist_now().strftime('%d %b %Y, %H:%M IST')
        st.session_state['scan_duration'] = _scan_duration
        st.session_state['scan_total']    = total

        # ── Auto-save scan history for ML training data ──
        _ctx_for_csv = st.session_state.get('nifty_context', {})
        save_scan_history(
            raw_results,
            interval,
            _ctx_for_csv.get('state', 'UNKNOWN'),
            _ctx_for_csv.get('vix'),
        )
        # ── Reset auto-refresh timer ──────────────────────
        reset_refresh_timer()
        st.rerun()

    all_results = st.session_state.get('scan_results', [])
    raw_results = st.session_state.get('scan_raw',     [])
    all_results = [r for r in raw_results if VERDICT_RANK.get(r.get('_verdict',''), 0) >= min_rank]
    all_results.sort(key=lambda x: x.get('_pick_score',0), reverse=True)

    if not all_results:
        _mkt_closed = not market_open()
        _raw_count  = len(st.session_state.get('scan_raw', []))
        _dur        = st.session_state.get('scan_duration', 0)

        if _mkt_closed:
            st.markdown(f"""
            <div style='background:#1a2035;border-radius:18px;padding:32px 36px;
                        text-align:center;margin:20px 0'>
                <div style='font-size:40px;margin-bottom:12px'>🔴</div>
                <div style='font-size:22px;font-weight:800;color:#ffffff;margin-bottom:8px'>
                    Market is Closed
                </div>
                <div style='font-size:15px;color:rgba(255,255,255,0.6);margin-bottom:20px'>
                    NSE trading hours are <b style='color:#f59e0b'>9:15 AM – 3:30 PM IST</b> on weekdays.<br>
                    Current time: <b style='color:#f59e0b'>{ist_now().strftime('%d %b %Y, %H:%M IST')}</b>
                </div>
                <div style='display:flex;gap:16px;justify-content:center;flex-wrap:wrap'>
                    <div style='background:#111827;border-radius:12px;padding:16px 24px;min-width:160px'>
                        <div style='font-size:11px;color:#6b7280;font-weight:700;letter-spacing:1px;text-transform:uppercase'>Stocks Scanned</div>
                        <div style='font-size:24px;font-weight:800;color:#f59e0b'>{st.session_state.get("scan_total",0)}</div>
                    </div>
                    <div style='background:#111827;border-radius:12px;padding:16px 24px;min-width:160px'>
                        <div style='font-size:11px;color:#6b7280;font-weight:700;letter-spacing:1px;text-transform:uppercase'>Returned Data</div>
                        <div style='font-size:24px;font-weight:800;color:#f59e0b'>{_raw_count}</div>
                    </div>
                    <div style='background:#111827;border-radius:12px;padding:16px 24px;min-width:160px'>
                        <div style='font-size:11px;color:#6b7280;font-weight:700;letter-spacing:1px;text-transform:uppercase'>Scan Time</div>
                        <div style='font-size:24px;font-weight:800;color:#f59e0b'>{_dur}s</div>
                    </div>
                </div>
                <div style='margin-top:24px;background:#0f172a;border-radius:12px;
                            padding:16px 20px;text-align:left;max-width:500px;margin-left:auto;margin-right:auto'>
                    <div style='font-size:13px;font-weight:700;color:#f59e0b;margin-bottom:8px'>
                        💡 What to do right now:
                    </div>
                    <div style='font-size:13px;color:rgba(255,255,255,0.7);line-height:1.8'>
                        1. Switch timeframe to <b style='color:#f59e0b'>15min or 1hr</b> — these have historical data<br>
                        2. Or come back at <b style='color:#f59e0b'>9:20 AM IST</b> Monday for live 1min data<br>
                        3. Use <b style='color:#f59e0b'>Custom Watchlist</b> mode with 5–10 stocks to test the app
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='background:#fffbeb;border:2px solid #f59e0b;border-radius:14px;
                        padding:24px 28px;text-align:center;margin:20px 0'>
                <div style='font-size:24px;font-weight:800;color:#92400e;margin-bottom:8px'>
                    🔍 No Stocks Matched "{min_verdict}"
                </div>
                <div style='font-size:14px;color:#92400e;margin-bottom:16px'>
                    {_raw_count} stocks returned data but none met the minimum verdict filter.
                </div>
                <div style='font-size:13px;color:#78350f'>
                    👉 Try lowering the filter to <b>⭐ WATCH</b> or <b>⚠️ NEUTRAL</b> in the sidebar
                </div>
            </div>""", unsafe_allow_html=True)
        st.stop()

    buy_count    = sum(1 for r in all_results if r.get('_verdict','') == '⭐⭐⭐ STRONG BUY')
    good_count   = sum(1 for r in all_results if r.get('_verdict','') == '⭐⭐ BUY')
    watch_count2 = sum(1 for r in all_results if r.get('_verdict','') == '⭐ WATCH')
    avg_conf     = int(sum(r['live_conf'] for r in all_results) / len(all_results)) if all_results else 0

    mc1, mc2, mc3, mc4 = st.columns(4)
    for col, label, val, sub, ic_bg, ic_col, val_cls, icon_svg in [
        (mc1, "Strong Buy",  buy_count,    "Highest conviction",
         "#c8f135","#1a2035","stat-green",
         "<svg xmlns='http://www.w3.org/2000/svg' width='20' height='20' fill='none' viewBox='0 0 24 24' stroke='#1a2035' stroke-width='2'><polyline points='8 21 12 21 16 21'/><line x1='12' y1='17' x2='12' y2='21'/><path d='M7 4H17l-1 7a5 5 0 0 1-4 4 5 5 0 0 1-4-4L7 4z'/></svg>"),
        (mc2, "Buy",         good_count,   "Good setups",
         "#dcfce7","#15803d","stat-green",
         "<svg xmlns='http://www.w3.org/2000/svg' width='20' height='20' fill='none' viewBox='0 0 24 24' stroke='#15803d' stroke-width='2'><polyline points='23 6 13.5 15.5 8.5 10.5 1 18'/></svg>"),
        (mc3, "Watch",       watch_count2, "Building setups",
         "#fef9c3","#d97706","stat-amber",
         "<svg xmlns='http://www.w3.org/2000/svg' width='20' height='20' fill='none' viewBox='0 0 24 24' stroke='#d97706' stroke-width='2'><circle cx='12' cy='12' r='10'/><polyline points='12 6 12 12 16 14'/></svg>"),
        (mc4, "Avg Confidence", f"{avg_conf}%", f"{len(all_results)} total matched",
         "#f0f4ff","#1d4ed8","",
         "<svg xmlns='http://www.w3.org/2000/svg' width='20' height='20' fill='none' viewBox='0 0 24 24' stroke='#1d4ed8' stroke-width='2'><rect x='3' y='12' width='4' height='9' rx='1'/><rect x='10' y='7' width='4' height='14' rx='1'/><rect x='17' y='3' width='4' height='18' rx='1'/></svg>"),
    ]:
        with col:
            col.markdown(f"""<div class='stat-card'>
                <div class='stat-card-icon' style='background:{ic_bg}'>{icon_svg}</div>
                <div class='stat-label'>{label}</div>
                <div class='stat-value {val_cls}'>{val}</div>
                <div class='stat-sub'>{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown(f"<div class='section-header'>⚡ Intraday Picks · {len(all_results)} results</div>", unsafe_allow_html=True)

    # ── Stock results — LAZY RENDER (only selected stock) ──────────
    # st.tabs renders ALL content for ALL tabs upfront — very slow.
    # Instead we use a compact summary table + single stock selectbox.
    # Only ONE stock is rendered at a time → instant page load.

    # ══════════════════════════════════════════════════════
    #  🎯 SHORTLIST PANEL
    #  Automatically applies all 5 filters + ranks top picks
    #  This is the ONLY section you need to look at daily
    # ══════════════════════════════════════════════════════

    st.markdown("""
    <div style='background:linear-gradient(135deg,#1a2035,#2d3561);
                border-radius:16px;padding:20px 24px;margin-bottom:16px'>
        <div style='font-size:20px;font-weight:800;color:#ffffff;
                    display:flex;align-items:center;gap:10px'>
            🎯 Today's Shortlist
            <span style='background:#f59e0b;color:#1a2035;border-radius:20px;
                         padding:3px 12px;font-size:11px;font-weight:800'>
                AUTO-FILTERED
            </span>
        </div>
        <div style='font-size:13px;color:rgba(255,255,255,0.6);margin-top:4px'>
            Stocks that pass ALL 10 filters — Score ≥ 75 · VWAP Above ·
            Volume ≥ 2.5× · Liquidity Excellent · RS > −0.5% · Signal Fresh · R:R ≥ 1.5
        </div>
    </div>""", unsafe_allow_html=True)

    # ── Apply all 5 filters ────────────────────────────────
    _shortlist = []
    _reject_reasons = {}

    # ── Nifty state for filter 7 ──────────────────────────
    _nifty_st = st.session_state.get('nifty_market_state', 'UNKNOWN')

    for _r in all_results:
        _sym    = _r['symbol'].replace('.NS','')
        _score  = _r.get('_pick_score', 0)
        _vwap   = _r.get('vwap', 'BELOW')
        _vol    = _r.get('vol_ratio', 0.0)
        _liq    = _r.get('liquidity', {}).get('grade', 'LOW')
        _rs     = _r.get('rs_vs_nifty', 0.0) or 0.0
        _cpr_w  = _r.get('cpr_width')
        _wu     = _r.get('warmup', 'READY')
        _iv     = _r.get('interval', '1minute')
        _cpd    = CANDLES_PER_DAY.get(_iv, 375)
        _verd   = _r.get('_verdict', '')
        _rsi    = _r.get('rsi', 50) or 50
        _sig_age= _r.get('sig_age_candles', 0)
        _tp     = _r.get('trade_plan') or {}
        _price  = _r.get('price', 0) or 0
        _sl_px  = _f(_tp.get('stop_loss', 0))
        _t1_px  = _f(_tp.get('t1', 0))
        _rr     = 0.0
        _sl_dist= 99.0
        if _price > 0 and _sl_px > 0 and _t1_px > 0:
            _risk_d  = _price - _sl_px
            _rew_d   = _t1_px - _price
            _rr      = round(_rew_d / _risk_d, 2) if _risk_d > 0 else 0
            _sl_dist = round(_risk_d / _price * 100, 2)

        # Skip warming up / partial
        if _wu == 'WARMING_UP':
            continue
        if _wu == 'PARTIAL' and _cpd >= 75:
            continue

        # ── Filter 1 — Score >= 75 (raised from 65) ───────
        if _score < 75:
            _reject_reasons[_sym] = f'Score {_score} < 75'
            continue

        # ── Filter 2 — VWAP Above ─────────────────────────
        if _vwap != 'ABOVE':
            _reject_reasons[_sym] = 'VWAP Below'
            continue

        # ── Filter 3 — Volume (BEAR day: lower threshold) ──
        # Defensive/commodity stocks move on 1.5-2× not 2.5×
        _vol_threshold = (1.5 if _cpd <= 10 else (2.0 if _cpd <= 25 else
                          (1.5 if _nifty_st == 'BEAR' else 2.5)))
        if _vol < _vol_threshold:
            _reject_reasons[_sym] = f'Vol {_vol:.1f}× < {_vol_threshold}×'
            continue

        # ── Filter 4 — Liquidity (BEAR day: allow HIGH too) ─
        _liq_required = ['EXCELLENT'] if _nifty_st != 'BEAR' else ['EXCELLENT','HIGH']
        if _liq not in _liq_required:
            _reject_reasons[_sym] = f'Liquidity {_liq} (need EXCELLENT{" or HIGH on BEAR days" if _nifty_st=="BEAR" else ""})'
            continue

        # ── Filter 5 — RS not severely underperforming ────
        # RS already scored in pick score (+15 to -15 pts)
        # Hard filter only rejects stocks significantly lagging Nifty
        # On bull days RS=0% is fine — stock moving with market
        if _rs < -0.5:
            _reject_reasons[_sym] = f'RS {_rs:.1f}% severely underperforming Nifty'
            continue

        # ── Filter 6 — Signal age <= 5 candles ────────────
        # 5-min chart: 5 candles = 25 min → valid at 9:35 AM scan ✅
        # 1-min chart: 5 candles = 5 min → better than 3 min
        if _sig_age > 5:
            _reject_reasons[_sym] = f'Signal {_sig_age} candles old (stale)'
            continue

        # ── Filter 7 — BEAR day RS gate ───────────────────
        # On BEAR days don't block everything — only block
        # stocks NOT outperforming Nifty significantly.
        # Stocks with RS > 1.0% are moving independently
        # (defensive, commodity, sector-specific) — allow them.
        if _nifty_st == 'BEAR' and _rs < 1.0:
            _reject_reasons[_sym] = f'Nifty BEAR + RS only {_rs:.1f}% (need > 1.0% on BEAR days)'
            continue

        # ── Filter 8 — RSI not overbought (< 72) ──────────
        if _rsi > 72:
            _reject_reasons[_sym] = f'RSI {_rsi:.0f} overbought (> 72)'
            continue

        # ── Filter 9 — SL within 1.5% of entry ───────────
        if _sl_dist > 1.5:
            _reject_reasons[_sym] = f'SL {_sl_dist:.1f}% away (> 1.5%)'
            continue

        # ── Filter 10 — R:R >= 1.5 ────────────────────────
        if _rr > 0 and _rr < 1.5:
            _reject_reasons[_sym] = f'R:R {_rr:.1f} < 1.5'
            continue

        # ── Filter 11 — BEAR day: Largecap only ──────────
        # On BEAR days midcap/smallcap losses are much larger
        # Only allow MIDCAP if RS > 2% (very strong outperformance)
        _cap_tier = _r.get('cap_tier', get_cap_tier(_sym))
        if _nifty_state == 'BEAR':
            if _cap_tier == 'SMALLCAP':
                _reject_reasons[_sym] = 'BEAR day — Smallcap blocked (too risky)'
                continue
            if _cap_tier == 'MIDCAP' and _rs < 2.0:
                _reject_reasons[_sym] = f'BEAR day — Midcap RS {_rs:.1f}% < 2% required'
                continue

        # ── Passed all 11 filters ─────────────────────────
        _cpr_ok  = _cpr_w is not None and _cpr_w < 0.6
        _mtf_key = f"mtf_{_r['symbol']}_{interval}"
        _align   = st.session_state.get(_mtf_key, {}).get('alignment', 'UNKNOWN')

        # Position size multiplier — market conditions
        _pos_mult, _pos_lbl, _pos_clr = get_position_size_multiplier(_rs)
        _adj_qty  = max(1, int(_f(_tp.get('qty', 0)) * _pos_mult))
        _adj_inv  = round(_price * _adj_qty, 2)

        _shortlist.append({
            'result':    _r,
            'sym':       _sym,
            'score':     _score,
            'verdict':   _verd,
            'price':     _price,
            'chg_pct':   _r.get('change_pct', 0),
            'vol':       _vol,
            'rs':        _rs,
            'liq':       _liq,
            'vwap':      _vwap,
            'cpr_ok':    _cpr_ok,
            'cpr_w':     _cpr_w,
            'rsi':       _rsi,
            'mtf_align': _align,
            'alerts':    _r.get('_alerts', []),
            'sig_age':   _sig_age,
            'sig_move':  _r.get('sig_move_since', 0.0),
            'sig_price': _r.get('sig_price_at_fire', _price),
            'entry':     _price,
            'sl':        _sl_px,
            't1':        _t1_px,
            't2':        _f(_tp.get('t2', 0)),
            'qty':       int(_f(_tp.get('qty', 0))),
            'adj_qty':   _adj_qty,       # position-size-adjusted qty
            'adj_inv':   _adj_inv,       # adjusted investment
            'pos_mult':  _pos_mult,      # 0.25 to 1.0
            'pos_lbl':   _pos_lbl,       # "70% size — Bull + High VIX"
            'pos_clr':   _pos_clr,
            'cap_tier':  _cap_tier,
            'rr':        _rr,
            'sl_dist':   _sl_dist,
            'investment':_f(_tp.get('investment', 0)),
            'risk_amt':  _f(_tp.get('risk_amount', 0)),
        })

    _shortlist.sort(key=lambda x: x['score'], reverse=True)

    # ── Show shortlist results ─────────────────────────────
    if not _shortlist:
        _now_ist2 = ist_now()
        _tm2      = _now_ist2.hour * 60 + _now_ist2.minute
        _reason   = (
            "Market opened less than 20 min ago — wait until 9:35 AM" if _tm2 < 575 else
            "VIX is EXTREME/CRISIS — signals capped today" if st.session_state.get('nifty_context',{}).get('vix_level') in ['HIGH','EXTREME','CRISIS'] else
            "Nifty is BEAR — only stocks with RS > 1.0% qualify today" if st.session_state.get('nifty_market_state') == 'BEAR' else
            "No stocks passed all 10 precision filters — do not force a trade today"
        )
        st.markdown(f"""
        <div style='background:#1a2035;border:2px solid #374151;border-radius:14px;
                    padding:24px;text-align:center;margin-bottom:16px'>
            <div style='font-size:32px;margin-bottom:10px'>🔍</div>
            <div style='font-size:17px;font-weight:800;color:#ffffff;margin-bottom:8px'>
                No stocks passed all 10 precision filters
            </div>
            <div style='font-size:13px;color:#f59e0b;font-weight:600;margin-bottom:16px'>
                {_reason}
            </div>
            <div style='font-size:12px;color:rgba(255,255,255,0.5);line-height:1.8'>
                ✅ This is the correct outcome — <b style='color:#34d399'>do not trade today</b><br>
                0 signals = no trade. Forcing trades when filters fail = guaranteed losses.<br>
                Wait for next scan cycle or tomorrow's session.
            </div>
        </div>""", unsafe_allow_html=True)

    else:
        # Header row
        _now_ist3 = ist_now()
        _time_msg = (
            "⚡ Best entry window active — 9:35 AM to 11:30 AM" if 575 <= _now_ist3.hour * 60 + _now_ist3.minute <= 690
            else "⚠️ Late session — reduce position size" if _now_ist3.hour * 60 + _now_ist3.minute > 810
            else "✅ Good trading window"
        )
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;align-items:center;"
            f"padding:6px 0;margin-bottom:8px'>"
            f"<span style='font-size:13px;font-weight:700;color:#1a2035'>"
            f"✅ {len(_shortlist)} stock{'s' if len(_shortlist)>1 else ''} passed all 10 filters — buy without review</span>"
            f"<span style='font-size:12px;color:#64748b'>{_time_msg}</span>"
            f"</div>", unsafe_allow_html=True)

        # Stock cards
        for _rank, _sl in enumerate(_shortlist[:5], 1):
            _chg_clr = "#16a34a" if _sl['chg_pct'] >= 0 else "#dc2626"
            _rs_clr  = "#16a34a" if _sl['rs'] >= 0 else "#dc2626"
            _vd      = _sl['verdict']
            _vd_bg   = {"⭐⭐⭐ STRONG BUY":"#dcfce7","⭐⭐ BUY":"#dbeafe"}.get(_vd,"#f8fafc")
            _vd_clr  = {"⭐⭐⭐ STRONG BUY":"#15803d","⭐⭐ BUY":"#1d4ed8"}.get(_vd,"#64748b")

            # Signal freshness
            _sig_age  = _sl.get('sig_age', 0)
            _sig_move = _sl.get('sig_move', 0.0)
            if _sig_age == 0:
                _fresh_badge = "<span style='background:#dcfce7;color:#15803d;font-size:10px;font-weight:700;border-radius:4px;padding:2px 8px'>🟢 Just fired</span>"
            elif _sig_age <= 2:
                _fresh_badge = f"<span style='background:#dcfce7;color:#15803d;font-size:10px;font-weight:700;border-radius:4px;padding:2px 8px'>🟢 {_sig_age}c ago · Fresh</span>"
            else:
                _fresh_badge = f"<span style='background:#fef3c7;color:#92400e;font-size:10px;font-weight:700;border-radius:4px;padding:2px 8px'>🟡 {_sig_age}c ago · +{_sig_move:.1f}%</span>"

            # Trade values
            _entry  = _sl.get('entry', 0)
            _sl_px  = _sl.get('sl', 0)
            _t1     = _sl.get('t1', 0)
            _t2     = _sl.get('t2', 0)
            _qty    = _sl.get('adj_qty', _sl.get('qty', 0))   # use adjusted qty
            _rr     = _sl.get('rr', 0)
            _sl_d   = _sl.get('sl_dist', 0)
            _inv    = _sl.get('adj_inv', _sl.get('investment', 0))
            _risk   = _sl.get('risk_amt', 0)
            _pos_lbl= _sl.get('pos_lbl', '100% size — Full position')
            _pos_clr= _sl.get('pos_clr', '#16a34a')
            _pos_mult=_sl.get('pos_mult', 1.0)
            _sl_pct = round((_entry - _sl_px) / _entry * 100, 2) if _entry > 0 else 0
            _t1_pct = round((_t1 - _entry) / _entry * 100, 2)    if _entry > 0 else 0
            _t2_pct = round((_t2 - _entry) / _entry * 100, 2)    if _entry > 0 else 0

            # Cap tier badge
            _cap     = _sl.get('cap_tier', 'SMALLCAP')
            _cap_ico, _cap_name, _cap_clr, _cap_bg = CAP_TIER_BADGE.get(
                _cap, ('🟠', 'Smallcap', '#c2410c', '#fff7ed'))
            _cap_badge = (
                f"<span style='background:{_cap_bg};color:{_cap_clr};"
                f"font-size:10px;font-weight:700;border-radius:4px;"
                f"padding:2px 8px;border:1px solid {_cap_clr}44;margin-left:6px'>"
                f"{_cap_ico} {_cap_name}</span>"
            )

            # Rank badge colour
            _rb = {1:"#f59e0b",2:"#94a3b8",3:"#b45309"}.get(_rank,"#e2e8f0")
            _rt = {1:"#1a2035",2:"#ffffff",3:"#ffffff"}.get(_rank,"#64748b")

            st.markdown(f"""
            <div style='background:#ffffff;border:2px solid {_vd_clr}33;
                        border-radius:16px;padding:18px 20px;margin-bottom:12px;
                        box-shadow:0 2px 12px rgba(0,0,0,0.06)'>

                <!-- Header row -->
                <div style='display:flex;align-items:flex-start;
                            justify-content:space-between;flex-wrap:wrap;gap:8px'>
                    <div style='display:flex;align-items:center;gap:12px'>
                        <div style='background:{_rb};color:{_rt};width:34px;height:34px;
                                    border-radius:50%;display:flex;align-items:center;
                                    justify-content:center;font-size:15px;
                                    font-weight:800;flex-shrink:0'>{_rank}</div>
                        <div>
                            <div style='display:flex;align-items:center;gap:8px;flex-wrap:wrap'>
                                <span style='font-size:22px;font-weight:800;color:#1a2035'>{_sl["sym"]}</span>
                                <span style='background:{_vd_bg};color:{_vd_clr};
                                             font-size:11px;font-weight:700;
                                             border-radius:6px;padding:3px 10px'>{_vd}</span>
                                {_fresh_badge}
                            </div>
                            <div style='font-size:12px;color:#64748b;margin-top:4px'>
                                Score <b style='color:#1a2035'>{_sl["score"]}/100</b>
                                &nbsp;·&nbsp; RSI {_sl["rsi"]:.0f}
                                &nbsp;·&nbsp; Vol {_sl["vol"]:.1f}×
                                &nbsp;·&nbsp; RS <span style='color:{_rs_clr}'>{_sl["rs"]:+.1f}%</span>
                            </div>
                            <div style='margin-top:5px'>
                                <span style='background:{_pos_clr}22;color:{_pos_clr};
                                             font-size:10px;font-weight:700;
                                             border-radius:4px;padding:2px 8px;
                                             border:1px solid {_pos_clr}44'>
                                    💰 {_pos_lbl}
                                </span>
                                {_cap_badge}
                            </div>
                        </div>
                    </div>
                    <div style='text-align:right'>
                        <div style='font-size:24px;font-weight:800;color:#1a2035;
                                    font-family:JetBrains Mono'>₹{_entry:,.2f}</div>
                        <div style='font-size:13px;font-weight:700;color:{_chg_clr}'>
                            {_sl["chg_pct"]:+.2f}% today
                        </div>
                    </div>
                </div>

                <!-- Trade plan — the key info -->
                <div style='display:flex;gap:8px;margin-top:14px;flex-wrap:wrap'>
                    <div style='background:#f0fdf4;border:1px solid #bbf7d0;
                                border-radius:10px;padding:10px 14px;flex:1;min-width:90px;text-align:center'>
                        <div style='font-size:9px;font-weight:700;color:#15803d;letter-spacing:1px'>ENTRY</div>
                        <div style='font-size:18px;font-weight:800;color:#15803d;
                                    font-family:JetBrains Mono;margin:3px 0'>₹{_entry:,.2f}</div>
                        <div style='font-size:10px;color:#15803d'>{_qty} shares · ₹{_inv:,.0f}</div>
                    </div>
                    <div style='background:#fff5f5;border:1px solid #fecaca;
                                border-radius:10px;padding:10px 14px;flex:1;min-width:90px;text-align:center'>
                        <div style='font-size:9px;font-weight:700;color:#dc2626;letter-spacing:1px'>STOP LOSS</div>
                        <div style='font-size:18px;font-weight:800;color:#dc2626;
                                    font-family:JetBrains Mono;margin:3px 0'>₹{_sl_px:,.2f}</div>
                        <div style='font-size:10px;color:#dc2626'>−{_sl_d:.1f}% · Risk ₹{_risk:,.0f}</div>
                    </div>
                    <div style='background:#eff6ff;border:1px solid #bfdbfe;
                                border-radius:10px;padding:10px 14px;flex:1;min-width:90px;text-align:center'>
                        <div style='font-size:9px;font-weight:700;color:#1d4ed8;letter-spacing:1px'>T1 TARGET</div>
                        <div style='font-size:18px;font-weight:800;color:#1d4ed8;
                                    font-family:JetBrains Mono;margin:3px 0'>₹{_t1:,.2f}</div>
                        <div style='font-size:10px;color:#1d4ed8'>+{_t1_pct:.1f}%</div>
                    </div>
                    <div style='background:#f5f3ff;border:1px solid #ddd6fe;
                                border-radius:10px;padding:10px 14px;flex:1;min-width:90px;text-align:center'>
                        <div style='font-size:9px;font-weight:700;color:#7c3aed;letter-spacing:1px'>T2 TARGET</div>
                        <div style='font-size:18px;font-weight:800;color:#7c3aed;
                                    font-family:JetBrains Mono;margin:3px 0'>₹{_t2:,.2f}</div>
                        <div style='font-size:10px;color:#7c3aed'>+{_t2_pct:.1f}% · R:R {_rr:.1f}</div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

            # One-click paper buy button — full width, prominent
            _pb_key = f"sl_paper_buy_{_sl['sym']}_{_rank}"
            if st.button(
                f"✅ Paper Buy  {_sl['sym']}  ·  Entry ₹{_entry:,.2f}  ·  SL ₹{_sl_px:,.2f}  ·  T1 ₹{_t1:,.2f}  ·  Qty {_qty}",
                key=_pb_key,
                use_container_width=True,
                type="primary",
            ):
                _port = load_portfolio()
                # Check if already have open position in this stock
                _already_open = any(
                    p.get('symbol') == _sl['sym'] and p.get('status') == 'OPEN'
                    for p in _port
                )
                if _already_open:
                    st.warning(f"⚠️ Already have an open position in {_sl['sym']} — skipping")
                else:
                    _new_pos = {
                        'symbol':      _sl['sym'],
                        'status':      'OPEN',
                        'entry':       round(_entry, 2),
                        'qty':         _qty,
                        'stop_loss':   round(_sl_px, 2),
                        't1':          round(_t1, 2),
                        't2':          round(_t2, 2),
                        't3':          round(_sl.get('result',{}).get('trade_plan',{}).get('t3',0) or 0, 2),
                        't4':          round(_sl.get('result',{}).get('trade_plan',{}).get('t4',0) or 0, 2),
                        'investment':  round(_inv, 2),
                        'actual_cost': round(_inv, 2),
                        'timeframe':   interval,
                        'date':        ist_now().strftime('%d %b %Y %H:%M'),
                        'entry_time':  ist_now().strftime('%H:%M'),
                        'nifty_state': st.session_state.get('nifty_market_state', 'UNKNOWN'),
                        'vix_level':   st.session_state.get('nifty_context', {}).get('vix_level', 'UNKNOWN'),
                        'score':       _sl['score'],
                        'verdict':     _vd,
                        'sig_age':     _sig_age,
                        'rs_vs_nifty': _sl['rs'],
                        'vol_ratio':   _sl['vol'],
                        'source':      'shortlist_quick_buy',
                        'exit_reason': '',
                    }
                    _port.append(_new_pos)
                    save_portfolio(_port)
                    st.session_state['paper_portfolio'] = _port
                    st.success(
                        f"✅ Paper bought {_qty} shares of {_sl['sym']} @ ₹{_entry:,.2f} · "
                        f"SL ₹{_sl_px:,.2f} · T1 ₹{_t1:,.2f} · Risk ₹{_risk:,.0f}"
                    )
                    st.rerun()

        # Show count of rejected stocks
        if _reject_reasons:
            with st.expander(f"👁 {len(_reject_reasons)} stocks scanned but filtered out — see why"):
                _rej_data = [{'Stock': k, 'Filtered because': v}
                             for k, v in list(_reject_reasons.items())[:30]]
                st.dataframe(pd.DataFrame(_rej_data), use_container_width=True,
                             hide_index=True)

    st.markdown("<hr style='border:none;border-top:1px solid #e2e8f0;margin:16px 0'>",
                unsafe_allow_html=True)

    # ── Compact summary table ──────────────────────────────────────
    _summary_rows = []
    for _r in all_results[:50]:
        _liq  = _r.get('liquidity', {})
        _alts = _r.get('_alerts', [])
        _alt_icons = ''.join(set(
            a.get('icon','') for a in _alts
            if a['type'] in ['STOP_LOSS','TARGET_T1','TARGET_T2','TARGET_T3','TARGET_T4',
                              'STRONG_BUY','BUY','VWAP_BREAK','VOL_SURGE']
        ))
        _wu   = _r.get('warmup', 'READY')
        _wu_display = {
            'WARMING_UP': '🚫 Not ready',
            'PARTIAL':    f"⏳ {_r.get('pct_ready',0)}% ready",
            'READY':      '✅ Ready',
        }.get(_wu, '✅ Ready')
        _summary_rows.append({
            'Symbol':     _r['symbol'].replace('.NS',''),
            'Price':      f"₹{_r['price']:,.2f}",
            'Change':     f"{'+' if _r['change_pct']>=0 else ''}{_r['change_pct']:.2f}%",
            'Score':      _r.get('_pick_score', 0),
            'Verdict':    _r.get('_verdict',''),
            'Signals':    _wu_display,
            'RSI':        f"{_r['rsi']:.0f}",
            'VWAP':       _r['vwap'],
            'Vol×':       (f"🏦{_r['vol_ratio']:.0f}×" if _r['vol_ratio'] >= 15 else
                           f"🔥{_r['vol_ratio']:.0f}×" if _r['vol_ratio'] >= 8  else
                           f"⚡{_r['vol_ratio']:.1f}×" if _r['vol_ratio'] >= 5  else
                           f"↑{_r['vol_ratio']:.1f}×"  if _r['vol_ratio'] >= 2  else
                           f"{_r['vol_ratio']:.1f}×"),
            'Liquidity':  _liq.get('grade','—'),
            'Conf%':      f"{_r['live_conf']}%",
            'Source':     '⚡ Kite' if _r.get('data_src') == 'kite' else '⏳ yfinance',
            'CPR':        ('⚡N' if (_r.get('cpr_width') or 99) < 0.4
                           else ('〰M' if (_r.get('cpr_width') or 99) < 0.8
                           else ('⚠️W' if _r.get('cpr_width') else '—'))),
            'RS':         (f"{'+' if (_r.get('rs_vs_nifty') or 0)>=0 else ''}"
                           f"{(_r.get('rs_vs_nifty') or 0):.1f}%"),
            'PDH':        (f"₹{_r['pdh']:,.1f}" if _r.get('pdh') else '—'),
            'Alerts':     _alt_icons if _alt_icons else '—',
        })
    _sum_df = pd.DataFrame(_summary_rows)

    _sh_col1, _sh_col2 = st.columns([4, 1])
    with _sh_col1:
        st.markdown("<div class='section-header'>📋 All Results — Click a stock below to analyse</div>",
                    unsafe_allow_html=True)
    with _sh_col2:
        if SCAN_HISTORY_FILE.exists():
            try:
                _hist_bytes = SCAN_HISTORY_FILE.read_bytes()
                st.download_button(
                    "📥 History CSV",
                    data=_hist_bytes,
                    file_name="investo_scan_history.csv",
                    mime="text/csv",
                    use_container_width=True,
                    help="Download all scan history for analysis / ML training"
                )
            except Exception:
                pass

    # Color-code verdict column
    def _color_verdict(val):
        colors = {
            '⭐⭐⭐ STRONG BUY': 'background-color:#dcfce7;color:#15803d;font-weight:700',
            '⭐⭐ BUY':          'background-color:#dbeafe;color:#1d4ed8;font-weight:700',
            '⭐ WATCH':          'background-color:#fef3c7;color:#92400e;font-weight:700',
            '⚠️ NEUTRAL':        'background-color:#f1f5f9;color:#64748b',
            '❌ AVOID':          'background-color:#fee2e2;color:#991b1b',
        }
        return colors.get(val, '')

    _styled = _sum_df.style.applymap(_color_verdict, subset=['Verdict'])
    st.dataframe(_styled, use_container_width=True, hide_index=True,
                 column_config={
                     'Score': st.column_config.ProgressColumn('Score', min_value=0, max_value=100, format='%d'),
                 })

    # ── Single stock selector ───────────────────────────────────────
    st.markdown("<div class='section-header'>🔬 Deep Analyse a Stock</div>", unsafe_allow_html=True)

    _sym_options = [r['symbol'].replace('.NS','') for r in all_results[:50]]

    # Auto-select stock from alert or top pick
    _default_idx = 0
    if st.session_state.get('_focus_stock'):
        _fs = st.session_state['_focus_stock']
        if _fs in _sym_options:
            _default_idx = _sym_options.index(_fs)

    st.markdown("<div class='section-header'>🔬 Deep Analyse a Stock</div>", unsafe_allow_html=True)

    _sym_options = [r['symbol'].replace('.NS','') for r in all_results[:50]]

    # Auto-select stock from alert or top pick
    _default_idx = 0
    if st.session_state.get('_focus_stock'):
        _fs = st.session_state['_focus_stock']
        if _fs in _sym_options:
            _default_idx = _sym_options.index(_fs)

    _sel_col1, _sel_col2, _sel_col3 = st.columns([3, 1, 1])
    with _sel_col1:
        _selected_sym = st.selectbox(
            "Select stock to analyse",
            _sym_options, index=_default_idx,
            key="stock_selector",
            help="Only the selected stock is rendered — keeps the page fast"
        )
    with _sel_col2:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if st.button("▶ Top Pick", key="auto_top"):
            st.session_state['_focus_stock'] = _sym_options[0]
            st.rerun()
    with _sel_col3:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        _da_stock_refresh = st.button(
            "🔄 Refresh Stock",
            key=f"refresh_stock_{_selected_sym}",
            use_container_width=True,
            type="primary",
            help="Re-fetch latest data and recalculate all indicators for this stock"
        )

    # Show last refresh time
    _da_stock_refresh_time = st.session_state.get(f'stock_refresh_time_{_selected_sym}', '')
    if _da_stock_refresh_time:
        st.markdown(
            f"<div style='font-size:11px;color:#94a3b8;margin:-8px 0 8px 0'>"
            f"🔄 Last refreshed: <b>{_da_stock_refresh_time}</b></div>",
            unsafe_allow_html=True)

    # ── On refresh: re-fetch data, recalculate indicators, update result ──
    if _da_stock_refresh:
        _refresh_sym = _selected_sym + '.NS'
        with st.spinner(f"🔄 Refreshing {_selected_sym}..."):
            try:
                _kite_now  = get_kite_client()
                _port_now  = load_portfolio()

                # Clear old cache for this stock so fresh data is fetched
                _old_ck = _cache_key(_refresh_sym, interval)
                if _old_ck in _DATA_CACHE:
                    del _DATA_CACHE[_old_ck]

                # Clear old chart cache
                for _k in [f'chart_fig_{_refresh_sym}_{interval}',
                            f'chart_interval_{_refresh_sym}']:
                    st.session_state.pop(_k, None)

                # Re-fetch fresh data
                _new_df, _new_src = fetch_intraday(_refresh_sym, interval, period, kite=_kite_now)

                if _new_df is not None:
                    _new_df = calculate_intraday_indicators(_new_df)
                    _new_df = generate_intraday_signals(_new_df)

                    _new_latest = _new_df.iloc[-1]
                    _new_prev   = _new_df.iloc[-2]
                    _sig_rows   = _new_df[_new_df['Signal'] != 0]
                    _new_last_sig = _sig_rows.iloc[-1] if len(_sig_rows) > 0 else None
                    _new_sig_val  = int(_new_last_sig['Signal']) if _new_last_sig is not None else 0
                    _new_bull  = int(_new_latest.get('Bull_Score', 0)) if not pd.isna(_new_latest.get('Bull_Score', 0)) else 0
                    _new_bear  = int(_new_latest.get('Bear_Score', 0)) if not pd.isna(_new_latest.get('Bear_Score', 0)) else 0
                    _new_conf  = min(int((_new_bull / 100) * 100), 100)
                    _new_price = float(_new_latest['Close']) if not pd.isna(_new_latest['Close']) else 0.0
                    _new_chg   = float(((_new_latest['Close']/_new_prev['Close'])-1)*100) \
                                 if not pd.isna(_new_prev['Close']) and _new_prev['Close'] != 0 else 0.0

                    _new_tp  = get_intraday_trade_plan(_new_df, capital, risk_pct)
                    _new_liq = compute_liquidity(_new_df, _new_price or 1, capital)
                    _wu, _nt, _mo, _pr = candle_warmup_status(_new_df, interval)

                    _new_r = {
                        'symbol':      _refresh_sym,
                        'data_src':    _new_src,
                        'df':          _new_df,
                        'latest':      _new_latest,
                        'prev':        _new_prev,
                        'last_signal': _new_last_sig,
                        'signal_val':  _new_sig_val,
                        'live_bull':   _new_bull,
                        'live_bear':   _new_bear,
                        'live_conf':   _new_conf,
                        'trade_plan':  _new_tp,
                        'liquidity':   _new_liq,
                        'warmup':      _wu,
                        'n_today':     _nt,
                        'mins_open':   _mo,
                        'pct_ready':   _pr,
                        'price':       _new_price,
                        'change_pct':  _new_chg,
                        'rsi':         float(_new_latest['RSI'])  if not pd.isna(_new_latest['RSI'])  else 50.0,
                        'adx':         float(_new_latest['ADX'])  if not pd.isna(_new_latest['ADX'])  else 0.0,
                        'macd':        float(_new_latest['MACD']) if not pd.isna(_new_latest['MACD']) else 0.0,
                        'supertrend':  int(_new_latest['Supertrend_Direction'])
                                       if not pd.isna(_new_latest['Supertrend_Direction']) else 0,
                        'ema_trend':   'BULL' if (not pd.isna(_new_latest['EMA_9']) and
                                       not pd.isna(_new_latest['EMA_21']) and
                                       _new_latest['EMA_9'] > _new_latest['EMA_21']) else 'BEAR',
                        'vwap':        'ABOVE' if (not pd.isna(_new_latest.get('VWAP', np.nan)) and
                                       _new_latest['Close'] > _new_latest['VWAP']) else 'BELOW',
                        'vol_ratio':   float(_new_latest['Volume_Ratio'])
                                       if not pd.isna(_new_latest['Volume_Ratio']) else 1.0,
                        'bb_pos':      'UPPER' if (not pd.isna(_new_latest['BB_Upper']) and
                                       _new_latest['Close'] >= _new_latest['BB_Upper']) else
                                       ('LOWER' if (not pd.isna(_new_latest['BB_Lower']) and
                                       _new_latest['Close'] <= _new_latest['BB_Lower']) else 'MID'),
                    }
                    _ps, _, _vrd  = compute_intraday_pick_score(_new_r)
                    _new_r['_pick_score'] = _ps
                    _new_r['_verdict']    = _vrd
                    _new_r['_alerts']     = evaluate_alerts(_new_r, _port_now)

                    # Update in scan_raw so result reflects fresh data
                    _raw = st.session_state.get('scan_raw', [])
                    for _i, _existing in enumerate(_raw):
                        if _existing['symbol'] == _refresh_sym:
                            _raw[_i] = _new_r
                            break
                    st.session_state['scan_raw'] = _raw

                    # Save refresh timestamp
                    st.session_state[f'stock_refresh_time_{_selected_sym}'] = \
                        ist_now().strftime('%H:%M:%S IST')

                    st.success(f"✅ {_selected_sym} refreshed @ ₹{_new_price:,.2f} · "
                               f"Score: {_ps}/100 · {_vrd} · "
                               f"{'⚡ Kite' if _new_src=='kite' else '⏳ yfinance'}")
                else:
                    st.warning(f"⚠️ Could not fetch fresh data for {_selected_sym}")

            except Exception as _e:
                st.error(f"Refresh failed: {_e}")

        st.rerun()

    # Find selected result (will use refreshed data if available)
    result = next((r for r in st.session_state.get('scan_raw', all_results)
                   if r['symbol'].replace('.NS','') == _selected_sym), None)

    if result:
        sym       = result['symbol']
        df        = result['df']
        latest    = result['latest']
        prev      = result['prev']
        ls        = result['last_signal']
        tp        = result['trade_plan']
        conf      = result['live_conf']
        grade, badge_cls = conf_label(conf)
        sig_val   = result['signal_val']
        chg       = result['change_pct']
        chg_color = "#16a34a" if chg >= 0 else "#dc2626"
        sym_clean = sym.replace('.NS', '')

    # ── Per-stock warmup warning ──────────────────────────
    _r_warmup  = result.get('warmup', 'READY')
    _r_ntoday  = result.get('n_today', 0)
    _r_pct     = result.get('pct_ready', 100)

    if _r_warmup == 'WARMING_UP':
        st.markdown(f"""
        <div style='background:#450a0a;border:2px solid #dc2626;border-radius:12px;
                    padding:12px 18px;margin-bottom:12px'>
            <div style='font-size:14px;font-weight:800;color:#fca5a5'>
                🚫 {sym_clean} — Indicators Not Ready ({_r_ntoday} candles today)
            </div>
            <div style='font-size:12px;color:#fca5a5;margin-top:4px'>
                Need minimum {MIN_CANDLES_HARD} candles. Score and verdict shown are <b>invalid</b>.
                Only Volume and VWAP are meaningful right now. Wait until 9:22 AM.
            </div>
        </div>""", unsafe_allow_html=True)

    elif _r_warmup == 'PARTIAL':
        st.markdown(f"""
        <div style='background:#451a03;border:1.5px solid #d97706;border-radius:12px;
                    padding:10px 16px;margin-bottom:12px;
                    display:flex;align-items:center;justify-content:space-between'>
            <div>
                <div style='font-size:13px;font-weight:700;color:#fbbf24'>
                    ⏳ {sym_clean} — Signals {_r_pct}% reliable ({_r_ntoday} candles today)
                </div>
                <div style='font-size:11px;color:#fde68a;margin-top:3px'>
                    RSI · EMA · Supertrend partially formed. Trust Volume + VWAP only until 9:35 AM.
                    Verdict capped at WATCH — BUY/STRONG BUY will show once fully warmed up.
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

    col_price, col_signal, col_conf = st.columns([2, 2, 3])

    # ── Price card ──
    with col_price:
        _vwap_val  = float(latest.get('VWAP', 0)) if not pd.isna(latest.get('VWAP', np.nan)) else None
        _vwap_str  = f"₹{_vwap_val:,.2f}" if _vwap_val else "—"
        _vwap_pos  = ("ABOVE" if _vwap_val and result['price'] > _vwap_val else "BELOW") if _vwap_val else "—"
        _vwap_cls  = "vwap-above" if _vwap_pos == "ABOVE" else "vwap-below"
        st.markdown(f"""
        <div style='background:#1a2035;border-radius:18px;padding:22px 26px;color:white;margin-bottom:8px'>
            <div style='font-size:11px;color:rgba(255,255,255,0.5);font-weight:700;letter-spacing:1.5px;text-transform:uppercase'>
                {sym_clean} · NSE · {interval_label.split("(")[0].strip()}
            </div>
            <div style='font-size:30px;font-weight:800;margin:6px 0;font-family:JetBrains Mono,monospace'>₹{result['price']:,.2f}</div>
            <div style='font-size:14px;font-weight:600;color:{chg_color}'>{'▲' if chg>=0 else '▼'} {abs(chg):.2f}% candle</div>
            <div style='margin-top:10px;display:flex;gap:8px;align-items:center'>
                <span style='font-size:11px;color:rgba(255,255,255,0.5)'>VWAP {_vwap_str}</span>
                <span class='{_vwap_cls}'>{_vwap_pos}</span>
            </div>
        </div>""", unsafe_allow_html=True)

        # ── Re-entry Warning ──────────────────────────────
        if tp:
            port        = load_portfolio()
            _today_str  = ist_now().strftime('%d %b %Y')
            _block_buy  = False   # default — overridden by re-entry checks below

            # Check 1 — Already have OPEN position in this stock
            _open_pos   = [p for p in port
                           if p.get('symbol') == sym_clean
                           and p.get('status') == 'OPEN']

            # Check 2 — Lost money on this stock TODAY
            _lost_today = [p for p in port
                           if p.get('symbol') == sym_clean
                           and p.get('status') != 'OPEN'
                           and p.get('exit_date','').startswith(_today_str)
                           and _f(p.get('net_pl', 0)) < 0]

            # Check 3 — Stop loss was hit on this stock TODAY
            _sl_hit_today = [p for p in port
                             if p.get('symbol') == sym_clean
                             and p.get('status') != 'OPEN'
                             and p.get('exit_date','').startswith(_today_str)
                             and 'Stop Loss' in p.get('exit_reason','')]

            # Show appropriate warning
            if _open_pos:
                _op = _open_pos[0]
                _op_entry  = _f(_op.get('entry', 0))
                _op_pl     = (_f(result['price']) - _op_entry) * int(_f(_op.get('qty', 0)))
                _op_pl_clr = '#16a34a' if _op_pl >= 0 else '#dc2626'
                st.markdown(
                    f"<div style='background:#fffbeb;border:1.5px solid #d97706;"
                    f"border-radius:10px;padding:10px 14px;margin-bottom:8px'>"
                    f"<div style='font-size:12px;font-weight:700;color:#92400e'>"
                    f"⚠️ Already have OPEN position in {sym_clean}</div>"
                    f"<div style='font-size:11px;color:#92400e;margin-top:3px'>"
                    f"Entry ₹{_op_entry:,.2f} · {int(_f(_op.get('qty',0)))} shares · "
                    f"P&L <b style='color:{_op_pl_clr}'>"
                    f"{'+' if _op_pl>=0 else ''}₹{_op_pl:,.0f}</b></div>"
                    f"</div>", unsafe_allow_html=True)
                _block_buy = False   # allow adding — just informing

            elif _sl_hit_today:
                _sl_trade  = _sl_hit_today[0]
                _sl_loss   = _f(_sl_trade.get('net_pl', 0))
                st.markdown(
                    f"<div style='background:#fef2f2;border:2px solid #dc2626;"
                    f"border-radius:10px;padding:10px 14px;margin-bottom:8px'>"
                    f"<div style='font-size:12px;font-weight:700;color:#991b1b'>"
                    f"🚫 STOP LOSS HIT today on {sym_clean}</div>"
                    f"<div style='font-size:11px;color:#991b1b;margin-top:3px'>"
                    f"Lost ₹{abs(_sl_loss):,.0f} · "
                    f"Re-entry after SL hit = revenge trade · "
                    f"<b>Strongly avoid</b></div>"
                    f"</div>", unsafe_allow_html=True)
                _reentry_confirmed = st.checkbox(
                    f"I understand the risk — re-enter {sym_clean} anyway",
                    key=f"reentry_confirm_sl_{sym_clean}")
                _block_buy = not _reentry_confirmed

            elif _lost_today:
                _lt        = _lost_today[0]
                _lt_loss   = _f(_lt.get('net_pl', 0))
                _lt_exit   = _lt.get('exit_date','')[-8:]
                st.markdown(
                    f"<div style='background:#fff5f5;border:1.5px solid #f87171;"
                    f"border-radius:10px;padding:10px 14px;margin-bottom:8px'>"
                    f"<div style='font-size:12px;font-weight:700;color:#dc2626'>"
                    f"⚠️ Previous loss on {sym_clean} today</div>"
                    f"<div style='font-size:11px;color:#dc2626;margin-top:3px'>"
                    f"Lost ₹{abs(_lt_loss):,.0f} at {_lt_exit} · "
                    f"Are you sure you want to re-enter?</div>"
                    f"</div>", unsafe_allow_html=True)
                _reentry_ok = st.checkbox(
                    f"Yes, I have a new reason to enter {sym_clean}",
                    key=f"reentry_confirm_loss_{sym_clean}")
                _block_buy = not _reentry_ok

            else:
                _block_buy = False

            # ── Paper Buy Button ───────────────────────────
            if not _block_buy:
                buy_label = f"📥 Paper Buy  {sym_clean}  @₹{result['price']:,.2f}"
                if st.button(buy_label, key=f"paper_buy_{sym_clean}", use_container_width=True):
                    already = any(
                        p.get('symbol') == sym_clean and
                        _f(p.get('entry')) == round(result['price'], 2)
                        for p in port
                    )
                    if not already:
                        port.append({
                            'symbol':      sym_clean,
                            'entry':       round(result['price'], 2),
                            'qty':         tp['qty'],
                            'stop_loss':   tp['stop_loss'],
                            't1':          tp['t1'], 't2': tp['t2'],
                            't3':          tp['t3'], 't4': tp['t4'],
                            'investment':  tp['investment'],
                            'actual_cost': tp['actual_cost'],
                            'charges':     tp['buy_charges']['total'],
                            'verdict':     result.get('_verdict', ''),
                            'pick_score':  result.get('_pick_score', 0),
                            'date':        ist_now().strftime('%d %b %Y %H:%M'),
                            'timeframe':   interval_label,
                            'status':      'OPEN',
                            'exit_price':  None,
                            'net_pl':      None,
                            'trade_type':  'INTRADAY',
                        })
                        save_portfolio(port)
                        st.session_state['paper_portfolio'] = port
                        st.success(f"✅ Added {sym_clean} · {tp['qty']} shares @ ₹{result['price']:,.2f}")
                    else:
                        st.info(f"ℹ️ {sym_clean} @ ₹{result['price']:,.2f} already in book.")

    with col_signal:
        if sig_val == 1:
            sig_html = f"<div class='signal-buy'><div style='font-size:20px;font-weight:700;color:#15803d'>🟢 BUY SIGNAL</div><div style='color:#64748b;font-size:12px;margin-top:4px'>Confidence: {ls['Confidence'] if ls is not None else 0}%</div></div>"
        elif sig_val == -1:
            sig_html = f"<div class='signal-sell'><div style='font-size:20px;font-weight:700;color:#dc2626'>🔴 SELL/SHORT</div><div style='color:#64748b;font-size:12px;margin-top:4px'>Confidence: {ls['Confidence'] if ls is not None else 0}%</div></div>"
        else:
            sig_html = "<div class='signal-none'><div style='font-size:20px;font-weight:700;color:#64748b'>⚪ NO SIGNAL</div><div style='color:#64748b;font-size:12px;margin-top:4px'>Watching for setup</div></div>"
        st.markdown(sig_html, unsafe_allow_html=True)

    with col_conf:
        bar_width  = int(conf)
        fill_color = conf_color(conf)
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Live Confidence</div>
            <div style='display:flex;align-items:center;gap:12px;margin-top:8px'>
                <div style='font-size:28px;font-weight:700;color:{fill_color}'>{conf}%</div>
                <span class='score-badge {badge_cls}'>{grade}</span>
            </div>
            <div class='conf-bar-bg' style='margin-top:8px'>
                <div class='conf-bar-fill' style='width:{bar_width}%;background:{fill_color}'></div>
            </div>
            <div style='display:flex;justify-content:space-between;margin-top:4px'>
                <span style='color:#64748b;font-size:11px'>Bull: {result['live_bull']} pts</span>
                <span style='color:#64748b;font-size:11px'>Bear: {result['live_bear']} pts</span>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Per-stock Alert Cards ──────────────────────────
    _stock_alerts = result.get('_alerts', [])
    if _stock_alerts:
        _alert_bg_map = {
            'STRONG_BUY': ('#dcfce7','#14532d','🚨 STRONG BUY ALERT'),
            'BUY':        ('#f0fdf4','#15803d','🔔 BUY ALERT'),
            'VOL_SURGE':  ('#fffbeb','#92400e','⚡ VOLUME SURGE'),
            'VWAP_BREAK': ('#fff5f5','#991b1b','⚠️ VWAP BREAKDOWN'),
            'RSI_OB':     ('#fff5f5','#991b1b','🔴 RSI OVERBOUGHT'),
            'STOP_LOSS':  ('#fef2f2','#7f1d1d','🛑 STOP LOSS HIT'),
            'TARGET_T1':  ('#f0fdf4','#15803d','🎯 T1 TARGET HIT'),
            'TARGET_T2':  ('#f0fdf4','#15803d','🎯 T2 TARGET HIT'),
            'TARGET_T3':  ('#ecfdf5','#065f46','🎯 T3 TARGET HIT'),
            'TARGET_T4':  ('#ecfdf5','#065f46','🎯 T4 TARGET HIT'),
            'TIME_WARN':  ('#fffbeb','#92400e','🕒 TIME WARNING'),
        }
        for _sa in _stock_alerts:
            _abg, _atc, _atitle = _alert_bg_map.get(_sa['type'], ('#f8fafc','#1a2035','📣 ALERT'))
            # Determine what action to take
            _action = {
                'STRONG_BUY': '✅ All criteria aligned — enter position now',
                'BUY':        '✅ Good setup — enter with normal size',
                'VOL_SURGE':  '👀 Monitor direction — wait for VWAP confirmation before entry',
                'VWAP_BREAK': '🚪 EXIT — price lost VWAP support, bears in control',
                'RSI_OB':     '📤 Book 50% — RSI-7 overbought, partial exit recommended',
                'STOP_LOSS':  '🚨 EXIT ALL — stop loss breached, no waiting',
                'TARGET_T1':  '📤 Book 50% quantity at T1 Scalp target',
                'TARGET_T2':  '📤 Book 30% quantity at T2 Target',
                'TARGET_T3':  '📤 Book 20% quantity at T3 Extended target',
                'TARGET_T4':  '📤 Book remaining at T4 Stretch target',
                'TIME_WARN':  '🕒 Start exiting — only 15 minutes to market close',
            }.get(_sa['type'], 'Review position')
            st.markdown(f"""
            <div style='background:{_abg};border:2px solid {_atc}44;border-radius:14px;
                        padding:14px 20px;margin-bottom:8px'>
                <div style='display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px'>
                    <div style='font-size:15px;font-weight:800;color:{_atc}'>{_atitle}</div>
                    <div style='font-size:11px;color:#94a3b8'>{_sa.get("time","")}</div>
                </div>
                <div style='font-size:13px;color:#374151;margin-top:6px'>{_sa["msg"]}</div>
                <div style='font-size:12px;font-weight:700;color:{_atc};margin-top:8px;
                            background:white;border-radius:8px;padding:6px 12px;display:inline-block'>
                    ➤ ACTION: {_action}
                </div>
            </div>""", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>📊 Key Intraday Indicators</div>", unsafe_allow_html=True)
    i1, i2, i3, i4, i5, i6 = st.columns(6)
    rsi_v   = float(latest['RSI'])    if not pd.isna(latest['RSI'])    else 50.0
    rsi_col = "#ef4444" if rsi_v>70 else ("#22c55e" if rsi_v<30 else "#f59e0b")
    rsi_lbl = "Overbought" if rsi_v>70 else ("Oversold" if rsi_v<30 else "Neutral")

    with i1:
        st.markdown(f"""<div class='stat-card'>
            <div class='stat-card-icon' style='background:#f5f0ff'>
                <svg xmlns='http://www.w3.org/2000/svg' width='20' height='20' fill='none' viewBox='0 0 24 24' stroke='#7c3aed' stroke-width='2'><polyline points='23 6 13.5 15.5 8.5 10.5 1 18'/></svg>
            </div>
            <div class='stat-label'>RSI-7</div>
            <div class='stat-value' style='color:{rsi_col};font-size:22px'>{rsi_v:.1f}</div>
            <div class='stat-sub'>{rsi_lbl}</div>
        </div>""", unsafe_allow_html=True)
    with i2:
        macd_v   = float(latest['MACD'])        if not pd.isna(latest['MACD'])        else 0.0
        macd_col = "#16a34a" if macd_v>0 else "#dc2626"
        st.markdown(f"""<div class='stat-card'>
            <div class='stat-card-icon' style='background:#f0fdf4'>
                <svg xmlns='http://www.w3.org/2000/svg' width='20' height='20' fill='none' viewBox='0 0 24 24' stroke='#15803d' stroke-width='2'><polyline points='23 18 13.5 8.5 8.5 13.5 1 6'/></svg>
            </div>
            <div class='stat-label'>MACD 5/13</div>
            <div class='stat-value' style='color:{macd_col};font-size:22px'>{macd_v:.2f}</div>
            <div class='stat-sub'>Sig: {float(latest["MACD_Signal"]):.2f}</div>
        </div>""", unsafe_allow_html=True)
    with i3:
        adx_v   = float(latest['ADX']) if not pd.isna(latest['ADX']) else 0.0
        adx_col = "#15803d" if adx_v>25 else "#94a3b8"
        st.markdown(f"""<div class='stat-card'>
            <div class='stat-card-icon' style='background:#fff7ed'>
                <svg xmlns='http://www.w3.org/2000/svg' width='20' height='20' fill='none' viewBox='0 0 24 24' stroke='#ea580c' stroke-width='2'><path d='M22 12h-4l-3 9L9 3l-3 9H2'/></svg>
            </div>
            <div class='stat-label'>ADX-7</div>
            <div class='stat-value' style='color:{adx_col};font-size:22px'>{adx_v:.1f}</div>
            <div class='stat-sub'>{"Strong" if adx_v>25 else ("Moderate" if adx_v>20 else "Weak")}</div>
        </div>""", unsafe_allow_html=True)
    with i4:
        vr_v   = float(latest['Volume_Ratio']) if not pd.isna(latest['Volume_Ratio']) else 1.0
        vr_col = "#16a34a" if vr_v>=2 else ("#f59e0b" if vr_v>=1.5 else "#94a3b8")
        vr_lbl = "🔥 Surge" if vr_v>=3 else ("High" if vr_v>=2 else ("Above Avg" if vr_v>=1.5 else "Normal"))
        st.markdown(f"""<div class='stat-card'>
            <div class='stat-card-icon' style='background:#fef9c3'>
                <svg xmlns='http://www.w3.org/2000/svg' width='20' height='20' fill='none' viewBox='0 0 24 24' stroke='#d97706' stroke-width='2'><polyline points='22 7 13.5 15.5 8.5 10.5 2 17'/><polyline points='16 7 22 7 22 13'/></svg>
            </div>
            <div class='stat-label'>Volume Ratio</div>
            <div class='stat-value' style='color:{vr_col};font-size:22px'>{vr_v:.1f}×</div>
            <div class='stat-sub'>{vr_lbl}</div>
        </div>""", unsafe_allow_html=True)
    with i5:
        vwap_v  = float(latest.get('VWAP', 0)) if not pd.isna(latest.get('VWAP', np.nan)) else None
        vwap_col= "#16a34a" if (vwap_v and result['price'] > vwap_v) else "#dc2626"
        vwap_lbl= "Above" if (vwap_v and result['price'] > vwap_v) else "Below"
        st.markdown(f"""<div class='stat-card'>
            <div class='stat-card-icon' style='background:#fef9c3'>
                <svg xmlns='http://www.w3.org/2000/svg' width='20' height='20' fill='none' viewBox='0 0 24 24' stroke='#f59e0b' stroke-width='2'><line x1='12' y1='1' x2='12' y2='23'/><path d='M17 5H9.5a3.5 3.5 0 1 0 0 7h5a3.5 3.5 0 1 1 0 7H6'/></svg>
            </div>
            <div class='stat-label'>VWAP</div>
            <div class='stat-value' style='color:{vwap_col};font-size:22px'>{f"₹{vwap_v:,.0f}" if vwap_v else "—"}</div>
            <div class='stat-sub'>{vwap_lbl}</div>
        </div>""", unsafe_allow_html=True)
    with i6:
        stk_v   = float(latest['Stoch_K']) if not pd.isna(latest['Stoch_K']) else 50.0
        stk_col = "#ef4444" if stk_v>80 else ("#22c55e" if stk_v<20 else "#64748b")
        st.markdown(f"""<div class='stat-card'>
            <div class='stat-card-icon' style='background:#f0f4ff'>
                <svg xmlns='http://www.w3.org/2000/svg' width='20' height='20' fill='none' viewBox='0 0 24 24' stroke='#1d4ed8' stroke-width='2'><circle cx='12' cy='12' r='10'/><polyline points='8 12 12 8 16 12'/><line x1='12' y1='16' x2='12' y2='8'/></svg>
            </div>
            <div class='stat-label'>Stoch %K</div>
            <div class='stat-value' style='color:{stk_col};font-size:22px'>{stk_v:.0f}</div>
            <div class='stat-sub'>{"Overbought" if stk_v>80 else ("Oversold" if stk_v<20 else "Neutral")}</div>
        </div>""", unsafe_allow_html=True)

    # ── Intraday Trade Plan ──
    # ── Liquidity Section ──────────────────────────────
    st.markdown("<div class='section-header'>💧 Liquidity Analysis</div>", unsafe_allow_html=True)
    liq = result.get('liquidity', {})
    _lg = liq.get('grade', 'UNKNOWN')
    _lt = liq.get('tradeable', True)
    _lw = liq.get('warnings', [])

    # Grade color
    _lg_cfg = {
        'EXCELLENT': ('#052e16','#4ade80','#16a34a','✅ EXCELLENT'),
        'HIGH':      ('#064e3b','#34d399','#059669','✅ HIGH'),
        'MEDIUM':    ('#451a03','#fbbf24','#d97706','⚠️ MEDIUM'),
        'LOW':       ('#450a0a','#fca5a5','#dc2626','❌ LOW'),
        'UNKNOWN':   ('#1f2937','#9ca3af','#6b7280','— UNKNOWN'),
    }
    _lbg, _ltc, _lbc, _llbl = _lg_cfg.get(_lg, _lg_cfg['UNKNOWN'])

    # Tradeable warning banner
    if not _lt:
        st.markdown(f"""
        <div style='background:#450a0a;border:2px solid #dc2626;border-radius:12px;
                    padding:12px 18px;margin-bottom:10px'>
            <div style='font-size:15px;font-weight:800;color:#fca5a5'>
                🚫 DO NOT TRADE — Insufficient Liquidity
            </div>
            {''.join(f"<div style='font-size:12px;color:#fca5a5;margin-top:4px'>{w}</div>" for w in _lw)}
        </div>""", unsafe_allow_html=True)
    elif _lw:
        for _lw_item in _lw:
            st.markdown(f"""
            <div style='background:#451a03;border:1px solid #d97706;border-radius:10px;
                        padding:8px 14px;margin-bottom:6px;font-size:12px;color:#fbbf24'>
                {_lw_item}
            </div>""", unsafe_allow_html=True)

    # 6 metric cards
    lc1, lc2, lc3, lc4, lc5, lc6 = st.columns(6)

    # 1. Grade
    with lc1:
        st.markdown(f"""<div style='background:{_lbg};border:1px solid {_lbc}44;
            border-radius:14px;padding:16px 14px;text-align:center'>
            <div style='font-size:10px;font-weight:700;color:{_ltc};letter-spacing:1px;text-transform:uppercase'>Liquidity</div>
            <div style='font-size:20px;font-weight:900;color:{_ltc};margin:8px 0'>{_llbl}</div>
            <div style='font-size:10px;color:{_ltc};opacity:0.7'>{"Trade ✅" if _lt else "Avoid ❌"}</div>
        </div>""", unsafe_allow_html=True)

    # 2. Avg Daily Turnover
    with lc2:
        _to  = liq.get('avg_turnover', 0)
        _tos = _fmt_turnover(_to)
        _to_col = "#16a34a" if _to >= 1_00_00_000 else ("#d97706" if _to >= 10_00_000 else "#dc2626")
        st.markdown(f"""<div style='background:#ffffff;border:1px solid #e8ecf3;
            border-radius:14px;padding:16px 14px;text-align:center'>
            <div style='font-size:10px;font-weight:700;color:#94a3b8;letter-spacing:1px;text-transform:uppercase'>Daily Turnover</div>
            <div style='font-size:18px;font-weight:800;color:{_to_col};margin:8px 0;font-family:JetBrains Mono'>{_tos}</div>
            <div style='font-size:10px;color:#94a3b8'>Est. ₹/day</div>
        </div>""", unsafe_allow_html=True)

    # 3. Avg Daily Volume
    with lc3:
        _dv  = liq.get('avg_daily_vol', 0)
        _dvs = f"{_dv/1_00_000:.1f}L" if _dv >= 1_00_000 else (f"{_dv/1_000:.1f}K" if _dv >= 1_000 else str(_dv))
        _dv_col = "#16a34a" if _dv >= 10_00_000 else ("#d97706" if _dv >= 1_00_000 else "#dc2626")
        st.markdown(f"""<div style='background:#ffffff;border:1px solid #e8ecf3;
            border-radius:14px;padding:16px 14px;text-align:center'>
            <div style='font-size:10px;font-weight:700;color:#94a3b8;letter-spacing:1px;text-transform:uppercase'>Daily Volume</div>
            <div style='font-size:18px;font-weight:800;color:{_dv_col};margin:8px 0;font-family:JetBrains Mono'>{_dvs}</div>
            <div style='font-size:10px;color:#94a3b8'>Shares/day est.</div>
        </div>""", unsafe_allow_html=True)

    # 4. Volume Consistency
    with lc4:
        _vc  = liq.get('consistency_pct', 0)
        _vc_col = "#16a34a" if _vc >= 70 else ("#d97706" if _vc >= 40 else "#dc2626")
        _vc_lbl = "Consistent" if _vc >= 70 else ("Moderate" if _vc >= 40 else "Erratic")
        st.markdown(f"""<div style='background:#ffffff;border:1px solid #e8ecf3;
            border-radius:14px;padding:16px 14px;text-align:center'>
            <div style='font-size:10px;font-weight:700;color:#94a3b8;letter-spacing:1px;text-transform:uppercase'>Vol Consistency</div>
            <div style='font-size:18px;font-weight:800;color:{_vc_col};margin:8px 0;font-family:JetBrains Mono'>{_vc}%</div>
            <div style='font-size:10px;color:#94a3b8'>{_vc_lbl}</div>
        </div>""", unsafe_allow_html=True)

    # 5. Slippage Risk
    with lc5:
        _sp  = liq.get('slippage', 'UNKNOWN')
        _atr_pct = liq.get('atr_pct', 0)
        _sp_col = {"VERY LOW":"#16a34a","LOW":"#16a34a","MEDIUM":"#d97706",
                   "HIGH":"#dc2626","VERY HIGH":"#7f1d1d"}.get(_sp,"#94a3b8")
        _sp_bg  = {"VERY LOW":"#f0fdf4","LOW":"#f0fdf4","MEDIUM":"#fffbeb",
                   "HIGH":"#fff5f5","VERY HIGH":"#fef2f2"}.get(_sp,"#f8fafc")
        st.markdown(f"""<div style='background:{_sp_bg};border:1px solid {_sp_col}33;
            border-radius:14px;padding:16px 14px;text-align:center'>
            <div style='font-size:10px;font-weight:700;color:#94a3b8;letter-spacing:1px;text-transform:uppercase'>Slippage Risk</div>
            <div style='font-size:16px;font-weight:800;color:{_sp_col};margin:8px 0'>{_sp}</div>
            <div style='font-size:10px;color:#94a3b8'>ATR {_atr_pct:.2f}% of price</div>
        </div>""", unsafe_allow_html=True)

    # 6. Position Fill
    with lc6:
        _pf  = liq.get('pos_liquidity', 'UNKNOWN')
        _pfr = liq.get('pos_fill_ratio', 0)
        _pf_col = {"EASY":"#16a34a","GOOD":"#16a34a","MODERATE":"#d97706",
                   "TIGHT":"#dc2626","ILLIQUID":"#7f1d1d"}.get(_pf,"#94a3b8")
        _pf_bg  = {"EASY":"#f0fdf4","GOOD":"#f0fdf4","MODERATE":"#fffbeb",
                   "TIGHT":"#fff5f5","ILLIQUID":"#fef2f2"}.get(_pf,"#f8fafc")
        st.markdown(f"""<div style='background:{_pf_bg};border:1px solid {_pf_col}33;
            border-radius:14px;padding:16px 14px;text-align:center'>
            <div style='font-size:10px;font-weight:700;color:#94a3b8;letter-spacing:1px;text-transform:uppercase'>Position Fill</div>
            <div style='font-size:16px;font-weight:800;color:{_pf_col};margin:8px 0'>{_pf}</div>
            <div style='font-size:10px;color:#94a3b8'>{_pfr:.1f}× your capital</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>🎯 Intraday Trade Plan</div>", unsafe_allow_html=True)
    if tp:
        tp_c1, tp_c2 = st.columns([3, 2])
        with tp_c1:
            entry_px = tp['entry']; sl_px = tp['stop_loss']
            rr_color = "#22c55e"
            for lbl, price, bg, bdr, rr in [
                ("Entry",           entry_px,  "#f8fafc","#e2e8f0","—"),
                ("Stop Loss",       sl_px,     "#fff5f5","#fecaca",f"-{abs(round((sl_px-entry_px)/entry_px*100,2))}%"),
                ("T1 — Scalp 0.5R", tp['t1'],  "#f0fdf4","#bbf7d0","+{:.2f}%".format((tp['t1']-entry_px)/entry_px*100)),
                ("T2 — Target 1R",  tp['t2'],  "#f0fdf4","#bbf7d0","+{:.2f}%".format((tp['t2']-entry_px)/entry_px*100)),
                ("T3 — Extend 1.5R",tp['t3'],  "#ecfdf5","#6ee7b7","+{:.2f}%".format((tp['t3']-entry_px)/entry_px*100)),
                ("T4 — Stretch 2R", tp['t4'],  "#f0fdf4","#86efac","+{:.2f}%".format((tp['t4']-entry_px)/entry_px*100)),
            ]:
                _col = "#dc2626" if "Stop" in lbl else ("#22c55e" if "T" in lbl else "#1a2035")
                st.markdown(f"""
                <div style='background:{bg};border:1px solid {bdr};border-radius:10px;
                            padding:10px 16px;margin-bottom:6px;
                            display:flex;align-items:center;justify-content:space-between'>
                    <span style='font-size:13px;font-weight:600;color:#4a5568'>{lbl}</span>
                    <span style='font-size:17px;font-weight:800;color:{_col};font-family:JetBrains Mono,monospace'>₹{price:,.2f}</span>
                    <span style='font-size:12px;font-weight:700;color:{_col}'>{rr}</span>
                </div>""", unsafe_allow_html=True)

        with tp_c2:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Position Size</div>
                <div style='font-size:26px;font-weight:800;color:#1a2035;font-family:JetBrains Mono,monospace'>{tp['qty']} shares</div>
                <div style='font-size:12px;color:#94a3b8;margin-top:4px'>@ ₹{entry_px:,.2f} entry</div>
                <hr style='margin:10px 0;border-color:#e8ecf3'>
                <div style='display:flex;justify-content:space-between;margin-bottom:6px'>
                    <span style='font-size:12px;color:#64748b'>Exposure</span>
                    <span style='font-size:13px;font-weight:700;color:#1a2035'>₹{tp["investment"]:,.0f}</span>
                </div>
                <div style='display:flex;justify-content:space-between;margin-bottom:6px'>
                    <span style='font-size:12px;color:#64748b'>Max Risk</span>
                    <span style='font-size:13px;font-weight:700;color:#dc2626'>₹{tp["risk_amount"]:,.0f}</span>
                </div>
                <div style='display:flex;justify-content:space-between;margin-bottom:6px'>
                    <span style='font-size:12px;color:#64748b'>ATR-7</span>
                    <span style='font-size:13px;font-weight:700;color:#1a2035'>₹{tp.get("atr",0):,.2f}</span>
                </div>
                <div style='display:flex;justify-content:space-between'>
                    <span style='font-size:12px;color:#64748b'>Buy Charges</span>
                    <span style='font-size:13px;font-weight:700;color:#1a2035'>₹{tp["buy_charges"]["total"]:,.2f}</span>
                </div>
                <div style='background:#fff7ed;border:1px solid #fed7aa;border-radius:8px;padding:8px 12px;margin-top:10px'>
                    <div style='font-size:10px;font-weight:700;color:#ea580c;letter-spacing:1px'>INTRADAY NOTE</div>
                    <div style='font-size:11px;color:#9a3412;margin-top:3px'>STT charged only on sell side (0.025%). No STCG — taxed as speculative income.</div>
                </div>
            </div>""", unsafe_allow_html=True)

        # P&L table
        st.markdown("<div class='section-header'>📊 P&L After All Charges (Intraday STT)</div>", unsafe_allow_html=True)
        pl_df = pd.DataFrame(tp['pl_table'])
        pl_df['Sell Value']   = pl_df['Sell Value'].apply(lambda x: f"₹{x:,.2f}")
        pl_df['Sell Charges'] = pl_df['Sell Charges'].apply(lambda x: f"₹{x:,.2f}")
        pl_df['Gross P&L']    = pl_df['Gross P&L'].apply(lambda x: f"₹{x:,.2f}")
        pl_df['Net P&L']      = pl_df['Net P&L'].apply(lambda x: f"₹{x:,.2f}")
        pl_df['Return%']      = pl_df['Return%'].apply(lambda x: f"{x:+.2f}%")
        st.dataframe(pl_df, use_container_width=True, hide_index=True)

    # ── Market Context Panel (VIX + Nifty + RS) ───────────
    _mkt_ctx2   = st.session_state.get('nifty_context', {})
    _vix2       = _mkt_ctx2.get('vix')
    _vix_level2 = _mkt_ctx2.get('vix_level', 'UNKNOWN')
    _rs2        = result.get('rs_vs_nifty')
    _nifty_chg2 = _mkt_ctx2.get('nifty_chg', 0)

    if _vix2 or _rs2 is not None:
        st.markdown("<div class='section-header'>🌐 Market Context</div>", unsafe_allow_html=True)
        _mx1, _mx2, _mx3 = st.columns(3)

        _vix_clr = {
            'CALM':    '#16a34a',
            'NORMAL':  '#16a34a',
            'ELEVATED':'#d97706',
            'HIGH':    '#ea580c',
            'EXTREME': '#dc2626',
            'CRISIS':  '#7f1d1d',
        }.get(_vix_level2, '#64748b')
        with _mx1:
            st.markdown(
                f"<div style='background:{_vix_clr}22;border:1px solid {_vix_clr}44;"
                f"border-radius:10px;padding:12px 14px'>"
                f"<div style='font-size:10px;font-weight:700;color:{_vix_clr};"
                f"letter-spacing:1px'>INDIA VIX</div>"
                f"<div style='font-size:26px;font-weight:800;color:{_vix_clr};"
                f"font-family:JetBrains Mono;margin:4px 0'>"
                f"{f'{_vix2:.2f}' if _vix2 else '—'}</div>"
                f"<div style='font-size:11px;color:{_vix_clr}'>{_vix_level2}</div>"
                f"</div>", unsafe_allow_html=True)

        _n_clr = '#16a34a' if _nifty_chg2 >= 0 else '#dc2626'
        with _mx2:
            _ns = st.session_state.get('nifty_market_state','—')
            st.markdown(
                f"<div style='background:{_n_clr}22;border:1px solid {_n_clr}44;"
                f"border-radius:10px;padding:12px 14px'>"
                f"<div style='font-size:10px;font-weight:700;color:{_n_clr};"
                f"letter-spacing:1px'>NIFTY 50</div>"
                f"<div style='font-size:22px;font-weight:800;color:{_n_clr};"
                f"font-family:JetBrains Mono;margin:4px 0'>"
                f"{'+' if _nifty_chg2>=0 else ''}{_nifty_chg2:.2f}%</div>"
                f"<div style='font-size:11px;color:{_n_clr}'>{_ns}</div>"
                f"</div>", unsafe_allow_html=True)

        if _rs2 is not None:
            _rs_clr = ('#16a34a' if _rs2 >= 1.5 else ('#d97706' if _rs2 >= -0.5 else '#dc2626'))
            _rs_lbl = ('🔥 Strongly Outperforming' if _rs2 >= 3.0 else
                       ('✅ Outperforming' if _rs2 >= 1.5 else
                        ('↔ In line' if _rs2 >= -0.5 else
                         ('⚠️ Underperforming' if _rs2 >= -1.5 else '🔴 Strongly Underperforming'))))
            with _mx3:
                st.markdown(
                    f"<div style='background:{_rs_clr}22;border:1px solid {_rs_clr}44;"
                    f"border-radius:10px;padding:12px 14px'>"
                    f"<div style='font-size:10px;font-weight:700;color:{_rs_clr};"
                    f"letter-spacing:1px'>RELATIVE STRENGTH</div>"
                    f"<div style='font-size:22px;font-weight:800;color:{_rs_clr};"
                    f"font-family:JetBrains Mono;margin:4px 0'>"
                    f"{'+' if _rs2>=0 else ''}{_rs2:.2f}%</div>"
                    f"<div style='font-size:11px;color:{_rs_clr}'>{_rs_lbl}</div>"
                    f"</div>", unsafe_allow_html=True)

    # ── Multi-Timeframe Panel ──────────────────────────────
    st.markdown("<div class='section-header'>⏱ Multi-Timeframe Confirmation</div>",
                unsafe_allow_html=True)

    _mtf_key = f"mtf_{sym}_{interval}"
    if _mtf_key not in st.session_state or st.button(
            "🔄 Fetch MTF", key=f"mtf_refresh_{sym_clean}", help="Re-fetch all 3 timeframes"):
        with st.spinner("Fetching 1min / 5min / 15min data..."):
            _kite_mtf = get_kite_client()
            _mtf_data = fetch_multi_timeframe(sym, kite=_kite_mtf)
            st.session_state[_mtf_key] = _mtf_data
    else:
        _mtf_data = st.session_state[_mtf_key]

    _tf_cols = st.columns(4)
    _align   = _mtf_data.get('alignment', 'UNKNOWN')
    _mtf_sc  = _mtf_data.get('mtf_score', 0)
    _align_colors = {
        'STRONG_BULL': '#15803d', 'BULL': '#16a34a', 'WEAK_BULL': '#65a30d',
        'CONFLICTING': '#d97706', 'WEAK_BEAR': '#ea580c',
        'BEAR': '#dc2626',        'STRONG_BEAR': '#991b1b', 'UNKNOWN': '#64748b',
    }
    _align_bg = {
        'STRONG_BULL': '#f0fdf4', 'BULL': '#f0fdf4', 'WEAK_BULL': '#f7fee7',
        'CONFLICTING': '#fffbeb', 'WEAK_BEAR': '#fff7ed',
        'BEAR': '#fff5f5',        'STRONG_BEAR': '#fff5f5', 'UNKNOWN': '#f8fafc',
    }
    _ac = _align_colors.get(_align, '#64748b')
    _ab = _align_bg.get(_align, '#f8fafc')

    with _tf_cols[0]:
        st.markdown(
            f"<div style='background:{_ab};border:1.5px solid {_ac}44;"
            f"border-radius:10px;padding:12px 14px;text-align:center'>"
            f"<div style='font-size:10px;font-weight:700;color:{_ac};"
            f"letter-spacing:1px'>ALIGNMENT</div>"
            f"<div style='font-size:14px;font-weight:800;color:{_ac};margin:4px 0'>"
            f"{_align.replace('_',' ')}</div>"
            f"<div style='font-size:13px;font-weight:800;color:{_ac}'>"
            f"Score: {'+' if _mtf_sc>=0 else ''}{_mtf_sc}</div>"
            f"</div>", unsafe_allow_html=True)

    for _tfi, _tf_lbl in enumerate(['1m','5m','15m'], 1):
        _tf_data = _mtf_data.get(_tf_lbl, {})
        if not _tf_data:
            with _tf_cols[_tfi]:
                st.markdown(
                    "<div style='background:#f8fafc;border:1px solid #e2e8f0;"
                    "border-radius:10px;padding:12px 14px;text-align:center;color:#94a3b8'>"
                    f"<div style='font-size:12px;font-weight:700'>{_tf_lbl}</div>"
                    "<div style='font-size:11px;margin-top:4px'>No data</div></div>",
                    unsafe_allow_html=True)
            continue
        _t = _tf_data['trend']
        _tc = '#16a34a' if _t == 'BULL' else ('#dc2626' if _t == 'BEAR' else '#d97706')
        _tb = '#f0fdf4' if _t == 'BULL' else ('#fff5f5' if _t == 'BEAR' else '#fffbeb')
        _ti = '🟢' if _t == 'BULL' else ('🔴' if _t == 'BEAR' else '🟡')
        with _tf_cols[_tfi]:
            st.markdown(
                f"<div style='background:{_tb};border:1.5px solid {_tc}44;"
                f"border-radius:10px;padding:12px 14px;text-align:center'>"
                f"<div style='font-size:13px;font-weight:700;color:#64748b'>{_tf_lbl}</div>"
                f"<div style='font-size:16px;font-weight:800;color:{_tc};margin:4px 0'>"
                f"{_ti} {_t}</div>"
                f"<div style='font-size:10px;color:{_tc}'>"
                f"RSI {_tf_data['rsi']:.0f} · EMA {('✅' if _tf_data['ema9']>_tf_data['ema21'] else '❌')}</div>"
                f"</div>", unsafe_allow_html=True)

    # ── MTF advice ────────────────────────────────────────
    _mtf_advice = {
        'STRONG_BULL': '✅ All 3 timeframes bullish — highest confidence BUY. Enter now.',
        'BULL':        '✅ 2 timeframes bullish — good confidence. Enter with normal position.',
        'WEAK_BULL':   '⚠️ Only 1 timeframe bullish — low confidence. Wait for 5min to confirm.',
        'CONFLICTING': '⚠️ Timeframes conflicting — skip this trade. Wait for alignment.',
        'WEAK_BEAR':   '🔴 1 timeframe bearish — avoid long entry.',
        'BEAR':        '🔴 2 timeframes bearish — do not enter long.',
        'STRONG_BEAR': '🚫 All 3 timeframes bearish — strong sell pressure. Avoid entirely.',
    }
    if _align in _mtf_advice:
        _adv_c = _align_colors.get(_align, '#64748b')
        _adv_b = _align_bg.get(_align, '#f8fafc')
        st.markdown(
            f"<div style='background:{_adv_b};border-left:4px solid {_adv_c};"
            f"border-radius:0 8px 8px 0;padding:10px 14px;margin:8px 0;"
            f"font-size:12px;color:{_adv_c}'>"
            f"⏱ <b>MTF Signal:</b> {_mtf_advice[_align]}</div>",
            unsafe_allow_html=True)

    # ── Previous Day High / Low Panel ─────────────────────
    _pdh2 = result.get('pdh')
    _pdl2 = result.get('pdl')
    _pr2  = result.get('price', 0)
    if _pdh2 and _pdl2 and _pr2 > 0:
        _d_pdh  = (_pdh2 - _pr2) / _pr2 * 100
        _d_pdl  = (_pr2 - _pdl2) / _pr2 * 100
        _pdh_clr = ('#dc2626' if _d_pdh < 0.3 else ('#d97706' if _d_pdh < 1.0 else '#16a34a'))
        _pdh_lbl = ('🚫 At/above PDH — strong resistance' if _d_pdh < 0
                    else '⚠️ Right at PDH — heavy resistance' if _d_pdh < 0.3
                    else '⚠️ Near PDH — caution' if _d_pdh < 1.0
                    else '✅ Room to run before PDH')
        _pdl_clr = ('#dc2626' if _d_pdl < 0 else ('#d97706' if _d_pdl < 0.5 else '#16a34a'))
        _pdl_lbl = ('🚫 Below PDL — broke support' if _d_pdl < 0
                    else '⚠️ Near PDL — weak base' if _d_pdl < 0.5
                    else '✅ Holding above PDL')

        st.markdown("<div class='section-header'>📏 Previous Day Levels</div>",
                    unsafe_allow_html=True)
        _pc1, _pc2, _pc3 = st.columns(3)
        with _pc1:
            st.markdown(
                f"<div style='background:{_pdh_clr}22;border:1px solid {_pdh_clr}44;"
                f"border-radius:10px;padding:12px 14px'>"
                f"<div style='font-size:10px;font-weight:700;color:{_pdh_clr};"
                f"letter-spacing:1px'>PREV DAY HIGH</div>"
                f"<div style='font-size:22px;font-weight:800;color:{_pdh_clr};"
                f"font-family:JetBrains Mono;margin:4px 0'>₹{_pdh2:,.2f}</div>"
                f"<div style='font-size:11px;color:{_pdh_clr}'>"
                f"{'+' if _d_pdh>=0 else ''}{_d_pdh:.2f}% away</div>"
                f"<div style='font-size:11px;color:{_pdh_clr};margin-top:2px'>{_pdh_lbl}</div>"
                f"</div>", unsafe_allow_html=True)
        with _pc2:
            st.markdown(
                f"<div style='background:{_pdl_clr}22;border:1px solid {_pdl_clr}44;"
                f"border-radius:10px;padding:12px 14px'>"
                f"<div style='font-size:10px;font-weight:700;color:{_pdl_clr};"
                f"letter-spacing:1px'>PREV DAY LOW</div>"
                f"<div style='font-size:22px;font-weight:800;color:{_pdl_clr};"
                f"font-family:JetBrains Mono;margin:4px 0'>₹{_pdl2:,.2f}</div>"
                f"<div style='font-size:11px;color:{_pdl_clr}'>"
                f"{'+' if _d_pdl>=0 else ''}{_d_pdl:.2f}% above</div>"
                f"<div style='font-size:11px;color:{_pdl_clr};margin-top:2px'>{_pdl_lbl}</div>"
                f"</div>", unsafe_allow_html=True)
        with _pc3:
            _range   = _pdh2 - _pdl2
            _pos_pct = ((_pr2 - _pdl2) / _range * 100) if _range > 0 else 50
            _pos_clr = '#16a34a' if _pos_pct > 60 else ('#d97706' if _pos_pct > 40 else '#dc2626')
            st.markdown(
                f"<div style='background:#f8fafc;border:1px solid #e2e8f0;"
                f"border-radius:10px;padding:12px 14px'>"
                f"<div style='font-size:10px;font-weight:700;color:#64748b;"
                f"letter-spacing:1px'>POSITION IN RANGE</div>"
                f"<div style='font-size:22px;font-weight:800;color:{_pos_clr};"
                f"font-family:JetBrains Mono;margin:4px 0'>{_pos_pct:.0f}%</div>"
                f"<div style='font-size:11px;color:#64748b'>"
                f"Range ₹{_range:,.2f} &nbsp;·&nbsp; "
                f"{'Upper half — momentum' if _pos_pct>50 else 'Lower half — recovery'}</div>"
                f"<div style='background:#e2e8f0;border-radius:3px;height:5px;margin-top:8px'>"
                f"<div style='background:{_pos_clr};height:5px;border-radius:3px;"
                f"width:{min(100,int(_pos_pct))}%'></div></div>"
                f"</div>", unsafe_allow_html=True)

    # ── CPR Panel ─────────────────────────────────────────
    _cpr_tc  = result.get('cpr_tc')
    _cpr_bc  = result.get('cpr_bc')
    _cpr_pv  = result.get('cpr_pivot')
    _cpr_r1  = result.get('cpr_r1')
    _cpr_s1  = result.get('cpr_s1')
    _cpr_r2  = result.get('cpr_r2')
    _cpr_s2  = result.get('cpr_s2')
    _cpr_w   = result.get('cpr_width')
    _price   = result.get('price', 0)

    if _cpr_tc and _cpr_bc and _cpr_pv:
        _w_pct   = _cpr_w or 0
        _w_lbl   = "NARROW ⚡ — Trending day expected" if _w_pct < 0.4 \
                   else ("MODERATE — Wait for direction" if _w_pct < 0.8 \
                   else "WIDE ⚠️ — Choppy day, avoid intraday")
        _w_bg    = "#f0fdf4" if _w_pct < 0.4 else ("#fffbeb" if _w_pct < 0.8 else "#fff5f5")
        _w_bc    = "#16a34a" if _w_pct < 0.4 else ("#d97706" if _w_pct < 0.8 else "#dc2626")

        _pos_lbl = "ABOVE TC 🟢 Bullish" if _price > _cpr_tc \
                   else ("INSIDE CPR 🟡 Neutral" if _price > _cpr_bc \
                   else "BELOW BC 🔴 Bearish")
        _pos_bg  = "#f0fdf4" if _price > _cpr_tc \
                   else ("#fffbeb" if _price > _cpr_bc else "#fff5f5")
        _pos_bc  = "#16a34a" if _price > _cpr_tc \
                   else ("#d97706" if _price > _cpr_bc else "#dc2626")

        st.markdown(f"<div class='section-header'>📐 CPR — Central Pivot Range</div>",
                    unsafe_allow_html=True)

        _ca, _cb, _cc = st.columns(3)

        with _ca:
            st.markdown(
                f"<div style='background:{_w_bg};border:1px solid {_w_bc}33;"
                f"border-radius:10px;padding:12px 14px'>"
                f"<div style='font-size:10px;font-weight:700;color:{_w_bc};"
                f"letter-spacing:1px;text-transform:uppercase'>CPR Width</div>"
                f"<div style='font-size:20px;font-weight:800;color:{_w_bc};"
                f"font-family:JetBrains Mono;margin:4px 0'>{_w_pct:.3f}%</div>"
                f"<div style='font-size:11px;color:{_w_bc}'>{_w_lbl}</div>"
                f"</div>", unsafe_allow_html=True)

        with _cb:
            st.markdown(
                f"<div style='background:{_pos_bg};border:1px solid {_pos_bc}33;"
                f"border-radius:10px;padding:12px 14px'>"
                f"<div style='font-size:10px;font-weight:700;color:{_pos_bc};"
                f"letter-spacing:1px;text-transform:uppercase'>Price vs CPR</div>"
                f"<div style='font-size:15px;font-weight:800;color:{_pos_bc};"
                f"margin:4px 0'>{_pos_lbl}</div>"
                f"<div style='font-size:11px;color:{_pos_bc}'>"
                f"TC ₹{_cpr_tc:,.2f} | BC ₹{_cpr_bc:,.2f}</div>"
                f"</div>", unsafe_allow_html=True)

        with _cc:
            st.markdown(
                f"<div style='background:#f8fafc;border:1px solid #e2e8f0;"
                f"border-radius:10px;padding:12px 14px'>"
                f"<div style='font-size:10px;font-weight:700;color:#64748b;"
                f"letter-spacing:1px;text-transform:uppercase'>Key Levels</div>"
                f"<div style='font-size:12px;color:#374151;margin-top:4px;line-height:1.8'>"
                f"<b style='color:#ef4444'>R2</b> ₹{_cpr_r2:,.2f} &nbsp;"
                f"<b style='color:#f87171'>R1</b> ₹{_cpr_r1:,.2f}<br>"
                f"<b style='color:#f59e0b'>Pivot</b> ₹{_cpr_pv:,.2f}<br>"
                f"<b style='color:#86efac'>S1</b> ₹{_cpr_s1:,.2f} &nbsp;"
                f"<b style='color:#22c55e'>S2</b> ₹{_cpr_s2:,.2f}"
                f"</div></div>", unsafe_allow_html=True)

        # CPR trade advice
        if _price > _cpr_tc:
            _cpr_advice = (f"Price is above TC ₹{_cpr_tc:,.2f}. "
                           f"Bullish bias — buy dips to TC. First target R1 ₹{_cpr_r1:,.2f}, "
                           f"stop below BC ₹{_cpr_bc:,.2f}.")
            _adv_bg, _adv_bc = "#f0fdf4", "#15803d"
        elif _price > _cpr_bc:
            _cpr_advice = (f"Price is inside CPR (BC ₹{_cpr_bc:,.2f} to TC ₹{_cpr_tc:,.2f}). "
                           f"Choppy/indecisive. Wait for a clean break above TC or below BC before trading.")
            _adv_bg, _adv_bc = "#fffbeb", "#92400e"
        else:
            _cpr_advice = (f"Price is below BC ₹{_cpr_bc:,.2f}. "
                           f"Bearish bias — selling pressure expected. "
                           f"Avoid long entries. Watch for bounce back into CPR.")
            _adv_bg, _adv_bc = "#fff5f5", "#dc2626"

        st.markdown(
            f"<div style='background:{_adv_bg};border-left:4px solid {_adv_bc};"
            f"border-radius:0 8px 8px 0;padding:10px 14px;margin:8px 0;font-size:12px;color:{_adv_bc}'>"
            f"📐 <b>CPR Signal:</b> {_cpr_advice}</div>",
            unsafe_allow_html=True)

    # ── Intraday Chart — On Demand ─────────────────────────
    st.markdown("<div class='section-header'>📈 Intraday Chart</div>", unsafe_allow_html=True)

    # Cache key — only rebuild if stock or interval changed
    _chart_cache_key = f"chart_fig_{sym}_{interval}"
    _chart_interval_key = f"chart_interval_{sym}"

    # Check if cached chart is still valid
    _cached_fig     = st.session_state.get(_chart_cache_key)
    _cached_interval= st.session_state.get(_chart_interval_key)
    _chart_stale    = (_cached_fig is None) or (_cached_interval != interval)

    _chart_col, _chart_btn_col = st.columns([5, 1])
    with _chart_btn_col:
        _rebuild_chart = st.button("🔄 Refresh Chart", key=f"rebuild_chart_{sym_clean}",
                                   help="Rebuild chart with latest data")
    with _chart_col:
        if _chart_stale or _rebuild_chart:
            with st.spinner("Building chart..."):
                _fig = build_intraday_chart(df, sym, interval_label.split("(")[0].strip())
                st.session_state[_chart_cache_key]  = _fig
                st.session_state[_chart_interval_key] = interval
        else:
            _fig = _cached_fig

    st.plotly_chart(_fig, use_container_width=True, key=f"chart_{sym}")

    # ── Signal history ──
    sig_history = df[df['Signal'] != 0][['Close','Signal','Signal_Type','Confidence','Bull_Score','Bear_Score']].tail(10)
    if len(sig_history):
        st.markdown("<div class='section-header'>🕐 Intraday Signal History</div>", unsafe_allow_html=True)
        sd = sig_history.copy()
        sd['Signal']     = sd['Signal'].map({1:'🟢 BUY', -1:'🔴 SELL'})
        sd['Close']      = sd['Close'].apply(lambda x: f"₹{x:,.2f}")
        sd['Confidence'] = sd['Confidence'].apply(lambda x: f"{x:.0f}%")
        st.dataframe(sd, use_container_width=True)

    # ── LSTM next-candle prediction ──
    st.markdown("<div class='section-header' style='margin-top:8px'>🤖 LSTM — Next 3 Candle Prediction</div>", unsafe_allow_html=True)
    _lstm_key     = f"lstm_result_{sym_clean}"
    _lstm_run_key = f"lstm_run_{sym_clean}"

    _rc1, _rc2 = st.columns([1, 2])
    with _rc1:
        _run_lstm = st.button(
            f"🤖 Predict Next 3 Candles · {sym_clean}",
            key=_lstm_run_key, use_container_width=True,
            help=f"Trains LSTM on {interval_label.split('(')[0].strip()} price history, predicts next 3 candle closes"
        )
    with _rc2:
        st.markdown(
            f"<div style='font-size:12px;color:#94a3b8;padding:10px 0'>"
            f"⚡ Pure NumPy LSTM · 80 candles lookback · 15-candle window · "
            f"predicts next 3 × {interval_label.split('(')[0].strip()} candle closes"
            f"</div>", unsafe_allow_html=True)

    if _run_lstm:
        with st.spinner(f"🧠 Training LSTM on {sym_clean} {interval_label.split('(')[0].strip()} data …"):
            _pred = lstm_predict_next_candles(df, sym_clean, n_candles=3)
        st.session_state[_lstm_key] = _pred

    _pred = st.session_state.get(_lstm_key)

    if _pred:
        if 'error' in _pred:
            st.error(f"⚠️ LSTM Error: {_pred['error']}")
        else:
            _lp     = _pred['last_price']
            _preds  = _pred['preds']
            _pcts   = _pred['pcts']
            _ts     = _pred['future_ts']
            _dir    = _pred['direction']
            _dc     = "#16a34a" if _dir == 'BULLISH' else "#dc2626"
            _db     = "#f0fdf4" if _dir == 'BULLISH' else "#fff5f5"
            _di     = "▲" if _dir == 'BULLISH' else "▼"

            st.markdown(f"""
            <div style='background:{_db};border:1.5px solid {_dc};border-radius:14px;
                        padding:14px 20px;margin:10px 0;display:flex;align-items:center;gap:16px'>
                <div style='font-size:28px;font-weight:900;color:{_dc}'>{_di}</div>
                <div>
                    <div style='font-size:16px;font-weight:800;color:{_dc}'>LSTM: {_dir}</div>
                    <div style='font-size:12px;color:#64748b;margin-top:2px'>
                        Next 3 candle trajectory from ₹{_lp:,.2f}
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

            _cc = st.columns(3)
            for _i, (_col, _p, _pct, _t) in enumerate(zip(_cc, _preds, _pcts, _ts)):
                _c  = "#16a34a" if _pct >= 0 else "#dc2626"
                _bg = "#f0fdf4" if _pct >= 0 else "#fff5f5"
                _ico= "▲" if _pct >= 0 else "▼"
                _s  = "+" if _pct >= 0 else ""
                _col.markdown(f"""
                <div style='background:#ffffff;border:1px solid #e8ecf3;border-radius:14px;
                            padding:18px 20px;text-align:center;box-shadow:0 1px 4px rgba(0,0,0,0.05)'>
                    <div style='font-size:10px;font-weight:700;color:#94a3b8;letter-spacing:1.5px;text-transform:uppercase'>Candle +{_i+1}</div>
                    <div style='font-size:11px;color:#94a3b8;margin-top:2px'>{_t} IST</div>
                    <div style='font-size:26px;font-weight:800;color:#1a2035;margin:10px 0;font-family:JetBrains Mono,monospace'>₹{_p:,.2f}</div>
                    <div style='background:{_bg};border-radius:20px;padding:4px 14px;display:inline-block;font-size:14px;font-weight:800;color:{_c}'>
                        {_ico} {_s}{_pct}%
                    </div>
                    <div style='font-size:10px;color:#94a3b8;margin-top:8px'>vs ₹{_lp:,.2f}</div>
                </div>""", unsafe_allow_html=True)

            # Forecast mini chart
            try:
                import plotly.graph_objects as _go2
                _hp  = _pred['history_prices']
                _n   = len(_hp)
                _hx  = list(range(-_n+1, 1))
                _fx  = [0, 1, 2, 3]
                _fy  = [_lp] + _preds
                _fig2 = _go2.Figure()
                _fig2.add_trace(_go2.Scatter(x=_hx, y=_hp, mode='lines',
                    line=dict(color='#1a2035', width=2), name='Historical'))
                _fig2.add_trace(_go2.Scatter(x=_fx, y=_fy, mode='lines+markers',
                    line=dict(color='#7c3aed', width=2.5, dash='dot'),
                    marker=dict(size=9, color='#7c3aed', line=dict(color='white',width=2)),
                    name='LSTM Forecast'))
                for _xi, _yi, _lbl in zip(_fx[1:], _fy[1:], [f"C+{k}" for k in range(1,4)]):
                    _fig2.add_annotation(x=_xi, y=_yi, text=f"₹{_yi:,.2f}",
                        showarrow=True, arrowhead=2,
                        font=dict(size=11, color='#7c3aed', family='JetBrains Mono'),
                        bgcolor='white', bordercolor='#7c3aed', borderwidth=1, ay=-36)
                _fig2.add_vline(x=0, line_dash='dash', line_color='#cbd5e1')
                _fig2.update_layout(
                    height=260, margin=dict(l=40,r=20,t=30,b=30),
                    paper_bgcolor='#ffffff', plot_bgcolor='#fafbfc',
                    font=dict(color='#4a5568', family='Outfit'),
                    xaxis=dict(gridcolor='#e8ecf0', title='Candles relative to now'),
                    yaxis=dict(gridcolor='#e8ecf0', title='Price (₹)', tickprefix='₹'),
                    title=dict(text=f'<b>{sym_clean}</b> — LSTM Next 3 Candle Forecast',
                               font=dict(size=13, color='#1a2035')),
                    legend=dict(orientation='h', y=1.08, x=0, font=dict(size=11)),
                )
                st.plotly_chart(_fig2, use_container_width=True, key=f"lstm_chart_{sym_clean}")
            except Exception:
                pass

            st.caption("⚠️ LSTM predictions are statistical estimates only. Intraday prices are highly volatile — use as one input alongside your own analysis.")

    # ── Portfolio position ──
    tab_positions = [p for p in load_portfolio() if p.get('symbol') == sym_clean]
    if tab_positions:
        st.markdown("<div class='section-header' style='margin-top:24px'>💼 MY INTRADAY POSITION</div>", unsafe_allow_html=True)

        # ── Refresh live price button ──────────────────────
        _da_ref_col1, _da_ref_col2 = st.columns([3, 1])
        with _da_ref_col2:
            _da_refresh = st.button(
                "🔄 Refresh P&L", key=f"refresh_pnl_{sym_clean}",
                use_container_width=True,
                help="Fetch latest price and recalculate P&L"
            )
        with _da_ref_col1:
            _da_last_refresh = st.session_state.get(f'da_refresh_time_{sym_clean}', '—')
            st.markdown(
                f"<div style='font-size:11px;color:#94a3b8;padding:10px 0'>"
                f"Live price · Last updated: {_da_last_refresh}</div>",
                unsafe_allow_html=True)

        # Fetch live price — on button click or first load
        _live_price_key = f"da_live_price_{sym_clean}"
        if _da_refresh or _live_price_key not in st.session_state:
            try:
                _ticker_sym = sym_clean + '.NS'
                _lh = yf.Ticker(_ticker_sym).history(period='1d', interval='1m')
                _live_px = float(_lh['Close'].iloc[-1]) if not _lh.empty else result['price']
            except Exception:
                _live_px = result['price']
            st.session_state[_live_price_key]            = _live_px
            st.session_state[f'da_refresh_time_{sym_clean}'] = ist_now().strftime('%H:%M:%S IST')
        else:
            _live_px = st.session_state[_live_price_key]

        for p in tab_positions:
            is_open    = p.get('status', 'OPEN') == 'OPEN'
            entry      = _f(p.get('entry', 0))
            qty        = int(_f(p.get('qty', 0)))
            actual_cost= _f(p.get('actual_cost', 0)) or 1

            # ── Bug fix: closed positions MUST use exit_price, not live price ──
            if is_open:
                cur_price  = _live_px
                price_label = "Live Price"
            else:
                _exit_px   = p.get('exit_price')
                cur_price  = _f(_exit_px) if _exit_px is not None else entry
                price_label = "Exit Price"

            unreal_pl  = (cur_price - entry) * qty
            unreal_pct = (unreal_pl / actual_cost) * 100
            pl_color   = "#16a34a" if unreal_pl >= 0 else "#dc2626"
            pl_sign    = "+" if unreal_pl >= 0 else ""
            status_lbl = "OPEN" if is_open else "CLOSED"
            status_col = "#16a34a" if is_open else "#64748b"

            # ── SL hit detection (only for OPEN positions) ──
            # Define sl first so _sl_hit_now can use it
            _sl_early = _f(p.get('stop_loss', 0))
            _sl_hit_now = is_open and _sl_early > 0 and cur_price <= _sl_early

            # Target hit highlights
            t1 = _f(p.get('t1', 0)); t2 = _f(p.get('t2', 0))
            t3 = _f(p.get('t3', 0)); t4 = _f(p.get('t4', 0))
            sl = _sl_early

            def _target_style(tval):
                if tval <= 0: return "#f8fafc", "#94a3b8"
                if cur_price >= tval: return "#dcfce7", "#15803d"  # hit
                return "#f0fdf4", "#16a34a"                         # pending

            sl_bg = "#fef2f2" if _sl_hit_now else "#fff5f5"
            sl_tc = "#dc2626"

            # ── SL Hit urgent banner (OPEN positions only) ──
            if _sl_hit_now:
                st.markdown(
                    f"<div style='background:#7f1d1d;border:2px solid #dc2626;"
                    f"border-radius:12px;padding:12px 18px;margin-bottom:8px;"
                    f"animation:pulse 1s infinite'>"
                    f"<div style='font-size:15px;font-weight:800;color:#fca5a5'>"
                    f"🛑 STOP LOSS HIT — EXIT {sym_clean} IMMEDIATELY</div>"
                    f"<div style='font-size:12px;color:#fca5a5;margin-top:4px'>"
                    f"Current ₹{cur_price:,.2f} · SL was ₹{sl:,.2f} · "
                    f"Loss: ₹{abs(unreal_pl):,.0f} · Click Square Off now</div>"
                    f"</div>", unsafe_allow_html=True)

            st.markdown(f"""
            <div style='background:#ffffff;border:1.5px solid {"#dc2626" if _sl_hit_now else "#e8ecf3"};
                        border-radius:16px;padding:18px 20px;margin-bottom:10px;
                        box-shadow:0 2px 8px rgba(0,0,0,0.04)'>

                <!-- Header row -->
                <div style='display:flex;justify-content:space-between;
                            align-items:flex-start;flex-wrap:wrap;gap:8px'>
                    <div>
                        <div style='display:flex;align-items:center;gap:10px'>
                            <span style='font-size:16px;font-weight:800;color:#1a2035'>{sym_clean}</span>
                            <span style='font-size:10px;font-weight:700;color:{status_col};
                                         background:{status_col}22;border-radius:20px;
                                         padding:2px 8px'>{status_lbl}</span>
                        </div>
                        <div style='font-size:12px;color:#64748b;margin-top:3px'>
                            {qty} shares · Entry ₹{entry:,.2f} · {p.get("timeframe","INTRADAY")} · {p.get("date","")}
                        </div>
                    </div>
                    <div style='text-align:right'>
                        <div style='font-size:11px;color:#94a3b8'>{price_label}</div>
                        <div style='font-size:22px;font-weight:800;color:#1a2035;
                                    font-family:JetBrains Mono'>₹{cur_price:,.2f}</div>
                        <div style='font-size:14px;font-weight:800;color:{pl_color}'>
                            {pl_sign}₹{unreal_pl:,.2f} ({pl_sign}{unreal_pct:.2f}%)
                        </div>
                    </div>
                </div>

                <!-- Target levels -->
                <div style='display:flex;gap:8px;margin-top:14px;flex-wrap:wrap'>
                    <div style='background:{sl_bg};border-radius:8px;padding:8px 12px;flex:1;min-width:80px;text-align:center'>
                        <div style='font-size:9px;font-weight:700;color:{sl_tc};letter-spacing:1px'>STOP LOSS</div>
                        <div style='font-size:14px;font-weight:800;color:{sl_tc};font-family:JetBrains Mono'>₹{sl:,.2f}</div>
                        <div style='font-size:9px;color:{sl_tc};margin-top:2px'>
                            {"🚨 HIT" if _sl_hit_now else f"{((cur_price-sl)/entry*100):+.1f}%"}
                        </div>
                    </div>
                    {"".join([f"""
                    <div style='background:{_target_style(tv)[0]};border-radius:8px;padding:8px 12px;flex:1;min-width:80px;text-align:center'>
                        <div style='font-size:9px;font-weight:700;color:{_target_style(tv)[1]};letter-spacing:1px'>{tlbl}</div>
                        <div style='font-size:14px;font-weight:800;color:{_target_style(tv)[1]};font-family:JetBrains Mono'>₹{tv:,.2f}</div>
                        <div style='font-size:9px;color:{_target_style(tv)[1]};margin-top:2px'>
                            {"✅ HIT" if cur_price >= tv and tv > 0 else f"{((tv-cur_price)/entry*100):+.1f}% away"}
                        </div>
                    </div>""" for tv, tlbl in [(t1,"T1 SCALP"),(t2,"T2 TARGET"),(t3,"T3 EXT"),(t4,"T4 MAX")] if tv > 0])}
                </div>
            </div>""", unsafe_allow_html=True)




# ─────────────────────────────────────────────
#  EARLY MOVERS PAGE
#  Catches gap-up stocks in first 10 minutes
#  No indicators needed — pure price + volume
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
#  ORB SCANNER PAGE
#  Opening Range Breakout — standalone page
# ─────────────────────────────────────────────
if _show_orb:
    _orb_kite_on = get_kite_client() is not None
    _orb_src_lbl = 'Kite API — Real-time' if _orb_kite_on else 'yfinance — 15 min delay'
    _orb_src_clr = '#16a34a' if _orb_kite_on else '#d97706'
    _orb_src_bg  = '#dcfce7' if _orb_kite_on else '#fef3c7'
    _orb_src_ico = '🟢' if _orb_kite_on else '🟡'

    st.markdown(f"""
    <div class='topbar'>
        <div>
            <div class='topbar-title'>🔓 Opening Range Breakout Scanner</div>
            <div class='topbar-subtitle'>
                Catches stocks that break above their first-candle high ·
                5 breakout rules · Best used 9:20 AM – 10:30 AM
            </div>
        </div>
        <div style='display:flex;align-items:center'>
            <div style='background:{_orb_src_bg};border:1px solid {_orb_src_clr}44;
                        border-radius:8px;padding:5px 12px;text-align:center'>
                <div style='font-size:10px;font-weight:700;color:{_orb_src_clr};
                            letter-spacing:1px'>DATA SOURCE</div>
                <div style='font-size:12px;font-weight:700;color:{_orb_src_clr};
                            margin-top:2px'>{_orb_src_ico} {_orb_src_lbl}</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    # ── What is ORB explanation ───────────────────────────
    st.markdown("""
    <div style='display:flex;gap:10px;margin-bottom:16px;flex-wrap:wrap'>
        <div style='flex:1;min-width:160px;background:#f5f3ff;border-radius:10px;
                    padding:12px 14px;border:1px solid #c4b5fd44'>
            <div style='font-size:18px;margin-bottom:6px'>📐</div>
            <div style='font-size:12px;font-weight:700;color:#7c3aed'>Opening Range</div>
            <div style='font-size:11px;color:#6d28d9;margin-top:3px'>
                High and Low of the first candle (9:15 AM).<br>
                This becomes the range to break.
            </div>
        </div>
        <div style='flex:1;min-width:160px;background:#fffbeb;border-radius:10px;
                    padding:12px 14px;border:1px solid #fbbf2444'>
            <div style='font-size:18px;margin-bottom:6px'>🔓</div>
            <div style='font-size:12px;font-weight:700;color:#d97706'>Breakout</div>
            <div style='font-size:11px;color:#b45309;margin-top:3px'>
                Price breaks ABOVE the first candle high<br>
                with volume confirmation.
            </div>
        </div>
        <div style='flex:1;min-width:160px;background:#f0fdf4;border-radius:10px;
                    padding:12px 14px;border:1px solid #bbf7d044'>
            <div style='font-size:18px;margin-bottom:6px'>📊</div>
            <div style='font-size:12px;font-weight:700;color:#15803d'>Difference from Early Movers</div>
            <div style='font-size:11px;color:#166534;margin-top:3px'>
                Early Movers catches gap at open (9:15 AM).<br>
                ORB catches consolidation breakout (9:20–10:30 AM).
            </div>
        </div>
        <div style='flex:1;min-width:160px;background:#fff7ed;border-radius:10px;
                    padding:12px 14px;border:1px solid #fdba7444'>
            <div style='font-size:18px;margin-bottom:6px'>⏰</div>
            <div style='font-size:12px;font-weight:700;color:#ea580c'>Best Time</div>
            <div style='font-size:11px;color:#c2410c;margin-top:3px'>
                9:20 AM – 10:30 AM on normal days.<br>
                10:00 AM – 10:30 AM on expiry days.
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    # ── Window status ─────────────────────────────────────
    _orb_now  = ist_now()
    _orb_tm   = _orb_now.hour * 60 + _orb_now.minute
    try:
        _orb_mkt_start = _orb_now.replace(hour=9, minute=15, second=0, microsecond=0)
        _orb_mins      = int((_orb_now - _orb_mkt_start.astimezone(_orb_now.tzinfo)).total_seconds() / 60)
    except Exception:
        _orb_mins = 999

    if not market_open():
        _orb_status = "🔴 Market Closed — Run at 9:20 AM"
        _orb_s_clr  = "#dc2626"; _orb_s_bg = "#fef2f2"
    elif _orb_mins < 5:
        _orb_status = f"⏳ Too early — {_orb_mins} min since open · Wait for first candle to form (9:20 AM)"
        _orb_s_clr  = "#d97706"; _orb_s_bg = "#fffbeb"
    elif _orb_mins <= 75:
        _orb_status = f"🟢 PRIME WINDOW — {_orb_mins} min since open · ORB breakouts most reliable now"
        _orb_s_clr  = "#15803d"; _orb_s_bg = "#f0fdf4"
    elif _orb_mins <= 120:
        _orb_status = f"🟡 Late window — {_orb_mins} min since open · Some breakouts still valid"
        _orb_s_clr  = "#d97706"; _orb_s_bg = "#fffbeb"
    else:
        _orb_status = f"⚪ Too late — {_orb_mins} min since open · ORB breakouts less reliable after 11:15 AM"
        _orb_s_clr  = "#64748b"; _orb_s_bg = "#f8fafc"

    st.markdown(
        f"<div style='background:{_orb_s_bg};border:1.5px solid {_orb_s_clr}44;"
        f"border-radius:12px;padding:10px 18px;margin-bottom:14px'>"
        f"<div style='font-size:13px;font-weight:700;color:{_orb_s_clr}'>{_orb_status}</div>"
        f"</div>", unsafe_allow_html=True)

    # ── Controls ──────────────────────────────────────────
    _ob1, _ob2 = st.columns([3, 1])
    with _ob1:
        _orb_universe = st.radio(
            "Scan universe",
            ["🔵 Largecap (Nifty 50)",
             "🟡 Midcap (Nifty Midcap 100)",
             "🟠 Smallcap",
             "📊 Nifty 500 (All)",
             "Custom Watchlist"],
            horizontal=True, key="orb_page_universe",
            help="Largecap = ~15s · Midcap = ~40s · Smallcap = ~55s · Nifty 500 = ~90s")
        _orb_stocks_universe = (
            LARGECAP_STOCKS  if _orb_universe == "🔵 Largecap (Nifty 50)"       else
            MIDCAP_STOCKS    if _orb_universe == "🟡 Midcap (Nifty Midcap 100)" else
            SMALLCAP_STOCKS  if _orb_universe == "🟠 Smallcap"                  else
            selected_stocks  if _orb_universe == "Custom Watchlist"              else
            POPULAR_STOCKS
        )
        _orb_count = len(_orb_stocks_universe)
        st.markdown(
            f"<div style='font-size:11px;color:#64748b;margin-top:-8px'>"
            f"⚡ {_orb_count} stocks · "
            f"{'~15 sec' if _orb_count <= 60 else '~40 sec' if _orb_count <= 150 else '~60 sec' if _orb_count <= 250 else '~90 sec'}"
            f" · <b style='color:#0369a1'>5-min candles (hardcoded — optimal for ORB)</b>"
            f"</div>", unsafe_allow_html=True)
    with _ob2:
        _run_orb_page = st.button(
            "🔓 Run ORB Scan",
            key="run_orb_page",
            use_container_width=True,
            type="primary",
            help="Scan for opening range breakouts")

    # ── Run scan ──────────────────────────────────────────
    if _run_orb_page:
        _orb_stocks = _orb_stocks_universe
        _kite_orb_pg = get_kite_client()
        _port_orb_pg = load_portfolio()

        with st.spinner(f"🔓 Scanning {len(_orb_stocks)} stocks for breakouts..."):
            _orb_page_results = run_breakout_screener(
                _orb_stocks, interval, _kite_orb_pg, _port_orb_pg)

        st.session_state['orb_results']   = _orb_page_results
        st.session_state['orb_scan_time'] = ist_now().strftime('%H:%M IST')
        st.rerun()

    # ── Results ───────────────────────────────────────────
    _orb_results = st.session_state.get('orb_results', [])
    _orb_time    = st.session_state.get('orb_scan_time', '')

    if not _orb_results:
        st.markdown(f"""
        <div style='background:#1a2035;border-radius:16px;padding:32px;
                    text-align:center;margin:20px 0'>
            <div style='font-size:40px;margin-bottom:12px'>🔓</div>
            <div style='font-size:18px;font-weight:800;color:#ffffff;margin-bottom:8px'>
                No results yet
            </div>
            <div style='font-size:13px;color:rgba(255,255,255,0.5);line-height:1.8'>
                Click <b style='color:#f59e0b'>🔓 Run ORB Scan</b> above.<br>
                Best used <b style='color:#f59e0b'>9:20 AM – 10:30 AM IST</b> on normal days.<br>
                After a stock consolidates for 5+ minutes and then breaks out.
            </div>
        </div>""", unsafe_allow_html=True)

    else:
        # Header
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;align-items:center;"
            f"flex-wrap:wrap;gap:8px;margin-bottom:12px'>"
            f"<div style='font-size:14px;font-weight:700;color:#1a2035'>"
            f"🔓 {len(_orb_results)} breakouts found · {_orb_time}</div>"
            f"<div style='display:flex;gap:8px'>"
            f"<span style='background:#f5f3ff;color:#7c3aed;font-size:11px;font-weight:700;"
            f"border-radius:6px;padding:3px 10px'>Top score: {_orb_results[0]['best']['score']}</span>"
            f"</div></div>", unsafe_allow_html=True)

        # Entry guide
        st.markdown("""
        <div style='background:#1a2035;border-radius:10px;padding:12px 18px;margin-bottom:14px'>
            <div style='font-size:12px;font-weight:700;color:#f59e0b;margin-bottom:6px'>
                📋 How to trade ORB — 3 steps
            </div>
            <div style='font-size:11px;color:rgba(255,255,255,0.7);line-height:2'>
                <b style='color:#34d399'>Step 1:</b>
                Pick ENTER NOW stocks with score ≥ 75 and Vol ≥ 2×.<br>
                <b style='color:#34d399'>Step 2:</b>
                Stop Loss = first candle LOW (ORB low). Not ATR-based.<br>
                <b style='color:#34d399'>Step 3:</b>
                Target = first candle range × 1.5 above the breakout point.
            </div>
        </div>""", unsafe_allow_html=True)

        # Result cards
        for _bo_r in _orb_results[:15]:
            _best  = _bo_r['best']
            _bc    = _best['color']
            _bbg   = _best['bg']
            _chg   = _bo_r['chg_pct']
            _chgc  = "#16a34a" if _chg >= 0 else "#dc2626"

            # Build patterns HTML separately to avoid nested f-string issues
            _orb_patterns_html = ""
            for _p in _bo_r['breakouts']:
                _p_msg = _p['msg'][_p['msg'].find('|')+2:] if '|' in _p['msg'] else _p['msg']
                _orb_patterns_html += (
                    f"<div style='background:{_p['bg']};border-left:4px solid {_p['color']};"
                    f"border-radius:0 8px 8px 0;padding:8px 12px;margin-bottom:4px'>"
                    f"<div style='font-size:12px;font-weight:700;color:{_p['color']}'>"
                    f"{_p['icon']} {_p['title']}</div>"
                    f"<div style='font-size:11px;color:#374151;margin-top:3px'>{_p_msg}</div>"
                    f"<div style='font-size:11px;font-weight:700;color:{_p['color']};"
                    f"margin-top:4px;background:white;border-radius:6px;"
                    f"padding:4px 10px;display:inline-block'>➤ {_p['action']}</div>"
                    f"</div>"
                )

            _oc1, _oc2 = st.columns([5, 1])
            with _oc1:
                st.markdown(f"""
                <div style='background:#ffffff;border:1.5px solid {_bc}33;
                            border-radius:14px;padding:16px 18px;margin-bottom:8px;
                            box-shadow:0 2px 8px rgba(0,0,0,0.04)'>
                    <div style='display:flex;align-items:flex-start;
                                justify-content:space-between;flex-wrap:wrap;gap:8px'>
                        <div style='display:flex;align-items:center;gap:12px'>
                            <div style='font-size:28px'>{_best['icon']}</div>
                            <div>
                                <div style='display:flex;align-items:center;gap:8px;flex-wrap:wrap'>
                                    <span style='font-size:20px;font-weight:800;color:#1a2035'>
                                        {_bo_r['sym_clean']}
                                    </span>
                                    <span style='background:{_bbg};color:{_bc};
                                                 font-size:12px;font-weight:700;
                                                 border-radius:6px;padding:3px 10px'>
                                        {_best['title']}
                                    </span>
                                    <span style='background:#f5f3ff;color:#7c3aed;
                                                 font-size:11px;font-weight:700;
                                                 border-radius:6px;padding:2px 8px'>
                                        Score {_best['score']}
                                    </span>
                                    <span style='font-size:10px;color:#94a3b8'>
                                        {'⚡ Kite' if _bo_r.get('src')=='kite' else '⏳ yfinance'}
                                    </span>
                                </div>
                                <div style='font-size:12px;color:#64748b;margin-top:4px'>
                                    Prev ₹{_bo_r['prev_close']:,.2f}
                                    &nbsp;·&nbsp; Now ₹{_bo_r['price']:,.2f}
                                    &nbsp;·&nbsp; Vol {_bo_r['vol_ratio']}×
                                </div>
                            </div>
                        </div>
                        <div style='text-align:right'>
                            <div style='font-size:22px;font-weight:800;color:#1a2035;
                                        font-family:JetBrains Mono'>
                                ₹{_bo_r['price']:,.2f}
                            </div>
                            <div style='font-size:13px;font-weight:700;color:{_chgc}'>
                                {_chg:+.2f}% from yesterday
                            </div>
                        </div>
                    </div>

                    <!-- All patterns found placeholder -->
                    <div id='orb_patterns_placeholder_{_bo_r["sym_clean"]}'></div>
                </div>""", unsafe_allow_html=True)
                # Render patterns separately to avoid f-string quote conflicts
                if _orb_patterns_html:
                    st.markdown(
                        f"<div style='margin-top:-8px;margin-bottom:8px;"
                        f"padding:0 18px;display:flex;flex-direction:column;gap:6px'>"
                        f"{_orb_patterns_html}</div>",
                        unsafe_allow_html=True)
            with _oc2:
                st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

                _orb_sym    = _bo_r['sym_clean']
                _orb_entry  = _bo_r['price']
                _orb_prev   = _bo_r['prev_close']
                _orb_range  = _bo_r.get('orb_range', _orb_entry * 0.005)
                # SL = first candle low (real support level, not fixed %)
                _orb_sl     = round(_bo_r.get('first_low', _orb_entry * 0.995), 2)
                _orb_risk_d = max(_orb_entry - _orb_sl, 0.01)
                # Targets: 1R, 2R, 3R from entry (minimum 1:1 R:R at T1)
                _orb_t1     = round(_orb_entry + _orb_risk_d * 1.0, 2)  # R:R 1:1
                _orb_t2     = round(_orb_entry + _orb_risk_d * 2.0, 2)  # R:R 2:1
                _orb_qty    = max(1, int((capital * risk_pct / 100) / _orb_risk_d))

                _orb_is_buy = _bo_r.get('best', {}).get('score', 0) >= 60

                if _orb_is_buy:
                    if st.button(
                        f"✅ Paper Buy",
                        key=f"orb_paper_buy_{_orb_sym}",
                        use_container_width=True,
                        type="primary",
                    ):
                        _port = load_portfolio()
                        _already = any(
                            p.get('symbol') == _orb_sym and p.get('status') == 'OPEN'
                            for p in _port
                        )
                        if _already:
                            st.warning(f"⚠️ Already open: {_orb_sym}")
                        else:
                            _port.append({
                                'symbol':      _orb_sym,
                                'status':      'OPEN',
                                'entry':       round(_orb_entry, 2),
                                'qty':         _orb_qty,
                                'stop_loss':   _orb_sl,
                                't1':          _orb_t1,
                                't2':          round(_orb_entry + _orb_range * 2.5, 2),
                                't3':          0, 't4': 0,
                                'investment':  round(_orb_entry * _orb_qty, 2),
                                'actual_cost': round(_orb_entry * _orb_qty, 2),
                                'timeframe':   '1min — ORB Breakout',
                                'date':        ist_now().strftime('%d %b %Y %H:%M'),
                                'entry_time':  ist_now().strftime('%H:%M'),
                                'nifty_state': st.session_state.get('nifty_market_state', 'UNKNOWN'),
                                'vix_level':   st.session_state.get('nifty_context', {}).get('vix_level', 'UNKNOWN'),
                                'score':       _bo_r.get('best', {}).get('score', 0),
                                'verdict':     _bo_r.get('best', {}).get('title', 'ORB BUY'),
                                'vol_ratio':   _bo_r.get('vol_ratio', 0),
                                'source':      'orb_scanner',
                                'exit_reason': '',
                            })
                            save_portfolio(_port)
                            st.session_state['paper_portfolio'] = _port
                            st.success(
                                f"✅ Bought {_orb_qty} × {_orb_sym} @ ₹{_orb_entry:,.2f} · "
                                f"SL ₹{_orb_sl:,.2f} · T1 ₹{_orb_t1:,.2f}"
                            )
                            st.rerun()
                else:
                    if st.button(
                        "🔬 Analyse",
                        key=f"orb_page_analyse_{_orb_sym}",
                        use_container_width=True,
                    ):
                        st.session_state['_focus_stock'] = _orb_sym
                        st.session_state['active_page']  = "📊  Scanner"
                        st.rerun()

        # Refresh button
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button("🔄 Refresh ORB Scan", key="orb_page_refresh",
                     use_container_width=True):
            st.session_state.pop('orb_results', None)
            st.rerun()


if _show_earlymovers:

    # ══════════════════════════════════════════════════════
    #  EXPIRY DAY DETECTION
    #  Nifty weekly = every Thursday
    #  Bank Nifty   = every Wednesday
    #  Monthly      = last Thursday of month
    # ══════════════════════════════════════════════════════
    _expiry_info = detect_expiry()  # uses top-level function
    _is_expiry   = _expiry_info['is_expiry']
    _is_monthly  = _expiry_info['is_monthly']

    # Banking stocks that get pinned on expiry
    _BANKING_PINNED = {
        'HDFCBANK','ICICIBANK','AXISBANK','KOTAKBANK','SBIN','INDUSINDBK',
        'BANDHANBNK','FEDERALBNK','IDFCFIRSTB','AUBANK','RBLBANK','PNB',
        'BANKBARODA','CANBK','YESBANK',
    }

    # ── Topbar ────────────────────────────────────────────
    _em_topbar_sub = (
        f"⚠️ {_expiry_info['expiry_label']} · Entry: {_expiry_info['best_entry_time']} · Exit by {_expiry_info['exit_time']}"
        if _is_expiry else
        "Gap-up stocks with volume explosion · No indicators needed · Best used 9:15 AM – 9:30 AM"
    )
    _em_topbar_clr = "#f59e0b" if _is_expiry else "rgba(255,255,255,0.6)"
    _em_kite_on    = get_kite_client() is not None
    _em_src_lbl    = 'Kite API — Real-time' if _em_kite_on else 'yfinance — 15 min delay'
    _em_src_clr    = '#16a34a' if _em_kite_on else '#d97706'
    _em_src_bg     = '#dcfce7' if _em_kite_on else '#fef3c7'
    _em_src_ico    = '🟢' if _em_kite_on else '🟡'
    _em_expiry_badge = (
        "<div class='topbar-badge' style='background:rgba(239,68,68,0.2);"
        "color:#fca5a5;border-color:rgba(239,68,68,0.4)'>⚠️ EXPIRY DAY</div>"
        if _is_expiry else ""
    )
    st.markdown(f"""
    <div class='topbar'>
        <div>
            <div class='topbar-title'>🚀 Early Movers — First 15 Minutes</div>
            <div class='topbar-subtitle' style='color:{_em_topbar_clr}'>
                {_em_topbar_sub}
            </div>
        </div>
        <div style='display:flex;align-items:center;gap:8px'>
            <div style='background:{_em_src_bg};border:1px solid {_em_src_clr}44;
                        border-radius:8px;padding:5px 12px;text-align:center'>
                <div style='font-size:10px;font-weight:700;color:{_em_src_clr};
                            letter-spacing:1px'>DATA SOURCE</div>
                <div style='font-size:12px;font-weight:700;color:{_em_src_clr};
                            margin-top:2px'>{_em_src_ico} {_em_src_lbl}</div>
            </div>
            {_em_expiry_badge}
        </div>
    </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════
    #  EXPIRY DAY BANNER
    # ══════════════════════════════════════════════════════
    if _is_expiry:
        _exp_bg  = "#450a0a" if _is_monthly else "#1c1917"
        _exp_bdr = "#dc2626" if _is_monthly else "#d97706"
        _exp_ttl = "#fca5a5" if _is_monthly else "#fbbf24"
        st.markdown(
            f"<div style='background:{_exp_bg};border:2px solid {_exp_bdr};"
            f"border-radius:14px;padding:18px 22px;margin-bottom:16px'>"
            f"<div style='font-size:16px;font-weight:800;color:{_exp_ttl};margin-bottom:10px'>"
            f"{'🚨' if _is_monthly else '⚠️'} {_expiry_info['expiry_label']}</div>"
            f"<div style='display:flex;gap:16px;flex-wrap:wrap'>"
            # Rules
            f"<div style='flex:1;min-width:200px'>"
            f"<div style='font-size:11px;font-weight:700;color:{_exp_ttl};letter-spacing:1px;margin-bottom:6px'>EXPIRY RULES</div>"
            f"<div style='font-size:11px;color:rgba(255,255,255,0.7);line-height:1.9'>"
            f"🚫 No entry before <b style='color:{_exp_ttl}'>10:00 AM</b> — fake moves<br>"
            f"⏳ Need <b style='color:{_exp_ttl}'>3 candle confirmation</b> before entry<br>"
            f"🎯 Target = <b style='color:{_exp_ttl}'>{_expiry_info['target_multiplier']}× gap</b> (take profit early)<br>"
            f"🚪 Exit ALL by <b style='color:{_exp_ttl}'>{_expiry_info['exit_time']}</b> — extreme volatility after<br>"
            f"📉 Gap fill probability = <b style='color:{_exp_ttl}'>{_expiry_info['gap_fill_prob']}%</b> (vs 30% normal)"
            f"</div></div>"
            # Entry window
            f"<div style='flex:1;min-width:200px'>"
            f"<div style='font-size:11px;font-weight:700;color:{_exp_ttl};letter-spacing:1px;margin-bottom:6px'>BEST ENTRY WINDOWS</div>"
            f"<div style='font-size:11px;color:rgba(255,255,255,0.7);line-height:1.9'>"
            f"🔴 <b>9:15–10:00 AM</b> — AVOID (fake moves, traps)<br>"
            f"🟡 <b>10:00–10:30 AM</b> — Only confirmed 3-candle breakouts<br>"
            f"🔴 <b>10:30–1:30 PM</b> — AVOID (time decay, choppy)<br>"
            f"🟢 <b>1:30–2:30 PM</b> — BEST window (genuine direction)<br>"
            f"🚫 <b>After 2:30 PM</b> — Close all, extreme volatility"
            f"</div></div>"
            # Avoid list
            f"<div style='flex:1;min-width:200px'>"
            f"<div style='font-size:11px;font-weight:700;color:{_exp_ttl};letter-spacing:1px;margin-bottom:6px'>AVOID THESE STOCKS</div>"
            f"<div style='font-size:11px;color:rgba(255,255,255,0.7);line-height:1.9'>"
            f"🏦 All banking stocks — options pinning<br>"
            f"🔢 Stocks near round numbers ₹100/200/500<br>"
            f"📊 Stocks that moved >3% already — chasing<br>"
            f"✅ <b style='color:{_exp_ttl}'>Prefer:</b> Mid-cap IT, Pharma, Consumer<br>"
            f"✅ These move on merit, not options pinning"
            f"</div></div>"
            f"</div></div>", unsafe_allow_html=True)

    # ── Time window status ────────────────────────────────
    _now_em    = ist_now()
    _tm_em     = _now_em.hour * 60 + _now_em.minute
    _mkt_start = _now_em.replace(hour=9, minute=15, second=0, microsecond=0)
    try:
        _mins_since = int((_now_em - _mkt_start.astimezone(_now_em.tzinfo)).total_seconds() / 60)
    except Exception:
        _mins_since = 999

    if not market_open():
        _em_status     = "🔴 Market Closed — Run at 9:15 AM for live results"
        _em_status_clr = "#dc2626"
        _em_status_bg  = "#fef2f2"
    elif _is_expiry:
        # Expiry-specific time windows
        if _tm_em < 615:    # before 10:15 AM
            _em_status     = f"🔴 EXPIRY — Too early ({_mins_since} min since open) · Fake moves likely · Wait until 10:00 AM"
            _em_status_clr = "#dc2626"
            _em_status_bg  = "#fef2f2"
        elif _tm_em <= 630:   # 10:00–10:30 AM
            _em_status     = f"🟡 EXPIRY — Confirmation window · Only enter with 3-candle breakout confirmed"
            _em_status_clr = "#d97706"
            _em_status_bg  = "#fffbeb"
        elif _tm_em < 810:   # 10:30 AM–1:30 PM
            _em_status     = f"🔴 EXPIRY — Choppy zone · Avoid new entries · Wait for 1:30 PM window"
            _em_status_clr = "#dc2626"
            _em_status_bg  = "#fef2f2"
        elif _tm_em <= 870:  # 1:30–2:30 PM
            _em_status     = f"🟢 EXPIRY BEST WINDOW — 1:30–2:30 PM · Genuine moves now · Enter with normal rules"
            _em_status_clr = "#15803d"
            _em_status_bg  = "#f0fdf4"
        else:
            _em_status     = f"🚫 EXPIRY — Past 2:30 PM · Close all positions · Do not enter"
            _em_status_clr = "#7f1d1d"
            _em_status_bg  = "#fef2f2"
    else:
        # Normal day windows
        if _mins_since <= 15:
            _em_status     = f"🟢 PRIME WINDOW — {_mins_since} min since open · Best time to catch moves"
            _em_status_clr = "#15803d"
            _em_status_bg  = "#f0fdf4"
        elif _mins_since <= 30:
            _em_status     = f"🟡 Good Window — {_mins_since} min since open · Most moves already started"
            _em_status_clr = "#d97706"
            _em_status_bg  = "#fffbeb"
        else:
            _em_status     = f"⚪ Late — {_mins_since} min since open · Use normal scanner instead"
            _em_status_clr = "#64748b"
            _em_status_bg  = "#f8fafc"

    st.markdown(
        f"<div style='background:{_em_status_bg};border:1.5px solid {_em_status_clr}44;"
        f"border-radius:12px;padding:12px 18px;margin-bottom:14px'>"
        f"<div style='font-size:14px;font-weight:700;color:{_em_status_clr}'>"
        f"{_em_status}</div>"
        f"<div style='font-size:11px;color:{_em_status_clr};margin-top:4px;opacity:0.8'>"
        f"{'Expiry mode: 3-candle confirmation required · Banking stocks flagged · Reduced targets'  if _is_expiry else 'Normal mode: Gap > 1% + Vol > 3× + Price holding above open'}"
        f"</div></div>", unsafe_allow_html=True)

    # ── Rules explanation (changes on expiry) ─────────────
    if _is_expiry:
        st.markdown(f"""
        <div style='display:flex;gap:10px;margin-bottom:16px;flex-wrap:wrap'>
            <div style='flex:1;min-width:160px;background:#fef2f2;border-radius:10px;
                        padding:12px 14px;border:1px solid #fecaca44'>
                <div style='font-size:20px;margin-bottom:6px'>📐</div>
                <div style='font-size:12px;font-weight:700;color:#dc2626'>Rule 1 — Gap Up</div>
                <div style='font-size:11px;color:#991b1b;margin-top:3px'>
                    Same as normal day.<br>
                    But gap fill chance = <b>{_expiry_info['gap_fill_prob']}%</b>
                </div>
            </div>
            <div style='flex:1;min-width:160px;background:#fffbeb;border-radius:10px;
                        padding:12px 14px;border:1px solid #fbbf2444'>
                <div style='font-size:20px;margin-bottom:6px'>📊</div>
                <div style='font-size:12px;font-weight:700;color:#d97706'>Rule 2 — Volume</div>
                <div style='font-size:11px;color:#b45309;margin-top:3px'>
                    Same check.<br>
                    High vol on expiry may be <b>hedging</b>, not buying.
                </div>
            </div>
            <div style='flex:1;min-width:160px;background:#fff7ed;border-radius:10px;
                        padding:12px 14px;border:1px solid #fdba7444'>
                <div style='font-size:20px;margin-bottom:6px'>3️⃣</div>
                <div style='font-size:12px;font-weight:700;color:#ea580c'>Rule 3 — 3 Candles</div>
                <div style='font-size:11px;color:#c2410c;margin-top:3px'>
                    <b>NEW on expiry.</b> Price must make<br>
                    3 consecutive higher highs.
                </div>
            </div>
            <div style='flex:1;min-width:160px;background:#f0fdf4;border-radius:10px;
                        padding:12px 14px;border:1px solid #bbf7d044'>
                <div style='font-size:20px;margin-bottom:6px'>⏰</div>
                <div style='font-size:12px;font-weight:700;color:#15803d'>Rule 4 — Time Gate</div>
                <div style='font-size:11px;color:#166534;margin-top:3px'>
                    <b>NEW on expiry.</b> No entry before<br>
                    10:00 AM regardless of signal.
                </div>
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='display:flex;gap:10px;margin-bottom:16px;flex-wrap:wrap'>
            <div style='flex:1;min-width:180px;background:#f5f3ff;border-radius:10px;
                        padding:12px 14px;border:1px solid #c4b5fd44'>
                <div style='font-size:20px;margin-bottom:6px'>📐</div>
                <div style='font-size:12px;font-weight:700;color:#7c3aed'>Rule 1 — Gap Up</div>
                <div style='font-size:11px;color:#6d28d9;margin-top:3px'>
                    Opened > 1% above yesterday's close.<br>
                    Shows overnight demand.
                </div>
            </div>
            <div style='flex:1;min-width:180px;background:#fffbeb;border-radius:10px;
                        padding:12px 14px;border:1px solid #fbbf2444'>
                <div style='font-size:20px;margin-bottom:6px'>📊</div>
                <div style='font-size:12px;font-weight:700;color:#d97706'>Rule 2 — Volume Surge</div>
                <div style='font-size:11px;color:#b45309;margin-top:3px'>
                    First candle volume > 3× average.<br>
                    Shows institutions are buying.
                </div>
            </div>
            <div style='flex:1;min-width:180px;background:#f0fdf4;border-radius:10px;
                        padding:12px 14px;border:1px solid #bbf7d044'>
                <div style='font-size:20px;margin-bottom:6px'>📌</div>
                <div style='font-size:12px;font-weight:700;color:#15803d'>Rule 3 — Gap Holding</div>
                <div style='font-size:11px;color:#166534;margin-top:3px'>
                    Current price still above opening price.<br>
                    Gap not fading = buyers in control.
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

    # ── Controls ──────────────────────────────────────────
    _em_c1, _em_c2, _em_c3 = st.columns([2, 1, 1])
    with _em_c1:
        _em_universe = st.radio(
            "Scan universe",
            ["🔵 Largecap (Nifty 50)",
             "🟡 Midcap (Nifty Midcap 100)",
             "🟠 Smallcap",
             "📊 Nifty 500 (All)",
             "Custom Watchlist"],
            horizontal=True, key="em_universe",
            help="Largecap = ~15s · Midcap = ~40s · Nifty 500 = ~90s")
        _em_stocks_universe = (
            LARGECAP_STOCKS  if _em_universe == "🔵 Largecap (Nifty 50)"       else
            MIDCAP_STOCKS    if _em_universe == "🟡 Midcap (Nifty Midcap 100)" else
            SMALLCAP_STOCKS  if _em_universe == "🟠 Smallcap"                  else
            selected_stocks  if _em_universe == "Custom Watchlist"              else
            POPULAR_STOCKS
        )
        _em_count = len(_em_stocks_universe)
        st.markdown(
            f"<div style='font-size:11px;color:#64748b;margin-top:-8px'>"
            f"⚡ {_em_count} stocks · "
            f"{'~30 sec' if _em_count <= 120 else '~60 sec' if _em_count <= 250 else '~90 sec'}"
            f" · <b style='color:#f59e0b'>1-min candles (hardcoded — fastest gap detection)</b>"
            f"</div>", unsafe_allow_html=True)
    with _em_c2:
        _em_gap_min = st.number_input(
            "Min gap %", min_value=0.5, max_value=5.0,
            value=1.0, step=0.5, format="%.1f", key="em_gap_min",
            help="Minimum gap-up % from previous close")
    with _em_c3:
        _em_vol_min = st.number_input(
            "Min volume ×", min_value=1.5, max_value=15.0,
            value=3.0, step=0.5, format="%.1f", key="em_vol_min",
            help="Minimum first-candle volume vs average")

    _run_em = st.button(
        "🚀 Scan Early Movers Now",
        key="run_early_movers", use_container_width=True, type="primary",
        help="Scans stocks for gap-up + volume surge.")

    # ══════════════════════════════════════════════════════
    #  SCANNER FUNCTION — EXPIRY AWARE
    # ══════════════════════════════════════════════════════
    def scan_early_movers(stocks, gap_min_pct, vol_min_x, kite, is_expiry_day):
        """
        Price + volume scan with optional expiry-day rules.
        On expiry: 3-candle confirmation, banking flagged, reduced targets.
        """
        results   = []
        total     = len(stocks)
        _prog_em  = st.progress(0, text="🚀 Scanning for early movers...")
        _stat_em  = st.empty()

        import pytz as _ptz_em
        _ist_em   = _ptz_em.timezone('Asia/Kolkata')
        _today_em = datetime.now(_ist_em).date()
        _now_min  = ist_now().hour * 60 + ist_now().minute

        for idx, symbol in enumerate(stocks):
            pct       = int(((idx + 1) / total) * 100)
            sym_clean = symbol.replace('.NS', '')
            _prog_em.progress(pct, text=f"🚀 {idx+1}/{total} · {sym_clean}")

            try:
                _ck = _cache_key(symbol, '1minute')
                if _ck in _DATA_CACHE:
                    df, src = _DATA_CACHE[_ck]
                else:
                    df, src = fetch_intraday(symbol, '1minute', '1d', kite=kite)
                    if df is None or len(df) < 3:
                        continue

                _idx_em = pd.to_datetime(df.index)
                if _idx_em.tzinfo is None:
                    _idx_em = _idx_em.tz_localize('UTC').tz_convert('Asia/Kolkata')
                else:
                    _idx_em = _idx_em.tz_convert('Asia/Kolkata')

                _today_df = df[_idx_em.date == _today_em]
                _prev_df  = df[_idx_em.date < _today_em]

                if len(_today_df) < 1 or len(_prev_df) < 5:
                    continue

                _prev_close  = float(_prev_df['Close'].iloc[-1])
                _open_price  = float(_today_df['Open'].iloc[0])
                _curr_price  = float(_today_df['Close'].iloc[-1])
                _first_vol   = float(_today_df['Volume'].iloc[0])
                _avg_vol     = float(_prev_df['Volume'].mean())

                if _avg_vol <= 0:
                    continue

                # ── Rule 1: Gap up ────────────────────────
                _gap_pct = (_open_price - _prev_close) / _prev_close * 100
                if _gap_pct < gap_min_pct:
                    continue

                # ── Rule 2: Volume ────────────────────────
                _vol_x = _first_vol / _avg_vol
                if _vol_x < vol_min_x:
                    continue

                # ── Rule 3: Gap holding ───────────────────
                _holding  = _curr_price >= _open_price * 0.998
                _fade_pct = (_curr_price - _open_price) / _open_price * 100
                _day_chg  = (_curr_price - _prev_close) / _prev_close * 100

                # ── Expiry Rule: 3-candle confirmation ────
                _three_candle_confirmed = False
                _candle_detail = ""
                if is_expiry_day and len(_today_df) >= 3:
                    _highs = _today_df['High'].values
                    _closes= _today_df['Close'].values
                    _vols  = _today_df['Volume'].values
                    # 3 consecutive higher highs AND volume not declining
                    _hh3   = all(_highs[i] > _highs[i-1] for i in range(1, min(3, len(_highs))))
                    _vol_ok= not (_vols[-1] < _vols[-2] < _vols[0]) if len(_vols) >= 3 else True
                    _three_candle_confirmed = _hh3 and _vol_ok and _holding
                    _candle_detail = (
                        "✅ 3 higher highs confirmed" if _three_candle_confirmed
                        else "⏳ Not yet confirmed — watch"
                    )
                elif not is_expiry_day:
                    _three_candle_confirmed = True   # no extra check on normal day

                # ── Banking check ─────────────────────────
                _is_banking = sym_clean in _BANKING_PINNED
                _banking_warn = "🏦 Options pinning risk" if (_is_expiry and _is_banking) else ""

                # ── Time gate on expiry ───────────────────
                _time_blocked = is_expiry_day and _now_min < 600  # before 10:00 AM

                # ── Strength score ────────────────────────
                _strength = round(_gap_pct * _vol_x, 1)

                # ── Target calculation ────────────────────
                _gap_amt    = _open_price - _prev_close
                # Normal day: T1=2×gap, T2=3×gap, T3=4×gap (raised from 1.5×)
                # Expiry day: T1=0.5×gap — volatile, take profit quickly
                _tgt_mult   = _expiry_info['target_multiplier'] if is_expiry_day else 2.0
                _target_px  = round(_open_price + _gap_amt * _tgt_mult, 2)
                _target_t2  = round(_open_price + _gap_amt * (_tgt_mult + 1.0), 2)
                _target_t3  = round(_open_price + _gap_amt * (_tgt_mult + 2.0), 2)
                _target_lbl = f"{_tgt_mult}× gap {'(expiry reduced)' if is_expiry_day else ''}"

                # ── Action label ──────────────────────────
                if _time_blocked:
                    _action     = "⏳ WAIT — Before 10 AM"
                    _action_clr = "#d97706"
                    _action_bg  = "#fffbeb"
                elif is_expiry_day and not _three_candle_confirmed:
                    _action     = "⏳ WAIT — Need 3 candles"
                    _action_clr = "#d97706"
                    _action_bg  = "#fffbeb"
                elif is_expiry_day and _is_banking:
                    _action     = "⚠️ CAUTION — Banking"
                    _action_clr = "#ea580c"
                    _action_bg  = "#fff7ed"
                elif not _holding:
                    _action     = "⚠️ FADING"
                    _action_clr = "#dc2626"
                    _action_bg  = "#fff5f5"
                elif _vol_x >= 8 and _gap_pct >= 2.0 and _holding:
                    _action     = "🏦 ENTER NOW"
                    _action_clr = "#15803d"
                    _action_bg  = "#dcfce7"
                elif _vol_x >= 5 and _gap_pct >= 1.5 and _holding:
                    _action     = "🔥 ENTER NOW"
                    _action_clr = "#16a34a"
                    _action_bg  = "#f0fdf4"
                elif _vol_x >= 3 and _gap_pct >= 1.0 and _holding:
                    _action     = "⚡ WATCH"
                    _action_clr = "#d97706"
                    _action_bg  = "#fffbeb"
                else:
                    _action     = "👀 MONITOR"
                    _action_clr = "#64748b"
                    _action_bg  = "#f8fafc"

                results.append({
                    'symbol':       symbol,
                    'sym_clean':    sym_clean,
                    'prev_close':   round(_prev_close, 2),
                    'open_price':   round(_open_price, 2),
                    'curr_price':   round(_curr_price, 2),
                    'gap_pct':      round(_gap_pct, 2),
                    'vol_x':        round(_vol_x, 1),
                    'holding':      _holding,
                    'fade_pct':     round(_fade_pct, 2),
                    'day_chg':      round(_day_chg, 2),
                    'strength':     _strength,
                    'action':       _action,
                    'action_clr':   _action_clr,
                    'action_bg':    _action_bg,
                    'src':          src,
                    'n_candles':    len(_today_df),
                    'is_banking':   _is_banking,
                    'banking_warn': _banking_warn,
                    'three_candle': _three_candle_confirmed,
                    'candle_detail':_candle_detail,
                    'time_blocked': _time_blocked,
                    'target_px':    _target_px,
                    'target_t2':    _target_t2,
                    'target_t3':    _target_t3,
                    'target_lbl':   _target_lbl,
                    'gap_amt':      round(_gap_amt, 2),
                    'gap_fill_prob':_expiry_info['gap_fill_prob'] if is_expiry_day else 30,
                })

                if len(results) % 5 == 0:
                    _stat_em.markdown(
                        f"<div style='font-size:12px;color:#7c3aed;padding:4px 0'>"
                        f"🚀 {len(results)} movers found so far...</div>",
                        unsafe_allow_html=True)

            except Exception:
                continue

        _prog_em.empty()
        _stat_em.empty()
        results.sort(key=lambda x: x['strength'], reverse=True)
        return results

    # ── Run scan ──────────────────────────────────────────
    if _run_em:
        _em_stocks = _em_stocks_universe
        _kite_em   = get_kite_client()
        with st.spinner(""):
            _em_results = scan_early_movers(
                _em_stocks, _em_gap_min, _em_vol_min, _kite_em, _is_expiry)
        st.session_state['early_movers']      = _em_results
        st.session_state['early_movers_time'] = ist_now().strftime('%H:%M:%S IST')
        st.session_state['early_movers_gap']  = _em_gap_min
        st.session_state['early_movers_vol']  = _em_vol_min
        st.rerun()

    # ── Show results ──────────────────────────────────────
    _em_results  = st.session_state.get('early_movers', [])
    _em_scantime = st.session_state.get('early_movers_time', '')

    if not _em_results:
        st.markdown(f"""
        <div style='background:#1a2035;border-radius:16px;padding:32px;
                    text-align:center;margin:20px 0'>
            <div style='font-size:40px;margin-bottom:12px'>🚀</div>
            <div style='font-size:18px;font-weight:800;color:#ffffff;margin-bottom:8px'>
                No results yet
            </div>
            <div style='font-size:13px;color:rgba(255,255,255,0.5);line-height:1.8'>
                Click <b style='color:#f59e0b'>🚀 Scan Early Movers Now</b> above.<br>
                {'<b style="color:#fbbf24">Expiry day:</b> Scan at 10:00 AM or 1:30 PM for best results.' if _is_expiry else
                 'Best used between <b style="color:#f59e0b">9:15 AM and 9:30 AM IST</b>.'}
            </div>
        </div>""", unsafe_allow_html=True)

    else:
        # Header stats
        _em_enter = sum(1 for r in _em_results if 'ENTER' in r['action'])
        _em_watch = sum(1 for r in _em_results if 'WATCH' in r['action'])
        _em_wait  = sum(1 for r in _em_results if 'WAIT' in r['action'])
        _em_fade  = sum(1 for r in _em_results if 'FADING' in r['action'])
        _em_caut  = sum(1 for r in _em_results if 'CAUTION' in r['action'])

        _expiry_badge = (
            f"<span style='background:#fef2f2;color:#dc2626;font-size:11px;"
            f"font-weight:700;border-radius:6px;padding:3px 10px'>⚠️ Expiry Mode</span>"
            if _is_expiry else ""
        )
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;align-items:center;"
            f"flex-wrap:wrap;gap:8px;margin-bottom:12px'>"
            f"<div style='font-size:14px;font-weight:700;color:#1a2035'>"
            f"🚀 {len(_em_results)} movers found · {_em_scantime}</div>"
            f"<div style='display:flex;gap:6px;flex-wrap:wrap'>"
            f"{_expiry_badge}"
            f"<span style='background:#dcfce7;color:#15803d;font-size:11px;font-weight:700;border-radius:6px;padding:3px 10px'>✅ {_em_enter} Enter</span>"
            f"<span style='background:#fffbeb;color:#d97706;font-size:11px;font-weight:700;border-radius:6px;padding:3px 10px'>👀 {_em_watch} Watch</span>"
            + (f"<span style='background:#fff7ed;color:#ea580c;font-size:11px;font-weight:700;border-radius:6px;padding:3px 10px'>⏳ {_em_wait} Wait</span>" if _em_wait else "")
            + (f"<span style='background:#fef2f2;color:#dc2626;font-size:11px;font-weight:700;border-radius:6px;padding:3px 10px'>⚠️ {_em_fade} Fading</span>" if _em_fade else "")
            + "</div></div>", unsafe_allow_html=True)

        # Entry guide (expiry-aware)
        if _is_expiry:
            _guide_bg    = "#451a03"
            _guide_steps = (
                f"<b style='color:#34d399'>Step 1:</b> Only look at <b>ENTER NOW</b> stocks — ignore WAIT and CAUTION.<br>"
                f"<b style='color:#34d399'>Step 2:</b> Check ✅ 3-candle confirmation badge — must be confirmed.<br>"
                f"<b style='color:#34d399'>Step 3:</b> SL = first candle LOW. Target = <b>{_expiry_info['target_multiplier']}× gap only</b> — book early.<br>"
                f"<b style='color:#fbbf24'>Expiry rule:</b> Exit by <b>2:30 PM</b> regardless of profit/loss."
            )
        else:
            _guide_bg    = "#1a2035"
            _guide_steps = (
                f"<b style='color:#34d399'>Step 1:</b> Look at ENTER NOW stocks only. Pick highest Vol× and gap%.<br>"
                f"<b style='color:#34d399'>Step 2:</b> Check current price is still near open (not already up 3% more).<br>"
                f"<b style='color:#34d399'>Step 3:</b> SL = first candle low. Target = 1.5× gap size. Exit if price falls below open."
            )
        st.markdown(
            f"<div style='background:{_guide_bg};border-radius:10px;padding:12px 18px;"
            f"margin-bottom:14px'>"
            f"<div style='font-size:12px;font-weight:700;color:#f59e0b;margin-bottom:6px'>"
            f"📋 How to trade {'(Expiry Mode)' if _is_expiry else '— 3 steps'}</div>"
            f"<div style='font-size:11px;color:rgba(255,255,255,0.7);line-height:2'>"
            f"{_guide_steps}</div></div>", unsafe_allow_html=True)

        # ── Result cards ──────────────────────────────────
        for _rank_em, _em in enumerate(_em_results[:15], 1):
            _gc  = "#16a34a" if _em['day_chg'] >= 0 else "#dc2626"
            _fc  = "#16a34a" if _em['holding'] else "#dc2626"
            _fl  = f"+{_em['fade_pct']:.2f}% holding" if _em['holding'] else f"{_em['fade_pct']:.2f}% fading"
            _vi  = ("🏦" if _em['vol_x'] >= 15 else "🔥" if _em['vol_x'] >= 8
                    else "⚡" if _em['vol_x'] >= 5 else "↑")
            _rb  = {1:"#f59e0b",2:"#94a3b8",3:"#b45309"}.get(_rank_em,"#e2e8f0")
            _rt  = {1:"#1a2035",2:"#ffffff",3:"#ffffff"}.get(_rank_em,"#64748b")

            # Extra badges for expiry
            _extra_badges = ""
            if _is_expiry:
                if _em['three_candle']:
                    _extra_badges += "<span style='background:#dcfce7;color:#15803d;font-size:10px;font-weight:700;border-radius:4px;padding:2px 7px'>✅ 3-candle confirmed</span> "
                elif _em['n_candles'] >= 3:
                    _extra_badges += "<span style='background:#fef3c7;color:#92400e;font-size:10px;font-weight:700;border-radius:4px;padding:2px 7px'>⏳ Not confirmed yet</span> "
                if _em['is_banking']:
                    _extra_badges += "<span style='background:#fef2f2;color:#dc2626;font-size:10px;font-weight:700;border-radius:4px;padding:2px 7px'>🏦 Pinning risk</span> "
                if _em['time_blocked']:
                    _extra_badges += "<span style='background:#fef2f2;color:#dc2626;font-size:10px;font-weight:700;border-radius:4px;padding:2px 7px'>⏰ Before 10 AM</span> "

            # Card border — red for banking/wait on expiry, normal otherwise
            _card_bdr = ("#fecaca" if (_is_expiry and (_em['is_banking'] or _em['time_blocked']))
                         else "#e8ecf3")

            _ec1, _ec2 = st.columns([5, 1])
            with _ec1:
                st.markdown(f"""
                <div style='background:#ffffff;border:1.5px solid {_card_bdr};
                            border-radius:14px;padding:16px 18px;margin-bottom:8px;
                            box-shadow:0 2px 8px rgba(0,0,0,0.04)'>
                    <div style='display:flex;align-items:flex-start;
                                justify-content:space-between;flex-wrap:wrap;gap:8px'>
                        <div style='display:flex;align-items:center;gap:12px'>
                            <div style='background:{_rb};color:{_rt};width:32px;height:32px;
                                        border-radius:50%;display:flex;align-items:center;
                                        justify-content:center;font-size:14px;
                                        font-weight:800;flex-shrink:0'>{_rank_em}</div>
                            <div>
                                <div style='display:flex;align-items:center;gap:8px;flex-wrap:wrap'>
                                    <span style='font-size:20px;font-weight:800;color:#1a2035'>
                                        {_em['sym_clean']}
                                    </span>
                                    <span style='background:{_em["action_bg"]};
                                                 color:{_em["action_clr"]};
                                                 font-size:12px;font-weight:700;
                                                 border-radius:6px;padding:3px 10px'>
                                        {_em['action']}
                                    </span>
                                    {_extra_badges}
                                    <span style='font-size:10px;color:#94a3b8'>
                                        {'⚡ Kite' if _em['src']=='kite' else '⏳ yfinance'}
                                        &nbsp;·&nbsp; {_em['n_candles']} candles today
                                    </span>
                                </div>
                                <div style='font-size:12px;color:#64748b;margin-top:4px'>
                                    Prev ₹{_em['prev_close']:,.2f}
                                    &nbsp;·&nbsp; Open ₹{_em['open_price']:,.2f}
                                    &nbsp;·&nbsp; Now ₹{_em['curr_price']:,.2f}
                                    {'&nbsp;·&nbsp; <b style="color:#d97706">Gap fill 65%</b>' if _is_expiry else ''}
                                </div>
                            </div>
                        </div>
                        <div style='text-align:right'>
                            <div style='font-size:22px;font-weight:800;color:#1a2035;
                                        font-family:JetBrains Mono'>₹{_em['curr_price']:,.2f}</div>
                            <div style='font-size:13px;font-weight:700;color:{_gc}'>
                                {'+' if _em['day_chg']>=0 else ''}{_em['day_chg']:.2f}% from yesterday
                            </div>
                        </div>
                    </div>

                    <div style='display:flex;gap:8px;margin-top:12px;flex-wrap:wrap'>
                        <div style='background:#f5f3ff;border-radius:8px;padding:8px 14px;text-align:center;min-width:80px'>
                            <div style='font-size:9px;font-weight:700;color:#7c3aed;letter-spacing:1px'>GAP UP</div>
                            <div style='font-size:18px;font-weight:800;color:#7c3aed;font-family:JetBrains Mono'>
                                +{_em['gap_pct']:.2f}%
                            </div>
                        </div>
                        <div style='background:#fffbeb;border-radius:8px;padding:8px 14px;text-align:center;min-width:80px'>
                            <div style='font-size:9px;font-weight:700;color:#d97706;letter-spacing:1px'>FIRST VOL</div>
                            <div style='font-size:18px;font-weight:800;color:#d97706;font-family:JetBrains Mono'>
                                {_vi}{_em['vol_x']:.1f}×
                            </div>
                        </div>
                        <div style='background:#f0f9ff;border-radius:8px;padding:8px 14px;text-align:center;min-width:80px'>
                            <div style='font-size:9px;font-weight:700;color:#0369a1;letter-spacing:1px'>STRENGTH</div>
                            <div style='font-size:18px;font-weight:800;color:#0369a1;font-family:JetBrains Mono'>
                                {_em['strength']:.0f}
                            </div>
                        </div>
                        <div style='background:{_em["action_bg"]};border-radius:8px;padding:8px 14px;text-align:center;min-width:100px;flex:1'>
                            <div style='font-size:9px;font-weight:700;color:{_em["action_clr"]};letter-spacing:1px'>GAP STATUS</div>
                            <div style='font-size:13px;font-weight:700;color:{_fc};margin-top:2px'>{_fl}</div>
                        </div>
                        <div style='background:#fff5f5;border-radius:8px;padding:8px 14px;text-align:center;min-width:100px'>
                            <div style='font-size:9px;font-weight:700;color:#dc2626;letter-spacing:1px'>
                                SL {'(1st candle low)' if _is_expiry else '(OPEN PRICE)'}
                            </div>
                            <div style='font-size:14px;font-weight:800;color:#dc2626;font-family:JetBrains Mono'>
                                ₹{_em['open_price']:,.2f}
                            </div>
                            <div style='font-size:9px;color:#dc2626'>exit if price falls below</div>
                        </div>
                        <div style='background:#f0fdf4;border-radius:8px;padding:8px 14px;text-align:center;min-width:100px'>
                            <div style='font-size:9px;font-weight:700;color:#15803d;letter-spacing:1px'>
                                TARGET ({_em['target_lbl']})
                            </div>
                            <div style='font-size:14px;font-weight:800;color:#15803d;font-family:JetBrains Mono'>
                                ₹{_em['target_px']:,.2f}
                            </div>
                            <div style='font-size:9px;color:#15803d'>
                                {'Gap fill prob: ' + str(_em['gap_fill_prob']) + '%' if _is_expiry else 'R:R ≈ 1.5:1'}
                            </div>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)

            with _ec2:
                st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

                # Only show Paper Buy for ENTER NOW signals
                _em_sym     = _em['sym_clean']
                _em_entry   = _em['curr_price']
                _em_sl      = _em['open_price']   # SL = open price (gap must hold)
                _em_target  = _em['target_px']    # T1 = 2× gap
                _em_t2      = _em.get('target_t2', _em_target)   # T2 = 3× gap
                _em_t3      = _em.get('target_t3', _em_target)   # T3 = 4× gap
                _em_gap_amt = _em.get('gap_amt', _em['open_price'] - _em['prev_close'])
                # Qty from capital and risk (risk = distance from entry to SL)
                _em_risk_d  = max(_em_entry - _em_sl, 0.01)
                _em_qty     = max(1, int((capital * risk_pct / 100) / _em_risk_d))

                if 'ENTER' in _em['action']:
                    if st.button(
                        f"✅ Paper Buy",
                        key=f"em_paper_buy_{_em_sym}_{_rank_em}",
                        use_container_width=True,
                        type="primary",
                    ):
                        _port = load_portfolio()
                        _already = any(
                            p.get('symbol') == _em_sym and p.get('status') == 'OPEN'
                            for p in _port
                        )
                        if _already:
                            st.warning(f"⚠️ Already open: {_em_sym}")
                        else:
                            _port.append({
                                'symbol':      _em_sym,
                                'status':      'OPEN',
                                'entry':       round(_em_entry, 2),
                                'qty':         _em_qty,
                                'stop_loss':   round(_em_sl, 2),
                                't1':          round(_em_target, 2),
                                't2':          round(_em_t2, 2),
                                't3':          round(_em_t3, 2),
                                't3':          0, 't4': 0,
                                'investment':  round(_em_entry * _em_qty, 2),
                                'actual_cost': round(_em_entry * _em_qty, 2),
                                'timeframe':   '1min — Early Mover',
                                'date':        ist_now().strftime('%d %b %Y %H:%M'),
                                'entry_time':  ist_now().strftime('%H:%M'),
                                'nifty_state': st.session_state.get('nifty_market_state', 'UNKNOWN'),
                                'vix_level':   st.session_state.get('nifty_context', {}).get('vix_level', 'UNKNOWN'),
                                'score':       0,
                                'verdict':     _em['action'],
                                'gap_pct':     _em['gap_pct'],
                                'vol_ratio':   _em['vol_x'],
                                'source':      'early_movers',
                                'exit_reason': '',
                            })
                            save_portfolio(_port)
                            st.session_state['paper_portfolio'] = _port
                            st.success(
                                f"✅ Bought {_em_qty} × {_em_sym} @ ₹{_em_entry:,.2f} · "
                                f"SL ₹{_em_sl:,.2f} · T1 ₹{_em_target:,.2f}"
                            )
                            st.rerun()
                else:
                    if st.button(f"🔬 Analyse", key=f"em_analyse_{_em_sym}_{_rank_em}",
                                 use_container_width=True):
                        st.session_state['_focus_stock'] = _em_sym
                        st.session_state['active_page']  = "📊  Scanner"
                        st.rerun()

        # ── Fading stocks ─────────────────────────────────
        _fading = [r for r in _em_results if 'FADING' in r['action']]
        if _fading:
            st.markdown("<hr style='border:none;border-top:1px solid #e2e8f0;margin:12px 0'>",
                        unsafe_allow_html=True)
            st.markdown(
                f"<div style='font-size:13px;font-weight:700;color:#dc2626;margin-bottom:8px'>"
                f"⚠️ {len(_fading)} stocks gap-up criteria met but now FADING — avoid</div>",
                unsafe_allow_html=True)
            for _fd in _fading:
                _fd_warn = " · 🏦 Banking pinning" if (_is_expiry and _fd['is_banking']) else ""
                st.markdown(
                    f"<div style='background:#fff5f5;border:1px solid #fecaca;"
                    f"border-radius:8px;padding:8px 14px;margin-bottom:4px;"
                    f"display:flex;justify-content:space-between;font-size:12px'>"
                    f"<span style='font-weight:700;color:#dc2626'>{_fd['sym_clean']}</span>"
                    f"<span style='color:#dc2626'>Gap +{_fd['gap_pct']:.2f}% but "
                    f"{_fd['fade_pct']:.2f}% below open{_fd_warn}</span>"
                    f"</div>", unsafe_allow_html=True)

        # ── Refresh ───────────────────────────────────────
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button("🔄 Refresh Early Movers", key="em_refresh",
                     use_container_width=True):
            st.session_state.pop('early_movers', None)
            st.rerun()



    # ── How it works explanation ──────────────────────────
    _now_em    = ist_now()
    _tm_em     = _now_em.hour * 60 + _now_em.minute
    _mkt_start = _now_em.replace(hour=9, minute=15, second=0, microsecond=0)
    try:
        _mins_since = int((_now_em - _mkt_start.astimezone(_now_em.tzinfo)).total_seconds() / 60)
    except Exception:
        _mins_since = 999

    # Window status
    if not market_open():
        _em_status     = "🔴 Market Closed — Run at 9:15 AM for live results"
        _em_status_clr = "#dc2626"
        _em_status_bg  = "#fef2f2"
    elif _mins_since <= 15:
        _em_status     = f"🟢 PRIME WINDOW — {_mins_since} min since open · Best time to catch moves"
        _em_status_clr = "#15803d"
        _em_status_bg  = "#f0fdf4"
    elif _mins_since <= 30:
        _em_status     = f"🟡 Good Window — {_mins_since} min since open · Most moves already started"
        _em_status_clr = "#d97706"
        _em_status_bg  = "#fffbeb"
    else:
        _em_status     = f"⚪ Late — {_mins_since} min since open · Early movers already ran · Use normal scanner"
        _em_status_clr = "#64748b"
        _em_status_bg  = "#f8fafc"
if _show_portfolio:
    _pf_kite_on  = get_kite_client() is not None
    _pf_src_lbl  = 'Kite API — Real-time' if _pf_kite_on else 'yfinance — 15 min delay'
    _pf_src_clr  = '#16a34a' if _pf_kite_on else '#d97706'
    _pf_src_bg   = '#dcfce7' if _pf_kite_on else '#fef3c7'
    _pf_src_ico  = '🟢' if _pf_kite_on else '🟡'
    st.markdown(f"""
    <div class='topbar'>
        <div>
            <div class='topbar-title'>💼 Intraday Paper Portfolio</div>
            <div class='topbar-subtitle'>Track your intraday paper trades</div>
        </div>
        <div style='display:flex;align-items:center'>
            <div style='background:{_pf_src_bg};border:1px solid {_pf_src_clr}44;
                        border-radius:8px;padding:5px 12px;text-align:center'>
                <div style='font-size:10px;font-weight:700;color:{_pf_src_clr};
                            letter-spacing:1px'>PRICE SOURCE</div>
                <div style='font-size:12px;font-weight:700;color:{_pf_src_clr};
                            margin-top:2px'>{_pf_src_ico} {_pf_src_lbl}</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    # ── Load portfolio first so open_pos is available everywhere ──
    port       = load_portfolio()
    open_pos   = [p for p in port if p.get('status') == 'OPEN']
    closed_pos = [p for p in port if p.get('status') != 'OPEN']

    if not port:
        st.info("📭 No positions yet. Go to the scanner and click 'Paper Buy' on any stock.")
        st.stop()

    # ── Portfolio Refresh Button ──────────────────────────
    _pf_col1, _pf_col2, _pf_col3 = st.columns([3, 1, 1])
    with _pf_col2:
        _pf_refresh = st.button("🔄 Refresh P&L", key="portfolio_refresh",
                                 use_container_width=True, type="primary",
                                 help="Fetch live prices and recalculate all P&L")
    with _pf_col3:
        _pf_auto = st.toggle("Auto 30s", value=st.session_state.get('pf_auto_refresh', False),
                              key="pf_auto_toggle",
                              help="Auto-refresh portfolio P&L every 30 seconds")
        st.session_state['pf_auto_refresh'] = _pf_auto
    with _pf_col1:
        _pf_last = st.session_state.get('pf_last_refresh', '—')
        _pf_src  = st.session_state.get('pf_price_source', 'yfinance')
        _src_clr = '#16a34a' if _pf_src == 'kite' else '#d97706'
        st.markdown(f"<div style='font-size:11px;color:#94a3b8;padding:10px 0'>"
                    f"<span style='color:{_src_clr};font-weight:700'>{_pf_src.upper()}</span>"
                    f" · Last refreshed: {_pf_last}</div>",
                    unsafe_allow_html=True)

    # Auto-refresh every 30s if enabled and market open
    if _pf_auto and market_open():
        _pf_elapsed = time.time() - st.session_state.get('pf_last_refresh_ts', 0)
        if _pf_elapsed >= 30:
            _pf_refresh = True

    # Fetch all live prices at once on refresh
    _pf_prices = st.session_state.get('pf_live_prices', {})
    if _pf_refresh or not _pf_prices:
        _fetch_spinner = st.empty()
        _fetch_spinner.markdown(
            "<div style='font-size:12px;color:#64748b;padding:4px 0'>⏳ Fetching live prices...</div>",
            unsafe_allow_html=True)
        _new_prices = {}
        for p in open_pos:
            _s = p.get('symbol', '')
            if _s and _s not in _new_prices:
                try:
                    # ── Kite first (real-time LTP) ────────────────────
                    _kite_pf = get_kite_client()
                    if _kite_pf is not None:
                        try:
                            _token = get_instrument_token(_kite_pf, _s + '.NS' if not _s.endswith('.NS') else _s)
                            if _token is not None:
                                _ltp_data = _kite_pf.ltp([f'NSE:{_s}'])
                                _ltp_key  = f'NSE:{_s}'
                                if _ltp_key in _ltp_data:
                                    _new_prices[_s] = float(_ltp_data[_ltp_key]['last_price'])
                                    continue
                        except Exception:
                            pass  # fall through to yfinance
                    # ── yfinance fallback ─────────────────────────────
                    _ticker_s = _s + '.NS' if not _s.endswith('.NS') else _s
                    _ph = yf.Ticker(_ticker_s).history(period='1d', interval='1m')
                    _new_prices[_s] = float(_ph['Close'].iloc[-1]) if not _ph.empty else _f(p.get('entry', 0))
                except Exception:
                    _new_prices[_s] = _f(p.get('entry', 0))
        st.session_state['pf_live_prices']    = _new_prices
        st.session_state['pf_last_refresh']   = ist_now().strftime('%H:%M:%S IST')
        st.session_state['pf_last_refresh_ts']= time.time()
        st.session_state['pf_price_source']   = 'kite' if get_kite_client() is not None else 'yfinance'
        _pf_prices = _new_prices
        _fetch_spinner.empty()
        # If auto-refresh triggered this fetch → rerun to update UI
        # then sleep 30s before next rerun cycle
        if _pf_auto and market_open():
            time.sleep(30)
            st.rerun()

    # Calculate total P&L using live prices
    total_inv    = sum(_f(p.get('actual_cost', _f(p.get('investment',0)))) for p in open_pos)
    total_unreal = sum(
        (_pf_prices.get(p.get('symbol',''), _f(p.get('entry',0))) - _f(p.get('entry',0)))
        * int(_f(p.get('qty',0)))
        for p in open_pos
    )
    total_unreal_pct = (total_unreal / total_inv * 100) if total_inv > 0 else 0
    pnl_color = "#16a34a" if total_unreal >= 0 else "#dc2626"
    pnl_sign  = "+" if total_unreal >= 0 else ""

    # Closed P&L
    closed_pnl = sum(_f(p.get('net_pl', 0)) for p in closed_pos)

    pf1, pf2, pf3, pf4 = st.columns(4)
    for _col, _label, _val, _cls, _sub in [
        (pf1, "Open Positions",  len(open_pos),                      "stat-green", "Active trades"),
        (pf2, "Total Exposure",  f"₹{total_inv:,.0f}",               "",           "Across open positions"),
        (pf3, "Unrealised P&L",  f"{pnl_sign}₹{total_unreal:,.0f}", "",           f"{pnl_sign}{total_unreal_pct:.2f}%"),
        (pf4, "Closed P&L",      f"{'+' if closed_pnl>=0 else ''}₹{closed_pnl:,.0f}", "", f"{len(closed_pos)} closed trades"),
    ]:
        _v_color = pnl_color if _label == "Unrealised P&L" else ("#16a34a" if _label == "Open Positions" else ("#16a34a" if closed_pnl >= 0 and _label == "Closed P&L" else "#dc2626" if closed_pnl < 0 and _label == "Closed P&L" else "#1a2035"))
        with _col:
            st.markdown(f"""<div class='stat-card'>
                <div class='stat-label'>{_label}</div>
                <div class='stat-value {_cls}' style='color:{_v_color};font-size:22px'>{_val}</div>
                <div class='stat-sub'>{_sub}</div>
            </div>""", unsafe_allow_html=True)

    # ── Manual Add Position ───────────────────────────
    with st.expander("➕ Add Position Manually", expanded=False):
        st.markdown(
            "<div style='font-size:12px;color:#64748b;margin-bottom:12px'>"
            "Add any stock manually — intraday, swing, delivery or just tracking"
            "</div>", unsafe_allow_html=True)

        _ma_c1, _ma_c2, _ma_c3 = st.columns(3)
        with _ma_c1:
            _ma_sym = st.text_input(
                "Stock Symbol", placeholder="e.g. HINDZINC or ATHERENERG",
                key="manual_add_sym",
                help="NSE symbol without .NS")
        with _ma_c2:
            _ma_entry = st.number_input(
                "Entry Price ₹", min_value=0.1, value=100.0,
                step=0.5, format="%.2f", key="manual_add_entry")
        with _ma_c3:
            _ma_qty = st.number_input(
                "Quantity (shares)", min_value=1, value=1,
                step=1, key="manual_add_qty")

        _ma_c4, _ma_c5, _ma_c6 = st.columns(3)
        with _ma_c4:
            _ma_sl = st.number_input(
                "Stop Loss ₹", min_value=0.0, value=0.0,
                step=0.5, format="%.2f", key="manual_add_sl",
                help="Leave 0 if not set")
        with _ma_c5:
            _ma_t1 = st.number_input(
                "Target T1 ₹", min_value=0.0, value=0.0,
                step=0.5, format="%.2f", key="manual_add_t1",
                help="Leave 0 if not set")
        with _ma_c6:
            _ma_t2 = st.number_input(
                "Target T2 ₹", min_value=0.0, value=0.0,
                step=0.5, format="%.2f", key="manual_add_t2",
                help="Leave 0 if not set")

        _ma_c7, _ma_c8, _ma_c9 = st.columns(3)
        with _ma_c7:
            _ma_type = st.selectbox(
                "Trade Type",
                ["Intraday", "SMA Weekly (3-7 days)",
                 "Monthly Swing (3-5 weeks)", "Delivery (Long term)"],
                key="manual_add_type")
        with _ma_c8:
            _ma_date = st.text_input(
                "Entry Date", value=ist_now().strftime('%d %b %Y'),
                key="manual_add_date",
                help="When did you buy?")
        with _ma_c9:
            _ma_note = st.text_input(
                "Notes (optional)", placeholder="e.g. Bought on breakout",
                key="manual_add_note")

        # Type → source mapping
        _ma_src_map = {
            "Intraday":                  "manual_intraday",
            "SMA Weekly (3-7 days)":     "sma_weekly",
            "Monthly Swing (3-5 weeks)": "monthly_swing",
            "Delivery (Long term)":      "manual_delivery",
        }

        if st.button("➕ Add to Portfolio", key="manual_add_btn",
                     use_container_width=True, type="primary"):
            _ma_sym_clean = _ma_sym.strip().upper().replace('.NS','')
            if not _ma_sym_clean:
                st.error("❌ Please enter a stock symbol")
            elif _ma_entry <= 0:
                st.error("❌ Entry price must be greater than 0")
            elif _ma_qty < 1:
                st.error("❌ Quantity must be at least 1")
            else:
                _ma_port = load_portfolio()
                _ma_inv  = round(_ma_entry * _ma_qty, 2)
                _ma_port.append({
                    'symbol':      _ma_sym_clean,
                    'status':      'OPEN',
                    'entry':       round(_ma_entry, 2),
                    'qty':         int(_ma_qty),
                    'stop_loss':   round(_ma_sl, 2),
                    't1':          round(_ma_t1, 2),
                    't2':          round(_ma_t2, 2),
                    't3':          0,
                    't4':          0,
                    'investment':  _ma_inv,
                    'actual_cost': _ma_inv,
                    'timeframe':   _ma_type,
                    'date':        _ma_date,
                    'entry_time':  ist_now().strftime('%H:%M'),
                    'nifty_state': 'MANUAL',
                    'vix_level':   'MANUAL',
                    'score':       0,
                    'verdict':     'Manual Entry',
                    'vol_ratio':   0,
                    'source':      _ma_src_map.get(_ma_type, 'manual'),
                    'exit_reason': '',
                    'notes':       _ma_note.strip(),
                })
                save_portfolio(_ma_port)
                st.session_state['paper_portfolio'] = _ma_port
                st.session_state.pop('pf_live_prices', None)  # force price refresh
                st.success(
                    f"✅ Added {_ma_sym_clean} · {int(_ma_qty)} shares @ ₹{_ma_entry:,.2f} · "
                    f"Investment ₹{_ma_inv:,.0f} · {_ma_type}")
                st.rerun()

    st.markdown("<div class='section-header'>📋 Open Positions</div>", unsafe_allow_html=True)

    for _pf_idx, p in enumerate(open_pos):
        sym_c  = p.get('symbol', '')
        entry  = _f(p.get('entry', 0))
        qty    = int(_f(p.get('qty', 0)))
        sl     = _f(p.get('stop_loss', 0))
        t1     = _f(p.get('t1', 0)); t2 = _f(p.get('t2', 0))
        t3     = _f(p.get('t3', 0)); t4 = _f(p.get('t4', 0))
        actual = _f(p.get('actual_cost', _f(p.get('investment',0))))

        # ── Detect if this is a swing position ───────────
        _is_swing   = p.get('source','') in ('monthly_swing','sma_weekly')
        _is_monthly = p.get('source','') == 'monthly_swing'
        _src_lbl    = ('📅 Monthly Swing' if _is_monthly
                       else '📈 SMA Weekly' if p.get('source','')=='sma_weekly'
                       else '⚡ Intraday' if p.get('source','').startswith('manual_intraday')
                       else '📦 Delivery' if p.get('source','').startswith('manual_delivery')
                       else '🏷️ Manual')

        # ── Fetch PSAR for swing positions ────────────────
        _sp_psar   = None
        _sp_psar_b = False
        _sp_signal = ''
        _sp_guidance = ''
        _sp_psar_clr = '#64748b'
        _sp_psar_bg  = '#f8fafc'
        _sp_psar_bdr = '#e2e8f0'
        _t1_hit      = False

        if _is_swing:
            try:
                import yfinance as _yf_pf
                _sp_ticker = _yf_pf.Ticker(sym_c + '.NS')
                _sp_df_d = _sp_ticker.history(period='5d', interval='1d',
                                              auto_adjust=True, actions=False)
                if _sp_df_d is not None and len(_sp_df_d) > 0:
                    _sp_df_d.columns = [c.split(' ')[0] if ' ' in str(c)
                                        else c for c in _sp_df_d.columns]
                    _sp_live = round(float(_sp_df_d['Close'].iloc[-1]), 2)
                else:
                    _sp_live = None

                if _is_monthly:
                    _sp_wdf = _sp_ticker.history(period='1y', interval='1wk',
                                                 auto_adjust=True, actions=False)
                    if _sp_wdf is not None and len(_sp_wdf) >= 10:
                        _sp_wdf.columns = [c.split(' ')[0] if ' ' in str(c)
                                           else c for c in _sp_wdf.columns]
                        _sp_wdf  = _sp_wdf[['Open','High','Low','Close','Volume']].dropna()
                        _sp_dfps = calc_psar(_sp_wdf.copy(), step=0.01, max_af=0.10)
                        _sp_psar = round(float(_sp_dfps['PSAR'].iloc[-1]), 2)
                        _sp_psar_b = bool(_sp_dfps['PSAR_bull'].iloc[-1])
                else:
                    _sp_ddf = _sp_ticker.history(period='1y', interval='1d',
                                                 auto_adjust=True, actions=False)
                    if _sp_ddf is not None and len(_sp_ddf) >= 20:
                        _sp_ddf.columns = [c.split(' ')[0] if ' ' in str(c)
                                           else c for c in _sp_ddf.columns]
                        _sp_ddf  = _sp_ddf[['Open','High','Low','Close','Volume']].dropna()
                        _sp_dfps = calc_psar(_sp_ddf.copy(), step=0.02, max_af=0.20)
                        _sp_psar = round(float(_sp_dfps['PSAR'].iloc[-1]), 2)
                        _sp_psar_b = bool(_sp_dfps['PSAR_bull'].iloc[-1])

                if _sp_psar and _sp_live:
                    _sp_psar_b = _sp_psar_b and _sp_live > _sp_psar
                    _t1_hit    = _sp_live >= t1 if t1 > 0 else False
                    _psar_gap  = round((_sp_live - _sp_psar) / _sp_live * 100, 1)
                    _psar_type = 'Weekly' if _is_monthly else 'Daily'
                    _check_freq= 'every Friday' if _is_monthly else 'every morning'

                    if not _t1_hit:
                        _sp_psar_clr = '#1d4ed8'; _sp_psar_bg='#eff6ff'; _sp_psar_bdr='#93c5fd'
                        _sp_signal   = f'⏳ Waiting for T1 ₹{t1:,.2f}'
                        _sp_guidance = (f'Needs +{round((t1-_sp_live)/_sp_live*100,1)}% to hit T1 · '
                                        f'Use original SL ₹{sl:,.2f} · PSAR activates after T1')
                    elif _sp_psar_b:
                        _sp_psar_clr = '#15803d'; _sp_psar_bg='#f0fdf4'; _sp_psar_bdr='#86efac'
                        _sp_signal   = '✅ HOLD — Above PSAR'
                        _sp_guidance = (f'T1 ✅ hit · Trail SL → ₹{_sp_psar:,.2f} · '
                                        f'Update Zerodha SL · Check {_check_freq}')
                    else:
                        _sp_psar_clr = '#dc2626'; _sp_psar_bg='#fef2f2'; _sp_psar_bdr='#fca5a5'
                        _sp_signal   = '🔴 EXIT — PSAR crossed after T1'
                        _sp_guidance = (f'T1 ✅ hit earlier · Below PSAR now · '
                                        f'Exit to lock profit · PSAR = ₹{_sp_psar:,.2f}')
            except Exception:
                pass

        # Use cached live price
        cur        = _pf_prices.get(sym_c, entry)
        unreal     = (cur - entry) * qty
        unreal_pct = (unreal / actual * 100) if actual else 0
        pl_color   = "#16a34a" if unreal >= 0 else "#dc2626"
        pl_sign    = "+" if unreal >= 0 else ""

        # ── Auto-sell session keys ──
        _as_key_pct     = f"autosell_pct_{sym_c}_{_pf_idx}"
        _as_key_enabled = f"autosell_on_{sym_c}_{_pf_idx}"
        _default_tp_pct = 2.0
        _default_sl_pct = 1.0
        _as_tp_pct  = max(0.1, float(st.session_state.get(_as_key_pct + '_tp', _default_tp_pct)))
        _as_sl_pct  = max(0.1, float(st.session_state.get(_as_key_pct + '_sl', _default_sl_pct)))
        _as_enabled = st.session_state.get(_as_key_enabled, False)

        # ── Check auto-sell trigger ──
        _auto_triggered = False
        _auto_reason    = ""
        if _as_enabled and cur > 0 and entry > 0:
            _cur_pct = (cur - entry) / entry * 100
            if _cur_pct >= _as_tp_pct:
                _auto_triggered = True; _auto_reason = 'T1_HIT'
            elif _cur_pct <= -_as_sl_pct:
                _auto_triggered = True; _auto_reason = 'SL_HIT'

        if _auto_triggered:
            for _p in port:
                if _p.get('symbol') == sym_c and _p.get('status') == 'OPEN':
                    _p['status']     = 'CLOSED'
                    _p['exit_price'] = round(cur, 2)
                    _p['net_pl']     = round(unreal, 2)
                    _p['exit_date']  = ist_now().strftime('%d %b %Y %H:%M IST')
                    _p['exit_reason']= _auto_reason
                    break
            save_portfolio(port)
            st.session_state['paper_portfolio'] = port
            st.success(f"🤖 AUTO SELL — {sym_c} @ ₹{cur:,.2f} · {_auto_reason} · "
                       f"P&L: {pl_sign}₹{unreal:,.2f} ({pl_sign}{unreal_pct:.2f}%)")
            st.rerun()

        # ── SL hit banner ─────────────────────────────────
        _pf_sl_hit = sl > 0 and cur <= sl
        if _pf_sl_hit:
            st.markdown(
                f"<div style='background:#7f1d1d;border:2px solid #dc2626;"
                f"border-radius:12px;padding:12px 18px;margin-bottom:8px'>"
                f"<div style='font-size:15px;font-weight:800;color:#fca5a5'>"
                f"🛑 STOP LOSS HIT — EXIT {sym_c} IMMEDIATELY</div>"
                f"<div style='font-size:12px;color:#fca5a5;margin-top:4px'>"
                f"Current ₹{cur:,.2f} · SL was ₹{sl:,.2f} · "
                f"Loss: ₹{abs(unreal):,.0f} · Click Square Off below</div>"
                f"</div>", unsafe_allow_html=True)

        # ── F&O Expiry Warning ────────────────────────────
        _pf_fno_info = get_fno_info(sym_c)
        _pf_zone     = _pf_fno_info['expiry_zone']
        _pf_is_fno   = _pf_fno_info['is_fno']
        _pf_dte      = _pf_fno_info['days_to_exp']
        if _pf_is_fno and _pf_zone in ('DANGER', 'CAUTION'):
            _pf_warn_bg  = '#fef2f2' if _pf_zone == 'DANGER' else '#fffbeb'
            _pf_warn_bdr = '#fca5a5' if _pf_zone == 'DANGER' else '#fde68a'
            _pf_warn_clr = '#dc2626' if _pf_zone == 'DANGER' else '#d97706'
            _pf_warn_ico = '🔴' if _pf_zone == 'DANGER' else '⚠️'
            _pf_warn_title = (f'Expiry in {_pf_dte} days — Price may be pinned'
                              if _pf_zone == 'DANGER'
                              else f'Expiry in {_pf_dte} days — Approaching expiry')
            _pf_warn_msg = ('F&O expiry week · Stock may be pinned near current price · '
                            'Targets may not hit this week · '
                            'Consider tightening SL to PSAR level'
                            if _pf_zone == 'DANGER'
                            else 'Second half of month · Slower movement expected · '
                                 'Reduce expectations for this week')
            st.markdown(
                f"<div style='background:{_pf_warn_bg};border:1.5px solid {_pf_warn_bdr};"
                f"border-radius:8px;padding:8px 14px;margin-bottom:6px'>"
                f"<div style='font-size:11px;font-weight:700;color:{_pf_warn_clr}'>"
                f"{_pf_warn_ico} {_pf_warn_title}</div>"
                f"<div style='font-size:10px;color:{_pf_warn_clr};margin-top:2px'>"
                f"📌 F&O listed stock · {_pf_warn_msg}</div>"
                f"</div>", unsafe_allow_html=True)
        elif _pf_is_fno and _pf_zone == 'FRESH':
            st.markdown(
                f"<div style='background:#f0fdf4;border:1.5px solid #86efac;"
                f"border-radius:8px;padding:8px 14px;margin-bottom:6px'>"
                f"<div style='font-size:11px;font-weight:700;color:#15803d'>"
                f"🟢 Post-Expiry — Fresh F&O cycle started</div>"
                f"<div style='font-size:10px;color:#15803d;margin-top:2px'>"
                f"📌 F&O listed · New positions being built · "
                f"Normal movement expected · Good time to hold</div>"
                f"</div>", unsafe_allow_html=True)


        # ── Progress bar ──────────────────────────────────
        _t2_pct  = round((t2 - entry) / entry * 100, 2) if entry > 0 and t2 > 0 else 2.0
        _sl_pct  = round((entry - sl)  / entry * 100, 2) if entry > 0 and sl > 0 else 0.5
        _cur_pct2= round((cur - entry) / entry * 100, 2) if entry > 0 else 0
        _bar_rng = (_t2_pct + _sl_pct) or 1
        _bar_pct = min(100, max(0, int((_cur_pct2 + _sl_pct) / _bar_rng * 100)))
        _bar_clr = "#16a34a" if _cur_pct2 >= 0 else "#dc2626"
        _au_badge= ("<span style='font-size:10px;font-weight:700;color:#7c3aed;"
                    "background:#f5f3ff;border-radius:20px;padding:2px 8px'>AUTO</span>"
                    if _as_enabled else "")

        # ── Target boxes ──────────────────────────────────
        _sl_hit2 = cur <= sl and sl > 0
        _t_html  = (
            "<div style='background:" + ("#fef2f2" if _sl_hit2 else "#fff5f5") +
            ";border-radius:8px;padding:8px 12px;flex:1;min-width:72px;text-align:center'>"
            "<div style='font-size:9px;color:#dc2626;font-weight:700'>SL</div>"
            "<div style='font-size:13px;font-weight:800;color:#dc2626;font-family:JetBrains Mono'>₹" +
            "{:,.2f}".format(sl) + "</div>"
            "<div style='font-size:9px;color:#dc2626'>" +
            ("🚨HIT" if _sl_hit2 else f"-{_sl_pct:.1f}%") + "</div></div>"
        ) if sl > 0 else ""

        for _tv, _tlbl in [(t1,"T1"),(t2,"T2"),(t3,"T3"),(t4,"T4")]:
            if _tv <= 0: continue
            _thit = cur >= _tv
            _tbg  = "#dcfce7" if _thit else "#f0fdf4"
            _tdsp = "✅HIT" if _thit else f"+{round((_tv-entry)/entry*100,1)}%"
            _t_html += (
                "<div style='background:" + _tbg +
                ";border-radius:8px;padding:8px 12px;flex:1;min-width:72px;text-align:center'>"
                "<div style='font-size:9px;color:#15803d;font-weight:700'>" + _tlbl + "</div>"
                "<div style='font-size:13px;font-weight:800;color:#15803d;font-family:JetBrains Mono'>₹" +
                "{:,.2f}".format(_tv) + "</div>"
                "<div style='font-size:9px;color:#15803d'>" + _tdsp + "</div></div>"
            )

        # ── Build card border colour ───────────────────────
        _card_border = ('#fca5a5' if _pf_sl_hit
                        else _sp_psar_bdr if _is_swing
                        else '#e8ecf3')

        # ── Pre-built strings ─────────────────────────────
        _notes_str  = f" · 📝 {p.get('notes','')}" if p.get('notes') else ""
        _manual_str = " · 🏷️ Manual" if p.get('source','').startswith('manual') else ""
        _detail_line= (f"{qty} shares · Entry ₹{entry:,.2f} · "
                       f"{p.get('timeframe','INTRADAY')} · {p.get('date','')}"
                       f"{_notes_str}{_manual_str}")
        _pnl_line   = (f"{pl_sign}₹{unreal:,.0f} "
                       f"<span style='font-size:13px;font-weight:700'>"
                       f"({pl_sign}{unreal_pct:.2f}%)</span>")
        _bar_html   = (f"<div style='background:#f1f5f9;border-radius:4px;"
                       f"height:8px;overflow:hidden'>"
                       f"<div style='background:{_bar_clr};height:8px;border-radius:4px;"
                       f"width:{_bar_pct}%;transition:width 0.4s'></div></div>")

        # ── PSAR strip for swing ───────────────────────────
        if _is_swing and _sp_psar:
            _psar_val_str   = f"₹{_sp_psar:,.2f}"
            _psar_lbl_sfx   = "(T1 ✅ active)" if _t1_hit else "(activates after T1)"
            _psar_type_lbl  = 'Weekly PSAR' if _is_monthly else 'Daily PSAR'
            _psar_gap_show  = f"{round((_sp_live-_sp_psar)/_sp_live*100,1):+.1f}% from price" if _t1_hit and _sp_live else ""
            _psar_strip = (
                f"<div style='background:{_sp_psar_bg};border:1px solid {_sp_psar_bdr};"
                f"border-radius:8px;padding:10px 14px;margin-top:10px;"
                f"display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px'>"
                f"<div>"
                f"<div style='font-size:9px;font-weight:700;color:{_sp_psar_clr};"
                f"letter-spacing:1px'>📍 {_psar_type_lbl.upper()} — TRAILING SL {_psar_lbl_sfx}</div>"
                f"<div style='font-size:17px;font-weight:800;color:{_sp_psar_clr};"
                f"font-family:JetBrains Mono;margin-top:2px'>"
                f"{_psar_val_str} "
                f"<span style='font-size:10px;font-weight:600;color:{_sp_psar_clr}'>"
                f"{_psar_gap_show}</span></div>"
                f"<div style='font-size:10px;color:{_sp_psar_clr};margin-top:3px'>{_sp_guidance}</div>"
                f"</div>"
                f"<div style='font-size:12px;font-weight:800;color:{_sp_psar_clr};"
                f"padding:7px 14px;background:white;border-radius:8px;"
                f"border:2px solid {_sp_psar_bdr};white-space:nowrap'>{_sp_signal}</div>"
                f"</div>"
            )
        else:
            _psar_strip = ""

        # ── Source badge ──────────────────────────────────
        _src_badge_clr = ('#7c3aed' if _is_monthly else
                          '#1d4ed8' if _is_swing else '#16a34a')
        _src_badge_bg  = ('#f5f3ff' if _is_monthly else
                          '#eff6ff' if _is_swing else '#dcfce7')

        _h = (
            f"<div style='background:#ffffff;border:1.5px solid {_card_border};"
            f"border-radius:16px;padding:18px 20px;margin-bottom:14px'>"

            # ── Header row ──
            f"<div style='display:flex;justify-content:space-between;"
            f"align-items:flex-start;flex-wrap:wrap;gap:8px'>"
            f"<div>"
            f"<div style='display:flex;align-items:center;gap:8px;flex-wrap:wrap'>"
            f"<span style='font-size:19px;font-weight:800;color:#1a2035'>{sym_c}</span>"
            f"<span style='background:{_src_badge_bg};color:{_src_badge_clr};font-size:10px;"
            f"font-weight:700;border-radius:4px;padding:2px 8px'>{_src_lbl}</span>"
            f"<span style='font-size:10px;font-weight:700;color:#16a34a;background:#dcfce7;"
            f"border-radius:20px;padding:2px 8px'>OPEN</span>"
            f"{_au_badge}</div>"
            f"<div style='font-size:11px;color:#64748b;margin-top:3px'>{_detail_line}</div>"
            f"</div>"
            f"<div style='text-align:right'>"
            f"<div style='font-size:11px;color:#94a3b8'>Live</div>"
            f"<div style='font-size:22px;font-weight:800;color:#1a2035;"
            f"font-family:JetBrains Mono'>₹{cur:,.2f}</div>"
            f"<div style='font-size:15px;font-weight:800;color:{pl_color}'>{_pnl_line}</div>"
            f"</div></div>"

            # ── Progress bar ──
            f"<div style='margin:10px 0 4px'>"
            f"<div style='display:flex;justify-content:space-between;"
            f"font-size:10px;color:#94a3b8;margin-bottom:3px'>"
            f"<span>SL -{_sl_pct:.2f}%</span>"
            f"<span style='color:{_bar_clr};font-weight:700'>{pl_sign}{_cur_pct2:.2f}% now</span>"
            f"<span>T2 +{_t2_pct:.2f}%</span>"
            f"</div>{_bar_html}</div>"

            # ── Target boxes ──
            f"<div style='display:flex;gap:6px;margin-top:10px;flex-wrap:wrap'>{_t_html}</div>"

            # ── PSAR strip (swing only) ──
            f"{_psar_strip}"

            # ── Footer note ──
            f"<div style='margin-top:8px;padding:5px 10px;background:#fffbeb;"
            f"border-radius:6px;font-size:10px;color:#92400e'>"
            f"{'Check every Friday · Trail SL to PSAR after T1' if _is_monthly else 'Check every morning · Trail SL after T1' if _is_swing else 'Square off before 3:20 PM IST'}"
            f"</div>"
            f"</div>"
        )
        st.markdown(_h, unsafe_allow_html=True)

        with st.container():
            _hh_col1, _hh_col2 = st.columns([4, 1])
            with _hh_col2:
                if st.button(f"✅ Square Off", key=f"sq_{sym_c}_{_pf_idx}",
                             use_container_width=True):
                    for _p in port:
                        if _p.get('symbol') == sym_c and _p.get('status') == 'OPEN':
                            _p['status']     = 'CLOSED'
                            _p['exit_price'] = round(cur, 2)
                            _p['net_pl']     = round(unreal, 2)
                            _p['exit_date']  = ist_now().strftime('%d %b %Y %H:%M IST')
                            _p['exit_reason']= 'MANUAL_PROFIT' if unreal >= 0 else 'MANUAL_LOSS'
                            break
                    save_portfolio(port)
                    st.session_state['paper_portfolio'] = port
                    st.success(f"✅ {sym_c} @ ₹{cur:,.2f} · {pl_sign}₹{unreal:,.0f} ({pl_sign}{unreal_pct:.2f}%)")
                    st.rerun()
                st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
                _new_as = st.toggle("🤖 Auto sell", value=_as_enabled,
                                    key=f"as_toggle_{sym_c}_{_pf_idx}",
                                    help="Auto square off when TP% or SL% is hit")
                st.session_state[_as_key_enabled] = _new_as

            with _hh_col1:
                if st.session_state.get(_as_key_enabled, False):
                    _asp1, _asp2 = st.columns(2)
                    with _asp1:
                        _new_tp = st.number_input(
                            f"Take Profit %", min_value=0.1, max_value=10.0,
                            value=float(max(0.1, round(_as_tp_pct, 2))),
                            step=0.05, format="%.2f", key=f"as_tp_{sym_c}_{_pf_idx}")
                        st.session_state[_as_key_pct + '_tp'] = _new_tp
                        st.markdown(
                            f"<div style='font-size:11px;color:#7c3aed;margin-top:-8px'>"
                            f"Sell at ₹{round(entry*(1+_new_tp/100),2):,.2f}</div>",
                            unsafe_allow_html=True)
                    with _asp2:
                        _new_sl2 = st.number_input(
                            f"Stop Loss %", min_value=0.1, max_value=5.0,
                            value=float(max(0.1, round(_as_sl_pct, 2))),
                            step=0.05, format="%.2f", key=f"as_sl_{sym_c}_{_pf_idx}")
                        st.session_state[_as_key_pct + '_sl'] = _new_sl2
                        st.markdown(
                            f"<div style='font-size:11px;color:#dc2626;margin-top:-8px'>"
                            f"Sell at ₹{round(entry*(1-_new_sl2/100),2):,.2f}</div>",
                            unsafe_allow_html=True)


    # Closed positions
    if closed_pos:
        st.markdown("<div class='section-header'>📁 Closed / Squared Off</div>", unsafe_allow_html=True)
        closed_data = []
        for p in closed_pos:
            net    = _f(p.get('net_pl', 0))
            entry  = _f(p.get('entry', 0))
            exit_p = _f(p.get('exit_price', 0))
            actual = _f(p.get('actual_cost', _f(p.get('investment', 0)))) or 1
            net_pct= round(net / actual * 100, 2)
            move_pct = round((exit_p - entry) / entry * 100, 2) if entry > 0 else 0
            closed_data.append({
                'Symbol':      p.get('symbol',''),
                'Entry':       f"₹{entry:,.2f}",
                'Exit':        f"₹{exit_p:,.2f}",
                'Move':        f"{'+' if move_pct>=0 else ''}{move_pct:.2f}%",
                'Qty':         int(_f(p.get('qty',0))),
                'Net P&L':     f"{'+' if net>=0 else ''}₹{net:,.0f}",
                'Return %':    f"{'+' if net_pct>=0 else ''}{net_pct:.2f}%",
                'Reason':      p.get('exit_reason','Manual'),
                'Exit Time':   p.get('exit_date',''),
                'Result':      '✅ Profit' if net>=0 else '❌ Loss',
            })
        st.dataframe(pd.DataFrame(closed_data), use_container_width=True, hide_index=True)

        if st.button("🗑️ Clear All Closed Positions", key="clear_closed"):
            save_portfolio([p for p in port if p.get('status') == 'OPEN'])
            st.rerun()

# ─────────────────────────────────────────────
#  ALERT LOG PAGE
# ─────────────────────────────────────────────
if _show_alertlog:
    st.markdown("""
    <div class='topbar'>
        <div>
            <div class='topbar-title'>🔔 Alert Log</div>
            <div class='topbar-subtitle'>All buy, exit, target and stop loss alerts this session</div>
        </div>
    </div>""", unsafe_allow_html=True)

    _init_alert_log()
    all_alerts = st.session_state.get(ALERT_LOG_KEY, [])

    if not all_alerts:
        st.info("📭 No alerts yet. Run the scanner to start generating alerts.")
        st.stop()

    # Summary counts
    _al_buy  = sum(1 for a in all_alerts if a['type'] in ['BUY','STRONG_BUY'])
    _al_exit = sum(1 for a in all_alerts if a['type'] in ['VWAP_BREAK','RSI_OB','TIME_WARN'])
    _al_sl   = sum(1 for a in all_alerts if a['type'] == 'STOP_LOSS')
    _al_tgt  = sum(1 for a in all_alerts if 'TARGET' in a['type'])
    _al_vol  = sum(1 for a in all_alerts if a['type'] == 'VOL_SURGE')

    ac1, ac2, ac3, ac4, ac5 = st.columns(5)
    for _acol, _albl, _aval, _abg, _atc in [
        (ac1, "Buy Signals",    _al_buy,  "#dcfce7", "#15803d"),
        (ac2, "Exit Warnings",  _al_exit, "#fff7ed", "#c2410c"),
        (ac3, "Stop Loss Hits", _al_sl,   "#fef2f2", "#dc2626"),
        (ac4, "Targets Hit",    _al_tgt,  "#f0fdf4", "#15803d"),
        (ac5, "Vol Surges",     _al_vol,  "#fffbeb", "#d97706"),
    ]:
        with _acol:
            st.markdown(f"""<div style='background:{_abg};border-radius:12px;padding:16px 18px;
                            border:1px solid {_atc}33;text-align:center'>
                <div style='font-size:11px;font-weight:700;color:{_atc};letter-spacing:1px;text-transform:uppercase'>{_albl}</div>
                <div style='font-size:30px;font-weight:800;color:{_atc};font-family:JetBrains Mono,monospace'>{_aval}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Filter by type
    _ftypes = ["All"] + sorted(list(set(a['type'] for a in all_alerts)))
    _fsel   = st.selectbox("Filter by Alert Type", _ftypes, key="alert_filter")
    _falerts = all_alerts if _fsel == "All" else [a for a in all_alerts if a['type'] == _fsel]

    st.markdown(f"<div class='section-header'>📋 {len(_falerts)} Alerts</div>", unsafe_allow_html=True)

    _full_bg = {
        'BUY':        ('#f0fdf4','#15803d'),
        'STRONG_BUY': ('#dcfce7','#14532d'),
        'VOL_SURGE':  ('#fffbeb','#92400e'),
        'VWAP_BREAK': ('#fff5f5','#991b1b'),
        'RSI_OB':     ('#fff5f5','#991b1b'),
        'STOP_LOSS':  ('#fef2f2','#7f1d1d'),
        'TARGET_T1':  ('#f0fdf4','#15803d'),
        'TARGET_T2':  ('#f0fdf4','#15803d'),
        'TARGET_T3':  ('#ecfdf5','#065f46'),
        'TARGET_T4':  ('#ecfdf5','#065f46'),
        'TIME_WARN':  ('#fffbeb','#92400e'),
    }
    _full_icons = {
        'BUY':'🔔','STRONG_BUY':'🚨','VOL_SURGE':'⚡',
        'VWAP_BREAK':'⚠️','RSI_OB':'🔴','STOP_LOSS':'🛑',
        'TARGET_T1':'🎯','TARGET_T2':'🎯','TARGET_T3':'🎯','TARGET_T4':'🎯',
        'TIME_WARN':'🕒',
    }

    for _al in _falerts:
        _abg2, _atc2 = _full_bg.get(_al['type'], ('#f8fafc','#1a2035'))
        _aic2 = _full_icons.get(_al['type'], '📣')
        _lc1, _lc2 = st.columns([5, 1])
        with _lc1:
            st.markdown(f"""
            <div style='background:{_abg2};border:1.5px solid {_atc2}44;border-radius:12px;
                        padding:14px 20px;margin-bottom:8px'>
                <div style='display:flex;align-items:center;gap:12px;flex-wrap:wrap'>
                    <span style='font-size:22px'>{_aic2}</span>
                    <div style='flex:1'>
                        <div style='display:flex;align-items:center;gap:10px;flex-wrap:wrap'>
                            <span style='font-size:15px;font-weight:800;color:{_atc2}'>{_al["symbol"]}</span>
                            <span style='font-size:11px;background:{_atc2}22;color:{_atc2};
                                         border-radius:20px;padding:2px 10px;font-weight:700'>{_al["type"].replace("_"," ")}</span>
                            <span style='font-size:12px;color:#94a3b8'>{_al["time"]} · {_al["date"]}</span>
                        </div>
                        <div style='font-size:13px;color:#374151;margin-top:5px'>{_al["message"]}</div>
                    </div>
                    <div style='font-size:16px;font-weight:800;color:{_atc2};font-family:JetBrains Mono'>₹{_al["price"]:,.2f}</div>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑️ Clear All Alerts", key="clear_all_alerts"):
        st.session_state[ALERT_LOG_KEY] = []; st.rerun()

# ─────────────────────────────────────────────────────────────
#  SMA WEEKLY SCANNER
#  Short-term swing trades (3–7 days hold)
#  Strategy: SMA20 + SMA50 crossover on daily chart
#  ATR-based SL and targets
# ─────────────────────────────────────────────────────────────

if _show_smaweekly:

    # ── Data source badge ─────────────────────────────────
    _sw_kite_on  = get_kite_client() is not None
    _sw_src_lbl  = 'Kite API — Real-time' if _sw_kite_on else 'yfinance — Daily data'
    _sw_src_clr  = '#16a34a' if _sw_kite_on else '#d97706'
    _sw_src_bg   = '#dcfce7' if _sw_kite_on else '#fef3c7'
    _sw_src_ico  = '🟢' if _sw_kite_on else '🟡'

    st.markdown(f"""
    <div class='topbar'>
        <div>
            <div class='topbar-title'>📈 SMA Weekly — Short Term Swing Scanner</div>
            <div class='topbar-subtitle'>
                SMA 20 + SMA 50 crossover · Daily chart · Hold 3–7 days ·
                ATR-based SL and targets
            </div>
        </div>
        <div style='display:flex;align-items:center'>
            <div style='background:{_sw_src_bg};border:1px solid {_sw_src_clr}44;
                        border-radius:8px;padding:5px 12px;text-align:center'>
                <div style='font-size:10px;font-weight:700;color:{_sw_src_clr};
                            letter-spacing:1px'>DATA SOURCE</div>
                <div style='font-size:12px;font-weight:700;color:{_sw_src_clr};
                            margin-top:2px'>{_sw_src_ico} {_sw_src_lbl}</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    # ── Strategy explanation ──────────────────────────────
    st.markdown("""
    <div style='display:flex;gap:10px;margin-bottom:16px;flex-wrap:wrap'>
        <div style='flex:1;min-width:160px;background:#eff6ff;border-radius:10px;
                    padding:12px 14px;border:1px solid #bfdbfe44'>
            <div style='font-size:20px;margin-bottom:6px'>🔀</div>
            <div style='font-size:12px;font-weight:700;color:#1d4ed8'>Signal 1 — Fresh Cross</div>
            <div style='font-size:11px;color:#1e40af;margin-top:3px'>
                SMA20 crossed above SMA50 in last 5 days.<br>
                Price above both SMAs. Highest score bonus.
            </div>
        </div>
        <div style='flex:1;min-width:160px;background:#f5f3ff;border-radius:10px;
                    padding:12px 14px;border:1px solid #ddd6fe44'>
            <div style='font-size:20px;margin-bottom:6px'>📉</div>
            <div style='font-size:12px;font-weight:700;color:#7c3aed'>Signal 2 — Pullback Bounce</div>
            <div style='font-size:11px;color:#6d28d9;margin-top:3px'>
                SMA20 already above SMA50 (5+ days).<br>
                Price pulled back to SMA20 and bouncing.
            </div>
        </div>
        <div style='flex:1;min-width:160px;background:#f0fdf4;border-radius:10px;
                    padding:12px 14px;border:1px solid #bbf7d044'>
            <div style='font-size:20px;margin-bottom:6px'>🎯</div>
            <div style='font-size:12px;font-weight:700;color:#15803d'>Targets — Weekly ATR</div>
            <div style='font-size:11px;color:#166534;margin-top:3px'>
                Uses actual weekly candle ATR-7<br>
                T1 = Entry + 0.5× weekly ATR (+3–5%)<br>
                T2 = Entry + 1.0× weekly ATR (+6–10%)<br>
                T3 = Entry + 1.5× weekly ATR (+9–15%)
            </div>
        </div>
        <div style='flex:1;min-width:160px;background:#fff5f5;border-radius:10px;
                    padding:12px 14px;border:1px solid #fecaca44'>
            <div style='font-size:20px;margin-bottom:6px'>🛑</div>
            <div style='font-size:12px;font-weight:700;color:#dc2626'>Stop Loss — ATR Based</div>
            <div style='font-size:11px;color:#991b1b;margin-top:3px'>
                SL = 1.5% below SMA20<br>
                    Or if SMA20 is far — ATR-7 based
            </div>
        </div>
        <div style='flex:1;min-width:160px;background:#fffbeb;border-radius:10px;
                    padding:12px 14px;border:1px solid #fde68a44'>
            <div style='font-size:20px;margin-bottom:6px'>⏱️</div>
            <div style='font-size:12px;font-weight:700;color:#d97706'>Hold Time</div>
            <div style='font-size:11px;color:#92400e;margin-top:3px'>
                3–7 trading days<br>
                Exit at T1 or T2. Never hold > 10 days.
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    # ── Controls ──────────────────────────────────────────
    _sw_c1, _sw_c2, _sw_c3 = st.columns([2, 1, 1])
    with _sw_c1:
        _sw_universe = st.radio(
            "Universe",
            ["🔵 Largecap (Nifty 50)",
             "🟡 Midcap (Nifty Midcap 100)",
             "🟠 Smallcap",
             "📊 Nifty 500 (All)",
             "📁 Upload My List"],
            horizontal=True, key="sw_universe",
            help="Largecap = safer weekly trades. Midcap = higher returns. "
                 "Upload My List = scan your own CSV of NSE symbols.")

        if _sw_universe == "📁 Upload My List":
            _sw_csv_file = st.file_uploader(
                "Upload CSV/Excel with NSE symbols",
                type=['csv', 'xlsx', 'xls'],
                key="sw_csv_upload",
                help="Any NSE export works — needs a 'Symbol' column "
                     "(or symbols in the first column). "
                     "e.g. ind_niftyautolist.csv from NSE website, "
                     "or your own watchlist export.")
            if _sw_csv_file is not None:
                _sw_csv_stocks, _sw_csv_err = parse_csv_stock_list(_sw_csv_file)
                if _sw_csv_err:
                    st.error(f"❌ {_sw_csv_err}")
                    _sw_stocks = POPULAR_STOCKS
                else:
                    st.success(f"✅ Loaded {len(_sw_csv_stocks)} symbols from "
                               f"{_sw_csv_file.name}")
                    _sw_stocks = _sw_csv_stocks
            else:
                st.info("⬆️ Upload a file to scan your own stock list")
                _sw_stocks = []
        else:
            _sw_stocks = (
                LARGECAP_STOCKS if _sw_universe == "🔵 Largecap (Nifty 50)"       else
                MIDCAP_STOCKS   if _sw_universe == "🟡 Midcap (Nifty Midcap 100)" else
                SMALLCAP_STOCKS if _sw_universe == "🟠 Smallcap"                  else
                POPULAR_STOCKS
            )
        st.markdown(
            f"<div style='font-size:11px;color:#64748b;margin-top:-8px'>"
            f"⚡ {len(_sw_stocks)} stocks · Daily chart · SMA20 + SMA50</div>",
            unsafe_allow_html=True)

    with _sw_c2:
        _sw_capital = st.number_input(
            "Capital ₹", min_value=50000, max_value=2000000,
            value=200000, step=50000, format="%d", key="sw_capital",
            help="Capital per trade for weekly swing")

    with _sw_c3:
        _sw_risk_pct = st.number_input(
            "Risk %", min_value=0.5, max_value=3.0,
            value=1.5, step=0.5, format="%.1f", key="sw_risk_pct",
            help="Max risk per trade as % of capital — auto-reduced based on Nifty + drawdown")

    # ── Drawdown Tracking ──────────────────────────────
    with st.expander("📉 Drawdown Tracking — Auto Position Sizing"):
        _sw_dd_col1, _sw_dd_col2 = st.columns(2)
        with _sw_dd_col1:
            _sw_peak_cap = st.number_input(
                "Peak Capital ₹ (your best balance)",
                min_value=50000, max_value=10000000,
                value=st.session_state.get('peak_capital', _sw_capital),
                step=10000, format="%d", key="sw_peak_capital",
                help="Your highest capital balance — used to calculate drawdown %")
        with _sw_dd_col2:
            _sw_curr_cap = st.number_input(
                "Current Capital ₹ (today's balance)",
                min_value=50000, max_value=10000000,
                value=st.session_state.get('current_capital', _sw_capital),
                step=10000, format="%d", key="sw_curr_capital",
                help="Your current capital balance")
        # Store in session
        st.session_state['peak_capital']    = _sw_peak_cap
        st.session_state['current_capital'] = _sw_curr_cap
        _sw_drawdown_pct = max(0.0, (_sw_peak_cap - _sw_curr_cap) / _sw_peak_cap * 100) \
                           if _sw_peak_cap > 0 else 0.0
        _sw_drawdown_pct = round(_sw_drawdown_pct, 2)

        # Show current drawdown status
        _sw_dd_clr = '#15803d' if _sw_drawdown_pct < 3 else \
                     '#d97706' if _sw_drawdown_pct < 7 else \
                     '#dc2626' if _sw_drawdown_pct < 12 else '#7f1d1d'
        _sw_dd_lbl = 'Normal ✅' if _sw_drawdown_pct < 3 else \
                     'Caution ⚠️' if _sw_drawdown_pct < 7 else \
                     'Reduced 🔴' if _sw_drawdown_pct < 12 else 'Danger ⛔'
        st.markdown(
            f"<div style='background:{_sw_dd_clr}22;border:1px solid {_sw_dd_clr}44;"
            f"border-radius:8px;padding:8px 14px;font-size:11px;font-weight:700;"
            f"color:{_sw_dd_clr}'>"
            f"📉 Current Drawdown: {_sw_drawdown_pct:.1f}% — {_sw_dd_lbl}"
            f"<br><span style='font-weight:400;font-size:10px'>"
            f"Position sizes will be automatically adjusted based on drawdown + Nifty state"
            f"</span></div>",
            unsafe_allow_html=True)

    # Min signal score filter
    _sw_min_score = st.slider(
        "Min signal score", min_value=50, max_value=90, value=65,
        step=5, key="sw_min_score",
        help="Higher = fewer but stronger signals")

    # ── Strict Entry Mode ──────────────────────────────
    _sw_strict = st.checkbox(
        "🛡️ Strict Entry Mode",
        value=True,
        key="sw_strict_mode",
        help=(
            "When ON — only shows stocks where:\n"
            "✅ PSAR is BULLISH (price above PSAR)\n"
            "✅ No bearish candles (Shooting Star, Doji excluded)\n"
            "✅ PA Signal is not 🔴 RISKY\n\n"
            "Mild Bull candles are allowed for SMA Weekly\n"
            "(daily candles have less conviction than weekly)\n\n"
            "When OFF — shows all signals including\n"
            "bearish PSAR and bearish candles"
        ))
    if _sw_strict:
        st.markdown(
            "<div style='background:#f0fdf4;border:1.5px solid #86efac;"
            "border-radius:8px;padding:8px 14px;font-size:11px;"
            "color:#15803d;margin-bottom:8px'>"
            "🛡️ <b>Strict Mode ON</b> — PSAR bullish required · "
            "Bearish candles filtered · PA RISKY excluded · "
            "Mild Bull allowed for weekly trades"
            "</div>",
            unsafe_allow_html=True)
    else:
        st.markdown(
            "<div style='background:#fffbeb;border:1.5px solid #fde68a;"
            "border-radius:8px;padding:8px 14px;font-size:11px;"
            "color:#d97706;margin-bottom:8px'>"
            "⚠️ <b>Strict Mode OFF</b> — All signals shown including "
            "weak candles and bearish PSAR · More stocks · Lower quality"
            "</div>",
            unsafe_allow_html=True)

    # Volatility filter
    _sw_vol_col1, _sw_vol_col2 = st.columns(2)
    with _sw_vol_col1:
        _sw_max_atr_pct = st.slider(
            "Volatility Score Penalty Starts At (Daily ATR%)",
            min_value=1.0, max_value=8.0, value=4.0, step=0.5,
            key="sw_max_atr_pct",
            help="Daily ATR% above this = score penalty applied. "
                 "NO hard reject — high vol stocks still show "
                 "but with lower score and 🔴 badge. "
                 "PA + SMA20 proximity gates handle bad entries.")
    with _sw_vol_col2:
        st.markdown(
            f"<div style='background:#f8fafc;border:1px solid #e2e8f0;"
            f"border-radius:8px;padding:10px 14px;margin-top:4px'>"
            f"<div style='font-size:10px;font-weight:700;color:#64748b;"
            f"letter-spacing:1px'>SCORE IMPACT</div>"
            f"<div style='font-size:11px;color:#374151;margin-top:4px;line-height:1.8'>"
            f"🟢 &lt;2% → +8 pts (bonus)<br>"
            f"🟡 2-4% → 0 pts (neutral)<br>"
            f"🔴 4-6% → -10 pts (penalty)<br>"
            f"❌ &gt;6% → -15 pts (heavy)"
            f"</div></div>",
            unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────
    #  CONFIDENT SCORE — 100 point combined rating
    #  Eliminates need for manual analysis after scan
    #  ≥ 80 = 🔥 CONFIDENT BUY (enter without question)
    #  60-79 = ✅ GOOD (quick chart check then enter)
    #  < 60  = filtered out — not shown in results
    # ─────────────────────────────────────────────────────
    def calc_confident_score(r):
        """
        Calculate confident score from result dict.
        Combines 6 factors into single 0-100+ score.
        Works for both Monthly Swing and SMA Weekly.
        """
        # ── Component 1: Technical Score (25 pts) ────────
        # Raw scanner score reflects all technical signals
        sc = r.get('score', 0)
        if   sc >= 90: c1 = 25
        elif sc >= 80: c1 = 22
        elif sc >= 70: c1 = 18
        elif sc >= 60: c1 = 14
        elif sc >= 50: c1 = 10
        else:          c1 = 6

        # ── Component 2: PSAR Status (20 pts) ────────────
        psar_bull = r.get('psar_bullish', False)
        if psar_bull:   c2 = 20
        else:           c2 = 0

        # ── Component 3: Structure HH+HL (15 pts) ────────
        _pa_struct = r.get('pa', {}).get('structure', '')
        if   'Bullish' in _pa_struct: c3 = 15
        elif 'Neutral' in _pa_struct: c3 = 8
        elif 'Broken'  in _pa_struct: c3 = 0
        else:
            hh = r.get('hh', False)
            hl = r.get('hl', False)
            if   hh and hl: c3 = 15
            elif hh or hl:  c3 = 8
            else:           c3 = 0

        # ── Component 4: Entry Badge (15 pts) ─────────────
        badge = r.get('entry_badge', 'ACCEPTABLE')
        if   badge == 'ENTER NOW':   c4 = 15
        elif badge == 'ACCEPTABLE':  c4 = 8
        else:                        c4 = 0

        # ── Component 5: R:R Quality (10 pts) ────────────
        rr2 = r.get('rr_t2', r.get('RR_T2', 0))
        try: rr2 = float(rr2)
        except: rr2 = 0
        if   rr2 >= 3.0: c5 = 10
        elif rr2 >= 2.0: c5 = 8
        elif rr2 >= 1.5: c5 = 5
        else:            c5 = 0

        # ── Component 6: Liquidity (5 pts) ───────────────
        liq = r.get('liq_grade', r.get('Liquidity', ''))
        if   liq == 'EXCELLENT': c6 = 5
        elif liq == 'HIGH':      c6 = 3
        elif liq == 'MEDIUM':    c6 = 1
        else:                    c6 = 0

        # ── Component 7: F&O Expiry (±15 pts) ────────────
        c7 = r.get('fno_penalty', 0)

        # ── Component 8: Sector Ranking (±10 pts) ─────────
        # Uses unified sector ranking — same as Sector Leaders tab
        _sym_r  = r.get('symbol', r.get('sym', ''))
        _sec_r, _sec_rank_r, _sec_rs_r, _sec_bull_r, _, c8, _, _ = \
            get_sector_score_for_stock(_sym_r, formula='weekly')

        total = c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8

        # ── Signal label ──────────────────────────────────
        if   total >= 130:
            label = '🔥 CONFIDENT BUY'
            clr   = '#15803d'
            bg    = '#f0fdf4'
            bdr   = '#86efac'
        elif total >= 100:
            label = '✅ STRONG SETUP'
            clr   = '#0369a1'
            bg    = '#f0f9ff'
            bdr   = '#7dd3fc'
        elif total >= 75:
            label = '👍 GOOD SETUP'
            clr   = '#1d4ed8'
            bg    = '#eff6ff'
            bdr   = '#93c5fd'
        elif total >= 55:
            label = '⚠️ WEAK'
            clr   = '#d97706'
            bg    = '#fffbeb'
            bdr   = '#fcd34d'
        else:
            label = '❌ SKIP'
            clr   = '#dc2626'
            bg    = '#fef2f2'
            bdr   = '#fca5a5'

        return {
            'confident_score': total,
            'confident_label': label,
            'confident_clr':   clr,
            'confident_bg':    bg,
            'confident_bdr':   bdr,
            'c1_tech':   c1,
            'c2_psar':   c2,
            'c3_struct': c3,
            'c4_badge':  c4,
            'c5_rr':     c5,
            'c6_liq':    c6,
            'c7_fno':    c7,
            'c8_sector': c8,
            'c8_sector_name': _sec_r,
            'c8_sector_rank': _sec_rank_r,
        }

    # ── Scan function ─────────────────────────────────────
    def scan_sma_weekly(stocks, capital, risk_pct, min_score, mode):
        """
        Mode 1 - Fresh Cross (5-day window)
        Mode 2 - Trend + Pullback (recommended, more signals)
          SMA20 > SMA50 established for 5+ days
          Price touched SMA20 within last 5 days
          Now bouncing above SMA20 with volume
        """
        results     = []
        total       = len(stocks)
        _prog_sw    = st.progress(0, text="📈 Scanning SMA20 + SMA50 signals...")
        _stat_sw    = st.empty()

        # ── Pre-fetch Nifty daily for beta + swing state ──
        _sw_nifty_df    = None
        _sw_nifty_swing = {'state': 'UNKNOWN'}
        try:
            import yfinance as _yf_sw_nf
            _nf_sw = _yf_sw_nf.Ticker('^NSEI').history(
                period='1y', interval='1d',
                auto_adjust=True, actions=False)
            if _nf_sw is not None and len(_nf_sw) >= 25:
                _nf_sw.columns = [c.split(' ')[0] if ' ' in str(c) else c
                                   for c in _nf_sw.columns]
                _nf_sw = _nf_sw[['Close']].dropna()
                _nf_sw['SMA20'] = _nf_sw['Close'].rolling(20).mean()
                _nf_sw['SMA50'] = _nf_sw['Close'].rolling(50).mean()
                _sw_nifty_df = _nf_sw.copy()

                _sw_close = float(_nf_sw['Close'].iloc[-1])
                _sw_sma20 = float(_nf_sw['SMA20'].iloc[-1])
                _sw_sma50 = float(_nf_sw['SMA50'].iloc[-1])
                _sw_prev5 = float(_nf_sw['SMA20'].iloc[-6]) if len(_nf_sw) >= 6 else _sw_sma20
                _sw_slope = (_sw_sma20 - _sw_prev5) / _sw_prev5 * 100 if _sw_prev5 > 0 else 0

                if   _sw_close > _sw_sma20 > _sw_sma50 and _sw_slope >= 0.5:
                    _sw_nifty_swing['state'] = 'BULLISH'
                elif _sw_close > _sw_sma20 > _sw_sma50 and _sw_slope >= 0.1:
                    _sw_nifty_swing['state'] = 'LATE_BULL'   # ← NEW: trend flattening
                elif _sw_close > _sw_sma20 > _sw_sma50:
                    _sw_nifty_swing['state'] = 'CAUTION'
                elif _sw_close > _sw_sma20:
                    _sw_nifty_swing['state'] = 'CAUTION'
                elif _sw_close > _sw_sma50:
                    _sw_nifty_swing['state'] = 'EARLY_BEAR'  # ← NEW: below SMA20, above SMA50
                else:
                    _sw_nifty_swing['state'] = 'BEARISH'

                _sw_nifty_swing['close'] = round(_sw_close, 2)
                _sw_nifty_swing['sma20'] = round(_sw_sma20, 2)
                _sw_nifty_swing['sma50'] = round(_sw_sma50, 2)
                # Cache for dashboard
                st.session_state['nifty_swing_daily'] = _sw_nifty_swing
        except Exception:
            pass

        # ── Sector ranking — use UNIFIED function ─────────
        # Same formula and data as Monthly Swing and Sector Leaders
        # Consistent ranking across all 3 tabs
        # Uses cached result (1hr) — no extra API calls
        _sw_rankings        = get_unified_sector_rankings(formula='weekly')
        _sw_sector_status   = _sw_rankings['status_map']
        _sw_sector_rs       = _sw_rankings['rs_map']
        _sw_sector_rank_map = _sw_rankings['rank_map']

        def _sw_get_sector(sym):
            """Thin wrapper — uses single authoritative classify_stock_sector()."""
            _sec = classify_stock_sector(sym)
            # Map extended sectors back to ETF keys available in SW
            _map = {
                'PSU_BANK':'BANK', 'PVT_BANK':'BANK',
                'HEALTHCARE':'PHARMA', 'UNKNOWN':'INFRA',
            }
            return _map.get(_sec, _sec) if _sec not in SECTOR_ETF_UNIFIED else _sec

        for idx, symbol in enumerate(stocks):
            pct       = int(((idx + 1) / total) * 100)
            sym_clean = symbol.replace('.NS', '')
            _prog_sw.progress(pct, text=f"📈 {idx+1}/{total} · {sym_clean}")

            try:
                # ── Fetch daily data — for signals ────────
                _ticker_sym = symbol if symbol.endswith('.NS') else symbol + '.NS'
                try:
                    import yfinance as _yf_sw
                    _t = _yf_sw.Ticker(_ticker_sym)
                    df = _t.history(
                        period='1y',
                        interval='1d',
                        auto_adjust=True,
                        actions=False,
                        prepost=False,
                    )
                except Exception:
                    continue
                if df is None or len(df) < 60:
                    continue
                df.columns = [c.split(' ')[0] if ' ' in str(c) else c for c in df.columns]
                df = df[['Open','High','Low','Close','Volume']].dropna()
                src_lbl = 'yfinance'

                # ── Fetch weekly data — for ATR only ──────
                # Weekly candle = actual 5-day High-Low range
                # More accurate for weekly swing trade targets
                # than daily ATR × multiplier
                _wk_atr = None
                try:
                    _df_wk = _t.history(
                        period='1y',
                        interval='1wk',
                        auto_adjust=True,
                        actions=False,
                    )
                    if _df_wk is not None and len(_df_wk) >= 8:
                        _df_wk.columns = [c.split(' ')[0] if ' ' in str(c) else c
                                          for c in _df_wk.columns]
                        _df_wk = _df_wk[['High','Low','Close']].dropna()
                        _wk_hl  = _df_wk['High'] - _df_wk['Low']
                        _wk_hpc = (_df_wk['High'] - _df_wk['Close'].shift(1)).abs()
                        _wk_lpc = (_df_wk['Low']  - _df_wk['Close'].shift(1)).abs()
                        _wk_tr  = pd.concat([_wk_hl, _wk_hpc, _wk_lpc], axis=1).max(axis=1)
                        _wk_atr = round(float(_wk_tr.rolling(7).mean().iloc[-1]), 2)
                except Exception:
                    pass
                    continue
                df.columns = [c.split(' ')[0] if ' ' in str(c) else c for c in df.columns]
                df = df[['Open','High','Low','Close','Volume']].dropna()
                src_lbl = 'yfinance'

                if len(df) < 55:
                    continue

                # ── Indicators ───────────────────────────
                df['SMA20']  = df['Close'].rolling(20).mean()
                df['SMA50']  = df['Close'].rolling(50).mean()
                df['HL']     = df['High'] - df['Low']
                df['HPC']    = (df['High'] - df['Close'].shift(1)).abs()
                df['LPC']    = (df['Low']  - df['Close'].shift(1)).abs()
                df['TR']     = df[['HL','HPC','LPC']].max(axis=1)
                df['ATR7']   = df['TR'].rolling(7).mean()
                df['ATR14']  = df['TR'].rolling(14).mean()
                df['VOL_MA'] = df['Volume'].rolling(20).mean()
                df['RSI14']  = 100 - (100 / (1 + (
                    df['Close'].diff().clip(lower=0).rolling(14).mean() /
                    (-df['Close'].diff().clip(upper=0)).rolling(14).mean()
                )))

                latest   = df.iloc[-1]
                prev     = df.iloc[-2]
                prev2    = df.iloc[-3]
                prev3    = df.iloc[-4]

                sma20    = float(latest['SMA20'])
                sma50    = float(latest['SMA50'])
                sma20_p  = float(prev['SMA20'])
                sma50_p  = float(prev['SMA50'])
                close    = float(latest['Close'])
                # Use min(ATR-7, ATR-14) to prevent spike inflation
                # ATR-7: recent volatility (responsive, good for weekly targets)
                # ATR-14: smoother (prevents inflated targets from single volatile week)
                # min() takes the smaller → tighter, more achievable targets
                _atr7    = float(latest['ATR7'])
                _atr14   = float(latest['ATR14'])
                _daily_atr = round(min(_atr7, _atr14), 2)

                # Use weekly ATR if available — more accurate for 3-7 day targets
                # Weekly ATR = actual High-Low range of a full 5-day candle
                # Fallback to daily ATR × 2.0 if weekly fetch failed
                if _wk_atr and _wk_atr > 0 and not pd.isna(_wk_atr):
                    atr = _wk_atr
                    _atr_label = 'weekly'
                else:
                    atr = round(_daily_atr * 2.0, 2)
                    _atr_label = 'est.weekly'
                vol      = float(latest['Volume'])
                vol_ma   = float(latest['VOL_MA'])
                rsi      = float(latest['RSI14'])

                if any(pd.isna(x) for x in [sma20, sma50, atr, vol_ma]) or atr <= 0:
                    continue

                vol_ratio   = vol / vol_ma if vol_ma > 0 else 1.0
                pct_above   = (close - sma20) / sma20 * 100 if sma20 > 0 else 0

                # ── Check 1: Fresh Cross (SMA20 crossed SMA50 in last 5 days) ──
                cross_today = sma20 > sma50 and sma20_p <= float(prev['SMA50'])
                cross_1d    = float(prev['SMA20']) > float(prev['SMA50']) and float(prev2['SMA20']) <= float(prev2['SMA50'])
                cross_2d    = float(prev2['SMA20']) > float(prev2['SMA50']) and float(prev3['SMA20']) <= float(prev3['SMA50'])
                cross_3d    = float(prev3['SMA20']) > float(prev3['SMA50']) and len(df)>5 and float(df['SMA20'].iloc[-5]) <= float(df['SMA50'].iloc[-5])
                cross_4d    = len(df)>6 and float(df['SMA20'].iloc[-5]) > float(df['SMA50'].iloc[-5]) and float(df['SMA20'].iloc[-6]) <= float(df['SMA50'].iloc[-6])
                has_fresh_cross = cross_today or cross_1d or cross_2d or cross_3d or cross_4d
                cross_age   = (0 if cross_today else 1 if cross_1d else 2 if cross_2d else 3 if cross_3d else 4) if has_fresh_cross else 99

                # ── SMA20 slope gate (HARD FILTER) ────────
                # SMA20 must be RISING — declining SMA20 means
                # momentum is weakening regardless of being above SMA50
                # This was the APARINDS bug — SMA20 declining but still
                # above SMA50 so it passed. Now rejected.
                _sma20_5d    = float(df['SMA20'].iloc[-5]) if len(df) >= 5 else sma20
                _sma20_slope = (sma20 - _sma20_5d) / _sma20_5d * 100 if _sma20_5d > 0 else 0
                if _sma20_slope <= 0:
                    continue  # SMA20 declining or flat → skip, not a valid uptrend

                # ── VOLATILITY GATE ───────────────────────
                # Daily ATR% = ATR / Price × 100
                # High ATR% = wide SL needed = bad R:R
                # Also calculate 4-week average for stability
                _daily_atr_pct = round(float(latest['ATR7']) / close * 100, 2) if close > 0 else 0
                _atr4w         = float(df['ATR7'].tail(20).mean())  # 4-week avg ATR
                _atr4w_pct     = round(_atr4w / close * 100, 2) if close > 0 else 0
                _vol_atr_use   = max(_daily_atr_pct, _atr4w_pct)  # use worse of two

                # Volatility grade
                if   _vol_atr_use < 2.0: _vol_grade='LOW';       _vol_clr='#15803d'; _vol_bg='#f0fdf4'; _vol_ico='🟢'
                elif _vol_atr_use < 4.0: _vol_grade='MEDIUM';    _vol_clr='#d97706'; _vol_bg='#fffbeb'; _vol_ico='🟡'
                elif _vol_atr_use < 6.0: _vol_grade='HIGH';      _vol_clr='#dc2626'; _vol_bg='#fff5f5'; _vol_ico='🔴'
                else:                    _vol_grade='VERY HIGH';  _vol_clr='#7f1d1d'; _vol_bg='#fef2f2'; _vol_ico='❌'

                # ── Soft scoring — NO hard reject ────────
                # High volatility = score penalty only
                # Good stocks (HINDCOPPER) still appear
                # Existing gates (SMA20 proximity, PA, Fib)
                # already protect against bad entries
                if   _vol_atr_use < 2.0: _vol_score =  8
                elif _vol_atr_use < 3.0: _vol_score =  5
                elif _vol_atr_use < 4.0: _vol_score =  0
                elif _vol_atr_use < 5.0: _vol_score = -5
                elif _vol_atr_use < 7.0: _vol_score = -10
                else:                    _vol_score = -15

                # ── Check 2: Pullback to SMA20 and bouncing ──
                trend_days   = 0
                pullback_found = False
                pullback_age   = 0

                if sma20 > sma50:
                    # Count consecutive days SMA20 > SMA50
                    for i in range(1, min(30, len(df))):
                        if float(df['SMA20'].iloc[-i]) > float(df['SMA50'].iloc[-i]):
                            trend_days += 1
                        else:
                            break

                    # Also verify SMA50 is rising (healthy trend)
                    _sma50_5d    = float(df['SMA50'].iloc[-5]) if len(df) >= 5 else sma50
                    _sma50_slope = (sma50 - _sma50_5d) / _sma50_5d * 100 if _sma50_5d > 0 else 0
                    if _sma50_slope < -0.5:
                        continue  # SMA50 declining → not an uptrend

                    if trend_days >= 5:
                        for i in range(1, 6):
                            row_low = float(df['Low'].iloc[-i])
                            row_sma = float(df['SMA20'].iloc[-i])
                            if abs(row_low - row_sma) / row_sma * 100 <= 1.5 or row_low <= row_sma:
                                pullback_found = True
                                pullback_age   = i
                                break

                has_pullback = pullback_found and close >= sma20 * 1.002 and pct_above <= 5

                # ── Volume Dry-Up on Pullback (SMA Weekly) ─
                _sw_pb_vol_score = 0
                _sw_pb_vol_label = ''
                _sw_pb_vol_clr   = '#64748b'
                _sw_pb_vol_ratio = 1.0
                if has_pullback and pullback_age > 0:
                    try:
                        _sw_vol_avg = float(df['VolMA'].iloc[-pullback_age]) if 'VolMA' in df.columns else float(df['Volume'].iloc[-10:-3].mean())
                        _sw_vol_pb  = float(df['Volume'].iloc[-pullback_age])
                        _sw_pb_vol_ratio = round(_sw_vol_pb / _sw_vol_avg if _sw_vol_avg > 0 else 1.0, 2)
                        if   _sw_pb_vol_ratio < 0.5:
                            _sw_pb_vol_score = +15; _sw_pb_vol_label = '💧 Vol dry-up (perfect)'; _sw_pb_vol_clr = '#15803d'
                        elif _sw_pb_vol_ratio < 0.7:
                            _sw_pb_vol_score = +10; _sw_pb_vol_label = '💧 Vol dry-up (healthy)'; _sw_pb_vol_clr = '#16a34a'
                        elif _sw_pb_vol_ratio < 1.0:
                            _sw_pb_vol_score = +5;  _sw_pb_vol_label = '✅ Vol below avg';        _sw_pb_vol_clr = '#d97706'
                        elif _sw_pb_vol_ratio < 1.5:
                            _sw_pb_vol_score = 0;   _sw_pb_vol_label = '⚠️ Vol normal';           _sw_pb_vol_clr = '#d97706'
                        elif _sw_pb_vol_ratio < 2.0:
                            _sw_pb_vol_score = -10; _sw_pb_vol_label = '🔴 Vol high on pullback'; _sw_pb_vol_clr = '#dc2626'
                        else:
                            _sw_pb_vol_score = -20; _sw_pb_vol_label = '❌ Distribution detected';_sw_pb_vol_clr = '#991b1b'
                    except Exception:
                        _sw_pb_vol_ratio = 1.0

                # ── Must have at least one signal ────────
                if not has_fresh_cross and not has_pullback:
                    continue

                # Price must be above both SMAs
                if close <= sma20 or close <= sma50:
                    continue

                # ── Signal label ──────────────────────────
                if has_fresh_cross and has_pullback:
                    signal_label = f"🔥 Fresh cross {cross_age}d ago + pullback bounce"
                    signal_type  = 'both'
                elif has_fresh_cross:
                    signal_label = f"🔀 Fresh cross {cross_age}d ago"
                    signal_type  = 'cross'
                else:
                    signal_label = f"📉 Pullback {pullback_age}d ago · bouncing"
                    signal_type  = 'pullback'

                # ── Scoring ──────────────────────────────
                score = 0
                pct_above_sma20 = (close - sma20) / sma20 * 100 if sma20 > 0 else 0
                pct_above_sma50 = (close - sma50) / sma50 * 100 if sma50 > 0 else 0

                # ── Pre-compute missing factors ───────────

                # Factor 1: Higher Highs over last 3 weeks
                # Each week = 5 trading days
                _w1h = float(df['High'].iloc[-5:].max())    # this week
                _w2h = float(df['High'].iloc[-10:-5].max()) # last week
                _w3h = float(df['High'].iloc[-15:-10].max())# 2 weeks ago
                _hh  = _w1h > _w2h > _w3h                  # true uptrend

                # Factor 2: Higher Lows over last 3 weeks
                _w1l = float(df['Low'].iloc[-5:].min())
                _w2l = float(df['Low'].iloc[-10:-5].min())
                _w3l = float(df['Low'].iloc[-15:-10].min())
                _hl  = _w1l > _w2l > _w3l                  # buyers defending higher levels

                # Factor 3: SMA20 slope strength (5-day change %)
                # Already computed as _sma20_slope above
                # Binary check already done (slope > 0 = hard gate)
                # Now SCORE how strong the rise is

                # Factor 4: Price vs SMA50 distance
                # If price is >15% above SMA50 = extended = risky entry
                # Penalise extended stocks — better to wait for pullback
                _extended = pct_above_sma50 > 15

                # ── Signal type bonus (max 30) ────────────
                _cross_bonus = (25 if cross_age == 0 else 20 if cross_age == 1 else
                                15 if cross_age == 2 else 10 if cross_age == 3 else 5) if has_fresh_cross else 0
                _pb_bonus    = (25 if pullback_age == 1 else 20 if pullback_age == 2 else
                                15 if pullback_age == 3 else 8) if has_pullback else 0
                if signal_type == 'both':
                    score += min(30, max(_cross_bonus, _pb_bonus) + 5)
                elif signal_type == 'cross':
                    score += _cross_bonus
                else:
                    score += _pb_bonus

                # ── Price position vs SMA20 (max 20) ─────
                score += (20 if pct_above_sma20 <= 1 else 15 if pct_above_sma20 <= 2 else
                          10 if pct_above_sma20 <= 3 else 5 if pct_above_sma20 <= 5 else 0)

                # ── Trend days (max 10) ───────────────────
                score += (10 if trend_days >= 20 else 7 if trend_days >= 10 else
                          4 if trend_days >= 5 else 0)

                # ── RSI (max 15) ──────────────────────────
                score += (15 if 45 <= rsi <= 65 else
                          8 if 40 <= rsi < 45 or 65 < rsi <= 70 else 0)

                # ── Volume (max 15) ───────────────────────
                score += (15 if vol_ratio >= 2.0 else 10 if vol_ratio >= 1.5 else
                          5 if vol_ratio >= 1.0 else 0)

                # ── SMA gap widening (max 10) ─────────────
                gap_now  = (sma20 - sma50) / sma50 * 100
                gap_prev = (sma20_p - sma50_p) / sma50_p * 100 if sma50_p > 0 else 0
                if gap_now > gap_prev:
                    score += 10

                # ── NEW: Higher Highs (max 10) ────────────
                # Both HH and HL together = full classical uptrend structure
                # Only HH = partial confirmation
                if _hh and _hl:
                    score += 10   # both confirmed = strongest trend
                elif _hh:
                    score += 6    # only higher highs
                elif _hl:
                    score += 4    # only higher lows

                # ── NEW: SMA20 slope strength (max 10) ────
                # Hard gate already ensured slope > 0
                # Now reward HOW STRONGLY SMA20 is rising
                score += (10 if _sma20_slope >= 1.0 else
                          7  if _sma20_slope >= 0.5 else
                          4  if _sma20_slope >= 0.2 else 1)

                # ── NEW: Price vs SMA50 distance penalty ──
                # Extended stocks (>15% above SMA50) are risky entries
                # Likely due for a deeper correction, not a shallow pullback
                if _extended:
                    score -= 15   # strong penalty — avoid chasing extended moves
                elif pct_above_sma50 > 10:
                    score -= 5    # mild penalty — slightly extended

                # ── ENTRY BADGE — filter + classify ──────
                # High ATR stocks need tighter proximity to SMA20
                # ATR > 5%: max 3% above SMA20
                # ATR ≤ 5%: max 5% above SMA20
                _sw_atr_tight = _vol_atr_use > 5.0
                _sw_max_prox  = 3.0 if _sw_atr_tight else 5.0

                # Hard filter — hide if too extended
                if pct_above_sma20 > _sw_max_prox:
                    continue

                # Assign entry badge
                if _sw_atr_tight:
                    # High ATR stock — tighter zones
                    if   pct_above_sma20 <= 1.0: _entry_badge = 'ENTER NOW';  _entry_clr = '#15803d'; _entry_bg = '#f0fdf4'; _entry_ico = '🟢'
                    else:                         _entry_badge = 'ACCEPTABLE'; _entry_clr = '#d97706'; _entry_bg = '#fffbeb'; _entry_ico = '🟡'
                else:
                    # Normal ATR stock
                    if   pct_above_sma20 <= 2.0: _entry_badge = 'ENTER NOW';  _entry_clr = '#15803d'; _entry_bg = '#f0fdf4'; _entry_ico = '🟢'
                    else:                         _entry_badge = 'ACCEPTABLE'; _entry_clr = '#d97706'; _entry_bg = '#fffbeb'; _entry_ico = '🟡'

                # ── Volatility Squeeze Detection ──────────────
                # TTM Squeeze — Bollinger inside Keltner
                # Squeeze fired + bullish = highest accuracy entry
                _sw_sq = detect_volatility_squeeze(df)

                # ── ADX — Trend Strength ───────────────────────
                # Measures HOW STRONG the trend is
                # ADX > 25 = strong trend = higher win rate
                _sw_adx, _sw_pdi, _sw_mdi = calc_adx(df, period=14)
                _sw_adx_score, _sw_adx_lbl, _sw_adx_clr = \
                    get_adx_score(_sw_adx, _sw_pdi, _sw_mdi)

                # ── RS vs Own Sector (Step 1 fix — 20-Jun-2026) ─
                # Finds stocks outperforming/lagging THEIR OWN sector,
                # regardless of overall sector rank. e.g. KIMS can be
                # a sector leader even while PHARMA overall is weak.
                # ISOLATED try/except — never relies on the scanner's
                # outer except, never silently empties the result list.
                # Defaults to neutral (0 pts) on any failure.
                _sw_rs_sec_diff  = 0.0
                _sw_rs_sec_score = 0
                _sw_rs_sec_label = ''
                _sw_rs_sec_clr   = '#64748b'
                try:
                    _sw_sec_for_rs = classify_stock_sector(sym_clean)
                    _sw_rs_sec_diff, _sw_rs_sec_score, _sw_rs_sec_label, _sw_rs_sec_clr = \
                        get_rs_vs_sector(df, _sw_sec_for_rs, _sw_rankings)
                except Exception as _sw_rs_exc:
                    st.session_state.setdefault('sw_scan_errors', []).append(
                        f"{symbol} RS-vs-sector (non-fatal, score=0): {str(_sw_rs_exc)[:80]}")
                score += _sw_rs_sec_score

                # ── Filter 1: Closing position in candle ──────
                # week_pos < 0.25 = sellers won = REJECT
                # week_pos > 0.75 = buyers won = +8 pts
                _wp, _wp_score, _wp_label, _wp_reject = \
                    get_candle_close_position(df, bars=1)
                if _wp_reject:
                    continue  # Sellers dominated — skip

                # ── Filter 2: Prior candle body comparison ─────
                # Growing body = +5, shrinking = -5
                _cb_ratio, _cb_score, _cb_label = \
                    get_candle_body_momentum(df)

                # ── Beta calculation (daily) ──────────────
                # MUST be here — before scoring — so score gets correct value
                # Bearish Nifty: low beta rewarded, high beta penalised
                # Bullish Nifty: high beta rewarded, low beta neutral
                _sw_beta_val   = 1.0
                _sw_beta_score = 0
                _sw_beta_label = '➡️ Neutral'
                _sw_beta_clr   = '#64748b'
                _sw_beta_grade = 'NEUTRAL'
                _sw_beta_bg    = '#f8fafc'
                _sw_beta_bdr   = '#e2e8f0'
                _sw_beta_ico   = '➡️'
                try:
                    if _sw_nifty_df is not None:
                        _sw_beta_val = calc_stock_beta(df, _sw_nifty_df, periods=52)
                        _sw_beta_score, _sw_beta_label, _sw_beta_clr = get_beta_score(
                            _sw_beta_val, _sw_nifty_swing)
                        _sw_beta_grade, _sw_beta_clr, _sw_beta_bg, _sw_beta_bdr, _sw_beta_ico = \
                            get_beta_grade(_sw_beta_val)
                except Exception:
                    pass

                # ── Sector check ──────────────────────────
                # Same logic as Monthly Swing
                # Sector bullish/bearish + RS rank scoring
                _sw_sec_name = _sw_get_sector(sym_clean)
                _sw_sec_bull, _sw_sec_gap = _sw_sector_status.get(
                    _sw_sec_name, (True, 0))
                _sw_sec_rank = _sw_sector_rank_map.get(_sw_sec_name, 5)
                _sw_sec_rs   = _sw_sector_rs.get(_sw_sec_name, 0.0)

                # Sector bullish/bearish — small context signal
                # Strong sector = bonus, weak = minimal penalty
                if   _sw_sec_bull and _sw_sec_gap > 1: score += 5
                elif _sw_sec_bull:                      score += 2
                else:                                   score -= 2

                # Sector RS rank — bonus only, minimal penalty
                # Stock quality (week_pos, RS vs sector) is
                # the real differentiator, not sector rank alone
                if   _sw_sec_rank <= 2: score += 10
                elif _sw_sec_rank <= 4: score += 7
                elif _sw_sec_rank <= 6: score += 3
                elif _sw_sec_rank <= 9: score += 0
                else:                   score -= 3

                # ── Volatility score adjustment ───────────
                score += _vol_score

                # ── Volume dry-up score (pullback only) ───
                score += _sw_pb_vol_score

                # ── Beta score (dynamic Nifty state) ──────
                score += _sw_beta_score

                # ── Filter 1: Candle close position ───────
                score += _wp_score

                # ── Filter 2: Candle body momentum ────────
                score += _cb_score

                # ── Volatility Squeeze score ───────────────
                score += _sw_sq.get('score', 0)

                # ── ADX score ──────────────────────────────
                score += _sw_adx_score

                # Raw score pre-filter
                if score < min_score:
                    continue

                # ── Trade plan ───────────────────────────
                entry  = close
                # Pullback/cross: SL just below SMA20 (tight)
                sl     = round(sma20 * 0.985, 2)
                risk_d = entry - sl
                if risk_d <= 0:
                    continue

                # ── Filter 3: Dynamic risk sizing ─────────
                # Adjusts position size based on Nifty state
                # and personal drawdown % automatically
                _sw_nifty_st = _sw_nifty_swing.get('state', 'UNKNOWN')
                _sw_dd_pct   = getattr(st.session_state, '__dict__', {}).get('sw_drawdown_val', 0)
                try:
                    _sw_pk = st.session_state.get('peak_capital', capital)
                    _sw_cr = st.session_state.get('current_capital', capital)
                    _sw_dd_pct = max(0.0, (_sw_pk-_sw_cr)/_sw_pk*100) if _sw_pk>0 else 0.0
                except Exception:
                    _sw_dd_pct = 0.0
                _adj_risk, _risk_lbl, _risk_clr, _risk_reason = \
                    get_dynamic_risk_pct(risk_pct, _sw_nifty_st, _sw_dd_pct)

                # Weekly ATR targets — correct for 3-7 day hold
                t1     = round(entry + 0.5 * atr, 2)
                t2     = round(entry + 1.0 * atr, 2)
                t3     = round(entry + 1.5 * atr, 2)
                qty    = max(1, int((capital * _adj_risk / 100) / risk_d))
                inv    = round(entry * qty, 2)
                rr_t1  = round((t1 - entry) / risk_d, 1)
                rr_t2  = round((t2 - entry) / risk_d, 1)
                wchg   = round((close - float(df.iloc[-6]['Close'])) / float(df.iloc[-6]['Close']) * 100, 2) if len(df) >= 6 else 0.0
                sma20_sl = round((sma20 - float(df['SMA20'].iloc[-5])) / float(df['SMA20'].iloc[-5]) * 100, 3) if float(df['SMA20'].iloc[-5]) > 0 else 0

                # ── PSAR (daily, step=0.02, max=0.20) ────
                # Trailing SL for 3-7 day hold
                # After T1 hit → move SL to PSAR level
                _sw_psar_val     = None
                _sw_psar_bullish = False
                try:
                    _df_sw_ps = calc_psar(df.copy(), step=0.02, max_af=0.20)
                    if len(_df_sw_ps) >= 1:
                        _spv = float(_df_sw_ps['PSAR'].iloc[-1])
                        _spb = bool(_df_sw_ps['PSAR_bull'].iloc[-1])
                        _sw_psar_val     = round(_spv, 2)
                        _sw_psar_bullish = _spb and close > _spv
                except Exception:
                    pass

                # ── Price Action Analysis (3 checks) ──────
                # Use daily df but group into weekly candles
                # for consistent PA analysis
                try:
                    import pandas as _pd
                    _df_for_pa = df.copy()
                    # Resample daily to weekly for PA checks
                    _df_for_pa.index = _pd.to_datetime(_df_for_pa.index)
                    _df_weekly_pa = _df_for_pa.resample('W').agg(
                        {'Open':'first','High':'max',
                         'Low':'min','Close':'last','Volume':'sum'}).dropna()
                    if len(_df_weekly_pa) >= 4:
                        # Fib levels for support proximity
                        _pa_swh = float(_df_weekly_pa['High'].tail(13).max())
                        _pa_swl = sma20
                        _pa_upm = _pa_swh - _pa_swl if _pa_swh > _pa_swl else 1
                        _pa_f382 = _pa_swh - _pa_upm * 0.382
                        _pa_f500 = _pa_swh - _pa_upm * 0.500
                        _pa_f618 = _pa_swh - _pa_upm * 0.618
                        _sw_pa = run_price_action_analysis(
                            _df_weekly_pa, entry, sma20, sma50,
                            _pa_f382, _pa_f500, _pa_f618)
                    else:
                        _sw_pa = run_price_action_analysis(
                            df, entry, sma20, sma50, 0, 0, 0)
                except Exception:
                    _sw_pa = {'pa_total_score':0,'pa_signal':'⚪ Unknown',
                              'pa_signal_clr':'#64748b','pa_signal_bg':'#f8fafc',
                              'candle_pattern':'Unknown','candle_score':0,
                              'candle_emoji':'⚪','candle_desc':'',
                              'support_name':'Unknown','support_score':0,
                              'support_pct':0,'support_desc':'',
                              'structure':'Unknown','structure_score':0,
                              'structure_reject':False,'structure_desc':''}

                # Hard reject if structure broken
                if _sw_pa.get('structure_reject', False):
                    continue

                # Add PA score
                score += _sw_pa.get('pa_total_score', 0)
                if score < min_score:
                    continue

                # ── Strict Entry Mode Filter ──────────────
                # Runs AFTER PA analysis so candle_pattern is available
                # Gate 1: PSAR bullish (mandatory)
                # Gate 2: No bearish candles
                # Gate 3: PA signal not RISKY
                # Mild Bull is ALLOWED for weekly (daily candles less dramatic)
                _sw_strict_on = st.session_state.get('sw_strict_mode', True)
                if _sw_strict_on:
                    # Gate 1 — PSAR must be bullish
                    if not _sw_psar_bullish:
                        continue  # PSAR bearish → skip

                    # Gate 2 — No bearish candles
                    _sw_candle_pat = _sw_pa.get('candle_pattern', '')
                    _bad_candles   = ('Shooting Star', 'Bearish Engulfing',
                                      'Bearish', 'Doji')
                    if any(c in _sw_candle_pat for c in _bad_candles):
                        continue  # bearish candle → skip

                    # Gate 3 — PA signal not RISKY or AVOID
                    _sw_pa_sig = _sw_pa.get('pa_signal', '')
                    if '🔴' in _sw_pa_sig:
                        continue  # PA risky → skip

                # ── Daily Liquidity check ─────────────────
                # For daily chart: use real daily volume (not estimated)
                # Grade based on avg daily turnover (₹ traded per day)
                _dv20     = float(df['Volume'].tail(20).mean())  # 20-day avg volume
                _turnover = _dv20 * close                         # ₹ turnover per day

                _vol_cv   = float(df['Volume'].tail(20).std() / _dv20) if _dv20 > 0 else 1.0
                _pos_days = (_dv20 * 0.05) / qty if qty > 0 and _dv20 > 0 else 999  # days to fill position at 5% ADV

                if _turnover >= 500_000_000:     # ≥₹50Cr/day
                    _liq_grade = 'EXCELLENT'
                    _liq_clr   = '#15803d'
                    _liq_bg    = '#dcfce7'
                    _liq_ico   = '✅'
                elif _turnover >= 100_000_000:   # ≥₹10Cr/day
                    _liq_grade = 'HIGH'
                    _liq_clr   = '#1d4ed8'
                    _liq_bg    = '#dbeafe'
                    _liq_ico   = '🔵'
                elif _turnover >= 20_000_000:    # ≥₹2Cr/day
                    _liq_grade = 'MEDIUM'
                    _liq_clr   = '#d97706'
                    _liq_bg    = '#fef3c7'
                    _liq_ico   = '🟡'
                else:
                    _liq_grade = 'LOW'
                    _liq_clr   = '#dc2626'
                    _liq_bg    = '#fee2e2'
                    _liq_ico   = '🔴'

                # Format turnover for display
                if _turnover >= 1_000_000_000:
                    _liq_turn_str = f"₹{_turnover/1_000_000_000:.1f}K Cr/day"
                elif _turnover >= 10_000_000:
                    _liq_turn_str = f"₹{_turnover/10_000_000:.0f} Cr/day"
                else:
                    _liq_turn_str = f"₹{_turnover/100_000:.0f} L/day"

                results.append({
                    'symbol': sym_clean, 'score': score,
                    'close': round(close,2), 'sma20': round(sma20,2), 'sma50': round(sma50,2),
                    'atr': round(atr,2), 'atr_label': _atr_label,
                    'atr7': round(_atr7,2), 'atr14': round(_daily_atr,2),
                    'wk_atr': round(_wk_atr,2) if _wk_atr else None, 'rsi': round(rsi,1), 'vol_ratio': round(vol_ratio,1),
                    'cross_age': cross_age, 'trend_days': trend_days,
                    'signal_label': signal_label, 'signal_type': signal_type,
                    'sma20_slope': round(_sma20_slope, 2),
                    'hh': _hh, 'hl': _hl,
                    'pct_above_sma50': round(pct_above_sma50, 1),
                    'extended': _extended,
                    'psar':         _sw_psar_val,
                    'psar_bullish': _sw_psar_bullish,
                    'pa':           _sw_pa,
                    'entry_badge':  _entry_badge,
                    'entry_clr':    _entry_clr,
                    'entry_bg':     _entry_bg,
                    'entry_ico':    _entry_ico,
                    'vol_atr_pct':  _vol_atr_use,
                    'vol_grade':    _vol_grade,
                    'vol_clr':      _vol_clr,
                    'vol_bg':       _vol_bg,
                    'vol_ico':      _vol_ico,
                    'entry': round(entry,2), 'sl': sl,
                    't1': t1, 't2': t2, 't3': t3, 'qty': qty, 'inv': inv,
                    'risk_d': round(risk_d,2), 'rr_t1': rr_t1, 'rr_t2': rr_t2,
                    'week_chg': wchg, 'cap_tier': get_cap_tier(sym_clean),
                    'liq_grade': _liq_grade, 'liq_clr': _liq_clr,
                    'liq_bg': _liq_bg, 'liq_ico': _liq_ico,
                    'liq_turn': _liq_turn_str,
                    'src': src_lbl, 'mode': signal_type,
                    'pb_vol_ratio':  _sw_pb_vol_ratio,
                    'pb_vol_score':  _sw_pb_vol_score,
                    'pb_vol_label':  _sw_pb_vol_label,
                    'pb_vol_clr':    _sw_pb_vol_clr,
                    # Filter 1 — candle close position
                    'week_pos':      _wp,
                    'wp_score':      _wp_score,
                    'wp_label':      _wp_label,
                    # Filter 2 — candle body momentum
                    'cb_ratio':      _cb_ratio,
                    'cb_score':      _cb_score,
                    'cb_label':      _cb_label,
                    # Volatility squeeze
                    'squeeze':       _sw_sq,
                    'squeeze_score': _sw_sq.get('score', 0),
                    'squeeze_fired': _sw_sq.get('squeeze_fired', False),
                    'squeeze_on':    _sw_sq.get('squeeze_on', False),
                    'squeeze_label': _sw_sq.get('label', ''),
                    'squeeze_weeks': _sw_sq.get('squeeze_weeks', 0),
                    # ADX
                    'adx':           _sw_adx,
                    'adx_pdi':       _sw_pdi,
                    'adx_mdi':       _sw_mdi,
                    'adx_score':     _sw_adx_score,
                    'adx_label':     _sw_adx_lbl,
                    'adx_clr':       _sw_adx_clr,
                    # RS vs own sector (Step 1 fix — 20-Jun-2026)
                    'rs_sec_diff':   _sw_rs_sec_diff,
                    'rs_sec_score':  _sw_rs_sec_score,
                    'rs_sec_label':  _sw_rs_sec_label,
                    'rs_sec_clr':    _sw_rs_sec_clr,
                    # Filter 3 — dynamic risk sizing
                    'adj_risk_pct':  _adj_risk,
                    'risk_label':    _risk_lbl,
                    'risk_clr':      _risk_clr,
                    'risk_reason':   _risk_reason,
                    'beta':          round(_sw_beta_val, 2),
                    'beta_score':    _sw_beta_score,
                    'beta_label':    _sw_beta_label,
                    'beta_grade':    _sw_beta_grade,
                    'beta_clr':      _sw_beta_clr,
                    'beta_bg':       _sw_beta_bg,
                    'beta_bdr':      _sw_beta_bdr,
                    'beta_ico':      _sw_beta_ico,
                    'nifty_swing_state': _sw_nifty_swing.get('state','UNKNOWN'),
                    'sec_name':    _sw_sec_name,
                    'sec_bull':    _sw_sec_bull,
                    'sec_gap':     _sw_sec_gap,
                    'sec_rank':    _sw_sec_rank,
                    'sec_rs_gap':  _sw_sec_rs,
                    **get_fno_info(sym_clean),
                })
                # Calculate confident score using local function
                _cs = calc_confident_score(results[-1])
                results[-1].update(_cs)

                if len(results) % 5 == 0:
                    _stat_sw.markdown(
                        f"<div style='font-size:12px;color:#1d4ed8;padding:4px 0'>"
                        f"📈 {len(results)} signals found so far...</div>",
                        unsafe_allow_html=True)

            except Exception as _sw_exc:
                import traceback as _tb
                _err = f"{symbol}: {str(_sw_exc)[:100]} | {_tb.format_exc().splitlines()[-1]}"
                st.session_state.setdefault('sw_scan_errors',[]).append(_err)
                continue
        _prog_sw.empty()
        _stat_sw.empty()
        # Sort by confident score (highest first)
        # CONFIDENT BUY (≥80) always appears first
        # Then GOOD SETUP (60-79)
        # Within each group sorted by confident score
        for r in results:
            if 'confident_score' not in r:
                _cs = calc_confident_score(r)
                r.update(_cs)
        results.sort(key=lambda x: x.get('confident_score', 0), reverse=True)
        return results

    _sw_run = st.button(
        "📈 Scan Now",
        key="sw_run_scan", use_container_width=True, type="primary",
        help="Scans daily chart for SMA20+SMA50 — fresh crosses AND pullback bounces")

    if _sw_run:
        if not _sw_stocks:
            st.warning("⚠️ No stocks to scan — upload a CSV file or pick a different universe above.")
        else:
            # Clear yfinance disk cache to force fresh data
            try:
                import yfinance as _yf_clear
                _yf_clear.set_tz_cache_location(None)
            except Exception:
                pass
            try:
                import shutil, pathlib
                _yf_cache = pathlib.Path.home() / '.cache' / 'py-yfinance'
                if _yf_cache.exists():
                    shutil.rmtree(_yf_cache, ignore_errors=True)
            except Exception:
                pass
            with st.spinner(f"📈 Scanning {len(_sw_stocks)} stocks on daily chart..."):
                _sw_results = scan_sma_weekly(
                    _sw_stocks, _sw_capital, _sw_risk_pct, _sw_min_score, '')
            st.session_state['sw_results']   = _sw_results
            st.session_state['sw_scan_time'] = ist_now().strftime('%d %b %Y %H:%M IST')
            st.rerun()

    # ── Show results ──────────────────────────────────────
    _sw_results  = st.session_state.get('sw_results', [])
    _sw_scantime = st.session_state.get('sw_scan_time', '')

    # ── Debug: Show scan errors (even if scan succeeded) ──
    # Non-fatal errors (e.g. RS-vs-sector isolated failures)
    # default to neutral and don't block results, but you
    # should still be able to see them happened.
    _sw_errors = st.session_state.get('sw_scan_errors', [])
    if _sw_errors:
        _sw_err_title = (f'🔍 Debug: {len(_sw_errors)} non-fatal error(s) during scan '
                          f'(results still shown — these defaulted to neutral)'
                          if len(_sw_results) > 0 else
                          f'🔍 Debug: {len(_sw_errors)} stocks had errors during scan')
        with st.expander(_sw_err_title):
            for _e in _sw_errors[:15]:
                st.code(_e)
        st.session_state['sw_scan_errors'] = []

    # ── Expiry Zone Banner ─────────────────────────────
    _dte  = days_to_expiry()
    _zone = get_expiry_zone(_dte)
    _exp  = get_monthly_expiry()
    _exp_str = _exp.strftime('%d %b %Y')
    if   _zone == 'FRESH':
        _ban_bg='#f0fdf4'; _ban_bdr='#86efac'; _ban_clr='#15803d'
        _ban_ico='🟢'; _ban_title='Post-Expiry — BEST Entry Window!'
        _ban_msg=(f'New F&O cycle started · Enter any stock freely · '
                  f'Fresh positions being built · Next expiry {_exp_str}')
    elif _zone == 'DANGER':
        _ban_bg='#fef2f2'; _ban_bdr='#fca5a5'; _ban_clr='#dc2626'
        _ban_ico='🔴'; _ban_title=f'Expiry Week — {_dte} days to {_exp_str}'
        _ban_msg=('F&O stocks may be price-pinned · '
                  'Non-F&O stocks shown first · '
                  'F&O stocks score penalised -15 pts · '
                  'Consider waiting for post-expiry entry')
    elif _zone == 'CAUTION':
        _ban_bg='#fffbeb'; _ban_bdr='#fde68a'; _ban_clr='#d97706'
        _ban_ico='⚠️'; _ban_title=f'Second Half — {_dte} days to {_exp_str}'
        _ban_msg=('F&O stocks may slow down · '
                  'Prefer Non-F&O stocks this week · '
                  'F&O stocks score -8 pts · Reduce position size on F&O')
    else:
        _ban_bg='#f0fdf4'; _ban_bdr='#bbf7d0'; _ban_clr='#15803d'
        _ban_ico='✅'; _ban_title=f'Safe Zone — {_dte} days to {_exp_str}'
        _ban_msg='First half of month · Enter freely · No expiry pinning risk'

    st.markdown(
        f"<div style='background:{_ban_bg};border:1.5px solid {_ban_bdr};"
        f"border-radius:10px;padding:10px 16px;margin-bottom:12px;"
        f"display:flex;align-items:center;gap:12px'>"
        f"<div style='font-size:22px'>{_ban_ico}</div>"
        f"<div>"
        f"<div style='font-size:12px;font-weight:800;color:{_ban_clr}'>"
        f"F&O EXPIRY — {_ban_title}</div>"
        f"<div style='font-size:11px;color:{_ban_clr};opacity:0.85;margin-top:2px'>"
        f"{_ban_msg}</div>"
        f"</div></div>",
        unsafe_allow_html=True)

    # ── Nifty State Banner for SMA Weekly ─────────────
    _sw_nifty_now = st.session_state.get('nifty_swing_daily', {})
    _sw_mkt_state = _sw_nifty_now.get('state', 'UNKNOWN')

    if _sw_mkt_state == 'BEARISH':
        st.markdown(
            "<div style='background:#1f0c0c;border:2px solid #dc2626;"
            "border-radius:12px;padding:14px 18px;margin-bottom:14px'>"
            "<div style='font-size:14px;font-weight:800;color:#fca5a5'>"
            "🔴 NIFTY BEARISH — SMA Weekly in Defensive Mode</div>"
            "<div style='font-size:11px;color:#fecaca;margin-top:6px;line-height:1.8'>"
            "⚡ <b>Only stocks outperforming Nifty (RS > 1.05)</b> are shown &nbsp;·&nbsp; "
            "⚡ <b>Minimum score raised to 90</b> — only strongest signals &nbsp;·&nbsp; "
            "⚡ <b>Position size auto-halved</b> — capital protection active &nbsp;·&nbsp; "
            "⚡ <b>Max 2 open trades</b> recommended in bearish market"
            "</div></div>",
            unsafe_allow_html=True)

    elif _sw_mkt_state == 'EARLY_BEAR':
        st.markdown(
            "<div style='background:#1c0d05;border:2px solid #ea580c;"
            "border-radius:12px;padding:14px 18px;margin-bottom:14px'>"
            "<div style='font-size:14px;font-weight:800;color:#fdba74'>"
            "🟠 NIFTY EARLY BEAR — Nifty below SMA20, above SMA50</div>"
            "<div style='font-size:11px;color:#fed7aa;margin-top:6px;line-height:1.8'>"
            "⚡ <b>Transitioning to bearish</b> — reduce exposure now &nbsp;·&nbsp; "
            "⚡ <b>Position size 35% of normal</b> &nbsp;·&nbsp; "
            "⚡ <b>Only beta &lt; 0.8 stocks</b> — defensive only &nbsp;·&nbsp; "
            "⚡ <b>No new Monthly Swing entries</b> recommended"
            "</div></div>",
            unsafe_allow_html=True)

    elif _sw_mkt_state == 'LATE_BULL':
        st.markdown(
            "<div style='background:#1c150a;border:1.5px solid #d97706;"
            "border-radius:10px;padding:10px 16px;margin-bottom:12px'>"
            "<div style='font-size:12px;font-weight:800;color:#fcd34d'>"
            "🟡 NIFTY LATE BULL — Trend Flattening · Transition Warning</div>"
            "<div style='font-size:11px;color:#fde68a;margin-top:4px'>"
            "SMA20 slope weakening — may be peaking · "
            "Position sizes reduced to 75% · "
            "Prefer defensive sectors (FMCG, Pharma, IT) · "
            "Tighten stop losses on existing positions"
            "</div></div>",
            unsafe_allow_html=True)

    elif _sw_mkt_state == 'CAUTION':
        st.markdown(
            "<div style='background:#1c150a;border:1.5px solid #d97706;"
            "border-radius:10px;padding:10px 16px;margin-bottom:12px'>"
            "<div style='font-size:12px;font-weight:800;color:#fcd34d'>"
            "⚠️ NIFTY CAUTION — Selective entry mode</div>"
            "<div style='font-size:11px;color:#fde68a;margin-top:4px'>"
            "Position sizes reduced · Only GOOD+ setups shown · "
            "Prefer beta &lt; 1.0 and top sector stocks"
            "</div></div>",
            unsafe_allow_html=True)

    if not _sw_results:
        st.markdown("""
        <div style='background:#1a2035;border-radius:16px;padding:32px;
                    text-align:center;margin:20px 0'>
            <div style='font-size:40px;margin-bottom:12px'>📈</div>
            <div style='font-size:18px;font-weight:700;color:white;margin-bottom:8px'>
                SMA Weekly Scanner Ready
            </div>
            <div style='font-size:13px;color:rgba(255,255,255,0.6)'>
                Select universe · Set capital · Click Scan
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        # ── Filter buttons ────────────────────────────
        _n_all   = len(_sw_results)
        _n_cross = len([r for r in _sw_results if r.get('signal_type') in ('cross','both')])
        _n_pb    = len([r for r in _sw_results if r.get('signal_type') in ('pullback','both')])
        _n_both  = len([r for r in _sw_results if r.get('signal_type') == 'both'])
        _n_sq    = len([r for r in _sw_results if r.get('squeeze_fired') or r.get('squeeze_on')])

        _sw_filter = st.radio(
            "Filter",
            [f"📊 All ({_n_all})",
             f"🔀 Fresh Cross ({_n_cross})",
             f"📉 Pullback Bounce ({_n_pb})",
             f"🔥 Both Signals ({_n_both})",
             f"🔥 Squeeze ({_n_sq})"],
            horizontal=True, key="sw_filter",
            help="Filter signals by type")

        # Apply filter
        if 'Fresh Cross' in _sw_filter:
            _sw_filtered = [r for r in _sw_results if r.get('signal_type') in ('cross','both')]
        elif 'Pullback Bounce' in _sw_filter:
            _sw_filtered = [r for r in _sw_results if r.get('signal_type') in ('pullback','both')]
        elif 'Both Signals' in _sw_filter:
            _sw_filtered = [r for r in _sw_results if r.get('signal_type') == 'both']
        elif 'Squeeze' in _sw_filter:
            _sw_filtered = [r for r in _sw_results
                            if r.get('squeeze_fired') or r.get('squeeze_on')]
            _sw_filtered = sorted(_sw_filtered,
                                   key=lambda x: (x.get('squeeze_fired',False),
                                                  x.get('squeeze_weeks',0)),
                                   reverse=True)
        else:
            _sw_filtered = _sw_results

        # ── Sort by confident score (CONFIDENT BUY first) ─
        _sw_filtered = sorted(
            _sw_filtered,
            key=lambda x: x.get('confident_score', 0),
            reverse=True)



        # ── CSV Export — SMA Weekly ────────────────────
        def _sw_to_csv(results):
            import csv, io
            _buf = io.StringIO()
            _cols = [
                'Symbol','Score','Rank_Score','Signal_Type','Signal_Label',
                'Entry','SMA20','SMA50','ATR','ATR_Label',
                'RSI','Vol_Ratio','Trend_Days','SMA20_Slope',
                'HH','HL','Pct_Above_SMA20','Pct_Above_SMA50','Extended',
                'Stop_Loss','T1','T2','T3',
                'Qty','Investment','RR_T1','RR_T2',
                'Liquidity','Liq_Turnover','Cap_Tier',
                'Week_Change','PSAR','PSAR_Bullish','Scan_Date',
            ]
            _w = csv.DictWriter(_buf, fieldnames=_cols, extrasaction='ignore')
            _w.writeheader()
            for r in results:
                _w.writerow({
                    'Symbol':          r.get('symbol',''),
                    'Score':           r.get('score',0),
                    'Rank_Score':      round(r.get('_rank_score',0),1),
                    'Signal_Type':     r.get('signal_type',''),
                    'Signal_Label':    r.get('signal_label',''),
                    'Entry':           r.get('entry',0),
                    'SMA20':           r.get('sma20',0),
                    'SMA50':           r.get('sma50',0),
                    'ATR':             r.get('atr',0),
                    'ATR_Label':       r.get('atr_label',''),
                    'RSI':             r.get('rsi',0),
                    'Vol_Ratio':       r.get('vol_ratio',0),
                    'Trend_Days':      r.get('trend_days',0),
                    'SMA20_Slope':     r.get('sma20_slope',0),
                    'HH':              r.get('hh',False),
                    'HL':              r.get('hl',False),
                    'Pct_Above_SMA20': r.get('pct_above20',0),
                    'Pct_Above_SMA50': r.get('pct_above50',0),
                    'Extended':        r.get('extended',False),
                    'Stop_Loss':       r.get('sl',0),
                    'T1':              r.get('t1',0),
                    'T2':              r.get('t2',0),
                    'T3':              r.get('t3',0),
                    'Qty':             r.get('qty',0),
                    'Investment':      r.get('inv',0),
                    'RR_T1':           r.get('rr_t1',0),
                    'RR_T2':           r.get('rr_t2',0),
                    'Liquidity':       r.get('liq_grade',''),
                    'Liq_Turnover':    r.get('liq_turn',''),
                    'Cap_Tier':        r.get('cap_tier',''),
                    'Week_Change':     r.get('week_chg',0),
                    'PSAR':            r.get('psar',''),
                    'PSAR_Bullish':    r.get('psar_bullish',False),
                    'Scan_Date':       _sw_scantime,
                })
            return _buf.getvalue().encode('utf-8')

        # Summary bar with CSV button
        _sw_hdr1, _sw_hdr2 = st.columns([4, 1])
        with _sw_hdr1:
            st.markdown(
                f"<div style='font-size:12px;color:#64748b;margin-bottom:12px'>"
                f"📈 {len(_sw_filtered)} signals shown · Scanned {_sw_scantime}</div>",
                unsafe_allow_html=True)
        with _sw_hdr2:
            _sw_csv_fname = f"sma_weekly_{ist_now().strftime('%d%b%Y')}.csv"
            st.download_button(
                label="📥 Download CSV",
                data=_sw_to_csv(_sw_filtered),
                file_name=_sw_csv_fname,
                mime="text/csv",
                use_container_width=True,
                help=f"Download all {len(_sw_filtered)} signals as CSV"
            )

        # ── Batch AI (SMA Weekly) ─────────────────
        _sw_batch_key = 'sw_batch_ai_btn'
        _sw_batch_res = 'sw_batch_ai_result'
        _sw_batch_tag = 'sw_batch_ai_tags'
        if st.button(
            f'🤖 AI Analyse All {len(_sw_filtered)} Stocks — Weekly Portfolio Recommendation',
            key=_sw_batch_key, use_container_width=True):
            with st.spinner('🤖 Analysing weekly stocks...'):
                try:
                    import requests as _sbr, json as _sbj
                    _ant_k2 = load_anthropic_key()
                    if not _ant_k2:
                        st.error("❌ Anthropic API key not set. Go to sidebar → 🤖 AI Validation → Set Anthropic API Key")
                        raise Exception("API key not configured")
                    _sw_sum = []
                    for _si,_ss in enumerate(_sw_filtered[:15],1):
                        _sw_sum.append(
                            f'STOCK {_si}: {_ss["symbol"]} Score={_ss.get("score",0)} '
                            f'Entry=Rs{_ss.get("entry",0):.2f} RSI={_ss.get("rsi",0):.1f} '
                            f'Vol={_ss.get("vol_ratio",0):.1f}x Trend={_ss.get("trend_days",0)}d '
                            f'Signal={_ss.get("signal_label","")} '
                            f'T1=Rs{_ss.get("t1",0):.2f}(RR {_ss.get("rr_t1",0)}:1) '
                            f'T2=Rs{_ss.get("t2",0):.2f}(RR {_ss.get("rr_t2",0)}:1) '
                            f'SL=Rs{_ss.get("sl",0):.2f} HH={_ss.get("hh",False)} HL={_ss.get("hl",False)}')
                    _sw_p = (
                        f'NSE SMA Weekly swing (3-7 day hold). Analyse {len(_sw_filtered[:15])} stocks.'
                        f' Capital Rs5L, max 2 positions. Which 2 to enter Monday morning?\n\n'
                        + '\n'.join(_sw_sum)
                        + '\n\nRank into enter_now(max 2), watchlist, avoid. '
                        'Reply ONLY valid JSON no markdown: '
                        '{"market_note":"str","portfolio_note":"str",'
                        '"enter_now":[{"symbol":"X","rank":1,"confidence":"HIGH/MEDIUM/LOW",'
                        '"reason":"str","risk":"str","entry_note":"str"}],'
                        '"watchlist":[{"symbol":"X","reason":"str","entry_condition":"str"}],'
                        '"avoid":[{"symbol":"X","reason":"str"}]}'
                    )
                    _sr = _sbr.post('https://api.anthropic.com/v1/messages',
                        headers={'Content-Type':'application/json',
                                 'x-api-key': _ant_k2,
                                 'anthropic-version':'2023-06-01'},
                        json={'model':'claude-sonnet-4-20250514','max_tokens':1200,
                              'system':'You are an expert NSE swing trading analyst. Always respond with valid JSON only. No markdown.',
                              'messages':[{'role':'user','content':_sw_p}]},timeout=45)
                    if _sr.status_code==200:
                        _sd2 = _sbj.loads(_sr.json()['content'][0]['text'].strip().replace('```json','').replace('```','').strip())
                        st.session_state[_sw_batch_res]=_sd2
                        _st2={}
                        for _e in _sd2.get('enter_now',[]): _st2[_e['symbol']]='enter'
                        for _w in _sd2.get('watchlist',[]): _st2[_w['symbol']]='watch'
                        for _a in _sd2.get('avoid',[]): _st2[_a['symbol']]='avoid'
                        st.session_state[_sw_batch_tag]=_st2
                    else:
                        try:
                            _err_b2 = _sr.json().get('error',{}).get('message','Unknown')
                        except Exception:
                            _err_b2 = _sr.text[:200]
                        st.error(f'AI error {_sr.status_code}: {_err_b2}')
                except Exception as _sex2: st.error(str(_sex2)[:100])

        if _sw_batch_res in st.session_state:
            _sd = st.session_state[_sw_batch_res]
            _sten=_sd.get('enter_now',[]); _stwl=_sd.get('watchlist',[]); _stav=_sd.get('avoid',[])
            st.markdown(
                f"<div style='background:white;border:2px solid #667eea44;border-radius:12px;"
                f"padding:14px;margin-bottom:12px'>"
                f"<b style='color:#1a2035'>🤖 AI Weekly Recommendation</b> &nbsp;"
                f"<span style='background:#dcfce7;color:#15803d;font-size:10px;font-weight:700;border-radius:4px;padding:2px 7px'>✅ {len(_sten)} ENTER</span> "
                f"<span style='background:#fef3c7;color:#d97706;font-size:10px;font-weight:700;border-radius:4px;padding:2px 7px'>⏳ {len(_stwl)} WATCH</span> "
                f"<span style='background:#fee2e2;color:#dc2626;font-size:10px;font-weight:700;border-radius:4px;padding:2px 7px'>❌ {len(_stav)} AVOID</span>"
                f"<div style='font-size:11px;color:#4c1d95;margin-top:8px;background:#f5f3ff;border-radius:6px;padding:6px 10px'>"
                f"💡 {_sd.get('portfolio_note','')}</div>",
                unsafe_allow_html=True)
            if _sd.get('market_note'):
                st.markdown(f"<div style='font-size:11px;color:#374151;background:#fffbeb;border:1px solid #fde68a;border-radius:6px;padding:6px 10px;margin-top:6px'>"
                    f"📊 {_sd['market_note']}</div>",unsafe_allow_html=True)
            for _e in _sten:
                _cr3={'HIGH':'🔥','MEDIUM':'📊','LOW':'⚠️'}.get(_e.get('confidence',''),'')
                st.markdown(
                    f"<div style='background:white;border:1px solid #86efac;border-radius:8px;padding:8px 12px;margin-top:6px'>"
                    f"<b>#{_e.get('rank',1)} {_e['symbol']}</b> "
                    f"<span style='background:#15803d;color:white;font-size:10px;font-weight:700;border-radius:4px;padding:2px 7px'>{_cr3} {_e.get('confidence','')} ✅ ENTER</span><br>"
                    f"<span style='font-size:11px;color:#374151'>{_e.get('reason','')}</span><br>"
                    f"<span style='font-size:11px;color:#d97706'>⚠️ {_e.get('risk','')}</span> &nbsp;"
                    f"<span style='font-size:11px;color:#1d4ed8'>🎯 {_e.get('entry_note','')}</span></div>",
                    unsafe_allow_html=True)
            for _w in _stwl:
                st.markdown(f"<div style='background:#fffbeb;border:1px solid #fde68a;border-radius:8px;padding:7px 12px;margin-top:4px;font-size:11px'>"
                    f"⏳ <b>{_w['symbol']}</b> — {_w.get('reason','')} "
                    f"<span style='color:#15803d'>📌 {_w.get('entry_condition','')}</span></div>",unsafe_allow_html=True)
            for _a in _stav:
                st.markdown(f"<div style='background:#fef2f2;border:1px solid #fca5a5;border-radius:7px;padding:6px 12px;margin-top:3px;font-size:11px'>"
                    f"❌ <b>{_a['symbol']}</b> — {_a.get('reason','')}</div>",unsafe_allow_html=True)
            st.markdown('</div>',unsafe_allow_html=True)
            if st.button('✕ Hide AI',key='sw_hide_batch'): del st.session_state[_sw_batch_res]; st.rerun()

        for _sw_r in _sw_filtered[:15]:
            _sc      = _sw_r['score']
            _sym     = _sw_r['symbol']
            _close   = _sw_r['close']
            _sma20   = _sw_r['sma20']
            _sma50   = _sw_r['sma50']
            _atr     = _sw_r['atr']
            _rsi     = _sw_r['rsi']
            _volx    = _sw_r['vol_ratio']
            _cage    = _sw_r['cross_age']
            _entry   = _sw_r['entry']
            _sl      = _sw_r['sl']
            _t1      = _sw_r['t1']
            _t2      = _sw_r['t2']
            _t3      = _sw_r['t3']
            _qty     = _sw_r['qty']
            _inv     = _sw_r['inv']
            _rr1     = _sw_r['rr_t1']
            _rr2     = _sw_r['rr_t2']
            _wchg    = _sw_r['week_chg']
            _cap     = _sw_r['cap_tier']
            _rd      = _sw_r['risk_d']
            _rk      = _sw_r.get('_rank_score', _sc)
            _liq_grade = _sw_r.get('liq_grade', 'MEDIUM')
            _liq_clr   = _sw_r.get('liq_clr',   '#d97706')
            _liq_bg    = _sw_r.get('liq_bg',    '#fef3c7')
            _liq_ico   = _sw_r.get('liq_ico',   '🟡')
            _liq_turn  = _sw_r.get('liq_turn',  '')
            _liq_border = _liq_clr + '44'

            # Score colour
            _sc_clr = '#15803d' if _sc >= 80 else ('#d97706' if _sc >= 65 else '#64748b')
            _sc_bg  = '#dcfce7' if _sc >= 80 else ('#fef3c7' if _sc >= 65 else '#f1f5f9')

            # Cross age label
            _cage_lbl = _sw_r.get('signal_label', f"Signal age {_cage}d")
            _cage_clr = '#15803d' if _cage <= 1 else '#d97706'

            # Cap badge
            _cap_ico, _cap_name, _cap_clr, _cap_bg = CAP_TIER_BADGE.get(
                _cap, ('🟠','Smallcap','#c2410c','#fff7ed'))
            _cap_border = _cap_clr + '44'

            # ── Confident Score ───────────────────────
            _conf       = _sw_r.get('confident_score', 0)
            _conf_lbl   = _sw_r.get('confident_label', '⚠️ WEAK')
            _conf_clr   = _sw_r.get('confident_clr',  '#d97706')
            _conf_bg    = _sw_r.get('confident_bg',   '#fffbeb')
            _conf_bdr   = _sw_r.get('confident_bdr',  '#fcd34d')
            _c1         = _sw_r.get('c1_tech',   0)
            _c2         = _sw_r.get('c2_psar',   0)
            _c3         = _sw_r.get('c3_struct', 0)
            _c4         = _sw_r.get('c4_badge',  0)
            _c5         = _sw_r.get('c5_rr',     0)
            _c6         = _sw_r.get('c6_liq',    0)

            # Card border = confident score colour
            _sc_border  = _conf_bdr

            # Weekly change colour
            _wchg_clr = '#15803d' if _wchg >= 0 else '#dc2626'

            # ── Card wrapper open ──────────────────────
            st.markdown(
                f"<div style='background:#ffffff;border:2px solid {_sc_border};"
                f"border-radius:16px;padding:18px 20px;margin-bottom:12px;'>",
                unsafe_allow_html=True)

            # ── Header ─────────────────────────────────
            _fno_note_html = (
                f"<div style='font-size:10px;color:{_sw_r.get('fno_clr','#64748b')};"
                f"margin-top:3px;padding:3px 8px;"
                f"background:{_sw_r.get('fno_bg','#f8fafc')};border-radius:4px'>"
                f"{_sw_r.get('fno_note','')}</div>"
            ) if _sw_r.get('fno_note') else ""
            _slope_clr = '#15803d' if _sw_r.get('sma20_slope',0)>0 else '#dc2626'
            _vol_ico   = _sw_r.get('vol_ico','⚪')
            _vol_atr   = _sw_r.get('vol_atr_pct',0)
            _vol_grd   = _sw_r.get('vol_grade','')
            _vol_bg2   = _sw_r.get('vol_bg','#f8fafc')
            _vol_clr2  = _sw_r.get('vol_clr','#64748b')
            _ent_ico   = _sw_r.get('entry_ico','🟡')
            _ent_badge = _sw_r.get('entry_badge','ACCEPTABLE')
            _ent_clr2  = _sw_r.get('entry_clr','#d97706')
            _ent_bg2   = _sw_r.get('entry_bg','#fffbeb')
            _fno_badge2= _sw_r.get('fno_badge','✅ Non-F&O')
            _fno_clr2  = _sw_r.get('fno_clr','#64748b')
            _fno_bg2   = _sw_r.get('fno_bg','#f8fafc')
            _fno_bdr2  = _sw_r.get('fno_bdr','#e2e8f0')
            _sw_slope  = _sw_r.get('sma20_slope',0)
            # Sector rank badge
            _sw_sec_name = _sw_r.get('sec_name', '')
            _sw_sec_rank = _sw_r.get('sec_rank', 5)
            _sw_sec_rs   = _sw_r.get('sec_rs_gap', 0.0)
            _sw_sec_bull = _sw_r.get('sec_bull', True)
            _sw_sec_rs_s = f'+{_sw_sec_rs:.1f}%' if _sw_sec_rs >= 0 else f'{_sw_sec_rs:.1f}%'
            if   _sw_sec_rank <= 2: _sw_sec_rank_clr='#15803d'; _sw_sec_rank_ico='🥇'
            elif _sw_sec_rank <= 4: _sw_sec_rank_clr='#16a34a'; _sw_sec_rank_ico='🥈'
            elif _sw_sec_rank <= 6: _sw_sec_rank_clr='#d97706'; _sw_sec_rank_ico='🥉'
            else:                   _sw_sec_rank_clr='#dc2626'; _sw_sec_rank_ico='⬇️'
            # Beta badge
            _sw_beta       = _sw_r.get('beta', 1.0)
            _sw_beta_grade = _sw_r.get('beta_grade', 'NEUTRAL')
            _sw_beta_clr   = _sw_r.get('beta_clr', '#64748b')
            _sw_beta_bg    = _sw_r.get('beta_bg', '#f8fafc')
            _sw_beta_bdr   = _sw_r.get('beta_bdr', '#e2e8f0')
            _sw_beta_ico   = _sw_r.get('beta_ico', '➡️')
            _sw_beta_score = _sw_r.get('beta_score', 0)
            _sw_beta_ss    = f'+{_sw_beta_score}' if _sw_beta_score > 0 else str(_sw_beta_score)

            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"align-items:flex-start;flex-wrap:wrap;gap:8px;margin-bottom:14px'>"
                f"<div>"
                f"<div style='display:flex;align-items:center;gap:8px;flex-wrap:wrap'>"
                f"<span style='font-size:22px;font-weight:800;color:#1a2035'>{_sym}</span>"
                f"<span style='background:{_conf_bg};color:{_conf_clr};font-size:13px;"
                f"font-weight:800;border-radius:8px;padding:4px 12px;"
                f"border:2px solid {_conf_bdr}'>⭐ {_conf}/100 · {_conf_lbl}</span>"
                f"<span style='background:{_sc_bg};color:{_sc_clr};font-size:10px;"
                f"font-weight:700;border-radius:6px;padding:2px 8px'>"
                f"Scanner {_sc}/100</span>"
                f"<span style='background:{_cap_bg};color:{_cap_clr};font-size:10px;"
                f"font-weight:700;border-radius:4px;padding:2px 8px;"
                f"border:1px solid {_cap_border}'>{_cap_ico} {_cap_name}</span>"
                f"<span style='background:{_liq_bg};color:{_liq_clr};font-size:10px;"
                f"font-weight:700;border-radius:4px;padding:2px 8px;"
                f"border:1px solid {_liq_border}'>{_liq_ico} {_liq_grade} · {_liq_turn}</span>"
                f"<span style='background:{_vol_bg2};color:{_vol_clr2};font-size:10px;"
                f"font-weight:700;border-radius:4px;padding:2px 8px;"
                f"border:1px solid {_vol_clr2}44'>{_vol_ico} Vol {_vol_atr:.1f}% {_vol_grd}</span>"
                f"<span style='background:{_ent_bg2};color:{_ent_clr2};font-size:10px;"
                f"font-weight:700;border-radius:4px;padding:2px 8px;"
                f"border:1px solid {_ent_clr2}44'>{_ent_ico} {_ent_badge}</span>"
                f"<span style='background:{_fno_bg2};color:{_fno_clr2};font-size:10px;"
                f"font-weight:700;border-radius:4px;padding:2px 8px;"
                f"border:1px solid {_fno_bdr2}'>{_fno_badge2}</span>"
                f"<span style='background:{_sw_beta_bg};color:{_sw_beta_clr};"
                f"font-size:10px;font-weight:700;border-radius:4px;padding:2px 8px;"
                f"border:1px solid {_sw_beta_bdr}'>"
                f"{_sw_beta_ico} β {_sw_beta:.2f} {_sw_beta_grade} ({_sw_beta_ss}pts)</span>"
                f"</div>"
                f"{_fno_note_html}"
                f"<div style='font-size:12px;color:#64748b;margin-top:5px'>"
                f"<span style='color:{_cage_clr};font-weight:700'>{_cage_lbl}</span>"
                f"&nbsp;·&nbsp; RSI {_rsi}"
                f"&nbsp;·&nbsp; Vol {_volx}×"
                f"&nbsp;·&nbsp; <span style='color:{_wchg_clr}'>Week {_wchg:+.1f}%</span>"
                f"&nbsp;·&nbsp; <span style='color:{_slope_clr}'>SMA20 slope {_sw_slope:+.2f}%</span>"
                f"&nbsp;·&nbsp; <span style='color:{_sw_sec_rank_clr};font-weight:700'>"
                f"{_sw_sec_rank_ico} Sector {_sw_sec_name} Rank #{_sw_sec_rank} ({_sw_sec_rs_s} vs Nifty)</span>"
                f"</div>"
                f"</div>"
                f"<div style='text-align:right'>"
                f"<div style='font-size:24px;font-weight:800;color:#1a2035;font-family:JetBrains Mono'>"
                f"₹{_close:,.2f}</div>"
                f"<div style='font-size:11px;color:#64748b'>"
                f"SMA20 ₹{_sma20:,.2f} · SMA50 ₹{_sma50:,.2f}</div>"
                f"</div></div>",
                unsafe_allow_html=True)

            # ── SMA bar ─────────────────────────────────
            _atr14_val = _sw_r.get('atr14', _atr)
            _atr_lbl2  = _sw_r.get('atr_label','weekly')
            st.markdown(
                f"<div style='background:#f8fafc;border-radius:8px;padding:10px 14px;"
                f"margin-bottom:12px;font-size:11px;color:#64748b'>"
                f"<div style='display:flex;justify-content:space-between;margin-bottom:4px'>"
                f"<span>SMA50 ₹{_sma50:,.2f}</span>"
                f"<span style='color:#d97706;font-weight:700'>SMA20 ₹{_sma20:,.2f}</span>"
                f"<span style='color:#1a2035;font-weight:700'>Price ₹{_close:,.2f}</span>"
                f"</div>"
                f"<div style='background:#e2e8f0;border-radius:4px;height:6px;position:relative'>"
                f"<div style='background:#d97706;height:6px;border-radius:4px;width:60%'></div>"
                f"<div style='background:#1a2035;height:10px;width:3px;border-radius:2px;"
                f"position:absolute;top:-2px;left:72%'></div></div>"
                f"<div style='font-size:10px;color:#94a3b8;margin-top:3px'>"
                f"ATR = ₹{_atr:,.2f} ({_atr_lbl2} ATR · daily ₹{_atr14_val:,.2f}) · Risk/share = ₹{_rd:,.2f}"
                f"</div>"
                + (f"<div style='font-size:10px;color:{_sw_r.get('pb_vol_clr','#64748b')};"
                   f"font-weight:700;margin-top:4px'>"
                   f"{_sw_r.get('pb_vol_label','')} "
                   f"({_sw_r.get('pb_vol_ratio',1.0):.2f}× avg vol on pullback)</div>"
                   if _sw_r.get('pb_vol_label') else "")
                + (f"<div style='font-size:10px;font-weight:700;margin-top:4px;"
                   f"color:{'#15803d' if _sw_r.get('wp_score',0)>0 else '#dc2626'}'>"
                   f"📍 Candle close: {_sw_r.get('wp_label','')} "
                   f"(pos {_sw_r.get('week_pos',0.5):.0%} of range)"
                   f"</div>")
                + (f"<div style='font-size:10px;font-weight:700;margin-top:4px;"
                   f"color:{'#15803d' if _sw_r.get('cb_score',0)>0 else '#d97706' if _sw_r.get('cb_score',0)==0 else '#dc2626'}'>"
                   f"📊 Body momentum: {_sw_r.get('cb_label','')} "
                   f"({'+' if _sw_r.get('cb_score',0)>=0 else ''}{_sw_r.get('cb_score',0)}pts)"
                   f"</div>")
                + (f"<div style='font-size:11px;font-weight:800;margin-top:5px;padding:4px 8px;"
                   f"background:{'#f0fdf4' if _sw_r.get('squeeze_fired') else '#fffbeb' if _sw_r.get('squeeze_on') else '#f8fafc'};"
                   f"border-radius:6px;border:1px solid "
                   f"{'#86efac' if _sw_r.get('squeeze_fired') else '#fcd34d' if _sw_r.get('squeeze_on') else '#e2e8f0'};"
                   f"color:{_sw_r.get('squeeze',{}).get('clr','#64748b')}'>"
                   f"{_sw_r.get('squeeze',{}).get('ico','➡️')} "
                   f"{_sw_r.get('squeeze_label','')}"
                   + (f" · BB {_sw_r.get('squeeze',{}).get('bb_width_change',0):+.0f}% width"
                      if _sw_r.get('squeeze_fired') else "")
                   + f" ({'+' if _sw_r.get('squeeze_score',0)>=0 else ''}"
                   f"{_sw_r.get('squeeze_score',0)}pts)</div>"
                   if _sw_r.get('squeeze_label') else "")
                + (f"<div style='font-size:10px;font-weight:700;margin-top:4px;"
                   f"color:{_sw_r.get('adx_clr','#64748b')}'>"
                   f"{_sw_r.get('adx_label','')}</div>"
                   if _sw_r.get('adx_label') else "")
                + (f"<div style='font-size:10px;font-weight:700;margin-top:4px;"
                   f"color:{_sw_r.get('rs_sec_clr','#64748b')}'>"
                   f"📊 {_sw_r.get('rs_sec_label','')} "
                   f"({'+' if _sw_r.get('rs_sec_score',0)>=0 else ''}{_sw_r.get('rs_sec_score',0)}pts)"
                   f"</div>"
                   if _sw_r.get('rs_sec_label') else "")
                + (f"<div style='font-size:10px;font-weight:700;margin-top:4px;"
                   f"color:{_sw_r.get('risk_clr','#64748b')}'>"
                   f"⚖️ Position sizing: {_sw_r.get('risk_label','')} "
                   f"{'— ' + _sw_r.get('risk_reason','') if _sw_r.get('risk_reason') else ''}"
                   f"</div>"
                   if _sw_r.get('risk_reason','') != 'No adjustment' else "")
                + "</div>",
                unsafe_allow_html=True)

            # ── Trend factors row ────────────────────────
            _hh      = _sw_r.get('hh', False)
            _hl      = _sw_r.get('hl', False)
            _ext     = _sw_r.get('extended', False)
            _sma50p  = _sw_r.get('pct_above_sma50', 0)
            _sl_str  = _sw_r.get('sma20_slope', 0)
            _sl_clr  = '#15803d' if _sl_str >= 0.5 else ('#d97706' if _sl_str > 0 else '#dc2626')
            _sl_lbl  = 'Strong ↑' if _sl_str >= 1.0 else ('Rising ↑' if _sl_str >= 0.5 else 'Weak ↑')
            _hh_clr  = '#15803d' if _hh else '#dc2626'
            _hl_clr  = '#15803d' if _hl else '#dc2626'
            _ext_clr = '#dc2626' if _ext else '#15803d'
            st.markdown(
                f"<div style='background:#f8fafc;border-radius:8px;padding:8px 14px;"
                f"margin-bottom:10px;display:flex;gap:16px;flex-wrap:wrap;font-size:11px'>"
                f"<span>📐 SMA20 slope: <b style='color:{_sl_clr}'>{_sl_lbl} {_sl_str:+.2f}%</b></span>"
                f"<span>📏 vs SMA50: <b style='color:{_ext_clr}'>{'⚠️ Extended' if _ext else '✅ Normal'} +{_sma50p:.1f}%</b></span>"
                f"<span>📅 Trend: <b style='color:#1d4ed8'>{_sw_r.get('trend_days',0)}d</b></span>"
                f"</div>",
                unsafe_allow_html=True)

            # ── Targets ─────────────────────────────────
            _sl_pct  = round((_entry - _sl)  / _entry * 100, 2) if _entry > 0 else 0
            _t1_pct  = round((_t1 - _entry)  / _entry * 100, 2) if _entry > 0 else 0
            _t2_pct  = round((_t2 - _entry)  / _entry * 100, 2) if _entry > 0 else 0
            _t3_pct  = round((_t3 - _entry)  / _entry * 100, 2) if _entry > 0 else 0
            _sw_atr_val   = _sw_r.get('atr', 0)
            _sw_rd        = _sw_r.get('risk_per_share', _entry - _sl)
            _sw_adj_risk  = _sw_r.get('adj_risk_pct', _sw_risk_pct)
            _sw_risk_lbl  = _sw_r.get('risk_label', '')
            _sw_risk_rsn  = _sw_r.get('risk_reason', '')
            st.markdown(
                f"<div style='display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px'>"
                # STOP LOSS box — enhanced with ATR info
                f"<div style='background:#fee2e2;border:2px solid #fca5a5;border-radius:10px;"
                f"padding:10px 14px;flex:1;min-width:80px;text-align:center'>"
                f"<div style='font-size:9px;font-weight:700;color:#dc2626;letter-spacing:1px'>STOP LOSS</div>"
                f"<div style='font-size:17px;font-weight:800;color:#dc2626;"
                f"font-family:JetBrains Mono;margin:3px 0'>₹{_sl:,.2f}</div>"
                f"<div style='font-size:10px;color:#dc2626'>−{_sl_pct:.2f}% below SMA20</div>"
                f"<div style='font-size:9px;color:#b91c1c;margin-top:3px;font-weight:700'>"
                f"ATR ₹{_sw_atr_val:,.2f} · Risk ₹{_sw_rd:,.2f}/share</div>"
                f"</div>"
                f"<div style='background:#eff6ff;border-radius:10px;padding:10px 14px;"
                f"flex:1;min-width:80px;text-align:center'>"
                f"<div style='font-size:9px;font-weight:700;color:#1d4ed8;letter-spacing:1px'>T1 — R:R {_rr1}:1</div>"
                f"<div style='font-size:17px;font-weight:800;color:#1d4ed8;font-family:JetBrains Mono;margin:3px 0'>₹{_t1:,.2f}</div>"
                f"<div style='font-size:10px;color:#1d4ed8'>+{_t1_pct:.2f}% · 0.5× ATR · Book 50%</div></div>"
                f"<div style='background:#f5f3ff;border-radius:10px;padding:10px 14px;"
                f"flex:1;min-width:80px;text-align:center'>"
                f"<div style='font-size:9px;font-weight:700;color:#7c3aed;letter-spacing:1px'>T2 — R:R {_rr2}:1</div>"
                f"<div style='font-size:17px;font-weight:800;color:#7c3aed;font-family:JetBrains Mono;margin:3px 0'>₹{_t2:,.2f}</div>"
                f"<div style='font-size:10px;color:#7c3aed'>+{_t2_pct:.2f}% · 1.0× ATR · Trail SL</div></div>"
                f"<div style='background:#f0fdf4;border-radius:10px;padding:10px 14px;"
                f"flex:1;min-width:80px;text-align:center'>"
                f"<div style='font-size:9px;font-weight:700;color:#15803d;letter-spacing:1px'>T3 — STRETCH</div>"
                f"<div style='font-size:17px;font-weight:800;color:#15803d;font-family:JetBrains Mono;margin:3px 0'>₹{_t3:,.2f}</div>"
                f"<div style='font-size:10px;color:#15803d'>+{_t3_pct:.2f}% · 1.5× ATR · Let run</div></div>"
                f"</div>",
                unsafe_allow_html=True)

            # ── Position sizing info ─────────────────────
            # Shows ATR-based qty + dynamic risk adjustment
            _sw_max_loss = int(_qty * _sw_rd)
            _sw_orig_qty = max(1, int((_sw_capital * _sw_risk_pct / 100) / _sw_rd)) \
                           if _sw_rd > 0 else _qty
            _sw_size_reduced = _qty < _sw_orig_qty * 0.95
            st.markdown(
                f"<div style='background:#f8fafc;border:1.5px solid "
                f"{'#fcd34d' if _sw_size_reduced else '#e2e8f0'};"
                f"border-radius:10px;padding:10px 14px;margin-bottom:8px'>"
                # Row 1 — main position info
                f"<div style='display:flex;gap:20px;flex-wrap:wrap;font-size:11px;color:#64748b'>"
                f"<span>📦 Qty: <b style='color:#1a2035'>{_qty} shares</b></span>"
                f"<span>💰 Invest: <b style='color:#1a2035'>₹{_inv:,.0f}</b></span>"
                f"<span>⚠️ Max loss if SL hit: <b style='color:#dc2626'>₹{_sw_max_loss:,}</b></span>"
                f"<span>🎯 Capital: <b style='color:#1a2035'>₹{_sw_capital:,.0f}</b></span>"
                f"</div>"
                # Row 2 — ATR sizing explanation
                f"<div style='margin-top:6px;padding-top:6px;border-top:1px solid #e2e8f0;"
                f"font-size:10px;color:#64748b'>"
                f"📐 <b>ATR sizing:</b> ₹{_sw_capital:,.0f} × {_sw_adj_risk:.2f}% ÷ ₹{_sw_rd:.2f}/share = "
                f"<b style='color:#1a2035'>{_qty} shares</b>"
                + (f" &nbsp;<span style='color:#d97706;font-weight:700'>"
                   f"(reduced from {_sw_orig_qty} — {_sw_risk_rsn})</span>"
                   if _sw_size_reduced else "")
                + f"</div>"
                # Row 3 — risk label if adjusted
                + (f"<div style='margin-top:4px;font-size:10px;font-weight:700;"
                   f"color:{_sw_r.get('risk_clr','#64748b')}'>"
                   f"⚖️ {_sw_risk_lbl}</div>"
                   if _sw_size_reduced else "")
                + f"</div>",
                unsafe_allow_html=True)

            # ── Confident Score Breakdown Strip ─────────────
            st.markdown(
                f"<div style='background:{_conf_bg};border:2px solid {_conf_bdr};"
                f"border-radius:10px;padding:10px 16px;margin-bottom:8px'>"
                f"<div style='display:flex;align-items:center;justify-content:space-between;"
                f"flex-wrap:wrap;gap:8px'>"
                f"<div>"
                f"<div style='font-size:11px;font-weight:700;color:{_conf_clr};"
                f"letter-spacing:1px'>⭐ CONFIDENT SCORE</div>"
                f"<div style='font-size:22px;font-weight:800;color:{_conf_clr};"
                f"margin-top:2px'>{_conf}/100 "
                f"<span style='font-size:13px'>{_conf_lbl}</span></div>"
                f"</div>"
                f"<div style='display:flex;gap:6px;flex-wrap:wrap'>"
                f"<div style='background:white;border-radius:6px;padding:4px 8px;"
                f"text-align:center;min-width:44px'>"
                f"<div style='font-size:8px;color:#94a3b8;font-weight:700'>TECH</div>"
                f"<div style='font-size:13px;font-weight:800;color:#1a2035'>{_c1}/30</div>"
                f"</div>"
                f"<div style='background:white;border-radius:6px;padding:4px 8px;"
                f"text-align:center;min-width:44px'>"
                f"<div style='font-size:8px;color:#94a3b8;font-weight:700'>PSAR</div>"
                f"<div style='font-size:13px;font-weight:800;"
                f"color:{'#15803d' if _c2>0 else '#dc2626'}'>{_c2}/25</div>"
                f"</div>"
                f"<div style='background:white;border-radius:6px;padding:4px 8px;"
                f"text-align:center;min-width:44px'>"
                f"<div style='font-size:8px;color:#94a3b8;font-weight:700'>STRUCT</div>"
                f"<div style='font-size:13px;font-weight:800;color:#1a2035'>{_c3}/15</div>"
                f"</div>"
                f"<div style='background:white;border-radius:6px;padding:4px 8px;"
                f"text-align:center;min-width:44px'>"
                f"<div style='font-size:8px;color:#94a3b8;font-weight:700'>ENTRY</div>"
                f"<div style='font-size:13px;font-weight:800;color:#1a2035'>{_c4}/15</div>"
                f"</div>"
                f"<div style='background:white;border-radius:6px;padding:4px 8px;"
                f"text-align:center;min-width:44px'>"
                f"<div style='font-size:8px;color:#94a3b8;font-weight:700'>R:R</div>"
                f"<div style='font-size:13px;font-weight:800;color:#1a2035'>{_c5}/10</div>"
                f"</div>"
                f"<div style='background:white;border-radius:6px;padding:4px 8px;"
                f"text-align:center;min-width:44px'>"
                f"<div style='font-size:8px;color:#94a3b8;font-weight:700'>LIQ</div>"
                f"<div style='font-size:13px;font-weight:800;color:#1a2035'>{_c6}/5</div>"
                f"</div>"
                f"<div style='background:white;border-radius:6px;padding:4px 8px;"
                f"text-align:center;min-width:44px'>"
                f"<div style='font-size:8px;color:#94a3b8;font-weight:700'>F&O</div>"
                f"<div style='font-size:13px;font-weight:800;"
                f"color:{'#15803d' if _sw_r.get('c7_fno',0)>0 else '#dc2626' if _sw_r.get('c7_fno',0)<0 else '#1a2035'}'>"
                f"{'+' if _sw_r.get('c7_fno',0)>0 else ''}{_sw_r.get('c7_fno',0)}</div>"
                f"</div>"
                f"<div style='background:white;border-radius:6px;padding:4px 8px;"
                f"text-align:center;min-width:54px'>"
                f"<div style='font-size:8px;color:#94a3b8;font-weight:700'>SECTOR</div>"
                f"<div style='font-size:13px;font-weight:800;"
                f"color:{'#15803d' if _sw_r.get('c8_sector',0)>0 else '#dc2626' if _sw_r.get('c8_sector',0)<0 else '#1a2035'}'>"
                f"{'+' if _sw_r.get('c8_sector',0)>0 else ''}{_sw_r.get('c8_sector',0)}</div>"
                f"<div style='font-size:8px;color:#64748b'>{_sw_r.get('c8_sector_name','')}</div>"
                f"</div>"
                f"</div></div></div>",
                unsafe_allow_html=True)

            # ── Price Action Analysis Strip ─────────────────
            _sw_pa_data = _sw_r.get('pa', {})
            if _sw_pa_data:
                _sw_pa_sig  = _sw_pa_data.get('pa_signal', '')
                _sw_pa_clr  = _sw_pa_data.get('pa_signal_clr', '#64748b')
                _sw_pa_bg   = _sw_pa_data.get('pa_signal_bg', '#f8fafc')
                _sw_pa_tot  = _sw_pa_data.get('pa_total_score', 0)
                _sw_pa_cico = _sw_pa_data.get('candle_emoji', '⚪')
                _sw_pa_cpat = _sw_pa_data.get('candle_pattern', '')
                _sw_pa_csc  = _sw_pa_data.get('candle_score', 0)
                _sw_pa_cdsc = _sw_pa_data.get('candle_desc', '')
                _sw_pa_snam = _sw_pa_data.get('support_name', '')
                _sw_pa_ssc  = _sw_pa_data.get('support_score', 0)
                _sw_pa_sdsc = _sw_pa_data.get('support_desc', '')
                _sw_pa_st   = _sw_pa_data.get('structure', '')
                _sw_pa_stsc = _sw_pa_data.get('structure_score', 0)
                _sw_pa_stds = _sw_pa_data.get('structure_desc', '')
                _sw_pa_cclr = '#15803d' if _sw_pa_csc > 0 else ('#dc2626' if _sw_pa_csc < 0 else '#64748b')
                _sw_pa_sclr = '#15803d' if _sw_pa_ssc > 0 else ('#dc2626' if _sw_pa_ssc < 0 else '#64748b')
                _sw_pa_stcl = '#15803d' if _sw_pa_stsc> 0 else ('#dc2626' if _sw_pa_stsc< 0 else '#64748b')
                _sw_pa_sign = '+' if _sw_pa_tot >= 0 else ''
                st.markdown(
                    f"<div style='background:{_sw_pa_bg};border:1.5px solid {_sw_pa_clr}33;"
                    f"border-radius:10px;padding:12px 16px;margin-bottom:8px'>"
                    f"<div style='display:flex;align-items:center;justify-content:space-between;"
                    f"flex-wrap:wrap;gap:8px;margin-bottom:8px'>"
                    f"<div style='font-size:10px;font-weight:700;color:{_sw_pa_clr};"
                    f"letter-spacing:1px'>📊 PRICE ACTION ANALYSIS</div>"
                    f"<div style='display:flex;align-items:center;gap:8px'>"
                    f"<span style='font-size:10px;font-weight:700;color:{_sw_pa_clr};"
                    f"background:white;border-radius:4px;padding:2px 8px;"
                    f"border:1px solid {_sw_pa_clr}44'>PA Score {_sw_pa_sign}{_sw_pa_tot}</span>"
                    f"<span style='font-size:12px;font-weight:800;color:{_sw_pa_clr}'>{_sw_pa_sig}</span>"
                    f"</div></div>"
                    f"<div style='display:flex;gap:8px;flex-wrap:wrap'>"
                    f"<div style='background:white;border-radius:8px;padding:8px 12px;flex:1;min-width:140px'>"
                    f"<div style='font-size:9px;font-weight:700;color:#94a3b8;letter-spacing:1px'>CANDLE</div>"
                    f"<div style='font-size:12px;font-weight:700;color:{_sw_pa_cclr};margin-top:2px'>"
                    f"{_sw_pa_cico} {_sw_pa_cpat} ({'+' if _sw_pa_csc>=0 else ''}{_sw_pa_csc})</div>"
                    f"<div style='font-size:10px;color:#64748b;margin-top:2px'>{_sw_pa_cdsc}</div>"
                    f"</div>"
                    f"<div style='background:white;border-radius:8px;padding:8px 12px;flex:1;min-width:140px'>"
                    f"<div style='font-size:9px;font-weight:700;color:#94a3b8;letter-spacing:1px'>SUPPORT</div>"
                    f"<div style='font-size:12px;font-weight:700;color:{_sw_pa_sclr};margin-top:2px'>"
                    f"{_sw_pa_snam} ({'+' if _sw_pa_ssc>=0 else ''}{_sw_pa_ssc})</div>"
                    f"<div style='font-size:10px;color:#64748b;margin-top:2px'>{_sw_pa_sdsc[:55]}</div>"
                    f"</div>"
                    f"<div style='background:white;border-radius:8px;padding:8px 12px;flex:1;min-width:140px'>"
                    f"<div style='font-size:9px;font-weight:700;color:#94a3b8;letter-spacing:1px'>STRUCTURE</div>"
                    f"<div style='font-size:12px;font-weight:700;color:{_sw_pa_stcl};margin-top:2px'>"
                    f"{_sw_pa_st} ({'+' if _sw_pa_stsc>=0 else ''}{_sw_pa_stsc})</div>"
                    f"<div style='font-size:10px;color:#64748b;margin-top:2px'>{_sw_pa_stds[:55]}</div>"
                    f"</div></div></div>",
                    unsafe_allow_html=True)

            # ── PSAR Trailing SL Display ──────────────────
            _sw_psar_v = _sw_r.get('psar', None)
            _sw_psar_b = _sw_r.get('psar_bullish', False)
            if _sw_psar_v:
                _sp_clr = '#15803d' if _sw_psar_b else '#dc2626'

                _sp_bg  = '#f0fdf4' if _sw_psar_b else '#fef2f2'
                _sp_bdr = '#86efac' if _sw_psar_b else '#fca5a5'
                _sp_ico = '✅' if _sw_psar_b else '⚠️'
                _sp_lbl = 'Bullish — hold' if _sw_psar_b else 'Weak — caution'
                _sp_pct = round((_entry - _sw_psar_v) / _entry * 100, 1)
                st.markdown(f"""
                <div style='background:{_sp_bg};border:1px solid {_sp_bdr};
                            border-radius:8px;padding:10px 16px;margin-bottom:6px;
                            display:flex;align-items:center;gap:16px;flex-wrap:wrap'>
                    <div>
                        <div style='font-size:10px;font-weight:700;color:{_sp_clr};
                                    letter-spacing:1px'>
                            📍 DAILY PSAR — TRAILING SL AFTER T1
                        </div>
                        <div style='font-size:18px;font-weight:800;color:{_sp_clr};
                                    font-family:JetBrains Mono;margin-top:2px'>
                            ₹{_sw_psar_v:,.2f}
                            <span style='font-size:11px;margin-left:8px'>
                                {_sp_ico} {_sp_lbl}
                            </span>
                        </div>
                    </div>
                    <div style='font-size:11px;color:#64748b;line-height:1.8'>
                        <b>{_sp_pct:.1f}% below entry</b> ·
                        After T1 hit → move Zerodha SL to ₹{_sw_psar_v:,.2f}<br>
                        Check each morning · If daily close &lt; PSAR → EXIT
                    </div>
                </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("</div>", unsafe_allow_html=True)

            # ── Paper Buy button ──────────────────────────
            _sw_pb_key = f"sw_paper_buy_{_sym}_{_sw_r.get('cross_age',0)}"
            if st.button(
                f"✅ Paper Buy  {_sym}  ·  Entry ₹{_entry:,.2f}  ·  SL ₹{_sl:,.2f}  ·  T1 ₹{_t1:,.2f}  ·  Qty {_qty}",
                key=_sw_pb_key,
                use_container_width=True,
                type="primary"
            ):
                _sw_port = load_portfolio()
                _sw_already = any(
                    p.get('symbol') == _sym and p.get('status') == 'OPEN'
                    for p in _sw_port
                )
                if _sw_already:
                    st.warning(f"⚠️ Already have open position in {_sym}")
                else:
                    _sw_port.append({
                        'symbol':      _sym,
                        'status':      'OPEN',
                        'entry':       round(_entry, 2),
                        'qty':         _qty,
                        'stop_loss':   _sl,
                        'atr':         round(_sw_r.get('atr', _sw_r.get('atr7', 0)), 2),
                        'risk_per_share': round(_entry - _sl, 2),
                        't1':          _t1,
                        't2':          _t2,
                        't3':          _t3,
                        't4':          0,
                        'investment':  _inv,
                        'actual_cost': _inv,
                        'timeframe':   'Daily — SMA Weekly',
                        'date':        ist_now().strftime('%d %b %Y %H:%M'),
                        'entry_time':  ist_now().strftime('%H:%M'),
                        'nifty_state': st.session_state.get('nifty_market_state', 'UNKNOWN'),
                        'vix_level':   st.session_state.get('nifty_context', {}).get('vix_level', 'UNKNOWN'),
                        'score':       _sc,
                        'verdict':     _cage_lbl,
                        'sig_age':     _sw_r.get('cross_age', 0),
                        'vol_ratio':   _volx,
                        'source':      'sma_weekly',
                        'exit_reason': '',
                        'cap_tier':    _cap,
                        'sma20':       _sma20,
                        'sma50':       _sma50,
                        'signal_type': _sw_r.get('signal_type', ''),
                    })
                    save_portfolio(_sw_port)
                    st.session_state['paper_portfolio'] = _sw_port
                    st.success(
                        f"✅ Paper bought {_qty} × {_sym} @ ₹{_entry:,.2f} · "
                        f"SL ₹{_sl:,.2f} · T1 ₹{_t1:,.2f} · "
                        f"Hold 3–7 days · Source: SMA Weekly"
                    )
                    st.rerun()

        # No more results
        if len(_sw_filtered) == 0:
            st.info("No signals match this filter. Try 'All' or lower the min score.")



# ─────────────────────────────────────────────────────────────
#  MONTHLY SWING SCANNER
#  For working professionals — weekend-only monitoring
#  Uses WEEKLY candles for signals and targets
#  Hold 3-5 weeks (one calendar month)
#  Check only on weekends — no intraday monitoring needed
# ─────────────────────────────────────────────────────────────

if _show_monthlyswing:

    # ── Local confident score function (dict-based) ────
    def calc_confident_score(r):
        # C1 Technical (25pts) — calibrated to real raw score range
        sc = r.get('score', 0)
        if   sc >= 90: c1 = 25
        elif sc >= 80: c1 = 22
        elif sc >= 70: c1 = 18
        elif sc >= 60: c1 = 14
        elif sc >= 50: c1 = 10
        else:          c1 = 6
        # C2 PSAR (20pts)
        c2 = 20 if r.get('psar_bullish', False) else 0
        # C3 PA Structure (15pts)
        _pa_struct = r.get('pa', {}).get('structure', '')
        if   'Bullish' in _pa_struct: c3 = 15
        elif 'Neutral' in _pa_struct: c3 = 8
        elif 'Broken'  in _pa_struct: c3 = 0
        else:
            hh = r.get('hh', False); hl = r.get('hl', False)
            c3 = 15 if (hh and hl) else (8 if (hh or hl) else 0)
        # C4 Entry Badge (15pts)
        badge = r.get('entry_badge', 'ACCEPTABLE')
        c4 = 15 if badge == 'ENTER NOW' else (8 if badge == 'ACCEPTABLE' else 0)
        # C5 R:R (10pts)
        rr2 = r.get('rr_t2', r.get('rr2', 0))
        try: rr2 = float(rr2)
        except: rr2 = 0
        c5 = 10 if rr2 >= 3.0 else (8 if rr2 >= 2.0 else (5 if rr2 >= 1.5 else 0))
        # C6 Liquidity (5pts)
        liq = r.get('liq_grade', '')
        c6 = 5 if liq == 'EXCELLENT' else (3 if liq == 'HIGH' else (1 if liq == 'MEDIUM' else 0))
        # C7 F&O Expiry (±15pts)
        c7 = r.get('fno_penalty', 0)
        # C8 Sector Ranking (±10pts) — unified function
        _sym_r = r.get('symbol', r.get('sym', ''))
        _sec_r, _sec_rank_r, _sec_rs_r, _sec_bull_r, _, c8, _, _ = \
            get_sector_score_for_stock(_sym_r, formula='monthly')
        total = c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8
        if   total >= 130: label='🔥 CONFIDENT BUY'; clr='#15803d'; bg='#f0fdf4'; bdr='#86efac'
        elif total >= 100: label='✅ STRONG SETUP';  clr='#0369a1'; bg='#f0f9ff'; bdr='#7dd3fc'
        elif total >= 75:  label='👍 GOOD SETUP';    clr='#1d4ed8'; bg='#eff6ff'; bdr='#93c5fd'
        elif total >= 55:  label='⚠️ WEAK SETUP';    clr='#d97706'; bg='#fffbeb'; bdr='#fcd34d'
        else:              label='❌ SKIP';           clr='#dc2626'; bg='#fef2f2'; bdr='#fca5a5'
        return {'confident_score':total,'confident_label':label,'confident_clr':clr,
                'confident_bg':bg,'confident_bdr':bdr,
                'c1_tech':c1,'c2_psar':c2,'c3_struct':c3,'c4_badge':c4,
                'c5_rr':c5,'c6_liq':c6,'c7_fno':c7,'c8_sector':c8,
                'c8_sector_name':_sec_r,'c8_sector_rank':_sec_rank_r}

    # ── Data source badge ──────────────────────────────
    _ms_kite = get_kite_client() is not None
    _ms_src  = 'Kite API' if _ms_kite else 'yfinance — Weekly candles'
    _ms_sclr = '#16a34a' if _ms_kite else '#d97706'
    _ms_sbg  = '#dcfce7' if _ms_kite else '#fef3c7'
    _ms_sico = '🟢' if _ms_kite else '🟡'

    # Nifty weekly status (cached from last scan)
    _ms_nifty_status = st.session_state.get('ms_nifty_bullish', None)
    _ms_nifty_lbl    = ('🟢 Nifty Weekly Bullish' if _ms_nifty_status == True
                        else '🔴 Nifty Weekly Bearish' if _ms_nifty_status == False
                        else '⚪ Nifty — Run scan')
    _ms_sico = '🟢' if _ms_kite else '🟡'

    st.markdown(f"""
    <div class='topbar'>
        <div>
            <div class='topbar-title'>📅 Monthly Swing Scanner</div>
            <div class='topbar-subtitle'>
                Weekly chart · SMA20 + SMA50 · Hold 3–5 weeks ·
                Weekend-only monitoring · 2–3 positions
            </div>
        </div>
        <div style='display:flex;align-items:center'>
            <div style='background:{_ms_sbg};border:1px solid {_ms_sclr}44;
                        border-radius:8px;padding:5px 12px;text-align:center'>
                <div style='font-size:10px;font-weight:700;color:{_ms_sclr};
                            letter-spacing:1px'>DATA SOURCE</div>
                <div style='font-size:12px;font-weight:700;color:{_ms_sclr};
                            margin-top:2px'>{_ms_sico} {_ms_src}</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    # ── Strategy cards ─────────────────────────────────
    st.markdown("""
    <div style='display:flex;gap:10px;margin-bottom:16px;flex-wrap:wrap'>
        <div style='flex:1;min-width:140px;background:#eff6ff;border-radius:10px;
                    padding:12px 14px;border:1px solid #bfdbfe44'>
            <div style='font-size:20px;margin-bottom:6px'>📅</div>
            <div style='font-size:12px;font-weight:700;color:#1d4ed8'>Hold Duration</div>
            <div style='font-size:11px;color:#1e40af;margin-top:3px'>
                3–5 weeks per trade<br>
                Check only on weekends<br>
                Perfect for 9–5 professionals
            </div>
        </div>
        <div style='flex:1;min-width:140px;background:#f0fdf4;border-radius:10px;
                    padding:12px 14px;border:1px solid #bbf7d044'>
            <div style='font-size:20px;margin-bottom:6px'>📊</div>
            <div style='font-size:12px;font-weight:700;color:#15803d'>Timeframe</div>
            <div style='font-size:11px;color:#166534;margin-top:3px'>
                Weekly candles only<br>
                1 candle = 1 trading week<br>
                Filters out daily noise
            </div>
        </div>
        <div style='flex:1;min-width:140px;background:#fff5f5;border-radius:10px;
                    padding:12px 14px;border:1px solid #fecaca44'>
            <div style='font-size:20px;margin-bottom:6px'>🎯</div>
            <div style='font-size:12px;font-weight:700;color:#dc2626'>Targets</div>
            <div style='font-size:11px;color:#991b1b;margin-top:3px'>
                T1 = +1.0× wkly ATR (~8–12%)<br>
                T2 = +2.0× wkly ATR (~15–20%)<br>
                T3 = +3.0× wkly ATR (~20–30%)
            </div>
        </div>
        <div style='flex:1;min-width:140px;background:#fffbeb;border-radius:10px;
                    padding:12px 14px;border:1px solid #fde68a44'>
            <div style='font-size:20px;margin-bottom:6px'>🛑</div>
            <div style='font-size:12px;font-weight:700;color:#d97706'>Stop Loss</div>
            <div style='font-size:11px;color:#92400e;margin-top:3px'>
                2× weekly ATR below entry<br>
                Or below weekly SMA20<br>
                Check only on Friday close
            </div>
        </div>
        <div style='flex:1;min-width:140px;background:#f5f3ff;border-radius:10px;
                    padding:12px 14px;border:1px solid #ddd6fe44'>
            <div style='font-size:20px;margin-bottom:6px'>⏰</div>
            <div style='font-size:12px;font-weight:700;color:#7c3aed'>Weekend Workflow</div>
            <div style='font-size:11px;color:#6d28d9;margin-top:3px'>
                Saturday: Run scan (15 min)<br>
                Sunday: Place limit orders<br>
                Monday: Confirm entries only
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    # ── Controls ───────────────────────────────────────
    _ms_c1, _ms_c2, _ms_c3 = st.columns([2, 1, 1])
    with _ms_c1:
        _ms_universe = st.radio(
            "Universe",
            ["🔵 Largecap (Nifty 50)",
             "🟡 Midcap (Nifty Midcap 100)",
             "🟠 Smallcap",
             "📊 Nifty 500 (All)",
             "📁 Upload My List"],
            horizontal=True, key="ms_universe",
            help="Midcap recommended for monthly swing — best return potential. "
                 "Upload My List = scan your own CSV of NSE symbols.")

        if _ms_universe == "📁 Upload My List":
            _ms_csv_file = st.file_uploader(
                "Upload CSV/Excel with NSE symbols",
                type=['csv', 'xlsx', 'xls'],
                key="ms_csv_upload",
                help="Any NSE export works — needs a 'Symbol' column "
                     "(or symbols in the first column). "
                     "e.g. ind_niftyautolist.csv from NSE website, "
                     "or your own watchlist export.")
            if _ms_csv_file is not None:
                _ms_csv_stocks, _ms_csv_err = parse_csv_stock_list(_ms_csv_file)
                if _ms_csv_err:
                    st.error(f"❌ {_ms_csv_err}")
                    _ms_stocks = POPULAR_STOCKS
                else:
                    st.success(f"✅ Loaded {len(_ms_csv_stocks)} symbols from "
                               f"{_ms_csv_file.name}")
                    _ms_stocks = _ms_csv_stocks
            else:
                st.info("⬆️ Upload a file to scan your own stock list")
                _ms_stocks = []
        else:
            _ms_stocks = (
                LARGECAP_STOCKS if _ms_universe == "🔵 Largecap (Nifty 50)"       else
                MIDCAP_STOCKS   if _ms_universe == "🟡 Midcap (Nifty Midcap 100)" else
                SMALLCAP_STOCKS if _ms_universe == "🟠 Smallcap"                  else
                POPULAR_STOCKS
            )
        st.markdown(
            f"<div style='font-size:11px;color:#64748b;margin-top:-8px'>"
            f"⚡ {len(_ms_stocks)} stocks · Weekly candles · SMA20 + SMA50</div>",
            unsafe_allow_html=True)
    with _ms_c2:
        _ms_capital = st.number_input(
            "Capital ₹", min_value=50000, max_value=5000000,
            value=500000, step=50000, format="%d", key="ms_capital",
            help="Capital per trade for monthly swing")
    with _ms_c3:
        _ms_risk = st.number_input(
            "Risk %", min_value=0.5, max_value=5.0,
            value=2.0, step=0.5, format="%.1f", key="ms_risk",
            help="Max risk per trade — wider for monthly holds")

    _ms_min_score = st.slider(
        "Min signal score", min_value=50, max_value=90, value=65,
        step=5, key="ms_min_score",
        help="Higher = fewer but stronger signals")

    # ── Strict Entry Mode ──────────────────────────────
    _ms_strict = st.checkbox(
        "🛡️ Strict Entry Mode",
        value=True,
        key="ms_strict_mode",
        help=(
            "When ON — only shows stocks where:\n"
            "✅ PSAR is BULLISH (price above PSAR)\n"
            "✅ Candle = Hammer OR Bullish Engulfing\n"
            "✅ Confident Score ≥ 70\n\n"
            "Fewer signals but much higher quality\n"
            "Stocks that hold even when Nifty dips\n\n"
            "When OFF — shows all signals including\n"
            "bearish PSAR and mild candles"
        ))
    if _ms_strict:
        st.markdown(
            "<div style='background:#f0fdf4;border:1.5px solid #86efac;"
            "border-radius:8px;padding:8px 14px;font-size:11px;"
            "color:#15803d;margin-bottom:8px'>"
            "🛡️ <b>Strict Mode ON</b> — Only Hammer/Engulfing candles "
            "with Bullish PSAR shown · Fewer stocks · Higher win rate"
            "</div>",
            unsafe_allow_html=True)
    else:
        st.markdown(
            "<div style='background:#fffbeb;border:1.5px solid #fde68a;"
            "border-radius:8px;padding:8px 14px;font-size:11px;"
            "color:#d97706;margin-bottom:8px'>"
            "⚠️ <b>Strict Mode OFF</b> — All signals shown including "
            "weak candles and bearish PSAR · More stocks · Lower quality"
            "</div>",
            unsafe_allow_html=True)

    # Volatility filter
    _ms_vol_col1, _ms_vol_col2 = st.columns(2)
    with _ms_vol_col1:
        _ms_max_atr_pct = st.slider(
            "Volatility Score Penalty Starts At (Weekly ATR%)",
            min_value=1.0, max_value=10.0, value=6.0, step=0.5,
            key="ms_max_atr_pct",
            help="Weekly ATR% above this = score penalty applied. "
                 "NO hard reject — high vol stocks still show "
                 "but with lower score and 🔴 HIGH badge. "
                 "HINDCOPPER (6.9%) will now appear with penalty. "
                 "PA + SMA20 proximity already protect bad entries.")
    with _ms_vol_col2:
        st.markdown(
            f"<div style='background:#f8fafc;border:1px solid #e2e8f0;"
            f"border-radius:8px;padding:10px 14px;margin-top:4px'>"
            f"<div style='font-size:10px;font-weight:700;color:#64748b;"
            f"letter-spacing:1px'>SCORE IMPACT</div>"
            f"<div style='font-size:11px;color:#374151;margin-top:4px;line-height:1.8'>"
            f"🟢 &lt;3% → +8 pts (bonus)<br>"
            f"🟡 3-5% → 0 pts (neutral)<br>"
            f"🔴 5-8% → -10 pts (penalty)<br>"
            f"❌ &gt;8% → -15 pts (heavy)"
            f"</div></div>",
            unsafe_allow_html=True)

    # ── Scan function ──────────────────────────────────
    def scan_monthly_swing(stocks, capital, risk_pct, min_score):
        """
        Monthly swing scanner — weekly candles.
        8 additional checks: Nifty alignment, RS, OBV, Sector,
        52W high, Earnings warning, MACD, Inside week.
        """
        results  = []
        total    = len(stocks)
        _prog_ms = st.progress(0, text="📅 Initialising — fetching Nifty + sector data...")
        _stat_ms = st.empty()

        # ── Funnel counter — tracks WHERE stocks get filtered ──
        # Lets us tell apart "genuine bearish market, nothing
        # qualifies" from "something's actually broken" with
        # hard numbers instead of guessing.
        _ms_funnel = {
            'total_scanned':      0,
            'data_ok':            0,
            'sma_trend_ok':       0,   # passed sma20>sma50, slopes, price>sma20
            'candle_recovering':  0,   # passed last3_red / recovering check
            'not_extended':       0,   # passed pct_above20 proximity gate
            'fib_retrace_ok':     0,   # passed fibonacci retrace window
            'weekly_move_ok':     0,   # passed this_wk_move / RSI gates
            'rs_vs_nifty_ok':     0,   # passed _rs_ratio >= 0.95
            'has_signal_pattern': 0,   # cross/pullback/breakout/52w detected
            'score_pre_filter':   0,   # passed score < 40 pre-filter
            'fundamentals_ok':    0,   # passed fundamental reject check
            'structure_ok':       0,   # passed PA structure_reject check
            'final_score_ok':     0,   # passed score >= min_score
        }

        # ══════════════════════════════════════════════════
        # PHASE 1 — Pre-scan: Nifty + Sector data (once)
        # ══════════════════════════════════════════════════

        # ── Nifty weekly alignment ────────────────────────
        _nifty_bullish = True
        _nifty_sma20   = None
        _nifty_sma50   = None
        _nifty_closes  = {}   # {weeks_ago: close}
        _nifty_df_weekly = None  # for beta calculation
        _ms_nifty_swing  = {'state': 'UNKNOWN'}  # swing state
        try:
            _nt = yf.Ticker('^NSEI')
            _nf = _nt.history(period='2y', interval='1wk', auto_adjust=True, actions=False)
            _nf.columns = [c.split(' ')[0] if ' ' in str(c) else c for c in _nf.columns]
            _nf = _nf[['Close']].dropna()
            _nf['SMA20'] = _nf['Close'].rolling(20).mean()
            _nf['SMA50'] = _nf['Close'].rolling(50).mean()
            _nifty_sma20 = float(_nf['SMA20'].iloc[-1])
            _nifty_sma50 = float(_nf['SMA50'].iloc[-1])
            _nifty_bullish = _nifty_sma20 > _nifty_sma50
            _nifty_df_weekly = _nf.copy()  # keep for beta calc
            _nifty_closes  = {
                0: float(_nf['Close'].iloc[-1]),
                4: float(_nf['Close'].iloc[-4]) if len(_nf)>=4 else float(_nf['Close'].iloc[-1]),
            }
            # Build swing state from fetched data
            _nf_close = float(_nf['Close'].iloc[-1])
            _nf_sma20 = float(_nf['SMA20'].iloc[-1])
            _nf_sma50 = float(_nf['SMA50'].iloc[-1])
            _nf_slope = 0.0
            if len(_nf) >= 6:
                _nf_prev = float(_nf['SMA20'].iloc[-6])
                _nf_slope = (_nf_sma20 - _nf_prev) / _nf_prev * 100 if _nf_prev > 0 else 0
            if   _nf_close > _nf_sma20 > _nf_sma50 and _nf_slope > 0:
                _ms_nifty_swing['state'] = 'BULLISH'
            elif _nf_close > _nf_sma20:
                _ms_nifty_swing['state'] = 'CAUTION'
            elif _nf_close > _nf_sma50:
                _ms_nifty_swing['state'] = 'CAUTION'
            else:
                _ms_nifty_swing['state'] = 'BEARISH'
            _ms_nifty_swing['close'] = round(_nf_close, 2)
            _ms_nifty_swing['sma20'] = round(_nf_sma20, 2)
            _ms_nifty_swing['sma50'] = round(_nf_sma50, 2)
            # Cache for dashboard
            st.session_state['nifty_swing_weekly'] = _ms_nifty_swing
        except Exception:
            _nifty_bullish = True  # assume bullish if fetch fails

        # Nifty weekly bearish → show WARNING only, continue scan
        # Don't stop — some stocks outperform even in weak Nifty
        # But apply score penalty and show caution badge on results
        try:
            st.session_state['_ms_nifty_temp'] = _nifty_bullish
        except Exception:
            pass
        if not _nifty_bullish and _nifty_sma20 and _nifty_sma50:
            st.warning(
                f"⚠️ Nifty Weekly is BEARISH — SMA20 ₹{_nifty_sma20:,.0f} < SMA50 ₹{_nifty_sma50:,.0f}  "
                f"Only stocks with strong RS vs Nifty will appear. Use smaller position size.")

        # ── Sector ranking — use UNIFIED function ─────────
        # Same formula and data as SMA Weekly and Sector Leaders
        # Consistent ranking across all 3 tabs
        # Uses cached result (1hr) — no extra API calls
        _ms_rankings        = get_unified_sector_rankings(formula='monthly')
        _sector_status      = _ms_rankings['status_map']
        _sector_rs          = _ms_rankings['rs_map']
        _sector_rank_map    = _ms_rankings['rank_map']
        _unique_sectors     = _ms_rankings['rs_map']

        def _get_sector_rank(sym):
            """Thin wrapper — uses single authoritative classify_stock_sector().
            Returns (sector_name, rank, rs) for MS scanner."""
            _sec = classify_stock_sector(sym)
            if _sec == 'UNKNOWN':
                return 'INFRA', 10, 0.0
            # Map extended proxy sectors to ones in _sector_rank_map
            _fallback = {
                'PSU_BANK':'BANK','PVT_BANK':'BANK',
                'HEALTHCARE':'PHARMA','TEXTILES':'INFRA',
                'AGRI':'INFRA','LOGISTICS':'INFRA',
                'DEFENCE':'INFRA','CONSUMPTION':'CONSUMER',
            }
            _sec_key = _fallback.get(_sec, _sec)
            _rank = _sector_rank_map.get(_sec_key, 10)
            _rs   = _sector_rs.get(_sec_key, 0.0)
            return _sec_key, _rank, _rs

        def _get_sector_status(sym):
            """Thin wrapper — uses single authoritative classify_stock_sector().
            Returns (bullish, gap%) for MS scanner."""
            _sec = classify_stock_sector(sym)
            if _sec == 'UNKNOWN':
                return (True, 0.0)
            _fallback = {
                'PSU_BANK':'BANK','PVT_BANK':'BANK',
                'HEALTHCARE':'PHARMA','TEXTILES':'INFRA',
                'AGRI':'INFRA','LOGISTICS':'INFRA',
                'DEFENCE':'INFRA','CONSUMPTION':'CONSUMER',
            }
            _sec_key = _fallback.get(_sec, _sec)
            return _sector_status.get(_sec_key, (True, 0.0))
        # Results season months — Q4: Apr-May, Q1: Jul-Aug, Q2: Oct-Nov, Q3: Jan-Feb
        _now_month = ist_now().month
        _results_season = _now_month in [1,2,4,5,7,8,10,11]

        _prog_ms.progress(5, text=f"📅 Nifty ✅ {'BULL' if _nifty_bullish else 'BEAR'} · Scanning {total} stocks...")

        # ══════════════════════════════════════════════════
        # PHASE 2 — Per-stock scan
        # ══════════════════════════════════════════════════
        for idx, symbol in enumerate(stocks):
            pct       = 5 + int(((idx+1)/total)*95)
            sym_clean = symbol.replace('.NS','')
            _prog_ms.progress(pct, text=f"📅 {idx+1}/{total} · {sym_clean}")

            try:
                _ticker_sym = symbol if symbol.endswith('.NS') else symbol+'.NS'
                _t = yf.Ticker(_ticker_sym)
                _ms_funnel['total_scanned'] += 1

                # ── Fetch weekly candles ──────────────────
                df = _t.history(period='2y', interval='1wk',
                                auto_adjust=True, actions=False)
                if df is None or len(df) < 30:
                    continue
                df.columns = [c.split(' ')[0] if ' ' in str(c) else c for c in df.columns]
                df = df[['Open','High','Low','Close','Volume']].dropna()
                if len(df) < 28:
                    continue
                _ms_funnel['data_ok'] += 1

                # ── Indicators ───────────────────────────
                df['SMA20'] = df['Close'].rolling(20).mean()
                df['SMA50'] = df['Close'].rolling(50).mean()
                df['HL']    = df['High'] - df['Low']
                df['HPC']   = (df['High'] - df['Close'].shift(1)).abs()
                df['LPC']   = (df['Low']  - df['Close'].shift(1)).abs()
                df['TR']    = df[['HL','HPC','LPC']].max(axis=1)
                df['ATR7']  = df['TR'].rolling(7).mean()
                df['VOLMA'] = df['Volume'].rolling(10).mean()
                df['RSI14'] = 100-(100/(1+(
                    df['Close'].diff().clip(lower=0).rolling(14).mean()/
                    (-df['Close'].diff().clip(upper=0)).rolling(14).mean()
                )))

                # NEW: OBV
                df['OBV'] = (df['Close'].diff().apply(
                    lambda x: 1 if x>0 else (-1 if x<0 else 0)) * df['Volume']).cumsum()

                # NEW: MACD on weekly
                df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
                df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
                df['MACD']  = df['EMA12'] - df['EMA26']
                df['MSIG']  = df['MACD'].ewm(span=9, adjust=False).mean()

                l=df.iloc[-1]; p=df.iloc[-2]; p2=df.iloc[-3]; p3=df.iloc[-4]

                close   = float(l['Close'])
                sma20   = float(l['SMA20'])
                sma50   = float(l['SMA50'])
                sma20_p = float(p['SMA20'])
                sma50_p = float(p['SMA50'])
                atr7    = float(l['ATR7'])
                vol     = float(l['Volume'])
                vol_ma  = float(l['VOLMA'])
                rsi     = float(l['RSI14'])
                macd    = float(l['MACD'])
                msig    = float(l['MSIG'])
                macd_p  = float(p['MACD'])
                msig_p  = float(p['MSIG'])
                obv_now = float(l['OBV'])
                obv_4w  = float(df['OBV'].iloc[-4]) if len(df)>=4 else obv_now

                if any(pd.isna(x) for x in [sma20,sma50,atr7,vol_ma]) or atr7<=0:
                    continue

                vol_ratio   = vol/vol_ma if vol_ma>0 else 1.0
                pct_above20 = (close-sma20)/sma20*100 if sma20>0 else 0
                pct_above50 = (close-sma50)/sma50*100 if sma50>0 else 0

                # Slopes
                sma20_2w     = float(df['SMA20'].iloc[-2])
                sma50_4w     = float(df['SMA50'].iloc[-4]) if len(df)>=4 else sma50
                sma20_4w     = float(df['SMA20'].iloc[-4]) if len(df)>=4 else sma20
                sma20_slope  = (sma20-sma20_2w)/sma20_2w*100 if sma20_2w>0 else 0
                sma20_slope_4w=(sma20-sma20_4w)/sma20_4w*100 if sma20_4w>0 else 0
                sma50_slope  = (sma50-sma50_4w)/sma50_4w*100 if sma50_4w>0 else 0

                c0=float(l['Close']); c1=float(p['Close'])
                c2=float(p2['Close']); c3=float(p3['Close'])
                o0=float(l['Open'])
                h0=float(l['High']); h1=float(p['High'])
                lo0=float(l['Low']); lo1=float(p['Low'])

                this_wk_move = abs(c0-c1)/c1*100 if c1>0 else 0

                # ── Fibonacci Retracement Calculation ─────
                # swing_low  = weekly SMA50 (base of trend)
                # swing_high = 13-week high (peak of last rally)
                # up_move    = full rally distance
                # Retrace %  = how much price pulled back from peak
                _swing_low  = sma50
                _swing_high = float(df['High'].iloc[-13:].max()) if len(df)>=13 else float(df['High'].max())
                _up_move    = _swing_high - _swing_low if _swing_high > _swing_low else 1
                _retrace_pct= (_swing_high - close) / _up_move * 100 if _up_move > 0 else 0

                # Fibonacci levels
                _fib_236 = _swing_high - _up_move * 0.236
                _fib_382 = _swing_high - _up_move * 0.382
                _fib_500 = _swing_high - _up_move * 0.500
                _fib_618 = _swing_high - _up_move * 0.618
                _fib_786 = _swing_high - _up_move * 0.786

                # ── HARD GATES ────────────────────────────
                if sma20<=sma50:          continue
                if sma20_slope<=0:        continue
                if sma50_slope<-0.5:      continue
                if close<=sma20:          continue
                _ms_funnel['sma_trend_ok'] += 1

                # ── VOLATILITY GATE ───────────────────────
                # Weekly ATR% = ATR / Price × 100
                # High weekly ATR = very wide SL needed
                # Use 4-week average for stability
                _wk_atr_curr  = float(atr7)
                _wk_atr_4w    = float(df['ATR7'].tail(4).mean()) if len(df) >= 4 else _wk_atr_curr
                _wk_atr_use   = max(_wk_atr_curr, _wk_atr_4w)
                _wk_atr_pct   = round(_wk_atr_use / close * 100, 2) if close > 0 else 0

                # Volatility grade
                if   _wk_atr_pct < 3.0: _ms_vol_grade='LOW';      _ms_vol_clr='#15803d'; _ms_vol_bg='#f0fdf4'; _ms_vol_ico='🟢'
                elif _wk_atr_pct < 6.0: _ms_vol_grade='MEDIUM';   _ms_vol_clr='#d97706'; _ms_vol_bg='#fffbeb'; _ms_vol_ico='🟡'
                elif _wk_atr_pct < 8.0: _ms_vol_grade='HIGH';     _ms_vol_clr='#dc2626'; _ms_vol_bg='#fff5f5'; _ms_vol_ico='🔴'
                else:                   _ms_vol_grade='VERY HIGH'; _ms_vol_clr='#7f1d1d'; _ms_vol_bg='#fef2f2'; _ms_vol_ico='❌'

                # ── Soft scoring — NO hard reject ─────────
                # High volatility = score penalty only
                # Good stocks (HINDCOPPER) still show up
                # PA + SMA20 proximity already protect
                # against bad high-vol entries
                if   _wk_atr_pct < 3.0: _ms_vol_score =  8
                elif _wk_atr_pct < 4.0: _ms_vol_score =  4
                elif _wk_atr_pct < 5.0: _ms_vol_score =  0
                elif _wk_atr_pct < 6.0: _ms_vol_score = -5
                elif _wk_atr_pct < 8.0: _ms_vol_score = -10
                else:                   _ms_vol_score = -15

                # ── ENTRY FILTERS ─────────────────────────
                last3_red = c0<c1 and c1<c2 and c2<c3
                if last3_red:             continue
                recovering = c0>c1 or c0>o0
                if not recovering:        continue
                _ms_funnel['candle_recovering'] += 1

                # ── ENTRY BADGE — filter + classify ───────
                # High ATR stocks (>5%) need tighter SMA20 proximity
                # because their natural weekly move can eat SL easily
                _ms_atr_tight = _wk_atr_pct > 5.0
                _ms_max_prox  = 3.0 if _ms_atr_tight else 5.0

                # Hard filter — hide if too extended
                if pct_above20 > _ms_max_prox:
                    continue
                _ms_funnel['not_extended'] += 1

                # Assign entry badge
                if _ms_atr_tight:
                    # High ATR — tighter zones
                    if   pct_above20 <= 1.0: _ms_entry_badge = 'ENTER NOW';  _ms_entry_clr = '#15803d'; _ms_entry_bg = '#f0fdf4'; _ms_entry_ico = '🟢'
                    else:                    _ms_entry_badge = 'ACCEPTABLE'; _ms_entry_clr = '#d97706'; _ms_entry_bg = '#fffbeb'; _ms_entry_ico = '🟡'
                else:
                    # Normal ATR — standard zones
                    if   pct_above20 <= 2.0: _ms_entry_badge = 'ENTER NOW';  _ms_entry_clr = '#15803d'; _ms_entry_bg = '#f0fdf4'; _ms_entry_ico = '🟢'
                    else:                    _ms_entry_badge = 'ACCEPTABLE'; _ms_entry_clr = '#d97706'; _ms_entry_bg = '#fffbeb'; _ms_entry_ico = '🟡'

                # Fibonacci gate: reject if retrace > 78.6% (trend broken)
                # or retrace < 15% (not pulled back enough — near high)
                if _retrace_pct > 78.6:   continue  # too deep — trend broken
                if _retrace_pct < 15:     continue  # not pulled back — near high
                _ms_funnel['fib_retrace_ok'] += 1

                if this_wk_move>8:        continue
                if rsi>70:                continue
                _ms_funnel['weekly_move_ok'] += 1

                # NEW: Relative strength vs Nifty (hard reject if badly underperforming)
                _rs_ratio = 1.0
                if _nifty_closes.get(0) and _nifty_closes.get(4):
                    _stk_4w  = float(df['Close'].iloc[-4]) if len(df)>=4 else close
                    _rs_ratio= (close/_stk_4w) / (_nifty_closes[0]/_nifty_closes[4]) if _stk_4w>0 else 1.0
                if _rs_ratio < 0.95:      continue  # badly underperforming Nifty → skip
                _ms_funnel['rs_vs_nifty_ok'] += 1

                # ── SIGNALS ───────────────────────────────
                cross_w0  = sma20>sma50 and sma20_p<=float(p['SMA50'])
                cross_w1  = float(p['SMA20'])>float(p['SMA50']) and float(p2['SMA20'])<=float(p2['SMA50'])
                cross_w2  = float(p2['SMA20'])>float(p2['SMA50']) and float(p3['SMA20'])<=float(p3['SMA50'])
                has_cross = cross_w0 or cross_w1 or cross_w2
                cross_age = 0 if cross_w0 else (1 if cross_w1 else 2)

                trend_weeks=0
                for i in range(1,min(20,len(df))):
                    if float(df['SMA20'].iloc[-i])>float(df['SMA50'].iloc[-i]): trend_weeks+=1
                    else: break

                pb_found=False; pb_age=0
                if trend_weeks>=4:
                    for i in range(1,3):
                        rl=float(df['Low'].iloc[-i]); rs=float(df['SMA20'].iloc[-i])
                        if abs(rl-rs)/rs*100<=1.5 or rl<=rs:
                            pb_found=True; pb_age=i; break
                has_pb = pb_found and close>=sma20*1.002 and pct_above20<=5

                # ── Volume Dry-Up on Pullback ─────────────
                # Healthy pullback = falling on LOW volume
                # = sellers exhausted, not distributing
                # Dangerous pullback = falling on HIGH volume
                # = institutions dumping = falling knife
                _pb_vol_score = 0
                _pb_vol_label = ''
                _pb_vol_clr   = '#64748b'
                if has_pb and pb_age > 0:
                    try:
                        # Volume during pullback weeks
                        _pb_vol_avg = float(df['Volume'].iloc[-10:-3].mean())
                        _pb_vol_now = float(df['Volume'].iloc[-pb_age])
                        _pb_vol_ratio = _pb_vol_now / _pb_vol_avg if _pb_vol_avg > 0 else 1.0
                        _pb_vol_ratio = round(_pb_vol_ratio, 2)

                        if   _pb_vol_ratio < 0.5:
                            # Very low volume = perfect dry-up ✅✅
                            _pb_vol_score = +15
                            _pb_vol_label = '💧 Vol dry-up (perfect)'
                            _pb_vol_clr   = '#15803d'
                        elif _pb_vol_ratio < 0.7:
                            # Low volume = healthy pullback ✅
                            _pb_vol_score = +10
                            _pb_vol_label = '💧 Vol dry-up (healthy)'
                            _pb_vol_clr   = '#16a34a'
                        elif _pb_vol_ratio < 1.0:
                            # Slightly below average = ok
                            _pb_vol_score = +5
                            _pb_vol_label = '✅ Vol below avg'
                            _pb_vol_clr   = '#d97706'
                        elif _pb_vol_ratio < 1.5:
                            # Average volume = neutral
                            _pb_vol_score = 0
                            _pb_vol_label = '⚠️ Vol normal'
                            _pb_vol_clr   = '#d97706'
                        elif _pb_vol_ratio < 2.0:
                            # High volume pullback = warning
                            _pb_vol_score = -10
                            _pb_vol_label = '🔴 Vol high on pullback'
                            _pb_vol_clr   = '#dc2626'
                        else:
                            # Very high volume = dangerous
                            _pb_vol_score = -20
                            _pb_vol_label = '❌ Distribution detected'
                            _pb_vol_clr   = '#991b1b'
                    except Exception:
                        _pb_vol_ratio = 1.0
                        _pb_vol_score = 0
                        _pb_vol_label = ''
                else:
                    _pb_vol_ratio = 1.0

                high_13w  = float(df['High'].iloc[-14:-1].max()) if len(df)>=14 else 0
                is_brkout = (close>high_13w and vol_ratio>=1.5) if high_13w>0 else False

                # ── 52-Week High Breakout ─────────────────
                # Annual high breakout = very strong signal
                # Institutions notice and chase these moves
                high_52w  = float(df['High'].iloc[-53:-1].max()) if len(df)>=53 else high_13w
                is_52w_brkout = (
                    close > high_52w and          # price above 52W high
                    vol_ratio >= 1.5 and           # volume confirming
                    close > sma20 > sma50          # trend intact
                ) if high_52w > 0 else False

                if not has_cross and not has_pb and not is_brkout and not is_52w_brkout:
                    continue
                _ms_funnel['has_signal_pattern'] += 1

                if rsi>70:     continue
                if this_wk_move>8: continue

                # ── SCORING ───────────────────────────────
                score=0

                # Signal (max 30)
                cb=(25 if cross_age==0 else 18 if cross_age==1 else 12) if has_cross else 0
                pb=(22 if pb_age==1 else 16) if has_pb else 0
                ib=20 if (is_brkout and vol_ratio>=1.5) else (12 if is_brkout else 0)
                # 52W breakout gets highest score — strongest signal
                i52=(30 if (is_52w_brkout and vol_ratio>=2.0) else
                     25 if is_52w_brkout else 0)
                sc=len([x for x in [has_cross,has_pb,is_brkout,is_52w_brkout] if x])
                if sc>=2:   score+=min(30,max(cb,pb,ib,i52)+5)
                elif sc==1: score+=max(cb,pb,ib,i52)

                # Volume dry-up on pullback (max +15, min -20)
                # Only applies when pullback signal triggered
                score += _pb_vol_score

                # Price vs SMA20 (max 20)
                score+=(20 if pct_above20<=1 else 15 if pct_above20<=2 else
                        10 if pct_above20<=3 else 5)

                # Trend weeks (max 10)
                score+=(10 if trend_weeks>=12 else 7 if trend_weeks>=8 else
                        4 if trend_weeks>=4 else 0)

                # RSI (max 15)
                score+=(15 if 45<=rsi<=65 else 8 if (40<=rsi<45 or 65<rsi<=70) else 0)

                # Volume (max 15)
                score+=(15 if vol_ratio>=2.0 else 10 if vol_ratio>=1.5 else
                        5 if vol_ratio>=1.0 else 0)

                # SMA20 slope 4wk (max 10)
                score+=(10 if sma20_slope_4w>=1.0 else 7 if sma20_slope_4w>=0.5 else
                        4 if sma20_slope_4w>=0.2 else 1)

                # HH + HL monthly (max 10)
                m1h=float(df['High'].iloc[-4:].max()); m2h=float(df['High'].iloc[-8:-4].max())
                m3h=float(df['High'].iloc[-12:-8].max())
                m1l=float(df['Low'].iloc[-4:].min());  m2l=float(df['Low'].iloc[-8:-4].min())
                m3l=float(df['Low'].iloc[-12:-8].min())
                hh=m1h>m2h>m3h; hl=m1l>m2l>m3l
                if hh and hl: score+=10
                elif hh: score+=6
                elif hl: score+=4

                # Gap widening (max 10)
                gap_now=(sma20-sma50)/sma50*100 if sma50>0 else 0
                gap_prev=(sma20_p-sma50_p)/sma50_p*100 if sma50_p>0 else 0
                if gap_now>gap_prev: score+=10

                # Vol direction (max 5)
                up_v=sum(float(df['Volume'].iloc[-i]) for i in range(1,5)
                         if float(df['Close'].iloc[-i])>float(df['Open'].iloc[-i]))
                dn_v=sum(float(df['Volume'].iloc[-i]) for i in range(1,5)
                         if float(df['Close'].iloc[-i])<=float(df['Open'].iloc[-i]))
                vol_healthy=up_v>=dn_v
                if vol_healthy: score+=5

                # Recovery bonus (max 5)
                if c0>o0 and c0>c1: score+=5

                # Nifty alignment (max +10, min −10)
                if _nifty_bullish:  score += 10
                else:               score -= 10   # bearish Nifty = penalty not block

                # NEW: Relative strength vs Nifty (max 10)
                if   _rs_ratio>=1.05: score+=10   # outperforming +5%+
                elif _rs_ratio>=1.02: score+=7    # outperforming +2%+
                elif _rs_ratio>=1.0:  score+=4    # slightly outperforming
                else:                 score-=5    # underperforming (but >0.95)

                # NEW: OBV rising (max 10)
                obv_slope = (obv_now-obv_4w)/abs(obv_4w)*100 if obv_4w!=0 else 0
                if   obv_slope>5:   score+=10  # strong accumulation
                elif obv_slope>0:   score+=6   # mild accumulation
                elif obv_slope>-5:  score+=0   # neutral
                else:               score-=10  # distribution — heavy penalty

                # NEW: Sector momentum (max 8)
                _sec_bull, _sec_gap = _get_sector_status(sym_clean)
                _sec_name, _sec_rank, _sec_rs_gap = _get_sector_rank(sym_clean)

                # Sector bullish/bearish — context signal, small weight
                if   _sec_bull and _sec_gap>1: score+=5   # sector strong
                elif _sec_bull:                score+=2   # sector mild bull
                else:                          score-=2   # sector bearish

                # Sector RS Ranking — BONUS ONLY, minimal penalty
                # Individual stock RS vs sector is the real differentiator
                # Extraordinary stocks exist even in weak sectors
                if   _sec_rank <= 2: score += 10  # top sector ✅✅
                elif _sec_rank <= 4: score += 7   # strong sector ✅
                elif _sec_rank <= 6: score += 3   # above average
                elif _sec_rank <= 9: score += 0   # neutral
                else:                score -= 3   # bottom sector ⚠️

                # NEW: 52-week high proximity (max 10)
                high_52w     = float(df['High'].tail(52).max()) if len(df)>=52 else float(df['High'].max())
                pct_from_52w = (high_52w-close)/high_52w*100 if high_52w>0 else 50
                if   pct_from_52w<=5:  score+=10  # within 5% — near breakout
                elif pct_from_52w<=10: score+=7   # within 10%
                elif pct_from_52w<=20: score+=3   # within 20%
                elif pct_from_52w>30:  score-=15  # far from high — downtrend

                # NEW: Weekly MACD (max 12)
                macd_cross_up  = macd>msig and macd_p<=msig_p  # just crossed
                macd_above_sig = macd>msig
                macd_above_zero= macd>0
                if   macd_cross_up:               score+=12  # fresh cross = strongest
                elif macd_above_sig and macd_above_zero: score+=8
                elif macd_above_sig:              score+=4
                else:                             score-=6   # MACD below signal

                # NEW: Inside week / tight base (max 8)
                inside_week  = h0<h1 and lo0>lo1
                week_range   = (h0-lo0)/lo0*100 if lo0>0 else 10
                if   inside_week:      score+=8   # energy building
                elif week_range<=5:    score+=5   # tight base
                elif week_range>10:    score-=5   # volatile/unsettled

                # ── Fibonacci Retracement Score ────────────
                # Replaces old "extended penalty"
                # Rewards stocks in ideal pullback zone
                # Penalises stocks near highs or in deep correction
                if   _retrace_pct < 23.6:   score -= 10  # too shallow — near high
                elif _retrace_pct < 38.2:   score += 20  # 23-38% — ideal pullback ✅✅
                elif _retrace_pct < 50.0:   score += 15  # 38-50% — good pullback ✅
                elif _retrace_pct < 61.8:   score += 10  # 50-61% — acceptable ✅
                elif _retrace_pct < 78.6:   score -= 5   # 61-78% — deep, still ok ⚠️
                # >78.6% already rejected as hard gate above

                if score < 40:  # low pre-filter
                    continue
                _ms_funnel['score_pre_filter'] += 1

                # ══════════════════════════════════════════
                # FUNDAMENTAL FILTER
                # Only fetched for stocks passing all technical
                # filters — minimal speed impact (5-15 stocks)
                # ══════════════════════════════════════════
                _de          = None   # debt to equity
                _promoter    = None   # insider/promoter holding
                _eps         = None   # trailing EPS
                _earn_date   = None   # next results date
                _earn_warn   = False  # results in next 4 weeks
                _fund_reject = False
                _fund_reason = ''

                try:
                    _fi = _t.info  # reuse existing ticker object
                    _de       = _fi.get('debtToEquity', None)
                    _promoter = _fi.get('heldPercentInsiders', None)
                    _eps      = _fi.get('trailingEps', None)
                    _roe      = _fi.get('returnOnEquity', None)

                    # Earnings date warning
                    _earn_raw = _fi.get('earningsTimestamps', None) or \
                                _fi.get('earningsDate', None)
                    if _earn_raw:
                        import datetime as _dt
                        try:
                            if isinstance(_earn_raw, (list, tuple)):
                                _earn_ts = _earn_raw[0]
                            else:
                                _earn_ts = _earn_raw
                            if hasattr(_earn_ts, 'timestamp'):
                                _earn_ts = _earn_ts.timestamp()
                            _earn_dt   = _dt.datetime.fromtimestamp(float(_earn_ts))
                            _days_away = (_earn_dt - _dt.datetime.now()).days
                            if 0 <= _days_away <= 28:
                                _earn_warn = True
                                _earn_date = _earn_dt.strftime('%d %b')
                        except Exception:
                            pass

                    # ── HARD REJECT ──────────────────────
                    # D/E > 2.0 → too much debt
                    if _de is not None and _de > 200:  # yfinance gives as %
                        _fund_reject = True
                        _fund_reason = f'High debt D/E={_de/100:.1f}'

                    # EPS < 0 → loss making
                    elif _eps is not None and _eps < 0:
                        _fund_reject = True
                        _fund_reason = f'Loss making EPS={_eps:.2f}'

                    # Promoter < 20% → no conviction
                    elif _promoter is not None and _promoter < 0.20:
                        _fund_reject = True
                        _fund_reason = f'Low promoter {_promoter*100:.0f}%'

                except Exception:
                    pass  # if fundamental fetch fails, continue without it

                if _fund_reject:
                    continue
                _ms_funnel['fundamentals_ok'] += 1

                # ── Fundamental scoring ───────────────────
                # D/E score
                if _de is not None:
                    _de_val = _de / 100  # yfinance gives as percentage
                    if   _de_val < 0.3:  score += 8   # nearly debt free
                    elif _de_val < 0.5:  score += 5
                    elif _de_val < 1.0:  score += 3
                    elif _de_val < 1.5:  score += 0
                    elif _de_val < 2.0:  score -= 5
                    # > 2.0 already rejected above

                # Promoter holding score
                if _promoter is not None:
                    if   _promoter >= 0.60: score += 8  # founder strongly backing
                    elif _promoter >= 0.50: score += 5
                    elif _promoter >= 0.35: score += 2
                    elif _promoter >= 0.20: score += 0
                    # < 0.20 already rejected above

                # Results warning penalty
                if _earn_warn:
                    score -= 8  # reduce score, don't reject
                    # Still show on card as warning

                # ── Beta score (dynamic — based on Nifty state) ──
                # Bearish Nifty: high beta penalised, low beta rewarded
                # Bullish Nifty: high beta rewarded, low beta neutral
                score += _beta_score

                # ── Volatility Squeeze Detection ──────────────
                # Weekly candles — more reliable squeeze detection
                # Squeeze fired on weekly = strongest signal
                _ms_sq = detect_volatility_squeeze(df)

                # ── ADX — Trend Strength ───────────────────────
                # Weekly ADX — better for 3-5 week hold
                # ADX > 25 on weekly = very strong trend
                _ms_adx, _ms_pdi, _ms_mdi = calc_adx(df, period=14)
                _ms_adx_score, _ms_adx_lbl, _ms_adx_clr = \
                    get_adx_score(_ms_adx, _ms_pdi, _ms_mdi)

                # ── RS vs Own Sector (Step 1 fix — 20-Jun-2026) ─
                # Same logic as SMA Weekly — finds stocks outperforming
                # their own sector regardless of overall sector rank.
                # ISOLATED try/except — never relies on the scanner's
                # outer except, never silently empties the result list.
                # Defaults to neutral (0 pts) on any failure.
                _ms_rs_sec_diff  = 0.0
                _ms_rs_sec_score = 0
                _ms_rs_sec_label = ''
                _ms_rs_sec_clr   = '#64748b'
                try:
                    _ms_sec_for_rs = classify_stock_sector(sym_clean)
                    _ms_rs_sec_diff, _ms_rs_sec_score, _ms_rs_sec_label, _ms_rs_sec_clr = \
                        get_rs_vs_sector(df, _ms_sec_for_rs, _ms_rankings)
                except Exception as _ms_rs_exc:
                    st.session_state.setdefault('ms_scan_errors', []).append(
                        f"{symbol} RS-vs-sector (non-fatal, score=0): {str(_ms_rs_exc)[:80]}")
                score += _ms_rs_sec_score

                # ── Filter 1: Candle close position ───────
                # Reject if price closed in bottom 25% of week
                _ms_wp, _ms_wp_score, _ms_wp_label, _ms_wp_reject = \
                    get_candle_close_position(df, bars=1)
                if _ms_wp_reject:
                    continue
                score += _ms_wp_score

                # ── Filter 2: Prior candle body comparison ─
                _ms_cb_ratio, _ms_cb_score, _ms_cb_label = \
                    get_candle_body_momentum(df)
                score += _ms_cb_score

                # ── Volatility Squeeze score ───────────────
                score += _ms_sq.get('score', 0)

                # ── ADX score ──────────────────────────────
                score += _ms_adx_score

                score += _ms_vol_score

                if score < min_score:
                    continue
                _ms_funnel['final_score_ok'] += 1

                # ── Fetch live price for entry/SL/targets ─
                # Weekly candle = used for signals and scoring (correct)
                # Live daily price = used for entry, SL, targets (correct)
                # This ensures SL and targets reflect TODAY's actual price
                _live_price = None
                try:
                    _df_live = _t.history(
                        period='5d', interval='1d',
                        auto_adjust=True, actions=False)
                    if _df_live is not None and len(_df_live) > 0:
                        _df_live.columns = [c.split(' ')[0] if ' ' in str(c)
                                            else c for c in _df_live.columns]
                        _live_price = round(float(_df_live['Close'].iloc[-1]), 2)
                except Exception:
                    pass

                # Use live price if available, else use weekly close
                entry = _live_price if _live_price else close
                _price_source = 'live' if _live_price else 'weekly_close'

                # ── Trade plan ────────────────────────────
                sl    = max(round(entry-2.0*atr7,2), round(sma20*0.97,2))
                risk_d= entry-sl
                if risk_d<=0: continue

                # ── Filter 3: Dynamic risk sizing ─────────
                _ms_nifty_st = st.session_state.get('nifty_swing_weekly', {}).get('state','UNKNOWN')
                try:
                    _ms_pk = st.session_state.get('peak_capital', capital)
                    _ms_cr = st.session_state.get('current_capital', capital)
                    _ms_dd_pct = max(0.0, (_ms_pk-_ms_cr)/_ms_pk*100) if _ms_pk>0 else 0.0
                except Exception:
                    _ms_dd_pct = 0.0
                _ms_adj_risk, _ms_risk_lbl, _ms_risk_clr, _ms_risk_reason = \
                    get_dynamic_risk_pct(risk_pct, _ms_nifty_st, _ms_dd_pct)

                t1 = round(entry+1.0*atr7,2)
                t2 = round(entry+2.0*atr7,2)
                t3 = round(entry+3.0*atr7,2)
                qty= max(1,int((capital*_ms_adj_risk/100)/risk_d))
                inv= round(entry*qty,2)
                rr1= round((t1-entry)/risk_d,1)
                rr2= round((t2-entry)/risk_d,1)
                rr3= round((t3-entry)/risk_d,1)
                mchg=round((close-float(df['Close'].iloc[-5]))/float(df['Close'].iloc[-5])*100,2) if len(df)>=5 else 0

                # ── Beta calculation ──────────────────────
                # Dynamic score based on Nifty swing state
                _beta_val   = 1.0
                _beta_score = 0
                _beta_label = '➡️ Neutral'
                _beta_clr   = '#64748b'
                _beta_grade = 'NEUTRAL'
                _beta_bg    = '#f8fafc'
                _beta_bdr   = '#e2e8f0'
                _beta_ico   = '➡️'
                try:
                    _beta_val = calc_stock_beta(df, _nifty_df_weekly, periods=52)
                    _beta_score, _beta_label, _beta_clr = get_beta_score(
                        _beta_val, _ms_nifty_swing)
                    _beta_grade, _beta_clr, _beta_bg, _beta_bdr, _beta_ico = get_beta_grade(_beta_val)
                except Exception:
                    pass

                # ── PSAR (weekly, step=0.01, max=0.10) ───
                # Used as trailing SL after T1 hit
                # Shows on card — user updates Zerodha SL to this level
                _psar_val      = None
                _psar_bullish  = False
                try:
                    _df_ps = calc_psar(df.copy(), step=0.01, max_af=0.10)
                    if len(_df_ps) >= 1:
                        _pv = float(_df_ps['PSAR'].iloc[-1])
                        _pb = bool(_df_ps['PSAR_bull'].iloc[-1])
                        _psar_val     = round(_pv, 2)
                        _psar_bullish = _pb and close > _pv
                except Exception:
                    pass

                # ── Price Action Analysis (3 checks) ─────
                _pa = run_price_action_analysis(
                    df, entry, sma20, sma50,
                    _fib_382, _fib_500, _fib_618)

                # Hard reject if structure broken
                if _pa['structure_reject']:
                    continue
                _ms_funnel['structure_ok'] += 1

                # Add PA score to total
                score += _pa['pa_total_score']
                if score < 40:  # low pre-filter
                    continue

                # ── Strict Entry Mode Filter ──────────────
                # Runs AFTER PA so candle_pattern is available
                # Monthly Swing: stricter than SMA Weekly
                # Requires Hammer/Engulfing/Strong Bull
                # because 3-5 week hold needs strong conviction
                _ms_strict_on = st.session_state.get('ms_strict_mode', True)
                if _ms_strict_on:
                    # Gate 1 — PSAR must be bullish
                    if not _psar_bullish:
                        continue  # PSAR bearish → skip

                    # Gate 2 — Strong candle required
                    _ms_candle_pat  = _pa.get('candle_pattern', '')
                    _strong_candles = ('Hammer', 'Bullish Engulfing', 'Strong Bull')
                    if not any(c in _ms_candle_pat for c in _strong_candles):
                        continue  # weak candle → skip

                # Liquidity
                _dv=vol_ma*close
                if _dv>=500_000_000:   _lg='EXCELLENT'; _lc='#15803d'; _lb='#dcfce7'; _li='✅'
                elif _dv>=100_000_000: _lg='HIGH';      _lc='#1d4ed8'; _lb='#dbeafe'; _li='🔵'
                elif _dv>=20_000_000:  _lg='MEDIUM';    _lc='#d97706'; _lb='#fef3c7'; _li='🟡'
                else:                  _lg='LOW';        _lc='#dc2626'; _lb='#fee2e2'; _li='🔴'
                if _dv>=1_000_000_000: _lt=f"₹{_dv/1_000_000_000:.1f}K Cr/wk"
                elif _dv>=10_000_000:  _lt=f"₹{_dv/10_000_000:.0f} Cr/wk"
                else:                  _lt=f"₹{_dv/100_000:.0f} L/wk"

                _rank=round(score*rr2,1)
                cap_tier=get_cap_tier(sym_clean)

                # Signal label
                sigs=[]
                if has_cross:   sigs.append(f"🔀 Weekly cross {cross_age}wk ago")
                if has_pb:      sigs.append(f"📉 Pullback {pb_age}wk ago")
                if is_brkout:     sigs.append(f"🚀 13wk breakout")
                if is_52w_brkout: sigs.append(f"🏆 52W High Breakout")
                signal_label=' + '.join(sigs)

                results.append({
                    'symbol':       sym_clean,
                    'score':        score,
                    '_rank':        _rank,
                    'close':        round(close,2),      # weekly close (for signals display)
                    'entry':        round(entry,2),      # live price (for SL/targets/trade)
                    'price_source': _price_source,       # 'live' or 'weekly_close'
                    'weekly_close': round(close,2),      # always weekly close
                    'sma20':        round(sma20,2),
                    'sma50':        round(sma50,2),
                    'atr7':         round(atr7,2),
                    'rsi':          round(rsi,1),
                    'vol_ratio':    round(vol_ratio,1),
                    'trend_weeks':  trend_weeks,
                    'signal_label': signal_label,
                    'signal_count': len(sigs),
                    'has_cross':    has_cross,
                    'has_pb':       has_pb,
                    'is_breakout':  is_brkout,
                    'is_52w_brkout':  is_52w_brkout,
                    'pb_vol_ratio':   round(_pb_vol_ratio, 2),
                    'pb_vol_score':   _pb_vol_score,
                    'pb_vol_label':   _pb_vol_label,
                    'pb_vol_clr':     _pb_vol_clr,
                    'cross_age':    cross_age if has_cross else 99,
                    'pb_age':       pb_age,
                    'hh': hh, 'hl': hl,
                    'sma20_slope':  round(sma20_slope,2),
                    'sma20_slope_4w':round(sma20_slope_4w,2),
                    'pct_above20':  round(pct_above20,1),
                    'pct_above50':  round(pct_above50,1),
                    'month_chg':    mchg,
                    # Fibonacci retracement
                    'fib_retrace':  round(_retrace_pct,1),
                    'fib_382':      round(_fib_382,2),
                    'fib_500':      round(_fib_500,2),
                    'fib_618':      round(_fib_618,2),
                    'swing_high':   round(_swing_high,2),
                    'swing_low':    round(_swing_low,2),
                    # Fundamentals
                    'de_ratio':     round(_de/100,2) if _de is not None else None,
                    'promoter':     round(_promoter*100,1) if _promoter is not None else None,
                    'eps':          round(_eps,2) if _eps is not None else None,
                    'earn_warn':    _earn_warn,
                    'earn_date':    _earn_date,
                    # PSAR
                    'psar':         _psar_val,
                    'psar_bullish': _psar_bullish,
                    'pa':           _pa,
                    'entry_badge':  _ms_entry_badge,
                    'entry_clr':    _ms_entry_clr,
                    'entry_bg':     _ms_entry_bg,
                    'entry_ico':    _ms_entry_ico,
                    'vol_atr_pct':  _wk_atr_pct,
                    'vol_grade':    _ms_vol_grade,
                    'vol_clr':      _ms_vol_clr,
                    'vol_bg':       _ms_vol_bg,
                    'vol_ico':      _ms_vol_ico,
                    'sl':           sl,
                    't1': t1, 't2': t2, 't3': t3,
                    'qty': qty, 'inv': inv,
                    'risk_d':       round(risk_d,2),
                    'rr_t1': rr1, 'rr_t2': rr2, 'rr_t3': rr3,
                    'liq_grade': _lg, 'liq_clr': _lc, 'liq_bg': _lb,
                    'liq_ico':  _li, 'liq_turn': _lt,
                    'cap_tier':     cap_tier,
                    # New fields
                    'rs_ratio':     round(_rs_ratio,3),
                    'obv_slope':    round(obv_slope,1),
                    'macd_cross':   macd_cross_up,
                    'macd_above':   macd_above_sig,
                    'sec_bull':     _sec_bull,
                    'sec_gap':      _sec_gap,
                    'sec_name':     _sec_name,
                    'sec_rank':     _sec_rank,
                    'sec_rs_gap':   _sec_rs_gap,
                    'pct_from_52w': round(pct_from_52w,1),
                    'inside_week':  inside_week,
                    'week_range':   round(week_range,1),
                    'results_season': _results_season,
                    'nifty_bullish': _nifty_bullish,
                    'beta':          round(_beta_val, 2),
                    'beta_score':    _beta_score,
                    'beta_label':    _beta_label,
                    'beta_grade':    _beta_grade,
                    'beta_clr':      _beta_clr,
                    'beta_bg':       _beta_bg,
                    'beta_bdr':      _beta_bdr,
                    'beta_ico':      _beta_ico,
                    'nifty_swing_state': _ms_nifty_swing.get('state', 'UNKNOWN'),
                    # Filter 1 — candle close position
                    'week_pos':      _ms_wp,
                    'wp_score':      _ms_wp_score,
                    'wp_label':      _ms_wp_label,
                    # Filter 2 — candle body momentum
                    'cb_ratio':      _ms_cb_ratio,
                    'cb_score':      _ms_cb_score,
                    'cb_label':      _ms_cb_label,
                    # Volatility squeeze
                    'squeeze':       _ms_sq,
                    'squeeze_score': _ms_sq.get('score', 0),
                    'squeeze_fired': _ms_sq.get('squeeze_fired', False),
                    'squeeze_on':    _ms_sq.get('squeeze_on', False),
                    'squeeze_label': _ms_sq.get('label', ''),
                    'squeeze_weeks': _ms_sq.get('squeeze_weeks', 0),
                    # ADX
                    'adx':           _ms_adx,
                    'adx_pdi':       _ms_pdi,
                    'adx_mdi':       _ms_mdi,
                    'adx_score':     _ms_adx_score,
                    'adx_label':     _ms_adx_lbl,
                    'adx_clr':       _ms_adx_clr,
                    # RS vs own sector (Step 1 fix — 20-Jun-2026)
                    'rs_sec_diff':   _ms_rs_sec_diff,
                    'rs_sec_score':  _ms_rs_sec_score,
                    'rs_sec_label':  _ms_rs_sec_label,
                    'rs_sec_clr':    _ms_rs_sec_clr,
                    # Filter 3 — dynamic risk sizing
                    'adj_risk_pct':  _ms_adj_risk,
                    'risk_label':    _ms_risk_lbl,
                    'risk_clr':      _ms_risk_clr,
                    'risk_reason':   _ms_risk_reason,
                    **get_fno_info(sym_clean),
                })
                # Calculate confident score
                _ms_cs = calc_confident_score(results[-1])
                results[-1].update(_ms_cs)

                if len(results)%3==0:
                    _stat_ms.markdown(
                        f"<div style='font-size:12px;color:#7c3aed;padding:4px 0'>"
                        f"📅 {len(results)} signals found...</div>",
                        unsafe_allow_html=True)

            except Exception as _e:
                continue  # silent — don't break scan for one bad stock

        _prog_ms.empty(); _stat_ms.empty()
        # Sort by confident score (CONFIDENT BUY first)
        for r in results:
            if 'confident_score' not in r:
                _ms_cs2 = calc_confident_score(r)
                r.update(_ms_cs2)
        results.sort(key=lambda x: x.get('confident_score', 0), reverse=True)

        # Save funnel data so we can show WHERE stocks dropped off,
        # regardless of whether the final result list is empty or not.
        st.session_state['ms_funnel'] = dict(_ms_funnel)

        return results
    # ── Scan button ────────────────────────────────────
    _ms_run = st.button(
        "📅 Scan for Monthly Swing Trades",
        key="ms_run_scan", use_container_width=True, type="primary",
        help="Scans weekly charts — best run on weekends after market close")

    if _ms_run:
        if not _ms_stocks:
            st.warning("⚠️ No stocks to scan — upload a CSV file or pick a different universe above.")
        else:
            # Clear yfinance disk cache
            try:
                import shutil, pathlib
                _yfc = pathlib.Path.home()/'.cache'/'py-yfinance'
                if _yfc.exists(): shutil.rmtree(_yfc, ignore_errors=True)
            except Exception:
                pass
            _ms_debug = st.empty()
            _ms_debug.info("📅 Starting scan — fetching Nifty + sector data...")
            try:
                _ms_results = scan_monthly_swing(
                    _ms_stocks, _ms_capital, _ms_risk, _ms_min_score)
                _ms_debug.empty()
            except Exception as _ms_err:
                _ms_debug.error(f"❌ Scan error: {str(_ms_err)}")
                _ms_results = []
            st.session_state['ms_results']   = _ms_results
            st.session_state['ms_scan_time'] = ist_now().strftime('%d %b %Y %H:%M IST')
            # Store Nifty status for topbar display
            st.session_state['ms_nifty_bullish'] = st.session_state.get('_ms_nifty_temp', None)
            if _ms_results:
                st.rerun()
            else:
                st.warning(f"⚠️ No signals found. Try lowering min score to 50 or select a broader universe.")

    # ── Results ────────────────────────────────────────
    _ms_results  = st.session_state.get('ms_results', [])
    _ms_scantime = st.session_state.get('ms_scan_time', '')

    # ── Debug: Show scan errors (even if scan succeeded) ──
    # Non-fatal errors (e.g. RS-vs-sector isolated failures)
    # default to neutral and don't block results, but you
    # should still be able to see them happened.
    _ms_errors = st.session_state.get('ms_scan_errors', [])
    if _ms_errors:
        _ms_err_title = (f'🔍 Debug: {len(_ms_errors)} non-fatal error(s) during scan '
                          f'(results still shown — these defaulted to neutral)'
                          if len(_ms_results) > 0 else
                          f'🔍 Debug: {len(_ms_errors)} stocks had errors during scan')
        with st.expander(_ms_err_title):
            for _e in _ms_errors[:15]:
                st.code(_e)
        st.session_state['ms_scan_errors'] = []

    # ── Funnel: Where are stocks getting filtered? ────────
    # Shows exact dropout counts at every gate — tells apart
    # "genuine bearish market, nothing qualifies" from
    # "something's actually broken" with hard numbers.
    _ms_funnel = st.session_state.get('ms_funnel', None)
    if _ms_funnel:
        with st.expander(
            f"📊 Scan Funnel — where did {_ms_funnel.get('total_scanned',0)} stocks go?",
            expanded=(len(_ms_results) == 0)):
            _funnel_steps = [
                ('Total scanned',                  'total_scanned'),
                ('Had valid price data',           'data_ok'),
                ('Passed SMA20>SMA50 + slope + price>SMA20', 'sma_trend_ok'),
                ('Passed candle recovering check', 'candle_recovering'),
                ('Passed proximity (not extended)','not_extended'),
                ('Passed Fibonacci retrace window','fib_retrace_ok'),
                ('Passed weekly move/RSI gates',   'weekly_move_ok'),
                ('Passed RS vs Nifty ≥ 0.95',      'rs_vs_nifty_ok'),
                ('Had a valid signal pattern',     'has_signal_pattern'),
                ('Passed raw score pre-filter (40)','score_pre_filter'),
                ('Passed fundamentals check',      'fundamentals_ok'),
                ('Passed price-action structure',  'structure_ok'),
                (f'Passed final min score ({_ms_min_score})', 'final_score_ok'),
            ]
            _prev = _ms_funnel.get('total_scanned', 0)
            for _label, _key in _funnel_steps:
                _count = _ms_funnel.get(_key, 0)
                _drop  = _prev - _count if _key != 'total_scanned' else 0
                _bar   = '█' * max(1, int(_count / max(_ms_funnel.get('total_scanned',1),1) * 30)) if _count > 0 else ''
                st.markdown(
                    f"<div style='font-family:monospace;font-size:11px;padding:2px 0'>"
                    f"{_label:<42} {_count:>5}"
                    f"{f'  <span style=\"color:#dc2626\">(-{_drop})</span>' if _drop > 0 else ''}"
                    f"</div>", unsafe_allow_html=True)
                _prev = _count
            st.caption(
                "If 'final min score' is 0 but earlier steps show plenty of "
                "survivors, the bottleneck is the score threshold — lower it. "
                "If it drops to 0 very early (e.g. at SMA trend gate), that's "
                "genuine broad-market weakness — nothing wrong, just no "
                "stocks currently satisfy a real uptrend definition.")

    # ── Nifty State Banner for Monthly Swing ──────────────
    _ms_nifty_disp = st.session_state.get('nifty_swing_weekly', {}).get('state', 'UNKNOWN')

    if _ms_nifty_disp == 'BEARISH':
        st.markdown(
            "<div style='background:#1f0c0c;border:2px solid #dc2626;"
            "border-radius:12px;padding:16px 20px;margin-bottom:16px'>"
            "<div style='font-size:15px;font-weight:800;color:#fca5a5'>"
            "⛔ NIFTY BEARISH — New Monthly Swing Entries NOT Recommended</div>"
            "<div style='font-size:11px;color:#fecaca;margin-top:8px;line-height:2'>"
            "📉 <b>Why:</b> 3-5 week hold in BEARISH market = high risk of further downside &nbsp;·&nbsp; "
            "Even strong stocks fall with a falling Nifty over weeks<br>"
            "✅ <b>What to do:</b> Use SMA Weekly tab instead (3-7 day holds) &nbsp;·&nbsp; "
            "Wait for Nifty to return to CAUTION or BULLISH state<br>"
            "⚠️ <b>If you scan:</b> Results shown for reference only — "
            "enter only if you have very strong conviction + tight SL"
            "</div></div>",
            unsafe_allow_html=True)

    elif _ms_nifty_disp == 'CAUTION':
        st.markdown(
            "<div style='background:#1c150a;border:1.5px solid #d97706;"
            "border-radius:10px;padding:10px 16px;margin-bottom:12px'>"
            "<div style='font-size:12px;font-weight:800;color:#fcd34d'>"
            "⚠️ NIFTY CAUTION — Monthly Swing in Selective Mode</div>"
            "<div style='font-size:11px;color:#fde68a;margin-top:4px'>"
            "Prefer SMA Weekly for shorter holds · "
            "Monthly Swing entries: only ✅ STRONG SETUP (≥100) or higher · "
            "Reduced position sizes active"
            "</div></div>",
            unsafe_allow_html=True)

    # ── Expiry Zone Banner ─────────────────────────────
    _ms_dte  = days_to_expiry()
    _ms_zone = get_expiry_zone(_ms_dte)
    _ms_exp  = get_monthly_expiry()
    _ms_exp_str = _ms_exp.strftime('%d %b %Y')
    if   _ms_zone == 'FRESH':
        _mb_bg='#f0fdf4'; _mb_bdr='#86efac'; _mb_clr='#15803d'
        _mb_ico='🟢'; _mb_title='Post-Expiry — BEST Entry Window!'
        _mb_msg=(f'New F&O cycle · Fresh positions building · '
                 f'Enter any stock freely · Next expiry {_ms_exp_str}')
    elif _ms_zone == 'DANGER':
        _mb_bg='#fef2f2'; _mb_bdr='#fca5a5'; _mb_clr='#dc2626'
        _mb_ico='🔴'; _mb_title=f'Expiry Week — {_ms_dte} days to {_ms_exp_str}'
        _mb_msg=('F&O stocks price-pinned this week · '
                 'Non-F&O stocks shown first · '
                 'F&O score -15 pts · Wait for post-expiry if possible')
    elif _ms_zone == 'CAUTION':
        _mb_bg='#fffbeb'; _mb_bdr='#fde68a'; _mb_clr='#d97706'
        _mb_ico='⚠️'; _mb_title=f'Second Half — {_ms_dte} days to {_ms_exp_str}'
        _mb_msg=('Approaching expiry · F&O stocks may slow · '
                 'Non-F&O stocks preferred · F&O score -8 pts')
    else:
        _mb_bg='#f0fdf4'; _mb_bdr='#bbf7d0'; _mb_clr='#15803d'
        _mb_ico='✅'; _mb_title=f'Safe Zone — {_ms_dte} days to {_ms_exp_str}'
        _mb_msg='First half of month · Enter freely · No expiry pinning risk'

    st.markdown(
        f"<div style='background:{_mb_bg};border:1.5px solid {_mb_bdr};"
        f"border-radius:10px;padding:10px 16px;margin-bottom:12px;"
        f"display:flex;align-items:center;gap:12px'>"
        f"<div style='font-size:22px'>{_mb_ico}</div>"
        f"<div>"
        f"<div style='font-size:12px;font-weight:800;color:{_mb_clr}'>"
        f"F&O EXPIRY — {_mb_title}</div>"
        f"<div style='font-size:11px;color:{_mb_clr};opacity:0.85;margin-top:2px'>"
        f"{_mb_msg}</div>"
        f"</div></div>",
        unsafe_allow_html=True)

    if not _ms_results:
        st.markdown("""
        <div style='background:#1a2035;border-radius:16px;padding:36px;
                    text-align:center;margin:20px 0'>
            <div style='font-size:44px;margin-bottom:12px'>📅</div>
            <div style='font-size:18px;font-weight:700;color:white;margin-bottom:8px'>
                Monthly Swing Scanner Ready
            </div>
            <div style='font-size:13px;color:rgba(255,255,255,0.6);line-height:1.8'>
                Best run on Saturday or Sunday evening<br>
                Scans weekly charts · Finds 3–5 week swing opportunities<br>
                No intraday monitoring needed
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        # Filter buttons
        _n_all   = len(_ms_results)
        _n_cross = len([r for r in _ms_results if r.get('has_cross')])
        _n_pb    = len([r for r in _ms_results if r.get('has_pb')])
        _n_brkout= len([r for r in _ms_results if r.get('is_breakout')])
        _n_52w   = len([r for r in _ms_results if r.get('is_52w_brkout')])

        _ms_filter = st.radio(
            "Filter",
            [f"📊 All ({_n_all})",
             f"🔀 Weekly Cross ({_n_cross})",
             f"📉 Weekly Pullback ({_n_pb})",
             f"🚀 13-Week Breakout ({_n_brkout})",
             f"🏆 52W High Breakout ({_n_52w})",
             f"🔥 Squeeze ({sum(1 for r in _ms_results if r.get('squeeze_fired') or r.get('squeeze_on'))})"],
            horizontal=True, key="ms_filter")

        if 'Weekly Cross' in _ms_filter:
            _ms_show = [r for r in _ms_results if r.get('has_cross')]
        elif 'Weekly Pullback' in _ms_filter:
            _ms_show = [r for r in _ms_results if r.get('has_pb')]
        elif '13-Week Breakout' in _ms_filter:
            _ms_show = [r for r in _ms_results if r.get('is_breakout')]
        elif '52W High Breakout' in _ms_filter:
            _ms_show = [r for r in _ms_results if r.get('is_52w_brkout')]
        elif 'Squeeze' in _ms_filter:
            _ms_show = [r for r in _ms_results
                        if r.get('squeeze_fired') or r.get('squeeze_on')]
            _ms_show = sorted(_ms_show,
                               key=lambda x: (x.get('squeeze_fired',False),
                                              x.get('squeeze_weeks',0)),
                               reverse=True)
        else:
            _ms_show = _ms_results

        # ── Sort by confident score (CONFIDENT BUY first) ─
        _ms_show = sorted(
            _ms_show,
            key=lambda x: x.get('confident_score', 0),
            reverse=True)


        # ── CSV Export — Monthly Swing ─────────────────
        def _ms_to_csv(results):
            import csv, io
            _buf = io.StringIO()
            _cols = [
                'Symbol','Score','Rank','Signal','Entry','Weekly_Close',
                'Price_Source','SMA20','SMA50','ATR','RSI','Vol_Ratio',
                'Trend_Weeks','SMA20_Slope','Fib_Retrace','Fib_Zone',
                'HH','HL','RS_vs_Nifty','OBV_Slope','MACD',
                'Sector_Bull','Pct_From_52W','Inside_Week',
                'Stop_Loss','T1','T2','T3',
                'Qty','Investment','Risk_Amt','RR_T1','RR_T2','RR_T3',
                'Liquidity','DE_Ratio','Promoter_Pct','EPS',
                'Results_Warning','Results_Date',
                'Cap_Tier','Scan_Date',
            ]
            _w = csv.DictWriter(_buf, fieldnames=_cols, extrasaction='ignore')
            _w.writeheader()
            for r in results:
                _fz = ("23-38% Ideal" if r.get('fib_retrace',50)<38.2 and r.get('fib_retrace',50)>=23.6
                       else "38-50% Good" if r.get('fib_retrace',50)<50
                       else "50-61% OK"   if r.get('fib_retrace',50)<61.8
                       else "61-78% Deep" if r.get('fib_retrace',50)<78.6
                       else ">78% Deep")
                _w.writerow({
                    'Symbol':          r.get('symbol',''),
                    'Score':           r.get('score',0),
                    'Rank':            round(r.get('_rank',0),1),
                    'Signal':          r.get('signal_label',''),
                    'Entry':           r.get('entry',0),
                    'Weekly_Close':    r.get('weekly_close',0),
                    'Price_Source':    r.get('price_source',''),
                    'SMA20':           r.get('sma20',0),
                    'SMA50':           r.get('sma50',0),
                    'ATR':             r.get('atr7',0),
                    'RSI':             r.get('rsi',0),
                    'Vol_Ratio':       r.get('vol_ratio',0),
                    'Trend_Weeks':     r.get('trend_weeks',0),
                    'SMA20_Slope':     r.get('sma20_slope',0),
                    'Fib_Retrace':     r.get('fib_retrace',0),
                    'Fib_Zone':        _fz,
                    'HH':              r.get('hh',False),
                    'HL':              r.get('hl',False),
                    'RS_vs_Nifty':     round((r.get('rs_ratio',1)-1)*100,2),
                    'OBV_Slope':       r.get('obv_slope',0),
                    'MACD':            ('FreshCross' if r.get('macd_cross') else
                                        'Above' if r.get('macd_above') else 'Below'),
                    'Sector_Bull':     r.get('sec_bull',True),
                    'Pct_From_52W':    r.get('pct_from_52w',0),
                    'Inside_Week':     r.get('inside_week',False),
                    'Stop_Loss':       r.get('sl',0),
                    'T1':              r.get('t1',0),
                    'T2':              r.get('t2',0),
                    'T3':              r.get('t3',0),
                    'Qty':             r.get('qty',0),
                    'Investment':      r.get('inv',0),
                    'Risk_Amt':        round(r.get('qty',0)*r.get('risk_d',0),0),
                    'RR_T1':           r.get('rr_t1',0),
                    'RR_T2':           r.get('rr_t2',0),
                    'RR_T3':           r.get('rr_t3',0),
                    'Liquidity':       r.get('liq_grade',''),
                    'DE_Ratio':        r.get('de_ratio',''),
                    'Promoter_Pct':    r.get('promoter',''),
                    'EPS':             r.get('eps',''),
                    'Results_Warning': r.get('earn_warn',False),
                    'Results_Date':    r.get('earn_date',''),
                    'Cap_Tier':        r.get('cap_tier',''),
                    'Scan_Date':       _ms_scantime,
                })
            return _buf.getvalue().encode('utf-8')

        _ms_hdr1, _ms_hdr2 = st.columns([4, 1])
        with _ms_hdr1:
            st.markdown(
                f"<div style='font-size:12px;color:#64748b;margin-bottom:12px'>"
                f"📅 {len(_ms_show)} signals · Scanned {_ms_scantime} · "
                f"Sorted by Score × R:R</div>",
                unsafe_allow_html=True)
        with _ms_hdr2:
            _ms_csv_fname = f"monthly_swing_{ist_now().strftime('%d%b%Y')}.csv"
            st.download_button(
                label="📥 Download CSV",
                data=_ms_to_csv(_ms_show),
                file_name=_ms_csv_fname,
                mime="text/csv",
                use_container_width=True,
                help=f"Download all {len(_ms_show)} signals as CSV"
            )

        # ── Batch AI Portfolio Recommendation ─────────
        _ms_batch_key = "ms_batch_ai_btn"
        _ms_batch_res = "ms_batch_ai_result"
        _ms_batch_tag = "ms_batch_ai_tags"  # per-symbol verdict tags

        if st.button(
            f"🤖 AI Analyse All {len(_ms_show)} Stocks — Get Portfolio Recommendation",
            key=_ms_batch_key, use_container_width=True,
            help="Single AI call analyses all stocks together and recommends which 2-3 to enter"
        ):
            with st.spinner("🤖 Analysing all stocks together..."):
                try:
                    import requests as _br, json as _bj
                    _ant_k = load_anthropic_key()
                    if not _ant_k:
                        st.error("❌ Anthropic API key not set. Go to sidebar → 🤖 AI Validation → Set Anthropic API Key")
                        raise Exception("API key not configured")
                    # Build compact summary for each stock
                    _stock_summaries = []
                    for _bi, _br_s in enumerate(_ms_show[:10], 1):
                        _bs = _br_s
                        _fz = ("23-38% Ideal" if _bs.get('fib_retrace',50)<38.2 and _bs.get('fib_retrace',50)>=23.6
                               else "38-50% Good" if _bs.get('fib_retrace',50)<50
                               else "50-61% OK"   if _bs.get('fib_retrace',50)<61.8
                               else "61-78% Deep" if _bs.get('fib_retrace',50)<78.6
                               else ">78% VeryDeep")
                        _stock_summaries.append(
                            f"STOCK {_bi}: {_bs['symbol']}\n"
                            f"  Score={_bs['score']}/145 | Entry=Rs{_bs['entry']:.2f} | "
                            f"SMA20=Rs{_bs['sma20']:.2f}({_bs.get('pct_above20',0):+.1f}%)\n"
                            f"  RSI={_bs.get('rsi',0):.1f} | Vol={_bs.get('vol_ratio',0):.1f}x | "
                            f"Trend={_bs.get('trend_weeks',0)}wk | Signal={_bs.get('signal_label','')}\n"
                            f"  MACD={'FreshCross' if _bs.get('macd_cross') else 'AboveSignal' if _bs.get('macd_above') else 'Below'} | "
                            f"OBV={'Accum' if _bs.get('obv_slope',0)>0 else 'Distrib'}({_bs.get('obv_slope',0):+.1f}%)\n"
                            f"  Fib={_fz}({_bs.get('fib_retrace',0):.1f}%) | "
                            f"RS={'+'if _bs.get('rs_ratio',1)>=1 else ''}{(_bs.get('rs_ratio',1)-1)*100:.1f}% | "
                            f"Sector={'Bull' if _bs.get('sec_bull',True) else 'Bear'}\n"
                            f"  D/E={_bs.get('de_ratio','N/A')} | "
                            f"Promoter={_bs.get('promoter','N/A')}% | "
                            f"EPS={'Rs'+str(_bs.get('eps','N/A')) if _bs.get('eps') else 'N/A'}\n"
                            f"  Results={'YES WARNING' if _bs.get('earn_warn') else 'No'} | "
                            f"T1=Rs{_bs.get('t1',0):.2f}(RR {_bs.get('rr_t1',0)}:1) | "
                            f"T2=Rs{_bs.get('t2',0):.2f}(RR {_bs.get('rr_t2',0)}:1)"
                        )
                    _nifty_ctx = "Bullish" if st.session_state.get("ms_nifty_bullish", True) else "Bearish"
                    _batch_prompt = (
                        f"You are an expert NSE swing trader. Analyse these {len(_ms_show[:10])} Monthly Swing candidates "
                        f"(3-5 week hold) and recommend which 2-3 to enter given Rs5L capital.\n\n"
                        f"Market: Nifty Weekly = {_nifty_ctx}\n"
                        f"Capital: Rs5,00,000 | Max positions: 2-3\n\n"
                        + "\n".join(_stock_summaries) +
                        f"\n\nRank ALL stocks into 3 categories: enter_now, watchlist, avoid.\n"
                        f"Consider: Fibonacci zone quality, signal freshness, volume confirmation,\n"
                        f"fundamental strength, results risk, sector momentum, R:R ratio.\n"
                        f"Reply ONLY with valid JSON (no markdown):\n"
                        f'{{"market_note":"1-2 sentences on current market condition",'
                        f'"portfolio_note":"1-2 sentences on capital allocation strategy",'
                        f'"enter_now":[{{"symbol":"X","rank":1,"confidence":"HIGH/MEDIUM/LOW",'
                        f'"reason":"2 sentences","risk":"main risk","entry_note":"entry guidance"}}],'
                        f'"watchlist":[{{"symbol":"X","reason":"why wait","entry_condition":"when to enter"}}],'
                        f'"avoid":[{{"symbol":"X","reason":"why avoid"}}]}}'
                    )
                    _br_resp = _br.post(
                        "https://api.anthropic.com/v1/messages",
                        headers={"Content-Type":"application/json",
                                 "x-api-key": load_anthropic_key(),
                                 "anthropic-version":"2023-06-01"},
                        json={"model":"claude-sonnet-4-20250514","max_tokens":1500,
                              "system":"You are an expert NSE swing trading analyst. Always respond with valid JSON only. No markdown, no explanation outside JSON.",
                              "messages":[{"role":"user","content":_batch_prompt}]},
                        timeout=45)
                    if _br_resp.status_code == 200:
                        _br_raw   = _br_resp.json()["content"][0]["text"].strip()
                        _br_clean = _br_raw.replace("```json","").replace("```","").strip()
                        _br_data  = _bj.loads(_br_clean)
                        st.session_state[_ms_batch_res] = _br_data
                        # Build tag dict for colouring stock cards
                        _tags = {}
                        for _be in _br_data.get("enter_now",[]): _tags[_be["symbol"]] = "enter"
                        for _bw in _br_data.get("watchlist",[]): _tags[_bw["symbol"]] = "watch"
                        for _ba in _br_data.get("avoid",[]):     _tags[_ba["symbol"]] = "avoid"
                        st.session_state[_ms_batch_tag] = _tags
                    else:
                        try:
                            _err_body = _br_resp.json()
                            _err_msg  = _err_body.get("error",{}).get("message","Unknown")
                        except Exception:
                            _err_msg = _br_resp.text[:200]
                        st.error(f"AI error {_br_resp.status_code}: {_err_msg}")
                except Exception as _bex:
                    st.error(f"AI error: {str(_bex)[:100]}")

        # Show batch AI result
        if _ms_batch_res in st.session_state:
            _bd    = st.session_state[_ms_batch_res]
            _btags = st.session_state.get(_ms_batch_tag, {})
            _ben   = _bd.get("enter_now", [])
            _bwl   = _bd.get("watchlist", [])
            _bav   = _bd.get("avoid", [])

            # Summary row
            st.markdown(
                f"<div style='background:white;border:2px solid #667eea44;border-radius:16px;"
                f"padding:16px 20px;margin-bottom:16px;box-shadow:0 4px 20px rgba(102,126,234,0.12)'>"
                f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:12px;flex-wrap:wrap'>"
                f"<span style='font-size:18px'>🤖</span>"
                f"<span style='font-size:15px;font-weight:800;color:#1a2035'>AI Portfolio Recommendation</span>"
                f"<span style='background:#eff6ff;color:#1d4ed8;font-size:10px;font-weight:700;"
                f"border-radius:4px;padding:2px 8px'>Monthly Swing · {_ms_scantime}</span>"
                f"</div>"
                f"<div style='display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap'>"
                f"<div style='background:#dcfce7;border:1px solid #86efac33;border-radius:10px;"
                f"padding:10px 16px;text-align:center;flex:1;min-width:80px'>"
                f"<div style='font-size:22px;font-weight:900;color:#15803d'>{len(_ben)}</div>"
                f"<div style='font-size:10px;font-weight:700;color:#15803d'>✅ ENTER NOW</div></div>"
                f"<div style='background:#fef3c7;border:1px solid #fde68a33;border-radius:10px;"
                f"padding:10px 16px;text-align:center;flex:1;min-width:80px'>"
                f"<div style='font-size:22px;font-weight:900;color:#d97706'>{len(_bwl)}</div>"
                f"<div style='font-size:10px;font-weight:700;color:#d97706'>⏳ WATCHLIST</div></div>"
                f"<div style='background:#fee2e2;border:1px solid #fca5a533;border-radius:10px;"
                f"padding:10px 16px;text-align:center;flex:1;min-width:80px'>"
                f"<div style='font-size:22px;font-weight:900;color:#dc2626'>{len(_bav)}</div>"
                f"<div style='font-size:10px;font-weight:700;color:#dc2626'>❌ AVOID</div></div>"
                f"<div style='background:#f5f3ff;border:1px solid #ddd6fe33;border-radius:10px;"
                f"padding:10px 16px;flex:2;min-width:160px'>"
                f"<div style='font-size:10px;font-weight:700;color:#7c3aed;margin-bottom:4px'>💡 STRATEGY</div>"
                f"<div style='font-size:11px;color:#4c1d95;line-height:1.5'>{_bd.get('portfolio_note','')}</div>"
                f"</div></div>",
                unsafe_allow_html=True)

            # Market note
            if _bd.get("market_note"):
                st.markdown(
                    f"<div style='background:#fffbeb;border:1px solid #fde68a;border-radius:8px;"
                    f"padding:10px 14px;margin-bottom:10px;font-size:12px;color:#374151'>"
                    f"<b style='color:#d97706'>📊 Market:</b> {_bd['market_note']}"
                    f"</div>", unsafe_allow_html=True)

            # Enter Now
            if _ben:
                st.markdown("<div style='font-size:11px;font-weight:700;color:#15803d;"
                            "letter-spacing:1px;margin-bottom:6px'>✅ ENTER NOW</div>",
                            unsafe_allow_html=True)
                for _be in _ben:
                    _cr = {"HIGH":"🔥","MEDIUM":"📊","LOW":"⚠️"}.get(_be.get("confidence",""),"")
                    st.markdown(
                        f"<div style='background:white;border:1.5px solid #86efac;"
                        f"border-radius:10px;padding:12px 14px;margin-bottom:8px'>"
                        f"<div style='display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:6px'>"
                        f"<span style='background:#15803d;color:white;font-size:11px;font-weight:900;"
                        f"border-radius:50%;width:22px;height:22px;display:flex;align-items:center;"
                        f"justify-content:center'>#{_be.get('rank',1)}</span>"
                        f"<span style='font-size:15px;font-weight:800;color:#1a2035'>{_be['symbol']}</span>"
                        f"<span style='background:#dcfce7;color:#15803d;font-size:10px;font-weight:700;"
                        f"border-radius:4px;padding:2px 8px'>✅ ENTER</span>"
                        f"<span style='background:#15803d;color:white;font-size:10px;font-weight:700;"
                        f"border-radius:4px;padding:2px 8px'>{_cr} {_be.get('confidence','')} CONFIDENCE</span>"
                        f"</div>"
                        f"<div style='font-size:11px;color:#374151;line-height:1.7;margin-bottom:6px'>"
                        f"{_be.get('reason','')}</div>"
                        f"<div style='display:flex;gap:8px;flex-wrap:wrap;font-size:11px'>"
                        f"<span style='background:#fff7ed;border:1px solid #fde68a;border-radius:6px;"
                        f"padding:4px 8px'>⚠️ <b>Risk:</b> {_be.get('risk','')}</span>"
                        f"<span style='background:#eff6ff;border:1px solid #bfdbfe;border-radius:6px;"
                        f"padding:4px 8px'>🎯 <b>Entry:</b> {_be.get('entry_note','')}</span>"
                        f"</div></div>", unsafe_allow_html=True)

            # Watchlist
            if _bwl:
                st.markdown("<div style='font-size:11px;font-weight:700;color:#d97706;"
                            "letter-spacing:1px;margin:8px 0 6px'>⏳ WATCHLIST — Wait for better entry</div>",
                            unsafe_allow_html=True)
                for _bw in _bwl:
                    st.markdown(
                        f"<div style='background:white;border:1.5px solid #fde68a;"
                        f"border-radius:10px;padding:10px 14px;margin-bottom:6px;"
                        f"display:flex;gap:10px;align-items:flex-start'>"
                        f"<span style='background:#fef3c7;color:#d97706;font-size:10px;font-weight:700;"
                        f"border-radius:6px;padding:3px 8px;flex-shrink:0'>⏳ WAIT</span>"
                        f"<div>"
                        f"<span style='font-weight:800;color:#1a2035;font-size:13px'>{_bw['symbol']}</span>"
                        f"<span style='font-size:11px;color:#374151;margin-left:8px'>{_bw.get('reason','')}</span>"
                        f"<div style='font-size:11px;background:#f0fdf4;border:1px solid #86efac;"
                        f"border-radius:6px;padding:4px 8px;margin-top:4px;color:#15803d'>"
                        f"📌 <b>When:</b> {_bw.get('entry_condition','')}</div>"
                        f"</div></div>", unsafe_allow_html=True)

            # Avoid
            if _bav:
                st.markdown("<div style='font-size:11px;font-weight:700;color:#dc2626;"
                            "letter-spacing:1px;margin:8px 0 6px'>❌ AVOID</div>",
                            unsafe_allow_html=True)
                for _ba in _bav:
                    st.markdown(
                        f"<div style='background:#fef2f2;border:1px solid #fca5a5;"
                        f"border-radius:8px;padding:8px 12px;margin-bottom:5px;"
                        f"display:flex;gap:8px;align-items:flex-start'>"
                        f"<span style='font-size:14px'>❌</span>"
                        f"<div><b style='color:#dc2626;font-size:12px'>{_ba['symbol']}</b>"
                        f"<span style='font-size:11px;color:#374151;margin-left:8px'>"
                        f"{_ba.get('reason','')}</span></div>"
                        f"</div>", unsafe_allow_html=True)

            # Close button
            st.markdown("</div>", unsafe_allow_html=True)
            if st.button("✕ Hide AI Recommendation", key="ms_hide_batch",
                         use_container_width=False):
                del st.session_state[_ms_batch_res]
                st.rerun()

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        for _ms_r in _ms_show[:10]:
            _sc   = _ms_r['score']
            _sym  = _ms_r['symbol']
            _rk   = _ms_r['_rank']
            _close= _ms_r['close']        # weekly close (for signals)
            _entry= _ms_r['entry']        # live price (for trade)
            _wk_cls = _ms_r.get('weekly_close', _close)
            _psrc   = _ms_r.get('price_source', 'weekly_close')
            _is_live= _psrc == 'live'
            _s20  = _ms_r['sma20']
            _s50  = _ms_r['sma50']
            _atr  = _ms_r['atr7']
            _rsi  = _ms_r['rsi']
            _volx = _ms_r['vol_ratio']
            _tw   = _ms_r['trend_weeks']
            _slab = _ms_r['signal_label']
            _sl   = _ms_r['sl']
            _t1   = _ms_r['t1']
            _t2   = _ms_r['t2']
            _t3   = _ms_r['t3']
            _qty  = _ms_r['qty']
            _inv  = _ms_r['inv']
            _rd   = _ms_r['risk_d']
            _rr1  = _ms_r['rr_t1']
            _rr2  = _ms_r['rr_t2']
            _rr3  = _ms_r['rr_t3']
            _mchg = _ms_r['month_chg']
            _cap  = _ms_r['cap_tier']
            _sl_str = _ms_r['sma20_slope']
            _p20  = _ms_r['pct_above20']
            _p50  = _ms_r['pct_above50']
            _hh   = _ms_r['hh']
            _hl   = _ms_r['hl']
            _lg   = _ms_r['liq_grade']
            _lc   = _ms_r['liq_clr']
            _lb   = _ms_r['liq_bg']
            _li   = _ms_r['liq_ico']
            _lt   = _ms_r['liq_turn']
            # Fibonacci
            _fib_ret  = _ms_r.get('fib_retrace', 0)
            _fib_382  = _ms_r.get('fib_382', 0)
            _fib_618  = _ms_r.get('fib_618', 0)
            _swh      = _ms_r.get('swing_high', 0)
            _swl      = _ms_r.get('swing_low', 0)
            _fib_zone = ('🟢 23-38% Ideal' if _fib_ret<38.2 and _fib_ret>=23.6
                         else '🟢 38-50% Good' if _fib_ret<50
                         else '🟡 50-61% OK'   if _fib_ret<61.8
                         else '⚠️ 61-78% Deep' if _fib_ret<78.6
                         else '🔴 >78% Weak')
            _fib_clr  = ('#15803d' if _fib_ret<50
                         else '#d97706' if _fib_ret<61.8
                         else '#dc2626')
            # Fundamentals
            _de_val   = _ms_r.get('de_ratio', None)
            _prom_val = _ms_r.get('promoter', None)
            _eps_val  = _ms_r.get('eps', None)
            _ew       = _ms_r.get('earn_warn', False)
            _ed       = _ms_r.get('earn_date', '')
            _de_clr   = ('#15803d' if _de_val is not None and _de_val<0.5
                         else '#d97706' if _de_val is not None and _de_val<1.0
                         else '#dc2626' if _de_val is not None
                         else '#64748b')
            _prom_clr = ('#15803d' if _prom_val is not None and _prom_val>=50
                         else '#1d4ed8' if _prom_val is not None and _prom_val>=35
                         else '#64748b')
            _de_str   = f"D/E {_de_val:.2f}" if _de_val is not None else "D/E N/A"
            _prom_str = f"Promoter {_prom_val:.0f}%" if _prom_val is not None else "Promoter N/A"
            _eps_str  = f"EPS ₹{_eps_val:.1f}" if _eps_val is not None else ""
            # New fields
            _rs   = _ms_r.get('rs_ratio', 1.0)
            _obv  = _ms_r.get('obv_slope', 0)
            _macd_x = _ms_r.get('macd_cross', False)
            _macd_a = _ms_r.get('macd_above', False)
            _sec_b  = _ms_r.get('sec_bull', True)
            _52w    = _ms_r.get('pct_from_52w', 20)
            _inw    = _ms_r.get('inside_week', False)
            _wkrng  = _ms_r.get('week_range', 5)
            _rseas  = _ms_r.get('results_season', False)

            # Colours
            _sc_clr = '#15803d' if _sc>=80 else ('#1d4ed8' if _sc>=70 else '#d97706')
            _sc_bg  = '#dcfce7' if _sc>=80 else ('#dbeafe' if _sc>=70 else '#fef3c7')
            _sc_bdr = _sc_clr+'33'
            # Override border if AI batch verdict available
            _ai_tag = st.session_state.get(_ms_batch_tag, {}).get(_sym, '')
            if _ai_tag == 'enter': _sc_bdr = '#86efac'; _ai_badge = "✅ AI: ENTER"
            elif _ai_tag == 'watch': _sc_bdr = '#fde68a'; _ai_badge = "⏳ AI: WATCH"
            elif _ai_tag == 'avoid': _sc_bdr = '#fca5a5'; _ai_badge = "❌ AI: AVOID"
            else: _ai_badge = ""
            _cap_ico,_cap_name,_cap_clr,_cap_bg = CAP_TIER_BADGE.get(
                _cap, ('🟠','Smallcap','#c2410c','#fff7ed'))
            _cap_bdr = _cap_clr+'44'
            _liq_bdr = _lc+'44'
            _mchg_clr= '#15803d' if _mchg>=0 else '#dc2626'
            _sl_clr2 = '#15803d' if _sl_str>=0.5 else ('#d97706' if _sl_str>0 else '#dc2626')
            _sl_lbl  = 'Strong ↑' if _sl_str>=1.0 else ('Rising ↑' if _sl_str>=0.5 else 'Weak ↑')

            # % changes
            _sl_pct = round((_entry-_sl)/_entry*100,2) if _entry>0 else 0
            _t1_pct = round((_t1-_entry)/_entry*100,2) if _entry>0 else 0
            _t2_pct = round((_t2-_entry)/_entry*100,2) if _entry>0 else 0
            _t3_pct = round((_t3-_entry)/_entry*100,2) if _entry>0 else 0

            # ── Card open + header ────────────────────
            st.markdown(
                f"<div style='background:#ffffff;border:1.5px solid {_sc_bdr};"
                f"border-radius:16px;padding:18px 20px;margin-bottom:14px;'>",
                unsafe_allow_html=True)

            # ── Card header ─────────────────────────────────────
            _live_clr    = "#15803d" if _is_live else "#d97706"
            _live_lbl    = "\U0001f7e2 Live" if _is_live else "\U0001f7e1 Wk Close"
            _wk_cls_str  = f"Wk close \u20b9{_wk_cls:,.2f} \u00b7 " if _is_live else ""
            if _ai_badge:
                _ai_badge_html = (
                    "<span style='background:#f0fdf4;color:#15803d;font-size:10px;"
                    "font-weight:700;border-radius:4px;padding:2px 8px;"
                    f"border:1px solid #86efac'>{_ai_badge}</span>"
                )
            else:
                _ai_badge_html = ""
            _ms_r_vol_ico   = _ms_r.get('vol_ico','⚪')
            _ms_r_vol_grade = _ms_r.get('vol_grade','')
            _ms_r_vol_pct   = _ms_r.get('vol_atr_pct', 0)
            _ms_r_vol_clr   = _ms_r.get('vol_clr','#64748b')
            _ms_r_vol_bg    = _ms_r.get('vol_bg','#f8fafc')
            _ms_r_eb_ico    = _ms_r.get('entry_ico','🟡')
            _ms_r_eb_badge  = _ms_r.get('entry_badge','ACCEPTABLE')
            _ms_r_eb_clr    = _ms_r.get('entry_clr','#d97706')
            _ms_r_eb_bg     = _ms_r.get('entry_bg','#fffbeb')
            _ms_r_fno_badge = _ms_r.get('fno_badge','✅ Non-F&O')
            _ms_r_fno_clr   = _ms_r.get('fno_clr','#64748b')
            _ms_r_fno_bg    = _ms_r.get('fno_bg','#f8fafc')
            _ms_r_fno_bdr   = _ms_r.get('fno_bdr','#e2e8f0')
            _ms_r_fno_note  = _ms_r.get('fno_note','')
            # Beta badge
            _ms_beta        = _ms_r.get('beta', 1.0)
            _ms_beta_grade  = _ms_r.get('beta_grade', 'NEUTRAL')
            _ms_beta_clr    = _ms_r.get('beta_clr', '#64748b')
            _ms_beta_bg     = _ms_r.get('beta_bg', '#f8fafc')
            _ms_beta_bdr    = _ms_r.get('beta_bdr', '#e2e8f0')
            _ms_beta_ico    = _ms_r.get('beta_ico', '➡️')
            _ms_beta_score  = _ms_r.get('beta_score', 0)
            _ms_beta_ss     = f'+{_ms_beta_score}' if _ms_beta_score > 0 else str(_ms_beta_score)

            # ── Confident Score ───────────────────────
            _ms_conf     = _ms_r.get('confident_score', 0)
            _ms_conf_lbl = _ms_r.get('confident_label', '⚠️ WEAK')
            _ms_conf_clr = _ms_r.get('confident_clr',  '#d97706')
            _ms_conf_bg  = _ms_r.get('confident_bg',   '#fffbeb')
            _ms_conf_bdr = _ms_r.get('confident_bdr',  '#fcd34d')
            _ms_c1       = _ms_r.get('c1_tech',   0)
            _ms_c2       = _ms_r.get('c2_psar',   0)
            _ms_c3       = _ms_r.get('c3_struct', 0)
            _ms_c4       = _ms_r.get('c4_badge',  0)
            _ms_c5       = _ms_r.get('c5_rr',     0)
            _ms_c6       = _ms_r.get('c6_liq',    0)
            _ms_c7       = _ms_r.get('c7_fno',    0)

            _hdr = (
                f"<div style='background:#ffffff;border:2px solid {_ms_conf_bdr};"
                "border-radius:16px;padding:18px 20px;margin-bottom:12px'>"
                "<div style='display:flex;justify-content:space-between;"
                "align-items:flex-start;flex-wrap:wrap;gap:8px;margin-bottom:12px'>"
                "<div>"
                "<div style='display:flex;align-items:center;gap:8px;flex-wrap:wrap'>"
                f"<span style='font-size:22px;font-weight:800;color:#1a2035'>{_sym}</span>"
                f"<span style='background:{_ms_conf_bg};color:{_ms_conf_clr};font-size:13px;"
                f"font-weight:800;border-radius:8px;padding:4px 12px;"
                f"border:2px solid {_ms_conf_bdr}'>"
                f"⭐ {_ms_conf}/100 · {_ms_conf_lbl}</span>"
                f"<span style='background:{_sc_bg};color:{_sc_clr};font-size:10px;"
                f"font-weight:700;border-radius:6px;padding:2px 8px'>"
                f"Scanner {_sc}/100</span>"
                f"{_ai_badge_html}"
                f"<span style='background:{_cap_bg};color:{_cap_clr};font-size:10px;"
                f"font-weight:700;border-radius:4px;padding:2px 8px;"
                f"border:1px solid {_cap_bdr}'>{_cap_ico} {_cap_name}</span>"
                f"<span style='background:{_lb};color:{_lc};font-size:10px;"
                f"font-weight:700;border-radius:4px;padding:2px 8px;"
                f"border:1px solid {_liq_bdr}'>{_li} {_lg} \u00b7 {_lt}</span>"
                f"<span style='background:{_ms_r_vol_bg};color:{_ms_r_vol_clr};font-size:10px;"
                f"font-weight:700;border-radius:4px;padding:2px 8px;"
                f"border:1px solid {_ms_r_vol_clr}44'>"
                f"{_ms_r_vol_ico} Vol {_ms_r_vol_pct:.1f}% {_ms_r_vol_grade}</span>"
                f"<span style='background:{_ms_r_eb_bg};color:{_ms_r_eb_clr};font-size:10px;"
                f"font-weight:700;border-radius:4px;padding:2px 8px;"
                f"border:1px solid {_ms_r_eb_clr}44'>"
                f"{_ms_r_eb_ico} {_ms_r_eb_badge}</span>"
                f"<span style='background:{_ms_r_fno_bg};color:{_ms_r_fno_clr};font-size:10px;"
                f"font-weight:700;border-radius:4px;padding:2px 8px;"
                f"border:1px solid {_ms_r_fno_bdr}'>"
                f"{_ms_r_fno_badge}</span>"
                f"<span style='background:{_ms_beta_bg};color:{_ms_beta_clr};"
                f"font-size:10px;font-weight:700;border-radius:4px;padding:2px 8px;"
                f"border:1px solid {_ms_beta_bdr}'>"
                f"{_ms_beta_ico} Beta {_ms_beta:.2f} {_ms_beta_grade} ({_ms_beta_ss}pts)</span>"
                "</div>"
                + (f"<div style='font-size:10px;color:{_ms_r_fno_clr};"
                   f"margin-top:3px;padding:3px 8px;"
                   f"background:{_ms_r_fno_bg};border-radius:4px'>"
                   f"{_ms_r_fno_note}</div>" if _ms_r_fno_note else "")
                + f"<div style='font-size:12px;color:#64748b;margin-top:6px'>"
                f"<span style='color:#7c3aed;font-weight:700'>{_slab}</span>"
                f" \u00b7 RSI {_rsi} \u00b7 Vol {_volx}\u00d7 \u00b7 Trend {_tw}wk \u00b7 "
                f"<span style='color:{_mchg_clr}'>Month {_mchg:+.1f}%</span>"
                "</div></div>"
                "<div style='text-align:right'>"
                f"<div style='font-size:26px;font-weight:800;color:#1a2035;"
                f"font-family:JetBrains Mono'>\u20b9{_entry:,.2f}"
                f"<span style='font-size:11px;font-weight:600;"
                f"color:{_live_clr};margin-left:6px'>{_live_lbl}</span></div>"
                f"<div style='font-size:11px;color:#64748b'>"
                f"{_wk_cls_str}SMA20 \u20b9{_s20:,.2f} \u00b7 SMA50 \u20b9{_s50:,.2f}"
                "</div></div></div>"
            )
            st.markdown(_hdr, unsafe_allow_html=True)

            # ── Trend quality row (all 8 checks) ──────
            _sl_clr2 = '#15803d' if _sl_str>=0.5 else ('#d97706' if _sl_str>0 else '#dc2626')
            _sl_lbl  = 'Strong' if _sl_str>=1.0 else ('Rising' if _sl_str>=0.5 else 'Weak')
            _rs_clr  = '#15803d' if _rs>=1.02 else ('#d97706' if _rs>=1.0 else '#dc2626')
            _obv_clr = '#15803d' if _obv>0 else '#dc2626'
            _macd_clr= '#15803d' if _macd_x else ('#1d4ed8' if _macd_a else '#dc2626')
            _macd_lbl= 'Fresh cross' if _macd_x else ('Above signal' if _macd_a else 'Below signal')
            _sec_clr = '#15803d' if _sec_b else '#dc2626'
            _52w_clr = '#15803d' if _52w<=10 else ('#d97706' if _52w<=20 else '#dc2626')
            # Pre-built conditional strings (avoid nested f-string issues)
            _p20_clr    = '#15803d' if _p20<=3 else '#d97706'
            _hh_ico     = '✅' if _hh else '❌'
            _hl_ico     = '✅' if _hl else '❌'
            _sec_ico    = '✅' if _sec_b else '❌'
            _inw_str    = '✅ Inside Week' if _inw else f'Wk range: {_wkrng:.1f}%'
            _rs_sign    = '+' if _rs>=1 else ''
            _obv_str    = '↑Accum' if _obv>0 else '↓Distrib'
            _rseas_html = '<span><b style="color:#d97706">⚠️Results</b></span>' if _rseas else ''
            _eps_html   = f'<span><b>{_eps_str}</b></span>' if _eps_str else ''
            _ew_html    = (f'<span style="color:#dc2626"><b>⚠️ Results {_ed} — verify</b></span>') if _ew else ''
            # Sector rank display
            _ms_sec_rank = _ms_r.get('sec_rank', 5)
            _ms_sec_name = _ms_r.get('sec_name', '')
            _ms_sec_rs   = _ms_r.get('sec_rs_gap', 0.0)
            _ms_sec_rs_s = f'+{_ms_sec_rs:.1f}%' if _ms_sec_rs >= 0 else f'{_ms_sec_rs:.1f}%'
            if   _ms_sec_rank <= 2: _ms_sec_rank_clr='#15803d'; _ms_sec_rank_ico='🥇'
            elif _ms_sec_rank <= 4: _ms_sec_rank_clr='#16a34a'; _ms_sec_rank_ico='🥈'
            elif _ms_sec_rank <= 6: _ms_sec_rank_clr='#d97706'; _ms_sec_rank_ico='🥉'
            else:                   _ms_sec_rank_clr='#dc2626'; _ms_sec_rank_ico='⬇️'
            st.markdown(
                f"<div style='background:#f8fafc;border-radius:8px;padding:10px 14px;"
                f"margin-bottom:10px;font-size:11px'>"
                f"<div style='display:flex;gap:12px;flex-wrap:wrap;margin-bottom:5px'>"
                f"<span>SMA20 slope:<b style='color:{_sl_clr2}'>{_sl_lbl} {_sl_str:+.2f}%</b></span>"
                f"<span>vs SMA20:<b style='color:{_p20_clr}'>+{_p20:.1f}%</b></span>"
                f"<span>vs SMA50:<b>+{_p50:.1f}%</b></span>"
                f"<span>Trend {_tw}wk</span>"
                f"<span>ATR ₹{_atr:,.2f}</span>"
                f"</div>"
                f"<div style='display:flex;gap:12px;flex-wrap:wrap;"
                f"padding-top:5px;border-top:1px solid #e2e8f0'>"
                f"<span>📐 Fib: <b style='color:{_fib_clr}'>{_fib_zone} ({_fib_ret:.1f}%)</b></span>"
                f"<span>Fib38=₹{_fib_382:,.0f} · Fib62=₹{_fib_618:,.0f}</span>"
                f"<span>RS:<b style='color:{_rs_clr}'>{_rs_sign}{(_rs-1)*100:.1f}%</b></span>"
                f"<span>OBV:<b style='color:{_obv_clr}'>{_obv_str}</b></span>"
                f"<span>MACD:<b style='color:{_macd_clr}'>{_macd_lbl}</b></span>"
                f"<span>Sector:<b style='color:{_sec_clr}'>{_sec_ico}</b></span>"
                f"<span style='color:{_ms_sec_rank_clr};font-weight:700'>"
                f"{_ms_sec_rank_ico} Rank #{_ms_sec_rank} {_ms_sec_name} ({_ms_sec_rs_s} vs Nifty)</span>"
                f"<span>52W:<b style='color:{_52w_clr}'>{_52w:.1f}%↓</b></span>"
                f"<span><b>{'✅ Inside Week' if _inw else f'Wk range: {_wkrng:.1f}%'}</b></span>"
                f"{_rseas_html}"
                + (f"<span style='color:{_ms_r.get('pb_vol_clr','#64748b')};font-weight:700'>"
                   f"{_ms_r.get('pb_vol_label','')} ({_ms_r.get('pb_vol_ratio',1.0):.2f}×)</span>"
                   if _ms_r.get('pb_vol_label') else "")
                + (f"<span style='color:{'#15803d' if _ms_r.get('wp_score',0)>0 else '#dc2626'};font-weight:700'>"
                   f"📍 {_ms_r.get('wp_label','')} ({_ms_r.get('week_pos',0.5):.0%})</span>")
                + (f"<span style='color:{'#15803d' if _ms_r.get('cb_score',0)>0 else '#d97706' if _ms_r.get('cb_score',0)==0 else '#dc2626'};font-weight:700'>"
                   f"📊 {_ms_r.get('cb_label','')}</span>")
                + (f"<span style='font-weight:800;"
                   f"color:{_ms_r.get('squeeze',{}).get('clr','#64748b')};"
                   f"background:{'#f0fdf4' if _ms_r.get('squeeze_fired') else '#fffbeb' if _ms_r.get('squeeze_on') else 'transparent'};"
                   f"padding:1px 6px;border-radius:4px'>"
                   f"{_ms_r.get('squeeze',{}).get('ico','➡️')} "
                   f"{_ms_r.get('squeeze_label','')}"
                   + (f" · BB {_ms_r.get('squeeze',{}).get('bb_width_change',0):+.0f}%"
                      if _ms_r.get('squeeze_fired') else "")
                   + f" ({'+' if _ms_r.get('squeeze_score',0)>=0 else ''}"
                   f"{_ms_r.get('squeeze_score',0)}pts)</span>"
                   if _ms_r.get('squeeze_label') else "")
                + (f"<span style='font-weight:700;"
                   f"color:{_ms_r.get('adx_clr','#64748b')}'>"
                   f"{_ms_r.get('adx_label','')}</span>"
                   if _ms_r.get('adx_label') else "")
                + (f"<span style='font-weight:700;"
                   f"color:{_ms_r.get('rs_sec_clr','#64748b')}'>"
                   f"📊 {_ms_r.get('rs_sec_label','')} "
                   f"({'+' if _ms_r.get('rs_sec_score',0)>=0 else ''}{_ms_r.get('rs_sec_score',0)}pts)"
                   f"</span>"
                   if _ms_r.get('rs_sec_label') else "")
                + "</div>"
                "<div style='display:flex;gap:12px;flex-wrap:wrap;"
                "padding-top:5px;border-top:1px solid #e2e8f0'>"
                f"<span>💰 <b style='color:{_de_clr}'>{_de_str}</b></span>"
                f"<span>👤 <b style='color:{_prom_clr}'>{_prom_str}</b></span>"
                f"{_eps_html}"
                f"{_ew_html}"
                + (f"<span style='color:{_ms_r.get('risk_clr','#64748b')};font-weight:700'>"
                   f"⚖️ {_ms_r.get('risk_label','')}"
                   f"{' — ' + _ms_r.get('risk_reason','') if _ms_r.get('risk_reason','') not in ('','No adjustment') else ''}"
                   f"</span>"
                   if _ms_r.get('risk_reason','') not in ('','No adjustment') else "")
                + "</div></div>",
                unsafe_allow_html=True)

            # ── Targets ───────────────────────────────
            _ms_atr_v    = _ms_r.get('atr7', _ms_r.get('atr', 0))
            _ms_rd_v     = _ms_r.get('risk_per_share', _entry - _sl if _entry > _sl else 1)
            _ms_adj_r    = _ms_r.get('adj_risk_pct', _ms_risk)
            _ms_risk_lbl2= _ms_r.get('risk_label', '')
            _ms_risk_rsn2= _ms_r.get('risk_reason', '')
            _ms_orig_qty = max(1, int((_ms_capital * _ms_risk / 100) / _ms_rd_v)) \
                           if _ms_rd_v > 0 else _qty
            _ms_size_cut = _qty < _ms_orig_qty * 0.95
            st.markdown(f"""
            <div style='display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px'>
                <div style='background:#fee2e2;border:2px solid #fca5a5;border-radius:10px;
                            padding:10px 14px;flex:1;min-width:80px;text-align:center'>
                    <div style='font-size:9px;font-weight:700;color:#dc2626;
                                letter-spacing:1px'>STOP LOSS</div>
                    <div style='font-size:17px;font-weight:800;color:#dc2626;
                                font-family:JetBrains Mono;margin:3px 0'>
                        ₹{_sl:,.2f}</div>
                    <div style='font-size:10px;color:#dc2626'>
                        −{_sl_pct:.1f}% below SMA20</div>
                    <div style='font-size:9px;color:#b91c1c;margin-top:3px;font-weight:700'>
                        ATR ₹{_ms_atr_v:,.2f} · Risk ₹{_ms_rd_v:,.2f}/share</div>
                </div>
                <div style='background:#eff6ff;border-radius:10px;padding:10px 14px;
                            flex:1;min-width:80px;text-align:center'>
                    <div style='font-size:9px;font-weight:700;color:#1d4ed8;
                                letter-spacing:1px'>T1 — R:R {_rr1}:1</div>
                    <div style='font-size:17px;font-weight:800;color:#1d4ed8;
                                font-family:JetBrains Mono;margin:3px 0'>
                        ₹{_t1:,.2f}</div>
                    <div style='font-size:10px;color:#1d4ed8'>
                        +{_t1_pct:.1f}% · 1× ATR · Exit 40%</div>
                </div>
                <div style='background:#f5f3ff;border-radius:10px;padding:10px 14px;
                            flex:1;min-width:80px;text-align:center'>
                    <div style='font-size:9px;font-weight:700;color:#7c3aed;
                                letter-spacing:1px'>T2 — R:R {_rr2}:1</div>
                    <div style='font-size:17px;font-weight:800;color:#7c3aed;
                                font-family:JetBrains Mono;margin:3px 0'>
                        ₹{_t2:,.2f}</div>
                    <div style='font-size:10px;color:#7c3aed'>
                        +{_t2_pct:.1f}% · 2× ATR · Exit 40%</div>
                </div>
                <div style='background:#f0fdf4;border-radius:10px;padding:10px 14px;
                            flex:1;min-width:80px;text-align:center'>
                    <div style='font-size:9px;font-weight:700;color:#15803d;
                                letter-spacing:1px'>T3 — R:R {_rr3}:1</div>
                    <div style='font-size:17px;font-weight:800;color:#15803d;
                                font-family:JetBrains Mono;margin:3px 0'>
                        ₹{_t3:,.2f}</div>
                    <div style='font-size:10px;color:#15803d'>
                        +{_t3_pct:.1f}% · 3× ATR · Trail 20%</div>
                </div>
            </div>""", unsafe_allow_html=True)

            # ── Position sizing info ──────────────────
            _ms_max_loss = int(_qty * _ms_rd_v)
            st.markdown(
                f"<div style='background:#f8fafc;border:1.5px solid "
                f"{'#fcd34d' if _ms_size_cut else '#e2e8f0'};"
                f"border-radius:10px;padding:10px 14px;margin-bottom:4px'>"
                f"<div style='display:flex;gap:20px;flex-wrap:wrap;"
                f"font-size:11px;color:#64748b'>"
                f"<span>📦 Qty: <b style='color:#1a2035'>{_qty} shares</b></span>"
                f"<span>💰 Invest: <b style='color:#1a2035'>₹{_inv:,.0f}</b></span>"
                f"<span>⚠️ Max loss if SL hit: <b style='color:#dc2626'>₹{_ms_max_loss:,}</b></span>"
                f"<span>⏳ Hold: <b style='color:#7c3aed'>3–5 weeks</b></span>"
                f"</div>"
                f"<div style='margin-top:6px;padding-top:6px;border-top:1px solid #e2e8f0;"
                f"font-size:10px;color:#64748b'>"
                f"📐 <b>ATR sizing:</b> ₹{_ms_capital:,.0f} × {_ms_adj_r:.2f}% "
                f"÷ ₹{_ms_rd_v:.2f}/share = "
                f"<b style='color:#1a2035'>{_qty} shares</b>"
                + (f" &nbsp;<span style='color:#d97706;font-weight:700'>"
                   f"(reduced from {_ms_orig_qty} — {_ms_risk_rsn2})</span>"
                   if _ms_size_cut and _ms_risk_rsn2 else "")
                + f"</div>"
                + (f"<div style='margin-top:4px;font-size:10px;font-weight:700;"
                   f"color:{_ms_r.get('risk_clr','#64748b')}'>"
                   f"⚖️ {_ms_risk_lbl2}</div>"
                   if _ms_size_cut else "")
                + f"</div>",
                unsafe_allow_html=True)

            # ── Confident Score Breakdown Strip ──────────
            st.markdown(
                f"<div style='background:{_ms_conf_bg};border:2px solid {_ms_conf_bdr};"
                f"border-radius:10px;padding:10px 16px;margin-bottom:8px'>"
                f"<div style='display:flex;align-items:center;justify-content:space-between;"
                f"flex-wrap:wrap;gap:8px'>"
                f"<div>"
                f"<div style='font-size:11px;font-weight:700;color:{_ms_conf_clr};"
                f"letter-spacing:1px'>⭐ CONFIDENT SCORE</div>"
                f"<div style='font-size:22px;font-weight:800;color:{_ms_conf_clr};"
                f"margin-top:2px'>{_ms_conf}/100 "
                f"<span style='font-size:13px'>{_ms_conf_lbl}</span></div>"
                f"</div>"
                f"<div style='display:flex;gap:6px;flex-wrap:wrap'>"
                f"<div style='background:white;border-radius:6px;padding:4px 8px;"
                f"text-align:center;min-width:44px'>"
                f"<div style='font-size:8px;color:#94a3b8;font-weight:700'>TECH</div>"
                f"<div style='font-size:13px;font-weight:800;color:#1a2035'>{_ms_c1}/30</div>"
                f"</div>"
                f"<div style='background:white;border-radius:6px;padding:4px 8px;"
                f"text-align:center;min-width:44px'>"
                f"<div style='font-size:8px;color:#94a3b8;font-weight:700'>PSAR</div>"
                f"<div style='font-size:13px;font-weight:800;"
                f"color:{'#15803d' if _ms_c2>0 else '#dc2626'}'>{_ms_c2}/25</div>"
                f"</div>"
                f"<div style='background:white;border-radius:6px;padding:4px 8px;"
                f"text-align:center;min-width:44px'>"
                f"<div style='font-size:8px;color:#94a3b8;font-weight:700'>STRUCT</div>"
                f"<div style='font-size:13px;font-weight:800;color:#1a2035'>{_ms_c3}/15</div>"
                f"</div>"
                f"<div style='background:white;border-radius:6px;padding:4px 8px;"
                f"text-align:center;min-width:44px'>"
                f"<div style='font-size:8px;color:#94a3b8;font-weight:700'>ENTRY</div>"
                f"<div style='font-size:13px;font-weight:800;color:#1a2035'>{_ms_c4}/15</div>"
                f"</div>"
                f"<div style='background:white;border-radius:6px;padding:4px 8px;"
                f"text-align:center;min-width:44px'>"
                f"<div style='font-size:8px;color:#94a3b8;font-weight:700'>R:R</div>"
                f"<div style='font-size:13px;font-weight:800;color:#1a2035'>{_ms_c5}/10</div>"
                f"</div>"
                f"<div style='background:white;border-radius:6px;padding:4px 8px;"
                f"text-align:center;min-width:44px'>"
                f"<div style='font-size:8px;color:#94a3b8;font-weight:700'>LIQ</div>"
                f"<div style='font-size:13px;font-weight:800;color:#1a2035'>{_ms_c6}/5</div>"
                f"</div>"
                f"<div style='background:white;border-radius:6px;padding:4px 8px;"
                f"text-align:center;min-width:44px'>"
                f"<div style='font-size:8px;color:#94a3b8;font-weight:700'>F&O</div>"
                f"<div style='font-size:13px;font-weight:800;"
                f"color:{'#15803d' if _ms_c7>0 else '#dc2626' if _ms_c7<0 else '#1a2035'}'>"
                f"{'+' if _ms_c7>0 else ''}{_ms_c7}</div>"
                f"</div>"
                f"<div style='background:white;border-radius:6px;padding:4px 8px;"
                f"text-align:center;min-width:54px'>"
                f"<div style='font-size:8px;color:#94a3b8;font-weight:700'>SECTOR</div>"
                + (lambda c8=_ms_r.get('c8_sector',0), nm=_ms_r.get('c8_sector_name',''):
                   f"<div style='font-size:13px;font-weight:800;"
                   f"color:{'#15803d' if c8>0 else '#dc2626' if c8<0 else '#1a2035'}'>"
                   f"{'+' if c8>0 else ''}{c8}</div>"
                   f"<div style='font-size:8px;color:#64748b'>{nm}</div>")()
                + f"</div>"
                f"</div></div></div>",
                unsafe_allow_html=True)

            # ── Price Action Analysis Strip ────────────
            _pa_data = _ms_r.get('pa', {})
            if _pa_data:
                _pa_sig     = _pa_data.get('pa_signal', '')
                _pa_clr     = _pa_data.get('pa_signal_clr', '#64748b')
                _pa_bg      = _pa_data.get('pa_signal_bg', '#f8fafc')
                _pa_total   = _pa_data.get('pa_total_score', 0)
                _pa_c_ico   = _pa_data.get('candle_emoji', '⚪')
                _pa_c_pat   = _pa_data.get('candle_pattern', '')
                _pa_c_score = _pa_data.get('candle_score', 0)
                _pa_c_desc  = _pa_data.get('candle_desc', '')
                _pa_s_name  = _pa_data.get('support_name', '')
                _pa_s_score = _pa_data.get('support_score', 0)
                _pa_s_desc  = _pa_data.get('support_desc', '')
                _pa_st      = _pa_data.get('structure', '')
                _pa_st_sc   = _pa_data.get('structure_score', 0)
                _pa_st_desc = _pa_data.get('structure_desc', '')

                _pa_c_clr  = '#15803d' if _pa_c_score > 0 else ('#dc2626' if _pa_c_score < 0 else '#64748b')
                _pa_s_clr  = '#15803d' if _pa_s_score > 0 else ('#dc2626' if _pa_s_score < 0 else '#64748b')
                _pa_st_clr = '#15803d' if _pa_st_sc  > 0 else ('#dc2626' if _pa_st_sc  < 0 else '#64748b')
                _pa_sign   = '+' if _pa_total >= 0 else ''

                st.markdown(
                    f"<div style='background:{_pa_bg};border:1.5px solid {_pa_clr}33;"
                    f"border-radius:10px;padding:12px 16px;margin-bottom:8px'>"
                    f"<div style='display:flex;align-items:center;justify-content:space-between;"
                    f"flex-wrap:wrap;gap:8px;margin-bottom:8px'>"
                    f"<div style='font-size:10px;font-weight:700;color:{_pa_clr};"
                    f"letter-spacing:1px'>📊 PRICE ACTION ANALYSIS</div>"
                    f"<div style='display:flex;align-items:center;gap:8px'>"
                    f"<span style='font-size:10px;font-weight:700;color:{_pa_clr};"
                    f"background:white;border-radius:4px;padding:2px 8px;"
                    f"border:1px solid {_pa_clr}44'>PA Score {_pa_sign}{_pa_total}</span>"
                    f"<span style='font-size:12px;font-weight:800;color:{_pa_clr}'>{_pa_sig}</span>"
                    f"</div></div>"
                    f"<div style='display:flex;gap:8px;flex-wrap:wrap'>"
                    f"<div style='background:white;border-radius:8px;padding:8px 12px;flex:1;min-width:150px'>"
                    f"<div style='font-size:9px;font-weight:700;color:#94a3b8;letter-spacing:1px'>CANDLE</div>"
                    f"<div style='font-size:13px;font-weight:700;color:{_pa_c_clr};margin-top:2px'>"
                    f"{_pa_c_ico} {_pa_c_pat} "
                    f"<span style='font-size:10px'>({'+' if _pa_c_score>=0 else ''}{_pa_c_score})</span></div>"
                    f"<div style='font-size:10px;color:#64748b;margin-top:2px'>{_pa_c_desc}</div>"
                    f"</div>"
                    f"<div style='background:white;border-radius:8px;padding:8px 12px;flex:1;min-width:150px'>"
                    f"<div style='font-size:9px;font-weight:700;color:#94a3b8;letter-spacing:1px'>SUPPORT</div>"
                    f"<div style='font-size:13px;font-weight:700;color:{_pa_s_clr};margin-top:2px'>"
                    f"{_pa_s_name} "
                    f"<span style='font-size:10px'>({'+' if _pa_s_score>=0 else ''}{_pa_s_score})</span></div>"
                    f"<div style='font-size:10px;color:#64748b;margin-top:2px'>{_pa_s_desc[:60]}</div>"
                    f"</div>"
                    f"<div style='background:white;border-radius:8px;padding:8px 12px;flex:1;min-width:150px'>"
                    f"<div style='font-size:9px;font-weight:700;color:#94a3b8;letter-spacing:1px'>STRUCTURE</div>"
                    f"<div style='font-size:13px;font-weight:700;color:{_pa_st_clr};margin-top:2px'>"
                    f"{_pa_st} "
                    f"<span style='font-size:10px'>({'+' if _pa_st_sc>=0 else ''}{_pa_st_sc})</span></div>"
                    f"<div style='font-size:10px;color:#64748b;margin-top:2px'>{_pa_st_desc[:60]}</div>"
                    f"</div></div></div>",
                    unsafe_allow_html=True)

            # ── PSAR Trailing SL Display ───────────────
            _psar_v   = _ms_r.get('psar', None)
            _psar_b   = _ms_r.get('psar_bullish', False)
            if _psar_v:
                _psar_clr  = '#15803d' if _psar_b else '#dc2626'
                _psar_bg   = '#f0fdf4' if _psar_b else '#fef2f2'

                _psar_bdr  = '#86efac' if _psar_b else '#fca5a5'
                _psar_ico  = '✅' if _psar_b else '⚠️'
                _psar_lbl  = 'Bullish — hold' if _psar_b else 'Check — may be weak'
                _psar_pct  = round((_entry - _psar_v) / _entry * 100, 1)
                st.markdown(f"""
                <div style='background:{_psar_bg};border:1px solid {_psar_bdr};
                            border-radius:8px;padding:10px 16px;margin-bottom:6px;
                            display:flex;align-items:center;gap:16px;flex-wrap:wrap'>
                    <div>
                        <div style='font-size:10px;font-weight:700;color:{_psar_clr};
                                    letter-spacing:1px'>
                            📍 WEEKLY PSAR — TRAILING SL AFTER T1
                        </div>
                        <div style='font-size:18px;font-weight:800;color:{_psar_clr};
                                    font-family:JetBrains Mono;margin-top:2px'>
                            ₹{_psar_v:,.2f}
                            <span style='font-size:11px;font-weight:600;margin-left:8px'>
                                {_psar_ico} {_psar_lbl}
                            </span>
                        </div>
                    </div>
                    <div style='font-size:11px;color:#64748b;line-height:1.8'>
                        <b>{_psar_pct:.1f}% below entry</b> ·
                        After T1 hit → move Zerodha SL to ₹{_psar_v:,.2f}<br>
                        Check every Friday · If price &lt; PSAR on weekly close → EXIT
                    </div>
                </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("</div>", unsafe_allow_html=True)


            # ── LLM Validation ────────────────────────
            _llm_key     = f"llm_ms_{_sym}_{_sc}"
            _llm_res_key = f"llm_ms_result_{_sym}_{_sc}"

            if st.button(
                f"🤖 AI Validate {_sym} — Should I enter this trade?",
                key=_llm_key, use_container_width=True
            ):
                with st.spinner(f"🤖 Analysing {_sym}..."):
                    try:
                        import requests as _req, json as _json
                        _fib_zone_txt = (
                            "23-38% Ideal" if _fib_ret<38.2 and _fib_ret>=23.6
                            else "38-50% Good" if _fib_ret<50
                            else "50-61% OK" if _fib_ret<61.8
                            else "61-78% Deep" if _fib_ret<78.6
                            else ">78% Very deep")
                        _psar_txt = f"Rs{_psar_v:.2f} bullish" if _psar_v and _psar_b else (f"Rs{_psar_v:.2f} bearish" if _psar_v else "N/A")
                        _nifty_txt = "Bullish" if st.session_state.get("ms_nifty_bullish", True) else "Bearish"
                        _prompt = (
                            f"You are an expert NSE swing trader. Analyse this Monthly Swing setup (3-5 week hold) and give verdict.\n\n"
                            f"STOCK: {_sym} | Price: Rs{_entry:.2f} | Score: {_sc}/145\n"
                            f"SMA20: Rs{_s20:.2f} ({_p20:+.1f}%) | SMA50: Rs{_s50:.2f} ({_p50:+.1f}%) | Slope: {_sl_str:+.2f}%\n"
                            f"RSI: {_rsi:.1f} | Vol: {_volx:.1f}x | Trend: {_tw}wk | Signal: {_slab}\n"
                            f"MACD: {'Fresh cross' if _macd_x else 'Above signal' if _macd_a else 'Below signal'}\n"
                            f"OBV: {'Accumulation' if _obv>0 else 'Distribution'} ({_obv:+.1f}%)\n"
                            f"RS vs Nifty: {((_rs-1)*100):+.1f}% | Sector: {'Bullish' if _sec_b else 'Bearish'}\n"
                            f"52W: {_52w:.1f}% below | HH: {_hh} | HL: {_hl} | Inside week: {_inw}\n"
                            f"Fibonacci: {_fib_zone_txt} ({_fib_ret:.1f}% retrace) | Fib38=Rs{_fib_382:.0f} Fib62=Rs{_fib_618:.0f}\n"
                            f"PSAR: {_psar_txt} | Monthly chg: {_mchg:+.1f}%\n"
                            f"D/E: {f'{_de_val:.2f}' if _de_val else 'N/A'} | Promoter: {f'{_prom_val:.0f}%' if _prom_val else 'N/A'} | EPS: {f'Rs{_eps_val:.1f}' if _eps_val else 'N/A'}\n"
                            f"Results warning: {'YES' if _ew else 'No'}{f' on {_ed}' if _ew and _ed else ''}\n"
                            f"Entry: Rs{_entry:.2f} | SL: Rs{_sl:.2f} ({((_entry-_sl)/_entry*100):.1f}%) | T1: Rs{_t1:.2f} R:R {_rr1}:1 | T2: Rs{_t2:.2f} R:R {_rr2}:1\n"
                            f"Nifty Weekly: {_nifty_txt}\n\n"
                            f"Reply ONLY with valid JSON, no markdown:\n"
                            f'{{"verdict":"BUY or WAIT or AVOID","confidence":"HIGH or MEDIUM or LOW",'
                            f'"reason":"2-3 sentences on key strengths or weaknesses",'
                            f'"main_risk":"single biggest risk for this trade",'
                            f'"best_entry":"ideal entry price zone or condition",'
                            f'"hold_target":"most realistic target T1 T2 or T3 and why",'
                            f'"score_comment":"one line on score {_sc}/145"}}'
                        )
                        _resp = _req.post(
                            "https://api.anthropic.com/v1/messages",
                            headers={"Content-Type":"application/json",
                                     "x-api-key": load_anthropic_key(),
                                     "anthropic-version":"2023-06-01"},
                            json={"model":"claude-sonnet-4-20250514","max_tokens":600,
                                  "system":"You are an expert NSE swing trading analyst. Always respond with valid JSON only. No markdown, no explanation outside JSON.",
                                  "messages":[{"role":"user","content":_prompt}]},
                            timeout=30)
                        if _resp.status_code == 200:
                            _raw = _resp.json()["content"][0]["text"].strip()
                            _clean = _raw.replace("```json","").replace("```","").strip()
                            st.session_state[_llm_res_key] = _json.loads(_clean)
                        else:
                            try:
                                _err_ps = _resp.json().get("error",{}).get("message","Unknown")
                            except Exception:
                                _err_ps = _resp.text[:150]
                            st.session_state[_llm_res_key] = {"verdict":"ERROR","confidence":"N/A",
                                "reason":f"API {_resp.status_code}: {_err_ps}","main_risk":"N/A",
                                "best_entry":"N/A","hold_target":"N/A","score_comment":"N/A"}
                    except Exception as _ex:
                        st.session_state[_llm_res_key] = {"verdict":"ERROR","confidence":"N/A",
                            "reason":str(_ex)[:120],"main_risk":"N/A",
                            "best_entry":"N/A","hold_target":"N/A","score_comment":"N/A"}

            if _llm_res_key in st.session_state:
                _lr  = st.session_state[_llm_res_key]
                _v   = _lr.get("verdict","")
                _c   = _lr.get("confidence","")
                _vclr= "#15803d" if _v=="BUY" else "#d97706" if _v=="WAIT" else "#dc2626" if _v=="AVOID" else "#64748b"
                _vbg = "#f0fdf4" if _v=="BUY" else "#fffbeb" if _v=="WAIT" else "#fef2f2" if _v=="AVOID" else "#f8fafc"
                _vico= "✅" if _v=="BUY" else "⏳" if _v=="WAIT" else "❌" if _v=="AVOID" else "⚪"
                _cbdge="🔥" if _c=="HIGH" else "📊" if _c=="MEDIUM" else "⚠️"
                _r1  = _lr.get("main_risk","").replace("<","&lt;").replace(">","&gt;")
                _r2  = _lr.get("best_entry","").replace("<","&lt;").replace(">","&gt;")
                _r3  = _lr.get("hold_target","").replace("<","&lt;").replace(">","&gt;")
                _r4  = _lr.get("reason","").replace("<","&lt;").replace(">","&gt;")
                _r5  = _lr.get("score_comment","").replace("<","&lt;").replace(">","&gt;")
                st.markdown(
                    f"<div style=\'background:{_vbg};border:2px solid {_vclr}44;"
                    f"border-radius:12px;padding:16px 18px;margin-bottom:8px\'>"
                    f"<div style=\'display:flex;align-items:center;gap:10px;margin-bottom:10px\'>"
                    f"<span style=\'font-size:20px;font-weight:900;color:{_vclr}\'>🤖 {_vico} {_v}</span>"
                    f"<span style=\'background:{_vclr};color:white;font-size:11px;"
                    f"font-weight:700;border-radius:6px;padding:3px 10px\'>{_cbdge} {_c} CONFIDENCE</span>"
                    f"</div>"
                    f"<div style=\'font-size:12px;color:#374151;line-height:1.7;margin-bottom:8px\'>"
                    f"<b>📝 Analysis:</b> {_r4}</div>"
                    f"<div style=\'display:flex;gap:10px;flex-wrap:wrap;font-size:11px\'>"
                    f"<span style=\'background:white;border-radius:6px;padding:5px 10px;"
                    f"border:1px solid {_vclr}33\'>⚠️ <b>Risk:</b> {_r1}</span>"
                    f"<span style=\'background:white;border-radius:6px;padding:5px 10px;"
                    f"border:1px solid {_vclr}33\'>🎯 <b>Entry:</b> {_r2}</span>"
                    f"<span style=\'background:white;border-radius:6px;padding:5px 10px;"
                    f"border:1px solid {_vclr}33\'>📈 <b>Target:</b> {_r3}</span>"
                    f"</div>"
                    f"<div style=\'font-size:11px;color:#64748b;margin-top:8px\'>📊 {_r5}</div>"
                    f"</div>",
                    unsafe_allow_html=True)

            # ── Paper Buy button ──────────────────────
            _ms_pb_key = f"ms_paper_{_sym}_{_sc}"
            if st.button(
                f"✅ Paper Buy  {_sym}  ·  ₹{_entry:,.2f}  ·  SL ₹{_sl:,.2f}  ·  T1 ₹{_t1:,.2f}  ·  T2 ₹{_t2:,.2f}  ·  Qty {_qty}",
                key=_ms_pb_key, use_container_width=True, type="primary"
            ):
                _ms_port = load_portfolio()
                _ms_dup  = any(p.get('symbol')==_sym and p.get('status')=='OPEN'
                               for p in _ms_port)
                if _ms_dup:
                    st.warning(f"⚠️ Already have open position in {_sym}")
                else:
                    _ms_port.append({
                        'symbol':      _sym,
                        'status':      'OPEN',
                        'entry':       round(_entry,2),
                        'qty':         _qty,
                        'stop_loss':   _sl,
                        'atr7':        round(_ms_r.get('atr7', _ms_r.get('atr', 0)), 2),
                        'risk_per_share': round(_entry - _sl, 2),
                        't1':          _t1,
                        't2':          _t2,
                        't3':          _t3,
                        't4':          0,
                        'investment':  _inv,
                        'actual_cost': _inv,
                        'timeframe':   'Weekly — Monthly Swing',
                        'date':        ist_now().strftime('%d %b %Y %H:%M'),
                        'entry_time':  ist_now().strftime('%H:%M'),
                        'nifty_state': st.session_state.get('nifty_market_state','UNKNOWN'),
                        'vix_level':   st.session_state.get('nifty_context',{}).get('vix_level','UNKNOWN'),
                        'score':       _sc,
                        'verdict':     _slab,
                        'sig_age':     _ms_r.get('pb_age',0),
                        'vol_ratio':   _volx,
                        'source':      'monthly_swing',
                        'exit_reason': '',
                        'cap_tier':    _cap,
                        'sma20':       _s20,
                        'sma50':       _s50,
                        'signal_type': _ms_r.get('signal_label',''),
                    })
                    save_portfolio(_ms_port)
                    st.session_state['paper_portfolio'] = _ms_port
                    st.success(
                        f"✅ Paper bought {_qty} × {_sym} @ ₹{_entry:,.2f} · "
                        f"SL ₹{_sl:,.2f} · T1 ₹{_t1:,.2f} · "
                        f"Hold 3–5 weeks · Source: Monthly Swing"
                    )
                    st.rerun()

        if len(_ms_show) == 0:
            st.info("No signals match this filter. Try 'All' or lower the min score.")


# ══════════════════════════════════════════════════════════════
#  🧪 BACKTEST PAGE
#  Tests 12 strategies on historical data
#  Validates app design decisions with real data
# ══════════════════════════════════════════════════════════════

if _show_backtest:

    st.markdown("""
    <div class='topbar'>
        <div class='topbar-title'>🧪 Strategy Backtester — Validate Your Edge</div>
    </div>
    """, unsafe_allow_html=True)

    # ── DIAGNOSTIC: RS vs Sector — isolated test ──────────────
    # Standalone verification tool, completely separate from the
    # live scanners. Run this FIRST and confirm numbers look right
    # before wiring get_rs_vs_sector() into scan_sma_weekly() or
    # scan_monthly_swing().
    with st.expander("🔬 Diagnostic: Test RS-vs-Sector Math (isolated, no scanner impact)"):
        st.caption(
            "Runs the corrected get_rs_vs_sector() on known stocks to verify "
            "the math before wiring it into live scanners. Zero impact on "
            "SMA Weekly / Monthly Swing — this only reads data and prints results.")

        _rs_test_stocks = st.text_input(
            "Stocks to test (comma-separated, no .NS)",
            value="HINDZINC,TORNTPHARM,PCBL,KIMS,INDUSTOWER",
            key="rs_diag_stocks")

        _rs_test_formula = st.radio(
            "Formula", ["weekly", "monthly"], horizontal=True, key="rs_diag_formula",
            help="weekly = SW periods (20/10/5 day) · monthly = MS periods (60/20/10 day)")

        if st.button("▶️ Run RS-vs-Sector Test", key="rs_diag_run"):
            _rs_diag_rankings = get_unified_sector_rankings(formula=_rs_test_formula)
            _rs_diag_syms = [s.strip().upper() for s in _rs_test_stocks.split(',') if s.strip()]

            st.markdown(f"**Nifty multi-period returns** "
                        f"(periods {_rs_diag_rankings['periods']}, "
                        f"weights {_rs_diag_rankings['weights']}): "
                        f"r1={_rs_diag_rankings['nifty_returns']['r1']:.2f}%, "
                        f"r2={_rs_diag_rankings['nifty_returns']['r2']:.2f}%, "
                        f"r3={_rs_diag_rankings['nifty_returns']['r3']:.2f}%")

            _rs_diag_rows = []
            for _sym in _rs_diag_syms:
                try:
                    _sec = classify_stock_sector(_sym)
                    _sec_rs = _rs_diag_rankings['rs_map'].get(_sec, None)

                    _interval = '1d' if _rs_test_formula == 'weekly' else '1wk'
                    _period   = '1y' if _rs_test_formula == 'weekly' else '3y'
                    _df_diag = yf.Ticker(f'{_sym}.NS').history(
                        period=_period, interval=_interval,
                        auto_adjust=True, actions=False)

                    if _df_diag is None or len(_df_diag) < 10:
                        _rs_diag_rows.append({
                            'Symbol': _sym, 'Sector': _sec, 'Error': 'No price data'})
                        continue

                    _diff, _sc, _lbl, _clr = get_rs_vs_sector(
                        _df_diag, _sec, _rs_diag_rankings)

                    _rs_diag_rows.append({
                        'Symbol':       _sym,
                        'Sector':       _sec,
                        'Sector RS vs Nifty': f"{_sec_rs:+.2f}%" if _sec_rs is not None else 'N/A',
                        'Stock vs Sector (diff)': f"{_diff:+.2f}pp",
                        'Score':        f"{_sc:+d}",
                        'Label':        _lbl,
                    })
                except Exception as _rs_diag_exc:
                    _rs_diag_rows.append({
                        'Symbol': _sym, 'Error': str(_rs_diag_exc)[:100]})

            if _rs_diag_rows:
                st.dataframe(pd.DataFrame(_rs_diag_rows), use_container_width=True)
            st.caption(
                "Sanity check: stocks known to be outperforming their sector "
                "should show positive diff + a leader label. Stocks known to "
                "be lagging/PSAR-bearish should show negative or zero diff.")

    # ── Helper functions ──────────────────────────────────────

    def bt_add_indicators(df):
        """Add all technical indicators needed for backtesting."""
        df = df.copy()
        df['SMA20']  = df['Close'].rolling(20).mean()
        df['SMA50']  = df['Close'].rolling(50).mean()
        df['RSI14']  = 100 - (100 / (1 + (
            df['Close'].diff().clip(lower=0).rolling(14).mean() /
            (-df['Close'].diff().clip(upper=0)).rolling(14).mean()
        )))
        df['ATR7']   = df['High'].combine(df['Close'].shift(1), max).subtract(
                       df['Low'].combine(df['Close'].shift(1), min)).rolling(7).mean()
        df['VolMA']  = df['Volume'].rolling(20).mean()
        return df.dropna()

    def bt_get_candle(row):
        """Classify weekly candle pattern."""
        o, h, l, c = row['Open'], row['High'], row['Low'], row['Close']
        body   = abs(c - o)
        rng    = h - l if h > l else 0.001
        lower  = min(o, c) - l
        upper  = h - max(o, c)
        if rng == 0: return 'Doji', 0
        if c > o and lower > 1.5 * body and upper < 0.3 * rng:
            return 'Hammer', 15
        if c > o and body > 0.6 * rng and lower < 0.2 * rng:
            prev_body = abs(row.get('prev_close', c) - row.get('prev_open', o))
            if body > prev_body: return 'Bullish Engulfing', 12
            if body > 0.7 * rng: return 'Strong Bull', 8
            return 'Mild Bull', 4
        if c < o and upper > 1.5 * body:
            return 'Shooting Star', -15
        if c < o and body > 0.6 * rng:
            return 'Bearish Engulfing', -12
        if body < 0.2 * rng:
            return 'Doji', -5
        return 'Neutral', 0

    def bt_get_nifty_state(nifty_df, idx):
        """Get Nifty state at a given index."""
        try:
            row = nifty_df.iloc[idx]
            sma20 = nifty_df['SMA20'].iloc[idx]
            sma50 = nifty_df['SMA50'].iloc[idx]
            close = row['Close']
            if   close > sma20 > sma50: return 'BULLISH'
            elif close > sma20:         return 'CAUTION'
            else:                       return 'BEARISH'
        except:
            return 'UNKNOWN'

    def bt_get_fno_zone(date):
        """Get F&O expiry zone for a given date."""
        import calendar
        from datetime import date as date_cls, timedelta
        if hasattr(date, 'date'):
            d = date.date()
        else:
            d = date
        year, month = d.year, d.month
        last_day  = calendar.monthrange(year, month)[1]
        last_date = date_cls(year, month, last_day)
        while last_date.weekday() != 3:
            last_date -= timedelta(days=1)
        if d > last_date:
            month = month + 1 if month < 12 else 1
            year  = year if month > 1 else year + 1
            last_day  = calendar.monthrange(year, month)[1]
            last_date = date_cls(year, month, last_day)
            while last_date.weekday() != 3:
                last_date -= timedelta(days=1)
        dte = (last_date - d).days
        if   dte <= 0:  return 'FRESH'
        elif dte <= 7:  return 'DANGER'
        elif dte <= 14: return 'CAUTION'
        else:           return 'SAFE'

    def bt_simulate_trade(df, entry_idx, entry, sl, t1, t2, max_hold):
        """Simulate a trade forward and return outcome."""
        for j in range(1, min(max_hold + 1, len(df) - entry_idx)):
            future = df.iloc[entry_idx + j]
            low    = future['Low']
            high   = future['High']
            close  = future['Close']
            psar   = future.get('PSAR', entry)
            psar_b = close > psar

            # SL hit
            if low <= sl:
                return {'outcome': 'LOSS', 'exit': sl,
                        'exit_reason': 'SL hit',
                        'hold': j, 'ret': round((sl - entry) / entry * 100, 2)}

            # T2 hit
            if high >= t2:
                return {'outcome': 'WIN', 'exit': t2,
                        'exit_reason': 'T2 hit',
                        'hold': j, 'ret': round((t2 - entry) / entry * 100, 2)}

            # T1 hit then trail with PSAR
            if high >= t1 and not psar_b:
                return {'outcome': 'WIN', 'exit': psar,
                        'exit_reason': 'PSAR after T1',
                        'hold': j, 'ret': round((psar - entry) / entry * 100, 2)}

        # Time stop
        exit_p = float(df['Close'].iloc[min(entry_idx + max_hold, len(df)-1)])
        outcome = 'WIN' if exit_p > entry else 'LOSS'
        return {'outcome': outcome, 'exit': exit_p,
                'exit_reason': 'Time stop',
                'hold': max_hold,
                'ret': round((exit_p - entry) / entry * 100, 2)}

    def run_backtest(sym_clean, scanner='monthly', years=3):
        """
        Full backtest for a single stock.
        Returns all trades and statistics.
        """
        import warnings
        warnings.filterwarnings('ignore')

        interval = '1wk' if scanner == 'monthly' else '1d'
        max_hold = 5 if scanner == 'monthly' else 10  # weeks/days

        # Fetch stock data
        t   = yf.Ticker(sym_clean + '.NS')
        df  = t.history(period=f'{years}y', interval=interval,
                        auto_adjust=True, actions=False)
        if df is None or len(df) < 60:
            return None

        df.columns = [c.split(' ')[0] if ' ' in str(c) else c for c in df.columns]
        df = bt_add_indicators(df)

        # Fetch Nifty for RS/state checks
        try:
            nf  = yf.Ticker('^NSEI')
            ndf = nf.history(period=f'{years}y', interval=interval,
                             auto_adjust=True, actions=False)
            ndf.columns = [c.split(' ')[0] if ' ' in str(c) else c for c in ndf.columns]
            ndf = bt_add_indicators(ndf)
            ndf = ndf.reindex(df.index, method='nearest')
        except:
            ndf = df.copy()

        # Add PSAR
        step   = 0.01 if scanner == 'monthly' else 0.02
        max_af = 0.10 if scanner == 'monthly' else 0.20
        try:
            df_ps = calc_psar(df.copy(), step=step, max_af=max_af)
            df['PSAR']       = df_ps['PSAR']
            df['PSAR_Bull']  = df_ps['PSAR_Bull'] if 'PSAR_Bull' in df_ps.columns else (df['Close'] > df_ps['PSAR'])
        except:
            df['PSAR']      = df['SMA20'] * 0.95
            df['PSAR_Bull'] = df['Close'] > df['PSAR']

        trades = []

        for i in range(52, len(df) - max_hold - 1):
            row    = df.iloc[i]
            sma20  = float(row['SMA20'])
            sma50  = float(row['SMA50'])
            close  = float(row['Close'])
            rsi    = float(row['RSI14'])
            atr    = float(row['ATR7'])
            psar   = float(row['PSAR'])
            psar_b = bool(row['PSAR_Bull'])

            if sma20 <= 0 or sma50 <= 0 or close <= 0:
                continue

            # ── Core entry gates ──────────────────────────
            if sma20 <= sma50:   continue
            if close <= sma20:   continue
            if rsi > 70:         continue

            sma20_prev = float(df['SMA20'].iloc[i-5])
            sma20_slope = (sma20 - sma20_prev) / sma20_prev * 100 if sma20_prev > 0 else 0
            if sma20_slope <= 0: continue

            pct_above = (close - sma20) / sma20 * 100

            # ── Candle ────────────────────────────────────
            row2 = row.copy()
            row2['prev_close'] = float(df['Close'].iloc[i-1])
            row2['prev_open']  = float(df['Open'].iloc[i-1])
            candle_name, candle_score = bt_get_candle(row2)

            if candle_name in ('Shooting Star', 'Bearish Engulfing',
                               'Bearish', 'Doji'):
                continue  # always skip bearish candles

            # ── Additional data ───────────────────────────
            nifty_state = bt_get_nifty_state(ndf, i)
            try:
                trade_date  = df.index[i]
                fno_zone    = bt_get_fno_zone(trade_date)
                _bt_fno_set, _ = load_custom_fno_list()
                is_fno      = sym_clean.upper() in _bt_fno_set
            except:
                fno_zone = 'SAFE'
                is_fno   = False

            # ── Signal type ───────────────────────────────
            # Check if SMA cross recently
            sma_cross = False
            for k in range(1, 4):
                if i-k >= 0:
                    prev_sma20 = float(df['SMA20'].iloc[i-k])
                    prev_sma50 = float(df['SMA50'].iloc[i-k])
                    if prev_sma20 <= prev_sma50 and sma20 > sma50:
                        sma_cross = True
            signal_type = 'SMA Cross' if sma_cross else 'Pullback Bounce'

            # ── Trade plan ────────────────────────────────
            entry = close
            sl    = round(sma20 * 0.97, 2)
            t1    = round(entry + 1 * atr, 2)
            t2    = round(entry + 2 * atr, 2)
            rr    = round((t2 - entry) / (entry - sl), 2) if entry > sl else 0

            if entry <= sl or rr < 1.0:
                continue

            # ── Simulate ──────────────────────────────────
            result = bt_simulate_trade(df, i, entry, sl, t1, t2, max_hold)

            trades.append({
                'date':         str(df.index[i])[:10],
                'entry':        round(entry, 2),
                'sl':           sl,
                't1':           t1,
                't2':           t2,
                'exit':         round(result['exit'], 2),
                'exit_reason':  result['exit_reason'],
                'hold':         result['hold'],
                'return_pct':   result['ret'],
                'outcome':      result['outcome'],
                'candle':       candle_name,
                'candle_score': candle_score,
                'psar_bullish': psar_b,
                'nifty_state':  nifty_state,
                'fno_zone':     fno_zone,
                'is_fno':       is_fno,
                'pct_above':    round(pct_above, 2),
                'signal_type':  signal_type,
                'rsi':          round(rsi, 1),
                'rr_t2':        rr,
                'strict_pass':  (psar_b and candle_name in
                                 ('Hammer','Bullish Engulfing','Strong Bull')),
            })

        return trades

    def calc_bt_stats(trades, filter_fn=None):
        """Calculate statistics for a list of trades."""
        if filter_fn:
            trades = [t for t in trades if filter_fn(t)]
        if not trades:
            return None
        wins   = [t for t in trades if t['outcome'] == 'WIN']
        losses = [t for t in trades if t['outcome'] == 'LOSS']
        win_rate  = len(wins) / len(trades) * 100 if trades else 0
        avg_win   = sum(t['return_pct'] for t in wins)   / len(wins)   if wins   else 0
        avg_loss  = abs(sum(t['return_pct'] for t in losses) / len(losses)) if losses else 0
        rr        = avg_win / avg_loss if avg_loss > 0 else 0
        # Max drawdown (equity curve)
        equity = 100.0
        peak   = 100.0
        max_dd = 0.0
        for t in trades:
            equity *= (1 + t['return_pct'] / 100)
            if equity > peak: peak = equity
            dd = (peak - equity) / peak * 100
            if dd > max_dd: max_dd = dd
        return {
            'total':    len(trades),
            'wins':     len(wins),
            'losses':   len(losses),
            'win_rate': round(win_rate, 1),
            'avg_win':  round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'rr':       round(rr, 2),
            'max_dd':   round(max_dd, 2),
            'total_ret':round(sum(t['return_pct'] for t in trades), 2),
            'equity':   round(equity, 2),
        }

    def show_stats_row(stats, label=''):
        """Display a stats row with colored metrics."""
        if not stats:
            st.warning(f"No trades found for {label}")
            return
        wr_clr = '🟢' if stats['win_rate'] >= 65 else ('🟡' if stats['win_rate'] >= 50 else '🔴')
        rr_clr = '🟢' if stats['rr'] >= 2.0 else ('🟡' if stats['rr'] >= 1.5 else '🔴')
        c1,c2,c3,c4,c5,c6 = st.columns(6)
        c1.metric("Trades",    stats['total'])
        c2.metric("Win Rate",  f"{wr_clr} {stats['win_rate']:.1f}%")
        c3.metric("Avg Win",   f"+{stats['avg_win']:.1f}%")
        c4.metric("Avg Loss",  f"-{stats['avg_loss']:.1f}%")
        c5.metric("R:R",       f"{rr_clr} {stats['rr']:.1f}:1")
        c6.metric("Max DD",    f"-{stats['max_dd']:.1f}%")

    def show_breakdown_chart(trades, group_fn, group_labels, title):
        """Show a breakdown bar chart by group."""
        groups = {}
        for t in trades:
            key = group_fn(t)
            if key not in groups:
                groups[key] = []
            groups[key].append(t)

        rows = []
        for label in group_labels:
            if label in groups:
                s = calc_bt_stats(groups[label])
                if s:
                    rows.append({'Group': label,
                                 'Win Rate': s['win_rate'],
                                 'Trades': s['total'],
                                 'Avg Win': s['avg_win']})

        if rows:
            import pandas as pd
            df_chart = pd.DataFrame(rows).set_index('Group')
            st.write(f"**{title}**")
            st.bar_chart(df_chart[['Win Rate']])
            st.dataframe(df_chart, use_container_width=True)

    # ─────────────────────────────────────────────────────────
    #  BACKTEST PAGE UI
    # ─────────────────────────────────────────────────────────

    st.markdown(
        "<div style='background:#0f172a;border:1px solid #1e3a8a;"
        "border-radius:12px;padding:14px 20px;margin-bottom:20px'>"
        "<div style='font-size:13px;font-weight:800;color:#93c5fd'>🧪 Strategy Backtester</div>"
        "<div style='font-size:11px;color:#64748b;margin-top:4px'>"
        "Tests your strategy on 2-3 years of historical data · "
        "Validates app design decisions with real numbers · "
        "12 strategies compared side by side"
        "</div></div>",
        unsafe_allow_html=True)

    # Controls
    _bt_col1, _bt_col2, _bt_col3, _bt_col4 = st.columns(4)
    with _bt_col1:
        _bt_sym = st.text_input(
            "Stock Symbol",
            value="KOTAKBANK",
            help="NSE symbol without .NS").upper().strip()
    with _bt_col2:
        _bt_scanner = st.selectbox(
            "Scanner Type",
            ["Monthly Swing", "SMA Weekly"],
            help="Which scanner rules to apply")
    with _bt_col3:
        _bt_years = st.selectbox(
            "Test Period",
            [2, 3, 5],
            index=1,
            help="Years of historical data")
    with _bt_col4:
        st.markdown("<br>", unsafe_allow_html=True)
        _bt_run = st.button(
            "🧪 Run Backtest",
            type="primary",
            use_container_width=True)

    if _bt_run:
        _scanner_key = 'monthly' if 'Monthly' in _bt_scanner else 'weekly'

        with st.spinner(f'Running backtest for {_bt_sym} ({_bt_years} years)...'):
            _bt_trades = run_backtest(_bt_sym, _scanner_key, _bt_years)

        if not _bt_trades:
            st.error(f"❌ No data found for {_bt_sym} or insufficient history")
        else:
            st.session_state['bt_trades']  = _bt_trades
            st.session_state['bt_sym']     = _bt_sym
            st.session_state['bt_scanner'] = _bt_scanner

    # Show results if available
    if 'bt_trades' in st.session_state:
        _bt_trades  = st.session_state['bt_trades']
        _bt_sym     = st.session_state.get('bt_sym', '')
        _bt_scanner = st.session_state.get('bt_scanner', '')

        if not _bt_trades:
            st.warning("No signals found for this stock/period")
        else:
            # ── Summary ──────────────────────────────────
            st.markdown("---")
            st.markdown(f"### 📊 {_bt_sym} — {_bt_scanner} Backtest Results")

            _all_stats = calc_bt_stats(_bt_trades)
            if _all_stats:
                _wr = _all_stats['win_rate']
                _verdict_clr = '#15803d' if _wr >= 65 else ('#d97706' if _wr >= 50 else '#dc2626')
                _verdict_bg  = '#f0fdf4' if _wr >= 65 else ('#fffbeb' if _wr >= 50 else '#fef2f2')
                _verdict_ico = '✅' if _wr >= 65 else ('⚠️' if _wr >= 50 else '❌')
                _verdict_txt = ('Strategy works well — trade with confidence' if _wr >= 65
                                else 'Marginal strategy — reduce size' if _wr >= 50
                                else 'Poor performance — review rules')

                st.markdown(
                    f"<div style='background:{_verdict_bg};border:1.5px solid {_verdict_clr}44;"
                    f"border-radius:10px;padding:12px 18px;margin-bottom:12px'>"
                    f"<span style='font-size:20px'>{_verdict_ico}</span>"
                    f"<span style='font-size:14px;font-weight:800;color:{_verdict_clr};margin-left:10px'>"
                    f"Win Rate {_wr:.1f}% — {_verdict_txt}</span>"
                    f"</div>",
                    unsafe_allow_html=True)

                show_stats_row(_all_stats)

            # ── 12 Strategy Breakdowns ────────────────────
            st.markdown("---")
            _strat_tab = st.tabs([
                "S1 Core", "S2 Strict vs Normal", "S3 Candle Quality",
                "S4 Entry Proximity", "S5 Nifty State", "S6 F&O Expiry",
                "S7 Confident Score", "S8 Signal Type", "S9 PA Signal",
                "S10 Hold Period", "S11 Sector", "S12 Trades"
            ])

            # S1 — Core strategy stats
            with _strat_tab[0]:
                st.markdown("**Strategy 1 — Core Strategy Performance**")
                st.caption("Overall win rate using all current app rules")
                show_stats_row(calc_bt_stats(_bt_trades), "Core")

                # Equity curve
                import pandas as pd
                _equity = [100.0]
                for t in _bt_trades:
                    _equity.append(_equity[-1] * (1 + t['return_pct'] / 100))
                _eq_df = pd.DataFrame({'Equity Curve': _equity})
                st.line_chart(_eq_df)

            # S2 — Strict vs Normal
            with _strat_tab[1]:
                st.markdown("**Strategy 2 — Strict Mode vs Normal Mode**")
                st.caption("Does PSAR + Hammer requirement improve win rate?")
                _sc1, _sc2 = st.columns(2)
                with _sc1:
                    st.markdown("🛡️ **Strict Mode** (PSAR + Hammer/Engulf)")
                    show_stats_row(
                        calc_bt_stats(_bt_trades,
                            lambda t: t['strict_pass']), "Strict")
                with _sc2:
                    st.markdown("📊 **Normal Mode** (All signals)")
                    show_stats_row(
                        calc_bt_stats(_bt_trades,
                            lambda t: not t['strict_pass']), "Normal")

            # S3 — Candle quality
            with _strat_tab[2]:
                st.markdown("**Strategy 3 — Candle Quality Impact**")
                st.caption("Which candle type performs best?")
                _candles = ['Hammer','Bullish Engulfing','Strong Bull','Mild Bull','Neutral']
                for _c in _candles:
                    _cs = calc_bt_stats(_bt_trades, lambda t,c=_c: t['candle'] == c)
                    if _cs and _cs['total'] > 0:
                        _ico = '🔨' if _c=='Hammer' else '🟢' if 'Engulf' in _c else '💚' if 'Strong' in _c else '🟡'
                        st.markdown(f"**{_ico} {_c}** — {_cs['total']} trades")
                        show_stats_row(_cs, _c)
                        st.markdown("---")

            # S4 — Entry proximity
            with _strat_tab[3]:
                st.markdown("**Strategy 4 — Entry Proximity Impact**")
                st.caption("Does entering closer to SMA20 improve results?")
                _prox_groups = [
                    ('≤1% (ENTER NOW tight)', lambda t: t['pct_above'] <= 1.0),
                    ('1-2% (ENTER NOW)',       lambda t: 1.0 < t['pct_above'] <= 2.0),
                    ('2-3% (ACCEPTABLE)',      lambda t: 2.0 < t['pct_above'] <= 3.0),
                    ('3-5% (ACCEPTABLE)',      lambda t: 3.0 < t['pct_above'] <= 5.0),
                    ('>5% (Extended)',         lambda t: t['pct_above'] > 5.0),
                ]
                for _label, _fn in _prox_groups:
                    _ps = calc_bt_stats(_bt_trades, _fn)
                    if _ps and _ps['total'] > 0:
                        st.markdown(f"**{_label}** — {_ps['total']} trades")
                        show_stats_row(_ps, _label)
                        st.markdown("---")

            # S5 — Nifty state
            with _strat_tab[4]:
                st.markdown("**Strategy 5 — Nifty State Impact**")
                st.caption("Does Nifty state at entry affect win rate?")
                for _ns in ['BULLISH', 'CAUTION', 'BEARISH']:
                    _nss = calc_bt_stats(_bt_trades,
                                         lambda t, n=_ns: t['nifty_state'] == n)
                    if _nss and _nss['total'] > 0:
                        _ic = '🟢' if _ns=='BULLISH' else '⚠️' if _ns=='CAUTION' else '🔴'
                        st.markdown(f"**{_ic} Nifty {_ns}** — {_nss['total']} trades")
                        show_stats_row(_nss, _ns)
                        st.markdown("---")

            # S6 — F&O expiry
            with _strat_tab[5]:
                st.markdown("**Strategy 6 — F&O Expiry Zone Impact**")
                st.caption("Does entering in DANGER zone reduce win rate?")
                for _fz in ['SAFE', 'CAUTION', 'DANGER', 'FRESH']:
                    _fzs = calc_bt_stats(_bt_trades,
                                          lambda t, z=_fz: t['fno_zone'] == z)
                    if _fzs and _fzs['total'] > 0:
                        _fic = '🟢' if _fz in ('SAFE','FRESH') else '⚠️' if _fz=='CAUTION' else '🔴'
                        st.markdown(f"**{_fic} {_fz} Zone** — {_fzs['total']} trades")
                        show_stats_row(_fzs, _fz)
                        st.markdown("---")

            # S7 — R:R quality (proxy for confident score)
            with _strat_tab[6]:
                st.markdown("**Strategy 7 — R:R Quality (Confident Score Proxy)**")
                st.caption("Does higher R:R at entry = better outcome?")
                _rr_groups = [
                    ('R:R ≥ 3.0 (excellent)',  lambda t: t['rr_t2'] >= 3.0),
                    ('R:R 2.0-3.0 (good)',     lambda t: 2.0 <= t['rr_t2'] < 3.0),
                    ('R:R 1.5-2.0 (ok)',       lambda t: 1.5 <= t['rr_t2'] < 2.0),
                    ('R:R < 1.5 (poor)',       lambda t: t['rr_t2'] < 1.5),
                ]
                for _label, _fn in _rr_groups:
                    _rrs = calc_bt_stats(_bt_trades, _fn)
                    if _rrs and _rrs['total'] > 0:
                        st.markdown(f"**{_label}** — {_rrs['total']} trades")
                        show_stats_row(_rrs, _label)
                        st.markdown("---")

            # S8 — Signal type
            with _strat_tab[7]:
                st.markdown("**Strategy 8 — Signal Type Performance**")
                st.caption("Pullback vs SMA Cross — which is more reliable?")
                for _st in ['Pullback Bounce', 'SMA Cross']:
                    _sts = calc_bt_stats(_bt_trades,
                                         lambda t, s=_st: t['signal_type'] == s)
                    if _sts and _sts['total'] > 0:
                        st.markdown(f"**{_st}** — {_sts['total']} trades")
                        show_stats_row(_sts, _st)
                        st.markdown("---")

            # S9 — PSAR state
            with _strat_tab[8]:
                st.markdown("**Strategy 9 — PSAR State at Entry**")
                st.caption("PSAR bullish vs bearish at entry")
                _pc1, _pc2 = st.columns(2)
                with _pc1:
                    st.markdown("✅ **PSAR Bullish at entry**")
                    show_stats_row(
                        calc_bt_stats(_bt_trades,
                            lambda t: t['psar_bullish']), "PSAR Bull")
                with _pc2:
                    st.markdown("❌ **PSAR Bearish at entry**")
                    show_stats_row(
                        calc_bt_stats(_bt_trades,
                            lambda t: not t['psar_bullish']), "PSAR Bear")

            # S10 — Hold period
            with _strat_tab[9]:
                st.markdown("**Strategy 10 — Hold Period Analysis**")
                st.caption("Optimal number of weeks/days to hold")
                for _hp in [1, 2, 3, 4, 5]:
                    _hps = calc_bt_stats(_bt_trades,
                                         lambda t, h=_hp: t['hold'] == h)
                    if _hps and _hps['total'] > 0:
                        unit = 'weeks' if 'Monthly' in _bt_scanner else 'days'
                        st.markdown(f"**Hold = {_hp} {unit}** — {_hps['total']} trades")
                        show_stats_row(_hps, f"{_hp}{unit}")
                        st.markdown("---")

            # S11 — Exit reason
            with _strat_tab[10]:
                st.markdown("**Strategy 11 — Exit Reason Analysis**")
                st.caption("T2 hit vs PSAR vs SL vs Time stop")
                for _er in ['T2 hit', 'PSAR after T1', 'SL hit', 'Time stop']:
                    _ers = calc_bt_stats(_bt_trades,
                                         lambda t, e=_er: t['exit_reason'] == e)
                    if _ers and _ers['total'] > 0:
                        st.markdown(f"**{_er}** — {_ers['total']} trades")
                        show_stats_row(_ers, _er)
                        st.markdown("---")

            # S12 — All trades table
            with _strat_tab[11]:
                st.markdown("**Strategy 12 — All Trades History**")
                st.caption(f"Complete list of {len(_bt_trades)} historical trades")
                import pandas as pd
                _df_trades = pd.DataFrame(_bt_trades)
                _df_trades['outcome_ico'] = _df_trades['outcome'].map(
                    {'WIN': '✅ WIN', 'LOSS': '❌ LOSS'})
                _display_cols = ['date','entry','exit','exit_reason','hold',
                                 'return_pct','candle','psar_bullish',
                                 'nifty_state','fno_zone','outcome_ico']
                _display_cols = [c for c in _display_cols if c in _df_trades.columns]
                st.dataframe(
                    _df_trades[_display_cols].rename(columns={
                        'date': 'Date', 'entry': 'Entry', 'exit': 'Exit',
                        'exit_reason': 'Exit Reason', 'hold': 'Hold',
                        'return_pct': 'Return %', 'candle': 'Candle',
                        'psar_bullish': 'PSAR ✅',
                        'nifty_state': 'Nifty', 'fno_zone': 'F&O Zone',
                        'outcome_ico': 'Result'
                    }),
                    use_container_width=True,
                    height=400)

                # Download
                _csv_bt = _df_trades.to_csv(index=False)
                st.download_button(
                    f"⬇️ Download {_bt_sym} Backtest CSV",
                    _csv_bt,
                    f"backtest_{_bt_sym}_{_bt_scanner.replace(' ','_')}.csv",
                    "text/csv")

# ══════════════════════════════════════════════════════════════
#  🏆 SECTOR LEADERS PAGE
#  Step 1: Rank sectors by ETF performance (weighted RS)
#  Step 2: Pick top 3 sectors automatically
#  Step 3: Fetch all stocks from those sectors (hardcoded)
#  Step 4: Apply full app logic to scan those stocks
#  Step 5: Show results grouped by sector
# ══════════════════════════════════════════════════════════════

# ── Sector ETF tickers for performance ranking ────────────────
SECTOR_ETF_TICKERS = {
    'BANKING':    '^NSEBANK',
    'IT':         '^CNXIT',
    'AUTO':       '^CNXAUTO',
    'PHARMA':     '^CNXPHARMA',
    'FMCG':       '^CNXFMCG',
    'METALS':     '^CNXMETAL',
    'ENERGY':     '^CNXENERGY',
    'REALTY':     '^CNXREALTY',
    'INFRA':      '^CNXINFRA',
    'FINANCE':    'NIFTY_FIN_SERVICE.NS',  # FIXED 20-Jun-2026: old ^CNXFIN was dead on Yahoo
    'PSU_BANK':   '^CNXPSUBANK',
    'MEDIA':      '^CNXMEDIA',
    'CONSUMER':   '^CNXCONSUM',
    'DEFENCE':    '^CNXINFRA',   # proxy
}

# ── Complete stock universe per sector ────────────────────────
# Based on NSE sector index constituents
# Updated as of June 2026
SECTOR_STOCKS_MAP = {

    'BANKING': [
        'HDFCBANK','ICICIBANK','KOTAKBANK','AXISBANK','SBIN',
        'INDUSINDBK','BANDHANBNK','IDFCFIRSTB','FEDERALBNK',
        'AUBANK','RBLBANK','CANBK','PNB','BANKBARODA',
        'UNIONBANK','INDIANB','CUB','KARURVYSYA',
    ],

    'PSU_BANK': [
        'SBIN','PNB','CANBK','BANKBARODA','UNIONBANK',
        'INDIANB','IOB','CENTRALBK','UCOBANK','BANKINDIA',
        'MAHABANK',
    ],

    'FINANCE': [
        'BAJFINANCE','BAJAJFINSV','CHOLAFIN','MUTHOOTFIN',
        'SHRIRAMFIN','MANAPPURAM','M&MFIN','PNBHOUSING',
        'LICHSGFIN','CANFINHOME','360ONE','ANANDRATHI',
        'MOTILALOFS','ANGELONE','HDFCAMC','ABSLAMC',
        'NAM-INDIA','CDSL','CAMS','BSE','MCX',
    ],

    'IT': [
        'TCS','INFY','WIPRO','HCLTECH','TECHM',
        'LTIM','MPHASIS','COFORGE','PERSISTENT','OFSS',
        'KPITTECH','TATAELXSI','CYIENT','NEWGEN',
        'LATENTVIEW','ECLERX','AFFLE','BSOFT',
    ],

    'PHARMA': [
        'SUNPHARMA','DRREDDY','CIPLA','DIVISLAB','AUROPHARMA',
        'ALKEM','LUPIN','TORNTPHARM','IPCALAB','GRANULES',
        'GLENMARK','NATCOPHARM','LAURUSLABS','MANKIND',
        'JBCHEPHARM','ZYDUSLIFE','BIOCON','GLAND',
        'APOLLOHOSP','MAXHEALTH','FORTIS',
    ],

    'FMCG': [
        'HINDUNILVR','ITC','NESTLEIND','BRITANNIA','DABUR',
        'MARICO','TATACONSUM','GODREJCP','COLPAL','EMAMILTD',
        'VBL','RADICO','UNITDSPR','JYOTHYLAB','BIKAJI',
        'PATANJALI','GILLETTE',
    ],

    'AUTO': [
        'MARUTI','TATAMOTORS','M&M','BAJAJ-AUTO','HEROMOTOCO',
        'EICHERMOT','TVSMOTOR','ASHOKLEY','ESCORTS',
        'MOTHERSON','BHARATFORG','BOSCHLTD','TIINDIA',
        'ENDURANCE','SONACOMS','APOLLOTYRE','CEATLTD',
        'BALKRISIND','MRF','EXIDEIND','UNOMINDA',
    ],

    'METALS': [
        'TATASTEEL','JSWSTEEL','HINDALCO','SAIL','VEDL',
        'NATIONALUM','NMDC','COALINDIA','HINDCOPPER',
        'WELCORP','JINDALSTEL','JSL','HINDZINC','MOIL',
        'GPIL','GRAVITA',
    ],

    'ENERGY': [
        'RELIANCE','ONGC','BPCL','IOC','NTPC',
        'POWERGRID','ADANIPOWER','TATAPOWER','GAIL',
        'PETRONET','IGL','MGL','ATGL','TORNTPOWER',
        'HINDPETRO','OIL','CESC',
    ],

    'INFRA': [
        'LT','SIEMENS','ABB','BHEL','THERMAX',
        'CUMMINSIND','KEC','KPIL','NCC','NBCC',
        'IRB','ENGINERSIN','RITES','IRCON',
        'CONCOR','GMRAIRPORT','JSWINFRA','AFCONS',
        'AIAENG','ACE','TITAGARH',
    ],

    'DEFENCE': [
        'HAL','BEL','BDL','BEML','COCHINSHIP',
        'GRSE','MAZDOCK','DATAPATTNS','PARAS',
        'RVNL','IRFC','IRCTC','RAILTEL',
    ],

    'REALTY': [
        'DLF','GODREJPROP','PRESTIGE','OBEROIRLTY',
        'BRIGADE','SOBHA','PHOENIXLTD','ANANTRAJ',
        'LODHA','SIGNATURE','CHALET','DBREALTY',
    ],

    'CONSUMER': [
        'HAVELLS','VOLTAS','CROMPTON','DIXON','AMBER',
        'BATAINDIA','VGUARD','POLYCAB','KEI','RRKABEL',
        'BLUESTARCO','CGPOWER','SOLARINDS','CERA',
        'KAJARIACER','CENTURYPLY','ASTRAL','APLAPOLLO',
        'SUPREMEIND','TRIDENT','APARINDS',
    ],

    'CHEMICALS': [
        'PIDILITIND','ASIANPAINT','BERGEPAINT','AARTIIND',
        'DEEPAKNTR','NAVINFLUOR','SRF','ATUL','PIIND',
        'TATACHEM','UPL','CLEAN','FLUOROCHEM',
        'DCMSHRIRAM','BASF','CASTROLIND',
    ],

    'TELECOM': [
        'BHARTIARTL','IDEA','TATACOMM','HFCL',
        'INDUSTOWER','TEJASNET','BHARTIHEXA',
    ],

    'SOLAR': [
        'WAAREEENER','SUZLON','ADANIGREEN','NHPC',
        'SJVN','INOXWIND','NTPCGREEN','JSWENERGY',
        'ADANIENSOL','TATAPOWER',
    ],
}

# ══════════════════════════════════════════════════════════════
#  🏆 SECTOR LEADERS PAGE — CSV Upload Based
#  Upload NSE sector CSVs (one per sector)
#  Auto-detects sector name from filename
#  Ranks sectors by weighted RS vs Nifty
#  Scans stocks from top N sectors
# ══════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════
#  🏆 SECTOR LEADERS PAGE — Folder-Based CSV Auto-Load
#  Place NSE sector CSVs in:
#    /Users/balaji/Desktop/Intraday_APP/Trading/sectors/
#  App reads all CSVs from that folder automatically
#  Just paste updated CSVs — no code change needed
# ══════════════════════════════════════════════════════════════

import os as _os
import glob as _glob

# ── Sector folder paths (checks both users) ───────────────────
_SECTOR_FOLDER_PATHS = [
    # Balaji's Mac
    _os.path.join(_os.path.expanduser('~'),
        'Desktop', 'Intraday_APP', 'Trading', 'sectors'),
    # Jeganath's Mac
    _os.path.join(_os.path.expanduser('~'),
        'Desktop', 'Balaji', 'Trading', 'sectors'),
    # Same folder as app file
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'sectors'),
    # Current working directory
    _os.path.join(_os.getcwd(), 'sectors'),
]

def _sl_find_sector_folder():
    """Find the first existing sectors folder."""
    for _p in _SECTOR_FOLDER_PATHS:
        if _os.path.isdir(_p):
            return _p
    return None

def _sl_load_sector_csvs(folder):
    """
    Load all CSVs from the sectors folder.
    Returns dict: {sector_name: [symbols]}
    """
    import pandas as _pd2
    import re as _re2

    _result = {}
    _errors = []
    _csvs   = _glob.glob(_os.path.join(folder, '*.csv'))

    def _extract_name(filename):
        fn = _os.path.basename(filename).lower().replace('.csv','')
        m  = _re2.search(r'ind_nifty(.+?)list', fn)
        if m:
            raw = m.group(1).strip('_').replace('_',' ').lower()
            _map = {
                'auto':             'AUTO',
                'banking':          'BANKING',
                'financialservices':'FINANCE',
                'financial':        'FINANCE',
                'fmcg':             'FMCG',
                'healthcare':       'HEALTHCARE',
                'it':               'IT',
                'media':            'MEDIA',
                'metal':            'METALS',
                'oilgas':           'ENERGY',
                'oil':              'ENERGY',
                'pharma':           'PHARMA',
                'privatebanking':   'PVTBANK',
                'private':          'PVTBANK',
                'psubanking':       'PSUBANK',
                'psu':              'PSUBANK',
                'realty':           'REALTY',
                'consumer':         'CONSUMER',
                'infra':            'INFRA',
                'defence':          'DEFENCE',
                'energy':           'ENERGY',
                'services':         'SERVICES',
                'midcap':           'MIDCAP',
                'smallcap':         'SMALLCAP',
            }
            for k, v in _map.items():
                if k in raw:
                    return v
            return raw.replace(' ','_').upper()[:12]
        # Non-NSE filename — use as-is
        base = _os.path.basename(filename).replace('.csv','')
        return base.upper()[:12]

    for _csv in sorted(_csvs):
        try:
            _df = _pd2.read_csv(_csv)
            # Find Symbol column
            _sym_col = next(
                (c for c in _df.columns
                 if c.strip().upper() in ('SYMBOL','NSE SYMBOL',
                                           'NSE_SYMBOL','SCRIP','TICKER')),
                None)
            if not _sym_col:
                _sym_col = next(
                    (c for c in _df.columns
                     if 'SYMBOL' in c.upper()),
                    None)
            if not _sym_col:
                _errors.append(
                    f"⚠️ {_os.path.basename(_csv)} — "
                    f"no Symbol column (found: {list(_df.columns[:3])})")
                continue
            _syms = (_df[_sym_col].dropna()
                     .str.strip().str.upper().tolist())
            _sec  = _extract_name(_csv)
            _result[_sec] = _syms
        except Exception as _e:
            _errors.append(f"⚠️ {_os.path.basename(_csv)} — {_e}")

    return _result, _errors


if _show_sectorleaders:

    st.markdown("""
    <div class='topbar'>
        <div class='topbar-title'>🏆 Sector Leaders — Auto-Load from Sectors Folder</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Find sectors folder ───────────────────────────────
    _sl_folder = _sl_find_sector_folder()

    # ── Folder path info banner ───────────────────────────
    if _sl_folder:
        _sl_csv_files = _glob.glob(_os.path.join(_sl_folder,'*.csv'))
        st.markdown(
            f"<div style='background:#0c1f12;border:1.5px solid #16a34a;"
            f"border-radius:10px;padding:12px 18px;margin-bottom:14px'>"
            f"<div style='font-size:12px;font-weight:800;color:#34d399'>✅ Sectors Folder Found</div>"
            f"<div style='font-size:11px;color:#86efac;margin-top:4px;font-family:monospace'>"
            f"{_sl_folder}</div>"
            f"<div style='font-size:11px;color:#64748b;margin-top:4px'>"
            f"{len(_sl_csv_files)} CSV files found · "
            f"Just paste updated CSVs here — app reloads automatically</div>"
            f"</div>",
            unsafe_allow_html=True)
    else:
        # Show setup instructions
        st.markdown(
            "<div style='background:#1f0c0c;border:1.5px solid #991b1b;"
            "border-radius:10px;padding:14px 18px;margin-bottom:14px'>"
            "<div style='font-size:12px;font-weight:800;color:#fca5a5'>"
            "⚠️ Sectors Folder Not Found — Create it first</div>"
            "<div style='font-size:11px;color:#fecaca;margin-top:8px;line-height:1.8'>"
            "Create this folder on your Mac:"
            "</div>",
            unsafe_allow_html=True)

        # Show paths to create
        for _p in _SECTOR_FOLDER_PATHS[:3]:
            st.code(_p, language=None)

        st.markdown(
            "<div style='background:#0f172a;border:1px solid #334155;"
            "border-radius:8px;padding:12px 16px;margin-top:8px'>"
            "<div style='font-size:12px;font-weight:700;color:#93c5fd'>"
            "Terminal command to create folder:</div>"
            "</div>",
            unsafe_allow_html=True)

        # Show terminal commands
        _balaji_path = _SECTOR_FOLDER_PATHS[0]
        st.code(
            f"mkdir -p \"{_balaji_path}\"",
            language="bash")

        st.info(
            "After creating the folder, download NSE sector CSVs and paste them there. "
            "App will auto-detect and load them.")

    # ── How it works ──────────────────────────────────────
    with st.expander("📋 Setup Guide — How to get NSE sector CSVs"):
        st.markdown(f"""
**Folder location (create this):**
```
{_SECTOR_FOLDER_PATHS[0]}
```

**Step 1 — Create the folder:**
```bash
mkdir -p "{_SECTOR_FOLDER_PATHS[0]}"
```

**Step 2 — Download CSVs from NSE:**
```
Go to: nseindia.com
→ Market Data → Indices
→ Equity Indices
→ Nifty Sectoral Indices
→ Click any sector → Download CSV
```

**Step 3 — Paste CSVs into folder:**
```
Just copy downloaded files into:
{_SECTOR_FOLDER_PATHS[0]}

App picks them up automatically
No restart needed
```

**Available sector CSVs from NSE:**
| File | Sector |
|------|--------|
| ind_niftyautolist.csv | AUTO |
| ind_niftybankinglist.csv | BANKING |
| ind_niftypharmalist.csv | PHARMA |
| ind_niftyfmcglist.csv | FMCG |
| ind_niftyitlist.csv | IT |
| ind_niftymetallist.csv | METALS |
| ind_niftyrealtylist.csv | REALTY |
| ind_niftyhealthcarelist.csv | HEALTHCARE |
| ind_niftyoilgaslist.csv | ENERGY |
| ind_niftypsubankinglist.csv | PSU BANK |
| ind_niftyprivatebankinglist.csv | PVT BANK |
| ind_niftyfinancialserviceslist.csv | FINANCE |
        """)

    if not _sl_folder:
        st.stop()

    # ── Load CSVs from folder ─────────────────────────────
    _sl_sector_stocks, _sl_errors = _sl_load_sector_csvs(_sl_folder)

    # Show errors if any
    for _err in _sl_errors:
        st.warning(_err)

    if not _sl_sector_stocks:
        st.error(
            "❌ No valid sector CSVs found in folder. "
            f"Add NSE sector CSV files to:\n{_sl_folder}")
        st.stop()

    # ── Sector summary ────────────────────────────────────
    _sl_total_stocks = sum(len(v) for v in _sl_sector_stocks.values())
    st.success(
        f"✅ {len(_sl_sector_stocks)} sectors loaded · "
        f"{_sl_total_stocks} total stocks · "
        f"Last refreshed: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}")

    # Show sector cards
    _sl_sec_list = sorted(_sl_sector_stocks.keys())
    _sl_card_cols = st.columns(min(len(_sl_sec_list), 6))
    for _si, _sec in enumerate(_sl_sec_list):
        with _sl_card_cols[_si % 6]:
            st.markdown(
                f"<div style='background:#1e293b;border-radius:8px;"
                f"padding:8px;text-align:center;font-size:11px;margin-bottom:6px'>"
                f"<div style='font-weight:700;color:#93c5fd'>{_sec}</div>"
                f"<div style='color:#94a3b8'>{len(_sl_sector_stocks[_sec])} stocks</div>"
                f"</div>",
                unsafe_allow_html=True)

    st.markdown("---")

    # ── Scanner controls ──────────────────────────────────
    _sl_c1, _sl_c2, _sl_c3 = st.columns(3)
    with _sl_c1:
        _sl_scanner = st.selectbox(
            "Scanner Type",
            ["SMA Weekly", "Monthly Swing"],
            key="sl_scanner_type")
    with _sl_c2:
        _sl_top_n = st.selectbox(
            "Top N Sectors",
            [1, 2, 3, 4, 5],
            index=2,
            key="sl_top_n")
    with _sl_c3:
        _sl_min_score = st.slider(
            "Min Score", 50, 85, 60, 5,
            key="sl_min_score")

    _sl_c4, _sl_c5 = st.columns(2)
    with _sl_c4:
        _sl_capital = st.number_input(
            "Capital per trade ₹",
            min_value=10000, max_value=500000,
            value=100000, step=10000,
            key="sl_capital")
    with _sl_c5:
        _sl_risk_pct = st.slider(
            "Risk %", 1.0, 3.0, 2.0, 0.5,
            key="sl_risk_pct")

    _sl_scanner_type = 'weekly' if 'Weekly' in _sl_scanner else 'monthly'

    # RS formula display
    if _sl_scanner_type == 'weekly':
        _w1,_w2,_w3 = 0.50,0.30,0.20
        _p1,_p2,_p3 = 20,10,5
        _l1,_l2,_l3 = '1M','2W','1W'
        _formula = "1M×50% + 2W×30% + 1W×20%"
    else:
        _w1,_w2,_w3 = 0.50,0.30,0.20
        _p1,_p2,_p3 = 60,20,10
        _l1,_l2,_l3 = '3M','1M','2W'
        _formula = "3M×50% + 1M×30% + 2W×20%"

    st.markdown(
        f"<div style='background:#1e293b;border:1px solid #334155;"
        f"border-radius:8px;padding:8px 14px;font-size:11px;"
        f"color:#94a3b8;margin-bottom:12px'>"
        f"📐 <b>Weighted RS Formula ({_sl_scanner}):</b> {_formula}"
        f"</div>",
        unsafe_allow_html=True)

    _sl_run = st.button(
        "🏆 Rank Sectors + Scan Best Stocks",
        type="primary",
        use_container_width=True)

    if _sl_run:
        _sl_prog   = st.progress(0, "🔍 Starting...")
        _sl_status = st.empty()

        # ══════════════════════════════════════════════
        # PHASE 1 — Nifty base returns
        # ══════════════════════════════════════════════
        _sl_status.info("📊 Phase 1 — Fetching Nifty base returns...")
        _sl_nf_r1=_sl_nf_r2=_sl_nf_r3=0.0
        _sl_nf_df=None
        try:
            _sl_nf_raw = yf.Ticker('^NSEI').history(
                period='6mo',interval='1d',
                auto_adjust=True,actions=False)
            _sl_nf_raw.columns=[c.split(' ')[0] if ' ' in str(c) else c for c in _sl_nf_raw.columns]
            _sl_nf_raw=_sl_nf_raw[['Close']].dropna()
            _sl_nf_df=_sl_nf_raw.copy()
            if len(_sl_nf_raw)>=_p1: _sl_nf_r1=float((_sl_nf_raw['Close'].iloc[-1]-_sl_nf_raw['Close'].iloc[-_p1])/_sl_nf_raw['Close'].iloc[-_p1]*100)
            if len(_sl_nf_raw)>=_p2: _sl_nf_r2=float((_sl_nf_raw['Close'].iloc[-1]-_sl_nf_raw['Close'].iloc[-_p2])/_sl_nf_raw['Close'].iloc[-_p2]*100)
            if len(_sl_nf_raw)>=_p3: _sl_nf_r3=float((_sl_nf_raw['Close'].iloc[-1]-_sl_nf_raw['Close'].iloc[-_p3])/_sl_nf_raw['Close'].iloc[-_p3]*100)
        except Exception: pass

        # ══════════════════════════════════════════════
        # PHASE 2 — Rank sectors
        # ══════════════════════════════════════════════
        _sl_status.info("📊 Phase 2 — Ranking sectors by weighted RS...")
        _sl_sec_scores={}; _sl_sec_detail={}; _sl_sec_bull={}
        _total_secs2=len(_sl_sector_stocks)

        for _si2,(_sec2,_syms2) in enumerate(_sl_sector_stocks.items()):
            _sl_prog.progress(int(_si2/_total_secs2*30),f"📊 {_sec2}...")
            _r1l=[];_r2l=[];_r3l=[];_bull=0
            for _ss in _syms2[:8]:  # use first 8 stocks as proxy
                try:
                    _ss_df=yf.Ticker(_ss+'.NS').history(
                        period='6mo',interval='1d',
                        auto_adjust=True,actions=False)
                    if _ss_df is None or len(_ss_df)<_p3+2: continue
                    _ss_df.columns=[c.split(' ')[0] if ' ' in str(c) else c for c in _ss_df.columns]
                    _ss_df=_ss_df[['Close']].dropna()
                    if len(_ss_df)>=_p1: _r1l.append(float((_ss_df['Close'].iloc[-1]-_ss_df['Close'].iloc[-_p1])/_ss_df['Close'].iloc[-_p1]*100)-_sl_nf_r1)
                    if len(_ss_df)>=_p2: _r2l.append(float((_ss_df['Close'].iloc[-1]-_ss_df['Close'].iloc[-_p2])/_ss_df['Close'].iloc[-_p2]*100)-_sl_nf_r2)
                    if len(_ss_df)>=_p3: _r3l.append(float((_ss_df['Close'].iloc[-1]-_ss_df['Close'].iloc[-_p3])/_ss_df['Close'].iloc[-_p3]*100)-_sl_nf_r3)
                    _ss_df['SMA20']=_ss_df['Close'].rolling(20).mean()
                    _ss_df['SMA50']=_ss_df['Close'].rolling(50).mean()
                    _ss_df=_ss_df.dropna()
                    if len(_ss_df)>0 and float(_ss_df['Close'].iloc[-1])>float(_ss_df['SMA20'].iloc[-1])>float(_ss_df['SMA50'].iloc[-1]): _bull+=1
                except Exception: continue
            _a1=sum(_r1l)/len(_r1l) if _r1l else 0
            _a2=sum(_r2l)/len(_r2l) if _r2l else 0
            _a3=sum(_r3l)/len(_r3l) if _r3l else 0
            _wt=round(_w1*_a1+_w2*_a2+_w3*_a3,2)
            _sl_sec_scores[_sec2]=_wt
            _sl_sec_detail[_sec2]={'r1':round(_a1,1),'r2':round(_a2,1),'r3':round(_a3,1),'wt':_wt}
            _sl_sec_bull[_sec2]=_bull

        _sl_ranked=sorted(_sl_sec_scores.items(),key=lambda x:x[1],reverse=True)
        _sl_top_secs=[s for s,_ in _sl_ranked[:_sl_top_n]]
        _sl_prog.progress(33,f"✅ Top {_sl_top_n}: {', '.join(_sl_top_secs)}")

        # ── Show sector ranking ───────────────────────
        st.markdown("---")
        st.markdown(f"### 📊 Sector Rankings — {_formula}")

        _rank_cols=st.columns(min(len(_sl_ranked),5))
        for _ri,(_rsec,_rsc) in enumerate(_sl_ranked[:10]):
            with _rank_cols[_ri%5]:
                _it  =_rsec in _sl_top_secs
                _rbg ='#0c1f12' if _it else '#1e293b'
                _rbdr='#16a34a' if _it else '#334155'
                _rclr='#34d399' if _rsc>=0 else '#f87171'
                _ric ='🥇' if _ri==0 else '🥈' if _ri==1 else '🥉' if _ri==2 else f'#{_ri+1}'
                _det =_sl_sec_detail.get(_rsec,{})
                _ns  =len(_sl_sector_stocks.get(_rsec,[]))
                st.markdown(
                    f"<div style='background:{_rbg};border:1.5px solid {_rbdr};"
                    f"border-radius:10px;padding:12px;text-align:center;margin-bottom:8px'>"
                    f"<div style='font-size:11px;font-weight:700;color:#94a3b8'>"
                    f"{_ric} {_rsec}{'  ✅' if _it else ''}</div>"
                    f"<div style='font-size:22px;font-weight:800;color:{_rclr}'>"
                    f"{'+' if _rsc>=0 else ''}{_rsc:.1f}</div>"
                    f"<div style='font-size:9px;color:#64748b'>Weighted RS vs Nifty</div>"
                    f"<div style='font-size:9px;color:#475569;margin-top:3px'>"
                    f"{_l1}:{'+' if _det.get('r1',0)>=0 else ''}{_det.get('r1',0):.1f}% · "
                    f"{_l2}:{'+' if _det.get('r2',0)>=0 else ''}{_det.get('r2',0):.1f}% · "
                    f"{_l3}:{'+' if _det.get('r3',0)>=0 else ''}{_det.get('r3',0):.1f}%</div>"
                    f"<div style='font-size:9px;color:#64748b;margin-top:2px'>"
                    f"{_sl_sec_bull.get(_rsec,0)} bullish · {_ns} stocks</div>"
                    f"</div>",
                    unsafe_allow_html=True)

        # ══════════════════════════════════════════════
        # PHASE 3 — Scan top sector stocks
        # ══════════════════════════════════════════════
        _sl_scan_list=[]; _sl_stock_sec={}
        for _ts in _sl_top_secs:
            for _stk in _sl_sector_stocks.get(_ts,[]):
                if _stk not in _sl_scan_list:
                    _sl_scan_list.append(_stk)
                    _sl_stock_sec[_stk]=_ts

        st.markdown("---")
        st.markdown(
            f"### 🔍 Scanning {len(_sl_scan_list)} stocks from "
            +' · '.join([f'**{s}**' for s in _sl_top_secs]))

        # Nifty swing state
        _sl_int ='1wk' if _sl_scanner_type=='monthly' else '1d'
        _sl_per2='2y'  if _sl_scanner_type=='monthly' else '1y'
        _sl_nf2=None; _sl_sw={'state':'UNKNOWN'}
        try:
            _sl_nf2t=yf.Ticker('^NSEI').history(period=_sl_per2,interval=_sl_int,auto_adjust=True,actions=False)
            _sl_nf2t.columns=[c.split(' ')[0] if ' ' in str(c) else c for c in _sl_nf2t.columns]
            _sl_nf2t['SMA20']=_sl_nf2t['Close'].rolling(20).mean()
            _sl_nf2t['SMA50']=_sl_nf2t['Close'].rolling(50).mean()
            _sl_nf2t=_sl_nf2t.dropna(); _sl_nf2=_sl_nf2t.copy()
            _nc=float(_sl_nf2t['Close'].iloc[-1]); _ns20=float(_sl_nf2t['SMA20'].iloc[-1]); _ns50=float(_sl_nf2t['SMA50'].iloc[-1])
            if   _nc>_ns20>_ns50: _sl_sw['state']='BULLISH'
            elif _nc>_ns20:       _sl_sw['state']='CAUTION'
            else:                 _sl_sw['state']='BEARISH'
            _sl_sw['sma20']=round(_ns20,2); _sl_sw['close']=round(_nc,2)
        except Exception: pass

        _sl_ns=_sl_sw.get('state','UNKNOWN')
        _sl_nc_clr={'BULLISH':'#15803d','CAUTION':'#d97706','BEARISH':'#dc2626','UNKNOWN':'#64748b'}.get(_sl_ns,'#64748b')
        _sl_nc_ico={'BULLISH':'✅','CAUTION':'⚠️','BEARISH':'🔴','UNKNOWN':'❓'}.get(_sl_ns,'❓')
        st.markdown(
            f"<div style='background:{_sl_nc_clr}22;border:1px solid {_sl_nc_clr}44;"
            f"border-radius:8px;padding:8px 14px;margin-bottom:12px;"
            f"font-size:11px;font-weight:700;color:{_sl_nc_clr}'>"
            f"{_sl_nc_ico} Nifty: {_sl_ns} · "
            f"{'High beta rewarded' if _sl_ns=='BULLISH' else 'Defensive preferred' if _sl_ns=='BEARISH' else 'Mixed'}"
            f"</div>",
            unsafe_allow_html=True)

        # Scan each stock
        _sl_results=[]; _sl_total3=len(_sl_scan_list)
        _sl_pstep=0.01 if _sl_scanner_type=='monthly' else 0.02
        _sl_pmax =0.10 if _sl_scanner_type=='monthly' else 0.20
        _sl_status.info(f"🔍 Phase 3 — Scanning {_sl_total3} stocks from top {_sl_top_n} sectors...")

        for _sli,_sl_sym in enumerate(_sl_scan_list):
            _sl_prog.progress(35+int(_sli/_sl_total3*60),f"🔍 {_sl_sym} ({_sli+1}/{_sl_total3})...")
            try:
                _sl_df=yf.Ticker(_sl_sym+'.NS').history(period=_sl_per2,interval=_sl_int,auto_adjust=True,actions=False)
                if _sl_df is None or len(_sl_df)<30: continue
                _sl_df.columns=[c.split(' ')[0] if ' ' in str(c) else c for c in _sl_df.columns]
                _sl_df['SMA20']=_sl_df['Close'].rolling(20).mean()
                _sl_df['SMA50']=_sl_df['Close'].rolling(50).mean()
                _sl_df['VolMA']=_sl_df['Volume'].rolling(20).mean()
                _sl_df=_sl_df.dropna()
                if len(_sl_df)<10: continue
                close=float(_sl_df['Close'].iloc[-1]); sma20=float(_sl_df['SMA20'].iloc[-1]); sma50=float(_sl_df['SMA50'].iloc[-1])
                vol=float(_sl_df['Volume'].iloc[-1]); volma=float(_sl_df['VolMA'].iloc[-1])
                atr=float(_sl_df['High'].iloc[-8:-1].max()-_sl_df['Low'].iloc[-8:-1].min())/7 if len(_sl_df)>=8 else close*0.03
                if close<=0 or sma20<=0 or sma50<=0: continue
                if sma20<=sma50 or close<=sma20: continue
                sma20_prev=float(_sl_df['SMA20'].iloc[-6]) if len(_sl_df)>=6 else sma20
                slope=(sma20-sma20_prev)/sma20_prev*100 if sma20_prev>0 else 0
                if slope<=0: continue
                pct_above=(close-sma20)/sma20*100; vol_ratio=vol/volma if volma>0 else 1.0
                # RS
                _sl_rs=1.0
                if _sl_nf2 is not None and len(_sl_nf2)>=5:
                    try:
                        _sr=float((_sl_df['Close'].iloc[-1]-_sl_df['Close'].iloc[-5])/_sl_df['Close'].iloc[-5]*100)
                        _nr=float((_sl_nf2['Close'].iloc[-1]-_sl_nf2['Close'].iloc[-5])/_sl_nf2['Close'].iloc[-5]*100)
                        _sl_rs=(_sr+100)/(_nr+100) if (_nr+100)>0 else 1.0
                    except: pass
                if _sl_rs<0.95: continue
                # Signal
                _has_cross=False; _has_pb=False; _cross_age=99; _pb_age=0
                _lb=3 if _sl_scanner_type=='monthly' else 5
                for _k in range(1,_lb+1):
                    if _k+1>=len(_sl_df): break
                    if float(_sl_df['SMA20'].iloc[-_k-1])<=float(_sl_df['SMA50'].iloc[-_k-1]) and float(_sl_df['SMA20'].iloc[-_k])>float(_sl_df['SMA50'].iloc[-_k]):
                        _has_cross=True; _cross_age=_k; break
                _tb=0
                for _k in range(1,min(20,len(_sl_df))):
                    if float(_sl_df['SMA20'].iloc[-_k])>float(_sl_df['SMA50'].iloc[-_k]): _tb+=1
                    else: break
                if _tb>=4:
                    for _k in range(1,4):
                        if _k>=len(_sl_df): break
                        if abs(float(_sl_df['Low'].iloc[-_k])-float(_sl_df['SMA20'].iloc[-_k]))/float(_sl_df['SMA20'].iloc[-_k])*100<=2.0 or float(_sl_df['Low'].iloc[-_k])<=float(_sl_df['SMA20'].iloc[-_k]):
                            _has_pb=True; _pb_age=_k; break
                if not _has_cross and not _has_pb: continue
                # Beta
                _sl_beta=1.0; _sl_beta_sc=0
                if _sl_nf2 is not None:
                    try: _sl_beta=calc_stock_beta(_sl_df,_sl_nf2,52); _sl_beta_sc,_,_=get_beta_score(_sl_beta,_sl_sw)
                    except: pass
                # PSAR
                _sl_psar_b=False; _sl_psar_v=0.0
                try:
                    _ps=calc_psar(_sl_df.copy(),step=_sl_pstep,max_af=_sl_pmax)
                    _sl_psar_v=round(float(_ps['PSAR'].iloc[-1]),2); _sl_psar_b=close>_sl_psar_v
                except: pass
                # Score
                _sc=0
                _cb=(25 if _cross_age==1 else 18 if _cross_age==2 else 12) if _has_cross else 0
                _pb2=(22 if _pb_age==1 else 16) if _has_pb else 0
                _sc+=max(_cb,_pb2)
                _sc+=(20 if pct_above<=1 else 15 if pct_above<=2 else 10 if pct_above<=3 else 5)
                _sc+=(10 if _tb>=12 else 7 if _tb>=6 else 4)
                _sc+=(8 if vol_ratio>=1.5 else 4 if vol_ratio>=1.0 else 0)
                _sc+=(8 if _sl_rs>=1.05 else 4 if _sl_rs>=1.0 else 0)
                _sc+=(20 if _sl_psar_b else 0); _sc+=_sl_beta_sc
                if _sc<_sl_min_score: continue
                _entry='ENTER NOW' if pct_above<=2.0 else 'ACCEPTABLE'
                _entry_clr='#15803d' if _entry=='ENTER NOW' else '#d97706'
                _sl_price=round(sma20*0.97,2); _risk_d=close-_sl_price
                if _risk_d<=0: continue
                _qty=max(1,int((_sl_capital*_sl_risk_pct/100)/_risk_d))
                _t1=round(close+1*atr,2); _t2=round(close+2*atr,2); _rr=round((_t2-close)/_risk_d,1)
                _unit='wk' if _sl_scanner_type=='monthly' else 'd'
                _sig=f"🔀 Cross {_cross_age}{_unit} ago" if _has_cross else f"📉 Pullback {_pb_age}{_unit} ago"
                _btg,_btc,_btbg,_btbdr,_btic=get_beta_grade(_sl_beta)
                _btss=f"+{_sl_beta_sc}" if _sl_beta_sc>0 else str(_sl_beta_sc)
                _sl_results.append({
                    'symbol':_sl_sym,'sector':_sl_stock_sec.get(_sl_sym,''),
                    'sec_rs':_sl_sec_scores.get(_sl_stock_sec.get(_sl_sym,''),0),
                    'close':round(close,2),'sma20':round(sma20,2),'pct_above':round(pct_above,2),
                    'score':_sc,'signal':_sig,'entry':_entry,'entry_clr':_entry_clr,
                    'sl':_sl_price,'t1':_t1,'t2':_t2,'rr':_rr,'qty':_qty,
                    'invest':round(_qty*close,0),'psar_bull':_sl_psar_b,'psar_val':_sl_psar_v,
                    'vol_ratio':round(vol_ratio,2),'rs':round(_sl_rs,3),
                    'beta':round(_sl_beta,2),'btg':_btg,'btc':_btc,'btbg':_btbg,
                    'btbdr':_btbdr,'btic':_btic,'btss':_btss,'trend_bars':_tb,
                })
            except Exception: continue

        _sl_prog.progress(100,f"✅ Done — {len(_sl_results)} signals found")
        _sl_status.empty()

        # ══════════════════════════════════════════════
        # PHASE 4 — Show results
        # ══════════════════════════════════════════════
        st.markdown("---")
        st.markdown(f"### 🎯 {len(_sl_results)} Stocks Found in Top {_sl_top_n} Sectors")

        if not _sl_results:
            st.warning("⚠️ No stocks passed filters. Try lowering min score.")
        else:
            import pandas as _sl_pd2
            _sl_results=sorted(_sl_results,key=lambda x:x['score'],reverse=True)
            _sl_by_sec={}
            for _r in _sl_results: _sl_by_sec.setdefault(_r['sector'],[]).append(_r)

            for _ts in _sl_top_secs:
                _ts_res=_sl_by_sec.get(_ts,[])
                if not _ts_res: continue
                _ts_sc=_sl_sec_scores.get(_ts,0)
                _ts_ri=next((i for i,(s,_) in enumerate(_sl_ranked) if s==_ts),0)
                _ts_ic='🥇' if _ts_ri==0 else '🥈' if _ts_ri==1 else '🥉'
                _ts_clr='#34d399' if _ts_sc>=0 else '#f87171'
                _ts_det=_sl_sec_detail.get(_ts,{})
                st.markdown(
                    f"<div style='background:#0c1f12;border:1.5px solid #16a34a;"
                    f"border-radius:10px;padding:12px 18px;margin:16px 0 8px'>"
                    f"<span style='font-size:15px;font-weight:800;color:#34d399'>"
                    f"{_ts_ic} Rank #{_ts_ri+1} — {_ts}</span>"
                    f"<span style='font-size:12px;font-weight:700;color:{_ts_clr};margin-left:14px'>"
                    f"RS: {'+' if _ts_sc>=0 else ''}{_ts_sc:.1f}</span>"
                    f"<span style='font-size:10px;color:#64748b;margin-left:14px'>"
                    f"{_l1}:{'+' if _ts_det.get('r1',0)>=0 else ''}{_ts_det.get('r1',0):.1f}% · "
                    f"{_l2}:{'+' if _ts_det.get('r2',0)>=0 else ''}{_ts_det.get('r2',0):.1f}% · "
                    f"{_l3}:{'+' if _ts_det.get('r3',0)>=0 else ''}{_ts_det.get('r3',0):.1f}%</span>"
                    f"<span style='font-size:10px;color:#64748b;margin-left:14px'>"
                    f"{len(_ts_res)} passing · {len(_sl_sector_stocks.get(_ts,[]))} total</span>"
                    f"</div>",
                    unsafe_allow_html=True)

                for _r in _ts_res:
                    _sc2=_r['score']
                    _sc_clr='#15803d' if _sc2>=75 else '#16a34a' if _sc2>=60 else '#d97706'
                    _sc_bg='#f0fdf4' if _sc2>=75 else '#dcfce7' if _sc2>=60 else '#fffbeb'
                    _sc_lbl='🔥 CONFIDENT BUY' if _sc2>=130 else '✅ STRONG' if _sc2>=100 else '👍 GOOD' if _sc2>=75 else '⚠️ WEAK'
                    st.markdown(
                        f"<div style='background:#111827;border:1px solid #1f2d45;"
                        f"border-radius:12px;padding:14px 18px;margin-bottom:10px'>"
                        f"<div style='display:flex;justify-content:space-between;align-items:center;"
                        f"flex-wrap:wrap;gap:8px;margin-bottom:10px'>"
                        f"<div><span style='font-size:20px;font-weight:800;color:#f1f5f9'>{_r['symbol']}</span>"
                        f"<span style='font-size:12px;color:#64748b;margin-left:8px'>₹{_r['close']:,.2f}</span></div>"
                        f"<div style='display:flex;gap:6px;flex-wrap:wrap'>"
                        f"<span style='background:{_sc_bg};color:{_sc_clr};font-size:11px;font-weight:700;"
                        f"border-radius:6px;padding:3px 10px'>{_sc_lbl} {_sc2}/100</span>"
                        f"<span style='background:{'#f0fdf4' if _r['entry']=='ENTER NOW' else '#fffbeb'};"
                        f"color:{_r['entry_clr']};font-size:11px;font-weight:700;border-radius:6px;padding:3px 10px'>"
                        f"{'🟢' if _r['entry']=='ENTER NOW' else '🟡'} {_r['entry']}</span>"
                        f"<span style='background:{'#eff6ff' if _r['psar_bull'] else '#fef2f2'};"
                        f"color:{'#1d4ed8' if _r['psar_bull'] else '#dc2626'};font-size:11px;"
                        f"font-weight:700;border-radius:6px;padding:3px 10px'>"
                        f"{'✅' if _r['psar_bull'] else '❌'} PSAR ₹{_r['psar_val']:,.2f}</span>"
                        f"<span style='background:{_r['btbg']};color:{_r['btc']};font-size:10px;"
                        f"font-weight:700;border-radius:6px;padding:3px 8px;border:1px solid {_r['btbdr']}'>"
                        f"{_r['btic']} β{_r['beta']:.2f} {_r['btg']} ({_r['btss']})</span>"
                        f"</div></div>"
                        f"<div style='display:flex;gap:14px;flex-wrap:wrap;font-size:11px;"
                        f"color:#94a3b8;margin-bottom:10px'>"
                        f"<span>📡 {_r['signal']}</span>"
                        f"<span>SMA20 ₹{_r['sma20']:,.2f} (+{_r['pct_above']:.1f}%)</span>"
                        f"<span>Vol {_r['vol_ratio']:.1f}×</span><span>RS {_r['rs']:.3f}</span>"
                        f"<span>Trend {_r['trend_bars']}{'wk' if _sl_scanner_type=='monthly' else 'd'}</span>"
                        f"</div>"
                        f"<div style='display:flex;gap:12px;flex-wrap:wrap;font-size:11px;"
                        f"background:#0f172a;border-radius:8px;padding:8px 12px'>"
                        f"<span>🔴 SL <b>₹{_r['sl']:,.2f}</b></span>"
                        f"<span>🎯 T1 <b>₹{_r['t1']:,.2f}</b></span>"
                        f"<span>🎯 T2 <b>₹{_r['t2']:,.2f}</b></span>"
                        f"<span>R:R <b>{_r['rr']}:1</b></span>"
                        f"<span>Qty <b>{_r['qty']}</b></span>"
                        f"<span>Invest <b>₹{_r['invest']:,.0f}</b></span>"
                        f"</div></div>",
                        unsafe_allow_html=True)

            _sl_csv=_sl_pd2.DataFrame(_sl_results).to_csv(index=False)
            st.download_button(
                f"⬇️ Download {len(_sl_results)} Sector Leader Signals",
                _sl_csv,
                f"sector_leaders_{_sl_scanner.replace(' ','_')}.csv",
                "text/csv", use_container_width=True)