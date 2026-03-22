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
KITE_CREDS_FILE = pathlib.Path.home() / "Downloads" / "kite_creds.json"

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

        # Rule 1 — Opening Volume Explosion (≥5× on first candle, bullish)
        if first_vol_ratio >= 5.0 and first_close > prev_close:
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

        # Rule 2 — Gap & Hold (>1.5% gap, holds above, volume confirming)
        if chg_pct >= 1.5 and last_close >= first_close * 0.998 and last_vol_ratio >= 2.0:
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
        if (vwap > 0 and last_close > vwap and mins_open <= 30 and
                last_vol_ratio >= 2.0 and first_close <= vwap * 1.003):
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

        # Rule 4 — ORB High Breakout (price breaks first 5-min candle high)
        if (last_close > first_high * 1.001 and last_vol_ratio >= 1.5 and mins_open <= 45):
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
            if all_bull and vr3 >= 1.5 and chg_pct >= 0.5:
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
    Reuses cached data so much faster on second run.
    """
    results = []
    total   = len(selected_stocks)
    _prog   = st.progress(0, text="🚀 Running Breakout Screener...")
    _stat   = st.empty()

    for idx, symbol in enumerate(selected_stocks):
        pct       = int(((idx + 1) / total) * 100)
        sym_clean = symbol.replace('.NS', '')
        _prog.progress(pct, text=f"🚀 {idx+1}/{total} · {sym_clean}")

        try:
            _ck = _cache_key(symbol, interval)
            if _ck in _DATA_CACHE:
                df, src = _DATA_CACHE[_ck]
            else:
                df, src = fetch_intraday(symbol, interval, period='1d', kite=kite)
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

def market_open():
    """True if current IST time is within 9:15–15:30 on a weekday."""
    now = ist_now()
    if now.weekday() >= 5:
        return False
    t = now.time()
    from datetime import time as _t
    return _t(9, 15) <= t <= _t(15, 30)

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
#  STOCK UNIVERSE
# ─────────────────────────────────────────────
POPULAR_STOCKS = sorted(list(set([
    "360ONE.NS","3MINDIA.NS","AADHARHFC.NS","AARTIIND.NS","AAVAS.NS","ABB.NS","ABBOTINDIA.NS","ABCAPITAL.NS",
    "ABFRL.NS","ABLBL.NS","ABREL.NS","ABSLAMC.NS","ACC.NS","ACE.NS","ACMESOLAR.NS","ADANIENSOL.NS",
    "ADANIENT.NS","ADANIGREEN.NS","ADANIPORTS.NS","ADANIPOWER.NS","AEGISLOG.NS","AEGISVOPAK.NS","AFCONS.NS","AFFLE.NS",
    "AGARWALEYE.NS","AIAENG.NS","AIIL.NS","AJANTPHARM.NS","AKUMS.NS","AKZOINDIA.NS","ALKEM.NS","ALKYLAMINE.NS",
    "ALOKINDS.NS","AMBER.NS","AMBUJACEM.NS","ANANDRATHI.NS","ANANTRAJ.NS","ANGELONE.NS","APARINDS.NS","APLAPOLLO.NS",
    "APLLTD.NS","APOLLOHOSP.NS","APOLLOTYRE.NS","APTUS.NS","ARE&M.NS","ASAHIINDIA.NS","ASHOKLEY.NS","ASIANPAINT.NS",
    "ASTERDM.NS","ASTRAL.NS","ASTRAZEN.NS","ATGL.NS","ATHERENERG.NS","ATUL.NS","AUBANK.NS","AUROPHARMA.NS",
    "AWL.NS","AXISBANK.NS","BAJAJ-AUTO.NS","BAJAJFINSV.NS","BAJAJHFL.NS","BAJAJHLDNG.NS","BAJFINANCE.NS","BALKRISIND.NS",
    "BALRAMCHIN.NS","BANDHANBNK.NS","BANKBARODA.NS","BANKINDIA.NS","BASF.NS","BATAINDIA.NS","BAYERCROP.NS","BBTC.NS",
    "BDL.NS","BEL.NS","BEML.NS","BERGEPAINT.NS","BHARATFORG.NS","BHARTIARTL.NS","BHARTIHEXA.NS","BHEL.NS",
    "BIKAJI.NS","BIOCON.NS","BLS.NS","BLUEDART.NS","BLUEJET.NS","BLUESTARCO.NS","BOSCHLTD.NS","BPCL.NS",
    "BRIGADE.NS","BRITANNIA.NS","BSE.NS","BSOFT.NS","CAMPUS.NS","CAMS.NS","CANBK.NS","CANFINHOME.NS",
    "CAPLIPOINT.NS","CARBORUNIV.NS","CASTROLIND.NS","CCL.NS","CDSL.NS","CEATLTD.NS","CENTRALBK.NS","CENTURYPLY.NS",
    "CERA.NS","CESC.NS","CGCL.NS","CGPOWER.NS","CHALET.NS","CHAMBLFERT.NS","CHENNPETRO.NS","CHOICEIN.NS",
    "CHOLAFIN.NS","CHOLAHLDNG.NS","CIPLA.NS","CLEAN.NS","COALINDIA.NS","COCHINSHIP.NS","COFORGE.NS","COHANCE.NS",
    "COLPAL.NS","CONCOR.NS","CONCORDBIO.NS","COROMANDEL.NS","CRAFTSMAN.NS","CREDITACC.NS","CRISIL.NS","CROMPTON.NS",
    "CUB.NS","CUMMINSIND.NS","CYIENT.NS","DABUR.NS","DALBHARAT.NS","DATAPATTNS.NS","DBREALTY.NS","DCMSHRIRAM.NS",
    "DEEPAKFERT.NS","DEEPAKNTR.NS","DELHIVERY.NS","DEVYANI.NS","DIVISLAB.NS","DIXON.NS","DLF.NS","DMART.NS",
    "DOMS.NS","DRREDDY.NS","ECLERX.NS","EICHERMOT.NS","EIDPARRY.NS","EIHOTEL.NS","ELECON.NS","ELGIEQUIP.NS",
    "EMAMILTD.NS","EMCURE.NS","ENDURANCE.NS","ENGINERSIN.NS","ENRIN.NS","ERIS.NS","ESCORTS.NS","ETERNAL.NS",
    "EXIDEIND.NS","FACT.NS","FEDERALBNK.NS","FINCABLES.NS","FINPIPE.NS","FIRSTCRY.NS","FIVESTAR.NS","FLUOROCHEM.NS",
    "FORCEMOT.NS","FORTIS.NS","FSL.NS","GAIL.NS","GESHIP.NS","GICRE.NS","GILLETTE.NS","GLAND.NS",
    "GLAXO.NS","GLENMARK.NS","GMDCLTD.NS","GMRAIRPORT.NS","GODFRYPHLP.NS","GODIGIT.NS","GODREJAGRO.NS","GODREJCP.NS",
    "GODREJIND.NS","GODREJPROP.NS","GPIL.NS","GRANULES.NS","GRAPHITE.NS","GRASIM.NS","GRAVITA.NS","GRSE.NS",
    "GSPL.NS","GUJGASLTD.NS","GVT&D.NS","HAL.NS","HAPPSTMNDS.NS","HAVELLS.NS","HBLENGINE.NS","HCLTECH.NS",
    "HDFCAMC.NS","HDFCBANK.NS","HDFCLIFE.NS","HEG.NS","HEROMOTOCO.NS","HEXT.NS","HFCL.NS","HINDALCO.NS",
    "HINDCOPPER.NS","HINDPETRO.NS","HINDUNILVR.NS","HINDZINC.NS","HOMEFIRST.NS","HONASA.NS","HONAUT.NS","HSCL.NS",
    "HUDCO.NS","HYUNDAI.NS","ICICIBANK.NS","ICICIGI.NS","ICICIPRULI.NS","IDBI.NS","IDEA.NS","IDFCFIRSTB.NS",
    "IEX.NS","IFCI.NS","IGIL.NS","IGL.NS","IIFL.NS","IKS.NS","INDGN.NS","INDHOTEL.NS",
    "INDIACEM.NS","INDIAMART.NS","INDIANB.NS","INDIGO.NS","INDUSINDBK.NS","INDUSTOWER.NS","INFY.NS","INOXINDIA.NS",
    "INOXWIND.NS","INTELLECT.NS","IOB.NS","IOC.NS","IPCALAB.NS","IRB.NS","IRCON.NS","IRCTC.NS",
    "IREDA.NS","IRFC.NS","ITC.NS","ITCHOTELS.NS","ITI.NS","J&KBANK.NS","JBCHEPHARM.NS","JBMA.NS",
    "JINDALSAW.NS","JINDALSTEL.NS","JIOFIN.NS","JKCEMENT.NS","JKTYRE.NS","JMFINANCIL.NS","JPPOWER.NS","JSL.NS",
    "JSWCEMENT.NS","JSWENERGY.NS","JSWINFRA.NS","JSWSTEEL.NS","JUBLFOOD.NS","JUBLINGREA.NS","JUBLPHARMA.NS","JWL.NS",
    "JYOTHYLAB.NS","JYOTICNC.NS","KAJARIACER.NS","KALYANKJIL.NS","KARURVYSYA.NS","KAYNES.NS","KEC.NS","KEI.NS",
    "KFINTECH.NS","KIMS.NS","KIRLOSBROS.NS","KIRLOSENG.NS","KOTAKBANK.NS","KPIL.NS","KPITTECH.NS","KPRMILL.NS",
    "KSB.NS","LALPATHLAB.NS","LATENTVIEW.NS","LAURUSLABS.NS","LEMONTREE.NS","LICHSGFIN.NS","LICI.NS","LINDEINDIA.NS",
    "LLOYDSME.NS","LODHA.NS","LT.NS","LTF.NS","LTFOODS.NS","LTM.NS","LTTS.NS","LUPIN.NS",
    "M&M.NS","M&MFIN.NS","MAHABANK.NS","MAHSCOOTER.NS","MAHSEAMLES.NS","MANAPPURAM.NS","MANKIND.NS","MANYAVAR.NS",
    "MAPMYINDIA.NS","MARICO.NS","MARUTI.NS","MAXHEALTH.NS","MAZDOCK.NS","MCX.NS","MEDANTA.NS","METROPOLIS.NS",
    "MFSL.NS","MGL.NS","MINDACORP.NS","MMTC.NS","MOTHERSON.NS","MOTILALOFS.NS","MPHASIS.NS","MRF.NS",
    "MRPL.NS","MSUMI.NS","MUTHOOTFIN.NS","NAM-INDIA.NS","NATCOPHARM.NS","NATIONALUM.NS","NAUKRI.NS","NAVA.NS",
    "NAVINFLUOR.NS","NBCC.NS","NCC.NS","NESTLEIND.NS","NETWEB.NS","NEULANDLAB.NS","NEWGEN.NS","NH.NS",
    "NHPC.NS","NIACL.NS","NIVABUPA.NS","NLCINDIA.NS","NMDC.NS","NSLNISP.NS","NTPC.NS","NTPCGREEN.NS",
    "NUVAMA.NS","NUVOCO.NS","NYKAA.NS","OBEROIRLTY.NS","OFSS.NS","OIL.NS","OLAELEC.NS","OLECTRA.NS",
    "ONESOURCE.NS","ONGC.NS","PAGEIND.NS","PATANJALI.NS","PAYTM.NS","PCBL.NS","PERSISTENT.NS","PETRONET.NS",
    "PFC.NS","PFIZER.NS","PGEL.NS","PGHH.NS","PHOENIXLTD.NS","PIDILITIND.NS","PIIND.NS","PNB.NS",
    "PNBHOUSING.NS","POLICYBZR.NS","POLYCAB.NS","POLYMED.NS","POONAWALLA.NS","POWERGRID.NS","POWERINDIA.NS","PPLPHARMA.NS",
    "PRAJIND.NS","PREMIERENE.NS","PRESTIGE.NS","PTCIL.NS","PVRINOX.NS","RADICO.NS","RAILTEL.NS","RAINBOW.NS",
    "RAMCOCEM.NS","RBLBANK.NS","RCF.NS","RECLTD.NS","REDINGTON.NS","RELIANCE.NS","RELINFRA.NS","RHIM.NS",
    "RITES.NS","RKFORGE.NS","RPOWER.NS","RRKABEL.NS","RVNL.NS","SAGILITY.NS","SAIL.NS","SAILIFE.NS",
    "SAMMAANCAP.NS","SAPPHIRE.NS","SARDAEN.NS","SAREGAMA.NS","SBFC.NS","SBICARD.NS","SBILIFE.NS","SBIN.NS",
    "SCHAEFFLER.NS","SCHNEIDER.NS","SCI.NS","SHREECEM.NS","SHRIRAMFIN.NS","SHYAMMETL.NS","SIEMENS.NS","SIGNATURE.NS",
    "SJVN.NS","SOBHA.NS","SOLARINDS.NS","SONACOMS.NS","SONATSOFTW.NS","SRF.NS","STARHEALTH.NS","SUMICHEM.NS",
    "SUNDARMFIN.NS","SUNDRMFAST.NS","SUNPHARMA.NS","SUNTV.NS","SUPREMEIND.NS","SUZLON.NS","SWANCORP.NS","SWIGGY.NS",
    "SYNGENE.NS","SYRMA.NS","TARIL.NS","TATACHEM.NS","TATACOMM.NS","TATACONSUM.NS","TATAELXSI.NS","TATAINVEST.NS",
    "TATAPOWER.NS","TATASTEEL.NS","TATATECH.NS","TBOTEK.NS","TCS.NS","TECHM.NS","TECHNOE.NS","TEJASNET.NS",
    "THELEELA.NS","THERMAX.NS","TIINDIA.NS","TIMKEN.NS","TITAGARH.NS","TITAN.NS","TMPV.NS","TORNTPHARM.NS",
    "TORNTPOWER.NS","TRENT.NS","TRIDENT.NS","TRITURBINE.NS","TRIVENI.NS","TTML.NS","TVSMOTOR.NS","UBL.NS",
    "UCOBANK.NS","ULTRACEMCO.NS","UNIONBANK.NS","UNITDSPR.NS","UNOMINDA.NS","UPL.NS","USHAMART.NS","UTIAMC.NS",
    "VBL.NS","VEDL.NS","VENTIVE.NS","VGUARD.NS","VIJAYA.NS","VMM.NS","VOLTAS.NS","VTL.NS",
    "WAAREEENER.NS","WELCORP.NS","WELSPUNLIV.NS","WHIRLPOOL.NS","WIPRO.NS","WOCKPHARMA.NS","YESBANK.NS","ZEEL.NS",
    "ZENSARTECH.NS","ZENTEC.NS","ZFCVINDIA.NS","ZYDUSLIFE.NS",
])))


# ─────────────────────────────────────────────
#  CORE ENGINE — INTRADAY DATA FETCH
# ─────────────────────────────────────────────




# ── Per-scan data cache (cleared on each new scan) ───────────────
# Prevents re-fetching the same stock if user re-runs with same settings
_DATA_CACHE: dict = {}

# ── Sector map ────────────────────────────────────────────
SECTOR_MAP = {
    # ── Banking ───────────────────────────────────────────
    'HDFCBANK':'BANKING','ICICIBANK':'BANKING','SBIN':'BANKING','AXISBANK':'BANKING',
    'KOTAKBANK':'BANKING','BANDHANBNK':'BANKING','IDFCFIRSTB':'BANKING','INDUSINDBK':'BANKING',
    'FEDERALBNK':'BANKING','CANBK':'BANKING','PNB':'BANKING','BANKBARODA':'BANKING',
    'AUBANK':'BANKING','RBLBANK':'BANKING','KARURVYSYA':'BANKING','CSBBANK':'BANKING',
    'DCBBANK':'BANKING','UJJIVANSFB':'BANKING','EQUITASBNK':'BANKING',

    # ── NBFC ──────────────────────────────────────────────
    'BAJFINANCE':'NBFC','BAJAJFINSV':'NBFC','CHOLAFIN':'NBFC','MUTHOOTFIN':'NBFC',
    'IIFL':'NBFC','M&MFIN':'NBFC','APTUS':'NBFC','CREDITACC':'NBFC',
    'MANAPPURAM':'NBFC','SHRIRAMFIN':'NBFC','LICHSGFIN':'NBFC','PNBHOUSING':'NBFC',
    'CANFINHOME':'NBFC','HOMEFIRST':'NBFC','AAVAS':'NBFC','ABCAPITAL':'NBFC',

    # ── Insurance ─────────────────────────────────────────
    'HDFCLIFE':'INSURANCE','SBILIFE':'INSURANCE','ICICIGI':'INSURANCE',
    'ICICIPRULI':'INSURANCE','STARHEALTH':'INSURANCE','NIACL':'INSURANCE',
    'GICRE':'INSURANCE','LICI':'INSURANCE',

    # ── IT ────────────────────────────────────────────────
    'TCS':'IT','INFY':'IT','WIPRO':'IT','HCLTECH':'IT','TECHM':'IT',
    'LTTS':'IT','LTIM':'IT','MPHASIS':'IT','COFORGE':'IT','PERSISTENT':'IT',
    'OFSS':'IT','KPITTECH':'IT','TATAELXSI':'IT','CYIENT':'IT',
    'BSOFT':'IT','MASTEK':'IT','SONATSOFTW':'IT','MINDTREE':'IT',

    # ── Auto ──────────────────────────────────────────────
    'MARUTI':'AUTO','TATAMOTORS':'AUTO','M&M':'AUTO','BAJAJ-AUTO':'AUTO',
    'HEROMOTOCO':'AUTO','EICHERMOT':'AUTO','TVSMOTORS':'AUTO','ASHOKLEY':'AUTO',
    'TVSMOTOR':'AUTO','ESCORTS':'AUTO','MOTHERSON':'AUTO','BHARATFORG':'AUTO',
    'BOSCHLTD':'AUTO','TIINDIA':'AUTO','ENDURANCE':'AUTO','SONACOMS':'AUTO',

    # ── Pharma & Healthcare ───────────────────────────────
    'SUNPHARMA':'PHARMA','DRREDDY':'PHARMA','CIPLA':'PHARMA','DIVISLAB':'PHARMA',
    'AUROPHARMA':'PHARMA','ALKEM':'PHARMA','LUPIN':'PHARMA','TORNTPHARM':'PHARMA',
    'IPCALAB':'PHARMA','GRANULES':'PHARMA','GLENMARK':'PHARMA','NATCOPHARM':'PHARMA',
    'ABBOTINDIA':'PHARMA','PFIZER':'PHARMA','GLAXO':'PHARMA','LAURUSLABS':'PHARMA',
    'APOLLOHOSP':'PHARMA','MAXHEALTH':'PHARMA','FORTIS':'PHARMA','METROPOLIS':'PHARMA',

    # ── Energy & Oil ──────────────────────────────────────
    'RELIANCE':'ENERGY','ONGC':'ENERGY','BPCL':'ENERGY','IOC':'ENERGY',
    'NTPC':'ENERGY','POWERGRID':'ENERGY','ADANIPOWER':'ENERGY','TATAPOWER':'ENERGY',
    'GAIL':'ENERGY','PETRONET':'ENERGY','GUJGASLTD':'ENERGY','MGL':'ENERGY',
    'IGL':'ENERGY','ATGL':'ENERGY','TORNTPOWER':'ENERGY','CESC':'ENERGY',

    # ── Solar/Renewables ──────────────────────────────────
    'WAAREEENER':'SOLAR','PREMIERENE':'SOLAR','SUZLON':'SOLAR','ADANIGREEN':'SOLAR',
    'NHPC':'SOLAR','SJVN':'SOLAR','INOXWIND':'SOLAR','WEBSOL':'SOLAR',

    # ── Metals & Mining ───────────────────────────────────
    'TATASTEEL':'METALS','JSWSTEEL':'METALS','HINDALCO':'METALS','SAIL':'METALS',
    'VEDL':'METALS','NATIONALUM':'METALS','NMDC':'METALS','COALINDIA':'METALS',
    'HINDCOPPER':'METALS','MOIL':'METALS','WELCORP':'METALS','RATNAMANI':'METALS',

    # ── FMCG ──────────────────────────────────────────────
    'HINDUNILVR':'FMCG','ITC':'FMCG','NESTLEIND':'FMCG','BRITANNIA':'FMCG',
    'DABUR':'FMCG','MARICO':'FMCG','TATACONSUM':'FMCG','GODREJCP':'FMCG',
    'COLPAL':'FMCG','EMAMILTD':'FMCG','VBL':'FMCG','RADICO':'FMCG',
    'MCDOWELL-N':'FMCG','UNITDSPR':'FMCG',

    # ── Real Estate ───────────────────────────────────────
    'DLF':'REALTY','GODREJPROP':'REALTY','PRESTIGE':'REALTY','OBEROIRLTY':'REALTY',
    'BRIGADE':'REALTY','SOBHA':'REALTY','PHOENIXLTD':'REALTY','ANANTRAJ':'REALTY',
    'MAHLIFE':'REALTY','KOLTEPATIL':'REALTY','PURVA':'REALTY','SUNTECK':'REALTY',
    'LODHA':'REALTY','SIGNATURE':'REALTY','DBREALTY':'REALTY','IBREALEST':'REALTY',

    # ── Cement ────────────────────────────────────────────
    'ULTRACEMCO':'CEMENT','AMBUJACEM':'CEMENT','ACC':'CEMENT','SHREECEM':'CEMENT',
    'DALBHARAT':'CEMENT','RAMCOCEM':'CEMENT','JKCEMENT':'CEMENT','HEIDELBERG':'CEMENT',
    'NUVOCO':'CEMENT','BIRLACORPN':'CEMENT',

    # ── Chemicals ─────────────────────────────────────────
    'PIDILITIND':'CHEMICALS','ASIANPAINT':'CHEMICALS','BERGERPAINTS':'CHEMICALS',
    'AARTIIND':'CHEMICALS','DEEPAKNTR':'CHEMICALS','NAVINFLUOR':'CHEMICALS',
    'FINEORG':'CHEMICALS','GALAXYSURF':'CHEMICALS','CLEAN':'CHEMICALS',
    'TATACHEM':'CHEMICALS','GNFC':'CHEMICALS','GUJFLUORO':'CHEMICALS',

    # ── Consumer Durables ─────────────────────────────────
    'HAVELLS':'CONSUMER','VOLTAS':'CONSUMER','BLUESTAR':'CONSUMER','WHIRLPOOL':'CONSUMER',
    'CROMPTON':'CONSUMER','ORIENTELEC':'CONSUMER','DIXON':'CONSUMER','AMBER':'CONSUMER',
    'BATAINDIA':'CONSUMER','VGUARD':'CONSUMER','POLYCAB':'CONSUMER','KEI':'CONSUMER',

    # ── Telecom ───────────────────────────────────────────
    'BHARTIARTL':'TELECOM','IDEA':'TELECOM','TATACOMM':'TELECOM','HFCL':'TELECOM',

    # ── Capital Goods / Infra ─────────────────────────────
    'LT':'INFRA','SIEMENS':'INFRA','ABB':'INFRA','BHEL':'INFRA',
    'THERMAX':'INFRA','CUMMINSIND':'INFRA','GRSE':'INFRA','BEL':'INFRA',
    'HAL':'INFRA','COCHINSHIP':'INFRA','MAZAGON':'INFRA','RVNL':'INFRA',
    'IRFC':'INFRA','RAILTEL':'INFRA','IRCTC':'INFRA',

    # ── Defence ───────────────────────────────────────────
    'SWANDEF':'DEFENCE','DATAPATTNS':'DEFENCE','PARAS':'DEFENCE',
    'MAZDOCK':'DEFENCE','ASTRAZEN':'DEFENCE',

    # ── Retail / Consumer Services ────────────────────────
    'DMART':'RETAIL','TRENT':'RETAIL','NYKAA':'RETAIL','DEVYANI':'RETAIL',
    'JUBLFOOD':'RETAIL','WESTLIFE':'RETAIL','SAPPHIRE':'RETAIL','CAMPUS':'RETAIL',

    # ── Media & Entertainment ─────────────────────────────
    'ZEEL':'MEDIA','SUNTV':'MEDIA','PVRINOX':'MEDIA','NETWORK18':'MEDIA',

    # ── Logistics ────────────────────────────────────────
    'DELHIVERY':'LOGISTICS','VRL':'LOGISTICS','MAHINDCIE':'LOGISTICS',
    'BLUEDART':'LOGISTICS','GATI':'LOGISTICS',
}

def get_nifty_market_state(kite=None):
    """
    Fetch Nifty 50 + India VIX Index (raw value).
    VIX thresholds calibrated for Indian market:
      < 12  → CALM      (very rare, ideal)
      12-15 → NORMAL    (best trading conditions)
      15-20 → ELEVATED  (slight caution)
      20-25 → HIGH      (reduce position size)
      > 25  → EXTREME   (avoid intraday)
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

        # Cache refreshes every 15 minutes
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

        # ── Fetch VIX — 5-min, today's latest value ───────
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

        # ── VIX Index classification ──────────────────────
        if _vix_val is not None:
            result['vix'] = round(_vix_val, 2)
            if _vix_val < 12:
                result['vix_level'] = 'CALM'      # Very rare — ideal conditions
            elif _vix_val < 15:
                result['vix_level'] = 'NORMAL'    # Best trading conditions
            elif _vix_val < 20:
                result['vix_level'] = 'ELEVATED'  # Slight caution
            elif _vix_val < 25:
                result['vix_level'] = 'HIGH'      # Reduce position size
            else:
                result['vix_level'] = 'EXTREME'   # VIX > 25 — avoid intraday

        # ── Market state: price % is primary ─────────────
        _above_vwap = _last > _vwap
        if   _chg >= 0.5 and _above_vwap:  result['state'] = 'BULL'
        elif _chg <= -0.5 and not _above_vwap: result['state'] = 'BEAR'
        elif _chg >= 0.5:  result['state'] = 'BULL'
        elif _chg <= -0.5: result['state'] = 'BEAR'
        elif _chg >= 0.2:  result['state'] = 'BULL'
        elif _chg <= -0.2: result['state'] = 'BEAR'
        else:              result['state'] = 'SIDEWAYS'

    except Exception:
        pass

    return result
    try:
        import pytz as _pytz
        _ist       = _pytz.timezone('Asia/Kolkata')
        _now_ist   = datetime.now(_ist)
        _today_ist = _now_ist.date()

        # Cache refreshes every 15 minutes
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

        # ── Fetch VIX — 1 YEAR daily for percentile rank ──
        _vix_val   = None
        _vix_score = None
        _vix_1y_lo = None
        _vix_1y_hi = None
        _vck = f"^VIX_1y_{_now_str[:10]}"   # daily cache key (date only)

        try:
            if _vck in _DATA_CACHE:
                _vdf_1y, _ = _DATA_CACHE[_vck]
            else:
                for _old in [k for k in list(_DATA_CACHE.keys()) if k.startswith('^VIX_1y_')]:
                    del _DATA_CACHE[_old]
                # Fetch 1 year of daily VIX data for percentile calculation
                _vdf_1y = yf.Ticker('^INDIAVIX').history(period='1y', interval='1d', auto_adjust=True)
                if _vdf_1y is not None and not _vdf_1y.empty:
                    _vdf_1y.columns = [c.split(' ')[0] if ' ' in str(c) else c for c in _vdf_1y.columns]
                    _DATA_CACHE[_vck] = (_vdf_1y, 'yfinance')

            if _vdf_1y is not None and not _vdf_1y.empty and len(_vdf_1y) >= 30:
                # Current VIX = latest daily close
                _vix_val   = float(_vdf_1y['Close'].iloc[-1])
                _vix_1y_lo = float(_vdf_1y['Low'].min())
                _vix_1y_hi = float(_vdf_1y['High'].max())

                # ── VIX Score = percentile rank ────────────
                # How many of the last 252 days had VIX BELOW today's value?
                _closes    = _vdf_1y['Close'].values
                _below     = float(np.sum(_closes < _vix_val))
                _vix_score = round((_below / len(_closes)) * 100, 1)
                # Score 0 = today's VIX is lowest in 1 year (very calm)
                # Score 100 = today's VIX is highest in 1 year (extreme fear)

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

        result['nifty_chg']   = round(_chg, 2)
        result['nifty_last']  = round(_last, 2)
        result['nifty_vwap']  = round(_vwap, 2)
        result['ema_trend']   = 'BULL' if _ema9 > _ema21 else 'BEAR'
        result['vix']         = round(_vix_val, 2)   if _vix_val   is not None else None
        result['vix_score']   = round(_vix_score, 1) if _vix_score is not None else None
        result['vix_1y_low']  = round(_vix_1y_lo, 2) if _vix_1y_lo is not None else None
        result['vix_1y_high'] = round(_vix_1y_hi, 2) if _vix_1y_hi is not None else None

        # ── VIX Score classification ──────────────────────
        # Based on percentile rank — not arbitrary fixed thresholds
        if _vix_score is not None:
            if _vix_score <= 20:
                result['vix_level'] = 'CALM'       # Bottom 20% of year — very calm
            elif _vix_score <= 40:
                result['vix_level'] = 'NORMAL'     # Below average fear
            elif _vix_score <= 60:
                result['vix_level'] = 'MODERATE'   # Average fear — neutral
            elif _vix_score <= 80:
                result['vix_level'] = 'ELEVATED'   # Above average fear — caution
            else:
                result['vix_level'] = 'EXTREME'    # Top 20% of year — avoid intraday
        elif _vix_val is not None:
            # Fallback if 1-year data unavailable
            result['vix_level'] = ('CALM' if _vix_val < 12 else
                                   'NORMAL' if _vix_val < 15 else
                                   'MODERATE' if _vix_val < 18 else
                                   'ELEVATED' if _vix_val < 22 else 'EXTREME')

        # ── Market state: price % is primary ─────────────
        _above_vwap = _last > _vwap
        if   _chg >= 0.5 and _above_vwap: result['state'] = 'BULL'
        elif _chg <= -0.5 and not _above_vwap: result['state'] = 'BEAR'
        elif _chg >= 0.5:  result['state'] = 'BULL'
        elif _chg <= -0.5: result['state'] = 'BEAR'
        elif _chg >= 0.2:  result['state'] = 'BULL'
        elif _chg <= -0.2: result['state'] = 'BEAR'
        else:              result['state'] = 'SIDEWAYS'

    except Exception:
        pass

    return result
    try:
        import pytz as _pytz
        _ist        = _pytz.timezone('Asia/Kolkata')
        _now_ist    = datetime.now(_ist)
        _today_ist  = _now_ist.date()   # Always IST date, not UTC

        # Cache key: refresh every 15 minutes
        _slot    = str(_now_ist.minute // 15)
        _now_str = _now_ist.strftime('%Y%m%d_%H') + _slot

        # ── Fetch Nifty ───────────────────────────────────
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
            # Convert index to IST for correct date comparison
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
                # Get today's VIX candles using IST date
                _vdf_today = _vdf[_vdf.index.date == _today_ist]
                if len(_vdf_today) > 0:
                    _vix_val = float(_vdf_today['Close'].iloc[-1])
                else:
                    _vix_val = float(_vdf['Close'].iloc[-1])  # fallback to latest
        except Exception:
            pass

        if len(_ndf) < 5:
            return result

        # ── Today's Nifty candles using IST date ──────────
        _today_df = _ndf[_ndf.index.date == _today_ist]
        if len(_today_df) < 3:
            # Market not open yet or weekend — use last trading day
            _all_dates  = sorted(set(_ndf.index.date))
            _today_df   = _ndf[_ndf.index.date == _all_dates[-1]] if _all_dates else _ndf.tail(20)

        _last  = float(_today_df['Close'].iloc[-1])
        _open_ = float(_today_df['Open'].iloc[0])
        _chg   = (_last - _open_) / _open_ * 100

        _ema9  = float(_today_df['Close'].ewm(span=9,  adjust=False).mean().iloc[-1])
        _ema21 = float(_today_df['Close'].ewm(span=21, adjust=False).mean().iloc[-1])

        _tp   = (_today_df['High'] + _today_df['Low'] + _today_df['Close']) / 3
        _vwap = float((_tp * _today_df['Volume']).sum() / (_today_df['Volume'].sum() + 1e-9))
        _above_vwap = _last > _vwap

        result['nifty_chg']  = round(_chg, 2)
        result['nifty_last'] = round(_last, 2)
        result['nifty_vwap'] = round(_vwap, 2)
        result['ema_trend']  = 'BULL' if _ema9 > _ema21 else 'BEAR'

        # ── VIX classification ────────────────────────────
        # Threshold raised: 22 is NOT extreme for Indian markets
        # Average VIX in volatile periods is 18-22 — this is HIGH, not EXTREME
        if _vix_val is not None:
            result['vix'] = round(_vix_val, 2)
            if _vix_val < 12:
                result['vix_level'] = 'CALM'
            elif _vix_val < 15:
                result['vix_level'] = 'NORMAL'
            elif _vix_val < 20:
                result['vix_level'] = 'ELEVATED'
            elif _vix_val < 25:
                result['vix_level'] = 'HIGH'      # 20-25: caution, not extreme
            else:
                result['vix_level'] = 'EXTREME'   # Only >25 is truly extreme

        # ── Market state: price % is primary ─────────────
        # VIX does NOT override state — it only adds scoring penalty
        # This prevents false BEAR when VIX is 22 but Nifty is up 0.5%
        if _chg >= 0.5 and _above_vwap:
            result['state'] = 'BULL'
        elif _chg <= -0.5 and not _above_vwap:
            result['state'] = 'BEAR'
        elif _chg >= 0.5:
            result['state'] = 'BULL'       # Up even if below VWAP = lean bull
        elif _chg <= -0.5:
            result['state'] = 'BEAR'       # Down even if above VWAP = lean bear
        elif _chg >= 0.2:
            result['state'] = 'BULL'
        elif _chg <= -0.2:
            result['state'] = 'BEAR'
        else:
            result['state'] = 'SIDEWAYS'

    except Exception:
        pass

    return result


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
    return f"{symbol}_{interval}_{datetime.now().strftime('%Y%m%d_%H')}"

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

    # ── Intraday targets: 0.5×, 1×, 1.5×, 2× ATR  (small but achievable) ──
    t1 = round(entry + rps * 0.5, 2)   # R:R 0.5:1 — quick scalp
    t2 = round(entry + rps * 1.0, 2)   # R:R 1:1
    t3 = round(entry + rps * 1.5, 2)   # R:R 1.5:1
    t4 = round(entry + rps * 2.0, 2)   # R:R 2:1 — stretch target

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
        ("T1 — Scalp (0.5R)",  t1),
        ("T2 — Target (1R)",   t2),
        ("T3 — Extended (1.5R)", t3),
        ("T4 — Stretch (2R)",  t4),
        ("Stop Loss",          stop_loss)
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
    _nifty_state = st.session_state.get('nifty_market_state', 'UNKNOWN')
    if _nifty_state == 'BEAR':
        scores['Market_Filter'] = -20
    elif _nifty_state == 'SIDEWAYS':
        scores['Market_Filter'] = -8
    elif _nifty_state == 'BULL':
        scores['Market_Filter'] = 8

    # ── Core indicators ───────────────────────────────────
    scores['Signal'] = 20 if r['signal_val'] == 1 else (0 if r['signal_val'] == 0 else -10)
    conf = r['live_conf']
    scores['Conf%']  = 15 if conf >= 80 else (12 if conf >= 70 else (9 if conf >= 60 else (5 if conf >= 40 else 0)))

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
    total = max(0, sum(scores.values()))

    # VIX Index caps
    _vix_level   = st.session_state.get('nifty_context', {}).get('vix_level', 'UNKNOWN')
    _nifty_state = st.session_state.get('nifty_market_state', 'UNKNOWN')
    if _vix_level == 'EXTREME':    # VIX > 25
        total = min(total, 50)     # No BUY
    elif _vix_level == 'HIGH':     # VIX 20-25
        total = min(total, 68)     # BUY possible, no STRONG BUY
    if _nifty_state == 'BEAR' and total >= 65:
        total = min(total, 60)
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
        st.session_state['active_page'] = "📊  Stock Summary"

    _NAV = [
        ("📊  Stock Summary", "Stock Summary"),
        ("📉  Price Charts",  "Price Charts"),
        ("🎯  Trade Signals", "Trade Signals"),
        ("🔍  Scanners",      "Scanners"),
        ("💼  Portfolio",     "Portfolio"),
        ("🔔  Alert Log",     "Alert Log"),
        ("📁  Scan History",  "Scan History"),
    ]
    st.markdown("<div class='sb-nav-section'>Navigation</div>", unsafe_allow_html=True)
    for _pkey, _plabel in _NAV:
        _active = st.session_state['active_page'] == _pkey
        _a_cls  = "active" if _active else ""
        _disp   = f"Portfolio ({_port_count})" if _plabel == "Portfolio" and _port_count > 0 else _plabel
        _al_cnt = len(st.session_state.get(ALERT_LOG_KEY, []))
        _badge  = ""
        if _plabel == "Portfolio" and _port_count > 0:
            _badge = f"<span class='sb-nav-badge'>{_port_count}</span>"
        elif _plabel == "Alert Log" and _al_cnt > 0:
            _badge = f"<span class='sb-nav-badge'>{_al_cnt}</span>"
        _icon_color = "#f59e0b" if _active else "#475569"
        st.markdown(f"""
        <div class='sb-nav-item {_a_cls}'>
            <span style='font-size:15px'>{_pkey.split()[0]}</span>
            <span class='sb-nav-text {_a_cls}'>{_disp}</span>
            {_badge}
        </div>""", unsafe_allow_html=True)
        if st.button(_disp, key=f"navbtn_{_pkey}", use_container_width=True):
            st.session_state['active_page'] = _pkey; st.rerun()

    active_page = st.session_state.get('active_page', "📊  Stock Summary")

    # ── Config ────────────────────────────────────────────
    st.markdown("<hr class='sb-section-divider'>", unsafe_allow_html=True)
    st.markdown("<div class='sb-section-label'>⚙ Config</div>", unsafe_allow_html=True)

    interval_label = st.selectbox("Timeframe",
        ["1min — Real-Time", "3min — Fast", "5min — Standard", "15min — Swing", "60min — Positional"])
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

    scan_mode = st.radio("Mode", ["Full NSE 500", "Custom Watchlist"])
    custom_stocks = []
    if scan_mode == "Custom Watchlist":
        custom_stocks = st.multiselect("Stocks", POPULAR_STOCKS,
            default=["RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS"])
        extra = st.text_input("Add symbol", "")
        if extra and extra.upper() not in custom_stocks:
            custom_stocks.append(extra.upper())
    selected_stocks = custom_stocks if scan_mode == "Custom Watchlist" else POPULAR_STOCKS

    min_verdict = st.select_slider("Min Verdict",
        options=["❌ AVOID","⚠️ NEUTRAL","⭐ WATCH","⭐⭐ BUY","⭐⭐⭐ STRONG BUY"],
        value="⭐ WATCH")


    run_btn = st.button("▶  Scan Now", use_container_width=True, type="primary")

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

    st.markdown("""
    <div class='sb-disclaimer'>
        ⚠️ High risk · Educational only<br>Not financial advice
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MAIN CONTENT
# ─────────────────────────────────────────────

if active_page == "💼  Portfolio":
    _show_scanner   = False
    _show_portfolio = True
    _show_alertlog  = False
elif active_page == "🔔  Alert Log":
    _show_scanner   = False
    _show_portfolio = False
    _show_alertlog  = True
else:
    _show_scanner   = True
    _show_portfolio = False
    _show_alertlog  = False

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

    # ── Nifty + VIX live bar (refreshes independently of scan) ──
    _sb1, _sb2 = st.columns([4, 1])
    with _sb1:
        st.markdown(f"""
        <div style='background:#f1f5f9;border:1px solid #e2e8f0;border-radius:10px;
                    padding:8px 18px;margin-bottom:8px;font-size:12px;color:#475569'>
            {_mkt_str} &nbsp;·&nbsp; Last scan: {st.session_state.get('scan_time','—')}{_dur_str2}{_missing_str}{_interval_warn}
        </div>""", unsafe_allow_html=True)
    with _sb2:
        _refresh_mkt = st.button("🔄 Market Data", key="refresh_market_ctx",
                                  use_container_width=True,
                                  help="Re-fetch Nifty + VIX without running full scan")

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
            'CALM':'#16a34a','NORMAL':'#16a34a','ELEVATED':'#d97706',
            'HIGH':'#dc2626','EXTREME':'#7f1d1d','UNKNOWN':'#64748b'
        }.get(_vix_level_bar,'#64748b')
        _vix_adv = {
            'CALM':     '✅ Very calm — ideal trading day',
            'NORMAL':   '✅ Normal — best conditions',
            'ELEVATED': '⚠️ Slight caution — trade normally',
            'HIGH':     '⚠️ Reduce position size by 50%',
            'EXTREME':  '🚫 VIX > 25 — avoid intraday',
            'UNKNOWN':  '',
        }.get(_vix_level_bar, '')

        _bar_cols = st.columns([1, 2, 1])
        with _bar_cols[0]:
            _ni = {'BULL':'📈','SIDEWAYS':'↔️','BEAR':'📉','UNKNOWN':'❓'}.get(_nifty_state_bar,'❓')
            st.markdown(
                f"<div style='background:{_nc_bar}22;border:1px solid {_nc_bar}44;"
                f"border-radius:8px;padding:10px 14px;margin-bottom:8px'>"
                f"<div style='font-size:11px;font-weight:700;color:{_nc_bar}'>"
                f"{_ni} Nifty: {_nifty_state_bar}</div>"
                f"<div style='font-size:18px;font-weight:800;color:{_nc_bar};"
                f"font-family:JetBrains Mono'>"
                f"{'+' if _nifty_chg_bar>=0 else ''}{_nifty_chg_bar:.2f}%</div>"
                f"</div>", unsafe_allow_html=True)

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

        # Only show EXTREME warning when truly current AND VIX is still extreme
        if _vix_level_bar == 'EXTREME':
            _ext_col1, _ext_col2 = st.columns([5, 1])
            with _ext_col1:
                st.warning("🚫 India VIX > 22 — Intraday trading extremely risky. "
                           "All verdicts capped at WATCH. Consider staying out.")
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
    _mkt_status = f"<span class='topbar-time'>● MARKET OPEN</span>" if market_open() else "<span class='topbar-time-closed'>● MARKET CLOSED</span>"
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
            <span class='topbar-badge'>NSE · INTRADAY</span>
            {_mkt_status}
        </div>
    </div>
    """, unsafe_allow_html=True)

    VERDICT_RANK = {"❌ AVOID":0,"⚠️ NEUTRAL":1,"⭐ WATCH":2,"⭐⭐ BUY":3,"⭐⭐⭐ STRONG BUY":4}
    min_rank     = VERDICT_RANK.get(min_verdict, 2)
    cache_key    = f"{interval}_{period}_{capital}_{risk_pct}_{','.join(sorted(selected_stocks))}"

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
        _vix_colors   = {'CALM':'#16a34a','NORMAL':'#16a34a','ELEVATED':'#d97706',
                         'HIGH':'#dc2626','EXTREME':'#7f1d1d','UNKNOWN':'#64748b'}
        _vix_advice   = {
            'CALM':     'VIX < 12 — Calm market, trend days likely ✅',
            'NORMAL':   'VIX 12–15 — Normal conditions ✅',
            'ELEVATED': 'VIX 15–18 — Elevated fear, reduce position size ⚠️',
            'HIGH':     'VIX 18–22 — High fear, trade only strongest signals ⚠️',
            'EXTREME':  'VIX > 22 — EXTREME FEAR — Avoid intraday entirely 🚫',
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

        if _vix_level == 'EXTREME':
            st.session_state['vix_extreme_warned'] = True

        # ── Progress UI ───────────────────────────────────
        _prog    = st.progress(0, text="Starting scan...")
        _stat    = st.empty()
        _sym_ph  = st.empty()

        for idx, symbol in enumerate(selected_stocks):
            pct        = int(((idx + 1) / total) * 100)
            elapsed    = int(time.time() - scan_start)
            eta        = int((elapsed / (idx + 1)) * (total - idx - 1)) if idx > 0 else 0
            eta_str    = f"{eta//60}m {eta%60}s" if eta >= 60 else f"{eta}s"
            sym_clean  = symbol.replace('.NS', '')

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

    # ── BREAKOUT SCREENER ─────────────────────────────────
    # Runs fast ORB rules — catches moves before indicators warm up
    # Best used 9:15–10:00 AM

    _orb_expander_label = "🚀 Opening Range Breakout Screener"
    _orb_results        = st.session_state.get('orb_results', [])
    _orb_time           = st.session_state.get('orb_scan_time', '')

    with st.expander(
        f"{_orb_expander_label}"
        f"{f' · ✅ {len(_orb_results)} breakouts found · {_orb_time}' if _orb_results else ' · Click to scan'}",
        expanded=len(_orb_results) > 0
    ):
        # Context info
        _now_ist  = ist_now()
        _mins_open= 0
        try:
            _mkt_st   = _now_ist.replace(hour=9, minute=15, second=0, microsecond=0)
            _mins_open= int((_now_ist - _mkt_st.astimezone(_now_ist.tzinfo)).total_seconds() / 60)
        except Exception:
            pass

        if market_open() and _mins_open <= 75:
            _orb_status_color = "#15803d"
            _orb_status = f"🟢 ACTIVE — {_mins_open} min into session · Best window: 9:15–10:30 AM"
        elif market_open():
            _orb_status_color = "#d97706"
            _orb_status = f"🟡 LATE — {_mins_open} min into session · Opening breakouts less reliable now"
        else:
            _orb_status_color = "#dc2626"
            _orb_status = "🔴 Market closed · Run at 9:20 AM for best results"

        st.markdown(f"""
        <div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;
                    padding:10px 16px;margin-bottom:12px'>
            <div style='font-size:12px;font-weight:700;color:{_orb_status_color}'>{_orb_status}</div>
            <div style='font-size:11px;color:#64748b;margin-top:4px'>
                Detects 5 patterns: Volume Explosion · Gap & Hold · VWAP Reclaim ·
                ORB High Break · Momentum Burst — before RSI/MACD/Supertrend warm up
            </div>
        </div>""", unsafe_allow_html=True)

        _ob1, _ob2 = st.columns([2, 1])
        with _ob1:
            _orb_universe = st.radio(
                "Scan universe",
                ["Current scan results", "Custom watchlist", "Full NSE 500"],
                horizontal=True, key="orb_universe",
                help="Current scan results is fastest — uses already-fetched data"
            )
        with _ob2:
            _run_orb = st.button("🚀 Run Breakout Scan", key="run_orb_scan",
                                  use_container_width=True, type="primary")

        if _run_orb:
            _orb_stocks = (
                [r['symbol'] for r in all_results] if _orb_universe == "Current scan results"
                else selected_stocks if _orb_universe == "Custom watchlist"
                else POPULAR_STOCKS
            )
            _kite_orb = get_kite_client()
            _port_orb = load_portfolio()

            with st.spinner(f"🚀 Scanning {len(_orb_stocks)} stocks for breakouts..."):
                _orb_results = run_breakout_screener(_orb_stocks, interval, _kite_orb, _port_orb)

            st.session_state['orb_results']   = _orb_results
            st.session_state['orb_scan_time'] = ist_now().strftime('%H:%M IST')
            st.rerun()

        # Show breakout results
        if _orb_results:
            st.markdown(f"<div class='section-header'>🚀 {len(_orb_results)} Breakouts Found</div>",
                        unsafe_allow_html=True)

            for _bo_r in _orb_results[:15]:
                _best  = _bo_r['best']
                _bc    = _best['color']
                _bbg   = _best['bg']
                _chg   = _bo_r['chg_pct']
                _chgc  = "#16a34a" if _chg >= 0 else "#dc2626"

                with st.container():
                    _bc1, _bc2, _bc3 = st.columns([3, 3, 2])

                    with _bc1:
                        st.markdown(f"""
                        <div style='background:{_bbg};border:2px solid {_bc}44;
                                    border-radius:14px;padding:16px 18px'>
                            <div style='display:flex;align-items:center;gap:10px'>
                                <span style='font-size:24px'>{_best['icon']}</span>
                                <div>
                                    <div style='font-size:16px;font-weight:800;color:#1a2035'>
                                        {_bo_r['sym_clean']}
                                    </div>
                                    <div style='font-size:11px;font-weight:700;color:{_bc}'>
                                        {_best['title']}
                                    </div>
                                </div>
                            </div>
                            <div style='display:flex;justify-content:space-between;
                                        margin-top:12px;flex-wrap:wrap;gap:8px'>
                                <div style='text-align:center'>
                                    <div style='font-size:10px;color:#94a3b8;font-weight:700'>PRICE</div>
                                    <div style='font-size:18px;font-weight:800;color:#1a2035;
                                                font-family:JetBrains Mono'>₹{_bo_r["price"]:,.2f}</div>
                                </div>
                                <div style='text-align:center'>
                                    <div style='font-size:10px;color:#94a3b8;font-weight:700'>CHANGE</div>
                                    <div style='font-size:18px;font-weight:800;color:{_chgc};
                                                font-family:JetBrains Mono'>{_chg:+.2f}%</div>
                                </div>
                                <div style='text-align:center'>
                                    <div style='font-size:10px;color:#94a3b8;font-weight:700'>VOLUME</div>
                                    <div style='font-size:18px;font-weight:800;color:#d97706;
                                                font-family:JetBrains Mono'>{_bo_r["vol_ratio"]}×</div>
                                </div>
                                <div style='text-align:center'>
                                    <div style='font-size:10px;color:#94a3b8;font-weight:700'>SCORE</div>
                                    <div style='font-size:18px;font-weight:800;color:{_bc};
                                                font-family:JetBrains Mono'>{_best["score"]}</div>
                                </div>
                            </div>
                        </div>""", unsafe_allow_html=True)

                    with _bc2:
                        # Show all patterns found
                        for _pat in _bo_r['breakouts']:
                            _pc = _pat['color']
                            _pbg = _pat['bg']
                            st.markdown(f"""
                            <div style='background:{_pbg};border-left:4px solid {_pc};
                                        border-radius:0 8px 8px 0;padding:8px 12px;
                                        margin-bottom:6px'>
                                <div style='font-size:12px;font-weight:700;color:{_pc}'>
                                    {_pat['icon']} {_pat['title']}
                                </div>
                                <div style='font-size:11px;color:#374151;margin-top:3px;
                                            line-height:1.4'>
                                    {_pat['msg'][_pat['msg'].find('|')+2:] if '|' in _pat['msg'] else _pat['msg']}
                                </div>
                            </div>""", unsafe_allow_html=True)

                    with _bc3:
                        st.markdown(f"""
                        <div style='background:#1a2035;border-radius:12px;padding:14px 16px;
                                    height:100%;display:flex;flex-direction:column;justify-content:space-between'>
                            <div>
                                <div style='font-size:10px;font-weight:700;color:#f59e0b;
                                            letter-spacing:1px;text-transform:uppercase'>Action</div>
                                <div style='font-size:12px;color:#e2e8f0;margin-top:6px;
                                            line-height:1.5'>{_best['action']}</div>
                            </div>
                            <div style='font-size:10px;color:#475569;margin-top:8px'>
                                Source: {"⚡ Kite" if _bo_r.get("src")=="kite" else "⏳ yfinance"}
                            </div>
                        </div>""", unsafe_allow_html=True)
                        if st.button(f"Analyse {_bo_r['sym_clean']}", key=f"orb_analyse_{_bo_r['sym_clean']}",
                                     use_container_width=True):
                            st.session_state['_focus_stock'] = _bo_r['sym_clean']
                            st.rerun()

                    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

        elif _run_orb:
            st.info("No breakout patterns found. Market may be slow or data unavailable.")
        else:
            st.markdown("""
            <div style='text-align:center;padding:20px;color:#94a3b8;font-size:13px'>
                Click <b>🚀 Run Breakout Scan</b> above to detect opening breakouts
            </div>""", unsafe_allow_html=True)

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
            Stocks that pass ALL 5 filters — Score ≥ 65 · VWAP Above ·
            Volume ≥ 1.5× · Liquidity Excellent/High · RS Positive
        </div>
    </div>""", unsafe_allow_html=True)

    # ── Apply all 5 filters ────────────────────────────────
    _shortlist = []
    _reject_reasons = {}

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

        # Skip WARMING_UP always
        # Skip PARTIAL only for fast timeframes (1min/3min) — not for 15min/60min
        # 60min: PARTIAL = 2+ candles = 2 hours of data = enough to trade
        if _wu == 'WARMING_UP':
            continue
        if _wu == 'PARTIAL' and _cpd >= 75:   # 75+ cpd = 1min or 3min
            continue

        # Filter 1 — Score ≥ 65
        if _score < 65:
            _reject_reasons[_sym] = f'Score {_score} < 65'
            continue

        # Filter 2 — VWAP Above
        if _vwap != 'ABOVE':
            _reject_reasons[_sym] = 'VWAP Below'
            continue

        # Filter 3 — Volume threshold (interval-aware)
        # 1min/3min: need 1.5× (intraday momentum)
        # 5min/15min: need 1.3× (slightly lower — candles aggregate more)
        # 60min: need 1.2× (hourly candles rarely spike 1.5×)
        _vol_threshold = (1.2 if _cpd <= 10 else (1.3 if _cpd <= 25 else 1.5))
        if _vol < _vol_threshold:
            _reject_reasons[_sym] = f'Vol {_vol:.1f}× < {_vol_threshold}×'
            continue

        # Filter 4 — Liquidity EXCELLENT or HIGH
        if _liq not in ['EXCELLENT', 'HIGH']:
            _reject_reasons[_sym] = f'Liquidity {_liq}'
            continue

        # Filter 5 — RS Positive
        if _rs < 0:
            _reject_reasons[_sym] = f'RS {_rs:.1f}% negative'
            continue

        # ── Passed all 5 filters ──
        # Bonus check: CPR narrow = +1 confidence star
        _cpr_ok = _cpr_w is not None and _cpr_w < 0.6

        # MTF alignment from session state if available
        _mtf_key  = f"mtf_{_r['symbol']}_{interval}"
        _mtf_data = st.session_state.get(_mtf_key, {})
        _align    = _mtf_data.get('alignment', 'UNKNOWN')

        _shortlist.append({
            'result':    _r,
            'sym':       _sym,
            'score':     _score,
            'verdict':   _verd,
            'price':     _r.get('price', 0),
            'chg_pct':   _r.get('change_pct', 0),
            'vol':       _vol,
            'rs':        _rs,
            'liq':       _liq,
            'vwap':      _vwap,
            'cpr_ok':    _cpr_ok,
            'cpr_w':     _cpr_w,
            'rsi':       _r.get('rsi', 50),
            'mtf_align': _align,
            'alerts':    _r.get('_alerts', []),
        })

    # Sort by score descending
    _shortlist.sort(key=lambda x: x['score'], reverse=True)

    # ── Show shortlist results ─────────────────────────────
    if not _shortlist:
        _now_ist2 = ist_now()
        _tm2      = _now_ist2.hour * 60 + _now_ist2.minute
        _reason   = (
            "Market opened less than 20 min ago — wait until 9:35 AM" if _tm2 < 575 else
            "VIX is HIGH/EXTREME — filters are strict today" if st.session_state.get('nifty_context',{}).get('vix_level') in ['HIGH','EXTREME'] else
            "Nifty is in BEAR state — most stocks failing VWAP/RS filters" if st.session_state.get('nifty_market_state') == 'BEAR' else
            "No stocks passed all 5 filters — do not force a trade today"
        )
        st.markdown(f"""
        <div style='background:#1a2035;border:2px solid #374151;border-radius:14px;
                    padding:24px;text-align:center;margin-bottom:16px'>
            <div style='font-size:32px;margin-bottom:10px'>🔍</div>
            <div style='font-size:17px;font-weight:800;color:#ffffff;margin-bottom:8px'>
                No stocks passed all 5 filters
            </div>
            <div style='font-size:13px;color:#f59e0b;font-weight:600;margin-bottom:16px'>
                {_reason}
            </div>
            <div style='font-size:12px;color:rgba(255,255,255,0.5);line-height:1.8'>
                ✅ This is the correct outcome — <b style='color:#34d399'>do not trade today</b><br>
                Forcing trades when filters fail = guaranteed losses<br>
                Come back tomorrow or wait for afternoon session
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
            f"✅ {len(_shortlist)} stock{'s' if len(_shortlist)>1 else ''} passed all filters</span>"
            f"<span style='font-size:12px;color:#64748b'>{_time_msg}</span>"
            f"</div>", unsafe_allow_html=True)

        # Stock cards
        for _rank, _sl in enumerate(_shortlist[:5], 1):
            _chg_clr  = "#16a34a" if _sl['chg_pct'] >= 0 else "#dc2626"
            _rs_clr   = "#16a34a" if _sl['rs'] >= 0 else "#dc2626"
            _vd       = _sl['verdict']
            _vd_bg    = {"⭐⭐⭐ STRONG BUY":"#dcfce7","⭐⭐ BUY":"#dbeafe"}.get(_vd,"#f8fafc")
            _vd_clr   = {"⭐⭐⭐ STRONG BUY":"#15803d","⭐⭐ BUY":"#1d4ed8"}.get(_vd,"#64748b")

            # MTF badge
            _mtf_badge = ""
            if _sl['mtf_align'] in ['STRONG_BULL','BULL']:
                _mtf_badge = "<span style='background:#dcfce7;color:#15803d;font-size:10px;font-weight:700;border-radius:6px;padding:2px 8px'>⏱ MTF BULL</span>"
            elif _sl['mtf_align'] == 'CONFLICTING':
                _mtf_badge = "<span style='background:#fef3c7;color:#92400e;font-size:10px;font-weight:700;border-radius:6px;padding:2px 8px'>⏱ CONFLICTING</span>"

            # CPR badge
            _cpr_badge = ""
            if _sl['cpr_ok']:
                _cpr_badge = "<span style='background:#f0f9ff;color:#0369a1;font-size:10px;font-weight:700;border-radius:6px;padding:2px 8px'>⚡ NARROW CPR</span>"

            # Alert badges
            _alert_types = set(a['type'] for a in _sl['alerts'])
            _alert_badge = ""
            if 'STRONG_BUY' in _alert_types:
                _alert_badge = "<span style='background:#dcfce7;color:#15803d;font-size:10px;font-weight:700;border-radius:6px;padding:2px 8px'>🚨 STRONG BUY ALERT</span>"
            elif 'BUY' in _alert_types:
                _alert_badge = "<span style='background:#dcfce7;color:#15803d;font-size:10px;font-weight:700;border-radius:6px;padding:2px 8px'>🔔 BUY ALERT</span>"

            # Rank colour
            _rank_bg = {1:"#f59e0b",2:"#94a3b8",3:"#b45309"}.get(_rank,"#e2e8f0")
            _rank_tc = {1:"#1a2035",2:"#ffffff",3:"#ffffff"}.get(_rank,"#64748b")

            _sc1, _sc2 = st.columns([5, 1])
            with _sc1:
                st.markdown(f"""
                <div style='background:#ffffff;border:1.5px solid #e8ecf3;border-radius:14px;
                            padding:16px 18px;margin-bottom:8px;
                            box-shadow:0 2px 8px rgba(0,0,0,0.04)'>
                    <div style='display:flex;align-items:flex-start;
                                justify-content:space-between;flex-wrap:wrap;gap:8px'>
                        <div style='display:flex;align-items:center;gap:12px'>
                            <div style='background:{_rank_bg};color:{_rank_tc};
                                        width:32px;height:32px;border-radius:50%;
                                        display:flex;align-items:center;justify-content:center;
                                        font-size:14px;font-weight:800;flex-shrink:0'>
                                {_rank}
                            </div>
                            <div>
                                <div style='display:flex;align-items:center;gap:8px;flex-wrap:wrap'>
                                    <span style='font-size:18px;font-weight:800;color:#1a2035'>
                                        {_sl['sym']}
                                    </span>
                                    <span style='background:{_vd_bg};color:{_vd_clr};
                                                 font-size:11px;font-weight:700;
                                                 border-radius:6px;padding:2px 10px'>
                                        {_vd}
                                    </span>
                                    {_mtf_badge} {_cpr_badge} {_alert_badge}
                                </div>
                                <div style='font-size:12px;color:#64748b;margin-top:4px'>
                                    Score <b style='color:#1a2035'>{_sl['score']}/100</b>
                                    &nbsp;·&nbsp; RSI {_sl['rsi']:.0f}
                                    &nbsp;·&nbsp; Vol <b>{_sl['vol']:.1f}×</b>
                                    &nbsp;·&nbsp; {_sl['liq']}
                                </div>
                            </div>
                        </div>
                        <div style='text-align:right'>
                            <div style='font-size:20px;font-weight:800;color:#1a2035;
                                        font-family:JetBrains Mono'>
                                ₹{_sl['price']:,.2f}
                            </div>
                            <div style='font-size:12px;font-weight:700;color:{_chg_clr}'>
                                {'+' if _sl['chg_pct']>=0 else ''}{_sl['chg_pct']:.2f}%
                                &nbsp;·&nbsp;
                                <span style='color:{_rs_clr}'>
                                    RS {'+' if _sl['rs']>=0 else ''}{_sl['rs']:.1f}%
                                </span>
                            </div>
                        </div>
                    </div>

                    <!-- 5 filter status badges -->
                    <div style='display:flex;gap:6px;margin-top:12px;flex-wrap:wrap'>
                        <span style='background:#dcfce7;color:#15803d;font-size:10px;
                                     font-weight:700;border-radius:4px;padding:3px 8px'>
                            ✅ Score {_sl['score']}
                        </span>
                        <span style='background:#dcfce7;color:#15803d;font-size:10px;
                                     font-weight:700;border-radius:4px;padding:3px 8px'>
                            ✅ VWAP Above
                        </span>
                        <span style='background:#dcfce7;color:#15803d;font-size:10px;
                                     font-weight:700;border-radius:4px;padding:3px 8px'>
                            ✅ Vol {_sl['vol']:.1f}×
                        </span>
                        <span style='background:#dcfce7;color:#15803d;font-size:10px;
                                     font-weight:700;border-radius:4px;padding:3px 8px'>
                            ✅ {_sl['liq']}
                        </span>
                        <span style='background:#dcfce7;color:#15803d;font-size:10px;
                                     font-weight:700;border-radius:4px;padding:3px 8px'>
                            ✅ RS +{_sl['rs']:.1f}%
                        </span>
                        {"<span style='background:#f0f9ff;color:#0369a1;font-size:10px;font-weight:700;border-radius:4px;padding:3px 8px'>⚡ Narrow CPR</span>" if _sl['cpr_ok'] else "<span style='background:#fef3c7;color:#92400e;font-size:10px;font-weight:700;border-radius:4px;padding:3px 8px'>〰 Moderate CPR</span>"}
                    </div>
                </div>""", unsafe_allow_html=True)

            with _sc2:
                st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
                if st.button(f"🔬 Analyse", key=f"shortlist_analyse_{_sl['sym']}",
                             use_container_width=True, type="primary"):
                    st.session_state['_focus_stock'] = _sl['sym']
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

        _vix_clr = {'CALM':'#16a34a','NORMAL':'#16a34a','ELEVATED':'#d97706',
                    'HIGH':'#dc2626','EXTREME':'#7f1d1d'}.get(_vix_level2, '#64748b')
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
            cur_price  = _live_px if is_open else _f(p.get('exit_price', entry))
            unreal_pl  = (cur_price - entry) * qty
            unreal_pct = (unreal_pl / actual_cost) * 100
            pl_color   = "#16a34a" if unreal_pl >= 0 else "#dc2626"
            pl_sign    = "+" if unreal_pl >= 0 else ""
            status_lbl = "OPEN" if is_open else "CLOSED"
            status_col = "#16a34a" if is_open else "#64748b"

            # Target hit highlights
            t1 = _f(p.get('t1', 0)); t2 = _f(p.get('t2', 0))
            t3 = _f(p.get('t3', 0)); t4 = _f(p.get('t4', 0))
            sl = _f(p.get('stop_loss', 0))

            def _target_style(tval):
                if tval <= 0: return "#f8fafc", "#94a3b8"
                if cur_price >= tval: return "#dcfce7", "#15803d"  # hit
                return "#f0fdf4", "#16a34a"                         # pending

            sl_bg = "#fef2f2" if cur_price <= sl and sl > 0 else "#fff5f5"
            sl_tc = "#dc2626"

            st.markdown(f"""
            <div style='background:#ffffff;border:1.5px solid #e8ecf3;border-radius:16px;
                        padding:18px 20px;margin-bottom:10px;
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
                        <div style='font-size:11px;color:#94a3b8'>Live Price</div>
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
                            {"🚨 HIT" if cur_price <= sl and sl > 0 else f"{((cur_price-sl)/entry*100):+.1f}%"}
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
#  PORTFOLIO PAGE
# ─────────────────────────────────────────────
if _show_portfolio:
    st.markdown("""
    <div class='topbar'>
        <div>
            <div class='topbar-title'>💼 Intraday Paper Portfolio</div>
            <div class='topbar-subtitle'>Track your intraday paper trades</div>
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
        st.markdown(f"<div style='font-size:11px;color:#94a3b8;padding:10px 0'>"
                    f"Live prices · Last refreshed: {_pf_last}</div>",
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
                    _ticker_s = _s + '.NS' if not _s.endswith('.NS') else _s
                    _ph = yf.Ticker(_ticker_s).history(period='1d', interval='1m')
                    _new_prices[_s] = float(_ph['Close'].iloc[-1]) if not _ph.empty else _f(p.get('entry', 0))
                except Exception:
                    _new_prices[_s] = _f(p.get('entry', 0))
        st.session_state['pf_live_prices']    = _new_prices
        st.session_state['pf_last_refresh']   = ist_now().strftime('%H:%M:%S IST')
        st.session_state['pf_last_refresh_ts']= time.time()
        _pf_prices = _new_prices
        _fetch_spinner.empty()

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

    st.markdown("<div class='section-header'>📋 Open Positions</div>", unsafe_allow_html=True)

    for p in open_pos:
        sym_c  = p.get('symbol', '')
        entry  = _f(p.get('entry', 0))
        qty    = int(_f(p.get('qty', 0)))
        sl     = _f(p.get('stop_loss', 0))
        t1     = _f(p.get('t1', 0)); t2 = _f(p.get('t2', 0))
        t3     = _f(p.get('t3', 0)); t4 = _f(p.get('t4', 0))
        actual = _f(p.get('actual_cost', _f(p.get('investment',0))))

        # Use cached live price
        cur        = _pf_prices.get(sym_c, entry)
        unreal     = (cur - entry) * qty
        unreal_pct = (unreal / actual * 100) if actual else 0
        pl_color   = "#16a34a" if unreal >= 0 else "#dc2626"
        pl_sign    = "+" if unreal >= 0 else ""

        # ── Auto-sell session keys ──
        _as_key_pct    = f"autosell_pct_{sym_c}"
        _as_key_type   = f"autosell_type_{sym_c}"
        _as_key_enabled= f"autosell_on_{sym_c}"

        # Defaults: take profit at T1 % gain, stop at SL %
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
                _auto_triggered = True
                _auto_reason    = f"Take Profit +{_as_tp_pct:.2f}% hit"
            elif _cur_pct <= -_as_sl_pct:
                _auto_triggered = True
                _auto_reason    = f"Stop Loss −{_as_sl_pct:.2f}% hit"

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

        # ── Position card ──
        with st.container():
            _h1, _h2 = st.columns([4, 1])

            with _h1:
                _t2_pct  = round((t2 - entry) / entry * 100, 2) if entry > 0 and t2 > 0 else 2.0
                _sl_pct  = round((entry - sl) / entry * 100, 2) if entry > 0 and sl > 0 else 0.5
                _cur_pct = round((cur - entry) / entry * 100, 2) if entry > 0 else 0
                _bar_rng = (_t2_pct + _sl_pct) or 1
                _bar_pct = min(100, max(0, int((_cur_pct + _sl_pct) / _bar_rng * 100)))
                _bar_clr = "#16a34a" if _cur_pct >= 0 else "#dc2626"
                _au_badge= ("<span style='font-size:10px;font-weight:700;color:#7c3aed;"
                            "background:#f5f3ff;border-radius:20px;padding:2px 8px'>AUTO</span>"
                            if _as_enabled else "")

                # Build target boxes
                _sl_hit = cur <= sl and sl > 0
                _t_html = (
                    "<div style='background:" + ("#fef2f2" if _sl_hit else "#fff5f5") +
                    ";border-radius:8px;padding:8px 12px;flex:1;min-width:80px;text-align:center'>"
                    "<div style='font-size:9px;color:#dc2626;font-weight:700'>SL</div>"
                    "<div style='font-size:13px;font-weight:800;color:#dc2626;font-family:JetBrains Mono'>₹" +
                    "{:,.2f}".format(sl) + "</div>"
                    "<div style='font-size:9px;color:#dc2626'>" +
                    ("HIT" if _sl_hit else "-{:.1f}%".format(_sl_pct)) + "</div></div>"
                )
                for _tv, _tlbl in [(t1,"T1"),(t2,"T2"),(t3,"T3"),(t4,"T4")]:
                    if _tv <= 0:
                        continue
                    _thit = cur >= _tv
                    _tbg  = "#dcfce7" if _thit else "#f0fdf4"
                    _tdsp = "HIT" if _thit else "+{:.1f}%".format(round((_tv-entry)/entry*100,1))
                    _t_html += (
                        "<div style='background:" + _tbg +
                        ";border-radius:8px;padding:8px 12px;flex:1;min-width:80px;text-align:center'>"
                        "<div style='font-size:9px;color:#15803d;font-weight:700'>" + _tlbl + "</div>"
                        "<div style='font-size:13px;font-weight:800;color:#15803d;font-family:JetBrains Mono'>₹" +
                        "{:,.2f}".format(_tv) + "</div>"
                        "<div style='font-size:9px;color:#15803d'>" + _tdsp + "</div></div>"
                    )

                _h = (
                    "<div class='port-card'>"
                    "<div style='display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:8px'>"
                    "<div><div style='display:flex;align-items:center;gap:8px'>"
                    "<span style='font-size:18px;font-weight:800;color:#1a2035'>" + str(sym_c) + "</span>"
                    "<span style='font-size:10px;font-weight:700;color:#16a34a;background:#dcfce7;"
                    "border-radius:20px;padding:2px 8px'>OPEN</span>" + _au_badge + "</div>"
                    "<div style='font-size:12px;color:#64748b;margin-top:3px'>" +
                    str(qty) + " shares · Entry ₹" + "{:,.2f}".format(entry) +
                    " · " + str(p.get("timeframe","INTRADAY")) +
                    " · " + str(p.get("date","")) + "</div></div>"
                    "<div style='text-align:right'>"
                    "<div style='font-size:11px;color:#94a3b8'>Live</div>"
                    "<div style='font-size:22px;font-weight:800;color:#1a2035;font-family:JetBrains Mono'>₹" +
                    "{:,.2f}".format(cur) + "</div>"
                    "<div style='font-size:16px;font-weight:800;color:" + pl_color + "'>" +
                    pl_sign + "₹" + "{:,.0f}".format(unreal) +
                    "<span style='font-size:13px;font-weight:700'> (" + pl_sign +
                    "{:.2f}%)</span></div></div></div>".format(unreal_pct) +
                    "<div style='margin:12px 0 4px'>"
                    "<div style='display:flex;justify-content:space-between;font-size:10px;color:#94a3b8;margin-bottom:3px'>"
                    "<span>SL -" + "{:.2f}%".format(_sl_pct) + "</span>"
                    "<span style='color:" + _bar_clr + ";font-weight:700'>" + pl_sign + "{:.2f}% now</span>".format(_cur_pct) +
                    "<span>T2 +" + "{:.2f}%".format(_t2_pct) + "</span></div>"
                    "<div style='background:#f1f5f9;border-radius:4px;height:8px;overflow:hidden'>"
                    "<div style='background:" + _bar_clr + ";height:8px;border-radius:4px;width:" +
                    str(_bar_pct) + "%;transition:width 0.4s'></div></div></div>"
                    "<div style='display:flex;gap:8px;margin-top:12px;flex-wrap:wrap'>" + _t_html + "</div>"
                    "<div style='margin-top:10px;padding:6px 10px;background:#fffbeb;"
                    "border-radius:6px;font-size:11px;color:#92400e'>"
                    "Square off before 3:20 PM IST</div></div>"
                )
                st.markdown(_h, unsafe_allow_html=True)


            with _h2:
                # Manual square off
                if st.button(f"✅ Square Off", key=f"sq_{sym_c}_{p.get('date','')}",
                             use_container_width=True):
                    for _p in port:
                        if _p.get('symbol') == sym_c and _p.get('status') == 'OPEN':
                            _p['status']     = 'CLOSED'
                            _p['exit_price'] = round(cur, 2)
                            _p['net_pl']     = round(unreal, 2)
                            _p['exit_date']  = ist_now().strftime('%d %b %Y %H:%M IST')
                            _p['exit_reason']= 'Manual'
                            break
                    save_portfolio(port)
                    st.session_state['paper_portfolio'] = port
                    st.success(f"✅ {sym_c} @ ₹{cur:,.2f} · {pl_sign}₹{unreal:,.0f} ({pl_sign}{unreal_pct:.2f}%)")
                    st.rerun()

                st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

                # ── Auto-sell toggle ──
                _new_as = st.toggle(
                    "🤖 Auto sell",
                    value=_as_enabled,
                    key=f"as_toggle_{sym_c}",
                    help="Auto square off when TP% or SL% is hit on next Refresh P&L"
                )
                st.session_state[_as_key_enabled] = _new_as

            # ── Auto-sell settings (shown when enabled) ──
            if st.session_state.get(_as_key_enabled, False):
                _asp1, _asp2 = st.columns(2)

                with _asp1:
                    _new_tp = st.number_input(
                        f"Take Profit % (T1 default: +{_default_tp_pct:.2f}%)",
                        min_value=0.1, max_value=10.0,
                        value=float(max(0.1, round(_as_tp_pct, 2))),
                        step=0.05, format="%.2f",
                        key=f"as_tp_{sym_c}",
                    )
                    st.session_state[_as_key_pct + '_tp'] = _new_tp
                    _tp_price = round(entry * (1 + _new_tp / 100), 2)
                    st.markdown(
                        f"<div style='font-size:11px;color:#7c3aed;margin-top:-8px'>"
                        f"Sell at ₹{_tp_price:,.2f} (+{_new_tp:.2f}%)</div>",
                        unsafe_allow_html=True)

                with _asp2:
                    _new_sl = st.number_input(
                        f"Stop Loss % (SL default: -{_default_sl_pct:.2f}%)",
                        min_value=0.1, max_value=5.0,
                        value=float(max(0.1, round(_as_sl_pct, 2))),
                        step=0.05, format="%.2f",
                        key=f"as_sl_{sym_c}",
                    )
                    st.session_state[_as_key_pct + '_sl'] = _new_sl
                    _sl_auto_price = round(entry * (1 - _new_sl / 100), 2)
                    st.markdown(
                        f"<div style='font-size:11px;color:#dc2626;margin-top:-8px'>"
                        f"Sell at ₹{_sl_auto_price:,.2f} (-{_new_sl:.2f}%)</div>",
                        unsafe_allow_html=True)

                _cur_pct_as = round((cur - entry) / entry * 100, 2) if entry > 0 else 0
                _status_clr = "#16a34a" if _cur_pct_as >= 0 else "#dc2626"
                st.markdown(
                    f"<div style='background:#f5f3ff;border:1px solid #c4b5fd;"
                    f"border-radius:8px;padding:8px 14px;margin-top:4px;font-size:11px;color:#7c3aed'>"
                    f"⚡ Active · Current P&L: "
                    f"<b style='color:{_status_clr}'>{'+' if _cur_pct_as >= 0 else ''}{_cur_pct_as:.2f}%</b>"
                    f" · Triggers on next <b>🔄 Refresh P&L</b>"
                    f"</div>",
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