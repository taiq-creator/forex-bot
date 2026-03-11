"""
╔══════════════════════════════════════════════════════════╗
║         FOREX AI BOT — Python + Streamlit                ║
║  Dữ liệu Twelve Data (~1s) | Không độ trễ               ║
║  Phân tích AI bằng Claude | Cảnh báo tín hiệu            ║
╚══════════════════════════════════════════════════════════╝

CÀI ĐẶT:
  pip install streamlit pandas plotly python-dotenv requests

CHẠY:
  streamlit run forex_bot_app.py

CẤU HÌNH API (tùy chọn):
  Tạo file .env cùng thư mục:
    GROQ_API_KEY=gsk_xxxxx         (miễn phí tại console.groq.com)
    TWELVE_DATA_API_KEY=xxxxx      (miễn phí tại twelvedata.com)
    TELEGRAM_BOT_TOKEN=xxxxx       (nếu muốn cảnh báo Telegram)
    TELEGRAM_CHAT_ID=xxxxx
"""

import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import os
import requests

# ── Đọc biến môi trường ──────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def _get_secret(key: str) -> str:
    """Đọc secret từ Streamlit Cloud hoặc .env local."""
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, "")

GROQ_API_KEY        = _get_secret("GROQ_API_KEY")
TWELVE_DATA_API_KEY = _get_secret("TWELVE_DATA_API_KEY")
TELEGRAM_BOT_TOKEN  = _get_secret("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID    = _get_secret("TELEGRAM_CHAT_ID")
FINAGE_API_KEY      = _get_secret("FINAGE_API_KEY")   # optional — finage.co free tier

# ── Cấu hình trang ───────────────────────────────────────
st.set_page_config(
    page_title="ForexAI Bot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="auto",
)

# ── CSS giao diện sáng ───────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Sora:wght@400;500;600;700;800&display=swap');

/* ══ RESET STREAMLIT ══ */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
.stApp { background: #070b14 !important; font-family: 'Sora', sans-serif; color: #e2e8f0; }
.block-container { padding: 0 0 80px 0 !important; max-width: 100% !important; }
#MainMenu, footer, header, .stDeployButton,
div[data-testid="stDecoration"],
div[data-testid="stStatusWidget"] { display: none !important; }

/* ── Xóa gap mặc định của Streamlit ── */
div[data-testid="stVerticalBlock"] > div { gap: 0 !important; }
div[data-testid="column"] { padding: 0 !important; gap: 0 !important; }
div[data-testid="stHorizontalBlock"] { gap: 6px !important; padding: 0 6px !important; }

/* ══ SIDEBAR ══ */
section[data-testid="stSidebar"] {
  background: #0c1121 !important;
  border-right: 1px solid #1a2540 !important;
}
section[data-testid="stSidebar"] * { color: #7c93b8 !important; }
section[data-testid="stSidebar"] h3 { color: #e2e8f0 !important; font-size: 13px !important; font-weight: 700 !important; }
section[data-testid="stSidebar"] .stSelectbox > div > div {
  background: #131d35 !important; border: 1px solid #1f3052 !important;
  color: #e2e8f0 !important; border-radius: 8px !important; font-size: 13px !important;
}
section[data-testid="stSidebar"] .stCheckbox label { color: #7c93b8 !important; font-size: 13px !important; }
section[data-testid="stSidebar"] .stButton > button {
  background: #1d4ed8 !important; border: none !important; border-radius: 8px !important;
  color: #fff !important; font-weight: 700 !important; font-size: 13px !important; padding: 10px !important;
}

/* ══ SELECTBOX MAIN PAGE ══ */
div[data-testid="stSelectbox"] > div > div {
  background: #0f172a !important; border: 1px solid #1e3a5f !important;
  color: #e2e8f0 !important; border-radius: 10px !important;
  font-size: 14px !important; font-weight: 600 !important;
  font-family: 'Sora', sans-serif !important;
}
div[data-testid="stSelectbox"] { padding: 0 !important; margin: 0 !important; }
div.row-widget.stSelectbox { padding: 3px 0 !important; }

/* ══ METRIC — ẩn hết, dùng HTML riêng ══ */
div[data-testid="metric-container"] { display: none !important; }

/* ══ EXPANDER ══ */
details { background: #0c1121 !important; border: 1px solid #1a2540 !important; border-radius: 8px !important; margin: 4px 6px !important; overflow: hidden; }
details summary { font-size: 12px !important; color: #64748b !important; font-weight: 600 !important; padding: 10px 14px !important; cursor: pointer; list-style: none; }
details summary::-webkit-details-marker { display: none; }
details[open] summary { border-bottom: 1px solid #1a2540; color: #94a3b8 !important; }
.streamlit-expanderContent { padding: 8px !important; }

/* ══ PLOTLY ══ */
div[data-testid="stPlotlyChart"] { padding: 0 6px !important; }
.js-plotly-plot { border-radius: 10px !important; overflow: hidden; border: 1px solid #1a2540 !important; }
.plotly .bg { fill: #0c1121 !important; }

/* ══ DATAFRAME ══ */
div[data-testid="stDataFrame"] { padding: 0 6px !important; }
.stDataFrame { border: 1px solid #1a2540; border-radius: 8px; overflow: hidden; }

/* ══ SCROLLBAR ══ */
::-webkit-scrollbar { width: 3px; height: 3px; }
::-webkit-scrollbar-track { background: #070b14; }
::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 2px; }

/* ══ SPINNER ══ */
.stSpinner { color: #3b82f6 !important; }

/* ══ ALERT / INFO ══ */
.stAlert { margin: 4px 6px !important; border-radius: 8px !important; font-size: 12px !important; }

/* ══ APP SHELL CLASSES ══ */

/* Header bar */
.fx-header {
  background: linear-gradient(90deg, #0f1e40 0%, #0c1832 100%);
  border-bottom: 1px solid #1a2d50;
  padding: 10px 14px;
  display: flex; align-items: center; gap: 10px;
}
.fx-header-logo {
  font-family: 'Sora', sans-serif;
  font-size: 17px; font-weight: 800; color: #fff; letter-spacing: -0.5px;
  flex-shrink: 0;
}
.fx-header-logo span { color: #60a5fa; font-weight: 500; }
.fx-live-dot {
  width: 6px; height: 6px; background: #22c55e; border-radius: 50%;
  box-shadow: 0 0 6px #22c55e; flex-shrink: 0;
  animation: pulse 1.4s ease-in-out infinite;
}
@keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.5;transform:scale(0.8)} }
.fx-badge {
  font-family: 'JetBrains Mono', monospace;
  font-size: 9px; color: #22c55e; letter-spacing: 1px; font-weight: 700;
  background: rgba(34,197,94,.1); border: 1px solid rgba(34,197,94,.25);
  padding: 2px 7px; border-radius: 20px;
}
.fx-header-right { margin-left: auto; font-size: 10px; color: #3b5280; font-family: 'JetBrains Mono', monospace; }

/* Section label */
.fx-sec {
  font-size: 9px; color: #334155; font-weight: 700;
  text-transform: uppercase; letter-spacing: 1.5px;
  padding: 10px 14px 4px;
  display: flex; align-items: center; gap: 6px;
}
.fx-sec::after { content:''; flex:1; height:1px; background:#0f1e35; }

/* Price card */
.fx-price-card {
  margin: 0 6px 4px;
  background: #0c1121; border: 1px solid #1a2d50;
  border-radius: 12px; padding: 12px 14px;
  display: flex; justify-content: space-between; align-items: center;
}
.fx-price { font-family: 'JetBrains Mono', monospace; font-size: 28px; font-weight: 700; color: #f8fafc; line-height: 1; }
.fx-pair  { font-size: 10px; color: #475569; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; }
.fx-chg.up   { font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #22c55e; font-weight: 600; margin-top: 4px; }
.fx-chg.down { font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #ef4444; font-weight: 600; margin-top: 4px; }
.fx-ohlc { text-align: right; }
.fx-ohlc-row { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #475569; margin-bottom: 2px; }
.fx-ohlc-row b { color: #94a3b8; }
.fx-rt-badge { font-size: 9px; font-family: 'JetBrains Mono', monospace; color: #22c55e; background: rgba(34,197,94,.08); border: 1px solid rgba(34,197,94,.2); padding: 2px 6px; border-radius: 4px; display: inline-block; margin-bottom: 6px; }

/* Signal cards */
.fx-sig-row { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin: 0 6px; }
.fx-sig {
  border-radius: 10px; padding: 11px 10px; position: relative; overflow: hidden;
}
.fx-sig.buy  { background: #031a0e; border: 1px solid #16a34a; }
.fx-sig.sell { background: #1a0303; border: 1px solid #dc2626; }
.fx-sig.neut { background: #161106; border: 1px solid #b45309; }
.fx-sig-lbl  { font-size: 8px; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase; color: #334155; margin-bottom: 5px; font-family: 'JetBrains Mono', monospace; }
.fx-sig-act  { font-size: 19px; font-weight: 800; line-height: 1; letter-spacing: -0.5px; }
.fx-sig-act.buy  { color: #4ade80; }
.fx-sig-act.sell { color: #f87171; }
.fx-sig-act.neut { color: #fbbf24; }
.fx-sig-conf { font-size: 10px; font-family: 'JetBrains Mono', monospace; margin-top: 3px; }
.fx-sig-conf.buy  { color: #86efac; }
.fx-sig-conf.sell { color: #fca5a5; }
.fx-sig-conf.neut { color: #fde68a; }

/* TP/SL inside signal */
.fx-tpsl { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 3px; margin-top: 8px; }
.fx-tpsl-item { border-radius: 5px; padding: 4px; text-align: center; }
.fx-tpsl-item.tp    { background: rgba(22,163,74,.15); }
.fx-tpsl-item.entry { background: rgba(30,58,138,.2); }
.fx-tpsl-item.sl    { background: rgba(220,38,38,.15); }
.fx-tpsl-lbl { font-size: 8px; color: #475569; text-transform: uppercase; letter-spacing: 0.3px; font-weight: 700; }
.fx-tpsl-val { font-family: 'JetBrains Mono', monospace; font-size: 10px; font-weight: 700; margin-top: 1px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.fx-tpsl-item.tp    .fx-tpsl-val { color: #4ade80; }
.fx-tpsl-item.entry .fx-tpsl-val { color: #93c5fd; }
.fx-tpsl-item.sl    .fx-tpsl-val { color: #f87171; }

/* Indicator chips */
.fx-ind-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 4px; margin: 0 6px; }
.fx-ind-chip {
  background: #0c1121; border: 1px solid #0f1e35;
  border-radius: 8px; padding: 7px 9px;
  display: flex; justify-content: space-between; align-items: center;
}
.fx-ind-chip.bull { border-left: 2px solid #16a34a; }
.fx-ind-chip.bear { border-left: 2px solid #dc2626; }
.fx-ind-chip.neut { border-left: 2px solid #b45309; }
.fx-ind-name { font-size: 9px; color: #334155; font-weight: 700; text-transform: uppercase; letter-spacing: 0.3px; }
.fx-ind-right { text-align: right; }
.fx-ind-val  { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #94a3b8; font-weight: 600; }
.fx-ind-sig  { font-size: 10px; color: #475569; margin-top: 1px; font-size: 9px; }

/* Live bar */
.fx-livebar {
  display: flex; align-items: center; gap: 7px;
  padding: 6px 14px; background: #040d08; border-bottom: 1px solid #0d2016;
}
.fx-livebar-dot { width: 5px; height: 5px; background: #22c55e; border-radius: 50%; animation: pulse 1.4s infinite; flex-shrink: 0; }
.fx-livebar-txt { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #22c55e; font-weight: 600; }
.fx-livebar-time { margin-left: auto; font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #1e3a5f; }

/* Sentiment bar */
.fx-sent { margin: 0 6px; background: #0c1121; border: 1px solid #1a2d50; border-radius: 10px; overflow: hidden; }
.fx-sent-top { padding: 11px 13px; display: flex; justify-content: space-between; align-items: center; }
.fx-sent-title { font-size: 8px; color: #334155; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 3px; }
.fx-sent-act  { font-size: 17px; font-weight: 800; font-family: 'JetBrains Mono', monospace; }
.fx-sent-sub  { font-size: 9px; color: #334155; font-family: 'JetBrains Mono', monospace; margin-top: 2px; }
.fx-sent-counts { display: grid; grid-template-columns: 1fr 1fr 1fr; border-top: 1px solid #0f1e35; }
.fx-sent-cnt { padding: 8px 4px; text-align: center; border-right: 1px solid #0f1e35; }
.fx-sent-cnt:last-child { border-right: none; }
.fx-sent-cnt-n { font-family: 'JetBrains Mono', monospace; font-size: 16px; font-weight: 700; }
.fx-sent-cnt-l { font-size: 8px; color: #334155; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; margin-top: 1px; }

/* News item */
.fx-news { margin: 4px 6px; background: #0c1121; border: 1px solid #0f1e35; border-radius: 8px; padding: 9px 11px; }
.fx-news.bull { border-left: 2px solid #16a34a; }
.fx-news.bear { border-left: 2px solid #dc2626; }
.fx-news.neut { border-left: 2px solid #b45309; }
.fx-news-title { font-size: 12px; color: #94a3b8; line-height: 1.35; font-weight: 500; }
.fx-news-foot  { display: flex; justify-content: space-between; align-items: center; margin-top: 5px; }
.fx-news-time  { font-size: 9px; color: #1e3a5f; font-family: 'JetBrains Mono', monospace; }
.fx-news-tag   { font-size: 8px; font-weight: 700; font-family: 'JetBrains Mono', monospace; padding: 2px 5px; border-radius: 3px; }
.fx-news-tag.bull { background: #031a0e; color: #4ade80; }
.fx-news-tag.bear { background: #1a0303; color: #f87171; }
.fx-news-tag.neut { background: #161106; color: #fbbf24; }

/* ══ MULTI-TIMEFRAME TABLE ══ */
.mtf-wrap {
  margin: 0 6px;
  background: #0a0e1a;
  border: 1px solid #0f1e35;
  border-radius: 10px;
  overflow: hidden;
}
.mtf-table {
  width: 100%;
  border-collapse: collapse;
  font-family: 'JetBrains Mono', monospace;
}
.mtf-table th {
  background: #0c1121;
  color: #334155;
  font-size: 9px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.8px;
  padding: 7px 6px;
  text-align: center;
  border-bottom: 1px solid #0f1e35;
  border-right: 1px solid #0f1e35;
  white-space: nowrap;
}
.mtf-table th:first-child { text-align: left; padding-left: 10px; }
.mtf-table td {
  padding: 6px 4px;
  text-align: center;
  border-bottom: 1px solid #080d16;
  border-right: 1px solid #080d16;
  font-size: 10px;
  font-weight: 700;
  white-space: nowrap;
}
.mtf-table td:first-child {
  text-align: left;
  padding-left: 10px;
  color: #475569;
  font-size: 9px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  background: #0c1121;
  border-right: 1px solid #0f1e35;
}
.mtf-table tr:last-child td { border-bottom: none; }
.mtf-table .c-buy     { background: #031a0e; color: #4ade80; }
.mtf-table .c-sell    { background: #1a0303; color: #f87171; }
.mtf-table .c-neutral { background: #0f1000; color: #ca8a04; }
.mtf-table .c-na      { background: #0c1121; color: #1e3a5f; }
.mtf-table .c-sum-buy  { background: #052e16; color: #4ade80; font-size: 12px; }
.mtf-table .c-sum-sell { background: #2d0a0a; color: #f87171; font-size: 12px; }
.mtf-table .c-sum-neut { background: #1c1400; color: #fbbf24; font-size: 12px; }
.mtf-table .tf-header-buy  { color: #4ade80 !important; border-bottom: 2px solid #16a34a !important; }
.mtf-table .tf-header-sell { color: #f87171 !important; border-bottom: 2px solid #dc2626 !important; }
.mtf-table .tf-header-neut { color: #fbbf24 !important; border-bottom: 2px solid #b45309 !important; }

/* AI box */
.fx-ai-box {
  margin: 0 6px;
  background: #0a1628; border: 1px solid #1e3a5f; border-left: 3px solid #3b82f6;
  border-radius: 10px; padding: 14px; font-size: 13px; line-height: 1.7; color: #94a3b8;
}
.fx-ai-title { font-size: 9px; font-weight: 700; letter-spacing: 1.5px; color: #3b82f6; text-transform: uppercase; font-family: 'JetBrains Mono', monospace; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
#  CÁC HÀM DỮ LIỆU
# ════════════════════════════════════════════════════════

# symbol: (twelve_data_symbol, yf_unused, type)
PAIRS = {
    # ── Forex Majors ──────────────────────────
    "🇪🇺 EUR/USD": ("EUR/USD",  "EURUSD=X",  "forex"),
    "🇬🇧 GBP/USD": ("GBP/USD",  "GBPUSD=X",  "forex"),
    "🇯🇵 USD/JPY": ("USD/JPY",  "JPY=X",     "forex"),
    "🇦🇺 AUD/USD": ("AUD/USD",  "AUDUSD=X",  "forex"),
    "🇨🇭 USD/CHF": ("USD/CHF",  "CHF=X",     "forex"),
    "🇨🇦 USD/CAD": ("USD/CAD",  "CAD=X",     "forex"),
    "🇳🇿 NZD/USD": ("NZD/USD",  "NZDUSD=X",  "forex"),
    # ── Forex Crosses ─────────────────────────
    "🇪🇺 EUR/GBP": ("EUR/GBP",  "EURGBP=X",  "forex"),
    "🇪🇺 EUR/JPY": ("EUR/JPY",  "EURJPY=X",  "forex"),
    "🇬🇧 GBP/JPY": ("GBP/JPY",  "GBPJPY=X",  "forex"),
    "🇦🇺 AUD/JPY": ("AUD/JPY",  "AUDJPY=X",  "forex"),
    "🇪🇺 EUR/AUD": ("EUR/AUD",  "EURAUD=X",  "forex"),
    # ── Kim loại quý ──────────────────────────
    "🥇 Vàng (XAU/USD)":   ("XAU/USD", "GC=F",    "commodity"),
    "🥈 Bạc (XAG/USD)":    ("XAG/USD", "SI=F",    "commodity"),
    # ── Năng lượng ────────────────────────────
    "🛢️ Dầu WTI (CL)":     ("WTI/USD", "CL=F",    "commodity"),
    "🛢️ Dầu Brent (BZ)":   ("BRENT/USD","BZ=F",   "commodity"),
    # ── Crypto ────────────────────────────────
    "₿  Bitcoin (BTC/USD)":  ("BTC/USD", "BTC-USD", "crypto"),
    "Ξ  Ethereum (ETH/USD)": ("ETH/USD", "ETH-USD", "crypto"),
}

# ── WebSocket symbol mapping ──
# Binance WS: wss://stream.binance.com:9443/ws/<symbol>@trade
# Finage  WS: wss://forex.finage.co.uk:8443/forex?apikey=<key>
WS_SYMBOLS = {
    # Crypto → Binance WS (free, no key)
    "₿  Bitcoin (BTC/USD)":   ("binance", "btcusdt"),
    "Ξ  Ethereum (ETH/USD)":  ("binance", "ethusdt"),
    # Forex → Finage WS (free key needed) hoặc polling fallback
    "🇪🇺 EUR/USD": ("finage", "EUR/USD"),
    "🇬🇧 GBP/USD": ("finage", "GBP/USD"),
    "🇯🇵 USD/JPY": ("finage", "USD/JPY"),
    "🇦🇺 AUD/USD": ("finage", "AUD/USD"),
    "🇨🇭 USD/CHF": ("finage", "USD/CHF"),
    "🇨🇦 USD/CAD": ("finage", "USD/CAD"),
    "🇳🇿 NZD/USD": ("finage", "NZD/USD"),
    "🇪🇺 EUR/GBP": ("finage", "EUR/GBP"),
    "🇪🇺 EUR/JPY": ("finage", "EUR/JPY"),
    "🇬🇧 GBP/JPY": ("finage", "GBP/JPY"),
    "🇦🇺 AUD/JPY": ("finage", "AUD/JPY"),
    "🇪🇺 EUR/AUD": ("finage", "EUR/AUD"),
    # Commodity → polling (không có WS free)
    "🥇 Vàng (XAU/USD)":   ("finage", "XAU/USD"),
    "🥈 Bạc (XAG/USD)":    ("finage", "XAG/USD"),
    "🛢️ Dầu WTI (CL)":     ("poll", "WTI/USD"),
    "🛢️ Dầu Brent (BZ)":   ("poll", "BRENT/USD"),
}

# interval: (twelve_data_interval, yf_interval, yf_period, outputsize)
TIMEFRAMES = {
    "M5 (5 phút)":   ("5min",  "5m",   "2d",   200),
    "M15 (15 phút)": ("15min", "15m",  "5d",   200),
    "M30 (30 phút)": ("30min", "30m",  "7d",   200),
    "H1 (1 giờ)":    ("1h",   "1h",   "10d",  200),
    "H4 (4 giờ)":    ("4h",   "1h",   "30d",  200),
    "D1 (Ngày)":     ("1day", "1d",   "180d", 200),
}

# Multi-Timeframe: (yf_interval, yf_period, label_short)
MTF_LIST = [
    ("5m",  "2d",   "5M"),
    ("15m", "5d",   "15M"),
    ("30m", "7d",   "30M"),
    ("1h",  "10d",  "1H"),
    ("1h",  "30d",  "4H"),   # resample từ 1h
    ("1d",  "180d", "1D"),
]

@st.cache_data(ttl=15)
def fetch_mtf_signals(yf_ticker: str) -> list:
    """
    Tính tín hiệu BUY/SELL/NEUTRAL cho 6 khung giờ cùng lúc.
    Trả về list dict: [{tf, action, score, indicators}, ...]
    """
    results = []
    for yf_int, yf_period, tf_label in MTF_LIST:
        try:
            df = yf.download(yf_ticker, interval=yf_int, period=yf_period,
                             progress=False, auto_adjust=True)
            if df.empty or len(df) < 30:
                results.append({"tf": tf_label, "action": "N/A", "score": 0,
                                 "indicators": {}, "error": True})
                continue
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            df = df[["Open","High","Low","Close","Volume"]].dropna()

            # Resample 4H từ 1H
            if tf_label == "4H":
                df = df.resample("4h").agg({
                    "Open":"first","High":"max","Low":"min",
                    "Close":"last","Volume":"sum"
                }).dropna()

            if len(df) < 30:
                results.append({"tf": tf_label, "action": "N/A", "score": 0,
                                 "indicators": {}, "error": True})
                continue

            df = add_indicators(df)
            last = df.iloc[-1]
            prev = df.iloc[-2]

            # Tính từng chỉ báo
            inds = {}

            # RSI
            rsi = float(last["RSI"])
            if rsi < 30:   inds["RSI"] = ("BUY",  "%.1f" % rsi)
            elif rsi > 70: inds["RSI"] = ("SELL", "%.1f" % rsi)
            else:          inds["RSI"] = ("NEUTRAL", "%.1f" % rsi)

            # MACD
            if last["MACD"] > last["MACD_signal"] and prev["MACD"] <= prev["MACD_signal"]:
                inds["MACD"] = ("BUY", "Cross↑")
            elif last["MACD"] < last["MACD_signal"] and prev["MACD"] >= prev["MACD_signal"]:
                inds["MACD"] = ("SELL", "Cross↓")
            elif last["MACD"] > last["MACD_signal"]:
                inds["MACD"] = ("BUY", "↑Sig")
            else:
                inds["MACD"] = ("SELL", "↓Sig")

            # EMA 20/50
            if last["EMA_20"] > last["EMA_50"]:
                inds["EMA"] = ("BUY", "20↑50")
            else:
                inds["EMA"] = ("SELL", "20↓50")

            # EMA 200
            c = float(last["Close"])
            if c > last["EMA_200"]:
                inds["EMA200"] = ("BUY", "↑200")
            else:
                inds["EMA200"] = ("SELL", "↓200")

            # Bollinger
            if c < float(last["BB_lower"]):
                inds["BB"] = ("BUY", "↓Band")
            elif c > float(last["BB_upper"]):
                inds["BB"] = ("SELL", "↑Band")
            else:
                inds["BB"] = ("NEUTRAL", "Mid")

            # Stochastic
            sk = float(last["Stoch_K"])
            sd = float(last["Stoch_D"])
            if sk < 20:   inds["Stoch"] = ("BUY",  "%.0f" % sk)
            elif sk > 80: inds["Stoch"] = ("SELL", "%.0f" % sk)
            else:         inds["Stoch"] = ("NEUTRAL", "%.0f" % sk)

            # Tổng điểm
            score = 0
            score_map = {"BUY": 1, "SELL": -1, "NEUTRAL": 0}
            for v in inds.values():
                score += score_map.get(v[0], 0)

            if score >= 3:    action = "BUY"
            elif score <= -3: action = "SELL"
            else:             action = "NEUTRAL"

            results.append({
                "tf": tf_label, "action": action,
                "score": score, "indicators": inds,
                "error": False,
                "rsi": rsi, "close": c,
            })
        except Exception as e:
            results.append({"tf": tf_label, "action": "N/A", "score": 0,
                             "indicators": {}, "error": True})
    return results


@st.cache_data(ttl=10)
def fetch_ohlcv_yahoo(yf_ticker: str, yf_interval: str, yf_period: str) -> pd.DataFrame:
    """Tải OHLCV từ Yahoo Finance — miễn phí, không giới hạn."""
    try:
        df = yf.download(yf_ticker, interval=yf_interval, period=yf_period,
                         progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return df[["Open","High","Low","Close","Volume"]].dropna()
    except Exception as e:
        return pd.DataFrame()


def get_ws_component(pair_name: str) -> str:
    """
    WebSocket price widget chạy trong iframe — không bị Streamlit rerun reset.
    - Crypto  : Binance WSS  → tick < 50ms
    - Forex   : Finage WSS   (nếu có key) HOẶC polling open.er-api 1s
    - Hiển thị: giá lớn, flash xanh/đỏ, H/L/ticks/latency
    """
    ws_info  = WS_SYMBOLS.get(pair_name, ("poll", ""))
    ws_type, ws_sym = ws_info
    td_sym   = PAIRS.get(pair_name, ("",))[0]

    finage_key = ""
    try:    finage_key = st.secrets.get("FINAGE_API_KEY", "")
    except: pass

    if "JPY" in td_sym:    dec = 3
    elif PAIRS.get(pair_name,("","","x"))[2] == "crypto": dec = 2
    else: dec = 5

    if ws_type == "binance":
        ws_js = (
            "var WS_URL='wss://stream.binance.com:9443/ws/" + ws_sym + "@trade';"
            "function buildWS(){"
            "  ws=new WebSocket(WS_URL);"
            "  ws.onopen=function(){setStatus('live');};"
            "  ws.onmessage=function(e){var d=JSON.parse(e.data);if(d.p)updatePrice(parseFloat(d.p));};"
            "  ws.onerror=function(){setStatus('error');};"
            "  ws.onclose=function(){setStatus('reconnect');setTimeout(buildWS,2000);};"
            "}"
            "buildWS();"
            "setInterval(function(){if(ws&&ws.readyState===1)ws.send(JSON.stringify({method:'ping'}));},20000);"
        )
    elif ws_type == "finage" and finage_key:
        ws_js = (
            "var WS_URL='wss://forex.finage.co.uk:8443/forex?apikey=" + finage_key + "';"
            "function buildWS(){"
            "  ws=new WebSocket(WS_URL);"
            "  ws.onopen=function(){ws.send(JSON.stringify({action:'subscribe',symbols:'" + ws_sym + "'}));setStatus('live');};"
            "  ws.onmessage=function(e){var d=JSON.parse(e.data);if(d.a)updatePrice(parseFloat(d.a));};"
            "  ws.onerror=function(){setStatus('error');};"
            "  ws.onclose=function(){setStatus('reconnect');setTimeout(buildWS,3000);};"
            "}"
            "buildWS();"
        )
    else:
        base, quote = (td_sym.split("/") + ["USD"])[:2]
        ws_js = (
            "function poll(){"
            "  fetch('https://open.er-api.com/v6/latest/" + base + "')"
            "  .then(function(r){return r.json();})"
            "  .then(function(d){if(d.result==='success'&&d.rates&&d.rates['" + quote + "'])updatePrice(d.rates['" + quote + "']);}).catch(function(){setStatus('error');});"
            "}"
            "function buildWS(){poll();setInterval(poll,1000);}"
            "buildWS();"
        )

    pair_label = (pair_name
        .replace("🇪🇺","").replace("🇬🇧","").replace("🇯🇵","")
        .replace("🇦🇺","").replace("🇨🇭","").replace("🇨🇦","")
        .replace("🇳🇿","").replace("🥇","").replace("🥈","")
        .replace("🛢️","").replace("₿ ","").replace("Ξ ","").strip()
    )

    src_label = "Binance WS" if ws_type=="binance" else ("Finage WS" if (ws_type=="finage" and finage_key) else "Polling 1s")

    html = (
        "<!DOCTYPE html><html><head>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<link href='https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Sora:wght@600;700;800&display=swap' rel='stylesheet'>"
        "<style>"
        "*{margin:0;padding:0;box-sizing:border-box;}"
        "body{background:#0a0e1a;font-family:'Sora',sans-serif;padding:10px 12px 8px;}"
        ".top{display:flex;justify-content:space-between;align-items:flex-start;}"
        ".left{}"
        ".right{text-align:right;padding-top:2px;}"
        "#pair-lbl{font-size:10px;color:#334155;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:3px;}"
        "#price{font-size:34px;font-weight:800;color:#f8fafc;letter-spacing:-1px;line-height:1;font-family:'JetBrains Mono',monospace;transition:color 0.12s;}"
        "#price.up{color:#22c55e;}"
        "#price.dn{color:#ef4444;}"
        "#chg{font-size:13px;font-weight:600;color:#475569;margin-top:5px;font-family:'JetBrains Mono',monospace;}"
        "#chg.up{color:#22c55e;}"
        "#chg.dn{color:#ef4444;}"
        ".ohlc-row{font-size:11px;color:#334155;font-family:'JetBrains Mono',monospace;margin-bottom:2px;}"
        ".ohlc-row b{color:#64748b;}"
        ".bottom{display:flex;align-items:center;gap:6px;margin-top:8px;border-top:1px solid #0f1e35;padding-top:6px;}"
        "#ws-dot{width:6px;height:6px;border-radius:50%;background:#334155;flex-shrink:0;transition:background .3s;}"
        "#ws-dot.live{background:#22c55e;box-shadow:0 0 5px #22c55e;animation:bl 1.2s infinite;}"
        "#ws-dot.error{background:#ef4444;}"
        "#ws-dot.rc{background:#f59e0b;}"
        "@keyframes bl{0%,100%{opacity:1}50%{opacity:.4}}"
        "#ws-lbl{font-size:9px;color:#334155;font-family:'JetBrains Mono',monospace;}"
        "#ts{margin-left:auto;font-size:9px;color:#1e3a5f;font-family:'JetBrains Mono',monospace;}"
        ".stats{display:flex;gap:12px;}"
        ".stat-item{text-align:center;}"
        ".stat-lbl{font-size:8px;color:#1e3a5f;text-transform:uppercase;letter-spacing:.3px;}"
        ".stat-val{font-size:10px;color:#475569;font-family:'JetBrains Mono',monospace;font-weight:600;}"
        "</style></head><body>"
        "<div class='top'>"
        "  <div class='left'>"
        "    <div id='pair-lbl'>" + pair_label + "</div>"
        "    <div id='price'>---</div>"
        "    <div id='chg'>-- ---%</div>"
        "  </div>"
        "  <div class='right'>"
        "    <div class='ohlc-row'><b>H</b> <span id='hv'>---</span></div>"
        "    <div class='ohlc-row'><b>L</b> <span id='lv'>---</span></div>"
        "    <div class='ohlc-row'><b>Ticks</b> <span id='tc'>0</span></div>"
        "  </div>"
        "</div>"
        "<div class='bottom'>"
        "  <div id='ws-dot'></div>"
        "  <span id='ws-lbl'>Connecting... (" + src_label + ")</span>"
        "  <span id='ts'>--:--:--.---</span>"
        "</div>"
        "<script>(function(){"
        "var ws,op=null,lp=null,hi=null,lo=null,tc=0,DEC=" + str(dec) + ";"
        "function fmt(p){return p.toLocaleString('en-US',{minimumFractionDigits:DEC,maximumFractionDigits:DEC});}"
        "function updatePrice(p){"
        "  if(op===null)op=p;"
        "  if(hi===null||p>hi){hi=p;document.getElementById('hv').textContent=fmt(hi);}"
        "  if(lo===null||p<lo){lo=p;document.getElementById('lv').textContent=fmt(lo);}"
        "  tc++;document.getElementById('tc').textContent=tc;"
        "  var el=document.getElementById('price');"
        "  var ce=document.getElementById('chg');"
        "  if(lp!==null){"
        "    el.classList.remove('up','dn');"
        "    void el.offsetWidth;"
        "    el.classList.add(p>lp?'up':'dn');"
        "    setTimeout(function(){el.classList.remove('up','dn');},180);"
        "    var chg=(p-op)/op*100;"
        "    var diff=p-op;"
        "    ce.className=chg>=0?'up':'dn';"
        "    ce.textContent=(chg>=0?'▲':'▼')+' '+Math.abs(diff).toFixed(DEC)+' ('+(chg>=0?'+':'')+chg.toFixed(3)+'%)';"
        "  }"
        "  el.textContent=fmt(p);"
        "  var d=new Date();"
        "  document.getElementById('ts').textContent=d.toLocaleTimeString('en-GB')+'.'+String(d.getMilliseconds()).padStart(3,'0');"
        "  lp=p;"
        "}"
        "function setStatus(s){"
        "  var d=document.getElementById('ws-dot'),l=document.getElementById('ws-lbl');"
        "  d.className='';"
        "  if(s==='live'){d.classList.add('live');l.textContent='● " + src_label + " LIVE';}"
        "  else if(s==='error'){d.classList.add('error');l.textContent='Connection error';}"
        "  else if(s==='reconnect'){d.classList.add('rc');l.textContent='Reconnecting...';}"
        "}"
        + ws_js +
        "setInterval(function(){"
        "  var d=new Date();"
        "  document.getElementById('ts').textContent=d.toLocaleTimeString('en-GB')+'.'+String(d.getMilliseconds()).padStart(3,'0');"
        "},100);"
        "})();</script>"
        "</body></html>"
    )
    return html

@st.cache_data(ttl=1)   # cache 1 giây — gần realtime
def fetch_realtime_price(pair_name: str) -> float | None:
    """
    Lấy giá tức thì từ ExchangeRate-API — miễn phí, không cần key.
    Hỗ trợ Forex + Kim loại. Crypto dùng Binance.
    Cache 1s để không spam API nhưng vẫn gần realtime.
    """
    try:
        _, _, asset_type = PAIRS[pair_name]
        td_sym = PAIRS[pair_name][0]  # "EUR/USD"

        if asset_type == "crypto":
            # Binance API — realtime hoàn toàn
            symbol_map = {
                "BTC/USD": "BTCUSDT",
                "ETH/USD": "ETHUSDT",
            }
            binance_sym = symbol_map.get(td_sym)
            if binance_sym:
                resp = requests.get(
                    f"https://api.binance.com/api/v3/ticker/price?symbol={binance_sym}",
                    timeout=5
                )
                return float(resp.json()["price"])

        elif asset_type in ("forex", "commodity"):
            # ExchangeRate-API — free, no key needed
            base, quote = td_sym.split("/")
            resp = requests.get(
                f"https://open.er-api.com/v6/latest/{base}",
                timeout=5
            )
            data = resp.json()
            if data.get("result") == "success":
                rate = data["rates"].get(quote)
                if rate:
                    return float(rate)
    except Exception:
        pass
    return None


@st.cache_data(ttl=10)   # cache 10 giây
def fetch_ohlcv_twelvedata(td_symbol: str, interval: str, outputsize: int) -> pd.DataFrame:
    """Tải dữ liệu từ Twelve Data API (~1 giây độ trễ)."""
    # Đọc key từ st.secrets (Streamlit Cloud) hoặc os.getenv (local)
    api_key = ""
    try:
        api_key = st.secrets["TWELVE_DATA_API_KEY"]
    except Exception:
        api_key = os.getenv("TWELVE_DATA_API_KEY", "")
    if not api_key:
        st.error("❌ Không tìm thấy TWELVE_DATA_API_KEY trong Secrets!")
        return pd.DataFrame()
    try:
        url = "https://api.twelvedata.com/time_series"
        params = {
            "symbol":     td_symbol,
            "interval":   interval,
            "outputsize": outputsize,
            "apikey":     api_key,
            "format":     "JSON",
            "order":      "ASC",
        }
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        if data.get("status") == "error" or "values" not in data:
            return pd.DataFrame()
        rows = data["values"]
        df = pd.DataFrame(rows)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")
        df = df.rename(columns={
            "open": "Open", "high": "High",
            "low": "Low", "close": "Close", "volume": "Volume"
        })
        for col in ["Open","High","Low","Close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0) if "Volume" in df.columns else 0
        return df[["Open","High","Low","Close","Volume"]].dropna()
    except Exception as e:
        st.error(f"❌ Twelve Data lỗi: {e}")
        return pd.DataFrame()



def fetch_ohlcv(pair_name: str, tf_label: str) -> tuple[pd.DataFrame, str]:
    """
    Lấy OHLCV từ Yahoo Finance + giá realtime từ ExchangeRate API.
    Trả về (DataFrame, source_name)
    """
    _, yf_ticker, _ = PAIRS[pair_name]
    _, yf_interval, yf_period, _ = TIMEFRAMES[tf_label]

    df = fetch_ohlcv_yahoo(yf_ticker, yf_interval, yf_period)
    if not df.empty:
        if tf_label == "H4 (4 giờ)":
            df = df.resample("4h").agg({
                "Open":"first","High":"max",
                "Low":"min","Close":"last","Volume":"sum"
            }).dropna()
        return df, "Yahoo Finance"

    return pd.DataFrame(), "Lỗi kết nối"




# ════════════════════════════════════════════════════════
#  TÍNH CHỈ BÁO KỸ THUẬT (thủ công, không cần ta-lib)
# ════════════════════════════════════════════════════════

def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calc_bollinger(close: pd.Series, period=20, std=2):
    sma = close.rolling(period).mean()
    stddev = close.rolling(period).std()
    upper = sma + std * stddev
    lower = sma - std * stddev
    return upper, sma, lower

def calc_stochastic(high, low, close, k=14, d=3):
    lowest_low = low.rolling(k).min()
    highest_high = high.rolling(k).max()
    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    stoch_d = stoch_k.rolling(d).mean()
    return stoch_k, stoch_d

def calc_atr(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calc_ema(close: pd.Series, period: int) -> pd.Series:
    return close.ewm(span=period, adjust=False).mean()

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm tất cả chỉ báo vào DataFrame."""
    if len(df) < 50:
        return df
    close, high, low = df["Close"], df["High"], df["Low"]

    df["RSI"]          = calc_rsi(close)
    df["EMA_20"]       = calc_ema(close, 20)
    df["EMA_50"]       = calc_ema(close, 50)
    df["EMA_200"]      = calc_ema(close, 200)
    df["MACD"], df["MACD_signal"], df["MACD_hist"] = calc_macd(close)
    df["BB_upper"], df["BB_mid"], df["BB_lower"]   = calc_bollinger(close)
    df["Stoch_K"], df["Stoch_D"]                   = calc_stochastic(high, low, close)
    df["ATR"]          = calc_atr(high, low, close)

    return df


# ════════════════════════════════════════════════════════
#  TIN TỨC & SENTIMENT ANALYSIS
# ════════════════════════════════════════════════════════

# Từ khóa tích cực/tiêu cực cho Forex
BULLISH_WORDS = [
    "rise", "rising", "surge", "surges", "gain", "gains", "rally", "rallies",
    "strengthen", "strengthens", "advances", "higher", "up", "bullish", "hawkish",
    "beat", "beats", "exceeds", "strong", "stronger", "growth", "tăng", "mạnh",
    "positive", "optimistic", "recovery", "recovers", "boost", "boosts",
    "rate hike", "hike", "hikes", "outperform", "buy", "demand"
]
BEARISH_WORDS = [
    "fall", "falling", "drop", "drops", "decline", "declines", "plunge", "plunges",
    "weaken", "weakens", "lower", "down", "bearish", "dovish", "miss", "misses",
    "weak", "weaker", "recession", "slow", "slows", "risk", "concern", "giảm", "yếu",
    "negative", "pessimistic", "selloff", "sell", "cut", "cuts", "rate cut",
    "disappoint", "disappoints", "crisis", "fear", "fears", "underperform"
]

# Map cặp tiền → từ khóa tìm kiếm tin tức
PAIR_KEYWORDS = {
    "EUR": ["EUR", "euro", "ECB", "eurozone", "European"],
    "GBP": ["GBP", "pound", "sterling", "BOE", "Bank of England", "UK", "Britain"],
    "USD": ["USD", "dollar", "Fed", "Federal Reserve", "FOMC", "US economy"],
    "JPY": ["JPY", "yen", "BOJ", "Bank of Japan", "Japan"],
    "AUD": ["AUD", "aussie", "RBA", "Reserve Bank Australia"],
    "CAD": ["CAD", "loonie", "BOC", "Bank of Canada", "oil"],
    "CHF": ["CHF", "franc", "SNB", "Swiss"],
    "NZD": ["NZD", "kiwi", "RBNZ"],
    "XAU": ["gold", "XAU", "bullion", "safe haven"],
    "XAG": ["silver", "XAG"],
    "WTI": ["oil", "crude", "WTI", "OPEC", "energy"],
    "BTC": ["bitcoin", "BTC", "crypto", "cryptocurrency"],
    "ETH": ["ethereum", "ETH", "crypto"],
}

@st.cache_data(ttl=120)  # cache 2 phút
def fetch_forex_news(pair_name: str) -> list:
    """Lấy tin tức Forex từ RSS feeds miễn phí."""
    td_sym = PAIRS[pair_name][0]  # "EUR/USD"
    currencies = td_sym.replace("/", " ").split()

    # Tổng hợp từ khóa tìm kiếm
    keywords = []
    for cur in currencies:
        keywords.extend(PAIR_KEYWORDS.get(cur, [cur]))
    keywords = list(dict.fromkeys(keywords))[:5]  # top 5 unique

    news_items = []

    # RSS feeds miễn phí cho Forex
    rss_feeds = [
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=EURUSD=X&region=US&lang=en-US",
        "https://www.forexlive.com/feed/news",
        "https://www.dailyfx.com/feeds/all",
        "https://feeds.bbci.co.uk/news/business/rss.xml",
    ]

    import xml.etree.ElementTree as ET
    for feed_url in rss_feeds[:3]:
        try:
            resp = requests.get(feed_url, timeout=5,
                                headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code != 200:
                continue
            root = ET.fromstring(resp.content)
            items = root.findall(".//item")[:10]
            for item in items:
                title = item.findtext("title", "")
                desc  = item.findtext("description", "")
                pub   = item.findtext("pubDate", "")
                link  = item.findtext("link", "")
                text  = (title + " " + desc).lower()

                # Lọc tin liên quan đến cặp tiền
                relevant = any(kw.lower() in text for kw in keywords)
                if not relevant and "forex" not in feed_url:
                    continue

                # Tính sentiment score
                bull = sum(1 for w in BULLISH_WORDS if w in text)
                bear = sum(1 for w in BEARISH_WORDS if w in text)
                if bull > bear:
                    sentiment = "bullish"
                    sent_score = min(bull - bear, 5)
                elif bear > bull:
                    sentiment = "bearish"
                    sent_score = -min(bear - bull, 5)
                else:
                    sentiment = "neutral"
                    sent_score = 0

                news_items.append({
                    "title":     title[:100],
                    "sentiment": sentiment,
                    "score":     sent_score,
                    "time":      pub[:25] if pub else "",
                    "link":      link,
                })
        except Exception:
            continue

    # Sắp xếp theo sentiment mạnh nhất
    news_items.sort(key=lambda x: abs(x["score"]), reverse=True)
    return news_items[:8]


def analyze_news_sentiment(news_items: list) -> dict:
    """Tổng hợp sentiment từ danh sách tin tức."""
    if not news_items:
        return {"action": "NEUTRAL", "score": 0, "confidence": 0,
                "bull_count": 0, "bear_count": 0, "neutral_count": 0}

    total_score = sum(n["score"] for n in news_items)
    bull_count  = sum(1 for n in news_items if n["sentiment"] == "bullish")
    bear_count  = sum(1 for n in news_items if n["sentiment"] == "bearish")
    neut_count  = sum(1 for n in news_items if n["sentiment"] == "neutral")
    total       = len(news_items)

    # Tín hiệu từ tin tức
    if total_score >= 3:
        action = "BUY"
    elif total_score <= -3:
        action = "SELL"
    else:
        action = "NEUTRAL"

    confidence = min(85, int(abs(total_score) / (total * 2) * 100) + 30) if total > 0 else 0

    return {
        "action":       action,
        "score":        total_score,
        "confidence":   confidence,
        "bull_count":   bull_count,
        "bear_count":   bear_count,
        "neutral_count": neut_count,
    }


# ════════════════════════════════════════════════════════
#  TÍNH TÍN HIỆU
# ════════════════════════════════════════════════════════

def _make_signal_result(action, score, max_score, atr_mult_tp, atr_mult_sl,
                        last, signals, rsi, sk):
    """Helper tạo dict kết quả tín hiệu."""
    confidence = min(95, int(abs(score) / max_score * 100) + 40)
    atr  = float(last["ATR"])
    entry = float(last["Close"])
    if action == "BUY":
        tp = entry + atr * atr_mult_tp
        sl = entry - atr * atr_mult_sl
    elif action == "SELL":
        tp = entry - atr * atr_mult_tp
        sl = entry + atr * atr_mult_sl
    else:
        tp = entry + atr * 1.5
        sl = entry - atr * 1.0
    return {
        "action": action, "confidence": confidence, "score": score,
        "signals": signals, "entry": entry, "tp": tp, "sl": sl,
        "rsi": float(rsi), "macd": float(last["MACD"]),
        "atr": atr, "stoch_k": float(sk),
    }


def compute_signal_short(df: pd.DataFrame) -> dict:
    """
    Tín hiệu NGẮN HẠN — scalping/intraday.
    Dựa trên: RSI, MACD crossover, Stochastic, Bollinger Bands.
    TP = 1.5×ATR, SL = 0.75×ATR → Risk/Reward 1:2
    """
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last
    signals = {}
    score = 0

    # RSI ngắn hạn — nhạy hơn
    rsi = last["RSI"]
    if rsi < 35:
        signals["RSI"] = ("🟢 Quá bán", "bullish"); score += 2
    elif rsi > 65:
        signals["RSI"] = ("🔴 Quá mua", "bearish"); score -= 2
    elif rsi < 50:
        signals["RSI"] = ("🟡 Hơi yếu", "neutral"); score += 0.5
    else:
        signals["RSI"] = ("🟡 Hơi mạnh", "neutral"); score -= 0.5

    # MACD crossover — tín hiệu mạnh nhất ngắn hạn
    if last["MACD"] > last["MACD_signal"] and prev["MACD"] <= prev["MACD_signal"]:
        signals["MACD"] = ("🟢 Cắt lên ↑", "bullish"); score += 3
    elif last["MACD"] < last["MACD_signal"] and prev["MACD"] >= prev["MACD_signal"]:
        signals["MACD"] = ("🔴 Cắt xuống ↓", "bearish"); score -= 3
    elif last["MACD"] > last["MACD_signal"]:
        signals["MACD"] = ("🟢 MACD > Signal", "bullish"); score += 1
    else:
        signals["MACD"] = ("🔴 MACD < Signal", "bearish"); score -= 1

    # Stochastic — momentum ngắn hạn
    sk = last["Stoch_K"]
    sd = last["Stoch_D"]
    if sk < 20 and sk > sd:
        signals["Stochastic"] = ("🟢 Đảo chiều tăng", "bullish"); score += 2
    elif sk > 80 and sk < sd:
        signals["Stochastic"] = ("🔴 Đảo chiều giảm", "bearish"); score -= 2
    elif sk < 20:
        signals["Stochastic"] = ("🟢 Quá bán", "bullish"); score += 1
    elif sk > 80:
        signals["Stochastic"] = ("🔴 Quá mua", "bearish"); score -= 1
    else:
        signals["Stochastic"] = ("⚪ Trung tính", "neutral")

    # Bollinger Bands — breakout/rebound
    c = float(last["Close"])
    if c < float(last["BB_lower"]):
        signals["Bollinger"] = ("🟢 Chạm đáy BB", "bullish"); score += 2
    elif c > float(last["BB_upper"]):
        signals["Bollinger"] = ("🔴 Chạm đỉnh BB", "bearish"); score -= 2
    elif c > float(last["BB_mid"]):
        signals["Bollinger"] = ("🟢 Trên MA20", "bullish"); score += 1
    else:
        signals["Bollinger"] = ("🔴 Dưới MA20", "bearish"); score -= 1

    score = int(score)
    action = "BUY" if score >= 4 else "SELL" if score <= -4 else "NEUTRAL"
    result = _make_signal_result(action, score, 8, 1.5, 0.75, last, signals, rsi, sk)
    result["news_score"] = 0  # sẽ được cập nhật ngoài
    return result


def compute_signal_long(df: pd.DataFrame) -> dict:
    """
    Tín hiệu DÀI HẠN — swing/position trading.
    Dựa trên: EMA 20/50/200, xu hướng tổng thể, MACD histogram.
    TP = 3×ATR, SL = 1.5×ATR → Risk/Reward 1:2
    """
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last
    signals = {}
    score = 0

    c = float(last["Close"])

    # EMA 20/50 — xu hướng trung hạn
    e20, e50 = float(last["EMA_20"]), float(last["EMA_50"])
    if e20 > e50 and prev["EMA_20"] <= prev["EMA_50"]:
        signals["EMA 20/50"] = ("🟢 Golden Cross ✨", "bullish"); score += 3
    elif e20 < e50 and prev["EMA_20"] >= prev["EMA_50"]:
        signals["EMA 20/50"] = ("🔴 Death Cross ☠️", "bearish"); score -= 3
    elif e20 > e50:
        signals["EMA 20/50"] = ("🟢 Xu hướng tăng", "bullish"); score += 1
    else:
        signals["EMA 20/50"] = ("🔴 Xu hướng giảm", "bearish"); score -= 1

    # EMA 200 — xu hướng dài hạn (quan trọng nhất)
    e200 = float(last["EMA_200"])
    if c > e200 and e20 > e200:
        signals["EMA 200"] = ("🟢 Trên EMA200 mạnh", "bullish"); score += 3
    elif c > e200:
        signals["EMA 200"] = ("🟢 Giá trên EMA200", "bullish"); score += 2
    elif c < e200 and e20 < e200:
        signals["EMA 200"] = ("🔴 Dưới EMA200 yếu", "bearish"); score -= 3
    else:
        signals["EMA 200"] = ("🔴 Giá dưới EMA200", "bearish"); score -= 2

    # MACD histogram — momentum dài hạn
    hist = float(last["MACD_hist"])
    prev_hist = float(prev["MACD_hist"])
    if hist > 0 and hist > prev_hist:
        signals["MACD Hist"] = ("🟢 Tăng tốc tăng", "bullish"); score += 2
    elif hist > 0:
        signals["MACD Hist"] = ("🟢 Dương", "bullish"); score += 1
    elif hist < 0 and hist < prev_hist:
        signals["MACD Hist"] = ("🔴 Tăng tốc giảm", "bearish"); score -= 2
    else:
        signals["MACD Hist"] = ("🔴 Âm", "bearish"); score -= 1

    # Giá so với EMA 20 — vị trí trong xu hướng
    rsi = float(last["RSI"])
    if c > e20 and rsi > 50:
        signals["Vị trí giá"] = ("🟢 Trên EMA20, RSI > 50", "bullish"); score += 1
    elif c < e20 and rsi < 50:
        signals["Vị trí giá"] = ("🔴 Dưới EMA20, RSI < 50", "bearish"); score -= 1
    else:
        signals["Vị trí giá"] = ("⚪ Trung tính", "neutral")

    sk = float(last["Stoch_K"])
    score = int(score)
    action = "BUY" if score >= 5 else "SELL" if score <= -5 else "NEUTRAL"
    return _make_signal_result(action, score, 9, 3.0, 1.5, last, signals, rsi, sk)


def compute_signal(df: pd.DataFrame) -> dict:
    """Tín hiệu tổng hợp (backward compat)."""
    return compute_signal_short(df)


# ════════════════════════════════════════════════════════
#  VẼ BIỂU ĐỒ PLOTLY
# ════════════════════════════════════════════════════════

def build_chart(df: pd.DataFrame, pair: str) -> go.Figure:
    """Vẽ biểu đồ nến + chỉ báo."""
    last_n = min(120, len(df))
    d = df.tail(last_n).copy()

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.2, 0.2],
        vertical_spacing=0.03,
        subplot_titles=(f"{pair} — Biểu đồ nến", "MACD", "RSI")
    )

    # ── Nến ──
    fig.add_trace(go.Candlestick(
        x=d.index, open=d["Open"], high=d["High"],
        low=d["Low"], close=d["Close"],
        name="Giá",
        increasing_fillcolor="#43a047",
        increasing_line_color="#2e7d32",
        decreasing_fillcolor="#ef5350",
        decreasing_line_color="#c62828",
    ), row=1, col=1)

    # ── EMA ──
    for col, color, name in [
        ("EMA_20",  "#1976d2", "EMA 20"),
        ("EMA_50",  "#f57c00", "EMA 50"),
        ("EMA_200", "#7b1fa2", "EMA 200"),
    ]:
        if col in d.columns:
            fig.add_trace(go.Scatter(
                x=d.index, y=d[col], name=name,
                line=dict(color=color, width=1.2),
                opacity=0.8
            ), row=1, col=1)

    # ── Bollinger Bands ──
    if "BB_upper" in d.columns:
        fig.add_trace(go.Scatter(
            x=d.index, y=d["BB_upper"], name="BB Upper",
            line=dict(color="rgba(150,150,255,0.4)", width=1, dash="dot"),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=d.index, y=d["BB_lower"], name="BB Lower",
            line=dict(color="rgba(150,150,255,0.4)", width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(100,100,255,0.04)"
        ), row=1, col=1)

    # ── MACD ──
    if "MACD" in d.columns:
        colors_hist = ["#43a047" if v >= 0 else "#e53935" for v in d["MACD_hist"]]
        fig.add_trace(go.Bar(
            x=d.index, y=d["MACD_hist"], name="MACD Hist",
            marker_color=colors_hist, opacity=0.6
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=d.index, y=d["MACD"], name="MACD",
            line=dict(color="#00e5ff", width=1.5)
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=d.index, y=d["MACD_signal"], name="Signal",
            line=dict(color="#ff6b35", width=1.5)
        ), row=2, col=1)

    # ── RSI ──
    if "RSI" in d.columns:
        fig.add_trace(go.Scatter(
            x=d.index, y=d["RSI"], name="RSI",
            line=dict(color="#ffcc00", width=1.5)
        ), row=3, col=1)
        for level, color in [(70, "rgba(229,57,53,0.4)"), (30, "rgba(56,142,60,0.4)")]:
            fig.add_hline(y=level, line_dash="dash",
                          line_color=color, row=3, col=1)

    fig.update_layout(
        height=600,
        paper_bgcolor="#f8fafd",
        plot_bgcolor="#ffffff",
        font=dict(color="#1a2332", family="Inter, sans-serif", size=11),
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#dce8f5", borderwidth=1,
            font=dict(size=10, color="#5a7a9a")
        ),
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_rangeslider_visible=False,
    )

    for i in range(1, 4):
        fig.update_xaxes(
            gridcolor="#e8f0fa", zeroline=False,
            showgrid=True, row=i, col=1,
            linecolor="#dce8f5",
        )
        fig.update_yaxes(
            gridcolor="#e8f0fa", zeroline=False,
            showgrid=True, row=i, col=1,
            linecolor="#dce8f5",
        )

    return fig


# ════════════════════════════════════════════════════════
#  GỌI CLAUDE AI
# ════════════════════════════════════════════════════════

def get_ai_analysis(pair: str, tf: str, sig: dict) -> str:
    """Gọi Groq API (Llama 3) để lấy phân tích — MIỄN PHÍ."""
    if not GROQ_API_KEY:
        return (
            "⚠️ Chưa cấu hình GROQ_API_KEY.\n\n"
            "**Cách lấy key miễn phí (2 phút):**\n"
            "1. Vào **console.groq.com** → Đăng ký bằng Google\n"
            "2. Nhấn **API Keys** → **Create API Key**\n"
            "3. Copy key (bắt đầu bằng `gsk_...`)\n"
            "4. Mở file `.env` → thêm: `GROQ_API_KEY=gsk_xxxxx`\n"
            "5. Khởi động lại bot"
        )

    signals_text = "\n".join([f"- {k}: {v[0]}" for k, v in sig["signals"].items()])
    prompt = f"""Bạn là chuyên gia phân tích kỹ thuật Forex cấp cao. Hãy viết phân tích (6-8 câu) bằng tiếng Việt cho:

Cặp tiền: {pair}
Khung thời gian: {tf}
Tín hiệu tổng hợp: {sig['action']} (Score: {sig['score']}, Confidence: {sig['confidence']}%)
RSI: {sig['rsi']:.1f}
MACD: {sig['macd']:.5f}
Stochastic: {sig['stoch_k']:.1f}
ATR: {sig['atr']:.5f}

Chi tiết chỉ báo:
{signals_text}

Mức giá:
- Vào lệnh: {sig['entry']:.5f}
- Take Profit: {sig['tp']:.5f}
- Stop Loss: {sig['sl']:.5f}

Yêu cầu: Viết đoạn văn liền mạch, phân tích chuyên sâu, nêu rõ xu hướng, ngưỡng kháng cự/hỗ trợ tiềm năng và lưu ý quản lý rủi ro. Không dùng bullet points."""

    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {GROQ_API_KEY}",
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "max_tokens": 600,
                "temperature": 0.7,
                "messages": [
                    {"role": "system", "content": "Bạn là chuyên gia phân tích kỹ thuật Forex, luôn trả lời bằng tiếng Việt, chuyên nghiệp và súc tích."},
                    {"role": "user", "content": prompt}
                ],
            },
            timeout=30,
        )
        data = resp.json()
        if "choices" in data and data["choices"]:
            return data["choices"][0]["message"]["content"]
        err = data.get("error", {}).get("message", str(data))
        return f"Lỗi Groq API: {err}"
    except Exception as e:
        return f"Không thể kết nối Groq API: {e}"


# ════════════════════════════════════════════════════════
#  GỬI TELEGRAM
# ════════════════════════════════════════════════════════

def send_telegram(pair: str, sig: dict) -> bool:
    """Gửi cảnh báo tín hiệu qua Telegram."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    emoji = "📈" if sig["action"] == "BUY" else "📉" if sig["action"] == "SELL" else "⏸"
    text = (
        f"{emoji} *ForexAI Signal*\n\n"
        f"*Cặp tiền:* {pair}\n"
        f"*Tín hiệu:* {sig['action']}\n"
        f"*Độ tin cậy:* {sig['confidence']}%\n\n"
        f"*Entry:* `{sig['entry']:.5f}`\n"
        f"*Take Profit:* `{sig['tp']:.5f}`\n"
        f"*Stop Loss:* `{sig['sl']:.5f}`\n\n"
        f"*RSI:* {sig['rsi']:.1f} | *ATR:* {sig['atr']:.5f}\n"
        f"_Cập nhật: {datetime.now().strftime('%H:%M %d/%m/%Y')}_"
    )
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"},
            timeout=10,
        )
        return r.status_code == 200
    except:
        return False


# ════════════════════════════════════════════════════════
#  GIAO DIỆN CHÍNH
# ════════════════════════════════════════════════════════

def main():
    # ── Header ──
    st.markdown("""
    <div class="fx-header">
      <div class="fx-live-dot"></div>
      <div class="fx-header-logo">FOREX<span>AI</span></div>
      <div class="fx-badge">LIVE</div>
      <div class="fx-header-right" id="fx-clock">--:--:--</div>
    </div>
    <script>
    (function() {
      function tick() {
        var el = document.getElementById('fx-clock');
        if (el) {
          var now = new Date();
          el.textContent = now.toLocaleTimeString('en-GB', {hour12:false}) +
            '.' + String(now.getMilliseconds()).padStart(3,'0');
        }
        requestAnimationFrame(tick);
      }
      tick();
    })();
    </script>
    """, unsafe_allow_html=True)

        # ── Selector nhanh ngay dưới header (luôn hiển thị, kể cả mobile) ──
    mc1, mc2 = st.columns([3, 2])
    with mc1:
        pair = st.selectbox("📌 Cặp tiền", list(PAIRS.keys()), index=0,
                            label_visibility="collapsed",
                            placeholder="Chọn cặp tiền...")
    with mc2:
        tf_label = st.selectbox("⏱ Khung TG", list(TIMEFRAMES.keys()), index=2,
                                label_visibility="collapsed")
    tf_short = tf_label.split(" ")[0]

    # ── Sidebar ── (ẩn trên mobile nhưng vẫn có trên desktop)
    with st.sidebar:
        st.markdown("### ⚙️ Cài đặt")
        st.markdown("""
        <div style="background:rgba(144,202,249,0.1);border:1px solid rgba(144,202,249,0.2);
                    border-radius:8px;padding:8px 10px;font-size:11px;color:#90caf9;margin-bottom:8px">
        💡 Chọn cặp tiền & khung TG ngay trên màn hình chính
        </div>
        """, unsafe_allow_html=True)

        # Đồng bộ với selector trên main (chỉ hiển thị lại để desktop tiện)
        pair_sb = st.selectbox("Cặp tiền tệ", list(PAIRS.keys()),
                               index=list(PAIRS.keys()).index(pair))
        tf_sb   = st.selectbox("Khung thời gian", list(TIMEFRAMES.keys()),
                               index=list(TIMEFRAMES.keys()).index(tf_label))
        # Nếu sidebar thay đổi thì dùng sidebar
        if pair_sb != pair:
            pair = pair_sb
        if tf_sb != tf_label:
            tf_label = tf_sb
            tf_short = tf_label.split(" ")[0]

        st.markdown("---")

        st.markdown("""
        <div style="background:rgba(67,160,71,0.1);border:1px solid rgba(67,160,71,0.4);
                    border-radius:8px;padding:8px 12px;font-size:12px;
                    color:#1b5e20;font-weight:600;text-align:center">
        ⚡ Twelve Data · Độ trễ ~1s
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        use_ai = st.checkbox("🤖 Phân tích AI (Groq/Llama 3)", value=bool(GROQ_API_KEY))
        use_telegram = st.checkbox("📱 Gửi Telegram", value=False)

        if not GROQ_API_KEY:
            st.markdown("""
            <div style="background:rgba(255,152,0,0.1);border:1px solid rgba(255,152,0,0.4);
                        border-radius:8px;padding:10px;font-size:11px;color:#e65100;margin-top:8px">
            ⚠️ Chưa có Groq API Key.<br>
            Lấy miễn phí tại <b>console.groq.com</b><br>
            Thêm vào file <code style="background:#fff3e0;color:#bf360c">.env</code>:<br>
            <code style="background:#fff3e0;color:#bf360c">GROQ_API_KEY=gsk_...</code>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        analyze_btn = st.button("⚡ Phân tích ngay", use_container_width=True)

        st.markdown("---")
        st.markdown("""
        <div style="background:rgba(255,152,0,0.12);border:1px solid rgba(255,152,0,0.3);
                    border-radius:8px;padding:10px 12px;font-size:11px;color:#bf360c;line-height:1.8">
        ⚠️ <b>Cảnh báo rủi ro</b><br>
        Thông tin chỉ mang tính tham khảo. Không phải tư vấn đầu tư tài chính.
        Giao dịch Forex có rủi ro cao.
        </div>
        """, unsafe_allow_html=True)

    # ── Định dạng số thập phân theo loại tài sản ──
    _, _, asset_type = PAIRS[pair]

    if asset_type == "crypto":
        fmt = lambda v: f"{v:,.2f}"
    elif asset_type == "commodity":
        fmt = lambda v: f"{v:,.3f}"
    elif "JPY" in pair:
        fmt = lambda v: f"{v:,.3f}"
    else:
        fmt = lambda v: f"{v:.5f}"

    # ══════════════════════════════════════════════
    # WebSocket PRICE — độc lập iframe, không reset khi rerun
    # ══════════════════════════════════════════════
    ws_html = get_ws_component(pair)
    components.html(ws_html, height=148, scrolling=False)

    # ══════════════════════════════════════════════
    # LIVE DASHBOARD: cập nhật chỉ báo mỗi 1 giây
    # ══════════════════════════════════════════════
    def live_dashboard():
        df_raw, data_src = fetch_ohlcv(pair, tf_label)
        if df_raw.empty:
            st.error("❌ Không tải được dữ liệu.")
            return
        df = add_indicators(df_raw.copy())
        if len(df) < 30:
            st.warning("⚠️ Không đủ dữ liệu.")
            return

        sig_s = compute_signal_short(df)
        sig_l = compute_signal_long(df)
        news  = fetch_forex_news(pair)
        sent  = analyze_news_sentiment(news)

        nb    = sent["score"]
        csc   = sig_s["score"] + nb
        sig_s["action"]     = "BUY" if csc>=4 else "SELL" if csc<=-4 else "NEUTRAL"
        sig_s["confidence"] = min(95, sig_s["confidence"] + (5 if sent["action"]==sig_s["action"] else 0))
        sig   = sig_s

        last  = df.iloc[-1]
        prev  = df.iloc[-2]
        # Giá realtime được hiển thị qua WS component (iframe)
        # Ở đây chỉ dùng Close để tính signal & OHLC bar
        price = float(last["Close"])
        chg   = (price - float(prev["Close"])) / float(prev["Close"]) * 100

        # âm thanh & telegram
        prev_act = st.session_state.get("prev_action","")
        if prev_act and prev_act != sig["action"] and sig["action"] != "NEUTRAL":
            freq = "880" if sig["action"]=="BUY" else "440"
            st.markdown(f"""<script>try{{var c=new AudioContext();var o=c.createOscillator();
            var g=c.createGain();o.connect(g);g.connect(c.destination);
            o.frequency.value={freq};g.gain.setValueAtTime(0.3,c.currentTime);
            g.gain.exponentialRampToValueAtTime(0.001,c.currentTime+0.6);
            o.start();o.stop(c.currentTime+0.6);}}catch(e){{}}</script>""",
            unsafe_allow_html=True)
            if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                send_telegram(pair, sig)
        st.session_state["prev_action"] = sig["action"]

        # ─── OHLC MINI BAR (từ candle data) ───
        from datetime import datetime as _dt
        now_s = _dt.now().strftime("%H:%M:%S")
        atr_s = fmt(float(last["ATR"])) if not np.isnan(last["ATR"]) else "—"
        st.markdown(
            '<div class="fx-livebar" style="margin:0 0 4px 0">' +
            '<div class="fx-livebar-dot"></div>' +
            '<span class="fx-livebar-txt">' + data_src.replace(" ⚡","") + ' · ' + tf_short + '</span>' +
            '<span style="margin-left:auto;font-family:monospace;font-size:10px;color:#475569">' +
            'H:' + fmt(float(last["High"])) + '  L:' + fmt(float(last["Low"])) + '  ATR:' + atr_s +
            '</span>' +
            '</div>',
            unsafe_allow_html=True
        )

        # ─── SIGNAL CARDS ───
        def _sig_html(s, title, icon):
            a   = s["action"]
            ac  = a.lower() if a != "NEUTRAL" else "neut"
            em  = "📈" if a == "BUY" else "📉" if a == "SELL" else "⏸"
            lbl = "MUA" if a == "BUY" else "BÁN" if a == "SELL" else "TRUNG TÍNH"
            rr  = abs(s["tp"] - s["entry"]) / max(abs(s["sl"] - s["entry"]), 1e-10)
            conf= s["confidence"]
            sc  = s["score"]
            tp  = fmt(s["tp"])
            en  = fmt(s["entry"])
            sl  = fmt(s["sl"])
            # Dùng % thay f-string để tránh conflict dấu {}
            html = (
                '<div class="fx-sig ' + ac + '">' +
                '<div class="fx-sig-lbl">' + icon + ' ' + title + '</div>' +
                '<div class="fx-sig-act ' + ac + '">' + em + ' ' + lbl + '</div>' +
                '<div class="fx-sig-conf ' + ac + '">' + str(conf) + '% · ' + ('%+d' % sc) + 'pt</div>' +
                '<div class="fx-tpsl">' +
                  '<div class="fx-tpsl-item tp"><div class="fx-tpsl-lbl">TP</div><div class="fx-tpsl-val">' + tp + '</div></div>' +
                  '<div class="fx-tpsl-item entry"><div class="fx-tpsl-lbl">Entry</div><div class="fx-tpsl-val">' + en + '</div></div>' +
                  '<div class="fx-tpsl-item sl"><div class="fx-tpsl-lbl">SL</div><div class="fx-tpsl-val">' + sl + '</div></div>' +
                '</div>' +
                '<div style="font-size:8px;color:#1e3a5f;margin-top:4px;font-family:monospace;text-align:right">R/R 1:' + ('%.1f' % rr) + '</div>' +
                '</div>'
            )
            return html

        html_short = _sig_html(sig_s, "NGẮN HẠN", "⚡")
        html_long  = _sig_html(sig_l, "DÀI HẠN", "📊")
        st.markdown('<div class="fx-sec">⚡ Tín hiệu giao dịch</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="fx-sig-row">' + html_short + html_long + '</div>',
            unsafe_allow_html=True
        )

        # ─── MULTI-TIMEFRAME SIGNALS TABLE ───
        st.markdown('<div class="fx-sec">📊 Phân tích đa khung giờ</div>', unsafe_allow_html=True)

        _, yf_ticker, _ = PAIRS[pair]
        mtf_data = fetch_mtf_signals(yf_ticker)

        IND_ROWS = [
            ("RSI",   "RSI 14"),
            ("MACD",  "MACD"),
            ("EMA",   "EMA 20/50"),
            ("EMA200","EMA 200"),
            ("BB",    "Bollinger"),
            ("Stoch", "Stoch %K"),
        ]

        def _cell(action, val=""):
            import html as _html
            safe_val = _html.escape(str(val))
            if action == "BUY":
                return '<td class="c-buy">▲' + safe_val + '</td>'
            elif action == "SELL":
                return '<td class="c-sell">▼' + safe_val + '</td>'
            elif action == "N/A":
                return '<td class="c-na">—</td>'
            else:
                return '<td class="c-neutral">–' + safe_val + '</td>'

        def _sum_class(action):
            if action == "BUY":    return "c-sum-buy"
            elif action == "SELL": return "c-sum-sell"
            else:                  return "c-sum-neut"

        def _th_class(action):
            if action == "BUY":    return "tf-header-buy"
            elif action == "SELL": return "tf-header-sell"
            else:                  return "tf-header-neut"

        # Header row — tên khung giờ
        thead = '<thead><tr><th>Chỉ báo</th>'
        for m in mtf_data:
            th_cls = _th_class(m["action"])
            thead += '<th class="' + th_cls + '">' + m["tf"] + '</th>'
        thead += '</tr></thead>'

        # Body — mỗi hàng 1 chỉ báo
        tbody = '<tbody>'
        for ind_key, ind_label in IND_ROWS:
            tbody += '<tr><td>' + ind_label + '</td>'
            for m in mtf_data:
                if m.get("error") or ind_key not in m["indicators"]:
                    tbody += '<td class="c-na">—</td>'
                else:
                    act, val = m["indicators"][ind_key]
                    tbody += _cell(act, " " + val)
            tbody += '</tr>'

        # Summary row — tổng hợp tín hiệu
        tbody += '<tr><td style="color:#e2e8f0;font-weight:700">TỔNG HỢP</td>'
        for m in mtf_data:
            sc   = m["score"]
            act  = m["action"]
            sc_cls = _sum_class(act)
            em   = "▲" if act == "BUY" else "▼" if act == "SELL" else "–"
            lbl  = "MUA" if act == "BUY" else "BÁN" if act == "SELL" else "NGANG" if act == "NEUTRAL" else "N/A"
            tbody += '<td class="' + sc_cls + '">' + em + ' ' + lbl + '</td>'
        tbody += '</tr></tbody>'

        mtf_html = (
            '<div class="mtf-wrap">' +
            '<table class="mtf-table">' + thead + tbody + '</table>' +
            '</div>'
        )
        st.markdown(mtf_html, unsafe_allow_html=True)

        # ─── SENTIMENT ───
        st.markdown('<div class="fx-sec">📰 Sentiment tin tức</div>', unsafe_allow_html=True)
        sa  = sent["action"]
        sc  = "#4ade80" if sa=="BUY" else "#ef4444" if sa=="SELL" else "#fbbf24"
        sem = "📈" if sa=="BUY" else "📉" if sa=="SELL" else "⏸"
        slbl= "TĂNG" if sa=="BUY" else "GIẢM" if sa=="SELL" else "TRUNG TÍNH"
        st.markdown(f"""
        <div class="fx-sent">
          <div class="fx-sent-top">
            <div>
              <div class="fx-sent-title">Sentiment tổng hợp</div>
              <div class="fx-sent-act" style="color:{sc}">{sem} {slbl} · {sent["confidence"]}%</div>
              <div class="fx-sent-sub">KT {sig_s["score"]:+d} + News {nb:+d} = <b style="color:{sc}">{csc:+d}</b></div>
            </div>
            <div style="font-size:28px">{sem}</div>
          </div>
          <div class="fx-sent-counts">
            <div class="fx-sent-cnt">
              <div class="fx-sent-cnt-n" style="color:#4ade80">{sent["bull_count"]}</div>
              <div class="fx-sent-cnt-l">Tích cực</div>
            </div>
            <div class="fx-sent-cnt">
              <div class="fx-sent-cnt-n" style="color:#ef4444">{sent["bear_count"]}</div>
              <div class="fx-sent-cnt-l">Tiêu cực</div>
            </div>
            <div class="fx-sent-cnt">
              <div class="fx-sent-cnt-n" style="color:#fbbf24">{sent["neutral_count"]}</div>
              <div class="fx-sent-cnt-l">Trung tính</div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

        # Tin tức chi tiết
        if news:
            with st.expander(f"📋 {len(news)} tin tức · bấm để xem"):
                for n in news:
                    c2 = "bull" if n["sentiment"]=="bullish" else "bear" if n["sentiment"]=="bearish" else "neut"
                    tag= "TĂNG" if n["sentiment"]=="bullish" else "GIẢM" if n["sentiment"]=="bearish" else "NGANG"
                    st.markdown(f"""
                    <div class="fx-news {c2}">
                      <div class="fx-news-title">{n["title"]}</div>
                      <div class="fx-news-foot">
                        <span class="fx-news-time">{n["time"][:16]}</span>
                        <span class="fx-news-tag {c2}">{tag} {n["score"]:+d}</span>
                      </div>
                    </div>""", unsafe_allow_html=True)

        # ─── CHART ───
        st.markdown('<div class="fx-sec">📈 Biểu đồ nến</div>', unsafe_allow_html=True)
        fig = build_chart(df, pair)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📋 OHLCV gần nhất"):
            st.dataframe(
                df.tail(10)[["Open","High","Low","Close","RSI","MACD","EMA_20"]].round(5),
                use_container_width=True,
            )

    # Gọi live_dashboard
    live_dashboard()

    # Auto-refresh mỗi 3s
    # Giá tức thì: ExchangeRate API (không cache)
    # OHLCV + chỉ báo: Twelve Data (cache 60s — tự hết hạn)
    time.sleep(1)
    st.rerun()

    # ── AI Phân tích (ngoài fragment vì chỉ chạy khi bấm nút) ──
    if use_ai and analyze_btn:
        st.markdown("---")
        st.markdown("**🤖 Phân tích AI — Groq Llama 3.3 70B (Miễn phí)**")
        with st.spinner("Đang phân tích..."):
            df_ai_raw, _ = fetch_ohlcv(pair, tf_label)
            if df_ai_raw.empty:
                st.error("❌ Không thể tải dữ liệu để phân tích AI.")
                return
            df_ai = df_ai_raw.copy()
            df_ai = add_indicators(df_ai)
            sig_ai = compute_signal(df_ai)
            analysis = get_ai_analysis(pair, tf_short, sig_ai)
        st.markdown(f"""
        <div class="fx-ai-box">
          <div style="font-size:11px;font-weight:700;letter-spacing:1.5px;
                      text-transform:uppercase;color:#3b82f6;margin-bottom:10px;font-family:JetBrains Mono,monospace">
            🤖 GROQ AI ANALYSIS
          </div>
          {analysis}
        </div>
        """, unsafe_allow_html=True)
    elif not analyze_btn:
        st.info("👆 Bấm **⚡ Phân tích ngay** để xem nhận định AI.")


if __name__ == "__main__":
    main()
