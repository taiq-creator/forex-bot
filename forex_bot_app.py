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

# ── Cấu hình trang ───────────────────────────────────────
st.set_page_config(
    page_title="ForexAI Bot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS giao diện sáng ───────────────────────────────────
st.markdown("""
<style>
  /* ── Nền sáng tổng thể ── */
  .stApp {
    background-color: #f0f4f8;
    color: #1a2332;
    font-family: 'Inter', sans-serif;
  }
  section[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #1e3a5f 0%, #0f2440 100%);
    border-right: 1px solid #2a4a7f;
  }

  /* ── Sidebar text ── */
  section[data-testid="stSidebar"] * { color: #c8dff5 !important; }
  section[data-testid="stSidebar"] h1,
  section[data-testid="stSidebar"] h2,
  section[data-testid="stSidebar"] h3 { color: #ffffff !important; }
  section[data-testid="stSidebar"] .stSelectbox label,
  section[data-testid="stSidebar"] .stCheckbox label { color: #a0c4e8 !important; font-size: 13px; }

  /* Selectbox trong sidebar */
  section[data-testid="stSidebar"] .stSelectbox > div > div {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    color: #ffffff !important;
    border-radius: 8px !important;
  }

  /* ── Tiêu đề main ── */
  h1, h2, h3 { color: #0d47a1 !important; font-family: 'Inter', sans-serif; font-weight: 700; }
  p, span, div { color: #1a2332; }

  /* ── Metric cards ── */
  div[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #dce8f5;
    border-radius: 12px;
    padding: 14px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
  }
  div[data-testid="metric-container"] label { color: #5a7a9a !important; font-size: 12px; font-weight: 500; }
  div[data-testid="metric-container"] div[data-testid="metric-value"] {
    color: #0d47a1 !important;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
  }
  div[data-testid="metric-container"] div[data-testid="metric-delta"] { font-size: 12px; }

  /* ── Signal boxes ── */
  .signal-buy {
    background: linear-gradient(135deg, #e8f5e9, #f1f8e9);
    border: 2px solid #43a047;
    border-radius: 14px;
    padding: 22px;
    text-align: center;
    box-shadow: 0 4px 16px rgba(67,160,71,0.15);
  }
  .signal-sell {
    background: linear-gradient(135deg, #ffebee, #fce4ec);
    border: 2px solid #e53935;
    border-radius: 14px;
    padding: 22px;
    text-align: center;
    box-shadow: 0 4px 16px rgba(229,57,53,0.15);
  }
  .signal-neutral {
    background: linear-gradient(135deg, #fff8e1, #fffde7);
    border: 2px solid #f9a825;
    border-radius: 14px;
    padding: 22px;
    text-align: center;
    box-shadow: 0 4px 16px rgba(249,168,37,0.15);
  }

  /* ── Nút bấm ── */
  .stButton > button {
    background: linear-gradient(135deg, #1565c0, #0d47a1) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    width: 100% !important;
    padding: 12px !important;
    font-size: 14px !important;
    letter-spacing: 0.3px !important;
    transition: all 0.2s !important;
  }
  .stButton > button:hover {
    background: linear-gradient(135deg, #1976d2, #1565c0) !important;
    box-shadow: 0 6px 20px rgba(13,71,161,0.35) !important;
    transform: translateY(-1px) !important;
  }

  /* ── Card thông tin chỉ báo ── */
  .ind-card {
    background: #ffffff;
    border: 1px solid #dce8f5;
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 7px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    transition: box-shadow 0.2s;
  }
  .ind-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
  .ind-card.bullish { border-left: 4px solid #43a047; background: #f1f8f1; }
  .ind-card.bearish { border-left: 4px solid #e53935; background: #fff5f5; }
  .ind-card.neutral { border-left: 4px solid #f9a825; background: #fffdf0; }

  /* ── AI analysis box ── */
  .ai-box {
    background: linear-gradient(135deg, #e3f2fd, #f0f7ff);
    border: 1px solid #90caf9;
    border-left: 4px solid #1565c0;
    border-radius: 12px;
    padding: 20px 24px;
    line-height: 1.8;
    font-size: 14px;
    color: #1a2332;
    box-shadow: 0 2px 12px rgba(21,101,192,0.08);
  }

  /* ── Divider ── */
  hr { border-color: #dce8f5 !important; margin: 16px 0 !important; }

  /* ── Expander ── */
  .streamlit-expanderHeader {
    background: #ffffff !important;
    border: 1px solid #dce8f5 !important;
    border-radius: 8px !important;
    color: #0d47a1 !important;
    font-weight: 600 !important;
  }

  /* ── Dataframe ── */
  .stDataFrame { border: 1px solid #dce8f5; border-radius: 10px; overflow: hidden; }

  /* ── Code ── */
  code { background: #e8f0fe !important; color: #1565c0 !important; border-radius: 4px; padding: 2px 6px; }

  /* ── Caption / footer ── */
  .stCaption { color: #7a9ab5 !important; font-size: 11px; }

  /* ── Warning/info boxes ── */
  .stAlert { border-radius: 10px !important; }

  /* Ẩn watermark */
  #MainMenu, footer { visibility: hidden; }
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

# interval: (twelve_data_interval, yf_interval, yf_period, outputsize)
TIMEFRAMES = {
    "M15 (15 phút)": ("15min", "15m",  "5d",   200),
    "H1 (1 giờ)":    ("1h",   "1h",   "10d",  200),
    "H4 (4 giờ)":    ("4h",   "1h",   "30d",  200),
    "D1 (Ngày)":     ("1day", "1d",   "180d", 200),
    "W1 (Tuần)":     ("1week","1wk",  "730d", 100),
}

@st.cache_data(ttl=20)
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


def fetch_realtime_price(pair_name: str) -> float | None:
    """
    Lấy giá tức thì từ ExchangeRate-API — miễn phí, không cần key, cập nhật mỗi 3s.
    Hỗ trợ Forex + Kim loại. Crypto dùng Binance.
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


@st.cache_data(ttl=60)   # cache 60 giây — tiết kiệm credits
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
            st.error(f"❌ Twelve Data: {data.get('message', data)}")
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
    Ưu tiên Twelve Data, fallback Yahoo Finance.
    Trả về (DataFrame, source_name)
    """
    td_sym, yf_ticker, _ = PAIRS[pair_name]
    td_interval, yf_interval, yf_period, outputsize = TIMEFRAMES[tf_label]

    # Thử Twelve Data
    df = fetch_ohlcv_twelvedata(td_sym, td_interval, outputsize)
    if not df.empty:
        return df, "Twelve Data ⚡"

    # Fallback Yahoo Finance
    df = fetch_ohlcv_yahoo(yf_ticker, yf_interval, yf_period)
    if not df.empty:
        # Resample 4H nếu cần (Yahoo không có 4H native)
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
#  TÍNH TÍN HIỆU
# ════════════════════════════════════════════════════════

def compute_signal(df: pd.DataFrame) -> dict:
    """Tổng hợp tín hiệu từ các chỉ báo."""
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last

    signals = {}
    score = 0   # dương → mua, âm → bán

    # RSI
    rsi = last["RSI"]
    if rsi < 30:
        signals["RSI"] = ("🟢 Quá bán", "bullish");  score += 2
    elif rsi > 70:
        signals["RSI"] = ("🔴 Quá mua", "bearish");  score -= 2
    else:
        signals["RSI"] = ("⚪ Trung tính", "neutral")

    # MACD
    if last["MACD"] > last["MACD_signal"] and prev["MACD"] <= prev["MACD_signal"]:
        signals["MACD"] = ("🟢 Cắt lên (BUY)", "bullish"); score += 3
    elif last["MACD"] < last["MACD_signal"] and prev["MACD"] >= prev["MACD_signal"]:
        signals["MACD"] = ("🔴 Cắt xuống (SELL)", "bearish"); score -= 3
    elif last["MACD"] > last["MACD_signal"]:
        signals["MACD"] = ("🟢 MACD trên Signal", "bullish"); score += 1
    else:
        signals["MACD"] = ("🔴 MACD dưới Signal", "bearish"); score -= 1

    # EMA
    c = last["Close"]
    if last["EMA_20"] > last["EMA_50"]:
        signals["EMA 20/50"] = ("🟢 EMA20 > EMA50", "bullish"); score += 1
    else:
        signals["EMA 20/50"] = ("🔴 EMA20 < EMA50", "bearish"); score -= 1

    if c > last["EMA_200"]:
        signals["EMA 200"] = ("🟢 Giá trên EMA200", "bullish"); score += 1
    else:
        signals["EMA 200"] = ("🔴 Giá dưới EMA200", "bearish"); score -= 1

    # Bollinger
    if c < last["BB_lower"]:
        signals["Bollinger"] = ("🟢 Chạm đáy BB", "bullish"); score += 2
    elif c > last["BB_upper"]:
        signals["Bollinger"] = ("🔴 Chạm đỉnh BB", "bearish"); score -= 2
    else:
        signals["Bollinger"] = ("⚪ Trong BB", "neutral")

    # Stochastic
    sk = last["Stoch_K"]
    if sk < 20:
        signals["Stochastic"] = ("🟢 Quá bán", "bullish"); score += 1
    elif sk > 80:
        signals["Stochastic"] = ("🔴 Quá mua", "bearish"); score -= 1
    else:
        signals["Stochastic"] = ("⚪ Trung tính", "neutral")

    # Tổng hợp
    if score >= 4:
        action = "BUY"
    elif score <= -4:
        action = "SELL"
    else:
        action = "NEUTRAL"

    max_score = 10
    confidence = min(95, int(abs(score) / max_score * 100) + 40)

    # TP / SL
    atr = last["ATR"]
    entry = float(last["Close"])
    if action == "BUY":
        tp = entry + atr * 2.0
        sl = entry - atr * 1.0
    elif action == "SELL":
        tp = entry - atr * 2.0
        sl = entry + atr * 1.0
    else:
        tp = entry + atr * 1.5
        sl = entry - atr * 1.0

    return {
        "action": action,
        "confidence": confidence,
        "score": score,
        "signals": signals,
        "entry": entry,
        "tp": tp,
        "sl": sl,
        "rsi": float(rsi),
        "macd": float(last["MACD"]),
        "atr": float(atr),
        "stoch_k": float(sk),
    }


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
    <div style="display:flex;align-items:center;gap:14px;margin-bottom:6px;
                background:linear-gradient(135deg,#1565c0,#0d47a1);
                padding:18px 24px;border-radius:14px;
                box-shadow:0 4px 20px rgba(13,71,161,0.2)">
      <div style="width:12px;height:12px;background:#69f0ae;border-radius:50%;
                  box-shadow:0 0 10px #69f0ae;flex-shrink:0"></div>
      <h1 style="margin:0;font-size:26px;letter-spacing:-0.5px;color:#ffffff !important">
        FOREX<span style="color:#90caf9;font-weight:400">AI</span> Bot
      </h1>
      <div style="background:rgba(105,240,174,0.15);border:1px solid rgba(105,240,174,0.4);
                  color:#69f0ae;padding:4px 12px;border-radius:20px;font-size:10px;
                  letter-spacing:1.5px;font-family:monospace;font-weight:700">● LIVE</div>
      <div style="margin-left:auto;font-family:monospace;font-size:11px;color:rgba(255,255,255,0.6)">
        Twelve Data · Groq AI
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("### ⚙️ Cài đặt")

        pair = st.selectbox("Cặp tiền tệ", list(PAIRS.keys()), index=0)
        tf_label = st.selectbox("Khung thời gian", list(TIMEFRAMES.keys()), index=2)
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
    # LIVE DASHBOARD: cập nhật mỗi 3 giây
    # ══════════════════════════════════════════════
    def live_dashboard():
        # Tải dữ liệu mới
        df_raw, data_source = fetch_ohlcv(pair, tf_label)
        if df_raw.empty:
            st.error("❌ Không thể tải dữ liệu. Kiểm tra API key hoặc kết nối mạng.")
            return

        df = df_raw.copy()
        df = add_indicators(df)
        if len(df) < 30:
            st.warning("⚠️ Không đủ dữ liệu.")
            return

        sig = compute_signal(df)
        last = df.iloc[-1]
        prev = df.iloc[-2]

        # Giá realtime từ ExchangeRate API
        realtime_price = fetch_realtime_price(pair)
        price = realtime_price if realtime_price else float(last["Close"])
        change_pct = (price - float(prev["Close"])) / float(prev["Close"]) * 100
        now_str = datetime.now().strftime("%H:%M:%S")

        # ── Thanh LIVE ──
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:10px;
                    background:#e8f5e9;border:1px solid #a5d6a7;
                    border-radius:8px;padding:7px 14px;margin-bottom:12px">
          <div style="width:8px;height:8px;background:#43a047;border-radius:50%;
                      animation:pulse 1s infinite;flex-shrink:0"></div>
          <span style="font-family:monospace;font-size:12px;color:#2e7d32;font-weight:600">
            ⚡ LIVE · {data_source} · {now_str} · {pair} {tf_short}
          </span>
        </div>
        <style>@keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:0.4}}}}</style>
        """, unsafe_allow_html=True)

        # ── Metrics ──
        st.markdown(f"### 📍 {pair} · {tf_short}")
        c1, c2, c3, c4, c5 = st.columns(5)
        price_label = "⚡ Giá realtime" if realtime_price else "Giá (60s)"
        c1.metric(price_label, fmt(price), f"{change_pct:+.2f}%")
        c2.metric("High (kỳ)", fmt(float(last["High"])))
        c3.metric("Low (kỳ)", fmt(float(last["Low"])))
        c4.metric("ATR", fmt(float(last["ATR"])) if not np.isnan(last["ATR"]) else "N/A")
        c5.metric("Cập nhật", now_str)

        st.markdown("---")

        # ── Tín hiệu + Chỉ báo ──
        col_sig, col_ind = st.columns([1, 2])
        with col_sig:
            action = sig["action"]
            conf   = sig["confidence"]
            css    = "signal-buy" if action=="BUY" else "signal-sell" if action=="SELL" else "signal-neutral"
            emoji  = "📈" if action=="BUY" else "📉" if action=="SELL" else "⏸"
            color  = "#2e7d32" if action=="BUY" else "#c62828" if action=="SELL" else "#e65100"
            label  = "TÍN HIỆU MUA" if action=="BUY" else "TÍN HIỆU BÁN" if action=="SELL" else "TRUNG TÍNH"
            st.markdown(f"""
            <div class="{css}">
              <div style="font-size:11px;font-family:monospace;color:#4a7a99;
                          letter-spacing:2px;text-transform:uppercase;margin-bottom:8px">
                Tín hiệu giao dịch
              </div>
              <div style="font-size:36px;font-weight:800;color:{color};letter-spacing:-1px">
                {emoji} {label}
              </div>
              <div style="font-size:22px;font-weight:700;color:{color};margin-top:4px">
                {conf}% Tin cậy
              </div>
              <div style="font-family:monospace;font-size:11px;color:#4a7a99;margin-top:8px">
                Score: {sig['score']:+d} / 10 | {pair} {tf_short}
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**📊 Mức giá giao dịch**")
            g1, g2, g3 = st.columns(3)
            g1.metric("📍 Entry",       fmt(sig["entry"]))
            g2.metric("🎯 Take Profit", fmt(sig["tp"]))
            g3.metric("🛑 Stop Loss",   fmt(sig["sl"]))
            rr = abs(sig["tp"]-sig["entry"]) / max(abs(sig["sl"]-sig["entry"]), 1e-10)
            st.caption(f"Risk/Reward ratio: 1:{rr:.1f}")

        with col_ind:
            st.markdown("**🔧 Chỉ báo kỹ thuật chi tiết**")
            ind_data = {
                "RSI (14)":     (f"{sig['rsi']:.1f}",   sig["signals"].get("RSI",("",""))[0]),
                "MACD":         (f"{sig['macd']:.5f}",  sig["signals"].get("MACD",("",""))[0]),
                "EMA 20/50":    (f"{fmt(float(last['EMA_20']))} / {fmt(float(last['EMA_50']))}",
                                 sig["signals"].get("EMA 20/50",("",""))[0]),
                "EMA 200":      (fmt(float(last["EMA_200"])), sig["signals"].get("EMA 200",("",""))[0]),
                "Bollinger":    (f"{fmt(float(last['BB_upper']))} / {fmt(float(last['BB_lower']))}",
                                 sig["signals"].get("Bollinger",("",""))[0]),
                "Stochastic %K":(f"{sig['stoch_k']:.1f}", sig["signals"].get("Stochastic",("",""))[0]),
                "ATR (14)":     (fmt(sig["atr"]), "📊 Biến động"),
            }
            for name, (val, signal_str) in ind_data.items():
                card_cls = ("ind-card bullish" if "🟢" in signal_str
                            else "ind-card bearish" if "🔴" in signal_str
                            else "ind-card neutral")
                st.markdown(f"""
                <div class="{card_cls}">
                  <span style="color:#5a7a9a;font-family:monospace;font-size:12px;font-weight:500">{name}</span>
                  <span style="color:#0d47a1;font-family:monospace;font-size:13px;font-weight:700">{val}</span>
                  <span style="font-size:12px">{signal_str}</span>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Biểu đồ ──
        st.markdown("**📈 Biểu đồ nến**")
        fig = build_chart(df, pair)
        st.plotly_chart(fig, use_container_width=True)

        # ── OHLCV table ──
        with st.expander("📋 Dữ liệu OHLCV gần nhất"):
            st.dataframe(
                df.tail(20)[["Open","High","Low","Close","Volume",
                             "RSI","MACD","EMA_20","EMA_50"]].round(5),
                use_container_width=True,
            )

        # ── Footer ──
        st.markdown(f"""
        <div style="text-align:center;font-family:monospace;font-size:11px;color:#7a9ab5;
                    background:#ffffff;border:1px solid #dce8f5;border-radius:8px;
                    padding:10px;margin-top:8px">
          📊 ForexAI Bot &nbsp;·&nbsp; {data_source} &nbsp;·&nbsp;
          Groq Llama 3 (Free) &nbsp;·&nbsp; 🕐 {now_str}
        </div>
        """, unsafe_allow_html=True)

    # Gọi live_dashboard
    live_dashboard()

    # Auto-refresh mỗi 3s
    # Giá tức thì: ExchangeRate API (không cache)
    # OHLCV + chỉ báo: Twelve Data (cache 60s — tự hết hạn)
    time.sleep(3)
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
        <div class="ai-box">
          <div style="font-size:11px;font-weight:700;letter-spacing:1.5px;
                      text-transform:uppercase;color:#1565c0;margin-bottom:10px;font-family:monospace">
            🤖 GROQ AI ANALYSIS
          </div>
          {analysis}
        </div>
        """, unsafe_allow_html=True)
    elif not analyze_btn:
        st.info("👆 Bấm **⚡ Phân tích ngay** để xem nhận định AI.")


if __name__ == "__main__":
    main()
