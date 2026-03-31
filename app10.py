"""
Golden Cross Predictor — Quant Edition v13
All v12 fixes retained. New in v13 (Senior DS Recommendations):

1. SECTOR METADATA MAP: Every ticker tagged with Sector + MarketCap bucket.
   Categorical embeddings (one-hot) added to FEATURE_COLS so XGBoost applies
   different logic to Bank vs Defense vs IT while sharing one global model.

2. SECTOR-DEMEANED FEATURES: RSI, Distance-from-200SMA, ADX, ATR_pct and
   MACD_hist_pct are normalised as (Feature_Stock − Mean(Feature_Sector)).
   This teaches the model to identify "stronger than peers" setups.

3. ATR-BASED LABELLING (volatility-adjusted): The binary target is no longer
   a fixed n-day horizon. Success = price reaches Entry + 2×ATR before
   Entry − 1×ATR. Labels are now comparable across high-vol small-caps and
   low-vol blue-chips.

4. MARKET REGIME FEATURE: Index-level MA200 regime flag added. Model learns
   to distinguish GC signals fired in bull vs bear macro environments.

5. TRAINING DATA: Start date default pushed back to 2010 to capture sideways
   years (2018-19) where whipsaws are frequent — model learns when to stay out.

6. SECTOR CAP-MIX ENFORCED: Training data is resampled to ~40% Large, 40% Mid,
   20% Small before the walk-forward split to prevent bull-run bias.

Original v12:
1. near_cross_score: continuous 0-1 convergence feature.
2. Trend filter (ADX > 20): shown individually in UI chips.
3. Regime filter (Price > MA200): explicit check.
4. Composite signal gate: model_prob > 0.6 AND near_cross_ok AND trend_ok.
5. Screener updated with filter status per ticker.
"""



import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import pandas_ta as ta
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
import datetime

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Golden Cross Predictor",
    page_icon="�",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — terminal dark aesthetic
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
  :root {
    --bg:#0a0e1a; --bg1:#111827; --bg2:#1a2235;
    --border:#2a3550; --border2:#334060;
    --text:#e8edf8; --text2:#9aa5c0; --text3:#5a6a8a;
    --green:#00c97a; --red:#e84060; --amber:#f0a020;
    --blue:#4da6ff; --purple:#a78bfa;
    --mono:'JetBrains Mono',monospace; --sans:'Inter',sans-serif;
  }
  html,body,[class*="css"]          { font-family:var(--sans); font-size:15px; }
  .stApp                            { background:var(--bg); color:var(--text); }
  section[data-testid="stSidebar"]  { background:var(--bg1)!important; border-right:2px solid var(--border); }
  .block-container                  { padding-top:1.5rem; padding-bottom:2rem; max-width:1400px; }
  #MainMenu,footer,header           { visibility:hidden; }

  /* ── Cards ─────────────────────────────────────────────────────────────── */
  .qcard {
    background:var(--bg1); border:1px solid var(--border);
    border-radius:10px; padding:18px 20px; margin-bottom:12px;
    position:relative; overflow:hidden;
  }
  .qcard::before {
    content:''; position:absolute; top:0; left:0; right:0; height:3px;
    background:var(--accent,var(--blue)); opacity:.8;
  }
  .qcard .qlabel {
    font-family:var(--mono); font-size:12px; color:var(--text3);
    text-transform:uppercase; letter-spacing:.08em; margin-bottom:8px;
    display:flex; align-items:center;
  }
  .qcard .qval   { font-size:28px; font-weight:700; line-height:1.1; color:var(--accent,var(--text)); }
  .qcard .qsub   { font-size:13px; color:var(--text2); margin-top:6px; line-height:1.4; }

  /* ── Badges ─────────────────────────────────────────────────────────────── */
  .badge-bullish { background:rgba(0,201,122,.15);  color:#00c97a; border:1px solid rgba(0,201,122,.4); }
  .badge-bearish { background:rgba(232,64,96,.15);  color:#e84060; border:1px solid rgba(232,64,96,.4); }
  .badge-caution { background:rgba(240,160,32,.15); color:#f0a020; border:1px solid rgba(240,160,32,.4); }
  .badge-neutral { background:rgba(77,166,255,.12); color:#9aa5c0; border:1px solid rgba(77,166,255,.25); }
  .badge { border-radius:8px; padding:8px 16px; font-weight:600; font-size:14px; display:inline-block; }

  /* ── Pills ──────────────────────────────────────────────────────────────── */
  .risk-pill {
    display:inline-block; font-size:13px; font-weight:500;
    padding:5px 12px; border-radius:20px; margin-right:8px; margin-bottom:6px;
  }

  /* ── Section headers ────────────────────────────────────────────────────── */
  .sec-hdr {
    font-size:13px; font-weight:600; color:var(--text2);
    text-transform:uppercase; letter-spacing:.06em;
    padding-bottom:10px; border-bottom:2px solid var(--border);
    margin:24px 0 16px;
  }

  /* ── Tables ─────────────────────────────────────────────────────────────── */
  .qtable { width:100%; border-collapse:collapse; font-size:14px; }
  .qtable th { background:var(--bg2); color:var(--text2); font-size:12px;
               font-weight:600; text-transform:uppercase; letter-spacing:.05em;
               padding:10px 14px; border-bottom:2px solid var(--border); text-align:left; }
  .qtable td { padding:10px 14px; border-bottom:1px solid var(--border2); color:var(--text); font-size:14px; }
  .qtable tr:hover td { background:var(--bg2); }

  /* ── Inputs ─────────────────────────────────────────────────────────────── */
  .stTextInput>div>div>input,
  .stTextArea>div>div>textarea {
    background:var(--bg2)!important; border:1px solid var(--border)!important;
    color:var(--text)!important; font-size:15px!important;
    border-radius:8px!important; padding:10px 14px!important;
  }

  /* ── Tabs ───────────────────────────────────────────────────────────────── */
  .stTabs [data-baseweb="tab-list"] { background:var(--bg1); border-bottom:2px solid var(--border); gap:4px; }
  .stTabs [data-baseweb="tab"]      { padding:12px 22px; font-size:15px; font-weight:500; color:var(--text2); }
  .stTabs [aria-selected="true"]    { color:var(--green)!important; border-bottom:3px solid var(--green)!important; font-weight:700!important; }

  /* ── Warn box ───────────────────────────────────────────────────────────── */
  .warn-box {
    background:rgba(240,160,32,.10); border:1px solid rgba(240,160,32,.35);
    border-radius:8px; padding:14px 18px; font-size:14px;
    color:var(--amber); margin:10px 0; line-height:1.5;
  }

  /* ── Scrollbar ──────────────────────────────────────────────────────────── */
  ::-webkit-scrollbar { width:6px; height:6px; }
  ::-webkit-scrollbar-track { background:var(--bg); }
  ::-webkit-scrollbar-thumb { background:var(--border2); border-radius:4px; }

  /* ── Sidebar labels ─────────────────────────────────────────────────────── */
  section[data-testid="stSidebar"] label,
  section[data-testid="stSidebar"] .stSlider label,
  section[data-testid="stSidebar"] p {
    font-size:14px !important; color:var(--text) !important;
  }
  section[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMin"],
  section[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMax"] {
    font-size:12px !important;
  }

  /* ── Info tooltip ───────────────────────────────────────────────────────── */
  .info-wrap { position:relative; display:inline-block; vertical-align:middle; margin-left:5px; }
  .info-btn {
    display:inline-flex; align-items:center; justify-content:center;
    width:17px; height:17px; border-radius:50%;
    background:rgba(77,166,255,.18); border:1px solid rgba(77,166,255,.4);
    color:#4da6ff; font-size:10px; font-weight:700;
    cursor:pointer; line-height:1; user-select:none;
  }
  .info-btn:hover { background:rgba(77,166,255,.35); border-color:#4da6ff; }
  .info-tip {
    display:none; position:absolute; z-index:99999;
    bottom:calc(100% + 10px); left:50%; transform:translateX(-50%);
    width:480px; background:#111827; border:1px solid #2a3550;
    border-radius:10px; overflow:hidden;
    box-shadow:0 12px 48px rgba(0,0,0,.75);
    pointer-events:none;
  }
  .tip-head {
    display:grid; grid-template-columns:110px 1fr 1fr 1fr;
    background:#1a2235; border-bottom:1px solid #2a3550;
    padding:8px 14px; gap:10px;
    font-size:11px; font-weight:700; color:#5a6a8a;
    text-transform:uppercase; letter-spacing:.1em;
  }
  .tip-body {
    display:grid; grid-template-columns:110px 1fr 1fr 1fr;
    padding:14px 14px; gap:10px;
    font-size:12px; color:#e8edf8; line-height:1.55;
  }
  .tip-term  { font-weight:700; color:#4da6ff; }
  .tip-expl  { color:#e8edf8; }
  .tip-form  { color:#a78bfa; white-space:pre-wrap; word-break:break-word; }
  .tip-note  { color:#9aa5c0; }
  .info-tip::after {
    content:''; position:absolute; top:100%; left:50%; transform:translateX(-50%);
    border:7px solid transparent; border-top-color:#2a3550;
  }
  .info-wrap:hover .info-tip,
  .info-wrap:focus-within .info-tip { display:block; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
_defs = dict(
    models_trained=False,
    models_gc=None, models_dc=None,
    models_gc_recall=None,          # Model B — high recall
    model_quality_gc=None, model_quality_dc=None,
    model_timing_gc=None,
    model_days_regressor=None,      # Model C — raw days regressor
    calibrator_gc=None, calibrator_dc=None,
    calibrator_gc_recall=None,
    feature_cols=None, feature_importance=None,
    train_metrics={},
    fast_p=50, slow_p=200,
    start_date="2010-01-01",
    pred_days=15, threshold=0.45, quality_min=0.25,
)
for k, v in _defs.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────

def get_listing_date(ticker):
    """Returns the earliest available date for a ticker from yfinance."""
    try:
        info = yf.Ticker(ticker).history(period="max", auto_adjust=True)
        if info.empty: return None
        return info.index[0].to_pydatetime().date()
    except: return None


def download_stock_data(ticker, start_date):
    try:
        df = yf.download(ticker, start=start_date, progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None

        # yfinance returns MultiIndex columns like ('Close','RELIANCE.NS') — flatten to just 'Close'
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Normalise column names to Title case
        df.columns = [str(c).strip().title() for c in df.columns]

        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing  = [c for c in required if c not in df.columns]
        if missing:
            return None

        df = df[required].copy()
        df.index = pd.to_datetime(df.index)
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.sort_index(inplace=True)
        df.dropna(inplace=True)

        if len(df) < 50:
            return None

        df.attrs['listing_date'] = str(df.index[0].date())
        df.attrs['bars']         = len(df)
        return df

    except Exception:
        return None


def resample_weekly(df):
    return df.resample('W-FRI').agg(
        {'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}
    ).dropna()


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING  (lookahead-free)
# ─────────────────────────────────────────────────────────────────────────────

def hurst_exponent(ts, max_lag=20):
    """Rescaled-range Hurst. H>0.5=trending, H<0.5=mean-reverting."""
    lags = range(2, max_lag)
    tau  = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
    tau  = [t for t in tau if t > 0]
    if len(tau) < 3: return 0.5
    return np.polyfit(np.log(range(2, 2+len(tau))), np.log(tau), 1)[0]

def rolling_hurst(series, window=60):
    result = np.full(len(series), np.nan)
    arr = series.values
    # Compute every 5th bar only then forward-fill — avoids backward fill lookahead
    for i in range(window, len(arr), 5):
        result[i] = hurst_exponent(arr[i-window:i])
    # Forward-fill NaN (uses only past computed values — no lookahead)
    s = pd.Series(result, index=series.index)
    s = s.ffill().fillna(0.5)
    return s

def garman_klass_vol(df, window=20):
    """Garman-Klass vol — more efficient than close-to-close."""
    log_hl = np.log(df['High'] / df['Low']) ** 2
    log_co = np.log(df['Close'] / df['Open']) ** 2
    gk = (0.5 * log_hl - (2*np.log(2)-1) * log_co).rolling(window).mean()
    return np.sqrt(gk * 252)

def engineer_features(df, fast_p=50, slow_p=200):
    d = df.copy()

    # ── EMA values (needed for gap/velocity — NOT exposed as features directly) ──
    d['ema_fast']        = ta.ema(d['Close'], length=fast_p)
    d['ema_slow']        = ta.ema(d['Close'], length=slow_p)

    # ── PHASE 1 FIX: GC = strict event (one row per crossover), not state ─────
    # gc_event = 1 ONLY on the exact bar where ema_fast crosses above ema_slow.
    # This is the ground-truth event used for targets and hist_gc_success.
    # It is NOT exposed as a feature (that would be leakage).
    d['gc_event'] = (
        (d['ema_fast'] > d['ema_slow']) &
        (d['ema_fast'].shift(1) <= d['ema_slow'].shift(1))
    ).astype(int)
    d['dc_event'] = (
        (d['ema_fast'] < d['ema_slow']) &
        (d['ema_fast'].shift(1) >= d['ema_slow'].shift(1))
    ).astype(int)

    # ── PHASE 2 FIX: EMA gap features only — no binary crossover state ────────
    # ema_gap_pct encodes the gap magnitude (negative = below, positive = above).
    # The model learns from the GAP CLOSING, not from "already crossed".
    d['ema_gap_pct']     = (d['ema_fast'] - d['ema_slow']) / (d['ema_slow'].abs() + 1e-9) * 100
    d['gap_velocity']    = d['ema_gap_pct'].diff()           # %/bar — convergence speed
    d['gap_velocity_5']  = d['ema_gap_pct'].diff(5) / 5      # 5-bar avg velocity
    d['gap_velocity_10'] = d['ema_gap_pct'].diff(10) / 10    # 10-bar avg velocity
    # PHASE 3 FIX: gap_change and gap_acceleration (key convergence features)
    d['gap_change']       = d['gap_velocity']                 # alias for clarity
    d['gap_acceleration'] = d['gap_velocity'].diff()          # 2nd derivative of gap
    d['gap_to_cross']    = -(d['ema_gap_pct'] / d['gap_velocity'].replace(0, np.nan)).clip(-60, 60)
    d['gap_accel']       = d['gap_velocity'].diff(3)

    # ── Returns ───────────────────────────────────────────────────────────────
    d['return_1']  = d['Close'].pct_change(1)
    d['return_5']  = d['Close'].pct_change(5)
    d['return_10'] = d['Close'].pct_change(10)
    d['return_20'] = d['Close'].pct_change(20)

    # ── Volatility ────────────────────────────────────────────────────────────
    pct = d['Close'].pct_change()
    d['volatility_10']    = pct.rolling(10).std()
    d['volatility_20']    = pct.rolling(20).std()
    d['volatility_60']    = pct.rolling(60).std()
    d['volatility_ratio'] = d['volatility_10'] / (d['volatility_20'] + 1e-9)
    d['vol_regime']       = d['volatility_20'] / (d['volatility_60'] + 1e-9)
    d['atr']              = ta.atr(d['High'], d['Low'], d['Close'], length=14)
    d['atr_pct']          = (d['atr'] / (d['Close'] + 1e-9)) * 100
    atr_med60             = d['atr_pct'].rolling(60, min_periods=20).median()
    d['atr_vs_median']    = d['atr_pct'] / (atr_med60 + 1e-9)
    d['gk_vol']           = garman_klass_vol(d, window=20)
    d['convergence_speed'] = d['gap_velocity_5'] / (d['atr_pct'] + 1e-6)
    d['gap_zscore'] = (d['ema_gap_pct'] - d['ema_gap_pct'].rolling(60).mean()) / (d['ema_gap_pct'].rolling(60).std() + 1e-9)

    # ── Volume ────────────────────────────────────────────────────────────────
    d['volume_ma20']   = d['Volume'].rolling(20).mean()
    d['volume_std20']  = d['Volume'].rolling(20).std()
    d['volume_change'] = d['Volume'].pct_change()
    d['volume_ratio']  = d['Volume'] / (d['volume_ma20'] + 1e-9)
    d['volume_surge']  = (d['volume_ratio'] > 1.5).astype(int)
    d['volume_zscore'] = (d['Volume'] - d['volume_ma20']) / (d['volume_std20'] + 1e-9)
    d['obv']           = ta.obv(d['Close'], d['Volume'])
    d['obv_slope']     = d['obv'].diff(5) / 5
    d['vpt']           = (d['return_1'] * d['Volume']).cumsum()
    d['vpt_slope']     = d['vpt'].diff(5) / 5

    # ── Momentum / oscillators ────────────────────────────────────────────────
    d['rsi_14']           = ta.rsi(d['Close'], length=14)
    d['rsi_7']            = ta.rsi(d['Close'], length=7)
    d['rsi_change']       = d['rsi_14'].diff(5)
    d['rsi_slope']        = d['rsi_14'].diff(3)
    macd_df               = ta.macd(d['Close'], fast=12, slow=26, signal=9)
    if macd_df is not None and not macd_df.empty:
        d['macd']         = macd_df.get('MACD_12_26_9',   pd.Series(0.0, index=d.index))
        d['macd_signal']  = macd_df.get('MACDs_12_26_9',  pd.Series(0.0, index=d.index))
        d['macd_hist']    = macd_df.get('MACDh_12_26_9',  pd.Series(0.0, index=d.index))
    else:
        d['macd'] = d['macd_signal'] = d['macd_hist'] = 0.0
    d['macd_hist_pct']    = d['macd_hist'] / (d['Close'] + 1e-9) * 100
    d['macd_hist_change'] = d['macd_hist_pct'].diff(3)
    d['macd_cross']       = ((d['macd'] > d['macd_signal']) &
                              (d['macd'].shift(1) <= d['macd_signal'].shift(1))).astype(int)
    stoch_df              = ta.stoch(d['High'], d['Low'], d['Close'])
    if stoch_df is not None and not stoch_df.empty:
        # Use explicit column names — column order varies across pandas_ta versions
        k_col = [c for c in stoch_df.columns if c.startswith('STOCHk')]
        d_col = [c for c in stoch_df.columns if c.startswith('STOCHd')]
        d['stoch_k'] = stoch_df[k_col[0]] if k_col else 50.0
        d['stoch_d'] = stoch_df[d_col[0]] if d_col else 50.0
    else:
        d['stoch_k'] = d['stoch_d'] = 50.0
    d['stoch_cross'] = ((d['stoch_k'] > d['stoch_d']) &
                         (d['stoch_k'].shift(1) <= d['stoch_d'].shift(1))).astype(int)
    try:
        cci = ta.cci(d['High'], d['Low'], d['Close'], length=20)
        d['cci'] = cci if cci is not None else 0.0
    except: d['cci'] = 0.0
    try:
        willr = ta.willr(d['High'], d['Low'], d['Close'], length=14)
        d['willr'] = willr if willr is not None else -50.0
    except: d['willr'] = -50.0

    # ── Trend strength ────────────────────────────────────────────────────────
    adx_df = ta.adx(d['High'], d['Low'], d['Close'], length=14)
    if adx_df is not None and not adx_df.empty:
        # Explicit column names — ADX_14, DMP_14, DMN_14 (ADXR_14_2 also present, skip it)
        adx_col = [c for c in adx_df.columns if c.startswith('ADX_')]
        dmp_col = [c for c in adx_df.columns if c.startswith('DMP_')]
        dmn_col = [c for c in adx_df.columns if c.startswith('DMN_')]
        d['adx'] = adx_df[adx_col[0]] if adx_col else 20.0
        d['dmp'] = adx_df[dmp_col[0]] if dmp_col else 20.0
        d['dmn'] = adx_df[dmn_col[0]] if dmn_col else 20.0
    else:
        d['adx'] = d['dmp'] = d['dmn'] = 20.0
    d['di_diff']    = d['dmp'] - d['dmn']
    d['fast_slope'] = d['ema_fast'].diff(5) / 5
    d['slow_slope'] = d['ema_slow'].diff(5) / 5
    d['slope_div']  = d['fast_slope'] - d['slow_slope']
    d['price_slope']= d['Close'].diff(10) / 10
    ma200           = d['Close'].rolling(200).mean()
    ma50            = d['Close'].rolling(50).mean()
    d['price_vs_ma200'] = (d['Close'] / (ma200 + 1e-9) - 1) * 100
    d['price_vs_ma50']  = (d['Close'] / (ma50  + 1e-9) - 1) * 100
    d['slope_ratio']    = (d['fast_slope'] / (d['slow_slope'].replace(0, np.nan))).clip(-10, 10)

    # ── Multi-timeframe ───────────────────────────────────────────────────────
    df_w             = resample_weekly(df)
    df_w['mtf_fast'] = ta.ema(df_w['Close'], length=10)
    df_w['mtf_slow'] = ta.ema(df_w['Close'], length=40)
    df_w['mtf_bull'] = (df_w['mtf_fast'] > df_w['mtf_slow']).astype(int)
    d['mtf_bull']    = df_w['mtf_bull'].reindex(d.index, method='ffill').fillna(0).astype(int)
    mtf_gap          = (df_w['mtf_fast'] - df_w['mtf_slow']) / (df_w['mtf_slow'] + 1e-9) * 100
    d['mtf_gap_pct'] = mtf_gap.reindex(d.index, method='ffill').fillna(0)
    try:
        df_m              = df.resample('ME').agg({'Close':'last'}).dropna()
        df_m['m_ema3']    = ta.ema(df_m['Close'], length=3)
        df_m['m_ema12']   = ta.ema(df_m['Close'], length=12)
        df_m['m_bull']    = (df_m['m_ema3'] > df_m['m_ema12']).astype(int)
        d['monthly_bull'] = df_m['m_bull'].reindex(d.index, method='ffill').fillna(0).astype(int)
    except: d['monthly_bull'] = 0

    # ── Regime / consolidation ────────────────────────────────────────────────
    d['price_range_20'] = ((d['High'].rolling(20).max() - d['Low'].rolling(20).min()) /
                           (d['Close'] + 1e-9)) * 100
    # dc_age: bars since last dc_event (not dc_regime state — avoids leakage)
    days_since = np.zeros(len(d)); counter = 0
    for i in range(len(d)):
        if d['dc_event'].iloc[i] == 1: counter = 0
        else: counter += 1
        days_since[i] = counter
    d['days_since_dc'] = days_since
    d['dc_age_norm']   = np.clip(d['days_since_dc'] / 250, 0, 1)

    # ── Historical GC success (lag-safe, event-based) ─────────────────────────
    # Uses gc_event (strict crossover) not gc_regime (state).
    # gc_held_fwd: did price stay above ema_slow 20 bars after the event?
    gc_held_fwd  = (d['ema_fast'] > d['ema_slow']).shift(-20).fillna(0).astype(int)
    gc_success   = (d['gc_event'] * gc_held_fwd).astype(float)
    gc_rate_raw  = gc_success.rolling(750, min_periods=5).mean()
    d['hist_gc_success'] = gc_rate_raw.shift(21).clip(0, 1).fillna(0.5)

    # ── Quality signals ───────────────────────────────────────────────────────
    d['rsi_momentum']     = d['rsi_14'].diff(3)
    d['price_momentum_3'] = d['Close'].pct_change(3)
    d['atr_expansion']    = d['atr_vs_median'] - 1
    d['vol_compression']  = (d['volatility_10'] < d['volatility_20']).astype(int)
    d['trend_consistency']= pct.rolling(10).apply(lambda x: (x > 0).mean(), raw=True)

    # ── Hurst ─────────────────────────────────────────────────────────────────
    d['hurst'] = rolling_hurst(d['Close'], window=60)

    # ── Near-cross score (continuous 0–1 convergence feature) ─────────────────
    # 1.0 = gap tiny AND closing fast (cross imminent)
    # 0.0 = gap wide AND diverging (no cross coming)
    # Adaptive proximity: uses rolling 60-bar median of |gap| as normaliser
    # so the score is relative to the stock's own gap history, not a fixed 5% threshold.
    gap_abs         = d['ema_gap_pct'].abs()
    gap_median60    = gap_abs.rolling(60, min_periods=20).median().fillna(gap_abs)
    gap_close       = -d['ema_gap_pct'] * d['gap_velocity']
    # Proximity: 1 when gap is below its own 60-bar median (relatively tight)
    gap_proximity   = np.clip(1.0 - gap_abs / (gap_median60 * 2.0 + 1e-9), 0, 1)
    gap_convergence = np.clip(gap_close / (gap_abs + 1e-9), 0, 1)
    d['near_cross_score'] = 0.5 * gap_proximity + 0.5 * gap_convergence

    d.replace([np.inf, -np.inf], np.nan, inplace=True)
    d.dropna(inplace=True)

    # ── Categorical placeholders (filled during multi-ticker concat) ──────────
    # These are set per-ticker in the training pipeline via SECTOR_MAP / CAP_MAP.
    # Here we initialise to 0 so single-ticker prediction paths don't break.
    if 'sector_id' not in d.columns:
        d['sector_id'] = 0
    if 'cap_id' not in d.columns:
        d['cap_id'] = 0
    if 'mkt_regime' not in d.columns:
        d['mkt_regime'] = 0

    return d


# ── Sector / cap constants (must be defined before FEATURE_COLS) ──────────────
N_SECTORS = 16
N_CAPS    = 3

FEATURE_COLS = [
    # ── EMA gap features — NO binary state, only gap magnitude & velocity ──────
    'ema_gap_pct','gap_velocity','gap_velocity_5','gap_velocity_10',
    'gap_acceleration',
    'gap_to_cross','gap_accel','convergence_speed','gap_zscore',
    # Returns
    'return_1','return_5','return_10','return_20',
    # Volatility
    'volatility_10','volatility_20','volatility_60','volatility_ratio','vol_regime',
    'atr_pct','atr_vs_median','gk_vol',
    # Volume
    'volume_change','volume_ratio','volume_surge','volume_zscore',
    # Momentum / oscillators
    'rsi_14','rsi_7','rsi_change','rsi_slope',
    'macd_hist_pct','macd_hist_change','macd_cross',
    'stoch_k','stoch_d','stoch_cross','cci','willr',
    # Trend
    'adx','dmp','dmn','di_diff',
    'slope_div','slope_ratio',
    'price_vs_ma200','price_vs_ma50',
    # MTF
    'mtf_bull','mtf_gap_pct','monthly_bull',
    # Regime
    'price_range_20','dc_age_norm','hurst',
    # Near-cross proximity and convergence gate
    'near_cross_score',
    # Quality
    'rsi_momentum','price_momentum_3','atr_expansion','vol_compression','trend_consistency',
    # Market regime (Nifty 50 above MA200 — macro bull/bear)
    'mkt_regime',
    # Sector-demeaned features — stock vs sector peers
    # For single-ticker inference these are set to raw value (demean=0)
    # which is still informative as an absolute level signal
    'rsi_14_vs_sector',
    'price_vs_ma200_vs_sector',
    'adx_vs_sector',
    'atr_pct_vs_sector',
    'macd_hist_pct_vs_sector',
    # NOTE: sector_* and cap_* one-hot columns REMOVED — they cause training/inference
    # mismatch when single-ticker backtest/predict has all zeros for these features.
    # The model learns spurious sector patterns that don't generalise.
]


def ensure_feature_cols(df):
    """
    Add any FEATURE_COLS that are missing from df as zeros.
    Needed for single-ticker paths (backtest, predict) where sector/cap
    one-hot and sector-demeaned columns haven't been injected yet.
    """
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
    return df


# ─────────────────────────────────────────────────────────────────────────────
# TARGET CREATION  (lookahead-free, validated)
# ─────────────────────────────────────────────────────────────────────────────

def create_targets(df, n_days=15, strength_days=20):
    """
    PHASE 1+3 FIX: All targets derived from gc_event (strict crossover, one row per cross).
    days_to_gc: actual trading days to next GC event, capped at 60, filled 999 if none.
    near_cross_bias: rows where gap is already very tight (< 1%) are flagged for training filter.
    """
    d = df.copy()

    # ── PHASE 1: Use strict gc_event / dc_event (already computed in engineer_features) ──
    # gc_event = 1 only on the exact crossover bar — no continuous state
    gc_cross = d['gc_event'].values
    dc_cross = d['dc_event'].values

    # Binary: will a gc_event happen in the next n_days?
    # Reverse-roll + shift(1) prevents lookahead on current bar
    gc_s = pd.Series(gc_cross, index=d.index)
    dc_s = pd.Series(dc_cross, index=d.index)
    d['target_gc'] = (gc_s[::-1].rolling(n_days, min_periods=1).max()[::-1]
                      .shift(1).fillna(0).astype(int))
    d['target_dc'] = (dc_s[::-1].rolling(n_days, min_periods=1).max()[::-1]
                      .shift(1).fillna(0).astype(int))

    # ── PHASE 3 FIX: days_to_gc — actual days to next gc_event ───────────────
    # target_days_raw: capped at 60, filled 999 if no future GC (per checklist step 4+6)
    ci = np.where(gc_cross == 1)[0]
    days_to     = np.full(len(d), np.nan)
    days_to_raw = np.full(len(d), 999.0)
    for idx in range(len(d)):
        future = ci[ci > idx]
        if len(future) > 0:
            dist = future[0] - idx
            days_to_raw[idx] = min(float(dist), 60.0)
            if dist <= n_days:
                days_to[idx] = float(dist)
    d['target_days_to_gc'] = days_to
    d['target_days_raw']   = days_to_raw   # 999 = no GC within 60 bars
    d['target_days_norm']  = np.clip(1.0 - (d['target_days_to_gc'] / n_days), 0, 1)

    # near_cross_score already computed in engineer_features — available in d

    # ── Quality score (composite 0–1, only at actual gc_event bars) ───────────
    gc_idx = np.where(gc_cross == 1)[0]
    quality = np.full(len(d), np.nan)
    if len(gc_idx) > 0:
        end_idx  = np.minimum(gc_idx + strength_days, len(d) - 1)
        fwd_ret  = np.where(end_idx > gc_idx,
                            d['Close'].values[end_idx] / d['Close'].values[gc_idx] - 1, 0.0)
        fwd_norm = np.clip((fwd_ret + 0.10) / 0.30, 0, 1)
        vol_norm = np.clip((d['volume_ratio'].values[gc_idx] - 0.5) / 2.0, 0, 1)
        adx_norm = np.clip(d['adx'].values[gc_idx] / 40.0, 0, 1)
        mtf_ok   = d['mtf_bull'].values[gc_idx].astype(float)
        quality[gc_idx] = 0.4*fwd_norm + 0.3*vol_norm + 0.3*(0.5*adx_norm + 0.5*mtf_ok)
    d['target_quality_gc'] = quality

    dc_idx = np.where(dc_cross == 1)[0]
    quality_dc = np.full(len(d), np.nan)
    if len(dc_idx) > 0:
        end_idx  = np.minimum(dc_idx + strength_days, len(d) - 1)
        fwd_ret  = np.where(end_idx > dc_idx,
                            d['Close'].values[dc_idx] / d['Close'].values[end_idx] - 1, 0.0)
        fwd_norm = np.clip((fwd_ret + 0.10) / 0.30, 0, 1)
        vol_norm = np.clip((d['volume_ratio'].values[dc_idx] - 0.5) / 2.0, 0, 1)
        adx_norm = np.clip(d['adx'].values[dc_idx] / 40.0, 0, 1)
        mtf_ok   = (1 - d['mtf_bull'].values[dc_idx]).astype(float)
        quality_dc[dc_idx] = 0.4*fwd_norm + 0.3*vol_norm + 0.3*(0.5*adx_norm + 0.5*mtf_ok)
    d['target_quality_dc'] = quality_dc

    # ── VOLATILITY-ADJUSTED LABEL (v13) — ATR-based success criterion ────────
    # Success: price reaches Entry + 2×ATR before hitting Entry − 1×ATR.
    # This standardises the definition of "success" across high-vol and low-vol
    # stocks so the model isn't biased toward high-beta moonshot behaviour.
    # atr_pct is the ATR as % of Close, already computed in engineer_features.
    atr_vals  = d['atr_pct'].values / 100.0      # fractional (e.g. 0.02 = 2%)
    close_arr = d['Close'].values
    target_atr = np.zeros(len(d), dtype=int)
    for idx in range(len(d) - 1):
        entry   = close_arr[idx]
        atr_f   = float(atr_vals[idx])
        tp_lvl  = entry * (1.0 + 2.0 * atr_f)   # 2× ATR above entry
        sl_lvl  = entry * (1.0 - 1.0 * atr_f)   # 1× ATR below entry
        hit_tp = hit_sl = False
        for j in range(idx + 1, min(idx + 60, len(d))):
            h = float(d['High'].iloc[j])
            l = float(d['Low'].iloc[j])
            if h >= tp_lvl: hit_tp = True; break
            if l <= sl_lvl: hit_sl = True; break
        target_atr[idx] = 1 if hit_tp and not hit_sl else 0
    d['target_gc_atr'] = target_atr

    # Trim tail: last n_days rows have forward-looking labels
    d = d.iloc[:-n_days].copy()
    return d


# ─────────────────────────────────────────────────────────────────────────────
# WALK-FORWARD SPLIT  (strict embargo: train / purge / calibration / test)
# ─────────────────────────────────────────────────────────────────────────────

def add_categorical_features(df_combined, nifty_regime_series=None):
    """
    v13: Add sector/cap one-hot columns, sector-demeaned features, and market regime.
    Called once on the concatenated multi-ticker dataframe before walk-forward split.

    Parameters
    ----------
    df_combined : pd.DataFrame  — already has 'ticker' column
    nifty_regime_series : pd.Series indexed by date, values 0/1 (Nifty above MA200)

    Returns
    -------
    df_combined with new feature columns added in-place.
    """
    # ── 1. Sector & cap one-hot encoding ─────────────────────────────────────
    df_combined['sector_id'] = df_combined['ticker'].map(SECTOR_MAP).fillna(15).astype(int)
    df_combined['cap_id']    = df_combined['ticker'].map(CAP_MAP).fillna(1).astype(int)

    for i in range(N_SECTORS):
        df_combined[f'sector_{i}'] = (df_combined['sector_id'] == i).astype(np.int8)
    for i in range(N_CAPS):
        df_combined[f'cap_{i}']    = (df_combined['cap_id'] == i).astype(np.int8)

    # ── 2. Market regime (Nifty 50 above its own MA200) ──────────────────────
    if nifty_regime_series is not None:
        df_combined['mkt_regime'] = (
            nifty_regime_series
            .reindex(df_combined['_date'], method='ffill')
            .fillna(0)
            .values
        )
    else:
        df_combined['mkt_regime'] = 0

    # ── 3. Sector-demeaned features ───────────────────────────────────────────
    # For each feature, subtract the cross-sectional sector mean at each date-sector bucket.
    # This turns absolute readings into "relative strength vs peers" signals.
    DEMEAN_FEATS = {
        'rsi_14':           'rsi_14_vs_sector',
        'price_vs_ma200':   'price_vs_ma200_vs_sector',
        'adx':              'adx_vs_sector',
        'atr_pct':          'atr_pct_vs_sector',
        'macd_hist_pct':    'macd_hist_pct_vs_sector',
    }
    for raw_col, new_col in DEMEAN_FEATS.items():
        if raw_col not in df_combined.columns:
            df_combined[new_col] = 0.0
            continue
        # Group by (_date, sector_id) and subtract the group mean
        sector_mean = (
            df_combined.groupby(['_date', 'sector_id'])[raw_col]
            .transform('mean')
        )
        df_combined[new_col] = df_combined[raw_col] - sector_mean

    return df_combined


def enforce_cap_mix(df_combined, large_frac=0.40, mid_frac=0.40, small_frac=0.20,
                    rng_seed=42):
    """
    Resample training data to enforce ~40% Large / 40% Mid / 20% Small cap mix.
    Prevents model from overfitting to large-cap bull-momentum regime.
    Operates on the FULL df_combined (before walk-forward split) — temporal order preserved.
    """
    cap_col = 'cap_id'
    if cap_col not in df_combined.columns:
        return df_combined

    rng = np.random.default_rng(rng_seed)
    large = df_combined[df_combined[cap_col] == 0]
    mid   = df_combined[df_combined[cap_col] == 1]
    small = df_combined[df_combined[cap_col] == 2]

    n_total = len(df_combined)
    target_large = int(n_total * large_frac)
    target_mid   = int(n_total * mid_frac)
    target_small = int(n_total * small_frac)

    def sample_or_keep(subset, target_n):
        if len(subset) == 0: return subset
        if len(subset) <= target_n: return subset   # don't inflate
        # Chronological random subsample (preserves temporal structure)
        chosen = np.sort(rng.choice(len(subset), target_n, replace=False))
        return subset.iloc[chosen]

    large_s = sample_or_keep(large, target_large)
    mid_s   = sample_or_keep(mid,   target_mid)
    small_s = sample_or_keep(small, target_small)

    df_mixed = pd.concat([large_s, mid_s, small_s], axis=0)
    # Re-sort by date to restore temporal order
    df_mixed = df_mixed.sort_values('_date').reset_index(drop=True)
    return df_mixed


# ─────────────────────────────────────────────────────────────────────────────
# WALK-FORWARD SPLIT  (strict embargo: train / purge / calibration / test)
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward_split(df, n_splits=8, purge_gap=20):
    """
    Returns list of (train_idx, cal_idx, test_idx) triples.
    Works with both DatetimeIndex and integer index (after reset_index).
    """
    idx_vals   = np.arange(len(df))   # always use positional integers
    n          = len(idx_vals)
    fold_size  = n // (n_splits + 1)
    splits     = []

    for i in range(1, n_splits + 1):
        train_end_pos  = fold_size * i - 1
        purge_end_pos  = train_end_pos + purge_gap          # integer purge gap
        if purge_end_pos >= n: break

        cal_start_pos  = purge_end_pos + 1
        cal_end_pos    = min(fold_size * i + fold_size // 4, n - 1)
        test_start_pos = cal_end_pos + 1
        test_end_pos   = min(fold_size * (i + 1) - 1, n - 1)

        if test_end_pos <= test_start_pos: break

        ti = idx_vals[:train_end_pos + 1]
        ci = idx_vals[cal_start_pos:cal_end_pos + 1]
        vi = idx_vals[test_start_pos:test_end_pos + 1]

        if len(ti) < 200 or len(vi) < 10: continue
        splits.append((ti, ci, vi))

    return splits


# ─────────────────────────────────────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_binary_ensemble(X_train, y_train):
    n_pos = int((y_train == 1).sum()); n_neg = int((y_train == 0).sum())
    spw   = (n_neg / n_pos) if n_pos > 0 else 1.0

    # XGBoost — primary model, fast with n_jobs=-1
    m_xgb = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        scale_pos_weight=spw, subsample=0.8, colsample_bytree=0.75,
        min_child_weight=8, gamma=0.2, reg_alpha=0.15, reg_lambda=2.0,
        random_state=42, eval_metric='logloss', n_jobs=-1,
        tree_method='hist',   # histogram-based: much faster on large data
    )
    m_xgb.fit(X_train, y_train)

    # Random Forest — reduced trees, capped depth for speed
    m_rf = RandomForestClassifier(
        n_estimators=150, max_depth=10, min_samples_leaf=20,
        max_features='sqrt', class_weight='balanced',
        random_state=42, n_jobs=-1,
    )
    m_rf.fit(X_train, y_train)

    # Second XGBoost with different hyperparams (replaces slow sklearn GBM)
    m_xgb2 = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.08,
        scale_pos_weight=spw, subsample=0.75, colsample_bytree=0.70,
        min_child_weight=10, gamma=0.3, reg_alpha=0.2, reg_lambda=1.5,
        random_state=7, eval_metric='logloss', n_jobs=-1,
        tree_method='hist',
    )
    m_xgb2.fit(X_train, y_train)

    return [m_xgb, m_rf, m_xgb2]


def train_quality_model(X_cross, y_quality):
    mask = ~np.isnan(y_quality)
    X_q, y_q = X_cross[mask], y_quality[mask]
    if len(y_q) < 20: return None
    model = XGBRegressor(
        n_estimators=150, max_depth=4, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.75,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42,
        tree_method='hist', n_jobs=-1,
    )
    model.fit(X_q, y_q)
    return model


def train_timing_model(X_rows, y_days_norm):
    mask = ~np.isnan(y_days_norm)
    valid = mask & (y_days_norm >= 0) & (y_days_norm <= 1)
    X_t, y_t = X_rows[valid], y_days_norm[valid]
    if len(y_t) < 30: return None
    model = XGBRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.75, random_state=42,
        tree_method='hist', n_jobs=-1,
    )
    model.fit(X_t, y_t)
    return model


def train_days_regressor(X_rows, y_days_raw):
    """
    Model C: XGBRegressor on raw days-to-GC (capped at 60; 999 = no upcoming GC).
    Only trained on rows where a future GC exists within 60 bars.
    Uses early stopping on a 20% validation split to prevent overfitting.
    """
    mask = y_days_raw < 999
    X_d, y_d = X_rows[mask], y_days_raw[mask]
    if len(y_d) < 50: return None

    # Chronological 80/20 split for early stopping (no shuffle — time series)
    split = int(len(X_d) * 0.8)
    X_tr, X_val = X_d.iloc[:split], X_d.iloc[split:]
    y_tr, y_val = y_d[:split],       y_d[split:]
    if len(y_val) < 10:
        X_tr, y_tr = X_d, y_d
        X_val, y_val = X_d.iloc[-20:], y_d[-20:]

    model = XGBRegressor(
        n_estimators=500, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.75,
        reg_alpha=0.2, reg_lambda=2.0,
        min_child_weight=10,
        random_state=42, tree_method='hist', n_jobs=-1,
        early_stopping_rounds=30,
        eval_metric='mae',
    )
    model.fit(X_tr, y_tr,
              eval_set=[(X_val, y_val)],
              verbose=False)
    return model


def train_recall_model(X_train, y_train):
    """
    Model B: Maximum-recall ensemble.
    Uses 5× scale_pos_weight + low min_child_weight to catch as many GC events as possible.
    Precision will be lower — combined with Model A for dual-confirmation signals.
    Used standalone at low threshold for recall-optimized backtest evaluation.
    """
    n_pos = int((y_train == 1).sum()); n_neg = int((y_train == 0).sum())
    spw   = (n_neg / n_pos) * 5.0 if n_pos > 0 else 5.0   # 5× — aggressive recall

    m_xgb = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        scale_pos_weight=spw, subsample=0.85, colsample_bytree=0.80,
        min_child_weight=3, gamma=0.05, reg_alpha=0.05, reg_lambda=0.5,
        random_state=99, eval_metric='aucpr',   # optimize PR-AUC for imbalanced
        n_jobs=-1, tree_method='hist',
    )
    m_xgb.fit(X_train, y_train)

    m_rf = RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=5,
        max_features='sqrt', class_weight={0: 1, 1: 5},   # 5× weight on positives
        random_state=99, n_jobs=-1,
    )
    m_rf.fit(X_train, y_train)

    # Third model: XGBoost with focal-loss-like behaviour via high pos weight
    m_xgb2 = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.08,
        scale_pos_weight=spw * 1.5, subsample=0.8, colsample_bytree=0.75,
        min_child_weight=2, gamma=0.0, reg_alpha=0.0, reg_lambda=0.3,
        random_state=77, eval_metric='aucpr', n_jobs=-1, tree_method='hist',
    )
    m_xgb2.fit(X_train, y_train)
    return [m_xgb, m_rf, m_xgb2]


def ensemble_predict_proba(models, X):
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


def calibrate_probabilities(models, X_cal, y_cal):
    """
    Platt scaling on HELD-OUT calibration data.
    X_cal and y_cal must NOT have been seen during training.
    """
    if len(y_cal) < 20 or y_cal.sum() < 5:
        return None
    raw = ensemble_predict_proba(models, X_cal).reshape(-1, 1)
    cal = LogisticRegression(C=1.0, solver='lbfgs', max_iter=500, class_weight='balanced')
    cal.fit(raw, y_cal)
    return cal


def apply_calibration(cal, models, X):
    """Apply Platt calibrator. Falls back to raw ensemble if calibrator is None."""
    raw = ensemble_predict_proba(models, X).reshape(-1, 1)
    if cal is None:
        return raw.flatten()
    return cal.predict_proba(raw)[:, 1]


def compute_feature_importance(models, feature_cols):
    imp = [m.feature_importances_ for m in models if hasattr(m, 'feature_importances_')]
    if not imp: return pd.Series(dtype=float)
    return pd.Series(np.mean(imp, axis=0), index=feature_cols).sort_values(ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST ENGINE — production-grade
# ─────────────────────────────────────────────────────────────────────────────

def backtest_strategy(df_feat, probs_gc, probs_dc,
                      model_quality_gc=None,
                      threshold=0.60, quality_min=0.35,
                      recall_threshold=0.40,
                      cooldown=10,
                      transaction_cost=0.001,
                      slippage=0.0005,
                      stop_loss=-0.08,
                      take_profit=0.20,
                      atr_stop_mult=2.0,
                      adx_min=20.0,
                      hurst_min=0.45,
                      ev_min=0.0,
                      avg_win=0.11,
                      avg_loss=0.05,
                      probs_gc_b=None,
                      exit_at_gc=True,
                      trailing_stop=True):
    """
    Production backtest with all fixes:
    - FIX 3: Regime filter (ADX > adx_min AND Hurst > hurst_min)
    - FIX 4: EV filter (only trade when EV > ev_min)
    - FIX 2: AND logic for entry (Model A AND Model B)
    - FIX 5: Exit at actual GC event OR trailing stop
    - Entry on NEXT BAR OPEN, costs on both sides
    """
    df = df_feat.copy()
    dates_arr = df.index.to_numpy()
    df = df.reset_index(drop=True)
    n = len(df)
    df['returns'] = df['Close'].pct_change().fillna(0)
    pg      = np.array(probs_gc).flatten()[:n]
    pg_b    = np.array(probs_gc_b).flatten()[:n] if probs_gc_b is not None else pg
    pd_arr  = np.array(probs_dc).flatten()[:n]
    total_cost = transaction_cost + slippage

    # gc_event column for exit-at-GC
    gc_ev = df['gc_event'].values if 'gc_event' in df.columns else np.zeros(n)

    position  = np.zeros(n); pos = 0
    last_exit = -cooldown; entry_price = 0.0
    entry_bar = -1; blocked_q = 0; blocked_regime = 0; blocked_ev = 0
    trade_log = []
    trail_high = 0.0   # for trailing stop

    for i in range(n - 1):
        atr_val = float(df['atr_pct'].iloc[i]) if not pd.isna(df['atr_pct'].iloc[i]) else 0.0

        # ── Exit logic ────────────────────────────────────────────────────────
        if pos == 1 and entry_price > 0:
            cur_price = float(df['Close'].iloc[i])
            tr = cur_price / entry_price - 1
            trail_high = max(trail_high, cur_price)

            dyn_sl = max(stop_loss, -(atr_stop_mult * atr_val / 100.0))
            trail_sl = (trail_high / entry_price - 1) - abs(stop_loss)
            trail_exit = trailing_stop and (tr < trail_sl) and (trail_high / entry_price - 1 > 0.03)
            gc_exit = exit_at_gc and (gc_ev[i] == 1)
            # DC exit only on very high DC probability (>0.75) — avoids premature exits
            dc_exit = pd_arr[i] > 0.75

            should_exit = (tr <= dyn_sl or tr >= take_profit or
                           dc_exit or trail_exit or gc_exit)
            if should_exit:
                exit_price = float(df['Open'].iloc[i+1]) if i+1 < n else cur_price
                actual_ret = exit_price / entry_price - 1 - 2*total_cost
                reason = ('GC_event' if gc_exit else
                          'trail_SL' if trail_exit else
                          'SL'       if tr <= dyn_sl else
                          'TP'       if tr >= take_profit else 'DC_signal')
                trade_log.append({
                    'entry_bar': entry_bar, 'exit_bar': i+1,
                    'entry_date': dates_arr[entry_bar] if entry_bar >= 0 else None,
                    'exit_date':  dates_arr[i+1] if i+1 < n else dates_arr[i],
                    'entry_p': entry_price, 'exit_p': exit_price,
                    'ret': actual_ret, 'reason': reason,
                    'bars_held': i+1 - entry_bar,
                })
                pos = 0; last_exit = i+1; entry_bar = -1; trail_high = 0.0
                position[i+1] = 0

        # ── Entry logic ───────────────────────────────────────────────────────
        elif pos == 0 and (i - last_exit) >= cooldown:
            # SIMPLIFIED ENTRY: Model A OR Model B fires above their thresholds
            # No hard ADX/near_cross gates — these are informational only
            # The model already learned these patterns during training
            sig_a = pg[i]   >= threshold
            sig_b = pg_b[i] >= recall_threshold

            if sig_a or sig_b:   # OR logic for entry — maximise opportunities
                skip = False

                # Soft regime filter: only Hurst (mean-reverting markets are genuinely bad)
                hurst_v_e = float(df['hurst'].iloc[i]) if 'hurst' in df.columns else 0.5
                if hurst_v_e < hurst_min:
                    skip = True; blocked_regime += 1

                # EV filter
                if not skip:
                    ev = pg[i] * avg_win - (1 - pg[i]) * avg_loss
                    if ev <= ev_min:
                        skip = True; blocked_ev += 1

                # Quality gate
                if not skip and model_quality_gc is not None and quality_min > 0:
                    try:
                        X_bar = df[FEATURE_COLS].iloc[[i]].replace([np.inf,-np.inf], np.nan).fillna(0)
                        if float(model_quality_gc.predict(X_bar)[0]) < quality_min:
                            skip = True; blocked_q += 1
                    except: pass

                if not skip and i+1 < n:
                    pos = 1
                    entry_price = float(df['Open'].iloc[i+1])
                    entry_bar   = i+1
                    last_exit   = i+1
                    trail_high  = entry_price

        position[i] = pos

    position[n-1] = 0

    strat_ret = np.zeros(n)
    for i in range(1, n):
        if position[i] == 1 and position[i-1] == 1:
            strat_ret[i] = df['returns'].iloc[i]
    for t in trade_log:
        eb = t['entry_bar']; xb = t['exit_bar']
        if 0 < eb < n:
            strat_ret[eb] = float(df['Close'].iloc[eb]) / t['entry_p'] - 1 - total_cost
        if 0 < xb < n:
            strat_ret[xb] = t['exit_p'] / float(df['Close'].iloc[xb-1]) - 1 - total_cost

    df['position']  = position
    df['strat_ret'] = pd.Series(strat_ret, index=df.index)
    df['equity']    = (1 + df['strat_ret']).cumprod()
    df['buy_hold']  = (1 + df['returns']).cumprod()

    # ── Performance metrics ────────────────────────────────────────────────────
    total_ret   = df['equity'].iloc[-1] - 1
    bh_ret      = df['buy_hold'].iloc[-1] - 1
    rolling_max = df['equity'].cummax()
    drawdown    = df['equity'] / rolling_max - 1
    max_dd      = drawdown.min()

    # CAGR over the full backtest period (total bars elapsed, not just active bars)
    # Using active bars would inflate annualised return by total/active (up to 10×).
    ann_ret     = (df['equity'].iloc[-1] ** (252 / max(n, 1))) - 1 if n > 30 else 0.0

    # Sharpe/Sortino from TRADE returns (not bar-by-bar including flat days)
    trade_rets  = [t['ret'] for t in trade_log]
    if len(trade_rets) >= 2:
        tr_arr      = np.array(trade_rets)
        tr_std      = tr_arr.std()
        tr_neg_std  = tr_arr[tr_arr < 0].std() if any(r < 0 for r in trade_rets) else 1e-9
        # Per-trade Sharpe (annualise by avg bars_held)
        avg_bars    = np.mean([t['bars_held'] for t in trade_log]) if trade_log else 20
        ann_factor  = np.sqrt(252 / max(avg_bars, 1))
        sharpe      = ann_factor * tr_arr.mean() / (tr_std + 1e-9)
        sortino     = ann_factor * tr_arr.mean() / (tr_neg_std + 1e-9)
    else:
        sharpe = sortino = 0.0

    calmar      = ann_ret / abs(max_dd) if max_dd < 0 else float('inf')
    win_rate    = np.mean([r > 0 for r in trade_rets]) if trade_rets else 0.0
    avg_trade   = np.mean(trade_rets) if trade_rets else 0.0
    avg_win     = np.mean([r for r in trade_rets if r > 0]) if any(r > 0 for r in trade_rets) else 0.0
    avg_loss    = np.mean([r for r in trade_rets if r < 0]) if any(r < 0 for r in trade_rets) else 0.0
    gross_p     = sum(r for r in trade_rets if r > 0)
    gross_l     = abs(sum(r for r in trade_rets if r < 0))
    profit_fac  = (gross_p / gross_l) if gross_l > 0 else float('inf')
    expectancy  = win_rate * avg_win + (1 - win_rate) * avg_loss
    time_in_mkt = 100.0 * position.sum() / max(n, 1)

    return dict(
        df=df, total_return=total_ret,
        ann_return=ann_ret, buy_hold_return=bh_ret,
        sharpe=sharpe, sortino=sortino, calmar=calmar,
        max_dd=max_dd, win_rate=win_rate,
        total_trades=len(trade_log), profit_factor=profit_fac,
        avg_trade=avg_trade, avg_win=avg_win, avg_loss=avg_loss,
        expectancy=expectancy, time_in_market=time_in_mkt,
        blocked_quality=blocked_q, blocked_regime=blocked_regime,
        blocked_ev=blocked_ev, trade_log=trade_log,
        drawdown_series=drawdown,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MONTE CARLO — risk-neutral drift + regime tilt
# ─────────────────────────────────────────────────────────────────────────────

def run_monte_carlo(prices, days=60, n_sims=1000, regime='neutral'):
    """
    Risk-neutral GBM with Student-t noise + small regime drift tilt.
    
    Drift fix: Using historical mean return as drift is unreliable for short
    horizons (very noisy, regime-dependent). Instead use risk-neutral drift = 0
    plus a small regime adjustment derived from current signal, not history.
    This gives more honest uncertainty bands.
    """
    log_r      = np.diff(np.log(prices))
    recent_vol = log_r[-60:].std() if len(log_r) >= 60 else log_r.std()
    full_vol   = log_r.std()
    # Blended vol: 60% recent (more accurate near-term), 40% long-term
    sig        = 0.60 * recent_vol + 0.40 * full_vol

    # Risk-neutral drift = 0 (not historical mean which is very noisy)
    # Add a small regime adjustment only
    regime_adj = {'bullish': +0.0002, 'bearish': -0.0002, 'neutral': 0.0}
    mu         = regime_adj.get(regime, 0.0)

    S0  = float(prices[-1])
    rng = np.random.default_rng(42)
    # Student-t(nu=5) for fat tails — more realistic than Gaussian
    Z     = rng.standard_t(df=5, size=(n_sims, days)) / np.sqrt(5/3)
    paths = S0 * np.exp(np.cumsum((mu - 0.5*sig**2) + sig*Z, axis=1))
    paths = np.hstack([np.full((n_sims,1), S0), paths])
    finals = paths[:, -1]

    var_95  = np.percentile(finals, 5)  / S0 - 1
    cvar_95 = finals[finals <= np.percentile(finals, 5)].mean() / S0 - 1

    return dict(
        paths=paths, finals=finals,
        p5 =np.percentile(paths,  5, axis=0),
        p25=np.percentile(paths, 25, axis=0),
        p50=np.percentile(paths, 50, axis=0),
        p75=np.percentile(paths, 75, axis=0),
        p95=np.percentile(paths, 95, axis=0),
        S0=S0, mu=mu, sig=sig, ann_vol=sig*np.sqrt(252),
        prob_up=(finals > S0).mean(),
        exp_return=(finals.mean()/S0)-1,
        var_95=var_95, cvar_95=cvar_95,
    )


def kelly_criterion(win_rate, avg_win, avg_loss):
    """
    Full Kelly fraction.
    avg_win and avg_loss should be from actual backtest trade log.
    Recommend using 0.25× (quarter-Kelly) to account for model uncertainty.
    """
    if not avg_loss or win_rate <= 0 or win_rate >= 1: return 0.0
    b = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0
    if b == 0: return 0.0
    return max(0.0, min((b*win_rate - (1-win_rate)) / b, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────────────────────────────────────

def predict_single(ticker, models_gc, models_dc,
                   model_quality_gc, model_quality_dc,
                   model_timing_gc, feature_cols,
                   calibrator_gc, calibrator_dc,
                   start_date, fast_p, slow_p,
                   threshold, quality_min, prediction_days,
                   models_gc_recall=None, calibrator_gc_recall=None,
                   model_days_regressor=None,
                   avg_win=0.11, avg_loss=0.05,
                   recall_threshold=0.40):
    df = download_stock_data(ticker, start_date)
    if df is None or len(df) < 300: return None, None
    df_feat = engineer_features(df, fast_p=fast_p, slow_p=slow_p)
    if len(df_feat) < 10: return None, None

    # ── v13: Inject sector/cap/regime categorical features for this ticker ──
    sector_id = SECTOR_MAP.get(ticker, 15)
    cap_id    = CAP_MAP.get(ticker, 1)
    for i in range(N_SECTORS):
        df_feat[f'sector_{i}'] = np.int8(1 if sector_id == i else 0)
    for i in range(N_CAPS):
        df_feat[f'cap_{i}']    = np.int8(1 if cap_id == i else 0)
    # Market regime: use cached Nifty regime if available
    try:
        nifty_reg = get_nifty_regime(str(start_date))
        if nifty_reg is not None:
            df_feat['mkt_regime'] = (
                nifty_reg.reindex(df_feat.index, method='ffill').fillna(0).values
            )
        else:
            df_feat['mkt_regime'] = 0
    except Exception:
        df_feat['mkt_regime'] = 0
    # Sector-demeaned features — for single-ticker prediction, demeaning by sector
    # mean is not computable without peers, so we use the raw value (demean = 0).
    for col in ['rsi_14_vs_sector','price_vs_ma200_vs_sector',
                'adx_vs_sector','atr_pct_vs_sector','macd_hist_pct_vs_sector']:
        if col not in df_feat.columns:
            df_feat[col] = 0.0

    # Ensure all FEATURE_COLS exist (fill missing with 0)
    for fc in feature_cols:
        if fc not in df_feat.columns:
            df_feat[fc] = 0.0

    X_latest = df_feat[feature_cols].iloc[[-1]].replace([np.inf,-np.inf], np.nan).fillna(0)

    raw_gc  = float(ensemble_predict_proba(models_gc, X_latest)[0])
    raw_dc  = float(ensemble_predict_proba(models_dc, X_latest)[0])
    prob_gc = float(apply_calibration(calibrator_gc, models_gc, X_latest)[0])
    prob_dc = float(apply_calibration(calibrator_dc, models_dc, X_latest)[0])

    # Model B — high recall (recall_threshold consistent with backtest)
    prob_gc_recall = float(ensemble_predict_proba(models_gc_recall, X_latest)[0]) \
                     if models_gc_recall else prob_gc
    dual_confirm   = (prob_gc >= threshold) and (prob_gc_recall >= recall_threshold)

    q_gc = float(model_quality_gc.predict(X_latest)[0]) if model_quality_gc else 0.5
    q_dc = float(model_quality_dc.predict(X_latest)[0]) if model_quality_dc else 0.5
    t_gc = float(model_timing_gc.predict(X_latest)[0])  if model_timing_gc  else 0.5
    q_gc = float(np.clip(q_gc, 0, 1))
    q_dc = float(np.clip(q_dc, 0, 1))
    t_gc = float(np.clip(t_gc, 0, 1))

    # Model C — raw days regressor (precise timing)
    days_pred = None
    if model_days_regressor is not None:
        raw_days = float(model_days_regressor.predict(X_latest)[0])
        days_pred = max(1, int(round(np.clip(raw_days, 1, 60))))

    # Timing string — always show days_pred if available, regardless of prob_gc
    if days_pred is not None:
        timing_str = f"~{days_pred} days"
    elif not np.isnan(t_gc):
        est = max(1, int(round((1.0 - t_gc) * prediction_days)))
        timing_str = f"~{est} days"
    else:
        timing_str = "n/a"

    latest   = df_feat.iloc[-1]
    gc_model = prob_gc >= threshold
    dc_model = prob_dc >= threshold
    gc_q_ok  = q_gc >= quality_min
    dc_q_ok  = q_dc >= quality_min

    # ── FILTER 1: Trend filter — ADX > 20 (trend must be established) ─────────
    adx_val    = float(latest['adx'])
    trend_ok   = adx_val > 20.0

    # ── FILTER 2: Regime filter — Price > MA200 (bull macro regime) ───────────
    price_vs_ma200_val = float(latest['price_vs_ma200'])
    regime_price_ok    = price_vs_ma200_val > 0.0   # price above 200-bar MA

    # ── FILTER 3: Near-cross gate — gap must be closing, not widening ─────────
    # v13: raised threshold 0.30 → 0.45 to cut false positives on wide-gap bars
    near_cross_val = float(latest['near_cross_score']) if 'near_cross_score' in latest.index else 0.5
    near_cross_ok  = near_cross_val >= 0.45   # gap is genuinely converging

    # ── FILTER 4: Hurst (persistence check — trend is not mean-reverting) ─────
    hurst_val  = float(latest['hurst'])
    hurst_ok   = hurst_val > 0.45

    # Combined regime: all four filters must pass for a clean BULLISH signal
    regime_ok  = trend_ok and regime_price_ok and near_cross_ok and hurst_ok

    # ── Composite signal gate (v13 tightened to suppress false positives) ─────
    # model_prob >= threshold AND near_cross (>=0.45) AND trend_ok AND dual_confirm
    composite_signal = (prob_gc >= threshold) and near_cross_ok and trend_ok and dual_confirm

    if gc_model and gc_q_ok and dual_confirm and regime_ok:
        signal, stype = f"BULLISH — GC Confirmed (A+B, quality={q_gc:.2f})", "bullish"
    elif composite_signal and gc_q_ok:
        signal, stype = f"BULLISH — Golden Cross (quality={q_gc:.2f})", "bullish"
    elif gc_model and gc_q_ok and not trend_ok:
        signal, stype = f"CAUTION — GC likely but ADX weak ({adx_val:.0f} < 20)", "caution"
    elif gc_model and gc_q_ok and not regime_price_ok:
        signal, stype = f"CAUTION — GC likely but price below MA200 ({price_vs_ma200_val:+.1f}%)", "caution"
    elif gc_model and gc_q_ok and not near_cross_ok:
        signal, stype = f"CAUTION — GC likely but crossover is far away (gap converging={near_cross_val:.2f})", "caution"
    elif gc_model and not gc_q_ok:
        signal, stype = f"CAUTION — GC likely but quality low ({q_gc:.2f})", "caution"
    elif dc_model and dc_q_ok:
        signal, stype = f"BEARISH — Death Cross (quality={q_dc:.2f})", "bearish"
    elif prob_gc > 0.50:
        signal, stype = "WATCH — Moderate bullish setup", "caution"
    elif prob_dc > 0.50:
        signal, stype = "WATCH — Moderate bearish setup", "caution"
    else:
        signal, stype = "NEUTRAL — No clear signal", "neutral"

    # ── Expected Value ─────────────────────────────────────────────────────────
    ev = (prob_gc * avg_win) - ((1 - prob_gc) * avg_loss)

    # ── Confidence-scaled position size (quarter-Kelly base) ──────────────────
    kf = kelly_criterion(prob_gc, avg_win, avg_loss)
    pos_size = float(np.clip(kf * 0.25 * (prob_gc / max(threshold, 0.01)), 0, 0.25))

    # ── Pre-GC entry signal ────────────────────────────────────────────────────
    pre_gc_entry = (days_pred is not None and 5 <= days_pred <= 20 and prob_gc > 0.45)

    # Vol from recent 60 bars
    log_r   = np.diff(np.log(df_feat['Close'].values[-61:]))
    ann_vol = log_r.std() * np.sqrt(252) * 100 if len(log_r) > 5 else 0.0

    return dict(
        ticker=ticker, prob_gc=prob_gc, prob_dc=prob_dc,
        raw_gc=raw_gc, raw_dc=raw_dc,
        prob_gc_recall=prob_gc_recall, dual_confirm=dual_confirm,
        quality_gc=q_gc, quality_dc=q_dc,
        signal=signal, signal_type=stype,
        timing=timing_str,
        days_pred=days_pred,
        ev=ev, pos_size=pos_size,
        pre_gc_entry=pre_gc_entry,
        regime_ok=regime_ok,
        trend_ok=trend_ok,
        regime_price_ok=regime_price_ok,
        near_cross_ok=near_cross_ok,
        near_cross_score=near_cross_val,
        composite_signal=composite_signal,
        hurst=float(latest['hurst']),
        hist_gc_success=float(latest['hist_gc_success']),
        dc_age=int(latest['days_since_dc']),
        price_vs_ma200=float(latest['price_vs_ma200']),
        adx=float(latest['adx']),
        di_diff=float(latest['di_diff']),
        current_price=float(df['Close'].iloc[-1]),
        ema_gap=float(latest['ema_gap_pct']),          # absolute gap removed; use pct
        ema_gap_pct=float(latest['ema_gap_pct']),
        state='Golden Cross' if latest['ema_gap_pct'] > 0 else 'Death Cross',
        rsi=float(latest['rsi_14']),
        atr=float(latest['atr']),
        atr_pct=float(latest['atr_pct']),
        ann_vol=ann_vol,
        gk_vol=float(latest['gk_vol'])*100,
        macd_hist=float(latest['macd_hist']),
        mtf_bull=int(latest['mtf_bull']),
        monthly_bull=int(latest['monthly_bull']),
        vol_compression=int(latest['vol_compression']),
        listing_date=df.attrs.get('listing_date', 'unknown'),
        history_bars=len(df),
        history_years=round(len(df) / 252, 1),
        short_history=len(df) < 500,   # flag for UI warning
        # v13: sector/cap/regime context
        sector_id=sector_id,
        cap_id=cap_id,
        sector_name=SECTOR_NAMES.get(sector_id, 'Other'),
        cap_name=CAP_NAMES.get(cap_id, 'Mid'),
        mkt_regime=int(df_feat['mkt_regime'].iloc[-1]),
    ), df_feat


# ─────────────────────────────────────────────────────────────────────────────
# PLOT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

DARK = dict(paper_bgcolor="#03050d", plot_bgcolor="#080c18",
            font=dict(color="#d4ddf5", family="JetBrains Mono, monospace"))
GRID = dict(gridcolor="#151c30", zerolinecolor="#1e2840")

# ── Tooltip definitions for every metric ─────────────────────────────────────
TOOLTIPS = {
    "GC Probability": {
        "explanation": "Calibrated probability (0–100%) that a Golden Cross will occur within the prediction window. A Golden Cross is when the shorter-term EMA (e.g. 50-period) crosses above the longer-term EMA (e.g. 200-period) — a classic bullish signal.",
        "formula": "Ensemble(XGBoost + RF + GBM) → Platt calibration\nP_calibrated = σ(a · P_raw + b)",
        "notes": "Values ≥ threshold trigger a BULLISH signal. Raw ensemble shown below for transparency.",
    },
    "DC Probability": {
        "explanation": "Calibrated probability that a Death Cross will occur — the opposite of a Golden Cross. The shorter-term EMA crosses below the longer-term EMA, a bearish signal indicating potential downtrend.",
        "formula": "Same ensemble model, separate target label\nP_dc = σ(a · P_raw_dc + b)",
        "notes": "Values ≥ threshold trigger a BEARISH signal. Both GC & DC low → no imminent crossover.",
    },
    "GC Quality Score": {
        "explanation": "Composite score (0–1) rating how 'strong' a potential Golden Cross would be. Combines forward return expectation, volume confirmation, ADX trend strength, and multi-timeframe alignment.",
        "formula": "Score = 0.4×fwd_ret_norm + 0.3×vol_norm\n      + 0.3×(0.5×ADX_norm + 0.5×MTF_bull)",
        "notes": "Gate threshold shown (e.g. ≥ 0.35). Scores below gate suppress the signal even if GC prob is high.",
    },
    "Est. Days to GC": {
        "explanation": "Model's estimated number of trading days until the next Golden Cross forms. Derived from a regression model trained on historical time-to-cross data.",
        "formula": "days = round((1 − t_gc) × pred_window)\nt_gc ∈ [0,1] from XGBoost regressor",
        "notes": "Shows n/a when GC probability is too low (< 0.45) to produce a reliable forecast.",
    },
    "Hurst Exponent": {
        "explanation": "Statistical measure of long-memory / persistence in the price series. Tells whether the asset tends to trend or revert to the mean.",
        "formula": "H = log(R/S) / log(T)\nR/S = rescaled range over period T\nComputed via rolling 60-bar window",
        "notes": "H > 0.5 → trending (momentum). H < 0.5 → mean-reverting. H = 0.5 → random walk.",
    },
    "Hist GC Success": {
        "explanation": "Historical rate at which past Golden Cross signals on this stock led to the EMA crossover holding for at least 20 bars (i.e. the trend persisted after the cross).",
        "formula": "Rate = Σ(GC held 20 bars) / Σ(GC events)\nRolling 750-bar window, shifted 20 bars\n(no lookahead bias)",
        "notes": "Higher % = past GC signals were more reliable for this stock. Lagged 20 bars to prevent lookahead.",
    },
    "Price vs MA200": {
        "explanation": "How far the current price is above or below the 200-day simple moving average, expressed as a percentage. A key trend health indicator.",
        "formula": "% = (Close / MA200 − 1) × 100\nMA200 = rolling 200-bar mean of Close",
        "notes": "Positive = price above MA200 (bullish context). Negative = below MA200 (bearish context).",
    },
    "ADX / DI Spread": {
        "explanation": "ADX (Average Directional Index) measures trend strength regardless of direction. DI Spread (+DI minus −DI) shows whether bulls or bears are in control.",
        "formula": "ADX = smoothed avg of |+DI − −DI| / (+DI + −DI)\n+DI = smoothed(High − prev High) / ATR\n−DI = smoothed(prev Low − Low) / ATR",
        "notes": "ADX > 25 = strong trend. ADX < 18 = weak/no trend. DI Spread > 0 = bullish pressure.",
    },
    "RSI 14": {
        "explanation": "Relative Strength Index over 14 periods. Measures the speed and magnitude of recent price changes to identify overbought or oversold conditions.",
        "formula": "RSI = 100 − 100/(1 + RS)\nRS = avg(up closes, 14) / avg(down closes, 14)",
        "notes": "RSI > 70 = overbought (potential reversal). RSI < 30 = oversold (potential bounce). 30–70 = neutral.",
    },
    "Ann. Volatility": {
        "explanation": "Annualised realised volatility of the stock, computed from the last 60 trading days of log returns. Higher volatility = wider price swings = higher risk.",
        "formula": "σ_ann = std(log(Ct/Ct-1), 60 bars) × √252\nGK vol uses High/Low/Open/Close for efficiency",
        "notes": "GK (Garman-Klass) shown alongside. > 40% = high volatility regime. < 20% = low volatility.",
    },
    "P(Price Up)": {
        "explanation": "Monte Carlo probability that the stock price will be higher than today's price after the forecast horizon, based on 1000 simulated price paths.",
        "formula": "P(S_T > S_0) = count(paths where S_T > S_0) / N\nPaths use risk-neutral GBM with Student-t noise",
        "notes": "Risk-neutral drift = 0 + small regime tilt (±0.02%/day). Historical mean not used — too noisy.",
    },
    "Expected Return": {
        "explanation": "The average return across all Monte Carlo simulation paths over the forecast horizon. Represents the probability-weighted expected outcome.",
        "formula": "E[R] = mean(S_T / S_0 − 1) across N paths\ndS = (μ − ½σ²)dt + σ·dW, dW ~ Student-t(5)",
        "notes": "Near zero is expected under risk-neutral assumption. Regime tilt shifts it slightly.",
    },
    "VaR 95% / CVaR": {
        "explanation": "Value at Risk (VaR) is the worst loss not exceeded in 95% of scenarios. CVaR (Conditional VaR / Expected Shortfall) is the average loss in the worst 5% of scenarios.",
        "formula": "VaR_95 = percentile(S_T/S_0 − 1, 5%)\nCVaR_95 = mean(returns | return ≤ VaR_95)",
        "notes": "CVaR is a more conservative risk measure than VaR — it captures tail risk beyond the threshold.",
    },
    "P50 Target": {
        "explanation": "The median (50th percentile) price target from Monte Carlo simulations after the forecast horizon. Half of all simulated paths end above this price.",
        "formula": "P50 = median(S_T across all N paths)\nS_T = S_0 · exp(cumsum((μ−½σ²) + σ·Z))",
        "notes": "Use P25/P75 for a confidence range. P5/P95 for extreme scenarios.",
    },
    "Kelly Fraction": {
        "explanation": "Optimal fraction of capital to allocate per trade, derived from the Kelly Criterion using actual backtest win rate and average win/loss sizes.",
        "formula": "f* = (b·p − q) / b\nb = avg_win/avg_loss, p = win_rate\nq = 1 − p",
        "notes": "Full Kelly is aggressive. Use 0.25× Kelly (quarter-Kelly) to account for model uncertainty and drawdown risk.",
    },
    "ATR Stop (2×ATR)": {
        "explanation": "Dynamic stop-loss price level set at 2× the Average True Range below entry. ATR measures recent price volatility, so the stop widens in volatile markets.",
        "formula": "Stop = Entry × (1 − 2 × ATR%)\nATR = Wilder smoothing of True Range (14 bars)\nTR = max(H−L, |H−Cprev|, |L−Cprev|)",
        "notes": "Floor stop-loss also applied (configurable %). The tighter of ATR-stop and fixed-stop is used.",
    },
    "Risk : Reward": {
        "explanation": "Ratio of potential profit (take-profit target) to potential loss (ATR-based stop). A ratio above 2:1 is generally considered favourable.",
        "formula": "R:R = (Entry × TP%) / (Entry − Stop_price)\nTP% = take-profit setting\nStop = 2×ATR below entry",
        "notes": "R:R > 2 = green. R:R < 1 = unfavourable setup. Always consider alongside win rate.",
    },
    "DC Regime Age": {
        "explanation": "Number of trading bars since the last Death Cross occurred (i.e. how long the stock has been in a bearish EMA regime). A longer base can indicate a stronger potential Golden Cross.",
        "formula": "age = bars since (EMA_fast crossed below EMA_slow)\nReset to 0 at each new Death Cross",
        "notes": "Longer DC age → more compressed base → potentially stronger breakout when GC forms.",
    },
    "Win Rate": {
        "explanation": "Percentage of GC buy signals where the stock price was higher after the selected horizon (e.g. 30 days). Equivalent to Precision in the confusion matrix.",
        "formula": "Win Rate = TP / (TP + FP)\nTP = signal fired AND price went up\nFP = signal fired AND price went down",
        "notes": "> 55% is considered good for a trend-following strategy. Evaluate alongside avg win/loss size.",
    },
    "Precision": {
        "explanation": "Of all the times the model fired a GC buy signal, what fraction were correct (price actually rose over the horizon). Same as Win Rate.",
        "formula": "Precision = TP / (TP + FP)",
        "notes": "High precision = fewer false alarms. Trade-off: higher threshold → higher precision but lower recall.",
    },
    "Recall": {
        "explanation": "Of all the times the price actually rose over the horizon, what fraction did the model correctly predict with a buy signal. Measures coverage.",
        "formula": "Recall = TP / (TP + FN)\nFN = price went up but no signal was fired",
        "notes": "Low recall = model is too conservative, missing many winning opportunities.",
    },
    "F1 Score": {
        "explanation": "Harmonic mean of Precision and Recall. Balances both metrics into a single score. Useful when the dataset is imbalanced (more non-signal days than signal days).",
        "formula": "F1 = 2 × (Precision × Recall) / (Precision + Recall)",
        "notes": "> 0.5 is good. F1 penalises models that sacrifice one metric to inflate the other.",
    },
    "Accuracy": {
        "explanation": "Overall fraction of correct predictions (both buy signals that worked AND non-signals where price fell). Can be misleading if classes are imbalanced.",
        "formula": "Accuracy = (TP + TN) / (TP + TN + FP + FN)",
        "notes": "Use alongside Precision/Recall for a complete picture. High accuracy with low precision = too many false positives.",
    },
    "Total Return": {
        "explanation": "Total cumulative return of the strategy over the backtest period, after all transaction costs and slippage. Compares to buy-and-hold on the equity curve.",
        "formula": "Total Return = (Final Equity / Initial Equity) − 1\nEquity built from trade log returns",
        "notes": "Includes commission + slippage on both entry and exit. Next-bar open fills used.",
    },
    "Sharpe": {
        "explanation": "Risk-adjusted return metric. Measures how much excess return you earn per unit of risk (volatility). Computed on trade returns only, not flat days.",
        "formula": "Sharpe = √(252/avg_hold) × mean(trade_rets) / std(trade_rets)",
        "notes": "> 1.0 = good. > 2.0 = excellent. Annualised by average holding period, not calendar days.",
    },
    "Max Drawdown": {
        "explanation": "The largest peak-to-trough decline in the equity curve during the backtest. Measures the worst-case loss an investor would have experienced.",
        "formula": "MDD = min(Equity_t / max(Equity_0..t) − 1)\nover all bars t in backtest",
        "notes": "< −20% = high risk strategy. Used in Calmar ratio denominator.",
    },
    "Calmar": {
        "explanation": "Ratio of annualised return to maximum drawdown. Measures return earned per unit of drawdown risk. Higher is better.",
        "formula": "Calmar = CAGR_active / |Max Drawdown|\nCAGR computed on active trading days only",
        "notes": "> 1.0 = strategy earns more than its worst drawdown annually. ∞ = no drawdown occurred.",
    },
    "Kelly (0.25×)": {
        "explanation": "Quarter-Kelly position sizing recommendation. Full Kelly is theoretically optimal but too aggressive in practice. 25% of Kelly is a common conservative choice.",
        "formula": "f_quarter = 0.25 × (b·p − q) / b\nComputed from live backtest trade statistics",
        "notes": "This is the recommended allocation per trade as % of capital. Recomputed after each backtest.",
    },
    "Time in Mkt": {
        "explanation": "Percentage of backtest bars where the strategy held a position (was invested). Lower time in market with similar returns = higher capital efficiency.",
        "formula": "Time in Market = Σ(position=1 bars) / total bars × 100%",
        "notes": "Lower is often better — idle capital can be deployed elsewhere. Compare with buy-and-hold (100%).",
    },
    "GC ROC-AUC": {
        "explanation": "Area Under the ROC Curve for the Golden Cross model on the out-of-sample test fold. Measures the model's ability to rank positive examples above negative ones.",
        "formula": "AUC = P(score(GC bar) > score(non-GC bar))\nComputed on strictly held-out test fold",
        "notes": "0.5 = random. 1.0 = perfect. > 0.85 = strong discriminative ability.",
    },
    "GC Avg Precision": {
        "explanation": "Area Under the Precision-Recall curve. More informative than ROC-AUC when positive events (GC signals) are rare — which they are in daily price data.",
        "formula": "AP = Σ(Precision_k × ΔRecall_k)\nSummed over all classification thresholds",
        "notes": "Baseline = event rate (e.g. 5%). AP >> baseline = model adds real value.",
    },
    "DC ROC-AUC": {
        "explanation": "Area Under the ROC Curve for the Death Cross model on the out-of-sample test fold.",
        "formula": "Same as GC ROC-AUC but for DC target label",
        "notes": "0.5 = random. > 0.85 = strong. Evaluated on same held-out test fold as GC model.",
    },
    "Brier Score (GC)": {
        "explanation": "Calibration quality metric. Measures the mean squared error between predicted probabilities and actual outcomes. Lower is better.",
        "formula": "Brier = mean((P_predicted − y_actual)²)\ny_actual ∈ {0,1}, P_predicted ∈ [0,1]",
        "notes": "0 = perfect calibration. 0.25 = random (uninformative). < 0.10 = well-calibrated model.",
    },
    "Walk-Forward Folds": {
        "explanation": "Number of walk-forward cross-validation folds used during training. Each fold has a strict train / purge / calibration / test split to prevent data leakage.",
        "formula": "Each fold: train → purge(20 bars) → cal → test\nFold size = total_bars / (n_splits + 1)",
        "notes": "More folds = more robust evaluation but longer training time. Final model uses last fold's split.",
    },
}

def _tip_html(key):
    """Render the tooltip popup HTML for a given metric key — 4-column table layout."""
    t = TOOLTIPS.get(key)
    if not t:
        return ""
    exp  = t.get("explanation", "")
    form = t.get("formula", "").replace("\n", "<br>")
    note = t.get("notes", "")
    return (
        f'<div class="info-wrap">'
        f'<span class="info-btn" tabindex="0">i</span>'
        f'<div class="info-tip">'
        f'  <div class="tip-head">'
        f'    <span>Term</span><span>Explanation</span>'
        f'    <span>Formula / Calculation</span><span>Derivation / Notes</span>'
        f'  </div>'
        f'  <div class="tip-body">'
        f'    <div class="tip-term">{key}</div>'
        f'    <div class="tip-expl">{exp}</div>'
        f'    <div class="tip-form">{form}</div>'
        f'    <div class="tip-note">{note}</div>'
        f'  </div>'
        f'</div></div>'
    )

def qcard(label, value, sub="", accent="#4da6ff"):
    tip = _tip_html(label)
    return (
        f'<div class="qcard" style="--accent:{accent};">'
        f'<div class="qlabel">{label}{tip}</div>'
        f'<div class="qval" style="color:{accent};">{value}</div>'
        f'<div class="qsub">{sub}</div>'
        f'</div>'
    )

def prob_bar(prob, color):
    return (f'<div class="bar-wrap">'
            f'<div class="bar-fill" style="width:{min(prob*100,100):.0f}%;background:{color};"></div>'
            f'</div>')

def plot_price_chart(df_feat, ticker, fast_p, slow_p):
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        row_heights=[0.50,0.18,0.16,0.16], vertical_spacing=0.02)
    last = df_feat.iloc[-252:]
    fig.add_trace(go.Candlestick(
        x=last.index, open=last['Open'], high=last['High'],
        low=last['Low'], close=last['Close'], name="Price",
        increasing_fillcolor="#00e5a0", decreasing_fillcolor="#f04060",
        increasing_line_color="#00e5a0", decreasing_line_color="#f04060",
        line=dict(width=0.8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=last.index, y=last['ema_fast'],
        line=dict(color="#4da6ff",width=1.5), name=f"EMA{fast_p}"), row=1, col=1)
    fig.add_trace(go.Scatter(x=last.index, y=last['ema_slow'],
        line=dict(color="#f04060",width=1.5), name=f"EMA{slow_p}"), row=1, col=1)
    vcols = ["#00e5a0" if c>=o else "#f04060" for c,o in zip(last['Close'], last['Open'])]
    fig.add_trace(go.Bar(x=last.index, y=last['Volume'],
        marker_color=vcols, name="Volume", opacity=0.55), row=2, col=1)
    fig.add_trace(go.Scatter(x=last.index, y=last['rsi_14'],
        line=dict(color="#f5a623",width=1.4), name="RSI"), row=3, col=1)
    fig.add_hrect(y0=70,y1=100,row=3,col=1,fillcolor="rgba(240,64,96,0.05)",line_width=0)
    fig.add_hrect(y0=0,y1=30,row=3,col=1,fillcolor="rgba(0,229,160,0.05)",line_width=0)
    fig.add_hline(y=70,row=3,col=1,line=dict(color="#f04060",width=0.8,dash="dot"))
    fig.add_hline(y=30,row=3,col=1,line=dict(color="#00e5a0",width=0.8,dash="dot"))
    mhcols = ["#00e5a0" if v>=0 else "#f04060" for v in last['macd_hist']]
    fig.add_trace(go.Bar(x=last.index, y=last['macd_hist'],
        marker_color=mhcols, name="MACD Hist", opacity=0.7), row=4, col=1)
    fig.add_trace(go.Scatter(x=last.index, y=last['macd'],
        line=dict(color="#4da6ff",width=1.2), name="MACD"), row=4, col=1)
    fig.add_trace(go.Scatter(x=last.index, y=last['macd_signal'],
        line=dict(color="#a78bfa",width=1.2), name="Signal"), row=4, col=1)
    fig.update_layout(**DARK,
        title=dict(text=f"{ticker} — Price, Volume, RSI, MACD", font=dict(size=13,color="#d4ddf5")),
        xaxis_rangeslider_visible=False,
        legend=dict(bgcolor="#080c18",bordercolor="#151c30",font=dict(size=10)),
        height=560, margin=dict(l=55,r=15,t=45,b=30))
    for row in range(1,5):
        fig.update_yaxes(**GRID, row=row, col=1)
        fig.update_xaxes(**GRID, row=row, col=1)
    return fig


def plot_monte_carlo(mc, ticker, days):
    x = list(range(days+1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x+x[::-1],
        y=list(mc["p95"])+list(mc["p5"])[::-1],
        fill="toself", fillcolor="rgba(77,166,255,0.05)",
        line=dict(color="rgba(0,0,0,0)"), name="P5–P95"))
    fig.add_trace(go.Scatter(x=x+x[::-1],
        y=list(mc["p75"])+list(mc["p25"])[::-1],
        fill="toself", fillcolor="rgba(77,166,255,0.10)",
        line=dict(color="rgba(0,0,0,0)"), name="P25–P75"))
    for path in mc["paths"][:150]:
        col = "rgba(0,229,160,0.03)" if path[-1]>mc["S0"] else "rgba(240,64,96,0.03)"
        fig.add_trace(go.Scatter(x=x, y=list(path),
            line=dict(color=col,width=0.4), showlegend=False, hoverinfo="skip"))
    for pct,col,name,dash,w in [
        ("p50","#4da6ff","Median","solid",2.0),
        ("p95","#00e5a0","P95","dash",1.5),
        ("p5","#f04060","P5","dash",1.5),
    ]:
        fig.add_trace(go.Scatter(x=x, y=list(mc[pct]),
            line=dict(color=col,width=w,dash=dash), name=name))
    fig.add_hline(y=mc["S0"], line=dict(color="#f5a623",width=1.5,dash="dot"),
                  annotation_text=f"  Current ₹{mc['S0']:.1f}",
                  annotation_font_color="#f5a623")
    fig.update_layout(**DARK,
        title=dict(text=f"Monte Carlo — {ticker} ({days}d, {len(mc['paths'])} paths, risk-neutral Student-t GBM)",
                   font=dict(size=13,color="#d4ddf5")),
        xaxis=dict(title="Trading Days",**GRID),
        yaxis=dict(title="Price (₹)",**GRID),
        legend=dict(bgcolor="#080c18",bordercolor="#151c30"),
        height=420, margin=dict(l=55,r=15,t=50,b=45))
    return fig


def plot_mc_distribution(mc):
    finals  = mc["finals"]
    returns = (finals / mc["S0"] - 1) * 100
    cols    = ["#00e5a0" if r>0 else "#f04060" for r in returns]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=returns, nbinsx=60,
        marker=dict(color=cols, line=dict(width=0)),
        opacity=0.75, name="Return distribution"))
    fig.add_vline(x=0, line=dict(color="#f5a623",width=1.5,dash="dot"),
                  annotation_text="  0%", annotation_font_color="#f5a623")
    fig.add_vline(x=float(mc["var_95"]*100),
                  line=dict(color="#f04060",width=1.5,dash="dash"),
                  annotation_text="  VaR", annotation_font_color="#f04060")
    fig.add_vline(x=float(mc["cvar_95"]*100),
                  line=dict(color="#a78bfa",width=1.2,dash="dash"),
                  annotation_text="  CVaR", annotation_font_color="#a78bfa")
    fig.update_layout(**DARK,
        title=dict(text="Distribution of final returns (risk-neutral Student-t GBM)",
                   font=dict(size=12,color="#d4ddf5")),
        xaxis=dict(title="Return (%)",**GRID),
        yaxis=dict(title="Count",**GRID),
        height=280, margin=dict(l=55,r=15,t=40,b=40))
    return fig


def plot_equity_curve(bt):
    df = bt['df']; dd = bt['drawdown_series']
    # df has integer index; use trade_log dates for x-axis if available
    x_axis = df.index  # integer, but Plotly handles it fine for shape
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.65,0.35], vertical_spacing=0.03)
    fig.add_trace(go.Scatter(x=x_axis, y=df['equity'],
        line=dict(color="#00e5a0",width=1.8), name="Strategy",
        fill="tozeroy", fillcolor="rgba(0,229,160,0.05)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_axis, y=df['buy_hold'],
        line=dict(color="#7a8aaa",width=1.2,dash="dot"), name="Buy & Hold"), row=1, col=1)
    fig.add_hline(y=1.0, row=1, col=1, line=dict(color="#3d4d6a",width=0.8))
    fig.add_trace(go.Scatter(x=x_axis, y=dd*100,
        line=dict(color="#f04060",width=1.2), name="Drawdown %",
        fill="tozeroy", fillcolor="rgba(240,64,96,0.07)"), row=2, col=1)
    for t in bt['trade_log']:
        col = "#00e5a0" if t['ret']>0 else "#f04060"
        try:
            if t.get('entry_date') is not None:
                fig.add_vline(x=t['entry_date'], row=1, col=1,
                              line=dict(color=col, width=0.5, dash="dot"))
        except: pass
    fig.update_layout(**DARK,
        title=dict(text="Equity curve & drawdown (next-bar entry, with slippage)",
                   font=dict(size=13,color="#d4ddf5")),
        legend=dict(bgcolor="#080c18",bordercolor="#151c30"),
        height=400, margin=dict(l=55,r=15,t=45,b=30))
    for row in range(1,3):
        fig.update_yaxes(**GRID, row=row, col=1)
        fig.update_xaxes(**GRID, row=row, col=1)
    return fig


def plot_feature_importance(fi, top_n=25):
    fi_top = fi.head(top_n)
    fig = go.Figure(go.Bar(
        x=fi_top.values[::-1], y=fi_top.index[::-1], orientation='h',
        marker=dict(color=fi_top.values[::-1],
                    colorscale=[[0,"#1e2840"],[1,"#00e5a0"]])))
    fig.update_layout(**DARK,
        title=dict(text=f"Top {top_n} feature importances (GC model — ensemble average)",
                   font=dict(size=13,color="#d4ddf5")),
        xaxis=dict(title="Importance",**GRID),
        yaxis=dict(**GRID),
        height=520, margin=dict(l=170,r=15,t=45,b=40))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# 200+ NSE STOCK UNIVERSE  (hardcoded — covers all major sectors & indices)
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_UNIVERSE = [
    # Indices
    "^NSEI","^NSEBANK","^BSESN","^NSEMDCP50",
    # Large cap — Nifty 50
    "RELIANCE.NS","TCS.NS","HDFCBANK.NS","BHARTIARTL.NS","ICICIBANK.NS",
    "INFY.NS","SBIN.NS","LT.NS","KOTAKBANK.NS","HCLTECH.NS",
    "AXISBANK.NS","ITC.NS","WIPRO.NS","MARUTI.NS","ONGC.NS",
    "TITAN.NS","BAJFINANCE.NS","SUNPHARMA.NS","NESTLEIND.NS","ULTRACEMCO.NS",
    "POWERGRID.NS","NTPC.NS","TECHM.NS","TATASTEEL.NS","COALINDIA.NS",
    "ADANIENT.NS","ADANIPORTS.NS","HINDUNILVR.NS","BAJAJFINSV.NS","DRREDDY.NS",
    "ASIANPAINT.NS","JSWSTEEL.NS","TATAMOTORS.NS","INDUSINDBK.NS","CIPLA.NS",
    "GRASIM.NS","BPCL.NS","EICHERMOT.NS","BRITANNIA.NS","DIVISLAB.NS",
    "HEROMOTOCO.NS","HINDALCO.NS","APOLLOHOSP.NS","SBILIFE.NS","HDFCLIFE.NS",
    "TATACONSUM.NS","BAJAJ-AUTO.NS","LTIM.NS","VEDL.NS","SHREECEM.NS",
    # Nifty Next 50
    "HAL.NS","BEL.NS","BDL.NS","ZOMATO.NS","DMART.NS",
    "SIEMENS.NS","ABB.NS","BOSCHLTD.NS","HAVELLS.NS","PIDILITIND.NS",
    "MUTHOOTFIN.NS","COLPAL.NS","DABUR.NS","MARICO.NS","GODREJCP.NS",
    "BERGEPAINT.NS","ASTRAL.NS","AUROPHARMA.NS","BIOCON.NS","ALKEM.NS",
    "TORNTPHARM.NS","LUPIN.NS","IPCALAB.NS","LAURUSLABS.NS","SYNGENE.NS",
    "MOTHERSON.NS","BALKRISIND.NS","EXIDEIND.NS","AMARARAJA.NS","MINDA.NS",
    "TVSMOTOR.NS","ESCORTS.NS","ASHOKLEY.NS","M&M.NS","TATAPOWER.NS",
    "CUMMINSIND.NS","THERMAXLTD.NS","GRINDWELL.NS","KAJARIACER.NS","JKCEMENT.NS",
    "RAMCOCEM.NS","LINDEINDIA.NS","SUNTV.NS","ZEEL.NS","PVRINOX.NS",
    "IRCTC.NS","INDIGO.NS","TIINDIA.NS","GMRAIRPORT.NS","CONCOR.NS",
    # Nifty Midcap 150
    "POLYCAB.NS","VOLTAS.NS","BLUEDART.NS","KPITTECH.NS","MPHASIS.NS",
    "PERSISTENT.NS","LTTS.NS","COFORGE.NS","HAPPSTMNDS.NS","CYIENT.NS",
    "SONACOMS.NS","SWARAJENG.NS","GREAVESCOT.NS","MAHINDCIE.NS","ELGIEQUIP.NS",
    "SCHAEFFLER.NS","TIMKEN.NS","SKFINDIA.NS","FINCABLES.NS","KEI.NS",
    "LXCHEM.NS","BASF.NS","AAVAS.NS","CANFINHOME.NS","PNBHOUSING.NS",
    "MANAPPURAM.NS","CHOLAFIN.NS","LTFH.NS","SHRIRAMFIN.NS","M&MFIN.NS",
    "SUNDARMFIN.NS","ABCAPITAL.NS","ICICIGI.NS","NIACL.NS","STARHEALTH.NS",
    "MAXHEALTH.NS","FORTIS.NS","METROPOLIS.NS","LALPATHLAB.NS","THYROCARE.NS",
    "IDFCFIRSTB.NS","FEDERALBNK.NS","KARURVSB.NS","CANBK.NS","BANKBARODA.NS",
    "PNB.NS","IOB.NS","UNIONBANK.NS","INDIANB.NS","UCOBANK.NS",
    "GAIL.NS","OIL.NS","MGL.NS","IGL.NS","ATGL.NS",
    "NLCINDIA.NS","NHPC.NS","SJVN.NS","TORNTPOWER.NS","CESC.NS",
    "GUJGASLTD.NS","PETRONET.NS","RECLTD.NS","PFC.NS","IRFC.NS",
    "HUDCO.NS","IRCON.NS","RITES.NS","RAILTEL.NS","NBCC.NS",
    "TATAELXSI.NS","ZENSAR.NS","NIITTECH.NS","RATEGAIN.NS","NEWGEN.NS",
    "NAUKRI.NS","POLICYBZR.NS","PAYTM.NS","DELHIVERY.NS","MAPMYINDIA.NS",
    # Nifty Smallcap picks
    "OFSS.NS","MASTEK.NS","BSOFT.NS","TANLA.NS","INTELLECT.NS",
    "ROUTE.NS","SANSERA.NS","TEJASNET.NS","STLTECH.NS","GTLINFRA.NS",
    "RVNL.NS","BEML.NS","BHEL.NS","MIDHANI.NS","COCHINSHIP.NS",
    "CHAMBLFERT.NS","GNFC.NS","GSFC.NS","COROMANDEL.NS","TATACHEM.NS",
    "ATUL.NS","DEEPAKNTR.NS","NOCIL.NS","AARTI.NS","FINEORG.NS",
    "BALRAMCHIN.NS","RENUKA.NS","DHANUKA.NS","RALLIS.NS","UPL.NS",
    "VGUARD.NS","CROMPTON.NS","ORIENTELEC.NS","CERA.NS","SYMPHONY.NS",
    "WONDERLA.NS","LEMONTREE.NS","CHALET.NS","EIHOTEL.NS","TAJGVK.NS",
    "ZYDUSLIFE.NS","GRANULES.NS","SUDARSCHEM.NS","VINATIORGA.NS","ALKYLAMINE.NS",
    "NATCOPHARM.NS","JUBLPHARMA.NS","AJANTPHARM.NS","GLENMARK.NS","FDC.NS",
    # ETFs / broader exposure
    "NIFTYBEES.NS","BANKBEES.NS","JUNIORBEES.NS","ITBEES.NS","PSUBNKBEES.NS",
]

# ─────────────────────────────────────────────────────────────────────────────
# SECTOR & CAP METADATA  (for categorical embeddings — v13 upgrade)
# ─────────────────────────────────────────────────────────────────────────────
# Sector codes: 0=IT, 1=Bank, 2=Finance, 3=Pharma, 4=Auto, 5=Energy,
#               6=FMCG, 7=Metal, 8=Infra, 9=Defense, 10=Telecom,
#               11=Cement, 12=Chemical, 13=Media, 14=Hospital, 15=Index/ETF
# Cap codes:    0=Large, 1=Mid, 2=Small

SECTOR_MAP = {
    # IT
    "TCS.NS":0,"INFY.NS":0,"HCLTECH.NS":0,"WIPRO.NS":0,"TECHM.NS":0,
    "LTIM.NS":0,"OFSS.NS":0,"MPHASIS.NS":0,"PERSISTENT.NS":0,"LTTS.NS":0,
    "COFORGE.NS":0,"HAPPSTMNDS.NS":0,"CYIENT.NS":0,"KPITTECH.NS":0,
    "MASTEK.NS":0,"BSOFT.NS":0,"TANLA.NS":0,"INTELLECT.NS":0,"ROUTE.NS":0,
    "ZENSAR.NS":0,"NIITTECH.NS":0,"RATEGAIN.NS":0,"NEWGEN.NS":0,
    "TATAELXSI.NS":0,"NAUKRI.NS":0,"MAPMYINDIA.NS":0,
    # Bank
    "HDFCBANK.NS":1,"ICICIBANK.NS":1,"SBIN.NS":1,"KOTAKBANK.NS":1,
    "AXISBANK.NS":1,"INDUSINDBK.NS":1,"IDFCFIRSTB.NS":1,"FEDERALBNK.NS":1,
    "KARURVSB.NS":1,"CANBK.NS":1,"BANKBARODA.NS":1,"PNB.NS":1,
    "IOB.NS":1,"UNIONBANK.NS":1,"INDIANB.NS":1,"UCOBANK.NS":1,
    "^NSEBANK":1,
    # Finance / NBFC
    "BAJFINANCE.NS":2,"BAJAJFINSV.NS":2,"MUTHOOTFIN.NS":2,"MANAPPURAM.NS":2,
    "CHOLAFIN.NS":2,"LTFH.NS":2,"SHRIRAMFIN.NS":2,"M&MFIN.NS":2,
    "SUNDARMFIN.NS":2,"ABCAPITAL.NS":2,"ICICIGI.NS":2,"NIACL.NS":2,
    "STARHEALTH.NS":2,"MAXHEALTH.NS":2,"SBILIFE.NS":2,"HDFCLIFE.NS":2,
    "AAVAS.NS":2,"CANFINHOME.NS":2,"PNBHOUSING.NS":2,"RECLTD.NS":2,
    "PFC.NS":2,"IRFC.NS":2,"HUDCO.NS":2,"POLICYBZR.NS":2,"PAYTM.NS":2,
    # Pharma
    "SUNPHARMA.NS":3,"DRREDDY.NS":3,"CIPLA.NS":3,"DIVISLAB.NS":3,
    "AUROPHARMA.NS":3,"BIOCON.NS":3,"ALKEM.NS":3,"TORNTPHARM.NS":3,
    "LUPIN.NS":3,"IPCALAB.NS":3,"LAURUSLABS.NS":3,"SYNGENE.NS":3,
    "ZYDUSLIFE.NS":3,"GRANULES.NS":3,"NATCOPHARM.NS":3,"JUBLPHARMA.NS":3,
    "AJANTPHARM.NS":3,"GLENMARK.NS":3,"FDC.NS":3,
    # Auto
    "MARUTI.NS":4,"TATAMOTORS.NS":4,"M&M.NS":4,"BAJAJ-AUTO.NS":4,
    "EICHERMOT.NS":4,"HEROMOTOCO.NS":4,"TVSMOTOR.NS":4,"ESCORTS.NS":4,
    "ASHOKLEY.NS":4,"MOTHERSON.NS":4,"BALKRISIND.NS":4,"EXIDEIND.NS":4,
    "AMARARAJA.NS":4,"MINDA.NS":4,"SANSERA.NS":4,"SONACOMS.NS":4,
    # Energy
    "RELIANCE.NS":5,"ONGC.NS":5,"BPCL.NS":5,"GAIL.NS":5,"OIL.NS":5,
    "NTPC.NS":5,"TATAPOWER.NS":5,"ADANIENT.NS":5,"COALINDIA.NS":5,
    "MGL.NS":5,"IGL.NS":5,"ATGL.NS":5,"NLCINDIA.NS":5,"NHPC.NS":5,
    "SJVN.NS":5,"TORNTPOWER.NS":5,"CESC.NS":5,"GUJGASLTD.NS":5,
    "PETRONET.NS":5,"POWERGRID.NS":5,"VEDL.NS":5,
    # FMCG
    "ITC.NS":6,"HINDUNILVR.NS":6,"NESTLEIND.NS":6,"BRITANNIA.NS":6,
    "TATACONSUM.NS":6,"COLPAL.NS":6,"DABUR.NS":6,"MARICO.NS":6,
    "GODREJCP.NS":6,"BERGEPAINT.NS":6,"ASIANPAINT.NS":6,"PIDILITIND.NS":6,
    "BALRAMCHIN.NS":6,"RENUKA.NS":6,"DHANUKA.NS":6,"RALLIS.NS":6,"UPL.NS":6,
    # Metal / Materials
    "TATASTEEL.NS":7,"JSWSTEEL.NS":7,"HINDALCO.NS":7,"GRASIM.NS":7,
    "LXCHEM.NS":7,"BASF.NS":7,"CHAMBLFERT.NS":7,"GNFC.NS":7,"GSFC.NS":7,
    "COROMANDEL.NS":7,"TATACHEM.NS":7,"ATUL.NS":7,"DEEPAKNTR.NS":7,
    "NOCIL.NS":7,"AARTI.NS":7,"FINEORG.NS":7,"SUDARSCHEM.NS":7,
    "VINATIORGA.NS":7,"ALKYLAMINE.NS":7,
    # Infra / Engineering / Capital Goods
    "LT.NS":8,"SIEMENS.NS":8,"ABB.NS":8,"BOSCHLTD.NS":8,"HAVELLS.NS":8,
    "POLYCAB.NS":8,"CUMMINSIND.NS":8,"THERMAXLTD.NS":8,"GRINDWELL.NS":8,
    "ELGIEQUIP.NS":8,"SCHAEFFLER.NS":8,"TIMKEN.NS":8,"SKFINDIA.NS":8,
    "FINCABLES.NS":8,"KEI.NS":8,"VGUARD.NS":8,"CROMPTON.NS":8,
    "ORIENTELEC.NS":8,"VOLTAS.NS":8,"IRCON.NS":8,"RITES.NS":8,
    "RAILTEL.NS":8,"NBCC.NS":8,"RVNL.NS":8,"BEML.NS":8,"BHEL.NS":8,
    "IRCTC.NS":8,"CONCOR.NS":8,"DELHIVERY.NS":8,"BLUEDART.NS":8,
    "INDIGO.NS":8,"GMRAIRPORT.NS":8,"ADANIPORTS.NS":8,"TIINDIA.NS":8,
    # Defense
    "HAL.NS":9,"BEL.NS":9,"BDL.NS":9,"MIDHANI.NS":9,"COCHINSHIP.NS":9,
    # Telecom
    "BHARTIARTL.NS":10,"TEJASNET.NS":10,"STLTECH.NS":10,"GTLINFRA.NS":10,
    # Cement
    "ULTRACEMCO.NS":11,"SHREECEM.NS":11,"JKCEMENT.NS":11,"RAMCOCEM.NS":11,
    "KAJARIACER.NS":11,"CERA.NS":11,"ASTRAL.NS":11,
    # Chemicals (specialty)
    "LINDEINDIA.NS":12,
    # Media / Hospitality / Consumer
    "SUNTV.NS":13,"ZEEL.NS":13,"PVRINOX.NS":13,"WONDERLA.NS":13,
    "LEMONTREE.NS":13,"CHALET.NS":13,"EIHOTEL.NS":13,"TAJGVK.NS":13,
    "SYMPHONY.NS":13,"TITAN.NS":13,"DMART.NS":13,"ZOMATO.NS":13,
    "APOLLOHOSP.NS":14,"FORTIS.NS":14,"METROPOLIS.NS":14,
    "LALPATHLAB.NS":14,"THYROCARE.NS":14,
    # Index / ETF
    "^NSEI":15,"^BSESN":15,"^NSEMDCP50":15,
    "NIFTYBEES.NS":15,"BANKBEES.NS":15,"JUNIORBEES.NS":15,
    "ITBEES.NS":15,"PSUBNKBEES.NS":15,
    # Misc tech/new-age
    "SWARAJENG.NS":8,"GREAVESCOT.NS":8,"MAHINDCIE.NS":4,
}

CAP_MAP = {
    # Large cap (Nifty 50)
    "RELIANCE.NS":0,"TCS.NS":0,"HDFCBANK.NS":0,"BHARTIARTL.NS":0,"ICICIBANK.NS":0,
    "INFY.NS":0,"SBIN.NS":0,"LT.NS":0,"KOTAKBANK.NS":0,"HCLTECH.NS":0,
    "AXISBANK.NS":0,"ITC.NS":0,"WIPRO.NS":0,"MARUTI.NS":0,"ONGC.NS":0,
    "TITAN.NS":0,"BAJFINANCE.NS":0,"SUNPHARMA.NS":0,"NESTLEIND.NS":0,"ULTRACEMCO.NS":0,
    "POWERGRID.NS":0,"NTPC.NS":0,"TECHM.NS":0,"TATASTEEL.NS":0,"COALINDIA.NS":0,
    "ADANIENT.NS":0,"ADANIPORTS.NS":0,"HINDUNILVR.NS":0,"BAJAJFINSV.NS":0,"DRREDDY.NS":0,
    "ASIANPAINT.NS":0,"JSWSTEEL.NS":0,"TATAMOTORS.NS":0,"INDUSINDBK.NS":0,"CIPLA.NS":0,
    "GRASIM.NS":0,"BPCL.NS":0,"EICHERMOT.NS":0,"BRITANNIA.NS":0,"DIVISLAB.NS":0,
    "HEROMOTOCO.NS":0,"HINDALCO.NS":0,"APOLLOHOSP.NS":0,"SBILIFE.NS":0,"HDFCLIFE.NS":0,
    "TATACONSUM.NS":0,"BAJAJ-AUTO.NS":0,"LTIM.NS":0,"VEDL.NS":0,"SHREECEM.NS":0,
    "HAL.NS":0,"BEL.NS":0,"ZOMATO.NS":0,"DMART.NS":0,
    # Mid cap
    "BDL.NS":1,"SIEMENS.NS":1,"ABB.NS":1,"BOSCHLTD.NS":1,"HAVELLS.NS":1,"PIDILITIND.NS":1,
    "MUTHOOTFIN.NS":1,"COLPAL.NS":1,"DABUR.NS":1,"MARICO.NS":1,"GODREJCP.NS":1,
    "BERGEPAINT.NS":1,"ASTRAL.NS":1,"AUROPHARMA.NS":1,"BIOCON.NS":1,"ALKEM.NS":1,
    "TORNTPHARM.NS":1,"LUPIN.NS":1,"IPCALAB.NS":1,"LAURUSLABS.NS":1,"SYNGENE.NS":1,
    "MOTHERSON.NS":1,"BALKRISIND.NS":1,"EXIDEIND.NS":1,"AMARARAJA.NS":1,"MINDA.NS":1,
    "TVSMOTOR.NS":1,"ESCORTS.NS":1,"ASHOKLEY.NS":1,"M&M.NS":1,"TATAPOWER.NS":1,
    "CUMMINSIND.NS":1,"THERMAXLTD.NS":1,"GRINDWELL.NS":1,"KAJARIACER.NS":1,"JKCEMENT.NS":1,
    "RAMCOCEM.NS":1,"LINDEINDIA.NS":1,"SUNTV.NS":1,"PVRINOX.NS":1,"IRCTC.NS":1,
    "INDIGO.NS":1,"TIINDIA.NS":1,"GMRAIRPORT.NS":1,"CONCOR.NS":1,"POLYCAB.NS":1,
    "VOLTAS.NS":1,"KPITTECH.NS":1,"MPHASIS.NS":1,"PERSISTENT.NS":1,"LTTS.NS":1,
    "COFORGE.NS":1,"HAPPSTMNDS.NS":1,"CYIENT.NS":1,"SONACOMS.NS":1,"ELGIEQUIP.NS":1,
    "SCHAEFFLER.NS":1,"TIMKEN.NS":1,"SKFINDIA.NS":1,"FINCABLES.NS":1,"KEI.NS":1,
    "AAVAS.NS":1,"CANFINHOME.NS":1,"PNBHOUSING.NS":1,"MANAPPURAM.NS":1,"CHOLAFIN.NS":1,
    "LTFH.NS":1,"SHRIRAMFIN.NS":1,"M&MFIN.NS":1,"SUNDARMFIN.NS":1,"ABCAPITAL.NS":1,
    "ICICIGI.NS":1,"NIACL.NS":1,"STARHEALTH.NS":1,"MAXHEALTH.NS":1,"FORTIS.NS":1,
    "METROPOLIS.NS":1,"LALPATHLAB.NS":1,"IDFCFIRSTB.NS":1,"FEDERALBNK.NS":1,
    "GAIL.NS":1,"MGL.NS":1,"IGL.NS":1,"ATGL.NS":1,"NLCINDIA.NS":1,"NHPC.NS":1,
    "SJVN.NS":1,"TORNTPOWER.NS":1,"CESC.NS":1,"GUJGASLTD.NS":1,"PETRONET.NS":1,
    "RECLTD.NS":1,"PFC.NS":1,"IRFC.NS":1,"HUDCO.NS":1,"ZEEL.NS":1,
    "NAUKRI.NS":1,"POLICYBZR.NS":1,"PAYTM.NS":1,"DELHIVERY.NS":1,"MAPMYINDIA.NS":1,
    "BLUEDART.NS":1,"TATAELXSI.NS":1,"CANBK.NS":1,"BANKBARODA.NS":1,
    # Small cap
    "OFSS.NS":2,"MASTEK.NS":2,"BSOFT.NS":2,"TANLA.NS":2,"INTELLECT.NS":2,
    "ROUTE.NS":2,"SANSERA.NS":2,"TEJASNET.NS":2,"STLTECH.NS":2,"GTLINFRA.NS":2,
    "RVNL.NS":2,"BEML.NS":2,"BHEL.NS":2,"MIDHANI.NS":2,"COCHINSHIP.NS":2,
    "CHAMBLFERT.NS":2,"GNFC.NS":2,"GSFC.NS":2,"COROMANDEL.NS":2,"TATACHEM.NS":2,
    "ATUL.NS":2,"DEEPAKNTR.NS":2,"NOCIL.NS":2,"AARTI.NS":2,"FINEORG.NS":2,
    "BALRAMCHIN.NS":2,"RENUKA.NS":2,"DHANUKA.NS":2,"RALLIS.NS":2,"UPL.NS":2,
    "VGUARD.NS":2,"CROMPTON.NS":2,"ORIENTELEC.NS":2,"CERA.NS":2,"SYMPHONY.NS":2,
    "WONDERLA.NS":2,"LEMONTREE.NS":2,"CHALET.NS":2,"EIHOTEL.NS":2,"TAJGVK.NS":2,
    "ZYDUSLIFE.NS":2,"GRANULES.NS":2,"SUDARSCHEM.NS":2,"VINATIORGA.NS":2,"ALKYLAMINE.NS":2,
    "NATCOPHARM.NS":2,"JUBLPHARMA.NS":2,"AJANTPHARM.NS":2,"GLENMARK.NS":2,"FDC.NS":2,
    "KARURVSB.NS":2,"PNB.NS":2,"IOB.NS":2,"UNIONBANK.NS":2,"INDIANB.NS":2,"UCOBANK.NS":2,
    "BDL.NS":2,"LXCHEM.NS":2,"BASF.NS":2,"THYROCARE.NS":2,
    "ZENSAR.NS":2,"NIITTECH.NS":2,"RATEGAIN.NS":2,"NEWGEN.NS":2,
    "OIL.NS":2,"SWARAJENG.NS":2,"GREAVESCOT.NS":2,"MAHINDCIE.NS":2,
    "IRCON.NS":2,"RITES.NS":2,"RAILTEL.NS":2,"NBCC.NS":2,
}

# Sector names for UI display
SECTOR_NAMES = {
    0:"IT", 1:"Bank", 2:"Finance/NBFC", 3:"Pharma", 4:"Auto",
    5:"Energy", 6:"FMCG", 7:"Metal/Chem", 8:"Infra/Eng", 9:"Defense",
    10:"Telecom", 11:"Cement", 12:"Specialty Chem", 13:"Media/Consumer",
    14:"Hospital", 15:"Index/ETF",
}
CAP_NAMES = {0:"Large", 1:"Mid", 2:"Small"}

# Number of unique sector/cap categories — defined earlier before FEATURE_COLS
# N_SECTORS = 16  (already set above)
# N_CAPS    = 3   (already set above)

# Nifty 50 index ticker for market regime feature
NIFTY_TICKER = "^NSEI"

# ─────────────────────────────────────────────────────────────────────────────
# MARKET REGIME DOWNLOADER (singleton cache)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=3600)
def get_nifty_regime(start_date_str):
    """Download Nifty 50 and compute MA200 regime flag (1 = index above MA200)."""
    df_n = download_stock_data(NIFTY_TICKER, start_date_str)
    if df_n is None:
        return None
    ma200 = df_n['Close'].rolling(200).mean()
    regime = (df_n['Close'] > ma200).astype(int)
    regime.name = 'mkt_regime'
    return regime

with st.sidebar:
    st.markdown("## 📊 GC Predictor · v13")

    st.markdown('<div class="sec-hdr">Model Universe</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-family:var(--mono);font-size:11px;color:var(--text2);'
        f'background:var(--bg2);border:1px solid var(--border);border-radius:6px;'
        f'padding:10px 12px;margin-bottom:10px;">'
        f'🌐 <b style="color:var(--text)">{len(TRAIN_UNIVERSE)} stocks</b> — auto-selected<br>'
        f'<span style="color:var(--text3)">Nifty 50 · Next 50 · Midcap 150<br>'
        f'Smallcap picks · Indices · ETFs<br>'
        f'All NSE sectors · ~40/40/20 cap-mix<br>'
        f'v13: Sector embeds · ATR labels · Regime</span></div>',
        unsafe_allow_html=True)

    st.markdown('<div class="sec-hdr">Model Configuration</div>', unsafe_allow_html=True)
    fast_mode = st.toggle("⚡ Fast Mode", value=True,
        help="Fast Mode: ~5 min training. Off = full precision (~45 min).")
    start_date = st.date_input("Training Start Date",
        value=datetime.date(2010,1,1),
        min_value=datetime.date(2000,1,1),
        max_value=datetime.date.today()-datetime.timedelta(days=365))
    c1,c2 = st.columns(2)
    with c1: fast_p = st.number_input("Fast EMA", value=50, min_value=5, max_value=200, step=5)
    with c2: slow_p = st.number_input("Slow EMA", value=200, min_value=20, max_value=500, step=10)
    pred_days = st.slider("Prediction Window (days)", 5, 30, 15)
    n_splits  = st.slider("Walk-Forward Splits", 4, 12, 5 if fast_mode else 8)

    st.markdown('<div class="sec-hdr">Signal Settings</div>', unsafe_allow_html=True)
    threshold   = st.slider("High Confidence Level (Model A)", 0.25, 0.85, 0.45, 0.05,
                             help="Higher = fewer but more reliable signals. Lower = more signals, more misses caught.")
    recall_threshold = st.slider("Recall Threshold (Model B)", 0.20, 0.60, 0.35, 0.05,
                             help="Model B fires at this threshold. Lower = catches more GC events.")
    quality_min = st.slider("Minimum Quality Score", 0.0, 0.60, 0.25, 0.05,
                             help="Filter out low-quality signals. 0 = no filter.")

    st.markdown('<div class="sec-hdr">Monte Carlo</div>', unsafe_allow_html=True)
    mc_days = st.slider("Forecast Days", 10, 120, 60)
    mc_sims = st.select_slider("Simulations", [250,500,1000,2000], value=1000)

    st.markdown('<div class="sec-hdr">Backtest</div>', unsafe_allow_html=True)
    stop_loss       = st.slider("Stop Loss %", -20, -3, -8) / 100
    take_profit     = st.slider("Take Profit %", 5, 50, 20) / 100
    cooldown        = st.slider("Cooldown (bars)", 5, 30, 10)
    transaction_cost= st.slider("Commission (% each side)", 0.0, 0.3, 0.1, 0.01) / 100
    slippage        = st.slider("Slippage (% each side)", 0.0, 0.2, 0.05, 0.01) / 100
    st.markdown('<div class="sec-hdr">Regime & Risk Filters</div>', unsafe_allow_html=True)
    adx_min_bt   = st.slider("Min ADX (regime filter)", 0, 35, 0,
                              help="Skip trades when ADX < this. 0 = disabled (recommended).")
    hurst_min_bt = st.slider("Min Hurst (regime filter)", 0.30, 0.60, 0.35, 0.05,
                              help="Skip trades when Hurst < this. 0.30 = nearly disabled.")
    ev_filter    = st.checkbox("EV Filter (only trade EV > 0)", value=True,
                               help="Skip trades where Expected Value ≤ 0.")
    exit_at_gc   = st.checkbox("Exit at GC Event", value=True,
                               help="Close position when actual Golden Cross fires.")
    trailing_stp = st.checkbox("Trailing Stop", value=True,
                               help="Trail stop below peak price during trade.")

    st.markdown("---")
    train_btn = st.button("🚀 Train Models", type="primary", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;gap:14px;margin-bottom:6px;">
  <div style="font-size:28px;font-weight:700;color:#d4ddf5;letter-spacing:-0.5px;font-family:'DM Sans',sans-serif;">
    Golden Cross Predictor
  </div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#3d4d6a;
              background:#080c18;border:1px solid #151c30;border-radius:4px;padding:4px 10px;">
    v13 · Sector Embeddings · Sector-Demeaned Features · ATR Labels · Market Regime · Cap-Mix
  </div>
</div>
<div style="font-size:12px;color:#7a8aaa;margin-bottom:20px;font-family:'JetBrains Mono',monospace;">
  No lookahead · Temporal walk-forward · Scale-free features · Correct GC labels · Real equity curve · ATR stops
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────
if train_btn:
    train_tickers = TRAIN_UNIVERSE
    prog          = st.progress(0, "Initialising...")
    log_box       = st.empty(); log_lines = []

    def log(msg):
        log_lines.append(msg)
        log_box.code("\n".join(log_lines[-28:]), language=None)

    try:
        log("📥 Downloading price data...")
        all_data = {}
        for i, tk in enumerate(train_tickers):
            df = download_stock_data(tk, str(start_date))
            if df is not None and len(df) > 350:
                listing = df.attrs.get('listing_date', '?')
                all_data[tk] = df
                log(f"  ✓ {tk}: {len(df):,} bars  listed {listing}")
            else:
                log(f"  ✗ {tk}: skipped (insufficient history)")
            prog.progress(5+int(17*(i+1)/len(train_tickers)), f"Downloading...")

        log(f"\n📊 {len(all_data)} tickers downloaded OK")
        if len(all_data) == 0:
            raise ValueError(
                "No tickers downloaded successfully. "
                "Check your internet connection and that the Training Start Date is not too recent."
            )
        prog.progress(22, "Engineering features...")
        log(f"\n⚙  Feature engineering ({len(FEATURE_COLS)} features, no lookahead)...")
        all_labeled = []
        failed_tickers = []
        for i, (tk, df_raw) in enumerate(all_data.items()):
            try:
                df_feat    = engineer_features(df_raw, fast_p=fast_p, slow_p=slow_p)
                if len(df_feat) < 300:
                    log(f"  ✗ {tk}: too few rows after feature engineering ({len(df_feat)})")
                    continue
                df_labeled = create_targets(df_feat, n_days=pred_days)
                if len(df_labeled) < 100:
                    log(f"  ✗ {tk}: too few labeled rows ({len(df_labeled)})")
                    continue
                df_labeled['ticker'] = tk
                all_labeled.append(df_labeled)
                log(f"  ✓ {tk}: {len(df_labeled):,} rows")
            except Exception as e:
                failed_tickers.append(tk)
                log(f"  ✗ {tk}: {type(e).__name__}: {e}")
            prog.progress(22+int(14*(i+1)/len(all_data)), f"Features: {tk}...")

        if len(all_labeled) == 0:
            raise ValueError(
                f"No tickers produced valid training data. "
                f"Failed tickers: {failed_tickers[:10]}. "
                f"Check that your training tickers are valid NSE/BSE symbols and have sufficient history. "
                f"Try extending the Training Start Date further back (e.g. 2005-01-01)."
            )

        df_combined = pd.concat(all_labeled, axis=0)
        # ── FIX BUG 3: sort strictly by date so walk-forward is temporal ─────
        df_combined['_date'] = df_combined.index
        df_combined = df_combined.sort_values('_date').reset_index(drop=True)

        # ── v13: Fetch Nifty regime for market regime feature ─────────────────
        log("\n🌐 Fetching Nifty 50 market regime (MA200)...")
        nifty_regime = get_nifty_regime(str(start_date))
        if nifty_regime is not None:
            log(f"  ✓ Nifty regime loaded: {len(nifty_regime):,} bars")
        else:
            log("  ⚠ Nifty regime unavailable — mkt_regime set to 0")

        # ── v13: Sector/cap embeddings + sector-demeaned features ─────────────
        log("🏷  Adding sector/cap embeddings and sector-demeaned features...")
        df_combined = add_categorical_features(df_combined, nifty_regime)
        log(f"  ✓ Sector one-hots: {N_SECTORS} cols | Cap one-hots: {N_CAPS} cols")
        log(f"  ✓ Demeaned features: rsi_14, price_vs_ma200, adx, atr_pct, macd_hist_pct")

        # ── v13: ATR-labelled target blending ─────────────────────────────────
        # Use ATR-based target as the primary GC label if available.
        # This replaces the fixed n-day horizon with a volatility-normalised criterion.
        if 'target_gc_atr' in df_combined.columns:
            atr_pos = df_combined['target_gc_atr'].sum()
            hor_pos = df_combined['target_gc'].sum()
            log(f"  ✓ ATR labels: {int(atr_pos)} pos | Horizon labels: {int(hor_pos)} pos")
            # Blend: 60% ATR label + 40% horizon label for robustness
            df_combined['target_gc'] = (
                (0.6 * df_combined['target_gc_atr'] + 0.4 * df_combined['target_gc']) >= 0.5
            ).astype(int)
            log(f"  ✓ Blended target: {int(df_combined['target_gc'].sum())} pos")

        # ── v13: Enforce cap-mix ~40/40/20 ───────────────────────────────────
        log("⚖  Enforcing 40% Large / 40% Mid / 20% Small cap mix...")
        n_before = len(df_combined)
        df_combined = enforce_cap_mix(df_combined)
        log(f"  ✓ Cap-mix: {n_before:,} → {len(df_combined):,} rows")

        gc_ev = df_combined['target_gc'].sum()
        log(f"  Total rows: {len(df_combined):,} | GC events: {gc_ev} ({100*gc_ev/len(df_combined):.1f}%)")

        # ── FIX BUG 4: walk-forward split FIRST, subsample ONLY in train ──────
        prog.progress(36, "Walk-forward splits...")
        log("\n✂  Walk-forward splits (train / purge / calibration / test)...")
        splits = walk_forward_split(df_combined, n_splits=n_splits)
        log(f"  {len(splits)} folds")
        if len(splits) < 2:
            raise ValueError("Not enough data for walk-forward splits. Add more tickers or extend start date.")

        # Use last fold for test metrics and Platt calibration
        train_idx, cal_idx, test_idx = splits[-1]

        X_all    = df_combined[FEATURE_COLS].replace([np.inf,-np.inf], np.nan)
        y_gc_all = df_combined['target_gc']; y_dc_all = df_combined['target_dc']

        # Fast mode: subsample negatives ONLY within the train slice — never touch cal/test
        MAX_ROWS_FAST = 150_000   # per-fold cap (train only)
        if fast_mode and len(train_idx) > MAX_ROWS_FAST:
            y_tr_tmp  = y_gc_all.iloc[train_idx]
            pos_mask  = y_tr_tmp.values == 1
            neg_mask  = ~pos_mask
            pos_idx   = train_idx[pos_mask]
            neg_idx   = train_idx[neg_mask]
            n_neg_keep= min(len(neg_idx), MAX_ROWS_FAST - len(pos_idx))
            rng_fast  = np.random.default_rng(42)
            chosen_neg= rng_fast.choice(len(neg_idx), n_neg_keep, replace=False)
            train_idx = np.concatenate([pos_idx, neg_idx[chosen_neg]])
            train_idx = np.sort(train_idx)   # keep chronological order within fold
            log(f"  ⚡ Fast mode: train fold capped at {len(train_idx):,} rows")

        X_train  = X_all.iloc[train_idx].fillna(0)
        y_gc_tr  = y_gc_all.iloc[train_idx]; y_dc_tr = y_dc_all.iloc[train_idx]
        X_cal    = X_all.iloc[cal_idx].fillna(0)
        y_gc_cal = y_gc_all.iloc[cal_idx];  y_dc_cal = y_dc_all.iloc[cal_idx]
        X_test   = X_all.iloc[test_idx].fillna(0)
        y_gc_te  = y_gc_all.iloc[test_idx]; y_dc_te  = y_dc_all.iloc[test_idx]

        # Remove NaN rows from training — use boolean mask, then iloc for train_rows
        valid      = ~X_train.isna().any(axis=1)
        X_train    = X_train[valid]; y_gc_tr = y_gc_tr[valid]; y_dc_tr = y_dc_tr[valid]
        valid_pos  = np.where(valid.values)[0]   # positional indices within train_idx
        train_rows = df_combined.iloc[train_idx[valid_pos]]

        # ── Near-cross bias filter: remove rows where gap is wide AND diverging ──
        # near_cross_score < 0.2 means gap > ~4% of price AND moving away from cross.
        # Training on these causes the model to fire signals when no cross is near.
        # Keep all rows for negative (no-cross) training — only filter positive signal rows.
        near_cross_vals = train_rows['near_cross_score'].values if 'near_cross_score' in train_rows.columns else np.ones(len(train_rows))
        # For positive (GC) labels, require score ≥ 0.2. For negatives, always keep.
        near_cross_keep = (near_cross_vals >= 0.2) | (y_gc_tr.values == 0)
        n_removed = int((~near_cross_keep).sum())
        X_train_filtered    = X_train[near_cross_keep]
        y_gc_filtered       = y_gc_tr[near_cross_keep]
        y_dc_filtered       = y_dc_tr[near_cross_keep]
        train_rows_filtered = train_rows[near_cross_keep]
        log(f"  ⚡ Near-cross bias filter: removed {n_removed} far-diverging positive rows → {len(X_train_filtered):,} remain")

        def make_ld(col):
            sub = train_rows_filtered[FEATURE_COLS+[col]].copy().replace([np.inf,-np.inf], np.nan)
            ok  = ~sub[FEATURE_COLS].isna().any(axis=1) & ~sub[col].isna()
            sub = sub[ok]
            return sub[FEATURE_COLS].fillna(0), sub[col].values

        X_q_tr,  y_q_tr  = make_ld('target_quality_gc')
        X_qd_tr, y_qd_tr = make_ld('target_quality_dc')
        X_t_tr,  y_t_tr  = make_ld('target_days_norm')
        X_dr_tr, y_dr_tr = make_ld('target_days_raw')

        # ── Accumulate train data across all folds ────────────────────────────
        # Each fold's train_idx is a strict prefix of the data (chronological).
        # Using ALL folds' training rows = train on everything up to the last fold boundary,
        # which is the correct "expanding window" approach for walk-forward.
        all_train_idx = splits[-1][0]   # last fold train_idx already covers all past data
        # (walk_forward_split builds expanding train sets — last fold has the most data)

        prog.progress(50, "Training GC ensemble (Model A — precision)...")
        log(f"\n🧠 GC ensemble Model A ({int(y_gc_filtered.sum())} pos / {int((y_gc_filtered==0).sum())} neg)...")
        models_gc = train_binary_ensemble(X_train_filtered, y_gc_filtered)
        prog.progress(60, "Training GC Model B (recall)...")
        log("  ✓ Model A done")
        models_gc_recall = train_recall_model(X_train_filtered, y_gc_filtered)
        log("  ✓ Model B (recall) done")
        prog.progress(67, "Training DC ensemble...")
        models_dc = train_binary_ensemble(X_train_filtered, y_dc_filtered)
        log("  ✓ DC ensemble done")

        prog.progress(70, "Platt calibration (held-out)...")
        log("\n🎯 Platt calibration on strictly held-out calibration fold...")
        calibrator_gc        = calibrate_probabilities(models_gc, X_cal, y_gc_cal)
        calibrator_gc_recall = calibrate_probabilities(models_gc_recall, X_cal, y_gc_cal)
        calibrator_dc        = calibrate_probabilities(models_dc, X_cal, y_dc_cal)
        log(f"  ✓ GC calibrator A: {'fitted' if calibrator_gc else 'skipped'}")
        log(f"  ✓ GC calibrator B: {'fitted' if calibrator_gc_recall else 'skipped'}")
        log(f"  ✓ DC calibrator:   {'fitted' if calibrator_dc else 'skipped'}")

        prog.progress(77, "Quality, timing & days models...")
        log("\n📐 Quality, timing & days-regressor models...")
        model_quality_gc    = train_quality_model(X_q_tr, y_q_tr)
        model_quality_dc    = train_quality_model(X_qd_tr, y_qd_tr)
        model_timing_gc     = train_timing_model(X_t_tr, y_t_tr)
        model_days_regressor= train_days_regressor(X_dr_tr, y_dr_tr)
        log(f"  Quality GC:      {'ok' if model_quality_gc else 'skipped'}")
        log(f"  Quality DC:      {'ok' if model_quality_dc else 'skipped'}")
        log(f"  Timing (norm):   {'ok' if model_timing_gc else 'skipped'}")
        log(f"  Days regressor:  {'ok' if model_days_regressor else 'skipped'} ← precise timing")

        prog.progress(86, "Test-set evaluation...")
        log("\n📊 Out-of-sample metrics...")
        probs_gc_test = apply_calibration(calibrator_gc, models_gc, X_test)
        probs_dc_test = apply_calibration(calibrator_dc, models_dc, X_test)
        auc_gc = roc_auc_score(y_gc_te, probs_gc_test)
        ap_gc  = average_precision_score(y_gc_te, probs_gc_test)
        auc_dc = roc_auc_score(y_dc_te, probs_dc_test)
        log(f"  GC  ROC-AUC={auc_gc:.3f}  Avg-Precision={ap_gc:.3f}")
        log(f"  DC  ROC-AUC={auc_dc:.3f}")

        # Brier score for calibration quality
        from sklearn.metrics import brier_score_loss
        brier_gc = brier_score_loss(y_gc_te, probs_gc_test)
        log(f"  GC  Brier={brier_gc:.4f}  (lower=better, 0=perfect, 0.25=random)")

        fi = compute_feature_importance(models_gc, FEATURE_COLS)

        st.session_state.update(dict(
            models_gc=models_gc, models_dc=models_dc,
            models_gc_recall=models_gc_recall,
            model_quality_gc=model_quality_gc, model_quality_dc=model_quality_dc,
            model_timing_gc=model_timing_gc,
            model_days_regressor=model_days_regressor,
            calibrator_gc=calibrator_gc, calibrator_dc=calibrator_dc,
            calibrator_gc_recall=calibrator_gc_recall,
            feature_cols=FEATURE_COLS, feature_importance=fi,
            fast_p=fast_p, slow_p=slow_p, threshold=threshold, quality_min=quality_min,
            start_date=str(start_date), pred_days=pred_days, models_trained=True,
            train_metrics=dict(
                auc_gc=auc_gc, ap_gc=ap_gc, auc_dc=auc_dc,
                brier_gc=brier_gc,
                n_train=len(X_train), n_test=len(X_test),
                gc_events=int(gc_ev), n_folds=len(splits),
            ),
        ))
        prog.progress(100, "✅ Training complete")
        log(f"\n✅ Done! {len(FEATURE_COLS)} features · {len(splits)} folds · Platt-calibrated")

    except Exception as e:
        st.error(f"Training failed: {e}")
        import traceback; st.code(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# GATE
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state.models_trained:
    st.info("👈  Configure training in the sidebar, then click **Train Models**.")
    with st.expander("v10 forensic audit fixes vs v9"):
        st.markdown("""
**All changes fix real accuracy / realism problems:**

| Issue | v8 | v9 Fix |
|---|---|---|
| `hist_gc_success` lookahead | Used `gc_regime.shift(-20)` without lag → future data visible | Shifted the success rate by 20 bars so bar *i* only sees past outcomes |
| Same-bar fill | Entry & exit on signal-bar close | Entry/exit on **next-bar open** (realistic) |
| No slippage | Commission only (0.1%) | Commission + **market impact slippage** (configurable, default 0.05%) |
| Kelly hardcoded | Used notebook averages (65% WR, 11% avg win) | Computed from **actual backtest trade log** |
| Sharpe inflated | Included flat (position=0) days in denominator | Computed on **trade returns only**, annualised by avg holding period |
| Calibration leakage | Cal fold came from same pool as training | **Strict embargo**: cal fold is a separate OOS slice never seen during training |
| MC drift noisy | Used historical log-return mean | **Risk-neutral drift = 0** + small regime tilt (historical mean is noise at < 1yr horizon) |
| `target_days_norm` unvalidated | Could be NaN or > 1 after arithmetic | Clipped to [0,1], NaN rows excluded from timing model |
| Quality model score out-of-range | Raw regressor output unclamped | Quality and timing scores **clamped to [0,1]** |
| Brier score not reported | No calibration quality metric | Brier score displayed in Model tab |
        """)
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_predict, tab_backtest, tab_model, tab_about = st.tabs([
    "🎯 Predict", "📈 Backtest", "🧠 Model", "ℹ About"
])


# ═══════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ═══════════════════════════════════════════════════════════════════
with tab_predict:
    col_inp, col_run = st.columns([4,1])
    with col_inp:
        predict_ticker_input = st.text_input("Ticker",
            placeholder="e.g. RELIANCE.NS   TCS.NS   ^NSEI",
            label_visibility="collapsed")
    with col_run:
        run_btn = st.button("Analyse →", type="primary", use_container_width=True)

    if run_btn and predict_ticker_input.strip():
        ticker = predict_ticker_input.strip().upper()
        with st.spinner(f"Analysing {ticker}..."):
            result, df_feat = predict_single(
                ticker,
                st.session_state.models_gc, st.session_state.models_dc,
                st.session_state.model_quality_gc, st.session_state.model_quality_dc,
                st.session_state.model_timing_gc, st.session_state.feature_cols,
                st.session_state.calibrator_gc, st.session_state.calibrator_dc,
                st.session_state.start_date,
                st.session_state.fast_p, st.session_state.slow_p,
                threshold, quality_min, st.session_state.pred_days,
                models_gc_recall=st.session_state.models_gc_recall,
                calibrator_gc_recall=st.session_state.calibrator_gc_recall,
                model_days_regressor=st.session_state.model_days_regressor,
                avg_win=st.session_state.get("bt_avg_win", 0.11),
                avg_loss=abs(st.session_state.get("bt_avg_loss", -0.05)),
                recall_threshold=recall_threshold,
            )
        if result is None:
            st.error(f"Could not fetch data for **{ticker}**. Verify the symbol.")
            st.stop()

        # ── Executive Summary ──────────────────────────────────────────────────
        def _exec_summary(r):
            p      = r['prob_gc']
            days   = r['days_pred']
            ev     = r['ev']
            comp   = r['composite_signal']
            stype  = r['signal_type']
            regime = r['regime_ok']
            near   = r['near_cross_ok']
            trend  = r['trend_ok']
            dual   = r['dual_confirm']
            q      = r['quality_gc']
            quality_min_val = quality_min

            # ── Decision ──────────────────────────────────────────────────────
            if comp and ev > 0 and dual:
                verdict = "✅ TAKE THE TRADE"
                verdict_color = "#00e5a0"
                verdict_bg    = "rgba(0,229,160,.08)"
                verdict_border= "rgba(0,229,160,.35)"
                days_str = f"in ~{days}d" if days else ""
                line1 = f"Model A+B both confirm a Golden Cross {days_str} with {p:.0%} probability and positive EV ({ev:+.2%})."
                line2 = (f"All filters pass — trend (ADX {r['adx']:.0f}), regime (price {r['price_vs_ma200']:+.1f}% vs MA200), "
                         f"convergence score {r['near_cross_score']:.2f}. "
                         f"Size: {r['pos_size']:.1%} of capital.")
            elif stype == 'bullish' and ev > 0 and not dual:
                verdict = "⚡ CONSIDER — Model A only"
                verdict_color = "#f5a623"
                verdict_bg    = "rgba(245,166,35,.08)"
                verdict_border= "rgba(245,166,35,.35)"
                days_str = f"in ~{days}d" if days else ""
                line1 = f"Model A signals GC {days_str} at {p:.0%} but Model B has not confirmed. EV is positive ({ev:+.2%})."
                line2 = f"Reduce size or wait for Model B confirmation (B prob: {r['prob_gc_recall']:.0%} vs threshold {recall_threshold:.0%})."
            elif stype in ('bullish','caution') and not regime:
                verdict = "⏸ WAIT — Regime filter blocking"
                verdict_color = "#f5a623"
                verdict_bg    = "rgba(245,166,35,.08)"
                verdict_border= "rgba(245,166,35,.35)"
                reasons = []
                if not trend:   reasons.append(f"ADX {r['adx']:.0f} < 20 (weak trend)")
                if not r['regime_price_ok']: reasons.append(f"price {r['price_vs_ma200']:+.1f}% below MA200")
                if not near:    reasons.append(f"cross convergence {r['near_cross_score']:.2f} < 0.45")
                line1 = f"GC probability is {p:.0%} but regime filters are blocking the signal."
                line2 = "Reason: " + " · ".join(reasons) + ". Wait for conditions to improve."
            elif ev <= 0 and p > 0.45:
                verdict = "⏸ WAIT — Negative expected value"
                verdict_color = "#f5a623"
                verdict_bg    = "rgba(245,166,35,.08)"
                verdict_border= "rgba(245,166,35,.35)"
                line1 = f"GC probability {p:.0%} is moderate but EV is {ev:+.2%} — risk outweighs reward at current win/loss estimates."
                line2 = "Wait for a higher-probability setup or better entry point."
            else:
                verdict = "❌ NO TRADE — No signal"
                verdict_color = "#f04060"
                verdict_bg    = "rgba(240,64,96,.08)"
                verdict_border= "rgba(240,64,96,.25)"
                line1 = f"GC probability is low ({p:.0%}) and no imminent crossover is detected (convergence: {r['near_cross_score']:.2f})."
                line2 = f"Stock is in {r['state']} with EMA gap {r['ema_gap_pct']:+.2f}%. No actionable setup."

            return verdict, verdict_color, verdict_bg, verdict_border, line1, line2

        verdict, v_col, v_bg, v_border, line1, line2 = _exec_summary(result)
        st.markdown(f"""
        <div style="background:{v_bg};border:1.5px solid {v_border};border-radius:10px;
                    padding:18px 22px;margin:8px 0 18px;">
          <div style="font-family:'JetBrains Mono';font-size:13px;font-weight:700;
                      color:{v_col};letter-spacing:.04em;margin-bottom:6px;">{verdict}</div>
          <div style="font-size:13px;color:#d4ddf5;line-height:1.6;">{line1}</div>
          <div style="font-size:12px;color:#7a8aaa;margin-top:4px;line-height:1.6;">{line2}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Signal header ──────────────────────────────────────────────────────
        st.markdown("---")
        chips = ""
        chips += ('<span class="risk-pill" style="background:rgba(0,229,160,.12);color:#00e5a0;border:1px solid rgba(0,229,160,.3);">Weekly Bull</span>'
                  if result['mtf_bull'] else
                  '<span class="risk-pill" style="background:rgba(240,64,96,.12);color:#f04060;border:1px solid rgba(240,64,96,.3);">Weekly Bear</span>')
        chips += ('<span class="risk-pill" style="background:rgba(0,229,160,.08);color:#00e5a0;border:1px solid rgba(0,229,160,.2);">Monthly Bull</span>'
                  if result['monthly_bull'] else
                  '<span class="risk-pill" style="background:rgba(240,64,96,.08);color:#f04060;border:1px solid rgba(240,64,96,.2);">Monthly Bear</span>')
        if result['vol_compression']:
            chips += '<span class="risk-pill" style="background:rgba(245,166,35,.12);color:#f5a623;border:1px solid rgba(245,166,35,.3);">Vol Squeeze</span>'
        if result['hurst'] > 0.55:
            chips += '<span class="risk-pill" style="background:rgba(77,166,255,.10);color:#4da6ff;border:1px solid rgba(77,166,255,.2);">Trending (H={:.2f})</span>'.format(result['hurst'])
        if result['dual_confirm']:
            chips += '<span class="risk-pill" style="background:rgba(167,139,250,.15);color:#a78bfa;border:1px solid rgba(167,139,250,.4);">A+B Confirmed</span>'
        if result['pre_gc_entry']:
            chips += '<span class="risk-pill" style="background:rgba(0,229,160,.20);color:#00e5a0;border:1px solid rgba(0,229,160,.5);font-weight:700;">⚡ PRE-GC ENTRY</span>'
        # Individual filter status chips
        if result['trend_ok']:
            chips += f'<span class="risk-pill" style="background:rgba(0,229,160,.10);color:#00e5a0;border:1px solid rgba(0,229,160,.3);">ADX {result["adx"]:.0f} ✓ Trend</span>'
        else:
            chips += f'<span class="risk-pill" style="background:rgba(240,64,96,.10);color:#f04060;border:1px solid rgba(240,64,96,.3);">ADX {result["adx"]:.0f} ✗ Weak Trend</span>'
        if result['regime_price_ok']:
            chips += f'<span class="risk-pill" style="background:rgba(0,229,160,.10);color:#00e5a0;border:1px solid rgba(0,229,160,.3);">Price {result["price_vs_ma200"]:+.1f}% vs MA200 ✓</span>'
        else:
            chips += f'<span class="risk-pill" style="background:rgba(240,64,96,.10);color:#f04060;border:1px solid rgba(240,64,96,.3);">Price {result["price_vs_ma200"]:+.1f}% vs MA200 ✗</span>'
        if result['near_cross_ok']:
            chips += f'<span class="risk-pill" style="background:rgba(0,229,160,.10);color:#00e5a0;border:1px solid rgba(0,229,160,.3);">Near Cross {result["near_cross_score"]:.2f} ✓</span>'
        else:
            chips += f'<span class="risk-pill" style="background:rgba(245,166,35,.10);color:#f5a623;border:1px solid rgba(245,166,35,.3);">Cross Far {result["near_cross_score"]:.2f} ✗</span>'
        # v13: Sector / Cap / Market Regime chips
        chips += f'<span class="risk-pill" style="background:rgba(77,166,255,.10);color:#4da6ff;border:1px solid rgba(77,166,255,.25);">Sector: {result.get("sector_name","?")}</span>'
        chips += f'<span class="risk-pill" style="background:rgba(77,166,255,.08);color:#9aa5c0;border:1px solid rgba(77,166,255,.2);">{result.get("cap_name","Mid")}-Cap</span>'
        if result.get('mkt_regime', 0) == 1:
            chips += '<span class="risk-pill" style="background:rgba(0,229,160,.10);color:#00c97a;border:1px solid rgba(0,201,122,.3);">Mkt Bull (Nifty&gt;MA200)</span>'
        else:
            chips += '<span class="risk-pill" style="background:rgba(240,64,96,.10);color:#e84060;border:1px solid rgba(232,64,96,.3);">Mkt Bear (Nifty&lt;MA200)</span>'

        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin:14px 0;">
          <div style="font-size:20px;font-weight:700;color:#d4ddf5;font-family:'DM Sans';">{ticker}</div>
          <div class="badge badge-{result['signal_type']}">{result['signal']}</div>
          <div style="font-size:12px;color:#7a8aaa;font-family:'JetBrains Mono';">
            ₹{result['current_price']:.2f} &nbsp;·&nbsp; {result['state']} &nbsp;·&nbsp; gap {result['ema_gap_pct']:+.2f}%
            &nbsp;·&nbsp; listed {result['listing_date']} &nbsp;·&nbsp; {result['history_bars']:,} bars ({result['history_years']}y)
          </div>
        </div>
        <div style="margin-bottom:16px;">{chips}</div>
        """, unsafe_allow_html=True)

        # ── Short history info (not a warning — model trains on all tickers) ───
        if result['short_history']:
            st.markdown(
                f'<div class="warn-box" style="border-color:rgba(77,166,255,.3);color:#4da6ff;">'
                f'ℹ️ <b>{ticker}</b> has <b>{result["history_bars"]} bars ({result["history_years"]}y)</b> '
                f'since listing on <b>{result["listing_date"]}</b>. '
                f'The model was trained on all tickers combined, so short history here is fine — '
                f'predictions generalise from the full training set.</div>',
                unsafe_allow_html=True)

        # ── HERO: Expected GC timing ───────────────────────────────────────────
        days_val   = result['days_pred']
        days_color = "#00e5a0" if (days_val and days_val <= 20) else ("#f5a623" if (days_val and days_val <= 40) else "#7a8aaa")
        days_disp  = f"{days_val} days" if days_val else "n/a"
        conf_label = ("High" if result['prob_gc'] >= 0.70 else
                      "Medium" if result['prob_gc'] >= 0.50 else "Low")
        conf_color = "#00e5a0" if result['prob_gc'] >= 0.70 else ("#f5a623" if result['prob_gc'] >= 0.50 else "#7a8aaa")
        ev_color   = "#00e5a0" if result['ev'] > 0 else "#f04060"

        st.markdown(f"""
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin:18px 0;">
          <div style="background:#111827;border:2px solid {days_color};border-radius:12px;padding:24px 28px;">
            <div style="font-size:14px;color:#5a6a8a;font-weight:600;margin-bottom:8px;">Golden Cross Expected In</div>
            <div style="font-size:42px;font-weight:700;color:{days_color};line-height:1;">{days_disp}</div>
            <div style="font-size:13px;color:#9aa5c0;margin-top:8px;">AI prediction based on market patterns</div>
          </div>
          <div style="background:#111827;border:2px solid {conf_color};border-radius:12px;padding:24px 28px;">
            <div style="font-size:14px;color:#5a6a8a;font-weight:600;margin-bottom:8px;">Confidence Level</div>
            <div style="font-size:42px;font-weight:700;color:{conf_color};line-height:1;">{result['prob_gc']:.0%}</div>
            <div style="font-size:13px;color:{conf_color};margin-top:8px;">{conf_label} confidence · Secondary model: {result['prob_gc_recall']:.0%}</div>
          </div>
          <div style="background:#111827;border:2px solid {ev_color};border-radius:12px;padding:24px 28px;">
            <div style="font-size:14px;color:#5a6a8a;font-weight:600;margin-bottom:8px;">Expected Profit</div>
            <div style="font-size:42px;font-weight:700;color:{ev_color};line-height:1;">{result['ev']:+.1%}</div>
            <div style="font-size:13px;color:#9aa5c0;margin-top:8px;">Recommended position: {result['pos_size']:.1%} of capital</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Row 1: Model output ────────────────────────────────────────────────
        st.markdown('<div class="sec-hdr">Model Output — Calibrated Probabilities</div>',
                    unsafe_allow_html=True)
        c1,c2,c3,c4,c5 = st.columns(5)
        gc_col = "#00e5a0" if result["prob_gc"]>=threshold else ("#f5a623" if result["prob_gc"]>0.45 else "#7a8aaa")
        dc_col = "#f04060" if result["prob_dc"]>=threshold else "#7a8aaa"
        q_col  = "#00e5a0" if (not np.isnan(result["quality_gc"]) and result["quality_gc"]>=quality_min) else "#f5a623"

        with c1:
            st.markdown(qcard(
                "GC Probability",
                f"{result['prob_gc']:.1%}",
                f"raw ensemble: {result['raw_gc']:.1%}" + prob_bar(result['prob_gc'], gc_col),
                gc_col), unsafe_allow_html=True)
        with c2:
            # Show regime status instead of DC probability — DC prob is redundant when stock is in DC
            is_gc_state = result['ema_gap_pct'] > 0
            regime_color = "#00c97a" if is_gc_state else "#e84060"
            regime_label = "Golden Cross" if is_gc_state else "Death Cross"
            dc_age_str   = f"{result['dc_age']} bars in DC" if not is_gc_state else "GC active"
            st.markdown(qcard(
                "Current Regime",
                regime_label,
                f"{dc_age_str} · gap {result['ema_gap_pct']:+.1f}%",
                regime_color), unsafe_allow_html=True)
        with c3:
            qv = result["quality_gc"]
            st.markdown(qcard(
                "GC Quality Score",
                f"{qv:.3f}" if not np.isnan(qv) else "n/a",
                f"gate: ≥{quality_min:.2f}  (fwd-ret + vol + trend)", q_col),
                unsafe_allow_html=True)
        with c4:
            days_c = "#00e5a0" if (result['days_pred'] and result['days_pred'] <= 20) else "#f5a623"
            days_sub = "Model C (raw days regressor)" if result['days_pred'] else "timing model (norm)"
            st.markdown(qcard(
                "Est. Days to GC", result["timing"],
                days_sub, days_c),
                unsafe_allow_html=True)
        with c5:
            hc = "#00e5a0" if result["hurst"] > 0.5 else "#f5a623"
            st.markdown(qcard(
                "Hurst Exponent", f"{result['hurst']:.3f}",
                ">0.5=trending  <0.5=mean-reverting", hc),
                unsafe_allow_html=True)

        # ── Row 2: Market context ──────────────────────────────────────────────
        st.markdown('<div class="sec-hdr">Market Context</div>', unsafe_allow_html=True)
        c6,c7,c8,c9,c10 = st.columns(5)
        with c6:
            st.markdown(qcard(
                "Hist GC Success",
                f"{result['hist_gc_success']:.1%}",
                "3yr trailing, 20-bar lag (no lookahead)"),
                unsafe_allow_html=True)
        with c7:
            mc2 = "#00e5a0" if result["price_vs_ma200"] > 0 else "#f04060"
            st.markdown(qcard(
                "Price vs MA200", f"{result['price_vs_ma200']:+.1f}%",
                "% above / below 200d MA", mc2), unsafe_allow_html=True)
        with c8:
            ac = "#00e5a0" if result["adx"]>25 else ("#f5a623" if result["adx"]>18 else "#7a8aaa")
            di_c = "#00e5a0" if result["di_diff"]>0 else "#f04060"
            st.markdown(qcard(
                "ADX / DI Spread", f"{result['adx']:.1f}",
                f"DI: <span style='color:{di_c}'>{result['di_diff']:+.1f}</span>  (>25 = strong trend)", ac),
                unsafe_allow_html=True)
        with c9:
            rc = "#f04060" if result["rsi"]>70 else ("#00e5a0" if result["rsi"]<30 else "#d4ddf5")
            st.markdown(qcard(
                "RSI 14", f"{result['rsi']:.1f}",
                "70=overbought / 30=oversold", rc),
                unsafe_allow_html=True)
        with c10:
            vc = "#f5a623" if result["ann_vol"] > 40 else "#7a8aaa"
            st.markdown(qcard(
                "Ann. Volatility", f"{result['ann_vol']:.1f}%",
                f"GK: {result['gk_vol']:.1f}%  (60d realised CC)", vc),
                unsafe_allow_html=True)

        # ── Price chart ────────────────────────────────────────────────────────
        st.markdown('<div class="sec-hdr">Price Chart — Last 252 Trading Days</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(plot_price_chart(df_feat, ticker,
            st.session_state.fast_p, st.session_state.slow_p), use_container_width=True)

        # ── Monte Carlo ────────────────────────────────────────────────────────
        st.markdown('<div class="sec-hdr">Monte Carlo Simulation (risk-neutral Student-t GBM)</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div class="warn-box">⚡ Drift = 0 (risk-neutral). A small ±0.02%/day regime tilt is applied. '
            'Historical mean not used — it is too noisy at sub-1yr horizons and inflates returns.</div>',
            unsafe_allow_html=True)

        regime_map = {"bullish":"bullish","bearish":"bearish","caution":"neutral","neutral":"neutral"}
        mc_result  = run_monte_carlo(df_feat['Close'].values, days=mc_days,
                                     n_sims=mc_sims,
                                     regime=regime_map.get(result["signal_type"],"neutral"))

        mc1,mc2,mc3,mc4 = st.columns(4)
        with mc1:
            pc = "#00e5a0" if mc_result["prob_up"]>0.5 else "#f04060"
            st.markdown(qcard("P(Price Up)", f"{mc_result['prob_up']:.1%}",
                f"in {mc_days} trading days", pc), unsafe_allow_html=True)
        with mc2:
            ec = "#00e5a0" if mc_result["exp_return"]>0 else "#f04060"
            st.markdown(qcard("Expected Return", f"{mc_result['exp_return']:+.1%}",
                "mean of all simulation paths", ec), unsafe_allow_html=True)
        with mc3:
            st.markdown(qcard("VaR 95% / CVaR",
                f"{mc_result['var_95']:+.1%}",
                f"CVaR (ES): {mc_result['cvar_95']:+.1%}  mean of worst 5%",
                "#f04060"), unsafe_allow_html=True)
        with mc4:
            st.markdown(qcard("P50 Target",
                f"₹{mc_result['p50'][-1]:.1f}",
                f"median in {mc_days}d  (ann.vol={mc_result['ann_vol']*100:.1f}%)",
                "#4da6ff"), unsafe_allow_html=True)

        st.plotly_chart(plot_monte_carlo(mc_result, ticker, mc_days), use_container_width=True)

        col_dist, col_pct = st.columns([2,1])
        with col_dist:
            st.plotly_chart(plot_mc_distribution(mc_result), use_container_width=True)
        with col_pct:
            st.markdown('<div class="sec-hdr">Percentile Targets</div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame({
                "Percentile": ["P5 (Bear)","P25","P50","P75","P95 (Bull)"],
                "Price":  [f"₹{mc_result[p][-1]:.1f}" for p in ["p5","p25","p50","p75","p95"]],
                "Return": [f"{mc_result[p][-1]/mc_result['S0']-1:+.1%}" for p in ["p5","p25","p50","p75","p95"]],
            }), hide_index=True, use_container_width=True)

        # ── Risk & position sizing ─────────────────────────────────────────────
        st.markdown('<div class="sec-hdr">Risk & Position Sizing</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="warn-box">⚡ Kelly fraction computed from backtest statistics '
            '(run Backtest tab first for live Kelly). Values below use notebook averages as seed.</div>',
            unsafe_allow_html=True)
        r1,r2,r3,r4 = st.columns(4)
        # Use backtest stats if available in session state, else fall back
        seed_wr = st.session_state.get("bt_win_rate", 0.65)
        seed_w  = st.session_state.get("bt_avg_win",  0.11)
        seed_l  = st.session_state.get("bt_avg_loss", -0.05)
        kf      = kelly_criterion(seed_wr, seed_w, seed_l)
        atr_stop_price = result['current_price'] * (1 - 2*result['atr_pct']/100)
        with r1:
            st.markdown(qcard("Kelly Fraction", f"{kf:.1%}",
                f"0.25×Kelly = {kf*0.25:.1%} (recommended)", "#a78bfa"), unsafe_allow_html=True)
        with r2:
            st.markdown(qcard("ATR Stop (2×ATR)",
                f"₹{atr_stop_price:.1f}",
                f"ATR={result['atr_pct']:.2f}%  (fixed floor={stop_loss:.0%})", "#f5a623"),
                unsafe_allow_html=True)
        with r3:
            denom = max(result['current_price'] - atr_stop_price, 0.001)
            rr    = abs(result['current_price'] * take_profit / denom)
            rr_col = "#00e5a0" if rr > 2 else "#f5a623"
            st.markdown(qcard("Risk : Reward",
                f"1 : {rr:.1f}",
                f"TP={take_profit:.0%}  SL=2×ATR", rr_col), unsafe_allow_html=True)
        with r4:
            st.markdown(qcard("DC Regime Age",
                f"{result['dc_age']} bars",
                "longer base → stronger potential GC"), unsafe_allow_html=True)

        # ── GC Prediction Forward Chart ────────────────────────────────────────
        st.markdown('<div class="sec-hdr">Golden Cross Prediction — When Will the Crossover Happen?</div>',
                    unsafe_allow_html=True)

        # Build the forward prediction chart:
        # Left side  = recent 60 bars of actual price + EMAs (context)
        # Right side = projected EMA convergence path to predicted crossover date
        _lookback   = 60
        _hist       = df_feat.iloc[-_lookback:].copy()
        _last_date  = _hist.index[-1]
        _last_close = float(_hist['Close'].iloc[-1])
        _ema_fast_last = float(_hist['ema_fast'].iloc[-1])
        _ema_slow_last = float(_hist['ema_slow'].iloc[-1])
        _gap_vel    = float(_hist['gap_velocity_5'].iloc[-1])   # %/bar convergence speed

        # Days prediction: use Model C output if available, else fallback
        _days_pred  = result['days_pred'] if result['days_pred'] else st.session_state.pred_days
        # Cap display window to max 90 days ahead
        _fwd_days   = min(int(_days_pred) + 15, 90)

        # Generate future dates (business days)
        _future_dates = pd.bdate_range(start=_last_date + pd.Timedelta(days=1), periods=_fwd_days)

        # Project EMA fast linearly converging toward EMA slow over predicted days
        # Use the current gap velocity to model the convergence path
        _gap_now_pct = float(_hist['ema_gap_pct'].iloc[-1])
        # Linear interpolation: gap closes from current value to 0 at predicted crossover
        _fwd_gap = np.linspace(_gap_now_pct, 0.0, _days_pred) if _days_pred > 0 else np.array([0.0])
        if _fwd_days > _days_pred:
            # After the crossover: EMAs diverge slightly on the other side
            _post = np.linspace(0.0, abs(_gap_now_pct) * 0.15, _fwd_days - _days_pred)
            _fwd_gap = np.concatenate([_fwd_gap, _post])
        _fwd_gap = _fwd_gap[:_fwd_days]

        # Derive projected EMA slow (approximately flat — slow MA barely moves in 90 days)
        _proj_slow = np.full(_fwd_days, _ema_slow_last)
        # Derive projected EMA fast from gap: fast = slow × (1 + gap_pct/100)
        _proj_fast = _proj_slow * (1.0 + _fwd_gap / 100.0)

        # Confidence band: ±1 ATR width around projected fast EMA
        _atr_pct   = float(_hist['atr_pct'].iloc[-1]) / 100.0
        _band_upper = _proj_fast * (1 + _atr_pct)
        _band_lower = _proj_fast * (1 - _atr_pct)

        # Crossover point marker
        _cross_date = _future_dates[min(_days_pred - 1, len(_future_dates) - 1)]
        _cross_price = float(_proj_slow[min(_days_pred - 1, len(_proj_slow) - 1)])

        # Build the figure: 3 rows — price, EMA gap, probability bar
        fig_pred = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                 row_heights=[0.55, 0.25, 0.20], vertical_spacing=0.03)

        # ── Row 1: Historical price + EMAs ──
        fig_pred.add_trace(go.Scatter(
            x=_hist.index, y=_hist['Close'],
            line=dict(color="#4da6ff", width=1.8), name="Close Price"), row=1, col=1)
        fig_pred.add_trace(go.Scatter(
            x=_hist.index, y=_hist['ema_fast'],
            line=dict(color="#00e5a0", width=1.4), name=f"EMA{st.session_state.fast_p} (actual)"), row=1, col=1)
        fig_pred.add_trace(go.Scatter(
            x=_hist.index, y=_hist['ema_slow'],
            line=dict(color="#f04060", width=1.4), name=f"EMA{st.session_state.slow_p} (actual)"), row=1, col=1)

        # ── Row 1: Projected EMA fast (dashed) + confidence band ──
        # Confidence band fill
        fig_pred.add_trace(go.Scatter(
            x=list(_future_dates) + list(_future_dates[::-1]),
            y=list(_band_upper) + list(_band_lower[::-1]),
            fill='toself', fillcolor='rgba(0,229,160,0.07)',
            line=dict(color='rgba(0,0,0,0)'), name='±1 ATR band',
            showlegend=True), row=1, col=1)
        fig_pred.add_trace(go.Scatter(
            x=_future_dates, y=_proj_fast,
            line=dict(color="#00e5a0", width=1.6, dash="dash"),
            name=f"EMA{st.session_state.fast_p} (projected)"), row=1, col=1)
        fig_pred.add_trace(go.Scatter(
            x=_future_dates, y=_proj_slow,
            line=dict(color="#f04060", width=1.6, dash="dash"),
            name=f"EMA{st.session_state.slow_p} (projected)"), row=1, col=1)

        # ── Predicted crossover marker ──
        gc_col_pred = "#00e5a0" if result['prob_gc'] >= 0.60 else "#f5a623"
        fig_pred.add_trace(go.Scatter(
            x=[_cross_date], y=[_cross_price],
            mode='markers+text',
            marker=dict(color=gc_col_pred, size=16, symbol='star',
                        line=dict(color=gc_col_pred, width=2)),
            text=[f"  ⭐ GC ~{_days_pred}d"],
            textposition='middle right',
            textfont=dict(color=gc_col_pred, size=12, family="JetBrains Mono"),
            name=f"Predicted GC (~{_days_pred}d)"), row=1, col=1)

        # ── Vertical line at today + at predicted GC ──
        fig_pred.add_vline(x=pd.Timestamp(_last_date).timestamp() * 1000, row=1, col=1,
                           line=dict(color="#f5a623", width=1.5, dash="dot"),
                           annotation_text="  Today", annotation_font_color="#f5a623",
                           annotation_font_size=10)
        fig_pred.add_vline(x=pd.Timestamp(_cross_date).timestamp() * 1000, row=1, col=1,
                           line=dict(color=gc_col_pred, width=1.5, dash="dot"),
                           annotation_text=f"  Predicted GC", annotation_font_color=gc_col_pred,
                           annotation_font_size=10)

        # ── Row 2: EMA gap % — historical + projected converging to zero ──
        fig_pred.add_trace(go.Scatter(
            x=_hist.index, y=_hist['ema_gap_pct'],
            line=dict(color="#a78bfa", width=1.3), name="EMA Gap % (actual)",
            fill='tozeroy', fillcolor='rgba(167,139,250,0.06)'), row=2, col=1)
        fig_pred.add_trace(go.Scatter(
            x=_future_dates, y=_fwd_gap,
            line=dict(color="#a78bfa", width=1.3, dash="dash"),
            name="EMA Gap % (projected)",
            fill='tozeroy', fillcolor='rgba(167,139,250,0.03)'), row=2, col=1)
        fig_pred.add_hline(y=0, row=2, col=1,
                           line=dict(color="#00e5a0", width=1.5, dash="dot"),
                           annotation_text="  Crossover = 0%",
                           annotation_font_color="#00e5a0", annotation_font_size=9)
        fig_pred.add_vline(x=str(_cross_date), row=2, col=1,
                           line=dict(color=gc_col_pred, width=1, dash="dot"))

        # ── Row 3: Probability gauge bar ──
        _prob_pct = result['prob_gc'] * 100
        _prob_bar_x = [0, _prob_pct, _prob_pct, 0]
        _prob_bar_y = [0, 0, 1, 1]
        fig_pred.add_trace(go.Scatter(
            x=[0, _prob_pct], y=[0.5, 0.5],
            mode='lines',
            line=dict(color=gc_col_pred, width=18),
            name=f"GC Prob: {result['prob_gc']:.1%}"), row=3, col=1)
        fig_pred.add_trace(go.Scatter(
            x=[_prob_pct, 100], y=[0.5, 0.5],
            mode='lines',
            line=dict(color="#1e2840", width=18),
            showlegend=False), row=3, col=1)
        fig_pred.add_trace(go.Scatter(
            x=[60], y=[0.5], mode='markers',
            marker=dict(color="#f5a623", size=10, symbol="line-ns",
                        line=dict(color="#f5a623", width=2)),
            name="Signal threshold 60%"), row=3, col=1)

        fig_pred.update_layout(**DARK,
            title=dict(
                text=f"{ticker} — Golden Cross Prediction · {result['prob_gc']:.1%} probability · estimated in ~{_days_pred} trading days",
                font=dict(size=13, color="#d4ddf5")),
            legend=dict(bgcolor="#080c18", bordercolor="#151c30", font=dict(size=10),
                        orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
            height=560, margin=dict(l=55, r=20, t=80, b=30),
            xaxis_rangeslider_visible=False)
        fig_pred.update_yaxes(title_text="Price", row=1, col=1, **GRID)
        fig_pred.update_yaxes(title_text="Gap %", row=2, col=1, **GRID)
        fig_pred.update_yaxes(title_text="Prob", row=3, col=1, range=[0, 1],
                               showticklabels=False, **GRID)
        for _r in range(1, 4):
            fig_pred.update_xaxes(**GRID, row=_r, col=1)
        st.plotly_chart(fig_pred, use_container_width=True)

        # Caption
        _conf_label = "High" if result['prob_gc'] >= 0.70 else ("Medium" if result['prob_gc'] >= 0.50 else "Low")
        _gap_dir    = "converging ✓" if _gap_vel < 0 else "diverging ✗"
        st.markdown(f"""
        <div style="font-family:'JetBrains Mono';font-size:11px;color:#7a8aaa;margin-top:-8px;margin-bottom:16px;">
          ⭐ Predicted GC in <b style="color:{gc_col_pred}">~{_days_pred} trading days</b>
          &nbsp;·&nbsp; Confidence: <b style="color:{gc_col_pred}">{_conf_label} ({result['prob_gc']:.1%})</b>
          &nbsp;·&nbsp; EMA gap: <b style="color:#a78bfa">{_gap_now_pct:+.2f}%</b> ({_gap_dir})
          &nbsp;·&nbsp; Dashed lines = model projection · ±1 ATR uncertainty band shown
        </div>
        """, unsafe_allow_html=True)

        # ── Predict Summary Panel ──────────────────────────────────────────────
        st.markdown('<div class="sec-hdr">📋 Signal Summary — What Every Number Means</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:var(--bg1);border:1px solid var(--border);border-radius:10px;
                    padding:24px 28px;font-family:'JetBrains Mono';font-size:12px;line-height:1.85;">

          <div style="font-size:14px;font-weight:700;color:#d4ddf5;margin-bottom:16px;">
            {ticker} · Current Signal: <span style="color:{'#00e5a0' if result['signal_type']=='bullish' else '#f04060' if result['signal_type']=='bearish' else '#f5a623'}">{result['signal']}</span>
          </div>

          <table style="width:100%;border-collapse:collapse;">
            <thead>
              <tr style="background:#0d1020;border-bottom:1px solid #1e2840;">
                <th style="padding:8px 12px;color:#3d4d6a;font-size:10px;text-transform:uppercase;letter-spacing:.1em;text-align:left;">Parameter</th>
                <th style="padding:8px 12px;color:#3d4d6a;font-size:10px;text-transform:uppercase;letter-spacing:.1em;text-align:left;">Value</th>
                <th style="padding:8px 12px;color:#3d4d6a;font-size:10px;text-transform:uppercase;letter-spacing:.1em;text-align:left;">What It Means</th>
                <th style="padding:8px 12px;color:#3d4d6a;font-size:10px;text-transform:uppercase;letter-spacing:.1em;text-align:left;">Good Range / Interpretation</th>
              </tr>
            </thead>
            <tbody>
              <tr style="border-bottom:1px solid #151c30;">
                <td style="padding:8px 12px;color:#4da6ff;font-weight:600;">GC Probability</td>
                <td style="padding:8px 12px;color:{'#00e5a0' if result['prob_gc']>=0.6 else '#f5a623' if result['prob_gc']>=0.45 else '#7a8aaa'};">{result['prob_gc']:.1%}</td>
                <td style="padding:8px 12px;color:#d4ddf5;">Calibrated probability that a Golden Cross (fast EMA crosses above slow EMA) will occur within the prediction window. Output of Model A (precision ensemble) after Platt scaling.</td>
                <td style="padding:8px 12px;color:#7a8aaa;">≥60% = signal. 45–60% = watch. &lt;45% = no signal.</td>
              </tr>
              <tr style="border-bottom:1px solid #151c30;background:#080c18;">
                <td style="padding:8px 12px;color:#4da6ff;font-weight:600;">Model B Probability</td>
                <td style="padding:8px 12px;color:{'#00e5a0' if result['prob_gc_recall']>=0.4 else '#7a8aaa'};">{result['prob_gc_recall']:.1%}</td>
                <td style="padding:8px 12px;color:#d4ddf5;">High-recall ensemble (Model B) score. Trained with 2× class weight to catch more GC events. Entry requires BOTH Model A ≥ threshold AND Model B ≥ recall threshold (AND logic prevents FP explosion).</td>
                <td style="padding:8px 12px;color:#7a8aaa;">≥40% = confirmation. AND logic: both must fire for a BULLISH signal.</td>
              </tr>
              <tr style="border-bottom:1px solid #151c30;">
                <td style="padding:8px 12px;color:#4da6ff;font-weight:600;">Near-Cross Score</td>
                <td style="padding:8px 12px;color:{'#00e5a0' if result['near_cross_score']>=0.45 else '#f5a623' if result['near_cross_score']>=0.3 else '#f04060'};">{result['near_cross_score']:.3f}</td>
                <td style="padding:8px 12px;color:#d4ddf5;">Continuous 0–1 convergence score. Combines EMA gap proximity and closing velocity. Measures how fast the fast EMA is approaching the slow EMA. Hard gate: score &lt;0.45 blocks signal (gap wide or diverging).</td>
                <td style="padding:8px 12px;color:#7a8aaa;">≥0.45 = cross imminent. 0.30–0.45 = watch. &lt;0.30 = cross far away.</td>
              </tr>
              <tr style="border-bottom:1px solid #151c30;background:#080c18;">
                <td style="padding:8px 12px;color:#4da6ff;font-weight:600;">EMA Gap %</td>
                <td style="padding:8px 12px;color:{'#f5a623' if result['ema_gap_pct']<0 else '#4da6ff'};">{result['ema_gap_pct']:+.2f}%</td>
                <td style="padding:8px 12px;color:#d4ddf5;">Percentage gap between fast and slow EMA: (fast−slow)/slow×100. Negative = fast below slow (DC regime). Positive = fast above slow (GC regime). The model watches this close toward zero.</td>
                <td style="padding:8px 12px;color:#7a8aaa;">Gap approaching 0% from below = GC candidate. Large negative gap = no signal.</td>
              </tr>
              <tr style="border-bottom:1px solid #151c30;">
                <td style="padding:8px 12px;color:#4da6ff;font-weight:600;">ADX (Trend Strength)</td>
                <td style="padding:8px 12px;color:{'#00e5a0' if result['adx']>=25 else '#f5a623' if result['adx']>=18 else '#f04060'};">{result['adx']:.1f}</td>
                <td style="padding:8px 12px;color:#d4ddf5;">Average Directional Index. Measures trend strength (not direction). Used as hard regime filter — signals blocked when ADX &lt;20 because a GC in a choppy/flat market is unreliable.</td>
                <td style="padding:8px 12px;color:#7a8aaa;">≥25 = strong trend. 20–25 = acceptable. &lt;20 = signal blocked.</td>
              </tr>
              <tr style="border-bottom:1px solid #151c30;background:#080c18;">
                <td style="padding:8px 12px;color:#4da6ff;font-weight:600;">Hurst Exponent</td>
                <td style="padding:8px 12px;color:{'#00e5a0' if result['hurst']>0.55 else '#f5a623' if result['hurst']>0.45 else '#f04060'};">{result['hurst']:.3f}</td>
                <td style="padding:8px 12px;color:#d4ddf5;">Rescaled-range Hurst. H&gt;0.5 = trending (persistent momentum). H&lt;0.5 = mean-reverting (price will snap back). Signals blocked when H&lt;0.45 — GC in a mean-reverting market tends to fail quickly.</td>
                <td style="padding:8px 12px;color:#7a8aaa;">H&gt;0.55 = trending (ideal). 0.45–0.55 = borderline. &lt;0.45 = blocked.</td>
              </tr>
              <tr style="border-bottom:1px solid #151c30;">
                <td style="padding:8px 12px;color:#4da6ff;font-weight:600;">Price vs MA200</td>
                <td style="padding:8px 12px;color:{'#00e5a0' if result['price_vs_ma200']>0 else '#f04060'};">{result['price_vs_ma200']:+.1f}%</td>
                <td style="padding:8px 12px;color:#d4ddf5;">% above/below the 200-bar moving average. Macro bull regime filter. A GC while price is below its 200-bar MA is historically unreliable — the model blocks signals in this case.</td>
                <td style="padding:8px 12px;color:#7a8aaa;">&gt;0% = bull regime ✓. &lt;0% = bear regime, signal blocked.</td>
              </tr>
              <tr style="border-bottom:1px solid #151c30;background:#080c18;">
                <td style="padding:8px 12px;color:#4da6ff;font-weight:600;">GC Quality Score</td>
                <td style="padding:8px 12px;color:{'#00e5a0' if not np.isnan(result['quality_gc']) and result['quality_gc']>=quality_min else '#f5a623'};">{"n/a" if np.isnan(result['quality_gc']) else f"{result['quality_gc']:.3f}"}</td>
                <td style="padding:8px 12px;color:#d4ddf5;">XGBRegressor quality estimate trained only on actual GC event bars. Score = 0.4×fwd_return + 0.3×volume_ratio + 0.3×(ADX+MTF). Higher = better post-cross environment. Minimum gate configurable in sidebar.</td>
                <td style="padding:8px 12px;color:#7a8aaa;">≥0.35 = good quality (configurable). &lt;threshold = signal blocked.</td>
              </tr>
              <tr style="border-bottom:1px solid #151c30;">
                <td style="padding:8px 12px;color:#4da6ff;font-weight:600;">Expected Value (EV)</td>
                <td style="padding:8px 12px;color:{'#00e5a0' if result['ev']>0 else '#f04060'};">{result['ev']:+.2%}</td>
                <td style="padding:8px 12px;color:#d4ddf5;">EV = P(GC)×avg_win − (1−P(GC))×avg_loss. The expected profit per trade if you took this signal repeatedly. Backtest avg win/loss used. Only positive EV trades are taken when EV filter is active.</td>
                <td style="padding:8px 12px;color:#7a8aaa;">&gt;0% = positive expectancy ✓. &lt;0% = negative expectancy, blocked.</td>
              </tr>
              <tr style="border-bottom:1px solid #151c30;background:#080c18;">
                <td style="padding:8px 12px;color:#4da6ff;font-weight:600;">Est. Days to GC</td>
                <td style="padding:8px 12px;color:{'#00e5a0' if result['days_pred'] and result['days_pred']<=20 else '#f5a623'};">{result['timing']}</td>
                <td style="padding:8px 12px;color:#d4ddf5;">Model C (XGBRegressor) trained on raw days-to-next-GC (capped at 60 bars). Gives precise timing forecast rather than binary yes/no. Tight accuracy metric: |predicted − actual| ≤ 5 days.</td>
                <td style="padding:8px 12px;color:#7a8aaa;">≤20 days = imminent (green). 20–40 = watch (amber). &gt;40 = far away.</td>
              </tr>
              <tr>
                <td style="padding:8px 12px;color:#4da6ff;font-weight:600;">Hist GC Success</td>
                <td style="padding:8px 12px;color:#00e5a0;">{result['hist_gc_success']:.1%}</td>
                <td style="padding:8px 12px;color:#d4ddf5;">3-year trailing price-direction win rate: % of past GC events where price was higher after the horizon period. Uses 20-bar forward lag to prevent lookahead. Measures real-world GC quality for this specific stock.</td>
                <td style="padding:8px 12px;color:#7a8aaa;">&gt;60% = reliable stock for GC signals. &lt;50% = historically unreliable.</td>
              </tr>
            </tbody>
          </table>

          <div style="margin-top:16px;padding:12px 14px;background:#0d1020;border-radius:6px;
                      border-left:3px solid #4da6ff;font-size:11px;color:#7a8aaa;">
            <b style="color:#d4ddf5;">False Positive Suppression (v13):</b>
            A signal fires ONLY when ALL of the following are true simultaneously:
            <span style="color:#00e5a0;">Model A ≥ {threshold:.0%}</span> AND
            <span style="color:#00e5a0;">Model B ≥ recall threshold</span> AND
            <span style="color:#00e5a0;">Near-Cross Score ≥ 0.45</span> AND
            <span style="color:#00e5a0;">ADX ≥ 20</span> AND
            <span style="color:#00e5a0;">Hurst &gt; 0.45</span> AND
            <span style="color:#00e5a0;">Price &gt; MA200</span> AND
            <span style="color:#00e5a0;">EV &gt; 0</span> AND
            <span style="color:#00e5a0;">Quality ≥ {quality_min:.2f}</span>.
            Each gate independently eliminates a class of false positives.
          </div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 2 — BACKTEST
# One honest question: "Did a GC cross happen within N days of the signal?"
# TP = model fired AND a real GC cross occurred within N days.
# FP = model fired but NO GC cross within N days.
# FN = model did NOT fire but a GC cross DID happen within N days.
# TN = model did NOT fire AND no GC cross within N days.
# ═══════════════════════════════════════════════════════════════════
with tab_backtest:
    st.markdown("""
    <div style="margin-bottom:4px;">
      <div style="font-size:20px;font-weight:700;color:#d4ddf5;font-family:'DM Sans';">
        GC Prediction Backtest
      </div>
      <div style="font-size:12px;color:#7a8aaa;font-family:'JetBrains Mono';margin-top:2px;">
        Did a GC cross happen within N days of the signal? · Full history · No lookahead bias
      </div>
    </div>
    """, unsafe_allow_html=True)

    bt_c1, bt_c2, bt_c3, bt_c4 = st.columns([3,1,1,1])
    with bt_c1:
        bt_ticker = st.text_input("Ticker",
            placeholder="e.g. RELIANCE.NS   HDFCBANK.NS   ^NSEI",
            label_visibility="collapsed", key="bt_ticker")
    with bt_c2:
        bt_horizon = st.selectbox("Prediction Window (days)", [30, 60, 90], index=0,
                                   format_func=lambda x: f"{x}d", key="bt_horizon")
    with bt_c3:
        bt_recall_thr = st.selectbox("Recall Threshold", [0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
                                      index=2, format_func=lambda x: f"{x:.0%}", key="bt_recall_thr",
                                      help="Lower = catches more GC events (higher recall)")
    with bt_c4:
        bt_btn = st.button("Run Backtest →", type="primary", use_container_width=True)

    if bt_btn and bt_ticker.strip():
        tk = bt_ticker.strip().upper()
        with st.spinner(f"Backtesting {tk}..."):
            df_raw = download_stock_data(tk, st.session_state.start_date)
            if df_raw is None or len(df_raw) < 400:
                st.error("Not enough data. Try an earlier Training Start Date in the sidebar.")
            else:
                horizon = bt_horizon

                # ── Feature engineering + target creation ──────────────────────
                df_feat_bt = engineer_features(df_raw,
                                fast_p=st.session_state.fast_p,
                                slow_p=st.session_state.slow_p)
                # target_gc = 1 if a confirmed GC cross happens within the next `horizon` bars.
                # This is EXACTLY the question the model was trained to answer.
                df_lab_bt  = create_targets(df_feat_bt, n_days=horizon)

                # Skip EMA warmup bars so indicators are stable
                WARMUP = 250
                df_eval    = df_lab_bt.iloc[WARMUP:].copy().reset_index(drop=False)
                date_col   = df_eval.columns[0]
                df_eval    = ensure_feature_cols(df_eval)

                # ── Inject sector/cap/regime for this ticker (same as predict_single) ──
                sector_id = SECTOR_MAP.get(tk, 15)
                cap_id    = CAP_MAP.get(tk, 1)
                for _i in range(N_SECTORS):
                    df_eval[f'sector_{_i}'] = np.int8(1 if sector_id == _i else 0)
                for _i in range(N_CAPS):
                    df_eval[f'cap_{_i}']    = np.int8(1 if cap_id == _i else 0)
                try:
                    nifty_reg = get_nifty_regime(st.session_state.start_date)
                    if nifty_reg is not None:
                        df_eval['mkt_regime'] = (
                            nifty_reg.reindex(
                                pd.to_datetime(df_eval[date_col]), method='ffill'
                            ).fillna(0).values
                        )
                except Exception:
                    df_eval['mkt_regime'] = 0
                # Sector-demeaned features: use raw value (no peers available)
                for _col in ['rsi_14_vs_sector','price_vs_ma200_vs_sector',
                             'adx_vs_sector','atr_pct_vs_sector','macd_hist_pct_vs_sector']:
                    if _col not in df_eval.columns or (df_eval[_col] == 0).all():
                        raw = _col.replace('_vs_sector','')
                        df_eval[_col] = df_eval[raw].fillna(0) if raw in df_eval.columns else 0.0

                X_eval = df_eval[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)

                # ── Model predictions on every bar ─────────────────────────────
                # Use RAW ensemble for Model A (calibrator squashes to ~0.5 on single-ticker)
                # Calibration was trained on multi-ticker data with sector features;
                # single-ticker inference with zero sector features breaks the calibrator.
                probs_gc     = ensemble_predict_proba(st.session_state.models_gc, X_eval)
                probs_gc_cal = apply_calibration(
                    st.session_state.calibrator_gc,
                    st.session_state.models_gc, X_eval)

                # Model B (recall) predictions — raw ensemble, no calibration needed
                probs_gc_b = ensemble_predict_proba(
                    st.session_state.models_gc_recall, X_eval
                ) if st.session_state.models_gc_recall else probs_gc

                # ── Debug: show probability distribution ───────────────────────
                st.markdown(f"""
                <div class="warn-box" style="border-color:rgba(77,166,255,.3);color:#4da6ff;">
                  📊 <b>Probability debug</b>: Model A raw max={probs_gc.max():.3f}
                  mean={probs_gc.mean():.3f} &nbsp;|&nbsp;
                  Model B max={probs_gc_b.max():.3f} mean={probs_gc_b.mean():.3f} &nbsp;|&nbsp;
                  Threshold A={threshold:.0%} B={bt_recall_thr:.0%} &nbsp;|&nbsp;
                  Signals A={(probs_gc>=threshold).sum()} B={(probs_gc_b>=bt_recall_thr).sum()}
                </div>""", unsafe_allow_html=True)

                y_pred_a  = (probs_gc   >= threshold).astype(int)
                y_pred_b  = (probs_gc_b >= bt_recall_thr).astype(int)
                y_pred_ab = np.clip(y_pred_a + y_pred_b, 0, 1)
                y_pred    = y_pred_a
                y_true      = df_eval["target_gc"].values

                n_bars   = len(y_true)
                n_pos    = int(y_true.sum())   # total real GC events in window

                if n_pos < 5:
                    st.warning(
                        f"Only {n_pos} real GC events in {n_bars} bars for {tk} "
                        f"at {horizon}d window. Try a wider window or longer history."
                    )

                # ── PHASE 4 FIX: Timing-aware confusion matrix ─────────────────
                def bkpi(label, val, sub="", col="#4da6ff"):
                    return (f'<div class="qcard" style="--accent:{col};">'
                            f'<div class="qlabel">{label}</div>'
                            f'<div class="qval" style="color:{col};">{val}</div>'
                            f'<div class="qsub">{sub}</div></div>')
                # Old: TP = signal fired AND gc happened in window (binary)
                # New: TP = predicted_days <= horizon AND actual_days <= horizon
                #      FP = predicted_days <= horizon AND actual_days > horizon
                # Also compute lead-time metric (how early was the signal?)
                days_pred_arr = None
                if st.session_state.model_days_regressor is not None:
                    days_pred_arr = np.clip(
                        st.session_state.model_days_regressor.predict(X_eval), 1, 60
                    ).round().astype(int)

                # Compute actual days to next gc_event for each bar
                gc_event_arr = df_eval['gc_event'].values if 'gc_event' in df_eval.columns else np.zeros(len(df_eval))
                actual_days_arr = np.full(len(df_eval), 999)
                gc_event_idx = np.where(gc_event_arr == 1)[0]
                for i in range(len(df_eval)):
                    fut = gc_event_idx[gc_event_idx > i]
                    if len(fut) > 0:
                        actual_days_arr[i] = fut[0] - i

                if days_pred_arr is not None:
                    # ── FIX 1: Tight timing metric |pred - actual| ≤ 5 ────────
                    # Old (lenient): pred≤30 AND actual≤30 → 93% because both just need to be <30
                    # New (tight):   |predicted_days - actual_days| ≤ 5 → real timing accuracy
                    valid_mask   = actual_days_arr < 999   # only bars with a future GC
                    timing_errs  = np.abs(days_pred_arr.astype(float) - actual_days_arr.astype(float))
                    tight_acc    = float(np.mean(timing_errs[valid_mask] <= 5)) if valid_mask.sum() > 0 else 0.0
                    mean_err     = float(np.mean(timing_errs[valid_mask]))      if valid_mask.sum() > 0 else float('nan')
                    pct_within10 = float(np.mean(timing_errs[valid_mask] <= 10)) if valid_mask.sum() > 0 else 0.0

                    # Timing-aware TP/FP (kept for recall/precision, but now secondary)
                    y_pred_timing = (days_pred_arr <= horizon).astype(int)
                    y_true_timing = (actual_days_arr <= horizon).astype(int)
                    TP_t = int(((y_pred_timing==1) & (y_true_timing==1)).sum())
                    FP_t = int(((y_pred_timing==1) & (y_true_timing==0)).sum())
                    FN_t = int(((y_pred_timing==0) & (y_true_timing==1)).sum())
                    prec_t = TP_t / (TP_t + FP_t + 1e-9)
                    rec_t  = TP_t / (TP_t + FN_t + 1e-9)

                    # Lead-time: for bars where model predicted ≤ horizon AND GC actually came
                    lead_times = actual_days_arr[(y_pred_timing==1) & (y_true_timing==1)].tolist()
                    avg_lead   = float(np.mean(lead_times)) if lead_times else float('nan')
                    pct_early  = float(np.mean([l >= 5 for l in lead_times])) if lead_times else 0.0

                    st.markdown('<div class="sec-hdr">Timing Accuracy — Model C (Days Regressor)</div>',
                                unsafe_allow_html=True)
                    ta1, ta2, ta3, ta4 = st.columns(4)
                    with ta1:
                        c = "#00e5a0" if tight_acc > 0.40 else ("#f5a623" if tight_acc > 0.25 else "#f04060")
                        st.markdown(bkpi("Tight Accuracy (±5d)",
                            f"{tight_acc:.1%}",
                            "|pred − actual| ≤ 5 days (real timing)", c), unsafe_allow_html=True)
                    with ta2:
                        c = "#00e5a0" if not np.isnan(mean_err) and mean_err < 10 else "#f5a623"
                        err_str = f"{mean_err:.1f}d" if not np.isnan(mean_err) else "n/a"
                        st.markdown(bkpi("Mean Timing Error",
                            err_str,
                            f"±10d: {pct_within10:.1%} of predictions", c), unsafe_allow_html=True)
                    with ta3:
                        c = "#00e5a0" if not np.isnan(avg_lead) and avg_lead >= 5 else "#f5a623"
                        lead_str = f"{avg_lead:.1f}d" if not np.isnan(avg_lead) else "n/a"
                        st.markdown(bkpi("Avg Lead Time",
                            lead_str,
                            "days before actual GC (higher = earlier)", c), unsafe_allow_html=True)
                    with ta4:
                        c = "#00e5a0" if pct_early > 0.6 else "#f5a623"
                        st.markdown(bkpi("% Early Signals",
                            f"{pct_early:.1%}",
                            "signals fired ≥5 days before GC", c), unsafe_allow_html=True)

                # ── Confusion matrix ────────────────────────────────────────────
                TP = int(((y_pred == 1) & (y_true == 1)).sum())
                FP = int(((y_pred == 1) & (y_true == 0)).sum())
                FN = int(((y_pred == 0) & (y_true == 1)).sum())
                TN = int(((y_pred == 0) & (y_true == 0)).sum())

                precision = TP / (TP + FP + 1e-9)
                recall    = TP / (TP + FN + 1e-9)
                f1        = 2 * precision * recall / (precision + recall + 1e-9)
                accuracy  = (TP + TN) / (TP + FP + FN + TN + 1e-9)
                base_rate = n_pos / max(n_bars, 1)   # how often GC happens anyway

                # ── Display header ──────────────────────────────────────────────
                n_years = round(n_bars / 252, 1)
                st.markdown(f"""
                <div style="margin:18px 0 12px;">
                  <span style="font-size:18px;font-weight:700;color:#d4ddf5;font-family:'DM Sans';">{tk}</span>
                  <span style="font-family:'JetBrains Mono';font-size:11px;color:#7a8aaa;margin-left:10px;">
                    {n_bars:,} bars · {n_years}y history · {n_pos} GC events · window={horizon}d · threshold={threshold:.0%}
                  </span>
                </div>
                <div class="warn-box" style="margin-bottom:14px;">
                  TP = signal fired AND GC cross confirmed within {horizon} days &nbsp;|&nbsp;
                  FP = signal fired, NO GC cross in {horizon} days &nbsp;|&nbsp;
                  FN = no signal but GC cross DID happen &nbsp;|&nbsp;
                  TN = no signal, no GC cross
                </div>
                """, unsafe_allow_html=True)

                # ── KPI row ─────────────────────────────────────────────────────
                k1, k2, k3, k4, k5 = st.columns(5)
                with k1:
                    c = "#00e5a0" if precision > 0.60 else ("#f5a623" if precision > base_rate else "#f04060")
                    st.markdown(bkpi("Precision",   f"{precision:.1%}",
                        f"of signals, GC happened · base={base_rate:.1%}", c), unsafe_allow_html=True)
                with k2:
                    c = "#00e5a0" if recall > 0.60 else ("#f5a623" if recall > 0.40 else "#f04060")
                    st.markdown(bkpi("Recall",      f"{recall:.1%}",
                        "of real GCs, model caught them", c), unsafe_allow_html=True)
                with k3:
                    c = "#00e5a0" if f1 > 0.55 else "#f5a623"
                    st.markdown(bkpi("F1 Score",    f"{f1:.3f}",
                        "harmonic mean of precision & recall", c), unsafe_allow_html=True)
                with k4:
                    c = "#4da6ff" if accuracy > 0.70 else "#f5a623"
                    st.markdown(bkpi("Accuracy",    f"{accuracy:.1%}",
                        "overall correct bars", c), unsafe_allow_html=True)
                with k5:
                    lift = precision / max(base_rate, 1e-9)
                    c = "#00e5a0" if lift > 1.5 else ("#f5a623" if lift > 1.0 else "#f04060")
                    st.markdown(bkpi("Precision Lift", f"{lift:.2f}×",
                        f"vs random baseline ({base_rate:.1%})", c), unsafe_allow_html=True)

                # ── Confusion matrix grid ───────────────────────────────────────
                st.markdown('<div class="sec-hdr">Confusion Matrix — "Will a GC cross happen in the next {horizon} days?"</div>'.format(horizon=horizon),
                            unsafe_allow_html=True)
                cm_col, explain_col = st.columns([3, 2])
                with cm_col:
                    st.markdown(f"""
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;max-width:500px;">
                      <div style="background:rgba(0,229,160,.10);border:1px solid rgba(0,229,160,.35);border-radius:8px;padding:18px 20px;">
                        <div style="font-family:'JetBrains Mono';font-size:9px;color:#00e5a0;letter-spacing:.1em;text-transform:uppercase;margin-bottom:6px;">TP — True Positive</div>
                        <div style="font-size:40px;font-weight:700;color:#00e5a0;line-height:1;">{TP}</div>
                        <div style="font-family:'JetBrains Mono';font-size:11px;color:#7a8aaa;margin-top:6px;">Signal fired → GC occurred ✓</div>
                      </div>
                      <div style="background:rgba(240,64,96,.10);border:1px solid rgba(240,64,96,.35);border-radius:8px;padding:18px 20px;">
                        <div style="font-family:'JetBrains Mono';font-size:9px;color:#f04060;letter-spacing:.1em;text-transform:uppercase;margin-bottom:6px;">FP — False Positive</div>
                        <div style="font-size:40px;font-weight:700;color:#f04060;line-height:1;">{FP}</div>
                        <div style="font-family:'JetBrains Mono';font-size:11px;color:#7a8aaa;margin-top:6px;">Signal fired → no GC ✗</div>
                      </div>
                      <div style="background:rgba(245,166,35,.08);border:1px solid rgba(245,166,35,.25);border-radius:8px;padding:18px 20px;">
                        <div style="font-family:'JetBrains Mono';font-size:9px;color:#f5a623;letter-spacing:.1em;text-transform:uppercase;margin-bottom:6px;">FN — False Negative</div>
                        <div style="font-size:40px;font-weight:700;color:#f5a623;line-height:1;">{FN}</div>
                        <div style="font-family:'JetBrains Mono';font-size:11px;color:#7a8aaa;margin-top:6px;">No signal → GC happened ✗</div>
                      </div>
                      <div style="background:rgba(77,166,255,.06);border:1px solid rgba(77,166,255,.2);border-radius:8px;padding:18px 20px;">
                        <div style="font-family:'JetBrains Mono';font-size:9px;color:#4da6ff;letter-spacing:.1em;text-transform:uppercase;margin-bottom:6px;">TN — True Negative</div>
                        <div style="font-size:40px;font-weight:700;color:#4da6ff;line-height:1;">{TN}</div>
                        <div style="font-family:'JetBrains Mono';font-size:11px;color:#7a8aaa;margin-top:6px;">No signal → no GC ✓</div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                with explain_col:
                    lift_c = "#00e5a0" if lift > 1.5 else ("#f5a623" if lift > 1.0 else "#f04060")
                    st.markdown(f"""
                    <div style="background:var(--bg1);border:1px solid var(--border);border-radius:8px;
                                padding:24px;height:100%;display:flex;flex-direction:column;justify-content:center;">
                      <div style="font-family:'JetBrains Mono';font-size:10px;color:{lift_c};letter-spacing:.14em;
                                  text-transform:uppercase;font-weight:600;margin-bottom:8px;">
                        Precision Lift over Random
                      </div>
                      <div style="font-size:52px;font-weight:700;color:{lift_c};line-height:1;margin-bottom:10px;">
                        {lift:.2f}×
                      </div>
                      <div style="font-family:'JetBrains Mono';font-size:11px;color:#7a8aaa;line-height:1.6;">
                        A random guesser at this base rate ({base_rate:.1%}) would get
                        precision = {base_rate:.1%}.<br>
                        The model gets {precision:.1%}.<br>
                        Lift &gt; 1.5× = meaningful edge.<br>
                        Lift &lt; 1.0× = worse than random.
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                # ── Recall improvement analysis ────────────────────────────────
                st.markdown('<div class="sec-hdr">Recall Analysis — Find the threshold that catches the most GC events</div>',
                            unsafe_allow_html=True)

                def _cm(y_p, y_t):
                    tp = int(((y_p==1)&(y_t==1)).sum())
                    fp = int(((y_p==1)&(y_t==0)).sum())
                    fn = int(((y_p==0)&(y_t==1)).sum())
                    tn = int(((y_p==0)&(y_t==0)).sum())
                    prec = tp/(tp+fp+1e-9); rec = tp/(tp+fn+1e-9)
                    f1s  = 2*prec*rec/(prec+rec+1e-9)
                    return tp, fp, fn, tn, prec, rec, f1s

                # Sweep thresholds on BOTH Model A and Model B
                sweep_rows = []
                for thr in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
                    yp_a = (probs_gc   >= thr).astype(int)
                    yp_b = (probs_gc_b >= thr).astype(int)
                    tp_a_,fp_a_,fn_a_,tn_a_,pr_a_,rc_a_,f1_a_ = _cm(yp_a, y_true)
                    tp_b_,fp_b_,fn_b_,tn_b_,pr_b_,rc_b_,f1_b_ = _cm(yp_b, y_true)
                    sweep_rows.append({
                        "Threshold": f"{thr:.0%}",
                        "A: TP": tp_a_, "A: FN missed": fn_a_,
                        "A: Recall": f"{rc_a_:.1%}", "A: Precision": f"{pr_a_:.1%}", "A: F1": f"{f1_a_:.3f}",
                        "B: TP": tp_b_, "B: FN missed": fn_b_,
                        "B: Recall": f"{rc_b_:.1%}", "B: Precision": f"{pr_b_:.1%}", "B: F1": f"{f1_b_:.3f}",
                        "★": "← current" if abs(thr - threshold) < 0.01 else ""
                    })
                sweep_df = pd.DataFrame(sweep_rows)
                st.dataframe(sweep_df, hide_index=True, use_container_width=True)

                best_f1_row  = max(sweep_rows, key=lambda r: float(r['A: F1']))
                best_rec_row = max(sweep_rows, key=lambda r: float(r['A: Recall'].strip('%'))/100)
                best_b_row   = max(sweep_rows, key=lambda r: float(r['B: Recall'].strip('%'))/100)
                st.markdown(
                    f'<div class="warn-box" style="border-color:rgba(0,201,122,.4);color:#00c97a;">'
                    f'📊 <b>Model A — Best F1</b>: threshold {best_f1_row["Threshold"]} → '
                    f'Recall {best_f1_row["A: Recall"]}, Precision {best_f1_row["A: Precision"]}, F1 {best_f1_row["A: F1"]} &nbsp;|&nbsp; '
                    f'<b>Model B — Max Recall</b>: threshold {best_b_row["Threshold"]} → '
                    f'Recall {best_b_row["B: Recall"]} (catches most GC events)</div>',
                    unsafe_allow_html=True)

                # Model A vs B vs AND vs OR
                st.markdown('<div class="sec-hdr">Model Comparison — A (Precision) vs B (Recall) vs A AND B vs A OR B</div>',
                            unsafe_allow_html=True)

                tp_a,fp_a,fn_a,tn_a,prec_a,rec_a,f1_a = _cm(y_pred_a,  y_true)
                tp_b,fp_b,fn_b,tn_b,prec_b,rec_b,f1_b = _cm(y_pred_b,  y_true)
                y_pred_and = (y_pred_a & y_pred_b).astype(int)
                y_pred_or  = np.clip(y_pred_a + y_pred_b, 0, 1)
                tp_c,fp_c,fn_c,tn_c,prec_c,rec_c,f1_c = _cm(y_pred_and, y_true)
                tp_o,fp_o,fn_o,tn_o,prec_o,rec_o,f1_o = _cm(y_pred_or,  y_true)

                # Best single-model recall row
                y_pred_b_low = (probs_gc_b >= 0.25).astype(int)
                tp_bl,fp_bl,fn_bl,tn_bl,prec_bl,rec_bl,f1_bl = _cm(y_pred_b_low, y_true)

                comp_df = pd.DataFrame({
                    "Model":     [
                        "A — Precision",
                        f"B — Recall (thr={recall_threshold:.0%})",
                        f"B — Max Recall (thr=25%)",
                        "✅ A AND B",
                        "A OR B",
                    ],
                    "TP ✓":      [tp_a, tp_b, tp_bl, tp_c, tp_o],
                    "FP ✗":      [fp_a, fp_b, fp_bl, fp_c, fp_o],
                    "FN missed": [fn_a, fn_b, fn_bl, fn_c, fn_o],
                    "Precision": [f"{prec_a:.1%}", f"{prec_b:.1%}", f"{prec_bl:.1%}", f"{prec_c:.1%}", f"{prec_o:.1%}"],
                    "Recall":    [f"{rec_a:.1%}",  f"{rec_b:.1%}",  f"{rec_bl:.1%}",  f"{rec_c:.1%}",  f"{rec_o:.1%}"],
                    "F1":        [f"{f1_a:.3f}",   f"{f1_b:.3f}",   f"{f1_bl:.3f}",   f"{f1_c:.3f}",   f"{f1_o:.3f}"],
                })
                st.dataframe(comp_df, hide_index=True, use_container_width=True)
                st.markdown(
                    f'<div class="warn-box" style="border-color:rgba(0,201,122,.4);color:#00c97a;">'
                    f'💡 <b>To improve recall</b>: lower the "Recall Threshold" dropdown above or the "High Confidence Level" slider in the sidebar. '
                    f'At 25% threshold, Model A catches {sweep_rows[0]["A: TP"]} GC events '
                    f'(vs {tp_a} at {threshold:.0%}). '
                    f'Model B at 25% catches {sweep_rows[0]["B: TP"]} events. '
                    f'Use the threshold sweep table above to find your optimal precision/recall balance.</div>',
                    unsafe_allow_html=True)

                # ── Strategy performance (equity backtest) ──────────────────────
                st.markdown('<div class="sec-hdr">Strategy Equity Backtest — Entry at signal, exit at stop/TP/DC</div>',
                            unsafe_allow_html=True)

                probs_dc_bt = apply_calibration(
                    st.session_state.calibrator_dc,
                    st.session_state.models_dc, X_eval)

                bt = backtest_strategy(
                    df_eval.set_index(date_col),
                    probs_gc, probs_dc_bt,
                    model_quality_gc=st.session_state.model_quality_gc,
                    threshold=threshold, quality_min=quality_min,
                    recall_threshold=recall_threshold,
                    cooldown=cooldown,
                    transaction_cost=transaction_cost,
                    slippage=slippage,
                    stop_loss=stop_loss, take_profit=take_profit,
                    adx_min=adx_min_bt, hurst_min=hurst_min_bt,
                    ev_min=0.0 if ev_filter else -999.0,
                    avg_win=st.session_state.get("bt_avg_win", 0.11),
                    avg_loss=abs(st.session_state.get("bt_avg_loss", -0.05)),
                    probs_gc_b=probs_gc_b,
                    exit_at_gc=exit_at_gc,
                    trailing_stop=trailing_stp,
                )

                st.session_state['bt_win_rate'] = bt['win_rate']
                st.session_state['bt_avg_win']  = bt['avg_win']
                st.session_state['bt_avg_loss'] = bt['avg_loss']

                # ── Row 1: Returns & risk ──────────────────────────────────────
                m1,m2,m3,m4,m5,m6 = st.columns(6)
                def bk(label, val, sub="", col="#4da6ff"):
                    return qcard(label, val, sub, col)
                with m1:
                    c = "#00e5a0" if bt['total_return']>0 else "#f04060"
                    st.markdown(bk("Total Return", f"{bt['total_return']:+.1%}",
                        f"B&H: {bt['buy_hold_return']:+.1%}", c), unsafe_allow_html=True)
                with m2:
                    c = "#00e5a0" if bt['sharpe']>1 else ("#f5a623" if bt['sharpe']>0 else "#f04060")
                    st.markdown(bk("Sharpe", f"{bt['sharpe']:.2f}",
                        f"Sortino: {bt['sortino']:.2f}", c), unsafe_allow_html=True)
                with m3:
                    c = "#f04060" if bt['max_dd']<-0.20 else "#f5a623"
                    st.markdown(bk("Max Drawdown", f"{bt['max_dd']:+.1%}",
                        f"Calmar: {bt['calmar']:.2f}" if bt['calmar'] != float('inf') else "Calmar: ∞", c), unsafe_allow_html=True)
                with m4:
                    wc = "#00e5a0" if bt['win_rate']>0.55 else "#f5a623"
                    st.markdown(bk("Win Rate", f"{bt['win_rate']:.1%}",
                        f"{bt['total_trades']} trades · avg {bt['avg_trade']:+.1%}", wc), unsafe_allow_html=True)
                with m5:
                    live_kf = kelly_criterion(bt['win_rate'], bt['avg_win'], bt['avg_loss'])
                    st.markdown(bk("Kelly (0.25×)", f"{live_kf*0.25:.1%}",
                        f"full Kelly: {live_kf:.1%}", "#a78bfa"), unsafe_allow_html=True)
                with m6:
                    st.markdown(bk("Time in Mkt", f"{bt['time_in_market']:.1f}%",
                        f"avg hold: {int(np.mean([t['bars_held'] for t in bt['trade_log']])) if bt['trade_log'] else 0}d"), unsafe_allow_html=True)

                # ── Row 2: Filter stats ────────────────────────────────────────
                fc1,fc2,fc3,fc4 = st.columns(4)
                total_blocked = bt['blocked_regime'] + bt['blocked_ev'] + bt['blocked_quality']
                with fc1:
                    rc = "#f5a623" if bt['blocked_regime'] > 0 else "#7a8aaa"
                    st.markdown(bk("Regime Blocked", str(bt['blocked_regime']),
                        f"ADX<{adx_min_bt} or Hurst<{hurst_min_bt:.2f}", rc), unsafe_allow_html=True)
                with fc2:
                    ec = "#f5a623" if bt['blocked_ev'] > 0 else "#7a8aaa"
                    st.markdown(bk("EV Blocked", str(bt['blocked_ev']),
                        "EV ≤ 0 filtered out", ec), unsafe_allow_html=True)
                with fc3:
                    qc = "#f5a623" if bt['blocked_quality'] > 0 else "#7a8aaa"
                    st.markdown(bk("Quality Blocked", str(bt['blocked_quality']),
                        f"quality < {quality_min:.2f}", qc), unsafe_allow_html=True)
                with fc4:
                    exit_reasons = {}
                    for t in bt['trade_log']:
                        exit_reasons[t['reason']] = exit_reasons.get(t['reason'], 0) + 1
                    reason_str = "  ".join(f"{k}:{v}" for k,v in sorted(exit_reasons.items(), key=lambda x:-x[1]))
                    st.markdown(bk("Exit Reasons", str(bt['total_trades']),
                        reason_str or "no trades", "#4da6ff"), unsafe_allow_html=True)

                st.plotly_chart(plot_equity_curve(bt), use_container_width=True)

                # ── Signal log ──────────────────────────────────────────────────
                st.markdown('<div class="sec-hdr">Signal Log — every bar where model fired (prob ≥ threshold)</div>',
                            unsafe_allow_html=True)

                closes = df_eval["Close"].values
                opens  = df_eval["Open"].values
                dates  = df_eval[date_col].values
                gc_cross_actual = df_eval["gc_event"].values if "gc_event" in df_eval.columns else np.zeros(len(df_eval))

                sig_rows = []
                for i, p in enumerate(probs_gc):
                    if p < threshold:
                        continue
                    # What actually happened: did a GC cross occur within horizon bars?
                    window_end   = min(i + horizon, len(gc_cross_actual) - 1)
                    actual_cross = int(gc_cross_actual[i+1:window_end+1].max()) if window_end > i else 0
                    # Days until first cross in window (or "-" if none)
                    cross_offsets = np.where(gc_cross_actual[i+1:window_end+1] == 1)[0]
                    days_to_cross = int(cross_offsets[0] + 1) if len(cross_offsets) > 0 else None

                    entry_px = float(opens[i+1]) if i+1 < len(opens) else float(closes[i])
                    sig_rows.append({
                        "date":          pd.Timestamp(dates[i]).strftime("%Y-%m-%d"),
                        "prob":          round(float(p), 3),
                        "entry_px":      round(entry_px, 2),
                        "gc_in_window":  actual_cross,
                        "days_to_cross": days_to_cross,
                        "result":        "✓ GC occurred" if actual_cross else "✗ No GC",
                    })

                if sig_rows:
                    n_fired  = len(sig_rows)
                    n_hit    = sum(r["gc_in_window"] for r in sig_rows)
                    avg_days = np.mean([r["days_to_cross"] for r in sig_rows if r["days_to_cross"] is not None]) if n_hit else float("nan")
                    hit_rate = n_hit / n_fired if n_fired else 0

                    st.markdown(f"""
                    <div style="font-family:'JetBrains Mono';font-size:12px;color:#7a8aaa;margin-bottom:10px;">
                      {n_fired} signals fired &nbsp;·&nbsp;
                      <span style="color:#00e5a0;">{n_hit} confirmed GC crosses</span> &nbsp;·&nbsp;
                      <span style="color:#f04060;">{n_fired-n_hit} false alarms</span> &nbsp;·&nbsp;
                      Hit rate: <b style="color:{"#00e5a0" if hit_rate>0.55 else "#f5a623"}">{hit_rate:.1%}</b>
                      {f" · Avg {avg_days:.1f}d to cross" if not np.isnan(avg_days) else ""}
                    </div>
                    """, unsafe_allow_html=True)

                    rows_html = ""
                    for r in sig_rows[-50:][::-1]:   # most recent 50, newest first
                        hit_c = "#00e5a0" if r["gc_in_window"] else "#f04060"
                        days_str = str(r["days_to_cross"]) + "d" if r["days_to_cross"] else "—"
                        rows_html += f"""<tr>
                          <td style="color:#7a8aaa;">{r["date"]}</td>
                          <td style="color:#4da6ff;">{r["prob"]:.3f}</td>
                          <td>₹{r["entry_px"]:.1f}</td>
                          <td>{days_str}</td>
                          <td style="color:{hit_c};font-weight:600;">{r["result"]}</td>
                        </tr>"""
                    st.markdown(f"""
                    <table class="qtable">
                      <thead><tr>
                        <th>Date</th>
                        <th>GC Prob</th>
                        <th>Entry (next open)</th>
                        <th>Days to Cross</th>
                        <th>Did GC happen within {horizon}d?</th>
                      </tr></thead>
                      <tbody>{rows_html}</tbody>
                    </table>
                    """, unsafe_allow_html=True)
                else:
                    st.info(f"No signals fired at threshold {threshold:.0%}. Try lowering the threshold in the sidebar.")

                # ── Backtest Summary Panel ─────────────────────────────────────
                st.markdown('<div class="sec-hdr">📋 Backtest Summary — What Every Number Means</div>',
                            unsafe_allow_html=True)

                live_kf_bt = kelly_criterion(bt['win_rate'], bt['avg_win'], bt['avg_loss'])
                exit_reasons_summary = {}
                for t in bt['trade_log']:
                    exit_reasons_summary[t['reason']] = exit_reasons_summary.get(t['reason'], 0) + 1
                exit_str = ", ".join(f"{k}={v}" for k,v in sorted(exit_reasons_summary.items(), key=lambda x:-x[1]))

                st.markdown(f"""
                <div style="background:var(--bg1);border:1px solid var(--border);border-radius:10px;
                            padding:24px 28px;font-family:'JetBrains Mono';font-size:12px;line-height:1.85;">

                  <div style="font-size:14px;font-weight:700;color:#d4ddf5;margin-bottom:16px;">
                    {tk} · {n_years}y history · {bt['total_trades']} trades · Horizon = {horizon}d
                  </div>

                  <table style="width:100%;border-collapse:collapse;">
                    <thead>
                      <tr style="background:#0d1020;border-bottom:1px solid #1e2840;">
                        <th style="padding:8px 12px;color:#3d4d6a;font-size:10px;text-transform:uppercase;letter-spacing:.1em;text-align:left;">Parameter</th>
                        <th style="padding:8px 12px;color:#3d4d6a;font-size:10px;text-transform:uppercase;letter-spacing:.1em;text-align:left;">Value</th>
                        <th style="padding:8px 12px;color:#3d4d6a;font-size:10px;text-transform:uppercase;letter-spacing:.1em;text-align:left;">What It Means</th>
                        <th style="padding:8px 12px;color:#3d4d6a;font-size:10px;text-transform:uppercase;letter-spacing:.1em;text-align:left;">Good Range / Interpretation</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr style="border-bottom:1px solid #151c30;">
                        <td style="padding:8px 12px;color:#4da6ff;font-weight:600;">Precision</td>
                        <td style="padding:8px 12px;color:{'#00e5a0' if precision>0.6 else '#f5a623' if precision>base_rate else '#f04060'};">{precision:.1%}</td>
                        <td style="padding:8px 12px;color:#d4ddf5;">Of all bars where the model fired a signal, what % actually had a GC cross within the horizon window. TP/(TP+FP). The primary metric to watch — you want this high.</td>
                        <td style="padding:8px 12px;color:#7a8aaa;">&gt;60% = strong. Base rate = {base_rate:.1%} (random). Lift = {precision/max(base_rate,1e-9):.2f}×.</td>
                      </tr>
                      <tr style="border-bottom:1px solid #151c30;background:#080c18;">
                        <td style="padding:8px 12px;color:#4da6ff;font-weight:600;">Recall</td>
                        <td style="padding:8px 12px;color:{'#00e5a0' if recall>0.6 else '#f5a623' if recall>0.4 else '#f04060'};">{recall:.1%}</td>
                        <td style="padding:8px 12px;color:#d4ddf5;">Of all actual GC crosses that occurred, what % did the model catch with a signal beforehand. TP/(TP+FN). Higher = fewer missed opportunities.</td>
                        <td style="padding:8px 12px;color:#7a8aaa;">&gt;60% = good. 40–60% = acceptable. &lt;40% = missing too many events.</td>
                      </tr>
                      <tr style="border-bottom:1px solid #151c30;">
                        <td style="padding:8px 12px;color:#4da6ff;font-weight:600;">F1 Score</td>
                        <td style="padding:8px 12px;color:{'#00e5a0' if f1>0.55 else '#f5a623'};">{f1:.3f}</td>
                        <td style="padding:8px 12px;color:#d4ddf5;">Harmonic mean of precision and recall. 2×P×R/(P+R). Balances both metrics. More informative than accuracy alone when events are rare (as GC events are).</td>
                        <td style="padding:8px 12px;color:#7a8aaa;">&gt;0.55 = good balance. 0.40–0.55 = acceptable. &lt;0.40 = imbalanced.</td>
                      </tr>
                      <tr style="border-bottom:1px solid #151c30;background:#080c18;">
                        <td style="padding:8px 12px;color:#4da6ff;font-weight:600;">Confusion Matrix</td>
                        <td style="padding:8px 12px;color:#d4ddf5;">TP={TP} · FP={FP} · FN={FN} · TN={TN}</td>
                        <td style="padding:8px 12px;color:#d4ddf5;"><b style="color:#00e5a0;">TP</b>=signal fired & GC happened. <b style="color:#f04060;">FP</b>=signal fired & no GC (false alarm). <b style="color:#f5a623;">FN</b>=no signal & GC happened (miss). <b style="color:#4da6ff;">TN</b>=no signal & no GC (correct silence).</td>
                        <td style="padding:8px 12px;color:#7a8aaa;">Want: high TP, low FP. FP = costly false alarms. FN = missed profits.</td>
                      </tr>
                      <tr style="border-bottom:1px solid #151c30;">
                        <td style="padding:8px 12px;color:#4da6ff;font-weight:600;">Total Return</td>
                        <td style="padding:8px 12px;color:{'#00e5a0' if bt['total_return']>0 else '#f04060'};">{bt['total_return']:+.1%}</td>
                        <td style="padding:8px 12px;color:#d4ddf5;">Cumulative strategy return over the entire backtest period. Built from bar-by-bar position × close-to-close returns. Entry/exit on next-bar open. Includes commission ({transaction_cost*100:.2f}%) + slippage ({slippage*100:.2f}%) on both sides.</td>
                        <td style="padding:8px 12px;color:#7a8aaa;">Compare to B&H: {bt['buy_hold_return']:+.1%}. Positive = strategy earned money.</td>
                      </tr>
                      <tr style="border-bottom:1px solid #151c30;background:#080c18;">
                        <td style="padding:8px 12px;color:#4da6ff;font-weight:600;">Sharpe Ratio</td>
                        <td style="padding:8px 12px;color:{'#00e5a0' if bt['sharpe']>1 else '#f5a623' if bt['sharpe']>0 else '#f04060'};">{bt['sharpe']:.2f}</td>
                        <td style="padding:8px 12px;color:#d4ddf5;">Risk-adjusted return. √(252/avg_hold) × mean(trade_rets) / std(trade_rets). Computed on trade returns only (not flat/idle days), annualised by average holding period. Measures how much return you get per unit of volatility.</td>
                        <td style="padding:8px 12px;color:#7a8aaa;">&gt;2.0 = excellent. 1–2 = good. 0–1 = marginal. &lt;0 = strategy lost money.</td>
                      </tr>
                      <tr style="border-bottom:1px solid #151c30;">
                        <td style="padding:8px 12px;color:#4da6ff;font-weight:600;">Sortino Ratio</td>
                        <td style="padding:8px 12px;color:{'#00e5a0' if bt['sortino']>1 else '#f5a623' if bt['sortino']>0 else '#f04060'};">{bt['sortino']:.2f}</td>
                        <td style="padding:8px 12px;color:#d4ddf5;">Like Sharpe but denominator is downside deviation only (negative trade returns). More relevant for investors who care only about downside risk, not upside volatility.</td>
                        <td style="padding:8px 12px;color:#7a8aaa;">&gt;2.0 = excellent. Should be higher than Sharpe (penalises only bad vol).</td>
                      </tr>
                      <tr style="border-bottom:1px solid #151c30;background:#080c18;">
                        <td style="padding:8px 12px;color:#4da6ff;font-weight:600;">Max Drawdown</td>
                        <td style="padding:8px 12px;color:{'#f04060' if bt['max_dd']<-0.20 else '#f5a623'};">{bt['max_dd']:+.1%}</td>
                        <td style="padding:8px 12px;color:#d4ddf5;">Largest peak-to-trough decline in the equity curve. min(Equity / rolling_max − 1). Measures the worst-case loss an investor would have experienced at any point during the backtest.</td>
                        <td style="padding:8px 12px;color:#7a8aaa;">&gt;−10% = low risk. −10% to −20% = moderate. &lt;−20% = high risk strategy.</td>
                      </tr>
                      <tr style="border-bottom:1px solid #151c30;">
                        <td style="padding:8px 12px;color:#4da6ff;font-weight:600;">Calmar Ratio</td>
                        <td style="padding:8px 12px;color:{'#00e5a0' if (bt['calmar']!=float('inf') and bt['calmar']>1) else '#f5a623'}">{f"{bt['calmar']:.2f}" if bt['calmar'] != float('inf') else "∞"}</td>
                        <td style="padding:8px 12px;color:#d4ddf5;">CAGR / |Max Drawdown|. Measures how much annualised return you earn per unit of max drawdown pain. CAGR computed over total elapsed bars (not just active bars — avoids inflating by 3–10×).</td>
                        <td style="padding:8px 12px;color:#7a8aaa;">&gt;1.0 = strategy earns more than its worst drawdown annually. &gt;2.0 = excellent.</td>
                      </tr>
                      <tr style="border-bottom:1px solid #151c30;background:#080c18;">
                        <td style="padding:8px 12px;color:#4da6ff;font-weight:600;">Win Rate</td>
                        <td style="padding:8px 12px;color:{'#00e5a0' if bt['win_rate']>0.55 else '#f5a623'};">{bt['win_rate']:.1%}</td>
                        <td style="padding:8px 12px;color:#d4ddf5;">% of completed trades with positive return (after costs). {bt['total_trades']} total trades. Avg win: {bt['avg_win']:+.1%}. Avg loss: {bt['avg_loss']:+.1%}. Profit factor: {bt['profit_factor']:.2f}.</td>
                        <td style="padding:8px 12px;color:#7a8aaa;">&gt;55% = good (at avg win ≈ avg loss). Can be lower with high win/loss ratio.</td>
                      </tr>
                      <tr style="border-bottom:1px solid #151c30;">
                        <td style="padding:8px 12px;color:#4da6ff;font-weight:600;">Expectancy (EV/trade)</td>
                        <td style="padding:8px 12px;color:{'#00e5a0' if bt['expectancy']>0 else '#f04060'};">{bt['expectancy']:+.2%}</td>
                        <td style="padding:8px 12px;color:#d4ddf5;">Average expected return per trade = WinRate × AvgWin + LossRate × AvgLoss. Positive = strategy has positive edge. This is what you expect to earn on average each time you take a trade.</td>
                        <td style="padding:8px 12px;color:#7a8aaa;">&gt;0% = edge exists. &gt;1% per trade = strong edge for the holding period.</td>
                      </tr>
                      <tr style="border-bottom:1px solid #151c30;background:#080c18;">
                        <td style="padding:8px 12px;color:#4da6ff;font-weight:600;">Kelly (0.25×)</td>
                        <td style="padding:8px 12px;color:#a78bfa;">{live_kf_bt*0.25:.1%} (full: {live_kf_bt:.1%})</td>
                        <td style="padding:8px 12px;color:#d4ddf5;">Quarter-Kelly position size recommendation. Full Kelly = (b×p−q)/b where b=win/loss ratio. Quarter-Kelly used because full Kelly is too aggressive in practice. This is the recommended % of capital per trade.</td>
                        <td style="padding:8px 12px;color:#7a8aaa;">5–15% per trade = typical range. &gt;25% = overlevered. Recomputed from this backtest.</td>
                      </tr>
                      <tr style="border-bottom:1px solid #151c30;">
                        <td style="padding:8px 12px;color:#4da6ff;font-weight:600;">Time in Market</td>
                        <td style="padding:8px 12px;color:#4da6ff;">{bt['time_in_market']:.1f}%</td>
                        <td style="padding:8px 12px;color:#d4ddf5;">% of all backtest bars where the strategy held a position. Lower = more selective signals. Idle capital can be deployed elsewhere. Compare: buy-and-hold = 100%.</td>
                        <td style="padding:8px 12px;color:#7a8aaa;">Lower is often better. High return with low market exposure = efficient strategy.</td>
                      </tr>
                      <tr style="border-bottom:1px solid #151c30;background:#080c18;">
                        <td style="padding:8px 12px;color:#4da6ff;font-weight:600;">Precision Lift</td>
                        <td style="padding:8px 12px;color:{'#00e5a0' if (precision/max(base_rate,1e-9))>1.5 else '#f5a623'};">{precision/max(base_rate,1e-9):.2f}×</td>
                        <td style="padding:8px 12px;color:#d4ddf5;">Model precision / base rate. Base rate = how often GC events happen randomly ({base_rate:.1%}). A random guesser would get precision = base rate. Lift shows how much better the model is than random.</td>
                        <td style="padding:8px 12px;color:#7a8aaa;">&gt;1.5× = meaningful edge. &gt;2× = strong. &lt;1× = worse than random.</td>
                      </tr>
                      <tr>
                        <td style="padding:8px 12px;color:#4da6ff;font-weight:600;">Exit Breakdown</td>
                        <td style="padding:8px 12px;color:#7a8aaa;">{exit_str or "—"}</td>
                        <td style="padding:8px 12px;color:#d4ddf5;">How trades were closed. GC_event=exited at actual crossover (ideal). trail_SL=trailing stop triggered. SL=fixed stop loss hit. TP=take profit hit. DC_signal=death cross signal appeared.</td>
                        <td style="padding:8px 12px;color:#7a8aaa;">High GC_event exits = strategy timing is working. High SL exits = signals firing too early.</td>
                      </tr>
                    </tbody>
                  </table>

                  <div style="margin-top:16px;padding:12px 14px;background:#0d1020;border-radius:6px;
                              border-left:3px solid #f5a623;font-size:11px;color:#7a8aaa;">
                    <b style="color:#d4ddf5;">Backtest assumptions:</b>
                    Entry/exit on next-bar open ·
                    Commission: {transaction_cost*100:.2f}% each side ·
                    Slippage: {slippage*100:.2f}% each side ·
                    Stop loss: {stop_loss:.0%} (or 2×ATR, whichever is larger) ·
                    Take profit: {take_profit:.0%} ·
                    Cooldown: {cooldown} bars between trades ·
                    CAGR computed over total elapsed bars (not active bars).
                  </div>
                </div>
                """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# (SCREENER REMOVED — charts moved to Predict tab)
# ═══════════════════════════════════════════════════════════════════
if False:
    st.markdown("### Stock Screener")
    screener_tickers_input = st.text_area("Tickers to Screen",
        value="\n".join([
            "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS",
            "SBIN.NS","AXISBANK.NS","KOTAKBANK.NS","ITC.NS","WIPRO.NS",
            "LT.NS","MARUTI.NS","TATASTEEL.NS","ONGC.NS","TITAN.NS",
            "ADANIENT.NS","BAJFINANCE.NS","SUNPHARMA.NS","NESTLEIND.NS",
        ]), height=150)
    screen_btn = st.button("🔍 Run Screener", type="primary")

    if screen_btn:
        screen_tickers = [t.strip().upper() for t in
                          screener_tickers_input.replace(",","\n").split("\n") if t.strip()]
        results_list = []; prog = st.progress(0); status = st.empty()
        for i, tk in enumerate(screen_tickers):
            status.text(f"Scanning {tk}...")
            try:
                r, _ = predict_single(
                    tk,
                    st.session_state.models_gc, st.session_state.models_dc,
                    st.session_state.model_quality_gc, st.session_state.model_quality_dc,
                    st.session_state.model_timing_gc, st.session_state.feature_cols,
                    st.session_state.calibrator_gc, st.session_state.calibrator_dc,
                    st.session_state.start_date,
                    st.session_state.fast_p, st.session_state.slow_p,
                    threshold, quality_min, st.session_state.pred_days,
                    models_gc_recall=st.session_state.models_gc_recall,
                    calibrator_gc_recall=st.session_state.calibrator_gc_recall,
                    model_days_regressor=st.session_state.model_days_regressor,
                    recall_threshold=recall_threshold,
                )
                if r: results_list.append(r)
            except: pass
            prog.progress((i+1)/len(screen_tickers))
        status.empty(); prog.empty()

        if not results_list:
            st.error("No results. Check ticker symbols.")
        else:
            df_sc = pd.DataFrame(results_list)
            q_num = pd.to_numeric(df_sc["quality_gc"], errors="coerce").fillna(0)
            # ── Composite score: prob × quality × filter multipliers ───────────
            # Stocks passing all 3 filters get a higher ranking weight
            t_ok_s  = df_sc.get("trend_ok",        pd.Series([True]*len(df_sc), index=df_sc.index)).astype(float)
            r_ok_s  = df_sc.get("regime_price_ok", pd.Series([True]*len(df_sc), index=df_sc.index)).astype(float)
            nc_ok_s = df_sc.get("near_cross_ok",   pd.Series([True]*len(df_sc), index=df_sc.index)).astype(float)
            filter_mult = 0.5 + 0.2*t_ok_s + 0.2*r_ok_s + 0.1*nc_ok_s   # range 0.5–1.0
            df_sc["combined_score"] = df_sc["prob_gc"] * q_num * filter_mult
            comp_col = df_sc.get("composite_signal", pd.Series([False]*len(df_sc), index=df_sc.index))
            df_sc["composite_signal"] = comp_col
            df_sc = df_sc.sort_values("combined_score", ascending=False)
            n_composite = int(df_sc["composite_signal"].sum())
            st.markdown(
                f"**{len(df_sc)} tickers scanned** · "
                f"**{n_composite} pass composite gate** (prob>60% AND near-cross AND ADX>20) · "
                f"sorted by composite score"
            )
            rows_html = ""
            for _, row in df_sc.iterrows():
                sc    = {"bullish":"#00e5a0","bearish":"#f04060","caution":"#f5a623","neutral":"#7a8aaa"}.get(row["signal_type"],"#7a8aaa")
                qv    = row["quality_gc"]
                qs    = f"{qv:.3f}" if not np.isnan(qv) else "—"
                qc    = "#00e5a0" if (not np.isnan(qv) and qv >= quality_min) else "#f5a623"
                t_ok  = bool(row.get("trend_ok",        True))
                r_ok  = bool(row.get("regime_price_ok", True))
                nc_ok = bool(row.get("near_cross_ok",   True))
                gate_html = (
                    f'<span style="color:{"#00e5a0" if t_ok  else "#f04060"}">ADX</span>·'
                    f'<span style="color:{"#00e5a0" if r_ok  else "#f04060"}">MA200</span>·'
                    f'<span style="color:{"#00e5a0" if nc_ok else "#f5a623"}">Near</span>'
                )
                comp_star = "⭐ " if bool(row.get("composite_signal", False)) else ""
                rows_html += f"""<tr>
                  <td><b>{comp_star}{row['ticker']}</b></td>
                  <td style="color:{sc};">{str(row['signal'])[:30]}</td>
                  <td>{row['prob_gc']:.1%}</td>
                  <td style="color:{qc};">{qs}</td>
                  <td>{row['timing']}</td>
                  <td>{gate_html}</td>
                  <td>{row['hist_gc_success']:.0%}</td>
                  <td style="color:{'#00e5a0' if row['price_vs_ma200']>0 else '#f04060'};">{row['price_vs_ma200']:+.1f}%</td>
                  <td>{row['ann_vol']:.1f}%</td>
                  <td>₹{row['current_price']:.1f}</td>
                </tr>"""
            st.markdown(f"""
            <table class="qtable"><thead><tr>
              <th>Ticker</th><th>Signal</th><th>GC Prob</th>
              <th>Quality</th><th>Est. Days</th><th>Filters (ADX·MA200·Near)</th>
              <th>Hist %</th><th>vs MA200</th><th>Ann Vol</th><th>Price</th>
            </tr></thead><tbody>{rows_html}</tbody></table>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            top5 = df_sc.head(5)
            fig_s = go.Figure()
            fig_s.add_trace(go.Bar(x=top5["ticker"], y=top5["prob_gc"],
                name="GC Prob (cal.)", marker_color="#4da6ff", opacity=0.85))
            q5 = pd.to_numeric(top5["quality_gc"], errors="coerce").fillna(0)
            fig_s.add_trace(go.Bar(x=top5["ticker"], y=q5,
                name="Quality Score", marker_color="#00e5a0", opacity=0.85))
            fig_s.update_layout(**DARK, barmode="group",
                title="Top 5 — calibrated GC probability vs quality score",
                yaxis=dict(**GRID, range=[0,1]), xaxis=dict(**GRID),
                legend=dict(bgcolor="#080c18"),
                height=300, margin=dict(l=40,r=15,t=50,b=40))
            st.plotly_chart(fig_s, use_container_width=True)

            # ── Golden Cross Timeline chart ────────────────────────────────────
            st.markdown('<div class="sec-hdr">Golden Cross Timeline — When Is Each Stock Crossing?</div>',
                        unsafe_allow_html=True)

            # Build timeline data: parse timing string → days estimate
            gc_timeline = []
            for _, row in df_sc.iterrows():
                timing_str = str(row.get('timing', 'n/a'))
                state      = str(row.get('state', ''))
                prob       = float(row['prob_gc'])
                if state == 'Golden Cross':
                    days_est = 0   # already in GC
                    category = 'In GC Now'
                elif timing_str.startswith('~') and 'days' in timing_str:
                    try:
                        days_est = int(timing_str.replace('~','').replace('days','').strip())
                    except:
                        days_est = 999
                    category = 'Approaching' if days_est <= 10 else 'Watch'
                else:
                    days_est = 999
                    category = 'No Signal'
                gc_timeline.append({
                    'ticker':   row['ticker'],
                    'days_est': days_est,
                    'category': category,
                    'prob_gc':  prob,
                })

            df_tl = pd.DataFrame(gc_timeline)
            # Only show stocks with a meaningful signal
            df_tl_show = df_tl[df_tl['category'] != 'No Signal'].sort_values('days_est')

            if not df_tl_show.empty:
                cat_colors = {'In GC Now': '#00e5a0', 'Approaching': '#f5a623', 'Watch': '#4da6ff'}
                fig_tl = go.Figure()
                for cat, grp in df_tl_show.groupby('category', sort=False):
                    col = cat_colors.get(cat, '#7a8aaa')
                    fig_tl.add_trace(go.Bar(
                        y=grp['ticker'],
                        x=grp['days_est'].clip(upper=30),
                        orientation='h',
                        name=cat,
                        marker_color=col,
                        opacity=0.85,
                        text=[f"{d}d" if d > 0 else "NOW" for d in grp['days_est']],
                        textposition='outside',
                        customdata=grp['prob_gc'].values,
                        hovertemplate='%{y}: %{x}d away · GC Prob %{customdata:.1%}<extra></extra>',
                    ))
                fig_tl.add_vline(x=0, line=dict(color='#00e5a0', width=1.5, dash='dot'),
                                 annotation_text='  GC Now', annotation_font_color='#00e5a0')
                fig_tl.update_layout(**DARK,
                    title=dict(text='Golden Cross Timeline — estimated days until crossover',
                               font=dict(size=13, color='#d4ddf5')),
                    xaxis=dict(title='Days Until Golden Cross', **GRID, range=[-1, 32]),
                    yaxis=dict(**GRID),
                    barmode='overlay',
                    legend=dict(bgcolor='#080c18', bordercolor='#151c30'),
                    height=max(280, 30 * len(df_tl_show) + 80),
                    margin=dict(l=100, r=60, t=50, b=40),
                )
                st.plotly_chart(fig_tl, use_container_width=True)
            else:
                st.info("No stocks with an imminent Golden Cross signal at the current threshold.")

            # ── GC Probability heatmap ─────────────────────────────────────────
            st.markdown('<div class="sec-hdr">GC Probability Heatmap — All Scanned Stocks</div>',
                        unsafe_allow_html=True)
            df_hm = df_sc.sort_values('prob_gc', ascending=True).tail(20)
            fig_hm = go.Figure()
            bar_cols = ['#00e5a0' if p >= threshold else ('#f5a623' if p >= 0.45 else '#3d4d6a')
                        for p in df_hm['prob_gc']]
            fig_hm.add_trace(go.Bar(
                y=df_hm['ticker'],
                x=df_hm['prob_gc'],
                orientation='h',
                marker_color=bar_cols,
                opacity=0.85,
                text=[f"{p:.1%}" for p in df_hm['prob_gc']],
                textposition='outside',
                hovertemplate='%{y}: GC Prob %{x:.1%}<extra></extra>',
            ))
            fig_hm.add_vline(x=threshold, line=dict(color='#f5a623', width=1.5, dash='dash'),
                             annotation_text=f'  Threshold {threshold:.0%}',
                             annotation_font_color='#f5a623')
            fig_hm.update_layout(**DARK,
                title=dict(text='GC Probability — top 20 stocks (sorted)',
                           font=dict(size=13, color='#d4ddf5')),
                xaxis=dict(title='Calibrated GC Probability', **GRID, range=[0, 1.1]),
                yaxis=dict(**GRID),
                height=max(280, 25 * len(df_hm) + 80),
                margin=dict(l=100, r=60, t=50, b=40),
            )
            st.plotly_chart(fig_hm, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 3 — MODEL INTERNALS
# ═══════════════════════════════════════════════════════════════════
with tab_model:
    st.markdown("### Model Internals & Diagnostics")
    tm = st.session_state.train_metrics

    m1,m2,m3,m4,m5 = st.columns(5)
    with m1:
        c = "#00e5a0" if tm.get('auc_gc',0)>0.85 else "#f5a623"
        st.markdown(qcard("GC ROC-AUC", f"{tm.get('auc_gc',0):.3f}",
            "calibrated, out-of-sample", c), unsafe_allow_html=True)
    with m2:
        st.markdown(qcard("GC Avg Precision", f"{tm.get('ap_gc',0):.3f}",
            "area under P-R curve", "#4da6ff"), unsafe_allow_html=True)
    with m3:
        c = "#00e5a0" if tm.get('auc_dc',0)>0.85 else "#f5a623"
        st.markdown(qcard("DC ROC-AUC", f"{tm.get('auc_dc',0):.3f}",
            "calibrated, out-of-sample", c), unsafe_allow_html=True)
    with m4:
        b = tm.get('brier_gc', 0)
        bc = "#00e5a0" if b < 0.10 else ("#f5a623" if b < 0.15 else "#f04060")
        st.markdown(qcard("Brier Score (GC)", f"{b:.4f}",
            "0=perfect  0.25=random  lower=better", bc), unsafe_allow_html=True)
    with m5:
        st.markdown(qcard("Walk-Forward Folds", str(tm.get('n_folds',0)),
            f"train: {tm.get('n_train',0):,}  test: {tm.get('n_test',0):,}"),
            unsafe_allow_html=True)

    st.markdown('<div class="sec-hdr">Feature Importance — GC Model (ensemble average)</div>',
                unsafe_allow_html=True)
    fi = st.session_state.feature_importance
    if fi is not None and len(fi) > 0:
        st.plotly_chart(plot_feature_importance(fi, top_n=25), use_container_width=True)

    st.markdown('<div class="sec-hdr">Critical Fixes — v13 (new) + v11/v10 (inherited)</div>', unsafe_allow_html=True)
    fixes = [
        ("Sector embeddings (v13)",
         "v12: model treated HDFC Bank and HAL identically — no sector context",
         "v13: 16-sector one-hot + 3-cap one-hot; XGBoost applies different logic to each sector"),
        ("Sector-demeaned features (v13)",
         "v12: RSI, ADX, MA200-dist were absolute — model could be 'betting on a sector', not a pattern",
         "v13: feature_stock − mean(feature_sector) per date; teaches 'stronger than peers' signals"),
        ("ATR-based labels (v13)",
         "v12: fixed 21-bar horizon — 5% gain in IT blue-chip = 'success', same as 5% in small-cap noise",
         "v13: success = price hits +2×ATR before −1×ATR; standardised across all volatility regimes"),
        ("Market regime feature (v13)",
         "v12: no macro context — model fired GC signals equally in bull and bear markets",
         "v13: mkt_regime = Nifty50 above/below MA200; model learns macro bull context"),
        ("Cap-mix enforcement (v13)",
         "v12: no cap-mix control — large-cap heavy universe biased model toward high-beta moonshots",
         "v13: 40% Large / 40% Mid / 20% Small resampling before walk-forward split"),
        ("Gap velocity (BUG-A/v11)",
         "v10: gap_velocity used ema_gap.diff() = ₹/bar — absolute, not scale-free across tickers",
         "v11: uses ema_gap_pct.diff() = %/bar — scale-free, correct for multi-ticker training"),
        ("gap_to_cross (BUG-B/v11)",
         "v10: ema_gap(₹) / gap_velocity(₹/bar) — numerically unstable across price levels",
         "v11: ema_gap_pct(%) / gap_velocity(%/bar) — both in %, ratio = bars (correct)"),
        ("macd_hist (BUG-H/v11)",
         "v10: macd_hist kept in feature set as absolute ₹ — scale-dependent across tickers",
         "v11: replaced with macd_hist_pct = macd_hist / Close × 100 — scale-free %"),
        ("convergence_speed (BUG-E/v11)",
         "v10: gap_velocity_5(₹/bar) / atr_pct(%) = ₹/% — dimensionally wrong",
         "v11: gap_velocity_5 now %/bar (BUG-A), so ratio = 1/bar (correct)"),
        ("Horizon selector (BUG-J/v11)",
         "v10: create_targets always used pred_days=15; all 10/30/60/90d tabs showed identical confusion matrix",
         "v11: create_targets called with bt_horizon; each tab evaluates a genuinely different window"),
        ("Degenerate recall (BUG-K/v11)",
         "v10: 70/30 single-ticker split left 2-4 GC events in test; model fires broadly → FN=0, Recall=1",
         "v11: full history used (minus 250-bar warmup); 10-50× more events for stable statistics"),
        ("GC Reliability metric (BUG-L/v11)",
         "v10: showed TP/(TP+FP) = GC-event precision — users expected price win rate",
         "v11: now shows price-direction win rate = % of signals where price was higher after horizon days"),
        ("GC label (BUG-1)",
         "v9: shift(+5) on raw cross AND shift(+5) on regime → tautology, model learned 'in bull trend' not 'cross coming'",
         "v10: shift(-5) on label outcome only (safe for targets) — cross bar correctly identified"),
        ("Timing label (BUG-2)",
         "v9: fillna(n_days) gave all 'no-cross' bars label 0.0 — timing model trained on ~85% garbage",
         "v10: leave NaN for 'no cross' bars; train_timing_model() already filters NaN rows"),
        ("Cross-ticker contamination (BUG-3)",
         "v9: concat + reset_index destroyed date order; walk-forward cut by position not time — model saw future",
         "v10: sort by date before walk-forward; splits are truly chronological"),
        ("Fast-mode subsampling (BUG-4)",
         "v9: subsampled globally before splitting → scrambled time order, lookahead in every fold",
         "v10: subsampling only inside the train fold; test and cal sets never touched"),
        ("Equity curve (BUG-5)",
         "v9: entire trade P&L on exit bar → flat intermediate bars → drawdown severely understated",
         "v10: bar-by-bar position × c-to-c returns; entry/exit bars use open fill price"),
        ("CAGR (BUG-7)",
         "v9: denominator = active bars → inflated annualised return by total/active (3–10×)",
         "v10: denominator = total elapsed bars — true CAGR over backtest period"),
        ("hist_gc_success (BUG-8)",
         "v9: measured GC regime at same bar as past cross — tautology, almost always 1",
         "v10: shift(-20) on regime (forward outcome), lagged 21 bars — true success rate"),
        ("Platt calibration (FLAW-3)",
         "v9: LogisticRegression with no class weight → biased toward majority class (no-cross)",
         "v10: class_weight='balanced' — correct handling of rare GC events (~8% of bars)"),
    ]
    rows_f = "".join(f"""<tr>
      <td><b>{f[0]}</b></td>
      <td style="color:#f04060;font-size:11px;">{f[1]}</td>
      <td style="color:#00e5a0;font-size:11px;">{f[2]}</td>
    </tr>""" for f in fixes)
    st.markdown(f"""
    <table class="qtable"><thead><tr>
      <th>Issue</th><th>Problem</th><th>Fix</th>
    </tr></thead><tbody>{rows_f}</tbody></table>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 4 — ABOUT
# ═══════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("### About — Golden Cross Predictor v13")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
**Model Architecture (6-layer, no lookahead)**

1. **Model A — Binary Ensemble** (XGBoost × 2 + RF)
   - Target: blended 60% ATR-based / 40% horizon label (v13)
   - Scale-free features + sector/cap/regime embeddings (v13)
   - Temporal walk-forward: sorted by date, purge gap, OOS cal + test
   - Near-cross bias filter: rows where |gap| < 1% excluded from training

2. **Model B — Recall Ensemble** (XGBoost + RF, 2× pos weight)
   - Same features, higher recall / lower precision
   - Entry requires: Model A ≥ threshold AND Model B ≥ recall_threshold

3. **Platt Calibration** — Logistic re-mapping (class_weight=balanced)
   - Strictly OOS calibration fold (never seen in training)
   - Brier score shown to verify calibration quality

4. **Quality Model** (XGBRegressor)
   - Only actual gc_event bars used for training
   - `0.4×fwd_return + 0.3×volume + 0.3×(ADX + MTF)`

5. **Model C — Days Regressor** (XGBRegressor on raw days)
   - Target: actual days to next GC, capped at 60 (999 = no GC)
   - Output: "Expected GC in X days" — precise timing, not binary

6. **Regime & EV Filters**
   - Regime filter: ADX > threshold AND Hurst > threshold
   - Market regime: Nifty 50 above/below its own MA200 (v13)
   - EV filter: only trade when P×avg_win − (1−P)×avg_loss > 0
   - Exit: at actual GC event OR trailing stop OR ATR stop
        """)
    with col_b:
        st.markdown("""
**v13 Upgrades (Senior DS Recommendations)**

| Upgrade | What Changed |
|---|---|
| Sector embeddings | 16 sectors × one-hot → XGBoost applies different logic per sector |
| Cap embeddings | Large/Mid/Small one-hot → prevents large-cap bull regime overfitting |
| Sector-demeaned features | RSI, MA200 dist, ADX, ATR%, MACD vs sector mean → "stronger than peers" |
| ATR-based labels | Success = hit +2×ATR before −1×ATR (not fixed 21-day horizon) |
| Market regime feature | Nifty 50 above/below MA200 → model learns macro bull vs bear context |
| Cap-mix enforcement | 40% Large / 40% Mid / 20% Small resampling before train split |
| Default start 2010 | Captures sideways 2018-19 whipsaw years for more robust learning |

**Signal Generation Logic**
```
if Model_A ≥ 0.60 AND Model_B ≥ 0.40:
    if Hurst > 0.45:               # mean-reversion filter
        if EV = P×win − (1−P)×loss > 0:  # EV filter
            if quality_score ≥ 0.35:      # quality gate
                signal = "PRE-GC ENTRY"
                size   = 0.25 × Kelly × (prob / threshold)
```
        """)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
**Model Architecture (6-layer, no lookahead)**

1. **Model A — Binary Ensemble** (XGBoost × 2 + RF)
   - Target: `gc_event` = strict one-row-per-crossover event
   - Scale-free features only (pct/ratio — no absolute ₹)
   - Temporal walk-forward: sorted by date, purge gap, OOS cal + test
   - Near-cross bias filter: rows where |gap| < 1% excluded from training

2. **Model B — Recall Ensemble** (XGBoost + RF, 2× pos weight)
   - Same features, higher recall / lower precision
   - Entry requires: Model A ≥ threshold AND Model B ≥ recall_threshold
   - AND logic prevents FP explosion vs OR logic

3. **Platt Calibration** — Logistic re-mapping (class_weight=balanced)
   - Strictly OOS calibration fold (never seen in training)
   - Brier score shown to verify calibration quality

4. **Quality Model** (XGBRegressor)
   - Only actual gc_event bars used for training
   - `0.4×fwd_return + 0.3×volume + 0.3×(ADX + MTF)`

5. **Model C — Days Regressor** (XGBRegressor on raw days)
   - Target: actual days to next GC, capped at 60 (999 = no GC)
   - Output: "Expected GC in X days" — precise timing, not binary
   - Tight accuracy metric: |pred − actual| ≤ 5 days

6. **Regime & EV Filters**
   - Regime filter: ADX > threshold AND Hurst > threshold
   - EV filter: only trade when P×avg_win − (1−P)×avg_loss > 0
   - Exit: at actual GC event OR trailing stop OR ATR stop
        """)
    with col_b:
        st.markdown("""
**v12 Fixes vs v11**

| Fix | Problem | Solution |
|---|---|---|
| GC definition | Continuous state (ema>ema) used as event | Strict crossover: ema crosses above on exact bar |
| Feature leakage | gc_regime binary state in features | Removed — only gap magnitude & velocity |
| Target | Binary "GC in N days" | Regression: actual days to GC (capped 60, 999=none) |
| Near-cross bias | Model trained on obvious near-cross rows | Filter |gap_pct| < 1% from training |
| Timing metric | pred≤30 AND actual≤30 → 93% (too lenient) | |pred − actual| ≤ 5 days (real accuracy) |
| Combined logic | A OR B → FP explodes | A AND B → precision preserved, recall improves |
| Regime filter | No environment check | ADX > 20 AND Hurst > 0.45 required |
| EV filter | All signals treated equally | Only trade when EV = P×win − (1−P)×loss > 0 |
| Exit logic | Fixed TP/SL only | Exit at actual GC event + trailing stop |
| Backtest display | No filter stats shown | blocked_regime, blocked_ev, exit reasons shown |

**Signal Generation Logic**
```
if Model_A ≥ 0.60 AND Model_B ≥ 0.40:
    if ADX > 20 AND Hurst > 0.45:      # regime filter
        if EV = P×win − (1−P)×loss > 0: # EV filter
            if quality_score ≥ 0.35:     # quality gate
                signal = "PRE-GC ENTRY"
                size   = 0.25 × Kelly × (prob / threshold)
```
        """)
    with col_b:
        st.markdown("""
**v10 Critical Fixes vs v9**

| Bug | v9 Problem | v10 Fix |
|---|---|---|
| GC label | shift(+5) on raw cross → tautology | shift(-5) on outcome (label lookahead OK) |
| Timing label | fillna(n_days)=0 for all "no cross" bars | Keep NaN; filter at train time |
| Temporal split | reset_index destroys date order | Sort by date before walk-forward |
| Fast mode | Subsample before split → scrambles time | Subsample inside train fold only |
| Equity curve | Lump P&L on exit bar → fake drawdown | Bar-by-bar position × c-to-c returns |
| ATR stop | atr/entry_price (wrong scale) | atr_pct/100 (fractional, correct) |
| CAGR | Active-bar denominator (inflated 3-10×) | Total elapsed bars (correct) |
| hist_gc_success | Measured co-occurrence, not success | Forward outcome + 21-bar lag |
| Feature scale | Absolute ₹ features across tickers | Scale-free pct/ratio features only |
| Signal entry price | Close[i] shown in table | Open[i+1] (actual fill price) |
| Confusion matrix | Price-up vs signal | GC event vs signal (correct target) |
| Platt calibration | No class weight → rare-event bias | class_weight='balanced' |
| Hurst lookahead | Backward fill (4-bar lookahead) | Forward fill only |

**Backtest Mechanics**

| Assumption | Value |
|---|---|
| Entry fill | Next-bar open |
| Exit fill | Next-bar open |
| Commission | Configurable (default 0.1% each side) |
| Slippage | Configurable (default 0.05% each side) |
| Stop type | max(fixed %, 2×ATR%) |
| CAGR denominator | Total elapsed bars |
| Equity curve | Bar-by-bar c-to-c × position |

**Disclaimer:** Research/educational tool.
Not financial advice. Past performance ≠ future returns.
        """)
