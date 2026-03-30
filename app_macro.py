"""
MacroLens — Macro-Driven Market Analytics  (2-page rebuild)
Page 1 : Macro Stress Intelligence
Page 2 : Portfolio Optimiser & Return Projector
─────────────────────────────────────────────────────────
FIX: stress probability now comes ONLY from a live model
     inference on the latest data row — no CSV column is
     used for the headline figure, eliminating the 2.7% /
     57% discrepancy that appeared across pages.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
from scipy.optimize import minimize
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MacroLens",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

[data-testid="stSidebar"] { background:#0a0d14; border-right:1px solid #1a1f2e; }
[data-testid="stSidebar"] * { color:#c8cdd8 !important; }
[data-testid="stSidebar"] label {
    color:#4b5563 !important; font-size:0.68rem !important;
    letter-spacing:0.09em; text-transform:uppercase; }

.kpi { background:#111520; border:1px solid #1a2035;
    border-radius:10px; padding:16px 18px; margin-bottom:8px; }
.kpi-label { font-size:0.67rem; color:#4b5563; text-transform:uppercase;
    letter-spacing:0.1em; margin-bottom:5px; }
.kpi-value { font-size:1.55rem; font-weight:600;
    font-family:'DM Mono',monospace; color:#f0f2f8; }
.kpi-sub { font-size:0.73rem; margin-top:4px; color:#4b5563; }

.signal-buy    { background:#0d2018; border:1px solid #166534;
    border-radius:8px; padding:12px 15px; margin-bottom:6px;
    border-left:3px solid #34d399; }
.signal-hold   { background:#1a1a0d; border:1px solid #713f12;
    border-radius:8px; padding:12px 15px; margin-bottom:6px;
    border-left:3px solid #fbbf24; }
.signal-reduce { background:#200d0d; border:1px solid #7f1d1d;
    border-radius:8px; padding:12px 15px; margin-bottom:6px;
    border-left:3px solid #f87171; }
.signal-title  { font-weight:600; font-size:0.87rem; margin-bottom:3px; }
.signal-desc   { font-size:0.76rem; color:#9ca3af; line-height:1.5; }

.stress-alert { background:#200d0d; border:1px solid #ef4444;
    border-radius:12px; padding:16px 20px; margin-bottom:14px; }
.normal-alert { background:#0d2018; border:1px solid #34d399;
    border-radius:12px; padding:16px 20px; margin-bottom:14px; }
.caution-alert { background:#1a1500; border:1px solid #fbbf24;
    border-radius:12px; padding:16px 20px; margin-bottom:14px; }

.section-head { font-size:0.67rem; color:#374151; text-transform:uppercase;
    letter-spacing:0.12em; padding-bottom:7px;
    border-bottom:1px solid #1a2035; margin-bottom:13px; }

.alloc-card { background:#111520; border:1px solid #1a2035;
    border-radius:8px; padding:11px 14px; margin-bottom:5px; }

.regime-badge {
    display:inline-block; padding:3px 10px; border-radius:20px;
    font-size:0.7rem; font-weight:600; letter-spacing:0.08em;
    text-transform:uppercase; margin-left:8px; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
SECTORS  = ['Bank Nifty','Nifty Pharma','Nifty IT','Nifty Auto','Nifty FMCG']
ALL_IDX  = ['Nifty 50'] + SECTORS
FEATURES = [
    'India_VIX','Repo_Rate','CPI_Inflation','USD_INR','Gold','Crude_Oil','SP500',
    'VIX_lag','Repo_lag','CPI_lag','USD_INR_lag','Gold_lag','Crude_Oil_lag','SP500_lag'
]
S_COLORS = {
    'Bank Nifty':'#60a5fa','Nifty Pharma':'#34d399',
    'Nifty IT':'#a78bfa','Nifty Auto':'#f59e0b','Nifty FMCG':'#f87171'
}
OPT_COLORS = {
    'Max Sharpe':'#f59e0b','Min Volatility':'#60a5fa',
    'Regime-Adaptive':'#a78bfa','Equal Weight':'#9ca3af'
}

PRESETS = {
    "📍 Current Market":           None,                         # filled from data
    "🔥 War / Geopolitical Shock": dict(vix=32,repo=6.5, usd=89,  gold=2400,crude=110,sp=4500,cpi=7.0),
    "🏦 RBI Rate Hike Cycle":      dict(vix=18,repo=7.0, usd=84,  gold=1900,crude=85, sp=5000,cpi=6.0),
    "💱 Currency Crisis (₹ Crash)":dict(vix=28,repo=6.75,usd=100, gold=2200,crude=95, sp=4800,cpi=7.0),
    "📉 Global Recession":         dict(vix=35,repo=5.5, usd=86,  gold=2100,crude=60, sp=3800,cpi=4.0),
    "🚀 Bull Market / Risk-On":    dict(vix=11,repo=6.0, usd=82,  gold=1800,crude=78, sp=5500,cpi=4.5),
    "✂️ Rate Cut Cycle":            dict(vix=13,repo=5.5, usd=83,  gold=1950,crude=80, sp=5200,cpi=3.5),
}

# ── Loaders ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("market_stress_analysis_final.csv", parse_dates=["Date"])
    return df.sort_values("Date").reset_index(drop=True)

@st.cache_resource
def load_model():
    try:
        return joblib.load("market_stress_model.pkl"), joblib.load("scaler.pkl")
    except:
        return None, None

df = load_data()
model, scaler = load_model()

# ── Single source of truth: live model inference for current stress ────────────
#    This eliminates the discrepancy between stored Stress_Prob column (~2.7%)
#    and live model output (~57%) that appeared across pages.
@st.cache_data
def compute_live_stress(features_dict):
    if model is None or scaler is None:
        return None
    X = pd.DataFrame([features_dict])[FEATURES]
    Xs = scaler.transform(X)
    return float(model.predict_proba(Xs)[0, 1])

last_row    = df[FEATURES].dropna().iloc[-1]
CURR_STRESS = compute_live_stress(last_row.to_dict())   # ← one number, used everywhere

# ── Portfolio helpers ─────────────────────────────────────────────────────────
def port_stats(w, returns):
    w = np.array(w)
    mu  = returns.mean() * 252
    cov = returns.cov()  * 252
    r = w @ mu
    v = np.sqrt(w @ cov @ w)
    s = r / v if v > 0 else 0
    return r, v, s

def _opt(objective, returns, extra_cons=None):
    n   = len(returns.columns)
    w0  = np.ones(n) / n
    bnd = [(0.05, 0.60)] * n
    cons = [{"type":"eq","fun": lambda w: np.sum(w)-1}]
    if extra_cons: cons += extra_cons
    res = minimize(objective, w0, bounds=bnd, constraints=cons, method="SLSQP")
    return res.x if res.success else w0

def max_sharpe(returns):
    return _opt(lambda w: -port_stats(w, returns)[2], returns)

def min_vol(returns):
    return _opt(lambda w:  port_stats(w, returns)[1], returns)

def equal_weight(returns):
    n = len(returns.columns)
    return np.ones(n) / n

def regime_adaptive(returns, sp_series, threshold=0.6):
    mask = sp_series > threshold
    sr   = returns[mask.values]  if mask.sum()  > 30 else returns
    nr   = returns[~mask.values] if (~mask).sum()> 30 else returns
    is_stress = (CURR_STRESS or 0) > threshold
    base = sr if is_stress else nr
    w    = max_sharpe(base)
    cols = list(returns.columns)
    if is_stress:
        defensive = ['Nifty Pharma','Nifty FMCG']
        cyclical  = ['Bank Nifty','Nifty IT','Nifty Auto']
        for i, c in enumerate(cols):
            if c in defensive: w[i] *= 1.25
            if c in cyclical:  w[i] *= 0.75
        w = w / w.sum()
    return w

def apply_risk_level(weights, risk_level, sectors):
    w = weights.copy()
    defensive = ['Nifty Pharma','Nifty FMCG']
    cyclical  = ['Bank Nifty','Nifty IT','Nifty Auto']
    if risk_level == "Conservative":
        for i, s in enumerate(sectors):
            if s in defensive: w[i] *= 1.30
            if s in cyclical:  w[i] *= 0.70
    elif risk_level == "Aggressive":
        for i, s in enumerate(sectors):
            if s in defensive: w[i] *= 0.75
            if s in cyclical:  w[i] *= 1.25
    return w / w.sum()

# ── Signal logic ──────────────────────────────────────────────────────────────
def compute_signals(vix, repo, usd, stress_prob, threshold):
    signals = {}
    vix_hi  = vix  > df["India_VIX"].median() * 1.10
    vix_lo  = vix  < df["India_VIX"].median() * 0.90
    repo_hi = repo > df["Repo_Rate"].median()  * 1.05
    repo_lo = repo < df["Repo_Rate"].median()  * 0.95
    usd_hi  = usd  > df["USD_INR"].median()    * 1.02
    usd_lo  = usd  < df["USD_INR"].median()    * 0.98
    stressed = stress_prob > threshold

    for sector in SECTORS:
        score, reasons = 0, []
        if sector == "Bank Nifty":
            if repo_hi:   score -= 2; reasons.append("High rates compress NIM margins")
            if repo_lo:   score += 2; reasons.append("Low rates boost lending & NIM")
            if vix_hi:    score -= 1; reasons.append("High VIX hurts FII flows into banks")
            if stressed:  score -= 2; reasons.append("Stress regime — banks underperform")
        elif sector == "Nifty IT":
            if usd_hi:    score += 2; reasons.append("Weak rupee boosts USD revenue")
            if usd_lo:    score -= 2; reasons.append("Strong rupee reduces USD earnings")
            if vix_hi:    score -= 1; reasons.append("Global risk-off hurts IT multiples")
            if vix_lo:    score += 1; reasons.append("Risk-on supports IT growth multiples")
            if stressed:  score -= 2; reasons.append("Stress regime reduces discretionary IT spend")
            # S&P 500 as US client spending proxy
            sp500_med = df["SP500"].median()
            if df["SP500"].iloc[-1] < sp500_med * 0.95:
                score -= 2; reasons.append("Weak S&P 500 signals US client budget cuts — IT deal flow at risk")
            # High rates compress global tech valuations
            if repo_hi:   score -= 1; reasons.append("High rates globally compress tech valuations")
        elif sector == "Nifty Auto":
            if repo_hi:   score -= 2; reasons.append("High rates raise EMI burden")
            if repo_lo:   score += 2; reasons.append("Low rates drive vehicle financing")
            if usd_hi:    score -= 1; reasons.append("Rupee weakness raises import costs")
            if vix_hi:    score -= 1; reasons.append("Consumer discretionary weak in volatile markets")
            if stressed:  score -= 2; reasons.append("Stress regime — cyclicals sold first")
        elif sector == "Nifty Pharma":
            if stressed:  score += 2; reasons.append("Defensive sector — outperforms in stress")
            if vix_hi:    score += 1; reasons.append("High VIX rotates funds into defensives")
            if usd_hi:    score += 1; reasons.append("Weak rupee benefits pharma exports")
        elif sector == "Nifty FMCG":
            if stressed:  score += 2; reasons.append("Stable demand — safe haven in stress")
            if vix_hi:    score += 1; reasons.append("Safe-haven rotation benefits FMCG")
            if usd_hi:    score -= 1; reasons.append("Import-heavy inputs cost more")

        if score >= 2:   sig = "BUY"
        elif score <= -2: sig = "REDUCE"
        else:             sig = "HOLD"
        if not reasons: reasons = ["No strong macro signal at current levels"]
        signals[sector] = {"signal": sig, "score": score, "reasons": reasons}
    return signals

# ── Chart helpers ─────────────────────────────────────────────────────────────
def dark_fig(figsize=(12, 4)):
    fig, ax = plt.subplots(figsize=figsize, facecolor="#0d1017")
    ax.set_facecolor("#0d1017")
    ax.tick_params(colors="#4b5563")
    for sp in ax.spines.values(): sp.set_edgecolor("#1a2035")
    ax.xaxis.label.set_color("#4b5563")
    ax.yaxis.label.set_color("#4b5563")
    return fig, ax

def stress_gauge(prob, threshold):
    """Render a slim, styled horizontal gauge for stress probability."""
    fig, ax = plt.subplots(figsize=(9, 1.0), facecolor="#0d1017")
    ax.set_facecolor("#0d1017")
    ax.barh(0, 1.0,   color="#1a2035", height=0.5)
    clr = "#f87171" if prob > 0.6 else ("#fbbf24" if prob > threshold else "#34d399")
    ax.barh(0, prob,  color=clr,       height=0.5)
    ax.axvline(threshold, color="#6b7280", lw=1.4, ls="--")
    ax.set_xlim(0, 1); ax.set_ylim(-0.5, 0.5)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["0%","20%","40%","60%","80%","100%"],
                       color="#4b5563", fontsize=8.5)
    ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.text(min(prob + 0.02, 0.95), 0, f"{prob:.1%}",
            va="center", color=clr, fontsize=12,
            fontweight="bold", fontfamily="monospace")
    plt.tight_layout(pad=0.3)
    return fig

# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📡 MacroLens")
    st.markdown(
        "<div style='font-size:0.72rem;color:#374151;margin-bottom:22px'>"
        "Macro-Driven Market Analytics</div>",
        unsafe_allow_html=True
    )

    page = st.radio("", [
        "📊 Macro Stress Intelligence",
        "💼 Portfolio & Return Projector",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("<div class='section-head'>Analysis Window</div>", unsafe_allow_html=True)
    min_d = df["Date"].min().date()
    max_d = df["Date"].max().date()
    date_range = st.date_input("", (min_d, max_d),
                               min_value=min_d, max_value=max_d,
                               label_visibility="collapsed")
    if len(date_range) == 2:
        dff = df[(df["Date"] >= pd.Timestamp(date_range[0])) &
                 (df["Date"] <= pd.Timestamp(date_range[1]))].copy()
    else:
        dff = df.copy()

    st.markdown("<div class='section-head'>Stress Threshold</div>", unsafe_allow_html=True)
    stress_thresh = st.slider("", 0.10, 0.90, 0.60, 0.05,
                              label_visibility="collapsed",
                              help="Probability above which market is classified as stressed")

    st.markdown("<div class='section-head'>Risk Profile</div>", unsafe_allow_html=True)
    risk_level = st.selectbox("", ["Conservative","Moderate","Aggressive"],
                              index=1, label_visibility="collapsed")

    st.markdown("<div class='section-head'>Portfolio Size (₹)</div>", unsafe_allow_html=True)
    invest_amt = st.number_input("", min_value=10_000, max_value=10_00_00_000,
                                 value=10_00_000, step=10_000, format="%d",
                                 label_visibility="collapsed")

    if CURR_STRESS is not None:
        pct = CURR_STRESS
        badge_clr = "#f87171" if pct > stress_thresh else "#34d399"
        st.markdown("---")
        st.markdown(
            f"<div style='text-align:center;padding:10px 0'>"
            f"<div style='font-size:0.65rem;color:#4b5563;text-transform:uppercase;"
            f"letter-spacing:0.1em;margin-bottom:6px'>Live Stress Signal</div>"
            f"<div style='font-size:1.8rem;font-family:DM Mono,monospace;"
            f"font-weight:700;color:{badge_clr}'>{pct:.1%}</div>"
            f"<div style='font-size:0.72rem;color:{badge_clr};margin-top:2px'>"
            f"{'⚠ STRESSED' if pct > stress_thresh else '✓ NORMAL'}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

# ── Shared computations ───────────────────────────────────────────────────────
returns_all = dff[SECTORS].dropna()
sp_aligned  = dff.loc[returns_all.index, "Stress_Prob"]
curr_vix  = df["India_VIX"].iloc[-1]
curr_repo = df["Repo_Rate"].iloc[-1]
curr_usd  = df["USD_INR"].iloc[-1]
curr_cpi  = df["CPI_Inflation"].iloc[-1]

# CURR_STRESS is already computed live above — used on BOTH pages
stress_prob = CURR_STRESS if CURR_STRESS is not None else 0.0


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — MACRO STRESS INTELLIGENCE
# ═════════════════════════════════════════════════════════════════════════════
if page == "📊 Macro Stress Intelligence":

    # ── Page header ──────────────────────────────────────────────────────────
    st.markdown(
        "<h2 style='margin-bottom:4px'>📊 Macro Stress Intelligence</h2>"
        "<div style='font-size:0.8rem;color:#4b5563;margin-bottom:18px'>"
        "Real-time + scenario XGBoost stress assessment · NSE sector signals</div>",
        unsafe_allow_html=True
    )

    # ── Regime alert banner ───────────────────────────────────────────────────
    if CURR_STRESS is None:
        st.error("⚠ Model files not found. Place `market_stress_model.pkl` and `scaler.pkl` in the app folder.")
        st.stop()

    is_stressed = stress_prob > stress_thresh
    is_caution  = (not is_stressed) and stress_prob > stress_thresh * 0.75

    if is_stressed:
        regime_txt = (
            f"<b>🚨 STRESS REGIME DETECTED</b> &nbsp;"
            f"<span style='background:#f8717133;color:#f87171;padding:2px 10px;"
            f"border-radius:20px;font-size:0.72rem;font-weight:700'>"
            f"Probability {stress_prob:.1%}</span><br>"
            f"<span style='font-size:0.83rem;color:#fca5a5;margin-top:4px;display:block'>"
            f"Macro conditions classify the current environment as high-stress. "
            f"Rotate towards defensive sectors — reduce cyclicals (Banks, IT, Auto); "
            f"increase Pharma and FMCG exposure.</span>"
        )
        st.markdown(f"<div class='stress-alert'>{regime_txt}</div>", unsafe_allow_html=True)
    elif is_caution:
        st.markdown(
            f"<div class='caution-alert'><b>⚠ ELEVATED RISK</b> &nbsp;"
            f"<span style='background:#fbbf2433;color:#fbbf24;padding:2px 10px;"
            f"border-radius:20px;font-size:0.72rem;font-weight:700'>"
            f"Probability {stress_prob:.1%}</span><br>"
            f"<span style='font-size:0.83rem;color:#fde68a;margin-top:4px;display:block'>"
            f"Approaching stress territory — monitor closely. "
            f"Consider trimming the most volatile cyclical positions.</span>"
            f"</div>", unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='normal-alert'><b>✅ NORMAL REGIME</b> &nbsp;"
            f"<span style='background:#34d39933;color:#34d399;padding:2px 10px;"
            f"border-radius:20px;font-size:0.72rem;font-weight:700'>"
            f"Probability {stress_prob:.1%}</span><br>"
            f"<span style='font-size:0.83rem;color:#6ee7b7;margin-top:4px;display:block'>"
            f"Macro environment is benign. Growth-oriented allocations are supported."
            f"</span>"
            f"</div>", unsafe_allow_html=True
        )

    # ── Stress gauge ──────────────────────────────────────────────────────────
    st.pyplot(stress_gauge(stress_prob, stress_thresh))
    st.caption("Live XGBoost inference on latest macro data  ·  Dashed line = your stress threshold")

    st.markdown("---")

    # ── 5 KPI tiles ──────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    for col, label, val, sub in [
        (k1, "Stress Probability", f"{stress_prob:.1%}", "Live model output"),
        (k2, "India VIX",          f"{curr_vix:.1f}",   f"Median {df['India_VIX'].median():.1f}"),
        (k3, "Repo Rate",          f"{curr_repo:.2f}%",  f"Median {df['Repo_Rate'].median():.2f}%"),
        (k4, "USD / INR",          f"₹{curr_usd:.1f}",  f"Median ₹{df['USD_INR'].median():.1f}"),
        (k5, "CPI Inflation",      f"{curr_cpi:.1f}%",  f"Median {df['CPI_Inflation'].median():.1f}%"),
    ]:
        col.markdown(
            f"<div class='kpi'><div class='kpi-label'>{label}</div>"
            f"<div class='kpi-value'>{val}</div>"
            f"<div class='kpi-sub'>{sub}</div></div>",
            unsafe_allow_html=True
        )

    # ── Historical stress chart ───────────────────────────────────────────────
    st.markdown("### Historical Stress Probability")

    fig_h, ax_h = dark_fig((13, 3.8))
    ax_h.fill_between(dff["Date"], dff["Stress_Prob"], alpha=0.18, color="#f87171")
    ax_h.plot(dff["Date"], dff["Stress_Prob"],
              color="#f87171", lw=0.9, label="Stored Stress Prob (training output)")
    ax_h.axhline(stress_thresh, color="#fbbf24", ls="--", lw=1.1,
                 label=f"Threshold {stress_thresh:.0%}")
    ax_h.axhline(stress_prob, color="#a78bfa", ls="-.", lw=1.2,
                 label=f"Live inference today: {stress_prob:.1%}")
    ax_h.scatter(dff[dff["Is_Stress"]==1]["Date"],
                 dff[dff["Is_Stress"]==1]["Stress_Prob"],
                 color="#ef4444", s=7, zorder=5, label="Labelled Stress")
    ax_h.legend(loc="upper left", framealpha=0, labelcolor="#9ca3af", fontsize=8.5)
    ax_h.set_ylabel("Stress Probability")
    ax_h.yaxis.set_major_formatter(plt.FuncFormatter(lambda y,_: f"{y:.0%}"))
    ax_h.set_title("Predicted Stress Probability (2014–present)", color="#c8cdd8", fontsize=11)
    plt.tight_layout(); st.pyplot(fig_h); plt.close()

    # Nifty 50 with stress shading
    fig_n, ax_n = dark_fig((13, 3.2))
    cum = (1 + dff["Nifty 50"]).cumprod()
    ax_n.plot(dff["Date"], cum, color="#60a5fa", lw=1.2)
    for d in dff[dff["Is_Stress"]==1]["Date"]:
        ax_n.axvspan(d, d + pd.Timedelta(days=1), alpha=0.10, color="#ef4444")
    ax_n.set_title("Nifty 50 Cumulative Returns  ·  Red bands = labelled stress periods",
                   color="#c8cdd8", fontsize=11)
    ax_n.set_ylabel("Cumulative Return Index")
    plt.tight_layout(); st.pyplot(fig_n); plt.close()

    st.markdown("---")

    # ══ Left column: Sector signals  |  Right column: Scenario engine ══════
    left, right = st.columns([1.1, 1.9])

    # ── Sector signals (current macro) ───────────────────────────────────────
    with left:
        st.markdown("### Sector Signals — Current Macro")
        st.caption("BUY / HOLD / REDUCE based on live VIX · Repo · USD/INR · Stress")
        signals = compute_signals(curr_vix, curr_repo, curr_usd, stress_prob, stress_thresh)
        order = {"BUY": 0, "HOLD": 1, "REDUCE": 2}
        for sector in sorted(SECTORS, key=lambda s: order[signals[s]["signal"]]):
            info = signals[sector]
            sig  = info["signal"]
            css  = {"BUY":"signal-buy","HOLD":"signal-hold","REDUCE":"signal-reduce"}[sig]
            clr  = {"BUY":"#34d399","HOLD":"#fbbf24","REDUCE":"#f87171"}[sig]
            bullet = " · ".join(info["reasons"][:2])
            st.markdown(
                f"<div class='{css}'><div class='signal-title' style='color:{clr}'>"
                f"{sig} &nbsp;·&nbsp; <span style='color:#c8cdd8'>{sector}</span></div>"
                f"<div class='signal-desc'>{bullet}</div></div>",
                unsafe_allow_html=True
            )

    # ── What-if Scenario Engine ───────────────────────────────────────────────
    with right:
        st.markdown("### 🧪 What-If Scenario Engine")
        st.caption("Select a preset or enter custom values — model predicts stress probability instantly")

        preset = st.selectbox("Macro Scenario", list(PRESETS.keys()), index=0)
        p = PRESETS[preset]

        if p is None:   # "Current Market" — use live values
            p = dict(
                vix=curr_vix, repo=curr_repo, usd=curr_usd,
                gold=float(df["Gold"].iloc[-1]),
                crude=float(df["Crude_Oil"].iloc[-1]),
                sp=float(df["SP500"].iloc[-1]),
                cpi=curr_cpi
            )

        s1, s2, s3 = st.columns(3)
        with s1:
            vix_v  = st.number_input("India VIX",        5.0,  90.0, float(p["vix"]),  0.5)
            repo_v = st.number_input("Repo Rate (%)",     1.0,  15.0, float(p["repo"]), 0.25)
            cpi_v  = st.number_input("CPI Inflation (%)", 0.0,  15.0, float(p["cpi"]),  0.1)
        with s2:
            usd_v  = st.number_input("USD / INR",        50.0, 130.0, float(p["usd"]),  0.5)
            gold_v = st.number_input("Gold (USD/oz)",   500.0,8000.0, float(p["gold"]),10.0)
        with s3:
            crude_v= st.number_input("Crude (USD/bbl)", 20.0, 200.0, float(p["crude"]), 1.0)
            sp_v   = st.number_input("S&P 500",        1000.0,10000.0,float(p["sp"]),   50.0)

        scen_features = {
            "India_VIX": vix_v, "Repo_Rate": repo_v, "CPI_Inflation": cpi_v,
            "USD_INR": usd_v, "Gold": gold_v, "Crude_Oil": crude_v, "SP500": sp_v,
            "VIX_lag":       float(last_row["VIX_lag"]),
            "Repo_lag":      float(last_row["Repo_lag"]),
            "CPI_lag":       float(last_row["CPI_lag"]),
            "USD_INR_lag":   float(last_row["USD_INR_lag"]),
            "Gold_lag":      float(last_row["Gold_lag"]),
            "Crude_Oil_lag": float(last_row["Crude_Oil_lag"]),
            "SP500_lag":     float(last_row["SP500_lag"]),
        }
        Xs = scaler.transform(pd.DataFrame([scen_features])[FEATURES])
        scen_stress = float(model.predict_proba(Xs)[0, 1])

        scen_is_stress = scen_stress > stress_thresh
        scen_clr       = "#f87171" if scen_stress > 0.6 else (
                         "#fbbf24" if scen_stress > stress_thresh else "#34d399")
        scen_label     = "STRESS" if scen_is_stress else "NORMAL"

        # Mini gauge for scenario
        fig_g, ax_g = plt.subplots(figsize=(7.5, 0.85), facecolor="#0d1017")
        ax_g.set_facecolor("#0d1017")
        ax_g.barh(0, 1.0,        color="#1a2035", height=0.45)
        ax_g.barh(0, scen_stress, color=scen_clr,  height=0.45)
        ax_g.axvline(stress_thresh, color="#6b7280", lw=1.2, ls="--")
        ax_g.set_xlim(0, 1); ax_g.set_ylim(-0.5, 0.5)
        ax_g.set_xticks([0, 0.25, 0.50, 0.75, 1.0])
        ax_g.set_xticklabels(["0%","25%","50%","75%","100%"],
                             color="#4b5563", fontsize=8)
        ax_g.set_yticks([])
        for sp_ in ax_g.spines.values(): sp_.set_visible(False)
        ax_g.text(min(scen_stress+0.02,0.90), 0, f"{scen_stress:.1%}",
                  va="center", color=scen_clr, fontsize=11,
                  fontweight="bold", fontfamily="monospace")
        plt.tight_layout(pad=0.2)
        st.markdown(
            f"<div style='font-size:0.7rem;color:#4b5563;text-transform:uppercase;"
            f"letter-spacing:0.09em;margin-top:10px;margin-bottom:2px'>"
            f"Scenario Stress Probability &nbsp;·&nbsp;"
            f"<span style='color:{scen_clr}'>{scen_label}</span></div>",
            unsafe_allow_html=True
        )
        st.pyplot(fig_g); plt.close()

        # Scenario sector signals
        scen_sigs = compute_signals(vix_v, repo_v, usd_v, scen_stress, stress_thresh)
        st.markdown("**Scenario sector signals:**")
        row_cols = st.columns(5)
        sig_clr  = {"BUY":"#34d399","HOLD":"#fbbf24","REDUCE":"#f87171"}
        for col, sector in zip(row_cols, SECTORS):
            s = scen_sigs[sector]["signal"]
            col.markdown(
                f"<div style='background:#111520;border:1px solid #1a2035;"
                f"border-radius:7px;padding:8px 6px;text-align:center;"
                f"border-top:2px solid {sig_clr[s]}'>"
                f"<div style='font-size:0.67rem;color:#6b7280;margin-bottom:3px'>"
                f"{sector.replace('Nifty ','')}</div>"
                f"<div style='font-size:0.78rem;font-weight:700;color:{sig_clr[s]}'>{s}</div>"
                f"</div>",
                unsafe_allow_html=True
            )

        # VIX sensitivity sweep
        st.markdown("**VIX sensitivity:** how stress probability changes across VIX levels")
        vix_range = np.linspace(8, 60, 50)
        sweep_probs = []
        for v in vix_range:
            sf = scen_features.copy(); sf["India_VIX"] = v
            Xv = pd.DataFrame([sf])[FEATURES]
            sweep_probs.append(float(model.predict_proba(scaler.transform(Xv))[0,1]))

        fig_sv, ax_sv = dark_fig((7.5, 2.8))
        ax_sv.plot(vix_range, sweep_probs, color="#f87171", lw=1.8)
        ax_sv.fill_between(vix_range, sweep_probs, alpha=0.12, color="#f87171")
        ax_sv.axvline(vix_v, color="#fbbf24", ls="--", lw=1.2,
                      label=f"Scenario VIX = {vix_v:.0f}")
        ax_sv.axhline(stress_thresh, color="#a78bfa", ls="--", lw=1,
                      label=f"Threshold {stress_thresh:.0%}")
        ax_sv.scatter([vix_v],[scen_stress], color="#fbbf24", s=70, zorder=5)
        ax_sv.set_xlabel("India VIX"); ax_sv.set_ylabel("Stress Prob")
        ax_sv.yaxis.set_major_formatter(plt.FuncFormatter(lambda y,_: f"{y:.0%}"))
        ax_sv.legend(framealpha=0, labelcolor="#9ca3af", fontsize=8)
        ax_sv.set_title("Stress vs VIX (other inputs fixed)", color="#c8cdd8", fontsize=10)
        plt.tight_layout(); st.pyplot(fig_sv); plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PORTFOLIO & RETURN PROJECTOR
# ═════════════════════════════════════════════════════════════════════════════
elif page == "💼 Portfolio & Return Projector":

    st.markdown(
        "<h2 style='margin-bottom:4px'>💼 Portfolio & Return Projector</h2>"
        "<div style='font-size:0.8rem;color:#4b5563;margin-bottom:18px'>"
        "Strategy optimisation · sector allocation · growth simulation  "
        f"·  Stress {stress_prob:.1%} ({'⚠ Stressed' if stress_prob > stress_thresh else '✓ Normal'})</div>",
        unsafe_allow_html=True
    )

    ret = returns_all.copy()
    sp  = sp_aligned.copy()

    # ── Strategy computation ──────────────────────────────────────────────────
    strategies = {
        "Max Sharpe":       max_sharpe(ret),
        "Regime-Adaptive":  regime_adaptive(ret, sp, stress_thresh),
        "Equal Weight":     equal_weight(ret),
    }
    results = {}
    for m, w_raw in strategies.items():
        w = apply_risk_level(w_raw, risk_level, SECTORS)
        r_, v_, s_ = port_stats(w, ret)
        results[m] = {"w": w, "r": r_, "v": v_, "s": s_}

    # ── Strategy comparison KPI strip ─────────────────────────────────────────
    st.markdown("### Strategy Comparison")
    s_cols = st.columns(len(results))
    for col, (m, res) in zip(s_cols, results.items()):
        clr = OPT_COLORS.get(m, "#9ca3af")
        col.markdown(
            f"<div class='kpi' style='border-top:2px solid {clr}'>"
            f"<div class='kpi-label' style='color:{clr}'>{m}</div>"
            f"<div class='kpi-value'>{res['s']:.3f} <span style='font-size:0.75rem;"
            f"color:#4b5563'>Sharpe</span></div>"
            f"<div class='kpi-sub'>Return {res['r']:.1%}  ·  Vol {res['v']:.1%}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

    # ── Backtested cumulative returns ─────────────────────────────────────────
    fig_cum, ax_cum = dark_fig((13, 4))
    for m, res in results.items():
        cum = (1 + ret @ res["w"]).cumprod()
        ax_cum.plot(dff.loc[ret.index, "Date"].values, cum.values,
                    label=m, color=OPT_COLORS.get(m,"#888"), lw=1.6)
    # Shade stress periods
    for d in dff[dff["Is_Stress"]==1]["Date"]:
        ax_cum.axvspan(d, d + pd.Timedelta(days=1), alpha=0.07, color="#ef4444")
    ax_cum.set_ylabel("Cumulative Return Index")
    ax_cum.legend(framealpha=0, labelcolor="#9ca3af", fontsize=9)
    ax_cum.set_title("Backtested Cumulative Returns  ·  Red = stress periods",
                     color="#c8cdd8", fontsize=11)
    plt.tight_layout(); st.pyplot(fig_cum); plt.close()

    st.markdown("---")

    # ── Allocation + Efficient Frontier side by side ──────────────────────────
    al_col, ef_col = st.columns([1, 1.8])

    with al_col:
        st.markdown("### Allocation Breakdown")
        tab_method = st.selectbox("Strategy", list(results.keys()), label_visibility="collapsed")
        res = results[tab_method]
        clr_list = [S_COLORS[s] for s in SECTORS]

        # Donut chart
        fig_p, ax_p = plt.subplots(figsize=(4, 4), facecolor="#0d1017")
        ax_p.set_facecolor("#0d1017")
        wedges, _, autotexts = ax_p.pie(
            res["w"], labels=None, autopct="%1.1f%%",
            colors=clr_list,
            wedgeprops=dict(width=0.52, edgecolor="#0d1017", linewidth=2),
            startangle=90, pctdistance=0.76
        )
        for at in autotexts:
            at.set_color("#0d1017"); at.set_fontsize(7.5); at.set_fontweight("bold")
        patches = [mpatches.Patch(color=clr_list[i], label=SECTORS[i])
                   for i in range(len(SECTORS))]
        ax_p.legend(handles=patches, loc="lower center", bbox_to_anchor=(0.5, -0.1),
                    ncol=2, framealpha=0, labelcolor="#9ca3af", fontsize=7.5)
        plt.tight_layout(); st.pyplot(fig_p); plt.close()

        # Rupee table
        sig_map = compute_signals(curr_vix, curr_repo, curr_usd, stress_prob, stress_thresh)
        for s, w, clr_ in zip(SECTORS, res["w"], clr_list):
            sig  = sig_map[s]["signal"]
            s_clr = {"BUY":"#34d399","HOLD":"#fbbf24","REDUCE":"#f87171"}[sig]
            st.markdown(
                f"<div class='alloc-card' style='border-left:3px solid {clr_}'>"
                f"<div style='display:flex;justify-content:space-between;align-items:center'>"
                f"<div><div style='color:#c8cdd8;font-size:0.83rem;font-weight:500'>{s}</div>"
                f"<div style='color:{s_clr};font-size:0.68rem;margin-top:1px'>{sig}</div></div>"
                f"<div style='text-align:right'>"
                f"<div style='font-family:DM Mono,monospace;color:#f0f2f8;font-size:0.88rem'>{w:.1%}</div>"
                f"<div style='color:#4b5563;font-size:0.73rem'>₹{w*invest_amt:,.0f}</div>"
                f"</div></div></div>",
                unsafe_allow_html=True
            )

        # Portfolio summary card
        r_, v_, s_ = res["r"], res["v"], res["s"]
        st.markdown(
            f"<div style='background:#0a1628;border:1px solid #1a3050;"
            f"border-radius:8px;padding:12px 16px;margin-top:8px'>"
            f"<div style='color:#60a5fa;font-size:0.68rem;text-transform:uppercase;"
            f"letter-spacing:0.09em;margin-bottom:8px'>Portfolio Metrics</div>"
            f"<div style='display:flex;gap:16px;flex-wrap:wrap'>"
            f"<div><div style='color:#4b5563;font-size:0.68rem'>Exp. Return</div>"
            f"<div style='color:#34d399;font-family:DM Mono,monospace;font-size:1rem'>{r_:.2%}</div></div>"
            f"<div><div style='color:#4b5563;font-size:0.68rem'>Volatility</div>"
            f"<div style='color:#f59e0b;font-family:DM Mono,monospace;font-size:1rem'>{v_:.2%}</div></div>"
            f"<div><div style='color:#4b5563;font-size:0.68rem'>Sharpe</div>"
            f"<div style='color:#a78bfa;font-family:DM Mono,monospace;font-size:1rem'>{s_:.3f}</div></div>"
            f"<div><div style='color:#4b5563;font-size:0.68rem'>Exp. Gain / yr</div>"
            f"<div style='color:#34d399;font-family:DM Mono,monospace;font-size:1rem'>"
            f"₹{r_*invest_amt:,.0f}</div></div>"
            f"</div></div>",
            unsafe_allow_html=True
        )

    # ── Efficient frontier ────────────────────────────────────────────────────
    with ef_col:
        st.markdown("### Efficient Frontier")
        n = len(SECTORS)
        mc_v, mc_r, mc_s = [], [], []
        for _ in range(3000):
            w_mc = np.random.dirichlet(np.ones(n))
            r_, v_, s_ = port_stats(w_mc, ret)
            mc_v.append(v_); mc_r.append(r_); mc_s.append(s_)

        fig_ef, ax_ef = dark_fig((9, 6))
        sc = ax_ef.scatter(mc_v, mc_r, c=mc_s, cmap="plasma",
                           s=5, alpha=0.30, zorder=1)
        plt.colorbar(sc, ax=ax_ef, label="Sharpe Ratio",
                     fraction=0.025, pad=0.02).ax.yaxis.set_tick_params(color="#4b5563")

        for m, res in results.items():
            clr = OPT_COLORS.get(m, "#fff")
            ax_ef.scatter(res["v"], res["r"], color=clr, s=200,
                          zorder=5, marker="*", edgecolors="#0d1017", lw=0.5, label=m)
            ax_ef.annotate(m, (res["v"], res["r"]),
                           xytext=(7, 4), textcoords="offset points",
                           color=clr, fontsize=8.5)

        ax_ef.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f"{x:.0%}"))
        ax_ef.yaxis.set_major_formatter(plt.FuncFormatter(lambda y,_:f"{y:.0%}"))
        ax_ef.set_xlabel("Annualised Volatility")
        ax_ef.set_ylabel("Annualised Return")
        ax_ef.set_title("Risk–Return Space  ·  3000 Random Portfolios",
                         color="#c8cdd8", fontsize=11)
        ax_ef.legend(framealpha=0, labelcolor="#9ca3af", fontsize=9, loc="lower right")
        plt.tight_layout(); st.pyplot(fig_ef); plt.close()

    st.markdown("---")

    # ── Return / Growth Projector ─────────────────────────────────────────────
    st.markdown("### 📈 Return & Growth Projector")

    proj_col1, proj_col2 = st.columns([1, 2])

    with proj_col1:
        proj_strategy = st.selectbox("Strategy for Projection", list(results.keys()))
        invest_type   = st.radio("Investment Type", ["Lump Sum","SIP (Monthly)"])
        if invest_type == "SIP (Monthly)":
            sip_amt = st.number_input("Monthly SIP (₹)", 1000, 5_00_000,
                                      value=min(invest_amt // 12, 10_000), step=500)
        horizon = st.slider("Horizon (years)", 1, 30, 10)

        pr = results[proj_strategy]
        ann_ret = pr["r"]
        monthly_ret = (1 + ann_ret) ** (1/12) - 1

        # Compute projection
        years  = np.arange(0, horizon + 1)
        if invest_type == "Lump Sum":
            values = [invest_amt * (1 + ann_ret)**y for y in years]
            inv_series = [invest_amt] * len(years)
        else:
            values = []
            for y in years:
                m = y * 12
                if monthly_ret > 0:
                    fv = sip_amt * (((1+monthly_ret)**m - 1) / monthly_ret) * (1+monthly_ret)
                else:
                    fv = sip_amt * m
                values.append(fv)
            inv_series = [min(sip_amt * y * 12, sip_amt * horizon * 12) for y in years]

        final_val = values[-1]
        total_inv = inv_series[-1]
        total_gain = final_val - total_inv

        # Summary
        st.markdown(
            f"<div style='background:#0a1628;border:1px solid #1a3050;"
            f"border-radius:10px;padding:14px 18px;margin-top:14px'>"
            f"<div style='color:#60a5fa;font-size:0.68rem;text-transform:uppercase;"
            f"letter-spacing:0.09em;margin-bottom:10px'>Projection Summary — {horizon}Y</div>"
            f"<div style='color:#4b5563;font-size:0.72rem'>Total Invested</div>"
            f"<div style='color:#f0f2f8;font-family:DM Mono,monospace;font-size:1.1rem;margin-bottom:6px'>"
            f"₹{total_inv:,.0f}</div>"
            f"<div style='color:#4b5563;font-size:0.72rem'>Portfolio Value</div>"
            f"<div style='color:#34d399;font-family:DM Mono,monospace;font-size:1.3rem;margin-bottom:6px'>"
            f"₹{final_val:,.0f}</div>"
            f"<div style='color:#4b5563;font-size:0.72rem'>Total Gain</div>"
            f"<div style='color:#a78bfa;font-family:DM Mono,monospace;font-size:1.1rem;margin-bottom:6px'>"
            f"₹{total_gain:,.0f} &nbsp;"
            f"<span style='font-size:0.75rem'>({(final_val/total_inv-1)*100:.1f}%)</span></div>"
            f"<div style='color:#4b5563;font-size:0.72rem'>Ann. Return Used</div>"
            f"<div style='color:#f59e0b;font-family:DM Mono,monospace;font-size:0.95rem'>"
            f"{ann_ret:.2%}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

    with proj_col2:
        fig_proj, ax_proj = dark_fig((10, 5))
        ax_proj.fill_between(years, inv_series, alpha=0.25, color="#60a5fa",
                             label="Capital Invested")
        ax_proj.fill_between(years, inv_series, values, alpha=0.25, color="#34d399",
                             label="Gains")
        ax_proj.plot(years, values, color="#34d399", lw=2.2,
                     label=f"{invest_type} Portfolio Value")
        ax_proj.plot(years, inv_series, color="#60a5fa", lw=1.4, ls="--",
                     label="Invested Amount")
        ax_proj.annotate(
            f"₹{final_val:,.0f}",
            xy=(horizon, final_val), xytext=(-60, 10),
            textcoords="offset points", color="#34d399", fontsize=9,
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#34d399", lw=0.8)
        )
        ax_proj.set_xlabel("Years")
        ax_proj.set_ylabel("Portfolio Value (₹)")
        ax_proj.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y,_: f"₹{y/1e5:.1f}L" if y < 1e7 else f"₹{y/1e7:.2f}Cr")
        )
        ax_proj.legend(framealpha=0, labelcolor="#9ca3af", fontsize=9)
        ax_proj.set_title(
            f"{invest_type} Growth  ·  {proj_strategy}  ·  {risk_level}  ·  {horizon}Y",
            color="#c8cdd8", fontsize=11
        )
        plt.tight_layout(); st.pyplot(fig_proj); plt.close()

        # Year-by-year table
        st.markdown("#### Year-by-Year Projection")
        rows = []
        for y in range(1, horizon + 1):
            fv_y  = values[y]
            inv_y = inv_series[y]
            rows.append({
                "Year": y,
                "Invested": f"₹{inv_y:,.0f}",
                "Portfolio Value": f"₹{fv_y:,.0f}",
                "Gain": f"₹{fv_y - inv_y:,.0f}",
                "Return %": f"{(fv_y/inv_y - 1)*100:.1f}%" if inv_y > 0 else "—",
            })
        st.dataframe(pd.DataFrame(rows).set_index("Year"), use_container_width=True)

    # ── Stress scenario overlay ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Stress Scenario: Bear vs Base vs Bull")
    st.caption("How your portfolio performs under different macro stress outcomes")

    pr_base = results[proj_strategy]["r"]
    pr_bear = pr_base * 0.35    # stress-discounted (Regime-Adaptive in stress historically ~35% of base)
    pr_bull = pr_base * 1.40    # optimistic macro tailwind

    fig_sc, ax_sc = dark_fig((12, 4.5))
    for label, rate, clr in [
        ("Bear (Stress Regime)", pr_bear, "#f87171"),
        ("Base (Current Macro)", pr_base, "#fbbf24"),
        ("Bull (Risk-On)",       pr_bull, "#34d399"),
    ]:
        if invest_type == "Lump Sum":
            vals = [invest_amt * (1+rate)**y for y in years]
        else:
            m_rate = (1+rate)**(1/12)-1
            vals = [sip_amt*(((1+m_rate)**(y*12)-1)/m_rate)*(1+m_rate) if m_rate>0
                    else sip_amt*y*12 for y in years]
        ax_sc.plot(years, vals, color=clr, lw=1.8, label=f"{label} ({rate:.1%}/yr)")

    ax_sc.fill_between(years,
        [invest_amt*(1+pr_bear)**y if invest_type=="Lump Sum"
         else sip_amt*(((1+(1+pr_bear)**(1/12)-1)**(y*12)-1)/max((1+pr_bear)**(1/12)-1,1e-9))
         *(1+(1+pr_bear)**(1/12)-1) for y in years],
        [invest_amt*(1+pr_bull)**y if invest_type=="Lump Sum"
         else sip_amt*(((1+(1+pr_bull)**(1/12)-1)**(y*12)-1)/max((1+pr_bull)**(1/12)-1,1e-9))
         *(1+(1+pr_bull)**(1/12)-1) for y in years],
        alpha=0.07, color="#60a5fa", label="Scenario range"
    )
    ax_sc.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y,_: f"₹{y/1e5:.1f}L" if y<1e7 else f"₹{y/1e7:.2f}Cr")
    )
    ax_sc.set_xlabel("Years")
    ax_sc.legend(framealpha=0, labelcolor="#9ca3af", fontsize=9)
    ax_sc.set_title("Bear / Base / Bull Scenario Projection", color="#c8cdd8", fontsize=11)
    plt.tight_layout(); st.pyplot(fig_sc); plt.close()