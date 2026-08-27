import os
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from scipy.signal import savgol_filter, find_peaks
from ultralytics import YOLO


st.set_page_config(
    page_title="KOSI | River Intelligence Platform",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# STYLE — Bioluminescent Deep-Water Terminal
# ============================================================

st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

    :root {
        --teal:    #00f5d4;
        --violet:  #7c3aed;
        --indigo:  #4f46e5;
        --sky:     #38bdf8;
        --amber:   #f59e0b;
        --rose:    #f43f5e;
        --green:   #22d3a0;
        --surface: rgba(255,255,255,0.036);
        --border:  rgba(255,255,255,0.075);
        --text-1:  #e8f3ff;
        --text-2:  #8ca8c5;
        --text-3:  #56738f;
    }

    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
        color: var(--text-1);
    }

    /* ── APP BACKGROUND ── */
    .stApp {
        background:
            radial-gradient(ellipse 80% 60% at 50% -20%, rgba(0,245,212,0.07) 0%, transparent 60%),
            radial-gradient(ellipse 60% 40% at 80% 80%, rgba(124,58,237,0.09) 0%, transparent 55%),
            radial-gradient(ellipse 40% 60% at 10% 90%, rgba(56,189,248,0.06) 0%, transparent 50%),
            linear-gradient(175deg, #030c18 0%, #040e1c 40%, #050f1e 100%);
        min-height: 100vh;
    }

    /* ── SIDEBAR ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #020b16 0%, #030d1a 100%) !important;
        border-right: 1px solid rgba(0,245,212,0.1);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 2rem;
    }

    [data-testid="stSidebar"] * {
        color: var(--text-1) !important;
    }

    /* ── INPUTS ── */
    .stSelectbox > div > div,
    .stRadio > div,
    .stNumberInput > div {
        background: rgba(0,245,212,0.04) !important;
        border: 1px solid rgba(0,245,212,0.15) !important;
        border-radius: 10px !important;
        color: var(--text-1) !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, rgba(0,245,212,0.15), rgba(124,58,237,0.20)) !important;
        border: 1px solid rgba(0,245,212,0.35) !important;
        color: var(--teal) !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 13px !important;
        font-weight: 700 !important;
        letter-spacing: 1.5px !important;
        border-radius: 12px !important;
        padding: 14px 28px !important;
        transition: all 0.25s ease !important;
        box-shadow: 0 0 20px rgba(0,245,212,0.08) !important;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, rgba(0,245,212,0.28), rgba(124,58,237,0.32)) !important;
        box-shadow: 0 0 35px rgba(0,245,212,0.20), 0 0 70px rgba(124,58,237,0.12) !important;
        transform: translateY(-2px) !important;
    }

    /* ── FILE UPLOADER ── */
    [data-testid="stFileUploader"] {
        background: rgba(0,245,212,0.03) !important;
        border: 1.5px dashed rgba(0,245,212,0.22) !important;
        border-radius: 16px !important;
        transition: border-color 0.2s;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: rgba(0,245,212,0.5) !important;
    }

    /* ── DATAFRAME ── */
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(0,245,212,0.12) !important;
        border-radius: 14px !important;
        overflow: hidden !important;
    }

    /* ── HERO ── */
    .pravaah-hero {
        position: relative;
        overflow: hidden;
        padding: 44px 44px 38px;
        border-radius: 28px;
        background:
            linear-gradient(135deg,
            rgba(0,245,212,0.07) 0%,
            rgba(124,58,237,0.10) 60%,
            rgba(56,189,248,0.06) 100%);
        border: 1px solid rgba(0,245,212,0.13);
        box-shadow:
            0 0 0 1px rgba(255,255,255,0.04),
            0 24px 80px rgba(0,0,0,0.38),
            inset 0 1px 0 rgba(255,255,255,0.08);
        margin-bottom: 32px;
    }

    .pravaah-hero::before {
        content: "";
        position: absolute;
        width: 320px; height: 320px;
        border-radius: 50%;
        right: -100px; top: -130px;
        background: radial-gradient(circle, rgba(0,245,212,0.12) 0%, transparent 70%);
        animation: orb-drift 8s ease-in-out infinite alternate;
    }

    .pravaah-hero::after {
        content: "";
        position: absolute;
        width: 180px; height: 180px;
        border-radius: 50%;
        left: 30%; bottom: -80px;
        background: radial-gradient(circle, rgba(124,58,237,0.10) 0%, transparent 70%);
        animation: orb-drift 11s ease-in-out infinite alternate-reverse;
    }

    @keyframes orb-drift {
        from { transform: translate(0, 0) scale(1); }
        to   { transform: translate(20px, -15px) scale(1.08); }
    }

    .hero-eyebrow {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        color: var(--teal);
        letter-spacing: 3px;
        margin-bottom: 12px;
        opacity: 0.8;
    }

    .hero-wordmark {
        font-size: 64px;
        font-weight: 700;
        letter-spacing: -3px;
        line-height: 1;
        background: linear-gradient(120deg, #00f5d4 0%, #38bdf8 45%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 14px;
        position: relative;
        z-index: 1;
    }

    .hero-tagline {
        color: var(--text-2);
        font-size: 15px;
        line-height: 1.75;
        max-width: 680px;
        position: relative;
        z-index: 1;
    }

    .hero-pills {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 24px;
        position: relative;
        z-index: 1;
    }

    .hero-pill {
        display: flex;
        align-items: center;
        gap: 8px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        color: var(--text-2);
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 999px;
        padding: 6px 14px;
        letter-spacing: 0.5px;
    }

    .live-dot {
        width: 7px; height: 7px;
        border-radius: 50%;
        background: #22d3a0;
        box-shadow: 0 0 8px #22d3a0, 0 0 16px #22d3a0;
        animation: live-blink 2s ease-in-out infinite;
        flex-shrink: 0;
    }

    @keyframes live-blink {
        0%, 100% { opacity: 1; }
        50%       { opacity: 0.2; }
    }

    /* ── METRIC CARD ── */
    .mc {
        padding: 22px 24px;
        border-radius: 20px;
        background: var(--surface);
        border: 1px solid var(--border);
        box-shadow: 0 8px 32px rgba(0,0,0,0.22);
        position: relative;
        overflow: hidden;
        transition: transform 0.2s, box-shadow 0.2s;
    }

    .mc:hover {
        transform: translateY(-3px);
        box-shadow: 0 16px 48px rgba(0,0,0,0.32), 0 0 0 1px rgba(0,245,212,0.12);
    }

    .mc::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, var(--teal), var(--violet));
        opacity: 0.6;
    }

    .mc-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        letter-spacing: 2px;
        color: var(--text-3);
        text-transform: uppercase;
        margin-bottom: 10px;
    }

    .mc-val {
        font-size: 32px;
        font-weight: 700;
        letter-spacing: -1px;
        background: linear-gradient(135deg, #e8f3ff, #a0cfe8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .mc-sub {
        font-size: 11px;
        color: var(--text-3);
        margin-top: 8px;
        line-height: 1.5;
    }

    /* ── SECTION HEADING ── */
    .sec-head {
        display: flex;
        align-items: baseline;
        gap: 14px;
        margin: 32px 0 6px;
    }

    .sec-title {
        font-size: 22px;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: var(--text-1);
    }

    .sec-num {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        color: var(--teal);
        letter-spacing: 2px;
        opacity: 0.7;
    }

    .sec-sub {
        font-size: 13px;
        color: var(--text-3);
        margin-bottom: 20px;
        line-height: 1.6;
    }

    /* ── GLASS CARD ── */
    .gc {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 12px 40px rgba(0,0,0,0.16);
        margin-bottom: 16px;
    }

    /* ── PIPELINE FLOW ── */
    .pipe-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 12px;
        margin: 20px 0;
    }

    .pipe-node {
        padding: 20px 12px;
        text-align: center;
        border-radius: 18px;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        position: relative;
        transition: all 0.2s;
        cursor: default;
    }

    .pipe-node:hover {
        background: rgba(0,245,212,0.06);
        border-color: rgba(0,245,212,0.2);
        transform: translateY(-4px);
        box-shadow: 0 12px 30px rgba(0,0,0,0.2), 0 0 20px rgba(0,245,212,0.07);
    }

    .pipe-icon {
        font-size: 30px;
        margin-bottom: 8px;
        display: block;
    }

    .pipe-name {
        font-size: 12px;
        font-weight: 600;
        color: var(--text-1);
        letter-spacing: 0.3px;
    }

    .pipe-desc {
        font-size: 10px;
        color: var(--text-3);
        margin-top: 5px;
        font-family: 'JetBrains Mono', monospace;
    }

    .pipe-arr {
        position: absolute;
        right: -14px;
        top: 50%;
        transform: translateY(-50%);
        color: var(--teal);
        font-size: 14px;
        opacity: 0.5;
        z-index: 1;
    }

    /* ── BADGE ── */
    .bdg {
        display: inline-block;
        padding: 5px 11px;
        border-radius: 999px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        font-weight: 500;
        letter-spacing: 1px;
        margin-bottom: 10px;
    }

    .bdg-teal   { color: #00f5d4; background: rgba(0,245,212,0.10); border: 1px solid rgba(0,245,212,0.22); }
    .bdg-violet { color: #c4b5fd; background: rgba(124,58,237,0.12); border: 1px solid rgba(124,58,237,0.26); }
    .bdg-amber  { color: #fcd34d; background: rgba(245,158,11,0.10); border: 1px solid rgba(245,158,11,0.22); }
    .bdg-sky    { color: #7dd3fc; background: rgba(56,189,248,0.10); border: 1px solid rgba(56,189,248,0.22); }
    .bdg-rose   { color: #fda4af; background: rgba(244,63,94,0.10); border: 1px solid rgba(244,63,94,0.22); }

    /* ── SIDEBAR NAV ── */
    .sidebar-brand {
        padding: 0 16px 24px;
        border-bottom: 1px solid rgba(0,245,212,0.10);
        margin-bottom: 20px;
    }

    .brand-mark {
        font-size: 26px;
        font-weight: 700;
        letter-spacing: -0.8px;
        background: linear-gradient(120deg, #00f5d4, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .brand-sub {
        font-family: 'JetBrains Mono', monospace;
        font-size: 9px;
        color: var(--text-3);
        letter-spacing: 2.5px;
        margin-top: 3px;
    }

    .module-status {
        padding: 14px 16px;
        background: rgba(0,245,212,0.03);
        border: 1px solid rgba(0,245,212,0.09);
        border-radius: 12px;
        margin-top: 20px;
    }

    .mod-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 9px;
        letter-spacing: 2px;
        color: var(--teal);
        margin-bottom: 10px;
        opacity: 0.8;
    }

    .mod-item {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 11px;
        color: var(--text-2);
        margin: 7px 0;
    }

    .mod-dot {
        width: 6px; height: 6px;
        border-radius: 50%;
        background: #22d3a0;
        box-shadow: 0 0 6px #22d3a0;
        animation: live-blink 2s ease-in-out infinite;
        flex-shrink: 0;
    }

    /* ── RESULT CALLOUT ── */
    .result-callout {
        padding: 28px;
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(0,245,212,0.06), rgba(124,58,237,0.08));
        border: 1px solid rgba(0,245,212,0.18);
        box-shadow: 0 0 40px rgba(0,245,212,0.05);
        text-align: center;
    }

    .result-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        letter-spacing: 2.5px;
        color: var(--text-3);
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .result-big {
        font-size: 42px;
        font-weight: 700;
        letter-spacing: -1.5px;
        background: linear-gradient(135deg, #00f5d4, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .result-note {
        font-size: 12px;
        color: var(--text-3);
        margin-top: 8px;
    }

    /* ── DIVIDER ── */
    .teal-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(0,245,212,0.25), transparent);
        margin: 30px 0;
    }

    /* ── FOOTER ── */
    .pravaah-footer {
        text-align: center;
        padding: 32px;
        margin-top: 60px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        letter-spacing: 2px;
        color: var(--text-3);
        border-top: 1px solid rgba(255,255,255,0.05);
    }

    /* scrollbar */
    ::-webkit-scrollbar { width: 6px; background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(0,245,212,0.15); border-radius: 4px; }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).parent.parent  # KOSI/

YOLO_PATH          = BASE_DIR / "models" / "yolo_best.pt"
RAMAN_MODEL_PATH   = BASE_DIR / "models" / "raman_svm.pkl"
RIVER_MODEL_PATH   = BASE_DIR / "models" / "river_rf.pkl"
DIGITAL_MODEL_PATH = BASE_DIR / "models" / "digital_twin_rf.pkl"
RAMAN_FOLDER       = BASE_DIR / "dataset" / "A Raman database of microplastics weathered under natural environments"

# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():
    yolo    = YOLO(YOLO_PATH)
    raman   = joblib.load(RAMAN_MODEL_PATH)
    river   = joblib.load(RIVER_MODEL_PATH)
    digital = joblib.load(DIGITAL_MODEL_PATH)
    return yolo, raman, river, digital


@st.cache_data
def load_river_data():
    ganga  = pd.read_csv("dataset/river_dataset/ganga.csv")
    sangam = pd.read_csv("dataset/river_dataset/sangam.csv")
    ganga["Date"]  = pd.to_datetime(ganga["Date"])
    sangam["Date"] = pd.to_datetime(sangam["Date"])
    return ganga, sangam


@st.cache_data
def build_raman_grid():
    metadata = pd.read_excel(RAMAN_FOLDER / "content.xlsx", header=1)
    metadata = metadata[metadata["ID"].astype(str).str.strip() != "ID"]
    spectra = []
    for _, row in metadata.iterrows():
        sample  = str(row["ID"]).strip()
        polymer = str(row["type"]).strip()
        if polymer == "/":
            continue
        file = RAMAN_FOLDER / f"{sample}.txt"
        if not file.exists():
            continue
        data = pd.read_csv(file, sep=r"\s+", header=None, names=["shift", "intensity"]).dropna()
        spectra.append(data)
    min_shift = max(item["shift"].min() for item in spectra)
    max_shift = min(item["shift"].max() for item in spectra)
    return np.linspace(min_shift, max_shift, 1400)


yolo_model, raman_model, river_model, digital_model = load_models()
ganga, sangam = load_river_data()
raman_grid    = build_raman_grid()


# ============================================================
# HELPERS
# ============================================================

def metric_card(label, value, info, accent="#00f5d4"):
    st.markdown(
        f"""
        <div class="mc">
            <div class="mc-label">{label}</div>
            <div class="mc-val">{value}</div>
            <div class="mc-sub">{info}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def section_head(title, num="", sub=""):
    num_html = f'<span class="sec-num">{num}</span>' if num else ""
    sub_html = f'<p class="sec-sub">{sub}</p>' if sub else ""
    st.markdown(
        f"""
        <div class="sec-head">
            <span class="sec-title">{title}</span>
            {num_html}
        </div>
        {sub_html}
        """,
        unsafe_allow_html=True
    )


def morphology(ratio):
    if 1 < ratio <= 1.3:
        return "Pellet",      (255,  80,  80)
    if 1.3 < ratio <= 4:
        return "Fragment",    ( 80, 220, 160)
    if 0.8 < ratio <= 1:
        return "Filament",    ( 80, 150, 255)
    return "Unclassified", (160, 160, 160)


def process_raman(data):
    x = data["shift"].values
    y = data["intensity"].values
    y = np.interp(raman_grid, x, y)
    baseline = savgol_filter(y, 101, 3)
    y = y - baseline
    y = savgol_filter(y, 11, 3)
    norm = np.linalg.norm(y)
    if norm != 0:
        y = y / norm
    return y


def plotly_theme(fig, h=480):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.015)",
        font=dict(color="#8ca8c5", family="Space Grotesk"),
        margin=dict(l=24, r=24, t=50, b=28),
        height=h,
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            linecolor="rgba(255,255,255,0.06)",
            showgrid=True
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            linecolor="rgba(255,255,255,0.06)",
            showgrid=True
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(255,255,255,0.07)",
            borderwidth=1
        )
    )
    return fig


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="brand-mark">🌊 KOSI</div>
            <div class="brand-sub">RIVER INTELLIGENCE PLATFORM</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    role = st.selectbox(
        "Workspace",
        ["Researcher", "Water Quality Analyst", "General User"]
    )

    st.markdown("<br>", unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "🏠  Command Center",
            "🔬  Microplastic Vision",
            "🧬  Raman Spectroscopy",
            "🌊  Water Quality",
            "🔮  Digital Twin",
            "📊  Research Lab"
        ]
    )

    st.markdown(
        """
        <div class="module-status">
            <div class="mod-title">ACTIVE MODULES</div>
            <div class="mod-item"><span class="mod-dot"></span> Vision AI — YOLOv8</div>
            <div class="mod-item"><span class="mod-dot"></span> Spectral AI — RBF SVM</div>
            <div class="mod-item"><span class="mod-dot"></span> River AI — Random Forest</div>
            <div class="mod-item"><span class="mod-dot"></span> Twin AI — Temporal RF</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="pravaah-hero">
        <div class="hero-eyebrow">◈ RIVER INTELLIGENCE PLATFORM</div>
        <div class="hero-wordmark">KOSI</div>
        <div class="hero-tagline">
            Named after India's most unpredictable river — KOSI watches so rivers don't become sorrows.
            AI-powered microplastic detection, polymer fingerprinting,
            water-quality estimation and next-state river forecasting,
            unified in a single intelligence platform.
        </div>
        <div class="hero-pills">
            <span class="hero-pill"><span class="live-dot"></span>YOLO ONLINE</span>
            <span class="hero-pill"><span class="live-dot"></span>RAMAN ONLINE</span>
            <span class="hero-pill"><span class="live-dot"></span>WATER MODEL ONLINE</span>
            <span class="hero-pill"><span class="live-dot"></span>DIGITAL TWIN ONLINE</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# COMMAND CENTER
# ============================================================

if page == "🏠  Command Center":

    section_head("Performance Snapshot", "01", "Live model metrics across all four intelligence modules.")

    cols = st.columns(4)
    cards = [
        ("YOLO mAP@50",     "73.4%", "Microplastic particle detection"),
        ("Raman Accuracy",  "93.9%", "Polymer identification via SVM"),
        ("River RF  R²",    "0.933", "Water Quality Index estimation"),
        ("Digital Twin R²", "0.987", "Future WQI state forecasting"),
    ]
    for col, (lbl, val, sub) in zip(cols, cards):
        with col:
            metric_card(lbl, val, sub)

    st.markdown('<div class="teal-divider"></div>', unsafe_allow_html=True)

    section_head("Multimodal Pipeline", "02", "Five-stage flow from raw image to future river state.")

    st.markdown(
        """
        <div class="pipe-grid">
            <div class="pipe-node">
                <span class="pipe-icon">📷</span>
                <div class="pipe-name">Vision</div>
                <div class="pipe-desc">Detect particles</div>
                <span class="pipe-arr">›</span>
            </div>
            <div class="pipe-node">
                <span class="pipe-icon">📐</span>
                <div class="pipe-name">Morphology</div>
                <div class="pipe-desc">Shape analysis</div>
                <span class="pipe-arr">›</span>
            </div>
            <div class="pipe-node">
                <span class="pipe-icon">🧬</span>
                <div class="pipe-name">Raman</div>
                <div class="pipe-desc">Polymer ID</div>
                <span class="pipe-arr">›</span>
            </div>
            <div class="pipe-node">
                <span class="pipe-icon">🌊</span>
                <div class="pipe-name">Water</div>
                <div class="pipe-desc">WQI analysis</div>
                <span class="pipe-arr">›</span>
            </div>
            <div class="pipe-node">
                <span class="pipe-icon">🔮</span>
                <div class="pipe-name">Digital Twin</div>
                <div class="pipe-desc">Future state</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="teal-divider"></div>', unsafe_allow_html=True)

    section_head("Live River State", "03", "WQI evolution for Ganga and Sangam monitoring sites.")

    g_plot = ganga.sort_values("Date").tail(300)
    s_plot = sangam.sort_values("Date").tail(300)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=g_plot["Date"], y=g_plot["WQI"],
        mode="lines", name="Ganga",
        line=dict(color="#00f5d4", width=2),
        fill="tozeroy",
        fillcolor="rgba(0,245,212,0.05)"
    ))
    fig.add_trace(go.Scatter(
        x=s_plot["Date"], y=s_plot["WQI"],
        mode="lines", name="Sangam",
        line=dict(color="#a78bfa", width=2),
        fill="tozeroy",
        fillcolor="rgba(167,139,250,0.05)"
    ))

    frames = []
    steps  = min(len(g_plot), len(s_plot))
    for i in range(20, steps, 10):
        frames.append(go.Frame(data=[
            go.Scatter(x=g_plot["Date"].iloc[:i], y=g_plot["WQI"].iloc[:i],
                       mode="lines", name="Ganga",
                       line=dict(color="#00f5d4", width=2),
                       fill="tozeroy", fillcolor="rgba(0,245,212,0.05)"),
            go.Scatter(x=s_plot["Date"].iloc[:i], y=s_plot["WQI"].iloc[:i],
                       mode="lines", name="Sangam",
                       line=dict(color="#a78bfa", width=2),
                       fill="tozeroy", fillcolor="rgba(167,139,250,0.05)")
        ]))
    fig.frames = frames

    fig.update_layout(
        title="WQI Evolution — Animated",
        updatemenus=[{
            "type": "buttons", "direction": "left",
            "x": 0.01, "y": 1.15,
            "buttons": [{
                "label": "▶  PLAY",
                "method": "animate",
                "args": [None, {"frame": {"duration": 80, "redraw": True}, "fromcurrent": True}]
            }]
        }]
    )
    st.plotly_chart(plotly_theme(fig, 480), use_container_width=True)

    st.markdown('<div class="teal-divider"></div>', unsafe_allow_html=True)
    section_head("River Status Distribution", "04")

    c1, c2 = st.columns(2)
    for col, df, site_name, color in [
        (c1, ganga,  "Ganga",  ["#00f5d4","#38bdf8","#7c3aed","#f59e0b","#f43f5e"]),
        (c2, sangam, "Sangam", ["#a78bfa","#38bdf8","#00f5d4","#fcd34d","#f43f5e"]),
    ]:
        with col:
            status = df["Status"].value_counts()
            fig = go.Figure(go.Pie(
                labels=status.index,
                values=status.values,
                hole=0.62,
                marker=dict(colors=color, line=dict(width=0)),
                textinfo="percent",
                textfont=dict(family="JetBrains Mono", size=11)
            ))
            fig.update_layout(
                title=site_name,
                annotations=[dict(text=site_name, showarrow=False,
                                  font=dict(size=14, color="#e8f3ff", family="Space Grotesk"))]
            )
            st.plotly_chart(plotly_theme(fig, 380), use_container_width=True)


# ============================================================
# MICROPLASTIC VISION
# ============================================================

elif page == "🔬  Microplastic Vision":

    section_head("Microplastic Vision", "01", "Upload a microscope image — YOLO detects particles and aspect-ratio rules classify morphology.")

    uploaded = st.file_uploader("Upload microscope image", type=["jpg","jpeg","png"], key="vision_upload")

    if uploaded:
        data   = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        image  = cv2.imdecode(data, cv2.IMREAD_COLOR)
        result = yolo_model.predict(image, conf=0.25, verbose=False)[0]
        output = image.copy()

        counts = {"Pellet": 0, "Fragment": 0, "Filament": 0, "Unclassified": 0}
        aspect_ratios = []
        detections    = []

        for box in result.boxes:
            x1, y1, x2, y2  = map(int, box.xyxy[0])
            confidence       = float(box.conf[0])
            width  = x2 - x1
            height = y2 - y1
            if height == 0:
                continue
            ratio = width / height
            label, color = morphology(ratio)
            counts[label] += 1
            aspect_ratios.append(ratio)
            detections.append({"Shape": label, "Aspect Ratio": round(ratio, 3), "Confidence": round(confidence, 3)})
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            cv2.putText(output, f"{label} {confidence:.2f}", (x1, max(y1-10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)

        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

        left, right = st.columns([1.8, 1])
        with left:
            st.image(output, caption="YOLO detections with morphology labels", use_container_width=True)
        with right:
            metric_card("Total Detected",   str(sum(counts.values())), "Particles above confidence threshold")
            st.write("")
            a, b = st.columns(2)
            with a: metric_card("Pellets",   counts["Pellet"],   "AR 1.0–1.3")
            with b: metric_card("Fragments", counts["Fragment"], "AR 1.3–4.0")
            st.write("")
            metric_card("Filaments", counts["Filament"], "AR 0.8–1.0")

        if detections:
            st.markdown('<div class="teal-divider"></div>', unsafe_allow_html=True)
            section_head("Morphology Analysis", "02")

            c1, c2 = st.columns(2)
            with c1:
                fig = go.Figure(go.Pie(
                    labels=list(counts.keys()),
                    values=list(counts.values()),
                    hole=0.60,
                    marker=dict(colors=["#f43f5e","#00f5d4","#38bdf8","#64748b"], line=dict(width=0)),
                    textinfo="percent+label",
                    textfont=dict(family="JetBrains Mono", size=10)
                ))
                fig.update_layout(title="Morphology Composition")
                st.plotly_chart(plotly_theme(fig, 400), use_container_width=True)

            with c2:
                fig = go.Figure(go.Histogram(
                    x=aspect_ratios, nbinsx=25,
                    marker=dict(color="#00f5d4", opacity=0.75, line=dict(width=0))
                ))
                for xv, lbl in [(0.8,"←Fil"), (1.0,"Fil/Pel→"), (1.3,"Pel/Fra→"), (4.0,"Fra→")]:
                    fig.add_vline(x=xv, line_dash="dot",
                                  line_color="rgba(0,245,212,0.4)",
                                  annotation_text=lbl,
                                  annotation_font=dict(size=9, color="#00f5d4"))
                fig.update_layout(title="Aspect Ratio Distribution",
                                  xaxis_title="Width / Height",
                                  yaxis_title="Count")
                st.plotly_chart(plotly_theme(fig, 400), use_container_width=True)

            section_head("Particle-level Results", "03")
            st.dataframe(pd.DataFrame(detections), use_container_width=True, hide_index=True)


# ============================================================
# RAMAN SPECTROSCOPY
# ============================================================

elif page == "🧬  Raman Spectroscopy":

    section_head("Raman Spectroscopy Lab", "01",
                 "Spectral preprocessing → peak detection → polymer classification. Shift in cm⁻¹, intensity normalized.")

    uploaded = st.file_uploader("Upload Raman spectrum (.txt)", type=["txt"], key="raman_upload")

    if uploaded:
        data      = pd.read_csv(uploaded, sep=r"\s+", header=None, names=["shift","intensity"]).dropna()
        processed = process_raman(data)
        X         = pd.DataFrame([processed], columns=[f"x{i}" for i in range(1400)])
        prediction   = raman_model.predict(X)[0]
        probabilities = raman_model.predict_proba(X)[0]
        confidence   = probabilities.max()
        peaks, _     = find_peaks(processed, prominence=0.01, distance=10)

        peak_df = pd.DataFrame({
            "Raman Shift (cm⁻¹)": raman_grid[peaks],
            "Intensity": processed[peaks]
        }).sort_values("Intensity", ascending=False).head(10)

        # spectrum chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data["shift"], y=data["intensity"],
            mode="lines", name="Raw Spectrum",
            line=dict(color="rgba(107,139,160,0.5)", width=1.2)
        ))
        fig.add_trace(go.Scatter(
            x=raman_grid, y=processed,
            mode="lines", name="Processed",
            line=dict(color="#00f5d4", width=2.2)
        ))
        if len(peaks) > 0:
            fig.add_trace(go.Scatter(
                x=raman_grid[peaks], y=processed[peaks],
                mode="markers+text",
                text=[f"{v:.0f}" for v in raman_grid[peaks]],
                textposition="top center",
                textfont=dict(family="JetBrains Mono", size=9, color="#f59e0b"),
                marker=dict(size=8, color="#f59e0b", symbol="circle",
                            line=dict(width=1.5, color="#030c18")),
                name="Peaks"
            ))
        fig.update_layout(title="Raman Shift vs Intensity",
                          xaxis_title="Raman Shift (cm⁻¹)",
                          yaxis_title="Normalized Intensity",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(plotly_theme(fig, 520), use_container_width=True)

        st.markdown('<div class="teal-divider"></div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1: metric_card("Predicted Polymer", str(prediction),    "RBF SVM classification")
        with c2: metric_card("Confidence",  f"{confidence*100:.1f}%", "Max class probability")
        with c3: metric_card("Peak Count",  str(len(peaks)),          "Detected spectral peaks")
        with c4: metric_card("Spectral Range", f"{raman_grid.min():.0f}–{raman_grid.max():.0f}", "cm⁻¹ shared grid")

        st.markdown('<div class="teal-divider"></div>', unsafe_allow_html=True)
        section_head("Polymer Probability Profile", "02")

        prob_df = pd.DataFrame({
            "Polymer": raman_model.classes_,
            "Probability": probabilities
        }).sort_values("Probability", ascending=True)

        fig = go.Figure(go.Bar(
            x=prob_df["Probability"],
            y=prob_df["Polymer"],
            orientation="h",
            marker=dict(
                color=prob_df["Probability"],
                colorscale=[[0,"rgba(0,245,212,0.15)"], [1,"rgba(0,245,212,0.90)"]],
                line=dict(width=0)
            )
        ))
        fig.update_layout(title="SVM Class Probabilities",
                          xaxis_title="Probability", yaxis_title="Polymer")
        st.plotly_chart(plotly_theme(fig, 380), use_container_width=True)

        section_head("Dominant Raman Peaks", "03")
        st.dataframe(peak_df, use_container_width=True, hide_index=True)

        st.markdown(
            """
            <div class="gc">
                <span class="bdg bdg-teal">SPECTROSCOPY</span>
                <h3 style="margin:10px 0 8px;font-size:16px;">What is being analyzed?</h3>
                <p style="color:#8ca8c5;font-size:13px;line-height:1.7;margin:0;">
                    Raman spectroscopy measures inelastic light scattering as a function of wavenumber shift.
                    The resulting spectrum is a material fingerprint. PRAVAAH applies baseline correction,
                    Savitzky–Golay smoothing and L2 normalization before passing the vector to the trained SVM.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# WATER QUALITY
# ============================================================

elif page == "🌊  Water Quality":

    section_head("Water Quality Intelligence", "01",
                 "Enter measured environmental parameters and estimate the Water Quality Index.")

    river_df = pd.read_csv("dataset/Results_MADE.csv")

    features = [
        "Temperature",
        "Dissolved Oxygen",
        "pH",
        "Bio-Chemical Oxygen Demand (mg/L)",
        "Faecal Streptococci (MPN/ 100 mL)",
        "Nitrate (mg/ L)",
        "Faecal Coliform (MPN/ 100 mL)",
        "Total Coliform (MPN/ 100 mL)",
        "Conductivity (mho/ Cm)"
    ]

    values   = {}
    sections = st.columns(3)
    for i, feature in enumerate(features):
        with sections[i % 3]:
            values[feature] = st.number_input(feature, value=float(river_df[feature].median()))

    st.write("")
    if st.button("⚡  ESTIMATE WATER QUALITY INDEX", use_container_width=True):
        input_df   = pd.DataFrame([values])
        prediction = float(river_model.predict(input_df)[0])
        status     = ("Good" if prediction <= 50 else
                      "Fair" if prediction <= 100 else
                      "Poor" if prediction <= 200 else "Very Poor")
        status_color = {"Good":"#22d3a0","Fair":"#f59e0b","Poor":"#f97316","Very Poor":"#f43f5e"}[status]

        st.write("")
        r1, r2, r3 = st.columns(3)
        with r1: metric_card("Predicted WQI", f"{prediction:.2f}", "Random Forest estimate")
        with r2: metric_card("Water Status",  status, "Estimated condition")
        with r3: metric_card("Dominant Predictor", "Conductivity", "Feature importance ≈ 0.945")

        st.markdown('<div class="teal-divider"></div>', unsafe_allow_html=True)
        section_head("Environmental Profile Radar", "02")

        radar_labels = ["Temperature","DO","pH","Nitrate","Conductivity"]
        radar_values = [
            values["Temperature"],
            values["Dissolved Oxygen"],
            values["pH"],
            values["Nitrate (mg/ L)"],
            values["Conductivity (mho/ Cm)"]
        ]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=radar_values, theta=radar_labels,
            fill="toself",
            fillcolor="rgba(0,245,212,0.07)",
            line=dict(color="#00f5d4", width=2),
            name="Current Profile"
        ))
        fig.update_layout(
            title="Selected Water Parameters",
            polar=dict(
                bgcolor="rgba(255,255,255,0.015)",
                radialaxis=dict(gridcolor="rgba(255,255,255,0.07)"),
                angularaxis=dict(gridcolor="rgba(255,255,255,0.07)")
            )
        )
        st.plotly_chart(plotly_theme(fig, 460), use_container_width=True)


# ============================================================
# DIGITAL TWIN
# ============================================================

elif page == "🔮  Digital Twin":

    section_head("River Digital Twin", "01",
                 "Lagged environmental observations drive a next-state WQI forecast.")

    site = st.selectbox("River System", ["Ganga", "Sangam"])
    data = (ganga if site == "Ganga" else sangam).copy().sort_values("Date")

    for col in ["WQI","DO","pH","ORP","Cond","Temp"]:
        data[f"{col}_lag1"] = data[col].shift(1)
        data[f"{col}_lag2"] = data[col].shift(2)
        data[f"{col}_lag3"] = data[col].shift(3)

    clean = data.dropna().copy()
    digital_features = [
        f"{p}_lag{l}"
        for p in ["WQI","DO","pH","ORP","Cond","Temp"]
        for l in [1,2,3]
    ]

    latest     = clean.iloc[-1:]
    future_wqi = float(digital_model.predict(latest[digital_features])[0])
    current_wqi = float(clean["WQI"].iloc[-1])
    delta       = future_wqi - current_wqi
    direction   = "Improving 📉" if delta < 0 else ("Worsening 📈" if delta > 0 else "Stable ━")

    c1, c2, c3 = st.columns(3)
    with c1: metric_card("Current WQI",       f"{current_wqi:.2f}", "Latest observed")
    with c2: metric_card("Forecast WQI",      f"{future_wqi:.2f}", "Next-state prediction")
    with c3: metric_card("Expected Direction", direction,            f"Δ = {delta:+.2f}")

    st.markdown('<div class="teal-divider"></div>', unsafe_allow_html=True)
    section_head("Historical + Forecast Timeline", "02")

    recent      = clean.tail(300)
    future_date = recent["Date"].iloc[-1] + pd.Timedelta(days=1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recent["Date"], y=recent["WQI"],
        mode="lines", name="Historical WQI",
        line=dict(color="#38bdf8", width=1.8),
        fill="tozeroy", fillcolor="rgba(56,189,248,0.04)"
    ))
    fig.add_trace(go.Scatter(
        x=[recent["Date"].iloc[-1], future_date],
        y=[current_wqi, future_wqi],
        mode="lines+markers", name="Forecast",
        line=dict(color="#00f5d4", width=3, dash="dot"),
        marker=dict(size=10, color="#00f5d4",
                    line=dict(width=2, color="#030c18"))
    ))
    fig.add_vline(
        x=recent["Date"].iloc[-1],
        line_dash="dash", line_color="rgba(0,245,212,0.3)",
        annotation_text="NOW", annotation_font=dict(color="#00f5d4", size=10)
    )
    fig.update_layout(title=f"{site} — Historical WQI & Next-State Forecast",
                      xaxis_title="Date", yaxis_title="WQI")
    st.plotly_chart(plotly_theme(fig, 500), use_container_width=True)

    st.markdown('<div class="teal-divider"></div>', unsafe_allow_html=True)
    section_head("Temporal WQI Heatmap", "03")

    heat = data.copy()
    heat["Month"] = heat["Date"].dt.month
    heat["Day"]   = heat["Date"].dt.day
    pivot = heat.pivot_table(index="Month", columns="Day", values="WQI", aggfunc="mean")

    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns, y=pivot.index,
        colorscale=[[0,"#030c18"],[0.3,"#0f2a3d"],[0.6,"#00789c"],[1,"#00f5d4"]],
        colorbar=dict(title="WQI", tickfont=dict(family="JetBrains Mono", size=10))
    ))
    fig.update_layout(title="River State Across Days and Months",
                      xaxis_title="Day of Month", yaxis_title="Month")
    st.plotly_chart(plotly_theme(fig, 500), use_container_width=True)

    st.markdown(
        """
        <div class="gc">
            <span class="bdg bdg-violet">TEMPORAL AI</span>
            <h3 style="margin:10px 0 8px;font-size:16px;">How the Digital Twin works</h3>
            <p style="color:#8ca8c5;font-size:13px;line-height:1.7;margin:0;">
                The forecasting model is trained on three-lag windows of WQI, DO, pH, ORP, Conductivity and
                Temperature. Given the most recent observation sequence, it predicts the next river state —
                enabling proactive environmental intervention before conditions deteriorate.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# RESEARCH LAB
# ============================================================

elif page == "📊  Research Lab":

    section_head("Research & Model Laboratory", "01", "Performance summary across all four PRAVAAH modules.")

    metrics = pd.DataFrame({
        "Module": ["YOLOv8","Raman SVM","River Random Forest","Digital Twin RF"],
        "Task":   ["Microplastic detection","Polymer identification","WQI estimation","WQI forecasting"],
        "Metric": ["mAP@50","Accuracy","R²","R²"],
        "Score":  [0.734, 0.939, 0.933, 0.987]
    })
    st.dataframe(metrics, use_container_width=True, hide_index=True)

    st.markdown('<div class="teal-divider"></div>', unsafe_allow_html=True)
    section_head("Performance Landscape", "02")

    radar = go.Figure()
    radar.add_trace(go.Scatterpolar(
        r=[0.734, 0.939, 0.933, 0.987],
        theta=["YOLO Detection","Raman SVM","River RF","Digital Twin"],
        fill="toself",
        fillcolor="rgba(0,245,212,0.07)",
        line=dict(color="#00f5d4", width=2.5),
        name="Model Score"
    ))
    radar.update_layout(
        title="Core Model Performance",
        polar=dict(
            bgcolor="rgba(255,255,255,0.015)",
            radialaxis=dict(visible=True, range=[0,1],
                            gridcolor="rgba(255,255,255,0.07)",
                            tickfont=dict(family="JetBrains Mono", size=9)),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.07)",
                             tickfont=dict(family="Space Grotesk", size=11))
        )
    )
    st.plotly_chart(plotly_theme(radar, 480), use_container_width=True)

    st.markdown('<div class="teal-divider"></div>', unsafe_allow_html=True)
    section_head("Module Overview", "03")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            <div class="gc">
                <span class="bdg bdg-teal">VISION</span>
                <h3 style="margin:10px 0 8px;font-size:15px;">Microplastic Detection</h3>
                <p style="color:#8ca8c5;font-size:13px;line-height:1.7;margin:0;">
                    YOLOv8 identifies particle bounding boxes in microscope images.
                    Aspect-ratio thresholds classify morphology into Pellet, Fragment and Filament.
                </p>
            </div>
            <div class="gc">
                <span class="bdg bdg-sky">SPECTRAL</span>
                <h3 style="margin:10px 0 8px;font-size:15px;">Polymer Identification</h3>
                <p style="color:#8ca8c5;font-size:13px;line-height:1.7;margin:0;">
                    Raman spectra are baseline-corrected, smoothed and L2-normalized
                    before classification with an RBF Support Vector Machine.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            """
            <div class="gc">
                <span class="bdg bdg-amber">ENVIRONMENTAL</span>
                <h3 style="margin:10px 0 8px;font-size:15px;">Water Quality</h3>
                <p style="color:#8ca8c5;font-size:13px;line-height:1.7;margin:0;">
                    Nine environmental parameters are mapped to a Water Quality Index score
                    using a trained Random Forest regressor (R² = 0.933).
                </p>
            </div>
            <div class="gc">
                <span class="bdg bdg-violet">TEMPORAL</span>
                <h3 style="margin:10px 0 8px;font-size:15px;">Digital Twin</h3>
                <p style="color:#8ca8c5;font-size:13px;line-height:1.7;margin:0;">
                    Three-lag temporal windows feed a Random Forest that forecasts
                    the next WQI state with R² = 0.987 — enabling proactive monitoring.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="pravaah-footer">
        KOSI &nbsp;·&nbsp; RIVER INTELLIGENCE PLATFORM
        <br><br>
        VISION &nbsp;·&nbsp; SPECTROSCOPY &nbsp;·&nbsp; WATER QUALITY &nbsp;·&nbsp; DIGITAL TWIN
    </div>
    """,
    unsafe_allow_html=True
)