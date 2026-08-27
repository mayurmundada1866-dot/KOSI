import os
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from scipy.signal import savgol_filter, find_peaks
from ultralytics import YOLO


st.set_page_config(
    page_title="KOSI | Environmental Intelligence",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

    :root {
        --teal: #00f5d4;
        --violet: #7c3aed;
        --sky: #38bdf8;
        --text1: #e8f3ff;
        --text2: #8ca8c5;
        --text3: #56738f;
        --surface: rgba(255,255,255,0.036);
        --border: rgba(255,255,255,0.075);
    }

    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
        color: var(--text1);
    }

    .stApp {
        background:
            radial-gradient(ellipse 80% 60% at 50% -20%, rgba(0,245,212,0.07) 0%, transparent 60%),
            radial-gradient(ellipse 60% 40% at 80% 80%, rgba(124,58,237,0.09) 0%, transparent 55%),
            radial-gradient(ellipse 40% 60% at 10% 90%, rgba(56,189,248,0.06) 0%, transparent 50%),
            linear-gradient(175deg, #030c18 0%, #040e1c 40%, #050f1e 100%);
        min-height: 100vh;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #020b16 0%, #030d1a 100%) !important;
        border-right: 1px solid rgba(0,245,212,0.1);
    }

    [data-testid="stSidebar"] * {
        color: var(--text1) !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, rgba(0,245,212,0.15), rgba(124,58,237,0.20)) !important;
        border: 1px solid rgba(0,245,212,0.35) !important;
        color: var(--teal) !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 700 !important;
        letter-spacing: 1px !important;
        border-radius: 12px !important;
        padding: 13px 25px !important;
        transition: all 0.25s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 0 35px rgba(0,245,212,0.18), 0 0 70px rgba(124,58,237,0.10) !important;
    }

    [data-testid="stFileUploader"] {
        background: rgba(0,245,212,0.03) !important;
        border: 1.5px dashed rgba(0,245,212,0.22) !important;
        border-radius: 16px !important;
    }

    .hero {
        position: relative;
        overflow: hidden;
        padding: 44px;
        border-radius: 28px;
        background: linear-gradient(135deg, rgba(0,245,212,0.07), rgba(124,58,237,0.10), rgba(56,189,248,0.06));
        border: 1px solid rgba(0,245,212,0.13);
        box-shadow: 0 0 0 1px rgba(255,255,255,0.04), 0 24px 80px rgba(0,0,0,0.38);
        margin-bottom: 30px;
    }

    .hero:before {
        content: "";
        position: absolute;
        width: 320px; height: 320px;
        border-radius: 50%;
        right: -100px; top: -130px;
        background: radial-gradient(circle, rgba(0,245,212,0.12), transparent 70%);
        animation: drift 8s ease-in-out infinite alternate;
    }

    .hero:after {
        content: "";
        position: absolute;
        width: 180px; height: 180px;
        border-radius: 50%;
        left: 30%; bottom: -80px;
        background: radial-gradient(circle, rgba(124,58,237,0.10), transparent 70%);
        animation: drift2 10s ease-in-out infinite alternate;
    }

    @keyframes drift {
        from { transform: translate(0,0) scale(1); }
        to { transform: translate(20px,-15px) scale(1.08); }
    }

    @keyframes drift2 {
        from { transform: translate(0,0) scale(1); }
        to { transform: translate(-15px,10px) scale(1.08); }
    }

    .eyebrow {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        letter-spacing: 3px;
        color: var(--teal);
        margin-bottom: 12px;
    }

    .wordmark {
        font-size: 64px;
        font-weight: 700;
        letter-spacing: -3px;
        line-height: 1;
        background: linear-gradient(120deg, #00f5d4, #38bdf8, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 15px;
        position: relative;
        z-index: 1;
    }

    .tagline {
        color: var(--text2);
        font-size: 15px;
        line-height: 1.7;
        max-width: 850px;
        position: relative;
        z-index: 1;
    }

    .status-bar {
        display: flex;
        gap: 18px;
        flex-wrap: wrap;
        margin-top: 22px;
    }

    .status-item {
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        color: var(--text2);
        display: flex;
        align-items: center;
        gap: 7px;
    }

    .dot {
        width: 7px; height: 7px;
        border-radius: 50%;
        background: #22d3a0;
        box-shadow: 0 0 8px #22d3a0, 0 0 16px #22d3a0;
        animation: blink 2s ease-in-out infinite;
    }

    @keyframes blink {
        0%,100% { opacity:1; }
        50% { opacity:0.2; }
    }

    .metric-card {
        padding: 22px;
        border-radius: 20px;
        background: var(--surface);
        border: 1px solid var(--border);
        box-shadow: 0 8px 32px rgba(0,0,0,0.22);
        min-height: 135px;
        transition: 0.2s;
        position: relative;
        overflow: hidden;
    }

    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 16px 48px rgba(0,0,0,0.32);
    }

    .metric-card:before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, var(--teal), var(--violet));
    }

    .metric-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        letter-spacing: 2px;
        color: var(--text3);
        margin-bottom: 10px;
    }

    .metric-value {
        font-size: 31px;
        font-weight: 700;
    }

    .metric-info {
        font-size: 11px;
        color: var(--text3);
        margin-top: 8px;
    }

    .section-head {
        display: flex;
        align-items: baseline;
        gap: 13px;
        margin-top: 30px;
        margin-bottom: 6px;
    }

    .section-title {
        font-size: 22px;
        font-weight: 700;
    }

    .section-number {
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        color: var(--teal);
        letter-spacing: 2px;
    }

    .section-sub {
        color: var(--text3);
        font-size: 12px;
        margin-bottom: 18px;
    }

    .glass {
        padding: 22px;
        border-radius: 20px;
        background: var(--surface);
        border: 1px solid var(--border);
        box-shadow: 0 12px 40px rgba(0,0,0,0.15);
        margin-bottom: 16px;
    }

    .pipeline {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 12px;
        margin: 20px 0;
    }

    .pipe {
        padding: 20px 12px;
        text-align: center;
        border-radius: 18px;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        transition: all .2s;
        position: relative;
    }

    .pipe:hover {
        transform: translateY(-4px);
        border-color: rgba(0,245,212,0.25);
        background: rgba(0,245,212,0.05);
    }

    .pipe-icon { font-size: 29px; }
    .pipe-name { font-size: 12px; font-weight: 700; margin-top: 8px; }
    .pipe-desc { font-family: 'JetBrains Mono', monospace; font-size: 9px; color: var(--text3); margin-top: 5px; }

    .teal-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(0,245,212,0.25), transparent);
        margin: 28px 0;
    }

    .bdg {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 9px;
        font-weight: 600;
        letter-spacing: 1px;
        margin-bottom: 10px;
    }

    .bdg-teal   { color: #00f5d4; background: rgba(0,245,212,0.10); border: 1px solid rgba(0,245,212,0.22); }
    .bdg-violet { color: #c4b5fd; background: rgba(124,58,237,0.12); border: 1px solid rgba(124,58,237,0.26); }
    .bdg-amber  { color: #fcd34d; background: rgba(245,158,11,0.10); border: 1px solid rgba(245,158,11,0.22); }
    .bdg-sky    { color: #7dd3fc; background: rgba(56,189,248,0.10); border: 1px solid rgba(56,189,248,0.22); }

    .small-muted { color: var(--text2); font-size: 13px; line-height: 1.7; margin: 0; }

    .footer {
        text-align: center;
        padding: 32px;
        margin-top: 50px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 9px;
        letter-spacing: 2px;
        color: var(--text3);
        border-top: 1px solid rgba(255,255,255,0.05);
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).parent.parent

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
    digital = joblib.load(DIGITAL_MODEL_PATH) if Path(DIGITAL_MODEL_PATH).exists() else None
    return yolo, raman, river, digital


@st.cache_data
def load_river_data():
    g = BASE_DIR / "dataset" / "river_dataset" / "ganga.csv"
    s = BASE_DIR / "dataset" / "river_dataset" / "sangam.csv"
    if not g.exists() or not s.exists():
        return None, None
    ganga  = pd.read_csv(g)
    sangam = pd.read_csv(s)
    ganga["Date"]  = pd.to_datetime(ganga["Date"])
    sangam["Date"] = pd.to_datetime(sangam["Date"])
    return ganga, sangam


@st.cache_data
def build_raman_grid():
    content = RAMAN_FOLDER / "content.xlsx"
    if not content.exists():
        return np.linspace(200, 3600, 1400)
    metadata = pd.read_excel(content, header=1)
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
        data = pd.read_csv(file, sep=r"\s+", header=None, names=["shift","intensity"]).dropna()
        spectra.append(data)
    if not spectra:
        return np.linspace(200, 3600, 1400)
    return np.linspace(
        max(s["shift"].min() for s in spectra),
        min(s["shift"].max() for s in spectra),
        1400
    )


yolo_model, raman_model, river_model, digital_model = load_models()
ganga, sangam = load_river_data()
raman_grid    = build_raman_grid()


# ============================================================
# HELPERS
# ============================================================

def metric_card(label, value, info):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-info">{info}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def section_head(title, number="", subtitle=""):
    st.markdown(
        f"""
        <div class="section-head">
            <span class="section-title">{title}</span>
            <span class="section-number">{number}</span>
        </div>
        <div class="section-sub">{subtitle}</div>
        """,
        unsafe_allow_html=True
    )


def morphology(ratio):
    if ratio > 1 and ratio <= 1.3:
        return "Pellet",      (255, 80,  80)
    if ratio > 1.3 and ratio <= 4:
        return "Fragment",    (80,  220, 160)
    if ratio > 0.8 and ratio <= 1:
        return "Filament",    (80,  150, 255)
    return "Unclassified", (150, 150, 150)


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


def plot_theme(fig, height=450):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.015)",
        font=dict(color="#8ca8c5", family="Space Grotesk"),
        margin=dict(l=25, r=25, t=50, b=30)
    )
    return fig


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="padding:0 16px 24px; border-bottom:1px solid rgba(0,245,212,0.10); margin-bottom:20px;">
            <div style="font-size:26px;font-weight:700;background:linear-gradient(120deg,#00f5d4,#38bdf8);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">🌊 KOSI</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:9px;color:#56738f;letter-spacing:2.5px;margin-top:3px;">ENVIRONMENTAL INTELLIGENCE</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    role = st.selectbox("Workspace", ["Researcher", "Water Quality Analyst", "General User"])

    st.markdown("---")

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

    st.markdown("---")

    st.markdown(
        """
        <div class="glass">
            <div class="metric-label">ACTIVE MODULES</div>
            <div class="status-item" style="margin:6px 0;display:flex;align-items:center;gap:8px;font-size:11px;color:#8ca8c5;"><span class="dot"></span> Vision AI</div>
            <div class="status-item" style="margin:6px 0;display:flex;align-items:center;gap:8px;font-size:11px;color:#8ca8c5;"><span class="dot"></span> Spectral AI</div>
            <div class="status-item" style="margin:6px 0;display:flex;align-items:center;gap:8px;font-size:11px;color:#8ca8c5;"><span class="dot"></span> Water AI</div>
            <div class="status-item" style="margin:6px 0;display:flex;align-items:center;gap:8px;font-size:11px;color:#8ca8c5;"><span class="dot"></span> Temporal AI</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">◈ MULTIMODAL ENVIRONMENTAL INTELLIGENCE</div>
        <div class="wordmark">KOSI</div>
        <div class="tagline">
            AI-powered microplastic detection, morphology analysis,
            Raman polymer identification, water-quality assessment
            and next-state river forecasting — unified in one intelligent platform.
        </div>
        <div class="status-bar">
            <div class="status-item"><span class="dot"></span> YOLO READY</div>
            <div class="status-item"><span class="dot"></span> RAMAN READY</div>
            <div class="status-item"><span class="dot"></span> WATER MODEL READY</div>
            <div class="status-item"><span class="dot"></span> DIGITAL TWIN READY</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# COMMAND CENTER
# ============================================================

if page == "🏠  Command Center":

    section_head("Performance Snapshot", "01", "Core model performance across the KOSI multimodal pipeline.")

    cols = st.columns(4)
    cards = [
        ("YOLO mAP@50",     "73.4%", "Microplastic detection"),
        ("Raman Accuracy",  "93.9%", "Polymer identification"),
        ("River RF R²",     "0.933", "WQI estimation"),
        ("Digital Twin R²", "0.987", "True WQI forecasting"),
    ]
    for col, (label, value, info) in zip(cols, cards):
        with col:
            metric_card(label, value, info)

    st.markdown('<div class="teal-divider"></div>', unsafe_allow_html=True)

    section_head("Multimodal Pipeline", "02", "From particle detection to future river-state intelligence.")

    st.markdown(
        """
        <div class="pipeline">
            <div class="pipe"><div class="pipe-icon">📷</div><div class="pipe-name">Vision</div><div class="pipe-desc">YOLO DETECTION</div></div>
            <div class="pipe"><div class="pipe-icon">📐</div><div class="pipe-name">Morphology</div><div class="pipe-desc">ASPECT RATIO</div></div>
            <div class="pipe"><div class="pipe-icon">🧬</div><div class="pipe-name">Raman</div><div class="pipe-desc">POLYMER ID</div></div>
            <div class="pipe"><div class="pipe-icon">🌊</div><div class="pipe-name">Water</div><div class="pipe-desc">WQI MODEL</div></div>
            <div class="pipe"><div class="pipe-icon">🔮</div><div class="pipe-name">Digital Twin</div><div class="pipe-desc">FORECAST</div></div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="teal-divider"></div>', unsafe_allow_html=True)

    section_head("River Intelligence", "03", "Real historical WQI behaviour from Ganga and Sangam.")

    if ganga is not None and sangam is not None:

        g = ganga.sort_values("Date").tail(300)
        s = sangam.sort_values("Date").tail(300)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=g["Date"], y=g["WQI"], mode="lines", name="Ganga",  line=dict(color="#00f5d4", width=2)))
        fig.add_trace(go.Scatter(x=s["Date"], y=s["WQI"], mode="lines", name="Sangam", line=dict(color="#a78bfa", width=2)))

        frames = []
        steps  = min(len(g), len(s))
        for i in range(20, steps, 10):
            frames.append(go.Frame(data=[
                go.Scatter(x=g["Date"].iloc[:i], y=g["WQI"].iloc[:i], mode="lines", name="Ganga",  line=dict(color="#00f5d4", width=2)),
                go.Scatter(x=s["Date"].iloc[:i], y=s["WQI"].iloc[:i], mode="lines", name="Sangam", line=dict(color="#a78bfa", width=2))
            ]))
        fig.frames = frames

        fig.update_layout(
            title="Animated WQI Evolution",
            xaxis_title="Date", yaxis_title="WQI",
            updatemenus=[{"type":"buttons","buttons":[{"label":"▶ PLAY","method":"animate","args":[None,{"frame":{"duration":90,"redraw":True},"fromcurrent":True}]}]}]
        )
        st.plotly_chart(plot_theme(fig, 470), use_container_width=True)

        st.markdown('<div class="teal-divider"></div>', unsafe_allow_html=True)
        section_head("River Status Distribution", "04")

        c1, c2 = st.columns(2)
        for col, df, site in [(c1, ganga, "Ganga"), (c2, sangam, "Sangam")]:
            with col:
                status = df["Status"].value_counts()
                fig = go.Figure(go.Pie(labels=status.index, values=status.values, hole=0.62))
                fig.update_layout(title=site)
                st.plotly_chart(plot_theme(fig, 360), use_container_width=True)

    else:
        st.warning("River datasets not available in this deployment.")


# ============================================================
# MICROPLASTIC VISION
# ============================================================

elif page == "🔬  Microplastic Vision":

    section_head("Microplastic Vision Intelligence", "01", "YOLO detection followed by aspect-ratio morphology analysis.")

    uploaded = st.file_uploader("Upload microscope image", type=["jpg","jpeg","png"])

    if uploaded:

        data   = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        image  = cv2.imdecode(data, cv2.IMREAD_COLOR)
        result = yolo_model.predict(image, conf=0.25, verbose=False)[0]
        output = image.copy()

        counts = {"Pellet":0, "Fragment":0, "Filament":0, "Unclassified":0}
        ratios = []
        detections = []

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            width  = x2 - x1
            height = y2 - y1
            if height == 0:
                continue
            ratio = width / height
            label, color = morphology(ratio)
            counts[label] += 1
            ratios.append(ratio)
            detections.append({"Shape": label, "Aspect Ratio": round(ratio, 3), "Confidence": round(confidence, 3)})
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            cv2.putText(output, f"{label} {confidence:.2f}", (x1, max(y1-10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)

        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

        left, right = st.columns([1.75, 1])
        with left:
            st.image(output, caption="YOLO microplastic detections", use_container_width=True)
        with right:
            metric_card("Total Detected", str(sum(counts.values())), "Confidence threshold ≥ 0.25")
            st.write("")
            a, b = st.columns(2)
            with a: metric_card("Pellet",   counts["Pellet"],   "AR 1.0–1.3")
            with b: metric_card("Fragment", counts["Fragment"], "AR 1.3–4.0")
            st.write("")
            metric_card("Filament", counts["Filament"], "AR 0.8–1.0")

        if ratios:
            st.markdown('<div class="teal-divider"></div>', unsafe_allow_html=True)
            section_head("Morphology Distribution", "02")

            c1, c2 = st.columns(2)
            with c1:
                morph_counts = {k: v for k, v in counts.items() if v > 0}
                fig = go.Figure(go.Pie(labels=list(morph_counts.keys()), values=list(morph_counts.values()), hole=0.58))
                fig.update_layout(title="Particle Composition")
                st.plotly_chart(plot_theme(fig, 390), use_container_width=True)

            with c2:
                fig = go.Figure(go.Histogram(x=ratios, nbinsx=25))
                for v in [0.8, 1.0, 1.3, 4.0]:
                    fig.add_vline(x=v, line_dash="dot")
                fig.update_layout(title="Aspect Ratio Distribution", xaxis_title="Width / Height", yaxis_title="Count")
                st.plotly_chart(plot_theme(fig, 390), use_container_width=True)

            section_head("Particle-level Results", "03")
            st.dataframe(pd.DataFrame(detections), use_container_width=True, hide_index=True)


# ============================================================
# RAMAN SPECTROSCOPY
# ============================================================

elif page == "🧬  Raman Spectroscopy":

    section_head("Raman Spectroscopy Laboratory", "01", "Signal processing, peak detection and polymer identification.")

    uploaded = st.file_uploader("Upload Raman spectrum (.txt)", type=["txt"])

    if uploaded:

        raw       = pd.read_csv(uploaded, sep=r"\s+", header=None, names=["shift","intensity"]).dropna()
        processed = process_raman(raw)
        X         = pd.DataFrame([processed], columns=[f"x{i}" for i in range(1400)])

        prediction    = raman_model.predict(X)[0]
        probabilities = raman_model.predict_proba(X)[0]
        confidence    = probabilities.max()
        peaks, _      = find_peaks(processed, prominence=0.01, distance=10)

        peak_df = pd.DataFrame({
            "Raman Shift (cm⁻¹)": raman_grid[peaks],
            "Intensity": processed[peaks]
        }).sort_values("Intensity", ascending=False).head(10)

        section_head("Raman Shift vs Intensity", "02")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=raw["shift"], y=raw["intensity"], mode="lines", name="Raw Spectrum",       line=dict(color="rgba(120,145,165,0.55)", width=1.2)))
        fig.add_trace(go.Scatter(x=raman_grid,   y=processed,        mode="lines", name="Processed Spectrum", line=dict(color="#00f5d4", width=2.3)))
        if len(peaks) > 0:
            fig.add_trace(go.Scatter(
                x=raman_grid[peaks], y=processed[peaks],
                mode="markers+text",
                text=[f"{x:.0f}" for x in raman_grid[peaks]],
                textposition="top center",
                marker=dict(size=8, color="#f59e0b"),
                name="Detected Peaks"
            ))
        fig.update_layout(title="Raw and Processed Raman Spectrum", xaxis_title="Raman Shift (cm⁻¹)", yaxis_title="Intensity", legend=dict(orientation="h"))
        st.plotly_chart(plot_theme(fig, 530), use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1: metric_card("Predicted Polymer", str(prediction),         "RBF SVM")
        with c2: metric_card("Confidence",        f"{confidence*100:.1f}%","Maximum class probability")
        with c3: metric_card("Detected Peaks",    str(len(peaks)),         "Prominence-based detection")
        with c4: metric_card("Feature Length",    "1400",                  "Common spectral grid")

        st.markdown('<div class="teal-divider"></div>', unsafe_allow_html=True)
        section_head("Polymer Probability Profile", "03")

        prob_df = pd.DataFrame({"Polymer": raman_model.classes_, "Probability": probabilities}).sort_values("Probability", ascending=True)
        fig = go.Figure(go.Bar(x=prob_df["Probability"], y=prob_df["Polymer"], orientation="h"))
        fig.update_layout(title="SVM Polymer Probabilities", xaxis_title="Probability")
        st.plotly_chart(plot_theme(fig, 390), use_container_width=True)

        section_head("Dominant Raman Peaks", "04")
        st.dataframe(peak_df, use_container_width=True, hide_index=True)

        st.markdown(
            """
            <div class="glass">
                <span class="bdg bdg-teal">SPECTROSCOPY</span>
                <h3>What is Raman spectroscopy measuring?</h3>
                <p class="small-muted">
                    Raman spectroscopy represents the intensity of scattered light as a function of Raman shift.
                    The resulting spectrum acts as a material fingerprint. KOSI applies interpolation onto a common
                    spectral grid, Savitzky–Golay baseline correction, smoothing and L2 normalization before polymer classification.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# WATER QUALITY
# ============================================================

elif page == "🌊  Water Quality":

    section_head("Water Quality Intelligence", "01", "Estimate Water Quality Index from environmental measurements.")

    river_csv = BASE_DIR / "dataset" / "Results_MADE.csv"
    if river_csv.exists():
        river_df = pd.read_csv(river_csv)
    else:
        st.warning("Reference dataset not available. Using default values.")
        river_df = None

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

    defaults = {f: float(river_df[f].median()) if river_df is not None else 0.0 for f in features}

    values = {}
    cols   = st.columns(3)
    for i, feature in enumerate(features):
        with cols[i % 3]:
            values[feature] = st.number_input(feature, value=defaults[feature])

    st.write("")

    if st.button("⚡ ESTIMATE WATER QUALITY", use_container_width=True):

        prediction = float(river_model.predict(pd.DataFrame([values]))[0])
        status     = ("Good" if prediction <= 50 else "Fair" if prediction <= 100 else "Poor" if prediction <= 200 else "Very Poor")

        c1, c2, c3 = st.columns(3)
        with c1: metric_card("Predicted WQI",      f"{prediction:.2f}", "Random Forest")
        with c2: metric_card("Water Status",        status,              "Estimated condition")
        with c3: metric_card("Dominant Predictor",  "Conductivity",      "Feature importance ≈ 0.945")

        section_head("Environmental Parameter Profile", "02")

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=[values["Temperature"], values["Dissolved Oxygen"], values["pH"], values["Nitrate (mg/ L)"], values["Conductivity (mho/ Cm)"]],
            theta=["Temperature","DO","pH","Nitrate","Conductivity"],
            fill="toself",
            line=dict(color="#00f5d4"),
            fillcolor="rgba(0,245,212,0.08)"
        ))
        fig.update_layout(title="Current Environmental Profile", polar=dict(bgcolor="rgba(255,255,255,0.015)"))
        st.plotly_chart(plot_theme(fig, 450), use_container_width=True)


# ============================================================
# DIGITAL TWIN
# ============================================================

elif page == "🔮  Digital Twin":

    section_head("River Digital Twin", "01", "Historical observations → temporal features → next-state WQI forecast.")

    if digital_model is None:
        st.warning("Digital Twin model not available in this deployment.")
    elif ganga is None or sangam is None:
        st.warning("River datasets not available in this deployment.")
    else:
        site = st.selectbox("Select River", ["Ganga", "Sangam"])
        data = (ganga.copy() if site == "Ganga" else sangam.copy()).sort_values("Date")

        for col in ["WQI","DO","pH","ORP","Cond","Temp"]:
            data[f"{col}_lag1"] = data[col].shift(1)
            data[f"{col}_lag2"] = data[col].shift(2)
            data[f"{col}_lag3"] = data[col].shift(3)

        clean = data.dropna().copy()

        digital_features = [f"{p}_lag{l}" for p in ["WQI","DO","pH","ORP","Cond","Temp"] for l in [1,2,3]]

        latest      = clean.iloc[-1:]
        future_wqi  = float(digital_model.predict(latest[digital_features])[0])
        current_wqi = float(clean["WQI"].iloc[-1])
        delta       = future_wqi - current_wqi
        direction   = "Improving" if delta < 0 else ("Worsening" if delta > 0 else "Stable")

        c1, c2, c3 = st.columns(3)
        with c1: metric_card("Current WQI",       f"{current_wqi:.2f}", "Latest observed state")
        with c2: metric_card("Forecast WQI",      f"{future_wqi:.2f}", "Next-state prediction")
        with c3: metric_card("Expected Direction", direction,            f"Δ = {delta:+.2f}")

        section_head("Historical + Forecast Timeline", "02")

        recent      = clean.tail(300)
        future_date = recent["Date"].iloc[-1] + pd.Timedelta(days=1)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recent["Date"], y=recent["WQI"], mode="lines", name="Historical WQI", line=dict(color="#38bdf8", width=1.8)))
        fig.add_trace(go.Scatter(
            x=[recent["Date"].iloc[-1], future_date],
            y=[current_wqi, future_wqi],
            mode="lines+markers", name="Forecast",
            line=dict(color="#00f5d4", width=3, dash="dot"),
            marker=dict(size=9)
        ))
        fig.add_vline(x=recent["Date"].iloc[-1], line_dash="dash")
        fig.update_layout(title=f"{site} — Historical vs Next-State Forecast", xaxis_title="Date", yaxis_title="WQI")
        st.plotly_chart(plot_theme(fig, 500), use_container_width=True)

        section_head("Temporal WQI Heatmap", "03")

        heat          = data.copy()
        heat["Month"] = heat["Date"].dt.month
        heat["Day"]   = heat["Date"].dt.day
        pivot         = heat.pivot_table(index="Month", columns="Day", values="WQI", aggfunc="mean")

        fig = go.Figure(go.Heatmap(
            z=pivot.values, x=pivot.columns, y=pivot.index,
            colorscale=[[0,"#030c18"],[0.3,"#0f2a3d"],[0.6,"#00789c"],[1,"#00f5d4"]],
            colorbar=dict(title="WQI")
        ))
        fig.update_layout(title="River WQI Temporal Heatmap", xaxis_title="Day", yaxis_title="Month")
        st.plotly_chart(plot_theme(fig, 500), use_container_width=True)


# ============================================================
# RESEARCH LAB
# ============================================================

elif page == "📊  Research Lab":

    section_head("Research & Model Laboratory", "01", "Core experimental results from the KOSI development pipeline.")

    metrics = pd.DataFrame({
        "Module": ["YOLOv8","Raman SVM","River Random Forest","Digital Twin Random Forest"],
        "Task":   ["Microplastic detection","Polymer identification","WQI estimation","Next-state WQI forecasting"],
        "Metric": ["mAP@50","Accuracy","R²","R²"],
        "Score":  [0.734, 0.939, 0.933, 0.987]
    })
    st.dataframe(metrics, use_container_width=True, hide_index=True)

    section_head("Performance Landscape", "02")

    radar = go.Figure()
    radar.add_trace(go.Scatterpolar(
        r=[0.734, 0.939, 0.933, 0.987],
        theta=["YOLO","Raman","River","Digital Twin"],
        fill="toself",
        line=dict(color="#00f5d4", width=2.5),
        fillcolor="rgba(0,245,212,0.07)"
    ))
    radar.update_layout(
        title="Core Model Performance",
        polar=dict(bgcolor="rgba(255,255,255,0.015)", radialaxis=dict(visible=True, range=[0,1]))
    )
    st.plotly_chart(plot_theme(radar, 480), use_container_width=True)

    section_head("Module Overview", "03")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            <div class="glass">
                <span class="bdg bdg-teal">VISION</span>
                <h3>Microplastic Detection</h3>
                <p class="small-muted">YOLOv8 identifies microplastic particles in microscope images. Bounding-box geometry is used for morphology analysis.</p>
            </div>
            <div class="glass">
                <span class="bdg bdg-sky">SPECTRAL</span>
                <h3>Raman Polymer Identification</h3>
                <p class="small-muted">Raman spectra are interpolated, baseline-corrected, smoothed and normalized before RBF SVM classification.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            """
            <div class="glass">
                <span class="bdg bdg-amber">ENVIRONMENTAL</span>
                <h3>Water Quality</h3>
                <p class="small-muted">Environmental parameters are mapped to Water Quality Index using a Random Forest regression model.</p>
            </div>
            <div class="glass">
                <span class="bdg bdg-violet">TEMPORAL</span>
                <h3>Digital Twin</h3>
                <p class="small-muted">Historical lagged observations are used to predict the next river WQI state and visualize temporal behaviour.</p>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        KOSI · MULTIMODAL ENVIRONMENTAL INTELLIGENCE
        <br><br>
        VISION · SPECTROSCOPY · WATER QUALITY · DIGITAL TWIN
    </div>
    """,
    unsafe_allow_html=True
)