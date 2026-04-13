# latest_changes/app.py
"""
PlantVillage AI Diagnosis System — Premium UI with GradCAM explainability.
"""

import base64
import io
import logging
import os
import sys

import streamlit as st
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import OOD_THRESHOLD, SUPPORTED_PLANTS
from models.loader import load_clip, load_vlm
from pipeline import run_pipeline
from utils.gradcam import generate_gradcam

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PlantMed AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Premium CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0f0a !important;
    color: #e8f0e8 !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 50% at 20% 10%, rgba(34,85,34,0.18) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(16,60,16,0.14) 0%, transparent 55%),
        #0a0f0a !important;
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { display: none; }
section.main > div { padding: 2rem 2.5rem 4rem; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0a0f0a; }
::-webkit-scrollbar-thumb { background: #2d5a2d; border-radius: 4px; }

/* ── Hero header ── */
.hero {
    text-align: center;
    padding: 3rem 0 2rem;
    position: relative;
}
.hero-badge {
    display: inline-block;
    background: rgba(52,199,89,0.1);
    border: 1px solid rgba(52,199,89,0.3);
    color: #34c759;
    font-family: 'DM Sans', monospace;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    padding: 6px 16px;
    border-radius: 20px;
    margin-bottom: 20px;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.8rem, 5vw, 4.5rem);
    font-weight: 800;
    line-height: 1.05;
    letter-spacing: -2px;
    color: #f0f7f0;
    margin-bottom: 12px;
}
.hero-title span {
    background: linear-gradient(135deg, #34c759, #a8e6a3, #34c759);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shimmer 4s linear infinite;
}
@keyframes shimmer {
    0% { background-position: 0% center; }
    100% { background-position: 200% center; }
}
.hero-sub {
    font-size: 15px;
    color: #7a9c7a;
    font-weight: 300;
    letter-spacing: 0.3px;
}

/* ── Divider ── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(52,199,89,0.2), transparent);
    margin: 2rem 0;
}

/* ── Panel cards ── */
.panel {
    background: rgba(16,26,16,0.7);
    border: 1px solid rgba(52,199,89,0.12);
    border-radius: 16px;
    padding: 1.8rem;
    backdrop-filter: blur(12px);
    margin-bottom: 1.2rem;
    transition: border-color 0.3s;
}
.panel:hover { border-color: rgba(52,199,89,0.25); }
.panel-title {
    font-family: 'Syne', sans-serif;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #34c759;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.panel-title::before {
    content: '';
    display: inline-block;
    width: 18px;
    height: 2px;
    background: #34c759;
    border-radius: 2px;
}

/* ── Streamlit widget overrides ── */
[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(52,199,89,0.2) !important;
    border-radius: 10px !important;
    color: #e8f0e8 !important;
}
[data-testid="stSelectbox"] label {
    color: #7a9c7a !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    letter-spacing: 0.5px !important;
}
[data-testid="stFileUploader"] {
    background: rgba(52,199,89,0.03) !important;
    border: 1.5px dashed rgba(52,199,89,0.25) !important;
    border-radius: 12px !important;
    transition: border-color 0.3s, background 0.3s;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(52,199,89,0.5) !important;
    background: rgba(52,199,89,0.06) !important;
}
[data-testid="stFileUploader"] label { color: #7a9c7a !important; }

/* ── Metric cards ── */
.metric-row {
    display: flex;
    gap: 12px;
    margin: 1.2rem 0;
    flex-wrap: wrap;
}
.metric-card {
    flex: 1;
    min-width: 110px;
    background: rgba(52,199,89,0.07);
    border: 1px solid rgba(52,199,89,0.18);
    border-radius: 12px;
    padding: 14px 16px;
    text-align: center;
}
.metric-card .mc-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #5a8a5a;
    margin-bottom: 6px;
}
.metric-card .mc-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #34c759;
    line-height: 1;
}
.metric-card .mc-sub {
    font-size: 11px;
    color: #5a8a5a;
    margin-top: 4px;
}

/* ── Diagnosis field rows ── */
.field-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin: 1rem 0;
}
.field-item {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(52,199,89,0.1);
    border-radius: 10px;
    padding: 14px 16px;
}
.field-item.full-width { grid-column: 1 / -1; }
.field-item .fi-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #5a8a5a;
    margin-bottom: 6px;
}
.field-item .fi-value {
    font-size: 14px;
    font-weight: 400;
    color: #d4ecd4;
    line-height: 1.5;
}
.field-item .fi-value.bold {
    font-family: 'Syne', sans-serif;
    font-size: 18px;
    font-weight: 700;
    color: #f0f7f0;
}

/* ── Severity badges ── */
.severity-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-top: 1rem;
}
.sev-none    { background: rgba(52,199,89,0.15);  border: 1px solid rgba(52,199,89,0.4);  color: #34c759; }
.sev-mild    { background: rgba(255,214,10,0.12);  border: 1px solid rgba(255,214,10,0.4);  color: #ffd60a; }
.sev-moderate{ background: rgba(255,149,0,0.12);   border: 1px solid rgba(255,149,0,0.4);   color: #ff9500; }
.sev-severe  { background: rgba(255,59,48,0.12);   border: 1px solid rgba(255,59,48,0.4);   color: #ff3b30; }

/* ── GradCAM section ── */
.gradcam-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #5a8a5a;
    text-align: center;
    margin-top: 8px;
}
.gradcam-legend {
    display: flex;
    justify-content: center;
    gap: 16px;
    margin-top: 10px;
    flex-wrap: wrap;
}
.legend-item {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
    color: #7a9c7a;
}
.legend-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
}

/* ── Error states ── */
.error-box {
    background: rgba(255,59,48,0.08);
    border: 1px solid rgba(255,59,48,0.25);
    border-radius: 12px;
    padding: 20px 24px;
    color: #ff6b6b;
}
.warn-box {
    background: rgba(255,149,0,0.08);
    border: 1px solid rgba(255,149,0,0.25);
    border-radius: 12px;
    padding: 20px 24px;
    color: #ff9500;
}
.error-title { font-family: 'Syne', sans-serif; font-size: 15px; font-weight: 700; margin-bottom: 8px; }
.error-body  { font-size: 13px; line-height: 1.6; opacity: 0.85; }

/* ── Waiting state ── */
.waiting {
    text-align: center;
    padding: 4rem 2rem;
    color: #3a5a3a;
}
.waiting-icon { font-size: 3rem; margin-bottom: 1rem; opacity: 0.4; }
.waiting-text { font-size: 14px; letter-spacing: 0.5px; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(52,199,89,0.1) !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary { color: #5a8a5a !important; font-size: 13px !important; }

/* ── Streamlit native overrides ── */
div[data-testid="stImage"] img { border-radius: 12px; }
.stSpinner > div { border-top-color: #34c759 !important; }
[data-testid="stMarkdownContainer"] p { color: #c8dcc8; }
</style>
""", unsafe_allow_html=True)


# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Initialising AI models…")
def _load_models():
    vlm, processor = load_vlm()
    clip_model, clip_proc = load_clip()
    return vlm, processor, clip_model, clip_proc


# ── Helpers ───────────────────────────────────────────────────────────────────
def _severity_badge(severity: str) -> str:
    s = severity.lower()
    if "severe" in s:
        return '<span class="severity-badge sev-severe">🔴 Severe — immediate action required</span>'
    elif "moderate" in s:
        return '<span class="severity-badge sev-moderate">🟠 Moderate — monitor closely</span>'
    elif "mild" in s:
        return '<span class="severity-badge sev-mild">🟡 Mild — early treatment advised</span>'
    else:
        return '<span class="severity-badge sev-none">🟢 None — plant appears healthy</span>'


def _field(label: str, value: str, full: bool = False, bold: bool = False) -> str:
    cls  = "field-item full-width" if full else "field-item"
    vcls = "fi-value bold" if bold else "fi-value"
    return f'''
    <div class="{cls}">
        <div class="fi-label">{label}</div>
        <div class="{vcls}">{value or "—"}</div>
    </div>'''


# ── Main app ──────────────────────────────────────────────────────────────────
def main():
    # Hero
    st.markdown("""
    <div class="hero">
        <div class="hero-badge">🌿 Powered by Qwen2.5-VL + CLIP</div>
        <div class="hero-title">Plant<span>Med</span> AI</div>
        <div class="hero-sub">Vision-Language Disease Diagnosis with Explainable AI</div>
    </div>
    <div class="divider"></div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1.4], gap="large")

    # ── LEFT: Input panel ─────────────────────────────────────────────────────
    with col_left:
        st.markdown('<div class="panel"><div class="panel-title">01 — Configure</div>', unsafe_allow_html=True)
        selected_plant = st.selectbox(
            "Select crop type",
            SUPPORTED_PLANTS,
            index=0,
            label_visibility="visible",
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="panel"><div class="panel-title">02 — Upload Leaf Image</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Drag & drop or click to browse",
            type=["jpg", "jpeg", "png"],
            label_visibility="visible",
        )
        st.markdown('</div>', unsafe_allow_html=True)

        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.markdown('<div class="panel-title" style="margin-bottom:8px;">03 — Preview</div>', unsafe_allow_html=True)
            st.image(image, use_column_width=True)

    # ── RIGHT: Results panel ──────────────────────────────────────────────────
    with col_right:
        if not uploaded:
            st.markdown("""
            <div class="waiting">
                <div class="waiting-icon">🔬</div>
                <div class="waiting-text">Upload a leaf image to begin diagnosis</div>
            </div>
            """, unsafe_allow_html=True)
            return

        # Run pipeline
        with st.spinner("Running AI pipeline…"):
            vlm, processor, clip_model, clip_proc = _load_models()
            result = run_pipeline(
                image, selected_plant,
                vlm, processor,
                clip_model, clip_proc,
                ood_threshold=OOD_THRESHOLD,
            )

        # ── Error states ──────────────────────────────────────────────────────
        if result["status"] == "error":
            err_type = result.get("type", "UNKNOWN")

            if err_type == "OOD":
                st.markdown(f"""
                <div class="error-box">
                    <div class="error-title">❌ Invalid Image Detected</div>
                    <div class="error-body">
                        This image was not recognised as a plant leaf.<br><br>
                        CLIP similarity score: <strong>{result['score']}</strong>
                        (threshold: {OOD_THRESHOLD})<br><br>
                        Please upload a clear, close-up photo of a crop leaf.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            elif err_type == "WRONG_PLANT":
                st.markdown(f"""
                <div class="warn-box">
                    <div class="error-title">⚠️ Crop Mismatch</div>
                    <div class="error-body">
                        Selected crop: <strong>{result['expected']}</strong><br>
                        Detected in image: <strong>{result['detected'] or 'Unknown'}</strong><br><br>
                        Please upload an image matching your selected crop,
                        or change the crop selection.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            elif err_type == "INVALID_OUTPUT":
                st.markdown(f"""
                <div class="error-box">
                    <div class="error-title">❌ Diagnosis Failed</div>
                    <div class="error-body">{result['message']}</div>
                </div>
                """, unsafe_allow_html=True)
            return

        # ── Success ───────────────────────────────────────────────────────────
        data       = result["data"]
        confidence = result.get("confidence", 0.0)
        severity   = data.get("Severity") or "None"

        # Confidence + stats row
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-card">
                <div class="mc-label">Confidence</div>
                <div class="mc-value">{confidence*100:.0f}%</div>
                <div class="mc-sub">CLIP alignment</div>
            </div>
            <div class="metric-card">
                <div class="mc-label">Severity</div>
                <div class="mc-value" style="font-size:1.1rem;padding-top:4px;">{severity}</div>
                <div class="mc-sub">assessed level</div>
            </div>
            <div class="metric-card">
                <div class="mc-label">Model</div>
                <div class="mc-value" style="font-size:0.85rem;padding-top:6px;">Qwen2.5-VL</div>
                <div class="mc-sub">7B LoRA</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Severity badge
        st.markdown(_severity_badge(severity), unsafe_allow_html=True)

        st.markdown('<div class="divider" style="margin:1.2rem 0;"></div>', unsafe_allow_html=True)

        # Diagnosis fields
        st.markdown('<div class="panel-title">Diagnosis</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="field-grid">
            {_field("🌱 Plant",     data.get("Plant"),     bold=True)}
            {_field("🦠 Condition", data.get("Condition"), bold=True)}
            {_field("🔬 Pathogen",  data.get("Pathogen"))}
            {_field("📊 Severity",  data.get("Severity"))}
            {_field("👁️ Visible Symptoms", data.get("Symptoms"),    full=True)}
            {_field("🧠 Explanation",       data.get("Explanation"), full=True)}
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="divider" style="margin:1.5rem 0;"></div>', unsafe_allow_html=True)

        # ── GradCAM ───────────────────────────────────────────────────────────
        st.markdown('<div class="panel-title">Explainability — GradCAM Attention Map</div>', unsafe_allow_html=True)

        with st.spinner("Generating attention heatmap…"):
            heatmap_img = generate_gradcam(image, vlm, processor)

        if heatmap_img:
            gc_col1, gc_col2 = st.columns(2, gap="small")
            with gc_col1:
                st.image(image, use_column_width=True)
                st.markdown('<div class="gradcam-label">Original Image</div>', unsafe_allow_html=True)
            with gc_col2:
                st.image(heatmap_img, use_column_width=True)
                st.markdown('<div class="gradcam-label">Model Attention Heatmap</div>', unsafe_allow_html=True)

            st.markdown("""
            <div class="gradcam-legend">
                <div class="legend-item">
                    <div class="legend-dot" style="background:#ff3b30;"></div>
                    High attention
                </div>
                <div class="legend-item">
                    <div class="legend-dot" style="background:#ff9500;"></div>
                    Medium attention
                </div>
                <div class="legend-item">
                    <div class="legend-dot" style="background:#34c759;"></div>
                    Low attention
                </div>
                <div class="legend-item">
                    <div class="legend-dot" style="background:#0a84ff;"></div>
                    No attention
                </div>
            </div>
            <p style="text-align:center;font-size:11px;color:#3a5a3a;margin-top:10px;">
                Red regions show where the model focused most during diagnosis
            </p>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align:center;padding:2rem;color:#3a5a3a;font-size:13px;">
                GradCAM unavailable for this image — attention weights could not be extracted.
            </div>
            """, unsafe_allow_html=True)

        # Raw output expander
        with st.expander("📄 Raw model output"):
            st.code(result["raw"], language=None)


if __name__ == "__main__":
    main()
