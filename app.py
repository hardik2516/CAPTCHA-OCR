"""
CAPTCHA OCR — Streamlit Application
ResNet18 trained from scratch · No pretrained weights · No transfer learning
"""

import io
import time
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import models

# ──────────────────────────────────────────────────────────────
# 1. Constants
# ──────────────────────────────────────────────────────────────
MODEL_PATH = "final_resnet18_captcha.pth"
VOCAB = [
    "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H",
    "J", "K", "M", "N", "P", "Q", "R", "S",
    "T", "U", "V", "W", "X", "Y", "Z",
]
IDX_TO_CHAR = {i: ch for i, ch in enumerate(VOCAB)}
NUM_CHARS = len(VOCAB)
SEQ_LEN = 6
MAX_UPLOADS = 5
ALLOWED_TYPES = ["png", "jpg", "jpeg"]

# ──────────────────────────────────────────────────────────────
# 2. Model Definition (identical to training)
# ──────────────────────────────────────────────────────────────
class ResNetCaptcha(nn.Module):
    """Modified ResNet-18 for grayscale 6-character CAPTCHA recognition."""

    def __init__(self, num_chars: int) -> None:
        super().__init__()
        backbone = models.resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.features = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
        )
        self.pool = nn.AdaptiveAvgPool2d((1, SEQ_LEN))
        self.classifier = nn.Linear(512, num_chars)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = x.squeeze(2)
        x = x.permute(0, 2, 1)
        x = self.classifier(x)
        return x


# ──────────────────────────────────────────────────────────────
# 3. Model Loading (cached)
# ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model() -> ResNetCaptcha:
    """Load trained weights once and cache in memory."""
    device = torch.device("cpu")
    model = ResNetCaptcha(NUM_CHARS)
    state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


# ──────────────────────────────────────────────────────────────
# 4. Preprocessing & Inference
# ──────────────────────────────────────────────────────────────
def preprocess(image_bytes: bytes) -> Optional[torch.Tensor]:
    """Convert uploaded bytes → grayscale float32 tensor [1,1,H,W]."""
    try:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        img = img.astype(np.float32) / 255.0
        return torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    except Exception:
        return None


def predict(model: ResNetCaptcha, tensor: torch.Tensor) -> dict:
    """Run single-image inference."""
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=2)
        max_probs, preds = probs.max(dim=2)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    text = "".join(IDX_TO_CHAR[idx.item()] for idx in preds[0])
    char_confs = [round(p.item() * 100, 1) for p in max_probs[0]]
    confidence = max_probs[0].mean().item() * 100.0
    return {"text": text, "confidence": confidence, "char_confs": char_confs, "time_ms": elapsed_ms}





# ──────────────────────────────────────────────────────────────
# 6. Custom CSS
# ──────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@500;700&display=swap');

/* ── Global ── */
html, body, .stApp { font-family: 'Inter', -apple-system, sans-serif; }
.stApp { background: #060810; }
.block-container { max-width: 1080px; padding-top: 1.2rem; padding-bottom: 1rem; }

/* Ambient glow */
.stApp::before {
    content: '';
    position: fixed; inset: 0;
    background: radial-gradient(ellipse 80% 50% at 50% -5%, rgba(99,102,241,0.06) 0%, transparent 60%);
    pointer-events: none; z-index: 0;
}

/* ── Hero ── */
.hero {
    position: relative;
    background: linear-gradient(160deg, #0c1020 0%, #111832 40%, #0c1020 100%);
    border-radius: 22px;
    padding: 2.4rem 2rem 1.8rem;
    margin-bottom: 1.8rem;
    text-align: center;
    overflow: hidden;
    border: 1px solid rgba(99,102,241,0.1);
}
.hero::before {
    content: '';
    position: absolute; inset: -1px;
    border-radius: 22px; padding: 1px;
    background: linear-gradient(160deg, rgba(99,102,241,0.25), transparent 35%, transparent 65%, rgba(168,85,247,0.18));
    -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    -webkit-mask-composite: xor; mask-composite: exclude;
    pointer-events: none;
}
.hero::after {
    content: '';
    position: absolute; top: -50%; right: -25%;
    width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(99,102,241,0.05) 0%, transparent 70%);
    pointer-events: none;
}
.hero-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(99,102,241,0.08); color: #818cf8;
    font-size: 0.6rem; font-weight: 700; letter-spacing: 2px; text-transform: uppercase;
    padding: 6px 16px; border-radius: 100px;
    border: 1px solid rgba(99,102,241,0.15); margin-bottom: 0.9rem;
}
.hero h1 {
    font-size: 2.4rem; font-weight: 900;
    background: linear-gradient(135deg, #f1f5f9 30%, #94a3b8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 0.4rem; line-height: 1.1; letter-spacing: -0.5px;
}
.hero-sub {
    color: #64748b; font-size: 0.85rem;
    max-width: 500px; margin: 0 auto; line-height: 1.6;
}

/* ── KPI Row ── */
.kpi-row { display: grid; grid-template-columns: repeat(4,1fr); gap: 0.6rem; margin-top: 1.5rem; }
@media(max-width:700px){ .kpi-row{grid-template-columns:repeat(2,1fr);} }
.kpi {
    background: rgba(12,16,32,0.7); border: 1px solid rgba(99,102,241,0.08);
    border-radius: 14px; padding: 0.9rem 0.6rem; text-align: center;
    backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
    transition: border-color 0.25s, transform 0.25s;
}
.kpi:hover { border-color: rgba(99,102,241,0.3); transform: translateY(-2px); }
.kpi-icon { font-size: 0.9rem; margin-bottom: 2px; }
.kpi-val {
    font-family: 'JetBrains Mono', monospace; font-size: 1.35rem; font-weight: 700;
    background: linear-gradient(135deg, #818cf8, #a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; line-height: 1.3;
}
.kpi-lbl { font-size: 0.56rem; color: #475569; text-transform: uppercase; letter-spacing: 1.2px; font-weight: 700; margin-top: 1px; }

/* ── Section Title ── */
.sec-title {
    color: #e2e8f0; font-size: 0.95rem; font-weight: 700;
    margin: 1.5rem 0 0.7rem; display: flex; align-items: center; gap: 8px;
}
.sec-title .dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: linear-gradient(135deg, #6366f1, #a78bfa);
}

/* ── Result Card (Vertical: image on top) ── */
.rcard {
    background: rgba(12,16,32,0.8);
    border: 1px solid rgba(99,102,241,0.08);
    border-radius: 16px; overflow: hidden;
    backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
    transition: transform 0.2s, box-shadow 0.2s, border-color 0.2s;
    height: 100%;
}
.rcard:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(99,102,241,0.08);
    border-color: rgba(99,102,241,0.22);
}
.rcard-img-wrap {
    background: #0a0d18; padding: 0.7rem;
    border-bottom: 1px solid rgba(99,102,241,0.06);
    display: flex; align-items: center; justify-content: center;
    min-height: 80px;
}
.rcard-img-wrap img {
    border-radius: 8px; max-width: 100%; height: auto;
}
.rcard-body { padding: 0.9rem 1rem 1rem; }
.rcard-filename {
    color: #475569; font-size: 0.65rem; font-weight: 600;
    margin-bottom: 0.5rem; letter-spacing: 0.3px;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}

/* ── Prediction String ── */
.pred-string {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.6rem; font-weight: 700; letter-spacing: 5px;
    text-align: center; margin: 0.3rem 0 0.6rem;
    background: linear-gradient(135deg, #c4b5fd, #818cf8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}

/* ── Confidence Bars ── */
.conf-bars { margin: 0.5rem 0 0.6rem; }
.conf-row {
    display: flex; align-items: center; gap: 6px;
    margin-bottom: 3px; font-size: 0.62rem;
}
.conf-char {
    font-family: 'JetBrains Mono', monospace; font-weight: 700;
    color: #94a3b8; width: 14px; text-align: center;
}
.conf-track {
    flex: 1; height: 6px; background: rgba(99,102,241,0.08);
    border-radius: 3px; overflow: hidden;
}
.conf-fill {
    height: 100%; border-radius: 3px;
    background: linear-gradient(90deg, #6366f1, #a78bfa);
    transition: width 0.6s ease;
}
.conf-pct {
    font-family: 'JetBrains Mono', monospace; font-weight: 600;
    color: #64748b; width: 34px; text-align: right; font-size: 0.58rem;
}

/* ── Badges ── */
.badge-row { display: flex; gap: 6px; flex-wrap: wrap; }
.badge {
    display: inline-flex; align-items: center; gap: 3px;
    font-size: 0.6rem; font-weight: 600; padding: 3px 9px; border-radius: 6px;
}
.b-ok  { background: rgba(34,197,94,0.08); color: #4ade80; border: 1px solid rgba(34,197,94,0.12); }
.b-time{ background: rgba(56,189,248,0.06); color: #7dd3fc; border: 1px solid rgba(56,189,248,0.1); }
.b-conf{ background: rgba(168,85,247,0.06); color: #c4b5fd; border: 1px solid rgba(168,85,247,0.1); }
.b-err { background: rgba(239,68,68,0.08); color: #f87171; border: 1px solid rgba(239,68,68,0.12); }

/* ── Summary Strip ── */
.summary {
    display: grid; grid-template-columns: repeat(4,1fr); gap: 0;
    background: rgba(12,16,32,0.8); border: 1px solid rgba(99,102,241,0.08);
    border-radius: 14px; overflow: hidden; margin: 1rem 0 0.3rem;
    backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
}
.summary-cell {
    text-align: center; padding: 0.85rem 0.4rem;
    border-right: 1px solid rgba(99,102,241,0.05);
}
.summary-cell:last-child { border-right: none; }
.s-val { font-family: 'JetBrains Mono', monospace; font-size: 1.15rem; font-weight: 700; color: #e2e8f0; }
.s-lbl { font-size: 0.52rem; color: #475569; text-transform: uppercase; letter-spacing: 1px; font-weight: 700; margin-top: 1px; }

/* ── Model Info ── */
.model-grid {
    display: grid; grid-template-columns: repeat(3,1fr); gap: 0.5rem;
    margin: 0.5rem 0;
}
@media(max-width:600px){ .model-grid{grid-template-columns:repeat(2,1fr);} }
.model-item {
    background: rgba(12,16,32,0.6); border: 1px solid rgba(99,102,241,0.06);
    border-radius: 10px; padding: 0.7rem 0.8rem;
    backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
}
.mi-label { font-size: 0.55rem; color: #475569; text-transform: uppercase; letter-spacing: 1px; font-weight: 700; }
.mi-value { font-size: 0.82rem; color: #cbd5e1; font-weight: 600; margin-top: 2px; }

/* ── Project Stats Footer ── */
.stats-strip {
    display: grid; grid-template-columns: repeat(4,1fr); gap: 0;
    background: rgba(12,16,32,0.6); border: 1px solid rgba(99,102,241,0.06);
    border-radius: 14px; overflow: hidden; margin: 1.5rem 0 0.5rem;
}
.stats-cell {
    text-align: center; padding: 0.8rem 0.4rem;
    border-right: 1px solid rgba(99,102,241,0.04);
}
.stats-cell:last-child { border-right: none; }
.st-val { font-family: 'JetBrains Mono', monospace; font-size: 1rem; font-weight: 700; color: #94a3b8; }
.st-lbl { font-size: 0.5rem; color: #334155; text-transform: uppercase; letter-spacing: 1px; font-weight: 700; margin-top: 1px; }

/* ── Error Card ── */
.err-card {
    background: rgba(20,10,10,0.6); border: 1px solid rgba(239,68,68,0.12);
    border-radius: 16px; padding: 1.2rem; text-align: center; height: 100%;
    backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
}
.err-msg { color: #f87171; font-size: 0.78rem; font-weight: 600; padding: 0.8rem 0; }

/* ── Sample Buttons ── */
.sample-hint {
    color: #475569; font-size: 0.75rem; font-weight: 500;
    margin-bottom: 0.5rem; text-align: center;
}

/* ── Footer ── */
.app-footer {
    text-align: center; padding: 1.5rem 0 0.5rem;
    color: #1e293b; font-size: 0.68rem; font-weight: 500; letter-spacing: 0.3px;
}
.app-footer span { color: #334155; }

/* ── Streamlit Overrides ── */
.stFileUploader > div { border: none !important; }
.stFileUploader [data-testid="stFileUploaderDropzone"] {
    background: rgba(12,16,32,0.4) !important;
    border: 2px dashed rgba(99,102,241,0.15) !important;
    border-radius: 14px !important;
}
.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; padding: 0.55rem 1.6rem !important;
    font-weight: 700 !important; font-size: 0.8rem !important;
    letter-spacing: 0.3px !important; transition: all 0.2s ease !important;
    box-shadow: 0 4px 14px rgba(79,70,229,0.2) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(79,70,229,0.3) !important;
}
.stDownloadButton > button {
    background: linear-gradient(135deg, #059669, #10b981) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; padding: 0.5rem 1.2rem !important;
    font-weight: 700 !important; font-size: 0.8rem !important;
    box-shadow: 0 4px 14px rgba(5,150,105,0.15) !important;
}
.stDownloadButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(5,150,105,0.25) !important;
}
div[data-testid="stImage"] img { border-radius: 10px; border: 1px solid rgba(99,102,241,0.08); }
hr { border-color: rgba(99,102,241,0.05) !important; }
#MainMenu, header, footer { visibility: hidden; }
div[data-testid="stExpander"] {
    background: transparent !important;
    border: 1px solid rgba(99,102,241,0.08) !important;
    border-radius: 14px !important;
}
div[data-testid="stExpander"] summary {
    color: #94a3b8 !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
}
</style>
"""


# ──────────────────────────────────────────────────────────────
# 7. UI Components
# ──────────────────────────────────────────────────────────────
def render_hero() -> None:
    st.markdown("""
    <div class="hero">
        <div class="hero-badge">⚡ Deep Learning · Computer Vision · PyTorch</div>
        <h1>CAPTCHA OCR</h1>
        <p class="hero-sub">
            ResNet-18 trained entirely from scratch on distorted CAPTCHA images.
            Upload images for instant character sequence recognition with
            per-character confidence scores.
        </p>
        <div class="kpi-row">
            <div class="kpi">
                <div class="kpi-icon">🎯</div>
                <div class="kpi-val">99.94%</div>
                <div class="kpi-lbl">Char Accuracy</div>
            </div>
            <div class="kpi">
                <div class="kpi-icon">✅</div>
                <div class="kpi-val">99.70%</div>
                <div class="kpi-lbl">Seq Accuracy</div>
            </div>
            <div class="kpi">
                <div class="kpi-icon">📉</div>
                <div class="kpi-val">0.06%</div>
                <div class="kpi-lbl">Error Rate</div>
            </div>
            <div class="kpi">
                <div class="kpi-icon">🧠</div>
                <div class="kpi-val">11.2M</div>
                <div class="kpi-lbl">Parameters</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def build_conf_bars_html(text: str, char_confs: list[float]) -> str:
    """Build HTML for character confidence mini-bars."""
    rows = ""
    for ch, cf in zip(text, char_confs):
        color = "#4ade80" if cf >= 95 else ("#818cf8" if cf >= 80 else "#f59e0b")
        rows += f"""
        <div class="conf-row">
            <span class="conf-char">{ch}</span>
            <div class="conf-track">
                <div class="conf-fill" style="width:{cf}%; background:linear-gradient(90deg,{color},{color}88);"></div>
            </div>
            <span class="conf-pct">{cf}%</span>
        </div>"""
    return f'<div class="conf-bars">{rows}</div>'


def render_result_card_html(filename: str, result: dict) -> str:
    """Return the inner HTML for a result card (no image — image rendered by Streamlit)."""
    conf_html = build_conf_bars_html(result["text"], result["char_confs"])
    return f"""
    <div class="rcard-body">
        <div class="rcard-filename">📄 {filename}</div>
        <div class="pred-string">{result['text']}</div>
        {conf_html}
        <div class="badge-row">
            <span class="badge b-ok">✓ Success</span>
            <span class="badge b-time">⏱ {result['time_ms']:.0f} ms</span>
            <span class="badge b-conf">⬥ {result['confidence']:.1f}%</span>
        </div>
    </div>"""


def render_summary(results: list[dict]) -> None:
    valid = [r for r in results if r.get("text")]
    total, success = len(results), len(valid)
    avg_time = np.mean([r["time_ms"] for r in valid]) if valid else 0
    avg_conf = np.mean([r["confidence"] for r in valid]) if valid else 0
    st.markdown(f"""
    <div class="summary">
        <div class="summary-cell"><div class="s-val">{total}</div><div class="s-lbl">Processed</div></div>
        <div class="summary-cell"><div class="s-val">{success}/{total}</div><div class="s-lbl">Successful</div></div>
        <div class="summary-cell"><div class="s-val">{avg_time:.0f} ms</div><div class="s-lbl">Avg Latency</div></div>
        <div class="summary-cell"><div class="s-val">{avg_conf:.1f}%</div><div class="s-lbl">Avg Confidence</div></div>
    </div>
    """, unsafe_allow_html=True)


def render_model_info() -> None:
    """Expandable model architecture section."""
    with st.expander("🧠  Model Architecture & Training Details"):
        st.markdown("""
        <div class="model-grid">
            <div class="model-item"><div class="mi-label">Architecture</div><div class="mi-value">ResNet-18</div></div>
            <div class="model-item"><div class="mi-label">Initialization</div><div class="mi-value">Random (from scratch)</div></div>
            <div class="model-item"><div class="mi-label">Input</div><div class="mi-value">1 × 100 × 200 grayscale</div></div>
            <div class="model-item"><div class="mi-label">Output</div><div class="mi-value">6 × 31 logits</div></div>
            <div class="model-item"><div class="mi-label">Vocabulary</div><div class="mi-value">31 chars (2-9, A-Z excl. I,L,O)</div></div>
            <div class="model-item"><div class="mi-label">Optimizer</div><div class="mi-value">AdamW (lr=3e-4)</div></div>
            <div class="model-item"><div class="mi-label">Loss</div><div class="mi-value">CE + 0.1 label smoothing</div></div>
            <div class="model-item"><div class="mi-label">Scheduler</div><div class="mi-value">ReduceLROnPlateau</div></div>
            <div class="model-item"><div class="mi-label">Epochs</div><div class="mi-value">40</div></div>
        </div>
        """, unsafe_allow_html=True)


def render_project_stats() -> None:
    st.markdown("""
    <div class="stats-strip">
        <div class="stats-cell"><div class="st-val">20,000</div><div class="st-lbl">Training Images</div></div>
        <div class="stats-cell"><div class="st-val">2,000</div><div class="st-lbl">Validation Images</div></div>
        <div class="stats-cell"><div class="st-val">6</div><div class="st-lbl">Wrong Predictions</div></div>
        <div class="stats-cell"><div class="st-val">40</div><div class="st-lbl">Epochs Trained</div></div>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# 8. Main Application
# ──────────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(page_title="CAPTCHA OCR — ResNet18", page_icon="🔍", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    render_hero()

    # Load model
    with st.spinner("Loading model weights…"):
        model = load_model()

    # Model info (expandable)
    render_model_info()

    # ── Upload ──
    st.markdown('<div class="sec-title"><span class="dot"></span> Upload CAPTCHA Images</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        f"Drop up to {MAX_UPLOADS} CAPTCHA images (PNG / JPG / JPEG)",
        type=ALLOWED_TYPES,
        accept_multiple_files=True,
        key="uploader",
    )

    if uploaded and len(uploaded) > MAX_UPLOADS:
        st.warning(f"Max {MAX_UPLOADS} images. Only the first {MAX_UPLOADS} will be processed.")
        uploaded = uploaded[:MAX_UPLOADS]

    # Session state
    if "results" not in st.session_state:
        st.session_state.results = []

    # ── Predict button ──
    if uploaded:
        c1, c2, _ = st.columns([1, 1, 3])
        with c1:
            predict_clicked = st.button("🚀  Predict All", use_container_width=True)
        with c2:
            if st.button("✕  Clear", use_container_width=True):
                st.session_state.results = []
                st.rerun()

        if predict_clicked:
            results = []
            with st.status("🔍 Running CAPTCHA OCR…", expanded=True) as status:
                st.write("📥 Loading uploaded images…")
                time.sleep(0.3)

                for i, f in enumerate(uploaded):
                    raw = f.read()
                    f.seek(0)
                    tensor = preprocess(raw)

                    st.write(f"🔬 Analyzing image {i+1}/{len(uploaded)}: `{f.name}`")

                    if tensor is not None:
                        res = predict(model, tensor)
                        res["filename"] = f.name
                        res["raw"] = raw
                    else:
                        res = {"filename": f.name, "text": None, "confidence": 0.0,
                               "char_confs": [], "time_ms": 0.0, "raw": raw}
                    results.append(res)

                st.write("✅ All images processed!")
                status.update(label=f"✅ {len(results)} image(s) processed!", state="complete")

            st.session_state.results = results
            st.rerun()

    # ── Display Results ──
    if st.session_state.results:
        results = st.session_state.results
        render_summary(results)
        st.markdown('<div class="sec-title"><span class="dot"></span> Prediction Results</div>', unsafe_allow_html=True)

        # Responsive grid: 3 columns for 3+, 2 for 2, 1 for 1
        n = len(results)
        cols_per_row = 3 if n >= 3 else n

        for row_start in range(0, n, cols_per_row):
            row_items = results[row_start : row_start + cols_per_row]
            cols = st.columns(cols_per_row, gap="medium")

            for idx, r in enumerate(row_items):
                with cols[idx]:
                    if r.get("text"):
                        # Image
                        pil = Image.open(io.BytesIO(r["raw"]))
                        st.image(pil, use_container_width=True)
                        # Card body
                        card_html = render_result_card_html(r["filename"], r)
                        st.markdown(f'<div class="rcard">{card_html}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="err-card">
                            <div class="rcard-filename">📄 {r['filename']}</div>
                            <div class="err-msg">⚠ Unable to process image</div>
                            <div class="badge-row"><span class="badge b-err">✗ Error</span></div>
                        </div>
                        """, unsafe_allow_html=True)

        # CSV export
        st.markdown("---")
        rows = [
            {"filename": r["filename"], "prediction": r.get("text", "ERROR"),
             "confidence_%": round(r.get("confidence", 0), 2),
             "inference_ms": round(r.get("time_ms", 0), 2),
             "status": "success" if r.get("text") else "error"}
            for r in results
        ]
        csv = pd.DataFrame(rows).to_csv(index=False).encode("utf-8")
        st.download_button("📥  Download Results as CSV", data=csv,
                           file_name="captcha_predictions.csv", mime="text/csv",
                           use_container_width=True)

    # ── Project Stats + Footer ──
    render_project_stats()
    st.markdown("""
    <div class="app-footer">
        Built with <span>PyTorch</span> & <span>Streamlit</span> &nbsp;·&nbsp;
        ResNet-18 from scratch &nbsp;·&nbsp; No pretrained weights
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
