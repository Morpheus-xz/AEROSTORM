#!/usr/bin/env python3
import os
import hashlib
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageOps, ImageEnhance

import torch
import torch.nn as nn
import streamlit as st

from openai import OpenAI
from dotenv import load_dotenv

# =======================
# 1. CONFIG & INIT
# =======================
st.set_page_config(
    page_title="AEROSTORM | Command Console",
    page_icon="🌪️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

load_dotenv()

# API Key Check & Client Init
if not os.getenv("OPENAI_API_KEY"):
    st.warning("⚠️ SYSTEM ALERT: OpenAI API Key missing in environment variables.")
else:
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

client = OpenAI()

BASE_DIR = Path(__file__).resolve().parent
# Ensure this matches where you saved your model in train_model.py
MODEL_PATH = BASE_DIR / "model" / "final_model.pth"
IMG_SIZE = (224, 224)

# =======================
# 2. 🧠 AEROSTORM MITRA: SYSTEM INTELLIGENCE
# =======================

SYSTEM_PROMPT = """
You are AEROSTORM MITRA, an advanced AI Disaster Management Commander specialized in Indian Cyclone Safety.
Your goal is to translate technical satellite telemetry into life-saving, actionable instructions.

---
CURRENT TELEMETRY CONTEXT:
Danger Level: {danger_level}
Cyclone Intensity Score: {intensity}
Visual Analysis: {visual_insights}
---

STRICT BEHAVIORAL PROTOCOLS:
1. TONE: Urgent but Calm. Authoritative. Do not use flowery or poetic language. Be direct.
2. AUDIENCE ADAPTATION:
   - If User is 'Fisherman': Focus on boats, nets, sea conditions, and coastal evacuation. Use simple terms (Hinglish/English).
   - If User is 'General Citizen': Focus on home safety, windows, food stock, and emergency kits.
   - If User is 'Official': Focus on evacuation zones, resource deployment, and crowd control.
   - If User is 'Meteorologist': Use technical jargon, discuss convection patterns and shear.
3. LANGUAGE:
   - Output in English (unless explicitly asked for Hindi).
   - If the danger is HIGH, use UPPERCASE for critical warnings like "EVACUATE NOW".
4. SAFETY GUARDRAILS:
   - NEVER suggest "waiting and watching" if Danger Level is HIGH.
   - Do NOT provide specific medical prescriptions (only basic First Aid).
   - Do NOT make up weather predictions that contradict the provided Telemetry.

FORMATTING:
- Use Bullet points for readability.
- Keep responses under 6 sentences unless asked for a detailed plan.
- End with one morale-boosting, resilient phrase suitable for the Indian context (e.g., "Stay strong, stay safe.").
"""


def genai_text(user_prompt, danger_level="UNKNOWN", intensity=0.0, visual_insights="N/A", role="General User"):
    """
    Wraps the user query with the System Prompt to ensure safety and persona consistency.
    """
    # Fill the slots in the System Prompt with live data
    formatted_system = SYSTEM_PROMPT.format(
        danger_level=danger_level,
        intensity=intensity,
        visual_insights=visual_insights
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": formatted_system},
                {"role": "user", "content": f"User Role: {role}\nRequest: {user_prompt}"}
            ],
            temperature=0.3  # Low temperature for deterministic safety
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠ COMMUNICATION LINK SEVERED: {e}"


# =======================
# 3. CSS - SCI-FI UI
# =======================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;500;700&family=Inter:wght@300;400;600&display=swap');

    :root {
        --primary: #00f2ff;
        --bg-dark: #020617;
        --text-main: #e2e8f0;
        --text-muted: #94a3b8;
    }

    .stApp {
        background: radial-gradient(circle at top left, #1e293b 0%, #020617 40%, #000000 100%);
        background-attachment: fixed;
        font-family: 'Inter', sans-serif;
        color: var(--text-main);
    }

    h1, h2, h3, h4 {
        font-family: 'Rajdhani', sans-serif;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: white;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1600px;
    }
    header, footer { visibility: hidden; }

    /* --- SCROLLABLE BOXES --- */
    .scroller-box-base {
        height: 400px;
        overflow-y: auto;
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 10px;
        position: relative;
    }
    .scroller-box-base::-webkit-scrollbar { width: 6px; }
    .scroller-box-base::-webkit-scrollbar-track { background: rgba(0,0,0,0.2); }
    .scroller-box-base::-webkit-scrollbar-thumb { background: var(--primary); border-radius: 3px; }

    /* Analysis: Transparent Blue */
    .analysis-box {
        background: rgba(0, 242, 255, 0.05); 
        border: 1px solid rgba(0, 242, 255, 0.2);
        box-shadow: inset 0 0 20px rgba(0, 242, 255, 0.02);
    }

    /* Protocol: Transparent Green */
    .protocol-box {
        background: rgba(16, 185, 129, 0.05); 
        border: 1px solid rgba(16, 185, 129, 0.2);
        box-shadow: inset 0 0 20px rgba(16, 185, 129, 0.02);
    }

    /* Chat: Neutral Transparent */
    .chat-box {
        background: rgba(255, 255, 255, 0.02); 
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .box-content {
        font-size: 0.95rem;
        line-height: 1.6;
        color: #e2e8f0;
        white-space: pre-wrap;
    }

    .standby-text {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        color: var(--text-muted);
        font-family: 'Rajdhani';
        opacity: 0.6;
        text-align: center;
        width: 100%;
        font-size: 1.2rem;
    }

    .panel-header {
        color: var(--primary);
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 8px;
        text-shadow: 0 0 15px rgba(0, 242, 255, 0.3);
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .chat-bubble-user {
        background: linear-gradient(90deg, #0ea5e9, #2563eb);
        color: white;
        padding: 8px 12px;
        border-radius: 12px 12px 0 12px;
        margin: 8px 0;
        text-align: right;
        font-size: 0.9rem;
        float: right;
        clear: both;
        max-width: 85%;
    }
    .chat-bubble-bot {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.1);
        color: #e2e8f0;
        padding: 8px 12px;
        border-radius: 12px 12px 12px 0;
        margin: 8px 0;
        text-align: left;
        font-size: 0.9rem;
        float: left;
        clear: both;
        max-width: 85%;
    }
    .clearfix::after { content: ""; clear: both; display: table; }

    .tele-label {
        font-family: 'Rajdhani', sans-serif;
        font-size: 0.8rem;
        color: var(--text-muted);
        text-align: center;
        margin-top: 5px;
        letter-spacing: 1px;
    }
    [data-testid="stImage"] img {
        border-radius: 4px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .risk-display {
        text-align: center;
        padding: 10px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        height: 100%;
        background: rgba(0,0,0,0.3);
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.05);
    }
    .risk-title { 
        font-size: 2.5rem; 
        font-weight: 800; 
        font-family: 'Rajdhani'; 
        text-shadow: 0 0 25px currentColor; 
        margin: 0;
        line-height: 1;
    }

    .stButton button {
        width: 100%;
        background: linear-gradient(90deg, rgba(0,242,255,0.1), transparent);
        border: 1px solid var(--primary);
        color: var(--primary);
        font-family: 'Rajdhani', sans-serif;
        font-weight: 600;
        text-transform: uppercase;
        transition: 0.3s;
        border-radius: 6px;
        letter-spacing: 1px;
    }
    .stButton button:hover {
        background: var(--primary);
        color: #000;
        box-shadow: 0 0 20px var(--primary);
    }
    .stTextInput input {
        border-radius: 8px;
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# =======================
# 4. BEAST MODEL ARCHITECTURE
# =======================
class BeastBlock(nn.Module):
    """Conv -> BatchNorm -> LeakyReLU -> MaxPool"""

    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True)
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CycloneBeast(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            BeastBlock(1, 32),  # 112x112
            BeastBlock(32, 64),  # 56x56
            BeastBlock(64, 128),  # 28x28
            BeastBlock(128, 256),  # 14x14
            BeastBlock(256, 512),  # 7x7
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 3)
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        feats = self.features(x)
        return self.classifier(feats), self.regressor(feats).view(-1)


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if MODEL_PATH.exists():
        try:
            ckpt = torch.load(MODEL_PATH, map_location=device)
            model = CycloneBeast().to(device)
            model.load_state_dict(ckpt["state"])
            model.eval()
            return model, device
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None, device
    else:
        return None, device


model, device = load_model()


# =======================
# 5. IMAGE HELPERS
# =======================
def preprocess_image(img_pil: Image.Image) -> torch.Tensor:
    """Matches the Beast Mode training: Grayscale + Contrast + 1 Channel"""
    img = img_pil.resize(IMG_SIZE)
    gray = ImageOps.grayscale(img)
    enhancer = ImageEnhance.Contrast(gray)
    gray = enhancer.enhance(1.5)

    arr = np.array(gray).astype("float32") / 255.0
    tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0).float().to(device)
    return tensor


def build_cloud_mask(gray_arr: np.ndarray) -> np.ndarray:
    # Top 45% brightest pixels = Cloud tops
    thr = np.percentile(gray_arr, 55)
    return (gray_arr > thr).astype("uint8")


def build_hotspot_map(gray_arr: np.ndarray) -> np.ndarray:
    # Non-linear scaling to emphasize the brightest spots (Eye/Wall)
    norm = gray_arr / (gray_arr.max() + 1e-8)
    g = np.power(norm, 2.2)
    return (g * 255).astype("uint8")


def to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.astype("uint8"))


def compute_risk_meter(pred_class: int, intensity: float):
    if pred_class == 0:
        label, color, base = "LOW DANGER", "#10b981", 25
    elif pred_class == 1:
        label, color, base = "MEDIUM DANGER", "#f59e0b", 55
    else:
        label, color, base = "HIGH DANGER", "#ef4444", 85

    # Normalize intensity roughly 0-100 for bar display
    adj = (intensity - 50.0) / 2.5
    percent = int(max(5, min(100, base + adj)))
    return label, color, percent


# =======================
# 6. MAIN UI LOGIC
# =======================

# Header
st.markdown("""
    <div style='text-align:center; margin-bottom: 30px;'>
        <h1 style='font-size: 3.5rem; margin-bottom:0; text-shadow: 0 0 30px #00f2ff;'>AEROSTORM</h1>
        <div style='color: #00f2ff; letter-spacing: 5px; font-size: 0.9rem; font-family:"Rajdhani";'>ADVANCED CYCLONE TELEMETRY SYSTEM</div>
    </div>
""", unsafe_allow_html=True)

# Session States
if "insight_cache" not in st.session_state: st.session_state.insight_cache = {}
if "safety_cache" not in st.session_state: st.session_state.safety_cache = {}
if "chat_history" not in st.session_state: st.session_state.chat_history = []

# File Upload
col_u1, col_u2, col_u3 = st.columns([1, 2, 1])
with col_u2:
    uploaded = st.file_uploader("INITIALIZE UPLINK (Upload IR Image)", type=["jpg", "png", "jpeg"])

if uploaded and model:
    # --- PROCESSING ---
    img_bytes = uploaded.getvalue()
    current_img_key = hashlib.sha1(img_bytes).hexdigest()

    orig = Image.open(uploaded).convert("RGB")

    # Generate Visual Proofs
    gray_visual = ImageOps.grayscale(orig).resize(IMG_SIZE)
    gray_np = np.array(gray_visual)

    mask_pil = to_pil(build_cloud_mask(gray_np) * 255)
    hotspot_pil = to_pil(build_hotspot_map(gray_np))

    # Neural Inference
    with torch.no_grad():
        x = preprocess_image(orig)
        out_cls, out_reg = model(x)
        p_cls = int(torch.argmax(out_cls, dim=1)[0].item())
        intensity = float(out_reg[0].item())

    label, color, pct = compute_risk_meter(p_cls, intensity)

    # --- ROW 1: TELEMETRY & RISK ---
    c1, c2 = st.columns([2.5, 1])
    with c1:
        st.markdown("<div class='panel-header'>📡 SATELLITE TELEMETRY</div>", unsafe_allow_html=True)
        ic1, ic2, ic3 = st.columns(3)
        with ic1:
            st.image(orig, use_container_width=True)
            st.markdown("<div class='tele-label'>OPTICAL IR</div>", unsafe_allow_html=True)
        with ic2:
            st.image(mask_pil, use_container_width=True)
            st.markdown("<div class='tele-label'>CLOUD MASK</div>", unsafe_allow_html=True)
        with ic3:
            st.image(hotspot_pil, use_container_width=True)
            st.markdown("<div class='tele-label'>HOTSPOT MAP</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='panel-header'>⚠️ THREAT LEVEL</div>", unsafe_allow_html=True)
        st.markdown(f"""
            <div class='risk-display'>
                <div class='risk-title' style='color:{color};'>{label}</div>
                <div style='color:{color}; margin-bottom:15px; font-family:"Rajdhani"; letter-spacing: 2px;'>INTENSITY: {intensity:.2f}</div>
                <div style='width:100%; background:rgba(255,255,255,0.1); height:12px; border-radius:6px; overflow:hidden;'>
                    <div style='width:{pct}%; background:{color}; height:100%; box-shadow:0 0 15px {color};'></div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)

    # --- ROW 2: GEN-AI INTERACTION ---
    col_a, col_b, col_c = st.columns(3, gap="medium")

    # 1. ANALYSIS BOX (BLUE)
    with col_a:
        st.markdown("<div class='panel-header'>🧬 SCIENTIFIC ANALYSIS</div>", unsafe_allow_html=True)

        insight_text = st.session_state.insight_cache.get(current_img_key)

        # Display Box
        content_html = f"<div class='box-content'>{insight_text}</div>" if insight_text else "<div class='standby-text'>SYSTEM STANDBY<br>AWAITING ANALYSIS</div>"
        st.markdown(f"<div class='scroller-box-base analysis-box'>{content_html}</div>", unsafe_allow_html=True)

        if st.button("RUN ANALYSIS", key="btn_run", use_container_width=True):
            with st.spinner("Processing Telemetry..."):
                res = genai_text(
                    "Analyze the cyclone structure based on the visuals. Explain the Hotspot Map and Cloud Mask findings.",
                    danger_level=label,
                    intensity=intensity,
                    visual_insights=f"Cloud Mask shows {pct}% coverage density.",
                    role="Meteorologist"
                )
                st.session_state.insight_cache[current_img_key] = res
                st.rerun()

    # 2. PROTOCOL BOX (GREEN)
    with col_b:
        c_h, c_d = st.columns([1, 1.2])
        with c_h:
            st.markdown("<div class='panel-header'>🛡️ PROTOCOL</div>", unsafe_allow_html=True)
        with c_d:
            role_options = ["General Citizen", "Fisherman", "Farmer", "Official"]
            role = st.selectbox("Role", role_options, label_visibility="collapsed")

        key_p = (current_img_key, role)
        plan_text = st.session_state.safety_cache.get(key_p)

        content_html = f"<div class='box-content'>{plan_text}</div>" if plan_text else f"<div class='standby-text'>AWAITING GENERATION FOR<br>{role.upper()}</div>"
        st.markdown(f"<div class='scroller-box-base protocol-box'>{content_html}</div>", unsafe_allow_html=True)

        if st.button(f"GENERATE PLAN ({role})", key="btn_proto", use_container_width=True):
            with st.spinner("Synthesizing Safety Protocols..."):
                current_insight = st.session_state.insight_cache.get(current_img_key, "Visual data analysis pending.")
                res = genai_text(
                    f"Generate a strict safety protocol checklist for a {role}. Prioritize immediate life-saving actions.",
                    danger_level=label,
                    intensity=intensity,
                    visual_insights=current_insight,
                    role=role
                )
                st.session_state.safety_cache[key_p] = res
                st.rerun()

    # 3. CHAT BOX (NEUTRAL)
    with col_c:
        st.markdown("<div class='panel-header'>🤖 MITRA UPLINK</div>", unsafe_allow_html=True)

        chat_html = ""
        if not st.session_state.chat_history:
            chat_html = "<div class='standby-text'>CHANNEL OPEN</div>"
        else:
            for msg in st.session_state.chat_history:
                if msg['role'] == 'user':
                    chat_html += f"<div class='chat-bubble-user'>{msg['content']}</div><div class='clearfix'></div>"
                else:
                    chat_html += f"<div class='chat-bubble-bot'>{msg['content']}</div><div class='clearfix'></div>"

        st.markdown(f"<div class='scroller-box-base chat-box' id='chat-box'>{chat_html}</div>", unsafe_allow_html=True)

        q = st.chat_input("Transmit Query...")
        if q:
            st.session_state.chat_history.append({"role": "user", "content": q})

            # Context Gathering
            curr_i = st.session_state.insight_cache.get(current_img_key, "N/A")

            ans = genai_text(
                q,
                danger_level=label,
                intensity=intensity,
                visual_insights=curr_i,
                role=role  # Maintain context of selected role
            )

            st.session_state.chat_history.append({"role": "assistant", "content": ans})
            st.rerun()

elif not uploaded:
    # Standby Screen
    st.markdown("""
    <div style='display:flex; justify-content:center; align-items:center; height:400px; margin-top:50px;'>
        <div style='text-align:center; color: var(--text-muted);'>
            <h2 style='color: var(--text-muted); opacity: 0.7; letter-spacing: 4px;'>SYSTEM OFFLINE</h2>
            <p style='letter-spacing: 2px; font-family:"Rajdhani"; font-size: 1.1rem; color: #475569;'>
                UPLOAD SATELLITE IMAGERY TO INITIALIZE COMMAND CONSOLE
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

elif uploaded and not model:
    st.error("🚨 CRITICAL ERROR: Model file not found. Please train 'CycloneBeast' and save to 'model/final_model.pth'.")