"""
╔══════════════════════════════════════════════════════════════╗
║     Assistant For Detection Of Retinal Diseases              ║
║     Built with Streamlit · EfficientNetB3 · Grad-CAM · Grok ║
╚══════════════════════════════════════════════════════════════╝
"""

# ==============================
# Imports
# ==============================
import os

import cv2
import gdown
import numpy as np
import requests
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model

# ==============================
# Constants
# ==============================
MODEL_PATH = "efficientnetb3_retinal.h5"
FILE_ID    = "1qnrKRAWa7UU5YbtT2UqGDbJij7uH6dIz"   # ← replace with your actual Google Drive file ID

CLASS_NAMES = [
    "Diabetic Retinopathy",
    "Disc Edema",
    "Healthy",
    "Myopia",
    "Pterygium",
    "Retinal Detachment",
    "Retinitis Pigmentosa",
]

SEVERITY_COLOR = {
    "Diabetic Retinopathy": "#ef4444",
    "Disc Edema":           "#ef4444",
    "Healthy":              "#22c55e",
    "Myopia":               "#f59e0b",
    "Pterygium":            "#f59e0b",
    "Retinal Detachment":   "#dc2626",
    "Retinitis Pigmentosa": "#ef4444",
}

DISEASE_INFO = {
    "Diabetic Retinopathy": {
        "icon": "🩺",
        "desc": "Damage to the retinal blood vessels caused by diabetes.",
        "action": "Consult an ophthalmologist immediately for further evaluation.",
    },
    "Disc Edema": {
        "icon": "🧠",
        "desc": "Swelling of the optic disc, often indicating raised intracranial pressure.",
        "action": "Urgent neurological and ophthalmological assessment required.",
    },
    "Healthy": {
        "icon": "✅",
        "desc": "No signs of retinal disease detected.",
        "action": "Maintain regular annual eye check-ups.",
    },
    "Myopia": {
        "icon": "👓",
        "desc": "Nearsightedness caused by elongation of the eyeball.",
        "action": "Consult an optometrist for corrective lenses.",
    },
    "Pterygium": {
        "icon": "🔬",
        "desc": "A benign growth of conjunctival tissue onto the cornea.",
        "action": "Monitor growth; surgical removal if vision is affected.",
    },
    "Retinal Detachment": {
        "icon": "🚨",
        "desc": "The retina separates from the underlying tissue — a medical emergency.",
        "action": "Seek emergency ophthalmological care immediately.",
    },
    "Retinitis Pigmentosa": {
        "icon": "🧬",
        "desc": "A genetic disorder causing progressive retinal degeneration.",
        "action": "Consult a retinal specialist for management and genetic counseling.",
    },
}

XAI_API_URL = "https://api.x.ai/v1/chat/completions"

PROMPT_TEMPLATE = """You are an ophthalmology AI assistant.

Write exactly 5 short medical lines about this eye disease prediction:

Prediction: {disease}
Confidence: {confidence:.1f}%

Structure (5 lines only, no headers, no repetition):
1. Prediction statement.
2. Short clinical definition.
3. Key symptoms the patient may notice.
4. Severity level (Mild / Moderate / Severe / Emergency).
5. Recommended next step."""


# ==============================
# Page Configuration
# ==============================
st.set_page_config(
    page_title="Assistant For Detection Of Retinal Diseases",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==============================
# Custom CSS
# ==============================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    html, body, [class*="css"], .stApp, .main {
        background-color: #f0fdf4 !important;
        color: #166534 !important;
    }
    .stApp {
        background: linear-gradient(150deg, #f0fdf4 0%, #dcfce7 40%, #f0fdf4 100%) !important;
        color: #166534;
    }

    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding: 0 2rem 4rem; max-width: 1200px; }

    /* ── Hero ── */
    .hero {
        position: relative;
        text-align: center;
        padding: 3.5rem 2rem 2.5rem;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        inset: 0;
        background:
            radial-gradient(ellipse 70% 50% at 50% 0%, rgba(22,163,74,0.12) 0%, transparent 65%),
            radial-gradient(ellipse 35% 25% at 10% 85%, rgba(21,128,61,0.08) 0%, transparent 60%),
            radial-gradient(ellipse 40% 30% at 90% 70%, rgba(74,222,128,0.1) 0%, transparent 55%);
        pointer-events: none;
    }
    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: clamp(2.4rem, 5vw, 4rem);
        font-weight: 800;
        line-height: 1.05;
        letter-spacing: -0.02em;
        color: #14532d;
        margin: 0 0 1rem;
    }
    .hero-title span {
        background: linear-gradient(135deg, #16a34a 0%, #15803d 60%, #4ade80 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .divider {
        height: 1.5px;
        background: linear-gradient(90deg, transparent, rgba(22,163,74,0.4), transparent);
        margin: 0 0 2.5rem;
    }

    /* ── Upload Section ── */
    .upload-section {
        background: #ffffff;
        border: 2px dashed rgba(22,163,74,0.35);
        border-radius: 20px;
        padding: 2.5rem 2rem;
        text-align: center;
        margin-bottom: 2rem;
        transition: all 0.2s ease;
        box-shadow: 0 2px 12px rgba(22,163,74,0.07);
    }
    .upload-section:hover {
        border-color: #16a34a;
        background: #f0fdf4;
        box-shadow: 0 4px 20px rgba(22,163,74,0.12);
    }
    .upload-label {
        font-family: 'Syne', sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
        color: #15803d;
        margin-bottom: 0.4rem;
    }
    .upload-hint { font-size: 0.82rem; color: #6aaa85; }

    [data-testid="stFileUploader"] { background: transparent !important; }
    [data-testid="stFileUploader"] > div { border: none !important; background: transparent !important; padding: 0 !important; }
    [data-testid="stFileUploader"] label { color: #16a34a !important; font-size: 0.9rem; }

    /* ── Image Cards ── */
    .img-card {
        background: #ffffff;
        border: 1px solid #bbf7d0;
        border-radius: 14px;
        padding: 0.6rem 0.6rem 0.5rem;
        text-align: center;
        max-width: 220px;
        margin: 0 auto;
        box-shadow: 0 4px 16px rgba(22,163,74,0.1);
    }
    .img-card-label {
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: #6aaa85;
        margin-top: 0.5rem;
    }
    [data-testid="stImage"] img {
        border-radius: 10px;
        width: 100%;
        max-height: 200px;
        object-fit: cover;
    }

    /* ── Widgets ── */
    .stSelectbox label, .stTextInput label, .stToggle label,
    .stRadio label, .stExpander summary, p, span, div {
        color: #166534 !important;
    }
    .stSelectbox > div > div,
    .stTextInput > div > div > input {
        background: #ffffff !important;
        border-color: #bbf7d0 !important;
        color: #14532d !important;
    }

    /* ── Progress Bar ── */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #22c55e, #16a34a) !important;
        border-radius: 999px !important;
    }
    .stProgress > div > div {
        background: #dcfce7 !important;
        border-radius: 999px !important;
        height: 8px !important;
    }

    /* ── Confidence Display ── */
    .confidence-label {
        font-size: 0.78rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #6aaa85;
        margin-bottom: 0.5rem;
    }
    .confidence-value {
        font-family: 'Syne', sans-serif;
        font-size: 2.4rem;
        font-weight: 800;
        color: #14532d;
        line-height: 1;
    }
    .confidence-value span { font-size: 1rem; font-weight: 400; color: #6aaa85; }

    /* ── Disease Card ── */
    .disease-card {
        background: #ffffff;
        border: 1px solid #bbf7d0;
        border-left: 4px solid #16a34a;
        border-radius: 16px;
        padding: 1.2rem 1.4rem;
        margin-top: 1rem;
        box-shadow: 0 2px 16px rgba(22,163,74,0.08);
    }
    .disease-card-title {
        font-family: 'Syne', sans-serif;
        font-size: 0.95rem;
        font-weight: 700;
        color: #15803d;
        margin-bottom: 0.4rem;
    }
    .disease-card-text { font-size: 0.85rem; color: #4b7a5e; line-height: 1.7; }

    /* ── LLM Explanation Card ── */
    .llm-card {
        background: #f0fdf4;
        border: 1px solid #86efac;
        border-left: 4px solid #4ade80;
        border-radius: 16px;
        padding: 1.2rem 1.4rem;
        margin-top: 1rem;
        box-shadow: 0 2px 16px rgba(74,222,128,0.1);
    }
    .llm-card-title {
        font-family: 'Syne', sans-serif;
        font-size: 0.95rem;
        font-weight: 700;
        color: #16a34a;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .llm-line {
        font-size: 0.86rem;
        color: #2d6a44;
        line-height: 1.75;
        margin-bottom: 0.45rem;
        padding-left: 0.6rem;
        border-left: 2px solid rgba(22,163,74,0.35);
    }
    .llm-error {
        font-size: 0.82rem;
        color: #b45309;
        background: rgba(245,158,11,0.08);
        border: 1px solid rgba(245,158,11,0.3);
        border-radius: 8px;
        padding: 0.7rem 1rem;
        margin-top: 0.5rem;
    }

    /* ── Disclaimer ── */
    .disclaimer {
        background: #fffbeb;
        border: 1px solid #fde68a;
        border-left: 4px solid #f59e0b;
        border-radius: 12px;
        padding: 0.9rem 1.2rem;
        font-size: 0.78rem;
        color: #92400e;
        text-align: center;
        margin-top: 2rem;
        line-height: 1.65;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f0fdf4 0%, #dcfce7 100%) !important;
        border-right: 2px solid #bbf7d0 !important;
    }

    /* ── Expander ── */
    .stExpander {
        background: #ffffff !important;
        border: 1px solid #bbf7d0 !important;
        border-radius: 12px !important;
    }

    /* ── Button ── */
    .stButton > button {
        background: linear-gradient(135deg, #16a34a, #15803d) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 8px rgba(22,163,74,0.3) !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 16px rgba(22,163,74,0.4) !important;
    }
</style>
""", unsafe_allow_html=True)


# ==============================
# Grok LLM Functions
# ==============================
def _clean_lines(text: str) -> str:
    """Keep the first 5 non-empty lines."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    return "\n".join(lines[:5])


def _test_grok_connection(api_key: str) -> tuple:
    """Test the xAI Grok API connection."""
    if not api_key.strip():
        return False, "❌ أدخل API Key أولاً."
    try:
        response = requests.post(
            XAI_API_URL,
            headers={
                "Authorization": f"Bearer {api_key.strip()}",
                "Content-Type": "application/json",
            },
            json={
                "model": "grok-3-mini",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "Hi"}],
            },
            timeout=10,
        )
        if response.status_code == 200:
            return True, "✅ Grok API يعمل بنجاح!"
        elif response.status_code == 401:
            return False, "❌ API Key غير صالح — تحقق من المفتاح."
        else:
            return False, f"⚠️ استجابة غير متوقعة: {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "❌ لا يمكن الاتصال — تحقق من اتصالك بالإنترنت."
    except requests.exceptions.Timeout:
        return False, "❌ انتهت المهلة — الخادم لا يستجيب."
    except Exception as e:
        return False, f"❌ خطأ: {e}"


def grok_llm_explain(disease: str, confidence: float, grok_model: str, api_key: str) -> str:
    """Generate a medical explanation using the xAI Grok API."""
    if not api_key.strip():
        return "ERROR: أدخل xAI API Key في إعدادات الشريط الجانبي."

    prompt = PROMPT_TEMPLATE.format(disease=disease, confidence=confidence * 100)

    try:
        response = requests.post(
            XAI_API_URL,
            headers={
                "Authorization": f"Bearer {api_key.strip()}",
                "Content-Type": "application/json",
            },
            json={
                "model": grok_model,
                "max_tokens": 300,
                "temperature": 0.1,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        response.raise_for_status()
        raw = response.json()["choices"][0]["message"]["content"].strip()
        return _clean_lines(raw)

    except requests.exceptions.ConnectionError:
        return "ERROR: تعذّر الاتصال بـ xAI API — تحقق من اتصالك بالإنترنت."
    except requests.exceptions.Timeout:
        return "ERROR: انتهت مهلة الاستجابة — حاول مجدداً."
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else "?"
        if status == 401:
            return "ERROR: API Key غير صالح — تحقق من المفتاح."
        if status == 429:
            return "ERROR: تجاوزت حد الطلبات — انتظر قليلاً وحاول مجدداً."
        return f"ERROR: HTTP {status} — {e}"
    except Exception as e:
        return f"ERROR: خطأ غير متوقع: {e}"


# ==============================
# Vision Model — Load & Cache
# ==============================
@st.cache_resource
def load_model_cached():
    """Download (if needed) and load the EfficientNetB3 model."""
    if not os.path.exists(MODEL_PATH):

        if FILE_ID == "YOUR_GOOGLE_DRIVE_FILE_ID":
            st.error("❌ لم يتم تعيين FILE_ID — أضف معرّف ملف Google Drive الحقيقي في الكود.")
            st.stop()

        with st.spinner("⬇️ جاري تحميل النموذج..."):
            success = False

            # Method 1: gdown with fuzzy=True (handles confirmation pages)
            try:
                gdown.download(
                    id=FILE_ID,
                    output=MODEL_PATH,
                    quiet=False,
                    fuzzy=True,
                )
                if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 5_000_000:
                    success = True
            except Exception as e:
                st.warning(f"⚠️ gdown (method 1) فشل: {e}")

            # Method 2: gdown with direct URL + confirm token
            if not success:
                try:
                    url = f"https://drive.google.com/uc?id={FILE_ID}&export=download&confirm=t"
                    gdown.download(url, MODEL_PATH, quiet=False)
                    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 5_000_000:
                        success = True
                except Exception as e:
                    st.warning(f"⚠️ gdown (method 2) فشل: {e}")

            # Method 3: requests session — bypasses large-file confirmation cookie
            if not success:
                try:
                    session = requests.Session()
                    url = f"https://drive.google.com/uc?id={FILE_ID}&export=download"
                    response = session.get(url, stream=True, timeout=30)

                    for key, value in response.cookies.items():
                        if key.startswith("download_warning"):
                            url = f"https://drive.google.com/uc?id={FILE_ID}&export=download&confirm={value}"
                            response = session.get(url, stream=True, timeout=60)
                            break

                    with open(MODEL_PATH, "wb") as f:
                        for chunk in response.iter_content(chunk_size=32768):
                            if chunk:
                                f.write(chunk)

                    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 5_000_000:
                        success = True
                except Exception as e:
                    st.warning(f"⚠️ requests fallback فشل: {e}")

            if not success:
                if os.path.exists(MODEL_PATH):
                    os.remove(MODEL_PATH)
                st.error(
                    "❌ فشل تحميل النموذج بجميع الطرق.\n\n"
                    "تحقق من:\n"
                    "- FILE_ID صحيح\n"
                    "- الملف مشارك للعموم (Anyone with the link → Viewer)\n"
                    "- حجم الملف أكبر من 5 MB"
                )
                st.stop()

    if os.path.getsize(MODEL_PATH) < 5_000_000:
        os.remove(MODEL_PATH)
        st.error("❌ ملف النموذج تالف أو غير مكتمل — أعد تشغيل التطبيق.")
        st.stop()

    try:
        return load_model(MODEL_PATH, compile=False)
    except Exception as exc:
        st.error(f"❌ فشل تحميل النموذج: {exc}")
        st.stop()


# ==============================
# Image Processing Helpers
# ==============================
def preprocess(img: Image.Image) -> np.ndarray:
    """Resize and preprocess a PIL image for EfficientNetB3."""
    img = img.resize((300, 300))
    arr = np.array(img)
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def predict(img: Image.Image, model) -> tuple[str, float, np.ndarray]:
    """Return (class_name, confidence, all_probabilities)."""
    preds = model.predict(preprocess(img))
    idx = int(np.argmax(preds[0]))
    return CLASS_NAMES[idx], float(np.max(preds)), preds[0]


def gradcam(img: Image.Image, model) -> np.ndarray:
    """Compute a Grad-CAM heatmap (BGR, uint8) for the top predicted class."""
    arr = np.array(img.resize((300, 300)))
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)

    target_layer = next(
        (layer for layer in reversed(model.layers) if isinstance(layer, tf.keras.layers.Conv2D)),
        None,
    )

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[target_layer.output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(arr)
        if isinstance(predictions, list):
            predictions = predictions[0]
        if predictions.shape[-1] == 1:
            loss = predictions[:, 0]
        else:
            class_idx = int(tf.argmax(predictions[0]).numpy())
            loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    grads = grads / (tf.reduce_mean(tf.abs(grads)) + 1e-8)

    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.reduce_sum(weights[:, None, None, :] * conv_outputs, axis=-1)[0].numpy()

    cam = np.maximum(cam, 0)
    if np.max(cam) > 0:
        cam /= np.max(cam)

    cam = np.power(cam, 0.3)
    cam = cv2.resize(cam, (300, 300))
    return cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)


def overlay_heatmap(img: Image.Image, heatmap: np.ndarray) -> np.ndarray:
    """Blend the original image with the Grad-CAM heatmap."""
    arr = np.array(img.resize((300, 300)))
    return cv2.addWeighted(arr, 0.75, heatmap, 0.25, 0)


# ==============================
# Hero Section
# ==============================
st.markdown("""
<div class="hero">
    <h1 class="hero-title">Assistant For Detection Of <span>Retinal Diseases</span></h1>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

# ==============================
# Load Model
# ==============================
model = load_model_cached()

# ==============================
# Sidebar — Grok LLM Settings
# ==============================
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Syne',sans-serif; font-size:1rem; font-weight:700;
                color:#16a34a; margin-bottom:1rem; padding-bottom:0.5rem;
                border-bottom:2px solid rgba(22,163,74,0.25);">
        🤖 إعدادات Grok (xAI)
    </div>
    """, unsafe_allow_html=True)

    enable_llm = st.toggle("🔘 تفعيل شرح LLM", value=True)

    grok_model = st.selectbox(
        "نموذج Grok",
        options=["grok-3", "grok-3-fast", "grok-3-mini", "grok-3-mini-fast"],
        index=2,
        help="اختر نموذج Grok من xAI",
    )

    grok_api_key = st.text_input(
        "🔑 xAI API Key",
        type="password",
        placeholder="xai-...",
        help="احصل على مفتاحك من: console.x.ai",
    )

    if st.button("🔌 اختبار الاتصال بـ Grok", use_container_width=True):
        ok, msg = _test_grok_connection(grok_api_key)
        (st.success if ok else st.error)(msg)

    st.markdown("""
    <div style="margin-top:1rem; font-size:0.75rem; color:#4b7a5e; line-height:2;">
        <span style="color:#15803d; font-weight:500;">للحصول على مفتاح API:</span><br>
        <a href="https://console.x.ai" target="_blank"
           style="color:#16a34a; text-decoration:none;">
            🔗 console.x.ai
        </a>
        <br><br>
        <span style="color:#15803d; font-weight:500;">النماذج المتاحة:</span><br>
        <code style="background:rgba(22,163,74,0.1); color:#15803d;
                     padding:0.15rem 0.5rem; border-radius:4px;">grok-3</code>
        &nbsp;·&nbsp;
        <code style="background:rgba(22,163,74,0.1); color:#15803d;
                     padding:0.15rem 0.5rem; border-radius:4px;">grok-3-fast</code>
        &nbsp;·&nbsp;
        <code style="background:rgba(22,163,74,0.1); color:#15803d;
                     padding:0.15rem 0.5rem; border-radius:4px;">grok-3-mini</code>
        &nbsp;·&nbsp;
        <code style="background:rgba(22,163,74,0.1); color:#15803d;
                     padding:0.15rem 0.5rem; border-radius:4px;">grok-3-mini-fast</code>
    </div>
    """, unsafe_allow_html=True)


# ==============================
# Main Layout — 3 Columns
# ==============================
diseases_col, left_col, right_col = st.columns([1, 1.1, 1.8], gap="medium")

# ── Column 1: Detectable Diseases Panel ──
with diseases_col:
    st.markdown("""
    <div style="font-family:'Syne',sans-serif; font-size:0.9rem; font-weight:700;
                color:#16a34a; margin-bottom:0.9rem; padding-bottom:0.5rem;
                border-bottom:2px solid rgba(22,163,74,0.25);">
        🔬 الأمراض المكتشفة
    </div>
    <div style="display:flex; flex-direction:column; gap:0.45rem;">
        <div style="display:flex; align-items:center; gap:0.55rem; background:#fff;
                    border:1px solid #bbf7d0; border-left:4px solid #ef4444;
                    border-radius:10px; padding:0.5rem 0.7rem;">
            <span style="font-size:1rem;">🩺</span>
            <div>
                <div style="font-size:0.78rem; font-weight:600; color:#14532d;">Diabetic Retinopathy</div>
                <div style="font-size:0.66rem; color:#6aaa85;">اعتلال الشبكية السكري</div>
            </div>
        </div>
        <div style="display:flex; align-items:center; gap:0.55rem; background:#fff;
                    border:1px solid #bbf7d0; border-left:4px solid #ef4444;
                    border-radius:10px; padding:0.5rem 0.7rem;">
            <span style="font-size:1rem;">🧠</span>
            <div>
                <div style="font-size:0.78rem; font-weight:600; color:#14532d;">Disc Edema</div>
                <div style="font-size:0.66rem; color:#6aaa85;">وذمة القرص البصري</div>
            </div>
        </div>
        <div style="display:flex; align-items:center; gap:0.55rem; background:#fff;
                    border:1px solid #bbf7d0; border-left:4px solid #22c55e;
                    border-radius:10px; padding:0.5rem 0.7rem;">
            <span style="font-size:1rem;">✅</span>
            <div>
                <div style="font-size:0.78rem; font-weight:600; color:#14532d;">Healthy</div>
                <div style="font-size:0.66rem; color:#6aaa85;">شبكية سليمة</div>
            </div>
        </div>
        <div style="display:flex; align-items:center; gap:0.55rem; background:#fff;
                    border:1px solid #bbf7d0; border-left:4px solid #f59e0b;
                    border-radius:10px; padding:0.5rem 0.7rem;">
            <span style="font-size:1rem;">👓</span>
            <div>
                <div style="font-size:0.78rem; font-weight:600; color:#14532d;">Myopia</div>
                <div style="font-size:0.66rem; color:#6aaa85;">قِصَر النظر</div>
            </div>
        </div>
        <div style="display:flex; align-items:center; gap:0.55rem; background:#fff;
                    border:1px solid #bbf7d0; border-left:4px solid #f59e0b;
                    border-radius:10px; padding:0.5rem 0.7rem;">
            <span style="font-size:1rem;">🔬</span>
            <div>
                <div style="font-size:0.78rem; font-weight:600; color:#14532d;">Pterygium</div>
                <div style="font-size:0.66rem; color:#6aaa85;">الظفرة</div>
            </div>
        </div>
        <div style="display:flex; align-items:center; gap:0.55rem; background:#fff;
                    border:1px solid #bbf7d0; border-left:4px solid #dc2626;
                    border-radius:10px; padding:0.5rem 0.7rem;">
            <span style="font-size:1rem;">🚨</span>
            <div>
                <div style="font-size:0.78rem; font-weight:600; color:#14532d;">Retinal Detachment</div>
                <div style="font-size:0.66rem; color:#6aaa85;">انفصال الشبكية</div>
            </div>
        </div>
        <div style="display:flex; align-items:center; gap:0.55rem; background:#fff;
                    border:1px solid #bbf7d0; border-left:4px solid #ef4444;
                    border-radius:10px; padding:0.5rem 0.7rem;">
            <span style="font-size:1rem;">🧬</span>
            <div>
                <div style="font-size:0.78rem; font-weight:600; color:#14532d;">Retinitis Pigmentosa</div>
                <div style="font-size:0.66rem; color:#6aaa85;">التهاب الشبكية الصباغي</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Column 2: Image Upload ──
with left_col:
    st.markdown('<div class="upload-label">رفع صورة العين</div>', unsafe_allow_html=True)
    st.markdown('<div class="upload-hint">الصيغ المدعومة: JPG · PNG</div>', unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        label="اختر صورة",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        thumb = image.copy()
        thumb.thumbnail((210, 210))
        st.markdown('<div class="img-card">', unsafe_allow_html=True)
        st.image(thumb, use_container_width=False, width=220)
        st.markdown('<div class="img-card-label">الصورة الأصلية</div></div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="upload-section">
            <div style="font-size:2.5rem; margin-bottom:0.75rem; opacity:0.5">👁️</div>
            <div style="font-size:0.88rem; color:#6aaa85;">اسحب وأفلت الصورة هنا<br>أو انقر للاختيار</div>
        </div>
        """, unsafe_allow_html=True)


# ── Column 3: Analysis Results ──
with right_col:
    if uploaded_file:
        with st.spinner("🔍 جاري التحليل..."):
            pred, conf, all_preds = predict(image, model)
            heatmap  = gradcam(image, model)
            overlay  = overlay_heatmap(image, heatmap)

        color = SEVERITY_COLOR.get(pred, "#38bdf8")
        info  = DISEASE_INFO.get(pred, {})

        # ── Diagnosis Result ──
        st.markdown(f"""
        <div style="margin-bottom:1.5rem;">
            <div style="font-size:0.72rem; letter-spacing:0.18em; text-transform:uppercase;
                        color:#6aaa85; margin-bottom:0.6rem;">نتيجة التشخيص</div>
            <div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:1rem;">
                <span style="font-size:1.8rem;">{info.get('icon', '🔬')}</span>
                <span style="font-family:'Syne',sans-serif; font-size:1.6rem; font-weight:800;
                             color:{color}; letter-spacing:-0.01em;">{pred}</span>
            </div>
            <div class="confidence-label">مستوى الثقة</div>
            <div class="confidence-value">{conf * 100:.1f}<span>%</span></div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(int(conf * 100))

        # ── Disease Info Card ──
        if info:
            st.markdown(f"""
            <div class="disease-card">
                <div class="disease-card-title">📋 عن هذه الحالة</div>
                <div class="disease-card-text">{info['desc']}</div>
                <div style="margin-top:0.7rem; padding-top:0.7rem;
                             border-top:1px solid rgba(56,189,248,0.1);">
                    <span style="font-size:0.75rem; color:#16a34a; font-weight:600;">التوصية: </span>
                    <span class="disease-card-text">{info['action']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── LLM Explanation (Grok) ──
        if enable_llm:
            with st.spinner(f"🤖 جاري توليد الشرح الطبي عبر Grok ({grok_model})..."):
                llm_result = grok_llm_explain(pred, conf, grok_model=grok_model, api_key=grok_api_key)

            if llm_result.startswith("ERROR:"):
                error_msg = llm_result.replace("ERROR:", "").strip()
                st.markdown(f"""
                <div class="llm-card">
                    <div class="llm-card-title">🤖 شرح النموذج اللغوي — Grok ({grok_model})</div>
                    <div class="llm-error">⚠️ {error_msg}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                lines_html = "".join(
                    f'<div class="llm-line">{line}</div>'
                    for line in llm_result.split("\n")
                    if line.strip()
                )
                st.markdown(f"""
                <div class="llm-card">
                    <div class="llm-card-title">🤖 شرح النموذج اللغوي — Grok ({grok_model})</div>
                    {lines_html}
                </div>
                """, unsafe_allow_html=True)

        # ── Grad-CAM Visualisation ──
        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.72rem; letter-spacing:0.18em; text-transform:uppercase;
                    color:#6aaa85; margin-bottom:0.75rem;">التحليل البصري — Grad-CAM</div>
        """, unsafe_allow_html=True)

        v1, v2 = st.columns(2)
        with v1:
            st.markdown('<div class="img-card">', unsafe_allow_html=True)
            st.image(heatmap, width=200, channels="BGR")
            st.markdown('<div class="img-card-label">خريطة الحرارة</div></div>', unsafe_allow_html=True)
        with v2:
            st.markdown('<div class="img-card">', unsafe_allow_html=True)
            st.image(overlay, width=200, channels="BGR")
            st.markdown('<div class="img-card-label">الصورة المدمجة</div></div>', unsafe_allow_html=True)

        # ── All Class Probabilities ──
        with st.expander("📊 جميع الاحتمالات"):
            for i in np.argsort(all_preds)[::-1]:
                pct       = float(all_preds[i]) * 100
                bar_color = color if CLASS_NAMES[i] == pred else "#bbf7d0"
                st.markdown(f"""
                <div style="display:flex; align-items:center; gap:0.75rem;
                             margin-bottom:0.5rem; font-size:0.82rem;">
                    <div style="width:160px; color:#4b7a5e; white-space:nowrap;
                                overflow:hidden; text-overflow:ellipsis;">{CLASS_NAMES[i]}</div>
                    <div style="flex:1; background:#dcfce7; border-radius:999px; height:6px; overflow:hidden;">
                        <div style="width:{pct:.1f}%; height:100%;
                                    background:{bar_color}; border-radius:999px;"></div>
                    </div>
                    <div style="width:44px; text-align:right; color:#6aaa85;">{pct:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="display:flex; flex-direction:column; align-items:center;
                    justify-content:center; height:300px; opacity:0.4; text-align:center;">
            <div style="font-size:3rem; margin-bottom:1rem;">🔬</div>
            <div style="font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:600; color:#15803d;">
                في انتظار صورة للتحليل
            </div>
        </div>
        """, unsafe_allow_html=True)


# ==============================
# Medical Disclaimer
# ==============================
st.markdown("""
<div class="disclaimer">
    ⚠️ <strong>تنبيه طبي:</strong> هذا النظام أداةٌ مساعدة للفحص الأولي ولا يُغني عن استشارة طبيب متخصص.
    يُرجى مراجعة طبيب عيون معتمد للتشخيص النهائي والعلاج المناسب.
</div>
""", unsafe_allow_html=True)
