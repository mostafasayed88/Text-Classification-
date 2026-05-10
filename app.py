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
    .hero-subtitle {
        font-size: 1rem;
        font-weight: 300;
        color: #4b7a5e;
        max-width: 600px;
        margin: 0 auto;
        line-height: 1.75;
        text-align: center;
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
# Constants
# ==============================
MODEL_PATH  = "best_efficientnetb3.h5"
FILE_ID     = "1qnrKRAWa7UU5YbtT2UqGDbJij7uH6dIz"

# Grok API (xAI — OpenAI-compatible)
GROK_API_BASE = "https://api.x.ai/v1"
GROK_MODELS   = ["grok-3-beta", "grok-3", "grok-3-latest"]

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
    "Healthy":               "#22c55e",
    "Myopia":                "#f59e0b",
    "Pterygium":             "#f59e0b",
    "Diabetic Retinopathy":  "#ef4444",
    "Disc Edema":            "#ef4444",
    "Retinal Detachment":    "#dc2626",
    "Retinitis Pigmentosa":  "#ef4444",
}

DISEASE_INFO = {
    "Diabetic Retinopathy": {
        "desc":   "تلف في أوعية الدم الدقيقة بشبكية العين نتيجة مرض السكري. يُعدّ من الأسباب الرئيسية للعمى لدى البالغين.",
        "action": "يُنصح بفحص دوري كل 6 أشهر ومراقبة مستوى السكر في الدم.",
        "icon":   "🩺",
    },
    "Disc Edema": {
        "desc":   "تورم في القرص البصري قد يشير إلى ارتفاع ضغط الدم داخل الجمجمة أو اضطرابات عصبية.",
        "action": "يتطلب تقييمًا عصبيًا عاجلاً وصور أشعة للدماغ.",
        "icon":   "🧠",
    },
    "Healthy": {
        "desc":   "لم يُكتشف أي مؤشر مرضي. تبدو شبكية العين سليمة وبحالة جيدة.",
        "action": "حافظ على فحوصات دورية سنوية للعين للاطمئنان على صحتها.",
        "icon":   "✅",
    },
    "Myopia": {
        "desc":   "قِصَر النظر: صعوبة في رؤية الأشياء البعيدة بوضوح بسبب طول محور مقلة العين.",
        "action": "يمكن تصحيحه بالنظارات أو العدسات اللاصقة أو جراحة الليزر.",
        "icon":   "👓",
    },
    "Pterygium": {
        "desc":   "نسيج ليفي وعائي ينمو على سطح القرنية من الملتحمة، وقد يؤثر على الرؤية.",
        "action": "قد يحتاج إلى استئصال جراحي إذا تقدّم نحو مركز القرنية.",
        "icon":   "🔬",
    },
    "Retinal Detachment": {
        "desc":   "انفصال الشبكية عن طبقة الظهارة الصباغية، وهو طارئ طبي يستوجب تدخلاً فوريًا.",
        "action": "توجّه فورًا إلى أقرب طوارئ عيون — يمكن أن يؤدي التأخير إلى فقدان البصر نهائيًا.",
        "icon":   "🚨",
    },
    "Retinitis Pigmentosa": {
        "desc":   "مجموعة اضطرابات وراثية تُسبب تدهورًا تدريجيًا في خلايا الشبكية المستقبلة للضوء.",
        "action": "لا يوجد علاج شافٍ حتى الآن؛ التدبير يركز على إبطاء التقدم وتحسين جودة الحياة.",
        "icon":   "🧬",
    },
}

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
# Grok LLM Helpers
# ==============================
def _clean_lines(text: str) -> str:
    """Return the first 5 non-empty lines of *text*."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return "\n".join(lines[:5])


def _explain_via_grok(disease: str, confidence: float, grok_model: str, api_key: str) -> str:
    """Call the xAI Grok API (OpenAI-compatible) and return a cleaned 5-line explanation."""
    prompt = PROMPT_TEMPLATE.format(disease=disease, confidence=confidence * 100)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": grok_model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 300,
        "temperature": 0.1,
    }
    response = requests.post(
        f"{GROK_API_BASE}/chat/completions",
        headers=headers,
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    raw = response.json()["choices"][0]["message"]["content"].strip()
    return _clean_lines(raw)


def _test_grok_connection(api_key: str) -> tuple[bool, str]:
    """Ping the xAI API with a minimal request. Return (success, message)."""
    if not api_key or not api_key.strip():
        return False, "❌ مفتاح API فارغ — أدخل مفتاحك من console.x.ai"
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key.strip()}",
        }
        payload = {
            "model": "grok-3-beta",
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 5,
        }
        r = requests.post(
            f"{GROK_API_BASE}/chat/completions",
            headers=headers,
            json=payload,
            timeout=15,
        )
        if r.status_code == 200:
            return True, "✅ Grok API يعمل بنجاح!"
        if r.status_code == 401:
            return False, "❌ مفتاح API غير صالح — تحقق من console.x.ai"
        return False, f"⚠️ استجابة غير متوقعة: {r.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "❌ لا يمكن الاتصال بـ api.x.ai"
    except requests.exceptions.Timeout:
        return False, "❌ انتهت المهلة — الخادم لا يستجيب"
    except Exception as exc:
        return False, f"❌ خطأ: {exc}"


def grok_llm_explain(
    disease: str,
    confidence: float,
    grok_model: str = "grok-3-beta",
    api_key: str = "",
) -> str:
    """Return an LLM explanation string, or an 'ERROR: …' string on failure."""
    if not api_key or not api_key.strip():
        return "ERROR: أدخل مفتاح xAI API في الشريط الجانبي."
    try:
        return _explain_via_grok(disease, confidence, grok_model, api_key.strip())
    except requests.exceptions.ConnectionError:
        return "ERROR: تعذّر الاتصال بـ api.x.ai — تحقق من اتصالك بالإنترنت."
    except requests.exceptions.Timeout:
        return "ERROR: انتهت مهلة الاستجابة — حاول مرة أخرى."
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "?"
        if status == 401:
            return "ERROR: مفتاح API غير صالح — تحقق من console.x.ai"
        if status == 429:
            return "ERROR: تجاوزت حد الطلبات — انتظر قليلاً ثم أعد المحاولة."
        return f"ERROR: HTTP {status} — {exc}"
    except Exception as exc:
        return f"ERROR: خطأ غير متوقع: {exc}"


# ==============================
# Legacy Keras 2 → Keras 3 Compatibility Patch
# ==============================
def _patch_input_layer():
    """
    Keras 3 / TF 2.16+ removed 'batch_shape' from InputLayer.__init__.
    Models saved with Keras 2 still store it in their JSON config, so
    deserialization crashes with:
        Unrecognized keyword arguments: ['batch_shape']

    This monkey-patch wraps InputLayer so that 'batch_shape' is silently
    converted to 'shape' (dropping the leading None batch dimension)
    before the real constructor runs.
    """
    try:
        original_init = tf.keras.layers.InputLayer.__init__

        def _patched_input_layer_init(self, *args, **kwargs):
    if "batch_shape" in kwargs:
        batch_shape = kwargs.pop("batch_shape")
        shape = tuple(batch_shape[1:]) if batch_shape[0] is None else tuple(batch_shape)
        kwargs.setdefault("shape", shape)
    _ORIGINAL_INPUT_LAYER_INIT(self, *args, **kwargs)  # ✅ always found

tf.keras.layers.InputLayer.__init__ = _patched_input_layer_init


_patch_input_layer()


# ==============================
# Vision Model — Load & Cache
# ==============================
def _patched_init(self, *args, **kwargs):
    if "batch_shape" in kwargs:
        batch_shape = kwargs.pop("batch_shape")   # remove the old arg
        kwargs.setdefault("shape", tuple(batch_shape[1:]))  # e.g. (300,300,3)
    original_init(self, *args, **kwargs)

tf.keras.layers.InputLayer.__init__ = _patched_init
@st.cache_resource
def load_model_cached():
    """Download (if needed) and load the EfficientNetB3 model."""
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⬇️ جاري تحميل النموذج..."):
            gdown.download(
                f"https://drive.google.com/uc?id={FILE_ID}",
                MODEL_PATH,
                quiet=False,
            )

    if not os.path.exists(MODEL_PATH):
        st.error("❌ النموذج غير موجود — تحقق من اتصالك بالإنترنت.")
        st.stop()

    if os.path.getsize(MODEL_PATH) < 5_000_000:
        st.error("❌ ملف النموذج تالف — احذفه وأعد تشغيل التطبيق.")
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
        🤖 إعدادات الشرح الذكي (Grok)
    </div>
    """, unsafe_allow_html=True)

    enable_llm = st.toggle("🔘 تفعيل شرح LLM", value=True)

    grok_model = st.selectbox(
        "نموذج Grok",
        options=GROK_MODELS,
        index=0,
        help="اختر نموذج Grok من xAI",
    )

    grok_api_key = st.text_input(
        "🔑 xAI API Key",
        type="password",
        placeholder="xai-xxxxxxxxxxxxxxxxxx",
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
                     padding:0.15rem 0.5rem; border-radius:4px;">grok-3-beta</code>
        &nbsp;·&nbsp;
        <code style="background:rgba(22,163,74,0.1); color:#15803d;
                     padding:0.15rem 0.5rem; border-radius:4px;">grok-3</code>
        &nbsp;·&nbsp;
        <code style="background:rgba(22,163,74,0.1); color:#15803d;
                     padding:0.15rem 0.5rem; border-radius:4px;">grok-3-latest</code>
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
