# =========================================================
# Assistant For Detection Of Retinal Diseases
# Streamlit + EfficientNetB3 + GradCAM + xAI Grok
# FIXED xAI API (Responses API)
# =========================================================

import os
import cv2
import gdown
import numpy as np
import requests
import streamlit as st
import tensorflow as tf

from PIL import Image
from tensorflow.keras.models import load_model


# =========================================================
# CONSTANTS
# =========================================================

MODEL_PATH = "efficientnetb3_retinal.h5"

FILE_ID = "1qnrKRAWa7UU5YbtT2UqGDbJij7uH6dIz"

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
    "Disc Edema": "#ef4444",
    "Healthy": "#22c55e",
    "Myopia": "#f59e0b",
    "Pterygium": "#f59e0b",
    "Retinal Detachment": "#dc2626",
    "Retinitis Pigmentosa": "#ef4444",
}

# =========================================================
# xAI API
# =========================================================

XAI_API_URL = "https://api.x.ai/v1/responses"

PROMPT_TEMPLATE = """
You are an ophthalmology AI assistant.

Write exactly 5 short medical lines.

Prediction: {disease}
Confidence: {confidence:.1f}%

Rules:
1. Prediction statement
2. Clinical definition
3. Symptoms
4. Severity level
5. Recommended next step

Only 5 lines.
"""


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Retinal Diseases Assistant",
    page_icon="👁️",
    layout="wide",
)

# =========================================================
# SIMPLE CSS
# =========================================================

st.markdown("""
<style>

body {
    background:#f0fdf4;
}

.stApp {
    background:#f0fdf4;
}

.main-title{
    font-size:42px;
    font-weight:800;
    color:#15803d;
    text-align:center;
    margin-bottom:30px;
}

.card{
    background:white;
    padding:20px;
    border-radius:16px;
    border:1px solid #bbf7d0;
    margin-bottom:20px;
}

.result{
    font-size:28px;
    font-weight:700;
}

.small{
    color:gray;
    font-size:14px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# xAI FUNCTIONS
# =========================================================

def clean_lines(text):
    lines = [x.strip() for x in text.split("\n") if x.strip()]
    return "\n".join(lines[:5])


def test_grok_connection(api_key):

    if not api_key.strip():
        return False, "❌ أدخل API Key"

    try:

        response = requests.post(
            XAI_API_URL,
            headers={
                "Authorization": f"Bearer {api_key.strip()}",
                "Content-Type": "application/json",
            },
            json={
                "model": "grok-4",
                "input": "Hello",
            },
            timeout=20,
        )

        if response.status_code == 200:
            return True, "✅ الاتصال ناجح"

        return False, f"{response.status_code} - {response.text}"

    except Exception as e:
        return False, str(e)


def grok_llm_explain(disease, confidence, model_name, api_key):

    if not api_key.strip():
        return "ERROR: أدخل API Key"

    prompt = PROMPT_TEMPLATE.format(
        disease=disease,
        confidence=confidence * 100,
    )

    try:

        response = requests.post(
            XAI_API_URL,
            headers={
                "Authorization": f"Bearer {api_key.strip()}",
                "Content-Type": "application/json",
            },
            json={
                "model": model_name,
                "input": prompt,
            },
            timeout=60,
        )

        response.raise_for_status()

        data = response.json()

        raw = data["output"][0]["content"][0]["text"]

        return clean_lines(raw)

    except requests.exceptions.HTTPError as e:

        return f"ERROR: HTTP {response.status_code} - {response.text}"

    except Exception as e:

        return f"ERROR: {e}"


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model_cached():

    if not os.path.exists(MODEL_PATH):

        with st.spinner("Downloading model..."):

            url = f"https://drive.google.com/uc?id={FILE_ID}"

            gdown.download(url, MODEL_PATH, quiet=False)

    return load_model(MODEL_PATH, compile=False)


# =========================================================
# IMAGE PROCESSING
# =========================================================

def preprocess(img):

    img = img.resize((300, 300))

    arr = np.array(img)

    arr = tf.keras.applications.efficientnet.preprocess_input(arr)

    return np.expand_dims(arr, axis=0)


def predict(img, model):

    preds = model.predict(preprocess(img))

    idx = np.argmax(preds[0])

    return (
        CLASS_NAMES[idx],
        float(np.max(preds)),
        preds[0]
    )


def gradcam(img, model):

    img_array = np.array(img.resize((300, 300)))

    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

    img_array = np.expand_dims(img_array, axis=0)

    # آخر Conv Layer
    last_conv_layer = None

    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break

    # Model for GradCAM
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_array)

        # لو predictions list
        if isinstance(predictions, list):
            predictions = predictions[0]

        # تحويل إلى numpy
        preds = predictions.numpy()

        class_idx = np.argmax(preds[0])

        loss = predictions[:, class_idx]

    # gradients
    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(
        pooled_grads * conv_outputs,
        axis=-1
    )

    heatmap = np.maximum(heatmap, 0)

    heatmap = heatmap / (np.max(heatmap) + 1e-8)

    heatmap = cv2.resize(heatmap, (300, 300))

    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    return heatmap

# =========================================================
# LOAD MODEL
# =========================================================

model = load_model_cached()

# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="main-title">👁️ Assistant For Detection Of Retinal Diseases</div>',
    unsafe_allow_html=True
)

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("🤖 xAI Grok")

    enable_llm = st.toggle("Enable Grok", value=True)

    grok_model = st.selectbox(
        "Model",
        [
            "grok-4",
            "grok-4-200-reasoning",
        ]
    )

    grok_api_key = st.text_input(
        "xAI API Key",
        type="password",
        placeholder="xai-..."
    )

    if st.button("Test Connection"):

        ok, msg = test_grok_connection(grok_api_key)

        if ok:
            st.success(msg)
        else:
            st.error(msg)

# =========================================================
# MAIN
# =========================================================

left, right = st.columns([1, 1.4])

# =========================================================
# LEFT
# =========================================================

with left:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload retinal image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(image, width=300)

    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# RIGHT
# =========================================================

with right:

    if uploaded_file:

        with st.spinner("Analyzing image..."):

            pred, conf, all_preds = predict(image, model)

            heatmap = gradcam(image, model)

def overlay_heatmap(img, heatmap):

    # resize original image
    img = img.resize((300, 300))

    original = np.array(img)

    # تأكد أن heatmap ثلاثي القنوات
    if len(heatmap.shape) == 2:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2BGR)

    # تأكد من نفس الحجم
    heatmap = cv2.resize(heatmap, (300, 300))

    # تحويل datatype
    original = original.astype(np.uint8)

    heatmap = heatmap.astype(np.uint8)

    # دمج الصور
    overlay = cv2.addWeighted(
        original,
        0.7,
        heatmap,
        0.3,
        0
    )

    return overlay
        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.markdown(
            f'<div class="result">Prediction: {pred}</div>',
            unsafe_allow_html=True
        )

        st.progress(int(conf * 100))

        st.write(f"Confidence: {conf*100:.2f}%")

        st.markdown('</div>', unsafe_allow_html=True)

        # =================================================
        # GROK
        # =================================================

        if enable_llm:

            with st.spinner("Generating AI explanation..."):

                result = grok_llm_explain(
                    pred,
                    conf,
                    grok_model,
                    grok_api_key
                )

            st.markdown('<div class="card">', unsafe_allow_html=True)

            st.subheader("🤖 Grok Explanation")

            if result.startswith("ERROR"):

                st.error(result)

            else:

                for line in result.split("\n"):
                    st.write("•", line)

            st.markdown('</div>', unsafe_allow_html=True)

        # =================================================
        # GRADCAM
        # =================================================

        c1, c2 = st.columns(2)

        with c1:

            st.markdown('<div class="card">', unsafe_allow_html=True)

            st.image(
                heatmap,
                channels="BGR",
                caption="GradCAM"
            )

            st.markdown('</div>', unsafe_allow_html=True)

        with c2:

            st.markdown('<div class="card">', unsafe_allow_html=True)

            st.image(
                overlay,
                channels="BGR",
                caption="Overlay"
            )

            st.markdown('</div>', unsafe_allow_html=True)

        # =================================================
        # PROBABILITIES
        # =================================================

        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.subheader("📊 All Probabilities")

        for i in np.argsort(all_preds)[::-1]:

            pct = float(all_preds[i]) * 100

            st.write(f"{CLASS_NAMES[i]} : {pct:.2f}%")

            st.progress(int(pct))

        st.markdown('</div>', unsafe_allow_html=True)

    else:

        st.info("Upload retinal image to start.")

# =========================================================
# FOOTER
# =========================================================

st.warning(
    "⚠️ This system is for educational purposes only and does not replace a doctor."
)
