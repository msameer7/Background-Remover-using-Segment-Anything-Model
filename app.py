import streamlit as st
import cv2
import numpy as np
import base64
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
import os
import urllib.request

# ---------------- CSS Styling ----------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #ffffff;
        font-family: 'Arial', sans-serif;
    }
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #222;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .upload-box {
        border: 2px dashed #ccc;
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        background: #f9f9f9;
        margin-bottom: 2rem;
    }
    .top-right {
        position: absolute;
        top: 20px;
        right: 30px;
        font-size: 1rem;
        font-weight: 600;
        color: #444;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- Download + Load Model ----------------
MODEL_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
MODEL_PATH = "./sam_vit_b_01ec64.pth"

def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⏳ Downloading SAM model (~358MB)..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        st.success("✅ Model downloaded!")

@st.cache_resource
def load_sam_model():
    download_model()
    sam = sam_model_registry["vit_b"](checkpoint=MODEL_PATH)
    predictor = SamPredictor(sam)
    return predictor

predictor = load_sam_model()

def remove_background(image: np.ndarray, x: int, y: int) -> np.ndarray:
    predictor.set_image(image)

    masks, scores, logits = predictor.predict(
        point_coords=np.asarray([[x, y]]),
        point_labels=np.asarray([1]),
        multimask_output=True
    )

    result_mask = np.any(masks, axis=0).astype(np.uint8)
    alpha_channel = np.where(result_mask == 1, 255, 0).astype(np.uint8)
    result_image = cv2.merge((image, alpha_channel))

    return result_image


# ---------------- Streamlit UI ----------------
st.markdown('<div class="top-right">Made by Muhammad Sameer</div>', unsafe_allow_html=True)

st.markdown('<div class="main-title">AI Background Remover</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Remove background from image instantly, fully automated and FREE</div>', unsafe_allow_html=True)

# Upload section
st.markdown('<div class="upload-box">⬆️ Upload Your Image</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file:
    pil_image = Image.open(uploaded_file).convert("RGB")
    image = np.array(pil_image)[:, :, ::-1]  # RGB -> BGR for OpenCV

    st.image(pil_image, caption="Uploaded Image", use_column_width=True)

    # Coordinates input
    st.write("### Select a point (x, y) on the image")
    x = st.number_input("X coordinate", min_value=0, value=image.shape[1] // 2)
    y = st.number_input("Y coordinate", min_value=0, value=image.shape[0] // 2)

    if st.button("✨ Remove Background"):
        result = remove_background(image, int(x), int(y))
        st.image(cv2.cvtColor(result, cv2.COLOR_BGRA2RGBA), caption="Background Removed", use_column_width=True)

        # Download option
        _, buffer = cv2.imencode(".png", result)
        b64 = base64.b64encode(buffer).decode()
        href = f'<a href="data:file/png;base64,{b64}" download="result.png">📥 Download Result</a>'
        st.markdown(href, unsafe_allow_html=True)
