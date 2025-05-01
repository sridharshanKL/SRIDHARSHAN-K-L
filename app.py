import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from unet_model import unet_model
import matplotlib.pyplot as plt
from PIL import Image

# Load model
model = unet_model()
model.load_weights("tumor_model.h5")

IMG_SIZE = 256

# Custom CSS for style and fun stickman button
st.markdown("""
    <style>
    .main {
        background-color: #000000;
    }
    .stButton>button {
        background-color: #fc0303;
        color: white;
        font-size: 18px;
        padding: 0.75em 2em;
        border-radius: 8px;
        border: none;
        transition: 0.3s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #ff7f7f;
        transform: scale(1.05);
    }
    .nothing {
        position: fixed;
        bottom: 10px;
        right: 10px;
        cursor: pointer;
        z-index: 100;
    }
    .nothing img {
        width: 50px;
        transition: transform 0.3s;
    }
    .nothing img:hover {
        transform: rotate(20deg) scale(1.1);
    }
    </style>
    <div class="nothing">
        <a href="https://www.youtube.com/watch?v=dQw4w9WgXcQ" target="_blank">
            <img src="https://img.pikbest.com/origin/09/24/02/75ZpIkbEsT86S.png!sw800" alt="Stickman Dance" title="Surprise!">
        </a>
    </div>
""", unsafe_allow_html=True)

# Page title and description
st.markdown("<h1 style='text-align: center; color: ##030ffc;'>🧠 Brain Tumor Segmentation App</h1>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("<p style='font-size:18px;'>Upload a grayscale MRI image. This app uses a U-Net deep learning model to predict and visualize brain tumor segmentation masks.</p>", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("\U0001F4C2 Upload an MRI image", type=["png", "jpg", "jpeg"])

# Main prediction logic
if uploaded_file is not None:
    # Read and preprocess image
    img = Image.open(uploaded_file).convert("L")
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized) / 255.0
    img_input = img_array.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    # Prediction
    prediction = model.predict(img_input)[0].squeeze()

    # Display results
    st.markdown("<h2 style='color:#ff4b4b;'>🖼️ Segmentation Result</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.image(img_resized, caption="Original MRI Image", use_container_width=True)
    with col2:
        st.image(prediction, caption="Predicted Tumor Mask", use_container_width=True, clamp=True)

    st.markdown("<br><p style='text-align:center; font-size:14px;'>Results generated using a trained U-Net model.</p>", unsafe_allow_html=True)
