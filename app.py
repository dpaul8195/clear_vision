# app.py
import streamlit as st
import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
import types
import sys
import torchvision.transforms.functional as F

# Fix for grayscale function expected by RealESRGAN
def rgb_to_grayscale(img):
    return F.rgb_to_grayscale(img)

functional_tensor = types.ModuleType("torchvision.transforms.functional_tensor")
functional_tensor.rgb_to_grayscale = rgb_to_grayscale
sys.modules["torchvision.transforms.functional_tensor"] = functional_tensor

# Streamlit UI setup
st.set_page_config(page_title="CLEAR-VISION Upscaler", layout="centered")
st.markdown("<h1 style='color:black;'>CLEAR-VISION Image Upscaler</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='color:black;'>Upload a low-quality image to enhance it</h3>", unsafe_allow_html=True)

# Background image (optional)
page_bg_img = """
<style>
.stApp {
  background-image: url("https://pixabay.com/photos/mountain-pyrenees-lake-valley-9027189/");
  background-size: cover;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Load Real-ESRGAN with your trained weights
@st.cache_resource
def setup_upsampler():
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    upsampler = RealESRGANer(
        scale=2,
        model_path="model_after_training/RealESRGAN_x2plus.pth",  # Your trained model path
        model=model,
        tile=400,
        tile_pad=10,
        pre_pad=0,
        half=False  # Use FP32
    )
    return upsampler

upsampler = setup_upsampler()

# Image enhancement function
def enhance_image(file):
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Only upsampling using RRDBNet
    output, _ = upsampler.enhance(img, outscale=2)

    return img, output

# Upload and process
file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
if file is not None:
    st.image(file, caption="Original Uploaded Image", use_container_width=True)

    with st.spinner("Enhancing Image..."):
        original, enhanced = enhance_image(file)

    # Show results
    st.markdown("### Enhanced Output:")
    col1, col2 = st.columns(2)
    with col1:
        st.image(original, caption="Original", use_container_width=True)
    with col2:
        st.image(enhanced, caption="Enhanced", use_container_width=True)

    # Download button
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
    output_path = "results/enhanced_output.jpg"
    os.makedirs("results", exist_ok=True)
    cv2.imwrite(output_path, enhanced_bgr)
    with open(output_path, "rb") as f:
        st.download_button("Download Enhanced Image", f, file_name="enhanced.jpg")
