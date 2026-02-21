import streamlit as st
import pydicom
import numpy as np
import cv2
import os
from engine import DiagnosisEngine

# 1. Page Config (MUST BE FIRST)
st.set_page_config(page_title="Pneumothorax AI", page_icon="🫁", layout="wide")

st.title("🫁 Pneumothorax Diagnostic Assistant")
st.write("System Status: **Online**")

# 2. Check if the "Brain" file exists
WEIGHTS_PATH = "best_efficientnet_model.pth"

if not os.path.exists(WEIGHTS_PATH):
    st.error(f"Critical Error: Model file '{WEIGHTS_PATH}' not found in repository!")
    st.stop()

# 3. Load AI Engine with a loading message
@st.cache_resource
def get_engine():
    try:
        return DiagnosisEngine(WEIGHTS_PATH)
    except Exception as e:
        st.error(f"Error loading AI Engine: {e}")
        return None

with st.spinner('Loading AI Model... please wait.'):
    engine = get_engine()

if engine is None:
    st.stop()

# 4. File Uploader
st.write("---")
uploaded_file = st.file_uploader("Upload a Chest X-ray (DICOM format)", type=["dcm"])

if uploaded_file is not None:
    try:
        # Read DICOM
        ds = pydicom.dcmread(uploaded_file)
        img = ds.pixel_array
        
        # Clinical Fixes
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            img = np.max(img) - img
        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-6) * 255
        img_rgb = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        
        # Run AI
        with st.spinner('AI is analyzing imagery...'):
            diagnosis, confidence, heatmap = engine.predict(img_rgb)
        
        # Show Results
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Radiograph")
            st.image(img_rgb, use_container_width=True)
        with col2:
            st.subheader("AI Pathology Focus (Grad-CAM)")
            st.image(heatmap, use_container_width=True)
            
        if "DETECTED" in diagnosis:
            st.error(f"**RESULT:** {diagnosis}  \n**Probability:** {confidence:.2%}")
        else:
            st.success(f"**RESULT:** {diagnosis}  \n**Probability:** {1-confidence:.2%}")
            
    except Exception as e:
        st.error(f"Error processing file: {e}")
