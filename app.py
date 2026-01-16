# File: app.py
import streamlit as st
import joblib
import numpy as np
import os

# Set page configuration
st.set_page_config(page_title="Breast Cancer Predictor", page_icon="⚕️", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
        width: 100%;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    .benign {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .malignant {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    </style>
    """, unsafe_allow_html=True)

# Application Header
st.title("⚕️ Breast Cancer Prediction System")
st.markdown("### Machine Learning Diagnostic Tool")
st.write("Enter the tumor features below to predict whether it is **Benign** or **Malignant**.")
st.info("Note: This tool uses 5 specific features for prediction: Radius, Texture, Perimeter, Area, and Smoothness.")

# Load Model and Scaler
# We use a try-except block to handle path issues during deployment
try:
    model_path = os.path.join('model', 'breast_cancer_model.pkl')
    scaler_path = os.path.join('model', 'scaler.pkl')
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except FileNotFoundError:
    st.error("Error: Model files not found. Please ensure 'breast_cancer_model.pkl' and 'scaler.pkl' are in the 'model' folder.")
    st.stop()

# Input Form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        radius = st.number_input("Radius Mean", min_value=0.0, max_value=50.0, value=14.0, step=0.1, help="Mean of distances from center to points on the perimeter")
        texture = st.number_input("Texture Mean", min_value=0.0, max_value=50.0, value=19.0, step=0.1, help="Standard deviation of gray-scale values")
        perimeter = st.number_input("Perimeter Mean", min_value=0.0, max_value=200.0, value=90.0, step=0.1)

    with col2:
        area = st.number_input("Area Mean", min_value=0.0, max_value=2500.0, value=600.0, step=1.0)
        smoothness = st.number_input("Smoothness Mean", min_value=0.0, max_value=0.3, value=0.09, step=0.001, format="%.4f")

    submit_button = st.form_submit_button("Analyze Tumor")

# Prediction Logic
if submit_button:
    # Prepare input array
    input_data = np.array([[radius, texture, perimeter, area, smoothness]])
    
    # Scale the input using the loaded scaler
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    
    # In sklearn breast cancer dataset: 0 = Malignant, 1 = Benign
    # Let's handle the output text
    if prediction[0] == 0:
        result_text = "MALIGNANT (Cancerous)"
        result_class = "malignant"
    else:
        result_text = "BENIGN (Safe)"
        result_class = "benign"

    # Display Result
    st.markdown("---")
    st.markdown(f'<div class="result-box {result_class}">Prediction: {result_text}</div>', unsafe_allow_html=True)
    
    # Disclaimer
    st.warning("⚠️ Disclaimer: This system is for educational purposes only and must not be used as a substitute for professional medical diagnosis.")