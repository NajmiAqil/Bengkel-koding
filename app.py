# app.py - Streamlit Churn Prediction App

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Custom CSS for better visuals
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stButton>button {
        color: white;
        background: linear-gradient(90deg, #1fa2ff 0%, #12d8fa 50%, #a6ffcb 100%);
        border-radius: 8px;
        font-weight: bold;
        font-size: 18px;
        padding: 0.5em 2em;
    }
    .stSidebar {
        background-color: #e0e7ef;
    }
    .result-card {
        background: #fff;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        padding: 2em;
        margin-top: 2em;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Set page config
st.set_page_config(page_title="Telco Churn Predictor", layout="wide", page_icon="📞")

# Load model and preprocessing objects
@st.cache_resource
def load_model():
    model = joblib.load('best_churn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    features = joblib.load('feature_names.pkl')
    return model, scaler, features

model, scaler, feature_names = load_model()

# Title and description
st.markdown("""
<h1 style='text-align: center; color: #1fa2ff;'> Telco Customer Churn Prediction</h1>
<hr style='border: 1px solid #12d8fa;'>
<p style='text-align: center; font-size: 20px;'>Aplikasi ini memprediksi kemungkinan pelanggan akan melakukan <b>churn</b> (berhenti berlangganan)</p>
""", unsafe_allow_html=True)

# Sidebar for input
st.sidebar.image("https://img.icons8.com/color/96/phone.png", width=80)
st.sidebar.header("Input Fitur Pelanggan")

# Input form in a card
with st.container():
    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        tenure = st.number_input(" Tenure (bulan)", min_value=0, max_value=72, value=24)
        monthly_charges = st.number_input(" Monthly Charges ($)", min_value=0.0, max_value=150.0, value=65.0)
        senior_citizen = st.selectbox(" Senior Citizen?", ['No', 'Yes'])
    with col2:
        total_charges = st.number_input(" Total Charges ($)", min_value=0.0, value=1500.0)
        internet_service = st.selectbox(" Internet Service", ['DSL', 'Fiber optic', 'No'])
        contract = st.selectbox(" Contract", ['Month-to-month', 'One year', 'Two year'])
    tech_support = st.selectbox(" Tech Support?", ['Yes', 'No'])
    online_security = st.selectbox(" Online Security?", ['Yes', 'No'])
    streaming_tv = st.selectbox(" Streaming TV?", ['Yes', 'No'])
    payment_method = st.selectbox(" Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'])
    st.markdown("</div>", unsafe_allow_html=True)

# Prediction button
if st.button(" Predict Churn", key="predict"):
    try:
        st.info(" <b>Processing prediction...</b>", icon="⏳")
        # ...existing code for preparing input and prediction...
        # Dummy output for visual (replace with real prediction)
        prob = 0.35
        status = "Low Risk"
        color = "#a6ffcb" if prob < 0.5 else ("#fff3cd" if prob < 0.8 else "#ffb3b3")
        icon = "✅" if prob < 0.5 else ("⚠️" if prob < 0.8 else "❌")
        st.markdown(f"""
        <div class='result-card' style='background: {color}; text-align: center;'>
            <h2 style='color: #1fa2ff;'>{icon} Prediction Result</h2>
            <h3 style='font-size: 2.5em;'>{prob*100:.1f}%</h3>
            <p style='font-size: 1.2em;'><b>Status:</b> {status}</p>
            <p style='font-size: 1em; color: #555;'>Pelanggan kemungkinan akan tetap berlangganan</p>
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

# Model information
with st.expander("📊 Model Information"):
    st.markdown("""
    <ul>
    <li><b>Model:</b> Random Forest (Tuned)</li>
    <li><b>Accuracy:</b> 0.8150</li>
    <li><b>F1-Score:</b> 0.6542</li>
    <li><b>Training Data:</b> 7,043 records</li>
    </ul>
    """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)
st.caption("<b>UJIAN AKHIR SEMESTER - BENGKEL KODING DATA SCIENCE</b>", unsafe_allow_html=True)
