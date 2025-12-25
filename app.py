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
    features = joblib.load('feature_names.pkl')
    metadata = joblib.load('model_metadata.pkl')
    
    # Load preprocessor/scaler if needed (for Preprocessing/Tuned models)
    scaler = None
    if metadata.get('processing_type') in ['Preprocessing', 'Tuned']:
        try:
            scaler = joblib.load('scaler.pkl')
        except FileNotFoundError:
            pass
    
    return model, scaler, features, metadata

model, scaler, feature_names, metadata = load_model()

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
        # Prepare input dictionary
        input_dict = {
            'tenure': tenure,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges,
            'SeniorCitizen': 1 if senior_citizen == 'Yes' else 0,
            'InternetService_DSL': 1 if internet_service == 'DSL' else 0,
            'InternetService_Fiber optic': 1 if internet_service == 'Fiber optic' else 0,
            'InternetService_No': 1 if internet_service == 'No' else 0,
            'Contract_Month-to-month': 1 if contract == 'Month-to-month' else 0,
            'Contract_One year': 1 if contract == 'One year' else 0,
            'Contract_Two year': 1 if contract == 'Two year' else 0,
            'TechSupport_Yes': 1 if tech_support == 'Yes' else 0,
            'TechSupport_No': 1 if tech_support == 'No' else 0,
            'OnlineSecurity_Yes': 1 if online_security == 'Yes' else 0,
            'OnlineSecurity_No': 1 if online_security == 'No' else 0,
            'StreamingTV_Yes': 1 if streaming_tv == 'Yes' else 0,
            'StreamingTV_No': 1 if streaming_tv == 'No' else 0,
            'PaymentMethod_Electronic check': 1 if payment_method == 'Electronic check' else 0,
            'PaymentMethod_Mailed check': 1 if payment_method == 'Mailed check' else 0,
            'PaymentMethod_Bank transfer': 1 if payment_method == 'Bank transfer' else 0,
            'PaymentMethod_Credit card': 1 if payment_method == 'Credit card' else 0
        }

        # Pastikan semua fitur yang dibutuhkan model ada
        input_features = {k: 0 for k in feature_names}
        for k, v in input_dict.items():
            if k in input_features:
                input_features[k] = v

        # Buat DataFrame input
        X_input = pd.DataFrame([input_features])
        # Scaling (hanya jika scaler tersedia)
        if scaler is not None:
            X_input_scaled = scaler.transform(X_input)
        else:
            X_input_scaled = X_input.values
        # Predict
        prob = model.predict_proba(X_input_scaled)[0][1]
        status = "Low Risk" if prob < 0.5 else ("Medium Risk" if prob < 0.8 else "High Risk")
        color = "#a6ffcb" if prob < 0.5 else ("#fff3cd" if prob < 0.8 else "#ffb3b3")
        icon = "✅" if prob < 0.5 else ("⚠️" if prob < 0.8 else "❌")
        st.markdown(f"""
        <div class='result-card' style='background: {color}; text-align: center;'>
            <h2 style='color: #1fa2ff;'>{icon} Prediction Result</h2>
            <h3 style='font-size: 2.5em;'>{prob*100:.1f}%</h3>
            <p style='font-size: 1.2em;'><b>Status:</b> {status}</p>
            <p style='font-size: 1em; color: #555;'>Pelanggan kemungkinan akan {'tetap berlangganan' if prob < 0.5 else ('berisiko churn' if prob < 0.8 else 'sangat berisiko churn')}</p>
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

# Model information
with st.expander("📊 Model Information"):
    st.markdown(f"""
    <ul>
    <li><b>Model:</b> {metadata.get('model_name', 'Unknown')}</li>
    <li><b>Processing Type:</b> {metadata.get('processing_type', 'Unknown')}</li>
    <li><b>Dataset Type:</b> {metadata.get('dataset_type', 'Normal')}</li>
    <li><b>Accuracy:</b> {metadata.get('accuracy', 0):.4f}</li>
    <li><b>Precision:</b> {metadata.get('precision', 0):.4f}</li>
    <li><b>Recall:</b> {metadata.get('recall', 0):.4f}</li>
    <li><b>F1-Score:</b> {metadata.get('f1_score', 0):.4f}</li>
    <li><b>Trained:</b> {metadata.get('trained_date', 'Unknown')}</li>
    </ul>
    """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)
st.caption("<b>UJIAN AKHIR SEMESTER - BENGKEL KODING DATA SCIENCE</b>", unsafe_allow_html=True)
