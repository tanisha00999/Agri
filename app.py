import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import google.generativeai as genai

# 🔐 Configure Gemini API key securely from Streamlit secrets
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# Load trained models and preprocessing tools
rf_model = joblib.load('rf_crop_model.pkl')
nn_model = load_model('nn_crop_model.h5')
scaler = joblib.load('scaler.pkl')
le = joblib.load('label_encoder.pkl')

# Streamlit app title and description
st.title("🌾 AI-Powered Crop Recommendation System")
st.write("Enter soil and weather parameters to predict the best crop and get AI-based fertilizer suggestions.")

# Input fields
N = st.number_input("Nitrogen (N)", 0, 200, 80)
P = st.number_input("Phosphorous (P)", 0, 200, 45)
K = st.number_input("Potassium (K)", 0, 200, 43)
temperature = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
humidity = st.number_input("Humidity (%)", 0.0, 100.0, 80.0)
ph = st.number_input("pH of soil", 0.0, 14.0, 7.0)
rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 250.0)

if st.button("Predict Crop"):
    # Prepare input for prediction
    sample = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    sample_scaled = scaler.transform(sample)

    # Model predictions
    rf_pred_num = rf_model.predict(sample_scaled)
    rf_pred_label = le.inverse_transform(rf_pred_num)[0]

    nn_pred_num = nn_model.predict(sample_scaled).argmax()
    nn_pred_label = le.inverse_transform([nn_pred_num])[0]

    st.success(f"🌾 Recommended Crop: {rf_pred_label}")

    # --- AI Explanation using Gemini ---
    prompt = f"""
    You are an agricultural expert.
    Given the following soil and weather data:
    - Nitrogen: {N}
    - Phosphorous: {P}
    - Potassium: {K}
    - Temperature: {temperature}°C
    - Humidity: {humidity}%
    - Soil pH: {ph}
    - Rainfall: {rainfall} mm

    The predicted crop is **{rf_pred_label}**.
    Suggest the most suitable fertilizer and explain clearly why it fits these conditions.
    """

    with st.spinner("🧠 AI analyzing fertilizer recommendation..."):
        model = genai.GenerativeModel("models/gemini-2.5-flash")  # ✅ updated model name
        response = model.generate_content(prompt)

    st.subheader("💬 AI Fertilizer Recommendation & Reasoning:")
    st.write(response.text or "No response received. Please try again.")
