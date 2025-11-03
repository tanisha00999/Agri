import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import google.generativeai as genai

# 🔐 Configure Gemini API key securely
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# Load models and preprocessing tools
rf_model = joblib.load('rf_crop_model.pkl')
nn_model = load_model('nn_crop_model.h5')
scaler = joblib.load('scaler.pkl')
le = joblib.load('label_encoder.pkl')

# 🌍 Sidebar language selection
st.sidebar.header("🌍 Language Settings")
language = st.sidebar.selectbox("Choose your language:", ["English", "Hindi", "Tamil", "Telugu", "Marathi"])

# 🌾 Translations for all text
texts = {
    "English": {
        "title": "🌾 AI-Powered Crop Recommendation System",
        "desc": "Enter soil and weather parameters to predict the best crop and get AI-based fertilizer suggestions.",
        "nitrogen": "Nitrogen (N)",
        "phosphorous": "Phosphorous (P)",
        "potassium": "Potassium (K)",
        "temperature": "Temperature (°C)",
        "humidity": "Humidity (%)",
        "ph": "pH of soil",
        "rainfall": "Rainfall (mm)",
        "predict": "Predict Crop",
        "recommendation": "🌾 Recommended Crop:",
        "ai_header": "💬 AI Fertilizer Recommendation & Reasoning"
    },
    "Hindi": {
        "title": "🌾 एआई-संचालित फसल सिफारिश प्रणाली",
        "desc": "मिट्टी और मौसम के मान दर्ज करें ताकि सर्वोत्तम फसल और उर्वरक की सिफारिश प्राप्त हो सके।",
        "nitrogen": "नाइट्रोजन (N)",
        "phosphorous": "फॉस्फोरस (P)",
        "potassium": "पोटेशियम (K)",
        "temperature": "तापमान (°C)",
        "humidity": "आर्द्रता (%)",
        "ph": "मिट्टी का pH",
        "rainfall": "वर्षा (mm)",
        "predict": "फसल की भविष्यवाणी करें",
        "recommendation": "🌾 अनुशंसित फसल:",
        "ai_header": "💬 एआई उर्वरक सिफारिश और कारण"
    },
    "Tamil": {
        "title": "🌾 செயற்கை நுண்ணறிவு பயிர் பரிந்துரை அமைப்பு",
        "desc": "மண் மற்றும் வானிலை அளவுருக்களை உள்ளிடவும் சிறந்த பயிரை பரிந்துரைக்கவும் உர பரிந்துரையைப் பெறவும்.",
        "nitrogen": "நைட்ரஜன் (N)",
        "phosphorous": "பாஸ்பரஸ் (P)",
        "potassium": "பொட்டாசியம் (K)",
        "temperature": "வெப்பநிலை (°C)",
        "humidity": "ஈரப்பதம் (%)",
        "ph": "மண் pH",
        "rainfall": "மழை (mm)",
        "predict": "பயிரை கணிக்கவும்",
        "recommendation": "🌾 பரிந்துரைக்கப்பட்ட பயிர்:",
        "ai_header": "💬 ஏஐ உர பரிந்துரை மற்றும் விளக்கம்"
    },
    "Telugu": {
        "title": "🌾 కృత్రిమ మేధ ఆధారిత పంట సిఫారసు వ్యవస్థ",
        "desc": "మట్టిలో మరియు వాతావరణ పరామితులను నమోదు చేసి ఉత్తమ పంటను మరియు ఎరువు సిఫారసును పొందండి.",
        "nitrogen": "నైట్రోజన్ (N)",
        "phosphorous": "ఫాస్ఫరస్ (P)",
        "potassium": "పొటాషియం (K)",
        "temperature": "ఉష్ణోగ్రత (°C)",
        "humidity": "తేమ (%)",
        "ph": "మట్టిలో pH",
        "rainfall": "వర్షపాతం (mm)",
        "predict": "పంటను అంచనా వేయండి",
        "recommendation": "🌾 సిఫారసు చేసిన పంట:",
        "ai_header": "💬 AI ఎరువు సిఫారసు మరియు వివరణ"
    },
    "Marathi": {
        "title": "🌾 एआय-आधारित पीक शिफारस प्रणाली",
        "desc": "माती आणि हवामानाचे मापदंड प्रविष्ट करा आणि सर्वोत्तम पीक व खताची शिफारस मिळवा.",
        "nitrogen": "नायट्रोजन (N)",
        "phosphorous": "फॉस्फरस (P)",
        "potassium": "पोटॅशियम (K)",
        "temperature": "तापमान (°C)",
        "humidity": "आर्द्रता (%)",
        "ph": "मातीचा pH",
        "rainfall": "पर्जन्यमान (mm)",
        "predict": "पीक भाकीत करा",
        "recommendation": "🌾 शिफारस केलेले पीक:",
        "ai_header": "💬 एआय खताची शिफारस आणि कारण"
    }
}

# Select language dictionary
t = texts[language]

# UI layout
st.title(t["title"])
st.write(t["desc"])

# Input fields
N = st.number_input(t["nitrogen"], 0, 200, 80)
P = st.number_input(t["phosphorous"], 0, 200, 45)
K = st.number_input(t["potassium"], 0, 200, 43)
temperature = st.number_input(t["temperature"], 0.0, 50.0, 25.0)
humidity = st.number_input(t["humidity"], 0.0, 100.0, 80.0)
ph = st.number_input(t["ph"], 0.0, 14.0, 7.0)
rainfall = st.number_input(t["rainfall"], 0.0, 500.0, 250.0)

# Predict crop
if st.button(t["predict"]):
    sample = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    sample_scaled = scaler.transform(sample)

    rf_pred_num = rf_model.predict(sample_scaled)
    rf_pred_label = le.inverse_transform(rf_pred_num)[0]

    nn_pred_num = nn_model.predict(sample_scaled).argmax()
    nn_pred_label = le.inverse_transform([nn_pred_num])[0]

    st.success(f"{t['recommendation']} {rf_pred_label}")

    # AI explanation using Gemini
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
    Suggest the best fertilizer and explain why it suits these conditions.
    Respond in {language}.
    """

    with st.spinner("🧠 AI analyzing fertilizer recommendation..."):
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        response = model.generate_content(prompt)

    st.subheader(f"{t['ai_header']} ({language})")
    st.write(response.text or "No response received. Please try again.")
