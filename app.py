import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Load models
rf_model = joblib.load('rf_crop_model.pkl')
nn_model = load_model('nn_crop_model.h5')

# Load scaler used in training
scaler = joblib.load('scaler.pkl')  # Make sure to save your scaler as well
# Example: joblib.dump(scaler, 'scaler.pkl') after training

# Streamlit UI
st.title("🌾 Crop Recommendation System")
st.write("Enter the soil and weather parameters to predict the best crop.")

# Input fields
N = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=80)
P = st.number_input("Phosphorous (P)", min_value=0, max_value=200, value=45)
K = st.number_input("Potassium (K)", min_value=0, max_value=200, value=43)
temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=80.0)
ph = st.number_input("pH of soil", min_value=0.0, max_value=14.0, value=7.0)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=250.0)

# Predict button
# Predict button
if st.button("Predict Crop"):
    # Prepare input
    sample = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    sample_scaled = scaler.transform(sample)

    # Load LabelEncoder
    le = joblib.load('label_encoder.pkl')

    # Random Forest prediction
    rf_pred_num = rf_model.predict(sample_scaled)          # numeric label
    rf_pred_label = le.inverse_transform(rf_pred_num)     # convert to crop name

    # Neural Network prediction
    nn_pred_num = nn_model.predict(sample_scaled).argmax()
    nn_pred_label = le.inverse_transform([nn_pred_num])[0]

    # Display predictions
    st.success(f"Random Forest Prediction: {rf_pred_label[0]}")
    #st.success(f"Neural Network Prediction: {nn_pred_label}")

