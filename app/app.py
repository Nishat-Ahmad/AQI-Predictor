import streamlit as st
import pandas as pd
import wandb
import joblib
import os

# 1. Dashboard Configuration
st.set_page_config(page_title="Lahore AQI Predictor", page_icon="🌬️", layout="wide")
st.title("🌬️ Real-Time AQI Predictor & Dashboard")
st.markdown("Enter the weather and pollution metrics below to generate an AQI prediction based on our Random Forest model.")

# 2. Download and Load Model
@st.cache_resource
def load_model():
    # Initialize a lightweight W&B run to pull the artifact
    run = wandb.init(project="pearls-aqi-predictor", job_type="inference")
    
    # Download the latest model artifact
    artifact = run.use_artifact("aqi_predictor_model:latest")
    model_dir = artifact.download()
    
    # Load the Joblib model file
    model_path = os.path.join(model_dir, "random_forest_aqi.pkl")
    rf_model = joblib.load(model_path)
    
    run.finish()
    return rf_model

try:
    with st.spinner("Downloading model from Weights & Biases..."):
        model = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load the model. Ensure you are logged into W&B. Error: {e}")
    st.stop()

# 3. Sidebar Input Form
st.sidebar.header("Input Metrics")
st.sidebar.markdown("Adjust the values to simulate different environmental conditions.")

hour = st.sidebar.slider("Hour of Day", 0, 23, 12)
day = st.sidebar.slider("Day of Month", 1, 31, 15)
month = st.sidebar.slider("Month", 1, 12, 8)

co = st.sidebar.number_input("Carbon Monoxide (CO)", value=800.0)
no2 = st.sidebar.number_input("Nitrogen Dioxide (NO2)", value=15.0)
o3 = st.sidebar.number_input("Ozone (O3)", value=50.0)
pm2_5 = st.sidebar.number_input("PM2.5", value=35.0)
pm10 = st.sidebar.number_input("PM10", value=45.0)

# 4. Prediction Logic & Alerts
if st.sidebar.button("Predict AQI"):
    # Format the input data to match what the model expects
    input_data = pd.DataFrame({
        'hour': [hour],
        'day': [day],
        'month': [month],
        'co': [co],
        'no2': [no2],
        'o3': [o3],
        'pm2_5': [pm2_5],
        'pm10': [pm10]
    })
    
    # Compute the prediction
    prediction = model.predict(input_data)[0]
    
    st.subheader("Prediction Result")
    st.metric(label="Predicted AQI Target", value=round(prediction, 2))
    
    # Generate Hazardous Alerts based on the blueprint specifications
    if prediction >= 300:
        st.error("🚨 **HAZARDOUS:** Health warning of emergency conditions. The entire population is more likely to be affected.")
    elif prediction >= 150:
        st.error("⚠️ **UNHEALTHY:** Everyone may begin to experience health effects; members of sensitive groups may experience more serious health effects.")
    elif prediction >= 50:
        st.warning("🟡 **MODERATE:** Air quality is acceptable; however, there may be a risk for some people, particularly those who are unusually sensitive to air pollution.")
    else:
        st.success("✅ **GOOD:** Air quality is considered satisfactory, and air pollution poses little or no risk.")