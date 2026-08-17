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
day_of_week_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
selected_day = st.sidebar.selectbox("Day of Week", list(day_of_week_map.keys()), index=0)
day_of_week = day_of_week_map[selected_day]

co = st.sidebar.number_input("Carbon Monoxide (CO)", value=800.0)
no2 = st.sidebar.number_input("Nitrogen Dioxide (NO2)", value=15.0)
o3 = st.sidebar.number_input("Ozone (O3)", value=50.0)
pm2_5 = st.sidebar.number_input("PM2.5", value=35.0)
pm10 = st.sidebar.number_input("PM10", value=45.0)

# Derived features
pm_ratio = round(pm2_5 / (pm10 + 1e-5), 4)
st.sidebar.caption(f"Calculated PM Ratio (PM2.5/PM10): **{pm_ratio}**")
aqi_change_rate = st.sidebar.slider("AQI Change Rate (vs Previous Hour)", min_value=-4.0, max_value=4.0, value=0.0, step=0.5)

# 4. Prediction Logic & Alerts
if st.sidebar.button("Predict AQI"):
    # Format the input data to match what the model expects
    input_data = pd.DataFrame({
        'hour': [hour],
        'day': [day],
        'month': [month],
        'day_of_week': [day_of_week],
        'co': [co],
        'no2': [no2],
        'o3': [o3],
        'pm2_5': [pm2_5],
        'pm10': [pm10],
        'pm_ratio': [pm_ratio],
        'aqi_change_rate': [aqi_change_rate]
    })
    
    # Align columns to model's expected features if available
    if hasattr(model, "feature_names_in_"):
        # Fill missing features with default 0 if model expects older/newer schema
        for col in model.feature_names_in_:
            if col not in input_data.columns:
                input_data[col] = 0.0
        input_data = input_data[model.feature_names_in_]
    
    # Compute the prediction
    prediction = model.predict(input_data)[0]
    
    st.subheader("Prediction Result")
    st.metric(label="Predicted AQI Target", value=round(prediction, 2))
    
    # Generate Hazardous Alerts based on OpenWeather 1-5 AQI scale
    if prediction >= 4.5:
        st.error(f"🚨 **VERY POOR / HAZARDOUS (Level {round(prediction, 1)}/5):** Health warning of emergency conditions. The entire population is likely to be affected.")
    elif prediction >= 3.5:
        st.error(f"⚠️ **POOR / UNHEALTHY (Level {round(prediction, 1)}/5):** Everyone may begin to experience health effects; sensitive groups may experience more serious effects.")
    elif prediction >= 2.5:
        st.warning(f"🟡 **MODERATE (Level {round(prediction, 1)}/5):** Air quality is acceptable; however, there may be moderate health concern for sensitive individuals.")
    elif prediction >= 1.5:
        st.info(f"🔵 **FAIR (Level {round(prediction, 1)}/5):** Air quality is acceptable, with low risk for most people.")
    else:
        st.success(f"✅ **GOOD (Level {round(prediction, 1)}/5):** Air quality is satisfactory, posing little or no health risk.")