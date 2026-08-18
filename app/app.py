import os
import sys
import pandas as pd

# Ensure app directory and project root are in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))

if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st

try:
    from components import (
        apply_custom_styles,
        render_system_status_bar,
        render_3day_forecast_section
    )
    from model_loader import load_all_models
    from explainability import render_shap_explanation
    from api_client import api_client
except ImportError:
    from app.components import (
        apply_custom_styles,
        render_system_status_bar,
        render_3day_forecast_section
    )
    from app.model_loader import load_all_models
    from app.explainability import render_shap_explanation
    from app.api_client import api_client

# 1. Page Configuration
st.set_page_config(
    page_title="Lahore AQI 3-Day Forecasting Platform",
    layout="wide"
)

# 2. Header & Styles
apply_custom_styles()

# 3. Model Loading & System Connectivity
is_backend_online = api_client.is_backend_online()
models = load_all_models()
render_system_status_bar(is_backend_online, models)

st.markdown("<br>", unsafe_allow_html=True)

# 4. Live 3-Day Forecast Controls & Synthesis
col_title, col_btn = st.columns([6, 1])
with col_btn:
    refresh_clicked = st.button("Refresh Forecast", use_container_width=True)

with st.spinner("Synthesizing real-time 72-hour multi-model air quality forecast for Lahore..."):
    # Query FastAPI backend or local fallback service
    forecast_data = None
    if is_backend_online:
        forecast_data = api_client.get_3day_forecast()
    
    if not forecast_data:
        try:
            from backend.forecast_service import generate_3day_forecast
            res = generate_3day_forecast()
            forecast_data = res.dict()
        except Exception as e:
            print(f"Local forecast error: {e}")

# 5. Render 3-Day Forecast Cards & 72-Hour Trajectory Chart
render_3day_forecast_section(forecast_data)

# 6. Live SHAP Feature Explainability for Current Forecast Conditions
if forecast_data and "hourly_forecast" in forecast_data and forecast_data["hourly_forecast"]:
    current_hourly = forecast_data["hourly_forecast"][0]
    feature_dict = {
        'hour': int(current_hourly.get('hour', 12)),
        'day': int(current_hourly.get('day', 15)),
        'month': int(current_hourly.get('month', 8)),
        'day_of_week': int(pd.to_datetime(current_hourly.get('datetime')).weekday()),
        'co': float(current_hourly.get('co', 800.0)),
        'no2': float(current_hourly.get('no2', 20.0)),
        'o3': float(current_hourly.get('o3', 45.0)),
        'pm2_5': float(current_hourly.get('pm2_5', 35.0)),
        'pm10': float(current_hourly.get('pm10', 48.0)),
        'pm_ratio': float(current_hourly.get('pm_ratio', 0.72)),
        'aqi_change_rate': float(current_hourly.get('aqi_change_rate', 0.0))
    }
    input_df = pd.DataFrame([feature_dict])
    
    render_shap_explanation(
        rf_model=models.get("rf"),
        rf_features=models.get("rf_features"),
        input_df=input_df
    )