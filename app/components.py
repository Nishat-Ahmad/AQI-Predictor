import streamlit as st
import pandas as pd
import numpy as np

def apply_custom_styles():
    """Applies clean UI styles and responsive headers."""
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.1rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            color: #6c757d;
            font-size: 1.05rem;
            margin-bottom: 1.2rem;
        }
        .status-badge {
            padding: 0.3rem 0.6rem;
            border-radius: 6px;
            font-size: 0.85rem;
            font-weight: 600;
        }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="main-header">Lahore AQI Multi-Model Intelligence Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Multi-algorithm real-time predictions (Random Forest, Ridge Regression, Deep Learning) with SHAP Explainability</div>', unsafe_allow_html=True)

def render_model_status_bar(models):
    """Renders readiness status for all 3 models."""
    col1, col2, col3 = st.columns(3)
    with col1:
        if models.get("rf") is not None:
            st.success("**Random Forest**: Ready")
        else:
            st.warning("**Random Forest**: Not Loaded")

    with col2:
        if models.get("ridge") is not None:
            st.success("**Ridge Regression**: Ready")
        else:
            st.warning("**Ridge Regression**: Not Loaded")

    with col3:
        if models.get("dl") is not None:
            st.success("**Deep Learning (PyTorch)**: Ready")
        else:
            st.warning("**Deep Learning**: Not Loaded")

def render_sidebar_inputs():
    """Renders interactive sidebar inputs and returns a DataFrame of feature values and submit flag."""
    st.sidebar.header("Environmental Inputs")

    hour = st.sidebar.slider("Hour of Day (0-23)", 0, 23, 14)
    day = st.sidebar.slider("Day of Month", 1, 31, 15)
    month = st.sidebar.slider("Month (1-12)", 1, 12, 8)
    day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
    selected_day = st.sidebar.selectbox("Day of Week", list(day_map.keys()), index=0)
    day_of_week = day_map[selected_day]

    st.sidebar.subheader("Pollution Metrics (ug/m3)")
    co = st.sidebar.number_input("Carbon Monoxide (CO)", value=850.0, step=10.0)
    no2 = st.sidebar.number_input("Nitrogen Dioxide (NO2)", value=22.0, step=1.0)
    o3 = st.sidebar.number_input("Ozone (O3)", value=48.0, step=1.0)
    pm2_5 = st.sidebar.number_input("PM2.5", value=42.0, step=1.0)
    pm10 = st.sidebar.number_input("PM10", value=55.0, step=1.0)

    # Derived Features
    pm_ratio = round(pm2_5 / (pm10 + 1e-5), 4)
    st.sidebar.caption(f"Calculated PM Ratio (PM2.5 / PM10): **{pm_ratio}**")
    aqi_change_rate = st.sidebar.slider("AQI Hourly Change Rate (Delta vs prev hour)", min_value=-4.0, max_value=4.0, value=0.0, step=0.5)

    predict_btn = st.sidebar.button("Predict AQI across All Models", type="primary", use_container_width=True)

    input_df = pd.DataFrame([{
        'hour': hour,
        'day': day,
        'month': month,
        'day_of_week': day_of_week,
        'co': co,
        'no2': no2,
        'o3': o3,
        'pm2_5': pm2_5,
        'pm10': pm10,
        'pm_ratio': pm_ratio,
        'aqi_change_rate': aqi_change_rate
    }])

    return input_df, predict_btn

def get_aqi_badge(val):
    """Categorizes AQI based on OpenWeather 1-5 scale."""
    if val >= 4.5:
        return "Hazardous (5/5)", "#dc3545"
    elif val >= 3.5:
        return "Poor / Unhealthy (4/5)", "#fd7e14"
    elif val >= 2.5:
        return "Moderate (3/5)", "#ffc107"
    elif val >= 1.5:
        return "Fair (2/5)", "#0dcaf0"
    else:
        return "Good (1/5)", "#198754"

def render_prediction_cards(predictions):
    """Renders side-by-side comparative cards and consensus for all 3 models."""
    st.markdown("---")
    st.subheader("Side-by-Side Multi-Model Predictions")

    if not predictions:
        st.error("No active model predictions found. Please run `python models/compare_models.py` to train all models.")
        return

    c1, c2, c3 = st.columns(3)
    rf_val = predictions.get("Random Forest")
    ridge_val = predictions.get("Ridge Regression")
    dl_val = predictions.get("Deep Learning (PyTorch MLP)")

    with c1:
        if rf_val is not None:
            tag, color = get_aqi_badge(rf_val)
            st.metric(label="Random Forest", value=f"{rf_val:.2f} / 5.0")
            st.markdown(f"<span style='color:{color}; font-weight:bold;'>{tag}</span>", unsafe_allow_html=True)
        else:
            st.info("Random Forest not trained.")

    with c2:
        if ridge_val is not None:
            tag, color = get_aqi_badge(ridge_val)
            st.metric(label="Ridge Regression", value=f"{ridge_val:.2f} / 5.0")
            st.markdown(f"<span style='color:{color}; font-weight:bold;'>{tag}</span>", unsafe_allow_html=True)
        else:
            st.info("Ridge Regression not trained.")

    with c3:
        if dl_val is not None:
            tag, color = get_aqi_badge(dl_val)
            st.metric(label="Deep Learning (PyTorch)", value=f"{dl_val:.2f} / 5.0")
            st.markdown(f"<span style='color:{color}; font-weight:bold;'>{tag}</span>", unsafe_allow_html=True)
        else:
            st.info("Deep Learning not trained.")

    # Multi-model consensus
    avg_aqi = float(np.mean(list(predictions.values())))
    ens_tag, ens_color = get_aqi_badge(avg_aqi)
    st.markdown("<br>", unsafe_allow_html=True)
    st.info(f"**Consensus Ensemble AQI:** **`{avg_aqi:.2f}`** | Status: **{ens_tag}**")
