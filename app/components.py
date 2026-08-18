import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def apply_custom_styles():
    """Applies clean UI styles and responsive headers."""
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            color: #6c757d;
            font-size: 1.05rem;
            margin-bottom: 1.2rem;
        }
        .forecast-card {
            background-color: #1e2130;
            border-radius: 10px;
            padding: 1rem;
            border: 1px solid #31374a;
            margin-bottom: 1rem;
        }
        .metric-title {
            font-size: 0.9rem;
            color: #8b949e;
        }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="main-header">Lahore AQI 3-Day Forecasting & Intelligence Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Real-time 72-Hour Multi-Model Air Quality Predictions (Random Forest, Ridge Regression, PyTorch Deep Learning) with SHAP Explainability</div>', unsafe_allow_html=True)

def render_system_status_bar(is_api_online: bool, models: dict):
    """Renders connection status of FastAPI backend and model readiness."""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if is_api_online:
            st.success("**FastAPI Backend**: Connected")
        else:
            st.info("**Backend Engine**: In-Process Engine (Active)")

    with col2:
        if models.get("rf") is not None or is_api_online:
            st.success("**Random Forest**: Ready")
        else:
            st.warning("**Random Forest**: Not Loaded")

    with col3:
        if models.get("ridge") is not None or is_api_online:
            st.success("**Ridge Regression**: Ready")
        else:
            st.warning("**Ridge Regression**: Not Loaded")

    with col4:
        if models.get("dl") is not None or is_api_online:
            st.success("**Deep Learning (PyTorch)**: Ready")
        else:
            st.warning("**Deep Learning**: Not Loaded")

def get_aqi_badge(val: float):
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

def render_3day_forecast_section(forecast_data: dict):
    """Renders 3-Day summary cards and 72-hour interactive trajectory chart."""
    if not forecast_data:
        st.error("Forecast data is currently unavailable. Please ensure OpenWeather API key is configured or local fallback is active.")
        return

    city = forecast_data.get("city", "Lahore, Pakistan")
    daily_summaries = forecast_data.get("daily_summaries", [])
    hourly_points = forecast_data.get("hourly_forecast", [])

    st.subheader(f"3-Day Air Quality Outlook: {city}")
    st.markdown("Multi-model 72-hour forecast synthesized from atmospheric pollution projections and our machine learning models.")

    # 1. Daily Summary Cards
    if daily_summaries:
        cols = st.columns(len(daily_summaries))
        for idx, day in enumerate(daily_summaries):
            with cols[idx]:
                st.markdown(f"### {day.get('day_name')} ({day.get('date')})")
                avg_val = day.get('avg_aqi', 3.0)
                badge = day.get('severity_badge', 'Moderate')
                color = day.get('severity_color', '#ffc107')
                
                st.metric(label="Average Expected AQI", value=f"{avg_val:.2f} / 5.0")
                st.markdown(f"<span style='color:{color}; font-weight:bold;'>Status: {badge}</span>", unsafe_allow_html=True)
                st.markdown(f"**Peak Pollution:** `{day.get('peak_aqi'):.2f}` at **{day.get('peak_hour')}**")
                st.markdown(f"**Primary Driver:** {day.get('dominant_pollutant')}")
                st.caption(f"Advisory: {day.get('health_advisory')}")

    st.markdown("---")

    # 2. 72-Hour Multi-Model Trajectory Chart
    st.subheader("72-Hour Multi-Model AQI Trajectory Comparison")
    
    if hourly_points:
        df_hourly = pd.DataFrame(hourly_points)
        df_hourly["datetime"] = pd.to_datetime(df_hourly["datetime"])

        fig, ax = plt.subplots(figsize=(12, 5))
        
        # Plot multi-model lines
        if "rf_aqi" in df_hourly and df_hourly["rf_aqi"].notna().any():
            ax.plot(df_hourly["datetime"], df_hourly["rf_aqi"], label="Random Forest", color="#2ecc71", linewidth=2)
        if "ridge_aqi" in df_hourly and df_hourly["ridge_aqi"].notna().any():
            ax.plot(df_hourly["datetime"], df_hourly["ridge_aqi"], label="Ridge Regression", color="#3498db", linewidth=2, linestyle="--")
        if "dl_aqi" in df_hourly and df_hourly["dl_aqi"].notna().any():
            ax.plot(df_hourly["datetime"], df_hourly["dl_aqi"], label="Deep Learning (PyTorch)", color="#9b59b6", linewidth=2, linestyle=":")
            
        ax.plot(df_hourly["datetime"], df_hourly["consensus_aqi"], label="Ensemble Consensus", color="#f1c40f", linewidth=3)

        # Reference Threshold Bands
        ax.axhline(y=2.5, color="#ffc107", linestyle="--", alpha=0.5, label="Moderate Threshold (2.5)")
        ax.axhline(y=3.5, color="#fd7e14", linestyle="--", alpha=0.5, label="Unhealthy Threshold (3.5)")
        ax.axhline(y=4.5, color="#dc3545", linestyle="--", alpha=0.5, label="Hazardous Threshold (4.5)")

        ax.set_title("Forecasted 72-Hour AQI Curve across Machine Learning Models", fontsize=13, pad=12)
        ax.set_xlabel("Time (Hourly Intervals)", fontsize=10)
        ax.set_ylabel("AQI Level (1.0 = Good, 5.0 = Hazardous)", fontsize=10)
        ax.set_ylim(0.5, 5.5)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", framealpha=0.8)
        fig.autofmt_xdate()
        plt.tight_layout()

        st.pyplot(fig)
        plt.close()

        # 3. Detailed Hourly Data Table (Collapsible)
        with st.expander("View Full 72-Hour Data Table & Pollutant Concentrations"):
            display_cols = [
                "datetime", "consensus_aqi", "rf_aqi", "ridge_aqi", "dl_aqi",
                "pm2_5", "pm10", "pm_ratio", "no2", "co", "o3", "severity_badge"
            ]
            valid_cols = [c for c in display_cols if c in df_hourly.columns]
            st.dataframe(
                df_hourly[valid_cols].rename(columns={
                    "datetime": "Timestamp",
                    "consensus_aqi": "Consensus AQI",
                    "rf_aqi": "Random Forest",
                    "ridge_aqi": "Ridge",
                    "dl_aqi": "Deep Learning",
                    "pm2_5": "PM2.5 (ug/m3)",
                    "pm10": "PM10 (ug/m3)",
                    "pm_ratio": "PM Ratio",
                    "no2": "NO2",
                    "co": "CO",
                    "o3": "Ozone",
                    "severity_badge": "Category"
                }),
                width="stretch"
            )
