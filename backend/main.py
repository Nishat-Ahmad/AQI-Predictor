import os
import sys
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

# Ensure root directory is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from backend.schemas import (
    EnvironmentalInput,
    PredictionResponse,
    MultiModelPrediction,
    ForecastResponse,
    ExplainRequest,
    ExplainResponse
)
from backend.model_service import model_service
from backend.forecast_service import generate_3day_forecast
import pandas as pd
import numpy as np

app = FastAPI(
    title="Global AQI Predictor & 3-Day Forecast API",
    description="Production REST API serving multi-algorithm AQI predictions (Random Forest, Ridge Regression, PyTorch Deep Learning) and real-time 72-hour air quality forecasts for world capitals.",
    version="1.0.0"
)

# Enable CORS for all frontend origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")

# Serve CSS and JS static folders if frontend exists
if os.path.exists(os.path.join(FRONTEND_DIR, "css")):
    app.mount("/css", StaticFiles(directory=os.path.join(FRONTEND_DIR, "css")), name="css")

if os.path.exists(os.path.join(FRONTEND_DIR, "js")):
    app.mount("/js", StaticFiles(directory=os.path.join(FRONTEND_DIR, "js")), name="js")

# Mount public directory if present
if os.path.exists(os.path.join(FRONTEND_DIR, "public")):
    app.mount("/public", StaticFiles(directory=os.path.join(FRONTEND_DIR, "public")), name="public")

# Serve Frontend HTML
@app.get("/", tags=["Frontend"])
def serve_index():
    # If Astro build output exists, serve from dist/
    dist_html = os.path.join(FRONTEND_DIR, "dist", "index.html")
    if os.path.exists(dist_html):
        return FileResponse(dist_html)
    
    # Otherwise serve raw frontend/index.html
    raw_html = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(raw_html):
        return FileResponse(raw_html)
    return {"message": "Frontend not found, please check frontend/ directory."}

@app.get("/api/info", tags=["General"])
def api_info():
    return {
        "service": "Global AQI Prediction & 3-Day Forecasting Service",
        "status": "healthy",
        "docs": "/docs",
        "endpoints": {
            "health": "GET /health",
            "predict": "POST /predict",
            "forecast_3day": "GET /forecast/3day",
            "explain": "POST /explain"
        }
    }

@app.get("/health", tags=["Health"])
def health_check():
    """Health check endpoint confirming API status, timestamp, and loaded model status."""
    status = model_service.get_status()
    all_ready = any([status.get("random_forest"), status.get("ridge_regression"), status.get("deep_learning")])
    return {
        "status": "online" if all_ready else "degraded",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": status,
        "service": "Global AQI Prediction & 3-Day Forecasting Service"
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_aqi(payload: EnvironmentalInput):
    """Computes multi-model predictions and consensus AQI for single atmospheric input values."""
    feature_dict = payload.dict()
    preds = model_service.predict(feature_dict)
    
    return PredictionResponse(
        status="success",
        input_features=feature_dict,
        predictions=MultiModelPrediction(
            random_forest=preds["random_forest"],
            ridge_regression=preds["ridge_regression"],
            deep_learning=preds["deep_learning"],
            consensus_aqi=preds["consensus_aqi"],
            severity_badge=preds["severity_badge"],
            severity_color=preds["severity_color"]
        )
    )

@app.get("/forecast/3day", response_model=ForecastResponse, tags=["Forecasting"])
def get_3day_forecast(
    lat: float = 51.5074,
    lon: float = -0.1278,
    city: str = "London, United Kingdom"
):
    """Fetches OpenWeather future pollution data for any global capital, executes 72-hour multi-model inference, and returns daily summaries."""
    try:
        forecast_data = generate_3day_forecast(lat=lat, lon=lon, city=city)
        return forecast_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate 3-day forecast: {str(e)}")

@app.post("/explain", response_model=ExplainResponse, tags=["Explainability"])
def explain_prediction(payload: ExplainRequest):
    """Computes SHAP/Feature importance contributions for a given input using the Random Forest model."""
    if model_service.rf is None:
        raise HTTPException(status_code=503, detail="Random Forest model not loaded for SHAP explanation")
    
    try:
        feat_dict = payload.features.dict()
        input_df = pd.DataFrame([feat_dict])
        
        # Fill standard lag & rolling defaults if single point
        pm2_5 = float(feat_dict.get("pm2_5", 25.0))
        pm10 = float(feat_dict.get("pm10", 40.0))
        if "pm_ratio" not in input_df.columns or input_df["pm_ratio"].iloc[0] is None:
            input_df["pm_ratio"] = round(pm2_5 / (pm10 + 1e-5), 4)
        if "current_aqi" not in input_df.columns:
            from backend.model_service import calculate_epa_aqi
            input_df["current_aqi"] = calculate_epa_aqi(pm2_5, pm10, feat_dict.get("no2", 15.0), feat_dict.get("o3", 30.0), feat_dict.get("co", 500.0))
            
        base_aqi = float(input_df["current_aqi"].iloc[0])
        for lag in [1, 2, 3, 6, 12, 24, 48]:
            input_df[f"aqi_lag_{lag}h"] = base_aqi
        input_df["pm2_5_lag_1h"] = pm2_5
        input_df["pm2_5_lag_24h"] = pm2_5
        input_df["pm10_lag_24h"] = pm10
        input_df["aqi_roll_mean_24h"] = base_aqi
        input_df["aqi_roll_std_24h"] = 4.0
        input_df["aqi_roll_max_24h"] = base_aqi + 10.0
        input_df["aqi_roll_min_24h"] = max(0.0, base_aqi - 10.0)
        input_df["pm2_5_roll_mean_24h"] = pm2_5
        
        hour = int(feat_dict.get("hour", 12))
        month = int(feat_dict.get("month", 8))
        input_df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        input_df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
        input_df["month_sin"] = np.sin(2 * np.pi * month / 12.0)
        input_df["month_cos"] = np.cos(2 * np.pi * month / 12.0)
        input_df["lat"] = 51.5074
        input_df["lon"] = -0.1278
        
        cols = model_service.rf_features or list(input_df.columns)
        aligned_df = input_df.reindex(columns=cols, fill_value=0.0)
        
        raw_pred = model_service.rf.predict(aligned_df)[0]
        pred = float(raw_pred[0] if isinstance(raw_pred, (list, np.ndarray)) else raw_pred)
        
        try:
            import shap
            explainer = shap.TreeExplainer(model_service.rf)
            shap_values = explainer.shap_values(aligned_df)
            
            if isinstance(shap_values, list):
                sv = shap_values[0][0]
            elif len(shap_values.shape) == 3:
                sv = shap_values[0, :, 0]
            else:
                sv = shap_values[0]
                
            base_value = float(explainer.expected_value[0] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value)
            
            contributions = []
            for feat_name, val, shap_val in zip(cols, aligned_df.iloc[0].values, sv):
                contributions.append({
                    "feature": feat_name,
                    "value": float(val),
                    "shap_value": float(shap_val)
                })
        except Exception:
            base_value = 50.0
            importances = getattr(model_service.rf, "feature_importances_", [1.0 / len(cols)] * len(cols))
            delta = pred - base_value
            contributions = []
            for feat_name, val, imp in zip(cols, aligned_df.iloc[0].values, importances):
                contributions.append({
                    "feature": feat_name,
                    "value": float(val),
                    "shap_value": round(float(delta * imp), 2)
                })

        contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        
        return ExplainResponse(
            base_value=base_value,
            prediction=pred,
            contributions=contributions[:10]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")

