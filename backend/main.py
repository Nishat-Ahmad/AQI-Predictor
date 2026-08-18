import os
import sys
from datetime import datetime

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
    """Computes SHAP feature importance contributions for a given input using the Random Forest model."""
    if model_service.rf is None:
        raise HTTPException(status_code=503, detail="Random Forest model not loaded for SHAP explanation")
    
    try:
        import shap
        feat_dict = payload.features.dict()
        cols = ["hour", "day", "month", "day_of_week", "co", "no2", "o3", "pm2_5", "pm10", "pm_ratio", "aqi_change_rate"]
        
        # Calculate derived features
        if feat_dict["pm_ratio"] is None and feat_dict["pm10"] > 0:
            feat_dict["pm_ratio"] = feat_dict["pm2_5"] / feat_dict["pm10"]
        elif feat_dict["pm_ratio"] is None:
            feat_dict["pm_ratio"] = 0.5
            
        row = [feat_dict[c] for c in cols]
        df_row = pd.DataFrame([row], columns=cols)
        
        explainer = shap.TreeExplainer(model_service.rf)
        shap_values = explainer.shap_values(df_row)
        base_value = float(explainer.expected_value[0] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value)
        
        contributions = []
        for feat_name, val, shap_val in zip(cols, row, shap_values[0]):
            contributions.append({
                "feature": feat_name,
                "value": float(val),
                "shap_value": float(shap_val)
            })
            
        contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        pred = float(model_service.rf.predict(df_row)[0])
        
        return ExplainResponse(
            base_value=base_value,
            prediction=pred,
            contributions=contributions
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SHAP explanation failed: {str(e)}")
