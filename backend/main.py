import os
import sys

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
    title="Lahore AQI Predictor & 3-Day Forecast API",
    description="Production REST API serving multi-algorithm AQI predictions (Random Forest, Ridge Regression, PyTorch Deep Learning) and real-time 72-hour air quality forecasts for Lahore, Pakistan.",
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

# Serve Custom Frontend Dashboard at Root URL
@app.get("/", tags=["UI"])
def serve_ui():
    index_file = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Frontend index.html not found, please check frontend/ directory."}

@app.get("/api/info", tags=["General"])
def api_info():
    return {
        "service": "Lahore AQI Prediction & 3-Day Forecasting Service",
        "status": "healthy",
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
            "predict": "POST /predict",
            "forecast_3day": "GET /forecast/3day",
            "explain": "POST /explain"
        }
    }

@app.get("/health", tags=["Health"])
def health_check():
    status = model_service.get_status()
    all_ready = any([status["random_forest"], status["ridge_regression"], status["deep_learning"]])
    return {
        "status": "online" if all_ready else "degraded",
        "models_loaded": status
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
    lat: float = 31.5204,
    lon: float = 74.3587,
    city: str = "Lahore, Pakistan"
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
        raise HTTPException(status_code=503, detail="Random Forest model not loaded for SHAP explanation.")

    try:
        import shap
        input_df = pd.DataFrame([payload.features.dict()])
        cols = model_service.rf_features or list(input_df.columns)
        aligned = input_df.reindex(columns=cols, fill_value=0.0)

        explainer = shap.TreeExplainer(model_service.rf)
        shap_vals = explainer(aligned)

        base_val = float(explainer.expected_value[0]) if isinstance(explainer.expected_value, (list, tuple)) else float(explainer.expected_value)
        pred_val = float(model_service.rf.predict(aligned)[0])

        contributions = []
        for feat, val in zip(aligned.columns, shap_vals[0].values):
            contributions.append({
                "feature": feat,
                "value": float(aligned[feat].iloc[0]),
                "shap_impact": round(float(val), 4)
            })

        contributions.sort(key=lambda x: abs(x["shap_impact"]), reverse=True)

        return ExplainResponse(
            base_value=round(base_val, 2),
            prediction=round(pred_val, 2),
            contributions=contributions
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explainability calculation error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=True)
