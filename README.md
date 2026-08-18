# Lahore AQI Multi-Model Forecasting & Intelligence Platform

An end-to-end Machine Learning and MLOps system for 3-Day Air Quality Index (AQI) forecasting and real-time inference in Lahore, Pakistan. Powered by OpenWeather data, Weights & Biases (W&B) for artifact tracking and model registry, a FastAPI backend service, and a custom glassmorphism web dashboard with live Chart.js trajectories and SHAP explainability.

---

## Architecture Overview

```
AQI-Predictor/
├── .env                       # API keys (OpenWeather, W&B)
├── requirements.txt           # Project dependencies (FastAPI, PyTorch, Scikit-Learn, SHAP)
├── Procfile                   # Cloud deployment entrypoint (Render / Railway)
├── render.yaml                # Render deployment specification
├── scripts/
│   ├── backfill_pipeline.py   # Historical data ingestion (2 years) & feature engineering
│   └── feature_pipeline.py    # Real-time data fetching & delta feature extraction
├── models/
│   ├── compare_models.py      # Unified benchmark script for all models
│   ├── randomForest.py        # Random Forest Regressor training with SHAP
│   ├── ridge_regression.py    # Ridge Regression baseline with CV
│   └── deep_learning.py       # PyTorch Deep Learning MLP Regressor
├── backend/
│   ├── main.py                # FastAPI app serving REST API & static web UI
│   ├── schemas.py             # Pydantic request/response validation
│   ├── model_service.py       # Multi-model inference engine (RF, Ridge, PyTorch)
│   └── forecast_service.py    # 72-hour OpenWeather forecast synthesizer
├── frontend/
│   ├── index.html             # Modern HTML5 dashboard layout
│   ├── css/style.css          # Custom glassmorphism dark theme
│   └── js/app.js              # Chart.js dynamic time-series & SHAP visualizer
└── artifacts/                 # Local model checkpoints and dataset artifacts
```

---

## Key Features

1. **3-Day Air Quality Forecast (72-Hour Horizon)**:
   - Fetches 72 hours of future atmospheric projections from OpenWeather.
   - Computes derived features (`pm_ratio` and `aqi_change_rate`) across the time series.
   - Generates multi-model predictions comparing **Random Forest**, **Ridge Regression**, and **Deep Learning (PyTorch)** along with daily summary cards (Avg AQI, Peak Hours, Dominant Pollutant).

2. **Multi-Model AI Suite**:
   - **Ridge Regression**: Linear statistical baseline with cross-validated regularization.
   - **Random Forest**: Non-linear tree ensemble with feature importances.
   - **Deep Learning**: PyTorch Multi-Layer Perceptron (MLP) with Batch Normalization and Dropout.
   - **Unified Benchmarking**: Side-by-side comparison script logging leaderboard metrics (RMSE, MAE, R2) to Weights & Biases.

3. **FastAPI Backend REST Service**:
   - Production REST API with automated Swagger documentation at `/docs`.
   - Endpoints:
     - `GET /` : Serves the custom web dashboard.
     - `GET /health` : Backend & model readiness status.
     - `POST /predict` : Single-point multi-model inference.
     - `GET /forecast/3day` : 72-hour forecast with daily aggregations.
     - `POST /explain` : SHAP feature contribution breakdown.

4. **Custom Frontend Dashboard**:
   - Built with Vanilla HTML5, CSS3, and JavaScript with zero heavy framework bloat.
   - Interactive 72-hour multi-model trajectory chart powered by Chart.js.
   - Live SHAP feature importance visualizer.
   - Day 1, Day 2, and Day 3 outlook summary cards with health advisories.

---

## Quickstart

### 1. Environment Setup
Create a `.env` file in the project root:
```env
OPENWEATHER_API_KEY=your_openweather_api_key
WANDB_API_KEY=your_wandb_api_key
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run Data Pipelines & Training
```bash
# 1. Fetch historical data & log to W&B
python scripts/backfill_pipeline.py

# 2. Train and benchmark all 3 models
python models/compare_models.py
```

### 3. Start the Web Application
```bash
uvicorn backend.main:app --port 8000 --reload
```
* **Web Dashboard**: Open [http://localhost:8000/](http://localhost:8000/)
* **Interactive API Swagger Docs**: Open [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Cloud Deployment (Render / Railway / Vercel)

### Deploying to Render
1. Push your repository to GitHub.
2. Create a new **Web Service** on [Render](https://render.com).
3. Connect your repository. Render automatically reads [`render.yaml`](file:///d:/Code/AQI-Predictor/render.yaml) and [`Procfile`](file:///d:/Code/AQI-Predictor/Procfile).
4. Add environment variables: `OPENWEATHER_API_KEY` and `WANDB_API_KEY`.
5. Your application will be live at `https://your-app-name.onrender.com/` serving both the custom web UI and the FastAPI REST API!
