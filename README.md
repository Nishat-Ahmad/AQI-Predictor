# Lahore AQI Multi-Model Forecasting & Intelligence Platform

An end-to-end Machine Learning and MLOps system for 3-Day Air Quality Index (AQI) forecasting and real-time inference in Lahore, Pakistan. Powered by OpenWeather data, Weights & Biases (W&B) for artifact tracking and model registry, a FastAPI backend service, and a modular Streamlit dashboard with SHAP explainability.

---

## Architecture Overview

```
AQI-Predictor/
├── .env                       # API keys (OpenWeather, W&B)
├── requirements.txt           # Project dependencies
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
│   ├── main.py                # FastAPI app with Swagger UI docs (/docs)
│   ├── schemas.py             # Pydantic request/response validation
│   ├── model_service.py       # Multi-model inference engine (RF, Ridge, PyTorch)
│   └── forecast_service.py    # 72-hour OpenWeather forecast synthesizer
├── app/
│   ├── app.py                 # Streamlit entrypoint with Dual Tabs
│   ├── api_client.py          # HTTP client communicating with FastAPI backend
│   ├── components.py          # 3-Day forecast charts, metric cards, status bars
│   ├── model_loader.py        # Local model loading fallback
│   └── explainability.py      # SHAP waterfall plots & feature contribution tables
└── artifacts/                 # Local model checkpoints and dataset artifacts
```

---

## Key Features

1. **3-Day Air Quality Forecast (72 Hours)**:
   - Fetches 72 hours of future atmospheric projections from OpenWeather.
   - Computes derived features (`pm_ratio` and `aqi_change_rate`) across the time series.
   - Generates multi-model predictions comparing **Random Forest**, **Ridge Regression**, and **Deep Learning (PyTorch)** along with daily summary cards (Avg AQI, Peak Hours, Dominant Pollutant).

2. **Multi-Model Suite**:
   - **Ridge Regression**: Linear statistical baseline with cross-validated regularization.
   - **Random Forest**: Non-linear tree ensemble with feature importances.
   - **Deep Learning**: PyTorch Multi-Layer Perceptron (MLP) with Batch Normalization and Dropout.
   - **Unified Benchmarking**: Side-by-side comparison script logging leaderboard metrics (RMSE, MAE, R2) to Weights & Biases.

3. **FastAPI Backend REST Service**:
   - Dedicated REST API service with automated Swagger documentation at `/docs`.
   - Endpoints:
     - `GET /health` : Backend & model readiness status.
     - `POST /predict` : Single-point multi-model inference.
     - `GET /forecast/3day` : 72-hour forecast with daily aggregations.
     - `POST /explain` : SHAP feature contribution breakdown.

4. **Streamlit Frontend with SHAP Explainability**:
   - **Tab 1: 3-Day Live Forecast**: Real-time 72-hour multi-model trajectory chart & daily cards.
   - **Tab 2: Scenario Simulator & SHAP**: Interactive slider simulation and live local SHAP waterfall diagrams.

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

### 3. Start the Services

#### Option A: Start FastAPI Backend + Streamlit Frontend
```bash
# Terminal 1: Start FastAPI REST API
uvicorn backend.main:app --port 8000 --reload

# Terminal 2: Start Streamlit Frontend
streamlit run app/app.py
```
* Interactive Swagger Docs: `http://localhost:8000/docs`
* Streamlit Web App: `http://localhost:8501`

#### Option B: Standalone Streamlit (with In-Process Engine)
```bash
streamlit run app/app.py
```

---

## Cloud Deployment (Render / Railway / Streamlit Cloud)

### FastAPI Backend on Render / Railway
1. Push your repository to GitHub.
2. Create a new **Web Service** on [Render](https://render.com) or [Railway](https://railway.app).
3. Set the start command:
   ```bash
   uvicorn backend.main:app --host 0.0.0.0 --port $PORT
   ```
4. Add environment variables (`OPENWEATHER_API_KEY`, `WANDB_API_KEY`).

### Streamlit Frontend on Streamlit Cloud
1. Deploy to [Streamlit Community Cloud](https://streamlit.io/cloud) pointing to `app/app.py`.
2. *(Optional)* Add `API_URL = "https://your-api.onrender.com"` under App Settings $\to$ Secrets. If omitted, the app will seamlessly run using its built-in in-process inference engine.
