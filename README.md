# AQI Predictor: Global Multi-Model Atmospheric Forecasting & Intelligence Platform

An end-to-end Machine Learning, MLOps, and Full-Stack Platform for **Global Air Quality Index (AQI) multi-horizon forecasting, interactive geospatial telemetry, and multi-model consensus across 195+ world capitals**. 

Powered by **1-Year Multi-Continent Historical Time-Series**, official **0–500 US EPA AQI Standard**, **Weights & Biases (W&B)** MLOps tracking and Model Registry, **FastAPI** backend services, and a modern **Astro + Vanilla CSS Glassmorphism** frontend featuring interactive Leaflet maps, 72-hour continuous trajectory curves, and SHAP explainability.

---

## Architectural Notes: Feature Store & MLOps Infrastructure

> [!NOTE]
> **Weights & Biases (W&B) Usage**: Due to credit-card and billing constraints on commercial feature platforms (such as Hopsworks / Vertex AI Feature Store), **Weights & Biases** is utilized as our **Experiment Tracking, Dataset Artifact Versioning, and Model Registry** system. Feature snapshots and time-series tables are versioned and logged as W&B Artifacts, serving as the versioned storage layer for historical training and evaluation datasets.

### True Feature Store $\to$ Model $\to$ Forecast Pipeline
1. **Feature Ingestion**: Real-time atmospheric observations (current reading + past 48 hours) are ingested for the target location.
2. **Feature Engineering**: Time-series features are computed strictly from observations available up to time $t$:
   - Temporal lags: `aqi_lag_1h`, `aqi_lag_2h`, `aqi_lag_3h`, `aqi_lag_6h`, `aqi_lag_12h`, `aqi_lag_24h`, `aqi_lag_48h`
   - Pollutant lags: `pm2_5_lag_1h`, `pm2_5_lag_24h`, `pm10_lag_24h`
   - Rolling 24h aggregations: `aqi_roll_mean_24h`, `aqi_roll_std_24h`, `aqi_roll_max_24h`, `aqi_roll_min_24h`, `pm2_5_roll_mean_24h`
   - Cyclical temporal signals: `hour_sin`, `hour_cos`, `month_sin`, `month_cos`, `day_of_week`
3. **Multi-Horizon ML Inference**: Features are fed directly into trained machine learning models (Random Forest, Ridge Regression, PyTorch MLP) to forecast true future horizons (**+24h, +48h, +72h**) without wrapping external pollutant predictions.
4. **Out-of-Time Temporal Evaluation**: Models are validated on an out-of-time test set (chronological 80/20 split) with zero future-to-past data leakage.

---

## System Architecture

```
AQI-Predictor/
├── .github/workflows/         # Automated CI/CD & MLOps GitHub Actions
│   ├── ci_pipeline.yml        # CI compilation, test suite & backend smoke tests
│   ├── training_pipeline.yml  # Scheduled daily model retraining & W&B logging
│   ├── feature_pipeline.yml   # Hourly live feature pipeline
│   └── backfill_pipeline.yml  # 1-Year multi-capital historical backfill & sync
├── api/
│   └── index.py               # Vercel serverless ASGI entrypoint
├── backend/                   # FastAPI REST API & Model Inference Engine
│   ├── main.py                # REST endpoints, static routing & CORS configuration
│   ├── schemas.py             # Pydantic data validation schemas
│   ├── model_service.py       # Multi-horizon inference (RF, Ridge, PyTorch MLP) on 0-500 scale
│   └── forecast_service.py    # Feature Store -> Multi-Model 72-Hour trajectory synthesis
├── frontend/                  # Modern Astro & Vanilla CSS Dashboard
│   ├── src/                   # Astro UI Components & Modular layouts
│   │   ├── components/        # HeroAtmosphere, ModelConsensus, GlobalMap, ForecastCards, TrajectoryChart
│   │   └── pages/index.astro  # Primary single-page dashboard entrypoint
│   ├── css/                   # Curated pastel theme variables, cards, map & responsive layouts
│   └── js/                    # Client app, 195+ capitals database, fuzzy search, Leaflet & Chart.js
├── models/                    # Machine Learning Architectures & Training Pipelines
│   ├── randomForest.py        # Multi-Horizon Random Forest Regressor (100 Trees) + SHAP values
│   ├── ridge_regression.py    # Multi-Horizon RidgeCV with StandardScaler & L2 regularization
│   ├── deep_learning.py       # Multi-Horizon PyTorch 4-Layer MLP with BatchNorm & Dropout
│   └── compare_models.py      # Unified out-of-time benchmarking & leaderboard
├── scripts/                   # Data Ingestion & MLOps Pipelines
│   ├── backfill_global_capitals.py # 1-Year historical backfill & temporal feature generation (151,440 rows)
│   ├── sync_to_wandb.py       # Weights & Biases dataset and model registry synchronization
│   └── feature_pipeline.py    # Real-time atmospheric feature extraction snapshot
├── data/
│   └── historical_aqi_features.csv # 1-Year multi-horizon feature matrix
├── artifacts/                 # Serialized production model weights (.pkl, .pt)
├── EDA/
├── requirements.txt           # Lean production backend dependencies for Vercel deployment
├── requirements-dev.txt       # Full ML, training, & MLOps dependencies (PyTorch, W&B, SHAP)
└── vercel.json                # Vercel deployment configuration
```

---

## Machine Learning Benchmarks (Out-of-Time Chronological Test Set)

Trained on **121,152 historical hourly samples** and evaluated on **30,288 out-of-time test samples** across 18 world capitals covering all global climate zones:

| Model Architecture | Training Pipeline | Test $R^2$ (+24h) | Test $R^2$ (+48h) | Test $R^2$ (+72h) | Mean $R^2$ | Test RMSE (+24h) | Test MAE (+24h) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ridge Regression** | StandardScaler + RidgeCV | **`0.7525`** | **`0.6551`** | **`0.6097`** | **`0.6724`** | `22.69` | `14.18` |
| **Random Forest** | Scikit-Learn (100 Trees, Depth 16) | **`0.7399`** | **`0.6360`** | **`0.6029`** | **`0.6596`** | `23.26` | `14.57` |
| **Deep Learning (MLP)** | PyTorch 4-Layer Neural Network | **`0.7021`** | **`0.6294`** | **`0.5886`** | **`0.6400`** | `24.90` | `15.78` |

*All runs, datasets, and model checkpoints are tracked on **Weights & Biases** under project `pearls-aqi-predictor`.*

---

## Key Features

### 1. Global Interactive Map & World Capitals Database
* **195+ Sovereign World Capitals**: Instant one-click selection across all continents (Tokyo, London, Washington D.C., Islamabad, Cairo, Brasília, Canberra, Paris, Berlin, etc.).
* **Fuzzy Search Dropdown**: Live real-time search with keyboard arrow navigation and clear button.
* **Custom Dark Matter Leaflet Map**: Smooth coordinate flying, dynamic pulse pins, and popup atmospheric metrics.

### 2. Official 0–500 US EPA AQI Standard
* Evaluates ground-truth concentrations of $\text{PM}_{2.5}$, $\text{PM}_{10}$, $\text{NO}_2$, $\text{O}_3$, and $\text{CO}$ using official piecewise-linear EPA breakpoints:
  $$I_p = \frac{I_{\text{Hi}} - I_{\text{Lo}}}{\text{BP}_{\text{Hi}} - \text{BP}_{\text{Lo}}} (C_p - \text{BP}_{\text{Lo}}) + I_{\text{Lo}}, \quad \text{AQI} = \max_p(I_p)$$
* **Standard Health Severity Bands**:
  * `0 – 50`: **Good** (Pastel Sage Green `#86efac`)
  * `51 – 100`: **Moderate** (Pastel Butter Yellow `#fde047`)
  * `101 – 150`: **Unhealthy for Sensitive Groups** (Pastel Apricot `#fdba74`)
  * `151 – 200`: **Unhealthy** (Pastel Soft Coral `#fca5a5`)
  * `201 – 300`: **Very Unhealthy** (Pastel Lavender `#c4b5fd`)
  * `301 – 500+`: **Hazardous** (Pastel Rose `#f472b6`)

### 3. Multi-Model AI Suite & Ensemble Consensus
* **Random Forest Regressor**: 100-tree ensemble capturing non-linear atmospheric dynamics.
* **PyTorch Deep Learning MLP**: 4-layer feedforward neural network with Batch Normalization, ReLU activations, and Dropout.
* **Ridge Regression**: $L_2$-regularized linear statistical baseline.
* **Ensemble Consensus**: Concurrent real-time inference providing calibrated weighted-mean projections across horizons.

### 4. 72-Hour Trajectory & SHAP Explainability
* **Continuous 72-Hour Trajectory Curves**: Hourly Chart.js forecast with filter toggles (*All, Consensus, RF, Ridge, DL*).
* **3-Day Summary Cards**: Daily average AQI, peak smog hour, dominant pollutant, and health advisories.
* **SHAP Explainability**: Dynamic feature contribution breakdown showing exact temporal lag and pollutant impacts on forecasted air quality.

---

## Getting Started

### 1. Clone & Setup Environment
```bash
git clone https://github.com/Nishat-Ahmad/AQI-Predictor.git
cd AQI-Predictor

# Create and activate Python virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install production backend dependencies (Lean Vercel bundle)
pip install -r requirements.txt

# Or install full ML training, pipeline & MLOps dependencies (PyTorch, W&B, SHAP)
pip install -r requirements-dev.txt
```

### 2. Configure Environment Variables
Create a `.env` file in the project root:
```ini
OPENWEATHER_API_KEY=your_openweather_api_key
WANDB_API_KEY=your_wandb_api_key
```

### 3. Run Historical Backfill & Model Retraining
```bash
# 1. Generate 1-year multi-horizon feature matrix
python scripts/backfill_global_capitals.py

# 2. Benchmark and train all 3 multi-horizon models
python models/compare_models.py

# 3. Synchronize dataset & artifacts to Weights & Biases Model Registry
python scripts/sync_to_wandb.py
```

### 4. Start the Application Locally
```bash
# Start FastAPI backend (port 8000)
uvicorn backend.main:app --port 8000 --reload

# In a separate terminal, start Astro frontend dev server (port 3000)
npm run dev --prefix frontend
```
Navigate to **`http://localhost:8000/`** (FastAPI) or **`http://localhost:3000/`** (Astro).

---

## API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the interactive AQI dashboard |
| `GET` | `/health` | Health check & model readiness status |
| `POST` | `/predict` | Multi-model AQI inference for custom pollutant inputs |
| `GET` | `/forecast/3day?lat={lat}&lon={lon}&city={city}` | 72-hour multi-horizon forecast and daily summaries |
| `POST` | `/explain` | SHAP feature contribution impacts for forecasted horizon |
| `GET` | `/docs` | Interactive Swagger API documentation |

---

## MLOps & CI/CD Pipelines
 
Automated via **GitHub Actions** with pip dependency caching, timeout protection, concurrency control, and artifact persistence:
* **`ci_pipeline.yml`**: Python syntax compilation, Pydantic schema validation, and multi-model smoke testing on every push and PR against lean production requirements.
* **`training_pipeline.yml`**: Scheduled daily retraining of multi-horizon Random Forest, Ridge Regression, and PyTorch MLP models, automatic upload of trained weights (`.pkl`/`.pt`) to GitHub Action artifacts, and cloud synchronization to Weights & Biases Model Registry.
* **`feature_pipeline.yml`**: Hourly automated live atmospheric feature extraction and logging to W&B Feature Store.
* **`backfill_pipeline.yml`**: On-demand workflow to execute global multi-continent backfill across 18 capitals, generating and preserving `historical_aqi_features.csv` as a downloadable artifact and syncing to W&B.
