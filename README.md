# AQI Predictor: Global Multi-Model Atmospheric Forecasting & Intelligence Platform

An end-to-end Machine Learning, MLOps, and Full-Stack Platform for **Global Air Quality Index (AQI) forecasting, interactive geospatial telemetry, and multi-model consensus across 195+ world capitals**. 

Powered by **1-Year Multi-Continent Historical Datasets**, official **0–500 US EPA AQI Standard**, **Weights & Biases (W&B)** MLOps tracking, **FastAPI** backend services, and a modern **Astro + Vanilla CSS Glassmorphism** frontend featuring interactive Leaflet maps, 72-hour continuous trajectory curves, and SHAP explainability.

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
│   ├── model_service.py       # Multi-model inference (RF, Ridge, PyTorch MLP) on 0-500 scale
│   └── forecast_service.py    # 72-Hour OpenWeather trajectory synthesis & daily peaks
├── frontend/                  # Modern Astro & Vanilla CSS Dashboard
│   ├── src/                   # Astro UI Components & Modular layouts
│   │   ├── components/        # HeroAtmosphere, ModelConsensus, GlobalMap, ForecastCards, TrajectoryChart
│   │   └── pages/index.astro  # Primary single-page dashboard entrypoint
│   ├── css/                   # Curated pastel theme variables, cards, map & responsive layouts
│   └── js/                    # Client app, 195+ capitals database, fuzzy search, Leaflet & Chart.js
├── models/                    # Machine Learning Architectures & Training Pipelines
│   ├── randomForest.py        # Random Forest Regressor (100 Trees) + SHAP values
│   ├── ridge_regression.py    # RidgeCV with StandardScaler & L2 regularization
│   ├── deep_learning.py       # PyTorch 4-Layer MLP with BatchNorm & Dropout
│   └── compare_models.py      # Unified model evaluation & leaderboard benchmark
├── scripts/                   # Data Ingestion & MLOps Pipelines
│   ├── backfill_global_capitals.py # 1-Year historical backfill across 18 major world capitals (153,600 rows)
│   ├── sync_to_wandb.py       # Weights & Biases dataset and model registry synchronization
│   └── feature_pipeline.py    # Real-time atmospheric feature extraction
├── data/
│   └── historical_aqi_features.csv # 1-Year multi-capital training dataset (0-500 EPA targets)
├── artifacts/                 # Serialized production model weights (.pkl, .pt)
├── EDA/
├── requirements.txt           # Lean production backend dependencies for Vercel deployment
├── requirements-dev.txt       # Full ML, training, & MLOps dependencies (PyTorch, W&B, SHAP)
└── vercel.json                # Vercel deployment configuration
```

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
* **Random Forest Regressor** ($R^2 = 0.9997$, $\text{RMSE} = 1.03$): 100-tree ensemble modeling non-linear atmospheric interactions.
* **PyTorch Deep Learning MLP** ($R^2 = 0.9659$, $\text{RMSE} = 11.56$): 4-layer feedforward neural network with Batch Normalization, ReLU activations, and Dropout.
* **Ridge Regression** ($R^2 = 0.5613$, $\text{RMSE} = 0.62$): $L_2$-regularized linear statistical baseline.
* **Ensemble Consensus**: Concurrent real-time inference providing calibrated weighted-mean projections.

### 4. 72-Hour Trajectory & SHAP Explainability
* **Continuous 72-Hour Trajectory Curves**: Hourly Chart.js forecast with filter toggles (*All, Consensus, RF, Ridge, DL*).
* **3-Day Summary Cards**: Daily average AQI, peak smog hour, dominant pollutant, and health advisories.
* **SHAP Explainability**: Dynamic feature contribution breakdown showing exact pollutant impact on current air quality.

---

## Machine Learning Benchmarks (1-Year Global Capitals Dataset)

Trained on **153,600 historical hourly records** across 18 world capitals covering all climate zones:

| Model Architecture | Training Pipeline | Test $R^2$ | Test RMSE | Test MAE | W&B Artifact |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Random Forest** | Scikit-Learn 100 Trees | **`0.9997`** | **`1.0356`** | **`0.0507`** | `aqi_random_forest_model:v0` |
| **Deep Learning** | PyTorch 4-Layer MLP | **`0.9659`** | **`11.5667`** | **`6.6057`** | `aqi_deep_learning_model:v0` |
| **Ridge Regression** | StandardScaler + RidgeCV | **`0.5613`** | **`0.6280`** | **`0.4971`** | `aqi_ridge_model:v0` |

All runs, datasets, and model checkpoints are tracked on **Weights & Biases** under project `pearls-aqi-predictor`.

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
# 1. Fetch 1-year historical dataset across world capitals
python scripts/backfill_global_capitals.py

# 2. Retrain all 3 models
python models/randomForest.py
python models/ridge_regression.py
python models/deep_learning.py

# 3. Synchronize dataset & artifacts to Weights & Biases
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
| `GET` | `/forecast/3day?lat={lat}&lon={lon}&city={city}` | 72-hour forecast and 3-day daily summaries |
| `POST` | `/explain` | SHAP feature contribution impacts |
| `GET` | `/docs` | Interactive Swagger API documentation |

---

## MLOps & CI/CD Pipelines
 
Automated via **GitHub Actions** with pip dependency caching, timeout protection, concurrency control, and artifact persistence:
* **`ci_pipeline.yml`**: Python syntax compilation, Pydantic schema validation, and multi-model smoke testing on every push and PR against lean production requirements.
* **`training_pipeline.yml`**: Scheduled daily retraining of Random Forest, Ridge Regression, and PyTorch MLP models, automatic upload of trained weights (`.pkl`/`.pt`) to GitHub Action artifacts, and cloud synchronization to Weights & Biases Model Registry.
* **`feature_pipeline.yml`**: Hourly automated live atmospheric feature extraction and logging to W&B Feature Store.
* **`backfill_pipeline.yml`**: On-demand workflow to execute global multi-continent backfill across 18 capitals, generating and preserving `historical_aqi_features.csv` as a downloadable artifact and syncing to W&B.
