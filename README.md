# Lahore AQI Predictor & Intelligence Dashboard

An end-to-end Machine Learning and MLOps pipeline for predicting Air Quality Index (AQI) in Lahore using OpenWeather data, Weights & Biases (W&B) for artifact tracking and experiment logging, and a modular Streamlit dashboard with SHAP explainability.

---

## Architecture Overview

```
AQI-Predictor/
├── .env                       # API keys (OpenWeather, W&B)
├── requirements.txt           # Project dependencies
├── scripts/
│   ├── backfill_pipeline.py   # Historical data ingestion (2 years) & feature engineering
│   └── feature_pipeline.py    # Real-time data fetching & delta feature extraction
├── models/
│   ├── compare_models.py      # Unified benchmark script for all models
│   ├── randomForest.py        # Random Forest Regressor training with SHAP
│   ├── ridge_regression.py    # Ridge Regression baseline with CV
│   └── deep_learning.py       # PyTorch Deep Learning MLP Regressor
├── app/
│   ├── app.py                 # Streamlit entrypoint
│   ├── components.py          # UI layout, inputs, and metric cards
│   ├── model_loader.py        # Multi-model caching and inference logic
│   └── explainability.py      # SHAP waterfall plots and contribution tables
└── artifacts/                 # Local model checkpoints and dataset artifacts
```

---

## Key Features

1. **Feature Engineering**:
   - **Time Features**: Hour, day, month, day of week.
   - **Derived Features**: Hourly AQI rate of change (`aqi_change_rate`) and fine-to-coarse particulate ratio (`pm_ratio` = PM2.5 / PM10).

2. **Multi-Model Suite**:
   - **Ridge Regression**: Linear statistical baseline with cross-validated regularization.
   - **Random Forest**: Non-linear ensemble model.
   - **Deep Learning**: PyTorch multi-layer perceptron (MLP) with batch normalization and dropout.
   - **Unified Benchmarking**: Side-by-side comparison script logging leaderboard metrics (RMSE, MAE, R2) to Weights & Biases.

3. **SHAP Explainability**:
   - Real-time local SHAP waterfall diagrams showing feature-by-feature impact on every individual prediction.
   - Color-coded feature contribution tables.

4. **Multi-Model Dashboard**:
   - Simultaneous side-by-side predictions across all 3 models with health advisory severity ratings and consensus average.

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

### 2. Run Data Pipelines
Fetch and engineer historical features, then log dataset artifact to W&B:
```bash
python scripts/backfill_pipeline.py
```

Test real-time feature extraction:
```bash
python scripts/feature_pipeline.py
```

### 3. Train & Benchmark Models
Train and compare all three models side-by-side:
```bash
python models/compare_models.py
```

*(Optional) Train individual models:*
```bash
python models/randomForest.py
python models/ridge_regression.py
python models/deep_learning.py
```

### 4. Launch the Dashboard
Run the Streamlit web application:
```bash
streamlit run app/app.py
```
