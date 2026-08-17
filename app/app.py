import os
import sys

# Ensure both the app directory and project root are in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))

if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st

try:
    from components import (
        apply_custom_styles,
        render_model_status_bar,
        render_sidebar_inputs,
        render_prediction_cards
    )
    from model_loader import load_all_models, predict_multi_models
    from explainability import render_shap_explanation
except ImportError:
    from app.components import (
        apply_custom_styles,
        render_model_status_bar,
        render_sidebar_inputs,
        render_prediction_cards
    )
    from app.model_loader import load_all_models, predict_multi_models
    from app.explainability import render_shap_explanation

# 1. Page Configuration
st.set_page_config(
    page_title="Lahore AQI Multi-Model Predictor",
    layout="wide"
)

# 2. Header & Styles
apply_custom_styles()

# 3. Model Loading & Status
models = load_all_models()
render_model_status_bar(models)

# 4. Sidebar Inputs
input_df, predict_clicked = render_sidebar_inputs()

# 5. Prediction Execution & SHAP Explainability
if predict_clicked:
    predictions = predict_multi_models(models, input_df)
    render_prediction_cards(predictions)
    render_shap_explanation(
        rf_model=models.get("rf"),
        rf_features=models.get("rf_features"),
        input_df=input_df
    )
else:
    st.info("Adjust inputs in the sidebar and click 'Predict AQI across All Models' to view comparative predictions and live SHAP explainability.")