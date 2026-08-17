import os
import joblib
import streamlit as st

# Check for PyTorch availability
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True

    class AQINeuralNet(nn.Module):
        def __init__(self, input_dim):
            super(AQINeuralNet, self).__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            )

        def forward(self, x):
            return self.network(x)
except ImportError:
    TORCH_AVAILABLE = False
    AQINeuralNet = None

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")

@st.cache_resource
def load_all_models():
    """Loads Random Forest, Ridge Regression, and Deep Learning models from local artifacts or W&B."""
    models = {
        "rf": None,
        "rf_features": None,
        "ridge": None,
        "ridge_features": None,
        "dl": None,
        "dl_scaler": None,
        "dl_features": None,
        "torch_available": TORCH_AVAILABLE
    }

    # 1. Load Random Forest
    rf_path = os.path.join(ARTIFACTS_DIR, "random_forest_aqi.pkl")
    if os.path.exists(rf_path):
        try:
            data = joblib.load(rf_path)
            if isinstance(data, dict):
                models["rf"] = data.get("model")
                models["rf_features"] = data.get("feature_names")
            else:
                models["rf"] = data
                models["rf_features"] = getattr(data, "feature_names_in_", None)
        except Exception:
            pass

    # 2. Load Ridge Regression
    ridge_path = os.path.join(ARTIFACTS_DIR, "ridge_regression_aqi.pkl")
    if os.path.exists(ridge_path):
        try:
            data = joblib.load(ridge_path)
            if isinstance(data, dict):
                models["ridge"] = data.get("pipeline")
                models["ridge_features"] = data.get("feature_names")
            else:
                models["ridge"] = data
        except Exception:
            pass

    # 3. Load Deep Learning Model (PyTorch)
    dl_path = os.path.join(ARTIFACTS_DIR, "deep_learning_aqi.pt")
    if os.path.exists(dl_path) and TORCH_AVAILABLE:
        try:
            try:
                checkpoint = torch.load(dl_path, map_location=torch.device("cpu"), weights_only=False)
            except TypeError:
                checkpoint = torch.load(dl_path, map_location=torch.device("cpu"))
            input_dim = checkpoint.get("input_dim", 11)
            net = AQINeuralNet(input_dim)
            net.load_state_dict(checkpoint["state_dict"])
            net.eval()
            models["dl"] = net
            models["dl_scaler"] = checkpoint.get("scaler")
            models["dl_features"] = checkpoint.get("feature_names")
        except Exception as e:
            print(f"Error loading PyTorch model: {e}")

    return models

def predict_multi_models(models, input_df):
    """Generates predictions across all available loaded models."""
    predictions = {}

    # 1. Random Forest
    if models.get("rf") is not None:
        rf_model = models["rf"]
        rf_cols = models.get("rf_features") or list(input_df.columns)
        aligned = input_df.reindex(columns=rf_cols, fill_value=0.0)
        predictions["Random Forest"] = float(rf_model.predict(aligned)[0])

    # 2. Ridge Regression
    if models.get("ridge") is not None:
        ridge_pipe = models["ridge"]
        ridge_cols = models.get("ridge_features") or list(input_df.columns)
        aligned = input_df.reindex(columns=ridge_cols, fill_value=0.0)
        predictions["Ridge Regression"] = float(ridge_pipe.predict(aligned)[0])

    # 3. Deep Learning Model (PyTorch)
    if models.get("dl") is not None and TORCH_AVAILABLE:
        dl_model = models["dl"]
        dl_scaler = models.get("dl_scaler")
        dl_cols = models.get("dl_features") or list(input_df.columns)
        aligned = input_df.reindex(columns=dl_cols, fill_value=0.0)

        if dl_scaler is not None:
            scaled_input = dl_scaler.transform(aligned)
        else:
            scaled_input = aligned.values

        with torch.no_grad():
            tensor_x = torch.tensor(scaled_input, dtype=torch.float32)
            val = dl_model(tensor_x).item()
        predictions["Deep Learning (PyTorch MLP)"] = float(val)

    return predictions
