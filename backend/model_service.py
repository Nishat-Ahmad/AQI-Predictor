import os
import joblib
import pandas as pd
import numpy as np

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

class ModelService:
    def __init__(self):
        self.rf = None
        self.rf_features = None
        self.ridge = None
        self.ridge_features = None
        self.dl = None
        self.dl_scaler = None
        self.dl_features = None
        self.load_models()

    def load_models(self):
        """Loads all available models from artifacts/"""
        # 1. Random Forest
        rf_path = os.path.join(ARTIFACTS_DIR, "random_forest_aqi.pkl")
        if os.path.exists(rf_path):
            try:
                data = joblib.load(rf_path)
                if isinstance(data, dict):
                    self.rf = data.get("model")
                    self.rf_features = data.get("feature_names")
                else:
                    self.rf = data
                    self.rf_features = getattr(data, "feature_names_in_", None)
            except Exception as e:
                print(f"Error loading RF: {e}")

        # 2. Ridge Regression
        ridge_path = os.path.join(ARTIFACTS_DIR, "ridge_regression_aqi.pkl")
        if os.path.exists(ridge_path):
            try:
                data = joblib.load(ridge_path)
                if isinstance(data, dict):
                    self.ridge = data.get("pipeline")
                    self.ridge_features = data.get("feature_names")
                else:
                    self.ridge = data
            except Exception as e:
                print(f"Error loading Ridge: {e}")

        # 3. Deep Learning (PyTorch)
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
                self.dl = net
                self.dl_scaler = checkpoint.get("scaler")
                self.dl_features = checkpoint.get("feature_names")
            except Exception as e:
                print(f"Error loading PyTorch model: {e}")

    def get_status(self):
        return {
            "random_forest": self.rf is not None,
            "ridge_regression": self.ridge is not None,
            "deep_learning": self.dl is not None,
            "torch_available": TORCH_AVAILABLE
        }

    @staticmethod
    def get_severity(val: float):
        if val >= 4.5:
            return "Hazardous (5/5)", "#dc3545", "Emergency health conditions. Avoid all outdoor exertion."
        elif val >= 3.5:
            return "Poor / Unhealthy (4/5)", "#fd7e14", "Unhealthy air. Sensitive groups should remain indoors."
        elif val >= 2.5:
            return "Moderate (3/5)", "#ffc107", "Air quality is acceptable; moderate health concern for sensitive individuals."
        elif val >= 1.5:
            return "Fair (2/5)", "#0dcaf0", "Air quality is acceptable with low risk for most people."
        else:
            return "Good (1/5)", "#198754", "Air quality is satisfactory with minimal or no health risk."

    def predict(self, feature_dict: dict) -> dict:
        """Executes inference across all 3 models on a single input row."""
        input_df = pd.DataFrame([feature_dict])
        
        # Calculate pm_ratio if not provided
        if "pm_ratio" not in feature_dict or feature_dict["pm_ratio"] is None:
            pm2_5 = feature_dict.get("pm2_5", 0.0)
            pm10 = feature_dict.get("pm10", 0.0)
            input_df["pm_ratio"] = round(pm2_5 / (pm10 + 1e-5), 4)

        if "aqi_change_rate" not in feature_dict or feature_dict["aqi_change_rate"] is None:
            input_df["aqi_change_rate"] = 0.0

        results = {}

        # 1. Random Forest
        if self.rf is not None:
            cols = self.rf_features or list(input_df.columns)
            aligned = input_df.reindex(columns=cols, fill_value=0.0)
            results["random_forest"] = round(float(self.rf.predict(aligned)[0]), 2)

        # 2. Ridge Regression
        if self.ridge is not None:
            cols = self.ridge_features or list(input_df.columns)
            aligned = input_df.reindex(columns=cols, fill_value=0.0)
            results["ridge_regression"] = round(float(self.ridge.predict(aligned)[0]), 2)

        # 3. Deep Learning (PyTorch)
        if self.dl is not None and TORCH_AVAILABLE:
            cols = self.dl_features or list(input_df.columns)
            aligned = input_df.reindex(columns=cols, fill_value=0.0)
            if self.dl_scaler is not None:
                scaled = self.dl_scaler.transform(aligned)
            else:
                scaled = aligned.values
            with torch.no_grad():
                tensor_x = torch.tensor(scaled, dtype=torch.float32)
                dl_val = self.dl(tensor_x).item()
            results["deep_learning"] = round(float(dl_val), 2)

        vals = [v for v in results.values() if v is not None]
        consensus = round(float(np.mean(vals)), 2) if vals else 1.0
        badge, color, advisory = self.get_severity(consensus)

        return {
            "random_forest": results.get("random_forest"),
            "ridge_regression": results.get("ridge_regression"),
            "deep_learning": results.get("deep_learning"),
            "consensus_aqi": consensus,
            "severity_badge": badge,
            "severity_color": color,
            "health_advisory": advisory
        }

model_service = ModelService()
