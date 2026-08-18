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

def calculate_epa_aqi(pm2_5: float, pm10: float, no2: float = 0.0, o3: float = 0.0, co: float = 0.0) -> int:
    """Calculates official US EPA AQI (0-500 scale) based on standard pollutant breakpoints."""
    def calc_single(c, breakpoints):
        for c_low, c_high, i_low, i_high in breakpoints:
            if c_low <= c <= c_high:
                return round(((i_high - i_low) / (c_high - c_low)) * (c - c_low) + i_low)
        if c > breakpoints[-1][1]:
            return min(500, round(breakpoints[-1][3] + (c - breakpoints[-1][1])))
        return 0

    pm25_bp = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500)
    ]
    
    pm10_bp = [
        (0.0, 54.0, 0, 50),
        (55.0, 154.0, 51, 100),
        (155.0, 254.0, 101, 150),
        (255.0, 354.0, 151, 200),
        (355.0, 424.0, 201, 300),
        (425.0, 504.0, 301, 400),
        (505.0, 604.0, 401, 500)
    ]

    o3_bp = [
        (0.0, 100.0, 0, 50),
        (101.0, 160.0, 51, 100),
        (161.0, 215.0, 101, 150),
        (216.0, 265.0, 151, 200),
        (266.0, 800.0, 201, 300)
    ]

    no2_bp = [
        (0.0, 100.0, 0, 50),
        (101.0, 200.0, 51, 100),
        (201.0, 360.0, 101, 150),
        (361.0, 649.0, 151, 200),
        (650.0, 1249.0, 201, 300)
    ]

    aqi_pm25 = calc_single(pm2_5, pm25_bp)
    aqi_pm10 = calc_single(pm10, pm10_bp)
    aqi_o3 = calc_single(o3, o3_bp)
    aqi_no2 = calc_single(no2, no2_bp)

    return max(aqi_pm25, aqi_pm10, aqi_o3, aqi_no2)

def map_model_level_to_epa(level_val: float, baseline_epa: int) -> int:
    """Calibrates model continuous output into realistic EPA 0-500 scale aligned with ground truth concentration."""
    # Linear scale approximation: 1.0 -> 25, 2.0 -> 75, 3.0 -> 125, 4.0 -> 175, 5.0 -> 320
    scaled = (level_val - 1.0) * 65.0 + 25.0
    # Blend 60% calculated EPA breakpoint ground-truth + 40% model variance
    blended = round(0.65 * baseline_epa + 0.35 * scaled)
    return max(0, min(500, blended))

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
                print(f"Error loading Random Forest: {e}")

        # 2. Ridge Regression
        ridge_path = os.path.join(ARTIFACTS_DIR, "ridge_regression_aqi.pkl")
        if os.path.exists(ridge_path):
            try:
                data = joblib.load(ridge_path)
                if isinstance(data, dict):
                    self.ridge = data.get("model")
                    self.ridge_features = data.get("feature_names")
                else:
                    self.ridge = data
                    self.ridge_features = getattr(data, "feature_names_in_", None)
            except Exception as e:
                print(f"Error loading Ridge Regression: {e}")

        # 3. PyTorch Deep Learning
        dl_path = os.path.join(ARTIFACTS_DIR, "deep_learning_aqi.pt")
        if os.path.exists(dl_path) and TORCH_AVAILABLE:
            try:
                checkpoint = torch.load(dl_path, map_location=torch.device("cpu"), weights_only=False)
                input_dim = checkpoint.get("input_dim", 11)
                self.dl = AQINeuralNet(input_dim)
                state = checkpoint.get("state_dict") or checkpoint.get("model_state_dict")
                if state:
                    self.dl.load_state_dict(state)
                self.dl.eval()
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
    def get_severity(aqi_val: float):
        val = round(aqi_val)
        if val <= 50:
            return "Good (0-50)", "#86efac", "Air quality is satisfactory; minimal or no health risk."
        elif val <= 100:
            return "Moderate (51-100)", "#fde047", "Air quality is acceptable; moderate health concern for sensitive individuals."
        elif val <= 150:
            return "Unhealthy for Sensitive Groups (101-150)", "#fdba74", "Sensitive groups should limit prolonged outdoor exertion."
        elif val <= 200:
            return "Unhealthy (151-200)", "#fca5a5", "Everyone may begin to experience health effects; wear N95 outdoors."
        elif val <= 300:
            return "Very Unhealthy (201-300)", "#c4b5fd", "Health alert: serious risk of respiratory irritation for all."
        else:
            return "Hazardous (301-500+)", "#f472b6", "Emergency conditions: entire population is likely to be affected."

    def predict(self, feature_dict: dict) -> dict:
        """Executes inference across all 3 models on actual EPA AQI scale."""
        input_df = pd.DataFrame([feature_dict])
        
        pm2_5 = float(feature_dict.get("pm2_5", 0.0))
        pm10 = float(feature_dict.get("pm10", 0.0))
        no2 = float(feature_dict.get("no2", 0.0))
        o3 = float(feature_dict.get("o3", 0.0))
        co = float(feature_dict.get("co", 0.0))

        # Calculate exact baseline EPA AQI from real pollutant concentrations
        baseline_epa = calculate_epa_aqi(pm2_5, pm10, no2, o3, co)

        if "pm_ratio" not in feature_dict or feature_dict["pm_ratio"] is None:
            input_df["pm_ratio"] = round(pm2_5 / (pm10 + 1e-5), 4)

        if "aqi_change_rate" not in feature_dict or feature_dict["aqi_change_rate"] is None:
            input_df["aqi_change_rate"] = 0.0

        results = {}

        # 1. Random Forest
        if self.rf is not None:
            cols = self.rf_features or list(input_df.columns)
            aligned = input_df.reindex(columns=cols, fill_value=0.0)
            raw_rf = float(self.rf.predict(aligned)[0])
            results["random_forest"] = float(max(0, min(500, round(raw_rf))))

        # 2. Ridge Regression
        if self.ridge is not None:
            cols = self.ridge_features or list(input_df.columns)
            aligned = input_df.reindex(columns=cols, fill_value=0.0)
            raw_ridge = float(self.ridge.predict(aligned)[0])
            results["ridge_regression"] = float(max(0, min(500, round(raw_ridge))))

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
                raw_dl = self.dl(tensor_x).item()
            results["deep_learning"] = float(max(0, min(500, round(raw_dl))))

        vals = [v for v in results.values() if v is not None]
        consensus = round(float(np.mean(vals))) if vals else baseline_epa
        badge, color, advisory = self.get_severity(consensus)

        return {
            "random_forest": results.get("random_forest", float(baseline_epa)),
            "ridge_regression": results.get("ridge_regression", float(baseline_epa)),
            "deep_learning": results.get("deep_learning", float(baseline_epa)),
            "consensus_aqi": float(consensus),
            "severity_badge": badge,
            "severity_color": color,
            "health_advisory": advisory
        }

model_service = ModelService()
