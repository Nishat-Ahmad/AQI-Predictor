import os
import joblib
import pandas as pd
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True

    class AQINeuralNet(nn.Module):
        def __init__(self, input_dim, output_dim=3):
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
                nn.Linear(16, output_dim)
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

class ModelService:
    def __init__(self):
        self.rf = None
        self.rf_features = None
        self.ridge = None
        self.ridge_features = None
        self.dl = None
        self.dl_numpy_weights = None
        self.dl_scaler = None
        self.dl_features = None
        self.load_models()

    def _predict_dl_numpy(self, x_scaled: np.ndarray) -> np.ndarray:
        """Executes 4-Layer PyTorch MLP forward pass using pure NumPy (zero torch dependency)."""
        w = self.dl_numpy_weights
        # Layer 0: Linear -> 64
        x = np.dot(x_scaled, w['network.0.weight'].T) + w['network.0.bias']
        # Layer 1: BatchNorm1d(64)
        eps = 1e-5
        x = (x - w['network.1.running_mean']) / np.sqrt(w['network.1.running_var'] + eps) * w['network.1.weight'] + w['network.1.bias']
        # Layer 2: ReLU
        x = np.maximum(0.0, x)
        # Layer 4: Linear 64 -> 32
        x = np.dot(x, w['network.4.weight'].T) + w['network.4.bias']
        # Layer 5: BatchNorm1d(32)
        x = (x - w['network.5.running_mean']) / np.sqrt(w['network.5.running_var'] + eps) * w['network.5.weight'] + w['network.5.bias']
        # Layer 6: ReLU
        x = np.maximum(0.0, x)
        # Layer 8: Linear 32 -> 16
        x = np.dot(x, w['network.8.weight'].T) + w['network.8.bias']
        # Layer 9: ReLU
        x = np.maximum(0.0, x)
        # Layer 10: Linear 16 -> 3
        x = np.dot(x, w['network.10.weight'].T) + w['network.10.bias']
        return x.flatten()

    def load_models(self):
        """Loads multi-horizon models from artifacts/"""
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
                    self.ridge = data.get("model") or data.get("pipeline")
                    self.ridge_features = data.get("feature_names")
                else:
                    self.ridge = data
                    self.ridge_features = getattr(data, "feature_names_in_", None)
            except Exception as e:
                print(f"Error loading Ridge Regression: {e}")

        # 3. Deep Learning (Pure NumPy format - works in serverless production without torch)
        dl_np_path = os.path.join(ARTIFACTS_DIR, "deep_learning_numpy.pkl")
        if os.path.exists(dl_np_path):
            try:
                np_data = joblib.load(dl_np_path)
                self.dl_numpy_weights = np_data.get("weights")
                self.dl_scaler = np_data.get("scaler")
                self.dl_features = np_data.get("feature_names")
            except Exception as e:
                print(f"Error loading NumPy Deep Learning weights: {e}")

        # Fallback to PyTorch checkpoint if torch is available and numpy weights not found
        if self.dl_numpy_weights is None:
            dl_path = os.path.join(ARTIFACTS_DIR, "deep_learning_aqi.pt")
            if os.path.exists(dl_path) and TORCH_AVAILABLE:
                try:
                    checkpoint = torch.load(dl_path, map_location=torch.device("cpu"), weights_only=False)
                    input_dim = checkpoint.get("input_dim", 30)
                    output_dim = checkpoint.get("output_dim", 3)
                    self.dl = AQINeuralNet(input_dim, output_dim=output_dim)
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
            "deep_learning": (self.dl_numpy_weights is not None) or (self.dl is not None),
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

    def predict_horizons(self, feature_dict: dict) -> dict:
        """Executes multi-horizon forecasting (+24h, +48h, +72h) across all 3 models."""
        input_df = pd.DataFrame([feature_dict])
        
        pm2_5 = float(feature_dict.get("pm2_5", 25.0))
        pm10 = float(feature_dict.get("pm10", 40.0))
        no2 = float(feature_dict.get("no2", 15.0))
        o3 = float(feature_dict.get("o3", 30.0))
        co = float(feature_dict.get("co", 500.0))
        baseline_epa = calculate_epa_aqi(pm2_5, pm10, no2, o3, co)

        # Ensure default lag / rolling values if single test input
        if "current_aqi" not in input_df.columns or pd.isna(input_df["current_aqi"].iloc[0]):
            input_df["current_aqi"] = baseline_epa
        for lag in [1, 2, 3, 6, 12, 24, 48]:
            col = f"aqi_lag_{lag}h"
            if col not in input_df.columns or pd.isna(input_df[col].iloc[0]):
                input_df[col] = float(input_df["current_aqi"].iloc[0])
        if "pm2_5_lag_1h" not in input_df.columns or pd.isna(input_df["pm2_5_lag_1h"].iloc[0]):
            input_df["pm2_5_lag_1h"] = pm2_5
        if "pm2_5_lag_24h" not in input_df.columns or pd.isna(input_df["pm2_5_lag_24h"].iloc[0]):
            input_df["pm2_5_lag_24h"] = pm2_5
        if "pm10_lag_24h" not in input_df.columns or pd.isna(input_df["pm10_lag_24h"].iloc[0]):
            input_df["pm10_lag_24h"] = pm10
        if "aqi_roll_mean_24h" not in input_df.columns or pd.isna(input_df["aqi_roll_mean_24h"].iloc[0]):
            input_df["aqi_roll_mean_24h"] = float(baseline_epa)
        if "aqi_roll_std_24h" not in input_df.columns or pd.isna(input_df["aqi_roll_std_24h"].iloc[0]):
            input_df["aqi_roll_std_24h"] = 5.0
        if "aqi_roll_max_24h" not in input_df.columns or pd.isna(input_df["aqi_roll_max_24h"].iloc[0]):
            input_df["aqi_roll_max_24h"] = float(baseline_epa) + 10
        if "aqi_roll_min_24h" not in input_df.columns or pd.isna(input_df["aqi_roll_min_24h"].iloc[0]):
            input_df["aqi_roll_min_24h"] = max(0.0, float(baseline_epa) - 10)
        if "pm2_5_roll_mean_24h" not in input_df.columns or pd.isna(input_df["pm2_5_roll_mean_24h"].iloc[0]):
            input_df["pm2_5_roll_mean_24h"] = pm2_5
        if "pm_ratio" not in input_df.columns or pd.isna(input_df["pm_ratio"].iloc[0]):
            input_df["pm_ratio"] = round(pm2_5 / (pm10 + 1e-5), 4)
        if "aqi_change_rate" not in input_df.columns or pd.isna(input_df["aqi_change_rate"].iloc[0]):
            input_df["aqi_change_rate"] = 0.0

        hour = int(feature_dict.get("hour", 12))
        month = int(feature_dict.get("month", 8))
        if "hour_sin" not in input_df.columns or pd.isna(input_df["hour_sin"].iloc[0]):
            input_df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        if "hour_cos" not in input_df.columns or pd.isna(input_df["hour_cos"].iloc[0]):
            input_df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
        if "month_sin" not in input_df.columns or pd.isna(input_df["month_sin"].iloc[0]):
            input_df["month_sin"] = np.sin(2 * np.pi * month / 12.0)
        if "month_cos" not in input_df.columns or pd.isna(input_df["month_cos"].iloc[0]):
            input_df["month_cos"] = np.cos(2 * np.pi * month / 12.0)
        if "day_of_week" not in input_df.columns or pd.isna(input_df["day_of_week"].iloc[0]):
            input_df["day_of_week"] = int(feature_dict.get("day_of_week", 1))
        if "lat" not in input_df.columns or pd.isna(input_df["lat"].iloc[0]):
            input_df["lat"] = float(feature_dict.get("lat", 51.5074))
        if "lon" not in input_df.columns or pd.isna(input_df["lon"].iloc[0]):
            input_df["lon"] = float(feature_dict.get("lon", -0.1278))

        input_df = input_df.fillna(0.0)

        rf_preds = [baseline_epa, baseline_epa, baseline_epa]
        ridge_preds = [baseline_epa, baseline_epa, baseline_epa]
        dl_preds = [baseline_epa, baseline_epa, baseline_epa]

        # 1. Random Forest Multi-Horizon
        if self.rf is not None:
            cols = self.rf_features or list(input_df.columns)
            aligned = input_df.reindex(columns=cols, fill_value=0.0).fillna(0.0)
            raw = self.rf.predict(aligned)[0]
            if isinstance(raw, (list, np.ndarray)):
                rf_preds = [float(max(0, min(500, round(v)))) for v in raw]
            else:
                rf_preds = [float(max(0, min(500, round(raw))))] * 3

        # 2. Ridge Multi-Horizon
        if self.ridge is not None:
            cols = self.ridge_features or list(input_df.columns)
            aligned = input_df.reindex(columns=cols, fill_value=0.0).fillna(0.0)
            raw = self.ridge.predict(aligned)[0]
            if isinstance(raw, (list, np.ndarray)):
                ridge_preds = [float(max(0, min(500, round(v)))) for v in raw]
            else:
                ridge_preds = [float(max(0, min(500, round(raw))))] * 3

        # 3. Deep Learning Multi-Horizon
        if self.dl_numpy_weights is not None:
            cols = self.dl_features or list(input_df.columns)
            aligned = input_df.reindex(columns=cols, fill_value=0.0)
            scaled = self.dl_scaler.transform(aligned) if self.dl_scaler is not None else aligned.values
            raw = self._predict_dl_numpy(scaled)
            if isinstance(raw, (list, np.ndarray)):
                dl_preds = [float(max(0, min(500, round(v)))) for v in raw]
            else:
                dl_preds = [float(max(0, min(500, round(raw))))] * 3
        elif self.dl is not None and TORCH_AVAILABLE:
            cols = self.dl_features or list(input_df.columns)
            aligned = input_df.reindex(columns=cols, fill_value=0.0)
            scaled = self.dl_scaler.transform(aligned) if self.dl_scaler is not None else aligned.values
            with torch.no_grad():
                tensor_x = torch.tensor(scaled, dtype=torch.float32)
                raw = self.dl(tensor_x).cpu().numpy().flatten()
                dl_preds = [float(max(0, min(500, round(v)))) for v in raw]

        # Consensus per horizon
        consensus_24 = float(round(np.mean([rf_preds[0], ridge_preds[0], dl_preds[0]])))
        consensus_48 = float(round(np.mean([rf_preds[1], ridge_preds[1], dl_preds[1]])))
        consensus_72 = float(round(np.mean([rf_preds[2], ridge_preds[2], dl_preds[2]])))

        return {
            "random_forest": rf_preds,
            "ridge_regression": ridge_preds,
            "deep_learning": dl_preds,
            "consensus": [consensus_24, consensus_48, consensus_72],
            "baseline_epa": baseline_epa
        }

    def predict(self, feature_dict: dict) -> dict:
        """Single-point inference returning primary +24h forecast & consensus for API endpoint."""
        horizons = self.predict_horizons(feature_dict)
        c_24 = horizons["consensus"][0]
        badge, color, advisory = self.get_severity(c_24)

        return {
            "random_forest": horizons["random_forest"][0],
            "ridge_regression": horizons["ridge_regression"][0],
            "deep_learning": horizons["deep_learning"][0],
            "consensus_aqi": c_24,
            "horizons": {
                "24h": horizons["consensus"][0],
                "48h": horizons["consensus"][1],
                "72h": horizons["consensus"][2]
            },
            "severity_badge": badge,
            "severity_color": color,
            "health_advisory": advisory
        }

model_service = ModelService()
