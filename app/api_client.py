import os
import requests
from typing import Dict, Any, Optional

DEFAULT_API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

class APIClient:
    def __init__(self, base_url: str = DEFAULT_API_URL):
        self.base_url = base_url

    def is_backend_online(self) -> bool:
        """Checks if the FastAPI backend service is reachable."""
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=2)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        return False

    def predict(self, feature_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Queries the FastAPI /predict endpoint."""
        try:
            resp = requests.post(f"{self.base_url}/predict", json=feature_dict, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                preds = data.get("predictions", {})
                return {
                    "Random Forest": preds.get("random_forest"),
                    "Ridge Regression": preds.get("ridge_regression"),
                    "Deep Learning (PyTorch MLP)": preds.get("deep_learning"),
                    "consensus_aqi": preds.get("consensus_aqi"),
                    "severity_badge": preds.get("severity_badge"),
                    "severity_color": preds.get("severity_color")
                }
        except Exception as e:
            print(f"APIClient predict fallback: {e}")
        return None

    def get_3day_forecast(self) -> Optional[Dict[str, Any]]:
        """Queries the FastAPI /forecast/3day endpoint."""
        try:
            resp = requests.get(f"{self.base_url}/forecast/3day", timeout=12)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            print(f"APIClient forecast fallback: {e}")
        return None

api_client = APIClient()
