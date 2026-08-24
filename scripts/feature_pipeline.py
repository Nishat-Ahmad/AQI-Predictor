import os
import sys
import requests
import pandas as pd
import numpy as np
import wandb
from datetime import datetime, timedelta

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

openweather_key = os.getenv("OPENWEATHER_API_KEY")
LAT = float(os.getenv("CITY_LAT", "31.5204"))
LON = float(os.getenv("CITY_LON", "74.3587"))

def calculate_epa_aqi(pm2_5: float, pm10: float, no2: float = 0.0, o3: float = 0.0, co: float = 0.0) -> int:
    def calc_single(c, breakpoints):
        for c_low, c_high, i_low, i_high in breakpoints:
            if c_low <= c <= c_high:
                return round(((i_high - i_low) / (c_high - c_low)) * (c - c_low) + i_low)
        if c > breakpoints[-1][1]:
            return min(500, round(breakpoints[-1][3] + (c - breakpoints[-1][1])))
        return 0

    pm25_bp = [(0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150), (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300), (250.5, 350.4, 301, 400), (350.5, 500.4, 401, 500)]
    pm10_bp = [(0.0, 54.0, 0, 50), (55.0, 154.0, 51, 100), (155.0, 254.0, 101, 150), (255.0, 354.0, 151, 200), (355.0, 424.0, 201, 300), (425.0, 504.0, 301, 400), (505.0, 604.0, 401, 500)]
    o3_bp = [(0.0, 100.0, 0, 50), (101.0, 160.0, 51, 100), (161.0, 215.0, 101, 150), (216.0, 265.0, 151, 200), (266.0, 800.0, 201, 300)]
    no2_bp = [(0.0, 100.0, 0, 50), (101.0, 200.0, 51, 100), (201.0, 360.0, 101, 150), (361.0, 649.0, 151, 200), (650.0, 1249.0, 201, 300)]

    return max(calc_single(pm2_5, pm25_bp), calc_single(pm10, pm10_bp), calc_single(o3, o3_bp), calc_single(no2, no2_bp))

def fetch_history(lat=LAT, lon=LON, hours=48):
    if not openweather_key:
        print("Warning: OPENWEATHER_API_KEY is not set.")
        return None
        
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(hours=hours)
    start_unix = int(start_dt.timestamp())
    end_unix = int(end_dt.timestamp())
    
    url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={lat}&lon={lon}&start={start_unix}&end={end_unix}&appid={openweather_key}"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            return resp.json().get("list", [])
    except Exception as e:
        print(f"Error fetching history: {e}")
    return None

def engineer_features():
    """Extracts recent observations and builds the temporal feature row."""
    items = fetch_history(LAT, LON, hours=48)
    
    if not items or len(items) < 2:
        print("Warning: Could not fetch real-time history, constructing nominal feature row.")
        base_pm25, base_pm10 = 25.0, 40.0
        now = datetime.now()
        base_aqi = calculate_epa_aqi(base_pm25, base_pm10, 15.0, 30.0, 500.0)
        return pd.DataFrame([{
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            "current_aqi": float(base_aqi),
            "co": 500.0, "no2": 15.0, "o3": 30.0, "pm2_5": base_pm25, "pm10": base_pm10,
            "pm_ratio": round(base_pm25 / (base_pm10 + 1e-5), 4),
            "aqi_change_rate": 0.0,
            "aqi_lag_1h": float(base_aqi), "aqi_lag_2h": float(base_aqi), "aqi_lag_3h": float(base_aqi),
            "aqi_lag_6h": float(base_aqi), "aqi_lag_12h": float(base_aqi), "aqi_lag_24h": float(base_aqi), "aqi_lag_48h": float(base_aqi),
            "pm2_5_lag_1h": base_pm25, "pm2_5_lag_24h": base_pm25, "pm10_lag_24h": base_pm10,
            "aqi_roll_mean_24h": float(base_aqi), "aqi_roll_std_24h": 4.0,
            "aqi_roll_max_24h": float(base_aqi + 10), "aqi_roll_min_24h": max(0.0, float(base_aqi - 10)),
            "pm2_5_roll_mean_24h": base_pm25,
            "hour_sin": np.sin(2 * np.pi * now.hour / 24.0),
            "hour_cos": np.cos(2 * np.pi * now.hour / 24.0),
            "month_sin": np.sin(2 * np.pi * now.month / 12.0),
            "month_cos": np.cos(2 * np.pi * now.month / 12.0),
            "day_of_week": now.weekday(),
            "lat": LAT, "lon": LON
        }])

    items.sort(key=lambda x: x["dt"])
    aqi_series = []
    pm25_series = []
    pm10_series = []

    for it in items:
        c = it.get("components", {})
        p25 = float(c.get("pm2_5", 20.0))
        p10 = float(c.get("pm10", 35.0))
        no2 = float(c.get("no2", 15.0))
        o3 = float(c.get("o3", 30.0))
        co = float(c.get("co", 500.0))
        epa = calculate_epa_aqi(p25, p10, no2, o3, co)
        aqi_series.append(epa)
        pm25_series.append(p25)
        pm10_series.append(p10)

    curr_item = items[-1]
    curr_c = curr_item.get("components", {})
    curr_p25 = float(curr_c.get("pm2_5", 25.0))
    curr_p10 = float(curr_c.get("pm10", 40.0))
    curr_no2 = float(curr_c.get("no2", 15.0))
    curr_o3 = float(curr_c.get("o3", 30.0))
    curr_co = float(curr_c.get("co", 500.0))
    curr_aqi = aqi_series[-1]
    
    prev_aqi = aqi_series[-2] if len(aqi_series) >= 2 else curr_aqi
    change_rate = float(curr_aqi - prev_aqi)

    def get_lag(arr, step):
        idx = len(arr) - 1 - step
        return float(arr[idx]) if idx >= 0 else float(arr[0])

    past_24_aqi = aqi_series[-24:] if len(aqi_series) >= 24 else aqi_series
    past_24_pm25 = pm25_series[-24:] if len(pm25_series) >= 24 else pm25_series

    dt_obj = datetime.fromtimestamp(curr_item["dt"])
    feature_dict = {
        "timestamp": dt_obj.strftime("%Y-%m-%d %H:%M:%S"),
        "current_aqi": float(curr_aqi),
        "co": curr_co, "no2": curr_no2, "o3": curr_o3, "pm2_5": curr_p25, "pm10": curr_p10,
        "pm_ratio": round(curr_p25 / (curr_p10 + 1e-5), 4),
        "aqi_change_rate": change_rate,
        "aqi_lag_1h": get_lag(aqi_series, 1),
        "aqi_lag_2h": get_lag(aqi_series, 2),
        "aqi_lag_3h": get_lag(aqi_series, 3),
        "aqi_lag_6h": get_lag(aqi_series, 6),
        "aqi_lag_12h": get_lag(aqi_series, 12),
        "aqi_lag_24h": get_lag(aqi_series, 24),
        "aqi_lag_48h": get_lag(aqi_series, 48),
        "pm2_5_lag_1h": get_lag(pm25_series, 1),
        "pm2_5_lag_24h": get_lag(pm25_series, 24),
        "pm10_lag_24h": get_lag(pm10_series, 24),
        "aqi_roll_mean_24h": float(np.mean(past_24_aqi)),
        "aqi_roll_std_24h": float(np.std(past_24_aqi)) if len(past_24_aqi) > 1 else 4.0,
        "aqi_roll_max_24h": float(np.max(past_24_aqi)),
        "aqi_roll_min_24h": float(np.min(past_24_aqi)),
        "pm2_5_roll_mean_24h": float(np.mean(past_24_pm25)),
        "hour_sin": np.sin(2 * np.pi * dt_obj.hour / 24.0),
        "hour_cos": np.cos(2 * np.pi * dt_obj.hour / 24.0),
        "month_sin": np.sin(2 * np.pi * dt_obj.month / 12.0),
        "month_cos": np.cos(2 * np.pi * dt_obj.month / 12.0),
        "day_of_week": dt_obj.weekday(),
        "lat": LAT, "lon": LON
    }
    return pd.DataFrame([feature_dict])

def save_to_feature_store(df: pd.DataFrame):
    """Logs hourly feature snapshot to Weights & Biases Feature Store."""
    os.environ.setdefault("WANDB_SILENT", "true")
    try:
        run = wandb.init(project="pearls-aqi-predictor", job_type="hourly-feature-pipeline")
    except Exception:
        run = wandb.init(project="pearls-aqi-predictor", job_type="hourly-feature-pipeline", mode="offline")

    table = wandb.Table(dataframe=df)
    artifact = wandb.Artifact("aqi_hourly_features", type="feature-snapshot")
    artifact.add(table, "latest_features")
    run.log_artifact(artifact)
    run.finish()
    print("Successfully logged hourly feature snapshot to W&B Feature Store!")

if __name__ == "__main__":
    features_df = engineer_features()
    if features_df is not None:
        print("Engineered hourly temporal feature vector:")
        print(features_df.to_dict(orient="records")[0])
        save_to_feature_store(features_df)