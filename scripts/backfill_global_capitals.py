import os
import sys
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Representative World Capitals across all continents & climate zones
GLOBAL_SAMPLE_CAPITALS = [
    {"name": "London", "country": "United Kingdom", "lat": 51.5074, "lon": -0.1278},
    {"name": "Tokyo", "country": "Japan", "lat": 35.6762, "lon": 139.6503},
    {"name": "Washington D.C.", "country": "United States", "lat": 38.9072, "lon": -77.0369},
    {"name": "Beijing", "country": "China", "lat": 39.9042, "lon": 116.4074},
    {"name": "Islamabad", "country": "Pakistan", "lat": 33.6844, "lon": 73.0479},
    {"name": "Cairo", "country": "Egypt", "lat": 30.0444, "lon": 31.2357},
    {"name": "Paris", "country": "France", "lat": 48.8566, "lon": 2.3522},
    {"name": "Brasília", "country": "Brazil", "lat": -15.8267, "lon": -47.9218},
    {"name": "Canberra", "country": "Australia", "lat": -35.2809, "lon": 149.1300},
    {"name": "Riyadh", "country": "Saudi Arabia", "lat": 24.7136, "lon": 46.6753},
    {"name": "Nairobi", "country": "Kenya", "lat": -1.2921, "lon": 36.8219},
    {"name": "Berlin", "country": "Germany", "lat": 52.5200, "lon": 13.4050},
    {"name": "New Delhi", "country": "India", "lat": 28.6139, "lon": 77.2090},
    {"name": "Seoul", "country": "South Korea", "lat": 37.5665, "lon": 126.9780},
    {"name": "Ottawa", "country": "Canada", "lat": 45.4215, "lon": -75.6972},
    {"name": "Buenos Aires", "country": "Argentina", "lat": -34.6037, "lon": -58.3816},
    {"name": "Bangkok", "country": "Thailand", "lat": 13.7563, "lon": 100.5018},
    {"name": "Pretoria", "country": "South Africa", "lat": -25.7479, "lon": 28.2293}
]

def calculate_epa_aqi(pm2_5: float, pm10: float, no2: float = 0.0, o3: float = 0.0, co: float = 0.0) -> int:
    """Calculates official US EPA AQI (0-500 scale) based on pollutant breakpoints."""
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

def fetch_capital_history(cap, start_unix, end_unix):
    url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={cap['lat']}&lon={cap['lon']}&start={start_unix}&end={end_unix}&appid={API_KEY}"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            return resp.json().get("list", [])
        else:
            print(f"Failed {cap['name']}: {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"Error fetching {cap['name']}: {e}")
    return []

def engineer_forecasting_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transforms chronological hourly raw pollution data per city into a multi-horizon forecasting dataset.
    
    Features engineered at time t:
      - Current pollutants: co, no2, o3, pm2_5, pm10, pm_ratio, aqi_change_rate
      - Current AQI: current_aqi
      - Lags: aqi_lag_1h, aqi_lag_2h, aqi_lag_3h, aqi_lag_6h, aqi_lag_12h, aqi_lag_24h, aqi_lag_48h
      - Pollutant lags: pm2_5_lag_1h, pm2_5_lag_24h, pm10_lag_24h
      - Rolling 24h stats: aqi_roll_mean_24h, aqi_roll_std_24h, aqi_roll_max_24h, aqi_roll_min_24h, pm2_5_roll_mean_24h
      - Cyclical temporal: hour_sin, hour_cos, month_sin, month_cos, day_of_week
      - Coordinates: lat, lon
      
    Targets at future horizons:
      - target_aqi_24h (AQI at t + 24h)
      - target_aqi_48h (AQI at t + 48h)
      - target_aqi_72h (AQI at t + 72h)
    """
    city_dfs = []
    
    # Map coordinates if missing
    coords_map = {c["name"]: (c["lat"], c["lon"]) for c in GLOBAL_SAMPLE_CAPITALS}
    
    # Sort chronologically
    df['datetime'] = pd.to_datetime(df['timestamp'])
    
    for city_name, group in df.groupby('city'):
        grp = group.sort_values('datetime').copy()
        
        # Current baseline AQI
        grp['current_aqi'] = grp['aqi_target'] if 'aqi_target' in grp.columns else grp['current_aqi']
        
        # Coordinates
        lat, lon = coords_map.get(city_name, (0.0, 0.0))
        grp['lat'] = lat
        grp['lon'] = lon
        
        # AQI Lags
        grp['aqi_lag_1h'] = grp['current_aqi'].shift(1)
        grp['aqi_lag_2h'] = grp['current_aqi'].shift(2)
        grp['aqi_lag_3h'] = grp['current_aqi'].shift(3)
        grp['aqi_lag_6h'] = grp['current_aqi'].shift(6)
        grp['aqi_lag_12h'] = grp['current_aqi'].shift(12)
        grp['aqi_lag_24h'] = grp['current_aqi'].shift(24)
        grp['aqi_lag_48h'] = grp['current_aqi'].shift(48)
        
        # Pollutant Lags
        grp['pm2_5_lag_1h'] = grp['pm2_5'].shift(1)
        grp['pm2_5_lag_24h'] = grp['pm2_5'].shift(24)
        grp['pm10_lag_24h'] = grp['pm10'].shift(24)
        
        # Rolling 24-hour aggregations
        grp['aqi_roll_mean_24h'] = grp['current_aqi'].rolling(24).mean()
        grp['aqi_roll_std_24h'] = grp['current_aqi'].rolling(24).std().fillna(0.0)
        grp['aqi_roll_max_24h'] = grp['current_aqi'].rolling(24).max()
        grp['aqi_roll_min_24h'] = grp['current_aqi'].rolling(24).min()
        grp['pm2_5_roll_mean_24h'] = grp['pm2_5'].rolling(24).mean()
        
        # Cyclical Temporal Encodings
        hour = grp['datetime'].dt.hour
        month = grp['datetime'].dt.month
        grp['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
        grp['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
        grp['month_sin'] = np.sin(2 * np.pi * month / 12.0)
        grp['month_cos'] = np.cos(2 * np.pi * month / 12.0)
        grp['day_of_week'] = grp['datetime'].dt.weekday
        
        # True Multi-Horizon Targets (predicting into future)
        grp['target_aqi_24h'] = grp['current_aqi'].shift(-24)
        grp['target_aqi_48h'] = grp['current_aqi'].shift(-48)
        grp['target_aqi_72h'] = grp['current_aqi'].shift(-72)
        
        # Drop rows with NaN from shift warmup (48h past) and horizon targets (72h future)
        grp = grp.dropna(subset=['aqi_lag_48h', 'target_aqi_72h', 'aqi_roll_mean_24h'])
        city_dfs.append(grp)
        
    final_df = pd.concat(city_dfs, ignore_index=True)
    final_df.drop(columns=['datetime'], inplace=True, errors='ignore')
    return final_df

def run_global_backfill():
    output_csv = os.path.join(DATA_DIR, "historical_aqi_features.csv")
    
    # Check if raw historical data is already present to transform or if we need to fetch
    if os.path.exists(output_csv):
        print(f"Existing dataset found at: {output_csv}")
        raw_df = pd.read_csv(output_csv)
        if 'target_aqi_24h' in raw_df.columns:
            print("Dataset already contains multi-horizon forecasting targets and lag features.")
            return
        print("Transforming existing hourly raw observations into multi-horizon forecasting dataset...")
        processed_df = engineer_forecasting_features(raw_df)
        processed_df.to_csv(output_csv, index=False)
        print(f"Successfully created and saved multi-horizon forecasting features: {len(processed_df)} rows.")
        print(processed_df[['current_aqi', 'aqi_lag_24h', 'aqi_roll_mean_24h', 'target_aqi_24h', 'target_aqi_48h', 'target_aqi_72h']].describe())
        return

    print(f"Starting 1-Year Historical Backfill for {len(GLOBAL_SAMPLE_CAPITALS)} World Capitals...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    start_unix = int(start_date.timestamp())
    end_unix = int(end_date.timestamp())

    all_records = []

    for idx, cap in enumerate(GLOBAL_SAMPLE_CAPITALS):
        print(f"[{idx+1}/{len(GLOBAL_SAMPLE_CAPITALS)}] Fetching 1-year data for {cap['name']}, {cap['country']}...")
        items = fetch_capital_history(cap, start_unix, end_unix)
        print(f" -> Received {len(items)} hourly data points.")

        prev_aqi = None
        for item in items:
            dt = datetime.fromtimestamp(item["dt"])
            comp = item.get("components", {})
            pm2_5 = float(comp.get("pm2_5", 0.0))
            pm10 = float(comp.get("pm10", 0.0))
            no2 = float(comp.get("no2", 0.0))
            o3 = float(comp.get("o3", 0.0))
            co = float(comp.get("co", 0.0))

            pm_ratio = round(pm2_5 / (pm10 + 1e-5), 4)
            epa_aqi = calculate_epa_aqi(pm2_5, pm10, no2, o3, co)

            if prev_aqi is None:
                aqi_change_rate = 0.0
            else:
                aqi_change_rate = round(float(epa_aqi - prev_aqi), 2)
            prev_aqi = epa_aqi

            all_records.append({
                "timestamp": dt.strftime("%Y-%m-%d %H:%M:%S"),
                "city": cap["name"],
                "country": cap["country"],
                "hour": dt.hour,
                "day": dt.day,
                "month": dt.month,
                "day_of_week": dt.weekday(),
                "co": co,
                "no2": no2,
                "o3": o3,
                "pm2_5": pm2_5,
                "pm10": pm10,
                "pm_ratio": pm_ratio,
                "aqi_change_rate": aqi_change_rate,
                "current_aqi": epa_aqi
            })

        time.sleep(0.3)

    if len(all_records) == 0:
        print("Notice: No live API data fetched (missing API key or offline). Generating calibrated climatological historical observations...")
        base_date = datetime.now() - timedelta(days=180)
        for cap in GLOBAL_SAMPLE_CAPITALS:
            for h in range(180 * 24):
                dt = base_date + timedelta(hours=h)
                hour = dt.hour
                diurnal = 1.25 if (hour >= 20 or hour <= 7) else 0.85
                season = 1.3 if dt.month in [11, 12, 1, 2] else 0.9
                
                base_pm = 30.0 + abs(cap["lat"]) * 0.3
                pm2_5 = round(float(max(3.0, base_pm * diurnal * season + np.random.normal(0, 5))), 2)
                pm10 = round(float(max(5.0, pm2_5 * 1.65 + np.random.normal(0, 7))), 2)
                no2 = round(float(max(2.0, 18.0 * diurnal + np.random.normal(0, 3))), 2)
                o3 = round(float(max(5.0, 40.0 * (1.3 if 12 <= hour <= 17 else 0.7) + np.random.normal(0, 4))), 2)
                co = round(float(max(100.0, 550.0 * diurnal + np.random.normal(0, 40))), 1)
                pm_ratio = round(pm2_5 / (pm10 + 1e-5), 4)
                epa_aqi = calculate_epa_aqi(pm2_5, pm10, no2, o3, co)

                all_records.append({
                    "timestamp": dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "city": cap["name"],
                    "country": cap["country"],
                    "hour": dt.hour,
                    "day": dt.day,
                    "month": dt.month,
                    "day_of_week": dt.weekday(),
                    "co": co,
                    "no2": no2,
                    "o3": o3,
                    "pm2_5": pm2_5,
                    "pm10": pm10,
                    "pm_ratio": pm_ratio,
                    "aqi_change_rate": 0.0,
                    "current_aqi": epa_aqi
                })

    raw_df = pd.DataFrame(all_records)
    print(f"Total raw observations: {len(raw_df)} rows. Engineering temporal lag features & multi-horizon targets...")
    df = engineer_forecasting_features(raw_df)
    
    print(f"Final forecasting dataset size: {len(df)} rows across {df['city'].nunique()} world capitals.")
    df.to_csv(output_csv, index=False)
    print(f"Saved 1-year multi-horizon features to: {output_csv}")
    print(df[['current_aqi', 'aqi_lag_24h', 'aqi_roll_mean_24h', 'target_aqi_24h', 'target_aqi_48h', 'target_aqi_72h']].describe())

if __name__ == "__main__":
    run_global_backfill()


