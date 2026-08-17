import os
import sys
import requests
import pandas as pd
import wandb
from datetime import datetime

# 1. Load API keys if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

openweather_key = os.getenv("OPENWEATHER_API_KEY")

# 2. Define the city you want to predict AQI for (e.g., Lahore)
LAT = "31.5204"
LON = "74.3587"

def fetch_weather_and_aqi():
    """Fetches raw weather and AQI data including recent history from the OpenWeather API."""
    print("Fetching data from OpenWeather...")
    now = datetime.now()
    end_unix = int(now.timestamp())
    start_unix = end_unix - (3600 * 3) # past 3 hours to capture previous hour reading
    
    # Try fetching recent history first to calculate hourly change rate
    history_url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={LAT}&lon={LON}&start={start_unix}&end={end_unix}&appid={openweather_key}"
    response = requests.get(history_url)
    
    if response.status_code == 200:
        data = response.json()
        items = data.get('list', [])
        if items:
            items.sort(key=lambda x: x['dt'])
            current_data = items[-1]
            aqi = current_data['main']['aqi']
            components = current_data['components']
            timestamp = current_data['dt']
            
            aqi_change_rate = 0.0
            if len(items) >= 2:
                prev_aqi = items[-2]['main']['aqi']
                aqi_change_rate = float(aqi - prev_aqi)
                
            return aqi, components, timestamp, aqi_change_rate
            
    # Fallback to current endpoint if history is unavailable
    current_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={openweather_key}"
    current_resp = requests.get(current_url)
    if current_resp.status_code == 200:
        data = current_resp.json()
        current_data = data['list'][0]
        return current_data['main']['aqi'], current_data['components'], current_data['dt'], 0.0
    else:
        print(f"Error fetching data: {current_resp.status_code}")
        return None, None, None, None

def engineer_features():
    """Computes features from raw data and returns a Pandas DataFrame."""
    aqi, components, timestamp, aqi_change_rate = fetch_weather_and_aqi()
    
    if aqi is None:
        return None
        
    print("Engineering features...")
    dt_object = datetime.fromtimestamp(timestamp)
    
    feature_dict = {
        "timestamp": dt_object.strftime("%Y-%m-%d %H:%M:%S"),
        "hour": dt_object.hour,
        "day": dt_object.day,
        "month": dt_object.month,
        "day_of_week": dt_object.weekday(),
        "aqi_target": aqi,
        "co": components['co'],
        "no2": components['no2'],
        "o3": components['o3'],
        "pm2_5": components['pm2_5'],
        "pm10": components['pm10'],
        "pm_ratio": round(components['pm2_5'] / (components['pm10'] + 1e-5), 4),
        "aqi_change_rate": aqi_change_rate
    }
    
    # Convert to a Pandas DataFrame
    df = pd.DataFrame([feature_dict])
    return df

def save_to_feature_store(df):
    """Logs the DataFrame to Weights & Biases as a Table Artifact."""
    print("Initializing Weights & Biases...")
    
    # 6. Initialize a W&B run
    with wandb.init(project="pearls-aqi-predictor", job_type="feature-engineering") as run:
        
        # 7. Convert the Pandas DataFrame into a W&B Table
        table = wandb.Table(dataframe=df)
        
        # 8. Create a W&B Artifact to store the table
        artifact = wandb.Artifact("aqi_features", type="dataset")
        artifact.add(table, "current_features")
        
        # Log the artifact to save it to the cloud
        run.log_artifact(artifact)
        print("Successfully logged features to W&B Feature Store!")

if __name__ == "__main__":
    if not openweather_key:
        print("Error: OPENWEATHER_API_KEY environment variable is not set.")
        sys.exit(1)
        
    features_df = engineer_features()
    if features_df is not None:
        print(features_df)
        save_to_feature_store(features_df)
    else:
        print("Error: Failed to engineer features.")
        sys.exit(1)