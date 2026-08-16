import os
import requests
import pandas as pd
import wandb
from dotenv import load_dotenv
from datetime import datetime

# 1. Load your API keys from the .env file
load_dotenv()
openweather_key = os.getenv("OPENWEATHER_API_KEY")

# 2. Define the city you want to predict AQI for (e.g., Lahore)
LAT = "31.5204"
LON = "74.3587"

def fetch_weather_and_aqi():
    """Fetches raw weather and AQI data from the OpenWeather API."""
    print("Fetching data from OpenWeather...")
    
    # OpenWeather Air Pollution API endpoint
    aqi_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={openweather_key}"
    response = requests.get(aqi_url)
    
    if response.status_code == 200:
        data = response.json()
        
        # 3. Extract the relevant raw data points
        current_data = data['list'][0]
        aqi = current_data['main']['aqi']
        components = current_data['components'] # Contains CO, NO2, O3, SO2, PM2.5, PM10
        timestamp = current_data['dt']
        
        return aqi, components, timestamp
    else:
        print(f"Error fetching data: {response.status_code}")
        return None, None, None

def engineer_features():
    """Computes features from raw data and returns a Pandas DataFrame."""
    aqi, components, timestamp = fetch_weather_and_aqi()
    
    if aqi is None:
        return None
        
    print("Engineering features...")
    
    # 4. Include time-based features (hour, day, month)
    dt_object = datetime.fromtimestamp(timestamp)
    
    # 5. Create a dictionary of all our features
    feature_dict = {
        "timestamp": dt_object.strftime("%Y-%m-%d %H:%M:%S"),
        "hour": dt_object.hour,
        "day": dt_object.day,
        "month": dt_object.month,
        "aqi_target": aqi,
        "co": components['co'],
        "no2": components['no2'],
        "o3": components['o3'],
        "pm2_5": components['pm2_5'],
        "pm10": components['pm10']
    }
    
    # Convert to a Pandas DataFrame
    df = pd.DataFrame([feature_dict])
    return df

def save_to_feature_store(df):
    """Logs the DataFrame to Weights & Biases as a Table Artifact."""
    print("Initializing Weights & Biases...")
    
    # 6. Initialize a W&B run
    with wandb.init(entity="pearl-aqi", project="pearls-aqi-predictor", job_type="feature-engineering") as run:
        
        # 7. Convert the Pandas DataFrame into a W&B Table
        table = wandb.Table(dataframe=df)
        
        # 8. Create a W&B Artifact to store the table
        artifact = wandb.Artifact("aqi_features", type="dataset")
        artifact.add(table, "current_features")
        
        # Log the artifact to save it to the cloud
        run.log_artifact(artifact)
        print("Successfully logged features to W&B Feature Store!")

if __name__ == "__main__":
    features_df = engineer_features()
    if features_df is not None:
        print(features_df)
        save_to_feature_store(features_df)