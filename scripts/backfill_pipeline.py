import os
import requests
import pandas as pd
import wandb
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")

# Lahore Coordinates
LAT = 31.5204
LON = 74.3587

def fetch_historical_data():
    print("Fetching historical data from OpenWeather (Past 2 years)...")
    
    # Calculate Unix timestamps for the past 2 years up to today
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 2)
    
    start_unix = int(start_date.timestamp())
    end_unix = int(end_date.timestamp())

    url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={LAT}&lon={LON}&start={start_unix}&end={end_unix}&appid={API_KEY}"
    
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data: {response.text}")
        
    return response.json()

def process_data(data):
    print("Engineering historical features...")
    records = []
    
    # Loop through the massive list of historical hourly records
    for item in data.get('list', []):
        dt = datetime.fromtimestamp(item['dt'])
        records.append({
            'timestamp': dt.strftime('%Y-%m-%d %H:%M:%S'),
            'hour': dt.hour,
            'day': dt.day,
            'month': dt.month,
            'aqi_target': item['main']['aqi'],
            'co': item['components']['co'],
            'no2': item['components']['no2'],
            'o3': item['components']['o3'],
            'pm2_5': item['components']['pm2_5'],
            'pm10': item['components']['pm10']
        })
        
    df = pd.DataFrame(records)
    return df

def log_to_wandb(df):
    print("Initializing Weights & Biases...")
    # job_type helps organize runs in the W&B dashboard
    run = wandb.init(entity="pearl-aqi", project="pearls-aqi-predictor", job_type="backfill")
    
    # Save the DataFrame locally as a CSV under the project's data directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    csv_filename = os.path.join(data_dir, "historical_aqi_features.csv")
    df.to_csv(csv_filename, index=False)
    
    # Create a dataset Artifact and upload the CSV to the cloud
    artifact = wandb.Artifact(
        name="historical_aqi_features", 
        type="dataset", 
        description="2 years of historical hourly AQI data for Lahore"
    )
    artifact.add_file(csv_filename)
    
    run.log_artifact(artifact)
    run.finish()
    
    print("Successfully logged historical features to W&B Feature Store!")

if __name__ == "__main__":
    if not API_KEY:
        print("Error: OPENWEATHER_API_KEY not found in .env file.")
    else:
        # 1. Fetch
        raw_data = fetch_historical_data()
        
        # 2. Process
        historical_df = process_data(raw_data)
        
        print(f"Successfully extracted {len(historical_df)} rows of historical data.")
        print(historical_df.head())
        
        # 3. Log
        log_to_wandb(historical_df)