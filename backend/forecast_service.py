import os
import requests
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from backend.model_service import model_service
from backend.schemas import HourlyForecastPoint, DaySummary, ForecastResponse

# Load environment
API_KEY = os.getenv("OPENWEATHER_API_KEY")
DEFAULT_LAT = float(os.getenv("CITY_LAT", "31.5204"))
DEFAULT_LON = float(os.getenv("CITY_LON", "74.3587"))
DEFAULT_CITY = "Lahore, Pakistan"

def fetch_openweather_forecast(lat: float, lon: float):
    """Fetches real-time 5-day hourly air pollution forecast from OpenWeather for any global coordinates."""
    if not API_KEY:
        return None
    url = f"http://api.openweathermap.org/data/2.5/air_pollution/forecast?lat={lat}&lon={lon}&appid={API_KEY}"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            return resp.json().get("list", [])
    except Exception as e:
        print(f"Warning: OpenWeather forecast API failed for ({lat}, {lon}): {e}")
    return None

def generate_fallback_forecast():
    """Generates a realistic 72-hour synthetic forecast based on diurnal atmospheric curves."""
    items = []
    base_time = datetime.now()
    for h in range(72):
        t = base_time + timedelta(hours=h)
        hour = t.hour
        diurnal_factor = 1.25 if (hour >= 20 or hour <= 7) else 0.85
        items.append({
            "dt": int(t.timestamp()),
            "main": {"aqi": 3},
            "components": {
                "co": round(750.0 * diurnal_factor + np.random.uniform(-40, 40), 1),
                "no2": round(20.0 * diurnal_factor + np.random.uniform(-3, 3), 1),
                "o3": round(45.0 * (1.4 if 12 <= hour <= 17 else 0.7), 1),
                "pm2_5": round(38.0 * diurnal_factor + np.random.uniform(-4, 6), 1),
                "pm10": round(52.0 * diurnal_factor + np.random.uniform(-5, 8), 1)
            }
        })
    return items

def generate_3day_forecast(lat: float = DEFAULT_LAT, lon: float = DEFAULT_LON, city: str = DEFAULT_CITY) -> ForecastResponse:
    """Generates 72-hour multi-model predictions and daily aggregations for any global capital or coordinate."""
    raw_list = fetch_openweather_forecast(lat, lon)
    if not raw_list or len(raw_list) < 24:
        raw_list = generate_fallback_forecast()

    # Limit to 72 hours (3 days)
    forecast_window = raw_list[:72]
    
    hourly_points = []
    prev_aqi = None

    for item in forecast_window:
        dt = datetime.fromtimestamp(item["dt"])
        comps = item.get("components", {})
        co = comps.get("co", 800.0)
        no2 = comps.get("no2", 20.0)
        o3 = comps.get("o3", 45.0)
        pm2_5 = comps.get("pm2_5", 35.0)
        pm10 = comps.get("pm10", 48.0)
        
        pm_ratio = round(pm2_5 / (pm10 + 1e-5), 4)
        raw_api_aqi = float(item.get("main", {}).get("aqi", 3.0))
        
        if prev_aqi is None:
            aqi_change_rate = 0.0
        else:
            aqi_change_rate = round(raw_api_aqi - prev_aqi, 2)
        prev_aqi = raw_api_aqi

        feature_dict = {
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
            "aqi_change_rate": aqi_change_rate
        }

        preds = model_service.predict(feature_dict)

        hourly_points.append(HourlyForecastPoint(
            datetime=dt.strftime("%Y-%m-%d %H:%M"),
            hour=dt.hour,
            day=dt.day,
            month=dt.month,
            day_name=dt.strftime("%A"),
            co=co,
            no2=no2,
            o3=o3,
            pm2_5=pm2_5,
            pm10=pm10,
            pm_ratio=pm_ratio,
            aqi_change_rate=aqi_change_rate,
            rf_aqi=preds["random_forest"],
            ridge_aqi=preds["ridge_regression"],
            dl_aqi=preds["deep_learning"],
            consensus_aqi=preds["consensus_aqi"],
            severity_badge=preds["severity_badge"],
            severity_color=preds["severity_color"]
        ))

    # Compute Daily Summaries
    df = pd.DataFrame([p.dict() for p in hourly_points])
    df["date"] = df["datetime"].apply(lambda x: x.split(" ")[0])
    
    unique_dates = df["date"].unique()[:3]
    daily_summaries = []

    for d in unique_dates:
        day_df = df[df["date"] == d]
        avg_aqi = round(float(day_df["consensus_aqi"].mean()), 2)
        peak_idx = day_df["consensus_aqi"].idxmax()
        peak_row = day_df.loc[peak_idx]
        peak_aqi = round(float(peak_row["consensus_aqi"]), 2)
        peak_hour = f"{int(peak_row['hour']):02d}:00"
        day_name = str(day_df.iloc[0]["day_name"])

        # Determine dominant pollutant
        if day_df["pm2_5"].mean() > 35:
            dom = "PM2.5 (Fine Particulates)"
        elif day_df["pm10"].mean() > 60:
            dom = "PM10 (Coarse Dust)"
        elif day_df["o3"].mean() > 60:
            dom = "Ozone (O3)"
        else:
            dom = "PM2.5"

        badge, color, advisory = model_service.get_severity(avg_aqi)

        daily_summaries.append(DaySummary(
            date=str(d),
            day_name=day_name,
            avg_aqi=avg_aqi,
            peak_aqi=peak_aqi,
            peak_hour=peak_hour,
            dominant_pollutant=dom,
            severity_badge=badge,
            severity_color=color,
            health_advisory=advisory
        ))

    return ForecastResponse(
        city=city,
        lat=lat,
        lon=lon,
        total_hours=len(hourly_points),
        daily_summaries=daily_summaries,
        hourly_forecast=hourly_points
    )
