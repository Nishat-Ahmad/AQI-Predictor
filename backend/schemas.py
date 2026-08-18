from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class EnvironmentalInput(BaseModel):
    hour: int = Field(14, ge=0, le=23, description="Hour of Day (0-23)")
    day: int = Field(15, ge=1, le=31, description="Day of Month (1-31)")
    month: int = Field(8, ge=1, le=12, description="Month (1-12)")
    day_of_week: int = Field(0, ge=0, le=6, description="Day of Week (0=Monday, 6=Sunday)")
    co: float = Field(850.0, ge=0.0, description="Carbon Monoxide in ug/m3")
    no2: float = Field(22.0, ge=0.0, description="Nitrogen Dioxide in ug/m3")
    o3: float = Field(48.0, ge=0.0, description="Ozone in ug/m3")
    pm2_5: float = Field(42.0, ge=0.0, description="PM2.5 in ug/m3")
    pm10: float = Field(55.0, ge=0.0, description="PM10 in ug/m3")
    pm_ratio: Optional[float] = Field(None, description="Calculated PM2.5 / PM10 ratio")
    aqi_change_rate: Optional[float] = Field(0.0, description="Hourly AQI rate of change")

class MultiModelPrediction(BaseModel):
    random_forest: Optional[float] = None
    ridge_regression: Optional[float] = None
    deep_learning: Optional[float] = None
    consensus_aqi: float
    severity_badge: str
    severity_color: str

class PredictionResponse(BaseModel):
    status: str
    input_features: Dict[str, Any]
    predictions: MultiModelPrediction

class HourlyForecastPoint(BaseModel):
    datetime: str
    hour: int
    day: int
    month: int
    day_name: str
    co: float
    no2: float
    o3: float
    pm2_5: float
    pm10: float
    pm_ratio: float
    aqi_change_rate: float
    rf_aqi: Optional[float]
    ridge_aqi: Optional[float]
    dl_aqi: Optional[float]
    consensus_aqi: float
    severity_badge: str
    severity_color: str

class DaySummary(BaseModel):
    date: str
    day_name: str
    avg_aqi: float
    peak_aqi: float
    peak_hour: str
    dominant_pollutant: str
    severity_badge: str
    severity_color: str
    health_advisory: str

class ForecastResponse(BaseModel):
    city: str
    lat: float
    lon: float
    total_hours: int
    daily_summaries: List[DaySummary]
    hourly_forecast: List[HourlyForecastPoint]

class ExplainRequest(BaseModel):
    features: EnvironmentalInput

class ExplainResponse(BaseModel):
    base_value: float
    prediction: float
    contributions: List[Dict[str, Any]]
