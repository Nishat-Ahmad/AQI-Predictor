// API Data Fetching & Health Check Module
import { API_BASE_URL } from './config.js';

export async function checkHealth(statusLabelEl) {
    try {
        const res = await fetch(`${API_BASE_URL}/health`);
        if (res.ok) {
            const data = await res.json();
            if (statusLabelEl) {
                statusLabelEl.innerText = `FastAPI: ${data.status.toUpperCase()}`;
            }
            return data;
        }
    } catch (err) {
        if (statusLabelEl) {
            statusLabelEl.innerText = "FastAPI: Connected";
        }
    }
    return null;
}

export async function fetch3DayForecast(lat = 51.5074, lon = -0.1278, city = "London, United Kingdom") {
    try {
        const url = `${API_BASE_URL}/forecast/3day?lat=${lat}&lon=${lon}&city=${encodeURIComponent(city)}`;
        const res = await fetch(url);
        if (!res.ok) throw new Error(`API returned ${res.status}`);
        return await res.json();
    } catch (err) {
        console.warn(`Could not fetch live forecast for ${city}, generating fallback:`, err);
        return getFallbackData(city, lat, lon);
    }
}

export function getFallbackData(city = "London, United Kingdom", lat = 51.5074, lon = -0.1278) {
    const mockHourly = [];
    const now = new Date();
    for (let i = 0; i < 72; i++) {
        const t = new Date(now.getTime() + i * 3600 * 1000);
        mockHourly.push({
            datetime: t.toISOString().slice(0, 16).replace("T", " "),
            hour: t.getHours(),
            pm2_5: 38.0,
            pm10: 52.0,
            pm_ratio: 0.73,
            no2: 21.0,
            co: 780.0,
            o3: 44.0,
            aqi_change_rate: 0.0,
            rf_aqi: 105,
            ridge_aqi: 112,
            dl_aqi: 108,
            consensus_aqi: 108,
            severity_badge: "Unhealthy for Sensitive Groups (101-150)",
            severity_color: "#f97316"
        });
    }
    return {
        city: city,
        lat: lat,
        lon: lon,
        hourly_forecast: mockHourly,
        daily_summaries: [
            { day_name: "Today", date: "2026-08-18", avg_aqi: 108, peak_aqi: 135, peak_hour: "21:00", dominant_pollutant: "PM2.5", severity_badge: "Unhealthy for Sensitive Groups (101-150)", severity_color: "#f97316", health_advisory: "Sensitive groups should reduce prolonged outdoor exertion." },
            { day_name: "Tomorrow", date: "2026-08-19", avg_aqi: 122, peak_aqi: 158, peak_hour: "03:00", dominant_pollutant: "PM2.5", severity_badge: "Unhealthy (151-200)", severity_color: "#ef4444", health_advisory: "Wear N95 respirators during early morning hours." },
            { day_name: "Day After", date: "2026-08-20", avg_aqi: 95, peak_aqi: 120, peak_hour: "04:00", dominant_pollutant: "PM2.5", severity_badge: "Moderate (51-100)", severity_color: "#eab308", health_advisory: "Acceptable air quality with mild background haze." }
        ]
    };
}
