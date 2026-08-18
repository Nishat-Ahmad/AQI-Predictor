// Global API Configuration & Constants
export const API_BASE_URL = (window.location.port === "5500" || window.location.port === "3000" || window.location.port === "5173")
    ? "http://localhost:8000"
    : window.location.origin;

export const ICON_LIGHT_SUN_CLOUD = `
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41"/>
        <circle cx="12" cy="12" r="4"/>
    </svg>
`;

export const ICON_DARK_LIGHTNING = `
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/>
    </svg>
`;

export const WEATHER_CLOUD_ICON = `
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M17.5 19H9a7 7 0 1 1 6.71-9h1.79a4.5 4.5 0 1 1 0 9Z"/>
    </svg>
`;

export function getAtmosphericHeadline(aqi, pm25) {
    if (aqi >= 4.5) return "Dense Winter Smog Inversion and Severe Particulate Buildup";
    if (aqi >= 3.5) return "Moderate to Heavy Smog Layer with Reduced Visibility";
    if (aqi >= 2.5) return "Hazy Atmospheric Layer with Fine Particulate Suspension";
    if (aqi >= 1.5) return "Clear Sky Horizon with Mild Background Emissions";
    return "Clean Atmospheric Dispersion with Optimal Air Quality";
}

export function getHealthAdvisory(aqi) {
    if (aqi >= 4.5) return "Hazardous Smog Conditions: Avoid all outdoor physical activity.";
    if (aqi >= 3.5) return "Unhealthy Air Pollution: Wear N95 respirators outdoors.";
    if (aqi >= 2.5) return "Moderate Air Quality: Sensitive individuals should limit prolonged outdoor exertion.";
    if (aqi >= 1.5) return "Fair Air Quality: Acceptable conditions with minimal health risk.";
    return "Satisfactory Air Quality: Little to no health risk for general population.";
}

export function getTrendHint(hourly) {
    if (hourly && hourly.length >= 24) {
        const next24 = hourly.slice(0, 24);
        const peak = next24.reduce((max, h) => h.consensus_aqi > max.consensus_aqi ? h : max, next24[0]);
        return `Next 24h trend: Smog intensity expected to peak at AQI ${Number(peak.consensus_aqi).toFixed(1)} around ${peak.hour}:00.`;
    }
    return "Next 24h trend: Regular diurnal cycles with higher concentrations during early morning hours.";
}
