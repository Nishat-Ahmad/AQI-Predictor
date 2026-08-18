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

export const ICON_DARK_MOON = `
    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/>
    </svg>
`;

export const WEATHER_CLOUD_ICON = `
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M17.5 19H9a7 7 0 1 1 6.71-9h1.79a4.5 4.5 0 1 1 0 9Z"/>
    </svg>
`;

export function getAtmosphericHeadline(aqi, pm25) {
    if (aqi >= 300) return "Emergency Hazardous Smog Layer and Extreme Particulate Buildup";
    if (aqi >= 200) return "Dense Smog Layer with Very Poor Visibility and Heavy Inversion";
    if (aqi >= 150) return "Unhealthy Particulate Suspension with Pronounced Smog Layer";
    if (aqi >= 100) return "Hazy Atmospheric Layer with Fine Particulate Suspension";
    if (aqi >= 50) return "Moderate Atmospheric Dispersion with Light Background Haze";
    return "Clean Atmospheric Dispersion with Optimal Air Quality";
}

export function getHealthAdvisory(aqi) {
    if (aqi >= 300) return "Hazardous Air Quality: Avoid all outdoor physical activity. Keep indoor air purifiers running.";
    if (aqi >= 200) return "Very Unhealthy Air Quality: Everyone should avoid outdoor exertion; wear N95 respirators.";
    if (aqi >= 150) return "Unhealthy Air: Active children and adults, and people with respiratory disease should avoid outdoor exertion.";
    if (aqi >= 100) return "Unhealthy for Sensitive Groups: People with respiratory issues should limit prolonged outdoor exertion.";
    if (aqi >= 50) return "Moderate Air Quality: Air quality is acceptable for most; unusually sensitive individuals should take care.";
    return "Good Air Quality: Air quality is satisfactory and poses little or no risk.";
}

export function getTrendHint(hourly) {
    if (hourly && hourly.length >= 24) {
        const next24 = hourly.slice(0, 24);
        const peak = next24.reduce((max, h) => h.consensus_aqi > max.consensus_aqi ? h : max, next24[0]);
        return `Next 24h trend: Air pollution expected to peak at AQI ${Math.round(peak.consensus_aqi)} around ${peak.hour}:00.`;
    }
    return "Next 24h trend: Regular diurnal cycles with higher concentrations during early morning hours.";
}
