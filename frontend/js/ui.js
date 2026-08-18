// DOM UI Rendering Module
import { getAtmosphericHeadline, getHealthAdvisory, getTrendHint, WEATHER_CLOUD_ICON } from './config.js';

export function renderHeroAtmosphere(current, hourly) {
    const currentAqiValEl = document.getElementById("current-aqi-val");
    const currentAqiBadgeEl = document.getElementById("current-aqi-badge");
    const currentAqiAdvisoryEl = document.getElementById("current-aqi-advisory");
    const conditionSummaryEl = document.getElementById("condition-summary");
    const currentTimestampEl = document.getElementById("current-timestamp");
    const forecastHintEl = document.getElementById("forecast-hint");

    const valPm25El = document.getElementById("val-pm25");
    const valPm10El = document.getElementById("val-pm10");
    const valPmRatioEl = document.getElementById("val-pmratio");
    const valNo2El = document.getElementById("val-no2");
    const valO3El = document.getElementById("val-o3");
    const valCoEl = document.getElementById("val-co");

    const aqiInt = Math.round(Number(current.consensus_aqi));

    if (currentAqiValEl) currentAqiValEl.innerText = aqiInt;
    if (currentAqiBadgeEl) {
        currentAqiBadgeEl.innerText = current.severity_badge;
        currentAqiBadgeEl.style.color = current.severity_color;
        currentAqiBadgeEl.style.borderColor = current.severity_color;
        currentAqiBadgeEl.style.backgroundColor = `${current.severity_color}18`;
    }

    if (conditionSummaryEl) conditionSummaryEl.innerText = `Atmospheric Summary: ${getAtmosphericHeadline(aqiInt, current.pm2_5)}`;
    if (currentTimestampEl) currentTimestampEl.innerText = `Forecast: ${current.datetime}`;
    if (currentAqiAdvisoryEl) currentAqiAdvisoryEl.innerText = getHealthAdvisory(aqiInt);
    if (forecastHintEl) forecastHintEl.innerText = getTrendHint(hourly);

    if (valPm25El) valPm25El.innerText = `${current.pm2_5} ug/m3`;
    if (valPm10El) valPm10El.innerText = `${current.pm10} ug/m3`;
    if (valPmRatioEl) valPmRatioEl.innerText = Number(current.pm_ratio).toFixed(4);
    if (valNo2El) valNo2El.innerText = `${current.no2} ug/m3`;
    if (valO3El) valO3El.innerText = `${current.o3} ug/m3`;
    if (valCoEl) valCoEl.innerText = `${current.co} ug/m3`;
}

export function renderModelTiles(current) {
    const rfScoreEl = document.getElementById("rf-score");
    const ridgeScoreEl = document.getElementById("ridge-score");
    const dlScoreEl = document.getElementById("dl-score");
    const consensusScoreEl = document.getElementById("consensus-score");

    if (rfScoreEl) rfScoreEl.innerText = current.rf_aqi ? `${Math.round(Number(current.rf_aqi))}` : "--";
    if (ridgeScoreEl) ridgeScoreEl.innerText = current.ridge_aqi ? `${Math.round(Number(current.ridge_aqi))}` : "--";
    if (dlScoreEl) dlScoreEl.innerText = current.dl_aqi ? `${Math.round(Number(current.dl_aqi))}` : "--";
    if (consensusScoreEl) consensusScoreEl.innerText = `${Math.round(Number(current.consensus_aqi))} AQI`;
}

export function renderDayCards(dailySummaries) {
    const dayCardsContainer = document.getElementById("day-cards-container");
    if (!dayCardsContainer) return;

    dayCardsContainer.innerHTML = "";

    dailySummaries.forEach((day) => {
        const card = document.createElement("div");
        card.className = "frost-card day-card";
        const avgAqiInt = Math.round(Number(day.avg_aqi));
        const peakAqiInt = Math.round(Number(day.peak_aqi));

        card.innerHTML = `
            <div>
                <div class="day-card-header">
                    <div>
                        <div class="d-dayname">${day.day_name}</div>
                        <span class="d-date">${day.date}</span>
                    </div>
                    <div class="day-weather-icon">${WEATHER_CLOUD_ICON}</div>
                </div>

                <div class="day-aqi-stat" style="color: ${day.severity_color};">
                    ${avgAqiInt} <span style="font-size: 0.95rem; color: var(--text-dim); font-weight: 500;">AQI</span>
                </div>

                <div class="day-severity-tag" style="color: ${day.severity_color}; border: 1px solid ${day.severity_color}40; background: ${day.severity_color}14;">
                    ${day.severity_badge}
                </div>
            </div>

            <div class="day-stats-list">
                <div>Peak AQI: <span>${peakAqiInt} at ${day.peak_hour}</span></div>
                <div>Dominant Driver: <span>${day.dominant_pollutant}</span></div>
                <div style="font-size: 0.72rem; color: var(--text-dim); margin-top: 0.3rem;">${day.health_advisory}</div>
            </div>
        `;
        dayCardsContainer.appendChild(card);
    });
}
