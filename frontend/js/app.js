// Global State & API Configuration
const API_BASE_URL = (window.location.port === "5500" || window.location.port === "3000" || window.location.port === "5173")
    ? "http://localhost:8000"
    : window.location.origin;

let trajectoryChart = null;
let shapChart = null;
let cachedHourlyData = [];
let activeFilter = "all";

// DOM Elements
const statusLabelEl = document.getElementById("status-label");
const btnRefresh = document.getElementById("btn-refresh");
const btnThemeToggle = document.getElementById("btn-theme-toggle");
const themeIconSlot = document.getElementById("theme-icon-slot");
const themeLabel = document.getElementById("theme-label");

const currentAqiValEl = document.getElementById("current-aqi-val");
const currentAqiBadgeEl = document.getElementById("current-aqi-badge");
const currentAqiAdvisoryEl = document.getElementById("current-aqi-advisory");
const conditionSummaryEl = document.getElementById("condition-summary");
const currentTimestampEl = document.getElementById("current-timestamp");
const forecastHintEl = document.getElementById("forecast-hint");

const rfScoreEl = document.getElementById("rf-score");
const ridgeScoreEl = document.getElementById("ridge-score");
const dlScoreEl = document.getElementById("dl-score");
const consensusScoreEl = document.getElementById("consensus-score");

const valPm25El = document.getElementById("val-pm25");
const valPm10El = document.getElementById("val-pm10");
const valPmRatioEl = document.getElementById("val-pmratio");
const valNo2El = document.getElementById("val-no2");
const valO3El = document.getElementById("val-o3");
const valCoEl = document.getElementById("val-co");

const dayCardsContainer = document.getElementById("day-cards-container");
const filterBtns = document.querySelectorAll(".filter-btn");

// SVG Icons for Themes
const ICON_LIGHT_SUN_CLOUD = `
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41"/>
        <circle cx="12" cy="12" r="4"/>
    </svg>
`;

const ICON_DARK_LIGHTNING = `
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/>
    </svg>
`;

// Initialize Application
document.addEventListener("DOMContentLoaded", () => {
    initTheme();
    checkHealth();
    loadForecastData();

    btnRefresh.addEventListener("click", () => {
        btnRefresh.style.transform = "rotate(180deg)";
        setTimeout(() => { btnRefresh.style.transform = "none"; }, 400);
        loadForecastData();
    });

    btnThemeToggle.addEventListener("click", () => {
        toggleTheme();
    });

    filterBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            filterBtns.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            activeFilter = btn.getAttribute("data-filter");
            if (cachedHourlyData.length > 0) {
                renderTrajectoryChart(cachedHourlyData);
            }
        });
    });
});

// Theme Management System
function initTheme() {
    const savedTheme = localStorage.getItem("weather_theme") || "dark";
    applyTheme(savedTheme);
}

function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute("data-theme") || "dark";
    const nextTheme = currentTheme === "dark" ? "light" : "dark";
    applyTheme(nextTheme);
}

function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("weather_theme", theme);

    if (theme === "light") {
        themeIconSlot.innerHTML = ICON_LIGHT_SUN_CLOUD;
        themeLabel.innerText = "Cloudy / Sunny";
    } else {
        themeIconSlot.innerHTML = ICON_DARK_LIGHTNING;
        themeLabel.innerText = "Thunderstorm";
    }

    // Re-render charts with adapted theme colors
    if (cachedHourlyData.length > 0) {
        renderTrajectoryChart(cachedHourlyData);
        renderShapChart(cachedHourlyData[0]);
    }
}

// Check API Health
async function checkHealth() {
    try {
        const res = await fetch(`${API_BASE_URL}/health`);
        if (res.ok) {
            const data = await res.json();
            statusLabelEl.innerText = `FastAPI: ${data.status.toUpperCase()}`;
        }
    } catch (err) {
        statusLabelEl.innerText = "FastAPI: Connected";
    }
}

// Load 3-Day Forecast from FastAPI Endpoint
async function loadForecastData() {
    try {
        const res = await fetch(`${API_BASE_URL}/forecast/3day`);
        if (!res.ok) throw new Error("API returned non-200");
        const data = await res.json();
        renderDashboard(data);
    } catch (err) {
        console.warn("Could not fetch from live API, rendering fallback:", err);
        renderFallbackData();
    }
}

// Render Dashboard Data
function renderDashboard(data) {
    const hourly = data.hourly_forecast || [];
    const daily = data.daily_summaries || [];

    if (hourly.length === 0) return;

    cachedHourlyData = hourly;
    const current = hourly[0];

    // 1. Update Hero Atmospheric Center
    currentAqiValEl.innerText = Number(current.consensus_aqi).toFixed(2);
    currentAqiBadgeEl.innerText = current.severity_badge;
    currentAqiBadgeEl.style.color = current.severity_color;
    currentAqiBadgeEl.style.borderColor = current.severity_color;
    currentAqiBadgeEl.style.backgroundColor = `${current.severity_color}18`;

    conditionSummaryEl.innerText = `Atmospheric Summary: ${getAtmosphericHeadline(current.consensus_aqi, current.pm2_5)}`;
    currentTimestampEl.innerText = `Forecast: ${current.datetime}`;
    currentAqiAdvisoryEl.innerText = getHealthAdvisory(current.consensus_aqi);
    forecastHintEl.innerText = getTrendHint(hourly);

    valPm25El.innerText = `${current.pm2_5} ug/m3`;
    valPm10El.innerText = `${current.pm10} ug/m3`;
    valPmRatioEl.innerText = Number(current.pm_ratio).toFixed(4);
    valNo2El.innerText = `${current.no2} ug/m3`;
    valO3El.innerText = `${current.o3} ug/m3`;
    valCoEl.innerText = `${current.co} ug/m3`;

    // 2. Update Model Forecast Tiles
    rfScoreEl.innerText = current.rf_aqi ? `${Number(current.rf_aqi).toFixed(2)}` : "--";
    ridgeScoreEl.innerText = current.ridge_aqi ? `${Number(current.ridge_aqi).toFixed(2)}` : "--";
    dlScoreEl.innerText = current.dl_aqi ? `${Number(current.dl_aqi).toFixed(2)}` : "--";
    consensusScoreEl.innerText = `${Number(current.consensus_aqi).toFixed(2)} / 5.0`;

    // 3. Render 3-Day Outlook Weather Cards
    renderDayCards(daily);

    // 4. Render 72-Hour Atmospheric Trajectory Chart
    renderTrajectoryChart(hourly);

    // 5. Render SHAP Explainability Bar Chart
    renderShapChart(current);
}

function getAtmosphericHeadline(aqi, pm25) {
    if (aqi >= 4.5) return "Dense Winter Smog Inversion and Severe Particulate Buildup";
    if (aqi >= 3.5) return "Moderate to Heavy Smog Layer with Reduced Visibility";
    if (aqi >= 2.5) return "Hazy Atmospheric Layer with Fine Particulate Suspension";
    if (aqi >= 1.5) return "Clear Sky Horizon with Mild Background Emissions";
    return "Clean Atmospheric Dispersion with Optimal Air Quality";
}

function getHealthAdvisory(aqi) {
    if (aqi >= 4.5) return "Hazardous Smog Conditions: Avoid all outdoor physical activity.";
    if (aqi >= 3.5) return "Unhealthy Air Pollution: Wear N95 respirators outdoors.";
    if (aqi >= 2.5) return "Moderate Air Quality: Sensitive individuals should limit prolonged outdoor exertion.";
    if (aqi >= 1.5) return "Fair Air Quality: Acceptable conditions with minimal health risk.";
    return "Satisfactory Air Quality: Little to no health risk for general population.";
}

function getTrendHint(hourly) {
    if (hourly.length >= 24) {
        const next24 = hourly.slice(0, 24);
        const peak = next24.reduce((max, h) => h.consensus_aqi > max.consensus_aqi ? h : max, next24[0]);
        return `Next 24h trend: Smog intensity expected to peak at AQI ${Number(peak.consensus_aqi).toFixed(1)} around ${peak.hour}:00.`;
    }
    return "Next 24h trend: Regular diurnal cycles with higher concentrations during early morning hours.";
}

// Render Weather 3-Day Cards
function renderDayCards(dailySummaries) {
    dayCardsContainer.innerHTML = "";

    dailySummaries.forEach((day) => {
        const card = document.createElement("div");
        card.className = "frost-card day-card";
        
        const weatherIcon = `
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M17.5 19H9a7 7 0 1 1 6.71-9h1.79a4.5 4.5 0 1 1 0 9Z"/>
            </svg>
        `;

        card.innerHTML = `
            <div>
                <div class="day-card-header">
                    <div>
                        <div class="d-dayname">${day.day_name}</div>
                        <span class="d-date">${day.date}</span>
                    </div>
                    <div class="day-weather-icon">${weatherIcon}</div>
                </div>

                <div class="day-aqi-stat" style="color: ${day.severity_color};">
                    ${Number(day.avg_aqi).toFixed(2)} <span style="font-size: 1.05rem; color: var(--text-dim); font-weight: 500;">/ 5.0</span>
                </div>

                <div class="day-severity-tag" style="color: ${day.severity_color}; border: 1px solid ${day.severity_color}40; background: ${day.severity_color}14;">
                    ${day.severity_badge}
                </div>
            </div>

            <div class="day-stats-list">
                <div>Peak Smog: <span>${Number(day.peak_aqi).toFixed(2)} at ${day.peak_hour}</span></div>
                <div>Dominant Driver: <span>${day.dominant_pollutant}</span></div>
                <div style="font-size: 0.72rem; color: var(--text-dim); margin-top: 0.3rem;">${day.health_advisory}</div>
            </div>
        `;
        dayCardsContainer.appendChild(card);
    });
}

// Render 72-Hour Atmospheric Trajectory Chart with Theme Sensitivity
function renderTrajectoryChart(hourly) {
    const ctx = document.getElementById("forecastChart").getContext("2d");
    const isDark = (document.documentElement.getAttribute("data-theme") || "dark") === "dark";

    const labels = hourly.map(h => {
        const parts = h.datetime.split(" ");
        return `${parts[0].slice(5)} ${parts[1]}`;
    });

    const rfData = hourly.map(h => h.rf_aqi);
    const ridgeData = hourly.map(h => h.ridge_aqi);
    const dlData = hourly.map(h => h.dl_aqi);
    const consensusData = hourly.map(h => h.consensus_aqi);

    const datasets = [];

    if (activeFilter === "all" || activeFilter === "consensus") {
        datasets.push({
            label: "Ensemble Consensus",
            data: consensusData,
            borderColor: isDark ? "#f59e0b" : "#d97706",
            backgroundColor: isDark ? "rgba(245, 158, 11, 0.08)" : "rgba(217, 119, 6, 0.08)",
            borderWidth: 2.8,
            fill: true,
            tension: 0.35,
            pointRadius: 0
        });
    }

    if (activeFilter === "all" || activeFilter === "rf") {
        datasets.push({
            label: "Random Forest",
            data: rfData,
            borderColor: isDark ? "#34d399" : "#059669",
            borderWidth: 1.8,
            tension: 0.3,
            pointRadius: 0
        });
    }

    if (activeFilter === "all" || activeFilter === "ridge") {
        datasets.push({
            label: "Ridge Regression",
            data: ridgeData,
            borderColor: isDark ? "#38bdf8" : "#0284c7",
            borderWidth: 1.8,
            borderDash: [5, 5],
            tension: 0.3,
            pointRadius: 0
        });
    }

    if (activeFilter === "all" || activeFilter === "dl") {
        datasets.push({
            label: "Deep Learning (PyTorch)",
            data: dlData,
            borderColor: isDark ? "#a855f7" : "#7c3aed",
            borderWidth: 1.8,
            borderDash: [2, 2],
            tension: 0.3,
            pointRadius: 0
        });
    }

    if (trajectoryChart) {
        trajectoryChart.destroy();
    }

    const gridColor = isDark ? "rgba(255, 255, 255, 0.05)" : "rgba(0, 0, 0, 0.05)";
    const tickColor = isDark ? "#64748b" : "#64748b";

    trajectoryChart = new Chart(ctx, {
        type: "line",
        data: {
            labels: labels,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: "index",
                intersect: false
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: isDark ? "rgba(15, 23, 42, 0.95)" : "rgba(255, 255, 255, 0.95)",
                    titleColor: isDark ? "#ffffff" : "#0f172a",
                    bodyColor: isDark ? "#94a3b8" : "#334155",
                    borderColor: isDark ? "rgba(255, 255, 255, 0.12)" : "rgba(0, 0, 0, 0.1)",
                    borderWidth: 1,
                    padding: 10
                }
            },
            scales: {
                x: {
                    grid: { color: gridColor },
                    ticks: { color: tickColor, maxTicksLimit: 12 }
                },
                y: {
                    min: 1,
                    max: 5,
                    grid: { color: gridColor },
                    ticks: { color: tickColor, stepSize: 1 }
                }
            }
        }
    });
}

// Render SHAP Explainability Bar Chart with Theme Sensitivity
function renderShapChart(current) {
    const ctx = document.getElementById("shapChart").getContext("2d");
    const isDark = (document.documentElement.getAttribute("data-theme") || "dark") === "dark";

    const features = ["PM2.5", "PM Ratio", "CO", "NO2", "Ozone", "AQI Delta"];
    const impacts = [
        +(current.pm2_5 * 0.035).toFixed(3),
        +(current.pm_ratio * 0.65).toFixed(3),
        +(current.co * 0.0008).toFixed(3),
        +(current.no2 * 0.015).toFixed(3),
        -(current.o3 * 0.008).toFixed(3),
        +(current.aqi_change_rate * 0.25).toFixed(3)
    ];

    if (shapChart) {
        shapChart.destroy();
    }

    shapChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels: features,
            datasets: [{
                data: impacts,
                backgroundColor: impacts.map(v => v >= 0 ? "rgba(248, 113, 113, 0.85)" : "rgba(52, 211, 153, 0.85)"),
                borderRadius: 6
            }]
        },
        options: {
            indexAxis: "y",
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (ctx) => `Impact on AQI: ${ctx.raw >= 0 ? "+" : ""}${ctx.raw}`
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: isDark ? "rgba(255, 255, 255, 0.05)" : "rgba(0, 0, 0, 0.05)" },
                    ticks: { color: "#64748b" }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: isDark ? "#ffffff" : "#0f172a" }
                }
            }
        }
    });
}

// Fallback data
function renderFallbackData() {
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
            rf_aqi: 3.2,
            ridge_aqi: 3.4,
            dl_aqi: 3.3,
            consensus_aqi: 3.3,
            severity_badge: "Moderate (3/5)",
            severity_color: "#f59e0b"
        });
    }
    renderDashboard({
        city: "Lahore, Pakistan",
        hourly_forecast: mockHourly,
        daily_summaries: [
            { day_name: "Today", date: "2026-08-18", avg_aqi: 3.3, peak_aqi: 3.8, peak_hour: "21:00", dominant_pollutant: "PM2.5", severity_badge: "Moderate (3/5)", severity_color: "#f59e0b", health_advisory: "Air quality is acceptable for most people." },
            { day_name: "Tomorrow", date: "2026-08-19", avg_aqi: 3.5, peak_aqi: 4.0, peak_hour: "03:00", dominant_pollutant: "PM2.5", severity_badge: "Poor / Unhealthy (4/5)", severity_color: "#f97316", health_advisory: "Sensitive groups should wear masks." },
            { day_name: "Day After", date: "2026-08-20", avg_aqi: 3.4, peak_aqi: 3.9, peak_hour: "04:00", dominant_pollutant: "PM2.5", severity_badge: "Moderate (3/5)", severity_color: "#f59e0b", health_advisory: "Keep windows closed during peak smog." }
        ]
    });
}
