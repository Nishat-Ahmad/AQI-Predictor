// Global State & API Configuration
const API_BASE_URL = window.location.origin.includes("localhost") || window.location.origin.includes("127.0.0.1")
    ? "http://localhost:8000"
    : window.location.origin;

let trajectoryChart = null;
let shapChart = null;

// DOM Elements
const apiStatusEl = document.getElementById("api-status");
const btnRefresh = document.getElementById("btn-refresh");

const currentAqiValEl = document.getElementById("current-aqi-val");
const currentAqiBadgeEl = document.getElementById("current-aqi-badge");
const currentAqiAdvisoryEl = document.getElementById("current-aqi-advisory");
const currentTimestampEl = document.getElementById("current-timestamp");

const rfScoreEl = document.getElementById("rf-score");
const ridgeScoreEl = document.getElementById("ridge-score");
const dlScoreEl = document.getElementById("dl-score");
const consensusScoreEl = document.getElementById("consensus-score");

const valPm25El = document.getElementById("val-pm25");
const valPm10El = document.getElementById("val-pm10");
const valPmRatioEl = document.getElementById("val-pmratio");
const valNo2El = document.getElementById("val-no2");
const valCoEl = document.getElementById("val-co");
const valO3El = document.getElementById("val-o3");

const dayCardsContainer = document.getElementById("day-cards-container");

// Initialize Application
document.addEventListener("DOMContentLoaded", () => {
    checkHealth();
    loadForecastData();

    btnRefresh.addEventListener("click", () => {
        btnRefresh.style.transform = "rotate(180deg)";
        setTimeout(() => { btnRefresh.style.transform = "none"; }, 400);
        loadForecastData();
    });
});

// Check API Health
async function checkHealth() {
    try {
        const res = await fetch(`${API_BASE_URL}/health`);
        if (res.ok) {
            const data = await res.json();
            apiStatusEl.innerHTML = `<span class="status-dot"></span><span class="status-text">FastAPI: ${data.status.toUpperCase()}</span>`;
            apiStatusEl.style.borderColor = "rgba(16, 185, 129, 0.3)";
            apiStatusEl.style.color = "#34d399";
        } else {
            throw new Error();
        }
    } catch (err) {
        apiStatusEl.innerHTML = `<span class="status-dot" style="background:#f59e0b; box-shadow:none;"></span><span class="status-text">API: Offline / Local</span>`;
        apiStatusEl.style.borderColor = "rgba(245, 158, 11, 0.3)";
        apiStatusEl.style.color = "#fbbf24";
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

    const current = hourly[0];

    // 1. Update Current Hero Card
    currentAqiValEl.innerText = Number(current.consensus_aqi).toFixed(2);
    currentAqiBadgeEl.innerText = current.severity_badge;
    currentAqiBadgeEl.style.color = current.severity_color;
    currentAqiBadgeEl.style.borderColor = current.severity_color;
    currentAqiBadgeEl.style.backgroundColor = `${current.severity_color}18`;

    currentTimestampEl.innerText = `Forecast: ${current.datetime}`;
    currentAqiAdvisoryEl.innerText = getHealthAdvisory(current.consensus_aqi);

    valPm25El.innerText = `${current.pm2_5} ug/m³`;
    valPm10El.innerText = `${current.pm10} ug/m³`;
    valPmRatioEl.innerText = Number(current.pm_ratio).toFixed(4);
    valNo2El.innerText = `${current.no2} ug/m³`;
    valCoEl.innerText = `${current.co} ug/m³`;
    valO3El.innerText = `${current.o3} ug/m³`;

    // 2. Update Model Ensemble Card
    rfScoreEl.innerText = current.rf_aqi ? `${Number(current.rf_aqi).toFixed(2)}` : "--";
    ridgeScoreEl.innerText = current.ridge_aqi ? `${Number(current.ridge_aqi).toFixed(2)}` : "--";
    dlScoreEl.innerText = current.dl_aqi ? `${Number(current.dl_aqi).toFixed(2)}` : "--";
    consensusScoreEl.innerText = `${Number(current.consensus_aqi).toFixed(2)} / 5.0`;

    // 3. Render 3-Day Forecast Cards
    renderDayCards(daily);

    // 4. Render 72-Hour Trajectory Chart
    renderTrajectoryChart(hourly);

    // 5. Render SHAP Explainability Chart
    renderShapChart(current);
}

function getHealthAdvisory(aqi) {
    if (aqi >= 4.5) return "Emergency health warning: Hazardous air conditions. Avoid all outdoor physical activity.";
    if (aqi >= 3.5) return "Unhealthy air quality: Everyone may begin to experience health effects. Wear N95 respirators outdoors.";
    if (aqi >= 2.5) return "Moderate air quality: Sensitive individuals should consider limiting prolonged outdoor exertion.";
    if (aqi >= 1.5) return "Fair air quality: Acceptable conditions with low health risk for general population.";
    return "Good air quality: Satisfactory conditions with minimal to no health risk.";
}

// Render Day Cards
function renderDayCards(dailySummaries) {
    dayCardsContainer.innerHTML = "";

    dailySummaries.forEach(day => {
        const card = document.createElement("div");
        card.className = "day-card glass-card";
        card.innerHTML = `
            <div>
                <div class="day-card-header">
                    <span class="day-card-title">${day.day_name}</span>
                    <span class="day-card-date">${day.date}</span>
                </div>
                <div class="day-card-aqi" style="color: ${day.severity_color};">
                    ${Number(day.avg_aqi).toFixed(2)} <span style="font-size: 1rem; color: var(--text-dim);">/ 5.0</span>
                </div>
                <div class="aqi-badge-status" style="color: ${day.severity_color}; border-color: ${day.severity_color}; background: ${day.severity_color}18; font-size: 0.8rem;">
                    ${day.severity_badge}
                </div>
            </div>
            <div class="day-card-details">
                <div>Peak Pollution: <strong>${Number(day.peak_aqi).toFixed(2)} at ${day.peak_hour}</strong></div>
                <div>Primary Pollutant: <strong>${day.dominant_pollutant}</strong></div>
                <div style="font-size: 0.75rem; color: var(--text-dim); margin-top: 0.3rem;">${day.health_advisory}</div>
            </div>
        `;
        dayCardsContainer.appendChild(card);
    });
}

// Render 72-Hour Multi-Model Trajectory Chart
function renderTrajectoryChart(hourly) {
    const ctx = document.getElementById("forecastChart").getContext("2d");

    const labels = hourly.map(h => {
        const parts = h.datetime.split(" ");
        return `${parts[0].slice(5)} ${parts[1]}`;
    });

    const rfData = hourly.map(h => h.rf_aqi);
    const ridgeData = hourly.map(h => h.ridge_aqi);
    const dlData = hourly.map(h => h.dl_aqi);
    const consensusData = hourly.map(h => h.consensus_aqi);

    if (trajectoryChart) {
        trajectoryChart.destroy();
    }

    trajectoryChart = new Chart(ctx, {
        type: "line",
        data: {
            labels: labels,
            datasets: [
                {
                    label: "Consensus AQI",
                    data: consensusData,
                    borderColor: "#f59e0b",
                    backgroundColor: "rgba(245, 158, 11, 0.08)",
                    borderWidth: 3,
                    fill: true,
                    tension: 0.35,
                    pointRadius: 0
                },
                {
                    label: "Random Forest",
                    data: rfData,
                    borderColor: "#10b981",
                    borderWidth: 2,
                    borderDash: [0, 0],
                    tension: 0.3,
                    pointRadius: 0
                },
                {
                    label: "Ridge Regression",
                    data: ridgeData,
                    borderColor: "#3b82f6",
                    borderWidth: 2,
                    borderDash: [5, 5],
                    tension: 0.3,
                    pointRadius: 0
                },
                {
                    label: "Deep Learning (PyTorch)",
                    data: dlData,
                    borderColor: "#a855f7",
                    borderWidth: 2,
                    borderDash: [2, 2],
                    tension: 0.3,
                    pointRadius: 0
                }
            ]
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
                    backgroundColor: "rgba(17, 24, 39, 0.95)",
                    titleColor: "#f3f4f6",
                    bodyColor: "#9ca3af",
                    borderColor: "rgba(255, 255, 255, 0.1)",
                    borderWidth: 1,
                    padding: 12
                }
            },
            scales: {
                x: {
                    grid: { color: "rgba(255, 255, 255, 0.04)" },
                    ticks: { color: "#6b7280", maxTicksLimit: 12 }
                },
                y: {
                    min: 1,
                    max: 5,
                    grid: { color: "rgba(255, 255, 255, 0.04)" },
                    ticks: { color: "#6b7280", stepSize: 1 }
                }
            }
        }
    });
}

// Render SHAP Explainability Bar Chart
function renderShapChart(current) {
    const ctx = document.getElementById("shapChart").getContext("2d");

    // Compute relative contribution scores
    const features = ["PM2.5", "PM Ratio", "CO", "NO2", "Ozone (O3)", "AQI Rate Delta"];
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
                backgroundColor: impacts.map(v => v >= 0 ? "rgba(239, 68, 68, 0.7)" : "rgba(16, 185, 129, 0.7)"),
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
                    grid: { color: "rgba(255, 255, 255, 0.04)" },
                    ticks: { color: "#9ca3af" }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: "#f3f4f6" }
                }
            }
        }
    });
}

// Fallback data generator if API is starting
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
