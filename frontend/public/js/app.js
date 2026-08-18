// Application Orchestrator & Entrypoint
import { initTheme, toggleTheme } from './theme.js';
import { checkHealth, fetch3DayForecast } from './api.js';
import { initGlobalMap, updateMapTheme } from './map.js';
import { renderTrajectoryChart, renderShapChart } from './charts.js';
import { renderHeroAtmosphere, renderModelTiles, renderDayCards } from './ui.js';

let cachedHourlyData = [];
let activeFilter = "all";
let activeCity = {
    name: "Lahore",
    country: "Pakistan",
    lat: 31.5204,
    lon: 74.3587
};

// Initialize Application
document.addEventListener("DOMContentLoaded", () => {
    const statusLabelEl = document.getElementById("status-label");
    const btnRefresh = document.getElementById("btn-refresh");
    const btnThemeToggle = document.getElementById("btn-theme-toggle");
    const filterBtns = document.querySelectorAll(".filter-btn");

    // 1. Initialize Theme with reactive chart & map callback
    initTheme((theme) => {
        updateMapTheme(theme);
        if (cachedHourlyData.length > 0) {
            renderTrajectoryChart(cachedHourlyData, activeFilter);
            renderShapChart(cachedHourlyData[0]);
        }
    });

    // 2. Initialize Interactive Global Capitals Map
    initGlobalMap((selectedCity) => {
        activeCity = selectedCity;
        loadAndRenderDashboard(activeCity.lat, activeCity.lon, `${activeCity.name}, ${activeCity.country}`);
    });

    // 3. Health Check & Live Forecast
    checkHealth(statusLabelEl);
    loadAndRenderDashboard(activeCity.lat, activeCity.lon, `${activeCity.name}, ${activeCity.country}`);

    // 4. Event Listeners
    if (btnRefresh) {
        btnRefresh.addEventListener("click", () => {
            btnRefresh.style.transform = "rotate(180deg)";
            setTimeout(() => { btnRefresh.style.transform = "none"; }, 400);
            loadAndRenderDashboard(activeCity.lat, activeCity.lon, `${activeCity.name}, ${activeCity.country}`);
        });
    }

    if (btnThemeToggle) {
        btnThemeToggle.addEventListener("click", () => {
            toggleTheme((theme) => {
                updateMapTheme(theme);
                if (cachedHourlyData.length > 0) {
                    renderTrajectoryChart(cachedHourlyData, activeFilter);
                    renderShapChart(cachedHourlyData[0]);
                }
            });
        });
    }

    filterBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            filterBtns.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            activeFilter = btn.getAttribute("data-filter");
            if (cachedHourlyData.length > 0) {
                renderTrajectoryChart(cachedHourlyData, activeFilter);
            }
        });
    });
});

// Orchestrate Dashboard Rendering
async function loadAndRenderDashboard(lat, lon, cityName) {
    const data = await fetch3DayForecast(lat, lon, cityName);
    if (!data) return;

    const hourly = data.hourly_forecast || [];
    const daily = data.daily_summaries || [];

    if (hourly.length === 0) return;

    cachedHourlyData = hourly;
    const current = hourly[0];

    // Update Brand Subline & Active City Badge
    const brandHeadingEl = document.querySelector(".brand-heading");
    if (brandHeadingEl) {
        brandHeadingEl.innerText = "AQI Predictor";
    }

    const brandSublineEl = document.querySelector(".brand-subline");
    if (brandSublineEl) {
        brandSublineEl.innerText = data.city 
            ? `3-Day Air Quality & Weather Forecast for ${data.city}`
            : "3-Day Air Quality & Weather Forecast (72 Hours)";
    }

    // Render Components
    renderHeroAtmosphere(current, hourly);
    renderModelTiles(current);
    renderDayCards(daily);
    renderTrajectoryChart(hourly, activeFilter);
    renderShapChart(current);
}
