// Application Orchestrator & Entrypoint
import { initTheme, toggleTheme } from './theme.js';
import { checkHealth, fetch3DayForecast } from './api.js';
import { renderTrajectoryChart, renderShapChart } from './charts.js';
import { renderHeroAtmosphere, renderModelTiles, renderDayCards } from './ui.js';

let cachedHourlyData = [];
let activeFilter = "all";

// Initialize Application
document.addEventListener("DOMContentLoaded", () => {
    const statusLabelEl = document.getElementById("status-label");
    const btnRefresh = document.getElementById("btn-refresh");
    const btnThemeToggle = document.getElementById("btn-theme-toggle");
    const filterBtns = document.querySelectorAll(".filter-btn");

    // 1. Initialize Theme with reactive chart update callback
    initTheme((theme) => {
        if (cachedHourlyData.length > 0) {
            renderTrajectoryChart(cachedHourlyData, activeFilter);
            renderShapChart(cachedHourlyData[0]);
        }
    });

    // 2. Health Check & Live Forecast
    checkHealth(statusLabelEl);
    loadAndRenderDashboard();

    // 3. Event Listeners
    if (btnRefresh) {
        btnRefresh.addEventListener("click", () => {
            btnRefresh.style.transform = "rotate(180deg)";
            setTimeout(() => { btnRefresh.style.transform = "none"; }, 400);
            loadAndRenderDashboard();
        });
    }

    if (btnThemeToggle) {
        btnThemeToggle.addEventListener("click", () => {
            toggleTheme((theme) => {
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
async function loadAndRenderDashboard() {
    const data = await fetch3DayForecast();
    if (!data) return;

    const hourly = data.hourly_forecast || [];
    const daily = data.daily_summaries || [];

    if (hourly.length === 0) return;

    cachedHourlyData = hourly;
    const current = hourly[0];

    // Render Components
    renderHeroAtmosphere(current, hourly);
    renderModelTiles(current);
    renderDayCards(daily);
    renderTrajectoryChart(hourly, activeFilter);
    renderShapChart(current);
}
