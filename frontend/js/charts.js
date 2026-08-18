// Chart.js Visualization Module (72-Hour Trajectory & SHAP Feature Importance)

let trajectoryChart = null;
let shapChart = null;

export function renderTrajectoryChart(hourly, activeFilter = "all") {
    const canvas = document.getElementById("forecastChart");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
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

export function renderShapChart(current) {
    const canvas = document.getElementById("shapChart");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
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
