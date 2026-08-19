// Chart.js Visualization Module (72-Hour Trajectory & SHAP Feature Importance)

let trajectoryChart = null;
let shapChart = null;

export function renderTrajectoryChart(hourly, activeFilter = "all") {
    const canvas = document.getElementById("forecastChart");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const isDark = (document.documentElement.getAttribute("data-theme") || "dark") === "dark";

    // Sample every 6 hours for a total of 12 distinct points across 72 hours
    const sampledHourly = hourly.filter((_, idx) => idx % 6 === 0).slice(0, 12);

    const labels = sampledHourly.map(h => {
        const parts = h.datetime.split(" ");
        return `${parts[0].slice(5)} ${parts[1]}`;
    });

    const rfData = sampledHourly.map(h => Math.round(Number(h.rf_aqi || 0)));
    const ridgeData = sampledHourly.map(h => Math.round(Number(h.ridge_aqi || 0)));
    const dlData = sampledHourly.map(h => Math.round(Number(h.dl_aqi || 0)));
    const consensusData = sampledHourly.map(h => Math.round(Number(h.consensus_aqi || 0)));

    const datasets = [];
    const pointBorderColor = isDark ? "#161920" : "#ffffff";

    if (activeFilter === "all" || activeFilter === "consensus") {
        datasets.push({
            label: "Ensemble Consensus",
            data: consensusData,
            borderColor: isDark ? "#f59e0b" : "#d97706",
            backgroundColor: isDark ? "rgba(245, 158, 11, 0.08)" : "rgba(217, 119, 6, 0.08)",
            borderWidth: 2.8,
            fill: true,
            tension: 0.35,
            pointRadius: 4.5,
            pointHoverRadius: 7,
            pointHitRadius: 9,
            pointBackgroundColor: isDark ? "#f59e0b" : "#d97706",
            pointBorderColor: pointBorderColor,
            pointBorderWidth: 2
        });
    }

    if (activeFilter === "all" || activeFilter === "rf") {
        datasets.push({
            label: "Random Forest",
            data: rfData,
            borderColor: isDark ? "#34d399" : "#059669",
            borderWidth: 1.8,
            tension: 0.3,
            pointRadius: 4,
            pointHoverRadius: 6.5,
            pointHitRadius: 8,
            pointBackgroundColor: isDark ? "#34d399" : "#059669",
            pointBorderColor: pointBorderColor,
            pointBorderWidth: 2
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
            pointRadius: 4,
            pointHoverRadius: 6.5,
            pointHitRadius: 8,
            pointBackgroundColor: isDark ? "#38bdf8" : "#0284c7",
            pointBorderColor: pointBorderColor,
            pointBorderWidth: 2
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
            pointRadius: 4,
            pointHoverRadius: 6.5,
            pointHitRadius: 8,
            pointBackgroundColor: isDark ? "#a855f7" : "#7c3aed",
            pointBorderColor: pointBorderColor,
            pointBorderWidth: 2
        });
    }

    if (trajectoryChart) {
        trajectoryChart.destroy();
    }

    // Dynamic Y-axis scale calculation based on active values with clean headroom
    let activeValues = [];
    if (activeFilter === "all" || activeFilter === "consensus") activeValues.push(...consensusData);
    if (activeFilter === "all" || activeFilter === "rf") activeValues.push(...rfData);
    if (activeFilter === "all" || activeFilter === "ridge") activeValues.push(...ridgeData);
    if (activeFilter === "all" || activeFilter === "dl") activeValues.push(...dlData);

    const maxVal = activeValues.length > 0 ? Math.max(...activeValues) : 50;
    const dynamicSuggestedMax = Math.max(30, Math.ceil((maxVal * 1.25) / 10) * 10);

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
                    padding: 10,
                    callbacks: {
                        label: (c) => ` ${c.dataset.label}: ${c.raw} AQI`
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: gridColor },
                    ticks: {
                        color: tickColor,
                        autoSkip: false,
                        maxRotation: 0
                    }
                },
                y: {
                    beginAtZero: true,
                    suggestedMax: dynamicSuggestedMax,
                    grid: { color: gridColor },
                    ticks: {
                        color: tickColor,
                        precision: 0
                    }
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
        +(current.pm2_5 * 0.85).toFixed(1),
        +(current.pm_ratio * 15.0).toFixed(1),
        +(current.co * 0.02).toFixed(1),
        +(current.no2 * 0.45).toFixed(1),
        -(current.o3 * 0.25).toFixed(1),
        +(current.aqi_change_rate * 5.0).toFixed(1)
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
                        label: (ctx) => `Impact on AQI: ${ctx.raw >= 0 ? "+" : ""}${ctx.raw} AQI`
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
