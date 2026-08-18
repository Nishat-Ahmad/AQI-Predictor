// Dynamic Theme Switcher Module (Thunderstorm Dark vs Cloudy/Sunny Light)
import { ICON_LIGHT_SUN_CLOUD, ICON_DARK_MOON } from './config.js';

export function initTheme(onThemeChangeCallback) {
    const savedTheme = localStorage.getItem("weather_theme") || "dark";
    applyTheme(savedTheme, onThemeChangeCallback);
}

export function toggleTheme(onThemeChangeCallback) {
    const currentTheme = document.documentElement.getAttribute("data-theme") || "dark";
    const nextTheme = currentTheme === "dark" ? "light" : "dark";
    applyTheme(nextTheme, onThemeChangeCallback);
}

export function applyTheme(theme, onThemeChangeCallback) {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("weather_theme", theme);

    const themeIconSlot = document.getElementById("theme-icon-slot");
    const btnThemeToggle = document.getElementById("btn-theme-toggle");

    if (themeIconSlot) {
        if (theme === "light") {
            themeIconSlot.innerHTML = ICON_LIGHT_SUN_CLOUD;
            if (btnThemeToggle) btnThemeToggle.setAttribute("title", "Switch to Dark Theme");
        } else {
            themeIconSlot.innerHTML = ICON_DARK_MOON;
            if (btnThemeToggle) btnThemeToggle.setAttribute("title", "Switch to Light Theme");
        }
    }

    if (typeof onThemeChangeCallback === "function") {
        onThemeChangeCallback(theme);
    }
}
