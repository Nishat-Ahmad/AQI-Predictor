// Dynamic Theme Switcher Module (Thunderstorm Dark vs Cloudy/Sunny Light)
import { ICON_LIGHT_SUN_CLOUD, ICON_DARK_LIGHTNING } from './config.js';

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
    const themeLabel = document.getElementById("theme-label");

    if (themeIconSlot && themeLabel) {
        if (theme === "light") {
            themeIconSlot.innerHTML = ICON_LIGHT_SUN_CLOUD;
            themeLabel.innerText = "Cloudy / Sunny";
        } else {
            themeIconSlot.innerHTML = ICON_DARK_LIGHTNING;
            themeLabel.innerText = "Thunderstorm";
        }
    }

    if (typeof onThemeChangeCallback === "function") {
        onThemeChangeCallback(theme);
    }
}
