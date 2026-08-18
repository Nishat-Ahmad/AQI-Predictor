// Leaflet Interactive World Map Module with Real-time Search
import { GLOBAL_CAPITALS } from './capitals.js';

let mapInstance = null;
let tileLayer = null;
let markersMap = {};
let selectedMarker = null;

const DARK_TILES = "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png";
const LIGHT_TILES = "https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png";
const TILE_ATTRIB = "&copy; <a href='https://www.openstreetmap.org/copyright'>OpenStreetMap</a>, &copy; <a href='https://carto.com/attributions'>CARTO</a>";

export function initGlobalMap(onCitySelectCallback) {
    const mapContainer = document.getElementById("world-map");
    if (!mapContainer || typeof L === "undefined") return;

    const isDark = (document.documentElement.getAttribute("data-theme") || "dark") === "dark";

    // Initialize Leaflet Map centered on world view
    mapInstance = L.map("world-map", {
        center: [28.0, 35.0],
        zoom: 2.2,
        minZoom: 1.8,
        maxZoom: 10,
        zoomControl: false,
        attributionControl: false
    });

    L.control.zoom({ position: "bottomright" }).addTo(mapInstance);

    tileLayer = L.tileLayer(isDark ? DARK_TILES : LIGHT_TILES, {
        attribution: TILE_ATTRIB,
        subdomains: "abcd",
        maxZoom: 19
    }).addTo(mapInstance);

    // Setup Search Bar & Render Markers
    setupCapitalSearchBar(onCitySelectCallback);
    renderCapitalMarkers(onCitySelectCallback);

    // Click anywhere on map to select custom coordinates
    mapInstance.on("click", (e) => {
        const { lat, lng } = e.latlng;
        const customCity = `Lat ${lat.toFixed(2)}, Lon ${lng.toFixed(2)}`;
        if (typeof onCitySelectCallback === "function") {
            onCitySelectCallback({
                name: customCity,
                country: "Custom Location",
                lat: lat,
                lon: lng
            });
        }
    });
}

function createCustomPin(isCurrent = false) {
    return L.divIcon({
        className: "custom-map-pin-container",
        html: `
            <div class="map-pulse-marker ${isCurrent ? 'active-pin' : ''}">
                <div class="marker-core"></div>
                <div class="marker-pulse"></div>
            </div>
        `,
        iconSize: [20, 20],
        iconAnchor: [10, 10]
    });
}

function renderCapitalMarkers(onCitySelectCallback) {
    GLOBAL_CAPITALS.forEach((cap) => {
        const marker = L.marker([cap.lat, cap.lon], {
            icon: createCustomPin(cap.name === "London")
        }).addTo(mapInstance);

        marker.bindTooltip(`<strong>${cap.name}</strong>, ${cap.country}`, {
            direction: "top",
            offset: [0, -8],
            className: "capital-map-tooltip"
        });

        marker.on("click", (e) => {
            L.DomEvent.stopPropagation(e);
            selectCityMarker(cap, onCitySelectCallback);
        });

        markersMap[`${cap.name}_${cap.country}`] = marker;

        if (cap.name === "London") {
            selectedMarker = marker;
        }
    });
}

export function selectCityMarker(cap, onCitySelectCallback) {
    if (!mapInstance) return;

    // Reset previous active marker icon
    if (selectedMarker) {
        selectedMarker.setIcon(createCustomPin(false));
    }

    const key = `${cap.name}_${cap.country}`;
    if (markersMap[key]) {
        selectedMarker = markersMap[key];
        selectedMarker.setIcon(createCustomPin(true));
    }

    mapInstance.flyTo([cap.lat, cap.lon], 5, {
        duration: 1.2
    });

    const mapTitleEl = document.getElementById("map-selected-city");
    if (mapTitleEl) {
        mapTitleEl.innerText = `${cap.name}, ${cap.country}`;
    }

    const searchInput = document.getElementById("capital-search-input");
    if (searchInput) {
        searchInput.value = `${cap.name}, ${cap.country}`;
    }

    const clearBtn = document.getElementById("btn-clear-search");
    if (clearBtn) {
        clearBtn.style.display = searchInput && searchInput.value ? "block" : "none";
    }

    const resultsDropdown = document.getElementById("search-results-dropdown");
    if (resultsDropdown) {
        resultsDropdown.style.display = "none";
    }

    if (typeof onCitySelectCallback === "function") {
        onCitySelectCallback(cap);
    }
}

function setupCapitalSearchBar(onCitySelectCallback) {
    const searchInput = document.getElementById("capital-search-input");
    const resultsDropdown = document.getElementById("search-results-dropdown");
    const clearBtn = document.getElementById("btn-clear-search");

    if (!searchInput || !resultsDropdown) return;

    let activeIndex = -1;

    function renderResults(query) {
        const q = query.trim().toLowerCase();
        if (!q) {
            resultsDropdown.style.display = "none";
            if (clearBtn) clearBtn.style.display = "none";
            return;
        }

        if (clearBtn) clearBtn.style.display = "block";

        const matches = GLOBAL_CAPITALS.filter(c => 
            c.name.toLowerCase().includes(q) || 
            c.country.toLowerCase().includes(q) ||
            c.region.toLowerCase().includes(q)
        ).slice(0, 10);

        resultsDropdown.innerHTML = "";
        activeIndex = -1;

        if (matches.length === 0) {
            resultsDropdown.innerHTML = `<div class="search-no-results">No world capitals found matching "${query}"</div>`;
            resultsDropdown.style.display = "block";
            return;
        }

        matches.forEach((c, idx) => {
            const item = document.createElement("div");
            item.className = "search-result-item";
            item.dataset.index = idx;
            item.innerHTML = `
                <div class="sr-left">
                    <span class="sr-city">${c.name}</span>
                    <span class="sr-country">${c.country}</span>
                </div>
                <span class="sr-region">${c.region}</span>
            `;

            item.addEventListener("click", () => {
                selectCityMarker(c, onCitySelectCallback);
                resultsDropdown.style.display = "none";
            });

            resultsDropdown.appendChild(item);
        });

        resultsDropdown.style.display = "block";
    }

    searchInput.addEventListener("input", (e) => {
        renderResults(e.target.value);
    });

    searchInput.addEventListener("focus", (e) => {
        if (e.target.value.trim()) {
            renderResults(e.target.value);
        }
    });

    // Keyboard navigation (Arrow keys + Enter)
    searchInput.addEventListener("keydown", (e) => {
        const items = resultsDropdown.querySelectorAll(".search-result-item");
        if (items.length === 0) return;

        if (e.key === "ArrowDown") {
            e.preventDefault();
            activeIndex = (activeIndex + 1) % items.length;
            updateActiveItem(items);
        } else if (e.key === "ArrowUp") {
            e.preventDefault();
            activeIndex = (activeIndex - 1 + items.length) % items.length;
            updateActiveItem(items);
        } else if (e.key === "Enter") {
            e.preventDefault();
            if (activeIndex >= 0 && activeIndex < items.length) {
                items[activeIndex].click();
            } else if (items.length > 0) {
                items[0].click();
            }
        } else if (e.key === "Escape") {
            resultsDropdown.style.display = "none";
        }
    });

    function updateActiveItem(items) {
        items.forEach((item, idx) => {
            if (idx === activeIndex) {
                item.classList.add("active-item");
                item.scrollIntoView({ block: "nearest" });
            } else {
                item.classList.remove("active-item");
            }
        });
    }

    if (clearBtn) {
        clearBtn.addEventListener("click", () => {
            searchInput.value = "";
            clearBtn.style.display = "none";
            resultsDropdown.style.display = "none";
            searchInput.focus();
        });
    }

    // Close dropdown on click outside
    document.addEventListener("click", (e) => {
        if (!searchInput.contains(e.target) && !resultsDropdown.contains(e.target)) {
            resultsDropdown.style.display = "none";
        }
    });
}

export function updateMapTheme(theme) {
    if (!tileLayer || !mapInstance) return;
    const isDark = theme === "dark";
    tileLayer.setUrl(isDark ? DARK_TILES : LIGHT_TILES);
}
