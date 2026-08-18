// Leaflet Interactive World Map Module for Global Capitals
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

    // Populate City Search Dropdown & Render Markers
    populateSearchDropdown(onCitySelectCallback);
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
            icon: createCustomPin(cap.name === "Lahore")
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

        if (cap.name === "Lahore") {
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

    const searchSelect = document.getElementById("capital-select");
    if (searchSelect) {
        searchSelect.value = `${cap.lat},${cap.lon},${cap.name}, ${cap.country}`;
    }

    const mapTitleEl = document.getElementById("map-selected-city");
    if (mapTitleEl) {
        mapTitleEl.innerText = `${cap.name}, ${cap.country}`;
    }

    if (typeof onCitySelectCallback === "function") {
        onCitySelectCallback(cap);
    }
}

function populateSearchDropdown(onCitySelectCallback) {
    const searchSelect = document.getElementById("capital-select");
    if (!searchSelect) return;

    searchSelect.innerHTML = "";

    // Group capitals by region
    const regions = {};
    GLOBAL_CAPITALS.forEach(c => {
        if (!regions[c.region]) regions[c.region] = [];
        regions[c.region].push(c);
    });

    for (const [region, cities] of Object.entries(regions)) {
        const optgroup = document.createElement("optgroup");
        optgroup.label = region;
        cities.forEach(c => {
            const opt = document.createElement("option");
            opt.value = `${c.lat},${c.lon},${c.name}, ${c.country}`;
            opt.innerText = `${c.name} (${c.country})`;
            if (c.name === "Lahore") opt.selected = true;
            optgroup.appendChild(opt);
        });
        searchSelect.appendChild(optgroup);
    }

    searchSelect.addEventListener("change", (e) => {
        const [lat, lon, name, country] = e.target.value.split(",");
        const cap = {
            lat: parseFloat(lat),
            lon: parseFloat(lon),
            name: name.trim(),
            country: country ? country.trim() : ""
        };
        selectCityMarker(cap, onCitySelectCallback);
    });
}

export function updateMapTheme(theme) {
    if (!tileLayer || !mapInstance) return;
    const isDark = theme === "dark";
    tileLayer.setUrl(isDark ? DARK_TILES : LIGHT_TILES);
}
