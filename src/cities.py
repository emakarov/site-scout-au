"""Study-area definitions. Adding a market = adding one entry here."""

CITIES = {
    "melbourne": {
        "label": "Greater Melbourne",
        # lon_min, lat_min, lon_max, lat_max
        "bbox": (144.55, -38.20, 145.55, -37.50),
        "cbd": (-37.8136, 144.9631),  # Melbourne GPO
        "zoom": 11,
    },
    "sydney": {
        "label": "Greater Sydney",
        "bbox": (150.60, -34.20, 151.40, -33.55),
        "cbd": (-33.8688, 151.2093),  # Sydney Town Hall
        "zoom": 11,
    },
}
