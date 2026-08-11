"""Interactive deliverable: a folium map of scores, archetypes and top sites."""

import branca.colormap as cm
import folium
import h3
import pandas as pd

ARCHETYPE_COLORS = {
    "CBD & Major Activity Centres": "#7b3294",
    "High-Street / Café Strip": "#008837",
    "Middle-Ring Mixed Use": "#a6dba0",
    "Local Neighbourhood Centre": "#fdb863",
    "Quiet Residential": "#d9d9d9",
}


def hex_geojson(row: pd.Series) -> dict:
    boundary = h3.cell_to_boundary(row["hex"])
    return {
        "type": "Polygon",
        "coordinates": [[[lng, lat] for lat, lng in boundary] +
                        [[boundary[0][1], boundary[0][0]]]],
    }


def tooltip_html(row: pd.Series) -> str:
    anchors = f"<br><i>{row['anchors']}</i>" if row["anchors"] else ""
    return (
        f"<b>{row['suburb']}</b> — {row['archetype']}<br>"
        f"Suitability: <b>{row['suitability']}</b> / 100 · "
        f"Opportunity: <b>{row['opportunity']}</b><br>"
        f"Bubble tea nearby: {int(row['c_bubble_tea'])} · "
        f"Cafés: {int(row['n_cafe'])} · Asian dining: {int(row['n_asian_dining'])} · "
        f"Retail: {int(row['n_retail'])}{anchors}"
    )


def build_map(hexes: pd.DataFrame, pois: pd.DataFrame, top: pd.DataFrame,
              city_cfg: dict) -> folium.Map:
    m = folium.Map(location=list(city_cfg["cbd"]),
                   zoom_start=city_cfg["zoom"],
                   tiles="cartodbpositron", prefer_canvas=True)

    colormap = cm.LinearColormap(
        ["#2c7fb8", "#7fcdbb", "#edf8b1", "#fec44f", "#d95f0e"],
        vmin=0, vmax=100, caption="Suitability score (0–100)",
    )
    colormap.add_to(m)

    # Layer 1 — suitability choropleth
    fg_score = folium.FeatureGroup(name="Suitability score", show=True)
    for _, row in hexes.iterrows():
        folium.GeoJson(
            {"type": "Feature", "geometry": hex_geojson(row), "properties": {}},
            style_function=(lambda _, c=colormap(row["suitability"]): {
                "fillColor": c, "color": "#666", "weight": 0.3,
                "fillOpacity": 0.55,
            }),
            tooltip=tooltip_html(row),
        ).add_to(fg_score)
    fg_score.add_to(m)

    # Layer 2 — archetypes (hidden by default)
    fg_arch = folium.FeatureGroup(name="Location archetypes (KMeans)", show=False)
    for _, row in hexes.iterrows():
        folium.GeoJson(
            {"type": "Feature", "geometry": hex_geojson(row), "properties": {}},
            style_function=(lambda _, c=ARCHETYPE_COLORS.get(
                row["archetype"], "#cccccc"): {
                "fillColor": c, "color": "#666", "weight": 0.3,
                "fillOpacity": 0.6,
            }),
            tooltip=tooltip_html(row),
        ).add_to(fg_arch)
    fg_arch.add_to(m)

    # Layer 3 — existing competitors
    fg_comp = folium.FeatureGroup(name="Existing bubble tea stores", show=True)
    stores = pois[pois["category"] == "bubble_tea"]
    for _, s in stores.iterrows():
        folium.CircleMarker(
            location=[s["lat"], s["lon"]], radius=3, color="#5e3c99",
            fill=True, fill_opacity=0.9, weight=1,
            tooltip=s["name"] or "Bubble tea store",
        ).add_to(fg_comp)
    fg_comp.add_to(m)

    # Layer 4 — recommended sites
    fg_top = folium.FeatureGroup(name="Top recommended sites", show=True)
    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        anchors = f"<br><i>Near: {row['anchors']}</i>" if row["anchors"] else ""
        folium.Marker(
            location=[row["lat"], row["lon"]],
            icon=folium.Icon(color="red", icon="star"),
            popup=folium.Popup(
                f"<b>#{rank} — {row['suburb']}</b><br>"
                f"{row['archetype']}<br>"
                f"Suitability {row['suitability']} · "
                f"Opportunity {row['opportunity']}{anchors}",
                max_width=280,
            ),
            tooltip=f"#{rank} {row['suburb']}",
        ).add_to(fg_top)
    fg_top.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    title = (
        '<div style="position:fixed;top:10px;left:50px;z-index:9999;'
        'background:rgba(255,255,255,.95);padding:10px 16px;border-radius:8px;'
        'box-shadow:0 1px 6px rgba(0,0,0,.3);font-family:sans-serif;">'
        f"<b>Milk Tea Location Scoring — {city_cfg['label']} (demo)</b><br>"
        '<span style="font-size:12px;">Overture Maps data · H3 grid · '
        "KMeans archetypes · RandomForest suitability · red stars = "
        "recommended underserved sites</span></div>"
    )
    m.get_root().html.add_child(folium.Element(title))
    return m
