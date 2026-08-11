"""Feature engineering: aggregate POIs onto an H3 hex grid.

Each hex (resolution 8, ~0.74 km²) becomes one row with:
  - in-hex counts per POI group (cafes, Asian dining, retail anchors, transit, ...)
  - catchment counts (hex + its 6 neighbours, ~2 km walkshed proxy)
  - distance to the Melbourne CBD
  - target: number of existing bubble tea stores

Suburb names are attached from locality polygons for readable reporting.
"""

import math

import h3
import numpy as np
import pandas as pd
from shapely import STRtree, wkt
from shapely.geometry import Point

from .cities import CITIES

H3_RES = 8
MIN_ACTIVITY = 3  # drop hexes with fewer total POIs (parkland, water, farmland)

ASIAN_DINING = {
    "chinese_restaurant", "asian_restaurant", "japanese_restaurant",
    "korean_restaurant", "vietnamese_restaurant", "thai_restaurant",
    "sushi_restaurant", "malaysian_restaurant", "ramen_restaurant",
    "taiwanese_restaurant", "asian_fusion_restaurant", "dim_sum_restaurant",
    "hot_pot_restaurant", "indonesian_restaurant", "filipino_restaurant",
}

GROUPS = {
    "cafe": {"cafe", "coffee_shop"},
    "dessert": {"desserts", "smoothie_juice_bar", "ice_cream_shop", "bakery"},
    "fast_food": {"fast_food_restaurant"},
    "retail": {"shopping", "shopping_center", "supermarket", "grocery_store",
               "department_store"},
    "transit": {"train_station", "bus_station", "transportation"},
    "education": {"college_university", "high_school", "private_school",
                  "language_school", "vocational_and_technical_school"},
    "leisure": {"gym", "movie_theater", "cinema", "hotel"},
}

FEATURE_COLS = [
    "n_cafe", "n_dessert", "n_fast_food", "n_restaurant", "n_asian_dining",
    "n_retail", "n_transit", "n_education", "n_leisure",
    "c_cafe", "c_dessert", "c_fast_food", "c_restaurant", "c_asian_dining",
    "c_retail", "c_transit", "c_education", "c_leisure",
    "dist_cbd_km",
]


def classify(category: str) -> str | None:
    if category == "bubble_tea":
        return "bubble_tea"
    if category in ASIAN_DINING:
        return "asian_dining"
    for group, cats in GROUPS.items():
        if category in cats:
            return group
    if category == "restaurant" or category.endswith("_restaurant"):
        return "restaurant"
    return None


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def build_hex_features(pois: pd.DataFrame, cbd: tuple[float, float]) -> pd.DataFrame:
    pois = pois.copy()
    pois["group"] = pois["category"].map(classify)
    pois = pois.dropna(subset=["group"])
    pois["hex"] = [
        h3.latlng_to_cell(lat, lon, H3_RES)
        for lat, lon in zip(pois["lat"], pois["lon"])
    ]

    counts = (
        pois.pivot_table(index="hex", columns="group", values="id",
                         aggfunc="count", fill_value=0)
        .rename(columns=lambda g: f"n_{g}")
    )
    all_groups = ["bubble_tea", "asian_dining", "restaurant", *GROUPS]
    for g in all_groups:
        counts[f"n_{g}"] = counts.get(f"n_{g}", 0)

    n_cols = [f"n_{g}" for g in all_groups]
    counts["total_poi"] = counts[n_cols].sum(axis=1)
    counts = counts[counts["total_poi"] >= MIN_ACTIVITY]

    # Catchment counts: hex + immediate neighbours (~2 km across).
    lookup = counts[n_cols].to_dict("index")
    catchment = {
        h: [
            sum(lookup.get(n, {}).get(col, 0) for n in h3.grid_disk(h, 1))
            for col in n_cols
        ]
        for h in counts.index
    }
    c_cols = [c.replace("n_", "c_", 1) for c in n_cols]
    counts[c_cols] = pd.DataFrame.from_dict(catchment, orient="index",
                                            columns=c_cols)

    centers = [h3.cell_to_latlng(h) for h in counts.index]
    counts["lat"] = [c[0] for c in centers]
    counts["lon"] = [c[1] for c in centers]
    counts["dist_cbd_km"] = [
        haversine_km(lat, lon, *cbd) for lat, lon in centers
    ]
    return counts.reset_index().rename(columns={"index": "hex"})


MAX_SUBURB_AREA = 0.01  # sq. degrees; drops metro-scale "Melbourne" polygon


def attach_suburbs(hexes: pd.DataFrame, localities: pd.DataFrame,
                   locality_points: pd.DataFrame) -> pd.DataFrame:
    """Suburb per hex: exact polygon match where Overture has a boundary,
    nearest named locality/macrohood point otherwise (inner-Melbourne suburbs
    like Prahran exist only as points in Overture)."""
    keep = [(wkt.loads(w), s) for w, s in
            zip(localities["wkt"], localities["suburb"])]
    keep = [(p, s) for p, s in keep if p.area <= MAX_SUBURB_AREA]
    polys = [p for p, _ in keep]
    poly_names = [s for _, s in keep]
    tree = STRtree(polys)

    pts = [Point(lon, lat) for lon, lat in
           zip(locality_points["lon"], locality_points["lat"])]
    pt_names = locality_points["suburb"].tolist()
    pt_tree = STRtree(pts)

    suburbs = []
    for lon, lat in zip(hexes["lon"], hexes["lat"]):
        pt = Point(lon, lat)
        idx = tree.query(pt, predicate="within")
        if len(idx):
            best = min(idx, key=lambda i: polys[i].area)
            suburbs.append(poly_names[best])
        else:
            suburbs.append(pt_names[pt_tree.nearest(pt)])
    hexes = hexes.copy()
    hexes["suburb"] = suburbs
    return hexes


def anchor_names(pois: pd.DataFrame, top_n: int = 3) -> pd.Series:
    """Best-known named anchors per hex (for map popups)."""
    anchors = pois[pois["category"].isin(
        {"shopping_center", "shopping", "train_station", "college_university",
         "supermarket", "department_store"}
    )].copy()
    anchors = anchors.dropna(subset=["name"])
    anchors["hex"] = [
        h3.latlng_to_cell(lat, lon, H3_RES)
        for lat, lon in zip(anchors["lat"], anchors["lon"])
    ]
    anchors = anchors.sort_values("confidence", ascending=False)
    return anchors.groupby("hex")["name"].agg(
        lambda s: ", ".join(s.drop_duplicates().head(top_n))
    )


def load_features(city: str = "melbourne") -> tuple[pd.DataFrame, pd.DataFrame]:
    pois = pd.read_parquet(f"data/{city}_pois.parquet")
    localities = pd.read_parquet(f"data/{city}_localities.parquet")
    locality_points = pd.read_parquet(f"data/{city}_locality_points.parquet")

    hexes = build_hex_features(pois, CITIES[city]["cbd"])
    hexes = attach_suburbs(hexes, localities, locality_points)
    hexes = hexes.merge(anchor_names(pois).rename("anchors"),
                        on="hex", how="left")
    hexes["anchors"] = hexes["anchors"].fillna("")
    return hexes, pois


if __name__ == "__main__":
    hexes, _ = load_features()
    print(f"{len(hexes):,} active hex cells, "
          f"{int(hexes['n_bubble_tea'].sum())} bubble tea stores, "
          f"{(hexes['n_bubble_tea'] > 0).mean():.1%} of cells have at least one.")
