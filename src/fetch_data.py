"""Fetch input data from Overture Maps (free, public S3 bucket — no API keys).

Downloads three datasets per city (see src/cities.py):
  data/<city>_pois.parquet             — points of interest (competitors + demand drivers)
  data/<city>_localities.parquet       — suburb boundary polygons (for readable reporting)
  data/<city>_locality_points.parquet  — suburb name points (fallback where
                                         Overture has no boundary polygon)

Usage: python -m src.fetch_data [melbourne|sydney]
"""

import sys

import duckdb

from .cities import CITIES

OVERTURE_RELEASE = "2026-07-22.0"
S3_ROOT = f"s3://overturemaps-us-west-2/release/{OVERTURE_RELEASE}"

POI_CATEGORIES = (
    "bubble_tea", "cafe", "coffee_shop", "desserts", "smoothie_juice_bar",
    "ice_cream_shop", "bakery", "fast_food_restaurant", "shopping",
    "shopping_center", "supermarket", "grocery_store", "department_store",
    "train_station", "bus_station", "transportation", "college_university",
    "high_school", "private_school", "language_school",
    "vocational_and_technical_school", "gym", "movie_theater", "cinema", "hotel",
)


def fetch(con: duckdb.DuckDBPyConnection, city: str) -> None:
    lon_min, lat_min, lon_max, lat_max = CITIES[city]["bbox"]
    cats = ", ".join(f"'{c}'" for c in POI_CATEGORIES)

    con.execute(f"""
        COPY (
          SELECT id,
                 names.primary AS name,
                 categories.primary AS category,
                 confidence,
                 ST_X(geometry) AS lon,
                 ST_Y(geometry) AS lat
          FROM read_parquet('{S3_ROOT}/theme=places/type=place/*', hive_partitioning=1)
          WHERE bbox.xmin > {lon_min} AND bbox.xmax < {lon_max}
            AND bbox.ymin > {lat_min} AND bbox.ymax < {lat_max}
            AND (categories.primary IN ({cats})
                 OR categories.primary LIKE '%_restaurant'
                 OR categories.primary = 'restaurant')
            AND confidence >= 0.3
        ) TO 'data/{city}_pois.parquet' (FORMAT parquet)
    """)

    con.execute(f"""
        COPY (
          SELECT names.primary AS suburb,
                 ST_AsText(geometry) AS wkt
          FROM read_parquet('{S3_ROOT}/theme=divisions/type=division_area/*', hive_partitioning=1)
          WHERE bbox.xmin < {lon_max} AND bbox.xmax > {lon_min}
            AND bbox.ymin < {lat_max} AND bbox.ymax > {lat_min}
            AND subtype = 'locality'
            AND country = 'AU'
        ) TO 'data/{city}_localities.parquet' (FORMAT parquet)
    """)

    con.execute(f"""
        COPY (
          SELECT names.primary AS suburb, subtype,
                 ST_X(geometry) AS lon,
                 ST_Y(geometry) AS lat
          FROM read_parquet('{S3_ROOT}/theme=divisions/type=division/*', hive_partitioning=1)
          WHERE bbox.xmin > {lon_min} AND bbox.xmax < {lon_max}
            AND bbox.ymin > {lat_min} AND bbox.ymax < {lat_max}
            AND subtype IN ('locality', 'macrohood', 'neighborhood')
            AND country = 'AU'
        ) TO 'data/{city}_locality_points.parquet' (FORMAT parquet)
    """)


def main() -> None:
    city = sys.argv[1] if len(sys.argv) > 1 else "melbourne"
    if city not in CITIES:
        sys.exit(f"Unknown city {city!r}; choose from {list(CITIES)}")
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial; INSTALL httpfs; LOAD httpfs;")
    con.execute("CREATE SECRET (TYPE S3, PROVIDER config, REGION 'us-west-2');")
    con.execute("SET geometry_always_xy = true;")
    fetch(con, city)
    n_pois = con.execute(f"SELECT count(*) FROM 'data/{city}_pois.parquet'").fetchone()[0]
    n_suburbs = con.execute(f"SELECT count(*) FROM 'data/{city}_localities.parquet'").fetchone()[0]
    print(f"{CITIES[city]['label']}: fetched {n_pois:,} POIs and {n_suburbs} suburb boundaries.")


if __name__ == "__main__":
    main()
