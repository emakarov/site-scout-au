"""Run the full demo pipeline: features → model → map + shortlist CSV.

Usage: python run_demo.py [melbourne|sydney]

Assumes data/ already contains the Overture extracts for that city
(run `python -m src.fetch_data <city>` first to refresh them).
"""

import sys

from src.cities import CITIES
from src.features import load_features
from src.make_map import build_map
from src.model import run


def main() -> None:
    city = sys.argv[1] if len(sys.argv) > 1 else "melbourne"
    if city not in CITIES:
        sys.exit(f"Unknown city {city!r}; choose from {list(CITIES)}")
    cfg = CITIES[city]

    hexes, pois = load_features(city)
    print(f"{cfg['label']}: {len(hexes):,} active hex cells · "
          f"{int(hexes['n_bubble_tea'].sum())} existing bubble tea stores")

    hexes, top, report = run(hexes)

    m = report["metrics"]
    print(f"\nRandomForest (5-fold out-of-fold): "
          f"ROC-AUC {m['roc_auc']:.3f} · avg precision {m['avg_precision']:.3f} "
          f"(base rate {m['base_rate']:.3f})")
    print("\nTop feature importances:")
    print(report["importances"].head(8).round(3).to_string())

    print("\nArchetype mix:")
    print(hexes["archetype"].value_counts().to_string())

    cols = ["suburb", "archetype", "suitability", "opportunity",
            "n_cafe", "n_asian_dining", "n_retail", "n_transit",
            "c_bubble_tea", "anchors", "lat", "lon"]
    top[cols].to_csv(f"docs/{city}_top_locations.csv", index=False)

    print("\nTop recommended sites:")
    print(top[["suburb", "archetype", "suitability", "opportunity"]]
          .to_string(index=False))

    build_map(hexes, pois, top, cfg).save(f"docs/{city}_milk_tea_map.html")
    print(f"\nWrote docs/{city}_milk_tea_map.html "
          f"and docs/{city}_top_locations.csv")


if __name__ == "__main__":
    main()
