"""Modelling: KMeans location archetypes + RandomForest suitability scoring.

Approach (revealed-demand):
  1. KMeans clusters every hex into an interpretable "location archetype"
     (CBD core, suburban retail hub, quiet residential, ...).
  2. A RandomForest learns what separates hexes that already sustain a bubble
     tea store from those that don't — using only demand-side features, never
     the bubble tea counts themselves.
  3. Suitability = out-of-fold predicted probability (no hex is scored by a
     model that saw it during training).
  4. Opportunity = suitability discounted by existing nearby supply →
     high-potential, underserved whitespace.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLS

RANDOM_STATE = 42
N_CLUSTERS = 5


def add_archetypes(hexes: pd.DataFrame) -> pd.DataFrame:
    hexes = hexes.copy()
    X = StandardScaler().fit_transform(np.log1p(hexes[FEATURE_COLS]))
    km = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=RANDOM_STATE)
    hexes["cluster"] = km.fit_predict(X)

    # Name clusters by commercial intensity (most active first). Names are
    # descriptive labels for the ranked profiles, not hand-tuned rules.
    profile = hexes.groupby("cluster")["total_poi"].mean()
    order = profile.sort_values(ascending=False).index.tolist()
    labels = [
        "CBD & Major Activity Centres",
        "High-Street / Café Strip",
        "Middle-Ring Mixed Use",
        "Local Neighbourhood Centre",
        "Quiet Residential",
    ]
    names = {c: labels[rank] for rank, c in enumerate(order)}
    hexes["archetype"] = hexes["cluster"].map(names)
    return hexes


def add_scores(hexes: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    hexes = hexes.copy()
    y = (hexes["n_bubble_tea"] > 0).astype(int)
    X = np.log1p(hexes[FEATURE_COLS])

    rf = RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    proba = cross_val_predict(rf, X, y, cv=cv, method="predict_proba")[:, 1]

    metrics = {
        "roc_auc": roc_auc_score(y, proba),
        "avg_precision": average_precision_score(y, proba),
        "base_rate": y.mean(),
        "n_cells": len(y),
        "n_positive": int(y.sum()),
    }

    rf.fit(X, y)  # full fit only for feature importances
    importances = (
        pd.Series(rf.feature_importances_, index=FEATURE_COLS)
        .sort_values(ascending=False)
    )

    hexes["suitability"] = (proba * 100).round(1)
    # Discount by existing supply in the catchment: whitespace bonus.
    hexes["opportunity"] = (
        hexes["suitability"] / (1 + hexes["c_bubble_tea"])
    ).round(1)
    return hexes, {"metrics": metrics, "importances": importances}


def top_opportunities(hexes: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    """Best underserved cells: no store in the hex itself, strong suitability."""
    cand = hexes[hexes["n_bubble_tea"] == 0].sort_values(
        "opportunity", ascending=False
    )
    # At most 2 picks per suburb so the shortlist spans the metro area.
    cand = cand.groupby("suburb").head(2)
    return cand.head(n)


def run(hexes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    hexes = add_archetypes(hexes)
    hexes, report = add_scores(hexes)
    top = top_opportunities(hexes)
    return hexes, top, report
