from __future__ import annotations

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .preprocessing import build_preprocessor


def get_models(X: pd.DataFrame, seed: int = 42) -> dict[str, Pipeline]:
    """Baseline estimators, each wrapped with its own preprocessing pipeline."""
    return {
        "logistic_regression": Pipeline([
            ("prep", build_preprocessor(X, scale=True)),
            ("model", LogisticRegression(max_iter=1000)),
        ]),
        "random_forest": Pipeline([
            ("prep", build_preprocessor(X, scale=False)),
            ("model", RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)),
        ]),
        "gradient_boosting": Pipeline([
            ("prep", build_preprocessor(X, scale=False)),
            ("model", HistGradientBoostingClassifier(random_state=seed)),
        ]),
    }
