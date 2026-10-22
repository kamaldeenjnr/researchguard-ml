"""Leakage-resistant preprocessing: every transform lives inside the CV pipeline,
so imputation/scaling/encoding statistics are learned from training folds only."""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


def split_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = [c for c in X.columns
               if pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c])]
    categorical = [c for c in X.columns if c not in numeric]
    return numeric, categorical


def build_preprocessor(X: pd.DataFrame, scale: bool = True,
                       encoding: str = "onehot") -> ColumnTransformer:
    """encoding="onehot" (sparse when wide) for linear models and random forests;
    encoding="ordinal" (always dense, one column per variable) for histogram gradient
    boosting, which does not accept sparse input and would need gigabytes for a dense
    one-hot matrix of high-cardinality codes such as diagnoses."""
    numeric, categorical = split_columns(X)
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    if encoding == "onehot":
        encoder = OneHotEncoder(handle_unknown="ignore")
    elif encoding == "ordinal":
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    else:
        raise ValueError(f"Unknown encoding: {encoding}")
    return ColumnTransformer([
        ("num", Pipeline(num_steps), numeric),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", encoder),
        ]), categorical),
    ], sparse_threshold=0 if encoding == "ordinal" else 0.3)
