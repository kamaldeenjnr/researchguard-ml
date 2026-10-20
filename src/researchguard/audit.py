from __future__ import annotations

import re

import pandas as pd
from sklearn.metrics import roc_auc_score

ID_NAME = re.compile(r"(^id$|_id$|^id_|uuid|identifier)", re.I)


def _warn(code: str, severity: str, details: dict, needs_confirmation: bool = False) -> dict:
    return {
        "code": code,
        "severity": severity,
        "details": details,
        "needs_user_confirmation": needs_confirmation,
    }


def audit_dataset(df: pd.DataFrame, target: str | None = None) -> dict:
    """Return basic, transparent dataset-quality checks.

    Detects observable problems only. It cannot prove the absence of
    leakage, bias or measurement error.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("The dataset has no rows.")
    if df.columns.has_duplicates:
        raise ValueError("The dataset contains duplicate column names.")
    if target is not None and target not in df.columns:
        raise ValueError(f"Target column '{target}' was not found.")

    warnings: list[dict] = []

    missing = df.isna().sum()
    missing = {str(k): int(v) for k, v in missing[missing > 0].items()}
    if missing:
        warnings.append(_warn("MISSING_VALUES", "warning", missing))

    duplicate_rows = int(df.duplicated().sum())
    if duplicate_rows:
        warnings.append(_warn("DUPLICATE_ROWS", "warning", {"count": duplicate_rows}))

    constant_columns = [str(c) for c in df.columns if df[c].nunique(dropna=False) <= 1]
    if constant_columns:
        warnings.append(_warn("CONSTANT_COLUMNS", "warning", {"columns": constant_columns}))

    id_like = [
        str(c) for c in df.columns
        if c != target and (
            ID_NAME.search(str(c))
            or (df[c].nunique() == len(df) and not pd.api.types.is_float_dtype(df[c]))
        )
    ]
    if id_like:
        warnings.append(_warn(
            "IDENTIFIER_COLUMNS", "warning", {"columns": id_like}, needs_confirmation=True))

    target_summary = None
    if target is not None:
        counts = df[target].value_counts(dropna=True)
        n_target_missing = int(df[target].isna().sum())
        minority = int(counts.min()) if len(counts) else 0
        target_summary = {
            "counts": {str(k): int(v) for k, v in counts.items()},
            "n_classes": int(len(counts)),
            "is_binary": len(counts) == 2,
            "minority_fraction": float(minority / counts.sum()) if counts.sum() else None,
            "n_missing": n_target_missing,
        }
        if n_target_missing:
            warnings.append(_warn("TARGET_MISSING", "error", {"count": n_target_missing}))
        if len(counts) != 2:
            warnings.append(_warn(
                "TARGET_NOT_BINARY", "error", {"n_classes": int(len(counts))}))
        elif target_summary["minority_fraction"] < 0.10:
            warnings.append(_warn(
                "CLASS_IMBALANCE", "warning",
                {"minority_fraction": target_summary["minority_fraction"]}))
        if len(counts) == 2 and minority < 30:
            warnings.append(_warn("SMALL_MINORITY_CLASS", "warning", {"count": minority}))

        # Possible leakage: a single numeric feature that almost perfectly separates the classes.
        if len(counts) == 2:
            suspects = {}
            y = df[target]
            for col in df.columns:
                if col == target or not pd.api.types.is_numeric_dtype(df[col]):
                    continue
                mask = df[col].notna() & y.notna()
                if mask.sum() < 20 or df.loc[mask, col].nunique() < 2:
                    continue
                try:
                    auc = roc_auc_score(y[mask], df.loc[mask, col])
                except ValueError:
                    continue
                if max(auc, 1 - auc) >= 0.98:
                    suspects[str(col)] = round(float(max(auc, 1 - auc)), 4)
            if suspects:
                warnings.append(_warn(
                    "POSSIBLE_LEAKAGE", "warning", suspects, needs_confirmation=True))

    return {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "missing_values": missing,
        "duplicate_rows": duplicate_rows,
        "constant_columns": constant_columns,
        "target_summary": target_summary,
        "warnings": warnings,
    }
