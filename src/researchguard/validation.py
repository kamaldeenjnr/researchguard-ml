"""Design-aware validation: choose a splitting strategy, then verify it.

Principle: random splitting is only appropriate when rows are independent.
Repeated participants, clusters (hospitals, communities) and time ordering
violate that assumption and inflate performance estimates.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    TimeSeriesSplit,
)

GROUP_NAME = re.compile(
    r"(patient|participant|subject|person|household|hospital|clinic|facility|"
    r"site|community|village|district|school|cluster|(^id$)|(_id$))", re.I)
TIME_NAME = re.compile(r"(date|time|year|month|visit|timestamp)", re.I)


@dataclass
class ValidationPlan:
    strategy: str  # stratified_kfold | stratified_group_kfold | time_series_split
    n_splits: int
    seed: int = 42
    group_col: str | None = None
    time_col: str | None = None
    reasons: list[str] = field(default_factory=list)
    warnings: list[dict] = field(default_factory=list)

    def split(self, df: pd.DataFrame, target: str) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield (train_idx, test_idx) as positional indices into df."""
        df = df.reset_index(drop=True)
        y = df[target].to_numpy()
        X = np.zeros((len(df), 1))
        if self.strategy == "stratified_kfold":
            yield from StratifiedKFold(
                self.n_splits, shuffle=True, random_state=self.seed).split(X, y)
        elif self.strategy == "stratified_group_kfold":
            yield from StratifiedGroupKFold(
                self.n_splits, shuffle=True, random_state=self.seed
            ).split(X, y, df[self.group_col].to_numpy())
        elif self.strategy == "time_series_split":
            order = np.argsort(df[self.time_col].to_numpy(), kind="stable")
            for tr, te in TimeSeriesSplit(self.n_splits).split(X):
                yield order[tr], order[te]
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")


def _warn(code: str, details: dict, severity: str = "warning") -> dict:
    return {"code": code, "severity": severity, "details": details,
            "needs_user_confirmation": code.startswith("UNCONFIRMED")}


def select_validation(
    df: pd.DataFrame,
    target: str,
    group_col: str | None = None,
    time_col: str | None = None,
    n_splits: int = 5,
    seed: int = 42,
) -> ValidationPlan:
    """Choose a validation strategy from declared study design.

    Structure the user declares is respected. Structure that is merely
    *suspected* from column names produces an UNCONFIRMED_* warning and is
    never silently acted on, because names alone cannot establish design.
    """
    for name, col in (("target", target), ("group_col", group_col), ("time_col", time_col)):
        if col is not None and col not in df.columns:
            raise ValueError(f"{name} '{col}' was not found in the dataset.")
    if df[target].isna().any():
        raise ValueError("Target contains missing values; handle them before validation.")
    if df[target].nunique() != 2:
        raise ValueError("Only binary classification is supported in this version.")
    if group_col and time_col:
        raise NotImplementedError(
            "Combined group + temporal validation is not supported in v0.1.")
    for col in (group_col, time_col):
        if col is not None and df[col].isna().any():
            raise ValueError(f"Column '{col}' contains missing values.")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    reasons: list[str] = []
    warnings: list[dict] = []

    minority = int(df[target].value_counts().min())
    if minority < n_splits:
        if minority < 2:
            raise ValueError("The minority class has fewer than 2 rows; cannot validate.")
        warnings.append(_warn("N_SPLITS_REDUCED",
                              {"requested": n_splits, "used": minority}))
        n_splits = minority

    if group_col:
        n_groups = int(df[group_col].nunique())
        if n_groups < n_splits:
            raise ValueError(
                f"Only {n_groups} groups in '{group_col}', fewer than n_splits={n_splits}.")
        strategy = "stratified_group_kfold"
        reasons.append(f"Rows share '{group_col}', so whole groups are held out together.")
    elif time_col:
        strategy = "time_series_split"
        reasons.append(f"'{time_col}' declared: models are always tested on later data.")
    else:
        strategy = "stratified_kfold"
        reasons.append("No group or time structure declared; assuming independent rows.")
        suspects = [
            str(c) for c in df.columns
            if c != target and GROUP_NAME.search(str(c)) and df[c].nunique() < len(df)
        ]
        if suspects:
            warnings.append(_warn("UNCONFIRMED_GROUP_RISK", {"columns": suspects}))
        time_suspects = [
            str(c) for c in df.columns
            if c != target and (pd.api.types.is_datetime64_any_dtype(df[c])
                                or TIME_NAME.search(str(c)))
        ]
        if time_suspects:
            warnings.append(_warn("UNCONFIRMED_TIME_RISK", {"columns": time_suspects}))

    return ValidationPlan(strategy, n_splits, seed, group_col, time_col, reasons, warnings)


def verify_plan(df: pd.DataFrame, target: str, plan: ValidationPlan) -> dict:
    """Check that the generated folds actually honour the plan."""
    df = df.reset_index(drop=True)
    feature_cols = [c for c in df.columns if c != target]
    hashes = pd.util.hash_pandas_object(df[feature_cols], index=False).to_numpy()

    group_overlap = time_violations = dup_leak = folds_missing_class = 0
    n_folds = 0
    for tr, te in plan.split(df, target):
        n_folds += 1
        if plan.group_col:
            group_overlap += len(set(df.loc[tr, plan.group_col]) & set(df.loc[te, plan.group_col]))
        if plan.time_col:
            if df.loc[tr, plan.time_col].max() > df.loc[te, plan.time_col].min():
                time_violations += 1
        dup_leak += int(np.isin(hashes[te], hashes[tr]).sum())
        if df.loc[te, target].nunique() < 2 or df.loc[tr, target].nunique() < 2:
            folds_missing_class += 1

    return {
        "n_folds": n_folds,
        "group_overlap": group_overlap,
        "time_order_violations": time_violations,
        "test_rows_duplicated_in_train": dup_leak,
        "folds_missing_a_class": folds_missing_class,
        "passed": not (group_overlap or time_violations or folds_missing_class),
    }
