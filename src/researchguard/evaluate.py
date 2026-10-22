"""Cross-validated model evaluation and the naive-vs-design-aware comparison."""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
from sklearn.base import clone

from .calibration import calibration_summary
from .metrics import compute_metrics
from .models import get_models
from .subgroup import subgroup_metrics
from .validation import ValidationPlan, select_validation, verify_plan

METRIC_KEYS = ["roc_auc", "pr_auc", "brier", "log_loss", "accuracy"]


def _prepare(df, target, group_col, time_col, drop_columns, positive_class=None):
    df = df.reset_index(drop=True)
    classes = sorted(df[target].unique(), key=str)
    if len(classes) != 2:
        raise ValueError("Target must be binary.")
    if positive_class is not None:
        match = [c for c in classes if str(c) == str(positive_class)]
        if not match:
            raise ValueError(f"positive_class '{positive_class}' is not one of {[str(c) for c in classes]}.")
        classes = [c for c in classes if str(c) != str(positive_class)] + match
    y = (df[target] == classes[1]).astype(int)
    excluded = {target, group_col, time_col, *(drop_columns or [])}
    features = [c for c in df.columns if c not in excluded]
    if not features:
        raise ValueError("No feature columns remain after exclusions.")
    work = df.copy()
    work[target] = y
    return work, features, classes


def evaluate_models(df, target, plan: ValidationPlan, subgroup_cols=None,
                    drop_columns=None, seed: int = 42, models=None, positive_class=None) -> dict:
    """Fit every model inside each fold; return fold metrics, calibration, subgroups.

    positive_class: the outcome value treated as 1. Defaults to the second value in
    sorted order (e.g. 1 for 0/1, "yes" for no/yes); set it explicitly for anything else.
    """
    work, features, classes = _prepare(df, target, plan.group_col, plan.time_col, drop_columns,
                                       positive_class)
    X, y = work[features], work[target].to_numpy()
    models = models or get_models(X, seed)
    folds = list(plan.split(work, target))
    out = {}
    for name, pipe in models.items():
        fold_rows, idx_all, p_all = [], [], []
        for tr, te in folds:
            m = clone(pipe).fit(X.iloc[tr], y[tr])
            p = m.predict_proba(X.iloc[te])[:, 1]
            fold_rows.append(compute_metrics(y[te], p))
            idx_all.append(te)
            p_all.append(p)
        idx, p = np.concatenate(idx_all), np.concatenate(p_all)
        summary = {}
        for k in METRIC_KEYS:
            vals = np.array([r[k] for r in fold_rows], dtype=float)
            ok = vals[~np.isnan(vals)]
            summary[k] = {"mean": float(ok.mean()) if len(ok) else None,
                          "std": float(ok.std(ddof=1)) if len(ok) > 1 else None}
        sub = {}
        for col in subgroup_cols or []:
            if col not in work.columns:
                raise ValueError(f"Subgroup column '{col}' not found.")
            sub[col] = subgroup_metrics(work.loc[idx, col], y[idx], p)
        out[name] = {"folds": fold_rows, "summary": summary,
                     "pooled": compute_metrics(y[idx], p),
                     "calibration": calibration_summary(y[idx], p),
                     "subgroups": sub}
    return {"positive_class": str(classes[1]), "features": features,
            "n_folds": len(folds), "models": out}


def compare_designs(df, target, group_col=None, time_col=None, drop_columns=None,
                    n_splits: int = 5, seed: int = 42, models=None, positive_class=None) -> dict:
    """Quantify how much conventional random CV overstates performance.

    optimism = AUC(random stratified CV) - AUC(design-aware CV). Positive values
    mean the conventional estimate was more flattering than the design-aware one.
    """
    if not (group_col or time_col):
        return {"applicable": False,
                "reason": "No group or time structure declared; nothing to compare."}
    naive = select_validation(df, target, n_splits=n_splits, seed=seed)
    design = select_validation(df, target, group_col, time_col, n_splits, seed)
    structure_cols = [c for c in (group_col, time_col) if c]
    naive_eval = evaluate_models(df, target, naive, drop_columns=(drop_columns or []) + structure_cols,
                                 seed=seed, models=models, positive_class=positive_class)
    design_eval = evaluate_models(df, target, design, drop_columns=drop_columns, seed=seed,
                                  models=models, positive_class=positive_class)
    probe = dataclasses.replace(naive, group_col=group_col, time_col=time_col)
    leakage = verify_plan(df, target, probe)
    rows = {}
    for name in naive_eval["models"]:
        a = naive_eval["models"][name]["summary"]["roc_auc"]["mean"]
        b = design_eval["models"][name]["summary"]["roc_auc"]["mean"]
        rows[name] = {"naive_auc": a, "design_aware_auc": b,
                      "optimism": None if a is None or b is None else a - b}
    return {"applicable": True, "naive_strategy": naive.strategy,
            "design_strategy": design.strategy,
            "naive_plan_violations": {k: leakage[k] for k in
                                      ("group_overlap", "distinct_groups_shared", "n_groups",
                                               "time_order_violations")},
            "models": rows}
