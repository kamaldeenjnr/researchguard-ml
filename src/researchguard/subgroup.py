from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import compute_metrics


def subgroup_metrics(values: pd.Series, y, p, min_n: int = 30, max_levels: int = 10) -> dict:
    """Per-subgroup performance. Numeric columns with many values are cut into quartiles.

    Subgroups below min_n rows, or lacking both classes, are reported but
    excluded from the AUC gap because their estimates are unreliable.
    """
    values = pd.Series(values).reset_index(drop=True)
    y = np.asarray(y)
    p = np.asarray(p)
    binned = False
    if pd.api.types.is_numeric_dtype(values) and values.nunique() > max_levels:
        values = pd.qcut(values, 4, duplicates="drop").astype(str)
        binned = True
    values = values.astype(str).where(values.notna(), "missing")

    rows = []
    for level in sorted(values.unique()):
        m = (values == level).to_numpy()
        r = compute_metrics(y[m], p[m])
        r["level"] = level
        r["reliable"] = bool(m.sum() >= min_n and len(np.unique(y[m])) == 2)
        rows.append(r)
    aucs = [r["roc_auc"] for r in rows if r["reliable"]]
    return {
        "binned_into_quartiles": binned,
        "min_n": min_n,
        "levels": rows,
        "auc_gap": float(max(aucs) - min(aucs)) if len(aucs) >= 2 else None,
    }
