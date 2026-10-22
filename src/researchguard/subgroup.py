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
    order = None
    if pd.api.types.is_numeric_dtype(values) and values.nunique() > max_levels:
        cut = pd.qcut(values, 4, duplicates="drop")
        order = [str(c) for c in cut.cat.categories]  # lowest quartile first
        values = cut.astype(str).where(cut.notna(), "missing")
        binned = True
    else:
        values = values.astype(str).where(values.notna(), "missing")
    levels = sorted(values.unique())
    if order is not None:
        levels = [lv for lv in order if lv in set(levels)] + [lv for lv in levels if lv not in order]

    rows = []
    for level in levels:
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
