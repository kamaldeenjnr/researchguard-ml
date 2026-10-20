from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from . import __version__
from .audit import audit_dataset
from .evaluate import compare_designs, evaluate_models
from .validation import select_validation, verify_plan

DISCLAIMER = ("These results describe predictive performance on the supplied data under the "
              "stated validation design. They do not establish causal effects, and they are "
              "not a clinical validation. Warnings marked 'needs confirmation' depend on "
              "study-design knowledge this tool cannot infer.")


def _data_hash(df: pd.DataFrame) -> str:
    h = pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes()
    return hashlib.sha256(h + "|".join(map(str, df.columns)).encode()).hexdigest()[:16]


def build_report(df, target, group_col=None, time_col=None, subgroup_cols=None,
                 drop_columns=None, n_splits=5, seed=42) -> dict:
    drop_columns = list(drop_columns or [])
    audit = audit_dataset(df, target)
    plan = select_validation(df, target, group_col, time_col, n_splits, seed)
    return {
        "researchguard_version": __version__,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                        "pandas": pd.__version__, "numpy": np.__version__},
        "config": {"target": target, "group_col": group_col, "time_col": time_col,
                   "subgroup_cols": subgroup_cols or [], "drop_columns": drop_columns,
                   "n_splits": plan.n_splits, "seed": seed},
        "data_fingerprint": _data_hash(df),
        "audit": audit,
        "validation": {"strategy": plan.strategy, "reasons": plan.reasons,
                       "warnings": plan.warnings, "checks": verify_plan(df, target, plan)},
        "evaluation": evaluate_models(df, target, plan, subgroup_cols, drop_columns, seed),
        "design_comparison": compare_designs(df, target, group_col, time_col,
                                             drop_columns, n_splits, seed),
        "disclaimer": DISCLAIMER,
    }


def _f(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def to_markdown(r: dict) -> str:
    a, v, e, c = r["audit"], r["validation"], r["evaluation"], r["design_comparison"]
    L = ["# ResearchGuard ML report", "",
         f"Data fingerprint `{r['data_fingerprint']}` · {a['n_rows']} rows · "
         f"{a['n_columns']} columns · seed {r['config']['seed']}", "",
         "## Data warnings", ""]
    L += [f"- **{w['code']}** ({w['severity']})"
          + (" — needs confirmation" if w.get("needs_user_confirmation") else "")
          + f": `{w['details']}`" for w in a["warnings"]] or ["- none detected"]
    L += ["", "## Validation design", "", f"Strategy: `{v['strategy']}`", ""]
    L += [f"- {x}" for x in v["reasons"]]
    L += [f"- **{w['code']}**: `{w['details']}`" + (" — needs confirmation"
          if w.get("needs_user_confirmation") else "") for w in v["warnings"]]
    L += ["", f"Fold checks: `{v['checks']}`", "", "## Model performance (mean ± sd over folds)", "",
          "| Model | ROC AUC | PR AUC | Brier | ECE |", "|---|---|---|---|---|"]
    for n, m in e["models"].items():
        s = m["summary"]
        L.append(f"| {n} | {_f(s['roc_auc']['mean'])} ± {_f(s['roc_auc']['std'])} | "
                 f"{_f(s['pr_auc']['mean'])} | {_f(s['brier']['mean'])} | "
                 f"{_f(m['calibration']['ece'])} |")
    if c.get("applicable"):
        L += ["", "## Conventional vs design-aware validation", "",
              f"Random CV placed the same group/time structure across train and test: "
              f"`{c['naive_plan_violations']}`", "",
              "| Model | Random-CV AUC | Design-aware AUC | Optimism |", "|---|---|---|---|"]
        for n, m in c["models"].items():
            L.append(f"| {n} | {_f(m['naive_auc'])} | {_f(m['design_aware_auc'])} | "
                     f"{_f(m['optimism'])} |")
    else:
        L += ["", "## Conventional vs design-aware validation", "", c["reason"]]
    for n, m in e["models"].items():
        for col, s in m["subgroups"].items():
            L += ["", f"## Subgroups — {n} by `{col}` (AUC gap among reliable groups: "
                      f"{_f(s['auc_gap'])})", "", "| Level | n | Prevalence | AUC | Reliable |",
                  "|---|---|---|---|---|"]
            L += [f"| {x['level']} | {x['n']} | {_f(x['prevalence'])} | {_f(x['roc_auc'])} | "
                  f"{x['reliable']} |" for x in s["levels"]]
    L += ["", "## Interpretation limits", "", r["disclaimer"], ""]
    return "\n".join(L)


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_report(report: dict, out_dir) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jp, mp = out / "report.json", out / "report.md"
    jp.write_text(json.dumps(report, indent=2, default=_default, allow_nan=True), encoding="utf-8")
    mp.write_text(to_markdown(report), encoding="utf-8")
    return jp, mp
