"""Real-data benchmark: Diabetes 130-US hospitals (1999-2008), UCI dataset 296.

Repeated hospital encounters from the same patient. Question: how much does ordinary
random cross-validation flatter 30-day readmission models compared with holding out
whole patients?

Data (CC BY 4.0, not redistributed here): download diabetic_data.csv from
https://archive.ics.uci.edu/dataset/296/diabetes-130-us-hospitals-for-years-1999-2008
Run:  python benchmarks/real/diabetes130.py --csv path/to/diabetic_data.csv --seeds 5

Pre-specified choices (fixed before looking at results):
- Outcome: readmitted within 30 days ("<30") = positive; ">30" and "NO" = negative.
- Encounters ending in death or hospice (discharge_disposition_id 11, 13, 14, 19, 20, 21)
  are removed, because those patients cannot be readmitted (as in Strack et al., 2014).
- Dropped: encounter_id (row identifier) and weight (97% missing).
- Group: patient_nbr. Coded IDs are treated as categories; diagnosis codes are cut to their
  3-character ICD-9 category.
"""
from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from researchguard import audit_dataset, compare_designs

EXCLUDE_DISCHARGE = {11, 13, 14, 19, 20, 21}


def load(csv: str) -> pd.DataFrame:
    df = pd.read_csv(csv, na_values=["?"], keep_default_na=False, low_memory=False)
    df = df[~df["discharge_disposition_id"].isin(EXCLUDE_DISCHARGE)].copy()
    df["readmit_30d"] = np.where(df["readmitted"] == "<30", "yes", "no")
    df = df.drop(columns=["readmitted", "encounter_id", "weight"])
    for c in ("admission_type_id", "discharge_disposition_id", "admission_source_id"):
        df[c] = df[c].astype(str)
    for c in ("diag_1", "diag_2", "diag_3"):
        df[c] = df[c].astype("string").str[:3]
    return df.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results"))
    args = ap.parse_args()

    df = load(args.csv)
    visits = df["patient_nbr"].value_counts()
    audit = audit_dataset(df, "readmit_30d", declared_columns=["patient_nbr"])
    rows = []
    for seed in range(args.seeds):
        t0 = time.time()
        res = compare_designs(df, "readmit_30d", group_col="patient_nbr", seed=seed,
                              positive_class="yes")
        for model, m in res["models"].items():
            rows.append({"seed": seed, "model": model, **m})
        nv = res["naive_plan_violations"]
        print(f"seed {seed}: {time.time() - t0:.0f}s, random CV shared "
              f"{nv['distinct_groups_shared']} of {nv['n_groups']} patients")

    r = pd.DataFrame(rows)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    r.to_csv(out / "real_diabetes130.csv", index=False)
    s = r.groupby("model").agg(random_cv_auc=("naive_auc", "mean"),
                               patient_grouped_auc=("design_aware_auc", "mean"),
                               optimism_mean=("optimism", "mean"),
                               optimism_sd=("optimism", "std"),
                               optimism_min=("optimism", "min"),
                               optimism_max=("optimism", "max")).round(4)
    n_pos = int((df["readmit_30d"] == "yes").sum())
    lines = [
        "# Real-data benchmark: Diabetes 130-US hospitals (UCI 296)", "",
        f"{len(df):,} encounters from {len(visits):,} patients after excluding death/hospice "
        f"discharges; {int((visits > 1).sum()):,} patients have more than one encounter "
        f"({int(visits[visits > 1].sum()):,} encounters). Outcome: readmitted within 30 days "
        f"({n_pos:,} positive, {n_pos / len(df):.1%}).", "",
        f"{args.seeds} seeds × 5 folds. optimism = random-CV AUC - patient-grouped-CV AUC.", "",
        s.reset_index().to_markdown(index=False), "",
        f"Audit warnings: {', '.join(w['code'] for w in audit['warnings']) or 'none'}.", "",
        f"Environment: Python {platform.python_version()}, scikit-learn {sklearn.__version__}, "
        f"pandas {pd.__version__}, numpy {np.__version__}.", "",
        "Source: Strack, B., et al. (2014). Impact of HbA1c measurement on hospital readmission "
        "rates: analysis of 70,000 clinical database patient records. BioMed Research "
        "International. Data: UCI Machine Learning Repository, CC BY 4.0.", ""]
    (out / "real_diabetes130.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
