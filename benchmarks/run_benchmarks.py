"""Synthetic benchmark: how much does random CV overstate AUC vs design-aware CV?

Scenarios with a known hazard should show positive optimism; controls should
show roughly zero. Controls matter: they measure the false-alarm side.
Run:  python benchmarks/run_benchmarks.py --seeds 10
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from researchguard.evaluate import compare_designs
from researchguard.synthetic import make_grouped_data, make_temporal_data

SCENARIOS = {
    "grouped_leaky": (lambda s: make_grouped_data(s, leaky=True), dict(group_col="group_id")),
    "grouped_control": (lambda s: make_grouped_data(s, leaky=False), dict(group_col="group_id")),
    "temporal_drift": (lambda s: make_temporal_data(s, drift=True), dict(time_col="visit_date")),
    "temporal_control": (lambda s: make_temporal_data(s, drift=False), dict(time_col="visit_date")),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default=str(Path(__file__).parent / "results"))
    args = ap.parse_args()
    rows = []
    for name, (make, kw) in SCENARIOS.items():
        for seed in range(args.seeds):
            res = compare_designs(make(seed), "outcome", seed=seed, **kw)
            for model, m in res["models"].items():
                rows.append({"scenario": name, "seed": seed, "model": model, **m})
    df = pd.DataFrame(rows)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "benchmark_results.csv", index=False)
    g = df.groupby(["scenario", "model"])
    summ = g.agg(naive_auc=("naive_auc", "mean"), design_auc=("design_aware_auc", "mean"),
                 optimism_mean=("optimism", "mean"), optimism_sd=("optimism", "std"),
                 optimism_min=("optimism", "min"), optimism_max=("optimism", "max")).round(3)
    lines = [f"# Synthetic benchmark ({args.seeds} seeds per scenario)", "",
             "optimism = random-CV AUC − design-aware-CV AUC. Data are synthetic and built to "
             "contain (or lack) a known hazard; results show the mechanism, not real-world "
             "prevalence.", "", summ.reset_index().to_markdown(index=False), ""]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
