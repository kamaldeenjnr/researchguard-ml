# ResearchGuard ML

**Auditable machine-learning evaluation for tabular research data.**

Most quick ML projects report a cross-validated score and stop. ResearchGuard asks a prior
question: *is that evaluation design appropriate for how the data were collected?* If the same
patient, hospital or time period appears on both sides of a split, the score measures recall,
not generalisation.

> **Status: v0.1, early research software.** Binary classification on tabular data only.
> Predictive association only: not causal inference, not clinical validation.

## What it does

| Step | Module | Output |
|---|---|---|
| Data audit | `audit.py` | missing values, duplicates, constants, identifier-like columns, class imbalance, single-feature leakage suspects |
| Validation design | `validation.py` | stratified / group-aware / forward-in-time splits chosen from the study design **you declare**; undeclared suspected structure is warned about, never silently assumed |
| Fold verification | `verify_plan` | group overlap, time-order violations, duplicate rows across train/test, folds missing a class |
| Leakage-safe modelling | `preprocessing.py`, `models.py` | logistic regression, random forest, gradient boosting; all preprocessing fitted inside each training fold |
| Reliability | `metrics.py`, `calibration.py`, `subgroup.py` | ROC/PR AUC, Brier, log-loss, ECE, per-subgroup metrics (small subgroups flagged unreliable) |
| Optimism estimate | `evaluate.compare_designs` | how much random CV overstates AUC compared with the design-aware split |
| Reproducible report | `reporting.py` | JSON + Markdown with config, seed, data fingerprint, library versions |

## Install and run

```bash
git clone https://github.com/kamaldeenjnr/researchguard-ml.git
cd researchguard-ml
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest -q
python examples/binary_classification.py
```

CLI:

```bash
researchguard audit data.csv --target outcome
researchguard run data.csv --target outcome --group-col patient_id --subgroup sex --out report/
```

Python:

```python
from researchguard import build_report, to_markdown
report = build_report(df, target="outcome", group_col="patient_id", subgroup_cols=["sex"])
print(to_markdown(report))
```

Web app: `pip install -e ".[app]" && streamlit run app/streamlit_app.py`
(do not upload identifiable patient data).

## Evidence so far (synthetic benchmark, 10 seeds)

`optimism = AUC under random CV − AUC under design-aware CV`. Mean over 10 seeds
(full table, SD and ranges: [`benchmarks/results/summary.md`](benchmarks/results/summary.md)).

| Scenario | Random forest | Gradient boosting | Logistic regression |
|---|---|---|---|
| Grouped data with group-recognisable features | **+0.18** | **+0.17** | +0.03 |
| Grouped control (no hazard) | −0.00 | −0.00 | +0.00 |
| Temporal concept drift | **+0.12** | **+0.13** | +0.07 |
| Temporal control (stable relationship) | +0.01 | −0.00 | +0.00 |

**How to read this honestly:**
- The datasets are *constructed* to contain the hazard. This shows the mechanism and that the
  controls do not false-alarm. It does **not** show how common the problem is in real research data.
- Flexible models (forests, boosting) exploit the hazard far more than logistic regression.
- Real-data benchmarks (e.g. OpenML datasets with known grouping) are the next required step
  and are not done yet. See [`docs/benchmark_protocol.md`](docs/benchmark_protocol.md).

Reproduce: `python benchmarks/run_benchmarks.py --seeds 10` (requires `pip install tabulate`).

## Limitations

See [`docs/limitations.md`](docs/limitations.md). In short: leakage cannot be reliably detected
from column names and structure alone, so those findings are labelled *needs confirmation*;
group + time validation together is not yet supported; no causal claims; no clinical use.

## Contributing

Issues and PRs welcome, especially real-data benchmark cases (with licence-compatible data),
new leakage checks with tests, and methodology critique. See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

MIT
