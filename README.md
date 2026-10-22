# ResearchGuard ML

[![tests](https://github.com/kamaldeenjnr/researchguard-ml/actions/workflows/tests.yml/badge.svg)](https://github.com/kamaldeenjnr/researchguard-ml/actions/workflows/tests.yml)
![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-0C203B)
[![license](https://img.shields.io/badge/license-MIT-0C203B)](LICENSE)
![status](https://img.shields.io/badge/status-v0.1.1%20alpha-C4412B)

**Auditable machine-learning evaluation for tabular research data.**

**In one sentence:** before you trust a model's accuracy, ResearchGuard checks whether the
way it was tested matches the way your data were collected, and shows how much the usual
shortcut would have flattered the result.

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

## What a report looks like

From [`examples/output/report.md`](examples/output/report.md) (synthetic data, 60 patients × 10 visits):

> Under ordinary random CV, **60 of 60 groups had rows in both a training and a test fold.**
>
> | Model | Random-CV AUC | Design-aware AUC | Optimism |
> |---|---|---|---|
> | logistic_regression | 0.655 | 0.644 | 0.011 |
> | random_forest | 0.727 | 0.624 | **0.103** |
> | gradient_boosting | 0.713 | 0.619 | **0.093** |

Read: with the same patients on both sides of the split, the random forest looked 0.10 AUC
better than it really is on new patients.

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
researchguard run data.csv --target outcome --positive-class yes \
    --group-col patient_id --subgroup sex --out report/
```

Python:

```python
from researchguard import build_report, to_markdown
report = build_report(df, target="outcome", positive_class="yes",
                      group_col="patient_id", subgroup_cols=["sex"])
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
| Grouped control (no hazard) | −0.00 | +0.00 | +0.00 |
| Temporal concept drift | **+0.12** | **+0.13** | +0.07 |
| Temporal control (stable relationship) | +0.01 | −0.00 | +0.00 |

**How to read this honestly:**
- The datasets are *constructed* to contain the hazard. This shows the mechanism and that the
  controls do not false-alarm. It does **not** show how common the problem is in real research data.
- Flexible models (forests, boosting) exploit the hazard far more than logistic regression.
- Real-data benchmarks (e.g. OpenML datasets with known grouping) are the next required step
  and are not done yet. See [`docs/benchmark_protocol.md`](docs/benchmark_protocol.md).

Reproduce: `python benchmarks/run_benchmarks.py --seeds 10`.

## Why you can check it rather than trust it

- **Checks are tested both ways:** firing on a known problem, and staying quiet on clean data
  (false alarms matter as much as misses). The suite runs on Python 3.10–3.12 on every push.
- **Controls in the benchmark:** scenarios with no hazard should show ~0 optimism, and do.
- **Reproducible reports:** each report records the seed, data fingerprint, configuration and
  library versions.
- **Folds are verified, not assumed:** `verify_plan` re-checks every split for shared groups,
  time-order violations, duplicated rows and missing classes.
- **Honest scope:** see the limitations below and the [changelog](CHANGELOG.md), which lists the
  bugs found and fixed.

The underlying problem is well documented: record-wise splits of repeated-measures data inflate
accuracy ([Saeb et al., 2017](https://doi.org/10.1093/gigascience/gix019)), structured data need
blocked cross-validation ([Roberts et al., 2017](https://doi.org/10.1111/ecog.02881)), and leakage
is a leading cause of irreproducible ML-based science
([Kapoor & Narayanan, 2023](https://doi.org/10.1016/j.patter.2023.100804)).

## Limitations

See [`docs/limitations.md`](docs/limitations.md). In short: leakage cannot be reliably detected
from column names and structure alone, so those findings are labelled *needs confirmation*;
group + time validation together is not yet supported; no causal claims; no clinical use.

## Contributing

Issues and PRs welcome, especially real-data benchmark cases (with licence-compatible data),
new leakage checks with tests, and methodology critique. See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Citing

If ResearchGuard helps your work, please cite it using [`CITATION.cff`](CITATION.cff)
(GitHub shows a "Cite this repository" button).

## License

MIT · built by [Amadu Kamal](https://modelanalysishub.com)
