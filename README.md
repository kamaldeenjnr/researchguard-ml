# ResearchGuard ML

[![tests](https://github.com/kamaldeenjnr/researchguard-ml/actions/workflows/tests.yml/badge.svg)](https://github.com/kamaldeenjnr/researchguard-ml/actions/workflows/tests.yml)
![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-0C203B)
[![license](https://img.shields.io/badge/license-MIT-0C203B)](LICENSE)
![status](https://img.shields.io/badge/status-v0.1.2%20alpha-C4412B)

ResearchGuard checks whether a model was tested in a way that matches how the data were
collected, and estimates how much a mismatch changes the reported score.

## In plain words

Think of a prediction model as a student and testing as an exam. If the exam reuses questions
the student already practised, a high score does not show the student can handle new questions.

Health data has the same trap. One patient often has several hospital visits. The usual method
shuffles all visits and splits them into a *training* set (practice) and a *test* set (the
exam). Some of a patient's visits can land in practice and another in the exam, so the model is
partly tested on someone it has already seen. Its score can then look better than it would be
for new patients.

ResearchGuard keeps all of a patient's visits on the same side, either all in practice or all
in the exam, checks that every split really does this, and reports the score both ways. If the
two scores are close, the shortcut did no harm. If they differ, the difference shows how much
the usual method overstated the result.

| Term | Meaning |
|---|---|
| Training / test data | The rows a model learns from, and the rows held back to check it. |
| Cross-validation (CV) | Repeating the train/test split several times (here, 5 "folds") so every row is tested once. |
| AUC | How well the model ranks people who had the outcome above those who did not. 0.5 is a coin toss, 1.0 is perfect. |
| Optimism | AUC with the usual random split minus AUC with patients kept together. Near 0 means no inflation. |

<img src="docs/img/fig1_splits.svg" width="100%" alt="Fold assignments for 8 patients with 3 visits each. Under random 4-fold CV every patient has visits in both train and test in at least one fold; under patient-grouped CV no patient is split.">

*Figure 1. Which visits are held out in each fold, for 8 patients with 3 visits each (dark = test).
Red lines mark a patient whose visits fall on both sides of that fold. Splits produced by
scikit-learn's `KFold` and `GroupKFold`.*

Status: v0.1.2, early research software. Binary outcomes and tabular data only. It describes
predictive performance; it is not causal inference or clinical validation.

## What it does

| Step | Module | Output |
|---|---|---|
| Data audit | `audit.py` | missing values, duplicates, constant columns, identifier-like columns, class imbalance, single features that separate the outcome almost perfectly |
| Validation design | `validation.py` | stratified, grouped or forward-in-time folds, chosen from the structure you declare. Structure that is only suspected from column names produces a warning; it is never assumed |
| Fold checks | `verify_plan` | groups shared across train and test, time-order violations, duplicated rows, folds missing a class |
| Modelling | `preprocessing.py`, `models.py` | logistic regression, random forest, gradient boosting; imputation, scaling and encoding fitted inside each training fold |
| Reliability | `metrics.py`, `calibration.py`, `subgroup.py` | ROC and PR AUC, Brier score, log-loss, calibration error, subgroup results (subgroups under 30 rows marked unreliable) |
| Optimism | `evaluate.compare_designs` | AUC under random CV minus AUC under the design-aware split |
| Report | `reporting.py` | JSON and Markdown, with configuration, seed, data fingerprint and library versions |

## A real run

<img src="docs/img/app_setup.png" width="100%" alt="ResearchGuard web app with a CSV of 8,401 hospital visits loaded; the outcome is 30-day readmission and the group column is patient_nbr.">

*ResearchGuard's web app with 8,401 real hospital visits from 6,000 patients, a random sample of
the Diabetes 130-US dataset. The outcome is readmission within 30 days, and `patient_nbr` tells
the tool which visits belong to the same person.*

<img src="docs/img/app_results.png" width="100%" alt="Report section showing that ordinary random CV split 1,206 of 6,000 patients across train and test, with AUC changing by less than 0.01.">

*Part of the report from that run. Every grouped fold passed its checks. An ordinary random split
would have put 1,206 of the 6,000 patients on both sides, yet AUC changed by less than 0.01, so on
this sample the shortcut did not inflate the score.*

An example report on synthetic data (60 patients, 10 visits each) is in
[`examples/output/report.md`](examples/output/report.md). In that run, random CV placed all 60
patients on both sides of the split, and the random forest's AUC fell from 0.727 to 0.624 once
patients were held out together.

## Install and run

```bash
git clone https://github.com/kamaldeenjnr/researchguard-ml.git
cd researchguard-ml
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest -q
python examples/binary_classification.py
```

Command line:

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

There is also a small web app: `pip install -e ".[app]" && streamlit run app/streamlit_app.py`.
Do not upload identifiable patient data to it.

## Results so far

<img src="docs/img/fig2_optimism.svg" width="100%" alt="Optimism by scenario and model. Up to about 0.18 AUC on synthetic data where patients are recognisable and about 0.13 under drift; close to zero for the controls and for the real hospital data.">

*Figure 2. Optimism (random-CV AUC minus design-aware AUC) by scenario and model. Markers are
means and lines are the range across seeds. Synthetic scenarios are built to contain or lack a
known problem; the bottom row is real hospital data.*

**Real data.** The Diabetes 130-US hospitals dataset has 99,343 encounters (after removing
deaths and hospice discharges) from 69,990 patients; 16,341 patients have more than one
encounter. The outcome is readmission within 30 days. Over 5 seeds of 5-fold CV:

| Model | Random-CV AUC | Patient-grouped AUC | Optimism, mean (range) |
|---|---|---|---|
| Logistic regression | 0.656 | 0.655 | 0.000 (−0.001 to 0.003) |
| Random forest | 0.659 | 0.657 | 0.003 (−0.001 to 0.005) |
| Gradient boosting | 0.672 | 0.672 | 0.000 (−0.002 to 0.001) |

Random CV put about 20% of patients on both sides of the split in every seed, but the scores
hardly moved. With these features and models, repeat patients did not make the task easier.
That is a useful finding in itself: shared patients only inflate a score when a model can
recognise them, and that has to be measured for each dataset rather than assumed. Details are in
[`benchmarks/results/real_diabetes130.md`](benchmarks/results/real_diabetes130.md) and the script
is [`benchmarks/real/diabetes130.py`](benchmarks/real/diabetes130.py). Running on this dataset
also turned up three bugs, listed in the [changelog](CHANGELOG.md).

**Synthetic data.** Four scenarios, 10 seeds each
([full table](benchmarks/results/summary.md)):

| Scenario | Random forest | Gradient boosting | Logistic regression |
|---|---|---|---|
| Grouped, patients recognisable from their features | +0.18 | +0.17 | +0.03 |
| Grouped, no hazard (control) | −0.00 | +0.00 | +0.00 |
| Relationship drifts over time | +0.12 | +0.13 | +0.07 |
| Stable over time (control) | +0.01 | −0.00 | +0.00 |

These datasets were built to contain the problem, so they show the mechanism and that the
controls stay near zero. They say nothing about how common the problem is. Forests and boosting
pick up group signatures much more than logistic regression does. Reproduce with
`python benchmarks/run_benchmarks.py --seeds 10`; the figures are rebuilt with
`python docs/img/make_figures.py`.

## Testing and reproducibility

Each check is tested on data where it should fire and on clean data where it should stay quiet.
The test suite runs on Python 3.10, 3.11 and 3.12 on every push. Reports record the seed, a
fingerprint of the data, the configuration and library versions, and `verify_plan` re-checks
every fold rather than trusting the splitter.

The problem itself is well described in the literature: record-wise splits of repeated-measures
data overstate accuracy ([Saeb et al., 2017](https://doi.org/10.1093/gigascience/gix019)),
structured data need blocked cross-validation
([Roberts et al., 2017](https://doi.org/10.1111/ecog.02881)), and leakage is a common cause of
irreproducible results in ML-based science
([Kapoor & Narayanan, 2023](https://doi.org/10.1016/j.patter.2023.100804)).

## Limitations

Leakage cannot be detected reliably from column names and structure alone, so those findings
are marked "needs confirmation". Group and time structure cannot yet be combined in one design,
and models use default hyperparameters. More real datasets, especially time-ordered ones, are
needed. Full list: [`docs/limitations.md`](docs/limitations.md).

## Contributing

Real-data benchmark cases (with licence-compatible data), new checks with tests, and criticism
of the method are all welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Citing and licence

Citation details are in [`CITATION.cff`](CITATION.cff). MIT licence.
Written by [Amadu Kamal](https://modelanalysishub.com).
