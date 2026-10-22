# Changelog

## 0.1.2 — 2026-10-22

### Fixed (both found by running on real hospital data)
- Columns with pandas nullable types (`Int64`, `boolean`, `string` with `pd.NA`) crashed the
  imputers. They are now converted to plain float/object columns before modelling.
- With many category levels (e.g. hundreds of diagnosis codes) the one-hot output became sparse
  and `HistGradientBoostingClassifier` rejected it. Gradient boosting now uses a dense ordinal
  encoding; logistic regression and random forest keep one-hot encoding.

- Columns named like identifiers (`*_id`) but with few distinct values (e.g. `admission_type_id`,
  8 codes) are no longer flagged as identifiers; a name match now also needs more than 30 levels.

### Added
- README figures (light and dark): how a random split mixes patients, and how much it inflated
  AUC across every benchmark scenario. Built from the results files by `docs/img/make_figures.py`.
- First real-data benchmark: Diabetes 130-US hospitals (UCI 296, CC BY 4.0), grouped by
  patient (`benchmarks/real/diabetes130.py`, registered in `benchmarks/datasets.yaml`).
  Result: about 20% of patients were shared across random-CV splits, but optimism was at most
  +0.005 AUC — the structure was detected and its effect correctly reported as negligible.

## 0.1.1 — 2026-10-22

### Fixed
- **Forward-in-time validation could split a single time point across train and test.**
  When several rows shared a timestamp (same day, same visit), the row-based split could put
  some of them in training and the rest in testing, and `verify_plan` still reported the plan
  as passed because it checked `train_max > test_min` instead of `>=`. Splits are now made on
  distinct time points, the check uses `>=`, and both behaviours are covered by tests.
- Too few distinct time points for the requested folds now raises a clear error.

### Added
- `positive_class` option (Python, CLI `--positive-class`, web app) so the event of interest is
  stated rather than inferred from alphabetical order. The chosen class is recorded in the report.
- `verify_plan` reports `distinct_groups_shared` and `n_groups`; the report now says
  "X of N groups had rows in both a training and a test fold" instead of a summed count.
- Committed example report (`examples/output/report.md`) so readers can see the output
  without installing anything.
- `CITATION.cff`, project metadata, literature references in `docs/methodology.md`.

### Changed
- Quartile subgroups are listed from lowest to highest instead of in text order.
- A declared group/time column is no longer flagged as a possible identifier in the audit.
- Fold checks in the Markdown report are written as a sentence rather than a raw dictionary.

## 0.1.0 — 2026-10-19
- Initial release: data audit, design-aware validation selection and verification,
  leakage-safe baselines, calibration and subgroup reliability, optimism estimate,
  JSON/Markdown reports, synthetic benchmark.
