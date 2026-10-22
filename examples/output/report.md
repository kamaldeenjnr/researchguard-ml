# ResearchGuard ML report

Data fingerprint `d7af900048560a3f` · 600 rows · 7 columns · seed 42 · positive class `1`

## Data warnings

- none detected

## Validation design

Strategy: `stratified_group_kfold`

- Rows share 'group_id', so whole groups are held out together.

Fold checks: **passed** across 5 folds — groups shared between train and test: 0; time-order violations: 0; test rows duplicated in train: 0; folds missing a class: 0.

## Model performance (mean ± sd over folds)

| Model | ROC AUC | PR AUC | Brier | ECE |
|---|---|---|---|---|
| logistic_regression | 0.644 ± 0.040 | 0.672 | 0.236 | 0.037 |
| random_forest | 0.624 ± 0.088 | 0.640 | 0.243 | 0.076 |
| gradient_boosting | 0.619 ± 0.042 | 0.648 | 0.285 | 0.197 |

## Conventional vs design-aware validation

Under ordinary random CV, 60 of 60 groups had rows in both a training and a test fold.

| Model | Random-CV AUC | Design-aware AUC | Optimism |
|---|---|---|---|
| logistic_regression | 0.655 | 0.644 | 0.011 |
| random_forest | 0.727 | 0.624 | 0.103 |
| gradient_boosting | 0.713 | 0.619 | 0.093 |

## Subgroups — logistic_regression by `x` (AUC gap among reliable groups: 0.071)

| Level | n | Prevalence | AUC | Reliable |
|---|---|---|---|---|
| (-3.9, -0.701] | 150 | 0.353 | 0.501 | True |
| (-0.701, -0.0502] | 150 | 0.447 | 0.538 | True |
| (-0.0502, 0.676] | 150 | 0.627 | 0.570 | True |
| (0.676, 3.066] | 150 | 0.653 | 0.572 | True |

## Subgroups — random_forest by `x` (AUC gap among reliable groups: 0.131)

| Level | n | Prevalence | AUC | Reliable |
|---|---|---|---|---|
| (-3.9, -0.701] | 150 | 0.353 | 0.540 | True |
| (-0.701, -0.0502] | 150 | 0.447 | 0.619 | True |
| (-0.0502, 0.676] | 150 | 0.627 | 0.489 | True |
| (0.676, 3.066] | 150 | 0.653 | 0.603 | True |

## Subgroups — gradient_boosting by `x` (AUC gap among reliable groups: 0.130)

| Level | n | Prevalence | AUC | Reliable |
|---|---|---|---|---|
| (-3.9, -0.701] | 150 | 0.353 | 0.554 | True |
| (-0.701, -0.0502] | 150 | 0.447 | 0.581 | True |
| (-0.0502, 0.676] | 150 | 0.627 | 0.515 | True |
| (0.676, 3.066] | 150 | 0.653 | 0.645 | True |

## Interpretation limits

These results describe predictive performance on the supplied data under the stated validation design. They do not establish causal effects, and they are not a clinical validation. Warnings marked 'needs confirmation' depend on study-design knowledge this tool cannot infer.
