# Methodology

## Research question
Can a transparent, design-aware evaluation workflow detect when conventional random
cross-validation gives misleading confidence, and quantify the consequence reproducibly?

## What is and is not claimed
Cross-validation, calibration assessment and leakage-safe pipelines are established
(see scikit-learn's model-evaluation and pipeline documentation). The problem ResearchGuard
targets is also well documented: record-wise splitting of repeated-measures data inflates
accuracy (Saeb et al., 2017), structured data need blocked or grouped cross-validation
(Roberts et al., 2017), and leakage is a leading cause of irreproducible ML-based science
(Kapoor & Narayanan, 2023). ResearchGuard does **not** claim
a new algorithm. Its candidate contribution is an auditable workflow that (1) selects the
validation design from declared study structure, (2) verifies the folds honour it, and (3)
reports the *optimism* of the conventional estimate. Novelty requires a literature review and
real-data benchmarks that have not yet been done.

## Pipeline
1. **Audit** observable data problems (`audit.py`).
2. **Design selection** (`validation.py`): group → `StratifiedGroupKFold`; time →
   `TimeSeriesSplit` over time-sorted rows; otherwise `StratifiedKFold`. Group and time columns are
   excluded from the feature set.
3. **Verification**: groups never straddle train/test; training data precede test data; duplicate
   feature rows spanning train/test are counted.
4. **Modelling**: three baselines, each in a `Pipeline` so imputation/scaling/encoding are learned on
   training folds only.
5. **Reliability**: ROC AUC, PR AUC, Brier, log-loss, expected calibration error (10 uniform bins),
   per-subgroup metrics. Subgroups under 30 rows or lacking both classes are excluded from the AUC gap.
6. **Optimism**: AUC(random stratified CV) − AUC(design-aware CV), same models, same seed,
   same features.

## Design decisions
- Declared structure is respected; *suspected* structure only produces `UNCONFIRMED_*` warnings.
- No composite "risk score" is shipped: an unvalidated weighted sum would be arbitrary. Any score
  must be justified and benchmarked first.
- Pooled out-of-fold predictions are used for calibration and subgroups; for temporal splits only
  the later rows receive predictions.

- Temporal splits use distinct time points: rows that share a timestamp always fall on the
  same side of a split (fixed in 0.1.1).

## References
- Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in
  machine-learning-based science. *Patterns*. https://doi.org/10.1016/j.patter.2023.100804
- Roberts, D. R., et al. (2017). Cross-validation strategies for data with temporal, spatial,
  hierarchical, or phylogenetic structure. *Ecography*, 40, 913–929.
  https://doi.org/10.1111/ecog.02881
- Saeb, S., et al. (2017). The need to approximate the use-case in clinical machine learning.
  *GigaScience*, 6(5). https://doi.org/10.1093/gigascience/gix019
