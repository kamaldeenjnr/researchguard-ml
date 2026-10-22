# Limitations

- Binary classification, tabular data only.
- Leakage detection is heuristic. The single-feature AUC ≥ 0.98 flag and name-based group/time
  suspicion can miss real leakage and flag legitimate strong predictors.
- Group + time validation cannot yet be combined; nested tuning is not implemented
  (models use fixed default hyperparameters).
- Optimism is estimated with the same fixed baselines; it is not a general measure of dataset quality.
- ECE with uniform bins is unstable on small samples; read it alongside the bin table.
- Benchmarks so far are synthetic by construction.
- Output describes predictive association, not causation, and is not clinical validation.
- Uncertainty is reported as fold SD, not a confidence interval.
- In temporal comparisons, forward-in-time folds also train on less data (an expanding window)
  than random CV, so part of the measured optimism can come from training-set size rather than
  leakage or drift. The stable-relationship control in the benchmark estimates this effect
  (about 0.00–0.01 AUC on the synthetic data), but it is not separated out on real data.
