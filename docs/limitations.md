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
