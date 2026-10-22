# Synthetic benchmark (10 seeds per scenario)

optimism = random-CV AUC - design-aware-CV AUC. Data are synthetic and built to contain (or lack) a known hazard; results show the mechanism, not real-world prevalence.

Environment: Python 3.11.15, scikit-learn 1.9.1, pandas 3.0.6, numpy 2.4.6. Logistic regression and random forest results are identical across recent scikit-learn versions; gradient boosting (HistGradientBoostingClassifier) can differ by about 0.01 to 0.03 AUC between versions.

| scenario         | model               |   naive_auc |   design_auc |   optimism_mean |   optimism_sd |   optimism_min |   optimism_max |
|:-----------------|:--------------------|------------:|-------------:|----------------:|--------------:|---------------:|---------------:|
| grouped_control  | gradient_boosting   |       0.564 |        0.563 |           0.001 |         0.017 |         -0.025 |          0.021 |
| grouped_control  | logistic_regression |       0.628 |        0.624 |           0.004 |         0.006 |         -0.003 |          0.013 |
| grouped_control  | random_forest       |       0.554 |        0.556 |          -0.001 |         0.017 |         -0.043 |          0.017 |
| grouped_leaky    | gradient_boosting   |       0.691 |        0.518 |           0.173 |         0.064 |          0.077 |          0.268 |
| grouped_leaky    | logistic_regression |       0.602 |        0.573 |           0.029 |         0.017 |         -0.004 |          0.052 |
| grouped_leaky    | random_forest       |       0.708 |        0.526 |           0.183 |         0.057 |          0.095 |          0.284 |
| temporal_control | gradient_boosting   |       0.821 |        0.821 |           0     |         0.01  |         -0.012 |          0.021 |
| temporal_control | logistic_regression |       0.862 |        0.862 |           0     |         0.002 |         -0.002 |          0.004 |
| temporal_control | random_forest       |       0.818 |        0.809 |           0.009 |         0.015 |         -0.008 |          0.031 |
| temporal_drift   | gradient_boosting   |       0.818 |        0.687 |           0.132 |         0.008 |          0.115 |          0.142 |
| temporal_drift   | logistic_regression |       0.49  |        0.423 |           0.067 |         0.021 |          0.036 |          0.098 |
| temporal_drift   | random_forest       |       0.814 |        0.693 |           0.121 |         0.015 |          0.095 |          0.142 |
