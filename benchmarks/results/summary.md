# Synthetic benchmark (10 seeds per scenario)

optimism = random-CV AUC − design-aware-CV AUC. Data are synthetic and built to contain (or lack) a known hazard; results show the mechanism, not real-world prevalence.

| scenario         | model               |   naive_auc |   design_auc |   optimism_mean |   optimism_sd |   optimism_min |   optimism_max |
|:-----------------|:--------------------|------------:|-------------:|----------------:|--------------:|---------------:|---------------:|
| grouped_control  | gradient_boosting   |       0.563 |        0.565 |          -0.002 |         0.013 |         -0.026 |          0.022 |
| grouped_control  | logistic_regression |       0.628 |        0.624 |           0.004 |         0.006 |         -0.003 |          0.013 |
| grouped_control  | random_forest       |       0.554 |        0.556 |          -0.001 |         0.017 |         -0.043 |          0.017 |
| grouped_leaky    | gradient_boosting   |       0.688 |        0.517 |           0.171 |         0.062 |          0.07  |          0.273 |
| grouped_leaky    | logistic_regression |       0.602 |        0.573 |           0.029 |         0.017 |         -0.004 |          0.052 |
| grouped_leaky    | random_forest       |       0.708 |        0.526 |           0.183 |         0.057 |          0.095 |          0.284 |
| temporal_control | gradient_boosting   |       0.821 |        0.821 |          -0     |         0.008 |         -0.013 |          0.017 |
| temporal_control | logistic_regression |       0.862 |        0.862 |           0     |         0.002 |         -0.002 |          0.004 |
| temporal_control | random_forest       |       0.818 |        0.809 |           0.009 |         0.015 |         -0.008 |          0.031 |
| temporal_drift   | gradient_boosting   |       0.817 |        0.686 |           0.131 |         0.009 |          0.115 |          0.142 |
| temporal_drift   | logistic_regression |       0.49  |        0.423 |           0.067 |         0.021 |          0.036 |          0.098 |
| temporal_drift   | random_forest       |       0.814 |        0.693 |           0.121 |         0.015 |          0.095 |          0.142 |
