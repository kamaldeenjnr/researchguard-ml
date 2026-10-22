# Real-data benchmark: Diabetes 130-US hospitals (UCI 296)

99,343 encounters from 69,990 patients after excluding death/hospice discharges; 16,341 patients have more than one encounter (45,694 encounters). Outcome: readmitted within 30 days (11,314 positive, 11.4%).

5 seeds × 5 folds. optimism = random-CV AUC − patient-grouped-CV AUC.

| model               |   random_cv_auc |   patient_grouped_auc |   optimism_mean |   optimism_sd |   optimism_min |   optimism_max |
|:--------------------|----------------:|----------------------:|----------------:|--------------:|---------------:|---------------:|
| gradient_boosting   |          0.6718 |                0.6723 |         -0.0005 |        0.0013 |        -0.0023 |         0.0009 |
| logistic_regression |          0.6558 |                0.6554 |          0.0004 |        0.0014 |        -0.0012 |         0.0025 |
| random_forest       |          0.659  |                0.6565 |          0.0025 |        0.0023 |        -0.0007 |         0.005  |

Audit warnings: MISSING_VALUES, CONSTANT_COLUMNS (`examide`, `citoglipton`). This run also
flagged `admission_type_id`, `discharge_disposition_id` and `admission_source_id` as possible
identifiers. That was a false alarm (they are short code lists) and the check was corrected
afterwards; the model results above are unaffected because audit warnings do not change modelling.

## Interpretation

Random CV put about 14,150 of 69,990 patients (20%) on both sides of a split in every seed, yet
AUC barely moved: optimism was at most +0.005 for any model and seed, and indistinguishable from
zero for logistic regression and gradient boosting. On this dataset, with these baseline models
and features, ignoring patient grouping did **not** materially inflate performance.

This is a useful result in both directions: ResearchGuard detected the structural problem
(shared patients) but did not exaggerate its consequence. It also shows that sharing patients
across folds only inflates scores when models can recognise individual patients from their
features, as in the synthetic `grouped_leaky` scenario. Whether that happens has to be measured
per dataset, which is what the optimism estimate is for.

Environment: Python 3.11.15, scikit-learn 1.9.1, pandas 3.0.6, numpy 2.4.6.

Source: Strack, B., et al. (2014). Impact of HbA1c measurement on hospital readmission rates: analysis of 70,000 clinical database patient records. BioMed Research International. Data: UCI Machine Learning Repository, CC BY 4.0.
