import numpy as np

from researchguard.calibration import calibration_summary
from researchguard.metrics import compute_metrics


def test_perfect_ranking():
    m = compute_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert m["roc_auc"] == 1.0 and m["accuracy"] == 1.0


def test_single_class_gives_nan_auc_not_crash():
    m = compute_metrics([1, 1, 1], [0.6, 0.7, 0.8])
    assert np.isnan(m["roc_auc"])


def test_calibration_ece_zero_when_perfect():
    p = np.array([0.0] * 50 + [1.0] * 50)
    y = np.array([0] * 50 + [1] * 50)
    assert calibration_summary(y, p)["ece"] == 0.0


def test_calibration_detects_overconfidence():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.5, 2000)
    assert calibration_summary(y, np.full(2000, 0.95))["ece"] > 0.4
