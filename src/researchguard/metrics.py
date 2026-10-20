from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             log_loss, roc_auc_score)


def compute_metrics(y, p, threshold: float = 0.5) -> dict:
    """Discrimination and probability-quality metrics for binary 0/1 outcomes."""
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype=float), 1e-15, 1 - 1e-15)
    two = len(np.unique(y)) == 2
    return {
        "n": int(len(y)),
        "prevalence": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, p)) if two else float("nan"),
        "pr_auc": float(average_precision_score(y, p)) if two else float("nan"),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "accuracy": float(accuracy_score(y, p >= threshold)),
    }
