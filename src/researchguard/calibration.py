from __future__ import annotations

import numpy as np


def calibration_summary(y, p, n_bins: int = 10) -> dict:
    """Reliability-bin table and expected calibration error (uniform bins)."""
    y = np.asarray(y).astype(float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    bins, ece = [], 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        mean_p, frac = float(p[m].mean()), float(y[m].mean())
        ece += m.mean() * abs(mean_p - frac)
        bins.append({"bin": b, "n": int(m.sum()), "mean_predicted": mean_p,
                     "observed_fraction": frac})
    return {"ece": float(ece), "n_bins": n_bins, "bins": bins}
