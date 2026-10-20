"""Synthetic datasets with known evaluation hazards (for tests and benchmarks)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def make_grouped_data(seed=0, n_groups=60, per_group=10, leaky=True) -> pd.DataFrame:
    """Repeated-measures data. Each group has a hidden outcome tendency.

    leaky=True adds group 'fingerprint' features (e.g. site-specific lab
    characteristics) that let a model recognise a group and recall its outcome
    rate. That works under random CV but cannot work on unseen groups.
    leaky=False is a control with independent rows and no such hazard.
    """
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(n_groups), per_group)
    n = len(g)
    u = rng.normal(0, 1.5, n_groups)[g] if leaky else 0.0
    x = rng.normal(size=n)
    y = rng.binomial(1, _sigmoid(0.5 * x + u))
    df = pd.DataFrame({"group_id": g, "x": x, "noise": rng.normal(size=n)})
    if leaky:
        fp = rng.normal(size=(n_groups, 3))[g] + rng.normal(0, 0.05, (n, 3))
        for i in range(3):
            df[f"site_signature_{i}"] = fp[:, i]
    else:
        df["group_id"] = rng.permutation(g)  # label unrelated to outcome or features
    df["outcome"] = y
    return df


def make_temporal_data(seed=0, n=1200, drift=True) -> pd.DataFrame:
    """Concept drift: the x->outcome relationship reverses in the later period.

    A period-proxy feature (e.g. calendar-linked measurement) lets models learn
    both regimes under random CV; forward-in-time validation exposes the shift.
    drift=False is the stable-relationship control.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    late = (t >= n // 2).astype(float)
    x = rng.normal(size=n)
    sign = 1 - 2 * late if drift else 1.0
    y = rng.binomial(1, _sigmoid(2.0 * sign * x))
    proxy = late * 2 - 1 + rng.normal(0, 0.1, n)
    return pd.DataFrame({"visit_date": t, "x": x, "period_proxy": proxy, "outcome": y})
