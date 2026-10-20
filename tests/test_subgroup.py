import numpy as np
import pandas as pd

from researchguard.subgroup import subgroup_metrics


def test_small_groups_are_flagged_unreliable():
    rng = np.random.default_rng(0)
    n = 200
    sex = pd.Series(["f"] * 190 + ["m"] * 10)
    y = rng.integers(0, 2, n)
    p = rng.random(n)
    r = subgroup_metrics(sex, y, p)
    by = {x["level"]: x for x in r["levels"]}
    assert by["f"]["reliable"] and not by["m"]["reliable"]
    assert r["auc_gap"] is None  # only one reliable level


def test_numeric_subgroup_is_binned():
    rng = np.random.default_rng(1)
    r = subgroup_metrics(pd.Series(rng.normal(size=400)), rng.integers(0, 2, 400), rng.random(400))
    assert r["binned_into_quartiles"] and len(r["levels"]) == 4
