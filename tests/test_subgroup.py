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


def test_quartiles_listed_low_to_high():
    rng = np.random.default_rng(3)
    v = pd.Series(rng.normal(size=400))
    v[:20] = np.nan
    r = subgroup_metrics(v, rng.integers(0, 2, 400), rng.random(400))
    levels = [x["level"] for x in r["levels"]]
    assert levels[-1] == "missing"
    lows = [float(lv.strip("([").split(",")[0]) for lv in levels[:-1]]
    assert lows == sorted(lows)
