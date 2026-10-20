import numpy as np
import pandas as pd
import pytest

from researchguard.validation import select_validation, verify_plan


def make_df(n_groups=20, per_group=5, seed=0):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(n_groups), per_group)
    y = rng.integers(0, 2, len(g))
    return pd.DataFrame({
        "patient_id": g,
        "visit_date": np.arange(len(g)),
        "age": rng.normal(40, 10, len(g)),
        "y": y,
    })


def test_group_plan_has_no_group_overlap():
    df = make_df()
    plan = select_validation(df, "y", group_col="patient_id")
    assert plan.strategy == "stratified_group_kfold"
    assert verify_plan(df, "y", plan)["group_overlap"] == 0


def test_random_split_leaks_groups():
    """Why this tool exists: naive CV puts the same patient in train and test."""
    df = make_df()
    plan = select_validation(df, "y")
    plan.group_col = "patient_id"  # measure overlap under the naive plan
    assert verify_plan(df, "y", plan)["group_overlap"] > 0


def test_unconfirmed_group_risk_is_warned_not_assumed():
    plan = select_validation(make_df(), "y")
    assert plan.strategy == "stratified_kfold"
    codes = {w["code"] for w in plan.warnings}
    assert {"UNCONFIRMED_GROUP_RISK", "UNCONFIRMED_TIME_RISK"} <= codes


def test_time_plan_trains_on_past_only():
    df = make_df().sample(frac=1, random_state=1)  # shuffled rows
    plan = select_validation(df, "y", time_col="visit_date")
    v = verify_plan(df, "y", plan)
    assert plan.strategy == "time_series_split"
    assert v["time_order_violations"] == 0


def test_duplicate_rows_across_folds_detected():
    df = make_df()
    df = pd.concat([df, df], ignore_index=True)
    plan = select_validation(df, "y")
    assert verify_plan(df, "y", plan)["test_rows_duplicated_in_train"] > 0


def test_n_splits_reduced_for_tiny_minority():
    df = pd.DataFrame({"x": range(20), "y": [1] * 3 + [0] * 17})
    plan = select_validation(df, "y", n_splits=5)
    assert plan.n_splits == 3
    assert any(w["code"] == "N_SPLITS_REDUCED" for w in plan.warnings)


def test_rejects_bad_inputs():
    df = make_df()
    with pytest.raises(ValueError):
        select_validation(df, "y", group_col="nope")
    with pytest.raises(NotImplementedError):
        select_validation(df, "y", group_col="patient_id", time_col="visit_date")
