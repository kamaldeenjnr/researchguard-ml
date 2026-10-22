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


def _same_day_df(seed=0):
    """43 days with 7 rows each: row-based splits would cut through days."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"day": np.repeat(np.arange(43), 7),
                         "x": rng.normal(size=301),
                         "y": rng.integers(0, 2, 301)})


def test_time_split_keeps_each_time_point_on_one_side():
    df = _same_day_df()
    plan = select_validation(df, "y", time_col="day", n_splits=5)
    for tr, te in plan.split(df, "y"):
        assert not set(df.loc[tr, "day"]) & set(df.loc[te, "day"])
        assert df.loc[tr, "day"].max() < df.loc[te, "day"].min()
    assert verify_plan(df, "y", plan)["passed"]


def test_verify_flags_shared_time_points():
    """A plan that puts the same day in train and test must not pass."""
    df = _same_day_df()
    plan = select_validation(df, "y", time_col="day", n_splits=5)

    def row_split(d, target):  # the old, row-based behaviour
        from sklearn.model_selection import TimeSeriesSplit
        order = np.argsort(d["day"].to_numpy(), kind="stable")
        for tr, te in TimeSeriesSplit(5).split(order):
            yield order[tr], order[te]

    plan.split = row_split
    checks = verify_plan(df, "y", plan)
    assert checks["time_order_violations"] > 0
    assert not checks["passed"]


def test_too_few_time_points_rejected():
    df = _same_day_df().assign(day=lambda d: d["day"] % 3)
    with pytest.raises(ValueError, match="distinct values"):
        select_validation(df, "y", time_col="day", n_splits=5)
