import pytest
import pandas as pd

from researchguard.evaluate import compare_designs, evaluate_models
from researchguard.synthetic import make_grouped_data, make_temporal_data
from researchguard.validation import select_validation


def test_evaluate_returns_all_models_and_metrics():
    df = make_grouped_data(0, leaky=False)
    plan = select_validation(df, "outcome")
    res = evaluate_models(df.drop(columns="group_id"), "outcome", plan, subgroup_cols=["x"])
    assert set(res["models"]) == {"logistic_regression", "random_forest", "gradient_boosting"}
    m = res["models"]["random_forest"]
    assert res["n_folds"] == 5 and "ece" in m["calibration"] and "x" in m["subgroups"]


def test_group_column_never_used_as_feature():
    df = make_grouped_data(0, leaky=True)
    plan = select_validation(df, "outcome", group_col="group_id")
    res = evaluate_models(df, "outcome", plan)
    assert "group_id" not in res["features"]


def test_random_cv_is_optimistic_on_leaky_grouped_data():
    gaps = []
    for s in range(3):
        r = compare_designs(make_grouped_data(s, leaky=True), "outcome", group_col="group_id", seed=s)
        gaps.append(r["models"]["random_forest"]["optimism"])
        assert r["naive_plan_violations"]["group_overlap"] > 0
    assert sum(gaps) / len(gaps) > 0.08


def test_control_shows_no_meaningful_optimism():
    gaps = []
    for s in range(3):
        r = compare_designs(make_grouped_data(s, leaky=False), "outcome", group_col="group_id", seed=s)
        gaps.append(r["models"]["random_forest"]["optimism"])
    assert abs(sum(gaps) / len(gaps)) < 0.05


def test_temporal_drift_detected():
    r = compare_designs(make_temporal_data(0, drift=True), "outcome", time_col="visit_date")
    assert r["models"]["gradient_boosting"]["optimism"] > 0.05


def test_compare_not_applicable_without_structure():
    df = make_grouped_data(0, leaky=False).drop(columns="group_id")
    assert compare_designs(df, "outcome")["applicable"] is False


def test_string_labels_and_categoricals_work():
    df = make_grouped_data(0, leaky=False).drop(columns="group_id")
    df["outcome"] = df["outcome"].map({0: "no", 1: "yes"})
    df["region"] = ["north", "south", "east", "west"] * (len(df) // 4)
    plan = select_validation(df, "outcome")
    res = evaluate_models(df, "outcome", plan)
    assert res["positive_class"] == "yes"


def test_positive_class_can_be_chosen():
    from researchguard.validation import select_validation
    df = make_grouped_data(seed=1, leaky=False)
    df["outcome"] = df["outcome"].map({0: "survived", 1: "died"})
    plan = select_validation(df, "outcome", n_splits=3)
    default = evaluate_models(df, "outcome", plan, drop_columns=["group_id"])
    chosen = evaluate_models(df, "outcome", plan, drop_columns=["group_id"], positive_class="died")
    assert default["positive_class"] == "survived"  # alphabetical default
    assert chosen["positive_class"] == "died"
    a = default["models"]["logistic_regression"]["summary"]["roc_auc"]["mean"]
    b = chosen["models"]["logistic_regression"]["summary"]["roc_auc"]["mean"]
    assert abs(a - b) < 1e-3  # AUC barely changes when the label is flipped
    with pytest.raises(ValueError, match="positive_class"):
        evaluate_models(df, "outcome", plan, positive_class="unknown")
