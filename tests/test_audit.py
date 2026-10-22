import numpy as np
import pandas as pd
import pytest

from researchguard.audit import audit_dataset


def codes(r):
    return {w["code"] for w in r["warnings"]}


def test_basic_dataset_summary():
    df = pd.DataFrame({"age": [20, 30, 40], "outcome": [0, 1, 0]})
    r = audit_dataset(df, target="outcome")
    assert r["n_rows"] == 3 and r["n_columns"] == 2
    assert r["target_summary"]["n_classes"] == 2


def test_detects_missing_values():
    df = pd.DataFrame({"age": [20, None, 40], "outcome": [0, 1, 0]})
    r = audit_dataset(df, target="outcome")
    assert r["missing_values"]["age"] == 1
    assert "MISSING_VALUES" in codes(r)


def test_rejects_missing_target_column():
    with pytest.raises(ValueError):
        audit_dataset(pd.DataFrame({"a": [1]}), target="outcome")


def test_detects_duplicates_and_constants():
    df = pd.DataFrame({"a": [1, 1, 2], "c": [5, 5, 5], "y": [0, 0, 1]})
    r = audit_dataset(df.iloc[[0, 0, 2]], target="y")
    assert {"DUPLICATE_ROWS", "CONSTANT_COLUMNS"} <= codes(r)


def test_flags_possible_leakage_for_confirmation():
    y = [0, 1] * 30
    df = pd.DataFrame({"leak": [v * 10 + 1 for v in y],
                       "noise": range(60), "y": y})
    r = audit_dataset(df, target="y")
    w = next(w for w in r["warnings"] if w["code"] == "POSSIBLE_LEAKAGE")
    assert "leak" in w["details"] and w["needs_user_confirmation"]


def _clean(n=400, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = rng.binomial(1, 1 / (1 + np.exp(-1.5 * x)))
    return pd.DataFrame({"age": rng.normal(40, 10, n).round(1), "x": x, "y": y})


def test_clean_data_raises_no_warnings():
    assert audit_dataset(_clean(), target="y")["warnings"] == []


def test_strong_but_plausible_predictor_not_flagged_as_leakage():
    codes = [w["code"] for w in audit_dataset(_clean(), target="y")["warnings"]]
    assert "POSSIBLE_LEAKAGE" not in codes  # AUC around 0.8 is a real signal, not leakage


def test_declared_group_column_not_flagged_as_identifier():
    df = _clean().assign(patient_id=np.repeat(np.arange(100), 4))
    flagged = audit_dataset(df, target="y")
    declared = audit_dataset(df, target="y", declared_columns=["patient_id"])
    assert "IDENTIFIER_COLUMNS" in [w["code"] for w in flagged["warnings"]]
    assert "IDENTIFIER_COLUMNS" not in [w["code"] for w in declared["warnings"]]
