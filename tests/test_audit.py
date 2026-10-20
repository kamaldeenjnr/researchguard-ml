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
