import json

import pandas as pd

from researchguard.cli import main
from researchguard.reporting import build_report, to_markdown, write_report
from researchguard.synthetic import make_grouped_data


def test_report_is_complete_and_reproducible(tmp_path):
    df = make_grouped_data(0, leaky=True)
    r1 = build_report(df, "outcome", group_col="group_id", subgroup_cols=["x"])
    r2 = build_report(df, "outcome", group_col="group_id", subgroup_cols=["x"])
    assert r1["data_fingerprint"] == r2["data_fingerprint"]
    assert (r1["evaluation"]["models"]["random_forest"]["summary"]
            == r2["evaluation"]["models"]["random_forest"]["summary"])
    jp, mp = write_report(r1, tmp_path)
    assert json.loads(jp.read_text())["config"]["seed"] == 42
    md = mp.read_text()
    assert "Interpretation limits" in md and "Optimism" in md


def test_markdown_states_causal_disclaimer():
    r = build_report(make_grouped_data(0, leaky=False).drop(columns="group_id"), "outcome")
    assert "causal" in to_markdown(r)


def test_cli_end_to_end(tmp_path, capsys):
    csv = tmp_path / "d.csv"
    make_grouped_data(0, leaky=True).to_csv(csv, index=False)
    assert main(["audit", str(csv), "--target", "outcome"]) == 0
    out = tmp_path / "rep"
    assert main(["run", str(csv), "--target", "outcome", "--group-col", "group_id",
                 "--out", str(out)]) == 0
    assert (out / "report.json").exists() and (out / "report.md").exists()
