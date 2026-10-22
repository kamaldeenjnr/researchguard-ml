"""Minimal example on synthetic repeated-measures data (no real patient data).

Run:  python examples/binary_classification.py
Writes examples/output/report.md and report.json (the committed copy shows what to expect).
"""
from pathlib import Path

from researchguard import build_report, write_report
from researchguard.synthetic import make_grouped_data

df = make_grouped_data(seed=0, leaky=True)
report = build_report(df, target="outcome", group_col="group_id", subgroup_cols=["x"], seed=42)
json_path, md_path = write_report(report, Path(__file__).parent / "output")
print(md_path.read_text(encoding="utf-8"))
