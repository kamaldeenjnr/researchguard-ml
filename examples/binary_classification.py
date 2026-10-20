"""Minimal example on synthetic repeated-measures data (no real patient data)."""
from researchguard import build_report, to_markdown
from researchguard.synthetic import make_grouped_data

df = make_grouped_data(seed=0, leaky=True)
report = build_report(df, target="outcome", group_col="group_id", subgroup_cols=["x"])
print(to_markdown(report))
