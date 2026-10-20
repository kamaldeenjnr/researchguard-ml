# Contributing

1. `pip install -e ".[dev]"` and run `pytest -q` before and after your change.
2. Every new check needs a test that shows it firing **and** a test showing it staying quiet on
   clean data (false alarms matter as much as misses).
3. Never commit real patient or confidential data. Use synthetic or openly licensed data.
4. Warnings that depend on study-design knowledge must set `needs_user_confirmation`.
5. Open an issue first for large changes (new model types, new validation designs).

Good first contributions: a real-data benchmark in `benchmarks/datasets.yaml`, additional
subgroup/uncertainty metrics, a combined group + temporal validation design.
