# Benchmark protocol

## Completed: synthetic (benchmarks/run_benchmarks.py)
Four scenarios (two hazards, two controls) × three models × N seeds. Reports mean/SD/range
of optimism. Controls estimate the false-alarm side.

## Required before any methodological claim: real data
1. Select public datasets with documented grouping or time structure (OpenML tasks, licensed
   public-health surveys). Record licence and source in `benchmarks/datasets.yaml`.
2. Pre-register metrics: optimism per model, fold-SD, runtime, false-alarm rate on datasets
   without structure, missed-risk rate where structure is documented.
3. Fix seeds and fold counts in advance; report all datasets, including failures.
4. Compare against: random CV alone, design-aware CV chosen by an expert, ResearchGuard's choice.
5. Publish code, configs, environment versions and raw results.
