from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from .audit import audit_dataset
from .reporting import build_report, write_report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="researchguard",
                                description="Auditable ML evaluation for research data")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("audit", help="Data-quality audit")
    a.add_argument("csv"); a.add_argument("--target", required=True)
    r = sub.add_parser("run", help="Full evaluation report")
    r.add_argument("csv"); r.add_argument("--target", required=True)
    r.add_argument("--group-col"); r.add_argument("--time-col")
    r.add_argument("--subgroup", action="append", default=[])
    r.add_argument("--drop", action="append", default=[], help="Column to exclude from features")
    r.add_argument("--n-splits", type=int, default=5)
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--out", default="researchguard_report")
    args = p.parse_args(argv)

    df = pd.read_csv(args.csv)
    if args.cmd == "audit":
        print(json.dumps(audit_dataset(df, args.target), indent=2, default=str))
        return 0
    report = build_report(df, args.target, args.group_col, args.time_col, args.subgroup,
                          args.drop, args.n_splits, args.seed)
    jp, mp = write_report(report, args.out)
    print(f"Wrote {jp} and {mp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
