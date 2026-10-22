"""ResearchGuard ML: auditable machine-learning evaluation for tabular research data."""
__version__ = "0.1.2"

from .audit import audit_dataset
from .evaluate import compare_designs, evaluate_models
from .reporting import build_report, to_markdown, write_report
from .validation import select_validation, verify_plan

__all__ = ["audit_dataset", "select_validation", "verify_plan", "evaluate_models",
           "compare_designs", "build_report", "to_markdown", "write_report"]
