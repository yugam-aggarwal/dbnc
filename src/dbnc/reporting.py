"""Aggregation and statistical analysis of benchmark results.

``generate_reports`` turns the per-run JSON records into the aggregated CSVs and
the ``main_statistical_tests.json`` (Friedman omnibus + Nemenyi post-hoc, plus
the per-baseline paired Wilcoxon tests) that back the paper's tables and figures.
"""
from dbnc._pipeline import (  # noqa: F401
    generate_reports,
    read_completed_results,
    records_frame,
)

__all__ = ["generate_reports", "records_frame", "read_completed_results"]
