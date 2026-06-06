"""Benchmark orchestration: fit, evaluate, and time each method per seed.

``final_evaluation`` fits one (method, dataset, seed) with its tuned
configuration and records predictive metrics, timing, and (for DBNC) graph
diagnostics; ``run_benchmark`` iterates over datasets x methods x seeds.
"""
from dbnc._pipeline import (  # noqa: F401
    environment_metadata,
    final_evaluation,
    read_completed_results,
    result_path,
    run_benchmark,
    set_seed,
)

__all__ = [
    "final_evaluation",
    "run_benchmark",
    "result_path",
    "read_completed_results",
    "environment_metadata",
    "set_seed",
]
