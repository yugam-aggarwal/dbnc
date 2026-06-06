"""Hyperparameter tuning with Optuna (TPE).

``suggest_configuration`` defines the per-method search spaces; ``tune_method``
runs 20 TPE trials per dataset and selects the configuration with the highest
validation accuracy, breaking ties by lower validation log-loss. Optuna is
imported lazily inside ``tune_method``.
"""
from dbnc._pipeline import (  # noqa: F401
    result_hp_id,
    suggest_configuration,
    tune_method,
    tuning_path,
    tuning_search_space_version,
)

__all__ = [
    "suggest_configuration",
    "tune_method",
    "tuning_path",
    "tuning_search_space_version",
    "result_hp_id",
]
