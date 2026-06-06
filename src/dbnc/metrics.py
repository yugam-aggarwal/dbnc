"""Predictive and probabilistic-quality metrics.

``predictive_metrics`` computes accuracy, macro-F1, AUC (binary or macro
one-vs-rest), log-loss, Brier score, and expected calibration error (ECE) from
true labels and predicted class probabilities.
"""
from dbnc._pipeline import predictive_metrics  # noqa: F401

__all__ = ["predictive_metrics"]
