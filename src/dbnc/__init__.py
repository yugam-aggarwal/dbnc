"""DBNC: a Differentiable Bayesian Network Classifier for categorical tabular data.

Quick start::

    from dbnc import DBNC

    model = DBNC(cards=[3, 4, 2], n_classes=2)
    model.fit(X_train, y_train, X_validation, y_validation)
    proba = model.predict_proba(X_test)

The experiment pipeline that reproduces the paper lives in the submodules
:mod:`dbnc.data` (OpenML loading + preprocessing), :mod:`dbnc.baselines` and
:mod:`dbnc.baselines_modern` (the eleven baselines), :mod:`dbnc.metrics`,
:mod:`dbnc.tuning` (Optuna search), :mod:`dbnc.benchmark`, and
:mod:`dbnc.reporting`. See the README for end-to-end commands.
"""
from __future__ import annotations

from dbnc.model import DBNC

__version__ = "1.0.0"
__all__ = ["DBNC", "__version__"]
