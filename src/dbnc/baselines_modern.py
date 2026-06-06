"""Modern tabular baselines and the shared fit/predict interface.

The boosted-tree baselines (XGBoost, CatBoost, LightGBM) and the neural
``FTTransformerClassifier`` are fitted through ``fit_predictor``, which returns a
``FittedPredictor`` exposing ``predict_proba`` plus timing and metadata. Their
optional dependencies are imported lazily inside ``fit_predictor``.
"""
from dbnc._pipeline import (  # noqa: F401
    FittedPredictor,
    FTTransformerClassifier,
    FTTransformerNetwork,
    categorical_tree_frame,
    fit_predictor,
    lightgbm_tree_frame,
)

__all__ = [
    "fit_predictor",
    "FittedPredictor",
    "FTTransformerClassifier",
    "FTTransformerNetwork",
    "categorical_tree_frame",
    "lightgbm_tree_frame",
]
