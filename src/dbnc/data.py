"""Data layer: OpenML loading, training-only preprocessing, and splits.

All transformations (rare-category grouping, missing-value imputation, quantile
discretisation of numeric features, standardisation) are fitted on the training
partition only. ``prepare_split`` returns a :class:`PreparedSplit` with the
discrete / tree / neural feature views used by the different model families.
"""
from dbnc._pipeline import (  # noqa: F401
    ALIAS_TO_ID,
    DATASETS,
    DatasetBundle,
    ID_TO_ALIAS,
    MatrixParts,
    PreparedSplit,
    TrainingPreprocessor,
    class_covering_training_subset,
    configure_openml_cache,
    load_openml_dataset,
    persisted_split,
    prepare_split,
    split_indices,
)

__all__ = [
    "DATASETS",
    "ALIAS_TO_ID",
    "ID_TO_ALIAS",
    "DatasetBundle",
    "MatrixParts",
    "PreparedSplit",
    "TrainingPreprocessor",
    "load_openml_dataset",
    "configure_openml_cache",
    "split_indices",
    "class_covering_training_subset",
    "persisted_split",
    "prepare_split",
]
