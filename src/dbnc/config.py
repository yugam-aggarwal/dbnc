"""Fixed experimental protocol: constants, method registry, dataset registry.

All values mirror the protocol used for the paper. They are re-exported from the
pipeline so there is a single source of truth.
"""
from dbnc._pipeline import (  # noqa: F401
    ALIAS_TO_ID,
    CORE_ABLATIONS,
    DATASETS,
    DBNC_MODEL_PARAMETER_KEYS,
    DBNC_SEARCH_SPACE_VERSION,
    DBNC_VARIANTS,
    ECE_BINS,
    GRAPH_THRESHOLD,
    ID_TO_ALIAS,
    MAIN_METHODS,
    METHOD_LABELS,
    MODES,
    N_BINS,
    RARE_MIN_COUNT,
    SEEDS,
)

__all__ = [
    "SEEDS",
    "N_BINS",
    "RARE_MIN_COUNT",
    "ECE_BINS",
    "GRAPH_THRESHOLD",
    "DBNC_SEARCH_SPACE_VERSION",
    "MODES",
    "DATASETS",
    "ALIAS_TO_ID",
    "ID_TO_ALIAS",
    "MAIN_METHODS",
    "METHOD_LABELS",
    "DBNC_VARIANTS",
    "CORE_ABLATIONS",
    "DBNC_MODEL_PARAMETER_KEYS",
]
