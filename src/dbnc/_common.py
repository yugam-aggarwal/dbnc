"""Shared leaf utilities for the :mod:`dbnc` package.

These are small, dependency-light helpers used by *both* the model
(:mod:`dbnc.model`) and the experiment pipeline (:mod:`dbnc._pipeline`).
Keeping them in one place gives a single source of truth and avoids a circular
import between the model and the pipeline.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "ExperimentError",
    "MissingDependency",
    "GRAPH_THRESHOLD",
    "is_acyclic",
    "validation_checkpoint_improved",
]

# Edge-weight above which a soft-adjacency entry is counted as a realised edge
# when computing graph diagnostics.
GRAPH_THRESHOLD = 0.05


class ExperimentError(RuntimeError):
    """Raised for an invalid or incomplete experimental operation."""


class MissingDependency(ExperimentError):
    """Raised when execution needs an optional package that is unavailable."""


def is_acyclic(adjacency: np.ndarray) -> bool:
    """Return ``True`` iff the directed graph induced by ``adjacency > 0`` is a DAG.

    Uses Kahn's algorithm (repeated removal of zero-in-degree nodes); the graph
    is acyclic iff every node is visited.
    """
    adjacency = np.asarray(adjacency) > 0
    incoming = adjacency.sum(axis=0).astype(int)
    queue = [node for node, degree in enumerate(incoming) if degree == 0]
    visited = 0
    while queue:
        node = queue.pop()
        visited += 1
        for child in np.flatnonzero(adjacency[node]):
            incoming[child] -= 1
            if incoming[child] == 0:
                queue.append(int(child))
    return visited == adjacency.shape[0]


def validation_checkpoint_improved(
    accuracy: float,
    log_loss_value: float,
    best_accuracy: float,
    best_log_loss: float,
) -> bool:
    """Validation-checkpoint rule: higher accuracy wins, log-loss breaks ties."""
    if accuracy > best_accuracy + 1e-12:
        return True
    return abs(accuracy - best_accuracy) <= 1e-12 and log_loss_value < best_log_loss - 1e-12
