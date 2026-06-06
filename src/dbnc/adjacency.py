"""Graph utilities for the rank-acyclic soft adjacency.

``is_acyclic`` checks that the support of a (thresholded) adjacency matrix is a
DAG; ``GRAPH_THRESHOLD`` is the edge-weight cutoff used for graph diagnostics.
The continuous soft adjacency itself is produced by ``DBNC.learned_adjacency``
(see :mod:`dbnc.model`).
"""
from dbnc._common import GRAPH_THRESHOLD, is_acyclic  # noqa: F401

__all__ = ["GRAPH_THRESHOLD", "is_acyclic"]
