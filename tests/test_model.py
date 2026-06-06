"""Tests for the DBNC model: acyclicity, fit/predict, determinism."""
import numpy as np
import torch

from dbnc import DBNC
from dbnc.adjacency import GRAPH_THRESHOLD, is_acyclic


def _synthetic(n=64, cards=(3, 4, 2, 5), n_classes=3, seed=0):
    rng = np.random.default_rng(seed)
    X = np.stack([rng.integers(0, c, size=n) for c in cards], axis=1)
    y = rng.integers(0, n_classes, size=n)
    return X, y, list(cards), n_classes


def test_learned_adjacency_is_acyclic():
    """The support of the rank-parameterised soft adjacency is always a DAG."""
    for seed in range(5):
        torch.manual_seed(seed)
        model = DBNC(cards=[3, 4, 2, 5, 6], n_classes=3, edge_logit_mean=0.0, rank_init_std=1.0)
        adjacency = model.adjacency().detach().cpu().numpy()
        assert is_acyclic(adjacency > 0)
        assert is_acyclic(adjacency > GRAPH_THRESHOLD)


def test_fit_predict_smoke():
    X, y, cards, n_classes = _synthetic()
    model = DBNC(cards=cards, n_classes=n_classes, d=8, n_heads=2, epochs=3, dropout=0.0)
    model.fit(X, y, X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (X.shape[0], n_classes)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert (proba >= 0).all()


def test_determinism():
    """Same seed and configuration reproduce identical predictions."""
    X, y, cards, n_classes = _synthetic()

    def run_once():
        torch.manual_seed(0)
        np.random.seed(0)
        model = DBNC(cards=cards, n_classes=n_classes, d=8, n_heads=2, epochs=3, dropout=0.0)
        model.fit(X, y, X, y)
        return model.predict_proba(X)

    assert np.allclose(run_once(), run_once())


def test_graph_diagnostics():
    model = DBNC(cards=[3, 4, 2], n_classes=2)
    diagnostics = model.graph_diagnostics()
    assert diagnostics["acyclic"] is True
    assert 0.0 <= diagnostics["thresholded_edge_density"] <= 1.0


def test_mlp_cpd_variant():
    X, y, cards, n_classes = _synthetic()
    model = DBNC(cards=cards, n_classes=n_classes, d=8, n_heads=2, epochs=2, cpd_type="mlp", dropout=0.0)
    model.fit(X, y, X, y)
    assert model.predict_proba(X).shape == (X.shape[0], n_classes)
