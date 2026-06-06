"""Tests for the classical Bayesian-network baselines and structure discovery."""
import numpy as np

from dbnc.baselines import (
    ChowLiuClassifier,
    ConditionalBNC,
    kdb_feature_parents,
    pairwise_mi,
    tan_feature_parents,
)


def _synthetic(n=200, cards=(3, 4, 2, 5), n_classes=3, seed=0):
    rng = np.random.default_rng(seed)
    X = np.stack([rng.integers(0, c, size=n) for c in cards], axis=1)
    y = rng.integers(0, n_classes, size=n)
    return X, y, list(cards), n_classes


def _valid_proba(proba, n, k):
    assert proba.shape == (n, k)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert (proba >= 0).all()


def test_naive_bayes():
    X, y, cards, n_classes = _synthetic()
    clf = ConditionalBNC(feature_parents=[[] for _ in cards], alpha=1.0).fit(X, y, cards, n_classes)
    _valid_proba(clf.predict_proba(X), X.shape[0], n_classes)


def test_tan_structure_and_fit():
    X, y, cards, n_classes = _synthetic()
    parents = tan_feature_parents(X, y)
    assert all(len(p) <= 1 for p in parents)  # TAN: at most one feature parent
    clf = ConditionalBNC(feature_parents=parents, alpha=1.0).fit(X, y, cards, n_classes)
    _valid_proba(clf.predict_proba(X), X.shape[0], n_classes)


def test_kdb_parent_budget():
    X, y, _, _ = _synthetic()
    for k in (1, 2, 3):
        parents = kdb_feature_parents(X, y, k)
        assert all(len(p) <= k for p in parents)


def test_chow_liu():
    X, y, cards, n_classes = _synthetic()
    clf = ChowLiuClassifier(alpha=1.0).fit(X, y, cards, n_classes)
    _valid_proba(clf.predict_proba(X), X.shape[0], n_classes)


def test_pairwise_mi_shape():
    X, _, cards, _ = _synthetic()
    mi = pairwise_mi(X)
    assert mi.shape == (len(cards), len(cards))
