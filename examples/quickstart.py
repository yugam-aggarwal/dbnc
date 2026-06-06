"""Minimal DBNC usage on synthetic categorical data (no downloads required).

    python examples/quickstart.py

For a real OpenML benchmark dataset, see the commented snippet at the bottom.
"""
import numpy as np
import torch

from dbnc import DBNC


def main() -> None:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    cards = [3, 4, 2, 5, 3]          # cardinality of each categorical feature
    n_classes = 3
    n = 600
    X = np.stack([rng.integers(0, c, size=n) for c in cards], axis=1)
    y = rng.integers(0, n_classes, size=n)
    Xtr, ytr = X[:400], y[:400]
    Xva, yva = X[400:500], y[400:500]
    Xte, yte = X[500:], y[500:]

    model = DBNC(cards=cards, n_classes=n_classes, d=16, n_heads=2,
                 epochs=50, hybrid_weight=0.1, dropout=0.1)
    model.fit(Xtr, ytr, Xva, yva)

    proba = model.predict_proba(Xte)
    print(f"test accuracy: {(proba.argmax(1) == yte).mean():.3f}")
    print("learned adjacency (rows = parents, cols = children):")
    print(np.round(model.adjacency().detach().cpu().numpy(), 2))
    print("graph diagnostics:", model.graph_diagnostics())


# --- Real OpenML dataset (needs `pip install dbnc[experiments]` and a network) -------
# from pathlib import Path
# from dbnc.data import load_openml_dataset, prepare_split
# bundle = load_openml_dataset("car", 40975)
# prepared = prepare_split(Path("."), bundle, seed=0)
# model = DBNC(prepared.cards, prepared.n_classes, d=32, epochs=100)
# model.fit(prepared.train.discrete, prepared.y_train,
#           prepared.validation.discrete, prepared.y_validation)
# acc = (model.predict_proba(prepared.test.discrete).argmax(1) == prepared.y_test).mean()
# print(f"car test accuracy: {acc:.3f}")


if __name__ == "__main__":
    main()
