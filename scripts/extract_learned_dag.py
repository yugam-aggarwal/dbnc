#!/usr/bin/env python3
"""Render a representative DBNC learned dependency graph for the paper.

The saved benchmark artifacts store only summary graph diagnostics, not the
adjacency matrices, so the learned structure must be re-derived by retraining a
DBNC model.  We retrain on the small, fully-interpretable ``car`` dataset
(6 named categorical predictors) and extract ``model.adjacency()``.

Reproducibility note: the recovered graph-tuned hyperparameter artifact for the
paper's v4 search space (``dbnc_v4_predictive_attention_graph_bins``) is used
directly.  For the car dataset this is ``hp_id=c28bc0ca22ecd67e``; the
corresponding development runs report thresholded edge density around 0.43.

Output: ``paper/figures/learned_dag_car.pdf``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "dbnc-dag-matplotlib"),
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))  # use the local package when not pip-installed
import paper_style as ps  # noqa: E402

DATASET_ID = 40975          # car
DATASET_ALIAS = "car"
SEED = 0
V4_SEARCH_SPACE_VERSION = "dbnc_v4_predictive_attention_graph_bins"


def train_and_extract():
    import torch

    from dbnc import DBNC
    from dbnc.config import GRAPH_THRESHOLD, N_BINS
    from dbnc.data import configure_openml_cache, load_openml_dataset, prepare_split
    from dbnc.variants import dbnc_variant_parameters

    warnings.filterwarnings("ignore")
    configure_openml_cache(ROOT)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    cfg_path = (
        ROOT
        / "configs"
        / "tuning"
        / "dbnc"
        / "full"
        / V4_SEARCH_SPACE_VERSION
        / f"{DATASET_ID}.json"
    )
    if not cfg_path.exists():
        raise FileNotFoundError(f"Recovered v4 tuning artifact not found: {cfg_path}")
    cfg = dict(json.loads(cfg_path.read_text())["selected_configuration"])
    numeric_bins = int(cfg.get("numeric_bins", N_BINS))

    bundle = load_openml_dataset(DATASET_ALIAS, DATASET_ID)
    prepared = prepare_split(ROOT, bundle, SEED, numeric_bins=numeric_bins)
    params = dbnc_variant_parameters("full", cfg, prepared.train.discrete, prepared.y_train)
    model = DBNC(prepared.cards, prepared.n_classes, device=device, **params)
    model.fit(prepared.train.discrete, prepared.y_train,
              prepared.validation.discrete, prepared.y_validation)

    adjacency = model.adjacency().detach().cpu().numpy()
    rank = model.rank.detach().cpu().numpy()
    names = list(prepared.transform_metadata["columns"])
    diagnostics = model.graph_diagnostics()
    return adjacency, rank, names, diagnostics, GRAPH_THRESHOLD


def build_graph(adjacency, rank, names, threshold):
    """adjacency[i, j] is the weight of edge parent i -> child j (rank[i] < rank[j])."""
    m = adjacency.shape[0]
    g = nx.DiGraph()
    g.add_nodes_from(range(m))
    for parent in range(m):
        for child in range(m):
            w = float(adjacency[parent, child])
            if child != parent and w > threshold:
                g.add_edge(parent, child, weight=w)
    assert nx.is_directed_acyclic_graph(g), "learned graph is not acyclic"
    # sanity: every edge points from lower learned rank to higher
    assert all(rank[p] <= rank[c] for p, c in g.edges()), "edge orientation inconsistent with rank"
    return g


def layered_positions(g, rank):
    """Longest-path (Sugiyama-style) layering; parents on top, children below."""
    layer = {}
    for node in nx.topological_sort(g):
        preds = list(g.predecessors(node))
        layer[node] = 0 if not preds else 1 + max(layer[p] for p in preds)
    by_layer = {}
    for node, lv in layer.items():
        by_layer.setdefault(lv, []).append(node)
    pos = {}
    n_layers = max(by_layer) + 1
    for lv, nodes in by_layer.items():
        nodes = sorted(nodes, key=lambda n: -rank[n])  # stable, parents-first within layer
        k = len(nodes)
        xs = np.linspace(-1.0, 1.0, k) if k > 1 else [0.0]
        for x, node in zip(xs, nodes):
            pos[node] = (float(x), float(n_layers - 1 - lv))
    return pos


def render(adjacency, rank, names, diagnostics, threshold, out):
    ps.apply_style()
    g = build_graph(adjacency, rank, names, threshold)
    pos = layered_positions(g, rank)

    fig, ax = plt.subplots(figsize=(ps.COL_W, 2.9))
    node_fill = "#D6E8F5"
    node_edge = ps.GROUP_COLORS["Classical BNC"]

    weights = np.array([g[u][v]["weight"] for u, v in g.edges()])
    wmax = weights.max() if weights.size else 1.0
    for (u, v) in g.edges():
        w = g[u][v]["weight"]
        frac = w / wmax
        ax.annotate(
            "", xy=pos[v], xytext=pos[u],
            arrowprops=dict(
                arrowstyle="-|>", color="#333333", alpha=0.35 + 0.55 * frac,
                lw=0.6 + 2.0 * frac, shrinkA=11, shrinkB=11,
                connectionstyle="arc3,rad=0.08",
            ),
            zorder=1,
        )
    for node, (x, y) in pos.items():
        ax.scatter([x], [y], s=520, color=node_fill, edgecolors=node_edge,
                   linewidths=1.0, zorder=2)
        ax.text(x, y, names[node], ha="center", va="center", fontsize=6.3, zorder=3)

    ax.set_axis_off()
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    ax.set_xlim(min(xs) - 0.45, max(xs) + 0.45)
    ax.set_ylim(min(ys) - 0.4, max(ys) + 0.4)
    n_edges = g.number_of_edges()
    fig.savefig(out)
    plt.close(fig)
    return out, n_edges


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paper", type=Path, default=ROOT / "paper")
    args = p.parse_args(argv)
    out = args.paper / "figures" / "learned_dag_car.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)

    adjacency, rank, names, diagnostics, threshold = train_and_extract()
    out, n_edges = render(adjacency, rank, names, diagnostics, threshold, out)
    print(f"wrote {out}  edges={n_edges}  density={diagnostics['thresholded_edge_density']:.3f}"
          f"  acyclic={diagnostics['acyclic']}")


if __name__ == "__main__":
    main()
