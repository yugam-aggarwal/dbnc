#!/usr/bin/env python3
"""Generate the publication figures for the DBNC paper as vector PDFs.

Produces the paper figures into ``paper/figures/`` from the aggregated benchmark
artifacts in ``reports/`` and the ablation summary in ``development/``:

  dbnc_architecture.pdf             -- single-column DBNC architecture diagram
  critical_difference_accuracy.pdf  -- Demsar CD diagram (with clique bars)
  paired_accuracy_deltas.pdf        -- DBNC-minus-comparator deltas + CIs + W/T/L
  rank_heatmap_accuracy.pdf         -- per-dataset accuracy ranks
  calibration_metrics.pdf           -- probabilistic-quality distributions
  diagnostic_runtime_summary.pdf    -- graph, runtime, and categorical summaries
  efficiency_pareto.pdf             -- accuracy vs training cost (log time)
  ablation_summary.pdf              -- single-column hybrid-weight sweep

All figures share ``scripts/paper_style.py`` for typography and colour, so they form
one consistent, colour-blind-safe, vector identity that matches the IEEE body text.

Usage:
    python3 scripts/make_paper_figures.py \
        --reports reports \
        --ablation development/reports_ablation_attention_5 \
        --paper paper
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "dbnc-paper-matplotlib"),
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_style as ps  # noqa: E402

CLASSICAL = ["NB", "Chow-Liu", "TAN", "BAN", "KDB-1", "KDB-2", "KDB-3"]
MODERN = ["XGBoost", "CatBoost", "LightGBM", "FT-Transformer"]
METRIC_COLORS = {"accuracy": "#0072B2", "log_loss": "#E69F00", "ece": "#009E73"}
CALIBRATION_METRICS = [("log_loss", "Log-loss"), ("brier", "Brier"), ("ece", "ECE")]


def _box(ax, xy, wh, text, fc="#F7F7F7", ec="#333333", lw=0.8, fontsize=6.1,
         color="#111111", weight="normal"):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.008,rounding_size=0.015",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=color,
        fontweight=weight,
        linespacing=1.1,
    )
    return patch


def _arrow(ax, start, end, color="#333333", lw=0.9, rad=0.0):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=7.5,
        linewidth=lw,
        color=color,
        shrinkA=1,
        shrinkB=1,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arr)
    return arr


def _poly_arrow(ax, points, color="#333333", lw=0.9):
    for start, end in zip(points[:-2], points[1:-1]):
        ax.plot([start[0], end[0]], [start[1], end[1]], color=color, lw=lw)
    arr = FancyArrowPatch(
        points[-2],
        points[-1],
        arrowstyle="-|>",
        mutation_scale=7.5,
        linewidth=lw,
        color=color,
        shrinkA=1,
        shrinkB=1,
    )
    ax.add_patch(arr)
    return arr


def accuracy_rank_order(summary: pd.DataFrame) -> list[str]:
    return (
        summary.sort_values("mean_rank_accuracy", na_position="last")["method_label"]
        .astype(str)
        .tolist()
    )


def filtered_graph(graph: pd.DataFrame, characteristics: pd.DataFrame) -> pd.DataFrame:
    paper_ids = set(characteristics["dataset_id"].astype(int))
    result = graph[graph["dataset_id"].astype(int).isin(paper_ids)].copy()
    result = result.merge(
        characteristics[["dataset_id", "dataset_alias", "n_predictors", "categorical_ratio"]],
        on="dataset_id",
        how="left",
    )
    result["thresholded_edges"] = (
        result["thresholded_edge_density"]
        * result["n_predictors"]
        * (result["n_predictors"] - 1)
    )
    return result


# ---------------------------------------------------------------------------
# F8 -- architecture diagram
# ---------------------------------------------------------------------------
def fig_architecture(out):
    blue = ps.GROUP_COLORS["Classical BNC"]
    green = ps.GROUP_COLORS["Boosted tree"]
    orange = ps.GROUP_COLORS["DBNC"]
    purple = ps.GROUP_COLORS["Neural tabular"]
    grey = "#333333"

    fig, ax = plt.subplots(figsize=(ps.COL_W, 3.7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def stage(y, text):
        ax.text(
            0.5,
            y,
            text,
            ha="center",
            va="center",
            fontsize=5.25,
            color="#4A4A4A",
            fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", pad=0.8),
            zorder=8,
        )

    stage(0.965, "DBNC architecture")

    # Top row: the three inputs to one local conditional distribution.
    _box(
        ax,
        (0.100, 0.835),
        (0.220, 0.105),
        "Features\n"
        "$\\mathbf{x}_n$",
        fc="#EEF6FF",
        ec=blue,
        fontsize=5.45,
    )
    _box(
        ax,
        (0.390, 0.835),
        (0.220, 0.105),
        "Class/target\n"
        "$y,\\ X_j$",
        fc="#F4EEFF",
        ec=purple,
        fontsize=5.45,
    )
    _box(
        ax,
        (0.680, 0.835),
        (0.220, 0.105),
        "Ranked graph\n"
        "$\\mathbf{r},\\mathbf{L}$",
        fc="#FFF5E6",
        ec=orange,
        fontsize=5.45,
    )

    # Middle row: transform each input branch into attention operands.
    _box(
        ax,
        (0.100, 0.635),
        (0.220, 0.105),
        "Keys/values\n"
        "$\\mathbf{k}_{n,i},\\mathbf{v}_{n,i}$",
        fc="#EEF6FF",
        ec=blue,
        fontsize=5.35,
    )
    _box(
        ax,
        (0.390, 0.635),
        (0.220, 0.105),
        "Query\n"
        "$\\mathbf{q}_{y,j}$",
        fc="#F4EEFF",
        ec=purple,
        fontsize=5.45,
    )
    _box(
        ax,
        (0.680, 0.635),
        (0.220, 0.105),
        "Soft DAG\n"
        "$\\mathbf{A}$ mask",
        fc="#FFF5E6",
        ec=orange,
        fontsize=5.45,
    )

    _arrow(ax, (0.210, 0.835), (0.210, 0.740), blue)
    _arrow(ax, (0.500, 0.835), (0.500, 0.740), purple)
    _arrow(ax, (0.790, 0.835), (0.790, 0.740), orange)

    _box(
        ax,
        (0.14, 0.430),
        (0.72, 0.105),
        "Graph-gated attention over admissible predecessors\n"
        "keys/values + query, masked by $\\mathbf{A}$",
        fc="#F2FBF6",
        ec=green,
        fontsize=5.2,
    )
    _arrow(ax, (0.210, 0.635), (0.34, 0.535), blue, rad=-0.08)
    _arrow(ax, (0.500, 0.635), (0.50, 0.535), purple)
    _arrow(ax, (0.790, 0.635), (0.66, 0.535), orange, rad=0.08)

    _box(
        ax,
        (0.22, 0.290),
        (0.56, 0.090),
        "Neural CPD\n"
        "$p_\\theta(X_j\\mid\\mathbf{x}_n,y,\\mathbf{A})$",
        fc="#F2FBF6",
        ec=green,
        fontsize=5.25,
    )
    _arrow(ax, (0.500, 0.430), (0.500, 0.380), green)

    _box(
        ax,
        (0.22, 0.155),
        (0.56, 0.080),
        "Class prior + feature factors\n"
        "$\\log p_\\theta(y,\\mathbf{x}_n)$",
        fc="#F7F7F7",
        ec=grey,
        fontsize=5.2,
    )
    _box(
        ax,
        (0.22, 0.015),
        (0.56, 0.080),
        "Normalize over classes\n"
        "$p_\\theta(y\\mid\\mathbf{x}_n),\\ \\hat{y}$",
        fc="#F7F7F7",
        ec=grey,
        fontsize=5.2,
    )
    _arrow(ax, (0.500, 0.290), (0.500, 0.235), green)
    _arrow(ax, (0.500, 0.155), (0.500, 0.095), grey)

    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# F1 -- critical-difference diagram
# ---------------------------------------------------------------------------
def _form_cliques(vals, cd):
    """Maximal contiguous rank intervals whose span is within the CD."""
    n = len(vals)
    reach = []
    for i in range(n):
        j = i
        while j + 1 < n and vals[j + 1] - vals[i] <= cd:
            j += 1
        reach.append(j)
    intervals = [(vals[i], vals[reach[i]]) for i in range(n) if reach[i] > i]
    maximal = []
    for a in set(intervals):
        if not any(b[0] <= a[0] and a[1] <= b[1] and b != a for b in intervals):
            maximal.append(a)
    return sorted(maximal)


def fig_cd(stats, out):
    acc = stats["accuracy"]
    ranks = acc["mean_ranks"]
    cd = float(acc["critical_difference"])
    methods = sorted(ranks, key=ranks.get)
    vals = [float(ranks[m]) for m in methods]
    n = len(methods)
    lo, hi = int(np.floor(min(vals))), int(np.ceil(max(vals)))

    fig, ax = plt.subplots(figsize=(ps.COL_W, 2.35))
    cline = 0.0          # y of the number line
    row_h = 0.50
    top_gap = 0.35
    half = (n + 1) // 2

    # number line + integer ticks
    ax.plot([lo, hi], [cline, cline], color="#444444", lw=1.0, zorder=2)
    for t in range(lo, hi + 1):
        ax.plot([t, t], [cline, cline + 0.10], color="#444444", lw=0.8, zorder=2)
        ax.text(t, cline + 0.16, str(t), ha="center", va="bottom", fontsize=6.4)

    def leader(rank, side, row, label, focal):
        y = cline - top_gap - (row + 1) * row_h
        edge = lo - 0.32 if side == "left" else hi + 0.32
        color = ps.GROUP_COLORS["DBNC"] if focal else "#222222"
        ax.plot([rank, rank], [cline, y], color=color, lw=0.7, zorder=3)
        ax.plot([rank, edge], [y, y], color=color, lw=0.7, zorder=3)
        ax.plot([rank], [cline], "o", ms=3.2, color=color, zorder=4)
        ha = "right" if side == "left" else "left"
        offs = -0.08 if side == "left" else 0.08
        weight = "bold" if focal else "normal"
        ax.text(edge + offs, y, ps.short(label), ha=ha, va="center",
                fontsize=6.6, color=color, fontweight=weight)

    left = methods[:half]
    right = methods[half:]
    for row, m in enumerate(left):
        leader(vals[methods.index(m)], "left", row, m, m == "DBNC")
    for row, m in enumerate(reversed(right)):
        leader(vals[methods.index(m)], "right", row, m, m == "DBNC")

    # critical-difference ruler
    yb = cline + 0.47
    ax.plot([lo, lo + cd], [yb, yb], color="black", lw=1.1)
    for x in (lo, lo + cd):
        ax.plot([x, x], [yb - 0.06, yb + 0.06], color="black", lw=1.1)
    ax.text(lo + cd / 2, yb + 0.08, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=6.4)

    # clique bars: methods joined by a bar are not significantly different
    bars = _form_cliques(vals, cd)
    last_end = -np.inf
    level = -1
    yb0 = cline - 0.14
    for a, b in bars:
        level = level + 1 if a <= last_end + 1e-9 else 0
        y = yb0 - level * 0.11
        ax.plot([a, b], [y, y], color="#3a3a3a", lw=1.8, solid_capstyle="round", zorder=5)
        last_end = b

    ax.set_xlim(lo - 1.15, hi + 1.15)
    ax.set_ylim(cline - top_gap - half * row_h - 0.25, yb + 0.32)
    ax.axis("off")
    ax.text((lo + hi) / 2, cline - top_gap - half * row_h - 0.18,
            "Mean accuracy rank (lower is better)", ha="center", va="top", fontsize=6.6)
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# F2 -- paired accuracy deltas with bootstrap CIs and W/T/L
# ---------------------------------------------------------------------------
def fig_paired_deltas(means, pairwise, out):
    wide = means.pivot_table(index="dataset_id", columns="method_label", values="accuracy")
    pw = pairwise[pairwise["metric"] == "accuracy"].set_index("comparator")
    rng = np.random.default_rng(0)

    def ci(comp):
        d = (wide["DBNC"] - wide[comp]).dropna().to_numpy()
        boot = rng.choice(d, size=(4000, d.size), replace=True).mean(axis=1)
        return d.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5)

    rows = []
    for comp in CLASSICAL + MODERN:
        mean, lo, hi = ci(comp)
        r = pw.loc[comp]
        rows.append(dict(comp=comp, group=ps.group_of(comp), mean=mean, lo=lo, hi=hi,
                         wtl=f"{int(r.wins)}/{int(r.ties)}/{int(r.losses)}"))
    df = pd.DataFrame(rows)
    # classical group on top (positive), modern at bottom; sort by delta within group
    cl = df[df.comp.isin(CLASSICAL)].sort_values("mean")
    mo = df[df.comp.isin(MODERN)].sort_values("mean")
    order = pd.concat([mo, cl], ignore_index=True)  # bottom-up => modern first
    y = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(ps.COL_W, 2.35))
    ax.axvline(0, color="#888888", lw=0.75, ls=(0, (4, 3)), zorder=1)
    for yi, (_, r) in zip(y, order.iterrows()):
        c = ps.GROUP_COLORS[r.group]
        ax.plot([r.lo, r.hi], [yi, yi], color=c, lw=1.25, solid_capstyle="round", zorder=2)
        ax.plot([r["mean"]], [yi], "o", ms=3.5, color=c, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([ps.short(m) for m in order.comp], fontsize=6.4)
    ax.set_xlabel("Accuracy difference (DBNC minus comparator)", fontsize=6.4, labelpad=1.5)
    ax.tick_params(axis="x", labelsize=5.8, pad=1.5)
    ax.tick_params(axis="y", pad=1.5)

    # W/T/L column at the right
    xr = order[["lo", "hi"]].to_numpy().max()
    xpad = (xr - order[["lo", "hi"]].to_numpy().min())
    xtext = xr + 0.08 * xpad
    for yi, (_, r) in zip(y, order.iterrows()):
        ax.text(xtext, yi, r.wtl, ha="left", va="center", fontsize=5.8, color="#333333")
    ax.text(xtext, y.max() + 0.65, "W/T/L", ha="left", va="center", fontsize=5.8,
            color="#333333", fontweight="bold")

    # group separator (modern group occupies the lower rows, classical the upper)
    sep = len(mo) - 0.5
    ax.axhline(sep, color="#cccccc", lw=0.7)
    xl = order[["lo", "hi"]].to_numpy().min()

    ax.set_xlim(xl - 0.045 * xpad, xtext + 0.18 * xpad)
    ax.set_ylim(-0.65, y.max() + 0.95)
    ax.margins(y=0)
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# F3 -- per-dataset accuracy-rank heatmap
# ---------------------------------------------------------------------------
def fig_rank_heatmap(means, summary, characteristics, out):
    methods = accuracy_rank_order(summary)
    paper_ids = list(characteristics.sort_values("dataset_id")["dataset_id"].astype(int))
    aliases = characteristics.set_index("dataset_id")["dataset_alias"]
    ranks = (
        means.pivot_table(index="dataset_id", columns="method_label", values="accuracy")
        .reindex(index=paper_ids, columns=methods)
        .rank(axis=1, ascending=False, method="average")
    )

    fig, ax = plt.subplots(figsize=(ps.FULL_W, 4.05))
    image = ax.imshow(ranks.to_numpy(dtype=float), aspect="auto", cmap=ps.SEQ_CMAP,
                      vmin=1, vmax=len(methods))
    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels([ps.short(m) for m in methods], rotation=45, ha="right", fontsize=6.2)
    labels = [f"{dataset_id} {aliases.loc[dataset_id]}" for dataset_id in ranks.index]
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=5.3)
    ax.set_xlabel("Method")
    ax.set_ylabel("Dataset")
    ax.tick_params(length=0)

    for row in range(ranks.shape[0]):
        for col in range(ranks.shape[1]):
            value = ranks.iat[row, col]
            if pd.notna(value):
                text = f"{value:.1f}" if abs(value - round(value)) > 1e-9 else f"{int(round(value))}"
                color = "white" if value > len(methods) / 2.0 else "#1a1a1a"
                ax.text(col, row, text, ha="center", va="center", fontsize=4.4, color=color)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.012)
    colorbar.set_label("Rank, lower is better", fontsize=7)
    colorbar.ax.tick_params(labelsize=6)
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# F4 -- calibration/probabilistic metric distributions
# ---------------------------------------------------------------------------
def fig_calibration(means, summary, out):
    methods = accuracy_rank_order(summary)
    fig, axes = plt.subplots(1, 3, figsize=(ps.FULL_W, 2.45), sharex=False)
    for ax, (metric, label) in zip(axes, CALIBRATION_METRICS):
        data = [
            means.loc[means["method_label"] == method, metric].dropna().to_numpy()
            for method in methods
        ]
        bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.55)
        for patch, method in zip(bp["boxes"], methods):
            patch.set_facecolor(ps.color_of(method))
            patch.set_alpha(0.72)
            patch.set_edgecolor("#333333")
        for median in bp["medians"]:
            median.set_color("#111111")
            median.set_linewidth(0.8)
        ax.set_title(label)
        ax.set_xticks(np.arange(1, len(methods) + 1))
        ax.set_xticklabels([ps.short(m) for m in methods], rotation=45, ha="right", fontsize=6)
        ax.grid(axis="y", alpha=0.55)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Lower is better")
    fig.savefig(out)
    plt.close(fig)
    return out


def categorical_delta_summary(means: pd.DataFrame, characteristics: pd.DataFrame) -> pd.DataFrame:
    wide = (
        means.pivot_table(index="dataset_id", columns="method_label", values="accuracy")
        .join(characteristics.set_index("dataset_id")[["categorical_ratio"]])
    )
    groups = [
        ("Cat. only", wide[wide["categorical_ratio"] == 1.0]),
        ("Cat. >= 0.5", wide[wide["categorical_ratio"] >= 0.5]),
        ("Cat. < 0.5", wide[wide["categorical_ratio"] < 0.5]),
    ]
    rows = []
    for label, frame in groups:
        best_classical = frame[[m for m in CLASSICAL if m in frame]].max(axis=1)
        best_modern = frame[[m for m in MODERN if m in frame]].max(axis=1)
        rows.append(
            {
                "group": label,
                "n": int(frame.shape[0]),
                "vs_classical": float((frame["DBNC"] - best_classical).mean()),
                "vs_modern": float((frame["DBNC"] - best_modern).mean()),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# F5 -- graph diagnostics, actual runtime, categorical subgroup deltas
# ---------------------------------------------------------------------------
def fig_diagnostic_runtime(graph, summary, means, characteristics, out):
    g = filtered_graph(graph, characteristics)
    methods = accuracy_rank_order(summary)
    s = summary.set_index("method_label").reindex(methods)
    cat = categorical_delta_summary(means, characteristics)

    fig, axes = plt.subplots(1, 3, figsize=(ps.FULL_W, 2.55),
                             gridspec_kw={"width_ratios": [0.9, 1.45, 1.05]})

    # Panel A: graph diagnostics over the 30 paper datasets only.
    ax = axes[0]
    graph_values = [
        float(g["acyclic"].fillna(False).mean()),
        float(g["thresholded_edge_density"].mean()),
        float(g["parent_limit_violation_rate"].mean()),
    ]
    graph_labels = ["Acyclic\nrate", "Mean\nedge density", "Mean parent\nviolation"]
    ax.bar(np.arange(3), graph_values, color=[ps.GROUP_COLORS["DBNC"], "#56B4E9", "#E69F00"])
    ax.set_ylim(0, 1.05)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(graph_labels, fontsize=6)
    ax.set_ylabel("Rate")
    ax.set_title("(a) Graph diagnostics")
    for x, y in enumerate(graph_values):
        ax.text(x, y + 0.025, f"{y:.2f}", ha="center", va="bottom", fontsize=6)
    ax.text(
        0.5,
        0.90,
        f"{int(g['acyclic'].sum())}/{len(g)} acyclic\n"
        f"{int((g['parent_limit_violation_rate'] > 0).sum())}/{len(g)} nonzero violations\n"
        f"{g['thresholded_edges'].mean():.1f} mean edges",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=5.7,
    )

    # Panel B: actual train/inference times.
    ax = axes[1]
    x = np.arange(len(methods))
    width = 0.38
    ax.bar(x - width / 2, s["geomean_training_seconds"], width=width,
           label="Training", color="#0072B2")
    ax.bar(x + width / 2, s["geomean_inference_seconds"], width=width,
           label="Inference", color="#E69F00")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([ps.short(m) for m in methods], rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("Seconds, geometric mean")
    ax.set_title("(b) Runtime")
    ax.grid(axis="y", alpha=0.55)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=6, handlelength=1.1)

    # Panel C: categorical subgroup accuracy deltas.
    ax = axes[2]
    x = np.arange(cat.shape[0])
    ax.axhline(0, color="#888888", lw=0.8)
    ax.bar(x - width / 2, cat["vs_classical"], width=width,
           label="vs best BNC", color=ps.GROUP_COLORS["Classical BNC"])
    ax.bar(x + width / 2, cat["vs_modern"], width=width,
           label="vs best modern", color=ps.GROUP_COLORS["Boosted tree"])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.group}\n(n={int(r.n)})" for r in cat.itertuples()], fontsize=6)
    ax.set_ylabel("DBNC accuracy delta")
    ax.set_title("(c) Categorical subsets")
    ax.grid(axis="y", alpha=0.55)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=5.8, handlelength=1.0)

    fig.tight_layout(w_pad=1.0)
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# F6 -- accuracy vs training cost (Pareto view)
# ---------------------------------------------------------------------------
def fig_pareto(summary, out):
    s = summary.set_index("method_label")
    fig, ax = plt.subplots(figsize=(ps.COL_W, 2.7))
    ax.set_xscale("log")
    ax.grid(True, which="major", axis="both", alpha=0.6)
    ax.set_axisbelow(True)

    for m in s.index:
        t, a = s.loc[m, "geomean_training_seconds"], s.loc[m, "median_accuracy"]
        focal = m == "DBNC"
        ax.scatter(t, a, s=58 if focal else 34, color=ps.color_of(m),
                   edgecolor="black" if focal else "none",
                   linewidth=0.8 if focal else 0, zorder=4 if focal else 3)

    # individually-labelled points (dx, dy in points; ha, va)
    labels = {
        "NB": (0, 8, "center", "bottom"),
        "Chow-Liu": (0, -9, "center", "top"),
        "LightGBM": (-5, -8, "right", "top"),
        "XGBoost": (4, 7, "left", "bottom"),
        "CatBoost": (0, -10, "center", "top"),
        "FT-Transformer": (-6, 8, "right", "bottom"),
        "DBNC": (-8, -1, "right", "center"),
    }
    for m, (dx, dy, ha, va) in labels.items():
        t, a = s.loc[m, "geomean_training_seconds"], s.loc[m, "median_accuracy"]
        ax.annotate(m, (t, a), textcoords="offset points", xytext=(dx, dy),
                    ha=ha, va=va, fontsize=7,
                    color=ps.GROUP_COLORS["DBNC"] if m == "DBNC" else "#222222",
                    fontweight="bold" if m == "DBNC" else "normal")

    # the tight classical-augmented cluster (TAN/BAN/KDB-1/2/3) gets one annotation
    clust = ["TAN", "BAN", "KDB-1", "KDB-2", "KDB-3"]
    cx = float(s.loc[clust, "geomean_training_seconds"].mean())
    cy = float(s.loc[clust, "median_accuracy"].mean())
    ax.annotate("TAN, BAN,\nKDB-1/2/3", (cx, cy), textcoords="offset points",
                xytext=(10, 16), ha="left", va="bottom", fontsize=7,
                color=ps.GROUP_COLORS["Classical BNC"],
                arrowprops=dict(arrowstyle="-", lw=0.5, color="#999999"))

    ax.set_xlabel("Training time (s, geometric mean, log scale)")
    ax.set_ylabel("Median accuracy")
    ax.set_xlim(0.045, 70)

    handles = [plt.Line2D([], [], marker="o", ls="", ms=5, color=c, label=g)
               for g, c in ps.GROUP_COLORS.items()]
    ax.legend(handles=handles, loc="lower right", fontsize=6.6, handletextpad=0.3)
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# F7 -- ablation summary (hybrid-weight sweep)
# ---------------------------------------------------------------------------
def fig_ablation(abl, out):
    a = abl.set_index("variant")
    metrics = ["accuracy", "log_loss", "ece"]
    mlabels = {"accuracy": "Accuracy", "log_loss": "Log-loss", "ece": "ECE"}
    sweep = ["loss_w_000", "loss_w_025", "loss_w_050", "loss_w_075", "loss_w_100"]
    w = [0.0, 0.25, 0.5, 0.75, 1.0]
    markers = {"accuracy": "o", "log_loss": "^", "ece": "s"}

    fig, ax = plt.subplots(figsize=(ps.COL_W, 1.65))
    ax.axhline(0, color="#888888", lw=0.75, ls=(0, (4, 3)), zorder=1)
    for met in metrics:
        vals = [a.loc[v, f"delta_{met}_mean"] for v in sweep]
        ax.plot(w, vals, marker=markers[met], color=METRIC_COLORS[met],
                label=mlabels[met], ms=3.6, lw=1.15, zorder=3)
    ax.set_xlabel("Joint-loss weight", fontsize=6.6, labelpad=1.5)
    ax.set_ylabel("Delta vs full model", fontsize=6.6, labelpad=1.5)
    ax.set_xticks(w)
    ax.tick_params(labelsize=5.8, pad=1.5)
    ax.legend(loc="upper left", fontsize=5.8, handlelength=1.3, handletextpad=0.35)
    fig.tight_layout(pad=0.25)
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reports", type=Path, default=Path("reports"))
    p.add_argument("--ablation", type=Path,
                   default=Path("development/reports_ablation_attention_5"))
    p.add_argument("--paper", type=Path, default=Path("paper"))
    args = p.parse_args(argv)

    ps.apply_style()
    figdir = args.paper / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    stats = json.loads((args.reports / "main_statistical_tests.json").read_text())
    means = pd.read_csv(args.reports / "main_dataset_means.csv")
    pairwise = pd.read_csv(args.reports / "tables" / "dbnc_pairwise_tests.csv")
    summary = pd.read_csv(args.reports / "tables" / "method_summary.csv")
    characteristics = pd.read_csv(args.reports / "tables" / "dataset_characteristics.csv")
    graph = pd.read_csv(args.reports / "graph_diagnostics.csv")
    abl = pd.read_csv(args.ablation / "core_ablation_summary.csv")

    written = [
        fig_architecture(figdir / "dbnc_architecture.pdf"),
        fig_cd(stats, figdir / "critical_difference_accuracy.pdf"),
        fig_paired_deltas(means, pairwise, figdir / "paired_accuracy_deltas.pdf"),
        fig_rank_heatmap(means, summary, characteristics, figdir / "rank_heatmap_accuracy.pdf"),
        fig_calibration(means, summary, figdir / "calibration_metrics.pdf"),
        fig_diagnostic_runtime(
            graph,
            summary,
            means,
            characteristics,
            figdir / "diagnostic_runtime_summary.pdf",
        ),
        fig_pareto(summary, figdir / "efficiency_pareto.pdf"),
        fig_ablation(abl, figdir / "ablation_summary.pdf"),
    ]
    for w in written:
        print("wrote", w)


if __name__ == "__main__":
    main()
