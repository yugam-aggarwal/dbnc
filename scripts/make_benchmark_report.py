#!/usr/bin/env python3
"""Generate paper-ready benchmark reports from completed DBNC artifacts.

This script is intentionally read-only with respect to experiment results: it
does not fit models, rerun evaluations, or require raw predictions. It derives
figures, summary tables, pairwise tests, and a short markdown report from the
CSV/JSON artifacts produced by ``run.py report``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "dbnc-report-matplotlib"),
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys as _sys

_sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_style  # noqa: E402

paper_style.apply_style()


METHOD_ORDER = [
    "DBNC",
    "NB",
    "Chow-Liu",
    "TAN",
    "BAN",
    "KDB-1",
    "KDB-2",
    "KDB-3",
    "XGBoost",
    "CatBoost",
    "LightGBM",
    "FT-Transformer",
]
CLASSICAL_BASELINES = ["NB", "Chow-Liu", "TAN", "BAN", "KDB-1", "KDB-2", "KDB-3"]
MODERN_BASELINES = ["XGBoost", "CatBoost", "LightGBM", "FT-Transformer"]
PRIMARY_COMPARATORS = [
    "NB",
    "Chow-Liu",
    "TAN",
    "BAN",
    "KDB-1",
    "KDB-2",
    "KDB-3",
    "XGBoost",
    "CatBoost",
    "LightGBM",
    "FT-Transformer",
]
HIGHER_IS_BETTER = {"accuracy", "macro_f1", "auc"}
SUMMARY_METRICS = ["accuracy", "macro_f1", "auc", "log_loss", "brier", "ece"]
RANK_METRICS = SUMMARY_METRICS + ["training_seconds", "inference_seconds"]
CALIBRATION_METRICS = ["ece", "brier", "log_loss"]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def available_methods(frame: pd.DataFrame) -> list[str]:
    observed = set(frame["method_label"].dropna().astype(str))
    ordered = [method for method in METHOD_ORDER if method in observed]
    extras = sorted(observed.difference(ordered))
    return ordered + extras


def metric_ascending(metric: str) -> bool:
    return metric not in HIGHER_IS_BETTER


def rank_table(means: pd.DataFrame, metric: str, methods: list[str]) -> pd.DataFrame:
    pivot = (
        means.pivot(index="dataset_id", columns="method_label", values=metric)
        .reindex(columns=methods)
        .dropna(how="all")
    )
    return pivot.rank(axis=1, ascending=metric_ascending(metric), method="average")


def best_methods(group: pd.DataFrame, metric: str) -> list[str]:
    values = group[["method_label", metric]].dropna()
    if values.empty:
        return []
    target = values[metric].min() if metric_ascending(metric) else values[metric].max()
    return values.loc[np.isclose(values[metric], target), "method_label"].tolist()


def win_counts(means: pd.DataFrame, metric: str, methods: list[str]) -> dict[str, int]:
    counts = {method: 0 for method in methods}
    for _, group in means.groupby("dataset_id"):
        for method in best_methods(group, metric):
            counts[method] = counts.get(method, 0) + 1
    return counts


def geometric_mean(values: Iterable[float], eps: float = 1e-12) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan")
    return float(np.exp(np.mean(np.log(np.clip(array, eps, None)))))


def create_method_summary(
    means: pd.DataFrame,
    per_seed: pd.DataFrame,
    ranks: pd.DataFrame,
    methods: list[str],
) -> pd.DataFrame:
    seed_std = (
        per_seed.groupby(["dataset_id", "method_label"], dropna=False)[["accuracy", "log_loss", "ece"]]
        .std(ddof=1)
        .reset_index()
    )
    rank_lookup: dict[tuple[str, str], float] = {}
    if not ranks.empty:
        for (metric, method), value in (
            ranks.groupby(["metric", "method_label"], dropna=False)["rank"].mean().items()
        ):
            rank_lookup[(str(metric), str(method))] = float(value)

    win_lookup = {metric: win_counts(means, metric, methods) for metric in SUMMARY_METRICS}
    rows: list[dict[str, Any]] = []
    for method in methods:
        method_means = means[means["method_label"] == method]
        method_seed_std = seed_std[seed_std["method_label"] == method]
        row: dict[str, Any] = {
            "method_label": method,
            "datasets": int(method_means["dataset_id"].nunique()),
        }
        for metric in RANK_METRICS:
            row[f"mean_rank_{metric}"] = rank_lookup.get((metric, method))
        for metric in SUMMARY_METRICS:
            row[f"median_{metric}"] = float(method_means[metric].median())
            row[f"mean_{metric}"] = float(method_means[metric].mean())
            row[f"win_count_{metric}"] = int(win_lookup[metric].get(method, 0))
        row["geomean_training_seconds"] = geometric_mean(method_means["training_seconds"])
        row["median_training_seconds"] = float(method_means["training_seconds"].median())
        row["geomean_inference_seconds"] = geometric_mean(method_means["inference_seconds"])
        row["median_inference_seconds"] = float(method_means["inference_seconds"].median())
        for metric in ("accuracy", "log_loss", "ece"):
            row[f"median_seed_std_{metric}"] = float(method_seed_std[metric].median())
            row[f"mean_seed_std_{metric}"] = float(method_seed_std[metric].mean())
        rows.append(row)
    summary = pd.DataFrame(rows)
    sort_column = "mean_rank_accuracy"
    if sort_column in summary:
        summary = summary.sort_values(sort_column, na_position="last").reset_index(drop=True)
    return summary


def wilcoxon_p_value(deltas: np.ndarray) -> tuple[float | None, float | None, str]:
    deltas = np.asarray(deltas, dtype=float)
    deltas = deltas[np.isfinite(deltas)]
    if deltas.size == 0:
        return None, None, "no_complete_pairs"
    if np.all(np.isclose(deltas, 0.0)):
        return 0.0, 1.0, "all_zero_deltas"
    try:
        statistic, p_value = stats.wilcoxon(deltas, zero_method="wilcox", alternative="two-sided")
    except ValueError as exc:
        return None, None, f"wilcoxon_unavailable:{exc}"
    return float(statistic), float(p_value), "ok"


def create_dbnc_pairwise_tests(means: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    pivot = (
        means.pivot(index="dataset_id", columns="method_label", values="accuracy")
        .reindex(columns=methods)
        .dropna(subset=["DBNC"])
    )
    rows: list[dict[str, Any]] = []
    for comparator in [method for method in PRIMARY_COMPARATORS if method in pivot.columns]:
        complete = pivot[["DBNC", comparator]].dropna()
        deltas = complete["DBNC"].to_numpy() - complete[comparator].to_numpy()
        statistic, p_value, status = wilcoxon_p_value(deltas)
        wins = int(np.sum(deltas > 1e-12))
        losses = int(np.sum(deltas < -1e-12))
        ties = int(deltas.size - wins - losses)
        median_delta = float(np.median(deltas)) if deltas.size else None
        if median_delta is None or math.isclose(median_delta, 0.0, abs_tol=1e-12):
            direction = "tie"
        elif median_delta > 0:
            direction = "DBNC higher"
        else:
            direction = f"{comparator} higher"
        rows.append(
            {
                "metric": "accuracy",
                "comparator": comparator,
                "n_datasets": int(deltas.size),
                "mean_delta_dbnc_minus_comparator": float(np.mean(deltas)) if deltas.size else None,
                "median_delta_dbnc_minus_comparator": median_delta,
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "wilcoxon_statistic": statistic,
                "wilcoxon_p_value": p_value,
                "test_status": status,
                "effect_direction": direction,
            }
        )
    return pd.DataFrame(rows)


def manifest_records(manifest: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in manifest.get("records", []):
        categorical = record.get("predictor_types", {}).get("categorical", [])
        numeric = record.get("predictor_types", {}).get("numeric", [])
        class_counts = record.get("class_counts", {})
        total_classes = sum(int(value) for value in class_counts.values())
        n_rows = int(record.get("n_rows", 0) or 0)
        n_predictors = int(record.get("n_predictors", 0) or 0)
        missing_total = int(record.get("missing_values", {}).get("total", 0) or 0)
        rows.append(
            {
                "dataset_id": int(record["dataset_id"]),
                "dataset_alias": record.get("alias"),
                "n_rows": n_rows,
                "n_predictors": n_predictors,
                "n_classes": int(record.get("n_classes", 0) or 0),
                "n_categorical": len(categorical),
                "n_numeric": len(numeric),
                "categorical_ratio": len(categorical) / max(1, len(categorical) + len(numeric)),
                "missing_total": missing_total,
                "missing_ratio": missing_total / max(1, n_rows * n_predictors),
                "majority_class_ratio": (
                    max(int(value) for value in class_counts.values()) / total_classes
                    if total_classes
                    else None
                ),
            }
        )
    return pd.DataFrame(rows)


def best_label_and_value(values: pd.Series, candidates: list[str], higher: bool = True) -> tuple[str | None, float | None]:
    present = values.reindex([candidate for candidate in candidates if candidate in values.index]).dropna()
    if present.empty:
        return None, None
    label = str(present.idxmax() if higher else present.idxmin())
    return label, float(present.loc[label])


def create_dataset_characteristics(means: pd.DataFrame, manifest: dict[str, Any], methods: list[str]) -> pd.DataFrame:
    characteristics = manifest_records(manifest)
    accuracy = means.pivot(index="dataset_id", columns="method_label", values="accuracy").reindex(columns=methods)
    log_loss = means.pivot(index="dataset_id", columns="method_label", values="log_loss").reindex(columns=methods)
    accuracy_ranks = accuracy.rank(axis=1, ascending=False, method="average")
    log_loss_ranks = log_loss.rank(axis=1, ascending=True, method="average")
    rows: list[dict[str, Any]] = []
    aliases = means[["dataset_id", "dataset_alias"]].drop_duplicates().set_index("dataset_id")["dataset_alias"]
    for dataset_id, values in accuracy.iterrows():
        if "DBNC" not in values.index or pd.isna(values["DBNC"]):
            continue
        best_classical, best_classical_value = best_label_and_value(values, CLASSICAL_BASELINES)
        best_modern, best_modern_value = best_label_and_value(values, MODERN_BASELINES)
        best_overall, best_overall_value = best_label_and_value(values, [m for m in methods if m != "DBNC"])
        dbnc_accuracy = float(values["DBNC"])
        rows.append(
            {
                "dataset_id": int(dataset_id),
                "dataset_alias": aliases.get(dataset_id),
                "dbnc_accuracy": dbnc_accuracy,
                "dbnc_accuracy_rank": float(accuracy_ranks.loc[dataset_id, "DBNC"]),
                "dbnc_log_loss": float(log_loss.loc[dataset_id, "DBNC"]),
                "dbnc_log_loss_rank": float(log_loss_ranks.loc[dataset_id, "DBNC"]),
                "best_classical_method": best_classical,
                "best_classical_accuracy": best_classical_value,
                "dbnc_delta_vs_best_classical": (
                    dbnc_accuracy - best_classical_value if best_classical_value is not None else None
                ),
                "best_modern_method": best_modern,
                "best_modern_accuracy": best_modern_value,
                "dbnc_delta_vs_best_modern": (
                    dbnc_accuracy - best_modern_value if best_modern_value is not None else None
                ),
                "best_non_dbnc_method": best_overall,
                "best_non_dbnc_accuracy": best_overall_value,
                "dbnc_delta_vs_best_non_dbnc": (
                    dbnc_accuracy - best_overall_value if best_overall_value is not None else None
                ),
            }
        )
    result = pd.DataFrame(rows)
    result = characteristics.merge(result, on=["dataset_id", "dataset_alias"], how="inner")
    return result.sort_values("dataset_id").reset_index(drop=True)


def filter_graph_to_paper_datasets(graph: pd.DataFrame, dataset_characteristics: pd.DataFrame) -> pd.DataFrame:
    """Keep graph diagnostics aligned with the complete 30-dataset comparison."""
    paper_ids = set(dataset_characteristics["dataset_id"].astype(int))
    return graph[graph["dataset_id"].astype(int).isin(paper_ids)].copy().reset_index(drop=True)


def write_heatmap(means: pd.DataFrame, metric: str, path: Path, methods: list[str]) -> None:
    ranks = rank_table(means, metric, methods).dropna(how="all")
    aliases = means[["dataset_id", "dataset_alias"]].drop_duplicates().set_index("dataset_id")["dataset_alias"]
    ranks = ranks.sort_index()
    labels = [f"{int(dataset_id)} {aliases.get(dataset_id, '')}" for dataset_id in ranks.index]
    figure_width = max(10, 0.68 * len(ranks.columns) + 3)
    figure_height = max(8, 0.26 * len(ranks.index) + 2)
    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    image = ax.imshow(ranks.to_numpy(dtype=float), aspect="auto", cmap=paper_style.SEQ_CMAP,
                      vmin=1, vmax=len(methods))
    ax.set_xticks(np.arange(len(ranks.columns)))
    ax.set_xticklabels(ranks.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(f"Per-dataset method ranks: {metric.replace('_', ' ')}")
    ax.set_xlabel("Method")
    ax.set_ylabel("Dataset")
    mid = len(methods) / 2.0
    for row in range(ranks.shape[0]):
        for col in range(ranks.shape[1]):
            value = ranks.iat[row, col]
            if pd.notna(value):
                # cividis_r is bright at low rank, dark at high rank -> contrast the text
                color = "white" if value > mid else "#1a1a1a"
                ax.text(col, row, f"{value:.0f}", ha="center", va="center", fontsize=6, color=color)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_label("Rank, lower is better")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def write_dbnc_delta_plot(means: pd.DataFrame, path: Path, methods: list[str]) -> None:
    pivot = means.pivot(index="dataset_id", columns="method_label", values="accuracy").reindex(columns=methods)
    comparators = [method for method in PRIMARY_COMPARATORS if method in pivot.columns]
    data = [(pivot["DBNC"] - pivot[method]).dropna().to_numpy() for method in comparators]
    fig, ax = plt.subplots(figsize=(9, max(5.5, 0.42 * len(comparators) + 1.5)))
    positions = np.arange(1, len(comparators) + 1)
    ax.boxplot(data, vert=False, positions=positions, widths=0.55, showfliers=False)
    rng = np.random.default_rng(7)
    for position, values in zip(positions, data):
        jitter = rng.normal(0.0, 0.035, size=len(values))
        ax.scatter(values, position + jitter, s=14, alpha=0.65, color="#2f6f9f", edgecolor="none")
    ax.axvline(0.0, color="black", linewidth=1.1, linestyle="--")
    ax.set_yticks(positions)
    ax.set_yticklabels(comparators)
    ax.set_xlabel("Accuracy delta: DBNC minus comparator")
    ax.set_title("DBNC accuracy deltas across datasets")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def pairwise_counts(means: pd.DataFrame, metric: str, methods: list[str]) -> tuple[np.ndarray, list[str], np.ndarray]:
    pivot = means.pivot(index="dataset_id", columns="method_label", values=metric).reindex(columns=methods)
    n = len(methods)
    labels = np.empty((n, n), dtype=object)
    net = np.zeros((n, n), dtype=float)
    ascending = metric_ascending(metric)
    for row, left in enumerate(methods):
        for col, right in enumerate(methods):
            if left == right:
                labels[row, col] = ""
                net[row, col] = np.nan
                continue
            pair = pivot[[left, right]].dropna()
            if ascending:
                left_better = pair[left] < pair[right] - 1e-12
                right_better = pair[left] > pair[right] + 1e-12
            else:
                left_better = pair[left] > pair[right] + 1e-12
                right_better = pair[left] < pair[right] - 1e-12
            wins = int(left_better.sum())
            losses = int(right_better.sum())
            ties = int(pair.shape[0] - wins - losses)
            labels[row, col] = f"{wins}/{ties}/{losses}"
            net[row, col] = wins - losses
    return net, methods, labels


def write_win_tie_loss_plot(means: pd.DataFrame, metric: str, path: Path, methods: list[str]) -> None:
    net, labels, cell_text = pairwise_counts(means, metric, methods)
    n_datasets = means["dataset_id"].nunique()
    fig, ax = plt.subplots(figsize=(11, 9.5))
    image = ax.imshow(net, cmap=paper_style.DIV_CMAP, vmin=-n_datasets, vmax=n_datasets)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(f"Pairwise win/tie/loss counts: {metric.replace('_', ' ')}")
    ax.set_xlabel("Column method")
    ax.set_ylabel("Row method")
    for row in range(len(labels)):
        for col in range(len(labels)):
            if row != col:
                # white text on saturated cells, dark on near-neutral ones
                color = "white" if abs(net[row, col]) > 0.55 * n_datasets else "#1a1a1a"
                ax.text(col, row, cell_text[row, col], ha="center", va="center",
                        fontsize=6.5, color=color)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.032, pad=0.02)
    colorbar.set_label("Row wins minus row losses")
    fig.text(0.5, 0.012, "Cell format is row wins / ties / losses across datasets.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    fig.savefig(path, dpi=300)
    plt.close(fig)


def write_seed_stability_plot(per_seed: pd.DataFrame, path: Path, methods: list[str]) -> None:
    std = (
        per_seed.groupby(["dataset_id", "method_label"], dropna=False)[["accuracy", "log_loss", "ece"]]
        .std(ddof=1)
        .reset_index()
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=False)
    for ax, metric in zip(axes, ["accuracy", "log_loss", "ece"]):
        data = [std.loc[std["method_label"] == method, metric].dropna().to_numpy() for method in methods]
        ax.boxplot(data, tick_labels=methods, showfliers=False)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylabel("Across-seed standard deviation")
        ax.tick_params(axis="x", labelrotation=60, labelsize=7)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Seed stability by method")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def write_calibration_plot(means: pd.DataFrame, path: Path, methods: list[str]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=False)
    for ax, metric in zip(axes, CALIBRATION_METRICS):
        data = [means.loc[means["method_label"] == method, metric].dropna().to_numpy() for method in methods]
        ax.boxplot(data, tick_labels=methods, showfliers=False)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylabel("Lower is better")
        ax.tick_params(axis="x", labelrotation=60, labelsize=7)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Probabilistic quality summary")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def write_efficiency_plot(means: pd.DataFrame, path: Path, methods: list[str]) -> None:
    summary = []
    for method in methods:
        group = means[means["method_label"] == method]
        summary.append(
            {
                "method_label": method,
                "training": geometric_mean(group["training_seconds"]),
                "inference": geometric_mean(group["inference_seconds"]),
                "accuracy": float(group["accuracy"].median()),
            }
        )
    frame = pd.DataFrame(summary).set_index("method_label")
    x = np.arange(len(frame))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    width = 0.38
    axes[0].bar(x - width / 2, frame["training"], width=width, label="Training", color="#0072B2")
    axes[0].bar(x + width / 2, frame["inference"], width=width, label="Inference", color="#E69F00")
    axes[0].set_yscale("log")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(frame.index, rotation=60, ha="right", fontsize=7)
    axes[0].set_ylabel("Geometric mean seconds")
    axes[0].set_title("Runtime summary")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    ax = axes[1]
    for method in frame.index:
        t, a = frame.loc[method, "training"], frame.loc[method, "accuracy"]
        focal = method == "DBNC"
        ax.scatter(t, a, s=70 if focal else 42, color=paper_style.color_of(method),
                   edgecolor="black" if focal else "none", linewidth=0.8 if focal else 0,
                   zorder=4 if focal else 3)
    # deterministic label offsets (avoid the overlapping-label problem)
    offsets = {
        "NB": (0, 8, "center", "bottom"), "Chow-Liu": (0, -9, "center", "top"),
        "LightGBM": (-5, -8, "right", "top"), "XGBoost": (4, 7, "left", "bottom"),
        "CatBoost": (0, -10, "center", "top"), "FT-Transformer": (-6, 8, "right", "bottom"),
        "DBNC": (-8, -1, "right", "center"),
    }
    for method, (dx, dy, ha, va) in offsets.items():
        if method in frame.index:
            t, a = frame.loc[method, "training"], frame.loc[method, "accuracy"]
            ax.annotate(method, (t, a), textcoords="offset points", xytext=(dx, dy),
                        ha=ha, va=va, fontsize=7,
                        fontweight="bold" if method == "DBNC" else "normal")
    cluster = [m for m in ["TAN", "BAN", "KDB-1", "KDB-2", "KDB-3"] if m in frame.index]
    if cluster:
        cx = float(frame.loc[cluster, "training"].mean())
        cy = float(frame.loc[cluster, "accuracy"].mean())
        ax.annotate("TAN, BAN,\nKDB-1/2/3", (cx, cy), textcoords="offset points",
                    xytext=(10, 16), ha="left", va="bottom", fontsize=7,
                    color=paper_style.GROUP_COLORS["Classical BNC"],
                    arrowprops=dict(arrowstyle="-", lw=0.5, color="#999999"))
    ax.set_xscale("log")
    ax.set_xlabel("Geometric mean training seconds")
    ax.set_ylabel("Median dataset accuracy")
    ax.set_title("Accuracy versus training cost")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def write_graph_diagnostics_plot(graph: pd.DataFrame, means: pd.DataFrame, path: Path) -> None:
    aliases = means[["dataset_id", "dataset_alias"]].drop_duplicates().set_index("dataset_id")["dataset_alias"]
    graph = graph.copy()
    graph["dataset_alias"] = graph["dataset_id"].map(aliases)
    by_dataset = (
        graph.groupby(["dataset_id", "dataset_alias"], dropna=False)[
            ["edge_mass", "thresholded_edge_density", "parent_limit_violation_rate"]
        ]
        .mean()
        .reset_index()
    )
    top_violations = by_dataset.sort_values("parent_limit_violation_rate", ascending=False).head(10)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes[0, 0].hist(graph["edge_mass"].dropna(), bins=24, color="#4c78a8", alpha=0.85)
    axes[0, 0].set_title("DBNC edge mass")
    axes[0, 0].set_xlabel("Edge mass")
    axes[0, 0].set_ylabel("Runs")
    axes[0, 1].hist(graph["thresholded_edge_density"].dropna(), bins=24, color="#59a14f", alpha=0.85)
    axes[0, 1].set_title("Thresholded edge density")
    axes[0, 1].set_xlabel("Density")
    axes[0, 1].set_ylabel("Runs")
    axes[1, 0].barh(
        [f"{int(row.dataset_id)} {row.dataset_alias}" for row in top_violations.itertuples()],
        top_violations["parent_limit_violation_rate"],
        color="#e15759",
    )
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_title("Highest parent-limit violation rate")
    axes[1, 0].set_xlabel("Mean violation rate")
    scatter = axes[1, 1].scatter(
        by_dataset["thresholded_edge_density"],
        by_dataset["edge_mass"],
        c=by_dataset["parent_limit_violation_rate"],
        cmap="magma",
        s=45,
    )
    axes[1, 1].set_title("Density, mass, and parent-limit pressure")
    axes[1, 1].set_xlabel("Mean thresholded edge density")
    axes[1, 1].set_ylabel("Mean edge mass")
    colorbar = fig.colorbar(scatter, ax=axes[1, 1], fraction=0.046, pad=0.04)
    colorbar.set_label("Parent-limit violation rate")
    acyclic_count = int(graph["acyclic"].fillna(False).sum()) if "acyclic" in graph else 0
    fig.suptitle(f"DBNC graph diagnostics ({acyclic_count}/{len(graph)} runs acyclic)")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def format_value(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.{digits}g}"


def create_markdown_report(
    path: Path,
    report_summary: dict[str, Any],
    method_summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    dataset_characteristics: pd.DataFrame,
    graph: pd.DataFrame,
    reports_root: Path,
) -> None:
    top_accuracy = method_summary.sort_values("mean_rank_accuracy").head(5)
    dbnc_summary = method_summary.loc[method_summary["method_label"] == "DBNC"].iloc[0]
    dbnc_pairwise = pairwise.set_index("comparator")

    def pair_line(comparator: str) -> str:
        if comparator not in dbnc_pairwise.index:
            return f"- {comparator}: not available"
        row = dbnc_pairwise.loc[comparator]
        return (
            f"- {comparator}: mean delta {format_value(row['mean_delta_dbnc_minus_comparator'])}, "
            f"median delta {format_value(row['median_delta_dbnc_minus_comparator'])}, "
            f"W/T/L {int(row['wins'])}/{int(row['ties'])}/{int(row['losses'])}, "
            f"Wilcoxon p={format_value(row['wilcoxon_p_value'], 3)}"
        )

    ablation_records = int(report_summary.get("core_ablation_records", 0) or 0)
    loss_weight_records = int(report_summary.get("loss_weight_records", 0) or 0)
    parent_violations = int((graph.get("parent_limit_violation_rate", pd.Series(dtype=float)) > 0).sum())
    acyclic_runs = int(graph.get("acyclic", pd.Series(dtype=bool)).fillna(False).sum()) if not graph.empty else 0
    expected_seeds = report_summary.get("expected_seeds", "n/a")
    n_seeds = len(expected_seeds) if isinstance(expected_seeds, list) else None
    paper_records = (
        int(report_summary.get(
            "paper_comparison_records",
            dataset_characteristics.shape[0] * method_summary.shape[0] * n_seeds,
        ))
        if n_seeds is not None
        else "n/a"
    )
    raw_completed_records = report_summary.get(
        "raw_completed_records", report_summary.get("completed_records", "n/a")
    )
    raw_main_records = report_summary.get("raw_main_records", report_summary.get("main_records", "n/a"))
    raw_graph_records = report_summary.get(
        "raw_graph_diagnostic_records", report_summary.get("graph_diagnostic_records", "n/a")
    )

    lines = [
        "# DBNC Benchmark Report",
        "",
        "## Protocol Snapshot",
        "",
        f"- Paper comparison records: {paper_records}",
        f"- Paper comparison datasets: {dataset_characteristics.shape[0]}",
        f"- Methods per complete dataset: {method_summary.shape[0]}",
        f"- Seeds per method: {expected_seeds}",
        f"- Raw completed records: {raw_completed_records}",
        f"- Raw main records: {raw_main_records}",
        f"- Raw graph diagnostic records: {raw_graph_records}",
        f"- Protocol complete: {report_summary.get('protocol_complete', 'n/a')}",
        f"- Admissibility enforced: {report_summary.get('admissibility_enforced', 'n/a')}",
        "",
        "## Main Findings",
        "",
        "Top methods by mean accuracy rank:",
        "",
    ]
    for _, row in top_accuracy.iterrows():
        lines.append(
            f"- {row['method_label']}: mean rank {format_value(row['mean_rank_accuracy'])}, "
            f"median accuracy {format_value(row['median_accuracy'])}, "
            f"accuracy wins {int(row['win_count_accuracy'])}"
        )
    lines.extend(
        [
            "",
            "DBNC summary:",
            "",
            f"- Mean accuracy rank: {format_value(dbnc_summary['mean_rank_accuracy'])}",
            f"- Median accuracy: {format_value(dbnc_summary['median_accuracy'])}",
            f"- Mean log-loss rank: {format_value(dbnc_summary['mean_rank_log_loss'])}",
            f"- Mean ECE rank: {format_value(dbnc_summary['mean_rank_ece'])}",
            f"- Geometric mean training time: {format_value(dbnc_summary['geomean_training_seconds'])} seconds",
            "",
            "DBNC accuracy deltas against selected baselines:",
            "",
            pair_line("NB"),
            pair_line("Chow-Liu"),
            pair_line("TAN"),
            pair_line("BAN"),
            pair_line("KDB-3"),
            pair_line("XGBoost"),
            pair_line("CatBoost"),
            pair_line("LightGBM"),
            pair_line("FT-Transformer"),
            "",
            "Interpretation: DBNC has positive paired accuracy deltas against all classical Bayesian-network baselines, including TAN, BAN, and KDB variants, but the Nemenyi rank differences against TAN/BAN/KDB are not significant. It remains behind the boosted-tree and deep tabular baselines on accuracy in this benchmark.",
            "",
            "## Dataset Effects",
            "",
            f"- Datasets in characteristic table: {dataset_characteristics.shape[0]}",
            f"- Mean DBNC delta vs best classical baseline: {format_value(dataset_characteristics['dbnc_delta_vs_best_classical'].mean())}",
            f"- Mean DBNC delta vs best modern baseline: {format_value(dataset_characteristics['dbnc_delta_vs_best_modern'].mean())}",
            "",
            "## Graph Diagnostics",
            "",
            f"- DBNC graph diagnostic records: {graph.shape[0]}",
            f"- Acyclic DBNC runs: {acyclic_runs}/{graph.shape[0]}",
            f"- Runs with nonzero parent-limit violation rate: {parent_violations}/{graph.shape[0]}",
            "",
            "## Ablation Evidence",
            "",
            f"- Main benchmark report-root core ablation records: {ablation_records}.",
            f"- Main benchmark report-root loss-weight records: {loss_weight_records}.",
            "- Paper ablation evidence is generated from `development/reports_ablation_attention_5`, which contains the focused component ablation and hybrid-weight sweep used in the Results section.",
            "",
            "## Missing Evidence",
            "",
            "- Raw test probabilities are not saved in the current aggregate artifacts, so reliability diagrams, confidence histograms, ROC curves, and PR curves are intentionally not generated here.",
            "",
            "## Generated Outputs",
            "",
            "- Figures are in `reports/figures/`.",
            "- Summary tables are in `reports/tables/`.",
            f"- Source reports root: `{reports_root}`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def validate_outputs(
    reports_root: Path,
    method_summary: pd.DataFrame,
    dataset_characteristics: pd.DataFrame,
    pairwise: pd.DataFrame,
    methods: list[str],
) -> None:
    required = [
        reports_root / "figures" / "rank_heatmap_accuracy.png",
        reports_root / "figures" / "rank_heatmap_log_loss.png",
        reports_root / "figures" / "dbnc_delta_accuracy.png",
        reports_root / "figures" / "win_tie_loss_accuracy.png",
        reports_root / "figures" / "seed_stability.png",
        reports_root / "figures" / "calibration_metrics.png",
        reports_root / "figures" / "efficiency_summary.png",
        reports_root / "figures" / "dbnc_graph_diagnostics.png",
        reports_root / "tables" / "method_summary.csv",
        reports_root / "tables" / "dbnc_pairwise_tests.csv",
        reports_root / "tables" / "dataset_characteristics.csv",
        reports_root / "benchmark_report.md",
    ]
    missing = [str(path) for path in required if not path.exists() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"Missing or empty report outputs: {missing}")
    if method_summary.shape[0] != len(methods):
        raise RuntimeError(f"method_summary.csv has {method_summary.shape[0]} rows, expected {len(methods)}")
    if dataset_characteristics.shape[0] != 30:
        raise RuntimeError(
            f"dataset_characteristics.csv has {dataset_characteristics.shape[0]} rows, expected 30"
        )
    expected_pairwise = len([method for method in PRIMARY_COMPARATORS if method in methods])
    if pairwise.shape[0] != expected_pairwise:
        raise RuntimeError(f"dbnc_pairwise_tests.csv has {pairwise.shape[0]} rows, expected {expected_pairwise}")


def generate(reports_root: Path, artifacts_root: Path) -> dict[str, Any]:
    reports_root = reports_root.resolve()
    artifacts_root = artifacts_root.resolve()
    figures_root = reports_root / "figures"
    tables_root = reports_root / "tables"
    figures_root.mkdir(parents=True, exist_ok=True)
    tables_root.mkdir(parents=True, exist_ok=True)

    per_seed = pd.read_csv(reports_root / "main_valid_per_seed_results.csv")
    means = pd.read_csv(reports_root / "main_dataset_means.csv")
    ranks = pd.read_csv(reports_root / "main_ranking_inputs.csv")
    graph = pd.read_csv(reports_root / "graph_diagnostics.csv")
    manifest = load_json(artifacts_root / "validated_manifest.json")
    report_summary = load_json(reports_root / "report_summary.json")
    methods = available_methods(means)

    method_summary = create_method_summary(means, per_seed, ranks, methods)
    pairwise = create_dbnc_pairwise_tests(means, methods)
    dataset_characteristics = create_dataset_characteristics(means, manifest, methods)
    graph_paper = filter_graph_to_paper_datasets(graph, dataset_characteristics)

    method_summary.to_csv(tables_root / "method_summary.csv", index=False)
    pairwise.to_csv(tables_root / "dbnc_pairwise_tests.csv", index=False)
    dataset_characteristics.to_csv(tables_root / "dataset_characteristics.csv", index=False)

    write_heatmap(means, "accuracy", figures_root / "rank_heatmap_accuracy.png", methods)
    write_heatmap(means, "log_loss", figures_root / "rank_heatmap_log_loss.png", methods)
    write_dbnc_delta_plot(means, figures_root / "dbnc_delta_accuracy.png", methods)
    write_win_tie_loss_plot(means, "accuracy", figures_root / "win_tie_loss_accuracy.png", methods)
    write_win_tie_loss_plot(means, "log_loss", figures_root / "win_tie_loss_log_loss.png", methods)
    write_win_tie_loss_plot(means, "ece", figures_root / "win_tie_loss_ece.png", methods)
    write_seed_stability_plot(per_seed, figures_root / "seed_stability.png", methods)
    write_calibration_plot(means, figures_root / "calibration_metrics.png", methods)
    write_efficiency_plot(means, figures_root / "efficiency_summary.png", methods)
    write_graph_diagnostics_plot(graph_paper, means, figures_root / "dbnc_graph_diagnostics.png")
    create_markdown_report(
        reports_root / "benchmark_report.md",
        report_summary,
        method_summary,
        pairwise,
        dataset_characteristics,
        graph_paper,
        reports_root,
    )

    validate_outputs(reports_root, method_summary, dataset_characteristics, pairwise, methods)
    return {
        "figures": 10,
        "tables": 3,
        "methods": len(methods),
        "datasets": int(dataset_characteristics.shape[0]),
        "pairwise_comparisons": int(pairwise.shape[0]),
        "benchmark_report": str(reports_root / "benchmark_report.md"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=Path("reports"), help="Directory containing benchmark CSV/JSON reports.")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"), help="Directory containing validated_manifest.json.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = generate(args.reports, args.artifacts)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
