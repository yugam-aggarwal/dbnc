#!/usr/bin/env python3
"""Generate focused visualizations for DBNC ablation reports.

The input directory should contain the CSV/JSON artifacts emitted by
``run.py report``: especially ``core_ablation_table.csv``,
``loss_weight_results.csv``, and ``graph_diagnostics.csv``.
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

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "dbnc-ablation-matplotlib"),
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys as _sys

_sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_style  # noqa: E402

# Shared publication style (serif typography, clean spines, colour-blind palette),
# consistent with the paper figures and the benchmark report.
paper_style.apply_style()

CORE_VARIANTS = [
    "full",
    "conditional_only",
    "fixed_tan",
    "mlp_cpd",
]
LOSS_WEIGHT_VARIANTS = [
    "loss_w_000",
    "loss_w_025",
    "loss_w_050",
    "loss_w_075",
    "loss_w_100",
]
VARIANT_ORDER = CORE_VARIANTS + LOSS_WEIGHT_VARIANTS
DEFAULT_EXCLUDED_VARIANTS = {"no_regularization"}
MAIN_TABLE_VARIANTS = CORE_VARIANTS
MAIN_TABLE_METRICS = [
    "accuracy",
    "accuracy_delta_vs_full",
    "log_loss",
    "log_loss_delta_vs_full",
    "ece",
    "ece_delta_vs_full",
]
VARIANT_LABELS = {
    "full": "Full",
    "conditional_only": "Conditional only",
    "fixed_tan": "Fixed TAN",
    "mlp_cpd": "MLP CPD",
    "no_regularization": "No regularization",
    "loss_w_000": "w=0.00",
    "loss_w_025": "w=0.25",
    "loss_w_050": "w=0.50",
    "loss_w_075": "w=0.75",
    "loss_w_100": "w=1.00",
}
LOSS_WEIGHTS = {
    "loss_w_000": 0.0,
    "loss_w_025": 0.25,
    "loss_w_050": 0.50,
    "loss_w_075": 0.75,
    "loss_w_100": 1.0,
}
METRIC_ORDER = ["accuracy", "macro_f1", "log_loss", "ece"]
HIGHER_IS_BETTER = {"accuracy", "macro_f1", "auc"}
GRAPH_METRICS = ["edge_mass", "thresholded_edge_density", "parent_limit_violation_rate"]
GOOD_COLOR = "#1b9e77"
BAD_COLOR = "#d95f02"
NEUTRAL_COLOR = "#386cb0"
MUTED_COLOR = "#8c9aa8"
GRID_COLOR = "#e2e7ec"


def default_reports_root() -> Path:
    candidates = [
        Path("development/reports_ablation_attention_5"),
        Path("development/reports_ablation_8_pruned_nursery"),
        Path("development/reports_ablation_5"),
        Path("reports"),
    ]
    for candidate in candidates:
        if (candidate / "core_ablation_table.csv").exists():
            return candidate
    return Path("reports")


def label_variant(variant: str) -> str:
    return VARIANT_LABELS.get(variant, variant.replace("_", " ").title())


def variant_sort_key(variant: str) -> tuple[int, str]:
    if variant in VARIANT_ORDER:
        return VARIANT_ORDER.index(variant), variant
    return len(VARIANT_ORDER), variant


def metric_label(metric: str) -> str:
    labels = {
        "auc": "AUC",
        "ece": "ECE",
        "macro_f1": "Macro F1",
        "log_loss": "Log Loss",
    }
    if metric in labels:
        return labels[metric]
    return metric.replace("_", " ").title()


def present_metrics(frame: pd.DataFrame, candidates: Iterable[str] = METRIC_ORDER) -> list[str]:
    return [metric for metric in candidates if metric in frame.columns]


def require_nonempty_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Required input has no rows: {path}")
    return frame


def read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def filter_variants(frame: pd.DataFrame, excluded: set[str]) -> pd.DataFrame:
    if frame.empty or "variant" not in frame.columns or not excluded:
        return frame
    return frame[~frame["variant"].astype(str).isin(excluded)].copy()


def dataset_label_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame[["dataset_id", "dataset_alias"]]
        .drop_duplicates()
        .sort_values(["dataset_id", "dataset_alias"], kind="mergesort")
        .reset_index(drop=True)
    )


def aggregate_dataset_metrics(core: pd.DataFrame, losses: pd.DataFrame) -> pd.DataFrame:
    metrics = present_metrics(core)
    required = ["dataset_id", "dataset_alias", "variant"]
    core_metrics = core[required + metrics].copy()
    pieces = [core_metrics]
    if not losses.empty:
        loss_metrics = present_metrics(losses)
        common_metrics = [metric for metric in metrics if metric in loss_metrics]
        if common_metrics:
            loss_frame = losses[losses["variant"].isin(LOSS_WEIGHT_VARIANTS)].copy()
            if not loss_frame.empty:
                loss_frame = (
                    loss_frame.groupby(required, dropna=False)[common_metrics]
                    .mean(numeric_only=True)
                    .reset_index()
                )
                pieces.append(loss_frame)
    combined = pd.concat(pieces, ignore_index=True)
    combined["_variant_order"] = combined["variant"].map(lambda value: variant_sort_key(str(value))[0])
    combined = combined.sort_values(["dataset_id", "_variant_order", "variant"], kind="mergesort")
    combined = combined.drop(columns=["_variant_order"]).reset_index(drop=True)
    return combined


def add_full_deltas(dataset_metrics: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    baseline = (
        dataset_metrics[dataset_metrics["variant"] == "full"]
        .set_index("dataset_id")[metrics]
        .rename(columns={metric: f"{metric}_full" for metric in metrics})
    )
    joined = dataset_metrics.join(baseline, on="dataset_id")
    for metric in metrics:
        joined[f"{metric}_delta_vs_full"] = joined[metric] - joined[f"{metric}_full"]
    return joined


def summarize_variants(
    dataset_metrics: pd.DataFrame,
    graph: pd.DataFrame,
    metrics: list[str],
) -> pd.DataFrame:
    summary = (
        dataset_metrics.groupby("variant", dropna=False)[metrics]
        .mean(numeric_only=True)
        .reset_index()
    )
    full = summary[summary["variant"] == "full"]
    if full.empty:
        raise ValueError("core_ablation_table.csv must include variant='full'")
    full_values = full.iloc[0]
    for metric in metrics:
        summary[f"{metric}_delta_vs_full"] = summary[metric] - float(full_values[metric])

    if not graph.empty:
        graph_metrics = [metric for metric in GRAPH_METRICS if metric in graph.columns]
        if graph_metrics:
            graph_summary = (
                graph.groupby("variant", dropna=False)[graph_metrics]
                .mean(numeric_only=True)
                .reset_index()
            )
            if "acyclic" in graph.columns:
                acyclic = (
                    graph.assign(acyclic=graph["acyclic"].astype(str).str.lower().isin(["true", "1", "yes"]))
                    .groupby("variant", dropna=False)["acyclic"]
                    .mean()
                    .reset_index(name="acyclic_rate")
                )
                graph_summary = graph_summary.merge(acyclic, on="variant", how="left")
            summary = summary.merge(graph_summary, on="variant", how="left")

    summary["variant_label"] = summary["variant"].map(label_variant)
    summary["loss_weight"] = summary["variant"].map(LOSS_WEIGHTS)
    summary["_variant_order"] = summary["variant"].map(lambda value: variant_sort_key(str(value))[0])
    summary = summary.sort_values(["_variant_order", "variant"], kind="mergesort").drop(columns="_variant_order")
    metric_columns: list[str] = []
    for metric in metrics:
        metric_columns.extend([metric, f"{metric}_delta_vs_full"])
    extra_columns = [
        column
        for column in ["loss_weight", *GRAPH_METRICS, "acyclic_rate"]
        if column in summary.columns
    ]
    return summary[["variant", "variant_label", *metric_columns, *extra_columns]]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, path: Path) -> None:
    ensure_parent(path)
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def colors_for_delta(metric: str, values: pd.Series) -> list[str]:
    higher_is_better = metric in HIGHER_IS_BETTER
    colors = []
    for value in values:
        if math.isclose(float(value), 0.0, abs_tol=1e-12):
            colors.append("#8d99a6")
        elif (value > 0 and higher_is_better) or (value < 0 and not higher_is_better):
            colors.append(GOOD_COLOR)
        else:
            colors.append(BAD_COLOR)
    return colors


def write_delta_bar_plot(summary: pd.DataFrame, metrics: list[str], path: Path) -> None:
    plot_frame = summary[summary["variant"] != "full"].copy()
    plot_frame = plot_frame.dropna(subset=[f"{metric}_delta_vs_full" for metric in metrics], how="all")
    if plot_frame.empty:
        return
    plot_metrics = [metric for metric in ("accuracy", "log_loss", "ece") if metric in metrics]
    height = max(3.2, 0.34 * plot_frame.shape[0] + 1.1)
    fig, axes = plt.subplots(1, len(plot_metrics), figsize=(3.15 * len(plot_metrics), height), sharey=True)
    if len(plot_metrics) == 1:
        axes = np.array([axes])
    y = np.arange(plot_frame.shape[0])
    labels = plot_frame["variant_label"].tolist()
    for axis, metric in zip(axes, plot_metrics):
        column = f"{metric}_delta_vs_full"
        values = plot_frame[column].astype(float)
        axis.barh(y, values, color=colors_for_delta(metric, values), edgecolor="none")
        axis.axvline(0.0, color="#27313a", linewidth=0.9)
        max_abs = float(np.nanmax(np.abs(values.to_numpy()))) if values.notna().any() else 0.0
        if max_abs > 0:
            axis.set_xlim(-max_abs * 1.15, max_abs * 1.15)
        axis.set_title(metric_label(metric))
        axis.set_xlabel("Delta vs Full")
        axis.grid(axis="x", color=GRID_COLOR, linewidth=0.7)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_visible(False)
        axis.tick_params(axis="y", length=0)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    fig.suptitle("Ablation Effects Relative to Full DBNC", y=1.02, fontsize=11, fontweight="semibold")
    fig.tight_layout()
    save_figure(fig, path)


def write_accuracy_heatmap(dataset_metrics: pd.DataFrame, path: Path) -> None:
    if "accuracy" not in dataset_metrics.columns:
        return
    metrics = add_full_deltas(dataset_metrics, ["accuracy"])
    heat = metrics.pivot(
        index="dataset_id",
        columns="variant",
        values="accuracy_delta_vs_full",
    )
    heat = heat.drop(columns=["full"], errors="ignore")
    ordered_columns = [variant for variant in VARIANT_ORDER if variant in heat.columns]
    ordered_columns.extend(sorted(set(heat.columns) - set(ordered_columns)))
    heat = heat.reindex(columns=ordered_columns)
    aliases = dataset_label_frame(dataset_metrics).set_index("dataset_id")["dataset_alias"]
    heat = heat.sort_index()
    if heat.empty:
        return
    values = heat.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    max_abs = float(np.max(np.abs(finite))) if finite.size else 1.0
    max_abs = max(max_abs, 1e-6)
    fig_width = max(7.2, 0.88 * heat.shape[1] + 1.7)
    fig_height = max(3.4, 0.36 * heat.shape[0] + 1.3)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(values, aspect="auto", cmap="RdBu_r", vmin=-max_abs, vmax=max_abs)
    ax.set_xticks(np.arange(heat.shape[1]))
    ax.set_xticklabels([label_variant(str(column)) for column in heat.columns], rotation=35, ha="right")
    ax.set_yticks(np.arange(heat.shape[0]))
    ax.set_yticklabels([str(aliases.get(dataset_id, dataset_id)) for dataset_id in heat.index])
    ax.set_title("Accuracy Delta by Dataset", fontsize=11, fontweight="semibold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    for row in range(heat.shape[0]):
        for col in range(heat.shape[1]):
            value = values[row, col]
            if np.isfinite(value):
                text_color = "white" if abs(value) > max_abs * 0.58 else "#1f2933"
                ax.text(col, row, f"{value:+.3f}", ha="center", va="center", fontsize=7, color=text_color)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.035)
    colorbar.set_label("Variant - Full")
    fig.tight_layout()
    save_figure(fig, path)


def loss_dataset_means(losses: pd.DataFrame) -> pd.DataFrame:
    if losses.empty:
        return pd.DataFrame()
    metrics = present_metrics(losses, ["accuracy", "macro_f1", "log_loss", "ece"])
    frame = losses[losses["variant"].isin(LOSS_WEIGHT_VARIANTS)].copy()
    if frame.empty or not metrics:
        return pd.DataFrame()
    frame["loss_weight"] = frame["variant"].map(LOSS_WEIGHTS)
    return (
        frame.groupby(["dataset_id", "dataset_alias", "variant", "loss_weight"], dropna=False)[metrics]
        .mean(numeric_only=True)
        .reset_index()
    )


def write_hybrid_weight_sweep(loss_means: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    if loss_means.empty:
        return
    metrics = [metric for metric in ["accuracy", "log_loss", "ece"] if metric in loss_means.columns]
    if not metrics:
        return
    full = summary[summary["variant"] == "full"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.25 * len(metrics), 3.1), sharex=True)
    if len(metrics) == 1:
        axes = np.array([axes])
    for axis, metric in zip(axes, metrics):
        for _, group in loss_means.groupby("dataset_id", dropna=False):
            group = group.sort_values("loss_weight")
            axis.plot(group["loss_weight"], group[metric], color=MUTED_COLOR, alpha=0.36, linewidth=0.9)
        by_weight = (
            loss_means.groupby("loss_weight", dropna=False)[metric]
            .agg(mean="mean", q25=lambda value: value.quantile(0.25), q75=lambda value: value.quantile(0.75))
            .reset_index()
            .sort_values("loss_weight")
        )
        axis.fill_between(by_weight["loss_weight"], by_weight["q25"], by_weight["q75"], color=NEUTRAL_COLOR, alpha=0.12)
        axis.plot(by_weight["loss_weight"], by_weight["mean"], color=NEUTRAL_COLOR, marker="o", markersize=4, linewidth=1.8)
        if not full.empty and metric in full.columns:
            axis.axhline(float(full.iloc[0][metric]), color="#27313a", linestyle="--", linewidth=0.9, alpha=0.68)
        axis.set_title(metric_label(metric))
        axis.set_xlabel("Joint-loss weight")
        axis.grid(axis="y", color=GRID_COLOR, linewidth=0.7)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].set_ylabel("Metric value")
    fig.suptitle("Hybrid Objective Sweep", y=1.03, fontsize=11, fontweight="semibold")
    fig.tight_layout()
    save_figure(fig, path)


def write_graph_diagnostics(summary: pd.DataFrame, path: Path) -> None:
    metrics = [metric for metric in GRAPH_METRICS if metric in summary.columns and summary[metric].notna().any()]
    if not metrics:
        return
    plot_frame = summary[summary["variant"] != "full"].copy()
    if plot_frame.empty:
        return
    labels = plot_frame["variant_label"].tolist()
    x = np.arange(plot_frame.shape[0])
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.25 * len(metrics), 3.3), sharex=True)
    if len(metrics) == 1:
        axes = np.array([axes])
    for axis, metric in zip(axes, metrics):
        axis.bar(x, plot_frame[metric], color="#6f9f8d", edgecolor="none")
        axis.set_title(metric_label(metric))
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=38, ha="right")
        axis.grid(axis="y", color=GRID_COLOR, linewidth=0.7)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.suptitle("Learned-Graph Diagnostics", y=1.03, fontsize=11, fontweight="semibold")
    fig.tight_layout()
    save_figure(fig, path)


def write_accuracy_vs_graph(summary: pd.DataFrame, path: Path) -> None:
    if "accuracy" not in summary.columns or "edge_mass" not in summary.columns:
        return
    plot_frame = summary.dropna(subset=["accuracy", "edge_mass"]).copy()
    if plot_frame.empty:
        return
    fig, ax = plt.subplots(figsize=(5.9, 4.1))
    colors = []
    for variant in plot_frame["variant"]:
        if variant == "full":
            colors.append("#27313a")
        elif variant in LOSS_WEIGHT_VARIANTS:
            colors.append("#d08b39")
        else:
            colors.append(NEUTRAL_COLOR)
    sizes = 95 + 420 * plot_frame.get("parent_limit_violation_rate", pd.Series(0.0, index=plot_frame.index)).fillna(0.0)
    ax.scatter(
        plot_frame["edge_mass"],
        plot_frame["accuracy"],
        s=sizes,
        color=colors,
        alpha=0.92,
        edgecolor="white",
        linewidth=0.7,
    )
    offsets = [(6, 0), (6, 8), (6, -8), (-6, 8), (-6, -8)]
    for index, row in enumerate(plot_frame.itertuples()):
        dx, dy = offsets[index % len(offsets)]
        ax.annotate(
            row.variant_label,
            (row.edge_mass, row.accuracy),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8,
            ha="left" if dx >= 0 else "right",
            va="center",
        )
    ax.set_title("Accuracy vs Graph Edge Mass", fontsize=11, fontweight="semibold")
    ax.set_xlabel("Mean edge mass")
    ax.set_ylabel("Mean accuracy")
    ax.grid(color=GRID_COLOR, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_figure(fig, path)


def html_table(frame: pd.DataFrame, columns: list[str]) -> str:
    rows = []
    for _, row in frame.iterrows():
        cells = []
        for column in columns:
            value = row.get(column)
            if isinstance(value, float):
                cells.append(f"<td>{value:.4g}</td>")
            elif pd.isna(value):
                cells.append("<td></td>")
            else:
                cells.append(f"<td>{value}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    header = "".join(f"<th>{column.replace('_', ' ')}</th>" for column in columns)
    return "<table><thead><tr>" + header + "</tr></thead><tbody>" + "\n".join(rows) + "</tbody></table>"


def main_ablation_table(summary: pd.DataFrame) -> pd.DataFrame:
    table = summary[summary["variant"].isin(MAIN_TABLE_VARIANTS)].copy()
    table["_variant_order"] = table["variant"].map(lambda value: variant_sort_key(str(value))[0])
    table = table.sort_values(["_variant_order", "variant"], kind="mergesort").drop(columns="_variant_order")
    columns = ["variant", "variant_label", *[column for column in MAIN_TABLE_METRICS if column in table.columns]]
    return table[columns].reset_index(drop=True)


def write_main_table_outputs(table: pd.DataFrame, reports_root: Path) -> dict[str, str]:
    tables_root = reports_root / "tables"
    tables_root.mkdir(parents=True, exist_ok=True)
    csv_path = tables_root / "ablation_main_table.csv"
    md_path = tables_root / "ablation_main_table.md"
    table.to_csv(csv_path, index=False)
    markdown = table.rename(
        columns={
            "variant_label": "Variant",
            "accuracy": "Accuracy",
            "accuracy_delta_vs_full": "Delta Accuracy",
            "log_loss": "Log Loss",
            "log_loss_delta_vs_full": "Delta Log Loss",
            "ece": "ECE",
            "ece_delta_vs_full": "Delta ECE",
        }
    ).drop(columns=["variant"], errors="ignore")
    md_path.write_text(markdown.to_markdown(index=False, floatfmt=".4f") + "\n", encoding="utf-8")
    return {"main_table_csv": str(csv_path), "main_table_markdown": str(md_path)}


def write_html_report(
    reports_root: Path,
    summary: pd.DataFrame,
    main_table: pd.DataFrame,
    report_summary: dict[str, Any],
    figure_paths: list[Path],
) -> Path:
    complete = report_summary.get("datasets_complete_for_ablation") or report_summary.get(
        "datasets_complete_for_comparison", []
    )
    protocol = report_summary.get("protocol_complete", "n/a")
    main_figure = reports_root / "figures" / "hybrid_weight_sweep.png"
    supplementary = [path for path in figure_paths if path != main_figure]

    def figure_section(paths: list[Path]) -> str:
        figure_items = []
        for path in paths:
            if not path.exists() or path.stat().st_size <= 0:
                continue
            rel = path.relative_to(reports_root)
            title = path.stem.replace("_", " ").title()
            figure_items.append(
                f"<section><h2>{title}</h2><img src=\"{rel.as_posix()}\" alt=\"{title}\"></section>"
            )
        return "".join(figure_items)

    main_figure_html = figure_section([main_figure])
    supplementary_html = figure_section(supplementary)
    table_columns = [
        column
        for column in ["variant_label", *MAIN_TABLE_METRICS]
        if column in main_table.columns
    ]
    supplementary_table_columns = [
        column
        for column in [
            "variant_label",
            "accuracy",
            "accuracy_delta_vs_full",
            "macro_f1_delta_vs_full",
            "log_loss_delta_vs_full",
            "ece_delta_vs_full",
            "edge_mass",
            "thresholded_edge_density",
            "parent_limit_violation_rate",
        ]
        if column in summary.columns
    ]
    pdf_links = []
    for path in figure_paths:
        if path.exists() and path.stat().st_size > 0:
            pdf = path.with_suffix(".pdf")
            if pdf.exists():
                pdf_links.append(
                    f"<li><a href=\"{pdf.relative_to(reports_root).as_posix()}\">{pdf.name}</a></li>"
                )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DBNC Ablation Visual Report</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #17212b; background: #f7f8fa; }}
    main {{ max-width: 1080px; margin: 0 auto; padding: 28px 24px 48px; }}
    header {{ margin-bottom: 24px; }}
    h1 {{ font-size: 28px; margin: 0 0 8px; }}
    h2 {{ font-size: 19px; margin: 0 0 12px; }}
    h3 {{ font-size: 15px; margin: 8px 0 8px; color: #344251; }}
    p {{ margin: 0 0 6px; color: #52616f; }}
    section {{ background: #fff; border: 1px solid #dfe5eb; border-radius: 8px; padding: 18px; margin: 18px 0; }}
    img {{ width: 100%; height: auto; display: block; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; background: #fff; }}
    th, td {{ padding: 8px 9px; border-bottom: 1px solid #e6ebf0; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ color: #344251; background: #eef2f5; }}
    .note {{ color: #667686; font-size: 13px; }}
    .links {{ margin: 8px 0 0 18px; color: #52616f; }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>DBNC Ablation Report</h1>
      <p>Complete datasets: {len(complete)}. Protocol complete: {protocol}.</p>
      <p>Main-paper table: architecture/component variants only. Main-paper figure: hybrid loss-weight sweep.</p>
      <p class="note">Full is the reference. Positive accuracy deltas are better; negative log-loss and ECE deltas are better. The no-regularization variant is excluded from this visual report.</p>
    </header>
    <section>
      <h2>Main Paper Table: Component Ablations</h2>
      {html_table(main_table, table_columns)}
    </section>
    {main_figure_html}
    <section>
      <h2>Supplementary Variant Summary</h2>
      <p class="note">Includes loss-weight rows for traceability; use the curve figure in the main paper.</p>
      {html_table(summary, supplementary_table_columns)}
      <h3>Vector figure files</h3>
      <ul class="links">{''.join(pdf_links)}</ul>
    </section>
    {supplementary_html}
  </main>
</body>
</html>
"""
    output = reports_root / "ablation_visual_report.html"
    output.write_text(html, encoding="utf-8")
    return output


def generate(reports_root: Path) -> dict[str, Any]:
    reports_root = reports_root.resolve()
    figures_root = reports_root / "figures"
    figures_root.mkdir(parents=True, exist_ok=True)

    core = require_nonempty_csv(reports_root / "core_ablation_table.csv")
    losses = read_optional_csv(reports_root / "loss_weight_results.csv")
    graph = read_optional_csv(reports_root / "graph_diagnostics.csv")
    core = filter_variants(core, DEFAULT_EXCLUDED_VARIANTS)
    losses = filter_variants(losses, DEFAULT_EXCLUDED_VARIANTS)
    graph = filter_variants(graph, DEFAULT_EXCLUDED_VARIANTS)
    report_summary = read_optional_json(reports_root / "report_summary.json") or read_optional_json(
        reports_root / "ablation_summary.json"
    )

    metrics = present_metrics(core)
    dataset_metrics = aggregate_dataset_metrics(core, losses)
    summary = summarize_variants(dataset_metrics, graph, metrics)
    summary_path = figures_root / "ablation_figure_summary.csv"
    summary.to_csv(summary_path, index=False)
    main_table = main_ablation_table(summary)
    table_outputs = write_main_table_outputs(main_table, reports_root)

    loss_means = loss_dataset_means(losses if not losses.empty else dataset_metrics)
    figure_paths = [
        figures_root / "ablation_delta_vs_full.png",
        figures_root / "accuracy_delta_heatmap.png",
        figures_root / "hybrid_weight_sweep.png",
        figures_root / "graph_diagnostics_by_variant.png",
        figures_root / "accuracy_vs_edge_mass.png",
    ]
    write_delta_bar_plot(summary, metrics, figure_paths[0])
    write_accuracy_heatmap(dataset_metrics, figure_paths[1])
    write_hybrid_weight_sweep(loss_means, summary, figure_paths[2])
    write_graph_diagnostics(summary, figure_paths[3])
    write_accuracy_vs_graph(summary, figure_paths[4])
    html_path = write_html_report(reports_root, summary, main_table, report_summary, figure_paths)

    written_figures = [path for path in figure_paths if path.exists() and path.stat().st_size > 0]
    if len(written_figures) < 2:
        raise RuntimeError("Generated too few ablation figures; check input CSV contents.")
    return {
        "reports_root": str(reports_root),
        "summary_csv": str(summary_path),
        **table_outputs,
        "html_report": str(html_path),
        "figures": [str(path) for path in written_figures],
        "pdf_figures": [str(path.with_suffix(".pdf")) for path in written_figures if path.with_suffix(".pdf").exists()],
        "excluded_variants": sorted(DEFAULT_EXCLUDED_VARIANTS),
        "variants": int(summary.shape[0]),
        "datasets": int(dataset_metrics["dataset_id"].nunique()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reports",
        type=Path,
        default=default_reports_root(),
        help="Directory containing ablation CSV/JSON reports.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = generate(args.reports)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
