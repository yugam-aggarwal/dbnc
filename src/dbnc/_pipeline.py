#!/usr/bin/env python3
"""DBNC experimental runner for Phase 0 validation and Phase 1 benchmarking.

The file is deliberately self-contained: reportable experiment execution does
not import the prototype in ``dbnc.py``.  The classical BNC factorizations are:

* NB: ``P(Y) prod_j P(X_j | Y)``.
* Chow-Liu: a maximum-mutual-information tree over ``(Y, X)`` rooted at Y,
  evaluated as the generative tree factorization.
* TAN: ``P(Y) prod_j P(X_j | Y, X_parent(j))`` where feature parents form a
  maximum conditional-mutual-information tree learned on training data.
* BAN: the same class-conditioned factorization, adding up to two earlier
  feature parents per node in decreasing conditional-mutual-information order.
* KDB-k: features are ordered by mutual information with Y and each feature
  receives up to k higher-ranked parents with the greatest positive
  conditional mutual information.

All structure discovery, encoders, imputers, binning, and scaling are fitted
on training partitions only.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import shutil
import sys
import tempfile
import time
import traceback
import warnings
from dataclasses import asdict, dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mutual_info_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from ._common import ExperimentError, MissingDependency, GRAPH_THRESHOLD, is_acyclic, validation_checkpoint_improved
from .model import DBNC

# Some installed model packages import plotting support eagerly.  Keep their
# cache writable even when the user's home configuration directory is read-only.
_MPL_CACHE = Path(tempfile.gettempdir()) / "dbnc-matplotlib"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))


# ---------------------------------------------------------------------------
# Fixed protocol declarations
# ---------------------------------------------------------------------------

SEEDS = (0, 1, 2, 3, 4)
N_BINS = 8
RARE_MIN_COUNT = 5
ECE_BINS = 15
DBNC_SEARCH_SPACE_VERSION = "dbnc_v4_predictive_attention_graph_bins"
MODES = ("retuned", "full_config_toggle", "main")

DATASETS: tuple[tuple[str, int], ...] = (
    ("heart-statlog", 53),
    ("vote", 56),
    ("wdbc", 1510),
    ("credit-approval", 29),
    ("breast-w", 15),
    ("car", 40975),
    ("spambase", 44),
    ("optdigits", 28),
    ("mushroom", 24),
    ("PhishingWebsites", 4534),
    ("MagicTelescope", 1120),
    ("letter", 6),
    ("nomao", 1486),
    ("bank-marketing", 1461),
    ("adult", 1590),
    ("jannis", 41168),
    ("MiniBooNE", 41150),
    ("GiveMeSomeCredit", 46929),
    ("diabetes", 37),
    ("tic-tac-toe", 50),
    ("vehicle", 54),
    ("cmc", 23),
    ("kr-vs-kp", 3),
    ("segment", 36),
    ("phoneme", 1489),
    ("satimage", 182),
    ("electricity", 151),
    ("default-of-credit-card-clients", 42477),
    ("covertype-binary", 293),
    ("poker-hand", 1567),
)
ALIAS_TO_ID = dict(DATASETS)
ID_TO_ALIAS = {dataset_id: alias for alias, dataset_id in DATASETS}

MAIN_METHODS = (
    "dbnc",
    "nb",
    "chow_liu",
    "tan",
    "ban",
    "kdb_1",
    "kdb_2",
    "kdb_3",
    "xgboost",
    "catboost",
    "lightgbm",
    "ft_transformer",
)
METHOD_LABELS = {
    "dbnc": "DBNC",
    "nb": "NB",
    "chow_liu": "Chow-Liu",
    "tan": "TAN",
    "ban": "BAN",
    "kdb_1": "KDB-1",
    "kdb_2": "KDB-2",
    "kdb_3": "KDB-3",
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
    "lightgbm": "LightGBM",
    "ft_transformer": "FT-Transformer",
}
DBNC_VARIANTS = (
    "full",
    "fixed_tan",
    "conditional_only",
    "mlp_cpd",
    "no_regularization",
    "no_edges",
    "no_sparse",
    "no_degree",
    "loss_w_000",
    "loss_w_025",
    "loss_w_050",
    "loss_w_075",
    "loss_w_100",
)
CORE_ABLATIONS = (
    "full",
    "fixed_tan",
    "conditional_only",
    "mlp_cpd",
    "no_regularization",
)
DBNC_MODEL_PARAMETER_KEYS = frozenset(
    {
        "d",
        "n_heads",
        "epochs",
        "lr",
        "batch_size",
        "lambda_sparse",
        "lambda_degree",
        "max_parents",
        "hybrid_weight",
        "device",
        "dropout",
        "cpd_type",
        "fixed_adjacency",
        "early_stopping_rounds",
        "min_epochs_before_checkpoint",
        "min_epochs_before_stopping",
        "structure_warmup_epochs",
        "edge_logit_mean",
        "rank_init_std",
    }
)


# ---------------------------------------------------------------------------
# Generic artifact and reproducibility helpers
# ---------------------------------------------------------------------------




def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def stable_id(value: Any, length: int = 16) -> str:
    payload = json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Mapping[str, Any], *, immutable: bool = False) -> None:
    normalized = jsonable(payload)
    if immutable and path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != normalized:
            raise ExperimentError(f"Immutable artifact already exists with different content: {path}")
        return
    ensure_parent(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(normalized, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def append_failure(root: Path, record: Mapping[str, Any]) -> None:
    path = root / "results" / "execution_failures.jsonl"
    ensure_parent(path)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(jsonable(record), sort_keys=True) + "\n")


def package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def environment_metadata(device: str, *, concurrent: bool = False) -> dict[str, Any]:
    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "device_requested": device,
        "torch_device": str(torch.device(device)),
        "cuda_available": torch.cuda.is_available(),
        "packages": {
            name: package_version(name)
            for name in (
                "numpy",
                "pandas",
                "scikit-learn",
                "scipy",
                "torch",
                "openml",
                "optuna",
                "xgboost",
                "catboost",
                "lightgbm",
                "matplotlib",
            )
        },
        "thread_limits": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "jobs_concurrent": bool(concurrent),
    }
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        metadata["gpu_name"] = torch.cuda.get_device_name(torch.device(device))
    return metadata


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync_device(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(device))


def parse_dataset_selection(values: Sequence[str] | None) -> list[tuple[str, int]]:
    if not values:
        return list(DATASETS)
    chosen: list[tuple[str, int]] = []
    for value in values:
        if value in ALIAS_TO_ID:
            chosen.append((value, ALIAS_TO_ID[value]))
            continue
        try:
            dataset_id = int(value)
        except ValueError as exc:
            raise ExperimentError(f"Unknown dataset alias or OpenML ID: {value}") from exc
        if dataset_id not in ID_TO_ALIAS:
            raise ExperimentError(f"OpenML ID is outside the declared manifest: {dataset_id}")
        chosen.append((ID_TO_ALIAS[dataset_id], dataset_id))
    return list(dict.fromkeys(chosen))


def result_path(
    root: Path,
    method: str,
    variant: str,
    dataset_id: int,
    seed: int,
    mode: str,
    hp_id: str,
) -> Path:
    if mode not in MODES:
        raise ExperimentError(f"Unsupported result mode: {mode}")
    values = (method, variant, str(dataset_id), str(seed), mode, hp_id)
    if any("/" in value or ".." in value for value in values):
        raise ExperimentError("Result identifiers must be path-safe")
    return root / "results" / method / variant / str(dataset_id) / str(seed) / mode / f"{hp_id}.json"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def normalized_probabilities(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ExperimentError("Predicted probabilities must be an N by K array with K >= 2")
    if not np.isfinite(probabilities).all():
        raise ExperimentError("Predicted probabilities contain non-finite values")
    probabilities = np.clip(probabilities, 0.0, None)
    totals = probabilities.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ExperimentError("Predicted probability row has no positive mass")
    return probabilities / totals


def multiclass_brier_score(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[np.asarray(y_true, dtype=int)]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = ECE_BINS,
) -> float:
    prediction = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = prediction == np.asarray(y_true)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    value = 0.0
    for index in range(n_bins):
        if index == 0:
            mask = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            mask = (confidence > edges[index]) & (confidence <= edges[index + 1])
        if np.any(mask):
            value += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(value)


def predictive_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = normalized_probabilities(probabilities)
    prediction = probabilities.argmax(axis=1)
    n_classes = probabilities.shape[1]
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "macro_f1": float(f1_score(y_true, prediction, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y_true, probabilities, labels=np.arange(n_classes))),
        "brier": multiclass_brier_score(y_true, probabilities),
        "ece": expected_calibration_error(y_true, probabilities),
    }
    present = np.unique(y_true)
    if present.shape[0] != n_classes:
        metrics["auc"] = None
        metrics["auc_status"] = "undefined_missing_test_class"
    else:
        try:
            if n_classes == 2:
                metrics["auc"] = float(roc_auc_score(y_true, probabilities[:, 1]))
            else:
                metrics["auc"] = float(
                    roc_auc_score(
                        y_true,
                        probabilities,
                        labels=np.arange(n_classes),
                        average="macro",
                        multi_class="ovr",
                    )
                )
            metrics["auc_status"] = "defined"
        except ValueError as exc:
            metrics["auc"] = None
            metrics["auc_status"] = f"undefined:{exc}"
    return metrics


# ---------------------------------------------------------------------------
# Dataset inspection, splits, and training-only transformations
# ---------------------------------------------------------------------------


@dataclass
class DatasetBundle:
    alias: str
    dataset_id: int
    version: int | None
    openml_name: str
    target: str
    X: pd.DataFrame
    y_raw: pd.Series
    categorical_columns: list[str]
    numeric_columns: list[str]

    @property
    def class_labels(self) -> list[str]:
        return sorted(self.y_raw.astype(str).unique().tolist())

    @property
    def y(self) -> np.ndarray:
        mapping = {label: index for index, label in enumerate(self.class_labels)}
        return self.y_raw.astype(str).map(mapping).to_numpy(dtype=np.int64)


def load_openml_dataset(alias: str, dataset_id: int) -> DatasetBundle:
    try:
        import openml
    except ImportError as exc:
        raise MissingDependency("OpenML inspection requires the 'openml' package") from exc

    dataset = openml.datasets.get_dataset(
        dataset_id,
        download_data=True,
        download_qualities=False,
        download_features_meta_data=True,
    )
    target = dataset.default_target_attribute
    if not target:
        raise ExperimentError(f"OpenML dataset {dataset_id} has no declared default target")
    try:
        X, y, categorical_indicator, _ = dataset.get_data(
            dataset_format="dataframe",
            target=target,
        )
    except TypeError as exc:
        # OpenML 0.15 calls pandas.factorize on ARFF nominal lists, which
        # pandas 3 rejects unless they are converted to an array first.
        if "factorize requires" not in str(exc):
            raise
        factorize = pd.factorize

        def compatible_factorize(values: Any, *args: Any, **kwargs: Any) -> Any:
            if isinstance(values, list):
                values = np.asarray(values, dtype=object)
            return factorize(values, *args, **kwargs)

        pd.factorize = compatible_factorize  # type: ignore[assignment]
        try:
            X, y, categorical_indicator, _ = dataset.get_data(
                dataset_format="dataframe",
                target=target,
            )
        finally:
            pd.factorize = factorize  # type: ignore[assignment]
    if y is None or y.isna().any():
        raise ExperimentError(f"Dataset {dataset_id} has missing target observations")
    if categorical_indicator is None:
        categorical_indicator = [
            not pd.api.types.is_numeric_dtype(X[column]) for column in X.columns
        ]
    categorical_columns = [
        str(column) for column, is_cat in zip(X.columns, categorical_indicator) if is_cat
    ]
    numeric_columns = [str(column) for column in X.columns if str(column) not in categorical_columns]
    X = X.copy()
    X.columns = [str(column) for column in X.columns]
    return DatasetBundle(
        alias=alias,
        dataset_id=dataset_id,
        version=getattr(dataset, "version", None),
        openml_name=str(dataset.name),
        target=str(target),
        X=X,
        y_raw=pd.Series(y).reset_index(drop=True),
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
    )


def configure_openml_cache(root: Path) -> None:
    try:
        import openml
    except ImportError as exc:
        raise MissingDependency("OpenML inspection requires the 'openml' package") from exc
    cache_directory = root / "cache" / "openml"
    cache_directory.mkdir(parents=True, exist_ok=True)
    openml.config.set_root_cache_directory(str(cache_directory))


def split_indices(y: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    positions = np.arange(len(y), dtype=np.int64)
    train, remainder = train_test_split(
        positions,
        test_size=0.30,
        random_state=seed,
        shuffle=True,
        stratify=y,
    )
    validation, test = train_test_split(
        remainder,
        test_size=0.50,
        random_state=seed,
        shuffle=True,
        stratify=y[remainder],
    )
    return {
        "train": np.sort(train),
        "validation": np.sort(validation),
        "test": np.sort(test),
    }


def class_covering_training_subset(
    indices: np.ndarray,
    y: np.ndarray,
    count: int,
    seed: int,
) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if count >= indices.shape[0]:
        return np.sort(indices)
    labels = np.asarray(y, dtype=np.int64)[indices]
    classes = np.unique(labels)
    if count < classes.shape[0]:
        raise ExperimentError(
            f"Training cap {count} is smaller than the {classes.shape[0]} observed classes"
        )
    rng = np.random.default_rng(seed)
    selected_positions: list[int] = []
    available = np.ones(indices.shape[0], dtype=bool)
    for label in classes:
        class_positions = np.flatnonzero(labels == label)
        chosen = int(rng.choice(class_positions))
        selected_positions.append(chosen)
        available[chosen] = False
    remaining_needed = count - len(selected_positions)
    if remaining_needed:
        remaining_positions = np.flatnonzero(available)
        selected_positions.extend(
            int(position)
            for position in rng.choice(remaining_positions, size=remaining_needed, replace=False)
        )
    return np.sort(indices[np.asarray(selected_positions, dtype=np.int64)])


def persisted_split(root: Path, bundle: DatasetBundle, seed: int) -> tuple[dict[str, np.ndarray], str]:
    try:
        indices = split_indices(bundle.y, seed)
    except ValueError as exc:
        raise ExperimentError(f"Dataset {bundle.dataset_id} cannot support split seed {seed}: {exc}") from exc
    serial = {
        "dataset_id": bundle.dataset_id,
        "dataset_version": bundle.version,
        "target": bundle.target,
        "seed": seed,
        "strategy": "stratified_holdout_70_15_15",
        "indices": {name: values.tolist() for name, values in indices.items()},
    }
    split_id = stable_id(serial)
    serial["split_id"] = split_id
    write_json(root / "cache" / "splits" / str(bundle.dataset_id) / f"{seed}.json", serial)
    return indices, split_id


def inspect_dataset(root: Path, alias: str, dataset_id: int) -> tuple[dict[str, Any], DatasetBundle | None]:
    try:
        configure_openml_cache(root)
        bundle = load_openml_dataset(alias, dataset_id)
        class_counts = bundle.y_raw.astype(str).value_counts().sort_index()
        missing_by_feature = bundle.X.isna().sum()
        exclusions: list[str] = []
        for seed in SEEDS:
            try:
                persisted_split(root, bundle, seed)
            except ExperimentError as exc:
                exclusions.append(str(exc))
        if dataset_id == 293 and len(class_counts) != 2:
            raise ExperimentError(
                "OpenML ID 293 did not retrieve the mandatory binary covertype target"
            )
        record = {
            "alias": alias,
            "dataset_id": dataset_id,
            "openml_name": bundle.openml_name,
            "version": bundle.version,
            "target": bundle.target,
            "n_rows": int(bundle.X.shape[0]),
            "n_predictors": int(bundle.X.shape[1]),
            "predictor_types": {
                "categorical": bundle.categorical_columns,
                "numeric": bundle.numeric_columns,
            },
            "n_classes": int(len(class_counts)),
            "class_counts": {str(key): int(value) for key, value in class_counts.items()},
            "missing_values": {
                "total": int(missing_by_feature.sum()),
                "by_predictor": {str(key): int(value) for key, value in missing_by_feature.items()},
            },
            "split_seeds_checked": list(SEEDS),
            "status": "excluded" if exclusions else "accepted",
            "exclusion_reasons": exclusions,
        }
        return record, bundle
    except Exception as exc:
        return {
            "alias": alias,
            "dataset_id": dataset_id,
            "status": "failed",
            "failure": f"{type(exc).__name__}: {exc}",
        }, None


def validate_manifest(root: Path, selected: list[tuple[str, int]]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for alias, dataset_id in selected:
        record, _ = inspect_dataset(root, alias, dataset_id)
        records.append(record)
        print(f"[manifest] {alias} ({dataset_id}): {record['status']}", flush=True)
    ids = {dataset_id for _, dataset_id in selected}
    complete_scope = ids == set(ID_TO_ALIAS)
    no_failed_retrievals = all(record["status"] != "failed" for record in records)
    output = {
        "protocol": "phase_0_dataset_manifest",
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "complete_declared_scope": complete_scope,
        "validation_complete": complete_scope and no_failed_retrievals,
        "records": records,
        "accepted_dataset_ids": [
            record["dataset_id"] for record in records if record["status"] == "accepted"
        ],
        "excluded_dataset_ids": [
            record["dataset_id"] for record in records if record["status"] == "excluded"
        ],
        "failed_dataset_ids": [
            record["dataset_id"] for record in records if record["status"] == "failed"
        ],
    }
    write_json(root / "artifacts" / "validated_manifest.json", output)
    return output


def _categorical_value(value: Any) -> str:
    if pd.isna(value):
        return "<MISSING>"
    return "VALUE:" + str(value)


def _numeric_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if isinstance(numeric.dtype, pd.SparseDtype):
        return numeric.sparse.to_dense()
    return numeric


@dataclass
class MatrixParts:
    discrete: np.ndarray
    tree: np.ndarray
    neural_numeric: np.ndarray
    neural_categorical: np.ndarray


class TrainingPreprocessor:
    """Builds discrete, tree, and neural views using only training rows."""

    def __init__(
        self,
        columns: Sequence[str],
        categorical_columns: Sequence[str],
        numeric_columns: Sequence[str],
        numeric_bins: int = N_BINS,
    ) -> None:
        self.columns = list(columns)
        self.categorical_columns = list(categorical_columns)
        self.numeric_columns = list(numeric_columns)
        self.numeric_bins = int(numeric_bins)
        if self.numeric_bins < 2:
            raise ValueError("numeric_bins must be at least 2")
        self.category_maps: dict[str, dict[str, int]] = {}
        self.rare_values: dict[str, set[str]] = {}
        self.numeric_medians: dict[str, float] = {}
        self.numeric_means: dict[str, float] = {}
        self.numeric_scales: dict[str, float] = {}
        self.binners: dict[str, KBinsDiscretizer] = {}
        self.bin_edges: dict[str, list[float]] = {}
        self.effective_numeric_bins: dict[str, int] = {}
        self.discrete_cards: list[int] = []
        self.categorical_cards: list[int] = []
        self.fitted = False

    def fit(self, X_train: pd.DataFrame) -> "TrainingPreprocessor":
        for column in self.categorical_columns:
            values = X_train[column].map(_categorical_value)
            counts = values.value_counts(dropna=False)
            common = sorted(
                value
                for value, count in counts.items()
                if value == "<MISSING>" or int(count) >= RARE_MIN_COUNT
            )
            self.rare_values[column] = {
                value
                for value, count in counts.items()
                if value != "<MISSING>" and int(count) < RARE_MIN_COUNT
            }
            levels = ["<MISSING>", "<RARE>", "<UNKNOWN>"]
            for value in common:
                if value not in levels:
                    levels.append(value)
            self.category_maps[column] = {value: index for index, value in enumerate(levels)}
        for column in self.numeric_columns:
            numeric = _numeric_series(X_train[column])
            median = float(numeric.median()) if not numeric.dropna().empty else 0.0
            imputed = numeric.fillna(median).to_numpy(dtype=np.float64).reshape(-1, 1)
            mean = float(imputed.mean())
            scale = float(imputed.std())
            if not np.isfinite(scale) or scale == 0.0:
                scale = 1.0
            binner = KBinsDiscretizer(
                n_bins=self.numeric_bins,
                encode="ordinal",
                strategy="quantile",
                quantile_method="linear",
            )
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Bins whose width are too small.*",
                    category=UserWarning,
                )
                warnings.filterwarnings(
                    "ignore",
                    message="Feature .* is constant and will be replaced with 0.*",
                    category=UserWarning,
                )
                binner.fit(imputed)
            self.numeric_medians[column] = median
            self.numeric_means[column] = mean
            self.numeric_scales[column] = scale
            self.binners[column] = binner
            self.bin_edges[column] = binner.bin_edges_[0].tolist()
            self.effective_numeric_bins[column] = int(binner.n_bins_[0])
        for column in self.columns:
            if column in self.category_maps:
                self.discrete_cards.append(len(self.category_maps[column]))
            else:
                self.discrete_cards.append(int(self.binners[column].n_bins_[0]))
        self.categorical_cards = [
            len(self.category_maps[column]) for column in self.categorical_columns
        ]
        self.fitted = True
        return self

    def _map_category(self, column: str, series: pd.Series) -> np.ndarray:
        mapping = self.category_maps[column]
        raw_values = series.map(_categorical_value)
        known = raw_values.map(mapping)
        values: list[int] = []
        for raw, code in zip(raw_values, known):
            if not pd.isna(code):
                values.append(int(code))
            elif raw in self.rare_values[column]:
                values.append(mapping["<RARE>"])
            else:
                values.append(mapping["<UNKNOWN>"])
        return np.asarray(values, dtype=np.int64)

    def transform(self, X: pd.DataFrame) -> MatrixParts:
        if not self.fitted:
            raise ExperimentError("Preprocessor must be fitted before transform")
        discrete_columns: list[np.ndarray] = []
        tree_columns: list[np.ndarray] = []
        numeric_neural: list[np.ndarray] = []
        cat_neural: list[np.ndarray] = []
        for column in self.columns:
            if column in self.category_maps:
                codes = self._map_category(column, X[column])
                discrete_columns.append(codes)
                tree_columns.append(codes.astype(np.float64))
                cat_neural.append(codes)
            else:
                numeric = _numeric_series(X[column])
                values = numeric.fillna(self.numeric_medians[column]).to_numpy(dtype=np.float64)
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="Feature .* is constant and will be replaced with 0.*",
                        category=UserWarning,
                    )
                    binned = (
                        self.binners[column]
                        .transform(values.reshape(-1, 1))
                        .reshape(-1)
                        .astype(np.int64)
                    )
                discrete_columns.append(binned)
                tree_columns.append(values)
                numeric_neural.append(
                    (values - self.numeric_means[column]) / self.numeric_scales[column]
                )
        size = X.shape[0]
        return MatrixParts(
            discrete=np.column_stack(discrete_columns).astype(np.int64)
            if discrete_columns else np.empty((size, 0), dtype=np.int64),
            tree=np.column_stack(tree_columns).astype(np.float64)
            if tree_columns else np.empty((size, 0), dtype=np.float64),
            neural_numeric=np.column_stack(numeric_neural).astype(np.float32)
            if numeric_neural else np.empty((size, 0), dtype=np.float32),
            neural_categorical=np.column_stack(cat_neural).astype(np.int64)
            if cat_neural else np.empty((size, 0), dtype=np.int64),
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "columns": self.columns,
            "categorical_columns": self.categorical_columns,
            "numeric_columns": self.numeric_columns,
            "categorical_levels": {
                column: sorted(mapping, key=mapping.get)
                for column, mapping in self.category_maps.items()
            },
            "rare_training_levels": {
                column: sorted(values) for column, values in self.rare_values.items()
            },
            "numeric_medians": self.numeric_medians,
            "numeric_means": self.numeric_means,
            "numeric_scales": self.numeric_scales,
            "bin_edges": self.bin_edges,
            "effective_numeric_bins": self.effective_numeric_bins,
            "discrete_cardinalities": self.discrete_cards,
            "categorical_cardinalities": self.categorical_cards,
            "rare_min_count": RARE_MIN_COUNT,
            "numeric_bins_requested": self.numeric_bins,
            "numeric_bin_quantile_method": "linear",
            "fitted_on": "training_partition_only",
        }


@dataclass
class PreparedSplit:
    dataset: DatasetBundle
    seed: int
    split_id: str
    training_subset_id: str | None
    transform_id: str
    transform_metadata: dict[str, Any]
    cards: list[int]
    categorical_cards: list[int]
    categorical_tree_indices: list[int]
    y_train: np.ndarray
    y_validation: np.ndarray
    y_test: np.ndarray
    train: MatrixParts
    validation: MatrixParts
    test: MatrixParts

    @property
    def n_classes(self) -> int:
        return len(self.dataset.class_labels)


def prepare_split(
    root: Path,
    bundle: DatasetBundle,
    seed: int,
    *,
    numeric_bins: int = N_BINS,
    tuning_training_cap: int | None = None,
) -> PreparedSplit:
    indices, split_id = persisted_split(root, bundle, seed)
    train_indices = indices["train"]
    training_subset_id: str | None = None
    if (
        tuning_training_cap is not None
        and bundle.X.shape[0] > 200_000
        and train_indices.shape[0] > tuning_training_cap
    ):
        train_indices = class_covering_training_subset(
            train_indices,
            bundle.y,
            tuning_training_cap,
            seed,
        )
        subset_record = {
            "dataset_id": bundle.dataset_id,
            "seed": seed,
            "split_id": split_id,
            "purpose": "tuning_training_cap",
            "cap": tuning_training_cap,
            "sampling_policy": "class_covering_uniform_remainder",
            "indices": train_indices.tolist(),
        }
        training_subset_id = stable_id(subset_record)
        subset_record["training_subset_id"] = training_subset_id
        write_json(
            root
            / "cache"
            / "tuning_subsets"
            / str(bundle.dataset_id)
            / f"{seed}_{training_subset_id}.json",
            subset_record,
        )
    transform = TrainingPreprocessor(
        bundle.X.columns.tolist(),
        bundle.categorical_columns,
        bundle.numeric_columns,
        numeric_bins=numeric_bins,
    ).fit(bundle.X.iloc[train_indices])
    metadata = {
        "dataset_id": bundle.dataset_id,
        "dataset_version": bundle.version,
        "target": bundle.target,
        "seed": seed,
        "split_id": split_id,
        "training_subset_id": training_subset_id,
        **transform.metadata(),
    }
    transform_id = stable_id(metadata)
    metadata["transform_id"] = transform_id
    write_json(
        root / "cache" / "transforms" / str(bundle.dataset_id) / str(seed) / f"{transform_id}.json",
        metadata,
    )
    return PreparedSplit(
        dataset=bundle,
        seed=seed,
        split_id=split_id,
        training_subset_id=training_subset_id,
        transform_id=transform_id,
        transform_metadata=metadata,
        cards=transform.discrete_cards,
        categorical_cards=transform.categorical_cards,
        categorical_tree_indices=[
            list(bundle.X.columns).index(column) for column in bundle.categorical_columns
        ],
        y_train=bundle.y[train_indices],
        y_validation=bundle.y[indices["validation"]],
        y_test=bundle.y[indices["test"]],
        train=transform.transform(bundle.X.iloc[train_indices]),
        validation=transform.transform(bundle.X.iloc[indices["validation"]]),
        test=transform.transform(bundle.X.iloc[indices["test"]]),
    )


# ---------------------------------------------------------------------------
# Classical BNC structures and discrete probability estimators
# ---------------------------------------------------------------------------


def pairwise_mi(X: np.ndarray) -> np.ndarray:
    count = X.shape[1]
    weights = np.zeros((count, count), dtype=np.float64)
    for left in range(count):
        for right in range(left):
            weight = mutual_info_score(X[:, left], X[:, right])
            weights[left, right] = weights[right, left] = float(weight)
    return weights


def conditional_mutual_information(left: np.ndarray, right: np.ndarray, y: np.ndarray) -> float:
    result = 0.0
    for cls in np.unique(y):
        mask = y == cls
        result += float(mask.mean()) * float(mutual_info_score(left[mask], right[mask]))
    return result


def conditional_mi_matrix(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    count = X.shape[1]
    weights = np.zeros((count, count), dtype=np.float64)
    for left in range(count):
        for right in range(left):
            value = conditional_mutual_information(X[:, left], X[:, right], y)
            weights[left, right] = weights[right, left] = value
    return weights


def maximum_spanning_tree(weights: np.ndarray, root: int = 0) -> list[tuple[int, int]]:
    """Return directed parent-child edges of a deterministic maximum tree."""
    n_nodes = weights.shape[0]
    if n_nodes < 2:
        return []
    selected = {root}
    edges: list[tuple[int, int]] = []
    while len(selected) < n_nodes:
        candidates = [
            (float(weights[parent, child]), -parent, -child, parent, child)
            for parent in selected
            for child in range(n_nodes)
            if child not in selected
        ]
        if not candidates:
            raise ExperimentError("Unable to construct a spanning tree")
        _, _, _, parent, child = max(candidates)
        selected.add(child)
        edges.append((parent, child))
    return edges


def tan_feature_parents(X: np.ndarray, y: np.ndarray) -> list[list[int]]:
    n_features = X.shape[1]
    if n_features == 0:
        return []
    relevance = [mutual_info_score(X[:, feature], y) for feature in range(n_features)]
    root = int(np.argmax(relevance))
    edges = maximum_spanning_tree(conditional_mi_matrix(X, y), root=root)
    parents = [[] for _ in range(n_features)]
    for parent, child in edges:
        parents[child] = [parent]
    return parents


def tan_adjacency(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    adjacency = np.zeros((X.shape[1], X.shape[1]), dtype=np.float32)
    for child, parents in enumerate(tan_feature_parents(X, y)):
        for parent in parents:
            adjacency[parent, child] = 1.0
    return adjacency


def ban_feature_parents(X: np.ndarray, y: np.ndarray, max_parents: int = 2) -> list[list[int]]:
    n_features = X.shape[1]
    relevance = np.asarray([mutual_info_score(X[:, feature], y) for feature in range(n_features)])
    order = list(np.argsort(-relevance, kind="stable"))
    before: list[int] = []
    parents = [[] for _ in range(n_features)]
    cmi = conditional_mi_matrix(X, y)
    for child in order:
        ranked = sorted(before, key=lambda parent: (-cmi[parent, child], parent))
        parents[child] = [
            parent for parent in ranked[:max_parents] if cmi[parent, child] > 0.0
        ]
        before.append(int(child))
    return parents


def kdb_feature_parents(X: np.ndarray, y: np.ndarray, k: int) -> list[list[int]]:
    n_features = X.shape[1]
    relevance = np.asarray([mutual_info_score(X[:, feature], y) for feature in range(n_features)])
    order = list(np.argsort(-relevance, kind="stable"))
    cmi = conditional_mi_matrix(X, y)
    parents = [[] for _ in range(n_features)]
    seen: list[int] = []
    for child in order:
        choices = sorted(seen, key=lambda parent: (-cmi[parent, child], parent))
        parents[child] = [parent for parent in choices[:k] if cmi[parent, child] > 0.0]
        seen.append(int(child))
    return parents




class ConditionalBNC:
    """Class-parent BNC with arbitrary acyclic feature parent sets."""

    def __init__(self, feature_parents: list[list[int]], alpha: float = 1.0) -> None:
        self.feature_parents = feature_parents
        self.alpha = float(alpha)
        self.cards: list[int] = []
        self.n_classes = 0
        self.class_counts: np.ndarray | None = None
        self.counts: list[dict[tuple[int, ...], np.ndarray]] = []

    def fit(self, X: np.ndarray, y: np.ndarray, cards: Sequence[int], n_classes: int) -> "ConditionalBNC":
        self.cards = list(map(int, cards))
        self.n_classes = int(n_classes)
        self.class_counts = np.bincount(y, minlength=n_classes).astype(np.float64)
        self.counts = [dict() for _ in self.cards]
        for row, cls in zip(X, y):
            for feature, parents in enumerate(self.feature_parents):
                key = (int(cls), *(int(row[parent]) for parent in parents))
                if key not in self.counts[feature]:
                    self.counts[feature][key] = np.zeros(self.cards[feature], dtype=np.float64)
                self.counts[feature][key][int(row[feature])] += 1.0
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.class_counts is not None
        probabilities = np.empty((X.shape[0], self.n_classes), dtype=np.float64)
        prior = np.log(
            (self.class_counts + self.alpha)
            / (self.class_counts.sum() + self.alpha * self.n_classes)
        )
        for row_index, row in enumerate(X):
            logp = prior.copy()
            for cls in range(self.n_classes):
                for feature, parents in enumerate(self.feature_parents):
                    key = (cls, *(int(row[parent]) for parent in parents))
                    values = self.counts[feature].get(
                        key, np.zeros(self.cards[feature], dtype=np.float64)
                    )
                    logp[cls] += math.log(
                        (values[int(row[feature])] + self.alpha)
                        / (values.sum() + self.alpha * self.cards[feature])
                    )
            logp -= logp.max()
            probabilities[row_index] = np.exp(logp) / np.exp(logp).sum()
        return probabilities


class ChowLiuClassifier:
    """Joint Chow-Liu tree rooted at Y, with exact posterior evaluation."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = float(alpha)
        self.n_classes = 0
        self.cards: list[int] = []
        self.edges: list[tuple[int, int]] = []
        self.class_counts: np.ndarray | None = None
        self.edge_counts: dict[tuple[int, int], dict[int, np.ndarray]] = {}

    def fit(self, X: np.ndarray, y: np.ndarray, cards: Sequence[int], n_classes: int) -> "ChowLiuClassifier":
        self.n_classes = int(n_classes)
        self.cards = [self.n_classes, *map(int, cards)]
        variables = np.column_stack([y, X])
        self.edges = maximum_spanning_tree(pairwise_mi(variables), root=0)
        self.class_counts = np.bincount(y, minlength=n_classes).astype(np.float64)
        self.edge_counts = {}
        for parent, child in self.edges:
            child_card = self.cards[child]
            count_table: dict[int, np.ndarray] = {}
            for row in variables:
                key = int(row[parent])
                count_table.setdefault(key, np.zeros(child_card, dtype=np.float64))
                count_table[key][int(row[child])] += 1.0
            self.edge_counts[(parent, child)] = count_table
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.class_counts is not None
        result = np.empty((X.shape[0], self.n_classes), dtype=np.float64)
        for index, features in enumerate(X):
            scores = np.empty(self.n_classes, dtype=np.float64)
            for cls in range(self.n_classes):
                values = np.concatenate(([cls], features.astype(int)))
                score = math.log(
                    (self.class_counts[cls] + self.alpha)
                    / (self.class_counts.sum() + self.alpha * self.n_classes)
                )
                for parent, child in self.edges:
                    table = self.edge_counts[(parent, child)].get(
                        int(values[parent]), np.zeros(self.cards[child], dtype=np.float64)
                    )
                    score += math.log(
                        (table[int(values[child])] + self.alpha)
                        / (table.sum() + self.alpha * self.cards[child])
                    )
                scores[cls] = score
            scores -= scores.max()
            result[index] = np.exp(scores) / np.exp(scores).sum()
        return result


# ---------------------------------------------------------------------------
# DBNC: reference model and validation variants
# ---------------------------------------------------------------------------






def dbnc_variant_parameters(
    variant: str,
    base: Mapping[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> dict[str, Any]:
    if variant not in DBNC_VARIANTS:
        raise ExperimentError(f"Unknown DBNC variant: {variant}")
    parameters = {
        key: value
        for key, value in base.items()
        if key in DBNC_MODEL_PARAMETER_KEYS and key != "device"
    }
    parameters.setdefault("cpd_type", "attention")
    if variant == "fixed_tan":
        parameters["fixed_adjacency"] = tan_adjacency(X_train, y_train)
    if variant == "conditional_only":
        parameters["hybrid_weight"] = 0.0
    if variant == "mlp_cpd":
        parameters["cpd_type"] = "mlp"
    if variant == "no_regularization":
        parameters["lambda_sparse"] = 0.0
        parameters["lambda_degree"] = 0.0
    if variant == "no_edges":
        parameters["fixed_adjacency"] = np.zeros((X_train.shape[1], X_train.shape[1]), dtype=np.float32)
    if variant == "no_sparse":
        parameters["lambda_sparse"] = 0.0
    if variant == "no_degree":
        parameters["lambda_degree"] = 0.0
    loss_values = {
        "loss_w_000": 0.0,
        "loss_w_025": 0.25,
        "loss_w_050": 0.50,
        "loss_w_075": 0.75,
        "loss_w_100": 1.0,
    }
    if variant in loss_values:
        parameters["hybrid_weight"] = loss_values[variant]
    return parameters


# ---------------------------------------------------------------------------
# Boosted-tree and neural tabular wrappers
# ---------------------------------------------------------------------------


def parameter_count(model: Any) -> int | None:
    if isinstance(model, nn.Module):
        return int(sum(parameter.numel() for parameter in model.parameters()))
    if hasattr(model, "network") and isinstance(model.network, nn.Module):
        return int(sum(parameter.numel() for parameter in model.network.parameters()))
    return None


def combined_neural(parts: MatrixParts) -> np.ndarray:
    return np.column_stack([parts.neural_numeric, parts.neural_categorical]).astype(np.float32)


def categorical_tree_frame(parts: MatrixParts, prepared: PreparedSplit) -> pd.DataFrame:
    columns = prepared.transform_metadata["columns"]
    frame = pd.DataFrame(parts.tree, columns=columns)
    levels = prepared.transform_metadata["categorical_levels"]
    for index in prepared.categorical_tree_indices:
        name = columns[index]
        frame[name] = pd.Categorical(
            frame[name].astype(int),
            categories=list(range(len(levels[name]))),
        )
    return frame


def lightgbm_tree_frame(parts: MatrixParts, prepared: PreparedSplit) -> pd.DataFrame:
    frame = categorical_tree_frame(parts, prepared)
    frame.columns = [f"feature_{index}" for index in range(frame.shape[1])]
    return frame


class FTTransformerNetwork(nn.Module):
    """Compact FT-Transformer implementation for the declared neural baseline."""

    def __init__(
        self,
        n_numeric: int,
        categorical_cards: Sequence[int],
        n_classes: int,
        *,
        d_token: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if d_token % n_heads != 0:
            raise ValueError("FT-Transformer d_token must be divisible by n_heads")
        self.n_numeric = int(n_numeric)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        self.numeric_weight = nn.Parameter(torch.empty(self.n_numeric, d_token))
        self.numeric_bias = nn.Parameter(torch.empty(self.n_numeric, d_token))
        self.cat_embeddings = nn.ModuleList(
            nn.Embedding(int(card), d_token) for card in categorical_cards
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=4 * d_token,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, n_classes)
        nn.init.normal_(self.cls_token, std=d_token ** -0.5)
        nn.init.normal_(self.numeric_weight, std=d_token ** -0.5)
        nn.init.zeros_(self.numeric_bias)

    def forward(self, numeric: torch.Tensor, categorical: torch.Tensor) -> torch.Tensor:
        tokens: list[torch.Tensor] = [
            self.cls_token.expand(numeric.shape[0], -1, -1)
        ]
        if self.n_numeric:
            tokens.append(
                numeric[:, :, None] * self.numeric_weight[None, :, :]
                + self.numeric_bias[None, :, :]
            )
        if self.cat_embeddings:
            tokens.append(
                torch.stack(
                    [
                        embedding(categorical[:, index])
                        for index, embedding in enumerate(self.cat_embeddings)
                    ],
                    dim=1,
                )
            )
        encoded = self.transformer(torch.cat(tokens, dim=1))
        return self.head(self.norm(encoded[:, 0]))


class FTTransformerClassifier:
    def __init__(
        self,
        n_numeric: int,
        categorical_cards: Sequence[int],
        n_classes: int,
        config: Mapping[str, Any],
        device: str,
    ) -> None:
        self.device = torch.device(device)
        self.batch_size = int(config.get("batch_size", 256))
        self.epochs = int(config.get("epochs", 50))
        self.patience = int(config.get("early_stopping_rounds", 10))
        self.lr = float(config.get("lr", 1e-3))
        self.n_numeric = int(n_numeric)
        self.network = FTTransformerNetwork(
            n_numeric,
            categorical_cards,
            n_classes,
            d_token=int(config.get("d_token", 32)),
            n_heads=int(config.get("n_heads", 4)),
            n_layers=int(config.get("n_layers", 2)),
            dropout=float(config.get("dropout", 0.1)),
        ).to(self.device)
        self.best_epoch = 0
        self.epochs_trained = 0

    def _tensor_parts(self, parts: MatrixParts) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.as_tensor(parts.neural_numeric, dtype=torch.float32),
            torch.as_tensor(parts.neural_categorical, dtype=torch.long),
        )

    def fit(
        self,
        train: MatrixParts,
        y_train: np.ndarray,
        validation: MatrixParts,
        y_validation: np.ndarray,
    ) -> "FTTransformerClassifier":
        numeric, categorical = self._tensor_parts(train)
        labels = torch.as_tensor(y_train, dtype=torch.long)
        loader = DataLoader(
            TensorDataset(numeric, categorical, labels),
            batch_size=self.batch_size,
            shuffle=True,
        )
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=self.lr)
        best_loss = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        stale = 0
        for epoch in range(1, self.epochs + 1):
            self.network.train()
            for num_batch, cat_batch, label_batch in loader:
                logits = self.network(
                    num_batch.to(self.device),
                    cat_batch.to(self.device),
                )
                loss = F.cross_entropy(logits, label_batch.to(self.device))
                if not torch.isfinite(loss):
                    raise ExperimentError("FT-Transformer encountered a non-finite training loss")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            self.epochs_trained = epoch
            probabilities = self.predict_proba(validation)
            validation_loss = float(
                log_loss(
                    y_validation,
                    probabilities,
                    labels=np.arange(probabilities.shape[1]),
                )
            )
            if validation_loss < best_loss - 1e-12:
                best_loss = validation_loss
                best_state = copy.deepcopy(self.network.state_dict())
                self.best_epoch = epoch
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break
        if best_state is not None:
            self.network.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def predict_proba(self, parts: MatrixParts) -> np.ndarray:
        numeric, categorical = self._tensor_parts(parts)
        loader = DataLoader(
            TensorDataset(numeric, categorical),
            batch_size=self.batch_size,
            shuffle=False,
        )
        training = self.network.training
        self.network.eval()
        output = []
        for num_batch, cat_batch in loader:
            logits = self.network(num_batch.to(self.device), cat_batch.to(self.device))
            output.append(logits.softmax(1).cpu().numpy())
        self.network.train(training)
        return np.concatenate(output, axis=0)


@dataclass
class FittedPredictor:
    predict_proba: Callable[[MatrixParts], np.ndarray]
    training_seconds: float
    parameter_count: int | None
    metadata: dict[str, Any]


def _time_fit(device: str, fit: Callable[[], None]) -> float:
    sync_device(device)
    start = time.perf_counter()
    fit()
    sync_device(device)
    return float(time.perf_counter() - start)


def fit_predictor(
    method: str,
    variant: str,
    config: Mapping[str, Any],
    prepared: PreparedSplit,
    device: str,
) -> FittedPredictor:
    """Fit a Phase 1 predictor; validation is never combined into training."""
    if method == "dbnc":
        parameters = dbnc_variant_parameters(
            variant,
            config,
            prepared.train.discrete,
            prepared.y_train,
        )
        n_restarts = int(config.get("n_restarts", 1))
        if n_restarts < 1:
            raise ExperimentError("DBNC n_restarts must be at least 1")
        model: DBNC | None = None
        best_restart = -1
        best_validation_accuracy = -float("inf")
        best_validation_log_loss = float("inf")
        elapsed = 0.0
        restart_records: list[dict[str, Any]] = []
        for restart in range(n_restarts):
            candidate = DBNC(
                prepared.cards,
                prepared.n_classes,
                device=device,
                **parameters,
            )
            restart_elapsed = _time_fit(
                device,
                lambda candidate=candidate: candidate.fit(
                    prepared.train.discrete,
                    prepared.y_train,
                    prepared.validation.discrete,
                    prepared.y_validation,
                ),
            )
            elapsed += restart_elapsed
            validation_probabilities = candidate.predict_proba(prepared.validation.discrete)
            validation_accuracy = float(
                accuracy_score(prepared.y_validation, validation_probabilities.argmax(axis=1))
            )
            validation_log_loss = float(
                log_loss(
                    prepared.y_validation,
                    validation_probabilities,
                    labels=np.arange(prepared.n_classes),
                )
            )
            restart_records.append(
                {
                    "restart": restart,
                    "validation_accuracy": validation_accuracy,
                    "validation_log_loss": validation_log_loss,
                    "best_epoch": candidate.best_epoch,
                    "epochs_trained": candidate.epochs_trained,
                    "training_seconds": restart_elapsed,
                    "graph_diagnostics": candidate.graph_diagnostics(),
                }
            )
            if validation_checkpoint_improved(
                validation_accuracy,
                validation_log_loss,
                best_validation_accuracy,
                best_validation_log_loss,
            ):
                model = candidate
                best_restart = restart
                best_validation_accuracy = validation_accuracy
                best_validation_log_loss = validation_log_loss
        assert model is not None
        details = {
            "representation": "discrete",
            "numeric_bins": int(prepared.transform_metadata["numeric_bins_requested"]),
            "device_assignment": str(torch.device(device)),
            "structure_type": model.structure_type,
            "cpd_type": model.cpd_type,
            "hybrid_weight": model.hybrid_weight,
            "lambda_sparse": model.lambda_sparse,
            "lambda_degree": model.lambda_degree,
            "max_parents": model.max_parents,
            "min_epochs_before_checkpoint": model.min_epochs_before_checkpoint,
            "min_epochs_before_stopping": model.min_epochs_before_stopping,
            "structure_warmup_epochs": model.structure_warmup_epochs,
            "edge_logit_mean": model.edge_logit_mean,
            "rank_init_std": model.rank_init_std,
            "n_restarts": n_restarts,
            "selected_restart": best_restart,
            "restart_selection_rule": "maximum_validation_accuracy_then_lower_log_loss_then_restart_number",
            "restart_records": restart_records,
            "fit_policy": "train_only_with_validation_early_stopping",
            "checkpoint_rule": "maximum_validation_accuracy_then_lower_log_loss",
            "best_validation_accuracy": model.best_validation_accuracy,
            "best_validation_log_loss": model.best_validation_log_loss,
            "best_epoch": model.best_epoch,
            "epochs_trained": model.epochs_trained,
            "graph_diagnostics": model.graph_diagnostics(),
        }
        return FittedPredictor(
            lambda parts: model.predict_proba(parts.discrete),
            elapsed,
            parameter_count(model),
            details,
        )

    if method in {"nb", "tan", "ban", "kdb_1", "kdb_2", "kdb_3"}:
        if method == "nb":
            parents = [[] for _ in prepared.cards]
            structure = "class_parent_only"
        elif method == "tan":
            parents = tan_feature_parents(prepared.train.discrete, prepared.y_train)
            structure = "training_conditional_mi_maximum_tree"
        elif method == "ban":
            parents = ban_feature_parents(prepared.train.discrete, prepared.y_train, max_parents=2)
            structure = "training_cmi_ordered_bounded_network_max_parents_2"
        else:
            k = int(method.rsplit("_", 1)[1])
            parents = kdb_feature_parents(prepared.train.discrete, prepared.y_train, k=k)
            structure = f"training_mi_ordered_kdb_{k}"
        model = ConditionalBNC(parents, alpha=float(config["alpha"]))
        elapsed = _time_fit(
            "cpu",
            lambda: model.fit(
                prepared.train.discrete,
                prepared.y_train,
                prepared.cards,
                prepared.n_classes,
            ),
        )
        adjacency = np.zeros((len(prepared.cards), len(prepared.cards)), dtype=int)
        for child, values in enumerate(parents):
            for parent in values:
                adjacency[parent, child] = 1
        return FittedPredictor(
            lambda parts: model.predict_proba(parts.discrete),
            elapsed,
            None,
            {
                "representation": "discrete",
                "device_assignment": "cpu",
                "structure_rule": structure,
                "structure_acyclic": is_acyclic(adjacency),
                "fit_policy": "training_partition_only",
            },
        )

    if method == "chow_liu":
        model = ChowLiuClassifier(alpha=float(config["alpha"]))
        elapsed = _time_fit(
            "cpu",
            lambda: model.fit(
                prepared.train.discrete,
                prepared.y_train,
                prepared.cards,
                prepared.n_classes,
            ),
        )
        return FittedPredictor(
            lambda parts: model.predict_proba(parts.discrete),
            elapsed,
            None,
            {
                "representation": "discrete",
                "device_assignment": "cpu",
                "structure_rule": "maximum_mi_tree_over_y_and_features_rooted_at_y",
                "fit_policy": "training_partition_only",
            },
        )

    if method == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise MissingDependency("XGBoost execution requires the 'xgboost' package") from exc
        multiclass_parameters = (
            {} if prepared.n_classes == 2 else {"num_class": prepared.n_classes}
        )
        model = XGBClassifier(
            n_estimators=int(config["n_estimators"]),
            max_depth=int(config["max_depth"]),
            learning_rate=float(config["learning_rate"]),
            subsample=float(config["subsample"]),
            colsample_bytree=float(config["colsample_bytree"]),
            objective="binary:logistic" if prepared.n_classes == 2 else "multi:softprob",
            eval_metric="logloss" if prepared.n_classes == 2 else "mlogloss",
            random_state=prepared.seed,
            n_jobs=1,
            tree_method="hist",
            enable_categorical=True,
            **multiclass_parameters,
        )
        elapsed = _time_fit(
            "cpu",
            lambda: model.fit(categorical_tree_frame(prepared.train, prepared), prepared.y_train),
        )
        return FittedPredictor(
            lambda parts: model.predict_proba(categorical_tree_frame(parts, prepared)),
            elapsed,
            None,
            {
                "representation": "undiscretized_numeric_with_native_categorical_columns",
                "device_assignment": "cpu",
                "fit_policy": "training_partition_only",
            },
        )

    if method == "catboost":
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            raise MissingDependency("CatBoost execution requires the 'catboost' package") from exc

        def cat_frame(parts: MatrixParts) -> pd.DataFrame:
            frame = pd.DataFrame(parts.tree)
            for index in prepared.categorical_tree_indices:
                frame[index] = frame[index].astype(int).astype(str)
            return frame

        model = CatBoostClassifier(
            iterations=int(config["iterations"]),
            depth=int(config["depth"]),
            learning_rate=float(config["learning_rate"]),
            l2_leaf_reg=float(config["l2_leaf_reg"]),
            loss_function="Logloss" if prepared.n_classes == 2 else "MultiClass",
            random_seed=prepared.seed,
            thread_count=1,
            verbose=False,
            allow_writing_files=False,
        )
        elapsed = _time_fit(
            "cpu",
            lambda: model.fit(
                cat_frame(prepared.train),
                prepared.y_train,
                cat_features=prepared.categorical_tree_indices,
                verbose=False,
            ),
        )
        return FittedPredictor(
            lambda parts: model.predict_proba(cat_frame(parts)),
            elapsed,
            None,
            {
                "representation": "undiscretized_numeric_with_native_categorical_columns",
                "device_assignment": "cpu",
                "fit_policy": "training_partition_only",
            },
        )

    if method == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise MissingDependency("LightGBM execution requires the 'lightgbm' package") from exc
        categorical_names = [
            f"feature_{index}"
            for index in prepared.categorical_tree_indices
        ]
        model = LGBMClassifier(
            n_estimators=int(config["n_estimators"]),
            num_leaves=int(config["num_leaves"]),
            learning_rate=float(config["learning_rate"]),
            subsample=float(config["subsample"]),
            colsample_bytree=float(config["colsample_bytree"]),
            objective="binary" if prepared.n_classes == 2 else "multiclass",
            **({} if prepared.n_classes == 2 else {"num_class": prepared.n_classes}),
            random_state=prepared.seed,
            n_jobs=1,
            verbosity=-1,
            subsample_freq=1,
        )
        elapsed = _time_fit(
            "cpu",
            lambda: model.fit(
                lightgbm_tree_frame(prepared.train, prepared),
                prepared.y_train,
                categorical_feature=categorical_names,
            ),
        )
        return FittedPredictor(
            lambda parts: model.predict_proba(lightgbm_tree_frame(parts, prepared)),
            elapsed,
            None,
            {
                "representation": "undiscretized_numeric_with_native_categorical_columns",
                "device_assignment": "cpu",
                "fit_policy": "training_partition_only",
                "feature_name_policy": "adapter_safe_positional_labels",
            },
        )

    if method == "ft_transformer":
        model = FTTransformerClassifier(
            prepared.train.neural_numeric.shape[1],
            prepared.categorical_cards,
            prepared.n_classes,
            config,
            device,
        )
        elapsed = _time_fit(
            device,
            lambda: model.fit(
                prepared.train,
                prepared.y_train,
                prepared.validation,
                prepared.y_validation,
            ),
        )
        return FittedPredictor(
            model.predict_proba,
            elapsed,
            parameter_count(model.network),
            {
                "representation": "standardized_numeric_with_categorical_embeddings",
                "device_assignment": str(torch.device(device)),
                "implementation": "in_file_ft_transformer",
                "fit_policy": "train_only_with_validation_early_stopping",
                "best_epoch": model.best_epoch,
                "epochs_trained": model.epochs_trained,
            },
        )

    raise ExperimentError(f"Unsupported Phase 1 method: {method}")


# ---------------------------------------------------------------------------
# Hyperparameter selection and final Phase 1 fitting
# ---------------------------------------------------------------------------


def method_variant(method: str) -> str:
    return "full" if method == "dbnc" else "baseline"


def suggest_configuration(method: str, trial: Any, *, quick: bool = False) -> dict[str, Any]:
    if method == "dbnc":
        d = trial.suggest_categorical("d", [16] if quick else [16, 32, 64])
        heads = trial.suggest_categorical("n_heads", [head for head in (2, 4, 8) if d % head == 0])
        return {
            "numeric_bins": trial.suggest_categorical(
                "numeric_bins", [N_BINS] if quick else [N_BINS, 16, 32]
            ),
            "d": d,
            "n_heads": heads,
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "dropout": trial.suggest_categorical("dropout", [0.0, 0.1, 0.2]),
            "cpd_type": "attention",
            "hybrid_weight": trial.suggest_categorical(
                "hybrid_weight",
                [0.0] if quick else [0.0, 0.01, 0.025, 0.05, 0.1, 0.2],
            ),
            "lambda_sparse": trial.suggest_float("lambda_sparse", 1e-6, 3e-3, log=True),
            "lambda_degree": trial.suggest_float("lambda_degree", 1e-5, 3e-3, log=True),
            "max_parents": trial.suggest_categorical("max_parents", [3] if quick else [3, 5, 8]),
            "epochs": trial.suggest_categorical("epochs", [2] if quick else [50, 100, 200]),
            "batch_size": 64 if quick else 1024,
            "early_stopping_rounds": 10,
            "min_epochs_before_checkpoint": 1 if quick else 15,
            "min_epochs_before_stopping": 1 if quick else 25,
            "structure_warmup_epochs": 0 if quick else trial.suggest_categorical("structure_warmup_epochs", [5, 10, 20]),
            "edge_logit_mean": trial.suggest_categorical(
                "edge_logit_mean", [-1.0] if quick else [-1.0, -0.5, 0.0]
            ),
            "rank_init_std": trial.suggest_categorical(
                "rank_init_std", [0.5] if quick else [0.1, 0.5, 1.0]
            ),
            "n_restarts": 1,
        }
    if method in {"nb", "chow_liu", "tan", "ban", "kdb_1", "kdb_2", "kdb_3"}:
        return {"alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True)}
    if method == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 5, 5) if quick else trial.suggest_int("n_estimators", 100, 600, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 3) if quick else trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.30, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
    if method == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 5, 5) if quick else trial.suggest_int("iterations", 100, 600, step=100),
            "depth": trial.suggest_int("depth", 4, 4) if quick else trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.30, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        }
    if method == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 5, 5) if quick else trial.suggest_int("n_estimators", 100, 600, step=100),
            "num_leaves": trial.suggest_categorical("num_leaves", [15] if quick else [15, 31, 63, 127]),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.30, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
    if method == "ft_transformer":
        d_token = trial.suggest_categorical("d_token", [16] if quick else [16, 32, 64])
        return {
            "d_token": d_token,
            "n_heads": trial.suggest_categorical(
                "n_heads", [head for head in (2, 4, 8) if d_token % head == 0]
            ),
            "n_layers": trial.suggest_categorical("n_layers", [1] if quick else [1, 2, 3]),
            "dropout": trial.suggest_categorical("dropout", [0.0, 0.1, 0.2]),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "epochs": trial.suggest_categorical("epochs", [2] if quick else [50, 100, 200]),
            "batch_size": 64 if quick else 256,
            "early_stopping_rounds": 10,
        }
    raise ExperimentError(f"No search space declared for method: {method}")


def tuning_search_space_version(method: str) -> str | None:
    return DBNC_SEARCH_SPACE_VERSION if method == "dbnc" else None


def tuning_path(root: Path, method: str, variant: str, dataset_id: int) -> Path:
    version = tuning_search_space_version(method)
    if version is not None:
        return root / "results" / "tuning" / method / variant / version / f"{dataset_id}.json"
    return root / "results" / "tuning" / method / variant / f"{dataset_id}.json"


def result_hp_id(method: str, configuration: Mapping[str, Any]) -> str:
    version = tuning_search_space_version(method)
    if version is None:
        return stable_id(configuration)
    return stable_id({"configuration": dict(configuration), "search_space_version": version})


def tune_method(
    root: Path,
    method: str,
    variant: str,
    bundle: DatasetBundle,
    *,
    trials: int,
    device: str,
    execution_scope: str,
    quick: bool = False,
) -> dict[str, Any]:
    path = tuning_path(root, method, variant, bundle.dataset_id)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("requested_trial_budget") != trials:
            raise ExperimentError(
                f"Existing immutable tuning artifact {path} used a different trial budget"
            )
        if existing.get("execution_scope") != execution_scope:
            raise ExperimentError(
                f"Existing tuning artifact {path} belongs to another execution scope"
            )
        if existing.get("quick_budget", False) != quick:
            raise ExperimentError(f"Existing tuning artifact {path} used another budget profile")
        expected_version = tuning_search_space_version(method)
        if expected_version is not None and existing.get("search_space_version") != expected_version:
            raise ExperimentError(
                f"Existing tuning artifact {path} belongs to another DBNC search-space version"
            )
        return existing
    try:
        import optuna
    except ImportError as exc:
        raise MissingDependency(
            "Phase 1 hyperparameter selection requires Optuna TPE; install 'optuna'"
        ) from exc
    if trials < 1 or trials > 20:
        raise ExperimentError("The confirmatory tuning budget must be between 1 and 20 trials")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    prepared_cache: dict[int, PreparedSplit] = {}
    default_prepared: PreparedSplit | None = None
    if method != "dbnc":
        default_prepared = prepare_split(root, bundle, 0, tuning_training_cap=50_000)

    def prepared_for_configuration(configuration: Mapping[str, Any]) -> PreparedSplit:
        if method != "dbnc":
            assert default_prepared is not None
            return default_prepared
        numeric_bins = int(configuration.get("numeric_bins", N_BINS))
        if numeric_bins not in prepared_cache:
            prepared_cache[numeric_bins] = prepare_split(
                root,
                bundle,
                0,
                numeric_bins=numeric_bins,
                tuning_training_cap=50_000,
            )
        return prepared_cache[numeric_bins]

    sampler = optuna.samplers.TPESampler(seed=0)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: Any) -> float:
        set_seed(10_000 + trial.number)
        configuration = suggest_configuration(method, trial, quick=quick)
        trial.set_user_attr("configuration", jsonable(configuration))
        try:
            prepared = prepared_for_configuration(configuration)
            predictor = fit_predictor(method, variant, configuration, prepared, device)
            validation = predictive_metrics(
                prepared.y_validation,
                predictor.predict_proba(prepared.validation),
            )
            trial.set_user_attr("validation_metrics", jsonable(validation))
            trial.set_user_attr("training_seconds", predictor.training_seconds)
            trial.set_user_attr("split_id", prepared.split_id)
            trial.set_user_attr("transform_id", prepared.transform_id)
            trial.set_user_attr("training_subset_id", prepared.training_subset_id)
            return float(validation["accuracy"])
        except Exception as exc:
            trial.set_user_attr("failure", f"{type(exc).__name__}: {exc}")
            raise

    study.optimize(objective, n_trials=trials, catch=(Exception,))
    completed = [
        trial
        for trial in study.trials
        if trial.state.name == "COMPLETE" and "validation_metrics" in trial.user_attrs
    ]
    trial_records = [
        {
            "number": trial.number,
            "state": trial.state.name,
            "configuration": trial.user_attrs.get("configuration"),
            "validation_metrics": trial.user_attrs.get("validation_metrics"),
            "training_seconds": trial.user_attrs.get("training_seconds"),
            "failure": trial.user_attrs.get("failure"),
        }
        for trial in study.trials
    ]
    if not completed:
        raise ExperimentError(f"No tuning trial completed for {method} on dataset {bundle.dataset_id}")
    if len(completed) != trials:
        raise ExperimentError(
            f"Only {len(completed)} of {trials} required tuning trials completed for "
            f"{method} on dataset {bundle.dataset_id}"
        )
    selected = min(
        completed,
        key=lambda trial: (
            -float(trial.user_attrs["validation_metrics"]["accuracy"]),
            float(trial.user_attrs["validation_metrics"]["log_loss"]),
            trial.number,
        ),
    )
    result = {
        "phase": "phase_1",
        "execution_scope": execution_scope,
        "quick_budget": quick,
        "kind": "hyperparameter_selection",
        "method": method,
        "method_label": METHOD_LABELS[method],
        "variant": variant,
        "dataset_id": bundle.dataset_id,
        "dataset_version": bundle.version,
        "target": bundle.target,
        "split_seed": 0,
        "split_id": selected.user_attrs["split_id"],
        "transform_id": selected.user_attrs["transform_id"],
        "training_subset_id": selected.user_attrs["training_subset_id"],
        "optimizer": "optuna_tpe",
        "search_space_version": tuning_search_space_version(method),
        "selection_rule": "maximum_validation_accuracy_then_lower_log_loss_then_trial_number",
        "requested_trial_budget": trials,
        "completed_trials": len(completed),
        "selected_trial": selected.number,
        "selected_configuration": selected.user_attrs["configuration"],
        "selected_validation_metrics": selected.user_attrs["validation_metrics"],
        "trials": trial_records,
        "environment": environment_metadata(device),
    }
    write_json(path, result, immutable=True)
    return result


def slice_parts(parts: MatrixParts, count: int) -> MatrixParts:
    return MatrixParts(
        discrete=parts.discrete[:count],
        tree=parts.tree[:count],
        neural_numeric=parts.neural_numeric[:count],
        neural_categorical=parts.neural_categorical[:count],
    )


def final_evaluation(
    root: Path,
    method: str,
    variant: str,
    bundle: DatasetBundle,
    seed: int,
    tuning: Mapping[str, Any],
    *,
    device: str,
    admissible: bool,
    execution_scope: str,
) -> dict[str, Any]:
    configuration = dict(tuning["selected_configuration"])
    hp_id = result_hp_id(method, configuration)
    path = result_path(root, method, variant, bundle.dataset_id, seed, "main", hp_id)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("admissible") != admissible or existing.get("execution_scope") != execution_scope:
            raise ExperimentError(f"Stored result belongs to another execution scope: {path}")
        return existing
    set_seed(seed)
    numeric_bins = int(configuration.get("numeric_bins", N_BINS)) if method == "dbnc" else N_BINS
    prepared = prepare_split(root, bundle, seed, numeric_bins=numeric_bins)
    fitted = fit_predictor(method, variant, configuration, prepared, device)
    probabilities = fitted.predict_proba(prepared.test)
    metrics = predictive_metrics(prepared.y_test, probabilities)
    timing_count = min(10_000, prepared.y_test.shape[0])
    timing_parts = slice_parts(prepared.test, timing_count)
    sync_device(device)
    start = time.perf_counter()
    fitted.predict_proba(timing_parts)
    sync_device(device)
    inference_seconds = float(time.perf_counter() - start)
    record = {
        "status": "completed",
        "phase": "phase_1",
        "admissible": admissible,
        "execution_scope": execution_scope,
        "dataset_id": bundle.dataset_id,
        "dataset_alias": bundle.alias,
        "dataset_version": bundle.version,
        "target": bundle.target,
        "split_seed": seed,
        "split_id": prepared.split_id,
        "method": method,
        "method_label": METHOD_LABELS[method],
        "variant": variant,
        "mode": "main",
        "hp_id": hp_id,
        "search_space_version": tuning_search_space_version(method),
        "selected_configuration": configuration,
        "tuning_provenance": {
            "artifact": str(tuning_path(root, method, variant, bundle.dataset_id)),
            "split_seed": tuning["split_seed"],
            "selected_trial": tuning["selected_trial"],
            "selection_rule": tuning["selection_rule"],
            "search_space_version": tuning.get("search_space_version"),
        },
        "feature_representation": fitted.metadata["representation"],
        "transform_id": prepared.transform_id,
        "metrics": metrics,
        "timing": {
            "training_seconds": fitted.training_seconds,
            "inference_seconds": inference_seconds,
            "inference_rows": timing_count,
            "excludes_preprocessing_and_download": True,
        },
        "environment": environment_metadata(device),
        "parameter_count": fitted.parameter_count,
        "model": fitted.metadata,
    }
    write_json(path, record, immutable=True)
    return record


def accepted_manifest_records(root: Path) -> dict[int, dict[str, Any]]:
    path = root / "artifacts" / "validated_manifest.json"
    if not path.exists():
        raise ExperimentError("Validated manifest is missing; run the 'phase0' command first")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return {
        int(record["dataset_id"]): record
        for record in manifest["records"]
        if record["status"] == "accepted"
    }


def require_phase0_gate(root: Path) -> None:
    path = root / "artifacts" / "phase0_acceptance.json"
    if not path.exists():
        raise ExperimentError("Phase 0 acceptance record is missing; run 'phase0' before 'benchmark'")
    record = json.loads(path.read_text(encoding="utf-8"))
    if not record.get("accepted"):
        raise ExperimentError("Phase 0 has not passed; final benchmark runs are not admissible")


def run_benchmark(
    root: Path,
    selected: list[tuple[str, int]],
    methods: Sequence[str],
    seeds: Sequence[int],
    *,
    trials: int,
    device: str,
    skip_phase0_gate: bool,
    manifest_root: Path | None = None,
    admissible: bool = True,
    execution_scope: str = "confirmatory",
    quick: bool = False,
) -> list[dict[str, Any]]:
    manifest_root = root if manifest_root is None else manifest_root
    if not skip_phase0_gate:
        require_phase0_gate(manifest_root)
    manifest = accepted_manifest_records(manifest_root)
    configure_openml_cache(manifest_root)
    records: list[dict[str, Any]] = []
    for alias, dataset_id in selected:
        if dataset_id not in manifest:
            print(f"[benchmark] skipping {alias}: not accepted in manifest", flush=True)
            continue
        bundle = load_openml_dataset(alias, dataset_id)
        accepted = manifest[dataset_id]
        if bundle.version != accepted.get("version") or bundle.target != accepted.get("target"):
            raise ExperimentError(f"Retrieved dataset identity changed since validation: {dataset_id}")
        for method in methods:
            variant = method_variant(method)
            try:
                print(f"[tuning] {alias}: {METHOD_LABELS[method]}", flush=True)
                tuning = tune_method(
                    root,
                    method,
                    variant,
                    bundle,
                    trials=trials,
                    device=device,
                    execution_scope=execution_scope,
                    quick=quick,
                )
                for seed in seeds:
                    try:
                        print(f"[final] {alias}: {METHOD_LABELS[method]} seed={seed}", flush=True)
                        records.append(
                            final_evaluation(
                                root,
                                method,
                                variant,
                                bundle,
                                int(seed),
                                tuning,
                                device=device,
                                admissible=admissible,
                                execution_scope=execution_scope,
                            )
                        )
                    except Exception as exc:
                        failure = {
                            "phase": "phase_1",
                            "dataset_id": dataset_id,
                            "dataset_alias": alias,
                            "method": method,
                            "variant": variant,
                            "seed": int(seed),
                            "stage": "final_evaluation",
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(),
                        }
                        append_failure(root, failure)
                        print(
                            f"[failed] {alias}: {METHOD_LABELS[method]} seed={seed}: {exc}",
                            file=sys.stderr,
                            flush=True,
                        )
            except Exception as exc:
                failure = {
                    "phase": "phase_1",
                    "dataset_id": dataset_id,
                    "dataset_alias": alias,
                    "method": method,
                    "variant": variant,
                    "stage": "tuning",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
                append_failure(root, failure)
                print(f"[failed] {alias}: {METHOD_LABELS[method]}: {exc}", file=sys.stderr, flush=True)
    return records


# ---------------------------------------------------------------------------
# Aggregation, statistical outputs, and plots
# ---------------------------------------------------------------------------


def read_completed_results(root: Path, *, admissible_only: bool = True) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    results_root = root / "results"
    if not results_root.exists():
        return records
    for path in results_root.rglob("*.json"):
        if "tuning" in path.parts:
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if (
            record.get("status") == "completed"
            and "metrics" in record
            and (not admissible_only or record.get("admissible") is True)
        ):
            records.append(record)
    return records


def read_failure_records(root: Path) -> list[dict[str, Any]]:
    path = root / "results" / "execution_failures.jsonl"
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def records_frame(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        row = {
            "phase": record.get("phase"),
            "dataset_id": record.get("dataset_id"),
            "dataset_alias": record.get("dataset_alias"),
            "seed": record.get("split_seed"),
            "method": record.get("method"),
            "method_label": record.get("method_label", record.get("method")),
            "variant": record.get("variant"),
            "mode": record.get("mode"),
            "hp_id": record.get("hp_id"),
            "search_space_version": record.get("search_space_version"),
            "admissible": record.get("admissible", False),
            "execution_scope": record.get("execution_scope"),
            "training_seconds": record.get("timing", {}).get("training_seconds"),
            "inference_seconds": record.get("timing", {}).get("inference_seconds"),
        }
        row.update(record.get("metrics", {}))
        rows.append(row)
    return pd.DataFrame(rows)


def prefer_current_main_records(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "method" not in frame.columns:
        return frame
    main_mask = (frame["phase"] == "phase_1") & (frame["mode"] == "main")
    main = frame[main_mask].copy()
    other = frame[~main_mask]
    if main.empty:
        return frame
    main["_current_priority"] = np.where(
        (main["method"] == "dbnc") & (main["search_space_version"] == DBNC_SEARCH_SPACE_VERSION),
        2,
        np.where(main["method"] == "dbnc", 1, 2),
    )
    key_columns = [
        "phase",
        "dataset_id",
        "seed",
        "method",
        "variant",
        "mode",
        "admissible",
        "execution_scope",
    ]
    main = (
        main.sort_values("_current_priority", kind="mergesort")
        .drop_duplicates(subset=key_columns, keep="last")
        .drop(columns=["_current_priority"])
    )
    return pd.concat([other, main], ignore_index=True)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    ensure_parent(path)
    frame.to_csv(path, index=False)


def _plot_pareto(frame: pd.DataFrame, x: str, y: str, path: Path) -> None:
    if frame.empty or frame[x].dropna().empty or frame[y].dropna().empty:
        return
    matplotlib_cache = path.parent / ".matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 5))
    for method, group in frame.groupby("method_label"):
        axis.scatter(group[x], group[y], label=method)
    axis.set_xlabel(x.replace("_", " ").title())
    axis.set_ylabel(y.replace("_", " ").title())
    if x.endswith("seconds"):
        axis.set_xscale("log")
    axis.legend(fontsize=7, loc="best")
    figure.tight_layout()
    ensure_parent(path)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_loss_weights(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        return
    matplotlib_cache = path.parent / ".matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    weight_map = {
        "loss_w_000": 0.0,
        "loss_w_025": 0.25,
        "loss_w_050": 0.50,
        "loss_w_075": 0.75,
        "loss_w_100": 1.0,
    }
    values = frame.copy()
    values["loss_weight"] = values["variant"].map(weight_map)
    values = values.dropna(subset=["loss_weight"])
    if values.empty:
        return
    means = values.groupby("loss_weight", as_index=False)[["accuracy", "log_loss", "ece"]].mean()
    figure, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    for axis, metric in zip(axes, ("accuracy", "log_loss", "ece")):
        axis.plot(means["loss_weight"], means[metric], marker="o")
        axis.set_xlabel("Hybrid weight")
        axis.set_ylabel(metric.replace("_", " ").title())
    figure.tight_layout()
    ensure_parent(path)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_critical_difference(
    mean_ranks: pd.Series,
    critical_difference: float,
    path: Path,
    metric: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = mean_ranks.sort_values()
    figure_height = max(2.2, 0.35 * len(ordered) + 1.2)
    figure, axis = plt.subplots(figsize=(8, figure_height))
    for index, (method, rank) in enumerate(ordered.items()):
        axis.plot([rank], [index], marker="o", color="black")
        axis.text(rank + 0.03, index, method, va="center", fontsize=8)
    upper = len(ordered) + 0.05
    start = float(ordered.min())
    axis.plot([start, start + critical_difference], [upper, upper], color="black", linewidth=2)
    axis.plot([start, start], [upper - 0.1, upper + 0.1], color="black")
    axis.plot([start + critical_difference, start + critical_difference], [upper - 0.1, upper + 0.1], color="black")
    axis.text(start + critical_difference / 2, upper + 0.13, f"CD={critical_difference:.3f}", ha="center", fontsize=8)
    axis.set_xlabel("Mean rank (lower is better)")
    axis.set_yticks([])
    axis.set_ylim(-0.7, upper + 0.45)
    axis.set_title(f"Critical-difference input: {metric}")
    figure.tight_layout()
    ensure_parent(path)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def nemenyi_records(mean_ranks: pd.Series, n_datasets: int) -> list[dict[str, Any]]:
    methods = mean_ranks.index.tolist()
    k = len(methods)
    standard_error = math.sqrt(k * (k + 1) / (6.0 * n_datasets))
    comparisons: list[dict[str, Any]] = []
    for index, left in enumerate(methods):
        for right in methods[index + 1 :]:
            difference = abs(float(mean_ranks[left]) - float(mean_ranks[right]))
            statistic = difference / standard_error * math.sqrt(2.0)
            p_value = float(stats.studentized_range.sf(statistic, k, np.inf))
            comparisons.append(
                {
                    "left": left,
                    "right": right,
                    "rank_difference": difference,
                    "p_value": p_value,
                }
            )
    return comparisons


def generate_reports(
    records: Sequence[Mapping[str, Any]],
    output: Path,
    *,
    expected_methods: Sequence[str] = MAIN_METHODS,
    expected_seeds: Sequence[int] = SEEDS,
    accepted_dataset_ids: Sequence[int] | None = None,
    failures: Sequence[Mapping[str, Any]] = (),
    exclusions: Sequence[Mapping[str, Any]] = (),
    enforce_admissible: bool = True,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    frame = prefer_current_main_records(records_frame(records))
    if frame.empty:
        raise ExperimentError("No completed result records are available for reporting")
    _write_csv(frame, output / "per_seed_results.csv")
    main = frame[(frame["phase"] == "phase_1") & (frame["mode"] == "main")]
    if enforce_admissible:
        main = main[main["admissible"] == True]  # noqa: E712 - pandas element-wise comparison
    expected_methods = tuple(expected_methods)
    expected_seeds = tuple(sorted(set(int(seed) for seed in expected_seeds)))
    dataset_ids = (
        list(dict.fromkeys(int(dataset_id) for dataset_id in accepted_dataset_ids))
        if accepted_dataset_ids is not None
        else sorted(int(dataset_id) for dataset_id in main["dataset_id"].dropna().unique())
    )
    pair_rows: list[dict[str, Any]] = []
    complete_pairs: set[tuple[int, str]] = set()
    for dataset_id in dataset_ids:
        for method in expected_methods:
            group = main[(main["dataset_id"] == dataset_id) & (main["method"] == method)]
            observed = sorted(int(seed) for seed in group["seed"].dropna().unique())
            has_duplicate_seed = group.shape[0] != len(observed)
            complete = observed == list(expected_seeds) and not has_duplicate_seed
            if complete:
                complete_pairs.add((dataset_id, method))
            pair_rows.append(
                {
                    "dataset_id": dataset_id,
                    "method": method,
                    "method_label": METHOD_LABELS.get(method, method),
                    "expected_seeds": list(expected_seeds),
                    "observed_seeds": observed,
                    "duplicate_seed_record": has_duplicate_seed,
                    "complete": complete,
                }
            )
    pair_status = pd.DataFrame(pair_rows)
    _write_csv(pair_status, output / "main_completion_status.csv")
    valid_dataset_ids = [
        dataset_id
        for dataset_id in dataset_ids
        if all((dataset_id, method) in complete_pairs for method in expected_methods)
    ]
    valid_main = main[
        main["dataset_id"].isin(valid_dataset_ids) & main["method"].isin(expected_methods)
    ]
    _write_csv(valid_main, output / "main_valid_per_seed_results.csv")
    metrics = [
        "accuracy",
        "macro_f1",
        "auc",
        "log_loss",
        "brier",
        "ece",
        "training_seconds",
        "inference_seconds",
    ]
    present_metrics = [metric for metric in metrics if metric in valid_main.columns]
    means = (
        valid_main.groupby(["dataset_id", "dataset_alias", "method", "method_label"], dropna=False)[present_metrics]
        .mean(numeric_only=True)
        .reset_index()
        if not valid_main.empty
        else pd.DataFrame()
    )
    _write_csv(means, output / "main_dataset_means.csv")
    rank_frames: list[pd.DataFrame] = []
    tests: dict[str, Any] = {}
    higher_is_better = {"accuracy", "macro_f1", "auc"}
    ranking_metrics = (
        "accuracy",
        "macro_f1",
        "auc",
        "log_loss",
        "brier",
        "ece",
        "training_seconds",
        "inference_seconds",
    )
    expected_labels = [METHOD_LABELS.get(method, method) for method in expected_methods]
    for metric in ranking_metrics:
        if means.empty or metric not in means.columns:
            continue
        pivot = (
            means.pivot(index="dataset_id", columns="method_label", values=metric)
            .reindex(columns=expected_labels)
            .dropna()
        )
        if pivot.empty:
            continue
        ascending = metric not in higher_is_better
        metric_ranks = pivot.rank(axis=1, ascending=ascending, method="average")
        long_ranks = metric_ranks.reset_index().melt(
            id_vars="dataset_id",
            var_name="method_label",
            value_name="rank",
        )
        long_ranks["metric"] = metric
        rank_frames.append(long_ranks)
        if metric in {"accuracy", "log_loss", "ece"}:
            record: dict[str, Any] = {
                "datasets_complete": int(pivot.shape[0]),
                "methods_complete": int(pivot.shape[1]),
                "mean_ranks": metric_ranks.mean(axis=0).to_dict(),
            }
            if pivot.shape[0] >= 2 and pivot.shape[1] >= 3:
                statistic, p_value = stats.friedmanchisquare(
                    *(pivot[column].to_numpy() for column in pivot.columns)
                )
                record["friedman"] = {"statistic": float(statistic), "p_value": float(p_value)}
                if p_value < 0.05:
                    k = pivot.shape[1]
                    q_value = float(stats.studentized_range.ppf(0.95, k, np.inf) / math.sqrt(2.0))
                    critical_difference = q_value * math.sqrt(k * (k + 1) / (6.0 * pivot.shape[0]))
                    record["nemenyi"] = nemenyi_records(metric_ranks.mean(axis=0), pivot.shape[0])
                    record["critical_difference"] = critical_difference
                    _plot_critical_difference(
                        metric_ranks.mean(axis=0),
                        critical_difference,
                        output / f"critical_difference_{metric}.png",
                        metric,
                    )
            else:
                record["friedman"] = {"status": "insufficient_complete_methods_or_datasets"}
            tests[metric] = record
    ranks = pd.concat(rank_frames, ignore_index=True) if rank_frames else pd.DataFrame()
    _write_csv(ranks, output / "main_ranking_inputs.csv")
    write_json(output / "main_statistical_tests.json", tests)
    if not means.empty:
        _plot_pareto(means, "training_seconds", "accuracy", output / "accuracy_vs_training_time.png")
        _plot_pareto(means, "ece", "accuracy", output / "accuracy_vs_ece.png")

    ablation_nonfull = frame[
        (frame["method"] == "dbnc") & (frame["variant"].isin(CORE_ABLATIONS[1:]))
    ]
    full_candidates = frame[(frame["method"] == "dbnc") & (frame["variant"] == "full")]
    phase2_full = full_candidates[full_candidates["phase"] == "phase_2"]
    reused_main_full = full_candidates[
        (full_candidates["phase"] == "phase_1") & (full_candidates["mode"] == "main")
    ]
    if ablation_nonfull.empty:
        ablation = ablation_nonfull
    else:
        ablation = pd.concat(
            [phase2_full if not phase2_full.empty else reused_main_full, ablation_nonfull],
            ignore_index=True,
        )
    ablation_means = (
        ablation.groupby(["dataset_id", "dataset_alias", "variant"], dropna=False)[
            [metric for metric in ("accuracy", "log_loss", "ece", "macro_f1", "auc", "brier") if metric in ablation]
        ]
        .mean(numeric_only=True)
        .reset_index()
        if not ablation.empty
        else pd.DataFrame()
    )
    _write_csv(ablation_means, output / "core_ablation_table.csv")
    losses = frame[frame["variant"].isin(("loss_w_000", "loss_w_025", "loss_w_050", "loss_w_075", "loss_w_100"))]
    _write_csv(losses, output / "loss_weight_results.csv")
    _plot_loss_weights(losses, output / "loss_weight_curves.png")
    diagnostic_record_keys = {
        (
            row["phase"],
            row["dataset_id"],
            row["seed"],
            row["method"],
            row["variant"],
            row["mode"],
            row["admissible"],
            row["execution_scope"],
            row["search_space_version"],
        )
        for _, row in frame.iterrows()
    }
    diagnostics: list[dict[str, Any]] = []
    for record in records:
        record_key = (
            record.get("phase"),
            record.get("dataset_id"),
            record.get("split_seed"),
            record.get("method"),
            record.get("variant"),
            record.get("mode"),
            record.get("admissible", False),
            record.get("execution_scope"),
            record.get("search_space_version"),
        )
        if record_key not in diagnostic_record_keys:
            continue
        graph = record.get("model", {}).get("graph_diagnostics")
        if graph:
            diagnostics.append(
                {
                    "dataset_id": record.get("dataset_id"),
                    "seed": record.get("split_seed"),
                    "variant": record.get("variant"),
                    **graph,
                }
            )
    diagnostic_frame = pd.DataFrame(diagnostics)
    _write_csv(diagnostic_frame, output / "graph_diagnostics.csv")
    failure_frame = pd.DataFrame(list(failures))
    exclusion_frame = pd.DataFrame(list(exclusions))
    _write_csv(failure_frame, output / "execution_failures.csv")
    _write_csv(exclusion_frame, output / "excluded_datasets.csv")
    protocol_complete = bool(dataset_ids) and set(valid_dataset_ids) == set(dataset_ids)
    summary = {
        "completed_records": int(frame.shape[0]),
        "main_records": int(main.shape[0]),
        "expected_methods": list(expected_methods),
        "expected_seeds": list(expected_seeds),
        "datasets_expected": dataset_ids,
        "datasets_complete_for_comparison": valid_dataset_ids,
        "protocol_complete": protocol_complete,
        "admissibility_enforced": enforce_admissible,
        "execution_failure_records": int(failure_frame.shape[0]),
        "excluded_dataset_records": int(exclusion_frame.shape[0]),
        "core_ablation_records": int(ablation.shape[0]),
        "loss_weight_records": int(losses.shape[0]),
        "graph_diagnostic_records": int(diagnostic_frame.shape[0]),
    }
    write_json(output / "report_summary.json", summary)
    return summary


# ---------------------------------------------------------------------------
# Phase 0 self-check and command-line entry points
# ---------------------------------------------------------------------------


def synthetic_reporting_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    variants = list(CORE_ABLATIONS) + [
        "loss_w_000",
        "loss_w_025",
        "loss_w_050",
        "loss_w_075",
        "loss_w_100",
    ]
    for dataset_id in (9001, 9002):
        for seed in (0, 1):
            for index, variant in enumerate(variants):
                records.append(
                    {
                        "status": "completed",
                        "phase": "phase_2" if variant in CORE_ABLATIONS else "phase_3",
                        "dataset_id": dataset_id,
                        "dataset_alias": f"synthetic-{dataset_id}",
                        "split_seed": seed,
                        "method": "dbnc",
                        "method_label": "DBNC",
                        "variant": variant,
                        "mode": "full_config_toggle",
                        "metrics": {
                            "accuracy": 0.80 - index * 0.005,
                            "macro_f1": 0.79 - index * 0.005,
                            "auc": 0.85 - index * 0.004,
                            "log_loss": 0.40 + index * 0.005,
                            "brier": 0.20 + index * 0.003,
                            "ece": 0.05 + index * 0.002,
                        },
                        "timing": {"training_seconds": 0.01, "inference_seconds": 0.001},
                        "model": {
                            "graph_diagnostics": {
                                "edge_mass": 1.0,
                                "thresholded_edge_density": 0.2,
                                "parent_limit_violation_rate": 0.0,
                                "acyclic": True,
                            }
                        },
                    }
                )
            for method, accuracy in (("dbnc", 0.84), ("nb", 0.76), ("tan", 0.78)):
                records.append(
                    {
                        "status": "completed",
                        "phase": "phase_1",
                        "dataset_id": dataset_id,
                        "dataset_alias": f"synthetic-{dataset_id}",
                        "split_seed": seed,
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "variant": method_variant(method),
                        "mode": "main",
                        "metrics": {
                            "accuracy": accuracy,
                            "macro_f1": accuracy,
                            "auc": accuracy,
                            "log_loss": 1.0 - accuracy,
                            "brier": 1.0 - accuracy,
                            "ece": 0.1 - accuracy / 20,
                        },
                        "timing": {"training_seconds": 0.01, "inference_seconds": 0.001},
                        "model": {},
                    }
                )
    return records


def run_self_check(root: Path, device: str = "cpu") -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}

    def check(name: str, operation: Callable[[], None]) -> None:
        try:
            operation()
            checks[name] = {"passed": True}
        except Exception as exc:
            checks[name] = {
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }

    rng = np.random.default_rng(14)
    X_discrete = rng.integers(0, 3, size=(36, 4), dtype=np.int64)
    y_discrete = (X_discrete[:, 0] + X_discrete[:, 1] > 2).astype(np.int64)
    cards = [3, 3, 3, 3]
    base = {
        "d": 8,
        "n_heads": 2,
        "epochs": 1,
        "lr": 0.003,
        "batch_size": 16,
        "lambda_sparse": 0.001,
        "lambda_degree": 0.001,
        "max_parents": 2,
        "hybrid_weight": 0.5,
        "dropout": 0.0,
        "early_stopping_rounds": 1,
    }
    trained_models: dict[str, DBNC] = {}

    def variants_train() -> None:
        for index, variant in enumerate(DBNC_VARIANTS):
            set_seed(index)
            parameters = dbnc_variant_parameters(variant, base, X_discrete, y_discrete)
            model = DBNC(cards, 2, device=device, **parameters)
            model.fit(X_discrete, y_discrete)
            probabilities = model.predict_proba(X_discrete[:8])
            if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1.0):
                raise AssertionError(f"{variant} did not return normalized finite probabilities")
            trained_models[variant] = model

    check("dbnc_variants_finite_probabilities", variants_train)

    def graph_checks() -> None:
        if "full" not in trained_models:
            raise AssertionError("variant training prerequisite failed")
        learned = trained_models["full"].adjacency().detach().cpu().numpy()
        if not is_acyclic(learned > 0):
            raise AssertionError("rank adjacency is cyclic")
        tan = tan_adjacency(X_discrete, y_discrete)
        if not is_acyclic(tan) or int(tan.sum()) != X_discrete.shape[1] - 1:
            raise AssertionError("TAN structure is not an acyclic feature tree")
        no_edges = trained_models["no_edges"].adjacency().detach().cpu().numpy()
        if np.count_nonzero(no_edges) != 0:
            raise AssertionError("no_edges supplied non-zero adjacency")

    check("graph_structure_variants", graph_checks)

    def endpoint_and_penalty_checks() -> None:
        for variant, weight in (("loss_w_000", 0.0), ("loss_w_100", 1.0)):
            if trained_models[variant].hybrid_weight != weight:
                raise AssertionError(f"{variant} did not configure endpoint loss weight")
        if trained_models["no_regularization"].lambda_sparse != 0.0 or trained_models["no_regularization"].lambda_degree != 0.0:
            raise AssertionError("no_regularization did not disable both penalties")
        if trained_models["no_sparse"].lambda_sparse != 0.0 or trained_models["no_sparse"].lambda_degree == 0.0:
            raise AssertionError("no_sparse disabled the wrong penalties")
        if trained_models["no_degree"].lambda_degree != 0.0 or trained_models["no_degree"].lambda_sparse == 0.0:
            raise AssertionError("no_degree disabled the wrong penalties")

    check("endpoint_losses_and_penalties", endpoint_and_penalty_checks)

    def dbnc_config_filter_check() -> None:
        parameters = dbnc_variant_parameters(
            "full",
            {
                **base,
                "numeric_bins": 16,
                "cpd_type": "attention",
                "edge_logit_mean": -1.0,
                "rank_init_std": 0.5,
                "n_restarts": 1,
            },
            X_discrete,
            y_discrete,
        )
        if "numeric_bins" in parameters:
            raise AssertionError("DBNC model parameters leaked numeric_bins")
        if "n_restarts" in parameters:
            raise AssertionError("DBNC model parameters leaked n_restarts")
        model = DBNC(cards, 2, device=device, **parameters)
        if model.cpd_type != "attention":
            raise AssertionError("DBNC model parameters dropped cpd_type")
        if model.edge_logit_mean != -1.0 or model.rank_init_std != 0.5:
            raise AssertionError("DBNC model parameters dropped graph initialization controls")

    check("dbnc_model_config_filtering", dbnc_config_filter_check)

    def dbnc_predictive_search_space_check() -> None:
        class RecordingTrial:
            def __init__(self) -> None:
                self.categorical_choices: dict[str, Sequence[Any]] = {}

            def suggest_categorical(self, name: str, choices: Sequence[Any]) -> Any:
                self.categorical_choices[name] = choices
                return choices[0]

            def suggest_float(self, name: str, low: float, high: float, **kwargs: Any) -> float:
                return low

            def suggest_int(self, name: str, low: int, high: int, **kwargs: Any) -> int:
                return low

        trial = RecordingTrial()
        configuration = suggest_configuration("dbnc", trial, quick=False)
        if list(trial.categorical_choices["numeric_bins"]) != [N_BINS, 16, 32]:
            raise AssertionError("DBNC search space did not expose wider numeric bins")
        if configuration["cpd_type"] != "attention":
            raise AssertionError("DBNC search space did not pin CPD to attention")
        if 0.0 not in trial.categorical_choices["hybrid_weight"]:
            raise AssertionError("DBNC search space did not include conditional-only loss")
        if configuration["n_restarts"] != 1:
            raise AssertionError("DBNC search space did not disable multiple restarts")
        if configuration["min_epochs_before_checkpoint"] < 2 or configuration["min_epochs_before_stopping"] < 2:
            raise AssertionError("DBNC search space did not guard against early graph collapse")
        if configuration["edge_logit_mean"] > -1.0 or configuration["rank_init_std"] < 0.1:
            raise AssertionError("DBNC search space did not loosen graph initialization")

    check("dbnc_v4_predictive_search_space", dbnc_predictive_search_space_check)

    def preprocessing_check() -> None:
        train = pd.DataFrame(
            {
                "category": ["common"] * 6 + ["rare"],
                "numeric": [1.0, 2.0, np.nan, 3.0, 4.0, 5.0, 2.0],
            }
        )
        heldout = pd.DataFrame({"category": ["rare", "unseen"], "numeric": [999.0, np.nan]})
        transform = TrainingPreprocessor(
            ["category", "numeric"], ["category"], ["numeric"]
        ).fit(train)
        encoded = transform.transform(heldout).discrete
        if encoded[0, 0] != transform.category_maps["category"]["<RARE>"]:
            raise AssertionError("training rare category was not deterministically mapped to <RARE>")
        if encoded[1, 0] != transform.category_maps["category"]["<UNKNOWN>"]:
            raise AssertionError("held-out unseen category was not mapped to <UNKNOWN>")
        if transform.numeric_medians["numeric"] >= 100:
            raise AssertionError("held-out values affected the fitted imputer")
        binned_train = pd.DataFrame({"numeric": np.arange(64, dtype=np.float64)})
        binned_transform = TrainingPreprocessor(
            ["numeric"],
            [],
            ["numeric"],
            numeric_bins=16,
        ).fit(binned_train)
        if binned_transform.metadata()["numeric_bins_requested"] != 16:
            raise AssertionError("numeric_bins was not recorded in preprocessing metadata")
        if binned_transform.discrete_cards != [16]:
            raise AssertionError("numeric_bins did not control fitted numeric cardinality")
        sparse_train = pd.DataFrame(
            {
                "numeric": pd.Series(
                    [0.0, 1.0, np.nan, 2.0],
                    dtype=pd.SparseDtype("float32", np.nan),
                )
            }
        )
        sparse_transform = TrainingPreprocessor(["numeric"], [], ["numeric"]).fit(sparse_train)
        sparse_encoded = sparse_transform.transform(sparse_train).discrete
        if sparse_encoded.shape != (4, 1) or not np.isfinite(sparse_encoded).all():
            raise AssertionError("sparse numeric preprocessing did not produce finite encoded data")

    check("training_only_preprocessing", preprocessing_check)

    def class_covering_subset_check() -> None:
        y = np.asarray([0] * 1000 + [1] * 2 + [2] * 20, dtype=np.int64)
        indices = np.arange(y.shape[0], dtype=np.int64)
        subset = class_covering_training_subset(indices, y, 50, seed=0)
        observed = set(int(value) for value in y[subset])
        if observed != {0, 1, 2} or subset.shape[0] != 50:
            raise AssertionError("class-covering tuning subset dropped a rare class")

    check("class_covering_tuning_subset", class_covering_subset_check)

    def checkpoint_rule_check() -> None:
        if not validation_checkpoint_improved(0.8, 1.0, 0.7, 0.1):
            raise AssertionError("checkpoint rule did not prefer higher validation accuracy")
        if validation_checkpoint_improved(0.7, 0.01, 0.8, 1.0):
            raise AssertionError("checkpoint rule preferred log-loss over lower accuracy")
        if not validation_checkpoint_improved(0.8, 0.1, 0.8, 0.2):
            raise AssertionError("checkpoint rule did not use log-loss as an accuracy tie-breaker")

    check("dbnc_accuracy_first_checkpoint_rule", checkpoint_rule_check)

    def result_path_check() -> None:
        paths = {
            result_path(root, "dbnc", variant, 1, 0, mode, "hp")
            for variant in DBNC_VARIANTS
            for mode in MODES
        }
        if len(paths) != len(DBNC_VARIANTS) * len(MODES):
            raise AssertionError("variant or mode result paths collide")
        dbnc_tuning = tuning_path(root, "dbnc", "full", 1)
        old_dbnc_tuning = root / "results" / "tuning" / "dbnc" / "full" / "1.json"
        if dbnc_tuning == old_dbnc_tuning:
            raise AssertionError("DBNC tuning path was not versioned")
        if DBNC_SEARCH_SPACE_VERSION not in dbnc_tuning.parts:
            raise AssertionError("DBNC tuning path omitted the search-space version")
        baseline_tuning = tuning_path(root, "nb", "baseline", 1)
        if DBNC_SEARCH_SPACE_VERSION in baseline_tuning.parts:
            raise AssertionError("non-DBNC tuning path unexpectedly used DBNC versioning")
        configuration = {"numeric_bins": N_BINS, "cpd_type": "attention"}
        if result_hp_id("dbnc", configuration) == stable_id(configuration):
            raise AssertionError("DBNC result hp_id did not include the search-space version")
        if result_hp_id("nb", configuration) != stable_id(configuration):
            raise AssertionError("non-DBNC result hp_id unexpectedly included DBNC versioning")

    check("result_identifier_non_collision", result_path_check)

    def current_main_record_preference_check() -> None:
        duplicate_records = [
            {
                "status": "completed",
                "phase": "phase_1",
                "dataset_id": 1,
                "dataset_alias": "synthetic",
                "split_seed": 0,
                "method": "dbnc",
                "method_label": METHOD_LABELS["dbnc"],
                "variant": "full",
                "mode": "main",
                "metrics": {"accuracy": 0.1},
                "timing": {},
            },
            {
                "status": "completed",
                "phase": "phase_1",
                "dataset_id": 1,
                "dataset_alias": "synthetic",
                "split_seed": 0,
                "method": "dbnc",
                "method_label": METHOD_LABELS["dbnc"],
                "variant": "full",
                "mode": "main",
                "search_space_version": DBNC_SEARCH_SPACE_VERSION,
                "metrics": {"accuracy": 0.9},
                "timing": {},
            },
        ]
        preferred = prefer_current_main_records(records_frame(duplicate_records))
        if preferred.shape[0] != 1 or float(preferred.iloc[0]["accuracy"]) != 0.9:
            raise AssertionError("report records did not prefer the current DBNC search-space version")

    check("current_dbnc_result_preference", current_main_record_preference_check)

    def metrics_check() -> None:
        perfect = predictive_metrics(
            np.asarray([0, 1, 2]),
            np.eye(3, dtype=np.float64),
        )
        if perfect["accuracy"] != 1.0 or perfect["brier"] != 0.0 or perfect["ece"] != 0.0:
            raise AssertionError("perfect multiclass toy metrics are incorrect")
        calibration = expected_calibration_error(
            np.asarray([0, 1]),
            np.asarray([[0.8, 0.2], [0.4, 0.6]]),
        )
        if not np.isclose(calibration, 0.3):
            raise AssertionError(f"known ECE expected 0.3, obtained {calibration}")

    check("known_metric_values", metrics_check)

    def reporting_check() -> None:
        report_root = root / "cache" / "self_check_reports"
        if report_root.exists():
            shutil.rmtree(report_root)
        summary = generate_reports(
            synthetic_reporting_records(),
            report_root,
            expected_methods=("dbnc", "nb", "tan"),
            expected_seeds=(0, 1),
            accepted_dataset_ids=(9001, 9002),
            enforce_admissible=False,
        )
        required = (
            "core_ablation_table.csv",
            "loss_weight_curves.png",
            "graph_diagnostics.csv",
            "main_ranking_inputs.csv",
        )
        if any(not (report_root / filename).exists() for filename in required):
            raise AssertionError("synthetic report generation omitted required outputs")
        if summary["main_records"] == 0 or summary["core_ablation_records"] == 0:
            raise AssertionError("synthetic reports contain no required records")
        if not summary["protocol_complete"]:
            raise AssertionError("complete synthetic main records were not admitted")
        significant_records: list[dict[str, Any]] = []
        for dataset_id in range(9100, 9110):
            for seed in (0, 1):
                for method, accuracy in (("dbnc", 0.95), ("nb", 0.70), ("tan", 0.45)):
                    significant_records.append(
                        {
                            "status": "completed",
                            "phase": "phase_1",
                            "dataset_id": dataset_id,
                            "dataset_alias": f"synthetic-{dataset_id}",
                            "split_seed": seed,
                            "method": method,
                            "method_label": METHOD_LABELS[method],
                            "variant": method_variant(method),
                            "mode": "main",
                            "metrics": {
                                "accuracy": accuracy,
                                "macro_f1": accuracy,
                                "auc": accuracy,
                                "log_loss": 1.0 - accuracy,
                                "brier": 1.0 - accuracy,
                                "ece": 1.0 - accuracy,
                            },
                            "timing": {"training_seconds": 1.0, "inference_seconds": 0.1},
                            "model": {},
                        }
                    )
        cd_root = root / "cache" / "self_check_cd_reports"
        if cd_root.exists():
            shutil.rmtree(cd_root)
        generate_reports(
            significant_records,
            cd_root,
            expected_methods=("dbnc", "nb", "tan"),
            expected_seeds=(0, 1),
            accepted_dataset_ids=tuple(range(9100, 9110)),
            enforce_admissible=False,
        )
        for metric in ("accuracy", "log_loss", "ece"):
            if not (cd_root / f"critical_difference_{metric}.png").exists():
                raise AssertionError(f"critical-difference figure missing for {metric}")

    check("report_generation", reporting_check)

    def incomplete_reporting_check() -> None:
        incomplete = [
            record
            for record in synthetic_reporting_records()
            if not (
                record.get("phase") == "phase_1"
                and record.get("method") == "nb"
                and record.get("split_seed") == 1
            )
        ]
        report_root = root / "cache" / "self_check_incomplete_reports"
        if report_root.exists():
            shutil.rmtree(report_root)
        summary = generate_reports(
            incomplete,
            report_root,
            expected_methods=("dbnc", "nb", "tan"),
            expected_seeds=(0, 1),
            accepted_dataset_ids=(9001, 9002),
            enforce_admissible=False,
        )
        if summary["protocol_complete"] or summary["datasets_complete_for_comparison"]:
            raise AssertionError("incomplete method-seed coverage was admitted to comparison")

    check("incomplete_runs_excluded_from_reporting", incomplete_reporting_check)
    passed = all(value["passed"] for value in checks.values())
    output = {
        "protocol": "phase_0_self_check",
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "device": device,
        "passed": passed,
        "checks": checks,
        "environment": environment_metadata(device),
    }
    write_json(root / "artifacts" / "self_check.json", output)
    return output


def execute_phase0(root: Path, selected: list[tuple[str, int]], device: str) -> dict[str, Any]:
    print("[phase0] running synthetic self-checks", flush=True)
    self_check = run_self_check(root, device=device)
    write_json(root / "artifacts" / "environment.json", environment_metadata(device))
    print("[phase0] inspecting declared OpenML datasets", flush=True)
    manifest = validate_manifest(root, selected)
    accepted = bool(self_check["passed"] and manifest["validation_complete"])
    record = {
        "phase": "phase_0",
        "accepted": accepted,
        "self_check_passed": self_check["passed"],
        "manifest_validation_complete": manifest["validation_complete"],
        "complete_declared_scope": manifest["complete_declared_scope"],
        "accepted_dataset_ids": manifest["accepted_dataset_ids"],
        "excluded_dataset_ids": manifest["excluded_dataset_ids"],
        "failed_dataset_ids": manifest["failed_dataset_ids"],
        "rule": "Final benchmark results are admissible only when accepted is true.",
    }
    write_json(root / "artifacts" / "phase0_acceptance.json", record)
    return record


def synthetic_pipeline_bundle() -> DatasetBundle:
    rng = np.random.default_rng(501)
    n_rows = 240
    labels = np.repeat(np.arange(3), n_rows // 3)
    rng.shuffle(labels)
    category = np.asarray([f"class_{label}" for label in labels], dtype=object)
    corrupt = rng.choice(n_rows, size=24, replace=False)
    category[corrupt] = np.asarray([f"class_{value}" for value in rng.integers(0, 3, size=24)])
    category[::31] = None
    rare = np.asarray(["common"] * n_rows, dtype=object)
    rare[::37] = [f"rare_{index}" for index in range(rare[::37].shape[0])]
    numeric_missing = labels.astype(float) + rng.normal(scale=0.35, size=n_rows)
    numeric_missing[::29] = np.nan
    X = pd.DataFrame(
        {
            "numeric_signal": labels.astype(float) + rng.normal(scale=0.25, size=n_rows),
            "numeric_missing": numeric_missing,
            "categorical_signal": category,
            "categorical_rare": rare,
        }
    )
    return DatasetBundle(
        alias="synthetic-pipeline",
        dataset_id=990001,
        version=1,
        openml_name="synthetic-pipeline",
        target="target",
        X=X,
        y_raw=pd.Series(labels.astype(str)),
        categorical_columns=["categorical_signal", "categorical_rare"],
        numeric_columns=["numeric_signal", "numeric_missing"],
    )


def fairness_audit(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_methods: Sequence[str],
    expected_seeds: Sequence[int],
    required_device: str,
    execution_scope: str = "smoke",
) -> dict[str, Any]:
    issues: list[str] = []
    frame = records_frame(records)
    for method in expected_methods:
        observed = sorted(
            int(value)
            for value in frame.loc[frame["method"] == method, "seed"].dropna().unique()
        )
        if observed != sorted(int(seed) for seed in expected_seeds):
            issues.append(f"{method} has seeds {observed}, expected {sorted(expected_seeds)}")
    for seed in expected_seeds:
        subset = [record for record in records if record.get("split_seed") == seed]
        if len({record["split_id"] for record in subset}) != 1:
            issues.append(f"seed {seed} is not evaluated on one identical persisted split")
        if len({record["transform_id"] for record in subset}) != 1:
            issues.append(f"seed {seed} is not evaluated using one identical fitted transform")
    expected_representations = {
        "dbnc": "discrete",
        "nb": "discrete",
        "chow_liu": "discrete",
        "tan": "discrete",
        "ban": "discrete",
        "kdb_1": "discrete",
        "kdb_2": "discrete",
        "kdb_3": "discrete",
        "xgboost": "undiscretized_numeric_with_native_categorical_columns",
        "catboost": "undiscretized_numeric_with_native_categorical_columns",
        "lightgbm": "undiscretized_numeric_with_native_categorical_columns",
        "ft_transformer": "standardized_numeric_with_categorical_embeddings",
    }
    neural_methods = {"dbnc", "ft_transformer"}
    for record in records:
        method = str(record["method"])
        metrics = record["metrics"]
        if not (0.0 <= float(metrics["accuracy"]) <= 1.0):
            issues.append(f"{method} produced an invalid accuracy")
        for metric in ("macro_f1", "log_loss", "brier", "ece"):
            if metrics.get(metric) is None or not np.isfinite(float(metrics[metric])):
                issues.append(f"{method} produced non-finite {metric}")
        if record.get("feature_representation") != expected_representations[method]:
            issues.append(f"{method} used an unexpected representation")
        if record.get("model", {}).get("device_assignment") != required_device:
            issues.append(f"{method} was not assigned to the required smoke device {required_device}")
        if method in neural_methods and record.get("parameter_count") is None:
            issues.append(f"{method} omitted its neural parameter count")
        if record.get("admissible") is not False or record.get("execution_scope") != execution_scope:
            issues.append(f"{method} development record was not isolated as non-admissible")
        provenance = record.get("tuning_provenance", {})
        if provenance.get("split_seed") != 0:
            issues.append(f"{method} selected hyperparameters outside split seed 0")
    return {
        "passed": not issues,
        "issues": issues,
        "comparison_scope": f"{execution_scope}_only",
        "same_split_and_transform_per_seed": not any("identical" in issue for issue in issues),
        "runtime_device_policy": f"all smoke models assigned to {required_device}",
        "fairness_notes": [
            "All methods use identical train/validation/test indices for a seed.",
            "Representations differ only by the family-appropriate protocol rule.",
            "Neural models use validation early stopping; non-neural models fit on training only.",
            "One quick tuning trial and two final seeds exercise the pipeline but are not evidence of comparative superiority.",
        ],
    }


def run_pipeline_smoke(root: Path, device: str = "cpu") -> dict[str, Any]:
    if str(torch.device(device)) != "cpu":
        raise ExperimentError("The fairness smoke run requires --device cpu so every method uses the same device")
    smoke_root = root / "smoke_pipeline"
    self_check = run_self_check(smoke_root, device=device)
    bundle = synthetic_pipeline_bundle()
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    smoke_seeds = (0, 1)
    for method in MAIN_METHODS:
        variant = method_variant(method)
        try:
            tuning = tune_method(
                smoke_root,
                method,
                variant,
                bundle,
                trials=1,
                device=device,
                execution_scope="smoke",
                quick=True,
            )
            for seed in smoke_seeds:
                records.append(
                    final_evaluation(
                        smoke_root,
                        method,
                        variant,
                        bundle,
                        seed,
                        tuning,
                        device=device,
                        admissible=False,
                        execution_scope="smoke",
                    )
                )
        except Exception as exc:
            failure = {
                "phase": "phase_1_smoke",
                "dataset_id": bundle.dataset_id,
                "method": method,
                "variant": variant,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            append_failure(smoke_root, failure)
    summary: dict[str, Any] | None = None
    if records:
        summary = generate_reports(
            records,
            smoke_root / "reports",
            expected_methods=MAIN_METHODS,
            expected_seeds=smoke_seeds,
            accepted_dataset_ids=(bundle.dataset_id,),
            failures=failures,
            enforce_admissible=False,
        )
    audit = fairness_audit(
        records,
        expected_methods=MAIN_METHODS,
        expected_seeds=smoke_seeds,
        required_device="cpu",
        execution_scope="smoke",
    )
    passed = bool(
        self_check["passed"]
        and not failures
        and summary is not None
        and summary["protocol_complete"]
        and audit["passed"]
    )
    result = {
        "passed": passed,
        "self_check_passed": self_check["passed"],
        "completed_results": len(records),
        "expected_results": len(MAIN_METHODS) * len(smoke_seeds),
        "failures": failures,
        "report_summary": summary,
        "fairness_audit": audit,
        "artifact_root": str(smoke_root),
    }
    write_json(smoke_root / "artifacts" / "pipeline_smoke.json", result)
    return result


def manifest_artifact(root: Path) -> dict[str, Any]:
    path = root / "artifacts" / "validated_manifest.json"
    if not path.exists():
        raise ExperimentError("Validated manifest is missing; run the 'phase0' command first")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_confirmatory_scope(
    root: Path,
    selected: Sequence[tuple[str, int]],
    methods: Sequence[str],
    seeds: Sequence[int],
    trials: int,
) -> dict[str, Any]:
    require_phase0_gate(root)
    manifest = manifest_artifact(root)
    accepted_ids = set(int(value) for value in manifest["accepted_dataset_ids"])
    selected_ids = set(dataset_id for _, dataset_id in selected)
    if not accepted_ids.issubset(selected_ids):
        missing = sorted(accepted_ids - selected_ids)
        raise ExperimentError(
            f"Confirmatory Phase 1 must run every accepted dataset; missing IDs: {missing}"
        )
    if len(methods) != len(MAIN_METHODS) or set(methods) != set(MAIN_METHODS):
        raise ExperimentError(
            "Confirmatory Phase 1 must execute DBNC and all eleven declared baselines; "
            "use --development for reduced method smoke runs"
        )
    if len(seeds) != len(SEEDS) or set(int(seed) for seed in seeds) != set(SEEDS):
        raise ExperimentError(
            "Confirmatory Phase 1 must execute all five seeds {0,1,2,3,4}; "
            "use --development for reduced seed smoke runs"
        )
    if trials != 20:
        raise ExperimentError(
            "Confirmatory Phase 1 requires the declared 20-trial tuning budget; "
            "use --development for small-budget runs"
        )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DBNC Phase 0 validation and Phase 1 benchmark runner (Python 3)."
    )
    parser.add_argument("--root", type=Path, default=Path("."), help="Artifact root directory.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    self_check = subparsers.add_parser("self-check", help="Run synthetic Phase 0 checks.")
    self_check.add_argument("--device", default="cpu")

    inspect = subparsers.add_parser("inspect", help="Retrieve and persist the dataset manifest.")
    inspect.add_argument("--datasets", nargs="*", help="Declared aliases or OpenML IDs; default is all.")

    phase0 = subparsers.add_parser("phase0", help="Run self-checks and complete manifest validation.")
    phase0.add_argument("--datasets", nargs="*", help="Declared aliases or OpenML IDs; default is all.")
    phase0.add_argument("--device", default="cpu")

    pipeline_smoke = subparsers.add_parser(
        "pipeline-smoke",
        help="Run all Phase 1 method adapters end-to-end on a development-only synthetic dataset.",
    )
    pipeline_smoke.add_argument("--device", default="cpu")

    benchmark = subparsers.add_parser("benchmark", help="Execute the Phase 1 comparative benchmark.")
    benchmark.add_argument("--datasets", nargs="*", help="Accepted aliases or OpenML IDs; default is all.")
    benchmark.add_argument("--methods", nargs="*", choices=MAIN_METHODS, default=list(MAIN_METHODS))
    benchmark.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    benchmark.add_argument("--trials", type=int, default=20)
    benchmark.add_argument("--device", default="cpu")
    benchmark.add_argument(
        "--development",
        action="store_true",
        help="Write non-admissible reduced-scope runs under development/ and report only that scope.",
    )
    benchmark.add_argument(
        "--quick",
        action="store_true",
        help="Development-only reduced training settings for integration smoke runs.",
    )
    benchmark.add_argument(
        "--skip-phase0-gate",
        action="store_true",
        help="Development-only: bypass Phase 0; output remains non-admissible under development/.",
    )

    report = subparsers.add_parser("report", help="Aggregate completed stored runs into tables and plots.")
    report.add_argument("--output", type=Path, default=None)
    report.add_argument("--development", action="store_true", help="Report the non-admissible development subtree.")
    report.add_argument("--methods", nargs="*", choices=MAIN_METHODS, default=list(MAIN_METHODS))
    report.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    root = arguments.root.resolve()
    try:
        if arguments.command == "self-check":
            record = run_self_check(root, arguments.device)
            print(json.dumps({"passed": record["passed"], "artifact": str(root / "artifacts" / "self_check.json")}))
            return 0 if record["passed"] else 1
        if arguments.command == "inspect":
            manifest = validate_manifest(root, parse_dataset_selection(arguments.datasets))
            print(json.dumps({"validation_complete": manifest["validation_complete"]}))
            return 0 if manifest["validation_complete"] else 1
        if arguments.command == "phase0":
            result = execute_phase0(root, parse_dataset_selection(arguments.datasets), arguments.device)
            print(json.dumps(result, sort_keys=True))
            return 0 if result["accepted"] else 1
        if arguments.command == "pipeline-smoke":
            result = run_pipeline_smoke(root, arguments.device)
            print(json.dumps(result, sort_keys=True))
            return 0 if result["passed"] else 1
        if arguments.command == "benchmark":
            invalid_seeds = set(arguments.seeds) - set(SEEDS)
            if invalid_seeds:
                raise ExperimentError(f"Phase 1 final evaluation seeds must be selected from {SEEDS}")
            selected = parse_dataset_selection(arguments.datasets)
            development = bool(arguments.development or arguments.skip_phase0_gate)
            if arguments.quick and not development:
                raise ExperimentError("--quick is development-only and cannot generate confirmatory results")
            if not development:
                manifest = validate_confirmatory_scope(
                    root,
                    selected,
                    arguments.methods,
                    arguments.seeds,
                    arguments.trials,
                )
                artifact_root = root
                admissible = True
                execution_scope = "confirmatory"
                enforce_admissible = True
            else:
                manifest = manifest_artifact(root)
                artifact_root = root / "development"
                admissible = False
                execution_scope = "development"
                enforce_admissible = False
            accepted_selected = [
                dataset_id
                for _, dataset_id in selected
                if dataset_id in set(int(value) for value in manifest["accepted_dataset_ids"])
            ]
            records = run_benchmark(
                artifact_root,
                selected,
                arguments.methods,
                arguments.seeds,
                trials=arguments.trials,
                device=arguments.device,
                skip_phase0_gate=arguments.skip_phase0_gate,
                manifest_root=root,
                admissible=admissible,
                execution_scope=execution_scope,
                quick=bool(arguments.quick),
            )
            summary = generate_reports(
                read_completed_results(artifact_root, admissible_only=enforce_admissible),
                artifact_root / "reports",
                expected_methods=arguments.methods,
                expected_seeds=arguments.seeds,
                accepted_dataset_ids=accepted_selected,
                failures=read_failure_records(artifact_root),
                exclusions=[
                    item for item in manifest["records"] if item["status"] == "excluded"
                ],
                enforce_admissible=enforce_admissible,
            )
            print(json.dumps({"new_or_resumed_records": len(records), "report": summary}, sort_keys=True))
            return 0 if summary["protocol_complete"] else 1
        if arguments.command == "report":
            invalid_seeds = set(arguments.seeds) - set(SEEDS)
            if invalid_seeds:
                raise ExperimentError(f"Report seeds must be selected from {SEEDS}")
            if arguments.development:
                artifact_root = root / "development"
                enforce_admissible = False
                dataset_ids = None
            else:
                require_phase0_gate(root)
                artifact_root = root
                enforce_admissible = True
                dataset_ids = manifest_artifact(root)["accepted_dataset_ids"]
            output = arguments.output.resolve() if arguments.output else artifact_root / "reports"
            summary = generate_reports(
                read_completed_results(artifact_root, admissible_only=enforce_admissible),
                output,
                expected_methods=arguments.methods,
                expected_seeds=arguments.seeds,
                accepted_dataset_ids=dataset_ids,
                failures=read_failure_records(artifact_root),
                exclusions=(
                    [item for item in manifest_artifact(root)["records"] if item["status"] == "excluded"]
                    if not arguments.development
                    else []
                ),
                enforce_admissible=enforce_admissible,
            )
            print(json.dumps(summary, sort_keys=True))
            return 0 if summary["protocol_complete"] else 1
    except ExperimentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
