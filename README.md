# DBNC — Differentiable Bayesian Network Classifier

Reference implementation for the paper **"Rethinking Bayesian Network Classifiers for Categorical Tabular Data with Differentiable Structure Learning"**.

`DBNC` learns a **rank-parameterised soft DAG** (acyclic by construction) together with **attention-based neural conditional probability models**, trained end-to-end under a hybrid generative–discriminative objective. It outperforms classical Bayesian-network classifiers (Naive Bayes, Chow-Liu, TAN, BAN, *k*-DB) while preserving an explicit, interpretable learned dependency graph.

## Installation

```bash
pip install -e .                 # core: the DBNC model + package API
pip install -e ".[experiments]"  # + Optuna, OpenML, XGBoost, LightGBM, CatBoost
pip install -e ".[figures]"      # + matplotlib (figure generation)
pip install -e ".[all]"          # everything
```

Python ≥ 3.10. Published results were produced with Python 3.11, PyTorch 2.5, scikit-learn 1.8, Optuna 4.8, XGBoost 3.2, LightGBM 4.6, and CatBoost 1.2.

## Quick Start

```python
from dbnc import DBNC

model = DBNC(cards=[3, 4, 2, 5], n_classes=3, d=32, n_heads=4, epochs=100)
model.fit(X_train, y_train, X_validation, y_validation)   # integer-encoded features
proba = model.predict_proba(X_test)
adjacency = model.adjacency()        # learned rank-acyclic soft graph
diagnostics = model.graph_diagnostics()
```

A runnable end-to-end example (synthetic data, no downloads required):

```bash
python examples/quickstart.py
```

## Package Layout

```
src/dbnc/
  model.py             # DBNC model (rank-soft-DAG + attention CPDs + hybrid loss)
  adjacency.py         # acyclicity utilities (is_acyclic, GRAPH_THRESHOLD)
  config.py            # protocol constants, method & dataset registries
  variants.py          # ablation/variant parameterisation
  data.py              # OpenML loading, training-only preprocessing, splits
  baselines.py         # NB, Chow-Liu, TAN, BAN, k-DB + structure discovery
  baselines_modern.py  # XGBoost, CatBoost, LightGBM, FT-Transformer
  metrics.py           # accuracy, macro-F1, AUC, log-loss, Brier, ECE
  tuning.py            # Optuna (TPE) search spaces + selection
  benchmark.py         # fit / evaluate / time per (method, dataset, seed)
  reporting.py         # aggregation + Friedman / Nemenyi / Wilcoxon
  cli.py               # `dbnc` command-line interface
scripts/               # figure & report generation (shared paper_style.py)
configs/tuning/        # selected hyperparameters per (method, dataset)
results/               # aggregated results backing all tables and figures
tests/                 # pytest: acyclicity, fit/predict, determinism, baselines
examples/              # quickstart + figure reproduction
```

## Reproducing Figures and Tables

The aggregated results backing every table and figure are provided in `results/`. Figures can be regenerated without retraining:

```bash
pip install -e ".[figures]"
python scripts/make_paper_figures.py --reports results --ablation results/ablation --paper paper_figures
# -> paper_figures/figures/*.pdf
```

| Artifact | Source | Command |
|---|---|---|
| Tables III–IV (benchmark, pairwise) | `results/tables/method_summary.csv`, `results/tables/dbnc_pairwise_tests.csv` | shipped; see `scripts/make_benchmark_report.py` to regenerate |
| Statistical tests (Friedman/Nemenyi) | `results/main_statistical_tests.json` | shipped |
| Figs. 1–4, 6 (architecture, CD, deltas, rank heatmap, Pareto) | `results/*` | `scripts/make_paper_figures.py` |
| Fig. 5 (learned `car` graph) | retrains DBNC | `scripts/extract_learned_dag.py` |
| Table V + ablation figure | `results/ablation/core_ablation_summary.csv` | `scripts/make_paper_figures.py` |

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the full experimental protocol.

## Re-running the Full Benchmark

```bash
dbnc phase0                                          # validate the dataset manifest
dbnc benchmark --device cuda:0 --seeds 0 1 2 3 4     # 30 datasets × 12 methods × 5 seeds
dbnc report --output reports                         # aggregate -> CSVs + statistical tests
```

This re-tunes every method per dataset with Optuna (20 trials) evaluated across five seeds — 1,800 final records in total. Pre-selected configurations that produced the published results are provided in `configs/tuning/`, including DBNC v4 configs under `configs/tuning/dbnc/full/dbnc_v4_predictive_attention_graph_bins/`.

## Tests

```bash
pip install -e ".[all]" pytest
pytest -q
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{dbnc,
  title   = {Rethinking {Bayesian} Network Classifiers for Categorical Tabular Data
             with Differentiable Structure Learning},
  author  = {},
  year    = {2026}
}
```

## License

[MIT](LICENSE).
