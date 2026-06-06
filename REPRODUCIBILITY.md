# Reproducibility

This document describes how to reproduce the results in the paper and records the
exact protocol. The completed ML Reproducibility Checklist (v2.0) is in
[reproducibility_checklist.md](reproducibility_checklist.md).

## Environment

Results were produced with **Python 3.11** and:

| Package | Version |
|---|---|
| PyTorch | 2.5 |
| scikit-learn | 1.8 |
| Optuna | 4.8 |
| XGBoost | 3.2 |
| LightGBM | 4.6 |
| CatBoost | 1.2 |
| NumPy / pandas / SciPy | see `requirements.txt` |

Hardware: NVIDIA Quadro RTX 5000 GPU(s) for the neural methods (DBNC,
FT-Transformer); the classical and boosted-tree baselines run on CPU.

## Datasets

30 public OpenML classification datasets, identified by OpenML ID (see
`dbnc.config.DATASETS` and Table II of the paper). They are downloaded and cached
automatically by `dbnc.data.load_openml_dataset`. No datasets were excluded.

## Protocol

- **Splits:** stratified 70/15/15 train/validation/test per seed; all preprocessing
  (rare-category grouping, missing-value imputation, quantile discretisation of
  numeric features, standardisation) is fitted on the training partition only.
- **Seeds:** 0–4 for the five final evaluations.
- **Tuning:** Optuna TPE, seed 0, 20 trials per (method, dataset); the selected
  configuration maximises validation accuracy, breaking ties by lower validation
  log-loss. For datasets with > 200,000 instances, tuning uses a class-covering
  subset capped at 50,000 examples; final evaluation uses the full training split.
- **Metrics:** accuracy, macro-F1, AUC, log-loss, Brier score, ECE, and train/inference
  time (`dbnc.metrics.predictive_metrics`).
- **Statistics:** Friedman omnibus + Nemenyi post-hoc across methods, and paired
  Wilcoxon signed-rank tests (Holm-corrected) for DBNC vs each baseline.

The selected per-(method, dataset) configurations that produced the published
results are in `configs/tuning/`. For DBNC, the published v4 selections are in
`configs/tuning/dbnc/full/dbnc_v4_predictive_attention_graph_bins/`; the older
`dbnc_v3_bins8_attention_hybrid/` directory is retained for traceability.

## Reproduce the figures and tables (minutes, CPU-only)

The aggregated results are shipped in `results/`, so the paper's figures/tables can
be regenerated without retraining:

```bash
pip install -e ".[figures]"
python scripts/make_paper_figures.py --reports results --ablation results/ablation --paper paper_figures
```

`results/main_statistical_tests.json` and `results/tables/*.csv` directly back the
statistical results and Tables III–IV; `results/ablation/` backs Table V and the
ablation figure.

## Re-run the full benchmark (GPU, multi-day)

```bash
dbnc phase0
dbnc benchmark --device cuda:0 --seeds 0 1 2 3 4
dbnc report --output reports
```

This re-tunes and re-evaluates all 30 × 12 × 5 = 1,800 records.

## Reproducing the learned-graph density

The DBNC graph diagnostics in the paper use the recovered v4 tuning artifacts in
`configs/tuning/dbnc/full/dbnc_v4_predictive_attention_graph_bins/`. These
artifacts include the selected values of `edge_logit_mean`,
`structure_warmup_epochs`, and `rank_init_std`, so graph retraining no longer
depends on a hand-applied override.

`scripts/extract_learned_dag.py` loads the recovered v4 `car` configuration
directly (`hp_id=c28bc0ca22ecd67e`) and reproduces the paper's representative
graph (Fig. 5):

```bash
pip install -e ".[experiments,figures]"
python scripts/extract_learned_dag.py --paper paper_figures
```

The benchmark **accuracy / probabilistic** tables reproduce directly from the
shipped aggregate results. Re-running graph diagnostics from the released configs
should use the v4 DBNC directory above; re-running from the legacy v3 DBNC configs
will produce substantially sparser learned graphs.

## What is shipped vs regenerated

- **Shipped (small, version-controlled):** aggregated results (`results/`), per-dataset
  tuning configs (`configs/tuning/`, including recovered DBNC v4 selections),
  figure/report scripts (`scripts/`).
- **Regenerated on demand:** OpenML data caches, raw per-run JSON outputs, and the
  full benchmark — none of which are committed (see `.gitignore`).
