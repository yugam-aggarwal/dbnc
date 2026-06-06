# ML Reproducibility Checklist v2.0 — DBNC (ICDM 2026 submission)

For each item: **Answer** ∈ {Yes, No, Not applicable} and a free-form comment.
Cross-references are to the paper (sections/equations/tables/figures) and to this
anonymized supplementary package.

## For all models and algorithms

**1. A clear description of the mathematical setting, algorithm, and/or model.**
**Yes.** Section *Method* gives the full formulation: problem setup and notation,
the rank-parameterised soft DAG (Eqs. 2–3), the attention-based conditional model
(Eqs. 4–12), the joint likelihood and class posterior (Eqs. 13–15), the training
objective (Eq. 16), and the training procedure (Algorithm 1). Implementation:
`src/dbnc/model.py`.

**2. A clear explanation of any assumptions.**
**Yes.** Inputs are discrete (continuous features discretised by training-only
supervised binning, Section *Problem Setup* / *Reproducibility*); acyclicity holds
by construction via the rank ordering (Proposition 1); the objective is a hybrid of
joint and conditional likelihood with structural regularisers (Section *Training
Objective*).

**3. An analysis of the complexity (time, space, sample size) of any algorithm.**
**Yes.** Section *Inference and Complexity*: forward-pass time is
O(B·K·M²·d) (adjacency-gated attention), and the number of conditional-probability
parameters grows linearly in Σⱼcⱼ, independent of parent-set size.

## For any theoretical claim

**4. A clear statement of the claim.**
**Yes.** Proposition 1 (Acyclicity): the directed graph induced by the non-zero
entries of the soft adjacency is acyclic for any parameter setting.

**5. A complete proof of the claim.**
**Yes.** The proof immediately follows Proposition 1 (a directed cycle would require
a strictly increasing chain of ranks returning to its start). Tested in
`tests/test_model.py::test_learned_adjacency_is_acyclic`.

## For all datasets used

**6. The relevant statistics, such as number of examples.**
**Yes.** Table II lists all 30 datasets with N, number of predictors M, classes K,
and categorical/numerical counts. Programmatic registry: `dbnc.config.DATASETS`.

**7. The details of train / validation / test splits.**
**Yes.** Stratified 70/15/15 per seed (seeds 0–4), fitted on train only
(Section *Reproducibility*; `dbnc.data.split_indices` / `prepare_split`).

**8. An explanation of any data that were excluded, and all pre-processing steps.**
**Yes.** No datasets were excluded (all 30 are reported). Preprocessing —
rare-category grouping, missing-value imputation, training-only quantile
discretisation, standardisation — is described in Section *Protocol* /
*Reproducibility* and implemented in `dbnc.data.TrainingPreprocessor`.

**9. A link to a downloadable version of the dataset or simulation environment.**
**Yes.** All datasets are public on OpenML and are downloaded by OpenML ID via
`dbnc.data.load_openml_dataset` (IDs in Table II).

**10. For new data collected, a complete description of the data collection process.**
**Not applicable.** No new data were collected; all datasets are existing public
OpenML benchmarks.

## For all shared code

**11. Specification of dependencies.**
**Yes.** `pyproject.toml` / `requirements.txt`; exact paper versions in
`REPRODUCIBILITY.md` (Python 3.11, PyTorch 2.5, scikit-learn 1.8, Optuna 4.8,
XGBoost 3.2, LightGBM 4.6, CatBoost 1.2).

**12. Training code.**
**Yes.** `src/dbnc/` (model, data, tuning, benchmark) and the `dbnc` CLI are included
in this anonymized supplementary package.

**13. Evaluation code.**
**Yes.** `dbnc.metrics`, `dbnc.benchmark`, `dbnc.reporting`, and `scripts/` (figures
and statistical reports).

**14. (Pre-)trained model(s).**
**Not applicable.** No pretrained model checkpoints are required; all models are
trained from the released code, fixed seeds (0–4), and the selected configurations
in `configs/tuning/`.

**15. README file includes table of results accompanied by precise command to run.**
**Yes.** `README.md` includes a results↔command table; the figures regenerate from
the shipped `results/` via `python scripts/make_paper_figures.py --reports results
--ablation results/ablation --paper paper_figures`.

## For all reported experimental results

**16. The range of hyper-parameters considered, the selection method, and the final values.**
**Yes.** Section *Reproducibility* lists the full search ranges for every method; the
selection rule is max validation accuracy then min validation log-loss (Optuna TPE,
20 trials). Search spaces: `dbnc.tuning.suggest_configuration`. Final per-dataset
values: `configs/tuning/`, including the recovered DBNC v4 selections used for the
reported graph diagnostics.

**17. The exact number of training and evaluation runs.**
**Yes.** 30 datasets × 12 methods × 5 seeds = 1,800 final evaluations, plus 20
tuning trials per (method, dataset) (Section *Protocol*).

**18. A clear definition of the specific measure or statistics used to report results.**
**Yes.** Section *Metrics* (accuracy, macro-F1, AUC, log-loss, Brier, ECE; ECE uses
15 bins) and Section *Statistical Analysis* (Friedman, Nemenyi, Wilcoxon + Holm).

**19. A description of results with central tendency and variation.**
**Yes.** Tables III–V report mean ranks and mean values across the five seeds; Fig. 3
shows per-dataset accuracy deltas with bootstrap 95% confidence intervals; per-seed
standard deviations are computed in `results/tables/method_summary.csv`.

**20. The average runtime for each result, or estimated energy cost.**
**Yes.** Section *Efficiency and Scalability* reports geometric-mean training and
inference times per method; per-run times are in `results/main_valid_per_seed_results.csv`.

**21. A description of the computing infrastructure used.**
**Yes.** Section *Reproducibility* and `REPRODUCIBILITY.md` specify the software
stack and NVIDIA Quadro RTX 5000 GPU(s) used for neural methods.
