#!/usr/bin/env bash
# Reproduce the paper's figures from the shipped aggregated results.
# Fast (a few minutes) and CPU-only -- no model retraining or downloads.
set -euo pipefail
cd "$(dirname "$0")/.."

# Install with figure extras (matplotlib).
pip install -e ".[figures]"

# Regenerate the figures into ./paper_figures/figures/ from results/.
python scripts/make_paper_figures.py \
    --reports results \
    --ablation results/ablation \
    --paper paper_figures

echo
echo "Figures written to paper_figures/figures/"
echo "The learned-graph figure (Fig. 5) retrains DBNC on the 'car' dataset and"
echo "needs the experiment extras + a network:"
echo "    pip install -e '.[experiments,figures]'"
echo "    python scripts/extract_learned_dag.py --paper paper_figures"
