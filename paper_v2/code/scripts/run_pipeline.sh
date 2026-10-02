#!/bin/bash
# Full evaluation pipeline: integrity checks, segmentation, then classification.
#
# Sequential throughout, because the GPU is shared with other users' jobs.
# Every stage writes into results/ and logs into logs/.
#
#   ./scripts/run_pipeline.sh              # evaluation only, uses existing checkpoints
#   ./scripts/run_pipeline.sh --with-features   # also re-extract radiomic features (~1.5 h)
#
# This does not train. See README for training entry points.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs results
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WITH_FEATURES=0
[[ "${1:-}" == "--with-features" ]] && WITH_FEATURES=1

step() { echo; echo "[pipeline] === $* ==="; }

step "0/4 data-integrity checks"
# Fail fast: a misplaced data directory should not cost GPU hours. The
# segmentation-summary check is expected to fail on a clean checkout, so
# tolerate a non-zero exit here and re-run the check at the end.
python -u scripts/verify_no_leakage.py 2>&1 | tee logs/verify_no_leakage.log || true

step "1/4 BraTS2021 internal-validation summary"
python -u scripts/summarize_brats.py 2>&1 | tee logs/summarize_brats.log

step "2/4 UPenn-GBM segmentation evaluation"
python -u scripts/evaluate_upenn.py 2>&1 | tee logs/evaluate_upenn.log

if [[ "$WITH_FEATURES" == "1" ]]; then
  step "3/4 radiomic feature extraction"
  python -u scripts/extract_radiomic_features.py 2>&1 | tee logs/extract_features.log
else
  step "3/4 radiomic feature extraction — skipped (pass --with-features to run)"
fi

step "4/4 IDH1 classification"
python -u scripts/train_classifier_idh1.py 2>&1 | tee logs/train_classifier.log

step "final integrity check"
python -u scripts/verify_no_leakage.py 2>&1 | tee logs/verify_no_leakage.log

echo
echo "[pipeline] ALL DONE at $(date '+%F %T')"
echo "[pipeline] results in $ROOT/results"
