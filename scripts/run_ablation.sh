#!/usr/bin/env bash
# Downsampling ablation (paper Table IX): DWT vs width-matched max-pooling.
#
# Trains BOTH arms from scratch under one recorded protocol on the identical
# seed-42 1000/251 partition, then scores each through the reported evaluation
# pipeline (sliding window + Gaussian blending + 8-flip TTA, fixed threshold 0.5,
# standard component cleanup).
#
# Why both arms are retrained rather than reusing segmentor_epoch_650.pth: that
# checkpoint is a bare state_dict with no recorded training configuration, and
# the repository's training script could not run as released (see the import
# defect fixed in brats_gbm/data/brats.py), so its protocol cannot be matched
# after the fact. The pair produced here is self-contained and is used ONLY for
# the ablation; the headline results elsewhere in the paper continue to come
# from the original checkpoint.
#
# Cost: ~50 h per arm on an RTX 5090 (measured 211 ms/case, 1000 cases,
# 750 epochs), plus ~2-4 h per arm for TTA evaluation. Budget ~4.5 days.
#
#   bash scripts/run_ablation.sh            # both arms, then evaluate
#   bash scripts/run_ablation.sh evaluate   # evaluation only (after training)
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
CKPT_DIR=checkpoints/ablation
LOG_DIR=logs
RESULT_DIR=results/brats/ablation
# 200 epochs, matched across arms. Not arbitrary: under this project's
# ReduceLROnPlateau(patience=3) the max-pooling arm reached its 1e-7 learning-rate
# floor at epoch 76 and its best validation Dice at epoch 43, then oscillated
# within noise for the remaining 150 epochs. 200 therefore clears convergence
# with a wide margin while avoiding ~53 h per arm of no-op compute. The 750 in
# train_brats.py's default is a *resume* target (that run continued an existing
# epoch-650 checkpoint), not a from-scratch budget.
EPOCHS=${EPOCHS:-200}
WORKERS=${WORKERS:-8}
SEED=${SEED:-42}
mkdir -p "$CKPT_DIR" "$LOG_DIR" "$RESULT_DIR"

train_arm () {
  local arm=$1 tag=$2
  if [ -f "$CKPT_DIR/${tag}_best.pth" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "[skip] $tag already trained ($CKPT_DIR/${tag}_best.pth); FORCE=1 to redo"
    return 0
  fi
  echo "[$(date '+%F %T')] training $tag (downsample=$arm, ${EPOCHS} epochs)"
  # -u: unbuffered, so the log streams live instead of block-buffering for
  # hours, which is what makes a multi-day run observable and failures visible.
  python -u scripts/train_brats.py \
      --downsample "$arm" \
      --tag "$tag" \
      --save-dir "$CKPT_DIR" \
      --resume "" \
      --epochs "$EPOCHS" \
      --num-workers "$WORKERS" \
      --seed "$SEED" \
      > "$LOG_DIR/ablation_${tag}.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "[FAIL] $tag exited $rc — see $LOG_DIR/ablation_${tag}.log"
    tail -20 "$LOG_DIR/ablation_${tag}.log"
    return $rc
  fi
  echo "[$(date '+%F %T')] finished $tag"
}

eval_arm () {
  local tag=$1
  local ckpt="$CKPT_DIR/${tag}_best.pth"
  [ -f "$ckpt" ] || { echo "[FAIL] missing $ckpt"; return 1; }
  echo "[$(date '+%F %T')] evaluating $tag"
  python -u scripts/evaluate_brats.py \
      --checkpoint "$ckpt" \
      --partition internal_validation \
      --out-dir "$RESULT_DIR" \
      --tag "$tag" \
      > "$LOG_DIR/ablation_eval_${tag}.log" 2>&1
  local rc=$?
  [ $rc -eq 0 ] || { echo "[FAIL] eval $tag exited $rc"; tail -20 "$LOG_DIR/ablation_eval_${tag}.log"; return $rc; }
  echo "[$(date '+%F %T')] evaluated $tag"
}

if [ "${1:-all}" != "evaluate" ]; then
  train_arm maxpool_matched ablation_maxpool || exit 1
  train_arm dwt             ablation_dwt     || exit 1
fi

eval_arm ablation_maxpool || exit 1
eval_arm ablation_dwt     || exit 1

echo "[$(date '+%F %T')] ablation complete — per-case CSVs in $RESULT_DIR"
