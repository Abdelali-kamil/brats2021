#!/usr/bin/env bash
#
# End-to-end augmentation experiment: continue-train the BraTS segmentor WITH
# augmentation from a base checkpoint, then automatically evaluate and compare
# against that base checkpoint. Runs all five steps and saves every log,
# checkpoint, and CSV. Designed to be launched once on a GPU machine that has
# the BraTS (and, for step 3, UPenn) data.
#
# Isolates the effect of augmentation only: default BatchNorm throughout.
#
# Usage:
#   ./scripts/run_augmentation_experiment.sh
#
# Override anything via environment variables:
#   BASE_CKPT=checkpoints/segmentor_epoch_650.pth
#   EPOCHS=800                       # target epoch (resumes from BASE_CKPT)
#   AUG_SAVE_DIR=checkpoints_aug
#   BRATS_DATA_DIR=/path/to/BraTS/data
#   UPENN_NIFTI_DIR=upenn_nifti
#   SKIP_UPENN=1                     # skip step 3 if UPenn data is unavailable
#
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

# ---- config ---------------------------------------------------------------
BASE_CKPT="${BASE_CKPT:-checkpoints/segmentor_epoch_650.pth}"
EPOCHS="${EPOCHS:-800}"
AUG_SAVE_DIR="${AUG_SAVE_DIR:-checkpoints_aug}"
UPENN_NIFTI_DIR="${UPENN_NIFTI_DIR:-upenn_nifti}"
SKIP_UPENN="${SKIP_UPENN:-0}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-logs}"
# Output roots (override to keep a dry-run out of the tracked results/ tree).
BRATS_OUT="${BRATS_OUT:-results/brats}"
UPENN_OUT="${UPENN_OUT:-results/upenn}"
# ALLOW_CPU=1 bypasses the GPU requirement — for a CPU plumbing/dry-run only;
# real training needs a GPU. NO_CACHE=1 forces UPenn inference to recompute.
ALLOW_CPU="${ALLOW_CPU:-0}"
NO_CACHE="${NO_CACHE:-0}"
# PREFLIGHT_ONLY=1 resolves and prints all paths, runs every check, then exits
# BEFORE training — so paths can be confirmed before the long run.
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
mkdir -p "$LOG_DIR" "$AUG_SAVE_DIR" "$BRATS_OUT" "$UPENN_OUT"

# BRATS_DATA_DIR is read by train_brats.py/evaluate_brats.py; default to <repo>/data.
export BRATS_DATA_DIR="${BRATS_DATA_DIR:-$ROOT/data}"

log() { echo -e "\n\033[1m[$(date +%H:%M:%S)] $*\033[0m"; }
die() { echo -e "\n\033[31mERROR: $*\033[0m" >&2; exit 1; }

# ---- step 0: preflight ----------------------------------------------------
log "Step 0/5  Preflight checks"
if [ "$ALLOW_CPU" = "1" ]; then
  echo "  [ALLOW_CPU=1] skipping GPU requirement — CPU plumbing/dry-run only, not a real experiment"
else
  python - <<'PY' || die "GPU not available. This experiment needs CUDA; run it on the GPU machine (or set ALLOW_CPU=1 for a CPU plumbing dry-run)."
import torch, sys
ok = torch.cuda.is_available()
print(f"  CUDA available : {ok}")
if ok:
    print(f"  GPU            : {torch.cuda.get_device_name(0)}")
sys.exit(0 if ok else 1)
PY
fi

[ -f "$BASE_CKPT" ] || die "base checkpoint not found: $BASE_CKPT"
n_brats=$(find "$BRATS_DATA_DIR" -maxdepth 1 -type d -name 'BraTS2021_*' 2>/dev/null | wc -l)
[ "$n_brats" -gt 0 ] || die "no BraTS2021_* folders under BRATS_DATA_DIR=$BRATS_DATA_DIR"
echo "  base checkpoint: $BASE_CKPT"
echo "  BraTS cases    : $n_brats  (under $BRATS_DATA_DIR)"
if [ "$SKIP_UPENN" != "1" ]; then
  n_upenn=$(find "$UPENN_NIFTI_DIR" -maxdepth 1 -name '*_seg.nii.gz' 2>/dev/null | wc -l)
  if [ "$n_upenn" -eq 0 ]; then
    echo "  [warn] no UPenn *_seg.nii.gz under $UPENN_NIFTI_DIR — set SKIP_UPENN=1 to skip step 3, or fix UPENN_NIFTI_DIR"
    die "UPenn data missing; aborting before the long training run so you don't lose step 3"
  fi
  echo "  UPenn subjects : $n_upenn  (under $UPENN_NIFTI_DIR)"
fi

# Resolved absolute paths, for confirmation before the long run.
echo ""
echo "  Resolved paths:"
echo "    1. repository root : $ROOT"
echo "    2. BraTS data      : $(cd "$BRATS_DATA_DIR" 2>/dev/null && pwd || echo "$BRATS_DATA_DIR")  ($n_brats cases)"
if [ "$SKIP_UPENN" != "1" ]; then
  echo "    3. UPenn data      : $(cd "$UPENN_NIFTI_DIR" 2>/dev/null && pwd || echo "$UPENN_NIFTI_DIR")  (${n_upenn:-?} subjects)"
else
  echo "    3. UPenn data      : (SKIP_UPENN=1)"
fi
echo "    4. base checkpoint : $(cd "$(dirname "$BASE_CKPT")" 2>/dev/null && pwd || echo .)/$(basename "$BASE_CKPT")"
echo "    -> train $((EPOCHS)) target epoch, augment=on, norm=batch (default)"
echo "    -> outputs: BraTS=$BRATS_OUT  UPenn=$UPENN_OUT  logs=$LOG_DIR  ckpts=$AUG_SAVE_DIR"

if [ "$PREFLIGHT_ONLY" = "1" ]; then
  log "PREFLIGHT_ONLY=1 — all checks passed, paths resolved above. Not training."
  echo "Re-run without PREFLIGHT_ONLY to start the real experiment."
  exit 0
fi

# ---- step 1: augmented training -------------------------------------------
log "Step 1/5  Augmented training (resume from $BASE_CKPT, target epoch $EPOCHS)"
TRAIN_LOG="$LOG_DIR/train_aug_${STAMP}.log"
python scripts/train_brats.py \
    --augment \
    --epochs "$EPOCHS" \
    --resume "$BASE_CKPT" \
    --save-dir "$AUG_SAVE_DIR" \
    2>&1 | tee "$TRAIN_LOG"

# The augmented checkpoint = the highest-epoch file in AUG_SAVE_DIR.
AUG_CKPT=$(ls -1 "$AUG_SAVE_DIR"/segmentor_epoch_*.pth 2>/dev/null \
    | sort -t_ -k3 -n | tail -1 || true)
[ -n "${AUG_CKPT:-}" ] && [ -f "$AUG_CKPT" ] \
    || die "training produced no checkpoint in $AUG_SAVE_DIR (see $TRAIN_LOG)"
echo "  augmented checkpoint: $AUG_CKPT"

# ---- step 2: BraTS internal-validation evaluation -------------------------
log "Step 2/5  BraTS internal-validation evaluation (baseline + augmented)"
python scripts/evaluate_brats.py --partition internal_validation \
    --checkpoint "$BASE_CKPT" --tag base650 --out-dir "$BRATS_OUT" \
    2>&1 | tee "$LOG_DIR/eval_brats_base_${STAMP}.log"
python scripts/evaluate_brats.py --partition internal_validation \
    --checkpoint "$AUG_CKPT" --tag aug --out-dir "$BRATS_OUT" \
    2>&1 | tee "$LOG_DIR/eval_brats_aug_${STAMP}.log"

# ---- step 3: UPenn zero-shot evaluation -----------------------------------
if [ "$SKIP_UPENN" != "1" ]; then
  log "Step 3/5  UPenn zero-shot evaluation (baseline + augmented)"
  NC=""; [ "$NO_CACHE" = "1" ] && NC="--no-cache"
  python scripts/evaluate_upenn.py \
      --nifti-dir "$UPENN_NIFTI_DIR" \
      --out-dir "$UPENN_OUT" $NC \
      --setups baseline_brats candidate_zeroshot \
      --extra-checkpoint "$AUG_CKPT" --extra-name candidate_zeroshot \
      --extra-preproc brats \
      2>&1 | tee "$LOG_DIR/eval_upenn_zeroshot_${STAMP}.log"
else
  log "Step 3/5  SKIPPED (SKIP_UPENN=1)"
fi

# ---- step 4: per-case CSVs (produced by steps 2-3) ------------------------
log "Step 4/5  Per-case CSVs written:"
ls -1 "$BRATS_OUT"/per_case_base650_recomputed.csv \
      "$BRATS_OUT"/per_case_aug_recomputed.csv 2>/dev/null || true
[ "$SKIP_UPENN" != "1" ] && ls -1 \
      "$UPENN_OUT"/per_case_baseline_brats.csv \
      "$UPENN_OUT"/per_case_candidate_zeroshot.csv 2>/dev/null || true

# ---- step 5: paired A/B comparisons ---------------------------------------
log "Step 5/5  Paired A/B comparison (augmented - baseline)"
echo -e "\n### BraTS internal validation ###"
python scripts/compare_runs.py \
    --baseline  "$BRATS_OUT"/per_case_base650_recomputed.csv \
    --candidate "$BRATS_OUT"/per_case_aug_recomputed.csv \
    --name-baseline epoch650 --name-candidate augmented \
    --out-csv "$BRATS_OUT"/ab_augmentation_${STAMP}.csv \
    2>&1 | tee "$LOG_DIR/ab_brats_${STAMP}.log"

if [ "$SKIP_UPENN" != "1" ]; then
  echo -e "\n### UPenn zero-shot (the generalisation question) ###"
  python scripts/compare_runs.py \
      --baseline  "$UPENN_OUT"/per_case_baseline_brats.csv \
      --candidate "$UPENN_OUT"/per_case_candidate_zeroshot.csv \
      --name-baseline zeroshot650 --name-candidate zeroshotAug \
      --out-csv "$UPENN_OUT"/ab_augmentation_zeroshot_${STAMP}.csv \
      2>&1 | tee "$LOG_DIR/ab_upenn_${STAMP}.log"
fi

# ---- summary --------------------------------------------------------------
log "DONE. Outputs:"
echo "  checkpoint : $AUG_CKPT"
echo "  logs       : $LOG_DIR/*_${STAMP}.log"
echo "  BraTS      : $BRATS_OUT/{summary_base650,summary_aug}_recomputed.csv  (Dice+HD95+CIs)"
echo "               $BRATS_OUT/ab_augmentation_${STAMP}.csv                  (paired Δ)"
if [ "$SKIP_UPENN" != "1" ]; then
echo "  UPenn      : $UPENN_OUT/segmentation_summary.csv                      (Dice+HD95+CIs)"
echo "               $UPENN_OUT/ab_augmentation_zeroshot_${STAMP}.csv         (paired Δ)"
fi
echo ""
echo "The paired-Δ tables above are the headline before/after result. A change"
echo "whose 95% CI excludes zero is credible; note UPenn's 29-case test set"
echo "cannot resolve differences under ~0.05 Dice (confirm with crossval_upenn.py)."
