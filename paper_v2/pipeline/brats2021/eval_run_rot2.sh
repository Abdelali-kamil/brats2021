#!/usr/bin/env bash
# PROTOCOL_v2 Amendments 12/15: candidate <tag> (default runC_rot) = frozen <src> weights, its VAL-chosen
# axis mode (both) + 90-degree in-plane rotation TTA. VAL (126) -> Amendment-5 Dice sweep ->
# TEST (125) scored ONCE with the VAL-selected setting. Idempotent; refuses a second scoring.
set -uo pipefail
TAG=${1:-runC_rot}; SRC=${2:-runC}; V=/mnt/data1/kamil_research/experiments/eval_v2; CODE=$V/code
PY=/home/kamilabdelali/anaconda3/envs/torchfix/bin/python; RES=/home/kamilabdelali/brats2021/results/brats
W=$V/work_$TAG; mkdir -p $W; LOG=$V/eval_$TAG.log
log() { echo "[eval $TAG $(date '+%F %T')] $*" | tee -a "$LOG"; }; die() { log "ABORT: $*"; exit 1; }
[ -e "$RES/summary_${TAG}_test_tuned_recomputed.csv" ] && die "test already scored for $TAG"
FROZEN=$(cut -d' ' -f5 $V/work_$SRC/selection.txt); ORIENT=$(cut -d' ' -f7 $V/work_$SRC/selection.txt)
[ -f "$FROZEN" ] || die "no frozen $SRC checkpoint"
log "weights $FROZEN sha256 $(sha256sum "$FROZEN" | cut -c1-16); axis-order $ORIENT + rotation TTA"
wait_gpu() { for i in $(seq 1 288); do f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
  [ -n "$f" ] && [ "$f" -ge 7000 ] && return 0; sleep 300; done; die "GPU not available"; }
if [ "$(ls $W/probs_val/*.npz 2>/dev/null | wc -l)" -ne 126 ]; then
  wait_gpu; log "[1/4] validation (126)"
  $PY $CODE/scripts/evaluate_brats.py --partition internal_validation --cases $CODE/baselines/common/val_ids.txt \
      --checkpoint "$FROZEN" --axis-order $ORIENT --rot-tta --save-probs $W/probs_val --tag ${TAG}_val >> $W/val.log 2>&1 || die "val failed"
fi
log "[1/4] $(grep -E '^MEAN' $W/val.log | tail -1)"
if ! grep -q '^WINNER' $W/sweep.log 2>/dev/null; then
  log "[2/4] Dice sweep on VAL (30 settings)"
  $PY $CODE/baselines/common/sweep_postproc_dice.py --probs $W/probs_val --tag ${TAG}_val > $W/sweep.log 2>&1 || die "sweep failed"
fi
read -r _ WT MULT ETMIN VALDICE <<< "$(grep '^WINNER' $W/sweep.log | tail -1)"
[ -n "${WT:-}" ] || die "selection produced nothing"
if [ "$ETMIN" = 0 ]; then ETARGS=(--et-policy none); else ETARGS=(--et-policy min_volume --et-min-volume "$ETMIN"); fi
log "[3/4] val-selected: wt_policy=$WT floor_mult=$MULT et_min_volume=$ETMIN -> val mean Dice $VALDICE"
echo "$TAG $VALDICE $WT $MULT $FROZEN $ETMIN ${ORIENT}+rot" > $W/selection.txt
[ -e "$RES/summary_${TAG}_test_tuned_recomputed.csv" ] && die "test already scored for $TAG"
wait_gpu; log "[4/4] TEST (125), once"
$PY $CODE/scripts/evaluate_brats.py --partition internal_validation --cases $CODE/baselines/common/test_ids.txt \
    --checkpoint "$FROZEN" --axis-order $ORIENT --rot-tta --wt-policy $WT --floor-mult $MULT "${ETARGS[@]}" \
    --save-probs $W/probs_test --tag ${TAG}_test_tuned >> $W/test.log 2>&1 || die "test failed"
log "[4/4] $(grep -E '^MEAN' $W/test.log | tail -1)"; log "=== done ==="
