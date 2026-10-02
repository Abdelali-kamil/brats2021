#!/usr/bin/env bash
# PROTOCOL_v2 per-model evaluation, for any of our checkpoints:
#   1. full VAL evaluation (126), probabilities saved
#   2. post-processing sweep on VAL
#   3. setting chosen by the fixed rule (lowest mean HD95 within 0.002 Dice of
#      the as-shipped setting) -- identical to eval_ep253/run_eval_ep253.sh
#   4. TEST (125) scored ONCE with that setting; refuses a second scoring
# Architecture (2D/3D DWT, deep supervision) is read from the checkpoint.
#
#   bash eval_run.sh <tag> <checkpoint>      e.g. bash eval_run.sh runA runA_ds400_best.pth
set -uo pipefail
TAG=$1; CKPT=$2
V=/mnt/data1/kamil_research/experiments/eval_v2
CODE=$V/code
PY=/home/kamilabdelali/anaconda3/envs/torchfix/bin/python
RES=/home/kamilabdelali/brats2021/results/brats
W=$V/work_$TAG; mkdir -p "$W"
LOG=$V/eval_$TAG.log
log() { echo "[eval $TAG $(date '+%F %T')] $*" | tee -a "$LOG"; }
die() { log "ABORT: $*"; exit 1; }
cd "$V" || exit 1
[ -f "$CKPT" ] || die "no checkpoint $CKPT"
[ -e "$RES/summary_${TAG}_test_tuned_recomputed.csv" ] && die "test already scored for $TAG"
# Freeze the exact weights being evaluated.
FROZEN=$W/$(basename "$CKPT" .pth)_evaluated.pth
[ -f "$FROZEN" ] || cp -p "$CKPT" "$FROZEN"
chmod a-w "$FROZEN"
log "checkpoint $CKPT -> frozen $FROZEN sha256 $(sha256sum "$FROZEN" | cut -c1-16)"

wait_gpu() { for i in $(seq 1 288); do
  f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
  [ -n "$f" ] && [ "$f" -ge 7000 ] && return 0; sleep 300; done; die "GPU not available"; }

# Amendment 10: VAL in two inference modes (nib order, and orientation TTA "both"),
# each with the Amendment-5 Dice sweep; the higher VAL Dice decides the mode.
for O in nib both; do
  PV="$W/probs_val_$O"
  n=$(ls "$PV"/*.npz 2>/dev/null | wc -l)
  if [ "$n" -ne 126 ]; then
    wait_gpu; log "[1/4] validation (126, full protocol, axis-order $O)"
    "$PY" "$CODE/scripts/evaluate_brats.py" --partition internal_validation \
        --cases "$CODE/baselines/common/val_ids.txt" --checkpoint "$FROZEN" --axis-order $O \
        --save-probs "$PV" --tag ${TAG}_val_$O >> "$W/val_$O.log" 2>&1 || die "val eval ($O) failed"
    n=$(ls "$PV"/*.npz 2>/dev/null | wc -l); [ "$n" -eq 126 ] || die "expected 126 val maps ($O), got $n"
  fi
  log "[1/4] $O: $(grep -E '^MEAN' "$W/val_$O.log" | tail -1)"
  if ! grep -q '^WINNER' "$W/sweep_$O.log" 2>/dev/null; then
    log "[2/4] post-processing sweep on validation ($O, Dice objective, 30 settings)"
    "$PY" "$CODE/baselines/common/sweep_postproc_dice.py" --probs "$PV" \
        --tag ${TAG}_val_$O > "$W/sweep_$O.log" 2>&1 || die "sweep ($O) failed"
  fi
done
read -r _ PN MN EN DN <<< "$(grep '^WINNER' "$W/sweep_nib.log" | tail -1)"
read -r _ PB MB EB DB <<< "$(grep '^WINNER' "$W/sweep_both.log" | tail -1)"
if python3 -c "import sys; sys.exit(0 if float('$DB') > float('$DN') else 1)"; then
  ORIENT=both; WT_POLICY=$PB; FLOOR_MULT=$MB; ET_MIN=$EB; VALDICE=$DB
else
  ORIENT=nib;  WT_POLICY=$PN; FLOOR_MULT=$MN; ET_MIN=$EN; VALDICE=$DN; fi
[ -n "${WT_POLICY:-}" ] || die "selection produced nothing"
ln -sfn "$W/probs_val_$ORIENT" "$W/probs_val"
if [ "$ET_MIN" = 0 ]; then ETARGS=(--et-policy none); else ETARGS=(--et-policy min_volume --et-min-volume "$ET_MIN"); fi
log "[3/4] val-selected: axis-order=$ORIENT (nib $DN vs both $DB) wt_policy=$WT_POLICY floor_mult=$FLOOR_MULT et_min_volume=$ET_MIN -> val mean Dice $VALDICE"
# fields: tag val_dice wt_policy floor_mult frozen_ckpt et_min_volume axis_order
echo "$TAG $VALDICE $WT_POLICY $FLOOR_MULT $FROZEN $ET_MIN $ORIENT" > "$W/selection.txt"

[ -e "$RES/summary_${TAG}_test_tuned_recomputed.csv" ] && die "test already scored for $TAG"
wait_gpu; log "[4/4] TEST (125), once"
"$PY" "$CODE/scripts/evaluate_brats.py" --partition internal_validation \
    --cases "$CODE/baselines/common/test_ids.txt" --checkpoint "$FROZEN" \
    --wt-policy "$WT_POLICY" --floor-mult "$FLOOR_MULT" "${ETARGS[@]}" --axis-order "$ORIENT" --save-probs "$W/probs_test" \
    --tag ${TAG}_test_tuned >> "$W/test.log" 2>&1 || die "test eval failed"
log "[4/4] $(grep -E '^MEAN' "$W/test.log" | tail -1)"
log "=== done; results in $RES/*${TAG}* ==="
