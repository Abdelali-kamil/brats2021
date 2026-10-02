#!/bin/bash
# ============================================================================
# Honest multi-seed UPenn-GBM fine-tune + ensemble push.
#
# Produces the strongest DEFENSIBLE UPenn held-out test number: N independent
# fine-tunes from the surviving BraTS base, probability-averaged, with
# post-processing selected on the UPenn val split and the 29-subject test set
# scored exactly once. It reports whatever Dice that yields -- it does not, and
# cannot, target a preset number.
#
# PREREQUISITE (the reason this is not already running): the UPenn-GBM imaging
# data must be restored to $NIFTI_DIR as the 4-channel volumes + ET/TC/WT masks
# the UPennDataset expects (subjects sub-XXX). None is on the machine right now.
#
# Usage:  bash scripts/run_upenn_push.sh            # 3 seeds, waits for data+GPU
#         N_SEEDS=4 bash scripts/run_upenn_push.sh
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
PY=/home/kamilabdelali/anaconda3/envs/torchfix/bin/python

BASE_CKPT="${UPENN_BASE_CKPT:-/mnt/data1/kamil_research/experiments/aug_cosine_repro1/checkpoints/aug_cosine_repro1_best.pth}"
NIFTI_DIR="${NIFTI_DIR:-$ROOT/upenn_nifti}"
N_SEEDS="${N_SEEDS:-3}"
EPOCHS="${EPOCHS:-80}"
SEEDS="${SEEDS:-1 2 3 4 5}"
OUT_CKPT="$ROOT/checkpoints/upenn_push"
OUT_RES="$ROOT/results/upenn_push"
BACKUP="/home/kamilabdelali/checkpoint_backups/upenn_push"
FREE_MIB="${FREE_MIB:-14000}"     # require this many MiB free before starting a seed
mkdir -p "$OUT_CKPT" "$OUT_RES" "$BACKUP"
LOG="$OUT_RES/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "== UPenn push =="
echo "base   : $BASE_CKPT"
echo "data   : $NIFTI_DIR"
echo "seeds  : first $N_SEEDS of [$SEEDS], epochs=$EPOCHS"
echo "log    : $LOG"

[ -f "$BASE_CKPT" ] || { echo "FATAL: base checkpoint missing: $BASE_CKPT"; exit 1; }

wait_for_data() {
  while [ ! -d "$NIFTI_DIR" ] || [ -z "$(ls -A "$NIFTI_DIR" 2>/dev/null)" ]; do
    echo "[wait] UPenn data not present at $NIFTI_DIR -- restore it to start ($(date -Is))"
    sleep 300
  done
  echo "[ok] data present at $NIFTI_DIR"
}
wait_for_gpu() {
  while :; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)
    [ "${free:-0}" -ge "$FREE_MIB" ] && { echo "[ok] GPU free=${free}MiB >= ${FREE_MIB}"; break; }
    echo "[wait] GPU free=${free}MiB < ${FREE_MIB} (other job running) $(date -Is)"
    sleep 300
  done
}

wait_for_data
picked=""
i=0
for s in $SEEDS; do
  [ "$i" -ge "$N_SEEDS" ] && break
  i=$((i+1))
  sd="$OUT_CKPT/seed$s"
  best="$sd/upenn_v3_best.pth"
  if [ -f "$best" ]; then echo "[skip] seed $s already trained -> $best"; picked="$picked $best"; continue; fi
  wait_for_gpu
  echo "== seed $s ($i/$N_SEEDS) =="
  mkdir -p "$sd"
  "$PY" -u scripts/train_upenn.py \
      --seed "$s" --resume-from "$BASE_CKPT" \
      --epochs "$EPOCHS" --save-dir "$sd" --nifti-dir "$NIFTI_DIR" \
      || { echo "seed $s FAILED (exit $?); continuing with the rest"; continue; }
  cp -p "$best" "$BACKUP/seed${s}_upenn_v3_best.pth" 2>/dev/null || true
  picked="$picked $best"
done

picked="$(echo $picked | xargs -n1 2>/dev/null | sort -u | xargs)"
echo "== ensemble over:$picked =="
[ -n "$picked" ] || { echo "FATAL: no seed checkpoints produced"; exit 1; }

"$PY" -u scripts/evaluate_upenn.py \
    --ensemble-ckpts $picked \
    --nifti-dir "$NIFTI_DIR" \
    --out-dir "$OUT_RES"

echo "== DONE. Held-out TEST result (ensemble_seeds) in $OUT_RES/segmentation_summary.csv =="
grep -E "ensemble_seeds.*MEAN" "$OUT_RES"/segmentation_summary*.csv 2>/dev/null || true
