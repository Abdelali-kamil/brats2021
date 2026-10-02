#!/usr/bin/env bash
# Resume aug_cosine_repro1 after the GPU fault of 2026-09-16 02:01.
#
# The run died at epoch 133/300 with `CUDA error: unspecified launch failure`
# and the card wedged -- the same fault that killed aug_cosine_v1 at epoch 205
# on 2026-09-05. Recovery needs a reboot (root). This script waits for the GPU
# to come back, resumes from the last checkpoint, and restarts the run if the
# fault happens a third time.
#
#   bash scripts/resume_repro.sh              # wait for GPU, then resume
#   MAX_RETRIES=0 bash scripts/resume_repro.sh   # no watchdog, single attempt
#
# Safe to run while the GPU is still down: it waits rather than failing.
set -uo pipefail

EXP=/mnt/data1/kamil_research/experiments/aug_cosine_repro1
REPO=/mnt/data1/kamil_research/experiments/repo_aug_cosine
BACKUP=/home/kamilabdelali/checkpoint_backups/aug_cosine_repro1
PY=/home/kamilabdelali/anaconda3/envs/torchfix/bin/python
CKPT="$EXP/checkpoints/aug_cosine_repro1_last.pth"
LOG="$EXP/logs/aug_cosine_repro1.log"

MAX_RETRIES="${MAX_RETRIES:-3}"
GPU_WAIT_SECS="${GPU_WAIT_SECS:-86400}"   # give the reboot up to 24h to happen

log() { echo "[resume $(date '+%F %T')] $*"; }

gpu_ok() { nvidia-smi -L >/dev/null 2>&1; }

wait_for_gpu() {
  local waited=0
  if gpu_ok; then log "GPU is available."; return 0; fi
  log "GPU is down. Waiting for the reboot (checking every 60s, up to $((GPU_WAIT_SECS/3600))h)..."
  while ! gpu_ok; do
    sleep 60
    waited=$((waited + 60))
    if [ "$waited" -ge "$GPU_WAIT_SECS" ]; then
      log "GPU still down after $((waited/3600))h. Giving up."
      return 1
    fi
    [ $((waited % 600)) -eq 0 ] && log "  still waiting ($((waited/60)) min)"
  done
  log "GPU came back after $((waited/60)) min."
}

backup() {
  mkdir -p "$BACKUP"
  cp -f "$EXP"/checkpoints/*.pth "$BACKUP"/ 2>/dev/null
  cp -f "$LOG" "$BACKUP"/ 2>/dev/null
  log "checkpoints backed up to $BACKUP"
}

run_once() {
  local epoch
  epoch=$("$PY" -c "
import torch; print(torch.load('$CKPT', map_location='cpu', weights_only=False)['epoch'])
" 2>/dev/null)
  if [ -z "$epoch" ]; then log "cannot read epoch from $CKPT -- aborting"; return 2; fi
  log "resuming from epoch $epoch -> 300"

  cd "$REPO" || return 2
  # --start-epoch is NOT passed: the checkpoint's own epoch drives it, and
  # T_max now spans the full 300 either way. Passing it would only risk
  # disagreeing with the checkpoint.
  "$PY" scripts/train_brats.py \
      --epochs 300 --tag aug_cosine_repro1 \
      --num-workers 8 --batch-size 2 --accum-steps 4 --scheduler cosine \
      --save-dir "$EXP/checkpoints" \
      --resume "$CKPT" \
      >> "$LOG" 2>&1
  return $?
}

main() {
  wait_for_gpu || exit 1
  backup

  local attempt=0
  while :; do
    run_once
    local rc=$?
    if [ "$rc" -eq 0 ]; then
      log "training finished cleanly."
      backup
      exit 0
    fi

    backup   # save whatever the crashed run had reached
    attempt=$((attempt + 1))
    log "run exited rc=$rc (attempt $attempt/$MAX_RETRIES)"

    if [ "$attempt" -ge "$MAX_RETRIES" ]; then
      log "retry limit reached. Stopping."
      exit "$rc"
    fi

    if ! gpu_ok; then
      # The GPU wedged again. No amount of retrying fixes that without a
      # reboot, so wait for one rather than spinning.
      log "GPU wedged again -- this needs another reboot (root)."
      wait_for_gpu || exit 1
    else
      log "GPU still healthy; retrying in 60s."
      sleep 60
    fi
  done
}

main "$@"
