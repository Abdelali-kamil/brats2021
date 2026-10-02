#!/usr/bin/env bash
# Overnight queue: wait for the running evaluation to finish, then train nnU-Net
# alongside the aug_cosine_repro1 run.
#
# Why wait rather than launch immediately: the repro training holds ~12.5 GB and
# the evaluation ~5.3 GB of the card's 32.6 GB. nnU-Net 3d_fullres needs roughly
# 11-14 GB, which would not fit beside both. The evaluation finishes in ~2 h, so
# queueing costs almost nothing against a 2-3 day nnU-Net run.
#
# Fold "all" trains on all 1000 cases -- exactly the repro model's training set,
# one model against one model. Fold 0 would use only 800, since nnU-Net carves
# its own internal 5-fold split.
#
#   bash baselines/nnunet/run_overnight.sh
set -uo pipefail

ROOT=/home/kamilabdelali/brats2021
LOG="$ROOT/logs/nnunet_train_all.log"
QLOG="$ROOT/logs/nnunet_queue.log"
MIN_FREE_MB=14000        # refuse to start without real headroom

log() { echo "[queue $(date '+%F %T')] $*" | tee -a "$QLOG"; }

free_mb() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1
}

# 0. Never start a second trainer. finish_fast.sh also calls this script, so
#    without this guard an early manual launch and the queued one would both
#    train the same fold, corrupting each other's checkpoints.
if pgrep -f "nnUNetv2_train" >/dev/null 2>&1; then
  log "nnU-Net is already training; nothing to do"
  exit 0
fi

# 1. Wait for any evaluation to finish.
# An evaluation needs only ~5.5 GB, so it can share the card rather than
# block a 27 h training run. Only wait if the card is genuinely full; the
# free-memory check below is the real gate.
if pgrep -f "evaluate_brats" >/dev/null 2>&1; then
  log "an evaluation is running; sharing the card (it needs ~5.5 GB)"
else
  log "no evaluation running"
fi

# 2. Wait until the card actually has room.
for _ in $(seq 1 60); do
  f=$(free_mb)
  [ -n "$f" ] && [ "$f" -ge "$MIN_FREE_MB" ] && break
  log "only ${f:-?} MB free, need ${MIN_FREE_MB}; waiting"
  sleep 120
done
f=$(free_mb)
if [ -z "$f" ] || [ "$f" -lt "$MIN_FREE_MB" ]; then
  log "gave up: ${f:-?} MB free, need ${MIN_FREE_MB}. nnU-Net NOT started."
  exit 1
fi
log "${f} MB free -- starting nnU-Net"

# 3. Train.
source "$ROOT/baselines/nnunet/env.sh"
# Keep dataloader workers modest: the repro run already holds 8, and this
# machine's two GPU faults both struck under sustained heavy load.
export nnUNet_n_proc_DA=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --c continues from checkpoint_latest.pth when one exists. Without it a
# restart would throw away every epoch already trained (203 of 1000 as of
# the 2026-09-17 fault).
CKPT="$nnUNet_results/Dataset137_BraTS2021/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_all/checkpoint_latest.pth"
CONT=""
if [ -f "$CKPT" ]; then
  CONT="--c"
  log "found checkpoint_latest.pth -- continuing, not restarting"
else
  log "no checkpoint yet -- starting from scratch"
fi

log "nnUNetv2_train 137 3d_fullres all $CONT -> $LOG"
"$ROOT/baselines/nnunet/.venv/bin/nnUNetv2_train" 137 3d_fullres all $CONT >> "$LOG" 2>&1
rc=$?
log "nnU-Net exited rc=$rc"
exit $rc
