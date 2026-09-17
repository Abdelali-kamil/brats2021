#!/usr/bin/env bash
# Finish aug_cosine_repro1 to the full 300 epochs, after push089 has run.
#
# Why finish it at all, given the run has almost certainly plateaued (the
# original peaked at epoch 195 and this one is at 203/0.8849): the paper
# describes a 300-epoch cosine schedule, and a reproduction that stops at 203 is
# not a reproduction of that protocol. Fidelity is the reason, not the score.
#
# Why after push089 rather than alongside: three concurrent jobs do not fit.
# repro ~12.5 GB + nnU-Net ~13 GB + push089 ~12.5 GB is ~38 GB against 32.6 GB.
# push089 is the shorter run and answers the open question, so it goes first.
#
# 97 epochs at ~5.2 min is ~8.4 h alone, longer sharing the card with nnU-Net.
# That exceeds the window this GPU keeps faulting in, so resume_repro.sh's
# watchdog matters here: it restarts from the last checkpoint rather than
# losing the run.
set -uo pipefail

ROOT=/home/kamilabdelali/brats2021
LOG="$ROOT/logs/chain_to_300.log"
log() { echo "[to300 $(date '+%F %T')] $*" | tee -a "$LOG"; }

log "=== waiting for push089 to finish ==="

# Wait for chain_089 to report a terminal state.
for _ in $(seq 1 960); do          # up to 8 h
  grep -qE "finished cleanly|retry limit reached|GPU wedged again" \
       "$ROOT/logs/chain_089.log" 2>/dev/null && break
  sleep 30
done
log "push089 stage over: $(tail -n 1 "$ROOT/logs/chain_089.log" 2>/dev/null)"

# Only proceed while the card is actually alive.
if ! nvidia-smi -L >/dev/null 2>&1; then
  log "GPU is down; not starting. The @reboot cron runs finish_fast, not this."
  exit 1
fi

EP=$(/home/kamilabdelali/anaconda3/envs/torchfix/bin/python -c "
import torch
p='/mnt/data1/kamil_research/experiments/aug_cosine_repro1/checkpoints/aug_cosine_repro1_last.pth'
print(torch.load(p, map_location='cpu', weights_only=False)['epoch'])" 2>/dev/null)
log "resuming aug_cosine_repro1 from epoch ${EP:-?} -> 300"

if [ "${EP:-0}" -ge 300 ]; then
  log "already at 300; nothing to do"
  exit 0
fi

# resume_repro.sh carries the scheduler fix (continues the original cosine
# rather than resetting the LR) and its own crash watchdog.
exec bash "$ROOT/scripts/resume_repro.sh"
