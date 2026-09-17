#!/usr/bin/env bash
# Start push_for_089 as soon as finish_fast's evaluations are done.
#
# finish_fast.sh runs three evaluations and then launches nnU-Net itself. This
# waits for the test evaluation to finish and starts the 0.89 attempt, so the
# two training jobs share the card: nnU-Net needs ~11-14 GB and push089 ~12.5 GB
# against 32.6 GB total, which fits.
#
# Sharing is a deliberate trade. Two concurrent jobs is what preceded the
# shortest GPU fault so far (8.3 h vs 11.4 and 13.7), but push089 is only 40
# epochs -- ~3.5 h alone, maybe 5-6 h shared -- so it should finish inside the
# window, and both jobs now resume from checkpoints if it does fault.
#
# Retries push089 up to twice, since a fault here costs hours rather than days.
set -uo pipefail

ROOT=/home/kamilabdelali/brats2021
LOG="$ROOT/logs/chain_089.log"
MAX_RETRIES=2

log() { echo "[chain089 $(date '+%F %T')] $*" | tee -a "$LOG"; }

log "=== waiting for finish_fast evaluations ==="

# finish_fast writes "[3/4] done" once the test evaluation has returned.
for _ in $(seq 1 480); do          # up to 4 h
  grep -q "\[3/4\] done" "$ROOT/logs/finish_fast.log" 2>/dev/null && break
  if ! pgrep -f "finish_fast" >/dev/null 2>&1 \
     && ! grep -q "\[1/4\]" "$ROOT/logs/finish_fast.log" 2>/dev/null; then
    log "finish_fast is not running and never started; aborting"
    exit 1
  fi
  sleep 30
done

if ! grep -q "\[3/4\] done" "$ROOT/logs/finish_fast.log" 2>/dev/null; then
  log "evaluations did not complete within 4 h; starting anyway"
else
  log "evaluations complete"
fi

# Wait for enough free memory beside nnU-Net.
for _ in $(seq 1 60); do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
  [ -n "$free" ] && [ "$free" -ge 13000 ] && break
  log "only ${free:-?} MB free, need 13000; waiting"
  sleep 60
done

attempt=0
while :; do
  attempt=$((attempt + 1))
  log "starting push_for_089 (attempt $attempt/$((MAX_RETRIES + 1)))"
  bash "$ROOT/scripts/push_for_089.sh" >> "$ROOT/logs/push089_run.log" 2>&1
  rc=$?
  log "push_for_089 exited rc=$rc"

  # Back up whatever it reached, crash or not.
  mkdir -p /home/kamilabdelali/checkpoint_backups/push089_v1
  cp -f /mnt/data1/kamil_research/experiments/push089_v1/checkpoints/*.pth \
        /home/kamilabdelali/checkpoint_backups/push089_v1/ 2>/dev/null
  cp -f /mnt/data1/kamil_research/experiments/push089_v1/logs/*.log \
        /home/kamilabdelali/checkpoint_backups/push089_v1/ 2>/dev/null

  [ "$rc" -eq 0 ] && { log "finished cleanly"; break; }
  [ "$attempt" -gt "$MAX_RETRIES" ] && { log "retry limit reached"; break; }

  if ! nvidia-smi -L >/dev/null 2>&1; then
    log "GPU wedged again -- needs a reboot; the @reboot cron will not rerun this."
    break
  fi
  sleep 60
done

log "=== done ==="
