#!/usr/bin/env bash
# Bring both GPU jobs back after a reboot, without anyone having to log in.
#
# Why this exists: this machine's GPU has now faulted three times
# (2026-09-05, 09-16, 09-17), each needing a reboot. The nohup watchdog does not
# survive a reboot, so on 09-16 the card came back at 18:02 and sat idle until
# 22:04 -- four hours lost because nothing restarted it. A @reboot cron entry
# does survive, and needs no root (systemd user services would not, since
# linger is off for this account).
#
# Install (already done, but this is how):
#   (crontab -l 2>/dev/null; echo "@reboot /home/kamilabdelali/brats2021/scripts/resume_all_on_boot.sh") | crontab -
#
# Remove:
#   crontab -l | grep -v resume_all_on_boot | crontab -
set -uo pipefail

ROOT=/home/kamilabdelali/brats2021
LOG="$ROOT/logs/boot_resume.log"
mkdir -p "$ROOT/logs"

log() { echo "[boot $(date '+%F %T')] $*" >> "$LOG"; }

log "=== boot resume starting (uptime $(uptime -p 2>/dev/null)) ==="

# Cron's environment is minimal; nothing here may rely on a login shell.
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# 1. Wait for the driver to be ready. After a reboot that follows a driver
#    upgrade, nvidia-smi can fail for a while until the new module loads.
for i in $(seq 1 60); do
  if nvidia-smi -L >/dev/null 2>&1; then
    log "GPU ready after ${i} check(s): $(nvidia-smi -L 2>/dev/null | head -1)"
    break
  fi
  sleep 30
done
if ! nvidia-smi -L >/dev/null 2>&1; then
  log "GPU still unavailable after 30 min. Nothing started."
  exit 1
fi

# 2. BraTS repro first -- it is the paper's critical path and gets the GPU
#    to itself until it has settled.
if pgrep -f "scripts/train_brats" >/dev/null 2>&1; then
  log "BraTS training already running; leaving it alone"
else
  log "starting BraTS resume watchdog"
  setsid nohup bash "$ROOT/scripts/resume_repro.sh" \
      >> "$ROOT/logs/resume_boot_$(date +%Y%m%d_%H%M).log" 2>&1 < /dev/null &
fi

# 3. nnU-Net second, once there is real headroom. run_overnight.sh does its own
#    waiting and free-memory check, and continues from checkpoint_latest.
if pgrep -f "nnUNetv2_train" >/dev/null 2>&1; then
  log "nnU-Net already running; leaving it alone"
else
  log "queueing nnU-Net (continue from latest checkpoint)"
  setsid nohup bash "$ROOT/baselines/nnunet/run_overnight.sh" \
      >> "$ROOT/logs/nnunet_boot_$(date +%Y%m%d_%H%M).log" 2>&1 < /dev/null &
fi

log "boot resume dispatched"
