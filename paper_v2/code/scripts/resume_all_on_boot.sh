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

# 2. NOT resuming BraTS training. Stopped deliberately at epoch 203: the
#    original run peaked at 195 and got worse after, this run is already past
#    that point, and another ~8 h run on a card that faults every 8-13 h buys
#    nothing. finish_fast.sh evaluates what we have and starts nnU-Net itself.
#    To go back to training instead: run scripts/resume_repro.sh by hand.
if pgrep -f "finish_fast" >/dev/null 2>&1; then
  log "finish_fast already running; leaving it alone"
else
  log "starting finish_fast (evaluate epoch 203, tune post-processing, then nnU-Net)"
  setsid nohup bash "$ROOT/scripts/finish_fast.sh" \
      >> "$ROOT/logs/finish_boot_$(date +%Y%m%d_%H%M).log" 2>&1 < /dev/null &
fi

# 3. nnU-Net is started by finish_fast.sh once the evaluations are done,
#    so it is not launched here -- two launches would collide.

log "boot resume dispatched"
