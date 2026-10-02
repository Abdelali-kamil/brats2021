#!/usr/bin/env bash
# Relaunch the Swin UNETR baseline after the validation-threshold fix.
# Stops a running swin_unetr_frozen job, archives its run dir if it holds
# checkpoints, starts fresh with the identical command.
# Run: bash baselines/swin_unetr/relaunch_fixed.sh
set -euo pipefail
REPO=/home/kamilabdelali/brats2021/baselines/repos/swin_unetr/SwinUNETR/BRATS21
LOGS=/home/kamilabdelali/brats2021/logs
cd "$REPO"
grep -q 'AsDiscrete(argmax=False, threshold=0.5)' main.py || { echo "fix not applied to main.py"; exit 1; }

# DataLoader workers hand MONAI MetaTensors (dozens of small metadata tensors
# each) to the main process as file descriptors. An SSH login shell's soft
# limit of 1024 runs out within a few batches and the job dies with
# "received 0 items of ancdata". Process limit only; training is unchanged.
ulimit -n "$(ulimit -Hn)"

PIDS=$(pgrep -f 'main.py .*--logdir swin_unetr_frozen' || true)
if [ -n "$PIDS" ]; then
  echo "stopping $PIDS"; kill $PIDS
  for i in $(seq 1 60); do pgrep -f 'main.py .*--logdir swin_unetr_frozen' >/dev/null || break; sleep 1; done
fi
if [ -d runs/swin_unetr_frozen ]; then
  if compgen -G 'runs/swin_unetr_frozen/*.pt' >/dev/null; then
    mv runs/swin_unetr_frozen "runs/swin_unetr_frozen_STOPPED_$(date +%Y%m%d_%H%M%S)"
  else
    rm -rf runs/swin_unetr_frozen   # crashed before its first checkpoint
  fi
fi

STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$LOGS/swin_unetr_${STAMP}.log"
echo "launch stamp: $STAMP (relaunch, validation threshold fixed, nofile $(ulimit -n))" > "$LOG"
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 nohup \
  /home/kamilabdelali/brats2021/baselines/swin_unetr/.venv_overlay/bin/python main.py \
  --json_list /home/kamilabdelali/brats2021/baselines/swin_unetr/datalist_frozen_train1000_val126.json \
  --data_dir /mnt/data1/kamil_research/data/brats2021 --fold 0 --feature_size 48 \
  --use_checkpoint --save_checkpoint --max_epochs 300 --val_every 25 --batch_size 1 \
  --workers 8 --logdir swin_unetr_frozen >> "$LOG" 2>&1 &
echo "started PID $! -> $LOG"
