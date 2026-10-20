#!/bin/bash
# Amendment 22: Henry et al. (BraTS 2020) deeply supervised 3D U-Net, pipeline A "U-Net like", official code, our split.
# The repository cannot resume mid-run; a failed run is moved to ~/delete/unused_checkpoints and restarted (max 3 attempts).
set -uo pipefail
E=/mnt/data1/kamil_research/experiments/henry_dsunet
PY=$E/.venv/bin/python; LOG=$E/train_wrapper.log
exec 9> $E/.lock_train; flock -n 9 || exit 0
[ -e $E/TRAINED ] && exit 0
export FIXED_SPLIT_CSV=/home/kamilabdelali/brats2021/baselines/common/brats_split_frozen.csv
export BRATS_FOLDERS=/mnt/data1/kamil_research/data/brats2021 PYTHONPATH=$E/ranger_pkg:$E/code PYTHONUNBUFFERED=1
cd $E/code; ulimit -n "$(ulimit -Hn)"
for att in 1 2 3; do
  until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge 12000 ]; do sleep 300; done
  [ -d runs ] && { mkdir -p ~/delete/unused_checkpoints; mv runs ~/delete/unused_checkpoints/henry_runs_failed_$(date +%Y%m%d_%H%M%S); }
  L=$E/train_$(date +%Y%m%d_%H%M%S).log
  echo "[henry $(date '+%F %T')] attempt $att -> $L" >> $LOG
  FP=(); [ -e $E/USE_FP32 ] && FP=(--no_fp16)
  $PY -m src.train --devices 0 --width 48 --arch EquiUnet --deep_sup --swa -j 8 "${FP[@]}" > $L 2>&1; rc=$?
  LASTVAL=$(grep "Val : \[" $L | tail -1 | grep -o "[0-9]\.[0-9]*" | awk '{s+=$1;n++} END{if(n) print s/n; else print 0}')
  if [ $rc -eq 0 ] && python3 -c "import sys; sys.exit(0 if float('$LASTVAL') < 0.5 else 1)"; then
    echo "[henry $(date '+%F %T')] collapse rule: last validation mean $LASTVAL < 0.5" >> $LOG
    [ -e $E/USE_FP32 ] && { touch $E/FAILED; exit 1; }; touch $E/USE_FP32; continue; fi
  RUN=$(ls -d runs/*EquiUnet_48* 2>/dev/null | head -1)
  if [ $rc -eq 0 ] && [ -n "$RUN" ] && [ -f "$RUN/model_best.pth.tar" ] && grep -q "SWA Model initialised" $L; then
    echo "$RUN" > $E/RUN_DIR; touch $E/TRAINED; echo "[henry $(date '+%F %T')] trained: $RUN" >> $LOG; exit 0
  fi
  echo "[henry $(date '+%F %T')] attempt $att failed rc=$rc: $(tail -2 $L | tr '\n' ' ' | cut -c1-200)" >> $LOG; sleep 120
done
touch $E/FAILED
