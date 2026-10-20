#!/bin/bash
# Amendment 24: TransBTS training on our fixed 1,000-case TRAIN split (official recipe; single GPU = batch 2 x 4 accumulation).
# Waits for the GPU, runs a 1-epoch smoke test on 8 cases, then the full 1000-epoch run with automatic resume (max 3 retries).
set -uo pipefail
W=/mnt/data1/kamil_research/experiments/transbts; PY=$W/.venv/bin/python; LOG=$W/train_wrapper.log
exec 9> $W/.lock_train; flock -n 9 || exit 0
log() { echo "[transbts $(date '+%F %T')] $*" | tee -a $LOG; }
[ -e $W/TRAINED ] && exit 0
while pgrep -f "[e]val_ensemble3_external.py" > /dev/null; do sleep 60; done
# Amendment 25: baseline efficiency benchmarks need the idle GPU first
until [ -e /mnt/data1/kamil_research/experiments/efficiency/BASELINES_DONE ]; do sleep 120; done
# (2026-10-14 03:06, user: 'use full server, save time') runs alongside the UNETR CV; only the free-memory check below gates it
until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge 24000 ]; do sleep 120; done
cd $W/code
run() {   # run <experiment> <train list> <end_epoch> [resume ckpt]
  local R=(); [ -n "${4:-}" ] && R=(--resume "$4")
  TRANSBTS_NPY_CACHE=$W/npy_cache $PY -m torch.distributed.run --nproc_per_node=1 --master_port 20013 train.py --root /mnt/data1/kamil_research/data \
    --train_dir brats2021 --train_file "$2" --mode train --gpu 0 --batch_size 2 --accum_steps 4 --num_workers 12 \
    --end_epoch "$3" --save_freq 100 --date 20261014 --experiment "$1" --user transbts --criterion softmax_dice2 "${R[@]}"   # addendum 2: background-inclusive loss
}
if [ ! -e $W/SMOKE_OK ]; then
  log "smoke: 4 epochs on 8 cases (end_epoch>=4: the checkpoint rule divides by end_epoch-1..3)"
  if run smoke_ $W/smoke_train.txt 4 > $W/smoke.log 2>&1 && [ -s checkpoint/smoke_20261014/model_epoch_last.pth ]; then
    log "smoke OK: $(grep -o 'Current epoch time consumption: [0-9.]* minutes' $W/smoke.log | tail -1)"; touch $W/SMOKE_OK
  else log "SMOKE FAILED (see smoke.log)"; touch $W/FAILED; exit 1; fi
fi
CK=checkpoint/TransBTS_dice2_20261014
for att in 1 2 3 4; do
  RES=""; [ -s $CK/model_last_resume.pth ] && RES=$CK/model_last_resume.pth
  log "attempt $att ${RES:+(resume $RES)}"
  run TransBTS_dice2_ $W/train.txt 1000 $RES >> $W/train_$(date +%Y%m%d_%H%M%S).log 2>&1
  if [ -s $CK/model_epoch_last.pth ]; then log "TRAINED: $CK/model_epoch_last.pth"; echo "$W/code/$CK" > $W/RUN_PATH; touch $W/TRAINED; exit 0; fi
  log "attempt $att ended without final model; retry in 120 s"; sleep 120
done
log "FAILED after retries"; touch $W/FAILED
