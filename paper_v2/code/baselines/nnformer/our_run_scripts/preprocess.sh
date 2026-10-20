#!/bin/bash
# Amendment 26: nnFormer's own planning + preprocessing (CPU only) of Task003_tumor (TRAIN+VAL; TEST is not in imagesTr).
# Waits for the baseline efficiency benchmarks (their timing includes CPU preprocessing). Idempotent; flock.
W=/mnt/data1/kamil_research/experiments/nnformer; source $W/env.sh; LOG=$W/preprocess.log
exec 9> $W/.lock_pre; flock -n 9 || exit 0
[ -e $W/PREPROCESSED ] && exit 0
until [ -e /mnt/data1/kamil_research/experiments/efficiency/BASELINES_DONE ]; do sleep 120; done
echo "[nnformer-pre $(date '+%F %T')] start" >> $LOG
cd $W && nnFormer_plan_and_preprocess -t 3 -tl 8 -tf 8 >> $LOG 2>&1 \
  && [ "$(ls $nnFormer_preprocessed/Task003_tumor/nnFormerData_plans_v2.1_stage0/*.npz | wc -l)" -eq 1126 ] \
  && { echo "[nnformer-pre $(date '+%F %T')] done: 1126 cases" >> $LOG; touch $W/PREPROCESSED; } \
  || echo "[nnformer-pre $(date '+%F %T')] FAILED" >> $LOG
