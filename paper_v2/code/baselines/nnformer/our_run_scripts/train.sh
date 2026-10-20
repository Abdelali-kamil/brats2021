#!/bin/bash
# Amendment 26: nnFormer training (authors' tumour trainer, 1000 epochs, our TRAIN/VAL split via NNFORMER_FIXED_SPLIT).
# Fixed GPU order: after preprocessing AND after TransBTS's training + evaluation. Resumes with -c (max 3 retries). flock.
W=/mnt/data1/kamil_research/experiments/nnformer; source $W/env.sh; LOG=$W/train_wrapper.log
exec 9> $W/.lock_train; flock -n 9 || exit 0
log() { echo "[nnformer $(date '+%F %T')] $*" | tee -a $LOG; }
[ -e $W/TRAINED ] && exit 0
until [ -e $W/PREPROCESSED ]; do sleep 300; done
until [ -e /mnt/data1/kamil_research/experiments/transbts/EVALUATED ] || [ -e $W/START_NOW ]; do sleep 300; done   # START_NOW = user OK to run in parallel
until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge 10000 ]; do sleep 120; done
OUT=$RESULTS_FOLDER/nnFormer/3d_fullres/Task003_tumor/nnFormerTrainerV2_nnformer_tumor__nnFormerPlansv2.1/fold_0
for att in 1 2 3 4; do
  C=""; [ -s $OUT/model_latest.model ] && C="-c"
  log "attempt $att ${C:+(continue)}"
  cd $W && CUDA_VISIBLE_DEVICES=0 nnFormer_train 3d_fullres nnFormerTrainerV2_nnformer_tumor 3 0 $C >> $W/train_$(date +%Y%m%d_%H%M%S).log 2>&1
  if [ -s $OUT/model_final_checkpoint.model ] && [ -s $OUT/model_best.model ]; then log "TRAINED: $OUT"; echo $OUT > $W/RUN_PATH; touch $W/TRAINED; exit 0; fi
  log "attempt $att ended without final model; retry in 120 s"; sleep 120
done
log "FAILED after retries"; touch $W/FAILED
