#!/bin/bash
# Amendment 24/25: evaluation of TransBTS after training (train.sh -> TRAINED), same steps as every baseline:
#  0. VAL5 sanity gate (first 5 VAL cases; TEST only if mean WT >= 0.70 and mean Dice >= 0.60)
#  1. BraTS 2021 TEST (125) scored ONCE   2. BraTS-Africa zero-shot (146)   3. classification with its zero-shot masks
#  4. BraTS-Africa 5-fold CV fine-tuning (from its BraTS weights, 150 epochs over the 64 TRAIN cases, its recipe, final model)
#  5. efficiency (otherwise idle GPU)
# Inference = the repository's test.py/predict.py (8 crops, 8-flip TTA, argmax, labels 1/2/4). Idempotent; flock.
set -uo pipefail
E=/mnt/data1/kamil_research/experiments/transbts; X=/mnt/data1/kamil_research/experiments
PY=$E/.venv/bin/python; T=/home/kamilabdelali/anaconda3/envs/torchfix/bin/python
RES=/home/kamilabdelali/brats2021/results/brats; DATA=/mnt/data1/kamil_research/data; AF=$X/henry_dsunet/africa146_in
LOG=$E/evaluate.log; exec 9> $E/.lock_eval; flock -n 9 || exit 0
log() { echo "[transbts-eval $(date '+%F %T')] $*" | tee -a $LOG; }
wait_gpu() { until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge 16000 ]; do sleep 300; done; }
until [ -e $E/TRAINED ] || [ -e $E/FAILED ]; do sleep 600; done
[ -e $E/TRAINED ] || { log "training FAILED -> nothing to evaluate"; exit 1; }
CK=$E/code/checkpoint/TransBTS_dice2_20261014/model_epoch_last.pth
[ -s $CK ] || { log "no checkpoint at $CK"; exit 1; }
cd $E/code
infer() {   # infer <root> <dir> <list file> <out dir> <experiment> <ckpt file name> <name>
  local n=$(wc -l < $3); [ "$(ls $4/*.nii.gz 2>/dev/null | wc -l)" -eq "$n" ] && [ "$n" -gt 0 ] && return 0
  wait_gpu
  TRANSBTS_REF_ROOT=$1/$2 $PY test.py --root $1 --valid_dir $2 --valid_file $3 --experiment $5 --test_date 20261014 \
    --test_file $6 --submission $7 --visual vis_$7 --gpu 0 --num_workers 4 >> $E/inference.log 2>&1 || return 1
  mkdir -p $4; cp output/$7/${5}20261014/*.nii.gz $4/   # test.py writes to <output_dir>/<submission>/<experiment+date>/
  [ "$(ls $4/*.nii.gz | wc -l)" -eq "$n" ]
}
# 0. sanity gate
if [ ! -e $E/SANITY_OK ]; then
  head -5 $E/val.txt > $E/val5.txt; mkdir -p $E/sanity
  if infer $DATA brats2021 $E/val5.txt $E/val5_pred TransBTS_dice2_ model_epoch_last.pth val5 \
     && $T /home/kamilabdelali/brats2021/baselines/monai_common/score_monai_test.py $E/val5_pred $E/val5.txt $E/sanity/val5 transbts_val5 >> $E/sanity/score.log 2>&1 \
     && $T -c "import pandas as p,sys; d=p.read_csv('$E/sanity/val5_per_case.csv'); m=d[['Dice_ET','Dice_TC','Dice_WT']].mean(); print('VAL5 ET/TC/WT %.4f %.4f %.4f'%tuple(m)); sys.exit(0 if m['Dice_WT']>=0.70 and m.mean()>=0.60 else 1)" > $E/sanity/check.txt 2>&1; then
    log "sanity OK: $(cat $E/sanity/check.txt)"; touch $E/SANITY_OK
  else
    log "SANITY FAILED -> test NOT scored: $(tail -2 $E/sanity/check.txt $E/sanity/score.log 2>/dev/null | tr '\n' ' ' | cut -c1-300)"; touch $E/SANITY_FAILED; exit 1
  fi
fi
# 1. BraTS 2021 test, once
if [ ! -s $RES/transbts_test_per_case.csv ]; then
  infer $DATA brats2021 $E/test.txt $E/test125_pred TransBTS_dice2_ model_epoch_last.pth test125 \
    && $T /home/kamilabdelali/brats2021/baselines/monai_common/score_monai_test.py $E/test125_pred $E/test.txt $RES/transbts_test transbts_test >> $E/score.log 2>&1 \
    && log "TEST: $(grep -E '^\S+\s+MEAN' $E/score.log | tail -1 | cut -c1-150)" || { log "TEST FAILED"; exit 1; }
fi
# 2. BraTS-Africa zero-shot (same staged, reoriented inputs as every baseline)
if [ ! -s $X/external_africa/results/per_case_transbts.csv ]; then
  ls $AF > $E/africa146_ids.txt
  infer $(dirname $AF) $(basename $AF) $E/africa146_ids.txt $X/external_africa/pred_transbts TransBTS_dice2_ model_epoch_last.pth africa146 \
    && (cd $X/external_africa && $T score_labelmaps.py transbts $X/external_africa/pred_transbts brats21 >> $E/score_africa.log 2>&1) \
    && log "AFRICA zero-shot: $(grep -i mean $E/score_africa.log | tail -2 | tr '\n' ' ' | cut -c1-200)" || log "AFRICA FAILED"
fi
# 3. classification
C=$X/classification_africa
[ -s $C/features_transbts.csv ] || (cd $C && $T extract_baselines.py transbts >> $E/features.log 2>&1) || log "features FAILED"
[ -s $C/summary_transbts.txt ] || { (cd $C && $T classify.py transbts > $E/classify.log 2>&1) && log "classification: $(head -2 $C/summary_transbts.txt | tr '\n' ' ')"; }
# 4. BraTS-Africa CV fine-tuning (Amendment 6 folds)
A=$X/adapt_africa/cv
for k in 0 1 2 3 4; do
  F=$A/fold$k; [ "$(ls $F/pred_transbts/*.nii.gz 2>/dev/null | wc -l)" -eq 19 ] && continue
  CKK=checkpoint/cv_fold${k}_20261014/model_epoch_last.pth
  if [ ! -s $CKK ]; then
    wait_gpu; log "CV fold $k: fine-tuning"
    $PY -m torch.distributed.run --nproc_per_node=1 --master_port 2002$k train.py --root /mnt/data1/kamil_research/brats_africa \
      --train_dir brats21fmt --train_file $F/train_ids.txt --mode train --gpu 0 --batch_size 2 --accum_steps 4 --num_workers 8 \
      --end_epoch 150 --save_freq 1000 --date 20261014 --experiment cv_fold${k}_ --user transbts --criterion softmax_dice2 --init_weights $CK \
      > $E/cv_fold$k.log 2>&1 || { log "CV fold $k training FAILED"; continue; }
  fi
  infer /mnt/data1/kamil_research/brats_africa brats21fmt $F/test_ids.txt $F/pred_transbts cv_fold${k}_ model_epoch_last.pth cv_fold$k \
    && log "CV fold $k: $(ls $F/pred_transbts | wc -l) predictions" || log "CV fold $k inference FAILED"
done
$T $A/score_cv_monai.py transbts >> $LOG 2>&1
# 5. efficiency on the otherwise idle GPU
if [ ! -s $X/efficiency/results/transbts.json ]; then
  # running jobs in parallel (user: "use full server"): time only when none of OUR jobs is on the GPU (2026-10-14)
  ours_on_gpu() { for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do [ "$(ps -o user= -p $p 2>/dev/null)" = "kamilab+" ] || [ "$(ps -o user= -p $p 2>/dev/null)" = "kamilabdelali" ] && return 0; done; return 1; }
  while ours_on_gpu; do sleep 300; done
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader > $(dirname $LOG)/foreign_gpu_at_bench.txt
  (cd $X/efficiency && $PY bench_transbts.py && $PY bench_transbts.py --name transbts_noTTA --no-tta) >> $E/bench.log 2>&1 && log "efficiency done" || log "efficiency FAILED"
fi
log "=== evaluation finished ==="; touch $E/EVALUATED
