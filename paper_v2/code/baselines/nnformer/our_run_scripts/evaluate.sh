#!/bin/bash
# Amendments 25/26: evaluation of nnFormer after training (train.sh -> TRAINED), same steps as every baseline:
#  0. VAL5 sanity gate  1. BraTS 2021 TEST (125) ONCE  2. BraTS-Africa zero-shot (146)  3. classification
#  4. BraTS-Africa 5-fold CV (19 epochs x 250 x batch 2 from its BraTS weights, model_best on the fold's 12 VAL)  5. efficiency
# Inference = nnFormer_predict 3d_fullres -chk model_best (its defaults: step 0.5, Gaussian, mirroring TTA, mixed precision);
# predictions converted from the authors' labels to BraTS 1/2/4 for the common scorer. Idempotent; flock.
set -uo pipefail
W=/mnt/data1/kamil_research/experiments/nnformer; X=/mnt/data1/kamil_research/experiments; source $W/env.sh
T=/home/kamilabdelali/anaconda3/envs/torchfix/bin/python; IO="$T $W/nnformer_io.py"
RES=/home/kamilabdelali/brats2021/results/brats; DATA=/mnt/data1/kamil_research/data/brats2021; AF=$X/henry_dsunet/africa146_in
LOG=$W/evaluate.log; exec 9> $W/.lock_eval; flock -n 9 || exit 0
log() { echo "[nnformer-eval $(date '+%F %T')] $*" | tee -a $LOG; }
wait_gpu() { until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge 10000 ]; do sleep 300; done; }
until [ -e $W/TRAINED ] || [ -e $W/FAILED ]; do sleep 600; done
[ -e $W/TRAINED ] || { log "training FAILED -> nothing to evaluate"; exit 1; }
predict() {   # predict <input dir> <final BraTS-label dir> <n expected>  (uses current nnFormer_* env)
  [ "$(ls $2/*.nii.gz 2>/dev/null | wc -l)" -eq "$3" ] && return 0
  wait_gpu   # nnFormer_predict skips cases already in the output folder (resume-safe)
  CUDA_VISIBLE_DEVICES=0 nnFormer_predict -i $1 -o $2.raw_labels -m 3d_fullres -t 3 -f 0 -chk model_best \
    -tr nnFormerTrainerV2_nnformer_tumor >> $W/inference.log 2>&1 || return 1
  $IO to_brats $2.raw_labels $2 && [ "$(ls $2/*.nii.gz | wc -l)" -eq "$3" ]
}
# 0. sanity gate
if [ ! -e $W/SANITY_OK ]; then
  python3 -c "import json; print('\n'.join(json.load(open('$W/split_fixed.json'))['val'][:5]))" > $W/val5.txt; mkdir -p $W/sanity
  $IO link_inputs $W/val5_in $W/val5.txt $DATA
  if predict $W/val5_in $W/val5_pred 5 \
     && $T /home/kamilabdelali/brats2021/baselines/monai_common/score_monai_test.py $W/val5_pred $W/val5.txt $W/sanity/val5 nnformer_val5 >> $W/sanity/score.log 2>&1 \
     && $T -c "import pandas as p,sys; d=p.read_csv('$W/sanity/val5_per_case.csv'); m=d[['Dice_ET','Dice_TC','Dice_WT']].mean(); print('VAL5 ET/TC/WT %.4f %.4f %.4f'%tuple(m)); sys.exit(0 if m['Dice_WT']>=0.70 and m.mean()>=0.60 else 1)" > $W/sanity/check.txt 2>&1; then
    log "sanity OK: $(cat $W/sanity/check.txt)"; touch $W/SANITY_OK
  else
    log "SANITY FAILED -> test NOT scored: $(tail -2 $W/sanity/check.txt $W/sanity/score.log 2>/dev/null | tr '\n' ' ' | cut -c1-300)"; touch $W/SANITY_FAILED; exit 1
  fi
fi
# 1. BraTS 2021 test, once
if [ ! -s $RES/nnformer_test_per_case.csv ]; then
  predict $nnFormer_raw_data_base/nnFormer_raw_data/Task003_tumor/imagesTs $W/test125_pred 125 \
    && $T /home/kamilabdelali/brats2021/baselines/monai_common/score_monai_test.py $W/test125_pred $X/transbts/test.txt $RES/nnformer_test nnformer_test >> $W/score.log 2>&1 \
    && log "TEST: $(grep -E '^\S+\s+MEAN' $W/score.log | tail -1 | cut -c1-150)" || { log "TEST FAILED"; exit 1; }
fi
# 2. BraTS-Africa zero-shot (same staged, reoriented inputs as every baseline)
if [ ! -s $X/external_africa/results/per_case_nnformer.csv ]; then
  ls $AF > $W/africa146_ids.txt; $IO link_inputs $W/africa146_in $W/africa146_ids.txt $AF
  predict $W/africa146_in $X/external_africa/pred_nnformer 146 \
    && (cd $X/external_africa && $T score_labelmaps.py nnformer $X/external_africa/pred_nnformer brats21 >> $W/score_africa.log 2>&1) \
    && log "AFRICA zero-shot: $(grep -i mean $W/score_africa.log | tail -2 | tr '\n' ' ' | cut -c1-200)" || log "AFRICA FAILED"
fi
# 3. classification
C=$X/classification_africa
[ -s $C/features_nnformer.csv ] || (cd $C && $T extract_baselines.py nnformer >> $W/features.log 2>&1) || log "features FAILED"
[ -s $C/summary_nnformer.txt ] || { (cd $C && $T classify.py nnformer > $W/classify.log 2>&1) && log "classification: $(head -2 $C/summary_nnformer.txt | tr '\n' ' ')"; }
# 4. BraTS-Africa CV (each fold = its own Task003_tumor base; Amendment 6 budget for nnU-Net-type trainers)
A=$X/adapt_africa/cv; BRATS_BEST=$(cat $W/RUN_PATH)/model_best.model
for k in 0 1 2 3 4; do
  F=$A/fold$k; [ "$(ls $F/pred_nnformer/*.nii.gz 2>/dev/null | wc -l)" -eq 19 ] && continue
  B=$W/DATASET_cv/fold$k
  (
    export nnFormer_raw_data_base=$B/nnFormer_raw nnFormer_preprocessed=$B/nnFormer_preprocessed RESULTS_FOLDER=$B/nnFormer_trained_models
    export NNFORMER_FIXED_SPLIT=$B/split.json NNFORMER_MAX_EPOCHS=19
    OUT=$RESULTS_FOLDER/nnFormer/3d_fullres/Task003_tumor/nnFormerTrainerV2_nnformer_tumor__nnFormerPlansv2.1/fold_0
    [ -s $B/split.json ] || $IO cv_fold $k >> $W/cv_fold$k.log 2>&1
    [ -d $nnFormer_preprocessed/Task003_tumor ] || (cd $W && nnFormer_plan_and_preprocess -t 3 -tl 8 -tf 8 >> $W/cv_fold$k.log 2>&1) || exit 1
    if [ ! -s $OUT/model_best.model ] || [ ! -s $OUT/model_final_checkpoint.model ]; then
      wait_gpu; log "CV fold $k: fine-tuning"
      (cd $W && CUDA_VISIBLE_DEVICES=0 nnFormer_train 3d_fullres nnFormerTrainerV2_nnformer_tumor 3 0 -pretrained_weights $BRATS_BEST >> $W/cv_fold$k.log 2>&1) || exit 1
    fi
    predict $B/test_in $F/pred_nnformer 19
  ) && log "CV fold $k: $(ls $F/pred_nnformer | wc -l) predictions" || log "CV fold $k FAILED"
done
$T $A/score_cv_monai.py nnformer >> $LOG 2>&1
# 5. efficiency on the otherwise idle GPU
if [ ! -s $X/efficiency/results/nnformer.json ]; then
  for i in $(seq 1 24); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break; sleep 300; done
  (cd $X/efficiency && $W/.venv/bin/python bench_nnformer.py && $W/.venv/bin/python bench_nnformer.py --name nnformer_noTTA --no-tta) >> $W/bench.log 2>&1 && log "efficiency done" || log "efficiency FAILED"
fi
log "=== evaluation finished ==="; touch $W/EVALUATED
