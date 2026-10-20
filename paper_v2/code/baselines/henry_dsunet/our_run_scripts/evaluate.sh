#!/bin/bash
# Amendment 22: evaluation of Henry et al.'s deeply supervised 3D U-Net after training (train.sh -> TRAINED):
#  1. BraTS 2021 TEST (125) scored ONCE: the repository's inference.py with its TTA, common scorer;
#  2. BraTS-Africa zero-shot (146 subjects; gliomas primary), same inference, common scorer;
#  3. classification features from the zero-shot masks + the fixed classification CV;
#  4. BraTS-Africa 5-fold CV fine-tuning (Amendment 6 folds; 150 epochs from the trained weights, repository recipe without
#     SWA, so the final model is kept; inference with TTA on each TEST fold) and pooled scoring.
# Idempotent (each step skips if its output exists), flock-guarded, waits for GPU room.
set -uo pipefail
E=/mnt/data1/kamil_research/experiments/henry_dsunet; X=/mnt/data1/kamil_research/experiments
PY=$E/.venv/bin/python; T=/home/kamilabdelali/anaconda3/envs/torchfix/bin/python
RES=/home/kamilabdelali/brats2021/results/brats; FIXED_SPLIT_CSV_SANITY=/home/kamilabdelali/brats2021/baselines/common/brats_split_frozen.csv; IDS=/home/kamilabdelali/brats2021/baselines/common/test_ids.txt
LOG=$E/evaluate.log; exec 9> $E/.lock_eval; flock -n 9 || exit 0
log() { echo "[henry-eval $(date '+%F %T')] $*" | tee -a $LOG; }
wait_gpu() { until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge 12000 ]; do sleep 300; done; }
until [ -e $E/TRAINED ] || [ -e $E/FAILED ]; do sleep 600; done
[ -e $E/FAILED ] && { log "training failed (collapse rule) -> nothing to evaluate"; exit 0; }
if [ ! -s $E/RUN_PATH ]; then   # move the finished BraTS run out of code/runs, so later fine-tuning runs cannot disturb it
  mkdir -p $E/brats_run; mv $E/code/$(cat $E/RUN_DIR) $E/brats_run/ && echo "$E/brats_run/$(basename $(cat $E/RUN_DIR))" > $E/RUN_PATH; fi
RUN=$(cat $E/RUN_PATH); CFG=$(ls $RUN/*.yaml | head -1); CK=$RUN/model_best.pth.tar
[ -f $CK ] || { log "no checkpoint at $CK"; exit 1; }
export PYTHONPATH=$E/ranger_pkg:$E/code PYTHONUNBUFFERED=1
cd $E/code
infer() {   # infer <input folder of case dirs> <output folder>
  [ "$(ls $2/*.nii.gz 2>/dev/null | wc -l)" -eq "$(ls -d $1/*/ | wc -l)" ] && return 0
  wait_gpu; local before=$(ls -d preds/* 2>/dev/null | wc -l)
  BRATS_TEST_FOLDER=$1 $PY -m src.inference --config $CFG --devices 0 --on test --tta >> $E/inference.log 2>&1 || return 1
  local P=$(ls -td preds/*/test_segs_ttaTrue | head -1); mkdir -p $2; cp -n $P/*.nii.gz $2/
}
links_brats() { mkdir -p $1; while read -r c; do [ -e $1/$c ] || ln -s /mnt/data1/kamil_research/data/brats2021/$c $1/$c; done < $2; }
links_africa() {   # links_africa <folder> <ids file>: BraTS-2021-style names pointing to the reoriented staged images
  while read -r c; do d=$1/$c; mkdir -p $d; S=$X/external_africa/staged/$c
    for m in t1:t1n t1ce:t1c t2:t2w flair:t2f; do [ -e $d/${c}_${m%%:*}.nii.gz ] || ln -s $S/$c-${m##*:}.nii.gz $d/${c}_${m%%:*}.nii.gz; done
  done < $2; }
# 0. Sanity gate (added 2026-10-10, before any test scoring): the inference path was never verified (smoke OOM'd on the
#    shared GPU). Run the same inference + common scorer on the first 5 VAL cases of the frozen split; the one-time TEST
#    scoring only proceeds if mean WT Dice >= 0.70 and mean (ET+TC+WT)/3 >= 0.60. VAL only: no test case is touched.
if [ ! -e $E/SANITY_OK ]; then
  tr -d "\r" < $FIXED_SPLIT_CSV_SANITY | grep ",val$" | head -5 | cut -d, -f1 > $E/val5_ids.txt   # CRLF csv (fixed 2026-10-12)
  links_brats $E/val5_in $E/val5_ids.txt; mkdir -p $E/sanity
  if infer $E/val5_in $E/val5_pred && $T /home/kamilabdelali/brats2021/baselines/monai_common/score_monai_test.py $E/val5_pred $E/val5_ids.txt $E/sanity/val5 henry_val5 >> $E/sanity/score.log 2>&1 \
     && $T -c "import pandas as p,sys; d=p.read_csv('$E/sanity/val5_per_case.csv'); m=d[['Dice_ET','Dice_TC','Dice_WT']].mean(); print('VAL5 ET/TC/WT %.4f %.4f %.4f'%tuple(m)); sys.exit(0 if m['Dice_WT']>=0.70 and m.mean()>=0.60 else 1)" > $E/sanity/check.txt 2>&1; then
    log "sanity OK: $(cat $E/sanity/check.txt)"; touch $E/SANITY_OK
  else
    log "SANITY FAILED -> test NOT scored: $(tail -2 $E/sanity/check.txt $E/sanity/score.log 2>/dev/null | tr '\n' ' ' | cut -c1-300)"; touch $E/SANITY_FAILED; exit 1
  fi
fi
# 1. BraTS 2021 test, once
if [ ! -s $RES/henry_dsunet_test_per_case.csv ]; then
  links_brats $E/test125_in $IDS
  infer $E/test125_in $E/test125_pred && $T /home/kamilabdelali/brats2021/baselines/monai_common/score_monai_test.py $E/test125_pred $IDS $RES/henry_dsunet_test henry_dsunet_test >> $E/score.log 2>&1 \
    && log "TEST: $(grep MEAN $E/score.log | tail -1 | cut -c1-120)" || log "TEST FAILED"
fi
# 2. BraTS-Africa zero-shot
if [ ! -s $X/external_africa/results/per_case_henry_dsunet.csv ]; then
  tail -n +2 $X/external_africa/cases_gated.csv | cut -d, -f1 > $E/africa146_ids.txt; links_africa $E/africa146_in $E/africa146_ids.txt
  infer $E/africa146_in $X/external_africa/pred_henry_dsunet && (cd $X/external_africa && $T score_labelmaps.py henry_dsunet $X/external_africa/pred_henry_dsunet brats21 >> $E/score_africa.log 2>&1) \
    && log "AFRICA zero-shot: $(grep -i mean $E/score_africa.log | tail -2 | tr '\n' ' ' | cut -c1-200)" || log "AFRICA FAILED"
fi
# 3. classification
C=$X/classification_africa
[ -s $C/features_henry_dsunet.csv ] || (cd $C && $T extract_baselines.py henry_dsunet >> $E/features.log 2>&1) || log "features FAILED"
[ -s $C/summary_henry_dsunet.txt ] || { (cd $C && $T classify.py henry_dsunet > $E/classify.log 2>&1) && log "classification: $(head -2 $C/summary_henry_dsunet.txt | tr '\n' ' ')"; }
# 4. BraTS-Africa CV fine-tuning
A=$X/adapt_africa/cv
for k in 0 1 2 3 4; do
  F=$A/fold$k; [ "$(ls $F/pred_henry_dsunet/*.nii.gz 2>/dev/null | wc -l)" -eq 19 ] && continue
  if [ ! -s $E/cv/fold$k/RUN_DIR ]; then
    mkdir -p $E/cv/fold$k; { echo "case_id,partition"; sed 's/$/,train/' $F/train_ids.txt; sed 's/$/,val/' $F/val_ids.txt; } > $E/cv/fold$k/split.csv
    wait_gpu; log "CV fold $k: fine-tuning"
    [ -d runs ] && mv runs $E/cv/runs_before_fold${k}_$(date +%H%M%S)
    FIXED_SPLIT_CSV=$E/cv/fold$k/split.csv BRATS_FOLDERS=/mnt/data1/kamil_research/brats_africa/brats21fmt \
      $PY -m src.train --devices 0 --width 48 --arch EquiUnet --deep_sup -j 8 --epochs 150 --init_weights $CK > $E/cv/fold$k/train.log 2>&1 \
      || { log "CV fold $k training FAILED"; continue; }
    R=$(ls -d runs/*EquiUnet_48* | head -1); mkdir -p $E/cv/fold$k/run; mv $R $E/cv/fold$k/run/ && echo "$E/cv/fold$k/run/$(basename $R)" > $E/cv/fold$k/RUN_DIR; rmdir runs 2>/dev/null
  fi
  mkdir -p $E/cv/fold$k/test_in; while read -r c; do [ -e $E/cv/fold$k/test_in/$c ] || ln -s /mnt/data1/kamil_research/brats_africa/brats21fmt/$c $E/cv/fold$k/test_in/$c; done < $F/test_ids.txt
  CFGK=$(ls $(cat $E/cv/fold$k/RUN_DIR)/*.yaml | head -1)
  (CFG=$CFGK; infer $E/cv/fold$k/test_in $F/pred_henry_dsunet) && log "CV fold $k: $(ls $F/pred_henry_dsunet | wc -l) predictions" || log "CV fold $k inference FAILED"
done
$T $X/adapt_africa/cv/score_cv_monai.py henry_dsunet >> $LOG 2>&1
log "=== evaluation finished ==="; touch $E/EVALUATED
