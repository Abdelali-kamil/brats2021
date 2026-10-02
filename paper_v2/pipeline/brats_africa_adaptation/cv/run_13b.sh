#!/bin/bash
# Amendment 13b: after the Run C CV finishes, per-fold VAL-selected ensemble option. Idempotent.
A=/mnt/data1/kamil_research/experiments/adapt_africa; T=/home/kamilabdelali/anaconda3/envs/torchfix/bin/python
until grep -q "=== CV from Run C done" $A/cv/cv.log; do sleep 300; done
for k in 0 1 2 3 4; do D=$A/cv/fold$k; [ -s $D/per_case_ours_runC_sel.csv ] && continue   # folds in parallel (CPU)
  [ -s $D/per_case_ours_runC.csv ] || continue
  ( $T $A/cv/cv_fold_select_multi.py $k >> $D/select_MC_sel.log 2>&1 && echo "[cv-13b $(date '+%F %T')] fold $k: $(grep 'TEST mean' $D/select_MC_sel.log | tail -1)" >> $A/cv/cv.log ) &
done; wait
echo "[cv-13b $(date '+%F %T')] === 13b done ===" >> $A/cv/cv.log
