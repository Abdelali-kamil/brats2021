#!/bin/bash
# Amendment 17 (13b procedure): after the Run D2 CV finishes, per-fold VAL-selected ensemble option. Idempotent.
A=/mnt/data1/kamil_research/experiments/adapt_africa; T=/home/kamilabdelali/anaconda3/envs/torchfix/bin/python
until grep -q "=== CV from Run D2 done" $A/cv/cv.log; do sleep 300; done
for k in 0 1 2 3 4; do D=$A/cv/fold$k; [ -s $D/per_case_ours_runD2_sel.csv ] && continue   # folds in parallel (CPU)
  [ -s $D/per_case_ours_runD2.csv ] || continue
  ( $T $A/cv/cv_fold_select_multi_md2.py $k >> $D/select_MD2_sel.log 2>&1 && echo "[cv-17md2 $(date '+%F %T')] fold $k: $(grep 'TEST mean' $D/select_MD2_sel.log | tail -1)" >> $A/cv/cv.log ) &
done; wait
echo "[cv-17md2 $(date '+%F %T')] === 17md2 done ===" >> $A/cv/cv.log
