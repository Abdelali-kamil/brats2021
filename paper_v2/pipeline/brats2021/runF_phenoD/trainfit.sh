#!/bin/bash
# Amendment 29 TRAIN-fit check: <tag>'s frozen VAL-selected checkpoint, no TTA, nibabel order, its VAL-selected post-processing,
# on the 65 TRAIN target cases and 65 random other TRAIN cases (explicit lists; TEST never loaded). Idempotent.
set -euo pipefail
TAG=$1; V=/mnt/data1/kamil_research/experiments/eval_v2; C=$V/code; L=$V/next_step_20261019/trainfit
A=/mnt/data1/kamil_research/experiments/runF_phenoD; OUT=$A/trainfit_$TAG; mkdir -p $OUT/out
PY=/home/kamilabdelali/anaconda3/envs/torchfix/bin/python
read -r _ _ WT MULT CK ETMIN _ < $V/work_$TAG/selection.txt
if [ "$ETMIN" = 0 ]; then ETARGS=(--et-policy none); else ETARGS=(--et-policy min_volume --et-min-volume "$ETMIN"); fi
cd $C
for G in train_target65 train_other65; do
  [ -s $OUT/out/per_case_${TAG}_notta_${G}_recomputed.csv ] && continue
  $PY scripts/evaluate_brats.py --partition train --cases $L/$G.txt --no-tta --axis-order nib --wt-policy $WT --floor-mult $MULT \
      "${ETARGS[@]}" --checkpoint $CK --out-dir $OUT/out --tag ${TAG}_notta_$G > $OUT/log_$G.txt 2>&1
done
echo "trainfit $TAG done: $(ls $OUT/out/per_case_*.csv | wc -l) files"
