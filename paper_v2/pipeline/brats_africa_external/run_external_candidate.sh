#!/bin/bash
# Add one of our retrained candidates (runA / runC) to the BraTS-Africa evaluation,
# using ITS BraTS-val-selected post-processing and frozen weights from eval_v2.
#   bash run_external_candidate.sh runA
set -uo pipefail
TAG=$1; X=/mnt/data1/kamil_research/experiments/external_africa
SEL=/mnt/data1/kamil_research/experiments/eval_v2/work_$TAG/selection.txt
[ -s "$SEL" ] || { echo "no BraTS-val selection for $TAG yet"; exit 1; }
[ -s $X/results/per_case_ours_$TAG.csv ] && { echo "already done"; exit 0; }
read -r _ VALD POL MULT CK ETMIN ORIENT < "$SEL"; ETMIN=${ETMIN:-200}; ORIENT=${ORIENT:-nib}
echo "[ext $(date '+%F %T')] ours $TAG (val $VALD, $POL x$MULT, ET>=$ETMIN)" | tee -a $X/run_external.log
cd $X && /home/kamilabdelali/anaconda3/envs/torchfix/bin/python eval_ours_external.py ours_$TAG "$CK" "$POL" "$MULT" "$ETMIN" "$ORIENT" >> $X/ours_$TAG.log 2>&1
