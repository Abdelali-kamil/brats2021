#!/bin/bash
# Amendment 4: adapt ours (ep253) to BraTS-Africa glioma TRAIN 48, select on VAL 12.
# Own default recipe (AdamW 2e-4 cosine, bs2/accum4, same aug/loss, no DS), 300 epochs.
# Attempt 1 warm-starts from ep253 (weights only); retries resume from _last.pth.
set -uo pipefail
A=/mnt/data1/kamil_research/experiments/adapt_africa; TAG=adapt_ours_ep253
CK=$A/ours/checkpoints; LOGD=$A/ours/logs; PY=/home/kamilabdelali/anaconda3/envs/torchfix/bin/python
EP253=/mnt/data1/kamil_research/experiments/aug_cosine_repro1/checkpoints/aug_cosine_repro1_ep253_final.pth
pgrep -f "train_brats.py .*--tag $TAG" >/dev/null && { echo running; exit 0; }
cd $A/code_ours; ulimit -n "$(ulimit -Hn)"
ep() { [ -f $CK/${TAG}_last.pth ] && $PY -c "import torch;print(int(torch.load('$CK/${TAG}_last.pth',map_location='cpu',weights_only=False).get('epoch',0)))" 2>/dev/null || echo 0; }
for att in 1 2 3 4 5 6; do
  e=$(ep); echo "=== attempt $att epoch=$e $(date -Is)" >> $LOGD/run.log
  [ "$e" -ge 300 ] && break
  if [ -f $CK/${TAG}_last.pth ]; then INIT=(--resume $CK/${TAG}_last.pth); else INIT=(--resume $EP253 --warm-restart); fi
  $PY -u scripts/train_brats.py --epochs 300 --tag $TAG --num-workers 6 --batch-size 2 --accum-steps 4 \
      --scheduler cosine --seed 42 --save-dir $CK \
      --data-root /mnt/data1/kamil_research/brats_africa/brats21fmt \
      --train-ids $A/train_ids.txt --val-ids $A/val_ids.txt "${INIT[@]}" >> $LOGD/$TAG.log 2>&1
  sleep 60
done
echo "=== done epoch=$(ep) $(date -Is)" >> $LOGD/run.log
