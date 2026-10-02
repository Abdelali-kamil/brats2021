#!/bin/bash
# Amendment 4, Swin UNETR: upstream main.py, default recipe (AdamW 1e-4, warmup-cosine,
# roi 96, bs1), weights-only init from BraTS model.pt (ep199), 300 epochs over TRAIN 48,
# val every 10 on VAL 12 (fold 0); model.pt = best by val.
set -uo pipefail
A=/mnt/data1/kamil_research/experiments/adapt_africa; R=/home/kamilabdelali/brats2021/baselines/repos/swin_unetr/SwinUNETR/BRATS21
PY=/home/kamilabdelali/brats2021/baselines/swin_unetr/.venv_overlay/bin/python
cd $R; ulimit -n "$(ulimit -Hn)"
[ -f runs/adapt_africa_swin/model_final.pt ] && { echo "done already"; exit 0; }
rm -rf runs/adapt_africa_swin   # upstream cannot resume; a crashed run restarts clean
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 $PY main.py --json_list $A/swin/datalist_adapt.json \
  --data_dir /mnt/data1/kamil_research/brats_africa/brats21fmt --fold 0 --feature_size 48 --use_checkpoint \
  --save_checkpoint --max_epochs 300 --val_every 10 --batch_size 1 --workers 4 --logdir adapt_africa_swin \
  --checkpoint $A/swin/init_from_brats_ep199.pt > $A/swin/train.log 2>&1
