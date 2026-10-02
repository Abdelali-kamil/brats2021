#!/bin/bash
# Swin UNETR on OUR 125-case BraTS held-out test set, scored with our metric.
#
# PRIMARY (declared before scoring): model.pt = best-by-val checkpoint (ep199),
# the upstream recipe's selection rule and the same rule as Wavelet U-Net++.
# SECONDARY, reference only: model_final.pt (ep299).
#
#   bash baselines/swin_unetr/run_swin_test_eval.sh
set -euo pipefail
ROOT=/home/kamilabdelali/brats2021
PY=$ROOT/baselines/swin_unetr/.venv_overlay/bin/python       # inference (torch cu128 + monai 1.6)
SPY=/home/kamilabdelali/anaconda3/envs/nnunet_infer/bin/python # scoring (same env as nnU-Net's score)
RUNS=$ROOT/baselines/repos/swin_unetr/SwinUNETR/BRATS21/runs/swin_unetr_frozen
IDS=$ROOT/baselines/common/test_ids.txt
WORK=/mnt/data1/kamil_research/baselines/swin_unetr/eval
mkdir -p "$WORK"
LOG=$WORK/run_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1
ulimit -n "$(ulimit -Hn)"

for spec in "model.pt:best_ep199:swin_unetr_test" "model_final.pt:final_ep299:swin_unetr_final_ep299_test"; do
  IFS=: read -r ck tag prefix <<< "$spec"
  echo "== [$tag] inference $(date -Is) =="
  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "$PY" "$ROOT/baselines/swin_unetr/infer_test125.py" \
      --ckpt "$RUNS/$ck" --out "$WORK/test125_pred_$tag"
  echo "== [$tag] scoring $(date -Is) =="
  "$SPY" "$ROOT/baselines/swin_unetr/score_swin_test.py" \
      "$WORK/test125_pred_$tag" "$IDS" "$ROOT/results/brats/$prefix" "$prefix"
done
echo "== DONE $(date -Is) =="
