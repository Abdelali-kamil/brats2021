#!/bin/bash
# PROTOCOL_v2 Amendment 13 (SECONDARY): ours' BraTS-Africa 5-fold CV from Run C (label MC).
# Same split and Amendment-6 recipe; inference in Run C's VAL-chosen mode; per-fold VAL
# selection; TEST once per fold -> fold*/per_case_ours_runC.csv. Idempotent, flock, @reboot.
set -uo pipefail
A=/mnt/data1/kamil_research/experiments/adapt_africa; E=/mnt/data1/kamil_research/experiments; L=$A/cv/cv.log
T=/home/kamilabdelali/anaconda3/envs/torchfix/bin/python
exec 9> $A/cv/.runC.lock; flock -n 9 || exit 0
log() { echo "[cv-runC $(date '+%F %T')] $*" | tee -a $L; }
SCK=$(cut -d' ' -f5 $E/eval_v2/work_runC/selection.txt); O=$(cut -d' ' -f7 $E/eval_v2/work_runC/selection.txt)
FL=(--downsample dwt3d --base-filters 24 --deep-supervision --norm batch --normalisation minmax)
log "=== Amendment 13 CV from Run C: $SCK (axis $O) flags ${FL[*]} ==="
ep() { [ -f $1 ] && $T -c "import torch;print(int(torch.load('$1',map_location='cpu',weights_only=False).get('epoch',0)))" 2>/dev/null || echo 0; }
for k in 0 1 2 3 4; do
  D=$A/cv/fold$k; [ -s $D/per_case_ours_runC.csv ] && continue
  TAG=adapt_cv_mc_f$k; CKD=$D/ckpt_MC; mkdir -p $CKD
  for att in 1 2 3 4; do
    [ "$(ep $CKD/${TAG}_last.pth)" -ge 150 ] && break
    until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge 20500 ]; do sleep 300; done
    if [ -f $CKD/${TAG}_last.pth ]; then INIT=(--resume $CKD/${TAG}_last.pth); else INIT=(--resume $SCK --warm-restart); fi
    log "fold $k MC train (attempt $att)"
    ( cd $A/code_ours_v2 && ulimit -n "$(ulimit -Hn)" && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $T -u scripts/train_brats.py \
        --epochs 150 --tag $TAG --num-workers 6 --batch-size 2 --accum-steps 4 --scheduler cosine --seed 42 --save-dir $CKD "${FL[@]}" \
        --data-root /mnt/data1/kamil_research/brats_africa/brats21fmt --train-ids $D/train_ids.txt --val-ids $D/val_ids.txt \
        "${INIT[@]}" ) >> $D/train_MC.log 2>&1
  done
  [ "$(ep $CKD/${TAG}_last.pth)" -ge 150 ] || { log "FAILED fold $k MC training"; continue; }
  F=$CKD/best_evaluated.pth; [ -f $F ] || { cp -p $CKD/${TAG}_best.pth $F; chmod a-w $F; }
  ok=1; for s in val test; do
    [ "$(ls $D/probs_MC_$s/*.npz 2>/dev/null | wc -l)" -eq "$(grep -c . $D/${s}_ids.txt)" ] && continue
    until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge 7000 ]; do sleep 120; done
    $T $A/eval_code/scripts/infer_probs.py $F $D/${s}_ids.txt $D/probs_MC_$s $O >> $D/infer_MC.log 2>&1 || ok=0
  done
  [ $ok = 1 ] || { log "FAILED fold $k MC inference"; continue; }
  $T $A/cv/cv_fold_select_single.py $k MC per_case_ours_runC.csv >> $D/select_MC.log 2>&1 || { log "FAILED fold $k MC selection"; continue; }
  log "fold $k: $(grep -E 'TEST mean' $D/select_MC.log | tail -1)"
done
log "=== CV from Run C done ==="
