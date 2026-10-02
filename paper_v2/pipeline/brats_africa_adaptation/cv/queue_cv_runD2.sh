#!/bin/bash
# PROTOCOL_v2 Amendment 17: ours' BraTS-Africa 5-fold CV from Run D2 (label MD2), Amendment-13 procedure.
# Same split and Amendment-6 recipe; inference in Run D2's VAL-chosen axis mode; per-fold VAL
# selection; TEST once per fold -> fold*/per_case_ours_runD2.csv. Idempotent, flock, @reboot.
set -uo pipefail
A=/mnt/data1/kamil_research/experiments/adapt_africa; E=/mnt/data1/kamil_research/experiments; L=$A/cv/cv.log
T=/home/kamilabdelali/anaconda3/envs/torchfix/bin/python
exec 9> $A/cv/.runD2.lock; flock -n 9 || exit 0
log() { echo "[cv-runD2 $(date '+%F %T')] $*" | tee -a $L; }
until [ -e $E/queue_v2_state/runD2_gate_pass ]; do sleep 900; done   # Amendment 17 gate (VAL only), set by queue_v6
SCK=$(cut -d' ' -f5 $E/eval_v2/work_runD2/selection.txt); O=$(cut -d' ' -f7 $E/eval_v2/work_runD2/selection.txt); O=${O%+rot}
FL=(--downsample dwt3d --base-filters 24 --deep-supervision --norm instance --normalisation zscore)
log "=== Amendment 17 CV from Run D2: $SCK (axis $O) flags ${FL[*]} ==="
ep() { [ -f $1 ] && $T -c "import torch;print(int(torch.load('$1',map_location='cpu',weights_only=False).get('epoch',0)))" 2>/dev/null || echo 0; }
for k in 0 1 2 3 4; do
  D=$A/cv/fold$k; [ -s $D/per_case_ours_runD2.csv ] && continue
  TAG=adapt_cv_md2_f$k; CKD=$D/ckpt_MD2; mkdir -p $CKD
  for att in 1 2 3 4; do
    [ "$(ep $CKD/${TAG}_last.pth)" -ge 150 ] && break
    until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge 20500 ]; do sleep 300; done
    if [ -f $CKD/${TAG}_last.pth ]; then INIT=(--resume $CKD/${TAG}_last.pth); else INIT=(--resume $SCK --warm-restart); fi
    log "fold $k MD2 train (attempt $att)"
    ( cd $A/code_ours_v2 && ulimit -n "$(ulimit -Hn)" && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $T -u scripts/train_brats.py \
        --epochs 150 --tag $TAG --num-workers 6 --batch-size 2 --accum-steps 4 --scheduler cosine --seed 42 --save-dir $CKD "${FL[@]}" \
        --data-root /mnt/data1/kamil_research/brats_africa/brats21fmt --train-ids $D/train_ids.txt --val-ids $D/val_ids.txt \
        "${INIT[@]}" ) >> $D/train_MD2.log 2>&1
  done
  [ "$(ep $CKD/${TAG}_last.pth)" -ge 150 ] || { log "FAILED fold $k MD2 training"; continue; }
  F=$CKD/best_evaluated.pth; [ -f $F ] || { cp -p $CKD/${TAG}_best.pth $F; chmod a-w $F; }
  ok=1; for s in val test; do
    [ "$(ls $D/probs_MD2_$s/*.npz 2>/dev/null | wc -l)" -eq "$(grep -c . $D/${s}_ids.txt)" ] && continue
    until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge 7000 ]; do sleep 120; done
    $T $A/eval_code/scripts/infer_probs.py $F $D/${s}_ids.txt $D/probs_MD2_$s $O >> $D/infer_MD2.log 2>&1 || ok=0
  done
  [ $ok = 1 ] || { log "FAILED fold $k MD2 inference"; continue; }
  $T $A/cv/cv_fold_select_single.py $k MD2 per_case_ours_runD2.csv >> $D/select_MD2.log 2>&1 || { log "FAILED fold $k MD2 selection"; continue; }
  log "fold $k: $(grep -E 'TEST mean' $D/select_MD2.log | tail -1)"
done
log "=== CV from Run D2 done ==="
