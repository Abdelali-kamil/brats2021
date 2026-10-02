#!/bin/bash
# Amendments 6+7+8+10+11 (M1 and M2), ours' BraTS-Africa 5-fold CV. Amendment 11: starts immediately (no
# wait for Run D, which is cancelled); M1/M2 = top two by BraTS VAL among {ep253, runA}; M1 on all folds first.
# M1/M2 = best/second-best SINGLE model by BraTS VAL Dice among {ep253, runA, runC, runD}
# (ep253 from its VAL-only Dice sweep). Per fold: fine-tune M1 and M2 (150 epochs, own
# architecture flags, warm start), save VAL and TEST probabilities (no scoring), then
# cv_fold_select.py chooses on VAL and scores TEST once. Idempotent, flock, @reboot.
set -uo pipefail
A=/mnt/data1/kamil_research/experiments/adapt_africa; E=/mnt/data1/kamil_research/experiments; L=$A/cv/cv.log
T=/home/kamilabdelali/anaconda3/envs/torchfix/bin/python
exec 9> $A/cv/.ours.lock; flock -n 9 || exit 0
log() { echo "[cv-ours $(date '+%F %T')] $*" | tee -a $L; }
if [ ! -s $A/cv/ours_start.txt ]; then
  $T - > $A/cv/ours_start.txt <<'PY'
import pandas as pd, os
E = "/mnt/data1/kamil_research/experiments"; R = "/home/kamilabdelali/brats2021/results/brats"
d = pd.read_csv(f"{R}/postproc_sweep_dice_ep253_amend5check_val.csv"); top = d.Dice_Mean.max()
cand = {"ep253": (float(d[d.Dice_Mean >= top - 0.001].sort_values("HD95_mean").iloc[0].Dice_Mean),
                  f"{E}/aug_cosine_repro1/checkpoints/aug_cosine_repro1_ep253_final.pth", "nib")}
for t in ("runA",):   # Amendment 11: only models evaluated on 28 Sep
    p = f"{E}/eval_v2/work_{t}/selection.txt"
    if os.path.exists(p): f = open(p).read().split(); cand[t] = (float(f[1]), f[4], f[6] if len(f) > 6 else "nib")
order = sorted(cand, key=lambda k: -cand[k][0])
for k in order[:2]: print(k, cand[k][1], f"{cand[k][0]:.4f}", cand[k][2])
PY
fi
log "M1/M2 by BraTS VAL: $(tr '\n' ';' < $A/cv/ours_start.txt)"
flags() { $T - "$1" <<'PY'
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False); sd = ck["model_state"]; cfg = ck.get("config", {}) or {}
down = cfg.get("downsample") or {4: "dwt", 8: "dwt3d"}[sd["conv1_0.conv.0.weight"].shape[1] // sd["conv0_0.conv.0.weight"].shape[0]]
out = ["--downsample", down, "--base-filters", str(cfg.get("base_filters") or sd["conv0_0.conv.0.weight"].shape[0])]
if any(k.startswith("ds_heads.") for k in sd): out.append("--deep-supervision")
out += ["--norm", cfg.get("norm") or ("batch" if any(k.endswith("running_mean") for k in sd) else "instance"),
        "--normalisation", cfg.get("normalisation") or "minmax"]
print(" ".join(out))
PY
}
ep() { [ -f $1 ] && $T -c "import torch;print(int(torch.load('$1',map_location='cpu',weights_only=False).get('epoch',0)))" 2>/dev/null || echo 0; }
train() {  # fold model_label start_ckpt
  local k=$1 M=$2 SCK=$3 D=$A/cv/fold$1; local TAG=adapt_cv_${M,,}_f$k; local CKD=$D/ckpt_$M; mkdir -p $CKD
  local FL; read -r -a FL <<< "$(flags $SCK)"
  for att in 1 2 3 4; do
    [ "$(ep $CKD/${TAG}_last.pth)" -ge 150 ] && return 0
    until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge 11000 ]; do sleep 300; done
    if [ -f $CKD/${TAG}_last.pth ]; then INIT=(--resume $CKD/${TAG}_last.pth); else INIT=(--resume $SCK --warm-restart); fi
    log "fold $k $M train (attempt $att) flags: ${FL[*]}"
    ( cd $A/code_ours_v2 && ulimit -n "$(ulimit -Hn)" && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $T -u scripts/train_brats.py --epochs 150 --tag $TAG --num-workers 6 \
        --batch-size 2 --accum-steps 4 --scheduler cosine --seed 42 --save-dir $CKD "${FL[@]}" \
        --data-root /mnt/data1/kamil_research/brats_africa/brats21fmt --train-ids $D/train_ids.txt --val-ids $D/val_ids.txt \
        "${INIT[@]}" ) >> $D/train_$M.log 2>&1
  done
  [ "$(ep $CKD/${TAG}_last.pth)" -ge 150 ]
}
infer() {  # fold model_label
  local k=$1 M=$2 D=$A/cv/fold$1; local TAG=adapt_cv_${M,,}_f$k; local F=$D/ckpt_$M/best_evaluated.pth
  local O; O=$(sed -n "$([ $M = M1 ] && echo 1 || echo 2)p" $A/cv/ours_start.txt | awk '{print $4}'); O=${O:-nib}
  [ -f $F ] || { cp -p $D/ckpt_$M/${TAG}_best.pth $F; chmod a-w $F; }
  for s in val test; do
    [ "$(ls $D/probs_${M}_$s/*.npz 2>/dev/null | wc -l)" -eq "$(grep -c . $D/${s}_ids.txt)" ] && continue
    $T $A/eval_code/scripts/infer_probs.py $F $D/${s}_ids.txt $D/probs_${M}_$s $O >> $D/infer_$M.log 2>&1 || return 1
  done
}
read -r M1N M1C _ < <(sed -n 1p $A/cv/ours_start.txt); read -r M2N M2C _ < <(sed -n 2p $A/cv/ours_start.txt)
# Amendment 11: M1 on all five folds first, then M2, then per-fold selection.
for M in M1 M2; do
  [ $M = M1 ] && SCK=$M1C || SCK=$M2C
  for k in 0 1 2 3 4; do
    D=$A/cv/fold$k; [ -s $D/per_case_ours_sel.csv ] && continue
    train $k $M $SCK || { log "FAILED fold $k $M training"; continue; }
    infer $k $M || { log "FAILED fold $k $M inference"; continue; }
    log "fold $k $M: trained + probabilities saved"
  done
done
n2=0; for k in 0 1 2 3 4; do D=$A/cv/fold$k
  [ "$(ls $D/probs_M2_test/*.npz 2>/dev/null | wc -l)" -eq "$(grep -c . $D/test_ids.txt)" ] && \
  [ "$(ls $D/probs_M2_val/*.npz 2>/dev/null | wc -l)" -eq "$(grep -c . $D/val_ids.txt)" ] && n2=$((n2+1)); done
if [ $n2 -eq 5 ]; then MODE=M1M2; else MODE=M1only; log "M2 complete on $n2/5 folds -> M1-only selection (Amendment 11.4)"; fi
for k in 0 1 2 3 4; do
  D=$A/cv/fold$k; [ -s $D/per_case_ours_sel.csv ] && continue
  CV_MODE=$MODE $T $A/cv/cv_fold_select.py $k >> $D/select.log 2>&1 || { log "FAILED fold $k selection"; continue; }
  log "fold $k: $(grep -E 'TEST mean' $D/select.log | tr '\n' ' ')"
done
log "=== ours CV done ($MODE) ==="
