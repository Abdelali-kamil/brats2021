#!/bin/bash
# Amendment 21: non-wavelet control for Run C. Identical to experiments/runC_dwt3d/run.sh except the tag and
# --downsample maxpool3d_matched (2x2x2 max-pool + 1x1x1 projection C -> 8C). Fault-tolerant: resumes from _last.pth.
# Collapse rule (as Amendment 19b): a validation with Dice < 0.05 for ET, TC and WT, or a finished run whose best
# validation Dice is < 0.5, stops the run; it is moved to ~/delete/unused_checkpoints and retrained ONCE in fp32 (--no-amp).
# Safe to launch twice: exits if a trainer for this tag is already running.
set -uo pipefail
A=/mnt/data1/kamil_research/experiments/ctrl_maxpool3d
TAG=ctrl_maxpool3d
CKPT=$A/checkpoints
LOGD=$A/logs
BACKUP=/home/kamilabdelali/checkpoint_backups/$TAG
PARK=/home/kamilabdelali/delete/unused_checkpoints
PY=/home/kamilabdelali/anaconda3/envs/torchfix/bin/python
TARGET=400
MAX_RETRIES="${MAX_RETRIES:-10}"
FREE_MIB="${FREE_MIB:-14000}"
mkdir -p "$BACKUP" "$LOGD" "$CKPT"
if pgrep -f "train_brats.py .*--tag $TAG" >/dev/null; then echo "$TAG already training"; exit 0; fi
MASTER="$LOGD/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$MASTER") 2>&1
cd "$A/code"
ulimit -n "$(ulimit -Hn)"

cur_epoch() {
  [ -f "$CKPT/${TAG}_last.pth" ] || { echo 0; return; }
  "$PY" -c "import torch;print(int(torch.load('$CKPT/${TAG}_last.pth',map_location='cpu',weights_only=False).get('epoch',0)))" 2>/dev/null || echo 0
}
best_metric() {
  [ -f "$CKPT/${TAG}_best.pth" ] || { echo 0; return; }
  "$PY" -c "import torch;print(float(torch.load('$CKPT/${TAG}_best.pth',map_location='cpu',weights_only=False).get('best_metric',0)))" 2>/dev/null || echo 0
}
collapsed_in_log() {   # any [VAL] line with all three region Dice < 0.05
  grep "^\[VAL\] Dice ET/TC/WT:" "$LOGD/$TAG.log" 2>/dev/null | awk -F'[ /|]+' '{if ($6+0<0.05 && $7+0<0.05 && $8+0<0.05) c=1} END{exit !c}'
}
park_and_switch_fp32() {
  mkdir -p "$PARK"; local d="$PARK/${TAG}_collapsed_amp_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$d"
  mv "$CKPT" "$d/checkpoints"; mv "$LOGD/$TAG.log" "$d/" 2>/dev/null; mkdir -p "$CKPT"; touch "$A/USE_FP32"
  echo "=== COLLAPSE: run parked in $d; retraining once from scratch in fp32 $(date -Is) ==="
}

for att in $(seq 1 "$MAX_RETRIES"); do
  ep=$(cur_epoch)
  FP=(); [ -e "$A/USE_FP32" ] && FP=(--no-amp)
  echo "=== attempt $att/$MAX_RETRIES (last completed epoch=$ep, target=$TARGET, ${FP[*]:-AMP}) $(date -Is) ==="
  [ "${ep:-0}" -ge "$TARGET" ] && { echo "already at target"; break; }
  while :; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)
    [ "${free:-0}" -ge "$FREE_MIB" ] && { echo "[ok] GPU free=${free}MiB"; break; }
    echo "[wait] GPU free=${free}MiB < ${FREE_MIB} $(date -Is)"; sleep 300
  done
  RESUME=()
  [ -f "$CKPT/${TAG}_last.pth" ] && RESUME=(--resume "$CKPT/${TAG}_last.pth")
  "$PY" -u scripts/train_brats.py \
      --epochs "$TARGET" --tag "$TAG" \
      --num-workers 8 --batch-size 2 --accum-steps 4 \
      --scheduler cosine --seed 42 --deep-supervision --downsample maxpool3d_matched --base-filters 24 \
      --save-dir "$CKPT" "${RESUME[@]}" "${FP[@]}" >> "$LOGD/$TAG.log" 2>&1 &
  TP=$!
  while kill -0 $TP 2>/dev/null; do
    if collapsed_in_log; then
      echo "collapse detected: $(grep '^\[VAL\]' "$LOGD/$TAG.log" | tail -1)"
      pkill -P $TP 2>/dev/null; kill $TP 2>/dev/null; sleep 30; kill -9 $TP 2>/dev/null; break
    fi
    sleep 120
  done
  wait $TP 2>/dev/null; rc=$?
  cp -pu "$CKPT"/${TAG}_*.pth "$BACKUP/" 2>/dev/null || true
  ep=$(cur_epoch)
  if collapsed_in_log || { [ "${ep:-0}" -ge "$TARGET" ] && python3 -c "import sys; sys.exit(0 if float('$(best_metric)') < 0.5 else 1)"; }; then
    if [ -e "$A/USE_FP32" ]; then echo "=== collapsed again in fp32: control NOT trained $(date -Is) ==="; touch "$A/FAILED"; break; fi
    park_and_switch_fp32; continue
  fi
  echo "=== attempt $att exited rc=$rc, now at epoch=$ep $(date -Is) ==="
  if [ "$rc" -eq 0 ] && [ "${ep:-0}" -ge "$TARGET" ]; then echo "TRAINING COMPLETE at epoch $ep"; touch "$A/TRAINED"; break; fi
  echo "[retry] waiting 120 s for the GPU to settle"; sleep 120
done
echo "=== wrapper finished, final epoch=$(cur_epoch) $(date -Is) ==="
