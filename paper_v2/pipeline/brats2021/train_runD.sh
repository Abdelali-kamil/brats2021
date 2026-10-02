#!/bin/bash
# Run D (PROTOCOL_v2 Amendment 15): Run C architecture (arch_flags.txt) + InstanceNorm + z-score
# inputs + STRONG augmentation, 200 epochs (Amendment 15b), fault-tolerant (resume from _last.pth after a GPU fault).
set -uo pipefail
A=/mnt/data1/kamil_research/experiments/runD_in_z; TAG=runD_in_z
CKPT=$A/checkpoints; LOGD=$A/logs; BACKUP=/home/kamilabdelali/checkpoint_backups/$TAG
PY=/home/kamilabdelali/anaconda3/envs/torchfix/bin/python; TARGET=200; FREE_MIB="${FREE_MIB:-20000}"
[ -s $A/arch_flags.txt ] || { echo "no arch_flags.txt"; exit 1; }
read -r -a ARCH < $A/arch_flags.txt
mkdir -p "$BACKUP" "$LOGD"
pgrep -f "train_brats.py .*--tag $TAG" >/dev/null && { echo "$TAG already training"; exit 0; }
MASTER="$LOGD/run_$(date +%Y%m%d_%H%M%S).log"; exec > >(tee -a "$MASTER") 2>&1
cd "$A/code"; ulimit -n "$(ulimit -Hn)"
cur_epoch() { [ -f "$CKPT/${TAG}_last.pth" ] || { echo 0; return; }
  "$PY" -c "import torch;print(int(torch.load('$CKPT/${TAG}_last.pth',map_location='cpu',weights_only=False).get('epoch',0)))" 2>/dev/null || echo 0; }
echo "architecture flags: ${ARCH[*]} --norm instance --normalisation zscore --aug strong"
for att in $(seq 1 10); do
  ep=$(cur_epoch); echo "=== attempt $att (last completed epoch=$ep, target=$TARGET) $(date -Is) ==="
  [ "${ep:-0}" -ge "$TARGET" ] && { echo "already at target"; break; }
  while :; do free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)
    [ "${free:-0}" -ge "$FREE_MIB" ] && break; echo "[wait] GPU free=${free}MiB $(date -Is)"; sleep 300; done
  RESUME=(); [ -f "$CKPT/${TAG}_last.pth" ] && RESUME=(--resume "$CKPT/${TAG}_last.pth")
  "$PY" -u scripts/train_brats.py --epochs "$TARGET" --tag "$TAG" --num-workers 8 --batch-size 2 --accum-steps 4 \
      --scheduler cosine --seed 42 "${ARCH[@]}" --norm instance --normalisation zscore --aug strong \
      --save-dir "$CKPT" "${RESUME[@]}" >> "$LOGD/$TAG.log" 2>&1
  rc=$?; cp -pu "$CKPT"/${TAG}_*.pth "$BACKUP/" 2>/dev/null || true
  ep=$(cur_epoch); echo "=== attempt $att exited rc=$rc, now at epoch=$ep $(date -Is) ==="
  [ "$rc" -eq 0 ] && [ "${ep:-0}" -ge "$TARGET" ] && { echo "TRAINING COMPLETE at epoch $ep"; break; }
  sleep 120
done
