#!/bin/bash
# Run D2 (PROTOCOL_v2 Amendment 17): Run D's final weights (epoch 200) + one more 200-epoch cosine cycle
# (warm restart: fresh AdamW at 2e-4), identical flags -> 400 epochs in total. Fault-tolerant: the first
# attempt warm-starts from runD_in_z_last.pth, later attempts resume runD2_last.pth.
set -uo pipefail
E=/mnt/data1/kamil_research/experiments; A=$E/runD2_warm; TAG=runD2
CKPT=$A/checkpoints; LOGD=$A/logs; BACKUP=/home/kamilabdelali/checkpoint_backups/$TAG
SRC=$E/runD_in_z/checkpoints/runD_in_z_last.pth
PY=/home/kamilabdelali/anaconda3/envs/torchfix/bin/python; TARGET=200; FREE_MIB="${FREE_MIB:-20000}"
read -r -a ARCH < $E/runD_in_z/arch_flags.txt
mkdir -p "$CKPT" "$BACKUP" "$LOGD"
pgrep -f "train_brats.py .*--tag $TAG " >/dev/null && { echo "$TAG already training"; exit 0; }
MASTER="$LOGD/run_$(date +%Y%m%d_%H%M%S).log"; exec > >(tee -a "$MASTER") 2>&1
cd "$E/runD_in_z/code"; ulimit -n "$(ulimit -Hn)"
cur_epoch() { [ -f "$CKPT/${TAG}_last.pth" ] || { echo 0; return; }
  "$PY" -c "import torch;print(int(torch.load('$CKPT/${TAG}_last.pth',map_location='cpu',weights_only=False).get('epoch',0)))" 2>/dev/null || echo 0; }
echo "architecture flags: ${ARCH[*]} --norm instance --normalisation zscore --aug strong; warm start from $SRC"
for att in $(seq 1 10); do
  ep=$(cur_epoch); echo "=== attempt $att (last completed epoch=$ep, target=$TARGET) $(date -Is) ==="
  [ "${ep:-0}" -ge "$TARGET" ] && { echo "already at target"; break; }
  while :; do free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)
    [ "${free:-0}" -ge "$FREE_MIB" ] && break; echo "[wait] GPU free=${free}MiB $(date -Is)"; sleep 300; done
  if [ -f "$CKPT/${TAG}_last.pth" ]; then INIT=(--resume "$CKPT/${TAG}_last.pth"); else INIT=(--resume "$SRC" --warm-restart); fi
  "$PY" -u scripts/train_brats.py --epochs "$TARGET" --tag "$TAG" --num-workers 8 --batch-size 2 --accum-steps 4 \
      --scheduler cosine --seed 42 "${ARCH[@]}" --norm instance --normalisation zscore --aug strong \
      --save-dir "$CKPT" "${INIT[@]}" >> "$LOGD/$TAG.log" 2>&1
  rc=$?; cp -pu "$CKPT"/${TAG}_*.pth "$BACKUP/" 2>/dev/null || true
  ep=$(cur_epoch); echo "=== attempt $att exited rc=$rc, now at epoch=$ep $(date -Is) ==="
  [ "$rc" -eq 0 ] && [ "${ep:-0}" -ge "$TARGET" ] && { echo "TRAINING COMPLETE at epoch $ep"; break; }
  sleep 120
done
