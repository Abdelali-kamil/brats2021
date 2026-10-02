#!/bin/bash
# Efficiency timing of Run C (3D DWT, F=24) under its VAL-selected inference protocol.
# Waits until the ensemble is done and the GPU is otherwise idle. Idempotent.
set -uo pipefail
E=/mnt/data1/kamil_research/experiments; B=$E/efficiency; cd $B
[ -s $B/results/ours_runC.json ] && exit 0
until [ -e $E/queue_v2_state/ensemble_done ] && [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 300; done
L=$B/bench_runC_$(date +%Y%m%d_%H%M%S).log
{ echo "start $(date -Is)"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv; } >> $L
T=/home/kamilabdelali/anaconda3/envs/torchfix/bin/python
CK=$(cut -d' ' -f5 $E/eval_v2/work_runC/selection.txt); O=$(cut -d' ' -f7 $E/eval_v2/work_runC/selection.txt)
AX=(); [ "$O" = both ] && AX=(--axis-both)
$T bench_ours.py --name ours_runC --downsample dwt3d --base-filters 24 --ckpt $CK "${AX[@]}" >> $L 2>&1
[ -s $E/eval_v2/work_runC_rot/selection.txt ] && $T bench_ours.py --name ours_runC_rot --downsample dwt3d --base-filters 24 --ckpt $CK "${AX[@]}" --rot-tta >> $L 2>&1
$T bench_ours.py --name ours_runC_noTTA --downsample dwt3d --base-filters 24 --ckpt $CK --no-tta >> $L 2>&1
{ echo "end $(date -Is)"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv; } >> $L
