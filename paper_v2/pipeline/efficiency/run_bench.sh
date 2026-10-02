#!/bin/bash
# Efficiency benchmark; run ONLY on an otherwise idle GPU (records what else is on it).
set -uo pipefail
B=/mnt/data1/kamil_research/experiments/efficiency; cd $B
L=$B/bench_$(date +%Y%m%d_%H%M%S).log
{ echo "start $(date -Is)"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv; } >> $L
T=/home/kamilabdelali/anaconda3/envs/torchfix/bin/python
$T bench_ours.py --name ours_ep253 --ckpt /mnt/data1/kamil_research/experiments/aug_cosine_repro1/checkpoints/aug_cosine_repro1_ep253_final.pth >> $L 2>&1
$T bench_ours.py --name ours_ep253_noTTA --no-tta --ckpt /mnt/data1/kamil_research/experiments/aug_cosine_repro1/checkpoints/aug_cosine_repro1_ep253_final.pth >> $L 2>&1
$T bench_ours.py --name ours_dwt3d_arch --downsample dwt3d >> $L 2>&1
$T bench_ours.py --name ours_dwt3d_arch_noTTA --downsample dwt3d --no-tta >> $L 2>&1
/home/kamilabdelali/brats2021/baselines/swin_unetr/.venv_overlay/bin/python bench_swin.py >> $L 2>&1
/home/kamilabdelali/anaconda3/envs/nnunet_infer/bin/python bench_nnunet.py >> $L 2>&1
{ echo "end $(date -Is)"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv; } >> $L
