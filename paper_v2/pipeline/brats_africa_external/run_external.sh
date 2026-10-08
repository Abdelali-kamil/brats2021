#!/bin/bash
# BraTS-Africa external evaluation (PROTOCOL_v2 Amendments 1-2), baselines + ep253.
# Run A / Run C are added by run_external_candidate.sh once their BraTS-val
# selection exists. Steps stop at the first failure; finished steps are skipped.
set -uo pipefail
X=/mnt/data1/kamil_research/experiments/external_africa; cd $X; mkdir -p results
T=/home/kamilabdelali/anaconda3/envs/torchfix/bin/python
N=/home/kamilabdelali/anaconda3/envs/nnunet_infer
S=/home/kamilabdelali/brats2021/baselines/swin_unetr/.venv_overlay/bin/python
LOG=$X/run_external.log
log() { echo "[ext $(date '+%F %T')] $*" | tee -a "$LOG"; }
die() { log "ABORT: $*"; exit 1; }

[ -s cases.csv ] || { $T prepare.py >> $LOG 2>&1 || die prepare; }
[ -s cases_gated.csv ] || { log "overlap gate"; $T gate_overlap.py >> $LOG 2>&1 || die gate; }
log "gate: $(python3 -c "import pandas as p;d=p.read_csv('$X/cases_gated.csv');print(int(d.overlap_flag.sum()),'flagged of',len(d))")"

# ours, ep253, with its BraTS-val-selected post-processing (eval_ep253: WT largest, x1)
[ -s results/per_case_ours_ep253.csv ] || { log "ours ep253"; $T eval_ours_external.py ours_ep253 \
  /mnt/data1/kamil_research/experiments/aug_cosine_repro1/checkpoints/aug_cosine_repro1_ep253_final.pth largest 1 \
  >> $X/ours_ep253.log 2>&1 || die "ours ep253"; }

# Swin UNETR (model.pt, upstream recipe)
[ -s results/per_case_swin_unetr.csv ] || { log "Swin UNETR"; $S infer_swin_external.py >> $X/swin.log 2>&1 || die swin
  $T score_labelmaps.py swin_unetr $X/pred_swin brats21 >> $X/swin.log 2>&1 || die "swin score"; }

# nnU-Net (fold all, checkpoint_final, native mirroring TTA; nnUNetv2_predict applies no post-processing)
if [ ! -s results/per_case_nnunet.csv ]; then
  log "nnU-Net"
  export nnUNet_raw=/mnt/data1/kamil_research/baselines/nnunet/nnUNet_raw
  export nnUNet_preprocessed=/mnt/data1/kamil_research/baselines/nnunet/nnUNet_preprocessed
  export nnUNet_results=/mnt/data1/kamil_research/baselines/nnunet/nnUNet_results
  $N/bin/nnUNetv2_predict -i $X/nnunet_in -o $X/pred_nnunet -d 137 -c 3d_fullres -f all \
     -tr nnUNetTrainer -p nnUNetPlans -chk checkpoint_final.pth -device cuda >> $X/nnunet.log 2>&1 || die nnunet
  $T score_labelmaps.py nnunet $X/pred_nnunet nnunet_regions >> $X/nnunet.log 2>&1 || die "nnunet score"
fi
log "=== baselines + ep253 done ==="
