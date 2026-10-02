#!/bin/bash
# Amendment 6: baselines' 5-fold CV adaptation. nnU-Net folds 0-4, then Swin folds 0-4.
# Idempotent (skips folds with 19 test predictions), flock-guarded, in the @reboot hook.
# Waits until Run A's BraTS evaluation and its BraTS-Africa zero-shot run are done
# (GPU memory beside Run C).
set -uo pipefail
A=/mnt/data1/kamil_research/experiments/adapt_africa; E=/mnt/data1/kamil_research/experiments; L=$A/cv/cv.log
exec 9> $A/cv/.baselines.lock; flock -n 9 || exit 0
log() { echo "[cv $(date '+%F %T')] $*" | tee -a $L; }
N=/mnt/data1/kamil_research/baselines/nnunet; B=/home/kamilabdelali/anaconda3/envs/nnunet_infer/bin
export nnUNet_raw=$N/nnUNet_raw nnUNet_preprocessed=$N/nnUNet_preprocessed nnUNet_results=$N/nnUNet_results
S=/home/kamilabdelali/brats2021/baselines/swin_unetr/.venv_overlay/bin/python
R=/home/kamilabdelali/brats2021/baselines/repos/swin_unetr/SwinUNETR/BRATS21
until [ -e $E/queue_v2_state/evalA_done ] && [ -s $E/external_africa/results/per_case_ours_runA.csv ]; do sleep 600; done
wait_gpu() { until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge 11000 ]; do sleep 300; done; }
npred() { ls $1/*.nii.gz 2>/dev/null | wc -l; }
log "=== baselines CV start ==="
for k in 0 1 2 3 4; do
  F=$N/nnUNet_results/Dataset139_BraTSAfricaCV/nnUNetTrainer_adaptcv19__nnUNetPlans__3d_fullres/fold_$k
  [ "$(npred $A/cv/fold$k/pred_nnunet)" -eq 19 ] && continue
  wait_gpu; log "nnU-Net fold $k"
  if [ ! -f $F/checkpoint_final.pth ]; then
    if [ -f $F/checkpoint_latest.pth ]; then C=(--c); else C=(-pretrained_weights $N/nnUNet_results/Dataset137_BraTS2021/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_all/checkpoint_final.pth); fi
    $B/nnUNetv2_train 139 3d_fullres $k -tr nnUNetTrainer_adaptcv19 "${C[@]}" >> $A/cv/nnunet_f$k.log 2>&1 || { log "FAILED nnU-Net train fold $k"; continue; }
  fi
  $B/nnUNetv2_predict -i $A/cv/fold$k/nnunet_in -o $A/cv/fold$k/pred_nnunet -d 139 -c 3d_fullres -f $k -tr nnUNetTrainer_adaptcv19 \
     -p nnUNetPlans -chk checkpoint_final.pth -device cuda >> $A/cv/nnunet_f$k.log 2>&1 || log "FAILED nnU-Net predict fold $k"
  log "nnU-Net fold $k: $(npred $A/cv/fold$k/pred_nnunet) predictions"
done
for k in 0 1 2 3 4; do
  [ "$(npred $A/cv/fold$k/pred_swin)" -eq 19 ] && continue
  wait_gpu; log "Swin fold $k"
  if [ ! -f $R/runs/adapt_cv_swin_f$k/model_final.pt ]; then
    rm -rf $R/runs/adapt_cv_swin_f$k
    ( cd $R && ulimit -n "$(ulimit -Hn)" && CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 $S main.py --json_list $A/cv/fold$k/swin_datalist.json \
      --data_dir /mnt/data1/kamil_research/brats_africa/brats21fmt --fold 0 --feature_size 48 --use_checkpoint --save_checkpoint \
      --max_epochs 150 --val_every 5 --batch_size 1 --workers 4 --logdir adapt_cv_swin_f$k \
      --checkpoint $A/swin/init_from_brats_ep199.pt ) >> $A/cv/swin_f$k.log 2>&1 || { log "FAILED Swin train fold $k"; continue; }
  fi
  $S $A/cv/infer_swin_fold.py $k >> $A/cv/swin_f$k.log 2>&1 || log "FAILED Swin infer fold $k"
  log "Swin fold $k: $(npred $A/cv/fold$k/pred_swin) predictions"
done
log "=== baselines CV done ==="
