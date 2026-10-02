#!/bin/bash
# Amendment 4, nnU-Net: official -pretrained_weights fine-tuning, identical plans to
# Dataset137, nnUNetTrainer_adapt29 (29 epochs = 14,500 samples), fold 0 = TRAIN 48 / VAL 12.
set -uo pipefail
N=/mnt/data1/kamil_research/baselines/nnunet; B=/home/kamilabdelali/anaconda3/envs/nnunet_infer/bin; A=/mnt/data1/kamil_research/experiments/adapt_africa
export nnUNet_raw=$N/nnUNet_raw nnUNet_preprocessed=$N/nnUNet_preprocessed nnUNet_results=$N/nnUNet_results
F=$N/nnUNet_results/Dataset138_BraTSAfricaAdapt/nnUNetTrainer_adapt29__nnUNetPlans__3d_fullres/fold_0
if [ ! -f $F/checkpoint_final.pth ]; then
  if [ -f $F/checkpoint_latest.pth ]; then CONT=(--c); else CONT=(-pretrained_weights $N/nnUNet_results/Dataset137_BraTS2021/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_all/checkpoint_final.pth); fi
  $B/nnUNetv2_train 138 3d_fullres 0 -tr nnUNetTrainer_adapt29 "${CONT[@]}" >> $A/nnunet/train.log 2>&1 || exit 1
fi
$B/nnUNetv2_predict -i $N/nnUNet_raw/Dataset138_BraTSAfricaAdapt/imagesTs -o $A/nnunet/pred_test -d 138 -c 3d_fullres -f 0 \
   -tr nnUNetTrainer_adapt29 -p nnUNetPlans -chk checkpoint_final.pth -device cuda >> $A/nnunet/predict.log 2>&1
