#!/bin/bash
# Amendment 19: train + evaluate three extra baselines (3D U-Net, SegResNet, UNETR) one after another.
# Each: train 300 epochs (official BRATS21 pipeline, only the network differs) -> BraTS 2021 test scored ONCE ->
# BraTS-Africa zero-shot -> features for classification. Then classification for all three. Idempotent, flock, @reboot.
set -uo pipefail
X=/home/kamilabdelali/brats2021/baselines/extra3; B=$X/BRATS21; E=/mnt/data1/kamil_research/experiments; Q=$E/queue_v2_state
PY=/home/kamilabdelali/brats2021/baselines/swin_unetr/.venv_overlay/bin/python      # training/inference (as Swin UNETR)
SPY=/home/kamilabdelali/anaconda3/envs/nnunet_infer/bin/python                      # scoring (as Swin UNETR)
T=/home/kamilabdelali/anaconda3/envs/torchfix/bin/python                            # BraTS-Africa scoring + classification
RES=/home/kamilabdelali/brats2021/results/brats; IDS=/home/kamilabdelali/brats2021/baselines/common/test_ids.txt
LOG=$X/queue_extra3.log; exec 9> $Q/.lock_extra3; flock -n 9 || exit 0
log() { echo "[extra3 $(date '+%F %T')] $*" | tee -a $LOG; }
ulimit -n "$(ulimit -Hn)"
log "=== queue_extra3 start ==="
for spec in "unet3d:96:96:96:" "segresnet:224:224:144:--squared_dice --smooth_nr 0 --smooth_dr 1e-5" "unetr:96:96:96:"; do
  IFS=: read -r n rx ry rz extra <<< "$spec"; R=$B/runs/extra3_$n
  if [ ! -e $Q/extra3_${n}_trained ]; then
    for att in 1 2 3; do
      until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge 14000 ]; do sleep 300; done
      L=/home/kamilabdelali/brats2021/logs/extra3_${n}_$(date +%Y%m%d_%H%M%S).log
      log "$n training attempt $att -> $L"
      (cd $B && CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 $PY main.py --json_list /home/kamilabdelali/brats2021/baselines/swin_unetr/datalist_frozen_train1000_val126.json \
         --data_dir /mnt/data1/kamil_research/data/brats2021 --fold 0 --model_name $n --roi_x $rx --roi_y $ry --roi_z $rz $extra \
         --save_checkpoint --max_epochs 300 --val_every 25 --batch_size 1 --workers 8 --logdir extra3_$n) > $L 2>&1
      grep -q "Training Finished" $L && [ -f $R/model.pt ] && { touch $Q/extra3_${n}_trained; log "$n trained: $(grep -m1 'Training Finished' $L)"; break; }
      log "$n training attempt $att failed: $(tail -2 $L | tr '\n' ' ' | cut -c1-200)"; sleep 120
    done
  fi
  [ -e $Q/extra3_${n}_trained ] || { log "$n NOT trained -> skipping its evaluation"; continue; }
  W=/mnt/data1/kamil_research/baselines/extra3/eval/$n; mkdir -p $W
  if [ ! -s $RES/${n}_test_per_case.csv ]; then
    log "$n BraTS 2021 test inference (125, once)"
    CUDA_VISIBLE_DEVICES=0 $PY $X/infer_test125_extra3.py --model_name $n --ckpt $R/model.pt --out $W/test125_pred >> $W/infer.log 2>&1 \
      && $SPY /home/kamilabdelali/brats2021/baselines/swin_unetr/score_swin_test.py $W/test125_pred $IDS $RES/${n}_test ${n}_test >> $W/score.log 2>&1 \
      && log "$n TEST: $(grep -E 'MEAN' $W/score.log | tail -1 | cut -c1-120)" || log "$n test FAILED"
  fi
  if [ ! -s $E/external_africa/results/per_case_$n.csv ]; then
    log "$n BraTS-Africa zero-shot"
    (cd $E/external_africa && CUDA_VISIBLE_DEVICES=0 $PY infer_extra3_external.py $n >> $W/africa_infer.log 2>&1 \
      && $T score_labelmaps.py $n $E/external_africa/pred_$n brats21 >> $W/africa_score.log 2>&1) \
      && log "$n AFRICA: $(grep -E 'MEAN' $W/africa_score.log | tail -2 | tr '\n' ' ' | cut -c1-200)" || log "$n Africa FAILED"
  fi
  [ -s $E/classification_africa/features_$n.csv ] || (cd $E/classification_africa && $T extract_baselines.py $n >> $W/features.log 2>&1) || log "$n features FAILED"
done
if [ ! -s $E/classification_africa/summary_unet3d_segresnet_unetr.txt ]; then
  (cd $E/classification_africa && $T classify.py unet3d segresnet unetr > extra3_cls.log 2>&1) && log "classification: $(grep -E 'logreg' $E/classification_africa/extra3_cls.log | tr '\n' ' ' | cut -c1-300)"
fi
log "=== queue_extra3 finished ==="
