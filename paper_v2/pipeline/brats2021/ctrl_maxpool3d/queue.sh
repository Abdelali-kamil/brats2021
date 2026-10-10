#!/bin/bash
# Amendment 21 queue: train the non-wavelet control -> evaluate exactly like Run C (eval_run.sh: VAL selection, TEST once)
# -> BraTS-Africa zero-shot -> runtime/GFLOPs -> paired comparison with Run C. Idempotent (each step skips if its output
# exists), flock-guarded, restarted by the @reboot hook (scripts/resume_all_on_boot.sh).
set -uo pipefail
E=/mnt/data1/kamil_research/experiments; A=$E/ctrl_maxpool3d; TAG=ctrl_maxpool3d
T=/home/kamilabdelali/anaconda3/envs/torchfix/bin/python
LOG=$A/queue.log; exec 9> $A/.lock_queue; flock -n 9 || exit 0
log() { echo "[ctrl $(date '+%F %T')] $*" | tee -a $LOG; }
log "=== queue start ==="
if [ ! -e $A/TRAINED ]; then
  [ -e $A/FAILED ] && { log "training failed twice (collapse rule) -> nothing to evaluate"; exit 0; }
  bash $A/run.sh
  [ -e $A/TRAINED ] || { log "training not complete (see logs/) -> stop; the boot hook will resume"; exit 0; }
fi
log "trained: best $(python3 -c "print(open('$A/logs/$TAG.log').read().count('New best'))") improvements; evaluating"
if [ ! -s /home/kamilabdelali/brats2021/results/brats/summary_${TAG}_test_tuned_recomputed.csv ]; then
  bash $E/eval_v2/eval_run.sh $TAG $A/checkpoints/${TAG}_best.pth || { log "eval_run.sh failed"; exit 1; }
fi
log "BraTS 2021: $(grep -E '\[4/4\] MEAN' $E/eval_v2/eval_$TAG.log | tail -1 | cut -c1-160)"
if [ ! -s $E/external_africa/results/per_case_ours_$TAG.csv ]; then
  bash $E/external_africa/run_external_candidate.sh $TAG || log "BraTS-Africa run failed"
fi
if [ ! -s $E/efficiency/results/ours_$TAG.json ]; then
  until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 300; done
  CK=$(cut -d' ' -f5 $E/eval_v2/work_$TAG/selection.txt); O=$(cut -d' ' -f7 $E/eval_v2/work_$TAG/selection.txt)
  AX=(); [ "$O" = both ] && AX=(--axis-both)
  (cd $E/efficiency && $T bench_ours.py --name ours_$TAG --downsample maxpool3d_matched --base-filters 24 --ckpt $CK "${AX[@]}" \
     && $T bench_ours.py --name ours_${TAG}_noTTA --downsample maxpool3d_matched --base-filters 24 --ckpt $CK --no-tta) >> $A/bench.log 2>&1 \
     || log "benchmark failed"
fi
$T $A/compare_ctrl.py > $A/results_amendment21.txt 2>&1 && log "comparison written: $A/results_amendment21.txt" && touch $A/DONE
log "=== queue finished ==="
