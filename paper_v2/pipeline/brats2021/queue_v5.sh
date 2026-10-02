#!/bin/bash
# PROTOCOL_v2 Amendment 15 queue. Starts only after tonight's jobs are finished (user: "wait until after 6am"):
#   ensemble v1 done + Run C Africa CV done (+13b)  ->  speed benchmark on the idle GPU  ->  Run D training
#   -> eval runD (VAL nib/both -> TEST once) -> [CV from Run D starts: queue_cv_runD.sh]
#   -> runD_rot (VAL -> TEST once) -> Africa zero-shot of the VAL-better of {runD, runD_rot}
#   -> ensemble2 (VAL -> TEST once). Idempotent (markers), flock, in the @reboot hook.
set -uo pipefail
E=/mnt/data1/kamil_research/experiments; Q=$E/queue_v2_state; LOG=$E/queue_v5.log; A=$E/adapt_africa
PY=/home/kamilabdelali/anaconda3/envs/torchfix/bin/python
exec 9> $Q/.lock_v5; flock -n 9 || exit 0
log() { echo "[queue5 $(date '+%F %T')] $*" | tee -a $LOG; }
log "=== queue_v5 start ==="
until [ -e $Q/ensemble_done ] && grep -q "=== CV from Run C done" $A/cv/cv.log && grep -q "=== 13b done" $A/cv/cv.log; do sleep 600; done
if [ ! -s $E/efficiency/results/ours_runC_noTTA.json ]; then
  until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 120; done
  log "speed benchmark (idle GPU)"; bash $E/efficiency/run_bench_runC.sh; log "benchmark done: $(ls $E/efficiency/results | tr '\n' ' ')"
fi
ep() { $PY -c "import torch;print(int(torch.load('$E/runD_in_z/checkpoints/runD_in_z_last.pth',map_location='cpu',weights_only=False).get('epoch',0)))" 2>/dev/null || echo 0; }
[ "$(ep)" -ge 200 ] || { log "Run D training starts/resumes (epoch $(ep))"; bash $E/runD_in_z/run.sh; }   # Amendment 15b: 200 epochs
[ "$(ep)" -ge 200 ] || { log "Run D training incomplete (epoch $(ep)) -> stop"; exit 1; }
log "Run D complete (epoch 200)"
if [ ! -e $Q/evalD_done ]; then
  bash $E/eval_v2/eval_run.sh runD $E/runD_in_z/checkpoints/runD_in_z_best.pth && touch $Q/evalD_done \
    && log "Run D evaluated: $(cat $E/eval_v2/work_runD/selection.txt)"
fi
[ -e $Q/evalD_done ] || { log "Run D evaluation failed -> stop"; exit 1; }
if [ ! -e $Q/evalDrot_done ]; then
  bash $E/eval_v2/eval_run_rot2.sh runD_rot runD && grep -q "=== done ===" $E/eval_v2/eval_runD_rot.log && touch $Q/evalDrot_done \
    && log "runD_rot evaluated: $(cat $E/eval_v2/work_runD_rot/selection.txt)"
fi
W=runD; [ -e $Q/evalDrot_done ] && python3 -c "import sys; a=float(open('$E/eval_v2/work_runD/selection.txt').read().split()[1]); b=float(open('$E/eval_v2/work_runD_rot/selection.txt').read().split()[1]); sys.exit(0 if b > a else 1)" && W=runD_rot
[ -s $E/external_africa/results/per_case_ours_$W.csv ] || { log "BraTS-Africa zero-shot: $W"; bash $E/external_africa/run_external_candidate.sh $W; }
log "Africa zero-shot $W: $(grep MEAN $E/external_africa/ours_$W.log | tail -2 | tr '\n' ' ')"
# Amendment 15b: ensemble2 (and the CV from Run D) only after the user decides to continue
until [ -e $Q/runD_continue ]; do sleep 900; done
if [ ! -e $Q/ensemble2_done ]; then
  log "ensemble2 search (VAL) -> TEST once"
  $PY $E/eval_v2/ensemble2_search.py > $E/eval_v2/ensemble2/ensemble2.log 2>&1 && touch $Q/ensemble2_done \
    && log "ensemble2: $(tail -8 $E/eval_v2/ensemble2/ensemble2.log | grep -E 'MEAN|VAL-selected' | tr '\n' ' ')"
fi
log "=== queue_v5 finished ==="
