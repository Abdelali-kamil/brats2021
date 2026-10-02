#!/bin/bash
# PROTOCOL_v2 Amendment 17 queue: after the CV from Run D (one GPU) -> Run D2 training (warm restart, +200 ep)
#   -> eval runD2 (VAL nib/both -> TEST once) -> runD2_rot (VAL -> TEST once) -> Africa zero-shot of the VAL-better
#   -> gate (VAL only): best VAL of {runD2, runD2_rot} > best VAL of {runD, runD_rot}
#      pass -> marker runD2_gate_pass (releases queue_cv_runD2.sh) + ensemble3 (VAL -> TEST once).
# Idempotent (markers), flock, in the @reboot hook.
set -uo pipefail
E=/mnt/data1/kamil_research/experiments; Q=$E/queue_v2_state; LOG=$E/queue_v6.log; A=$E/adapt_africa; V=$E/eval_v2
PY=/home/kamilabdelali/anaconda3/envs/torchfix/bin/python
exec 9> $Q/.lock_v6; flock -n 9 || exit 0
log() { echo "[queue6 $(date '+%F %T')] $*" | tee -a $LOG; }
log "=== queue_v6 start ==="
until grep -q "=== CV from Run D done" $A/cv/cv.log; do sleep 600; done   # one GPU
ep() { $PY -c "import torch;print(int(torch.load('$E/runD2_warm/checkpoints/runD2_last.pth',map_location='cpu',weights_only=False).get('epoch',0)))" 2>/dev/null || echo 0; }
[ "$(ep)" -ge 200 ] || { log "Run D2 training starts/resumes (epoch $(ep))"; bash $E/runD2_warm/run.sh; }
[ "$(ep)" -ge 200 ] || { log "Run D2 training incomplete (epoch $(ep)) -> stop"; exit 1; }
log "Run D2 complete (epoch 200 of the new cycle; 400 in total)"
if [ ! -e $Q/evalD2_done ]; then
  bash $V/eval_run.sh runD2 $E/runD2_warm/checkpoints/runD2_best.pth && touch $Q/evalD2_done \
    && log "Run D2 evaluated: $(cat $V/work_runD2/selection.txt)"
fi
[ -e $Q/evalD2_done ] || { log "Run D2 evaluation failed -> stop"; exit 1; }
if [ ! -e $Q/evalD2rot_done ]; then
  bash $V/eval_run_rot2.sh runD2_rot runD2 && grep -q "=== done ===" $V/eval_runD2_rot.log && touch $Q/evalD2rot_done \
    && log "runD2_rot evaluated: $(cat $V/work_runD2_rot/selection.txt)"
fi
vd() { [ -s $V/work_$1/selection.txt ] && cut -d' ' -f2 $V/work_$1/selection.txt || echo 0; }
W=runD2; python3 -c "import sys; sys.exit(0 if float('$(vd runD2_rot)') > float('$(vd runD2)') else 1)" && W=runD2_rot
[ -s $E/external_africa/results/per_case_ours_$W.csv ] || { log "BraTS-Africa zero-shot: $W"; bash $E/external_africa/run_external_candidate.sh $W; }
log "Africa zero-shot $W: $(grep MEAN $E/external_africa/ours_$W.log | tail -2 | tr '\n' ' ')"
NEW=$(python3 -c "print(max(float('$(vd runD2)'), float('$(vd runD2_rot)')))"); OLD=$(python3 -c "print(max(float('$(vd runD)'), float('$(vd runD_rot)')))")
if python3 -c "import sys; sys.exit(0 if $NEW > $OLD else 1)"; then
  log "gate PASSED: best VAL Run D2 $NEW > Run D $OLD -> CV from Run D2 + ensemble3"; touch $Q/runD2_gate_pass
  if [ ! -e $Q/ensemble3_done ]; then
    $PY $V/ensemble3_search.py > $V/ensemble3/ensemble3.log 2>&1 && touch $Q/ensemble3_done \
      && log "ensemble3: $(tail -8 $V/ensemble3/ensemble3.log | grep -E 'MEAN|VAL-selected' | tr '\n' ' ')"
  fi
else
  log "gate FAILED: best VAL Run D2 $NEW <= Run D $OLD -> no CV from Run D2, no ensemble3"; touch $Q/runD2_gate_fail
fi
log "=== queue_v6 finished ==="
