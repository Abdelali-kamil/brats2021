#!/bin/bash
# PROTOCOL_v2 Amendment 12 queue (replaces queue_v3): runC_rot evaluation -> ensemble search over
# {ep253, runA, runC, runC_rot} (VAL) -> TEST once. If runC_rot is not evaluated by
# 2026-09-30 08:00, the ensemble runs without it (ensemble_search only uses complete candidates).
set -uo pipefail
E=/mnt/data1/kamil_research/experiments; Q=$E/queue_v2_state; LOG=$E/queue_v4.log
PY=/home/kamilabdelali/anaconda3/envs/torchfix/bin/python
exec 9> $Q/.lock_v4; flock -n 9 || exit 0
log() { echo "[queue4 $(date '+%F %T')] $*" | tee -a $LOG; }
log "=== queue_v4 start ==="
until [ -e $Q/evalC_done ]; do sleep 600; done
DEADLINE=$(date -d "2026-09-30 08:00" +%s)
if [ ! -e $Q/evalCrot_done ]; then
  bash $E/eval_v2/eval_run_rot.sh & RP=$!
  while kill -0 $RP 2>/dev/null && [ "$(date +%s)" -lt "$DEADLINE" ]; do sleep 60; done
  if kill -0 $RP 2>/dev/null; then log "runC_rot not done by the deadline -> stopped; ensemble without it"
    pkill -P $RP; kill $RP
  elif grep -q "=== done ===" $E/eval_v2/eval_runC_rot.log; then touch $Q/evalCrot_done
    log "runC_rot evaluated: $(cat $E/eval_v2/work_runC_rot/selection.txt)"
  else log "runC_rot evaluation failed -> ensemble without it"; fi
fi
if [ ! -e $Q/ensemble_done ]; then
  mkdir -p $E/eval_v2/ensemble; log "ensemble search (VAL) -> TEST once"
  $PY $E/eval_v2/ensemble_search.py > $E/eval_v2/ensemble/ensemble.log 2>&1 && touch $Q/ensemble_done \
    && log "ensemble: $(tail -8 $E/eval_v2/ensemble/ensemble.log | grep -E 'MEAN|VAL-selected' | tr '\n' ' ')"
fi
log "=== queue_v4 finished ==="
