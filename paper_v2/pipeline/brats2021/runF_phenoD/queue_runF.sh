#!/bin/bash
# Amendment 29 queue: train Run F (run.sh, 200 epochs, resumable) -> VAL-only evaluation (eval_run_valonly.sh runF) ->
# rotation variant VAL-only (eval_run_rot_valonly.sh runF_rot runF) -> TRAIN-fit (Run F) -> criteria vs Run D. NO TEST. Idempotent; flock.
set -uo pipefail
E=/mnt/data1/kamil_research/experiments; A=$E/runF_phenoD; V=$E/eval_v2; LOG=$A/queue.log
PY=/home/kamilabdelali/anaconda3/envs/torchfix/bin/python
exec 9> $A/.lock_queue; flock -n 9 || exit 0
log() { echo "[runF $(date '+%F %T')] $*" | tee -a $LOG; }
[ -e $A/DONE ] && exit 0
ep() { $PY -c "import torch;print(int(torch.load('$A/checkpoints/runF_phenoD_last.pth',map_location='cpu',weights_only=False).get('epoch',0)))" 2>/dev/null || echo 0; }
if [ "$(ep)" -lt 200 ]; then log "training starts/resumes (epoch $(ep))"; bash $A/run.sh > /dev/null 2>&1; fi
[ "$(ep)" -ge 200 ] || { log "training did not reach 200 epochs (epoch $(ep)) -> stop"; touch $A/FAILED; exit 1; }
log "training complete: $(grep -a 'New best' $A/logs/runF_phenoD.log | tail -1)"
[ -s $V/work_runF/selection.txt ] || bash $V/eval_run_valonly.sh runF $A/checkpoints/runF_phenoD_best.pth
[ -s $V/work_runF/selection.txt ] || { log "VAL eval runF FAILED (see $V/eval_runF.log)"; touch $A/FAILED; exit 1; }
log "runF VAL: $(cat $V/work_runF/selection.txt)"
bash $A/trainfit.sh runD >> $LOG 2>&1 || log "trainfit runD FAILED"   # control; normally already done before Run F results exist
bash $A/trainfit.sh runF >> $LOG 2>&1 || log "trainfit runF FAILED"
mkdir -p $A/results
$PY $A/criteria.py runF runD --trainfit-a $A/trainfit_runF/out --trainfit-b $A/trainfit_runD/out > $A/results/criteria_base.log 2>&1
log "criteria (base): $(grep VERDICT $A/results/criteria_base.log)"
[ -s $V/work_runF_rot/selection.txt ] || bash $V/eval_run_rot_valonly.sh runF_rot runF
log "runF_rot VAL: $(cat $V/work_runF_rot/selection.txt 2>/dev/null)"
$PY $A/criteria.py runF_rot runD_rot --trainfit-a $A/trainfit_runF/out --trainfit-b $A/trainfit_runD/out > $A/results/criteria_rot.log 2>&1
log "criteria (rot): $(grep VERDICT $A/results/criteria_rot.log)"
touch $A/DONE; log "=== Run F finished (VAL only; no TEST) ==="
