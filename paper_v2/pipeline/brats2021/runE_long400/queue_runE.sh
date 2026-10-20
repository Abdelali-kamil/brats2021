#!/bin/bash
# Amendment 27: Run E queue. Waits until (a) 2026-10-17 or later and (b) the TransBTS and nnFormer evaluations have ended
# (EVALUATED / FAILED / SANITY_FAILED each), then trains Run E (run.sh, 400 epochs, resumable) and evaluates it exactly like
# Run D2: eval_run.sh runE (VAL -> TEST once), eval_run_rot2.sh runE_rot runE (VAL -> TEST once). Idempotent; flock.
set -uo pipefail
E=/mnt/data1/kamil_research/experiments; A=$E/runE_long400; V=$E/eval_v2; LOG=$A/queue.log; RES=/home/kamilabdelali/brats2021/results/brats
exec 9> $A/.lock_queue; flock -n 9 || exit 0
log() { echo "[runE $(date '+%F %T')] $*" | tee -a $LOG; }
ended() { [ -e $1/EVALUATED ] || [ -e $1/FAILED ] || [ -e $1/SANITY_FAILED ]; }
[ -e $A/DONE ] && exit 0
until [ "$(date +%Y%m%d)" -ge 20261017 ] && ended $E/transbts && ended $E/nnformer; do sleep 600; done
log "gate open (transbts: $(ls $E/transbts | grep -E '^(EVALUATED|FAILED|SANITY_FAILED)$' | tr '\n' ' ')nnformer: $(ls $E/nnformer | grep -E '^(EVALUATED|FAILED|SANITY_FAILED)$' | tr '\n' ' '))"
ep() { /home/kamilabdelali/anaconda3/envs/torchfix/bin/python -c "import torch;print(int(torch.load('$A/checkpoints/runE_long400_last.pth',map_location='cpu',weights_only=False).get('epoch',0)))" 2>/dev/null || echo 0; }
if [ "$(ep)" -lt 400 ]; then log "training starts/resumes (epoch $(ep))"; bash $A/run.sh > /dev/null 2>&1; fi
[ "$(ep)" -ge 400 ] || { log "training did not reach 400 epochs (epoch $(ep)) -> stop"; touch $A/FAILED; exit 1; }
log "training complete: $(grep -a 'New best' $A/logs/runE_long400.log | tail -1)"
[ -e $RES/summary_runE_test_tuned_recomputed.csv ] || bash $V/eval_run.sh runE $A/checkpoints/runE_long400_best.pth
[ -e $RES/summary_runE_test_tuned_recomputed.csv ] || { log "eval runE FAILED (see $V/eval_runE.log)"; touch $A/FAILED; exit 1; }
log "runE evaluated: $(cat $V/work_runE/selection.txt)"
[ -e $RES/summary_runE_rot_test_tuned_recomputed.csv ] || bash $V/eval_run_rot2.sh runE_rot runE
log "runE_rot evaluated: $(cat $V/work_runE_rot/selection.txt 2>/dev/null)"
log "VAL reference: Run D2 rot 0.8950 | Run C rot 0.9006 | Ensemble 3 0.9079 (Amendment 27)"
touch $A/DONE; log "=== Run E finished ==="
