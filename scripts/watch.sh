#!/bin/bash
# Follow whichever pipeline stage is currently running.
#
#   ./scripts/watch.sh            # auto-follow the active stage
#   ./scripts/watch.sh status     # one-shot snapshot, no tailing
#   ./scripts/watch.sh train      # force-follow v4 training
#   ./scripts/watch.sh eval       # force-follow UPenn evaluation
#   ./scripts/watch.sh brats      # force-follow the BraTS recompute
#   ./scripts/watch.sh chain      # follow stage transitions only
#
# Ctrl-C stops watching; it does not stop the pipeline.

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
L=logs

snapshot() {
  echo "──────── jobs ────────"
  ps -eo pid,etime,comm,args --no-headers \
    | grep -E "train_upenn|evaluate_upenn|evaluate_brats|extract_radiomic|train_classifier|opt_chain" \
    | grep -v grep | grep -v "bash -c" | grep -v "tail -f" \
    | while read -r pid etime _ rest; do
        # Show just the script being run, not the full argv.
        script=$(printf '%s\n' "$rest" | grep -oE '[a-z_]+\.(py|sh)' | head -1)
        printf "  %-8s up %-9s %s\n" "$pid" "$etime" "${script:-$rest}"
      done
  [ -z "$(pgrep -f 'scripts/(train_upenn|evaluate_upenn|evaluate_brats)\.py')" ] && echo "  (no compute job running)"

  echo "──────── gpu ────────"
  nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader | sed 's/^/  /'

  echo "──────── stages ────────"
  [ -f "$L/opt_chain.log" ] && sed 's/^/  /' "$L/opt_chain.log" || echo "  (chain not started)"

  if [ -f "$L/finetune_v4.log" ]; then
    echo "──────── v4 training (last 5 epochs) ────────"
    grep -E "^Epoch" "$L/finetune_v4.log" | tail -5 | sed 's/^/  /'
    echo "  epochs done: $(grep -cE '^Epoch' "$L/finetune_v4.log")/80"
  fi

  for f in eval_final evaluate_brats; do
    if [ -f "$L/$f.log" ]; then
      echo "──────── $f (tail) ────────"
      tail -3 "$L/$f.log" | sed 's/^/  /'
    fi
  done
}

pick_active() {
  for f in finetune_v4 eval_final evaluate_brats extract_features train_classifier eval_corrected; do
    if [ -f "$L/$f.log" ] && [ -n "$(find "$L/$f.log" -mmin -2 2>/dev/null)" ]; then
      echo "$L/$f.log"; return
    fi
  done
  echo "$L/opt_chain.log"
}

case "${1:-auto}" in
  status) snapshot ;;
  train)  tail -f "$L/finetune_v4.log" ;;
  eval)   tail -f "$L/eval_final.log" ;;
  brats)  tail -f "$L/evaluate_brats.log" ;;
  chain)  tail -f "$L/opt_chain.log" ;;
  auto)
    snapshot
    f=$(pick_active)
    echo
    echo "──────── following $f (Ctrl-C to stop) ────────"
    tail -f "$f"
    ;;
  *) echo "usage: $0 [status|auto|train|eval|brats|chain]"; exit 1 ;;
esac
