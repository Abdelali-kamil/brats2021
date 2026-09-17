#!/usr/bin/env bash
# Finish the BraTS result in one unattended pass, then hand the GPU to nnU-Net.
#
# Decision this encodes: stop training at epoch 203. The original aug_cosine_v1
# run peaked at epoch 195 (val 0.8865) and every later epoch was worse; this run
# is at 203 with 0.8849, i.e. already past that peak. The remaining 97 epochs
# are very unlikely to add anything, and finishing them would mean exposing
# another ~8 h run to a machine that has faulted three times in twelve days
# (2026-09-05, 09-16, 09-17) at 13.7 h, 11.4 h and 8.3 h.
#
# So: evaluate what we have, fix HD95, report, and spend the GPU on the nnU-Net
# baseline instead.
#
#   1. Evaluate best checkpoint on VALIDATION, saving probabilities
#   2. Sweep post-processing on those validation probabilities only
#   3. Apply the winning setting to TEST, once
#   4. Start nnU-Net continuing from its saved checkpoint
#
# Step 2 tunes on validation and step 3 touches test exactly once, which is the
# protocol the paper already claims. Do not reorder them.
set -uo pipefail

ROOT=/home/kamilabdelali/brats2021
PY=/home/kamilabdelali/anaconda3/envs/torchfix/bin/python
EXP=/mnt/data1/kamil_research/experiments/aug_cosine_repro1
CKPT="$EXP/checkpoints/aug_cosine_repro1_best.pth"
ANA=/mnt/data1/kamil_research/experiments/analysis
LOG="$ROOT/logs/finish_fast.log"

mkdir -p "$ROOT/logs" "$ANA"
log() { echo "[finish $(date '+%F %T')] $*" | tee -a "$LOG"; }

log "=== starting ==="

for i in $(seq 1 60); do
  nvidia-smi -L >/dev/null 2>&1 && break
  [ "$i" -eq 1 ] && log "waiting for GPU"
  sleep 30
done
nvidia-smi -L >/dev/null 2>&1 || { log "no GPU after 30 min; aborting"; exit 1; }
log "GPU ready: $(nvidia-smi -L | head -1)"

EP=$("$PY" -c "import torch;print(torch.load('$CKPT',map_location='cpu',weights_only=False)['epoch'])" 2>/dev/null)
log "evaluating checkpoint epoch $EP"

# --- 1. validation, with probabilities kept so step 2 needs no re-inference ---
# NOTE: --partition internal_validation is the 2-way split's 251 cases, which
# CONTAINS the 125 test cases. Tuning on it and then "testing" would leak. Use
# the frozen 3-way split: 126 val for tuning, 125 test scored once.
log "[1/4] validation (126 cases, full protocol)"
"$PY" "$ROOT/scripts/evaluate_brats.py" --partition internal_validation \
    --cases "$ROOT/baselines/common/val_ids.txt" \
    --checkpoint "$CKPT" --save-probs "$ANA/probs_val" \
    --tag final_ep${EP}_val >> "$ROOT/logs/eval_final_val.log" 2>&1
log "[1/4] done rc=$?"

# --- 2. tune post-processing on validation ONLY ---
log "[2/4] post-processing sweep on validation"
"$PY" "$ROOT/baselines/common/sweep_postproc_hd95.py" \
    --probs "$ANA/probs_val" --tag final_val >> "$ROOT/logs/sweep_final_val.log" 2>&1
log "[2/4] done rc=$?"

# Pick the setting with the lowest mean HD95 that does not cost more than
# 0.002 mean Dice -- HD95 is the weak metric, but not at the cost of the
# headline one.
read -r WT_POLICY FLOOR_MULT <<< "$("$PY" - <<'PYEOF'
import pandas as pd, pathlib, re
p = pathlib.Path("/home/kamilabdelali/brats2021/results/brats/postproc_sweep_final_val.csv")
if not p.exists():
    print("components 1"); raise SystemExit
d = pd.read_csv(p)
base = d.iloc[0]
d["hd95_mean"] = d[["HD95_ET", "HD95_TC", "HD95_WT"]].mean(axis=1)
ok = d[d["Dice_Mean"] >= base["Dice_Mean"] - 0.002]
win = ok.loc[ok["hd95_mean"].idxmin()] if len(ok) else base
mult = 1
m = re.search(r"x(\d+)", str(win["setting"]))
if m: mult = int(m.group(1))
print(f"{win['wt_policy']} {mult}")
PYEOF
)"
log "[3/4] winner on validation: wt_policy=$WT_POLICY floor_mult=$FLOOR_MULT"

# --- 3. test, once, with the validation-selected setting ---
log "[3/4] test (125 cases), applied once"
"$PY" "$ROOT/scripts/evaluate_brats.py" --partition internal_validation \
    --cases "$ROOT/baselines/common/test_ids.txt" \
    --checkpoint "$CKPT" --wt-policy "$WT_POLICY" --floor-mult "$FLOOR_MULT" \
    --save-probs "$ANA/probs_test" \
    --tag final_ep${EP}_test_tuned >> "$ROOT/logs/eval_final_test.log" 2>&1
log "[3/4] done rc=$?"

# --- 4. hand the GPU to the baseline ---
log "[4/4] starting nnU-Net (continues from its checkpoint)"
setsid nohup bash "$ROOT/baselines/nnunet/run_overnight.sh" \
    >> "$ROOT/logs/nnunet_after_finish.log" 2>&1 < /dev/null &

log "=== done. results in $ROOT/results/brats/ ==="
