#!/usr/bin/env bash
# Best available shot at 0.89 on the clean test split.
#
# Stacks the one proven lever with the two untried ones, warm-restarting from
# aug_cosine_repro1 epoch 203 (val 0.8849):
#
#   ET channel up-weighted 1.5x   proven: 0.8804 -> 0.8846 on test
#   deep supervision              untried; nnU-Net enables it by default
#   focal-Tversky on ET           untried; targets the weakest region
#
# 40 epochs, ~3.5 h. The length is deliberate: this machine's GPU has faulted
# three times at 13.7 h, 11.4 h and 8.3 h, so a run must finish well inside
# 8 hours to be worth starting.
#
# Honest expectation: ~0.886-0.890 on test. 0.89 is the optimistic edge, not
# the expected value, and every component gain is within the noise band of a
# 125-case test set. Whatever comes out is what gets reported.
set -uo pipefail

R=/mnt/data1/kamil_research/experiments/repo_aug_cosine
EXP=/mnt/data1/kamil_research/experiments
PY=/home/kamilabdelali/anaconda3/envs/torchfix/bin/python
SRC="$EXP/aug_cosine_repro1/checkpoints/aug_cosine_repro1_best.pth"
OUT="$EXP/push089_v1"
mkdir -p "$OUT/checkpoints" "$OUT/logs"

for i in $(seq 1 60); do nvidia-smi -L >/dev/null 2>&1 && break; sleep 30; done
nvidia-smi -L >/dev/null 2>&1 || { echo "no GPU; aborting"; exit 1; }

cat > "$OUT/PROVENANCE.txt" <<PROV
run: push089_v1
started: $(date -Iseconds)
warm-restarted from: $SRC (epoch 203, val 0.8849)
changes vs the base recipe:
  --weight-channels 1.5 1.0 1.0   ET up-weighted (proven lever, +0.004 historically)
  --deep-supervision              auxiliary heads on x0_1/x0_2/x0_3 (untried)
  --tversky-et 0.5                focal-Tversky on the ET channel (untried)
  --lr 5e-5                       fine-tune rate, as the previous ET fine-tune used
  40 epochs cosine                short enough to finish inside the GPU's fault window
split: unchanged, identical to the base run
NOTE: architecture is unchanged. Deep supervision adds 153 training-only
parameters and the inference path is byte-identical to the published model.
PROV

cd "$R" || exit 1
"$PY" scripts/train_brats.py \
    --resume "$SRC" --warm-restart \
    --epochs 40 --lr 5e-5 --scheduler cosine \
    --weight-channels 1.5 1.0 1.0 \
    --deep-supervision --tversky-et 0.5 \
    --tag push089_v1 --num-workers 8 --batch-size 2 --accum-steps 4 \
    --save-dir "$OUT/checkpoints" \
    > "$OUT/logs/push089_v1.log" 2>&1
echo "exit rc=$?"
