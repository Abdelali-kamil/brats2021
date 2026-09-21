# Segmentation improvements — levers, protocol, and how to run them

This document describes changes added to improve BraTS2021 segmentation and,
especially, generalisation to UPenn-GBM. Everything here is **opt-in**: default
behaviour, the frozen splits, and every reported number are unchanged unless a
flag is passed. No change tunes on or otherwise touches held-out/test data, so
none introduces leakage.

The actual before/after Dice must be produced on a machine with the BraTS/UPenn
data and a GPU — this repository ships the code and the comparison harness, not
the trained checkpoints.

## What was wrong (and is now fixable)

1. **BraTS base training had no augmentation.** `get_datasets()` returned a
   dataset with `data_aug=False` and `on="test"` → `training=False`, so the
   train subset used a **deterministic centre crop every epoch and no
   augmentation at all**. The model that must transfer zero-shot to UPenn was
   the least-regularised one — the likely dominant cause of the ~0.19 zero-shot
   gap (BraTS 0.883 → UPenn 0.695). The UPenn fine-tune script, by contrast,
   already augmented. This asymmetry is now removed.

2. **Broken helper import in `brats.py`** (`from image import …`) silently
   failed in the package layout, leaving the crop/normalise helpers undefined.
   Fixed with a robust relative/absolute import.

3. **`evaluate_brats.py` hard-coded the checkpoint** (`segmentor_epoch_650.pth`)
   and only recognised the `model_state_dict` key — not the `model_state` key
   that `train_brats.py` actually saves. It now takes `--checkpoint` and
   accepts all common key layouts, so any checkpoint can be evaluated.

4. **Hard-coded data paths** now honour `$BRATS_DATA_DIR` / `--data-root`.

## New levers

### 1. On-the-fly 3D augmentation (`brats_gbm/data/augment.py`)

Intensity-heavy (this is what buys cross-cohort robustness) plus light spatial:

| Transform | Default p | Notes |
|---|---|---|
| flips (3 axes) | 0.5 | shared image+label |
| small affine (±10°, scale 0.85–1.15) | 0.2 | label via nearest-neighbour |
| gamma | 0.3 | foreground, rescaled → valid for any normalisation |
| brightness/contrast | 0.3 | per-channel foreground |
| MRI bias field | 0.3 | smooth multiplicative low-freq field |
| Gaussian blur | 0.2 | |
| low-resolution simulation | 0.25 | down/up-sample (resolution shift) |
| Gaussian noise | 0.2 | foreground |

Applied to the **training subset only**; the validation subset stays
deterministic and its subjects are **byte-identical** to the frozen seed-42
split (verified: 1000 train / 251 val). Each sample draws a fresh RNG so
DataLoader workers never share augmentation state.

Enable with `train_brats.py --augment`.

### 2. Batch-independent normalisation (`WaveletUNetPlusPlus(norm=…)`)

`--norm {batch,instance,group}`. Training runs at effective batch ≈ 1, where
BatchNorm running statistics are noisy and specific to the BraTS intensity
distribution — a known drag on cross-scanner transfer. `instance`/`group` do
not depend on batch statistics (the nnU-Net-standard choice). Default `batch`
keeps existing checkpoints loadable; `instance`/`group` change the parameter
set and therefore require training **from scratch** (pass `--resume ""`).

### 3. A/B comparison harness (`scripts/compare_runs.py`)

Paired bootstrap comparison of two per-case CSVs on their shared cases,
reporting each region's Dice with 95% CIs and the paired difference with a CI.
A change is credible only when its paired CI excludes zero. (Validated: it
reproduces the published fine-tuning result +0.109 [0.038, 0.192].)

## Experiment protocol

### A. Does augmentation improve BraTS internal validation?

```bash
# Baseline: score the current checkpoint (no retrain)
python scripts/evaluate_brats.py --partition internal_validation \
    --checkpoint checkpoints/segmentor_epoch_650.pth --tag base650

# Continue-train the same checkpoint WITH augmentation (~100–150 epochs)
python scripts/train_brats.py --augment --epochs 800 \
    --resume checkpoints/segmentor_epoch_650.pth --save-dir checkpoints_aug

# Score the augmented checkpoint
python scripts/evaluate_brats.py --partition internal_validation \
    --checkpoint checkpoints_aug/segmentor_epoch_800.pth --tag aug800

python scripts/compare_runs.py \
    --baseline  results/brats/per_case_base650_recomputed.csv \
    --candidate results/brats/per_case_aug800_recomputed.csv \
    --name-baseline epoch650 --name-candidate augmented \
    --out-csv results/brats/ab_augmentation.csv
```

### B. Does it improve zero-shot generalisation to UPenn? (the key question)

Score both checkpoints zero-shot on the UPenn test set (BraTS preprocessing
convention — see `docs/METHODOLOGY.md`) and compare paired. Use
`scripts/evaluate_upenn.py` with each checkpoint, then `compare_runs.py` on the
resulting per-case CSVs. A rise in zero-shot UPenn Dice with a paired CI
excluding zero is the evidence that generalisation improved.

### C. Batch-independent norm (larger commitment)

```bash
python scripts/train_brats.py --augment --norm instance --resume "" \
    --epochs 700 --save-dir checkpoints_in
```
Then evaluate with `--norm instance` and compare as above. Because the 29-case
UPenn test set cannot resolve differences < ~0.05, confirm any promising result
with `scripts/crossval_upenn.py` over 147 subjects.

## Guardrails

- Augmentation touches only the training subset; validation/test are untouched.
- The frozen splits are unchanged (reproduction verified in `tests/`).
- Post-processing / thresholds are not re-tuned here.
- Correctness is covered by `tests/test_augment.py` and `tests/test_model.py`;
  performance claims require the runs above.
