# Pushing internal-validation mean Dice toward 0.90

Current held-out (251-case internal validation), from `results/brats/`:

| Region | Dice | note |
|---|---|---|
| ET | 0.850 | **bottleneck** — small, near-threshold, all-or-nothing on ET-negative cases |
| TC | 0.894 | |
| WT | 0.925 | already strong |
| **Mean** | **0.883–0.889** | depends on the post-processing point (see the two summary CSVs) |

The mean is dragged down almost entirely by ET. Lifting ET by ~0.03 and TC by
~0.01 is what crosses 0.90 — WT needs nothing.

> These are the levers, not a result. This repo does not ship numbers it cannot
> reproduce (see `docs/METHODOLOGY.md`). Every figure below has to be produced by
> actually rerunning training/evaluation on the data + GPU; none is asserted.

## 1. Training augmentation was disabled (the big one)

`get_datasets()` previously returned a single dataset built with
`training=False, data_aug=False`, so the *training* split was fed
deterministic centre crops and **zero augmentation** — the `augment()` method
was never called. A 3D segmentor trained on ~1000 cases with no augmentation
overfits, and the hard, small ET region is where that shows up first.

Fixed in `brats_gbm/data/brats.py`:

- `get_datasets()` now returns `(train_ds, val_ds)` from the *same* seeded 80/20
  split used by `brats_gbm.splits.brats_split` / `scripts/evaluate_brats.py`, so
  the internal-validation set is unchanged and disjoint. The train split gets
  random crops + augmentation; the val split stays a deterministic centre crop.
- `augment()` now does independent flips on all three spatial axes plus
  per-channel intensity scale / shift / gamma inside the brain mask (background
  zeros preserved). Previously it only flipped two axes and, more importantly,
  never ran.

This is checkpoint-compatible (architecture unchanged), so you can either
retrain from scratch or **fine-tune from `segmentor_epoch_650.pth`** with
augmentation on.

```bash
# fine-tune the existing checkpoint with augmentation + a cosine LR push
python scripts/train_brats.py --epochs 750 --scheduler cosine --lr 1e-4
```

## 2. Loss weighting toward ET/TC

The loss weights ET/TC/WT equally. Since ET/TC are the harder regions, mildly
upweighting them is worth trying (validate — over-weighting destabilises):

```bash
python scripts/train_brats.py --weight-channels 1.3 1.1 1.0 --scheduler cosine --lr 1e-4
```

## 3. No-retrain lever: tune the operating point on TRAIN, apply to VAL

`evaluate_brats.py` fixes the threshold at 0.5 on purpose — its 251-case set
doubled as the model-selection set, so tuning there is fitting on your own test
data. `scripts/tune_thresholds_brats.py` does it honestly: it selects per-region
thresholds and the ET min-volume rule on the **train** partition, then applies
the chosen point to internal validation (disjoint split). A slightly lower ET
threshold recovers true enhancement while min-volume kills the stray-voxel false
positives that zero out ET-negative cases.

```bash
python scripts/tune_thresholds_brats.py --tune-limit 200
```

This costs nothing but inference time and targets ET directly.

## Bigger levers (need a from-scratch retrain — not done here)

Not applied because they change the architecture and would abandon the existing
checkpoint, and because none can be validated without a training run:

- **InstanceNorm / GroupNorm instead of BatchNorm.** Training uses
  `batch_size=1` with `accum_steps=8`; BatchNorm statistics on a batch of one
  are noise. 3D medical segmentation (nnU-Net et al.) uses InstanceNorm for
  exactly this reason. This is the largest untapped gain but requires retraining
  from scratch.
- **Deep supervision.** U-Net++ naturally supports auxiliary losses on the
  `x0_1..x0_3` heads; standard for this family and stabilises the small regions.
- **Model / checkpoint ensembling** across seeds or folds.

Recommended order: (3) for a free check today → (1) fine-tune with augmentation
→ (2) loss weighting → then a from-scratch run with InstanceNorm + deep
supervision if 0.90 is still short.
