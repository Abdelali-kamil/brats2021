# MRI segmentation figures — BraTS2021 and UPenn-GBM

Qualitative figures for both cohorts. Both are produced by the same rendering
code (`brats_gbm/viz.py`), so palette, layout and typography are identical by
construction rather than by convention.

| Cohort | Script | Output |
|---|---|---|
| BraTS2021 | `scripts/visualize_brats.py` | `results/brats/figures/brats_<case>.png` |
| UPenn-GBM | `scripts/visualize_upenn.py` | `results/upenn/figures/upenn_<subject>.png` |

```bash
python scripts/visualize_brats.py                 # best / median / worst
python scripts/visualize_upenn.py
python scripts/visualize_brats.py BraTS2021_00413 # a specific case
python scripts/visualize_upenn.py --modality T2w  # a different modality
```

## Layout

Three panels, left to right:

1. **Image** — the chosen modality at the selected slice, no overlay, so the
   underlying anatomy is visible before any annotation covers it.
2. **Expert annotation** — the reference segmentation.
3. **Prediction** — the model output, captioned with that case's per-region
   Dice.

The figure title states the cohort, which partition the case came from, whether
it is the best, median or worst case, and its mean Dice.

## Colour scheme

| Colour | Region | RGB |
|---|---|---|
| Blue | Enhancing tumour (ET) | `(0.165, 0.471, 0.839)` |
| Orange | Necrotic core (TC \ ET) | `(0.922, 0.408, 0.204)` |
| Green | Oedema (WT \ TC) | `(0.106, 0.686, 0.478)` |

Overlays are drawn at 55% opacity over a grayscale base.

The three regions are **nested** — ET ⊆ TC ⊆ WT — and are painted
largest-first, so each colour that remains visible is that region *minus* the
one nested inside it. This is why the legend names them as set differences
rather than as the raw regions, and it is what makes the figures readable: on a
typical glioblastoma you see an orange necrotic core, ringed by blue enhancing
tumour, ringed by green oedema, which is the expected anatomy.

## Slice selection

The axial slice carrying the most whole-tumour voxels in the reference
annotation, so the figure shows the lesion rather than an arbitrary plane. If
the reference is empty the prediction is used instead; if both are empty the
mid-volume slice is shown.

## Case selection

By default the **best, median and worst** case by mean Dice, so the figures
span the performance range instead of showing only successes. A worst case that
fails outright is included deliberately.

Both scripts refuse case ids outside the held-out partition — BraTS2021 figures
come from the 251-case internal-validation partition and UPenn-GBM figures from
the 29-subject test partition. Rendering a training case would be misleading,
so it is blocked rather than merely discouraged.

## Preprocessing (important)

Each script feeds its checkpoint the convention that checkpoint was trained
under. They differ, and the difference is not cosmetic:

| Script | Channel order | Normalisation |
|---|---|---|
| `visualize_brats.py` | `[t1, t1ce, t2, flair]` | percentile-clipped min-max to [0,1] |
| `visualize_upenn.py` | `[flair, t1, t1ce, t2]` | z-score over non-zero voxels |

Feeding a BraTS-trained checkpoint the UPenn convention costs roughly 0.56 mean
Dice. See `docs/METHODOLOGY.md`.

## Operating point

`visualize_upenn.py` reads the thresholds `evaluate_upenn.py` selected on the
validation split from `results/upenn/segmentation_summary.csv`, so the masks
drawn are the masks the reported Dice was computed from.

`visualize_brats.py` uses a fixed 0.5, the value `train_brats.py` validates
against. BraTS has no separate tuning partition in this project, so no
threshold is fitted.

## Reading the numbers on the figure

The Dice printed in the third panel is **recomputed by the figure script** from
that case's prediction, not copied from a results CSV. It is therefore always
consistent with the mask you are looking at.

For BraTS this can disagree with `per_case_all_1251.csv`, which came from an
archived pipeline whose settings were not recorded and which the current code
cannot reproduce. `scripts/evaluate_brats.py` regenerates BraTS per-case scores
under the current protocol; prefer `per_case_internal_validation_recomputed.csv`
over the legacy file wherever both exist.
