# Methodology and evaluation protocol

This document records how every reported number is produced and, where an
earlier version of this project got something wrong, what the error was and how
it was corrected. It is written for a reviewer who wants to check the claims
rather than for someone rerunning the code — for that, see the README.

## Datasets and partitions

### BraTS2021 — 1251 cases

`scripts/train_brats.py` makes a two-way 80/20 partition (1000 / 251) with
`torch.utils.data.random_split` under a fixed seed of 42.

The 251-case partition served as the model-selection set during training:
validation Dice was computed on it every epoch and used to judge checkpoints.
It is therefore **internal validation, not a held-out test set**, and is
labelled that way everywhere in `results/brats/`. Reporting it as a test set
would understate the optimism by an unknown amount.

The honest external test in this project is UPenn-GBM, which the BraTS model
never saw in any form.

Two things bound how much that optimism can be worth. Checkpoints were written
every epoch rather than being chosen adaptively from a small pool, and the
train-minus-validation generalisation gap is +0.0087 mean Dice with a 95%
confidence interval of [-0.0089, +0.0284] — an interval containing zero, so no
overfitting to the training cases is statistically detectable.

### UPenn-GBM — 147 expert-segmented subjects

UPenn-GBM ships 147 subjects with expert-approved segmentations and a further
~480 with automated ones. Only the expert-segmented subjects are used for
segmentation training and evaluation.

The partition is 104 train / 14 validation / 29 test, drawn by permuting the
sorted subject list with `numpy.random.default_rng(42)`. All three partitions
are produced by a single function, `brats_gbm.splits.upenn_split`, and the
resulting assignment is written to `results/upenn/split_assignment.csv` so it
can be audited.

> Previously the split function was duplicated verbatim in the training script
> and the evaluation script. The two copies agreed, so no contamination
> occurred, but nothing enforced that they would keep agreeing. A single
> definition now serves both.

## What is selected on what

The distinction that matters is between parameters *fitted on validation* and
parameters *reported on test*.

| Decision | Selected on | Never sees |
|---|---|---|
| Network weights | UPenn train (104) | val, test |
| Checkpoint (EMA-best) | UPenn validation (14) | test |
| Region thresholds (ET/TC/WT) | UPenn validation (14) | test |
| ET post-processing policy and cutoff | UPenn validation (14) | test |
| Ensemble membership | UPenn validation (14) | test |
| Reported Dice / HD95 | — | scored once on test (29) |

`scripts/evaluate_upenn.py` enforces this ordering structurally: it computes
validation probability maps, makes every selection from them, and only then
loads the test maps. The chosen configuration is written to
`results/upenn/validation_selection.csv` alongside the validation score that
justified it.

### The test-set tuning that was removed

An earlier reporting path (`final_results_all.py`, `final_test.py`,
`results_test.py`, all now in the archive) swept per-region thresholds
*separately for each of seven model setups*, then ranked the setups against
each other **on the test set** and reported the winner. Two compounding
problems:

1. The thresholds were fitted to the data they were then scored on.
2. Setup selection used the same test set, so the reported maximum was the
   maximum of seven noisy estimates rather than an unbiased estimate of any
   one of them.

The winning configuration used `et_thr = 0.015`, which labels nearly any voxel
as enhancing tumour. That produced the reported ET Dice of 0.931 against a WT
Dice of 0.793 — an ordering that is backwards from every published BraTS
result, because whole tumour is by far the easiest region and enhancing tumour
by far the hardest. The number was an artefact of fitting the threshold to the
test set, not a finding.

Those scripts and every output derived from them have been removed from the
repository. `docs/ARCHIVE.md` lists what went where.

## Input preprocessing is a property of the checkpoint

This was the most consequential error found in the project, and it invalidated
the original headline result.

The two loaders disagree about what the network's four input channels mean:

| | Channel order | Normalisation |
|---|---|---|
| `brats_gbm/data/brats.py` (BraTS training) | `[t1, t1ce, t2, flair]` | percentile-clipped min-max to [0,1] |
| `brats_gbm/data/upenn.py` (UPenn fine-tune + eval) | `[flair, t1, t1ce, t2]` | z-score over non-zero voxels |

That is a channel permutation *and* a different intensity distribution. The
zero-shot baseline — the BraTS-trained checkpoint applied to UPenn-GBM — was
being fed the UPenn convention, so it received scrambled input. Measured on 5
UPenn validation subjects with `segmentor_epoch_650`, mean Dice:

| Input convention | ET | TC | WT | Mean |
|---|---|---|---|---|
| UPenn (what was previously used) | 0.017 | 0.121 | 0.472 | **0.203** |
| Correct order, wrong normalisation | 0.310 | 0.389 | 0.368 | 0.356 |
| Wrong order, correct normalisation | 0.023 | 0.130 | 0.328 | 0.160 |
| BraTS (what it was trained on) | 0.725 | 0.846 | 0.727 | **0.766** |

**Almost the entire reported BraTS-to-UPenn "domain gap" was this bug.** The
original claim — that a BraTS model collapses to ~0.16 Dice on UPenn-GBM and
fine-tuning recovers it to ~0.82, a gain of roughly +0.66 — does not survive.
Given its own preprocessing the same checkpoint reaches roughly 0.77 zero-shot,
so the genuine benefit of fine-tuning is on the order of +0.05, not +0.66.

Each setup now declares its `preprocessing` in `scripts/evaluate_upenn.py`, and
the probability cache is keyed by it so variants cannot be silently mixed.

Two consequences worth stating plainly. First, the fine-tuned checkpoints were
themselves fine-tuned under the UPenn convention, which means fine-tuning had to
spend capacity re-learning a channel permutation; their reported numbers are
valid, because they are evaluated under the same convention they were trained
under, but a fine-tune started from correctly-preprocessed inputs might do
better and has not been tried. Second, the apparent "unstripped skull causes the
domain gap" explanation that appears in the archived code comments was a
rationalisation of a bug.

### The threshold grid, and the edge-pinning diagnostic

Region thresholds are swept on validation over
{0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80} for each region.

The grid got that wide because of a diagnostic worth keeping. A selection that
lands on the edge of a grid is not a selection at all — it is an artefact of
where the sweep stops — so `select_thresholds` warns whenever the chosen value
sits at either extreme, and the grid was widened until nothing was pinned.

That warning is what first exposed the preprocessing bug. Under the broken
preprocessing the baseline selected the **lowest** threshold offered, and
sweeping three orders of magnitude down showed its Dice rising monotonically to
0.0002 with no interior optimum at all: a model fed scrambled input emits
uniformly tiny probabilities, so there is no calibrated decision boundary to
find and lowering the cut merely labels more voxels.

With the correct convention the picture inverts, which is the signature of a
model that is now properly calibrated:

| ET=TC threshold | 0.0002 | 0.01 | 0.10 | 0.30 | 0.50 | 0.70 | **0.80** | 0.90 |
|---|---|---|---|---|---|---|---|---|
| Validation mean Dice | 0.454 | 0.608 | 0.649 | 0.731 | 0.756 | 0.766 | **0.769** | 0.760 |

The optimum is now interior — it peaks at 0.80 and falls again at 0.90 — so the
grid is adequate and the selected threshold is a real choice. The full curve is
regenerated by `scripts/probe_baseline_threshold.py` into
`results/upenn/baseline_threshold_sensitivity.csv`.

The lesson generalises: an edge-pinned hyperparameter is usually evidence of a
defect upstream, not evidence that the grid needs extending. Extending the grid
here would have produced a slightly better number and left the real bug in
place.

Region thresholds are swept on validation over
{0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50} for each of ET, TC and WT.

The fine-tuned models select interior points, so the grid is adequate for them.
The zero-shot baseline does not: it selects whatever minimum it is offered. That
was worth investigating rather than papering over, because the baseline is the
"before" term in the domain-adaptation comparison and handicapping it would
inflate the reported gain.

`scripts/probe_baseline_threshold.py` sweeps three orders of magnitude on the
cached validation maps. The baseline's Dice rises **monotonically** all the way
down to 0.0002 with no interior optimum:

| ET=TC threshold | 0.50 | 0.30 | 0.10 | 0.02 | 0.002 | 0.0002 |
|---|---|---|---|---|---|---|
| Validation mean Dice | 0.144 | 0.151 | 0.163 | 0.177 | 0.201 | 0.227 |

This is not a grid that is too narrow. The baseline's enhancing-tumour and
tumour-core probabilities on UPenn-GBM lie almost entirely below any sensible
cut, so lowering the threshold does not recover a calibrated decision boundary —
it just labels progressively more voxels. **Threshold tuning cannot repair the
domain gap.**

The reporting consequence is that the baseline's Dice is a function of where the
grid stops, so no single value is canonical. The grid floor is held at 0.02 for
every setup, giving one uniform protocol, and the full curve is recorded in
`results/upenn/baseline_threshold_sensitivity.csv` so the reported figure can be
read against it. The conclusion is invariant across the whole range: the
baseline is catastrophic at every threshold, and fine-tuning moves it to
roughly 0.82. Quote the domain-adaptation improvement as approximate rather than
to four decimal places, since its "before" term carries this dependence.

## Post-processing

Applied in `brats_gbm/eval/postprocess.py`, in order:

1. **Hierarchy.** ET ⊆ TC ⊆ WT. The three channels are independent sigmoids,
   so the network does not guarantee this.
2. **Connected components.** Drop components below a per-region voxel floor
   (ET 5, TC 20, WT 50), keeping the largest if that would empty the mask.
3. **ET policy.** See below.

> The hierarchy constraint was previously applied innermost-outwards — ET was
> intersected with TC, and only then was TC intersected with WT. The second
> step can remove TC voxels that the first step had already admitted ET into,
> leaving ET outside the final TC. The original code happened to compensate by
> invoking the constraint three times per case, but threshold selection invoked
> it once and so swept over slightly inconsistent masks. Applying the
> constraints outermost-inwards makes a single pass correct. This is covered by
> `tests/test_eval.py::test_hierarchy_is_enforced`.

### The ET policy

Dice is all-or-nothing on a region whose ground truth is empty: a handful of
stray predicted voxels turns a correct 1.0 into 0.0. Glioblastoma cohorts
contain cases with no enhancing tumour, so how small ET predictions are handled
is worth real Dice points.

Three policies are implemented and the choice is made on validation:

- `min_volume` — zero ET when its total volume is below a cutoff. This is
  standard BraTS practice and protects true-negative ET cases. Cutoff swept
  over {0, 50, 100, 200, 300, 500}.
- `none` — leave ET as thresholded (equivalent to `min_volume` with cutoff 0).
- `rescue` — if ET is empty, re-threshold at 0.02 inside the TC mask to force
  a non-empty prediction.

`rescue` was the original hard-coded behaviour. It is backwards: it can only
ever convert a correct empty prediction into a zero-scoring one, and it cannot
help a case that already has ET. It is retained as a candidate purely so that
validation rejects it on the evidence rather than by assertion. The policy
each setup actually selected is recorded in
`results/upenn/validation_selection.csv`.

## Inference

Sliding-window at 128³ with 64³ stride, Gaussian-weighted blending so voxels
near a patch edge contribute less than voxels near its centre, and test-time
augmentation over all eight axis-flip combinations. Neither is fitted to data,
so both are applied uniformly to every setup including the zero-shot baseline.

Ensembling averages **probability maps**, not binarised masks, so the same
threshold sweep applies to single models and ensembles alike.

## Metrics

- **Dice.** Both masks empty scores 1.0.
- **HD95.** Symmetric 95th-percentile Hausdorff distance, computed over
  **surface** voxels, in millimetres at 1 mm isotropic spacing. Undefined (NaN)
  when exactly one mask is empty; 0.0 when both are.

> The previous implementation computed HD95 over *all* mask voxels rather than
> surface voxels. Interior voxels sit at distance zero from the other mask, so
> they drag the 95th percentile down and make the metric look better than it
> is. HD95 values in `results/` are consequently larger than, and not
> comparable to, any HD95 reported in the archived files. The surface-based
> definition is the standard one.

Undefined HD95 is dropped before averaging rather than imputed as zero or as
some large sentinel; the number of contributing cases is reported alongside.

## Confidence intervals

Every reported metric carries a 95% bootstrap confidence interval from 2000
resamples with a fixed seed. **Patients are the resampling unit**, not voxels,
because patients are the independent observations.

Before/after fine-tuning is compared with a **paired** bootstrap
(`brats_gbm.eval.stats.paired_diff_ci`): each resample keeps a patient's two
scores together, so the interval describes the within-patient improvement
rather than the difference of two independent means. This is considerably
tighter and is the right comparison, since both models are evaluated on the
same 29 patients.

## Statistical power

The UPenn test set has 29 subjects. That places roughly a ±0.04–0.05 interval
on mean Dice, which is wider than several of the differences between setups.
Rankings among the fine-tuned variants should not be read as established;
the fine-tuned-versus-baseline difference is far larger than the interval and
is the claim worth making.

`scripts/crossval_upenn.py` implements 5-fold cross-validation over all 147
expert-segmented subjects, which would raise the evaluated count from 29 to 147
and tighten the intervals correspondingly. Fold construction is verified
(`--dry-run`) but the folds have **not been trained** — five fine-tuning runs
cost roughly 10–14 GPU-hours. No cross-validated number is reported anywhere;
where cross-validation results are requested, the answer is that they were not
run.

## Classification: cohort definition

This is where the most consequential leakage was found.

`scripts/extract_radiomic_features.py` builds features from segmentation masks,
taking the **expert mask** where one exists (147 subjects) and a **model-predicted
mask** otherwise (482 subjects). Mask provenance is not random with respect to
the outcome:

| Mask source | n (labelled) | IDH1-mutant prevalence |
|---|---|---|
| ground truth | 116 | 0.9% |
| model predicted | 416 | 4.1% |

A 4.5-fold prevalence difference. Because provenance changes the distribution
of every shape feature — sphericity, component count, surface area — a
classifier can learn to recognise which kind of mask it is looking at and use
that as a proxy for the label. Dropping `mask_source` as an explicit column,
which the original script did, does not remove the signal; it survives in the
geometry.

Compounding it, the ground-truth subjects are exactly the subjects the
segmentor was fine-tuned on, so their masks are optimistic in a second,
independent way.

**Primary analysis** therefore uses only the 416 model-predicted subjects: one
uniform mask provenance, and zero overlap with segmentation training by
construction (a subject has a predicted mask precisely because it lacks an
expert mask, and fine-tuning required an expert mask). The mixed cohort is
reported as a sensitivity analysis so the size of the artefact is visible.

`scripts/verify_no_leakage.py` asserts the disjointness and uniform-provenance
properties and exits non-zero if either fails.

### Decision threshold

Thresholds are fitted by maximising Youden's J on **cross-fitted** scores from
within the training folds, never on the held-out fold.

> Fitting the threshold on the model's own training-fold predictions
> (resubstitution) is the obvious approach and it is wrong for high-variance
> models. A random forest separates its training data almost perfectly, so the
> resulting threshold sits near 1.0 and no held-out case ever reaches it. That
> presented as a random forest with 0.000 sensitivity — a statement about the
> threshold, not the model. With cross-fitted threshold selection the same
> model reaches 0.83 sensitivity.

### Power

17 positives in the primary cohort. Every metric is reported with a confidence
interval and those intervals are wide; the point estimates should not be
quoted without them. Repeated stratified 5-fold cross-validation (10 repeats)
is used rather than a single split, because with 17 positives one unlucky
partition moves AUC by roughly 0.1.

No hyperparameter search is performed. With this many positives a grid search
would overfit the selection itself, so both models are specified a priori and
only the decision threshold is fitted.

## Known limitations

1. BraTS2021 performance is internal validation, not a clean test set.
2. The UPenn test set has 29 subjects; cross-validation is implemented but
   unrun.
3. IDH1 classification rests on 17 positives and is underpowered. It should be
   read as a feasibility result.
4. Radiomic features for the classification cohort come from predicted masks
   whose accuracy is itself uncertain; segmentation error propagates into the
   features and is not modelled.
5. Both cohorts are glioblastoma-enriched, so the IDH1-mutant prevalence here
   is far below what a general glioma population would show. Positive
   predictive value in particular will not transfer.
