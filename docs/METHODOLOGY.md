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

## The reported configuration involves no selection

**Primary results use a fixed configuration: threshold 0.5 in every region,
standard connected-component cleanup, no ET volume rule, no WT
largest-component rule.** It is identical for every setup and fitted to
nothing.

That is a retreat from the validation-selection machinery described below, and
it was forced by evidence. Widening the threshold grid and adding the WT policy
pushed selection to six parameters — three thresholds, ET policy, ET cutoff, WT
policy — chosen on a 14-subject validation split. Six parameters on fourteen
subjects does not work, and the damage is measurable. For `upenn_v3_best`:

| Search space | Validation | Test |
|---|---|---|
| Narrow grid, no WT policy | 0.8380 | **0.8157** |
| Wide grid + WT policy | 0.8512 ↑ | **0.8070** ↓ |

Validation improved because the search found configurations fitting those 14
subjects; test degraded because they did not generalise. The fingerprint is in
the chosen parameters: `upenn_v3_best` was the only setup to select the unusual
combination `WT=largest, ET cutoff 300`, and it fell furthest.

`scripts/selection_sensitivity.py` scores every setup both ways:

| Setup | Fixed | Validation-selected | Δ |
|---|---|---|---|
| `ensemble_v3_v4` | **0.8204** | 0.8128 | +0.0075 |
| `upenn_v3_best` | 0.8156 | 0.8070 | +0.0086 |
| `ensemble_top5` | 0.8131 | 0.8095 | +0.0036 |
| `upenn_v4_best` | 0.8071 | 0.8048 | +0.0023 |
| `upenn_v3_last` | 0.8060 | 0.8056 | +0.0005 |
| `baseline_brats` | 0.6948 | 0.6968 | −0.0019 |

Fixed wins on five of six. Selection was not merely failing to help; it was
costing performance.

### On the honesty of this choice

Choosing a selection strategy by its test performance is test-set tuning one
level up, and it would be self-defeating to fix one form of leakage by
introducing another. The decision to lead with the fixed configuration rests on
the sample-size argument — 14 subjects cannot support six selected parameters —
which holds regardless of how the test numbers landed. The table above is
reported so a reader can check that reasoning rather than accept it, and both
configurations ship in `results/upenn/`.

The proper remedy is not a better selection heuristic but a larger selection
set. `scripts/crossval_upenn.py` would give 147 subjects instead of 14; it is
implemented and unrun.

Everything below describes the validation-selection machinery, which remains in
the code and is reported as a sensitivity analysis.

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

## The KAN + Transformer + GNN module: BraTS primary, UPenn external

`scripts/ablation_kan_gnn.py` and `brats_gbm/gnn.py` implement the deep
classification module (KAN encoders, a region-token Transformer branch, a
retrieval-augmented memory-bank GNN, and adaptive gated fusion). Its **primary
task is BraTS 2021 MGMT methylation** — the only classification label BraTS 2021
carries; there is no tumour-grade label, so grading is not attempted. UPenn-GBM
is used **only for external validation**.

Two constraints are forced by the data and are stated rather than worked around.
BraTS 2021 ships no clinical metadata, so on the primary task the module is
**imaging-only**: the clinical encoder and the gate have nothing to fuse, and
are exercised only on the UPenn cohort. And MGMT-from-MRI is a task with
contested signal, so a near-chance result is an expected, legitimate outcome.

The primary ablation is nested 5×3 cross-validation, 3 repeats, on the 577-case
BraTS cohort, restricted to the 51 radiomic features shared with UPenn so the
representation is identical to the external test. Point estimates with 95%
bootstrap CIs (`results/classification/kan_gnn_brats.json`):

| Configuration | AUC [95% CI] |
|---|---|
| MLP (baseline) | 0.617 [0.568, 0.663] |
| KAN | 0.602 [0.553, 0.648] |
| KAN + Transformer | 0.645 [0.599, 0.692] |
| KAN + GNN | 0.615 [0.568, 0.662] |
| KAN + Transformer + GNN (full) | 0.624 [0.575, 0.670] |

Every interval overlaps every other; the mean 95% CI half-width is ±0.047, and
no contrast between rungs exceeds it. The full module (0.624) is not
distinguishable from the plain MLP (0.617); KAN does not beat MLP, matching the
parameter-count argument in `kan.py`; the memory bank draws ~25% of its
representation from neighbours (`graph_reliance` ≈ 0.25) but that does not become
an AUC gain. These BraTS numbers sit where the radiomic baselines do
(best AUC 0.583, logistic regression) and where the RSNA-MICCAI 2021 challenge
landed (~0.62).

**External validation.** The BraTS-trained full model, applied to the 227-subject
UPenn MGMT cohort on the shared feature space, scores **0.537 [0.460, 0.612]** —
an interval spanning chance. The weak BraTS-internal signal does not transfer.
UPenn feature provenance differs (predicted rather than expert masks), which is
itself part of what external validation exposes.

**Multimodal reference.** Because clinical covariates exist only for UPenn, the
imaging+clinical gated variant can be shown only there: nested CV on UPenn gives
**0.599 [0.523, 0.671]**, no better than BraTS imaging-only. Adding clinical data
and the gate does not rescue the task.

The reported conclusion is therefore a negative one, and it is reported as such:
the module runs, the leakage controls hold, and MGMT methylation is not
recoverable from these features at a level that clears chance out of sample. No
configuration is selected on the quantity being reported.

## The downsampling ablation: DWT vs max-pooling

The paper's segmentation contribution is a single architectural change — a fixed
Haar DWT replacing max-pooling in the encoder — so the claim rests on comparing
against an otherwise-identical max-pooling backbone. `scripts/run_ablation.sh`
runs that comparison.

**The baseline is width-matched, not a plain swap.** The DWT concatenates four
sub-bands, so every encoder block receives 4C channels; plain max-pooling emits
C. Swapping the operator alone therefore also removes 34% of the parameters
(6.87M vs 10.40M), and any resulting gap would confound capacity with the
frequency content the claim is actually about. `WaveletUNetPlusPlus(downsample=
"maxpool_matched")` pools over the same two in-plane axes and then applies a 1x1
projection back to 4C, so both arms present identically shaped tensors at every
encoder level and sit at 10.48M vs 10.40M parameters — a 0.8% difference.

**Both arms are retrained.** The released `segmentor_epoch_650.pth` is a bare
state_dict carrying no optimiser state, seed, epoch count or configuration, no
training log survives, and the training script could not run as released (see
the import defect below), so its protocol cannot be reconstructed and matched.
Comparing a freshly trained baseline against it would not be a controlled
experiment. Both arms are therefore trained from scratch under one recorded
protocol, on the identical seed-42 1000/251 partition, and this pair is used
**only** for the ablation. Every other number in the project continues to come
from the original checkpoint, which is untouched.

Training settings are identical across arms and unchanged from the rest of the
project: AdamW at 2e-4, `ReduceLROnPlateau` on validation Dice (factor 0.5,
patience 3, floor 1e-7), focal+Dice loss, effective batch 8, mixed precision,
min-max normalisation, deterministic 128^3 centre crops and no augmentation.
Only `--downsample` differs. Scoring uses the reported evaluation pipeline
(sliding window with Gaussian blending, 8-flip TTA, fixed threshold 0.5,
standard component cleanup) with patient-level bootstrap intervals.

**Epoch budget: 200, matched, best-validation checkpoint reported.** This was
not the initial setting and the correction is worth recording. `train_brats.py`
defaults to `--epochs 750`, but that is a *resume* target: the original run
continued an existing epoch-650 checkpoint for a further 100 epochs. Carried
over as a from-scratch budget it is roughly an order of magnitude too large for
this schedule. The max-pooling arm reached its best validation Dice (0.8199) at
**epoch 43** and its 1e-7 learning-rate floor at **epoch 76**, then oscillated
between 0.773 and 0.801 — noise around a converged model — for 150 further
epochs. Running to 750 would have added ~53 h per arm at a learning rate too
small to change the weights. 200 epochs clears convergence with a wide margin
while keeping the two arms matched; `patience=3` is aggressive enough that
eleven halvings fit between epochs 31 and 76.

**An import defect blocked training entirely.** `brats_gbm/data/brats.py` used a
bare `from image import ...`, which resolves only when `brats_gbm/data/` happens
to be on `sys.path` — which `train_brats.py` never arranged. An
`except ImportError` swallowed the failure and printed a warning, so it surfaced
much later as a `NameError` inside a DataLoader worker. Every other module in
the repository imports those helpers as `from brats_gbm.data.image import ...`;
this file was the sole exception and `train_brats.py` its only consumer, so
BraTS training could not run from the repository as it stood. Now fixed to match
the rest of the codebase, with the fallback left to raise rather than swallow.

Two further consequences are worth recording. Because the released checkpoint
cannot have been produced by the code as released, its training protocol is
genuinely unknown, and the Methods section describes the current code rather
than a verified history. And every checkpoint written from now on embeds the
configuration that produced it — operator, seed, optimiser, schedule, loss,
normalisation, augmentation state, split sizes, torch version, timestamp — so
this particular gap cannot recur.

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
6. The KAN + Transformer + GNN module does not exceed a plain MLP on BraTS MGMT
   and transfers to UPenn at chance. This is reported as a negative result; it
   is not evidence that the architecture is useful for this task.
