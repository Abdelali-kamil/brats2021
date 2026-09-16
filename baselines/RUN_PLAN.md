# Reproducing the comparison table — run plan

Companion to `baselines/SOURCES.md`. That file records **what the code is**
(repo, pinned commit, license); this one records **how we run it**, what is
done, and what is still open.

Rule that governs everything here: every number in the paper's comparison table
is produced by the method's own official code, run on this server, on our split,
and scored by our scorer. Nothing is copied from a source paper.

Last updated: 2026-09-15.

## Status

| Method | Task | Env | Data prep | Train | Scored |
|---|---|:--:|:--:|:--:|:--:|
| `xgboost` | cls | done | done | done | **done** |
| `nnunet` | seg | done | running | – | – |
| `swin_unetr` | seg | – | – | – | – |
| `selfmedmae` | seg (+cls) | – | – | – | – |
| `mtanet` | seg + cls | **dropped** | – | – | – |
| `mmgl` | cls | – | – | – | – |
| `daft` | cls | – | – | – | – |
| `ours` | seg | done | done | **running** | – |

`ours` is the `aug_cosine_repro1` run regenerating the segmentation model after
the original checkpoints were lost; see that run's `PROVENANCE.txt`. Until it
finishes, there is no model to produce the `ours` row from.

## The split every method must use

Frozen to `baselines/common/brats_split_frozen.{json,csv}`, generated from
`brats_gbm.splits.brats_split_3way(seed=42)`:

- **train — 1000 cases.** The only cases any method may train on.
- **val — 126 cases.** Checkpoint selection and any tuning.
- **test — 125 cases.** Scored once, at the end.
- **heldout_251 = val + test.** The internal-validation partition the paper's
  current comparison table reports on.

Verified 2026-09-15: the baselines repo's 2-way `brats_split` and the training
repo's 3-way `brats_split_3way` yield the **identical** 1000 training cases, and
val + test is exactly the 251-case held-out set, with no train overlap. That
equality is what makes baselines trained here comparable to `ours`.

## Order of work

GPU is a single RTX 5090, **shared with another user**. One training job at a
time; CPU-only steps run alongside.

1. **CPU now, while `ours` trains.** nnU-Net environment, BraTS→nnU-Net
   conversion, and `nnUNetv2_plan_and_preprocess`. Planning is CPU/disk work and
   must happen before training regardless, so it costs nothing to do now.
2. **Classification baselines next — `daft`, `mmgl`.** Far cheaper than the
   segmentation networks and small enough to fit without starving the repro run.
3. **`nnunet` once `ours` finishes.** The baseline reviewers will look for first.
4. **`swin_unetr`, then `selfmedmae`.** Both full segmentation trainings.
5. **Score everything, including `ours`, through
   `baselines/common/score_segmentation.py`** — one scorer, no method scored by
   its own repo's metric — then `make_comparison_table.py`.

## GPU budget — read before committing

Rough orders of magnitude on one shared RTX 5090, **not measured yet**; treat as
planning figures and replace with real timings as runs complete.

| Method | Default schedule | Estimate |
|---|---|---|
| `nnunet` 3d_fullres | 1000 epochs × 250 iters, per fold | ~2–4 days / fold |
| `swin_unetr` | BRATS21 recipe default | ~2–3 days |
| `selfmedmae` | pre-train + fine-tune | ~3–5 days |
| `daft` | 3D CNN + tabular | hours |
| `mmgl` | graph over tabular cohort | hours |

nnU-Net's standard recipe is 5 folds. Running one fold is defensible and far
cheaper, but it must be **stated in the paper** as single-fold rather than the
5-fold ensemble the nnU-Net paper reports.

Full roster at default settings is a multi-week commitment. Shortening any
method's own training schedule would defeat the purpose — the number would no
longer be what their code produces — so the lever is *which* methods run, not
how long each trains.

## Open decisions

**1. Which partition does the table report?**
The current `comparison.md` and the paper report the **251-case internal
validation**. The **125-case clean test** is the stronger claim, and since no
baseline has trained yet, switching now is free; after they train it would mean
redoing them. Decide before step 2.

**2. MTANet.** Its repo is model-definition code only — no trainer, no loader,
no config, no license (`SOURCES.md` has the inventory). Producing a row means
writing the training pipeline ourselves, which the paper would have to disclose
as ours rather than the authors'. The alternative is to cite it as not
reproduced, the same treatment as MMCL and ResGANet. **Unresolved.**

**3. Single-fold vs 5-fold nnU-Net** — see the budget table above.

## Scoring contract

Segmentation, every method including ours:

```bash
python baselines/common/score_segmentation.py --key <key> \
       --pred <pred_dir> --gt <gt_seg_dir>      # nnU-Net: add --et-label 3
```

Writes `results/baselines/<key>_segmentation.json`. Classification baselines
write `results/baselines/<key>_classification.json`. Then:

```bash
python baselines/common/make_comparison_table.py   # -> comparison.md + .csv
```

Never hand-write a result JSON. A row with no result file stays `pending`, which
is the honest state — an empty cell is a fact, a borrowed number is not.

## Expected honesty caveat

On our radiomic features under our CV protocol, XGBoost reaches **AUC ≈ 0.57 on
MGMT — near chance**, consistent with the project's own fusion (~0.63) and
imaging-only (~0.54) numbers. The 0.80–0.89 AUCs some baseline papers report came
from their own cohorts. Expect the classification baselines here to land in the
same near-chance range, and report what the runs give.


## Measured: parameter counts (2026-09-15)

Each model instantiated from its own repo at the pinned commit, with that repo's
own BraTS configuration, counted by one function
(`baselines/common/count_parameters.py` -> `results/baselines/parameter_counts.json`).

| Method | Parameters | Config |
|---|---|---|
| **Wavelet U-Net++ (ours)** | **10.40 M** | in=4, classes=3 |
| Swin UNETR | **62.19 M** | feature_size 48, roi 96^3 (BRATS21 defaults) |
| SelfMedMAE | needs own env | timm <0.9; see below |
| MTANet | n/a | see below |
| nnU-Net | pending planning | self-configures per dataset |

Ours is **6x smaller than Swin UNETR**. If the Dice gap turns out to be small,
that is a real efficiency result -- and one only stateable because both numbers
were measured here rather than quoted.

## MTANet is dropped from the reproduced table

Two independent blockers, verified in the pinned clone:

1. **It is a 2D network.** `MATNet.py` and `pvtv2.py` hold 39 and 8 `Conv2d`
   layers and **zero** `Conv3d`. BraTS is 3D volumetric data, so the published
   model cannot accept this task's input at all.
2. `MTANet.__init__` unconditionally loads `lib/pvt_v2_b2.pth`, a pretrained
   backbone the repository never ships.

This has a consequence for the paper independent of any reproduction: the
**88.7 Dice currently credited to MTANet is a 2D result** being compared against
3D BraTS Dice. That row must go regardless, alongside MMCL and ResGaNet, which
have no code at all.

## SelfMedMAE needs its own environment

Its pinned 2024 code calls `timm.models.vision_transformer.Block(drop=...)`,
renamed `proj_drop` in timm 1.x, and imports `timm.models.layers.helpers`, moved
to `timm.layers`. It needs a venv with timm <0.9 -- which is exactly why the
harness mandates one environment per method. Worth keeping on the roster: it is
the only joint segmentation+classification method in the table that ships code.

## Correction to nnU-Net's SETUP.md (2026-09-15)

Two errors found while running it, now fixed:

* It specified the **cu121** torch wheel. cu121 does **not** support this
  machine's RTX 5090 (Blackwell, sm_120) -- following it would have installed a
  torch that cannot use the GPU. Let pip resolve a CUDA 13 build.
* It showed `Dataset137_BraTS21.py -i /path/to/data`. That script takes **no
  arguments** and copies every case into `imagesTr`, which would put the 251
  held-out cases into nnU-Net's training set. Use
  `baselines/nnunet/convert_split.py`, which calls nnU-Net's own label
  conversion and dataset-json generation but honours the frozen split.
