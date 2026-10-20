# paper_v2 — files behind each result

Everything here is the code and output that produced the paper's numbers. Imaging data and
model weights are not included (licence and size). Scripts under `pipeline/` are kept exactly as
run and contain the original machine's absolute paths.

## Protocol

`protocol/PROTOCOL_v2.md` — written 2026-09-24 before Run A finished; amendments are dated and
were appended before the results they govern. Rules: checkpoint = best-by-validation epoch;
post-processing chosen on validation (30-setting grid, highest Dice); test scored once per
candidate; headline = highest validation Dice; every candidate's test result reported.

Before this protocol, the same BraTS 2021 test split had been scored during development for
earlier versions of the base model; the surviving scores are in
`results/brats2021_earlier_test_scorings/` (mean Dice 0.882–0.885; the base model reported in
the paper scored 0.886).

## Code (`code/`)

| Path | Purpose |
|---|---|
| `brats_gbm/model.py` | Wavelet U-Net++ (2D or 3D Haar DWT, width F, deep supervision, BatchNorm or InstanceNorm) |
| `brats_gbm/data/brats.py`, `data/strong_aug.py` | data loading, min-max or z-score inputs, standard or strong augmentation |
| `brats_gbm/eval/` | sliding-window inference, flip TTA, post-processing, Dice/HD95, bootstrap |
| `brats_gbm/features.py`, `classification.py` | 54 region features; repeated stratified CV classifiers |
| `scripts/train_brats.py` | training (BraTS 2021 split, or `--train-ids/--val-ids` for BraTS-Africa) |
| `scripts/evaluate_brats.py` | evaluation (`--axis-order`, `--rot-tta`, `--save-probs`) |
| `scripts/infer_probs.py` | probability maps for the BraTS-Africa CV folds |
| `baselines/common/` | common scoring and parameter-count code used for every method |
| `baselines/extra3/` | SegResNet, UNETR and 3D U-Net: the official MONAI BraTS 2021 pipeline (Project-MONAI research-contributions, BRATS21, commit 21ed8e5) with only the network switched (`BRATS21/models_extra3.py`, `main.py --model_name`), test inference and the run queues (Amendments 19/19b) |
| `baselines/transbts/`, `baselines/nnformer/` | our run scripts and the patches against the official code (TransBTS @ 7dcfc31, Apache-2.0, Amendment 24; nnFormer @ d48aaeb, MIT, Amendment 26); get the official code from upstream at those commits |
| `baselines/henry_dsunet/` | our run scripts for Henry et al. (open_brats2020 @ e5f9896, Amendment 22); the upstream repository has no licence, so neither its code nor a patch is included — the corrections are described in `SOURCES.txt` |

Model variants: base (2D DWT, F=16, 10.4 M); Run A (+ deep supervision); Run C (3D DWT, F=24,
34.0 M); Run D (Run C + InstanceNorm + z-score inputs + strong augmentation, 200 epochs);
Run D2 (Run D + a second 200-epoch cosine cycle); max-pool control (Run C with 2×2×2 max-pooling + 1×1×1
projection instead of the DWT, Amendment 21); Run E (the Run D recipe, 400 epochs in one cosine cycle, Amendment 27);
Run F (Run D + phenotype-balanced case sampling, validation only, Amendment 29).

## Pipeline (`pipeline/`)

| Folder | Paper section |
|---|---|
| `brats2021/` | per-model evaluation (`eval_run.sh`, `eval_run_rot2.sh`; validation-only `eval_run_valonly.sh`, `eval_run_rot_valonly.sh`), ensemble searches, `compare.py`, training launchers, queues; `ctrl_maxpool3d/`, `runE_long400/`, `runF_phenoD/` (launchers, queue, Amendment-29 `criteria.py`), `amendment28/` (validation-only ensemble alternatives), `validation_analyses/` (`missing_stats.py` = the final paired statistics, `compare_val.py`, `val_decomposition.py`, `diagnostics.py`) |
| `brats_africa_external/` | data preparation and reorientation, patient-overlap gate, zero-shot evaluation |
| `brats_africa_adaptation/cv/` | 5-fold CV fine-tuning, per-fold selection, `compare_cv.py` |
| `classification/` | metadata, feature extraction (incl. location features), classifiers |
| `efficiency/` | parameter / FLOP / runtime benchmarks |
| `figures/` | figure scripts |

## Results (`results/`)

| Folder | Content |
|---|---|
| `brats2021/per_case/` | per-case Dice/HD95 on the 125 test cases for every candidate, the max-pool control and all six baselines |
| `brats2021/selection/` | each candidate's validation-selected settings and validation Dice |
| `brats2021/amendment28_ensemble3_val_only_report.txt` | the 17 validation-only alternatives to Ensemble 3 (Amendment 28) |
| `statistics/` | `paired_all.csv` + `report.txt`: every paired comparison in the paper (BraTS test, BraTS-Africa zero-shot and CV, adaptation gains, classification AUC differences; script `pipeline/brats2021/validation_analyses/missing_stats.py`), the matched-control result (`results_amendment21.txt`), the validation-only Run C vs control and Run E vs Run D reports |
| `runF_validation_only/` | Run F (Amendment 29): criteria logs and verdicts, per-case validation results of Run F and Run D, selections, code diff, training target list |
| `brats_africa_zeroshot/` | per-case zero-shot results on the 146 BraTS-Africa subjects |
| `brats_africa_cv/` | pooled per-case CV results (95 gliomas) for our starts and the six baselines, per-fold selections |
| `classification/` | AUC summaries and full results (BraTS-Africa); `classification/brats2021_mgmt/` BraTS 2021 MGMT results quoted in the paper |
| `efficiency/` | benchmark outputs |

File-name key: `final_ep253` = base model; `runA`, `runC`, `runD`, `runD2` = variants above;
`_rot` = with 90° rotation test-time augmentation; `ensemble`, `ensemble2`, `ensemble3` =
validation-selected ensembles; `_sel` = per-fold validation-selected option in the CV.

## Baselines on the identical split

The six baselines (Henry et al., nnFormer, SegResNet, TransBTS, UNETR, 3D U-Net) and their results are summarised in the
[root README](../README.md). 3D U-Net and UNETR collapsed under mixed precision in their first runs and were retrained once
in full precision under the rule of Amendment 19b, fixed before any of the three was evaluated. The first TransBTS run
collapsed under its default foreground-only loss and was restarted, before test scoring, with the authors' alternative
background-inclusive loss (Amendment 24, addendum 2).
