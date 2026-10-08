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
the paper scored 0.886). The same folder holds the test score of Swin UNETR's last epoch (0.862);
the reported Swin UNETR checkpoint is its best-by-validation epoch, as for every model.

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
| `baselines/` | nnU-Net and Swin UNETR set-up, conversion and scoring on the identical split |
| `baselines/extra3/` | SegResNet, UNETR and 3D U-Net: the official MONAI BraTS 2021 pipeline (as for Swin UNETR) with only the network switched (`BRATS21/models_extra3.py`, `main.py --model_name`), test inference and the run queues (Amendments 19/19b) |

Model variants: base (2D DWT, F=16, 10.4 M); Run A (+ deep supervision); Run C (3D DWT, F=24,
34.0 M); Run D (Run C + InstanceNorm + z-score inputs + strong augmentation, 200 epochs);
Run D2 (Run D + a second 200-epoch cosine cycle).

## Pipeline (`pipeline/`)

| Folder | Paper section |
|---|---|
| `brats2021/` | per-model evaluation (`eval_run.sh`, `eval_run_rot2.sh`), ensemble searches, `compare.py` (paired statistics), training launchers, queues |
| `brats_africa_external/` | data preparation and reorientation, patient-overlap gate, zero-shot evaluation |
| `brats_africa_adaptation/cv/` | 5-fold CV fine-tuning, per-fold selection, `compare_cv.py` |
| `classification/` | metadata, feature extraction (incl. location features), classifiers |
| `efficiency/` | parameter / FLOP / runtime benchmarks |
| `figures/` | figure scripts |

## Results (`results/`)

| Folder | Content |
|---|---|
| `brats2021/per_case/` | per-case Dice/HD95 on the 125 test cases for every candidate and all five baselines |
| `brats2021/selection/` | each candidate's validation-selected settings and validation Dice |
| `brats2021/compare_test.txt` | paired bootstrap + Wilcoxon against nnU-Net, Swin UNETR, SegResNet, UNETR and 3D U-Net |
| `brats2021/extra_paired_stats.txt` | the remaining paired statistics quoted in the text (HD95 comparisons, Run C vs base, Run D vs Run C zero-shot, adaptation-gain differences); script `pipeline/brats2021/extra_paired_stats.py` |
| `brats_africa_zeroshot/` | per-case zero-shot results on the 146 BraTS-Africa subjects |
| `brats_africa_cv/` | pooled per-case CV results (95 gliomas), per-fold selections, `compare_cv.txt` |
| `classification/` | AUC summaries and full results (BraTS-Africa); `classification/brats2021_mgmt/` BraTS 2021 MGMT results quoted in the paper |
| `efficiency/` | benchmark outputs |

File-name key: `final_ep253` = base model; `runA`, `runC`, `runD`, `runD2` = variants above;
`_rot` = with 90° rotation test-time augmentation; `ensemble`, `ensemble2`, `ensemble3` =
validation-selected ensembles; `_sel` = per-fold validation-selected option in the CV.

## Baselines on the identical split (BraTS 2021 test, 125 cases, scored once)

| Method | Params | Mean Dice [95% CI] | HD95 (mm) | BraTS-Africa zero-shot (95 gliomas) |
|---|---|---|---|---|
| nnU-Net | 31.2 M | 0.910 [0.887, 0.929] | 2.89 | 0.848 |
| SegResNet | 4.7 M | 0.905 [0.884, 0.923] | 3.59 | 0.846 |
| Swin UNETR | 62.2 M | 0.903 [0.885, 0.918] | 4.10 | 0.836 |
| **Wavelet U-Net++ (Ensemble 3, headline)** | 78.3 M | 0.899 [0.874, 0.919] | 3.96 | 0.803 (Run D2) |
| UNETR | 102.2 M | 0.881 [0.860, 0.899] | 5.39 | 0.799 |
| 3D U-Net | 16.3 M | 0.862 [0.835, 0.887] | 7.45 | 0.805 |

3D U-Net and UNETR collapsed under mixed precision in their first runs and were retrained once in full precision
under the rule of Amendment 19b, fixed before any of the three was evaluated.
