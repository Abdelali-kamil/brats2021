# Wavelet U-Net++ for brain tumour segmentation and segmentation-based classification

Code, pre-registered evaluation protocol, data splits and per-case results for the paper

> **Wavelet U-Net++ for Brain Tumour Segmentation and Segmentation-Based Classification:
> A Pre-Registered Evaluation on BraTS 2021 and BraTS-Africa**

Wavelet U-Net++ is a 3D U-Net++ whose encoder replaces pooling with a fixed Haar discrete
wavelet transform (DWT). It is compared with **nnU-Net** and **Swin UNETR** trained on the
identical split, scored by identical code, with paired bootstrap confidence intervals and
Wilcoxon tests. Every model-selection and post-processing choice was fixed on validation data
before the corresponding test result existed; the dated protocol and all of its amendments are in
[`paper_v2/protocol/PROTOCOL_v2.md`](paper_v2/protocol/PROTOCOL_v2.md).

Everything for the paper is in [`paper_v2/`](paper_v2/) — see [`paper_v2/README.md`](paper_v2/README.md)
for a file-by-file map from each table to the script and result file behind it.

## Main results

**BraTS 2021**, held-out test set (125 cases; 1,000 train / 126 validation / 125 test, seed 42).
Mean Dice with 95% bootstrap CI.

| Method | Params | ET | TC | WT | Mean Dice [95% CI] |
|---|---|---|---|---|---|
| nnU-Net | 31.2 M | 0.881 | 0.918 | 0.932 | 0.910 [0.888, 0.929] |
| Swin UNETR | 62.2 M | 0.876 | 0.908 | 0.925 | 0.903 [0.885, 0.918] |
| Wavelet U-Net++, best single model (3D DWT, rotation TTA) | 34.0 M | 0.873 | 0.906 | 0.914 | 0.898 [0.873, 0.917] |
| Wavelet U-Net++, validation-selected ensemble | 78.4 M | 0.873 | 0.899 | 0.921 | 0.898 [0.872, 0.918] |

Statistically on par with Swin UNETR, below nnU-Net. All candidates (including those that did
worse) are in `paper_v2/results/brats2021/`.

**BraTS-Africa** (95 gliomas, verified to share no patient with BraTS 2021), 5-fold
cross-validated fine-tuning (64 train / 12 validation / 19 test per fold, every subject tested once):

| Method | Zero-shot | Fine-tuned (CV) |
|---|---|---|
| nnU-Net | 0.848 | 0.885 [0.854, 0.909] |
| Swin UNETR | 0.836 | 0.877 [0.852, 0.898] |
| Wavelet U-Net++, start model fixed in advance | 0.782 | 0.848 [0.814, 0.876] |
| Wavelet U-Net++, z-score + instance norm + strong aug. (secondary) | 0.810 | 0.883 [0.854, 0.907] |

**Classification** (BraTS-Africa, glioma vs other neoplasm, 146 subjects, repeated stratified
5-fold CV): AUC 0.762 [0.679, 0.836] from our own zero-shot masks, 0.826 with expert masks,
0.484 from centre and scanner alone.

Secondary and post-hoc analyses are marked as such in the paper and in the protocol.

## UPenn-GBM is not an external test set for BraTS 2021 models

At least **107 of the 147** UPenn-GBM subjects are the same patients and scans as BraTS 2021 cases
(identical tumour masks; FLAIR correlation 0.89–0.98); 84 of them fall in our BraTS training split.
The matching table is in `paper_v2/results/upenn_overlap/`. UPenn-GBM results reported in the first
phase of this project (see `docs/README_v1.md`) are therefore **not** external results and should
not be cited as such.

## Layout

```
paper_v2/
  code/            brats_gbm package + training/evaluation scripts used for the paper
  pipeline/        orchestration and analysis scripts as they were run (BraTS 2021, BraTS-Africa,
                   classification, efficiency, figures)
  protocol/        PROTOCOL_v2.md — the dated, pre-registered protocol with every amendment
  splits/          BraTS 2021 validation/test case lists; BraTS-Africa CV folds
  results/         per-case and summary results for every model, the selection files, statistics
brats_gbm/, scripts/, baselines/, docs/, results/   first project phase (kept for the record)
```

## Data

The imaging data are not redistributed. BraTS 2021 is available from the RSNA-ASNR-MICCAI BraTS
challenge; BraTS-Africa from The Cancer Imaging Archive (CC BY 4.0). Model weights (~400 MB each)
are not in this repository.

## Note on paths

The scripts under `paper_v2/pipeline/` are the exact files that produced the results and contain
the absolute paths of the machine they ran on (`/mnt/data1/kamil_research/...`,
`/home/kamilabdelali/...`). Point those to your own data and output locations before running them.
