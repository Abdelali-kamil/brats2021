# Wavelet U-Net++ for brain tumour segmentation and segmentation-based classification

Code, pre-specified evaluation protocol, data splits and per-case results for the paper

> **Wavelet U-Net++ for Brain Tumour Segmentation and Segmentation-Based Classification:
> Benchmarking and Domain-Shift Evaluation on BraTS 2021 and BraTS-Africa**

Wavelet U-Net++ is a 3D U-Net++ whose encoder replaces pooling with a fixed Haar discrete
wavelet transform (DWT). It is compared with five published methods — **nnU-Net**, **Swin UNETR**, **SegResNet**, **UNETR** and
**3D U-Net** — each trained with its own recipe on the identical split and scored by identical code, with paired bootstrap
confidence intervals and Wilcoxon tests. Checkpoint, post-processing and ensemble selection used validation data, and each
candidate was scored on the test set once; the test split had been scored during earlier development of the base model, and
analyses added later are marked as exploratory. The dated protocol and all of its amendments are in
[`paper_v2/protocol/PROTOCOL_v2.md`](paper_v2/protocol/PROTOCOL_v2.md).

Everything for the paper is in [`paper_v2/`](paper_v2/) — see [`paper_v2/README.md`](paper_v2/README.md)
for a file-by-file map from each table to the script and result file behind it.

## Main results

**BraTS 2021**, held-out test set (125 cases; 1,000 train / 126 validation / 125 test, seed 42).
Mean Dice with 95% bootstrap CI.

| Method | Params | ET | TC | WT | Mean Dice [95% CI] |
|---|---|---|---|---|---|
| nnU-Net | 31.2 M | 0.881 | 0.918 | 0.932 | 0.910 [0.887, 0.929] |
| SegResNet | 4.7 M | 0.871 | 0.917 | 0.927 | 0.905 [0.884, 0.923] |
| Swin UNETR | 62.2 M | 0.876 | 0.908 | 0.925 | 0.903 [0.885, 0.918] |
| UNETR | 102.2 M | 0.851 | 0.877 | 0.914 | 0.881 [0.860, 0.899] |
| 3D U-Net | 16.3 M | 0.832 | 0.858 | 0.896 | 0.862 [0.835, 0.887] |
| Wavelet U-Net++, best single model (3D DWT, rotation TTA) | 34.0 M | 0.873 | 0.906 | 0.914 | 0.898 [0.873, 0.916] |
| Wavelet U-Net++, validation-selected ensemble (headline) | 78.3 M | 0.875 | 0.902 | 0.921 | 0.899 [0.874, 0.919] |

Not significantly different from Swin UNETR and SegResNet (equivalence was not tested), better than UNETR and 3D U-Net,
below nnU-Net. After adaptation on BraTS-Africa our model does not differ significantly from nnU-Net (+0.002 [−0.011, +0.016];
ours higher on 35/95 cases). Mean HD95 leaves out undefined values (one empty mask); with them set to 373.13 mm the headline
ensemble's HD95 is 6.92 mm (nnU-Net 3.83) — see `paper_v2/results/review_2026-10-10/`. All candidates (including those that did
worse) are in `paper_v2/results/brats2021/`.

**BraTS-Africa** (95 gliomas; no overlap with BraTS 2021 detected by our screening), 5-fold
cross-validated fine-tuning (64 train / 12 validation / 19 test per fold, every subject tested once):

| Method | Zero-shot | Fine-tuned (CV) |
|---|---|---|
| nnU-Net | 0.848 | 0.885 [0.854, 0.909] |
| Swin UNETR | 0.836 | 0.877 [0.852, 0.898] |
| Wavelet U-Net++, start model fixed in advance | 0.782 | 0.848 [0.814, 0.876] |
| Wavelet U-Net++, z-score + instance norm + strong aug., 200 epochs (secondary) | 0.810 | 0.883 [0.854, 0.907] |
| Wavelet U-Net++, same, 400 epochs (secondary) | 0.803 | 0.887 [0.858, 0.911] |

**Classification** (BraTS-Africa, glioma vs other neoplasm, 146 subjects, repeated stratified
5-fold CV): AUC 0.762 [0.679, 0.836] from our own zero-shot masks, 0.826 with expert masks,
0.484 from centre and scanner alone. Paired differences in AUC between mask sources (expert vs ours +0.064
[−0.009, +0.137]; each baseline vs ours −0.014 to +0.009) are not significant.

Secondary and post-hoc analyses are marked as such in the paper and in the protocol.

## Peer-review revision (2026-10-10)

- `paper_v2/pipeline/review_2026-10-10/` and `paper_v2/results/review_2026-10-10/`: analyses added after an external
  review, on existing results only — undefined and penalised HD95, paired AUC differences between mask sources, and three
  small checks (Run A without its two collapsed cases, Run D2 vs Run D zero-shot, Run C vs base with penalised HD95).
- **Non-wavelet control (Amendment 21, running):** Run C retrained with 2×2×2 max-pooling and a 1×1×1 projection in place
  of the 3D Haar DWT, everything else identical. The design, the comparison and its interpretation rule were written in the
  protocol before training started; scripts in `paper_v2/pipeline/brats2021/ctrl_maxpool3d/`. Its result will be added here
  whatever it shows.

## Layout

```
paper_v2/
  code/            brats_gbm package + training/evaluation scripts used for the paper
  pipeline/        orchestration and analysis scripts as they were run (BraTS 2021, BraTS-Africa,
                   classification, efficiency, figures)
  protocol/        PROTOCOL_v2.md — the dated, pre-specified protocol with every amendment
  splits/          BraTS 2021 validation/test case lists; BraTS-Africa CV folds
  results/         per-case and summary results for every model, the selection files, statistics
```

## Data

The imaging data are not redistributed. BraTS 2021 is available from the RSNA-ASNR-MICCAI BraTS
challenge; BraTS-Africa from The Cancer Imaging Archive (CC BY 4.0). Model weights (~400 MB each)
are not in this repository.

## Note on paths

The scripts under `paper_v2/pipeline/` are the exact files that produced the results and contain
the absolute paths of the machine they ran on (`/mnt/data1/kamil_research/...`,
`/home/kamilabdelali/...`). Point those to your own data and output locations before running them.
