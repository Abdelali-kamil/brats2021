# Wavelet U-Net++ for brain tumour segmentation and segmentation-based classification

Code, pre-specified evaluation protocol, data splits and per-case results for the paper

> **Wavelet U-Net++ for Brain Tumour Segmentation and Segmentation-Based Classification:
> Benchmarking and Domain-Shift Evaluation on BraTS 2021 and BraTS-Africa**

Wavelet U-Net++ is a 3D U-Net++ whose encoder replaces pooling with a fixed Haar discrete
wavelet transform (DWT). It is compared with a matched max-pooling control and six published methods — **Henry et al.**
(deeply supervised 3D U-Net), **nnFormer**, **SegResNet**, **TransBTS**, **UNETR** and **3D U-Net** — each trained with its own
recipe on the identical split and scored by identical code, with paired bootstrap confidence intervals and Wilcoxon tests.
Checkpoint, post-processing and ensemble selection used validation data, and each candidate was scored on the test set once;
the test split had been scored during earlier development of the base model, and analyses added later are marked as
exploratory. The dated protocol and all of its amendments (1–29) are in
[`paper_v2/protocol/PROTOCOL_v2.md`](paper_v2/protocol/PROTOCOL_v2.md).

Everything for the paper is in [`paper_v2/`](paper_v2/) — see [`paper_v2/README.md`](paper_v2/README.md)
for a file-by-file map from each table to the script and result file behind it.

## Main results

**BraTS 2021**, held-out test set (125 cases; 1,000 train / 126 validation / 125 test, seed 42).
Mean Dice with 95% case-bootstrap CI; HD95\* replaces undefined regional distances by the image diagonal (373.13 mm).

| Method | Params | ET | TC | WT | Mean Dice [95% CI] | HD95\* (mm) |
|---|---|---|---|---|---|---|
| Henry et al. | 23.2 M | 0.881 | 0.914 | 0.931 | 0.909 [0.891, 0.924] | 3.32 |
| nnFormer | 37.6 M | 0.872 | 0.920 | 0.933 | 0.908 [0.890, 0.923] | 3.99 |
| SegResNet | 4.7 M | 0.871 | 0.917 | 0.927 | 0.905 [0.884, 0.923] | 4.52 |
| TransBTS | 33.0 M | 0.870 | 0.897 | 0.920 | 0.896 [0.872, 0.916] | 6.68 |
| UNETR | 102.2 M | 0.851 | 0.877 | 0.914 | 0.881 [0.860, 0.899] | 6.33 |
| 3D U-Net | 16.3 M | 0.832 | 0.858 | 0.896 | 0.862 [0.835, 0.887] | 10.27 |
| Wavelet U-Net++, Run C (3D DWT, single model) | 34.0 M | 0.871 | 0.907 | 0.914 | 0.897 [0.873, 0.916] | 6.66 |
| Wavelet U-Net++, Run C + rotation TTA | 34.0 M | 0.873 | 0.906 | 0.914 | 0.898 [0.873, 0.916] | 6.70 |
| Wavelet U-Net++, Ensemble 3 (validation-selected headline) | 78.3 M | 0.875 | 0.902 | 0.921 | 0.899 [0.874, 0.919] | 6.92 |

Paired mean-Dice differences, Ensemble 3 minus baseline: Henry et al. −0.0096 [−0.0236, +0.0004], nnFormer −0.0091
[−0.0241, +0.0020], SegResNet −0.0059 [−0.0151, +0.0016], TransBTS +0.0033 [−0.0070, +0.0138], UNETR +0.0184
[+0.0029, +0.0334], 3D U-Net +0.0370 [+0.0221, +0.0535]. The intervals include zero against the first four baselines
(neither superiority nor equivalence is established) and favour the ensemble against UNETR and 3D U-Net. Run C alone is
below Henry et al. (−0.0114 [−0.0259, −0.0005]). Every candidate, including those that did worse, is in
`paper_v2/results/brats2021/`; all paired comparisons are in `paper_v2/results/statistics/paired_all.csv`.

**Matched wavelet control** (Amendment 21; same width, decoder, loss, 400-epoch budget and seed; 2×2×2 max-pooling +
1×1×1 projection instead of the 3D Haar DWT; one seed per arm):

| Measure | 3D Haar DWT (Run C) | Max-pool control | DWT − control [95% CI] |
|---|---|---|---|
| BraTS validation Dice | 0.8994 | 0.9008 | −0.0014 [−0.0108, +0.0069] |
| BraTS test Dice (primary) | 0.8973 | 0.8883 | +0.0090 [+0.0023, +0.0173] |
| BraTS-Africa gliomas, zero-shot | 0.7775 | 0.7602 | +0.0173 [+0.0036, +0.0316] |
| BraTS-Africa other neoplasms, zero-shot | 0.7003 | 0.7228 | −0.0226 [−0.0670, +0.0210] |

**BraTS-Africa** (95 gliomas; no overlap with BraTS 2021 detected by our screening): zero-shot, and 5-fold
cross-validated fine-tuning (64 train / 12 validation / 19 test per fold, every subject tested once).

| Method | Zero-shot [95% CI] | Fine-tuned [95% CI] | Gain [95% CI] |
|---|---|---|---|
| Henry et al. | 0.828 [0.787, 0.863] | 0.867 [0.834, 0.897] | +0.039 [+0.023, +0.058] |
| nnFormer | 0.857 [0.828, 0.881] | 0.866 [0.832, 0.896] | +0.010 [−0.005, +0.023] |
| SegResNet | 0.846 [0.810, 0.876] | 0.879 [0.845, 0.906] | +0.033 [+0.020, +0.048] |
| TransBTS | 0.806 [0.769, 0.838] | 0.857 [0.817, 0.890] | +0.050 [+0.034, +0.067] |
| UNETR | 0.799 [0.762, 0.832] | 0.837 [0.802, 0.869] | +0.038 [+0.025, +0.051] |
| 3D U-Net | 0.805 [0.764, 0.840] | 0.816 [0.775, 0.855] | +0.012 [+0.002, +0.022] |
| Ours, Ensemble 3 (zero-shot rule; not fine-tuned) | 0.797 [0.750, 0.838] | – | – |
| Ours, Run A start (primary CV, fixed in advance) | 0.782 [0.743, 0.818] | 0.848 [0.813, 0.879] | +0.066 [+0.047, +0.087] |
| Ours, Run C start (secondary) | 0.778 [0.729, 0.820] | 0.869 [0.836, 0.896] | +0.091 [+0.065, +0.122] |
| Ours, Run D start (secondary) | 0.810 [0.768, 0.846] | 0.883 [0.854, 0.908] | +0.074 [+0.054, +0.097] |
| Ours, Run D2 start (secondary) | 0.803 [0.759, 0.842] | 0.887 [0.856, 0.911] | +0.083 [+0.058, +0.113] |

The secondary starts were chosen after earlier results on the same subjects and do not replace the primary Run A start.

**Classification** (BraTS-Africa, glioma vs other neoplasm, 146 subjects, stratified 5-fold CV repeated 10 times):
logistic-regression AUC 0.762 [0.679, 0.836] from our own zero-shot masks (Run C), 0.826 with expert masks, 0.748–0.799
with the six baselines' masks, 0.484 from centre and scanner alone. Every paired AUC difference from the Run C masks has an
interval that includes zero.

**Exploratory, validation-only analyses** (not used to change the headline): Run E (the Run D recipe for 400 epochs in one
cosine cycle; test 0.889), Amendment 28 (17 ensemble alternatives, none met the improvement rule) and Run F (Amendment 29,
phenotype-balanced case sampling: the targeted subgroup improved but the overall validation mean fell, 0.8905 vs 0.8937 for
Run D; not adopted, no test scoring; `paper_v2/results/runF_validation_only/`).

## Layout

```
paper_v2/
  code/            brats_gbm package + training/evaluation scripts; baselines/ = how each baseline was run on our split
  pipeline/        orchestration and analysis scripts as they were run (BraTS 2021, BraTS-Africa,
                   classification, efficiency, figures)
  protocol/        PROTOCOL_v2.md — the dated, pre-specified protocol with every amendment (1–29)
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
