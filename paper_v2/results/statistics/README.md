# Missing statistics (2026-10-19)

Analysis of saved results only: no inference, no training, and no TEST case scored again.
- Script: `missing_stats.py`. Outputs: `report.txt` (readable), `paired_all.csv` (every comparison; `status` column =
  `new` / `reused-verified` / `recomputed (recorded value differs)`), `run.log`.
- VAL-only Run C vs max-pool control: `val_runC_vs_ctrl_maxpool3d.txt`, made with `../runE_vs_runD/compare_val.py`. That
  script also writes `per_case_val_runC.csv` and `per_case_val_ctrl_maxpool3d.csv` into `../runE_vs_runD/`.

## Conventions
- Single-method CI: bootstrap over cases, 2,000 resamples, seed 42 (`brats_gbm.eval.stats.bootstrap_ci`), as in the paper.
- Paired difference: bootstrap, 10,000 resamples, a fresh seed-0 generator per comparison, plus two-sided Wilcoxon.
  This is the convention of extra_paired_stats.py, nnformer_paired.py and Amendments 21/28.
- Classification: logistic-regression out-of-fold probabilities, paired patient-level bootstrap (2,000, seed 0).

## Inventory: existing comparisons, reused after recomputation
| Comparison | Recorded | Recomputed | Status |
|---|---|---|---|
| Ens3 − nnFormer, BraTS TEST | −0.0091 [−0.0241, +0.0020] p=0.66 | identical | reused: same per-case files (Ens3 10-03, nnFormer 10-17 12:29, unchanged), same 125 cases, same method |
| Ens3 − SegResNet / Henry / UNETR / 3D U-Net, TEST | −0.0059 [−0.0151, +0.0016] / −0.010 [−0.024, +0.000] / +0.018 [+0.003, +0.033] / +0.037 [+0.022, +0.054] | match (3 dp) | reused |
| Run C − SegResNet / 3D U-Net, TEST | paper values | match | reused |
| Run C − max-pool control, TEST (Amendment 21) | +0.0090 [+0.0023, +0.0173], HD95* −2.26 | identical | reused |
| Africa ZS Run C − SegResNet / UNETR / 3D U-Net | −0.068 / −0.022 / −0.027 | match | reused |
| Africa CV D2 − SegResNet / Henry / UNETR / 3D U-Net | +0.008 / +0.020 / +0.050 / +0.071 | match | reused |
| CV gains: Run A start, nnFormer, TransBTS | +0.066 / +0.0095 / +0.0501 | match | reused |
| AUC (8 mask sources) and paired AUC: expert, SegResNet, UNETR, 3D U-Net − Run C | 10-10 / 10-17 values | identical | reused |

"Recorded value differs": point estimates match. Interval bounds differ by ≤ 0.002 because the earlier values came from
compare.py / compare_cv.py, which draw bootstraps from a shared random stream and were rounded to 3 dp. Cases:
Run C (and Run C + rot) − UNETR, Run C + rot − 3D U-Net, and the CV gains from Run C / D / D2.
**Recommendation:** quote every interval from `paired_all.csv`, so the paper uses one convention throughout.

## New comparisons (not computed before)
- BraTS TEST:
  - Run C, Run C + rot and Ens3 vs Henry, nnFormer and TransBTS (Ens3 vs Henry and nnFormer were reused);
  - HD95* paired comparisons for all pairs.
- Africa zero-shot (95 gliomas):
  - Ens3 (the BraTS-VAL winner, Amendment 1 rule) vs all six baselines;
  - Run C vs Henry, nnFormer and TransBTS;
  - Ens3 vs Run C.
- Africa CV:
  - primary (start model Run A, fixed in advance) vs all six baselines;
  - Run C / D starts vs all six;
  - D2 start vs nnFormer and TransBTS;
  - UNETR, SegResNet, Henry and 3D U-Net CIs and gains;
  - gain differences (Run A start vs each baseline).
- Classification: paired AUC for Henry, nnFormer and TransBTS minus Run C.
- VAL: Run C vs max-pool control, paired (−0.0014 [−0.0108, +0.0069], Wilcoxon p = 0.014, Run C better on 77/126).
  The mean gap comes from the two no-core cases (01179, 01624), where Run C predicts 297 and 23 TC voxels (TC Dice 0).
