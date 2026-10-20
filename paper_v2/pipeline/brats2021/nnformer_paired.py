#!/usr/bin/env python3
"""Amendment 26: Ensemble 3 (headline) minus nnFormer on the same 125 BraTS 2021 test cases, from the saved per-case
scores (no inference). Same paired statistics as the other baselines: paired bootstrap 95% CI of the per-case difference
(10,000 resamples, a fresh seed-0 generator per comparison, as extra_paired_stats.py) + two-sided Wilcoxon; mean Dice and
per region. Written 2026-10-17 after nnFormer's TEST was scored (user request)."""
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
R = "/home/kamilabdelali/brats2021/results/brats"
L = lambda p: pd.read_csv(p).set_index("Patient_ID")
ens3 = L(f"{R}/per_case_ensemble3_test_tuned_recomputed.csv")
for name, f in (("nnFormer", "nnformer_test_per_case.csv"), ("SegResNet (check vs compare.py)", "segresnet_test_per_case.csv")):
    b = L(f"{R}/{f}"); ids = ens3.index.intersection(b.index); assert len(ids) == 125 == len(b) == len(ens3), len(ids)
    print(f"== Ensemble 3 - {name}")
    for c in ("Dice_Mean", "Dice_ET", "Dice_TC", "Dice_WT"):
        x = (ens3.loc[ids, c] - b.loc[ids, c]).values
        bs = x[np.random.default_rng(0).integers(0, len(x), (10000, len(x)))].mean(1); lo, hi = np.percentile(bs, [2.5, 97.5])
        verdict = "Ens3 BETTER" if lo > 0 else ("Ens3 WORSE" if hi < 0 else "no sig. difference")
        print(f"  {c:9s} Ens3 {ens3.loc[ids, c].mean():.4f}  {name.split()[0]} {b.loc[ids, c].mean():.4f}  diff {x.mean():+.4f} "
              f"[{lo:+.4f}, {hi:+.4f}]  Wilcoxon p={wilcoxon(x).pvalue:.4g}  Ens3 higher {int((x > 0).sum())}/125  {verdict}")
