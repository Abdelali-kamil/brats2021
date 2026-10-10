#!/usr/bin/env python3
"""Amendment 21: Run C (3D Haar DWT) vs the non-wavelet control (3D max-pool + 1x1x1 projection), otherwise identical.
Primary: Run C minus control, mean Dice on the 125 BraTS 2021 test cases, paired bootstrap (10,000, fresh seed-0 generator
per comparison) 95% CI + two-sided Wilcoxon. Interpretation fixed in the amendment: the wavelet is said to improve Dice only
if the CI excludes zero in Run C's favour; a CI including zero = "no difference detected" (equivalence not tested).
Secondary: per-region Dice, HD95 (per case mean over defined regions; pairs defined for both) and HD95 with undefined values
set to 373.13 mm, BraTS-Africa zero-shot (95 gliomas; 51 other neoplasms separately)."""
import sys, numpy as np, pandas as pd
from scipy.stats import wilcoxon
sys.path.insert(0, "/mnt/data1/kamil_research/experiments/eval_v2/code")
from brats_gbm.eval.stats import bootstrap_ci
RES = "/home/kamilabdelali/brats2021/results/brats"; X = "/mnt/data1/kamil_research/experiments/external_africa/results"
V = "/mnt/data1/kamil_research/experiments/eval_v2"
P = float(np.sqrt(240**2 + 240**2 + 155**2))
REG = ("ET", "TC", "WT")
L = lambda p: pd.read_csv(p).set_index("Patient_ID")
md = lambda d: d[[f"Dice_{r}" for r in REG]].mean(axis=1)
hd = lambda d: d[[f"HD95_{r}" for r in REG]].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
hp = lambda d: d[[f"HD95_{r}" for r in REG]].apply(pd.to_numeric, errors="coerce").fillna(P).mean(axis=1)

def summary(name, d):
    m, lo, hi = bootstrap_ci(md(d))
    print(f"  {name:28s} n={len(d)}  ET {d.Dice_ET.mean():.4f}  TC {d.Dice_TC.mean():.4f}  WT {d.Dice_WT.mean():.4f}  "
          f"mean {m:.4f} [{lo:.4f}, {hi:.4f}]  HD95 {hd(d).mean():.2f} (n={int(hd(d).notna().sum())})  HD95* {hp(d).mean():.2f}")

def paired(label, a, b, lower_better=False):
    ids = a.index.intersection(b.index); x = (a.loc[ids] - b.loc[ids]).dropna().values
    bs = x[np.random.default_rng(0).integers(0, len(x), (10000, len(x)))].mean(1); lo, hi = np.percentile(bs, [2.5, 97.5])
    p = wilcoxon(x).pvalue if np.any(x != 0) else 1.0
    good, bad = (hi < 0, lo > 0) if lower_better else (lo > 0, hi < 0)
    verdict = "favours Run C" if good else ("favours the control" if bad else "no difference detected")
    print(f"  {label:34s} n={len(x):3d} {x.mean():+.6f} [{lo:+.6f}, {hi:+.6f}]  Wilcoxon p={p:.3g}  "
          f"Run C better on {int((x < 0).sum() if lower_better else (x > 0).sum())}/{len(x)}  -> {verdict}")

for tag in ("runC", "ctrl_maxpool3d"):
    try: print(f"{tag}: VAL selection: {open(f'{V}/work_{tag}/selection.txt').read().strip()}")
    except OSError: print(f"{tag}: no selection file")
c, k = L(f"{RES}/per_case_runC_test_tuned_recomputed.csv"), L(f"{RES}/per_case_ctrl_maxpool3d_test_tuned_recomputed.csv")
assert len(c) == len(k) == 125 and set(c.index) == set(k.index)
print("\n== BraTS 2021 TEST (125, each scored once)")
summary("Run C (3D Haar DWT)", c); summary("control (3D max-pool)", k)
print("\n  Run C minus control (positive Dice / negative HD95 = Run C better)")
paired("Dice_Mean  [PRIMARY]", md(c), md(k))
for r in REG: paired(f"Dice_{r}", c[f"Dice_{r}"], k[f"Dice_{r}"])
paired("HD95 (defined for both, mm)", hd(c), hd(k), lower_better=True)
paired("HD95* (undefined = 373.13 mm)", hp(c), hp(k), lower_better=True)
try:
    zc, zk = L(f"{X}/per_case_ours_runC.csv"), L(f"{X}/per_case_ours_ctrl_maxpool3d.csv")
    for g in ("glioma", "other_neoplasm"):
        a, b = zc[zc.group == g], zk[zk.group == g]
        print(f"\n== BraTS-Africa zero-shot, {g} ({len(a)})" + ("  [secondary]" if g == "glioma" else "  [stress test]"))
        summary("Run C (3D Haar DWT)", a); summary("control (3D max-pool)", b)
        paired("Dice_Mean", md(a), md(b))
        for r in REG: paired(f"Dice_{r}", a[f"Dice_{r}"], b[f"Dice_{r}"])
except FileNotFoundError as e:
    print("\nBraTS-Africa results not available yet:", e)
