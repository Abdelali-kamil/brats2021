#!/usr/bin/env python3
"""Paired statistics quoted in the paper text that are not printed by compare.py / compare_cv.py /
compare_external.py: HD95 comparisons on BraTS 2021, the 3D DWT vs the base model, Run D vs Run C on
BraTS-Africa zero-shot, per-case medians, and the difference in adaptation gain between methods.
Paired bootstrap of the per-case difference (10,000 resamples, a fresh seed-0 generator per comparison)
+ two-sided Wilcoxon. Mean HD95 per case = mean over the regions where it is defined (as in
summarise_segmentation); cases where it is undefined for either method are dropped from that pair."""
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
R = "/home/kamilabdelali/brats2021/results/brats"
X = "/mnt/data1/kamil_research/experiments/external_africa/results"
CV = "/home/kamilabdelali/release/brats2021/paper_v2/results/brats_africa_cv"
L = lambda p: pd.read_csv(p).set_index("Patient_ID")
hd = lambda d: d[["HD95_ET", "HD95_TC", "HD95_WT"]].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)

def paired(name, a, b, unit=""):
    ids = a.index.intersection(b.index); x = (a.loc[ids] - b.loc[ids]).dropna().values
    bs = x[np.random.default_rng(0).integers(0, len(x), (10000, len(x)))].mean(1); lo, hi = np.percentile(bs, [2.5, 97.5])
    p = wilcoxon(x).pvalue if np.any(x != 0) else 1.0
    print(f"{name:52s} n={len(x):3d} mean {x.mean():+.6f} [{lo:+.6f}, {hi:+.6f}]{unit}  median {np.median(x):+.6f}  "
          f"first higher on {int((x > 0).sum())}/{len(x)}  Wilcoxon p={p:.3g}")

t = {k: L(f"{R}/{f}") for k, f in {"base": "per_case_final_ep253_test_tuned_recomputed.csv", "runA": "per_case_runA_test_tuned_recomputed.csv",
     "runC": "per_case_runC_test_tuned_recomputed.csv", "runC_rot": "per_case_runC_rot_test_tuned_recomputed.csv",
     "ens3": "per_case_ensemble3_test_tuned_recomputed.csv", "nnU-Net": "nnunet_test_per_case.csv", "Swin UNETR": "swin_unetr_test_per_case.csv"}.items()}
print("== BraTS 2021 test (125)")
paired("Run A - base, Dice_Mean", t["runA"].Dice_Mean, t["base"].Dice_Mean)
paired("Run C - base, Dice_Mean", t["runC"].Dice_Mean, t["base"].Dice_Mean)
paired("Run C - base, HD95 (mm)", hd(t["runC"]), hd(t["base"]))
paired("Run C rot - Swin UNETR, Dice_Mean (median)", t["runC_rot"].Dice_Mean, t["Swin UNETR"].Dice_Mean)
for o in ("runC", "runC_rot", "ens3"):
    for b in ("Swin UNETR", "nnU-Net"):
        paired(f"{o} - {b}, HD95 (mm)", hd(t[o]), hd(t[b]))

print("== BraTS-Africa zero-shot, 95 gliomas")
z = {k: L(f"{X}/per_case_{k}.csv") for k in ("ours_runC_rot", "ours_runD_rot", "ours_runA", "nnunet", "swin_unetr")}
z = {k: d[d.group == "glioma"] for k, d in z.items()}
paired("Run D rot - Run C rot, Dice_Mean", z["ours_runD_rot"].Dice_Mean, z["ours_runC_rot"].Dice_Mean)

print("== BraTS-Africa CV, adaptation gain (adapted - own zero-shot), difference between methods")
ad = {"ours (Run A)": L(f"{CV}/per_case_cv_ours.csv"), "nnU-Net": L(f"{CV}/per_case_cv_nnunet.csv"), "Swin UNETR": L(f"{CV}/per_case_cv_swin.csv")}
zs = {"ours (Run A)": z["ours_runA"], "nnU-Net": z["nnunet"], "Swin UNETR": z["swin_unetr"]}
ids = ad["nnU-Net"].index
gain = {k: ad[k].loc[ids].Dice_Mean - zs[k].loc[ids].Dice_Mean for k in ad}
for b in ("nnU-Net", "Swin UNETR"):
    paired(f"gain ours (Run A) - gain {b}", gain["ours (Run A)"], gain[b])
