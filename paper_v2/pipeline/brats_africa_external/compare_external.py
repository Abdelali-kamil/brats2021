#!/usr/bin/env python3
"""BraTS-Africa: per-group table + paired bootstrap (10,000, seed 0) and Wilcoxon,
ours vs each baseline, same as the BraTS-test comparison. Primary = glioma."""
import glob, os, numpy as np, pandas as pd
from scipy.stats import wilcoxon
X = "/mnt/data1/kamil_research/experiments/external_africa/results"
load = lambda t: pd.read_csv(f"{X}/per_case_{t}.csv").set_index("Patient_ID")
tags = sorted(os.path.basename(f)[9:-4] for f in glob.glob(f"{X}/per_case_*.csv"))
ours = [t for t in tags if t.startswith("ours_")]; base = [t for t in ("nnunet", "swin_unetr") if t in tags]
rng = np.random.default_rng(0)
for g in ("glioma", "other_neoplasm"):
    print(f"\n===== {g} {'(PRIMARY)' if g == 'glioma' else '(secondary, out-of-distribution)'} =====")
    tab = []
    for t in ours + base:
        d = load(t); d = d[d.group == g]
        tab.append([t, len(d)] + [d[c].mean() for c in ("Dice_ET", "Dice_TC", "Dice_WT", "Dice_Mean")] + [d[["HD95_ET", "HD95_TC", "HD95_WT"]].mean().mean()])
    print(pd.DataFrame(tab, columns=["model", "n", "ET", "TC", "WT", "Mean", "HD95"]).round(4).to_string(index=False))
    for o in ours:
        a = load(o); a = a[a.group == g]
        for b in base:
            d = load(b); ids = a.index.intersection(d[d.group == g].index)
            for c in ("Dice_Mean", "Dice_ET", "Dice_TC", "Dice_WT"):
                x = (a.loc[ids, c] - d.loc[ids, c]).values
                bs = x[rng.integers(0, len(x), (10000, len(x)))].mean(1); lo, hi = np.percentile(bs, [2.5, 97.5])
                p = wilcoxon(x).pvalue if np.any(x != 0) else 1.0
                v = "ours BETTER" if lo > 0 else ("ours WORSE" if hi < 0 else "no sig. difference")
                print(f"{o:12s} vs {b:10s} {c:9s} n={len(ids)} {x.mean():+.4f} [{lo:+.4f}, {hi:+.4f}] p={p:.3g} wins {int((x>0).sum())}/{len(x)} {v}")
