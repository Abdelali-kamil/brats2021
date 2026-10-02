#!/usr/bin/env python3
"""PROTOCOL_v2 comparison table: every available candidate of ours vs nnU-Net
and Swin UNETR on the same 125 test cases. Paired bootstrap (10,000, seed 0)
95% CI of the per-case difference + two-sided Wilcoxon, mean Dice and per region.
Headline 'ours' = highest VAL Dice (from eval_v2/work_*/selection.txt; ep253
uses its recorded val 0.8893), never chosen by test."""
import pathlib, numpy as np, pandas as pd
from scipy.stats import wilcoxon
R = pathlib.Path("/home/kamilabdelali/brats2021/results/brats")
V = pathlib.Path("/mnt/data1/kamil_research/experiments/eval_v2")
ours = {"ep253": (R / "per_case_final_ep253_test_tuned_recomputed.csv", 0.8911)}  # ep253 VAL under the Amendment-5 Dice rule
for tag in ("runA", "runC", "runC_rot", "runD", "runD_rot", "runD2", "runD2_rot"):   # Amendments 11/12/15/17
    f, s = R / f"per_case_{tag}_test_tuned_recomputed.csv", V / f"work_{tag}/selection.txt"
    if f.exists() and s.exists():
        ours[tag] = (f, float(s.read_text().split()[1]))
import json
for _e in ("ensemble", "ensemble2", "ensemble3"):   # Amendments 11/15/17
    _ens = V / _e / "selection.json"
    if _ens.exists() and (R / f"per_case_{_e}_test_tuned_recomputed.csv").exists():
        ours[_e] = (R / f"per_case_{_e}_test_tuned_recomputed.csv", json.loads(_ens.read_text())["val_dice"])
base = {"nnU-Net": R / "nnunet_test_per_case.csv", "Swin UNETR": R / "swin_unetr_test_per_case.csv"}
load = lambda f: pd.read_csv(f).set_index("Patient_ID")
head = max(ours, key=lambda k: ours[k][1])
print("candidates (val Dice):", {k: v[1] for k, v in ours.items()}, "-> headline by VAL:", head)
rows = []
for n, (f, _) in ours.items():
    d = load(f); rows.append([n, len(d)] + [d[c].mean() for c in ("Dice_ET", "Dice_TC", "Dice_WT", "Dice_Mean")])
for n, f in base.items():
    d = load(f); rows.append([n, len(d)] + [d[c].mean() for c in ("Dice_ET", "Dice_TC", "Dice_WT", "Dice_Mean")])
print(pd.DataFrame(rows, columns=["model", "n", "ET", "TC", "WT", "Mean"]).round(4).to_string(index=False))
rng = np.random.default_rng(0)
print("\npaired: ours - baseline")
for n in ours:
    o = load(ours[n][0])
    for b, f in base.items():
        d = load(f); ids = o.index.intersection(d.index); assert len(ids) == 125, len(ids)
        for c in ("Dice_Mean", "Dice_ET", "Dice_TC", "Dice_WT"):
            x = (o.loc[ids, c] - d.loc[ids, c]).values
            bs = x[rng.integers(0, len(x), (10000, len(x)))].mean(1)
            lo, hi = np.percentile(bs, [2.5, 97.5]); p = wilcoxon(x).pvalue
            verdict = "ours BETTER" if lo > 0 else ("ours WORSE" if hi < 0 else "no sig. difference")
            print(f"{n:6s} vs {b:10s} {c:9s} {x.mean():+.4f} [{lo:+.4f}, {hi:+.4f}] p={p:.4g}  wins {int((x>0).sum())}/125  {verdict}")
