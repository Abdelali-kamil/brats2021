#!/usr/bin/env python3
"""Figure: BraTS 2021 test mean Dice (95% CI) vs parameter count, all methods, same split/scorer.
Colour = method family (3 validated slots), marker shape = secondary encoding, direct labels.
Re-run after the ensemble / rotation-TTA results exist; missing candidates are skipped."""
import os, sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/mnt/data1/kamil_research/experiments/eval_v2/code")
from brats_gbm.eval.stats import summarise_segmentation
R = "/home/kamilabdelali/brats2021/results/brats"; OUT = os.path.dirname(os.path.abspath(__file__))
FAM = {"nnU-Net": ("#2a78d6", "s"), "Swin UNETR": ("#eb6834", "^"), "Wavelet U-Net++ (ours)": ("#1baf7a", "o")}
P = {"ep253": 10.40, "runA": 10.40, "runC": 33.96, "runC_rot": 33.96}   # parameters (M) per candidate
ROWS = [("nnU-Net", "nnU-Net", "nnunet_test_per_case.csv", 31.2),
        ("Swin UNETR", "Swin UNETR", "swin_unetr_test_per_case.csv", 62.2),
        ("Wavelet U-Net++ (ours)", "base (2D DWT)", "per_case_final_ep253_test_tuned_recomputed.csv", P["ep253"]),
        ("Wavelet U-Net++ (ours)", "+ deep supervision", "per_case_runA_test_tuned_recomputed.csv", P["runA"]),
        ("Wavelet U-Net++ (ours)", "+ 3D DWT, F=24", "per_case_runC_test_tuned_recomputed.csv", P["runC"]),
        ("Wavelet U-Net++ (ours)", "+ rotation TTA", "per_case_runC_rot_test_tuned_recomputed.csv", P["runC_rot"])]
sel = "/mnt/data1/kamil_research/experiments/eval_v2/ensemble/selection.json"
if os.path.exists(sel):
    import json; members = json.load(open(sel))["combo"].split("+")
    weights = sorted({m.replace("_rot", "") for m in members})   # a model and its rotation-TTA variant share weights
    if len(weights) > 1:   # an ensemble of one set of weights is the same model as its members: not plotted separately
        ROWS.append(("Wavelet U-Net++ (ours)", "ensemble (" + "+".join(members) + ")", "per_case_ensemble_test_tuned_recomputed.csv", sum(P[w] for w in weights)))
pts = []
for fam, lab, f, prm in ROWS:
    if not os.path.exists(f"{R}/{f}"): continue
    d = pd.read_csv(f"{R}/{f}")
    if "Patient_ID" not in d.columns: d["Patient_ID"] = d.iloc[:, -1]
    s = summarise_segmentation(d[["Patient_ID"] + [c for c in d.columns if c.startswith(("Dice_", "HD95_")) and c != "Dice_Mean"]], label=lab)
    m = s[s.region == "MEAN"].iloc[0]; pts.append((fam, lab, prm, m.dice, m.dice_lo, m.dice_hi))
    print(f"{lab:32s} {prm:6.1f} M  {m.dice:.4f} [{m.dice_lo:.4f}, {m.dice_hi:.4f}]")
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(3.5, 2.9), dpi=300)
ax.grid(axis="y", color="#e4e3df", lw=0.5); ax.set_axisbelow(True)
seen = {}; BARS = []
for fam, lab, prm, dc, lo, hi in pts:
    c, mk = FAM[fam]
    k = seen.get(prm, 0); seen[prm] = k + 1; x = prm + (k - 0.5) * 1.6 if sum(p[2] == prm for p in pts) > 1 else prm   # dodge equal sizes
    BARS.append((x, lo, hi))
    ax.errorbar(x, dc, yerr=[[dc - lo], [hi - dc]], fmt=mk, ms=5, color=c, mec="#fcfcfb", mew=0.8, elinewidth=0.8, capsize=0, zorder=3)
    above = lab.startswith(("base", "+ rotation")) or lab == "nnU-Net"   # above its own CI bar: clears neighbouring bars
    below = lab.startswith(("+ 3D DWT", "+ deep supervision"))           # below its CI bar: clears neighbours
    if above: ax.annotate(lab, (x, hi), xytext=(-3 if lab.startswith(("base", "+ rotation")) else 0, 3), textcoords="offset points",
                          ha="left" if lab.startswith(("base", "+ rotation")) else "center", va="bottom", fontsize=6.5, color="#52514e")
    elif below: ax.annotate(lab, (x, lo), xytext=(-3, -3), textcoords="offset points", ha="left", va="top", fontsize=6.5, color="#52514e")
    else: ax.annotate(lab, (x, dc), xytext=(5, 0), textcoords="offset points", ha="left", va="center", fontsize=6.5, color="#52514e")
ax.set_xlabel("Parameters (M)"); ax.set_ylabel("BraTS 2021 test mean Dice")
ax.set_xlim(0, 75)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], marker=FAM[k][1], color=FAM[k][0], ls="", ms=5, label=k) for k in FAM], loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False, fontsize=6.5, handletextpad=0.3, columnspacing=1.0)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_acc_params.pdf"); fig.savefig(f"{OUT}/fig_acc_params.png")
print("saved", f"{OUT}/fig_acc_params.pdf")
