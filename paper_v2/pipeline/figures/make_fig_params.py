#!/usr/bin/env python3
"""Figure: BraTS 2021 test mean Dice (95% CI) vs parameter count, all methods, same split/scorer.
Colour = method family (3 validated slots), marker shape = secondary encoding, direct labels.
Re-run after the ensemble / rotation-TTA results exist; missing candidates are skipped."""
import os, sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/mnt/data1/kamil_research/experiments/eval_v2/code")
from brats_gbm.eval.stats import summarise_segmentation
R = "/home/kamilabdelali/brats2021/results/brats"; OUT = os.path.dirname(os.path.abspath(__file__))
FAM = {"nnU-Net": ("#2a78d6", "s"), "Swin UNETR": ("#eb6834", "^"), "Wavelet U-Net++ (ours)": ("#1baf7a", "o"),
       "other baselines": ("#7a7873", "D")}   # Amendment 19: SegResNet, UNETR, 3D U-Net (neutral grey, diamond)
P = {"ep253": 10.40, "runA": 10.40, "runC": 33.96, "runC_rot": 33.96, "runD2": 33.96, "runD2_rot": 33.96}   # parameters (M) per candidate
ROWS = [("nnU-Net", "nnU-Net", "nnunet_test_per_case.csv", 31.2),
        ("Swin UNETR", "Swin UNETR", "swin_unetr_test_per_case.csv", 62.2),
        ("other baselines", "SegResNet", "segresnet_test_per_case.csv", 4.70),
        ("other baselines", "UNETR", "unetr_test_per_case.csv", 102.24),
        ("other baselines", "3D U-Net", "unet3d_test_per_case.csv", 16.32),
        ("Wavelet U-Net++ (ours)", "base (2D DWT)", "per_case_final_ep253_test_tuned_recomputed.csv", P["ep253"]),
        ("Wavelet U-Net++ (ours)", "+ deep supervision", "per_case_runA_test_tuned_recomputed.csv", P["runA"]),
        ("Wavelet U-Net++ (ours)", "+ 3D DWT, F=24", "per_case_runC_test_tuned_recomputed.csv", P["runC"]),
        ("Wavelet U-Net++ (ours)", "+ rotation TTA", "per_case_runC_rot_test_tuned_recomputed.csv", P["runC_rot"]),
        ("Wavelet U-Net++ (ours)", "Run D2 (z-score, strong aug.)", "per_case_runD2_rot_test_tuned_recomputed.csv", P["runD2_rot"])]
sel = "/mnt/data1/kamil_research/experiments/eval_v2/ensemble3/selection.json"   # headline: highest VAL Dice of all candidates
if os.path.exists(sel):
    import json; members = json.load(open(sel))["combo"].split("+")
    weights = sorted({m.replace("_rot", "") for m in members})   # a model and its rotation-TTA variant share weights
    if len(weights) > 1:   # an ensemble of one set of weights is the same model as its members: not plotted separately
        ROWS.append(("Wavelet U-Net++ (ours)", "ensemble (headline)", "per_case_ensemble3_test_tuned_recomputed.csv", sum(P[w] for w in weights)))
pts = []
for fam, lab, f, prm in ROWS:
    if not os.path.exists(f"{R}/{f}"): continue
    d = pd.read_csv(f"{R}/{f}")
    if "Patient_ID" not in d.columns: d["Patient_ID"] = d.iloc[:, -1]
    s = summarise_segmentation(d[["Patient_ID"] + [c for c in d.columns if c.startswith(("Dice_", "HD95_")) and c != "Dice_Mean"]], label=lab)
    m = s[s.region == "MEAN"].iloc[0]; pts.append((fam, lab, prm, m.dice, m.dice_lo, m.dice_hi))
    print(f"{lab:32s} {prm:6.1f} M  {m.dice:.4f} [{m.dice_lo:.4f}, {m.dice_hi:.4f}]")
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(5.2, 3.2), dpi=300)
ax.grid(axis="y", color="#e4e3df", lw=0.5); ax.set_axisbelow(True)
seen = {}; BARS = []; TEXTS = []
# label position (data coords), alignment, and whether a leader line joins label and point
LAB = {"nnU-Net": (31.2, 0.9315, "center", "bottom", False),
       "Swin UNETR": (60.6, 0.9028, "right", "center", False),
       "SegResNet": (1.0, 0.9262, "left", "center", False),
       "UNETR": (102.24, 0.9010, "center", "bottom", False),
       "3D U-Net": (17.6, 0.8620, "left", "center", False),
       "base (2D DWT)": (13.0, 0.9180, "left", "center", True),
       "+ deep supervision": (11.2, 0.8325, "center", "top", False),
       "+ 3D DWT, F=24": (39.5, 0.8545, "left", "center", True),
       "+ rotation TTA": (39.5, 0.8665, "left", "center", True),
       "Run D2 (z-score, strong aug.)": (39.5, 0.8785, "left", "center", True),
       "ensemble (headline)": (78.3, 0.9215, "center", "bottom", False)}
for fam, lab, prm, dc, lo, hi in pts:
    c, mk = FAM[fam]
    n = sum(p[2] == prm for p in pts); k = seen.get(prm, 0); seen[prm] = k + 1; x = prm + (k - (n - 1) / 2) * 1.6   # dodge equal sizes
    BARS.append((x, lo, hi))
    ax.errorbar(x, dc, yerr=[[dc - lo], [hi - dc]], fmt=mk, ms=5, color=c, mec="#fcfcfb", mew=0.8, elinewidth=0.8, capsize=0, zorder=3)
    tx, ty, ha, va, leader = LAB[lab]
    if leader:
        ax.plot([tx - (0.6 if ha == "left" else -0.6), x], [ty, dc], color="#b5b3ad", lw=0.5, zorder=1)
    TEXTS.append(ax.text(tx, ty, lab, ha=ha, va=va, fontsize=6.5, color="#52514e", zorder=4))
ax.set_xlabel("Parameters (M)"); ax.set_ylabel("BraTS 2021 test mean Dice")
ax.set_xlim(0, 112); ax.set_ylim(0.822, 0.936)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], marker=FAM[k][1], color=FAM[k][0], ls="", ms=5, label=k) for k in FAM], loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, frameon=False, fontsize=6.5, handletextpad=0.3, columnspacing=1.0)
fig.tight_layout(); fig.canvas.draw(); rr = fig.canvas.get_renderer()
bb = [t.get_window_extent(rr).expanded(1.02, 1.1) for t in TEXTS]; ab = ax.get_window_extent(rr); bad = 0
for i in range(len(bb)):
    for j in range(i + 1, len(bb)):
        if bb[i].overlaps(bb[j]): bad += 1; print("TEXT OVERLAP:", TEXTS[i].get_text(), "|", TEXTS[j].get_text())
    if bb[i].x0 < ab.x0 - 1 or bb[i].x1 > ab.x1 + 1 or bb[i].y1 > fig.bbox.y1: bad += 1; print("OUTSIDE AXES:", TEXTS[i].get_text())
    for (x, lo, hi) in BARS:
        px0, py0 = ax.transData.transform((x, lo)); px1, py1 = ax.transData.transform((x, hi))
        if bb[i].x0 <= px0 <= bb[i].x1 and not (bb[i].y1 < py0 or bb[i].y0 > py1): bad += 1; print("TEXT ON BAR:", TEXTS[i].get_text(), "bar at", round(x, 1))
print("layout problems:", bad)
fig.savefig(f"{OUT}/fig_acc_params.pdf"); fig.savefig(f"{OUT}/fig_acc_params.png")
print("saved", f"{OUT}/fig_acc_params.pdf")
