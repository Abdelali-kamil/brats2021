#!/usr/bin/env python3
"""Qualitative BraTS 2021 test figure. Cases are chosen by a fixed rule (no cherry-picking): the test
cases nearest the 25th, 50th and 75th percentile of the HEADLINE model's per-case mean Dice. Axial slice
with the largest ground-truth whole-tumour area. Columns: T1ce, ground truth, ours, nnU-Net, Swin UNETR.
Regions: ED (WT minus TC), NCR (TC minus ET), ET - three validated categorical slots."""
import os, sys, json, numpy as np, pandas as pd, nibabel as nib, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt; from matplotlib.patches import Patch
E = "/mnt/data1/kamil_research/experiments"; sys.path.insert(0, f"{E}/eval_v2/code")
from brats_gbm.eval import postprocess as pp
R = "/home/kamilabdelali/brats2021/results/brats"; DATA = "/mnt/data1/kamil_research/data/brats2021"; OUT = os.path.dirname(os.path.abspath(__file__))
# headline = best VAL candidate among those evaluated
cand = {}
for t in ("runC", "runC_rot"):
    f = f"{E}/eval_v2/work_{t}/selection.txt"
    if os.path.exists(f) and os.path.exists(f"{R}/per_case_{t}_test_tuned_recomputed.csv"):
        x = open(f).read().split(); cand[t] = dict(val=float(x[1]), members=[t], wt=x[2], mult=int(x[3]), et=int(x[5]))
f = f"{E}/eval_v2/ensemble/selection.json"
if os.path.exists(f) and os.path.exists(f"{R}/per_case_ensemble_test_tuned_recomputed.csv"):
    s = json.load(open(f)); cand["ensemble"] = dict(val=s["val_dice"], members=s["combo"].split("+"), wt=s["wt_policy"], mult=s["floor_mult"], et=s["et_min_volume"])
H = max(cand, key=lambda k: cand[k]["val"]); h = cand[H]; print("headline:", H, h)
PROB = {"ep253": f"{E}/eval_ep253/probs_test"} | {t: f"{E}/eval_v2/work_{t}/probs_test" for t in ("runA", "runC", "runC_rot")}
per = pd.read_csv(f"{R}/per_case_{H}_test_tuned_recomputed.csv").set_index("Patient_ID")
nn = pd.read_csv(f"{R}/nnunet_test_per_case.csv"); nn = nn.set_index(nn.columns[-1] if "Patient_ID" not in nn else "Patient_ID")
sw = pd.read_csv(f"{R}/swin_unetr_test_per_case.csv"); sw = sw.set_index(sw.columns[-1] if "Patient_ID" not in sw else "Patient_ID")
dm = per[["Dice_ET", "Dice_TC", "Dice_WT"]].mean(1)
cases = [(q, (dm - dm.quantile(q)).abs().idxmin()) for q in (0.25, 0.50, 0.75)]
print(cases)
ORIG = dict(pp.MIN_VOXELS)
def ours(c):
    p = sum(np.load(f"{PROB[m]}/{c}.npz")["prob"].astype(np.float32) for m in h["members"]) / len(h["members"])
    pp.MIN_VOXELS = {k: v * h["mult"] for k, v in ORIG.items()}
    r = np.asarray(pp.postprocess(p, .5, .5, .5, wt_policy=h["wt"], et_policy="none" if h["et"] == 0 else "min_volume", et_min_volume=max(h["et"], 1))).astype(bool)
    pp.MIN_VOXELS = dict(ORIG); return r
def from_b21(l): return np.stack([l == 4, (l == 4) | (l == 1), l > 0])
def from_nn(l): return np.stack([l == 3, (l == 2) | (l == 3), l >= 1])
COL = {"ED": "#2a78d6", "NCR": "#eb6834", "ET": "#1baf7a"}
def overlay(ax, img, reg, title):
    ax.imshow(img.T, cmap="gray", origin="lower"); et, tc, wt = reg
    rgba = np.zeros(img.T.shape + (4,))
    for m, k in ((wt & ~tc, "ED"), (tc & ~et, "NCR"), (et, "ET")):
        c = matplotlib.colors.to_rgb(COL[k]); rgba[m.T] = (*c, 0.55)
    ax.imshow(rgba, origin="lower"); ax.set_title(title, fontsize=6.5, color="#0b0b0b", pad=2); ax.axis("off")
fig, axs = plt.subplots(3, 5, figsize=(7.0, 5.0), dpi=300)
for i, (q, c) in enumerate(cases):
    gtl = np.asarray(nib.load(f"{DATA}/{c}/{c}_seg.nii.gz").dataobj).astype(np.int16); gt = from_b21(gtl)
    z = int(gt[2].sum((0, 1)).argmax()); img = np.asarray(nib.load(f"{DATA}/{c}/{c}_t1ce.nii.gz").dataobj, np.float32)[:, :, z]
    img = np.clip(img / np.percentile(img[img > 0], 99.5), 0, 1)
    xs, ys = np.where(img > 0); x0, x1, y0, y1 = max(xs.min() - 4, 0), xs.max() + 5, max(ys.min() - 4, 0), ys.max() + 5   # crop to brain
    cr = lambda a: a[..., x0:x1, y0:y1]; img = cr(img)
    nnl = np.asarray(nib.load(f"{E}/../baselines/nnunet/eval/test125_pred/{c}.nii.gz").dataobj).astype(np.int16)
    swl = np.asarray(nib.load(f"{E}/../baselines/swin_unetr/eval/test125_pred_best_ep199/{c}.nii.gz").dataobj).astype(np.int16)
    md = lambda d: d.loc[c, ["Dice_ET", "Dice_TC", "Dice_WT"]].mean()
    axs[i, 0].imshow(img.T, cmap="gray", origin="lower"); axs[i, 0].axis("off")
    axs[i, 0].set_title(f"{c.replace('BraTS2021_', 'case ')} (P{int(q*100)})", fontsize=6.5, pad=2)
    overlay(axs[i, 1], img, cr(gt[:, :, :, z]), "ground truth")
    overlay(axs[i, 2], img, cr(ours(c)[:, :, :, z]), f"ours  {md(per):.3f}")
    overlay(axs[i, 3], img, cr(from_nn(nnl)[:, :, :, z]), f"nnU-Net  {md(nn):.3f}")
    overlay(axs[i, 4], img, cr(from_b21(swl)[:, :, :, z]), f"Swin UNETR  {md(sw):.3f}")
fig.legend(handles=[Patch(color=COL[k], alpha=0.8, label=l) for k, l in (("ED", "oedema (ED)"), ("NCR", "necrotic core (NCR)"), ("ET", "enhancing tumour (ET)"))],
           loc="lower center", ncol=3, frameon=False, fontsize=6.5)
fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=1.2, w_pad=0.2); fig.savefig(f"{OUT}/fig_qualitative.pdf"); fig.savefig(f"{OUT}/fig_qualitative.png")
json.dump(dict(headline=H, cases=cases), open(f"{OUT}/fig_qualitative_cases.json", "w")); print("saved")
