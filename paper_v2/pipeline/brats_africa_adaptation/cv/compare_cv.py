#!/usr/bin/env python3
"""Amendment 6: pool the 5 CV folds (n=95 gliomas, each tested once) and compare
adapted vs zero-shot (same method) and ours vs each baseline (both adapted).
Baseline predictions are scored with the external pipeline's scorer (same GT)."""
import sys, os, json, numpy as np, nibabel as nib, pandas as pd
from scipy.stats import wilcoxon
sys.path.insert(0, "/mnt/data1/kamil_research/experiments/external_africa")
from common_ext import score_case, gt_regions, pred_regions, summarise_segmentation
A = "/mnt/data1/kamil_research/experiments/adapt_africa"; X = "/mnt/data1/kamil_research/experiments/external_africa"
cv = json.load(open(f"{A}/cv/cv_split.json")); cases = pd.read_csv(f"{X}/cases.csv").set_index("case")
R = ("Dice_ET", "Dice_TC", "Dice_WT", "Dice_Mean")
def pooled_baseline(name, conv):
    out = f"{A}/cv/per_case_cv_{name}.csv"
    if os.path.exists(out): return pd.read_csv(out).set_index("Patient_ID")
    rows = []
    for k, f in enumerate(cv["folds"]):
        for c in f["test"]:
            p = f"{A}/cv/fold{k}/pred_{name}/{c}.nii.gz"
            if not os.path.exists(p): return None
            s = score_case(pred_regions(np.asarray(nib.load(p).dataobj).astype(np.int16), conv), gt_regions(cases.loc[c, "seg"]))
            s["Patient_ID"] = c; s["fold"] = k; rows.append(s)
    d = pd.DataFrame(rows); d.to_csv(out, index=False); return d.set_index("Patient_ID")
ad = {}
parts = [f"{A}/cv/fold{k}/per_case_ours.csv" for k in range(5)]
if all(os.path.exists(p) for p in parts): ad["ours"] = pd.concat([pd.read_csv(p) for p in parts]).set_index("Patient_ID")
parts = [f"{A}/cv/fold{k}/per_case_ours_sel.csv" for k in range(5)]   # Amendment 8 (val-selected, may be an ensemble)
if all(os.path.exists(p) for p in parts): ad["ours_sel"] = pd.concat([pd.read_csv(p) for p in parts]).set_index("Patient_ID")
parts = [f"{A}/cv/fold{k}/per_case_ours_runC.csv" for k in range(5)]   # Amendment 13 (secondary, from Run C)
if all(os.path.exists(p) for p in parts): ad["ours_runC"] = pd.concat([pd.read_csv(p) for p in parts]).set_index("Patient_ID")
parts = [f"{A}/cv/fold{k}/per_case_ours_runC_sel.csv" for k in range(5)]   # Amendment 13b (may be an ensemble)
if all(os.path.exists(p) for p in parts): ad["ours_runC_sel"] = pd.concat([pd.read_csv(p) for p in parts]).set_index("Patient_ID")
for lab, fn in (("ours_runD", "per_case_ours_runD.csv"), ("ours_runD_sel", "per_case_ours_runD_sel.csv"),
                ("ours_runD2", "per_case_ours_runD2.csv"), ("ours_runD2_sel", "per_case_ours_runD2_sel.csv")):   # Amendments 15/17
    parts = [f"{A}/cv/fold{k}/{fn}" for k in range(5)]
    if all(os.path.exists(p) for p in parts): ad[lab] = pd.concat([pd.read_csv(p) for p in parts]).set_index("Patient_ID")
for n, conv in (("nnunet", "nnunet_regions"), ("swin", "brats21")):
    d = pooled_baseline(n, conv)
    if d is not None: ad["nnunet" if n == "nnunet" else "swin_unetr"] = d
allc = sorted(sum((f["test"] for f in cv["folds"]), []))
zs = {k: pd.read_csv(f"{X}/results/per_case_{v}.csv").set_index("Patient_ID").loc[allc]
      for k, v in (("ours", "ours_runA"), ("nnunet", "nnunet"), ("swin_unetr", "swin_unetr"))}   # ours zero-shot = its start model (Run A)
if os.path.exists(f"{X}/results/per_case_ours_runC.csv"): zs["ours_runC"] = pd.read_csv(f"{X}/results/per_case_ours_runC.csv").set_index("Patient_ID").loc[allc]
if "ours_runC" in zs: zs["ours_runC_sel"] = zs["ours_runC"]
for zt in ("ours_runD_rot", "ours_runD"):   # zero-shot of the Run D variant evaluated externally (VAL-better one)
    if os.path.exists(f"{X}/results/per_case_{zt}.csv"):
        zs["ours_runD"] = zs["ours_runD_sel"] = pd.read_csv(f"{X}/results/per_case_{zt}.csv").set_index("Patient_ID").loc[allc]; break
for zt in ("ours_runD2_rot", "ours_runD2"):   # Amendment 17: zero-shot of the VAL-better Run D2 variant
    if os.path.exists(f"{X}/results/per_case_{zt}.csv"):
        zs["ours_runD2"] = zs["ours_runD2_sel"] = pd.read_csv(f"{X}/results/per_case_{zt}.csv").set_index("Patient_ID").loc[allc]; break
start = " / ".join(l.split()[0] for l in open(f"{A}/cv/ours_start.txt") if l.strip()) if os.path.exists(f"{A}/cv/ours_start.txt") else "?"
print(f"BraTS-Africa glioma, 5-fold CV adaptation, pooled n=95 (ours adapted from: {start}; ours zero-shot = Run A, ours_runC zero-shot = Run C; ours_sel zero-shot shown as Run A)")
rows = []
zs["ours_sel"] = zs["ours"]
for k in ("ours", "ours_sel", "ours_runC", "ours_runC_sel", "ours_runD", "ours_runD_sel", "ours_runD2", "ours_runD2_sel", "nnunet", "swin_unetr"):
    for kind, d in (("zero-shot", zs.get(k) if k not in ("ours_sel", "ours_runC_sel", "ours_runD_sel", "ours_runD2_sel") else None), ("adapted (CV)", ad.get(k))):
        if d is None: continue
        d = d.loc[allc]; m = summarise_segmentation(d.reset_index()[["Patient_ID"] + [c for c in d.columns if c.startswith(("Dice", "HD95"))]], label=k)
        m = m[m.region == "MEAN"].iloc[0]
        rows.append([k, kind] + [d[r].mean() for r in R] + [f"[{m.dice_lo:.3f}, {m.dice_hi:.3f}]"])
print(pd.DataFrame(rows, columns=["model", "setting", "ET", "TC", "WT", "Mean", "Mean 95% CI"]).round(4).to_string(index=False))
rng = np.random.default_rng(0)
def paired(a, b, lab):
    for c in R:
        x = (a.loc[allc, c] - b.loc[allc, c]).values; bs = x[rng.integers(0, len(x), (10000, len(x)))].mean(1)
        lo, hi = np.percentile(bs, [2.5, 97.5]); p = wilcoxon(x).pvalue if np.any(x != 0) else 1.0
        v = "BETTER" if lo > 0 else ("WORSE" if hi < 0 else "no sig. difference")
        print(f"{lab:34s} {c:9s} {x.mean():+.4f} [{lo:+.4f}, {hi:+.4f}] p={p:.3g} wins {int((x>0).sum())}/{len(x)} {v}")
print("\nadapted (CV) minus zero-shot (same start model)"); [paired(ad[k], zs[k], f"{k}") for k in ad if k in zs]
if "ours" in ad:
    print("\nours minus baseline (both adapted, CV)"); [paired(ad[o], ad[b], f"{o} - {b}") for o in ("ours", "ours_sel", "ours_runC", "ours_runC_sel", "ours_runD", "ours_runD_sel", "ours_runD2", "ours_runD2_sel") if o in ad for b in ("nnunet", "swin_unetr") if b in ad]
