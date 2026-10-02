#!/usr/bin/env python3
"""PROTOCOL_v2 gate: patient overlap with BraTS 2021, by the UPenn method
(best WT-mask Dice over all 1251 BraTS 2021 cases at 2x downsampling, then FLAIR
intensity correlation on that match). Flag = corr > 0.8 or WT Dice > 0.9."""
import glob, os, numpy as np, nibabel as nib, pandas as pd
from concurrent.futures import ProcessPoolExecutor
X = os.path.dirname(os.path.abspath(__file__)); B = "/mnt/data1/kamil_research/data/brats2021"
def wt(p): return (np.asarray(nib.load(p).dataobj)[::2, ::2, ::2] > 0).ravel()
cases = pd.read_csv(f"{X}/cases.csv")
bp = sorted(glob.glob(f"{B}/BraTS2021_*/BraTS2021_*_seg.nii.gz"))
with ProcessPoolExecutor(16) as ex:
    Bv = np.stack(list(ex.map(wt, bp, chunksize=20))).astype(np.float32)
    Av = np.stack(list(ex.map(wt, cases.seg, chunksize=5))).astype(np.float32)
d = 2 * (Av @ Bv.T) / (Av.sum(1)[:, None] + Bv.sum(1)[None, :] + 1e-9)
j = d.argmax(1)
def corr(a):
    i, k = a; x = np.asarray(nib.load(cases.t2f[i]).dataobj, float)
    b = bp[k].replace("_seg.nii.gz", "_flair.nii.gz"); y = np.asarray(nib.load(b).dataobj, float)
    m = (x > 0) & (y > 0); return float(np.corrcoef(x[m], y[m])[0, 1]) if m.sum() > 100 else 0.0
with ProcessPoolExecutor(16) as ex: cr = list(ex.map(corr, zip(range(len(cases)), j)))
cases["best_brats21"] = [os.path.basename(bp[k])[:15] for k in j]; cases["wt_dice"] = d.max(1); cases["flair_corr"] = cr
cases["overlap_flag"] = (cases.flair_corr > 0.8) | (cases.wt_dice > 0.9)
cases.to_csv(f"{X}/cases_gated.csv", index=False)
print(cases[["wt_dice", "flair_corr"]].describe().round(3).to_string())
print("FLAGGED:", int(cases.overlap_flag.sum())); print(cases[cases.overlap_flag][["case", "group", "best_brats21", "wt_dice", "flair_corr"]].to_string(index=False))
