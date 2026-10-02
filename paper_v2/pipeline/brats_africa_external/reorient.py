#!/usr/bin/env python3
"""PROTOCOL_v2 Amendment 3: reorient every BraTS-Africa volume and label map from
its own header to the BraTS 2021 array layout (LPS), by axis flips only (no
resampling). Writes staged/, repoints cases.csv and nnunet_in/, then verifies."""
import os, glob, shutil, numpy as np, nibabel as nib, pandas as pd
from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform, apply_orientation
X = os.path.dirname(os.path.abspath(__file__)); ST = f"{X}/staged"
REF = nib.load("/mnt/data1/kamil_research/data/brats2021/BraTS2021_00000/BraTS2021_00000_flair.nii.gz")
TGT = axcodes2ornt(nib.aff2axcodes(REF.affine))
d = pd.read_csv(f"{X}/cases.csv"); keys = ("t1n", "t1c", "t2w", "t2f", "seg")
for k in keys: d[f"{k}_orig"] = d[k]
for i, r in d.iterrows():
    os.makedirs(f"{ST}/{r.case}", exist_ok=True)
    for k in keys:
        img = nib.load(r[f"{k}_orig"]); t = ornt_transform(io_orientation(img.affine), TGT)
        arr = apply_orientation(np.asanyarray(img.dataobj), t)
        assert arr.shape == REF.shape, (r.case, k, arr.shape)
        out = f"{ST}/{r.case}/{r.case}-{k}.nii.gz"
        h = img.header.copy(); h.set_data_dtype(img.get_data_dtype())
        nib.save(nib.Nifti1Image(arr, REF.affine, h), out); d.at[i, k] = out
    d.at[i, "orig_axcodes"] = "".join(nib.aff2axcodes(nib.load(r["t2f_orig"]).affine))
d.to_csv(f"{X}/cases.csv", index=False)
nn = f"{X}/nnunet_in"; shutil.rmtree(nn, ignore_errors=True); os.makedirs(nn)
for r in d.itertuples():
    for j, k in enumerate(("t1n", "t1c", "t2w", "t2f")): os.symlink(getattr(r, k), f"{nn}/{r.case}_{j:04d}.nii.gz")
# verification: brain-mask Dice with the BraTS 2021 mean mask, before vs after
bm = np.mean([np.asarray(nib.load(f).dataobj) > 0 for f in
              sorted(glob.glob("/mnt/data1/kamil_research/data/brats2021/BraTS2021_0000*/BraTS2021_*_flair.nii.gz"))[:10]], 0) > 0.5
dc = lambda m: 2 * (m & bm).sum() / (m.sum() + bm.sum())
v = pd.DataFrame([dict(case=r.case, ax=r.orig_axcodes,
                       before=dc(np.asarray(nib.load(r.t2f_orig).dataobj) > 0), after=dc(np.asarray(nib.load(r.t2f).dataobj) > 0))
                  for r in d.itertuples()])
v.to_csv(f"{X}/reorient_check.csv", index=False)
print(v.groupby("ax")[["before", "after"]].describe().round(3).T.to_string())
print("cases where alignment got WORSE:", int((v.after < v.before - 1e-6).sum()))
print("nnU-Net inputs restaged:", len(os.listdir(nn)) // 4, "cases")
