#!/usr/bin/env python3
"""Amendment 14A: region features for the 146 BraTS-Africa subjects from a given mask source.
  extract.py expert            -> features_expert.csv   (expert masks, BraTS 2023 labels: ET=3)
  extract.py ours_runC         -> features_ours_runC.csv (Run C zero-shot predicted masks, ET=4)"""
import sys, numpy as np, nibabel as nib, pandas as pd
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, "/home/kamilabdelali/brats2021")
from brats_gbm.features import all_region_features, znorm
X = "/mnt/data1/kamil_research/experiments/external_africa"; D = "/mnt/data1/kamil_research/experiments/classification_africa"
SRC = sys.argv[1]
cases = pd.read_csv(f"{X}/cases.csv")
def one(i):
    r = cases.iloc[i]
    vols = {k: nib.load(p).get_fdata().astype(np.float32) for k, p in (("t1", r.t1n), ("t1ce", r.t1c), ("t2", r.t2w), ("flair", r.t2f))}
    if SRC == "expert":
        img = nib.load(r.seg); s = np.asarray(img.dataobj).astype(np.int16); et = s == 3; ncr, ed = s == 1, s == 2
    else:
        img = nib.load(f"{X}/results/masks_{SRC}/{r.case}.nii.gz"); s = np.asarray(img.dataobj).astype(np.int16); et = s == 4; ncr, ed = s == 1, s == 2
    tc = et | ncr; wt = tc | ed
    f = {"case": r.case, "y_glioma": int(r.group == "glioma"), "mask_source": SRC}
    f.update(all_region_features({"ET": et, "TC": tc, "WT": wt}, znorm(vols), float(abs(np.linalg.det(img.affine[:3, :3])))))
    return f
if __name__ == "__main__":
    with ProcessPoolExecutor(8) as ex: rows = list(ex.map(one, range(len(cases))))
    df = pd.DataFrame(rows); df.to_csv(f"{D}/features_{SRC}.csv", index=False)
    print(SRC, df.shape, "glioma", int(df.y_glioma.sum()), "| all-NaN feature columns:", int(df.isna().all().sum()))
