#!/usr/bin/env python3
"""Amendment 18: the Amendment-14A features from the baselines' zero-shot BraTS-Africa masks.
  extract_baselines.py nnunet | swin   ->  features_<src>.csv"""
import sys, numpy as np, nibabel as nib, pandas as pd
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, "/home/kamilabdelali/brats2021")
from brats_gbm.features import all_region_features, znorm
X = "/mnt/data1/kamil_research/experiments/external_africa"; D = "/mnt/data1/kamil_research/experiments/classification_africa"
SRC = sys.argv[1]; PRED = {"nnunet": "pred_nnunet", "swin": "pred_swin", "unet3d": "pred_unet3d", "segresnet": "pred_segresnet", "unetr": "pred_unetr"}[SRC]   # Amendment 19 adds the last three
cases = pd.read_csv(f"{X}/cases.csv")
def regions(lab):
    if SRC == "nnunet":            # nnU-Net region labels: WT={1,2,3}, TC={2,3}, ET={3}
        return lab == 3, (lab == 2) | (lab == 3), lab >= 1
    et = lab == 4; tc = et | (lab == 1); return et, tc, tc | (lab == 2)   # Swin: BraTS 2021 labels
def one(i):
    r = cases.iloc[i]
    vols = {k: nib.load(p).get_fdata().astype(np.float32) for k, p in (("t1", r.t1n), ("t1ce", r.t1c), ("t2", r.t2w), ("flair", r.t2f))}
    img = nib.load(f"{X}/{PRED}/{r.case}.nii.gz"); lab = np.asarray(img.dataobj).astype(np.int16)
    assert lab.shape == vols["t1"].shape, (r.case, lab.shape, vols["t1"].shape)
    et, tc, wt = regions(lab)
    g = np.asarray(nib.load(r.seg).dataobj).astype(np.int16) > 0      # alignment check: WT Dice vs expert
    f = {"case": r.case, "y_glioma": int(r.group == "glioma"), "mask_source": SRC,
         "_wt_dice_check": float(2 * (wt & g).sum() / max(wt.sum() + g.sum(), 1))}
    f.update(all_region_features({"ET": et, "TC": tc, "WT": wt}, znorm(vols), float(abs(np.linalg.det(img.affine[:3, :3])))))
    return f
if __name__ == "__main__":
    with ProcessPoolExecutor(8) as ex: rows = list(ex.map(one, range(len(cases))))
    df = pd.DataFrame(rows)
    ref = pd.read_csv(f"{X}/results/per_case_{ {'nnunet': 'nnunet', 'swin': 'swin_unetr'}.get(SRC, SRC) }.csv").set_index("Patient_ID").Dice_WT
    chk = df.set_index("case")._wt_dice_check
    print(SRC, "alignment: mean |WT Dice here - WT Dice in scoring| =", round(float((chk - ref.loc[chk.index]).abs().mean()), 4))
    df.drop(columns=["_wt_dice_check"]).to_csv(f"{D}/features_{SRC}.csv", index=False)
    print(SRC, df.shape, "glioma", int(df.y_glioma.sum()))
