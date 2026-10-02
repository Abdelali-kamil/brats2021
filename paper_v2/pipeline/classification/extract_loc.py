#!/usr/bin/env python3
"""Amendment 16: 8 tumour-location features appended to the 54 Amendment-14A features.
  extract_loc.py <src>  ->  features_<src>_loc.csv   (src: expert | ours_runC | ours_runD_rot)"""
import sys, numpy as np, nibabel as nib, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from scipy import ndimage as ndi
X = "/mnt/data1/kamil_research/experiments/external_africa"; D = "/mnt/data1/kamil_research/experiments/classification_africa"
SRC = sys.argv[1]
cases = pd.read_csv(f"{X}/cases.csv")
NAN = float("nan")

def one(i):
    r = cases.iloc[i]
    imgs = [nib.load(p) for p in (r.t1n, r.t1c, r.t2w, r.t2f)]
    assert all(nib.aff2axcodes(m.affine) == ("L", "P", "S") for m in imgs)
    brain = np.zeros(imgs[0].shape, bool)
    for m in imgs: brain |= np.asarray(m.dataobj) != 0
    brain = ndi.binary_fill_holes(brain)
    depth = ndi.distance_transform_edt(brain, sampling=imgs[0].header.get_zooms()[:3])
    R = float(depth.max())
    if SRC == "expert":
        s = np.asarray(nib.load(r.seg).dataobj).astype(np.int16); et = s == 3
    else:
        s = np.asarray(nib.load(f"{X}/results/masks_{SRC}/{r.case}.nii.gz").dataobj).astype(np.int16); et = s == 4
    tc = et | (s == 1); wt = tc | (s == 2)
    bc = np.array(ndi.center_of_mass(brain)); idx = np.argwhere(brain); lo, hi = idx.min(0), idx.max(0)
    f = {"case": r.case}
    f["loc_wt_surface_frac5"] = float((depth[wt] <= 5).mean()) if wt.any() else NAN
    f["loc_tc_surface_frac5"] = float((depth[tc] <= 5).mean()) if tc.any() else NAN
    f["loc_tc_min_depth_rel"] = float(depth[tc].min() / R) if tc.any() else NAN
    f["loc_wt_mean_depth_rel"] = float(depth[wt].mean() / R) if wt.any() else NAN
    if wt.any():
        c = np.array(ndi.center_of_mass(wt))
        # LPS voxel axes: 0 = L-R, 1 = towards posterior, 2 = towards superior
        f["loc_lateral_offset"] = float(abs(c[0] - bc[0]) / ((hi[0] - lo[0]) / 2))
        f["loc_ap_pos"] = float(1 - (c[1] - lo[1]) / (hi[1] - lo[1]))     # 0 = posterior, 1 = anterior
        f["loc_si_pos"] = float((c[2] - lo[2]) / (hi[2] - lo[2]))         # 0 = inferior, 1 = superior
        f["loc_centroid_dist_rel"] = float(np.linalg.norm(c - bc) / R)
    else:
        for k in ("loc_lateral_offset", "loc_ap_pos", "loc_si_pos", "loc_centroid_dist_rel"): f[k] = NAN
    return f

if __name__ == "__main__":
    with ProcessPoolExecutor(8) as ex: rows = list(ex.map(one, range(len(cases))))
    loc = pd.DataFrame(rows).set_index("case")
    base = pd.read_csv(f"{D}/features_{SRC}.csv").set_index("case")
    assert len(base) == len(loc) == 146 and set(base.index) == set(loc.index)
    out = base.join(loc); out["mask_source"] = f"{SRC}_loc"
    out.reset_index().to_csv(f"{D}/features_{SRC}_loc.csv", index=False)
    print(SRC, out.shape, "| NaN per loc column:", loc.isna().sum().to_dict())
