#!/usr/bin/env python3
"""Diagnostic for the next-experiment proposal (2026-10-17). TRAIN + VAL labels only; TEST not read.
(1) TRAIN: what the current patch sampler actually shows the network. Training crops a uniform random 128^3 window
    from the full SimpleITK (z,y,x) = (155,240,240) volume (brats_gbm/data/image.py get_crop_slice, no tumour-aware
    sampling). For every case, ALL possible crop offsets are evaluated exactly with a 3D prefix sum: fraction of crops
    with no TC, with >= 90 % of the case's TC, and the mean fraction of TC / WT voxels inside the crop.
(2) VAL: the same quantities for the deterministic centre crop used for training-time checkpoint selection.
(3) VAL: relation between E0 (Ensemble 3) TC errors and (a) tumour location (TC coverage by random training crops) and
    (b) tumour phenotype (ET fraction of TC, i.e. enhancing vs non-enhancing core). Descriptive Spearman correlations."""
import json, pathlib
import numpy as np, pandas as pd, SimpleITK as sitk
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import spearmanr

D = pathlib.Path("/mnt/data1/kamil_research/data/brats2021"); OUT = pathlib.Path(__file__).resolve().parent
A28 = pathlib.Path("/mnt/data1/kamil_research/experiments/eval_v2/amendment28")
S = 128


def box_sums(m):
    """Sum of mask m (z,y,x) inside every 128^3 window position the training sampler can pick (all offsets)."""
    c = np.pad(m.astype(np.int64), ((1, 0), (1, 0), (1, 0))).cumsum(0).cumsum(1).cumsum(2)
    z, y, x = [np.arange(0, n - S + 1) for n in m.shape]
    Z, Y, X = np.ix_(z, y, x)
    return (c[Z + S, Y + S, X + S] - c[Z, Y + S, X + S] - c[Z + S, Y, X + S] - c[Z + S, Y + S, X]
            + c[Z, Y, X + S] + c[Z, Y + S, X] + c[Z + S, Y, X] - c[Z, Y, X])


def case(cid):
    seg = sitk.GetArrayFromImage(sitk.ReadImage(str(D / cid / f"{cid}_seg.nii.gz")))
    assert seg.shape == (155, 240, 240), seg.shape
    et = seg == 4; tc = et | (seg == 1); wt = tc | (seg == 2)
    r = dict(Patient_ID=cid, et=int(et.sum()), tc=int(tc.sum()), wt=int(wt.sum()), ncr=int((seg == 1).sum()))
    for name, m in (("tc", tc), ("wt", wt)):
        if m.sum() == 0:
            continue
        b = box_sums(m) / m.sum()
        r[f"{name}_mean_cov"] = float(b.mean()); r[f"{name}_p_none"] = float((b == 0).mean())
        r[f"{name}_p_ge90"] = float((b >= 0.9).mean())
        cz, cy, cx = [(n - S) // 2 for n in m.shape]   # deterministic centre crop (validation)
        r[f"{name}_centre_cov"] = float(m[cz:cz + S, cy:cy + S, cx:cx + S].sum() / m.sum())
    return r


def main():
    sp = json.loads(pathlib.Path("/mnt/data1/kamil_research/experiments/nnformer/split_fixed.json").read_text())
    tr, va = sorted(sp["train"]), sorted(sp["val"])
    assert len(tr) == 1000 and len(va) == 126
    with ProcessPoolExecutor(4) as ex:
        rows = list(ex.map(case, tr + va, chunksize=8))
    df = pd.DataFrame(rows); df["set"] = ["train"] * len(tr) + ["val"] * len(va)
    df["et_frac_tc"] = np.where(df.tc > 0, df.et / df.tc.where(df.tc > 0, 1), np.nan)
    df.to_csv(OUT / "diag_crop_coverage.csv", index=False)
    L = []
    P = lambda s: (print(s), L.append(s))
    t = df[(df.set == "train") & (df.tc > 0)]
    P(f"== TRAIN (n={len(t)} with TC): uniform random 128^3 crop over all offsets")
    P(f"  TC: mean fraction of the case's TC inside a crop {t.tc_mean_cov.mean():.3f} (median {t.tc_mean_cov.median():.3f}); "
      f"P(crop has no TC) {t.tc_p_none.mean():.3f}; P(crop holds >= 90 % of TC) {t.tc_p_ge90.mean():.3f}")
    tw = df[(df.set == "train") & (df.wt > 0)]
    P(f"  WT: mean fraction inside a crop {tw.wt_mean_cov.mean():.3f}; P(no WT) {tw.wt_p_none.mean():.3f}; "
      f"P(>= 90 % of WT) {tw.wt_p_ge90.mean():.3f}")
    P(f"  cases where < 50 % of crops hold >= 90 % of TC: {int((t.tc_p_ge90 < 0.5).sum())}/{len(t)}; "
      f"where > 20 % of crops hold no TC: {int((t.tc_p_none > 0.2).sum())}/{len(t)}")
    v = df[(df.set == "val") & (df.tc > 0)]
    P(f"== VAL (n={len(v)} with TC): deterministic centre crop used for training-time checkpoint selection")
    P(f"  TC inside centre crop: mean {v.tc_centre_cov.mean():.3f}; cases with < 90 %: {int((v.tc_centre_cov < 0.9).sum())}; "
      f"< 50 %: {int((v.tc_centre_cov < 0.5).sum())}")
    vw = df[(df.set == "val") & (df.wt > 0)]
    P(f"  WT inside centre crop: mean {vw.wt_centre_cov.mean():.3f}; cases with < 90 %: {int((vw.wt_centre_cov < 0.9).sum())}")
    # (3) relation with E0's VAL TC errors
    e0 = pd.read_csv(A28 / "per_case_val.csv"); e0 = e0[e0.candidate == "E0"].set_index("Patient_ID")
    er = pd.read_csv(A28 / "errors_E0_val.csv"); er = er[er.region == "TC"].set_index("Patient_ID")
    m = v.set_index("Patient_ID").join(e0[["Dice_TC", "Dice_WT"]]).join(er[["fn", "fn_pred_as_edema", "fp_gt_edema"]])
    m["tc_err_conf_frac"] = (m.fn_pred_as_edema + m.fp_gt_edema) / m.tc
    P("== VAL: E0 TC Dice vs candidate explanations (Spearman rho, p; descriptive, 124 cases with TC)")
    for col, lab in (("tc_mean_cov", "location: TC coverage by random TRAIN crops"), ("tc_centre_cov", "TC in the centre crop"),
                     ("et_frac_tc", "phenotype: ET fraction of TC"), ("tc", "TC volume")):
        rho, p = spearmanr(m[col], m.Dice_TC); rho2, p2 = spearmanr(m[col], m.tc_err_conf_frac)
        P(f"  {lab:45s} vs Dice_TC rho {rho:+.3f} (p={p:.2g}); vs core<->edema error / TC vol rho {rho2:+.3f} (p={p2:.2g})")
    q = pd.qcut(m.et_frac_tc, 4, labels=["Q1 (least enhancing)", "Q2", "Q3", "Q4 (most enhancing)"])
    P("  E0 Dice_TC by quartile of ET fraction of TC: " + "; ".join(f"{k} {g.Dice_TC.mean():.3f} (n={len(g)}, "
      f"ET/TC {g.et_frac_tc.min():.2f}-{g.et_frac_tc.max():.2f})" for k, g in m.groupby(q, observed=True)))
    q2 = pd.qcut(m.tc_mean_cov, 4, labels=["Q1 (least covered)", "Q2", "Q3", "Q4 (best covered)"])
    P("  E0 Dice_TC by quartile of random-crop TC coverage: " + "; ".join(f"{k} {g.Dice_TC.mean():.3f} (n={len(g)})"
      for k, g in m.groupby(q2, observed=True)))
    tq = df[(df.set == "train") & (df.tc > 0)]
    P(f"== TRAIN phenotype: cases with ET fraction of TC < {m.et_frac_tc.quantile(0.25):.2f} (the VAL Q1 boundary): "
      f"{int((tq.et_frac_tc < m.et_frac_tc.quantile(0.25)).sum())}/{len(tq)}")
    m.to_csv(OUT / "diag_val_tc_errors.csv")
    (OUT / "diagnostics.txt").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
