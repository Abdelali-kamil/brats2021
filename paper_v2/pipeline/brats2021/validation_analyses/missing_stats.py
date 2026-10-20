#!/usr/bin/env python3
"""Missing paired statistics and uncertainty estimates for the compact paper structure (2026-10-19, user request).
Uses ONLY saved per-case result files and saved classification feature files: no inference, no training, no new TEST
scoring. Existing comparisons are recomputed and checked against the values recorded earlier (reuse if they match).

Conventions (as in extra_paired_stats.py / nnformer_paired.py / Amendments 21 and 28):
  * single-method mean: percentile bootstrap over cases, 2,000 resamples, seed 42 (brats_gbm.eval.stats.bootstrap_ci);
  * paired difference: bootstrap of the per-case difference, 10,000 resamples, a FRESH seed-0 generator per comparison,
    plus two-sided Wilcoxon signed-rank; "a higher/lower on k/n";
  * HD95 per case = mean over defined regions (case left out if none defined); HD95* = undefined set to 373.13 mm;
  * classification: logistic regression out-of-fold probabilities (classify.py cross_validate, repeated stratified
    5-fold x 10, seed 42), paired patient-level bootstrap (2,000, seed 0) of the AUC difference (paired_auc_logreg.py).
Selection rules followed (not re-chosen): BraTS 'ours' = Run C (single model) and Ensemble 3 (pre-specified headline);
Africa zero-shot 'ours' = the BraTS-VAL winner (Amendment 1: Ensemble 3) plus Run C as the single model; Africa CV primary
= start model fixed in advance (Run A, Amendments 6/11), Run C / D / D2 starts are secondary (Amendments 13/15/17);
classification primary = Run C zero-shot masks (Amendment 14)."""
import sys, json, pathlib
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, "/mnt/data1/kamil_research/experiments/eval_v2/code")
from brats_gbm.eval.stats import bootstrap_ci  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent
E = pathlib.Path("/mnt/data1/kamil_research/experiments")
B = pathlib.Path("/home/kamilabdelali/brats2021/results/brats")
X = E / "external_africa/results"
CV = E / "adapt_africa/cv"
REG = ("ET", "TC", "WT")
DIAG = float(np.sqrt(240 ** 2 + 240 ** 2 + 155 ** 2))  # 373.13 mm
L = lambda p: pd.read_csv(p).set_index("Patient_ID")
md = lambda d: d[[f"Dice_{r}" for r in REG]].mean(axis=1)
hdm = lambda d: d[[f"HD95_{r}" for r in REG]].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
hds = lambda d: d[[f"HD95_{r}" for r in REG]].apply(pd.to_numeric, errors="coerce").fillna(DIAG).mean(axis=1)

LINES, ROWS = [], []
def P(t=""):
    print(t); LINES.append(t)

def ci(v):
    m, lo, hi = bootstrap_ci(v); return m, lo, hi

def paired(section, a_name, b_name, metric, a, b, recorded=None, status="new"):
    ids = a.index.intersection(b.index)
    x = (a.loc[ids] - b.loc[ids]).dropna().to_numpy(float)
    bs = x[np.random.default_rng(0).integers(0, len(x), (10000, len(x)))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = wilcoxon(x).pvalue if np.any(x != 0) else 1.0
    hi_n, lo_n = int((x > 1e-12).sum()), int((x < -1e-12).sum())
    verdict = "a better" if lo > 0 else ("a worse" if hi < 0 else "n.s.")
    if metric.startswith("HD95"):  # lower is better
        verdict = "a better" if hi < 0 else ("a worse" if lo > 0 else "n.s.")
    chk = ""
    if recorded is not None:
        r = recorded
        same = all(abs(u - v) <= 0.0005 + 1e-12 for u, v in zip((x.mean(), lo, hi), r[:3]))
        chk = f"  | recorded {r[0]:+.4f} [{r[1]:+.4f}, {r[2]:+.4f}]" + (f" p={r[3]:.3g}" if len(r) > 3 else "") + \
              ("  -> MATCH (reused)" if same else "  -> DIFFERS (see note)")
        status = "reused-verified" if same else "recomputed (recorded value differs)"
    ROWS.append(dict(section=section, a=a_name, b=b_name, metric=metric, n=len(x), diff=x.mean(), lo=lo, hi=hi,
                     wilcoxon_p=p, a_higher=hi_n, a_lower=lo_n, verdict=verdict, status=status))
    P(f"  {a_name:22s} - {b_name:14s} {metric:9s} n={len(x):3d} {x.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]  p={p:.3g}  "
      f"higher {hi_n}/lower {lo_n}  {verdict}{chk}")

def summary(name, d, hd=True):
    m, lo, hi = ci(md(d))
    s = f"  {name:30s} n={len(d):3d}  mean {m:.4f} [{lo:.4f}, {hi:.4f}]  ET {d.Dice_ET.mean():.4f} TC {d.Dice_TC.mean():.4f} WT {d.Dice_WT.mean():.4f}"
    if hd:
        h = hdm(d); s += f"  HD95 {h.mean():.2f} (n={int(h.notna().sum())})  HD95* {hds(d).mean():.2f}"
    P(s)
    assert np.allclose(md(d), d.Dice_Mean, atol=1e-9), f"{name}: Dice_Mean != mean of regions"
    return m, lo, hi

# ------------------------------------------------------------------ A. BraTS 2021 TEST
P("== A. BraTS 2021 TEST (125 cases; each model's TEST scored once earlier; nothing re-scored)")
OURS = {"Run C (single)": "per_case_runC_test_tuned_recomputed.csv",
        "Ensemble 3 (headline)": "per_case_ensemble3_test_tuned_recomputed.csv",
        "Run C + rot (best single by VAL)": "per_case_runC_rot_test_tuned_recomputed.csv"}
BASE = {"Henry et al.": "henry_dsunet_test_per_case.csv", "nnFormer": "nnformer_test_per_case.csv",
        "SegResNet": "segresnet_test_per_case.csv", "TransBTS": "transbts_test_per_case.csv",
        "UNETR": "unetr_test_per_case.csv", "3D U-Net": "unet3d_test_per_case.csv"}
T = {k: L(B / f) for k, f in {**OURS, **BASE}.items()}
ids0 = T["Ensemble 3 (headline)"].index
for k, d in T.items():
    assert len(d) == 125 and set(d.index) == set(ids0), k
    summary(k, d)
# recorded values (mean, lo, hi[, p]) of comparisons that already exist
REC = {("Ensemble 3 (headline)", "nnFormer"): (-0.0091, -0.0241, 0.0020, 0.6636),      # eval_v2/nnformer_paired.txt
       ("Ensemble 3 (headline)", "SegResNet"): (-0.0059, -0.0151, 0.0016, 0.9049),     # nnformer_paired.txt check row
       ("Ensemble 3 (headline)", "Henry et al."): (-0.010, -0.024, 0.000, 0.97),       # 10-14 tables (fresh seed-0)
       ("Ensemble 3 (headline)", "UNETR"): (0.018, 0.003, 0.033),                      # 10-14 tables; paper +0.018 [+0.002, +0.033]
       ("Ensemble 3 (headline)", "3D U-Net"): (0.037, 0.022, 0.054),                   # 10-14 tables; paper [+0.022, +0.053]
       ("Run C (single)", "SegResNet"): (-0.008, -0.016, 0.002),                       # paper (compare.py shared stream)
       ("Run C (single)", "UNETR"): (0.017, 0.002, 0.031),
       ("Run C (single)", "3D U-Net"): (0.035, 0.021, 0.051),
       ("Run C + rot (best single by VAL)", "SegResNet"): (-0.007, -0.016, 0.002),
       ("Run C + rot (best single by VAL)", "UNETR"): (0.017, 0.002, 0.031),
       ("Run C + rot (best single by VAL)", "3D U-Net"): (0.036, 0.021, 0.052)}
P("\n  paired (ours - baseline); Dice: positive = ours better; HD95*: negative = ours better")
for o in OURS:
    for b in BASE:
        a_, b_ = T[o], T[b]
        paired("A_brats_test", o, b, "Dice_Mean", a_.Dice_Mean, b_.Dice_Mean, REC.get((o, b)))
        for r in REG:
            paired("A_brats_test", o, b, f"Dice_{r}", a_[f"Dice_{r}"], b_[f"Dice_{r}"])
        paired("A_brats_test", o, b, "HD95*", hds(a_), hds(b_))
    P()
P("  ablation (already reported under Amendment 21; recomputed as a check)")
ctrl = L(B / "per_case_ctrl_maxpool3d_test_tuned_recomputed.csv"); summary("max-pool control", ctrl)
paired("A_brats_test", "Run C (single)", "max-pool ctrl", "Dice_Mean", T["Run C (single)"].Dice_Mean, ctrl.Dice_Mean,
       (0.008980, 0.002327, 0.017252, 6.15e-06))
paired("A_brats_test", "Run C (single)", "max-pool ctrl", "HD95*", hds(T["Run C (single)"]), hds(ctrl),
       (-2.258360, -5.249364, -0.129450, 0.000568))

# ------------------------------------------------------------------ B. BraTS-Africa zero-shot
P("\n== B. BraTS-Africa zero-shot (95 gliomas primary; 51 other neoplasms reported separately)")
ZO = {"Ensemble 3 (BraTS-VAL winner)": "ours_ensemble3", "Run C (single)": "ours_runC",
      "Run C + rot": "ours_runC_rot", "Run D + rot": "ours_runD_rot", "Run D2 + rot": "ours_runD2_rot",
      "Run A": "ours_runA", "base (ep253)": "ours_ep253", "max-pool control": "ours_ctrl_maxpool3d"}
ZB = {"Henry et al.": "henry_dsunet", "nnFormer": "nnformer", "SegResNet": "segresnet", "TransBTS": "transbts",
      "UNETR": "unetr", "3D U-Net": "unet3d"}
Z = {k: L(X / f"per_case_{v}.csv") for k, v in {**ZO, **ZB}.items()}
for k, d in Z.items():
    assert len(d) == 146 and (d.group == "glioma").sum() == 95, k
G = {k: d[d.group == "glioma"] for k, d in Z.items()}
O = {k: d[d.group != "glioma"] for k, d in Z.items()}
P("  gliomas (95)")
for k in Z: summary(k, G[k])
P("  other neoplasms (51)")
for k in Z: summary(k, O[k], hd=False)
ZREC = {("Run C (single)", "SegResNet"): (-0.068, -0.101, -0.040), ("Run C (single)", "3D U-Net"): (-0.027, -0.058, -0.0002),
        ("Run C (single)", "UNETR"): (-0.022, -0.055, 0.007)}   # paper text (compare_external.py)
P("\n  paired on the 95 gliomas (ours - baseline)")
for o in ("Ensemble 3 (BraTS-VAL winner)", "Run C (single)"):
    for b in ZB:
        paired("B_africa_zs_glioma", o, b, "Dice_Mean", G[o].Dice_Mean, G[b].Dice_Mean, ZREC.get((o, b)))
        for r in REG:
            paired("B_africa_zs_glioma", o, b, f"Dice_{r}", G[o][f"Dice_{r}"], G[b][f"Dice_{r}"])
    P()
paired("B_africa_zs_glioma", "Ensemble 3 (BraTS-VAL winner)", "Run C (single)", "Dice_Mean",
       G["Ensemble 3 (BraTS-VAL winner)"].Dice_Mean, G["Run C (single)"].Dice_Mean)

# ------------------------------------------------------------------ C. BraTS-Africa 5-fold CV fine-tuning
P("\n== C. BraTS-Africa 5-fold CV fine-tuning (95 gliomas, pooled out-of-fold)")
def cvload(tag):
    d = pd.concat([pd.read_csv(CV / f"fold{i}" / f"per_case_{tag}.csv") for i in range(5)]).set_index("Patient_ID")
    assert len(d) == 95 and d.index.is_unique, tag
    return d
CO = {"from Run A (PRIMARY, fixed in advance)": ("ours", "ours_runA"), "from Run C (secondary)": ("ours_runC", "ours_runC"),
      "from Run D (secondary)": ("ours_runD", "ours_runD_rot"), "from Run D2 (secondary)": ("ours_runD2", "ours_runD2_rot")}
CB = {"Henry et al.": "henry_dsunet", "nnFormer": "nnformer", "SegResNet": "segresnet", "TransBTS": "transbts",
      "UNETR": "unetr", "3D U-Net": "unet3d"}
C = {k: cvload(v[0]) for k, v in CO.items()}
C.update({k: L(CV / f"per_case_cv_{v}.csv") for k, v in CB.items()})
TAG2ZO = {f: n for n, f in ZO.items()}
ZS_REF = {**{k: G[TAG2ZO[v[1]]] for k, v in CO.items()}, **{k: G[k] for k in CB}}
ids95 = C["from Run A (PRIMARY, fixed in advance)"].index
for k, d in C.items():
    assert set(d.index) == set(ids95) == set(ZS_REF[k].index), k
    summary(k, d)
CREC = {("from Run D2 (secondary)", "SegResNet"): (0.008, -0.000, 0.017, 0.051), ("from Run D2 (secondary)", "Henry et al."): (0.020, 0.006, 0.036),
        ("from Run D2 (secondary)", "UNETR"): (0.050, 0.034, 0.070), ("from Run D2 (secondary)", "3D U-Net"): (0.071, 0.051, 0.094)}
GREC = {"TransBTS": (0.0501, 0.0338, 0.0673), "nnFormer": (0.0095, -0.0047, 0.0229),
        "from Run A (PRIMARY, fixed in advance)": (0.066, 0.047, 0.087), "from Run C (secondary)": (0.091, 0.063, 0.121),
        "from Run D (secondary)": (0.074, 0.053, 0.097), "from Run D2 (secondary)": (0.083, 0.057, 0.112)}
P("\n  gain = adapted - own zero-shot on the same 95 gliomas")
for k in C:
    paired("C_africa_cv_gain", k, "own zero-shot", "Dice_Mean", C[k].Dice_Mean, ZS_REF[k].Dice_Mean, GREC.get(k))
P("\n  difference in gain (ours primary gain - baseline gain), same 95 gliomas")
gp = C["from Run A (PRIMARY, fixed in advance)"].Dice_Mean - ZS_REF["from Run A (PRIMARY, fixed in advance)"].Dice_Mean
for b in CB:
    paired("C_africa_cv_gain_diff", "gain Run A start", f"gain {b}", "Dice_Mean", gp, C[b].Dice_Mean - ZS_REF[b].Dice_Mean)
P("\n  paired (ours - baseline), adapted")
for o in CO:
    for b in CB:
        paired("C_africa_cv", o, b, "Dice_Mean", C[o].Dice_Mean, C[b].Dice_Mean, CREC.get((o, b)))
    P()

# ------------------------------------------------------------------ D. Classification (glioma vs other neoplasm)
P("== D. Classification, BraTS-Africa glioma (95) vs other neoplasm (51), logistic regression")
sys.path.insert(0, "/home/kamilabdelali/brats2021")
from brats_gbm.classification import build_models, cross_validate  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
CL = E / "classification_africa"
meta = pd.read_csv(CL / "africa_meta.csv").set_index("case")
SRC = {"Run C masks (PRIMARY)": "ours_runC", "expert masks": "expert", "Henry et al.": "henry_dsunet", "nnFormer": "nnformer",
       "SegResNet": "segresnet", "TransBTS": "transbts", "UNETR": "unetr", "3D U-Net": "unet3d"}
AREC = {"expert masks": 0.826, "Run C masks (PRIMARY)": 0.762, "Henry et al.": 0.781, "nnFormer": 0.799, "SegResNet": 0.772,
        "TransBTS": 0.776, "UNETR": 0.748, "3D U-Net": 0.771}
oof, y = {}, None
for k, s in SRC.items():
    f = pd.read_csv(CL / f"features_{s}.csv").set_index("case").loc[meta.index]
    yy = f.y_glioma.values.astype(int)
    Xf = f.drop(columns=["y_glioma", "mask_source"]).select_dtypes("number"); Xf = Xf.loc[:, ~Xf.isna().all()]
    y = yy if y is None else y; assert (y == yy).all()
    oof[k] = cross_validate(Xf.values.astype(float), yy, build_models()["logreg"])["oof_prob"]
    auc = roc_auc_score(y, oof[k])
    P(f"  {k:24s} AUC {auc:.3f}  (recorded {AREC[k]:.3f}{', MATCH' if abs(auc - AREC[k]) < 0.0005 else ', DIFFERS'})")
idx = np.random.default_rng(0).integers(0, len(y), (2000, len(y)))
DREC = {"expert masks": (0.064, -0.009, 0.137), "SegResNet": (0.009, -0.045, 0.068), "UNETR": (-0.014, -0.076, 0.048),
        "3D U-Net": (0.009, -0.048, 0.067)}   # paired_auc_logreg.txt (2026-10-10)
P("  paired AUC difference (source - Run C masks), patient-level bootstrap 2,000")
for k in SRC:
    if k == "Run C masks (PRIMARY)": continue
    pa, pb = oof[k], oof["Run C masks (PRIMARY)"]; d = roc_auc_score(y, pa) - roc_auc_score(y, pb)
    bs = [roc_auc_score(y[s], pa[s]) - roc_auc_score(y[s], pb[s]) for s in idx if len(np.unique(y[s])) == 2]
    lo, hi = np.percentile(bs, [2.5, 97.5]); r = DREC.get(k)
    same = r is not None and all(abs(u - v) <= 0.0005 + 1e-12 for u, v in zip((d, lo, hi), r))
    st = "new" if r is None else ("reused-verified" if same else "recomputed (recorded value differs)")
    ROWS.append(dict(section="D_classification_auc", a=k, b="Run C masks", metric="AUC_logreg", n=len(y), diff=d, lo=lo, hi=hi,
                     wilcoxon_p=np.nan, a_higher=np.nan, a_lower=np.nan, verdict="a better" if lo > 0 else ("a worse" if hi < 0 else "n.s."),
                     status=st))
    P(f"  {k:24s} - Run C masks  {d:+.3f} [{lo:+.3f}, {hi:+.3f}]  {st}")

pd.DataFrame(ROWS).to_csv(OUT / "paired_all.csv", index=False)
(OUT / "report.txt").write_text("\n".join(LINES) + "\n")
P(f"\nwrote {OUT/'paired_all.csv'} ({len(ROWS)} rows) and report.txt")
