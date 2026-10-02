#!/usr/bin/env python3
"""Amendments 6+8, one fold: from saved probabilities, choose on the fold's 12 VAL
cases among {M1, M2, M1+M2} x Amendment-5 grid (highest Dice; ties within 0.001 ->
M1 alone, then lower HD95), then score the 19 TEST cases once for
 (a) M1 alone with M1's own best VAL setting   -> per_case_ours.csv      (Amendment 6)
 (b) the VAL-selected candidate+setting         -> per_case_ours_sel.csv  (Amendment 8)
  cv_fold_select.py FOLD"""
import sys, json, itertools, pathlib, numpy as np, pandas as pd
A = pathlib.Path("/mnt/data1/kamil_research/experiments/adapt_africa"); C = A / "eval_code"
sys.path.insert(0, str(C)); sys.path.insert(0, str(C / "baselines" / "common"))
from brats_gbm.eval import postprocess as pp
from brats_gbm.eval.metrics import score_case
from sweep_postproc_hd95 import load_gt
k = int(sys.argv[1]); D = A / "cv" / f"fold{k}"
if (D / "per_case_ours_sel.csv").exists(): sys.exit(f"fold {k} already scored")
ids = lambda n: [l.strip() for l in open(D / f"{n}_ids.txt") if l.strip()]
GRID = list(itertools.product(("components", "largest"), (1, 5, 20), (0, 100, 200, 500, 1000))); ORIG = dict(pp.MIN_VOXELS)
def post(p, wt, mult, et):
    pp.MIN_VOXELS = {x: v * mult for x, v in ORIG.items()}
    o = pp.postprocess(p, .5, .5, .5, wt_policy=wt, et_policy="none" if et == 0 else "min_volume", et_min_volume=max(et, 1))
    pp.MIN_VOXELS = dict(ORIG); return o
def probs(model, split, c): return np.load(D / f"probs_{model}_{split}" / f"{c}.npz")["prob"].astype(np.float32)
CANDS = {"M1": ("M1",), "M2": ("M2",), "M1+M2": ("M1", "M2")}
import os
if os.environ.get("CV_MODE") == "M1only" or not (D / "probs_M2_val").exists():   # Amendments 9/11: M1 only
    CANDS = {"M1": ("M1",)}
rows = []
for c in ids("val"):
    gt = load_gt(c); pm = {m: probs(m, "val", c) for m in {x for mem in CANDS.values() for x in mem}}
    for name, mem in CANDS.items():
        p = sum(pm[m] for m in mem) / len(mem)
        for g in GRID:
            s = score_case(post(p, *g), gt); rows.append(dict(cand=name, wt=g[0], mult=g[1], et=g[2], Dice=s["Dice_Mean"],
                                                              HD95=np.nanmean([s["HD95_ET"], s["HD95_TC"], s["HD95_WT"]])))
v = pd.DataFrame(rows).groupby(["cand", "wt", "mult", "et"]).mean().reset_index(); v.to_csv(D / "cv_val_selection.csv", index=False)
def pick(df):
    top = df.Dice.max(); t = df[df.Dice >= top - 0.001].copy(); t["m1"] = (t.cand != "M1").astype(int)
    return t.sort_values(["m1", "HD95"]).iloc[0]
a = pick(v[v.cand == "M1"]); b = pick(v)
sel = {"primary_M1": dict(wt=a.wt, mult=int(a.mult), et=int(a.et), val=float(a.Dice)),
       "selected": dict(cand=b.cand, wt=b.wt, mult=int(b.mult), et=int(b.et), val=float(b.Dice))}
(D / "cv_selection.json").write_text(json.dumps(sel, indent=1)); print("fold", k, sel, flush=True)
for out, cand, s in (("per_case_ours.csv", "M1", sel["primary_M1"]), ("per_case_ours_sel.csv", sel["selected"]["cand"], sel["selected"])):
    res = []
    for c in ids("test"):
        p = sum(probs(m, "test", c) for m in CANDS[cand]) / len(CANDS[cand])
        r = score_case(post(p, s["wt"], s["mult"], s["et"]), load_gt(c)); r["Patient_ID"] = c; res.append(r)
    pd.DataFrame(res).to_csv(D / out, index=False)
    print(out, cand, "TEST mean Dice %.4f" % pd.DataFrame(res).Dice_Mean.mean(), flush=True)
