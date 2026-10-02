#!/usr/bin/env python3
"""Amendment 13b: one fold; choose on the fold's 12 VAL cases among {MC, MC+M1, MC+M2, MC+M1+M2}
(probability averaging) x Amendment-5 grid (highest Dice; ties within 0.001 -> fewer members,
then lower HD95); score the 19 TEST cases once -> per_case_ours_runC_sel.csv.
  cv_fold_select_multi.py FOLD"""
import sys, json, itertools, pathlib, numpy as np, pandas as pd
A = pathlib.Path("/mnt/data1/kamil_research/experiments/adapt_africa"); C = A / "eval_code"
sys.path.insert(0, str(C)); sys.path.insert(0, str(C / "baselines" / "common"))
from brats_gbm.eval import postprocess as pp
from brats_gbm.eval.metrics import score_case
from sweep_postproc_hd95 import load_gt
k = int(sys.argv[1]); D = A / "cv" / f"fold{k}"; OUT = "per_case_ours_runC_sel.csv"
if (D / OUT).exists(): sys.exit(f"fold {k} already scored")
ids = lambda n: [l.strip() for l in open(D / f"{n}_ids.txt") if l.strip()]
GRID = list(itertools.product(("components", "largest"), (1, 5, 20), (0, 100, 200, 500, 1000))); ORIG = dict(pp.MIN_VOXELS)
def post(p, wt, mult, et):
    pp.MIN_VOXELS = {x: v * mult for x, v in ORIG.items()}
    o = pp.postprocess(p, .5, .5, .5, wt_policy=wt, et_policy="none" if et == 0 else "min_volume", et_min_volume=max(et, 1))
    pp.MIN_VOXELS = dict(ORIG); return o
CANDS = {"MC": ("MC",), "MC+M1": ("MC", "M1"), "MC+M2": ("MC", "M2"), "MC+M1+M2": ("MC", "M1", "M2")}
probs = lambda m, s, c: np.load(D / f"probs_{m}_{s}" / f"{c}.npz")["prob"].astype(np.float32)
rows = []
for c in ids("val"):
    gt = load_gt(c); pm = {m: probs(m, "val", c) for m in ("MC", "M1", "M2")}
    for name, mem in CANDS.items():
        p = sum(pm[m] for m in mem) / len(mem)
        for g in GRID:
            s = score_case(post(p, *g), gt)
            rows.append(dict(cand=name, n=len(mem), wt=g[0], mult=g[1], et=g[2], Dice=s["Dice_Mean"],
                             HD95=np.nanmean([s["HD95_ET"], s["HD95_TC"], s["HD95_WT"]])))
v = pd.DataFrame(rows).groupby(["cand", "n", "wt", "mult", "et"]).mean().reset_index(); v.to_csv(D / "cv_val_selection_runC_sel.csv", index=False)
w = v[v.Dice >= v.Dice.max() - 0.001].sort_values(["n", "HD95"]).iloc[0]
sel = dict(cand=w.cand, wt=w.wt, mult=int(w.mult), et=int(w.et), val=float(w.Dice)); (D / "cv_selection_runC_sel.json").write_text(json.dumps(sel, indent=1))
print("fold", k, sel, flush=True); res = []
for c in ids("test"):
    p = sum(probs(m, "test", c) for m in CANDS[sel["cand"]]) / len(CANDS[sel["cand"]])
    r = score_case(post(p, sel["wt"], sel["mult"], sel["et"]), load_gt(c)); r["Patient_ID"] = c; res.append(r)
pd.DataFrame(res).to_csv(D / OUT, index=False)
print(OUT, sel["cand"], "TEST mean Dice %.4f" % pd.DataFrame(res).Dice_Mean.mean(), flush=True)
