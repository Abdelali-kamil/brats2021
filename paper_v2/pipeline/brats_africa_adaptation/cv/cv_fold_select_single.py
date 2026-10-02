#!/usr/bin/env python3
"""Amendment 13 (secondary): one fold, ONE adapted model (label, e.g. MC = from Run C).
Post-processing chosen on the fold's 12 VAL cases over the Amendment-5 grid (highest Dice;
ties within 0.001 -> lower HD95), then the fold's 19 TEST cases scored ONCE.
  cv_fold_select_single.py FOLD LABEL OUT_CSV"""
import sys, json, itertools, pathlib, numpy as np, pandas as pd
A = pathlib.Path("/mnt/data1/kamil_research/experiments/adapt_africa"); C = A / "eval_code"
sys.path.insert(0, str(C)); sys.path.insert(0, str(C / "baselines" / "common"))
from brats_gbm.eval import postprocess as pp
from brats_gbm.eval.metrics import score_case
from sweep_postproc_hd95 import load_gt
k, M, OUT = int(sys.argv[1]), sys.argv[2], sys.argv[3]; D = A / "cv" / f"fold{k}"
if (D / OUT).exists(): sys.exit(f"fold {k} {M} already scored")
ids = lambda n: [l.strip() for l in open(D / f"{n}_ids.txt") if l.strip()]
GRID = list(itertools.product(("components", "largest"), (1, 5, 20), (0, 100, 200, 500, 1000))); ORIG = dict(pp.MIN_VOXELS)
def post(p, wt, mult, et):
    pp.MIN_VOXELS = {x: v * mult for x, v in ORIG.items()}
    o = pp.postprocess(p, .5, .5, .5, wt_policy=wt, et_policy="none" if et == 0 else "min_volume", et_min_volume=max(et, 1))
    pp.MIN_VOXELS = dict(ORIG); return o
probs = lambda s, c: np.load(D / f"probs_{M}_{s}" / f"{c}.npz")["prob"].astype(np.float32)
rows = []
for c in ids("val"):
    gt = load_gt(c); p = probs("val", c)
    for g in GRID:
        s = score_case(post(p, *g), gt)
        rows.append(dict(wt=g[0], mult=g[1], et=g[2], Dice=s["Dice_Mean"], HD95=np.nanmean([s["HD95_ET"], s["HD95_TC"], s["HD95_WT"]])))
v = pd.DataFrame(rows).groupby(["wt", "mult", "et"]).mean().reset_index(); v.to_csv(D / f"cv_val_selection_{M}.csv", index=False)
w = v[v.Dice >= v.Dice.max() - 0.001].sort_values("HD95").iloc[0]
sel = dict(wt=w.wt, mult=int(w.mult), et=int(w.et), val=float(w.Dice)); (D / f"cv_selection_{M}.json").write_text(json.dumps(sel, indent=1))
print("fold", k, M, sel, flush=True)
res = []
for c in ids("test"):
    r = score_case(post(probs("test", c), sel["wt"], sel["mult"], sel["et"]), load_gt(c)); r["Patient_ID"] = c; res.append(r)
pd.DataFrame(res).to_csv(D / OUT, index=False)
print(OUT, M, "TEST mean Dice %.4f" % pd.DataFrame(res).Dice_Mean.mean(), flush=True)
