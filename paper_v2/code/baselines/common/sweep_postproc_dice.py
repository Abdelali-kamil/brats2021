#!/usr/bin/env python3
"""PROTOCOL_v2 Amendment 5: post-processing sweep on VAL probabilities, objective =
mean Dice. Grid: WT policy {components, largest} x floors {x1,x5,x20} x ET rule
{off, 100, 200, 500, 1000}. Winner = highest mean Dice; ties within 0.001 -> lower
mean HD95. Writes results/brats/postproc_sweep_dice_<tag>.csv and prints WINNER."""
import argparse, itertools, pathlib, sys
import numpy as np, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from brats_gbm.eval import postprocess as pp
from brats_gbm.eval.metrics import score_case
sys.path.insert(0, str(ROOT / "baselines" / "common"))
from sweep_postproc_hd95 import load_gt
ap = argparse.ArgumentParser(); ap.add_argument("--probs", required=True); ap.add_argument("--tag", required=True)
a = ap.parse_args()
files = sorted(pathlib.Path(a.probs).glob("*.npz")); assert files
GRID = list(itertools.product(("components", "largest"), (1, 5, 20), (0, 100, 200, 500, 1000)))
ORIG = dict(pp.MIN_VOXELS)

def one_case(f):
    """All grid settings for one case (parallel over cases; identical per-case computation)."""
    prob = np.load(f)["prob"].astype(np.float32); gt = load_gt(f.stem); out = []
    for wt, mult, et in GRID:
        pp.MIN_VOXELS = {k: v * mult for k, v in ORIG.items()}
        out.append(score_case(pp.postprocess(prob, 0.5, 0.5, 0.5, wt_policy=wt, et_policy="none" if et == 0 else "min_volume",
                                             et_min_volume=max(et, 1)), gt))
    pp.MIN_VOXELS = dict(ORIG)
    return out

from concurrent.futures import ProcessPoolExecutor
with ProcessPoolExecutor(16) as ex:
    per_case = list(ex.map(one_case, files))
rows = []
for g, (wt, mult, et) in enumerate(GRID):
    d = pd.DataFrame([pc[g] for pc in per_case])
    rows.append(dict(wt_policy=wt, floor_mult=mult, et_min_volume=et, Dice_ET=d.Dice_ET.mean(), Dice_TC=d.Dice_TC.mean(),
                     Dice_WT=d.Dice_WT.mean(), Dice_Mean=d.Dice_Mean.mean(),
                     HD95_mean=d[["HD95_ET", "HD95_TC", "HD95_WT"]].mean(skipna=True).mean()))
    print(f"  {wt:10s} x{mult:<3d} ET>={et:<5d} Dice {rows[-1]['Dice_Mean']:.4f} HD95 {rows[-1]['HD95_mean']:.2f}", flush=True)
df = pd.DataFrame(rows); out = ROOT / "results" / "brats" / f"postproc_sweep_dice_{a.tag}.csv"; df.to_csv(out, index=False)
top = df.Dice_Mean.max(); w = df[df.Dice_Mean >= top - 0.001].sort_values("HD95_mean").iloc[0]
print(f"WINNER {w.wt_policy} {int(w.floor_mult)} {int(w.et_min_volume)} {w.Dice_Mean:.4f}")
