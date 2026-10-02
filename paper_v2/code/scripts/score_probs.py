#!/usr/bin/env python3
"""Score saved probability maps with ONE fixed post-processing setting and the
reported statistics (score_case + summarise_segmentation, bootstrap CIs).
  score_probs.py PROBS_DIR TAG WT_POLICY FLOOR_MULT"""
import sys, pathlib
import numpy as np, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from brats_gbm.eval import postprocess as pp
from brats_gbm.eval.metrics import score_case
from brats_gbm.eval.stats import print_summary, summarise_segmentation
sys.path.insert(0, str(ROOT / "baselines" / "common"))
from sweep_postproc_hd95 import load_gt
pdir, tag, policy, mult = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3], float(sys.argv[4])
pp.MIN_VOXELS = {k: int(round(v * mult)) for k, v in pp.MIN_VOXELS.items()}
rows = []
for f in sorted(pdir.glob("*.npz")):
    prob = np.load(f)["prob"].astype(np.float32)
    pred = pp.postprocess(prob, 0.5, 0.5, 0.5, wt_policy=policy)
    r = {"Patient_ID": f.stem}; r.update(score_case(pred, load_gt(f.stem))); rows.append(r)
df = pd.DataFrame(rows)
out = ROOT / "results" / "brats"
df.to_csv(out / f"per_case_{tag}_recomputed.csv", index=False)
s = summarise_segmentation(df, label=tag); s["wt_policy"] = policy; s["floor_mult"] = mult
s.to_csv(out / f"summary_{tag}_recomputed.csv", index=False)
print_summary(s, f"{tag} (n={len(df)})")
