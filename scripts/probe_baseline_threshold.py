#!/usr/bin/env python3
"""Quantify how far threshold tuning alone can carry the zero-shot baseline.

Why this exists
---------------
Every setup gets its region thresholds chosen on validation. The fine-tuned
models select interior points of the grid, so the grid is adequate for them.
The zero-shot BraTS model does not: it selects whatever minimum it is offered,
and lowering the floor keeps raising its score.

That is not a grid that is too narrow. Sweeping down three orders of magnitude
shows the baseline's validation Dice rising **monotonically** all the way to
0.0002, with no interior optimum. Its enhancing-tumour and tumour-core
probabilities on UPenn-GBM sit almost entirely below any sensible threshold, so
lowering the cut does not recover a calibrated decision boundary — it just
labels progressively more voxels. Threshold tuning cannot repair the domain
gap.

The consequence for reporting: the baseline's Dice depends on where the grid
stops, so a single number would be arbitrary. This script records the whole
curve, so the reported figure can be read against it and nobody has to take the
grid floor on faith. The conclusion is unchanged across the entire range — the
baseline is catastrophic at every threshold, and fine-tuning moves it to ~0.82.

Runs entirely off the cached validation probability maps; no GPU.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brats_gbm.data.upenn import UPennDataset  # noqa: E402
from brats_gbm.eval.metrics import dice_score  # noqa: E402
from brats_gbm.splits import upenn_split  # noqa: E402

CACHE = ROOT / "cache" / "upenn_probs"
OUT_CSV = ROOT / "results" / "upenn" / "baseline_threshold_sensitivity.csv"

FLOORS = [0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05,
          0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
WT_THR = 0.20


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-stem", default="segmentor_epoch_650__brats")
    ap.add_argument("--nifti-dir", default=str(ROOT / "upenn_nifti"))
    ap.add_argument("--out", default=str(OUT_CSV))
    args = ap.parse_args()

    cache_dir = CACHE / args.checkpoint_stem / "val"
    if not cache_dir.exists():
        raise SystemExit(f"No cached maps at {cache_dir}; run evaluate_upenn.py first")

    _, val_subs, _ = upenn_split(args.nifti_dir)
    ds = UPennDataset(args.nifti_dir, val_subs)
    gt = {ds.subject_ids[i]: ds.labels_only(i) for i in range(len(ds))}
    probs = {p.stem: np.load(p) for p in sorted(cache_dir.glob("*.npy"))}
    print(f"checkpoint : {args.checkpoint_stem}")
    print(f"validation : {len(probs)} subjects (never used for reporting)\n")

    rows = []
    print(f"{'ET=TC thr':>10} {'ET':>8} {'TC':>8} {'WT':>8} {'MEAN':>8}")
    for t in FLOORS:
        per = {"ET": [], "TC": [], "WT": []}
        for pid, pr in probs.items():
            pr = pr.astype(np.float32)
            g = gt[pid]
            wt = pr[2] > WT_THR
            tc = (pr[1] > t) & wt
            et = (pr[0] > t) & tc
            per["ET"].append(dice_score(et, g[0]))
            per["TC"].append(dice_score(tc, g[1]))
            per["WT"].append(dice_score(wt, g[2]))
        means = {k: float(np.mean(v)) for k, v in per.items()}
        mean = float(np.mean(list(means.values())))
        rows.append({"et_tc_threshold": t, "wt_threshold": WT_THR,
                     "val_dice_ET": means["ET"], "val_dice_TC": means["TC"],
                     "val_dice_WT": means["WT"], "val_dice_mean": mean})
        print(f"{t:>10.4f} {means['ET']:>8.4f} {means['TC']:>8.4f} "
              f"{means['WT']:>8.4f} {mean:>8.4f}")

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    best_i = int(df.val_dice_mean.idxmax())
    best = df.iloc[best_i]
    print()
    print(f"peak validation Dice {best.val_dice_mean:.4f} at threshold "
          f"{best.et_tc_threshold}")
    if best_i == 0:
        print("That is the LOWEST threshold swept — the optimum lies below the "
              "grid, so this number is a function of where the sweep stops.")
    elif best_i == len(df) - 1:
        print("That is the HIGHEST threshold swept — the optimum lies above the "
              "grid, so this number is a function of where the sweep stops.")
    else:
        print("The optimum is interior to the sweep, so the grid is adequate and "
              "the selected threshold is a real choice rather than an edge "
              "artefact.")
    print(f"range over the sweep: {df.val_dice_mean.min():.4f} to "
          f"{df.val_dice_mean.max():.4f}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
