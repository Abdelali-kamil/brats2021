"""Does component removal actually fix our HD95? Measure it, don't assume.

`diagnose_hd95.py` showed the median HD95 is 1.4-2.2 mm while the mean is
4.5-8.1, because a few cases carry components tens of millimetres from the
tumour. `postprocess.py` already has the machinery to drop those -- per-region
component floors (`MIN_VOXELS`, currently ET 5 / TC 20 / WT 50 voxels) and a WT
`largest` policy -- but its docstring says those values were chosen for UPenn
cross-cohort transfer, where skull leaks into the whole-tumour mask, and were
never tuned for BraTS.

This sweeps those settings over saved probability maps and reports what each
does to HD95 *and* to Dice, because the trade-off is the whole question: keeping
only the largest component removes distant false positives but also deletes the
satellite lesions of a genuinely multifocal tumour.

  python baselines/common/sweep_postproc_hd95.py --probs <dir> --tag worst25

IMPORTANT -- this is a diagnostic, not a tuning run. Run on a subset chosen for
being the worst cases, its numbers are a best case and must not be reported.
Tuning happens on the full validation split, and the winner is applied to test
once.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from brats_gbm.eval import postprocess as pp  # noqa: E402
from brats_gbm.eval.metrics import score_case  # noqa: E402

DATA = ROOT / "data"
THR = 0.5

# (label, WT policy, multiplier applied to the per-region voxel floors)
SETTINGS = [
    ("baseline (as shipped)",      "components", 1),
    ("floors x5",                  "components", 5),
    ("floors x20",                 "components", 20),
    ("floors x50",                 "components", 50),
    ("WT largest only",            "largest",    1),
    ("WT largest + floors x20",    "largest",    20),
]


def load_gt(case_id: str) -> np.ndarray:
    m = nib.load(str(DATA / case_id / f"{case_id}_seg.nii.gz")).get_fdata()
    et = m == 4
    tc = np.logical_or(m == 4, m == 1)
    wt = np.logical_or(tc, m == 2)
    return np.stack([et, tc, wt])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", required=True, help="dir of <case>.npz from --save-probs")
    ap.add_argument("--tag", default="sweep")
    args = ap.parse_args()

    pdir = pathlib.Path(args.probs)
    files = sorted(pdir.glob("*.npz"))
    if not files:
        sys.exit(f"no .npz under {pdir}")
    print(f"{len(files)} cases from {pdir}\n")

    original = dict(pp.MIN_VOXELS)
    rows = []

    for label, wt_policy, mult in SETTINGS:
        # MIN_VOXELS is a module-level constant the function reads directly, so
        # the only way to vary it is to swap it and restore afterwards.
        pp.MIN_VOXELS = {k: v * mult for k, v in original.items()}
        per_case = []
        for f in files:
            case = f.stem
            prob = np.load(f)["prob"].astype(np.float32)
            gt = load_gt(case)
            pred = pp.postprocess(prob, THR, THR, THR, wt_policy=wt_policy)
            per_case.append(score_case(pred, gt))
        pp.MIN_VOXELS = dict(original)

        df = pd.DataFrame(per_case)
        rows.append({
            "setting": label,
            "floors": str({k: v * mult for k, v in original.items()}),
            "wt_policy": wt_policy,
            "HD95_ET": df["HD95_ET"].mean(skipna=True),
            "HD95_TC": df["HD95_TC"].mean(skipna=True),
            "HD95_WT": df["HD95_WT"].mean(skipna=True),
            "Dice_ET": df["Dice_ET"].mean(skipna=True),
            "Dice_TC": df["Dice_TC"].mean(skipna=True),
            "Dice_WT": df["Dice_WT"].mean(skipna=True),
            "Dice_Mean": df["Dice_Mean"].mean(skipna=True),
        })
        r = rows[-1]
        print(f"  {label:<26} HD95 ET/TC/WT {r['HD95_ET']:6.2f}/{r['HD95_TC']:6.2f}/"
              f"{r['HD95_WT']:6.2f}   Dice {r['Dice_Mean']:.4f}", flush=True)

    out = ROOT / "results" / "brats" / f"postproc_sweep_{args.tag}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")

    base, best = rows[0], min(rows, key=lambda r: r["HD95_WT"])
    if best is not base:
        print(f"\nlowest WT HD95: '{best['setting']}' "
              f"{base['HD95_WT']:.2f} -> {best['HD95_WT']:.2f} mm "
              f"({base['HD95_WT'] - best['HD95_WT']:+.2f}), "
              f"Dice {base['Dice_Mean']:.4f} -> {best['Dice_Mean']:.4f} "
              f"({best['Dice_Mean'] - base['Dice_Mean']:+.4f})")
    print("\nDiagnostic only -- these cases were selected for being the worst. "
          "Tune on the full validation split before reporting anything.")


if __name__ == "__main__":
    main()
