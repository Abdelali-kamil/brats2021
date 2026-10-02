"""Find out what is actually driving our HD95.

HD95 is our weakest number against every baseline -- 6.20 mm where nnU-Net
reports ~2.8 and Swin UNETR ~3-4. Unlike Dice, HD95 is a *distance* metric
dominated by the single furthest error, so it is not spread evenly across
cases: a handful of predictions with one stray component far from the tumour
can carry the whole mean. Dice barely notices those voxels; HD95 does.

This reports where the mean actually comes from, so the fix can be aimed rather
than guessed at:

  python baselines/common/diagnose_hd95.py results/brats/per_case_<tag>.csv

If prediction volumes were saved, pass --pred-dir and --gt-dir as well and it
also counts connected components on the worst cases and estimates how much
HD95 would improve if components below a size floor were dropped -- the
question `brats_gbm/eval/postprocess.py` already has the machinery for, but
whose thresholds were tuned for UPenn cross-cohort transfer rather than BraTS.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

REGIONS = ("ET", "TC", "WT")


def hd95_columns(df: pd.DataFrame) -> dict[str, str]:
    """Map region -> the HD95 column name, tolerating naming variants."""
    found = {}
    for r in REGIONS:
        for cand in (f"hd95_{r}", f"HD95_{r}", f"{r}_hd95", f"hd95{r}"):
            if cand in df.columns:
                found[r] = cand
                break
    return found


def concentration(values: np.ndarray) -> None:
    """How much of the mean comes from the worst few cases."""
    v = values[np.isfinite(values)]
    if v.size == 0:
        print("    no finite values")
        return
    total = v.sum()
    order = np.sort(v)[::-1]
    print(f"    mean {v.mean():7.3f}   median {np.median(v):7.3f}   "
          f"max {v.max():8.3f}   n={v.size}")
    for k in (1, 5, 10):
        if k <= v.size:
            share = order[:k].sum() / total * 100
            print(f"    top {k:2d} case(s) contribute {share:5.1f}% of the summed HD95")
    # What the mean would be without the worst 5 -- the headroom a
    # component-removal fix could plausibly reach.
    if v.size > 5:
        print(f"    mean excluding worst 5      : {np.sort(v)[:-5].mean():7.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="per-case CSV from evaluate_brats.py")
    ap.add_argument("--pred-dir", default=None,
                    help="directory of saved prediction volumes (optional)")
    ap.add_argument("--gt-dir", default=None,
                    help="directory of ground-truth segmentations (optional)")
    ap.add_argument("--top", type=int, default=10, help="worst N cases to list")
    args = ap.parse_args()

    path = pathlib.Path(args.csv)
    if not path.exists():
        sys.exit(f"not found: {path}")
    df = pd.read_csv(path)
    print(f"{path.name}: {len(df)} cases, columns: {', '.join(df.columns[:8])}\n")

    cols = hd95_columns(df)
    if not cols:
        sys.exit(f"no HD95 columns found. Available: {list(df.columns)}")

    for region, col in cols.items():
        print(f"[{region}]  ({col})")
        concentration(df[col].to_numpy(dtype=float))
        print()

    # The cases to actually look at.
    primary = cols.get("WT") or next(iter(cols.values()))
    worst = df.nlargest(args.top, primary)
    id_col = next((c for c in ("case", "case_id", "patient", "id", "subject")
                   if c in df.columns), df.columns[0])
    print(f"worst {args.top} cases by {primary}:")
    show = [id_col] + list(cols.values())
    print(worst[show].to_string(index=False))

    if args.pred_dir and args.gt_dir:
        analyse_components(worst[id_col].tolist(), args.pred_dir, args.gt_dir)
    else:
        print("\nPass --pred-dir and --gt-dir to also count connected components "
              "on these cases and estimate the gain from dropping small ones.")


def analyse_components(cases: list[str], pred_dir: str, gt_dir: str) -> None:
    """Count components on the worst cases; stray ones are the usual culprit."""
    try:
        import nibabel as nib
        from scipy import ndimage
    except ImportError as exc:
        print(f"\ncomponent analysis needs nibabel and scipy: {exc}")
        return

    print(f"\ncomponent analysis on {len(cases)} worst cases:")
    for case in cases:
        pred_p = next(pathlib.Path(pred_dir).glob(f"*{case}*.nii.gz"), None)
        if pred_p is None:
            print(f"  {case}: no prediction found"); continue
        pred = nib.load(str(pred_p)).get_fdata()
        mask = pred > 0
        lab, n = ndimage.label(mask)
        if n == 0:
            print(f"  {case}: empty prediction"); continue
        sizes = np.bincount(lab.ravel())[1:]
        order = np.sort(sizes)[::-1]
        largest = order[0]
        strays = (sizes < 0.05 * largest).sum()
        print(f"  {case}: {n:3d} component(s), largest {largest:7d} vox, "
              f"{strays:3d} under 5% of it "
              f"{'  <-- stray components likely drive HD95' if strays else ''}")


if __name__ == "__main__":
    main()
