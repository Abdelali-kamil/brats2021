#!/usr/bin/env python3
"""Score any method's segmentation predictions with THIS project's metrics.

The fairness backbone of the baseline comparison. Every segmentation method —
nnU-Net, Swin UNETR, SelfMedMAE, MTANet, and your own Wavelet U-Net++ — has its
predictions scored by the exact same code (`brats_gbm.eval.metrics.score_case`,
the ET/TC/WT Dice + HD95 used for the reported "Ours" numbers), on the exact same
cases. No method is scored by its own repo's metric, so no method gets a scoring
advantage. That is what makes the final table a fair comparison.

Inputs are label-map NIfTIs (one file per case). BraTS labels are assumed:
necrotic=1, edema=2, enhancing=ET (default label 4; pass --et-label 3 for the
nnU-Net-style relabelling). Regions follow brats_gbm/data/brats.py:
    ET = (seg == ET);  TC = ET | (seg == 1);  WT = TC | (seg == 2).

Usage (on your server, after a method has written predictions):
    python baselines/common/score_segmentation.py \
        --key nnunet --pred /path/to/nnunet_predictions --gt /path/to/gt_segs

Writes results/baselines/<key>_segmentation.json, which
baselines/common/make_comparison_table.py reads into the comparison table.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from brats_gbm.eval.metrics import score_case  # noqa: E402

OUT_DIR = ROOT / "results" / "baselines"


def to_regions(seg: np.ndarray, et_label: int) -> np.ndarray:
    """Label map -> stacked [ET, TC, WT] boolean masks (brats.py convention)."""
    et = seg == et_label
    tc = et | (seg == 1)
    wt = tc | (seg == 2)
    return np.stack([et, tc, wt])


def case_id(path: Path) -> str:
    """Best-effort case id, matching BraTS or UPenn naming, else the file stem."""
    for pat in (r"BraTS\d{4}_\d{3,6}",   # BraTS2021_00001 (case number, not the year)
                r"sub-?\d+"):             # UPenn sub-251 / sub251
        m = re.search(pat, path.name)
        if m:
            return m.group(0)
    # strip common channel/suffix decorations
    return re.sub(r"(_seg|_pred|_0000)?\.nii(\.gz)?$", "", path.name)


def index_dir(d: Path) -> dict[str, Path]:
    files = sorted([p for p in d.iterdir()
                    if p.name.endswith((".nii", ".nii.gz"))])
    return {case_id(p): p for p in files}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True, help="method key, e.g. nnunet / swin_unetr / mtanet / ours")
    ap.add_argument("--pred", required=True, help="dir of predicted label-map NIfTIs")
    ap.add_argument("--gt", required=True, help="dir of ground-truth seg NIfTIs")
    ap.add_argument("--et-label", type=int, default=4, help="enhancing-tumour label (4 BraTS raw, 3 nnU-Net-style)")
    ap.add_argument("--gt-et-label", type=int, default=None, help="override ET label for GT if it differs from --et-label")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    try:
        import nibabel as nib
    except ImportError:
        raise SystemExit("nibabel is required: pip install nibabel")

    preds = index_dir(Path(args.pred))
    gts = index_dir(Path(args.gt))
    common = sorted(set(preds) & set(gts))
    if not common:
        raise SystemExit(f"no matching case ids between {args.pred} and {args.gt}\n"
                         f"  pred ids (first 5): {list(preds)[:5]}\n"
                         f"  gt   ids (first 5): {list(gts)[:5]}")
    missing = sorted((set(preds) | set(gts)) - set(common))
    if missing:
        print(f"warning: {len(missing)} case(s) not in both dirs, skipped: {missing[:5]}")

    gt_et = args.gt_et_label if args.gt_et_label is not None else args.et_label
    rows = []
    for cid in common:
        pimg = nib.load(str(preds[cid]))
        gimg = nib.load(str(gts[cid]))
        spacing = tuple(float(z) for z in gimg.header.get_zooms()[:3])
        pred = to_regions(np.asarray(pimg.dataobj).astype(np.int16), args.et_label)
        gt = to_regions(np.asarray(gimg.dataobj).astype(np.int16), gt_et)
        s = score_case(pred, gt, spacing)
        s["case"] = cid
        rows.append(s)

    def agg(key):
        vals = np.array([r[key] for r in rows], float)
        vals = vals[np.isfinite(vals)]
        return (float(np.mean(vals)), float(np.std(vals))) if len(vals) else (None, None)

    dice_mean, dice_sd = agg("Dice_Mean")
    # HD95 overall = mean over the three regions' per-case finite values
    hd_all = np.array([r[f"HD95_{reg}"] for r in rows for reg in ("ET", "TC", "WT")], float)
    hd_all = hd_all[np.isfinite(hd_all)]
    hd95_mean = float(np.mean(hd_all)) if len(hd_all) else None
    hd95_sd = float(np.std(hd_all)) if len(hd_all) else None

    result = {
        "key": args.key,
        "n_cases": len(rows),
        "dice_mean": dice_mean, "dice_sd": dice_sd,
        "hd95_mean": hd95_mean, "hd95_sd": hd95_sd,
        "per_region": {
            reg: {
                "dice_mean": agg(f"Dice_{reg}")[0], "dice_sd": agg(f"Dice_{reg}")[1],
                "hd95_mean": agg(f"HD95_{reg}")[0], "hd95_sd": agg(f"HD95_{reg}")[1],
            } for reg in ("ET", "TC", "WT")
        },
        "scored_by": "brats_gbm.eval.metrics.score_case (identical to 'Ours')",
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.key}_segmentation.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"scored {len(rows)} cases | Dice {dice_mean:.4f}  HD95 {hd95_mean:.2f} mm")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
