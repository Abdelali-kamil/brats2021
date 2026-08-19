#!/usr/bin/env python3
"""Evaluate the segmentor on BraTS2021 with the current inference pipeline.

Why this exists
---------------
The BraTS per-case scores this project previously reported came from an
archived pipeline whose inference settings were not recorded. They cannot be
reproduced by the current code: on BraTS2021_01628, for instance, the stored
file reports Dice 0.065/0.206/0.203 while the current pipeline predicts nothing
at all above threshold 0.5 (peak probabilities 0.005/0.011/0.487). Whatever
produced those numbers used a different operating point, a different crop, or
different preprocessing.

Numbers that cannot be regenerated should not be published, so this script
recomputes them under exactly the protocol used for UPenn-GBM: full-volume
sliding-window inference at 128^3 with 64^3 stride, Gaussian blending, 8-flip
test-time augmentation, and the shared post-processing.

Operating point
---------------
Fixed at 0.5, the value `train_brats.py` validates against. Unlike UPenn there
is no separate partition to tune on — the 251-case partition doubled as the
model-selection set during training — so tuning a threshold here would be
fitting on the data being reported. It is used as specified, not selected.

Preprocessing
-------------
The BraTS training convention: [t1, t1ce, t2, flair] channel order with
percentile-clipped min-max normalisation. See docs/METHODOLOGY.md for why this
matters.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brats_gbm.data.image import irm_min_max_preprocess  # noqa: E402
from brats_gbm.eval import postprocess as pp  # noqa: E402
from brats_gbm.eval.inference import sliding_window_predict  # noqa: E402
from brats_gbm.eval.metrics import score_case  # noqa: E402
from brats_gbm.eval.stats import print_summary, summarise_segmentation  # noqa: E402
from brats_gbm.model import WaveletUNetPlusPlus  # noqa: E402
from brats_gbm.splits import brats_split  # noqa: E402

DATA = ROOT / "data"
CKPT = ROOT / "checkpoints" / "segmentor_epoch_650.pth"
OUT_DIR = ROOT / "results" / "brats"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BRATS_ORDER = ("t1", "t1ce", "t2", "flair")
THR = 0.5
ET_LABEL, NCR_LABEL, ED_LABEL = 4, 1, 2


def load_case(case_id: str) -> tuple[np.ndarray, np.ndarray]:
    d = DATA / case_id
    img = np.stack([
        irm_min_max_preprocess(
            nib.load(str(d / f"{case_id}_{m}.nii.gz")).get_fdata().astype(np.float32))
        for m in BRATS_ORDER
    ])
    seg = nib.load(str(d / f"{case_id}_seg.nii.gz")).get_fdata().astype(np.float32)
    et = seg == ET_LABEL
    tc = et | (seg == NCR_LABEL)
    wt = tc | (seg == ED_LABEL)
    return img, np.stack([et, tc, wt])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="internal_validation",
                    choices=["internal_validation", "train", "all"])
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate only the first N cases (for a quick check)")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--tag", default=None)
    ap.add_argument("--checkpoint", default=str(CKPT),
                    help="checkpoint to evaluate (default: the released segmentor)")
    ap.add_argument("--downsample", default=None,
                    choices=["dwt", "maxpool_matched"],
                    help="encoder downsampling arm; read from the checkpoint's "
                         "config block when omitted, else 'dwt'")
    args = ap.parse_args()

    train_ids, val_ids = brats_split(str(DATA))
    if args.partition == "internal_validation":
        cases = sorted(val_ids)
    elif args.partition == "train":
        cases = sorted(train_ids)
    else:
        cases = sorted(train_ids | val_ids)
    if args.limit:
        cases = cases[:args.limit]

    tag = args.tag or args.partition
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.checkpoint)
    ck = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=False)

    # Prefer the arm recorded alongside the weights; fall back to the default so
    # the released bare-state_dict checkpoint keeps working untouched.
    arm = args.downsample
    if arm is None:
        arm = (ck.get("config", {}) or {}).get("downsample", "dwt") \
            if isinstance(ck, dict) else "dwt"

    model = WaveletUNetPlusPlus(in_channels=4, n_classes=3, downsample=arm).to(DEVICE)
    if isinstance(ck, dict):
        state = ck.get("model_state", ck.get("model_state_dict", ck))
    else:
        state = ck
    model.load_state_dict(state, strict=True)
    model.eval()

    print(f"checkpoint    : {ckpt_path.name}")
    print(f"downsampling  : {arm}")
    print(f"partition     : {args.partition}  ({len(cases)} cases)")
    print(f"preprocessing : {BRATS_ORDER} + percentile min-max")
    print(f"threshold     : {THR} (fixed, not tuned)")
    print(f"TTA           : {not args.no_tta}\n")

    rows = []
    for i, case_id in enumerate(cases, 1):
        try:
            img, gt = load_case(case_id)
        except FileNotFoundError as e:
            print(f"  [{i}/{len(cases)}] {case_id}  SKIP ({e})")
            continue

        prob = sliding_window_predict(model, img, DEVICE, use_tta=not args.no_tta)
        pred = pp.postprocess(prob, THR, THR, THR)
        row = {"Patient_ID": case_id}
        row.update(score_case(pred, gt))
        rows.append(row)
        print(f"  [{i}/{len(cases)}] {case_id}  "
              f"ET {row['Dice_ET']:.3f}  TC {row['Dice_TC']:.3f}  "
              f"WT {row['Dice_WT']:.3f}", flush=True)
        del prob, pred
        if i % 25 == 0:
            torch.cuda.empty_cache()

    if not rows:
        raise SystemExit("No cases evaluated.")

    df = pd.DataFrame(rows)
    per_case = out_dir / f"per_case_{tag}_recomputed.csv"
    df.to_csv(per_case, index=False)

    summary = summarise_segmentation(df, label=tag)
    summary["threshold"] = THR
    summary["tta"] = not args.no_tta
    summary.to_csv(out_dir / f"summary_{tag}_recomputed.csv", index=False)
    print_summary(summary, f"BraTS2021 {args.partition} (n={len(df)}) — recomputed")
    print(f"\nWrote {per_case}")


if __name__ == "__main__":
    main()
