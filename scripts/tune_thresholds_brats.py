#!/usr/bin/env python3
"""Tune per-region thresholds and the ET min-volume rule on the TRAIN split.

Why this is here
----------------
`evaluate_brats.py` deliberately fixes the operating point at 0.5 because its
251-case partition doubled as the model-selection set during training — tuning a
threshold on the set you then report is fitting on your own test data.

This script keeps that honest: it selects the operating point on the *train*
partition and only *applies* the chosen point to the internal-validation
partition. The two partitions come from the same seeded split used everywhere
else (`brats_gbm.splits.brats_split`), so they are disjoint.

The enhancing tumour (ET) region is the one this helps most. ET Dice is
all-or-nothing on cases with no true ET, and the region sits near the
threshold, so a slightly lower ET threshold recovers true enhancement while the
min-volume rule suppresses the stray-voxel false positives that would otherwise
zero out a whole case.

Usage
-----
    # tune on 200 train cases, then score internal_validation at the result
    python scripts/tune_thresholds_brats.py --tune-limit 200

    # tune only (skip the validation scoring)
    python scripts/tune_thresholds_brats.py --tune-limit 200 --no-apply
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
from brats_gbm.eval.metrics import dice_score, score_case  # noqa: E402
from brats_gbm.eval.stats import print_summary, summarise_segmentation  # noqa: E402
from brats_gbm.model import WaveletUNetPlusPlus  # noqa: E402
from brats_gbm.splits import brats_split  # noqa: E402

DATA = ROOT / "data"
CKPT = ROOT / "checkpoints" / "segmentor_epoch_650.pth"
OUT_DIR = ROOT / "results" / "brats"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BRATS_ORDER = ("t1", "t1ce", "t2", "flair")
ET_LABEL, NCR_LABEL, ED_LABEL = 4, 1, 2

THR_GRID = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
ET_VOL_GRID = [0, 50, 100, 200, 300, 500]


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


def build_model() -> WaveletUNetPlusPlus:
    model = WaveletUNetPlusPlus(in_channels=4, n_classes=3).to(DEVICE)
    ck = torch.load(str(CKPT), map_location=DEVICE, weights_only=False)
    state = ck.get("model_state", ck.get("model_state_dict", ck)) if isinstance(ck, dict) else ck
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def tune(model, cases: list[str], use_tta: bool) -> dict:
    """Select each region's threshold (and ET min-volume) by mean per-region Dice.

    Thresholds are chosen per region on the raw channel probabilities. This
    ignores the ET-in-TC hierarchy coupling, which is a deliberate
    simplification: the final scoring in `apply()` runs the full post-processing
    (hierarchy, components, ET policy) at the selected point.
    """
    # ET is swept jointly over (threshold, min_volume); TC/WT over threshold.
    et_sum = {(t, v): 0.0 for t in THR_GRID for v in ET_VOL_GRID}
    tc_sum = {t: 0.0 for t in THR_GRID}
    wt_sum = {t: 0.0 for t in THR_GRID}
    n = 0

    for i, case_id in enumerate(cases, 1):
        try:
            img, gt = load_case(case_id)
        except FileNotFoundError as e:
            print(f"  [{i}/{len(cases)}] {case_id}  SKIP ({e})")
            continue

        prob = sliding_window_predict(model, img, DEVICE, use_tta=use_tta)
        for t in THR_GRID:
            et_bin = prob[0] > t
            et_count = int(et_bin.sum())
            for v in ET_VOL_GRID:
                pred = np.zeros_like(et_bin) if 0 < et_count < v else et_bin
                et_sum[(t, v)] += dice_score(pred, gt[0])
            tc_sum[t] += dice_score(prob[1] > t, gt[1])
            wt_sum[t] += dice_score(prob[2] > t, gt[2])
        n += 1
        print(f"  [{i}/{len(cases)}] {case_id}  accumulated", flush=True)
        del prob
        if i % 25 == 0 and DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    if n == 0:
        raise SystemExit("No tuning cases evaluated.")

    (et_thr, et_vol), et_best = max(et_sum.items(), key=lambda kv: kv[1])
    tc_thr, tc_best = max(tc_sum.items(), key=lambda kv: kv[1])
    wt_thr, wt_best = max(wt_sum.items(), key=lambda kv: kv[1])

    result = {
        "n_tune": n,
        "et_thr": et_thr, "et_min_volume": et_vol, "et_dice_tune": et_best / n,
        "tc_thr": tc_thr, "tc_dice_tune": tc_best / n,
        "wt_thr": wt_thr, "wt_dice_tune": wt_best / n,
    }
    print("\n=== Selected on TRAIN partition (n={}) ===".format(n))
    print(f"  ET: thr={et_thr}  min_volume={et_vol}   train-Dice={et_best / n:.4f}")
    print(f"  TC: thr={tc_thr}                     train-Dice={tc_best / n:.4f}")
    print(f"  WT: thr={wt_thr}                     train-Dice={wt_best / n:.4f}")
    print(f"  (baseline for reference: all thr=0.5, ET min_volume=200)\n")
    return result


def apply(model, cases: list[str], sel: dict, use_tta: bool, tag: str, out_dir: Path) -> None:
    rows = []
    for i, case_id in enumerate(cases, 1):
        try:
            img, gt = load_case(case_id)
        except FileNotFoundError as e:
            print(f"  [{i}/{len(cases)}] {case_id}  SKIP ({e})")
            continue
        prob = sliding_window_predict(model, img, DEVICE, use_tta=use_tta)
        pred = pp.postprocess(
            prob, sel["et_thr"], sel["tc_thr"], sel["wt_thr"],
            et_policy="min_volume", et_min_volume=sel["et_min_volume"],
        )
        row = {"Patient_ID": case_id}
        row.update(score_case(pred, gt))
        rows.append(row)
        print(f"  [{i}/{len(cases)}] {case_id}  "
              f"ET {row['Dice_ET']:.3f}  TC {row['Dice_TC']:.3f}  WT {row['Dice_WT']:.3f}",
              flush=True)
        del prob, pred
        if i % 25 == 0 and DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    if not rows:
        raise SystemExit("No apply cases evaluated.")

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_case = out_dir / f"per_case_{tag}_tuned.csv"
    df.to_csv(per_case, index=False)
    summary = summarise_segmentation(df, label=tag)
    summary["et_thr"] = sel["et_thr"]
    summary["tc_thr"] = sel["tc_thr"]
    summary["wt_thr"] = sel["wt_thr"]
    summary["et_min_volume"] = sel["et_min_volume"]
    summary.to_csv(out_dir / f"summary_{tag}_tuned.csv", index=False)
    print_summary(summary, f"BraTS2021 {tag} — tuned-on-train operating point")
    print(f"\nWrote {per_case}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune-limit", type=int, default=200,
                    help="number of TRAIN cases to tune on (None = all)")
    ap.add_argument("--apply-limit", type=int, default=None,
                    help="number of validation cases to score (None = all)")
    ap.add_argument("--no-apply", action="store_true",
                    help="tune only; do not score the validation partition")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    train_ids, val_ids = brats_split(str(DATA))
    tune_cases = sorted(train_ids)
    if args.tune_limit:
        tune_cases = tune_cases[:args.tune_limit]

    model = build_model()
    print(f"checkpoint : {CKPT.name}")
    print(f"tuning on  : {len(tune_cases)} TRAIN cases")
    print(f"TTA        : {not args.no_tta}\n")

    sel = tune(model, tune_cases, use_tta=not args.no_tta)

    if not args.no_apply:
        apply_cases = sorted(val_ids)
        if args.apply_limit:
            apply_cases = apply_cases[:args.apply_limit]
        print(f"\nApplying selected point to {len(apply_cases)} internal_validation cases...\n")
        apply(model, apply_cases, sel, use_tta=not args.no_tta,
              tag="internal_validation", out_dir=Path(args.out_dir))


if __name__ == "__main__":
    main()
