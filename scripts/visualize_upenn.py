#!/usr/bin/env python3
"""Render MRI / ground-truth / prediction figures for UPenn-GBM test subjects.

Same palette, layout and typography as the BraTS2021 figures — both go through
`brats_gbm.viz`, so they are identical by construction rather than by
convention.

Subjects are drawn from the 29-subject held-out test partition, and the best,
median and worst by mean Dice are shown unless explicit subject ids are given.
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

from brats_gbm import viz  # noqa: E402
from brats_gbm.data.upenn import PREPROCESSING, normalise, regions_from_seg  # noqa: E402
from brats_gbm.eval.inference import sliding_window_predict  # noqa: E402
from brats_gbm.eval.metrics import dice_score  # noqa: E402
from brats_gbm.eval.postprocess import postprocess  # noqa: E402
from brats_gbm.model import WaveletUNetPlusPlus  # noqa: E402

NIFTI = ROOT / "upenn_nifti"
CKPT = ROOT / "checkpoints" / "upenn_v3_best.pth"
RESULTS = ROOT / "results" / "upenn" / "per_case_upenn_v3_best.csv"
SELECTION = ROOT / "results" / "upenn" / "validation_selection.csv"
SUMMARY = ROOT / "results" / "upenn" / "segmentation_summary.csv"
OUTDIR = ROOT / "results" / "upenn" / "figures"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The fine-tuned checkpoint was trained under the UPenn convention.
PREPROC = "upenn"

# Fallback thresholds if the validation-selected ones are unavailable.
DEFAULT_THR = (0.40, 0.40, 0.40)


def selected_thresholds(setup: str = "upenn_v3_best") -> tuple[float, float, float]:
    """Reuse the thresholds evaluate_upenn.py chose on validation.

    Reading them back rather than hard-coding keeps the figures showing the
    same masks the reported Dice was computed from.
    """
    if SUMMARY.exists():
        s = pd.read_csv(SUMMARY)
        row = s[s.label == setup]
        if len(row):
            r = row.iloc[0]
            return (float(r.et_threshold), float(r.tc_threshold), float(r.wt_threshold))
    return DEFAULT_THR


def load_subject(sub: str, suffixes) -> dict[str, np.ndarray]:
    out = {}
    for suffix in suffixes:
        p = NIFTI / f"{sub}_{suffix}.nii.gz"
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")
        out[suffix] = nib.load(str(p)).get_fdata().astype(np.float32)
    return out


def main() -> None:
    suffixes, norm = PREPROCESSING[PREPROC]
    ap = argparse.ArgumentParser()
    ap.add_argument("subjects", nargs="*",
                    help="e.g. sub-009 (default: best/median/worst)")
    ap.add_argument("--modality", default="FLAIR", choices=list(suffixes))
    ap.add_argument("--out-dir", default=str(OUTDIR))
    ap.add_argument("--no-tta", action="store_true")
    args = ap.parse_args()

    if not RESULTS.exists():
        raise SystemExit(f"{RESULTS} missing — run scripts/evaluate_upenn.py first")
    res = pd.read_csv(RESULTS)

    if args.subjects:
        unknown = set(args.subjects) - set(res.Patient_ID)
        if unknown:
            raise SystemExit(
                f"not in the held-out test partition: {sorted(unknown)}\n"
                f"Figures are restricted to held-out subjects by design.")
        picks = [(s, "selected") for s in args.subjects]
    else:
        picks = viz.pick_cases(res)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = WaveletUNetPlusPlus(in_channels=4, n_classes=3).to(DEVICE)
    ck = torch.load(str(CKPT), map_location=DEVICE, weights_only=False)
    model.load_state_dict(
        ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck, strict=True)
    model.eval()

    et_thr, tc_thr, wt_thr = selected_thresholds()
    print(f"checkpoint   : {CKPT.name}")
    print(f"preprocessing: {suffixes} + {norm}")
    print(f"thresholds   : ET {et_thr} TC {tc_thr} WT {wt_thr} "
          f"(selected on validation)\n")

    for sub, kind in picks:
        vols = load_subject(sub, (*suffixes, "seg"))
        img = normalise(np.stack([vols[s] for s in suffixes]), norm)

        prob = sliding_window_predict(model, img, DEVICE, use_tta=not args.no_tta)
        pred_arr = postprocess(prob, et_thr, tc_thr, wt_thr).astype(np.float32)
        pred = {"ET": pred_arr[0], "TC": pred_arr[1], "WT": pred_arr[2]}

        g = regions_from_seg(vols["seg"])
        gt = {"ET": g[0], "TC": g[1], "WT": g[2]}

        dice = {r: dice_score(pred[r], gt[r]) for r in ("ET", "TC", "WT")}
        mean_dice = float(np.mean(list(dice.values())))

        z = viz.best_slice(gt["WT"], pred["WT"])
        base = vols[args.modality][:, :, z]

        out = out_dir / f"upenn_{sub}.png"
        viz.render_case(
            out, base,
            {k: v[:, :, z] for k, v in gt.items()},
            {k: v[:, :, z] for k, v in pred.items()},
            sub, args.modality, z, dice,
            f"UPenn-GBM held-out {kind}  ·  mean Dice {mean_dice:.3f}",
        )
        print(f"  {sub:<10} {kind:<12} mean={mean_dice:.4f}  -> {out}")

        del prob, pred_arr
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
