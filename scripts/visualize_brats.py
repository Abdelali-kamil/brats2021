#!/usr/bin/env python3
"""Render MRI / ground-truth / prediction figures for BraTS2021 cases.

Same palette, layout and typography as the UPenn-GBM figures — both go through
`brats_gbm.viz`, so they are identical by construction rather than by
convention.

Cases are drawn from the 251-case internal-validation partition, never from the
1000 training cases, and the best, median and worst by mean Dice are shown
unless explicit case ids are given.

The model is fed the BraTS training convention: [t1, t1ce, t2, flair] channel
order with percentile-clipped min-max normalisation. This matters — feeding it
the UPenn convention instead costs roughly 0.56 mean Dice, which is the bug
documented in docs/METHODOLOGY.md.
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
from brats_gbm.data.image import irm_min_max_preprocess  # noqa: E402
from brats_gbm.eval.inference import sliding_window_predict  # noqa: E402
from brats_gbm.eval.metrics import dice_score  # noqa: E402
from brats_gbm.eval.postprocess import postprocess  # noqa: E402
from brats_gbm.model import WaveletUNetPlusPlus  # noqa: E402

DATA = ROOT / "data"
CKPT = ROOT / "checkpoints" / "segmentor_epoch_650.pth"
RESULTS = ROOT / "results" / "brats" / "per_case_internal_validation.csv"
OUTDIR = ROOT / "results" / "brats" / "figures"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# BraTS training convention: channel order and normalisation both matter.
BRATS_ORDER = ("t1", "t1ce", "t2", "flair")

# The training-time operating point (train_brats.py evaluates at 0.5). BraTS has
# no separate validation partition in this project — the 251 cases doubled as
# the model-selection set — so no threshold is fitted here; 0.5 is used as
# specified rather than tuned.
ET_THR = TC_THR = WT_THR = 0.5

# BraTS2021 labels: 1 necrotic core, 2 oedema, 4 enhancing tumour.
ET_LABEL, NCR_LABEL, ED_LABEL = 4, 1, 2


def load_case(case_id: str) -> dict[str, np.ndarray]:
    d = DATA / case_id
    out = {}
    for suffix in (*BRATS_ORDER, "seg"):
        p = d / f"{case_id}_{suffix}.nii.gz"
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")
        out[suffix] = nib.load(str(p)).get_fdata().astype(np.float32)
    return out


def regions(seg: np.ndarray) -> dict[str, np.ndarray]:
    et = seg == ET_LABEL
    tc = et | (seg == NCR_LABEL)
    wt = tc | (seg == ED_LABEL)
    return {"ET": et.astype(np.float32),
            "TC": tc.astype(np.float32),
            "WT": wt.astype(np.float32)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="*",
                    help="e.g. BraTS2021_00000 (default: best/median/worst)")
    ap.add_argument("--modality", default="flair", choices=BRATS_ORDER)
    ap.add_argument("--out-dir", default=str(OUTDIR))
    ap.add_argument("--no-tta", action="store_true")
    args = ap.parse_args()

    if not RESULTS.exists():
        raise SystemExit(f"{RESULTS} missing — run scripts/summarize_brats.py first")
    res = pd.read_csv(RESULTS)

    if args.cases:
        unknown = set(args.cases) - set(res.Patient_ID)
        if unknown:
            raise SystemExit(
                f"not in the internal-validation partition: {sorted(unknown)}\n"
                f"Figures are restricted to held-out cases by design.")
        picks = [(c, "selected") for c in args.cases]
    else:
        picks = viz.pick_cases(res)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = WaveletUNetPlusPlus(in_channels=4, n_classes=3).to(DEVICE)
    ck = torch.load(str(CKPT), map_location=DEVICE, weights_only=False)
    model.load_state_dict(
        ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck, strict=True)
    model.eval()

    print(f"checkpoint   : {CKPT.name}")
    print(f"preprocessing: {BRATS_ORDER} + percentile min-max")
    print(f"threshold    : {ET_THR} (training-time operating point, not tuned)\n")

    for case_id, kind in picks:
        vols = load_case(case_id)
        img = np.stack([irm_min_max_preprocess(vols[m]) for m in BRATS_ORDER])

        prob = sliding_window_predict(model, img, DEVICE, use_tta=not args.no_tta)
        pred_arr = postprocess(prob, ET_THR, TC_THR, WT_THR).astype(np.float32)
        pred = {"ET": pred_arr[0], "TC": pred_arr[1], "WT": pred_arr[2]}
        gt = regions(vols["seg"])

        dice = {r: dice_score(pred[r], gt[r]) for r in ("ET", "TC", "WT")}
        mean_dice = float(np.mean(list(dice.values())))

        z = viz.best_slice(gt["WT"], pred["WT"])
        base = vols[args.modality][:, :, z]

        out = out_dir / f"brats_{case_id}.png"
        viz.render_case(
            out, base,
            {k: v[:, :, z] for k, v in gt.items()},
            {k: v[:, :, z] for k, v in pred.items()},
            case_id, args.modality, z, dice,
            f"BraTS2021 internal-validation {kind}  ·  mean Dice {mean_dice:.3f}",
        )
        print(f"  {case_id:<18} {kind:<12} mean={mean_dice:.4f}  -> {out}")

        del prob, pred_arr
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
