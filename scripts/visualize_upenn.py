#!/usr/bin/env python3
"""
Render MRI / ground-truth / prediction figures for UPENN-GBM test subjects.

Regions are drawn as nested overlays using the same colours as the results
report: whole tumour underneath, then tumour core, then enhancing tumour on top.
Picks the best, median and worst test subjects by mean Dice unless given
explicit subject ids.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brats_gbm.model import WaveletUNetPlusPlus  # noqa: E402
from brats_gbm.eval.inference import sliding_window_predict  # noqa: E402
from brats_gbm.eval.postprocess import apply_thresholds, enforce_hierarchy  # noqa: E402
from brats_gbm.data.upenn import ET_LABELS  # noqa: E402

UPENN = ROOT
NIFTI = UPENN / "upenn_nifti"
CKPT = ROOT / "checkpoints" / "upenn_v3_best.pth"
RESULTS = ROOT / "results" / "upenn" / "per_case_upenn_v3_best.csv"
OUTDIR = ROOT / "results" / "upenn" / "figures"

# thresholds selected on the validation split
ET_THR, TC_THR, WT_THR = 0.40, 0.40, 0.40

# report palette: WT aqua, TC orange, ET blue
COLORS = {"WT": (0.106, 0.686, 0.478), "TC": (0.922, 0.408, 0.204), "ET": (0.165, 0.471, 0.839)}


def load_subject(sub):
    def g(suffix):
        return nib.load(str(NIFTI / f"{sub}_{suffix}.nii.gz")).get_fdata().astype(np.float32)
    return g("FLAIR"), g("T1w"), g("ce-gd_T1w"), g("T2w")


def normalise(stack):
    img = np.stack(stack, 0)
    for c in range(img.shape[0]):
        m = img[c] > 0
        if m.sum():
            img[c] = (img[c] - img[c][m].mean()) / (img[c][m].std() + 1e-8)
            img[c][~m] = 0.0
    return img


def overlay(ax, base, masks, title):
    ax.imshow(np.rot90(base), cmap="gray", interpolation="nearest")
    for name in ("WT", "TC", "ET"):          # painted largest-first
        m = masks.get(name)
        if m is None or m.sum() == 0:
            continue
        rgba = np.zeros((*m.shape, 4), dtype=np.float32)
        rgba[..., :3] = COLORS[name]
        rgba[..., 3] = np.where(m > 0, 0.55, 0.0)
        ax.imshow(np.rot90(rgba), interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def best_slice(wt_gt, wt_pred):
    """Axial slice carrying the most tumour, so the figure shows something."""
    load = wt_gt.sum(axis=(0, 1)) if wt_gt.sum() else wt_pred.sum(axis=(0, 1))
    return int(np.argmax(load)) if load.sum() else wt_gt.shape[2] // 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subjects", nargs="*", help="e.g. sub-009 (default: best/median/worst)")
    ap.add_argument("--modality", default="FLAIR", choices=["FLAIR", "T1w", "ce-gd_T1w", "T2w"])
    args = ap.parse_args()

    res = pd.read_csv(RESULTS)
    res["mean"] = res[["Dice_ET", "Dice_TC", "Dice_WT"]].mean(axis=1)
    res = res.sort_values("mean", ascending=False).reset_index(drop=True)

    if args.subjects:
        picks = [(s, "selected") for s in args.subjects]
    else:
        picks = [
            (res.iloc[0].Patient_ID, "best case"),
            (res.iloc[len(res) // 2].Patient_ID, "median case"),
            (res.iloc[-1].Patient_ID, "worst case"),
        ]

    OUTDIR.mkdir(parents=True, exist_ok=True)
    model = load_model(str(CKPT))
    mod_idx = {"FLAIR": 0, "T1w": 1, "ce-gd_T1w": 2, "T2w": 3}[args.modality]

    for sub, kind in picks:
        raw = load_subject(sub)
        img = normalise(raw)

        prob = sliding_window_predict(model, img, SPATIAL_SIZE, SLIDE_STEP, use_tta=True)
        pred = apply_thresholds(prob, ET_THR, TC_THR, WT_THR)

        seg = nib.load(str(NIFTI / f"{sub}_seg.nii.gz")).get_fdata()
        gt = {
            "ET": np.isin(seg, ET_LABELS).astype(np.float32),
            "TC": (np.isin(seg, ET_LABELS) | (seg == 1)).astype(np.float32),
            "WT": (np.isin(seg, ET_LABELS) | (seg == 1) | (seg == 2)).astype(np.float32),
        }
        pr = {"ET": pred[0], "TC": pred[1], "WT": pred[2]}

        z = best_slice(gt["WT"], pr["WT"])
        base = raw[mod_idx][:, :, z]

        row = res[res.Patient_ID == sub].iloc[0]
        fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.9))
        overlay(axes[0], base, {}, f"{sub} — {args.modality}, slice {z}")
        overlay(axes[1], base, {k: v[:, :, z] for k, v in gt.items()}, "Expert annotation")
        overlay(axes[2], base, {k: v[:, :, z] for k, v in pr.items()},
                f"Prediction — Dice ET {row.Dice_ET:.3f} / TC {row.Dice_TC:.3f} / WT {row.Dice_WT:.3f}")

        # Regions are nested (ET ⊂ TC ⊂ WT) and painted largest-first, so each
        # colour that remains visible is that region minus the one inside it.
        handles = [mpatches.Patch(color=COLORS[k], label=lbl) for k, lbl in
                   [("ET", "Enhancing tumour (ET)"),
                    ("TC", "Necrotic core (TC \\ ET)"),
                    ("WT", "Oedema (WT \\ TC)")]]
        fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9)
        fig.suptitle(f"UPENN-GBM held-out {kind}  ·  mean Dice {row['mean']:.3f}",
                     fontsize=11, y=0.99)
        fig.tight_layout(rect=[0, 0.06, 1, 0.96])

        out = OUTDIR / f"upenn_{sub}.png"
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"  {sub:<10} {kind:<12} mean={row['mean']:.4f}  -> {out}")

        del prob, pred
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
