#!/usr/bin/env python3
"""Failure-mode analysis for BraTS segmentation (paper Section V-G).

Two questions the aggregate Dice in Table IV cannot answer:

  (a) At what lesion size does enhancing-tumour detection break down?
  (b) What happens to the tumour core when the tumour does not enhance?

Both are answered from the per-case scores already in
`results/brats/per_case_internal_validation_recomputed.csv` plus the
ground-truth volumes read from `data/`, so this script needs no GPU and no
model. It writes the figure and a machine-readable summary; the numbers quoted
in the paper come from the JSON, not from reading the plot.

Colours are the data-viz reference palette (colourblind-validated), matching
`scripts/make_ablation_figures.py` so the paper's figures read as one set.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 9,
    "text.color": INK,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK2,
    "xtick.color": MUTED,
    "ytick.color": INK,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})

# Enhancing volume below which we call a tumour non-enhancing for the purpose of
# panel (b). Chosen as the point the panel (a) bins show detection has already
# collapsed, not fitted to the outcome.
ET_ABSENT_VOXELS = 300


def collect(per_case_csv: str, data_root: str) -> list[dict]:
    """Join per-case Dice with ground-truth region volumes."""
    rows = list(csv.DictReader(open(per_case_csv)))
    out = []
    for r in rows:
        seg = glob.glob(str(pathlib.Path(data_root) / r["Patient_ID"] / "*_seg.nii.gz"))
        if not seg:
            continue
        s = sitk.GetArrayFromImage(sitk.ReadImage(seg[0]))
        out.append({
            "id": r["Patient_ID"],
            "et_vol": int((s == 4).sum()),
            "tc_vol": int(((s == 4) | (s == 1)).sum()),
            "wt_vol": int((s > 0).sum()),
            "dice_et": float(r["Dice_ET"]),
            "dice_tc": float(r["Dice_TC"]),
            "dice_wt": float(r["Dice_WT"]),
        })
    return out


def summarise(rec: list[dict]) -> dict:
    et_vol = np.array([r["et_vol"] for r in rec])
    d_et = np.array([r["dice_et"] for r in rec])
    d_tc = np.array([r["dice_tc"] for r in rec])

    bins = [(1, 100), (100, 300), (300, 1000), (1000, 5000), (5000, 10**9)]
    by_size = []
    for lo, hi in bins:
        m = (et_vol >= lo) & (et_vol < hi)
        if m.sum():
            by_size.append({"lo": lo, "hi": None if hi > 10**8 else hi,
                            "n": int(m.sum()), "mean_dice_et": float(d_et[m].mean())})

    empty = et_vol == 0
    absent = et_vol < ET_ABSENT_VOXELS
    present = ~absent
    return {
        "n": len(rec),
        "et_dice_by_gt_volume": by_size,
        "empty_gt_et": {"n": int(empty.sum()), "mean_dice_et": float(d_et[empty].mean())},
        "tc_when_et_absent": {
            "threshold_voxels": ET_ABSENT_VOXELS,
            "n": int(absent.sum()),
            "mean_dice_tc": float(d_tc[absent].mean()),
            "median_dice_tc": float(np.median(d_tc[absent])),
        },
        "tc_when_et_present": {
            "n": int(present.sum()),
            "mean_dice_tc": float(d_tc[present].mean()),
            "median_dice_tc": float(np.median(d_tc[present])),
        },
    }


def figure(rec: list[dict], out_stem: pathlib.Path) -> None:
    et_vol = np.array([r["et_vol"] for r in rec], dtype=float)
    d_et = np.array([r["dice_et"] for r in rec])
    d_tc = np.array([r["dice_tc"] for r in rec])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 3.0))

    # ---- (a) detection cliff -------------------------------------------------
    nz = et_vol > 0
    ax1.scatter(et_vol[nz], d_et[nz], s=13, color=BLUE, alpha=0.45,
                linewidths=0, zorder=3)

    edges = np.array([1, 100, 300, 1000, 5000, 20000, 10**9], dtype=float)
    cx, cy = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (et_vol >= lo) & (et_vol < hi)
        if m.sum() >= 2:
            cx.append(np.sqrt(lo * min(hi, et_vol[m].max() or hi)))
            cy.append(d_et[m].mean())
    ax1.plot(cx, cy, color=INK, lw=1.6, zorder=5, marker="o", ms=4.5,
             mfc=SURFACE, mec=INK, mew=1.4)

    ax1.axvline(1000, color=BASELINE, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax1.text(1050, 0.045, "1,000 voxels", color=MUTED, fontsize=7.5, ha="left")

    ax1.set_xscale("log")
    ax1.set_xlabel("Ground-truth enhancing-tumour volume (voxels)")
    ax1.set_ylabel("Dice, enhancing tumour")
    ax1.set_ylim(-0.04, 1.04)
    ax1.set_title("(a)  Detection collapses on small lesions",
                  fontsize=9.5, color=INK, loc="left", pad=8)
    ax1.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    ax1.set_axisbelow(True)
    for side in ("top", "right"):
        ax1.spines[side].set_visible(False)

    # ---- (b) enhancement dependence of the core -----------------------------
    absent = et_vol < ET_ABSENT_VOXELS
    groups = [d_tc[absent], d_tc[~absent]]
    labels = [f"Barely enhancing\n(ET < {ET_ABSENT_VOXELS} vox, n={absent.sum()})",
              f"Enhancing\n(n={(~absent).sum()})"]
    colors = [ORANGE, BLUE]

    bp = ax2.boxplot(groups, positions=[0, 1], widths=0.5, showfliers=False,
                     patch_artist=True, medianprops=dict(color=INK, lw=1.6),
                     whiskerprops=dict(color=BASELINE, lw=1.0),
                     capprops=dict(color=BASELINE, lw=1.0),
                     boxprops=dict(lw=0))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.22)
        patch.set_edgecolor(c)
        patch.set_linewidth(1.2)

    rng = np.random.default_rng(0)
    for pos, vals, c in zip((0, 1), groups, colors):
        jitter = rng.uniform(-0.13, 0.13, len(vals))
        ax2.scatter(pos + jitter, vals, s=14 if pos == 0 else 8, color=c,
                    alpha=0.85 if pos == 0 else 0.30, linewidths=0, zorder=4)

    for pos, vals in zip((0, 1), groups):
        med = float(np.median(vals))
        ax2.annotate(f"median {med:.3f}", (pos, med), textcoords="offset points",
                     xytext=(34 if pos == 0 else 0, 6 if pos == 0 else 10),
                     ha="left" if pos == 0 else "center", fontsize=8, color=INK2)

    # The two barely-enhancing cases scoring 1.0 have an empty ground-truth core
    # and were correctly predicted empty. They are the reason the mean for this
    # group (0.27) overstates it and the median (0.008) is the honest summary;
    # leaving them unlabelled would read as a cherry-picked group.
    n_correct_empty = int(((d_tc[absent] > 0.99)).sum())
    if n_correct_empty:
        ax2.text(-0.46, 0.90, f"{n_correct_empty} cases: true core\nalso empty (correct)",
                 ha="left", va="top", fontsize=7.2, color=MUTED, linespacing=1.35)

    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("Dice, tumour core")
    ax2.set_ylim(-0.04, 1.04)
    ax2.set_title("(b)  The core is found by finding enhancement",
                  fontsize=9.5, color=INK, loc="left", pad=8)
    ax2.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    ax2.set_axisbelow(True)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-case",
                    default="results/brats/per_case_internal_validation_recomputed.csv")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out-dir", default="results/figures")
    args = ap.parse_args()

    rec = collect(args.per_case, args.data_root)
    if not rec:
        raise SystemExit("no cases matched; check --per-case and --data-root")

    summary = summarise(rec)
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    figure(rec, out / "fig_failure_modes")

    json_path = pathlib.Path("results/brats/failure_modes.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(json_path, "w"), indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nwrote {out/'fig_failure_modes'}.{{pdf,png}} and {json_path}")


if __name__ == "__main__":
    main()
