#!/usr/bin/env python
"""Architecture diagram of the *implemented* pipeline (thesis Methods figure).

Draws what the code actually builds — verified against brats_gbm/model.py and
brats_gbm/gnn.py — NOT the original proposal. Deliberately omits the Transformer
branch and the learned lesion/slice/global encoders, which were never built.

Stage 1  Wavelet U-Net++ (DWT downsampling) -> 3-region mask (ET/TC/WT)
Stage 2  51 radiomic features (+ 3 clinical) -> KAN|MLP encoders -> concat/LN
         -> [adaptive gated fusion] -> [memory-bank GNN] -> linear head -> MGMT

Boxes with a dashed outline are ablation toggles (GNN = rungs 4-5, gate = rung 5);
the imaging and clinical encoders are KAN *or* MLP, swapped across the ablation.

Output: results/figures/fig_architecture_implemented.{pdf,png}
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(ROOT, "results", "figures")
os.makedirs(FIGDIR, exist_ok=True)

# reference palette (same as the ablation figures)
SURFACE = "#fcfcfb"; PLANE = "#f9f9f7"; INK = "#0b0b0b"; INK2 = "#52514e"
MUTED = "#898781"; GRID = "#e1e0d9"; BASE = "#c3c2b7"
BLUE = "#2a78d6"; ORANGE = "#eb6834"; AQUA = "#1baf7a"; VIOLET = "#4a3aa7"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "svg.fonttype": "none",
})

fig, ax = plt.subplots(figsize=(7.4, 9.7))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")


def box(cx, cy, w, h, text, *, fill=SURFACE, edge=INK, tcolor=INK,
        fs=9.5, bold=False, dashed=False, lw=1.6):
    p = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                       boxstyle="round,pad=0.35,rounding_size=1.4",
                       linewidth=lw, edgecolor=edge, facecolor=fill,
                       linestyle=(0, (4, 2.5)) if dashed else "solid", zorder=3)
    ax.add_patch(p)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=tcolor,
            fontweight="bold" if bold else "normal", zorder=4, linespacing=1.25)


def arrow(x0, y0, x1, y1, label=None, color=INK2, lw=1.7, rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                 arrowstyle="-|>", mutation_scale=15, lw=lw, color=color,
                 connectionstyle=f"arc3,rad={rad}", zorder=2,
                 shrinkA=1.5, shrinkB=1.5))
    if label:
        ax.text((x0 + x1) / 2 + 2.5, (y0 + y1) / 2, label, fontsize=8,
                color=MUTED, ha="left", va="center", style="italic", zorder=4)


def stage(y0, y1, title, subtitle):
    ax.add_patch(FancyBboxPatch((3, y0), 94, y1 - y0,
                 boxstyle="round,pad=0.4,rounding_size=1.6",
                 linewidth=1.0, edgecolor=GRID, facecolor=PLANE, zorder=0))
    ax.text(5.5, y1 - 1.2, title, fontsize=10.5, fontweight="bold", color=INK,
            ha="left", va="top", zorder=1)
    ax.text(5.5, y1 - 4.0, subtitle, fontsize=8.3, color=INK2, ha="left",
            va="top", zorder=1)


# ---- stage backgrounds --------------------------------------------------
stage(63.5, 98, "Stage 1 — Segmentation",
      "trained on BraTS2021 · applied to UPenn-GBM")
stage(2, 61, "Stage 2 — MGMT-methylation classification",
      "per-patient imaging + clinical fusion")

# ---- Stage 1 ------------------------------------------------------------
box(50, 86, 56, 6.6,
    "Multi-parametric MRI  ·  4 channels\n(T1, T1ce, T2, FLAIR)",
    fill="#eaf2fc", edge=BLUE, fs=9.3)
box(50, 76, 62, 7.6,
    "Wavelet U-Net++\nDWT downsampling  ·  dense (nested) skip connections",
    fill=BLUE, edge=BLUE, tcolor="#ffffff", fs=9.8, bold=True)
box(50, 66.7, 54, 6.6,
    "3-region segmentation mask   (ET ⊆ TC ⊆ WT)",
    fill=SURFACE, edge=BLUE, fs=9.3)
arrow(50, 82.7, 50, 79.9)
arrow(50, 72.2, 50, 70.1)

# ---- Stage 2: feature sources ------------------------------------------
box(28, 50, 40, 8.2,
    "Radiomics extraction\n51 features  (shape / intensity / texture)",
    fill=SURFACE, edge=AQUA, fs=9.2)
box(74, 50, 40, 8.2,
    "Clinical tabular  ·  3 features\nage,  sex,  GTR > 90%",
    fill=SURFACE, edge=ORANGE, fs=9.2)
arrow(46, 63.4, 30, 54.4)                            # mask -> radiomics
ax.text(33.5, 61.9, "mask + MRI", fontsize=8, color=MUTED, ha="right",
        va="center", style="italic")
ax.text(74, 58.4, "external input", fontsize=8, color=MUTED, ha="center",
        va="bottom", style="italic")
arrow(74, 57.8, 74, 54.3)

# ---- encoders ----------------------------------------------------------
box(28, 40, 34, 7.4, "Imaging encoder\n(KAN  or  MLP)",
    fill="#f2f4f7", edge=INK2, fs=9.2)
box(74, 40, 34, 7.4, "Clinical encoder\n(KAN  or  MLP)",
    fill="#f2f4f7", edge=INK2, fs=9.2)
arrow(28, 45.9, 28, 43.9)
arrow(74, 45.9, 74, 43.9)

# ---- fusion spine ------------------------------------------------------
box(51, 31, 44, 6.2, "Concatenate  +  LayerNorm", fill="#f2f4f7", edge=INK2,
    fs=9.4)
arrow(28, 36.3, 45, 34.3, rad=-0.12)
arrow(74, 36.3, 57, 34.3, rad=0.12)

box(51, 22.5, 50, 7.0,
    "Adaptive gated fusion\nper-sample imaging ⇄ clinical weight",
    fill=SURFACE, edge=VIOLET, dashed=True, fs=9.0)
ax.text(78.5, 22.5, "rung 5", fontsize=8, color=VIOLET, ha="left", va="center",
        fontweight="bold")
arrow(51, 27.9, 51, 26.1)

box(51, 13.5, 58, 7.4,
    "Memory-bank GNN\nGraphConv  ·  k = 8 neighbours  ·  memory → query",
    fill=SURFACE, edge=VIOLET, dashed=True, fs=9.0)
ax.text(82.5, 13.5, "rungs 4–5", fontsize=8, color=VIOLET, ha="left",
        va="center", fontweight="bold")
arrow(51, 19.0, 51, 17.3)

box(51, 5.5, 52, 6.4, "Linear head  →  P(MGMT methylated)",
    fill=BLUE, edge=BLUE, tcolor="#ffffff", fs=9.6, bold=True)
arrow(51, 9.8, 51, 8.8)

# ---- legend -------------------------------------------------------------
ax.add_patch(FancyBboxPatch((3, -3.6), 94, 4.2,
             boxstyle="round,pad=0.3,rounding_size=1.0", linewidth=0,
             facecolor="none", zorder=0))
ax.plot([7, 12], [-1.4, -1.4], color=INK2, lw=1.6, solid_capstyle="round")
ax.text(13, -1.4, "always active", fontsize=8.3, color=INK2, va="center")
ax.plot([31, 36], [-1.4, -1.4], color=VIOLET, lw=1.6, ls=(0, (4, 2.5)))
ax.text(37, -1.4, "ablation toggle", fontsize=8.3, color=INK2, va="center")
ax.text(60, -1.4,
        "Encoders are KAN or MLP (swapped across the ablation).",
        fontsize=8.3, color=MUTED, va="center", style="italic")
ax.set_ylim(-5, 100)

ax.text(3, 99.4, "Implemented pipeline", fontsize=13.5, fontweight="bold",
        color=INK, ha="left", va="top")

fig.subplots_adjust(left=0.02, right=0.98, top=0.985, bottom=0.01)
for ext in ("pdf", "png"):
    out = os.path.join(FIGDIR, f"fig_architecture_implemented.{ext}")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print("wrote", os.path.relpath(out, ROOT))
plt.close(fig)
