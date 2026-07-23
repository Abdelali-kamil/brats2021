"""Shared figure rendering for segmentation results.

Both cohorts render through this module so the BraTS2021 and UPenn-GBM figures
are guaranteed identical in palette, layout and typography rather than merely
similar.

Regions are nested (ET subset of TC subset of WT) and painted largest-first, so
each colour that stays visible is that region minus the one nested inside it —
which is why the legend names them as set differences.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Report palette: WT aqua, TC orange, ET blue.
COLORS = {
    "WT": (0.106, 0.686, 0.478),
    "TC": (0.922, 0.408, 0.204),
    "ET": (0.165, 0.471, 0.839),
}

LEGEND = [
    ("ET", "Enhancing tumour (ET)"),
    ("TC", "Necrotic core (TC \\ ET)"),
    ("WT", "Oedema (WT \\ TC)"),
]

OVERLAY_ALPHA = 0.55
FIGSIZE = (13.2, 4.9)
DPI = 140


def overlay(ax, base: np.ndarray, masks: dict[str, np.ndarray], title: str) -> None:
    """Grayscale slice with nested region overlays."""
    ax.imshow(np.rot90(base), cmap="gray", interpolation="nearest")
    for name in ("WT", "TC", "ET"):  # painted largest-first
        m = masks.get(name)
        if m is None or m.sum() == 0:
            continue
        rgba = np.zeros((*m.shape, 4), dtype=np.float32)
        rgba[..., :3] = COLORS[name]
        rgba[..., 3] = np.where(m > 0, OVERLAY_ALPHA, 0.0)
        ax.imshow(np.rot90(rgba), interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def best_slice(wt_gt: np.ndarray, wt_pred: np.ndarray) -> int:
    """Axial slice carrying the most tumour, so the figure shows something."""
    load = wt_gt.sum(axis=(0, 1)) if wt_gt.sum() else wt_pred.sum(axis=(0, 1))
    return int(np.argmax(load)) if load.sum() else wt_gt.shape[2] // 2


def render_case(
    out_path,
    base: np.ndarray,
    gt: dict[str, np.ndarray],
    pred: dict[str, np.ndarray],
    case_id: str,
    modality: str,
    z: int,
    dice: dict[str, float],
    suptitle: str,
) -> None:
    """Three-panel figure: image, expert annotation, prediction."""
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE)
    overlay(axes[0], base, {}, f"{case_id} — {modality}, slice {z}")
    overlay(axes[1], base, gt, "Expert annotation")
    overlay(
        axes[2], base, pred,
        f"Prediction — Dice ET {dice['ET']:.3f} / TC {dice['TC']:.3f} / WT {dice['WT']:.3f}",
    )

    handles = [mpatches.Patch(color=COLORS[k], label=lbl) for k, lbl in LEGEND]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle(suptitle, fontsize=11, y=0.99)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def pick_cases(df, id_col: str = "Patient_ID") -> list[tuple[str, str]]:
    """Best, median and worst case by mean Dice."""
    d = df.copy()
    d["_mean"] = d[["Dice_ET", "Dice_TC", "Dice_WT"]].mean(axis=1)
    d = d.sort_values("_mean", ascending=False).reset_index(drop=True)
    return [
        (d.iloc[0][id_col], "best case"),
        (d.iloc[len(d) // 2][id_col], "median case"),
        (d.iloc[-1][id_col], "worst case"),
    ]
