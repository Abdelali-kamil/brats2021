"""Dice and HD95 for binary 3D masks.

Empty-mask convention follows the BraTS challenge: when ground truth and
prediction are both empty the case scores Dice 1.0 and HD95 0.0, and when only
one is empty it scores Dice 0.0 with HD95 undefined (NaN). Enhancing tumour is
where this bites — a handful of stray predicted voxels on a case with no true
ET turns a perfect score into a zero, which is what `min_et_volume` in
postprocess.py exists to prevent.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if not pred.any() and not gt.any():
        return 1.0
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(pred, gt).sum() / denom)


def _surface_points(mask: np.ndarray) -> np.ndarray:
    """Voxels on the mask boundary (any 6-neighbour outside the mask)."""
    from scipy import ndimage

    if not mask.any():
        return np.empty((0, 3))
    eroded = ndimage.binary_erosion(mask, structure=ndimage.generate_binary_structure(3, 1))
    return np.argwhere(mask & ~eroded)


def hausdorff95(
    pred: np.ndarray,
    gt: np.ndarray,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> float:
    """Symmetric 95th-percentile Hausdorff distance in millimetres.

    NaN when exactly one mask is empty (the distance is undefined, not large),
    0.0 when both are empty.
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if not pred.any() and not gt.any():
        return 0.0
    if not pred.any() or not gt.any():
        return float("nan")

    sp = np.asarray(spacing, dtype=float)
    a = _surface_points(pred) * sp
    b = _surface_points(gt) * sp
    if len(a) == 0 or len(b) == 0:
        return float("nan")

    d_ab = cKDTree(b).query(a)[0]
    d_ba = cKDTree(a).query(b)[0]
    return float(np.percentile(np.concatenate([d_ab, d_ba]), 95))


def score_case(
    pred: np.ndarray,
    gt: np.ndarray,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> dict[str, float]:
    """Per-region Dice and HD95 for a stacked [ET, TC, WT] prediction."""
    out: dict[str, float] = {}
    for i, region in enumerate(("ET", "TC", "WT")):
        out[f"Dice_{region}"] = dice_score(pred[i], gt[i])
        out[f"HD95_{region}"] = hausdorff95(pred[i], gt[i], spacing)
    out["Dice_Mean"] = float(np.mean([out[f"Dice_{r}"] for r in ("ET", "TC", "WT")]))
    return out
