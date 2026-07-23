"""Prediction post-processing.

Three stages, applied in order:

  1. Hierarchy    ET subset of TC subset of WT, which the network does not
                  guarantee because the three channels are independent sigmoids.
  2. Components   Drop connected components below a per-region voxel count,
                  keeping the largest if that would empty the mask.
  3. ET policy    Decide what to do with very small enhancing-tumour
                  predictions (see below).

The ET policy is the consequential one. Dice for a region is all-or-nothing on
cases where the ground truth is empty: predict a handful of stray voxels on a
case with no true ET and a perfect 1.0 becomes 0.0. Glioblastoma cohorts do
contain such cases, so the policy is worth real Dice points.

The original code shipped a "rescue" policy that re-thresholded ET at 0.02
inside the TC mask whenever ET came out empty, forcing a non-empty prediction.
That is backwards: it can only ever convert correct empty predictions into
zeros. `min_volume` is the standard BraTS rule and does the opposite. Both are
implemented here so the choice can be made on validation data rather than
asserted — see scripts/evaluate_upenn.py, which sweeps them and reports which
one validation selected.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

# Per-region connected-component floors, carried over unchanged.
MIN_VOXELS = {"ET": 5, "TC": 20, "WT": 50}

# Candidate ET volume cutoffs swept on validation. 0 disables the rule.
ET_MIN_VOLUME_GRID = [0, 50, 100, 200, 300, 500]

ET_POLICIES = ("min_volume", "none", "rescue")


def remove_small_components(mask: np.ndarray, min_voxels: int) -> np.ndarray:
    """Drop components under `min_voxels`; keep the largest if all would go."""
    mask = mask.astype(bool)
    if not mask.any() or min_voxels <= 1:
        return mask

    labeled, n = ndimage.label(mask)
    if n == 0:
        return mask

    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    keep = np.flatnonzero(sizes >= min_voxels) + 1
    if keep.size == 0:
        keep = np.array([int(np.argmax(sizes)) + 1])
    return np.isin(labeled, keep)


def enforce_hierarchy(pred: np.ndarray) -> np.ndarray:
    """ET subset of TC subset of WT.

    Order matters. Constraining ET to TC first and only then shrinking TC into
    WT can leave ET voxels outside the final TC, because the second step
    removes TC voxels that the first step had already accepted ET into. Going
    outermost-inwards fixes each containing region before the region it
    contains, so one pass suffices.
    """
    pred = pred.astype(bool)
    pred[1] &= pred[2]
    pred[0] &= pred[1]
    return pred


def apply_thresholds(prob: np.ndarray, et_thr: float, tc_thr: float, wt_thr: float) -> np.ndarray:
    """Binarise the three probability channels. No post-processing here."""
    return np.stack([
        prob[0] > et_thr,
        prob[1] > tc_thr,
        prob[2] > wt_thr,
    ])


def apply_et_policy(
    pred: np.ndarray,
    prob: np.ndarray,
    policy: str = "min_volume",
    min_volume: int = 200,
    rescue_thr: float = 0.02,
) -> np.ndarray:
    """Resolve small or empty enhancing-tumour predictions.

    min_volume : zero ET when its total volume falls below `min_volume`
                 (standard BraTS practice; protects true-negative ET cases).
    none       : leave ET as thresholded.
    rescue     : if ET is empty, re-threshold at `rescue_thr` inside TC to
                 force a non-empty prediction (legacy behaviour, retained only
                 so the validation sweep can reject it on the evidence).
    """
    if policy not in ET_POLICIES:
        raise ValueError(f"unknown ET policy {policy!r}; expected one of {ET_POLICIES}")

    if policy == "min_volume":
        if 0 < pred[0].sum() < min_volume:
            pred[0] = np.zeros_like(pred[0])
    elif policy == "rescue":
        if pred[0].sum() == 0 and prob[0].max() > rescue_thr:
            et_in_tc = prob[0] * pred[1]
            if et_in_tc.max() > rescue_thr:
                pred[0] = et_in_tc > rescue_thr
    return pred


def postprocess(
    prob: np.ndarray,
    et_thr: float,
    tc_thr: float,
    wt_thr: float,
    et_policy: str = "min_volume",
    et_min_volume: int = 200,
) -> np.ndarray:
    """Threshold, clean and apply the ET policy. Returns a boolean [3,D,H,W]."""
    pred = apply_thresholds(prob, et_thr, tc_thr, wt_thr)
    pred = enforce_hierarchy(pred)

    for i, region in enumerate(("ET", "TC", "WT")):
        pred[i] = remove_small_components(pred[i], MIN_VOXELS[region])

    pred = enforce_hierarchy(pred)
    pred = apply_et_policy(pred, prob, policy=et_policy, min_volume=et_min_volume)
    return enforce_hierarchy(pred)
