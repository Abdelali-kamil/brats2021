"""Tests for the metric, post-processing and split logic.

Run with:  python -m pytest tests/ -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brats_gbm.eval import postprocess as pp
from brats_gbm.eval.metrics import dice_score, hausdorff95, score_case
from brats_gbm.eval.stats import bootstrap_ci, paired_diff_ci
from brats_gbm.splits import assert_disjoint


# --------------------------------------------------------------- metrics
def test_dice_identical_masks_is_one():
    m = np.zeros((10, 10, 10), bool)
    m[2:6, 2:6, 2:6] = True
    assert dice_score(m, m) == pytest.approx(1.0)


def test_dice_disjoint_masks_is_zero():
    a = np.zeros((10, 10, 10), bool); a[0:3, 0:3, 0:3] = True
    b = np.zeros((10, 10, 10), bool); b[7:10, 7:10, 7:10] = True
    assert dice_score(a, b) == pytest.approx(0.0)


def test_dice_both_empty_is_one():
    """The BraTS convention: correctly predicting 'no tumour here' scores 1."""
    z = np.zeros((5, 5, 5), bool)
    assert dice_score(z, z) == pytest.approx(1.0)


def test_dice_one_empty_is_zero():
    a = np.zeros((5, 5, 5), bool); a[1, 1, 1] = True
    assert dice_score(a, np.zeros((5, 5, 5), bool)) == pytest.approx(0.0)


def test_dice_half_overlap():
    a = np.zeros((10, 10, 10), bool); a[0:4, :, :] = True
    b = np.zeros((10, 10, 10), bool); b[2:6, :, :] = True
    # |A|=|B|=400, intersection=200 -> 2*200/800 = 0.5
    assert dice_score(a, b) == pytest.approx(0.5)


def test_hd95_identical_is_zero():
    m = np.zeros((20, 20, 20), bool); m[5:15, 5:15, 5:15] = True
    assert hausdorff95(m, m) == pytest.approx(0.0, abs=1e-9)


def test_hd95_undefined_when_one_mask_empty():
    a = np.zeros((10, 10, 10), bool); a[1, 1, 1] = True
    assert np.isnan(hausdorff95(a, np.zeros((10, 10, 10), bool)))


def test_hd95_both_empty_is_zero():
    z = np.zeros((10, 10, 10), bool)
    assert hausdorff95(z, z) == pytest.approx(0.0)


def test_hd95_uses_surface_not_interior():
    """A shifted cube's HD95 should reflect the shift, not be diluted to ~0.

    Computing over all voxels rather than surface voxels puts most distances at
    zero (interior voxels sit inside the other mask), which drags the 95th
    percentile down and flatters the metric. This guards that regression.
    """
    a = np.zeros((40, 40, 40), bool); a[10:30, 10:30, 10:30] = True
    b = np.zeros((40, 40, 40), bool); b[14:34, 10:30, 10:30] = True  # shift 4
    assert hausdorff95(a, b) > 2.0


def test_hd95_respects_spacing():
    a = np.zeros((30, 30, 30), bool); a[10:20, 10:20, 10:20] = True
    b = np.zeros((30, 30, 30), bool); b[13:23, 10:20, 10:20] = True
    iso = hausdorff95(a, b, spacing=(1.0, 1.0, 1.0))
    stretched = hausdorff95(a, b, spacing=(2.0, 1.0, 1.0))
    assert stretched > iso


def test_score_case_returns_all_regions():
    pred = np.zeros((3, 12, 12, 12), bool)
    gt = np.zeros((3, 12, 12, 12), bool)
    pred[:, 2:6, 2:6, 2:6] = True
    gt[:, 2:6, 2:6, 2:6] = True
    out = score_case(pred, gt)
    for r in ("ET", "TC", "WT"):
        assert out[f"Dice_{r}"] == pytest.approx(1.0)
    assert out["Dice_Mean"] == pytest.approx(1.0)


# --------------------------------------------------- post-processing
def test_hierarchy_is_enforced():
    """ET must end up inside TC, and TC inside WT."""
    pred = np.zeros((3, 8, 8, 8), bool)
    pred[0, 0:4, 0:4, 0:4] = True   # ET somewhere
    pred[1, 2:6, 2:6, 2:6] = True   # TC only partly overlapping
    pred[2, 4:8, 4:8, 4:8] = True   # WT elsewhere again
    out = pp.enforce_hierarchy(pred.copy())
    assert not (out[0] & ~out[1]).any()
    assert not (out[1] & ~out[2]).any()


def test_small_components_removed():
    m = np.zeros((20, 20, 20), bool)
    m[2:8, 2:8, 2:8] = True    # 216 voxels
    m[15, 15, 15] = True       # 1 voxel speck
    out = pp.remove_small_components(m, min_voxels=50)
    assert out[15, 15, 15] == False
    assert out[3, 3, 3] == True


def test_largest_component_kept_when_all_below_floor():
    """Never empty a mask entirely just because every piece is small."""
    m = np.zeros((20, 20, 20), bool)
    m[1:3, 1:3, 1:3] = True   # 8 voxels
    m[10, 10, 10] = True      # 1 voxel
    out = pp.remove_small_components(m, min_voxels=1000)
    assert out.sum() == 8


def test_et_min_volume_zeroes_small_et():
    pred = np.zeros((3, 20, 20, 20), bool)
    pred[0, 0:3, 0:3, 0:3] = True    # 27 voxels of ET
    pred[1] = True
    pred[2] = True
    prob = np.zeros((3, 20, 20, 20), np.float32)
    out = pp.apply_et_policy(pred.copy(), prob, policy="min_volume", min_volume=200)
    assert out[0].sum() == 0


def test_et_min_volume_keeps_large_et():
    pred = np.zeros((3, 20, 20, 20), bool)
    pred[0, 0:10, 0:10, 0:10] = True   # 1000 voxels
    prob = np.zeros((3, 20, 20, 20), np.float32)
    out = pp.apply_et_policy(pred.copy(), prob, policy="min_volume", min_volume=200)
    assert out[0].sum() == 1000


def test_et_policy_none_leaves_et_untouched():
    pred = np.zeros((3, 20, 20, 20), bool)
    pred[0, 0:2, 0:2, 0:2] = True
    prob = np.zeros((3, 20, 20, 20), np.float32)
    out = pp.apply_et_policy(pred.copy(), prob, policy="none")
    assert out[0].sum() == 8


def test_et_rescue_can_only_add_voxels():
    """The legacy policy forces a non-empty ET; documents why it is harmful."""
    pred = np.zeros((3, 10, 10, 10), bool)
    pred[1] = True                       # TC everywhere
    prob = np.zeros((3, 10, 10, 10), np.float32)
    prob[0, 5, 5, 5] = 0.9               # one confident ET voxel
    out = pp.apply_et_policy(pred.copy(), prob, policy="rescue")
    assert out[0].sum() > 0


def test_unknown_et_policy_raises():
    pred = np.zeros((3, 4, 4, 4), bool)
    prob = np.zeros((3, 4, 4, 4), np.float32)
    with pytest.raises(ValueError):
        pp.apply_et_policy(pred, prob, policy="nonsense")


def test_postprocess_output_is_boolean_and_hierarchical():
    rng = np.random.default_rng(0)
    prob = rng.random((3, 16, 16, 16)).astype(np.float32)
    out = pp.postprocess(prob, 0.3, 0.4, 0.5)
    assert out.dtype == bool
    assert not (out[0] & ~out[1]).any()
    assert not (out[1] & ~out[2]).any()


# --------------------------------------------------------------- stats
def test_bootstrap_ci_brackets_the_mean():
    vals = [0.8, 0.85, 0.9, 0.75, 0.82]
    p, lo, hi = bootstrap_ci(vals)
    assert p == pytest.approx(np.mean(vals))
    assert lo <= p <= hi


def test_bootstrap_ci_is_deterministic():
    vals = list(np.random.default_rng(1).random(30))
    assert bootstrap_ci(vals) == bootstrap_ci(vals)


def test_bootstrap_ci_ignores_nan():
    p, lo, hi = bootstrap_ci([0.5, np.nan, 0.7])
    assert p == pytest.approx(0.6)


def test_bootstrap_ci_all_nan_returns_nan():
    p, lo, hi = bootstrap_ci([np.nan, np.nan])
    assert np.isnan(p) and np.isnan(lo) and np.isnan(hi)


def test_paired_diff_detects_consistent_improvement():
    """A uniform +0.1 on every case must give an interval excluding zero."""
    base = np.linspace(0.4, 0.9, 40)
    better = base + 0.1
    d, lo, hi = paired_diff_ci(better, base)
    assert d == pytest.approx(0.1)
    assert lo > 0


def test_paired_diff_on_identical_inputs_is_zero():
    v = np.linspace(0.3, 0.8, 25)
    d, lo, hi = paired_diff_ci(v, v)
    assert d == pytest.approx(0.0)
    assert lo == pytest.approx(0.0) and hi == pytest.approx(0.0)


# --------------------------------------------------------------- splits
def test_assert_disjoint_passes_on_disjoint_sets():
    assert_disjoint(a=["x", "y"], b=["z"], c=["w"])


def test_assert_disjoint_raises_on_overlap():
    with pytest.raises(AssertionError, match="Leakage"):
        assert_disjoint(train=["s1", "s2"], test=["s2", "s3"])
