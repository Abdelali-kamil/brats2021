"""Augmentation and split tests.

The augmentation added in September 2026 transforms image and label together.
If a transform were ever applied to one and not the other, training would still
run, the loss would still fall, and the resulting model would be quietly wrong —
nothing else in the pipeline would notice. These tests are the thing that
notices.
"""
import numpy as np
import pytest

from brats_gbm.data.brats import Brats
from brats_gbm.splits import BRATS_VAL_FRAC_OF_HELDOUT, assert_disjoint


@pytest.fixture
def aug():
    """A Brats instance used only for its augment method (no data on disk)."""
    return Brats.__new__(Brats)


def _synthetic(seed=0):
    """An image whose channels differ, and a label marking one corner blob."""
    rng = np.random.default_rng(seed)
    img = rng.random((4, 16, 16, 16)).astype(np.float32)
    lbl = np.zeros((3, 16, 16, 16), dtype=np.float32)
    lbl[:, 2:6, 3:7, 4:8] = 1.0  # asymmetric, so any transform is detectable
    return img, lbl


def test_label_voxel_count_is_preserved(aug):
    """Flips and 90-degree rotations move voxels; they must not create or lose."""
    for seed in range(25):
        np.random.seed(seed)
        import random as _r
        _r.seed(seed)
        img, lbl = _synthetic(seed)
        before = lbl.sum()
        _, out = aug.augment(img, lbl)
        assert out.sum() == pytest.approx(before), "augmentation changed label volume"


def test_label_stays_binary(aug):
    """No interpolation is used, so labels must remain exactly 0 or 1."""
    for seed in range(25):
        np.random.seed(seed)
        import random as _r
        _r.seed(seed)
        img, lbl = _synthetic(seed)
        _, out = aug.augment(img, lbl)
        assert set(np.unique(out)).issubset({0.0, 1.0}), "label was interpolated"


def test_image_and_label_receive_the_same_spatial_transform(aug):
    """The core property: image and label must move together.

    A marker is written into the image at exactly the label's location. After
    augmentation the marker must still coincide with the label, whatever spatial
    transform was drawn.
    """
    for seed in range(40):
        np.random.seed(seed)
        import random as _r
        _r.seed(seed)
        img, lbl = _synthetic(seed)
        img[:] = 0.0
        img[0][lbl[0] > 0] = 1.0  # channel 0 marks the label region

        out_img, out_lbl = aug.augment(img, lbl)

        marked = out_img[0] > 0.5
        labelled = out_lbl[0] > 0.5
        assert np.array_equal(marked, labelled), (
            f"seed {seed}: image and label diverged under augmentation"
        )


def test_shape_is_unchanged(aug):
    for seed in range(10):
        np.random.seed(seed)
        import random as _r
        _r.seed(seed)
        img, lbl = _synthetic(seed)
        out_img, out_lbl = aug.augment(img, lbl)
        assert out_img.shape == img.shape
        assert out_lbl.shape == lbl.shape


def test_output_is_contiguous_and_float32(aug):
    """flip/rot90 return views; the loader downstream assumes a real array."""
    np.random.seed(0)
    import random as _r
    _r.seed(0)
    img, lbl = _synthetic(0)
    out_img, _ = aug.augment(img, lbl)
    assert out_img.flags["C_CONTIGUOUS"]
    assert out_img.dtype == np.float32


def test_augmentation_actually_changes_something(aug):
    """A no-op augmentation would pass every test above."""
    changed = 0
    for seed in range(20):
        np.random.seed(seed)
        import random as _r
        _r.seed(seed)
        img, lbl = _synthetic(seed)
        out_img, _ = aug.augment(img.copy(), lbl.copy())
        if not np.allclose(out_img, img):
            changed += 1
    assert changed >= 18, f"augmentation was a near no-op ({changed}/20 changed)"


def test_three_way_split_is_disjoint_and_covers_the_heldout():
    """train/val/test must partition, not overlap.

    Exercised on synthetic ids so the test needs no data on disk.
    """
    train = {f"BraTS2021_{i:05d}" for i in range(1000)}
    heldout = sorted(f"BraTS2021_{i:05d}" for i in range(1000, 1251))

    idx = np.random.default_rng(42 + 7).permutation(len(heldout))
    n_val = int(round(len(heldout) * BRATS_VAL_FRAC_OF_HELDOUT))
    val = {heldout[i] for i in idx[:n_val]}
    test = {heldout[i] for i in idx[n_val:]}

    assert_disjoint(train=train, val=val, test=test)
    assert val | test == set(heldout), "held-out cases went missing"
    assert len(val) + len(test) == len(heldout)
    assert len(test) > 100, "test partition too small to report a number from"
