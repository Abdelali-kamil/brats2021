"""Tests for the 3D augmentation pipeline (brats_gbm/data/augment.py).

These verify *correctness* — label preservation, registration, background
handling, determinism — not segmentation performance, which needs the imaging
data and a GPU.

Run with:  python -m pytest tests/ -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brats_gbm.data import augment as A


def _synthetic(seed=0):
    """A [4,D,H,W] image with a zero background and a foreground blob, plus a
    registered [3,D,H,W] binary label inside that blob."""
    rng = np.random.default_rng(seed)
    img = np.zeros((4, 24, 24, 24), np.float32)
    img[:, 6:18, 6:18, 6:18] = rng.uniform(0.2, 1.0, size=(4, 12, 12, 12))
    label = np.zeros((3, 24, 24, 24), np.float32)
    label[:, 8:16, 8:16, 8:16] = 1.0
    return img, label


# ------------------------------------------------------------------ pipeline
def test_pipeline_preserves_shape_and_dtype():
    img, label = _synthetic()
    aug = A.Augmentor3D(seed=1)
    out_img, out_lbl = aug(img, label)
    assert out_img.shape == img.shape
    assert out_lbl.shape == label.shape
    assert out_img.dtype == np.float32
    assert out_lbl.dtype == np.float32


def test_pipeline_keeps_labels_binary():
    img, label = _synthetic()
    aug = A.Augmentor3D(seed=2)
    for _ in range(10):
        _, out_lbl = aug(img, label)
        uniq = np.unique(out_lbl)
        assert set(uniq.tolist()).issubset({0.0, 1.0}), uniq


def test_pipeline_outputs_are_finite():
    img, label = _synthetic()
    aug = A.Augmentor3D(seed=3)
    for _ in range(10):
        out_img, _ = aug(img, label)
        assert np.isfinite(out_img).all()


def test_pipeline_is_deterministic_under_seed():
    img, label = _synthetic()
    a1 = A.Augmentor3D(seed=123)
    a2 = A.Augmentor3D(seed=123)
    i1, l1 = a1(img, label)
    i2, l2 = a2(img, label)
    assert np.array_equal(i1, i2)
    assert np.array_equal(l1, l2)


def test_pipeline_actually_changes_input():
    """With every probability at 1 the output must differ from the input."""
    img, label = _synthetic()
    aug = A.Augmentor3D(seed=5, p_flip=1.0, p_affine=1.0, p_gamma=1.0,
                        p_brightness=1.0, p_noise=1.0, p_blur=1.0,
                        p_lowres=1.0, p_bias=1.0)
    out_img, _ = aug(img, label)
    assert not np.allclose(out_img, img)


# ------------------------------------------------------------------ flips
def test_flip_applies_identically_to_image_and_label():
    img, label = _synthetic()
    rng = np.random.default_rng(0)
    # p=1 flips all three axes; the label must match the image's flip exactly.
    fi, fl = A.random_flip(img.copy(), label.copy(), rng, p=1.0)
    assert np.array_equal(fi, np.ascontiguousarray(np.flip(img, axis=(1, 2, 3))))
    assert np.array_equal(fl, np.ascontiguousarray(np.flip(label, axis=(1, 2, 3))))


# ------------------------------------------------------------------ intensity
@pytest.mark.parametrize("fn", [
    A.random_gamma, A.random_brightness_contrast,
    A.random_gaussian_noise, A.random_bias_field,
])
def test_intensity_transforms_do_not_touch_label(fn):
    img, label = _synthetic()
    rng = np.random.default_rng(7)
    _, out_lbl = fn(img.copy(), label.copy(), rng, p=1.0)
    assert np.array_equal(out_lbl, label)


@pytest.mark.parametrize("fn", [
    A.random_gamma, A.random_brightness_contrast,
    A.random_gaussian_noise, A.random_bias_field,
])
def test_foreground_intensity_transforms_preserve_zero_background(fn):
    img, label = _synthetic()
    bg = img == 0
    rng = np.random.default_rng(9)
    out_img, _ = fn(img.copy(), label.copy(), rng, p=1.0)
    assert np.all(out_img[bg] == 0), "background leaked non-zero values"


def test_gamma_safe_on_negative_zscored_input():
    """z-score normalisation produces negative values; gamma must not NaN."""
    img = np.zeros((1, 16, 16, 16), np.float32)
    img[0, 4:12, 4:12, 4:12] = np.random.default_rng(0).normal(0, 1, (8, 8, 8))
    label = np.zeros((3, 16, 16, 16), np.float32)
    rng = np.random.default_rng(0)
    out, _ = A.random_gamma(img.copy(), label, rng, p=1.0)
    assert np.isfinite(out).all()


# ------------------------------------------------------------------ affine
def test_affine_identity_is_near_noop():
    """Zero rotation and unit scale must leave the volume essentially unchanged."""
    img, label = _synthetic()
    rng = np.random.default_rng(0)
    out_img, out_lbl = A.random_affine(
        img.copy(), label.copy(), rng, p=1.0,
        rotation_deg=0.0, scale_range=(1.0, 1.0))
    assert np.allclose(out_img, img, atol=1e-4)
    assert np.array_equal(out_lbl, label)


def test_affine_keeps_label_binary_and_registered():
    img, label = _synthetic()
    rng = np.random.default_rng(11)
    out_img, out_lbl = A.random_affine(img.copy(), label.copy(), rng, p=1.0)
    assert set(np.unique(out_lbl).tolist()).issubset({0.0, 1.0})
    # Label must stay inside the transformed foreground: wherever the label is
    # on, the image should be non-zero (both went through the same transform).
    on = out_lbl[0] > 0.5
    if on.any():
        assert (out_img[0][on] != 0).mean() > 0.9
