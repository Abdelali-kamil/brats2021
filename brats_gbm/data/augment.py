"""On-the-fly 3D augmentation for BraTS/UPenn segmentation training.

The BraTS base segmentor was previously trained with *no* augmentation and a
fixed centre crop (see ``get_datasets`` in ``brats.py``), which is exactly the
model that has to transfer zero-shot to UPenn-GBM. Intensity augmentation is the
lever that stops a network locking onto one cohort's intensity profile, so the
emphasis here is on intensity transforms (gamma, brightness/contrast, noise,
blur, low-resolution simulation, MRI bias field) alongside light spatial
transforms (flips, small affine).

Design notes
------------
* Volumes are ``[C, D, H, W]`` float32; labels are ``[K, D, H, W]`` float32
  binary masks. Spatial transforms are applied *identically* to image and
  label so they stay registered; intensity transforms touch the image only.
* Every intensity transform is restricted to the non-zero foreground so the
  zero background — which both normalisers produce — is preserved.
* Each ``__call__`` draws from a fresh ``numpy`` Generator seeded from OS
  entropy unless a seed is fixed. That sidesteps the classic DataLoader pitfall
  where every worker shares one RNG state and emits identical augmentations.
  Pass ``seed=`` for deterministic behaviour in tests.

Only ``numpy`` and ``scipy`` are used, both already project dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.ndimage import affine_transform, gaussian_filter, zoom


# --------------------------------------------------------------------------- #
# Individual transforms. Each takes (img, label, rng) and returns (img, label).
# img is mutated on a copy; label is only touched by spatial transforms.
# --------------------------------------------------------------------------- #
def _foreground(channel: np.ndarray) -> np.ndarray:
    """Boolean mask of the non-zero foreground for one channel."""
    return channel != 0


def random_flip(img, label, rng, p=0.5):
    """Flip each spatial axis independently with probability ``p``."""
    for ax in (1, 2, 3):  # D, H, W in [C, D, H, W]
        if rng.random() < p:
            img = np.flip(img, axis=ax)
            label = np.flip(label, axis=ax)
    # np.flip returns views; downstream code and torch need contiguous arrays.
    return np.ascontiguousarray(img), np.ascontiguousarray(label)


def random_gamma(img, label, rng, p=0.3, gamma_range=(0.7, 1.5)):
    """Per-channel gamma on the foreground, rescaled to [0, 1] first so it is
    valid regardless of the incoming normalisation (z-score can be negative)."""
    if rng.random() >= p:
        return img, label
    out = img.copy()
    for c in range(out.shape[0]):
        fg = _foreground(out[c])
        if not fg.any():
            continue
        vals = out[c][fg]
        lo, hi = vals.min(), vals.max()
        rng_span = hi - lo
        if rng_span <= 0:
            continue
        g = rng.uniform(*gamma_range)
        scaled = (vals - lo) / rng_span
        scaled = np.clip(scaled, 0.0, 1.0) ** g
        out[c][fg] = scaled * rng_span + lo
    return out, label


def random_brightness_contrast(img, label, rng, p=0.3,
                               scale_range=(0.85, 1.15), shift_range=(-0.1, 0.1)):
    """Per-channel multiplicative scale + additive shift on the foreground."""
    if rng.random() >= p:
        return img, label
    out = img.copy()
    for c in range(out.shape[0]):
        fg = _foreground(out[c])
        if not fg.any():
            continue
        scale = rng.uniform(*scale_range)
        shift = rng.uniform(*shift_range)
        out[c][fg] = out[c][fg] * scale + shift
    return out, label


def random_gaussian_noise(img, label, rng, p=0.2, sigma_range=(0.0, 0.08)):
    """Additive Gaussian noise on the foreground."""
    if rng.random() >= p:
        return img, label
    out = img.copy()
    for c in range(out.shape[0]):
        fg = _foreground(out[c])
        if not fg.any():
            continue
        sigma = rng.uniform(*sigma_range)
        if sigma <= 0:
            continue
        out[c][fg] = out[c][fg] + rng.normal(0.0, sigma, size=fg.sum()).astype(out.dtype)
    return out, label


def random_gaussian_blur(img, label, rng, p=0.2, sigma_range=(0.5, 1.5)):
    """Per-channel isotropic Gaussian blur."""
    if rng.random() >= p:
        return img, label
    out = img.copy()
    for c in range(out.shape[0]):
        sigma = rng.uniform(*sigma_range)
        out[c] = gaussian_filter(out[c], sigma=sigma)
    return out, label


def random_low_resolution(img, label, rng, p=0.25, zoom_range=(0.5, 1.0)):
    """Simulate an acquisition at lower through-plane resolution: downsample by a
    random factor with linear interpolation, then restore the original size with
    nearest-neighbour. nnU-Net's sim-to-real transform for resolution shift."""
    if rng.random() >= p:
        return img, label
    factor = rng.uniform(*zoom_range)
    if factor >= 1.0:
        return img, label
    out = np.empty_like(img)
    for c in range(img.shape[0]):
        orig_shape = img[c].shape
        down = zoom(img[c], factor, order=1)
        if any(s == 0 for s in down.shape):
            out[c] = img[c]
            continue
        back = zoom(down, np.array(orig_shape) / np.array(down.shape), order=0)
        # zoom can be off by a voxel; crop/pad back to the exact shape.
        out[c] = _fit_to_shape(back, orig_shape)
    return out, label


def _fit_to_shape(arr: np.ndarray, shape) -> np.ndarray:
    slices = tuple(slice(0, min(a, s)) for a, s in zip(arr.shape, shape))
    fitted = np.zeros(shape, dtype=arr.dtype)
    fitted[tuple(slice(0, s.stop) for s in slices)] = arr[slices]
    return fitted


def random_bias_field(img, label, rng, p=0.3, coeff_range=(-0.3, 0.3), order=3):
    """Multiply by a smooth low-frequency field — the dominant MRI intensity
    non-uniformity artefact, and one of the most cohort-specific. A small random
    coefficient grid is upsampled to the volume and exponentiated."""
    if rng.random() >= p:
        return img, label
    out = img.copy()
    grid = rng.uniform(coeff_range[0], coeff_range[1], size=(order, order, order))
    vol_shape = out.shape[1:]
    field = zoom(grid, np.array(vol_shape) / np.array(grid.shape), order=3)
    field = _fit_to_shape(field, vol_shape)
    field = np.exp(field).astype(out.dtype)
    for c in range(out.shape[0]):
        fg = _foreground(out[c])
        out[c][fg] = out[c][fg] * field[fg]
    return out, label


def random_affine(img, label, rng, p=0.2,
                  rotation_deg=10.0, scale_range=(0.85, 1.15)):
    """Small random rotation (about each axis) + isotropic scale, applied to the
    image with linear interpolation and to the label with nearest-neighbour so
    masks stay binary and registered."""
    if rng.random() >= p:
        return img, label

    angles = np.deg2rad(rng.uniform(-rotation_deg, rotation_deg, size=3))
    scale = rng.uniform(*scale_range)
    matrix = _rotation_matrix(angles) * scale

    vol_shape = np.array(img.shape[1:], dtype=np.float64)
    centre = (vol_shape - 1) / 2.0
    offset = centre - matrix @ centre

    out_img = np.empty_like(img)
    for c in range(img.shape[0]):
        out_img[c] = affine_transform(
            img[c], matrix, offset=offset, order=1, mode="constant", cval=0.0
        )
    out_lbl = np.empty_like(label)
    for k in range(label.shape[0]):
        out_lbl[k] = affine_transform(
            label[k], matrix, offset=offset, order=0, mode="constant", cval=0.0
        )
    return out_img, out_lbl


def _rotation_matrix(angles) -> np.ndarray:
    ax, ay, az = angles
    rx = np.array([[1, 0, 0],
                   [0, np.cos(ax), -np.sin(ax)],
                   [0, np.sin(ax), np.cos(ax)]])
    ry = np.array([[np.cos(ay), 0, np.sin(ay)],
                   [0, 1, 0],
                   [-np.sin(ay), 0, np.cos(ay)]])
    rz = np.array([[np.cos(az), -np.sin(az), 0],
                   [np.sin(az), np.cos(az), 0],
                   [0, 0, 1]])
    return rz @ ry @ rx


# --------------------------------------------------------------------------- #
# Composed augmentor
# --------------------------------------------------------------------------- #
@dataclass
class Augmentor3D:
    """Compose the transforms above. Probabilities are the per-call chance each
    transform fires; the defaults favour intensity transforms, which is what
    buys cross-cohort generalisation.

    Set ``seed`` for reproducible output (tests); leave it ``None`` in training
    so every sample and every worker draws independently.
    """

    p_flip: float = 0.5
    p_affine: float = 0.2
    p_gamma: float = 0.3
    p_brightness: float = 0.3
    p_noise: float = 0.2
    p_blur: float = 0.2
    p_lowres: float = 0.25
    p_bias: float = 0.3
    rotation_deg: float = 10.0
    scale_range: tuple = (0.85, 1.15)
    seed: Optional[int] = None
    _rng: Optional[np.random.Generator] = field(default=None, repr=False)

    def __post_init__(self):
        if self.seed is not None:
            self._rng = np.random.default_rng(self.seed)

    def _get_rng(self) -> np.random.Generator:
        # Fixed seed -> reuse the stream (deterministic); otherwise draw a fresh
        # generator per call so DataLoader workers never share RNG state.
        return self._rng if self._rng is not None else np.random.default_rng()

    def __call__(self, img: np.ndarray, label: np.ndarray):
        img = np.ascontiguousarray(img.astype(np.float32))
        label = np.ascontiguousarray(label.astype(np.float32))
        rng = self._get_rng()

        # Spatial first (cheap flips, then affine), then intensity.
        img, label = random_flip(img, label, rng, self.p_flip)
        img, label = random_affine(img, label, rng, self.p_affine,
                                   self.rotation_deg, self.scale_range)
        img, label = random_gamma(img, label, rng, self.p_gamma)
        img, label = random_brightness_contrast(img, label, rng, self.p_brightness)
        img, label = random_bias_field(img, label, rng, self.p_bias)
        img, label = random_gaussian_blur(img, label, rng, self.p_blur)
        img, label = random_low_resolution(img, label, rng, self.p_lowres)
        img, label = random_gaussian_noise(img, label, rng, self.p_noise)

        return (np.ascontiguousarray(img.astype(np.float32)),
                np.ascontiguousarray(label.astype(np.float32)))
