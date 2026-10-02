"""Strong, nnU-Net-inspired augmentation for Run D (PROTOCOL_v2 Amendment 15).

Applied to one (C, D, H, W) training patch after the axis-aligned flips / rot90.
Spatial: with p=0.25 one joint random rotation (+-30 deg about each axis) and isotropic
scaling (0.7-1.4); the image is resampled linearly, the labels by nearest neighbour with the
SAME sampling grid for every region channel, so ET <= TC <= WT nesting is preserved exactly.
Intensity (per channel, image only): Gaussian noise, Gaussian blur, multiplicative brightness,
range-preserving contrast, simulated low resolution, range-preserving (inverted) gamma.
Background is 0 after z-scoring over non-zero voxels, so out-of-volume samples use cval=0.
"""
import math
import random

import numpy as np
from scipy import ndimage


def _rotation(ax, ay, az):
    cx, sx, cy, sy, cz, sz = math.cos(ax), math.sin(ax), math.cos(ay), math.sin(ay), math.cos(az), math.sin(az)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def spatial(img, lbl, max_angle=math.radians(30), scale=(0.7, 1.4)):
    m = _rotation(*(random.uniform(-max_angle, max_angle) for _ in range(3))) / random.uniform(*scale)
    c = (np.array(img.shape[1:]) - 1) / 2.0
    off = c - m @ c
    img = np.stack([ndimage.affine_transform(ch, m, offset=off, order=1, mode="constant", cval=0.0) for ch in img])
    lbl = np.stack([ndimage.affine_transform(ch.astype(np.float32), m, offset=off, order=0, mode="constant", cval=0.0) > 0.5
                    for ch in lbl])
    return img.astype(np.float32), lbl


def _low_res(ch, zoom_range=(0.5, 1.0)):
    f = random.uniform(*zoom_range)
    small = ndimage.zoom(ch, f, order=0)
    up = ndimage.zoom(small, [t / s for t, s in zip(ch.shape, small.shape)], order=3)
    out = np.zeros_like(ch)
    sl = tuple(slice(0, min(a, b)) for a, b in zip(ch.shape, up.shape))
    out[sl] = up[sl]
    return out


def _gamma(ch, gamma_range=(0.7, 1.5), invert=False):
    if invert: ch = -ch
    mn, sd = ch.mean(), ch.std()
    lo, hi = ch.min(), ch.max()
    if hi - lo > 1e-8:
        ch = ((ch - lo) / (hi - lo)) ** random.uniform(*gamma_range) * (hi - lo) + lo
        if ch.std() > 1e-8: ch = (ch - ch.mean()) / ch.std() * sd + mn   # retain statistics
    return -ch if invert else ch


def intensity(img):
    for c in range(img.shape[0]):
        ch = img[c]
        if random.random() < 0.15:
            ch = ch + np.random.normal(0.0, math.sqrt(random.uniform(0.0, 0.1)), ch.shape).astype(np.float32)
        if random.random() < 0.2:
            ch = ndimage.gaussian_filter(ch, random.uniform(0.5, 1.0))
        if random.random() < 0.15:
            ch = ch * random.uniform(0.75, 1.25)
        if random.random() < 0.15:
            lo, hi, mn = ch.min(), ch.max(), ch.mean()
            ch = np.clip((ch - mn) * random.uniform(0.75, 1.25) + mn, lo, hi)
        if random.random() < 0.25:
            ch = _low_res(ch)
        if random.random() < 0.3:
            ch = _gamma(ch, invert=random.random() < 0.1 / 0.3)   # inverted gamma overall p = 0.1
        img[c] = ch
    return img


def strong_augment(img, lbl):
    """img: float32 (C, D, H, W) contiguous; lbl: bool (3, D, H, W). Returns augmented copies."""
    if random.random() < 0.25:
        img, lbl = spatial(img, lbl)
    img = intensity(np.ascontiguousarray(img, dtype=np.float32))
    return img.astype(np.float32), np.ascontiguousarray(lbl)
