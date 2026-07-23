"""Sliding-window inference with Gaussian blending and flip test-time augmentation.

Patches overlap by half a window and are blended with a Gaussian weight so that
voxels near a patch edge, where the receptive field is truncated, contribute
less than voxels near its centre. TTA averages the sigmoid outputs over the
eight axis-flip combinations of the three spatial axes.

Neither is tuned on any data, so both are safe to apply everywhere.
"""
from __future__ import annotations

import numpy as np
import torch

PATCH_SIZE = (128, 128, 128)
STEP_SIZE = (64, 64, 64)

# All 8 combinations of flips over the three spatial axes of a [B,C,D,H,W]
# tensor. Axis indices are 2,3,4 because batch and channel come first.
TTA_FLIPS = [[], [2], [3], [4], [2, 3], [2, 4], [3, 4], [2, 3, 4]]

_GAUSS_CACHE: dict = {}


def gaussian_kernel_3d(shape: tuple, sigma_ratio: float = 0.125) -> np.ndarray:
    grids = [np.arange(s) - s // 2 for s in shape]
    mesh = np.meshgrid(*grids, indexing="ij")
    sigma = [s * sigma_ratio for s in shape]
    g = np.exp(-0.5 * sum((m / sg) ** 2 for m, sg in zip(mesh, sigma)))
    return (g / g.max()).astype(np.float32)


def _get_gauss(patch_shape: tuple, device) -> torch.Tensor:
    key = (patch_shape, str(device))
    if key not in _GAUSS_CACHE:
        _GAUSS_CACHE[key] = torch.from_numpy(gaussian_kernel_3d(patch_shape)).to(device)
    return _GAUSS_CACHE[key]


def _window_starts(extent: int, patch: int, step: int) -> list[int]:
    starts = list(range(0, extent - patch + 1, step))
    if not starts:
        return [0]
    if starts[-1] + patch < extent:
        starts.append(extent - patch)
    return starts


@torch.no_grad()
def sliding_window_predict(
    model,
    image_np: np.ndarray,
    device,
    patch_size: tuple = PATCH_SIZE,
    step_size: tuple = STEP_SIZE,
    use_tta: bool = True,
) -> np.ndarray:
    """Full-volume probability map, shape [3, D, H, W], values in [0, 1]."""
    _, D, H, W = image_np.shape
    pd_, ph_, pw_ = patch_size

    pad = [max(0, pd_ - D), max(0, ph_ - H), max(0, pw_ - W)]
    if any(pad):
        image_np = np.pad(
            image_np,
            [(0, 0), (0, pad[0]), (0, pad[1]), (0, pad[2])],
            mode="constant",
            constant_values=0,
        )

    _, D2, H2, W2 = image_np.shape
    out_sum = torch.zeros(3, D2, H2, W2, device=device)
    cnt_sum = torch.zeros(1, D2, H2, W2, device=device)
    gauss = _get_gauss(patch_size, device)
    image_t = torch.from_numpy(image_np).float().to(device).unsqueeze(0)

    def predict_patch(patch: torch.Tensor) -> torch.Tensor:
        if not use_tta:
            return torch.sigmoid(model(patch))
        acc = None
        for axes in TTA_FLIPS:
            x = torch.flip(patch, dims=axes) if axes else patch
            p = torch.sigmoid(model(x))
            if axes:
                p = torch.flip(p, dims=axes)
            acc = p if acc is None else acc + p
        return acc / len(TTA_FLIPS)

    for ds in _window_starts(D2, pd_, step_size[0]):
        for hs in _window_starts(H2, ph_, step_size[1]):
            for ws in _window_starts(W2, pw_, step_size[2]):
                patch = image_t[:, :, ds:ds + pd_, hs:hs + ph_, ws:ws + pw_]
                prob = predict_patch(patch)
                out_sum[:, ds:ds + pd_, hs:hs + ph_, ws:ws + pw_] += prob[0] * gauss
                cnt_sum[:, ds:ds + pd_, hs:hs + ph_, ws:ws + pw_] += gauss

    prob_full = (out_sum / (cnt_sum + 1e-8)).detach().cpu().numpy()
    return prob_full[:, :D, :H, :W]


def average_probability_maps(maps: list[np.ndarray]) -> np.ndarray:
    """Mean of several probability maps — the ensemble operator.

    Averaging probabilities rather than binarised masks keeps the ensemble
    differentiable with respect to the threshold, so the same threshold sweep
    applies to single models and ensembles alike.
    """
    if not maps:
        raise ValueError("no probability maps to average")
    acc = np.zeros_like(maps[0], dtype=np.float32)
    for m in maps:
        acc += m.astype(np.float32)
    return acc / len(maps)
