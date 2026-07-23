"""Radiomic feature extraction from segmentation masks.

Shared by both cohorts so the feature definitions are identical and the two
classification analyses are comparable. Per region (ET, TC, WT) it computes
size, per-modality intensity statistics, and shape descriptors.

Intensity features are computed on z-scored volumes rather than raw ones,
because absolute MRI intensities are not comparable across scanners or
sequences. That makes the values relative to each subject's own brain, which
is the only defensible scale without a phantom.
"""
from __future__ import annotations

import numpy as np
from skimage import measure

# Minimum region size below which shape descriptors are meaningless.
MIN_VOXELS_FOR_FEATURES = 5


def znorm(volumes: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Z-score each modality over its non-zero voxels."""
    out = {}
    for key, v in volumes.items():
        v = v.astype(np.float32).copy()
        mask = v > 0
        if mask.any():
            v = (v - v[mask].mean()) / (v[mask].std() + 1e-8)
            v[~mask] = 0.0
        out[key] = v
    return out


def region_features(
    region_mask: np.ndarray,
    volumes: dict[str, np.ndarray],
    prefix: str,
    voxel_vol: float,
) -> dict[str, float]:
    """Size, intensity and shape descriptors for one region.

    `volumes` maps a modality key to its (already normalised) array; the keys
    become part of the feature names, so both cohorts must use the same ones.
    Regions below `MIN_VOXELS_FOR_FEATURES` return NaN for statistics that
    would be undefined, rather than a misleading zero.
    """
    feats: dict[str, float] = {}
    n_vox = int(region_mask.sum())
    feats[f"{prefix}_voxels"] = n_vox
    feats[f"{prefix}_volume_mm3"] = n_vox * voxel_vol

    if n_vox < MIN_VOXELS_FOR_FEATURES:
        for m in volumes:
            feats[f"{prefix}_{m}_mean"] = np.nan
            feats[f"{prefix}_{m}_std"] = np.nan
            feats[f"{prefix}_{m}_max"] = np.nan
        feats[f"{prefix}_surface_area"] = 0.0
        feats[f"{prefix}_sphericity"] = np.nan
        feats[f"{prefix}_n_components"] = 0
        return feats

    region_bool = region_mask.astype(bool)
    for m, vol in volumes.items():
        vals = vol[region_bool]
        feats[f"{prefix}_{m}_mean"] = float(vals.mean())
        feats[f"{prefix}_{m}_std"] = float(vals.std())
        feats[f"{prefix}_{m}_max"] = float(vals.max())

    try:
        verts, faces, _, _ = measure.marching_cubes(
            region_mask.astype(np.float32), level=0.5)
        area = float(measure.mesh_surface_area(verts, faces))
    except Exception:
        area = np.nan
    feats[f"{prefix}_surface_area"] = area

    volume = n_vox * voxel_vol
    if area and area > 0 and not np.isnan(area):
        feats[f"{prefix}_sphericity"] = float(
            (np.pi ** (1 / 3)) * (6 * volume) ** (2 / 3) / area)
    else:
        feats[f"{prefix}_sphericity"] = np.nan

    _, n_components = measure.label(region_mask, return_num=True)
    feats[f"{prefix}_n_components"] = int(n_components)
    return feats


def all_region_features(
    regions: dict[str, np.ndarray],
    volumes: dict[str, np.ndarray],
    voxel_vol: float,
) -> dict[str, float]:
    """Features for every region, plus whole-tumour composition ratios.

    The ratios matter more than they look: absolute tumour volume varies
    enormously between patients, so the *proportion* of the lesion that is
    enhancing or necrotic carries information that raw volumes bury.
    """
    feats: dict[str, float] = {}
    for name, mask in regions.items():
        feats.update(region_features(mask, volumes, name, voxel_vol))

    wt = feats.get("WT_voxels", 0)
    feats["ET_over_WT"] = feats.get("ET_voxels", 0) / wt if wt else np.nan
    feats["TC_over_WT"] = feats.get("TC_voxels", 0) / wt if wt else np.nan
    tc = feats.get("TC_voxels", 0)
    feats["ET_over_TC"] = feats.get("ET_voxels", 0) / tc if tc else np.nan
    return feats
