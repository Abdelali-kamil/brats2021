#!/usr/bin/env python3
"""
Build a per-patient feature table for IDH1 classification.

For each of the 629 usable UPenn-GBM subjects (all except sub-123, which is
missing T2w):
  - If a ground-truth segmentation mask exists (147 subjects), use it.
  - Otherwise, run the trained segmentor (upenn_v2_best.pth) with a single
    forward pass per sliding-window patch (no TTA, for speed) and threshold
    with the same defaults evaluate_upenn.py falls back to.

From the resulting ET/TC/WT binary masks + the four MRI modalities, extract
per-region intensity and shape features, plus age/gender from the clinical
CSV. Output: upenn_radiomic_features.csv (629 rows, includes IDH1 label where
known and NaN where not).
"""
import os
import re
import sys
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
import torch
from skimage import measure

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brats_gbm.model import WaveletUNetPlusPlus  # noqa: E402
from brats_gbm.eval.inference import (  # noqa: E402
    PATCH_SIZE as SPATIAL_SIZE,
    STEP_SIZE as SLIDE_STEP,
    sliding_window_predict,
)
from brats_gbm.eval.postprocess import postprocess  # noqa: E402
from brats_gbm.data.upenn import ET_LABELS  # noqa: E402

UPENN_DIR = ROOT
NIFTI_DIR = UPENN_DIR / "upenn_nifti"
CLINICAL_CSV = UPENN_DIR / "upenn_data" / "UPENN-GBM_clinical_info_v2.1.csv"
# Use the UPenn-adapted segmentor (corrected ET labels). The BraTS checkpoint
# scores only ~0.15 Dice here because UPENN-GBM images are not skull-stripped,
# so it would produce unusable masks for the subjects lacking ground truth.
CHECKPOINT = ROOT / "checkpoints" / "upenn_v3_best.pth"
OUT_CSV = ROOT / "results" / "classification" / "radiomic_features.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Match the thresholds the fine-tuning run validated against, so the masks here
# correspond to the reported validation Dice.
ET_THR, TC_THR, WT_THR = 0.30, 0.40, 0.35

# Post-processing for predicted masks. Read from the validation-selected
# configuration written by scripts/evaluate_upenn.py so feature masks are built
# the same way the reported segmentation masks are, rather than by a second,
# independently chosen rule.
ET_POLICY, ET_MIN_VOLUME = "min_volume", 200
_selection = ROOT / "results" / "upenn" / "validation_selection.csv"
if _selection.exists():
    import pandas as _pd
    _sel = _pd.read_csv(_selection)
    _row = _sel[_sel.setup == "upenn_v3_best"].sort_values(
        "val_mean_dice", ascending=False)
    if len(_row):
        ET_POLICY = str(_row.iloc[0]["et_policy"])
        ET_MIN_VOLUME = int(_row.iloc[0]["et_min_volume"])

MODALITIES = {"flair": "FLAIR", "t1": "T1w", "t1ce": "ce-gd_T1w", "t2": "T2w"}


def sub_to_num(sub_id):
    m = re.search(r"sub-(\d+)", str(sub_id))
    return int(m.group(1)) if m else None


def list_usable_subjects():
    files = os.listdir(NIFTI_DIR)
    subs = sorted(set(m.group(1) for f in files if (m := re.match(r"^(sub-\d+)_", f))))
    usable = []
    for s in subs:
        if all((NIFTI_DIR / f"{s}_{suffix}.nii.gz").exists() for suffix in MODALITIES.values()):
            usable.append(s)
    return usable


def load_modalities(sub_id):
    vols = {}
    for key, suffix in MODALITIES.items():
        p = NIFTI_DIR / f"{sub_id}_{suffix}.nii.gz"
        vols[key] = nib.load(str(p)).get_fdata().astype(np.float32)
    return vols


def normalize(vols):
    img = np.stack([vols["flair"], vols["t1"], vols["t1ce"], vols["t2"]], axis=0)
    for c in range(4):
        mask = img[c] > 0
        if mask.sum() > 0:
            img[c] = (img[c] - img[c][mask].mean()) / (img[c][mask].std() + 1e-8)
            img[c][~mask] = 0.0
    return img


def region_features(region_mask, vols, prefix, voxel_vol):
    feats = {}
    n_vox = int(region_mask.sum())
    feats[f"{prefix}_voxels"] = n_vox
    feats[f"{prefix}_volume_mm3"] = n_vox * voxel_vol

    if n_vox < 5:
        for m in MODALITIES:
            feats[f"{prefix}_{m}_mean"] = np.nan
            feats[f"{prefix}_{m}_std"] = np.nan
            feats[f"{prefix}_{m}_max"] = np.nan
        feats[f"{prefix}_surface_area"] = 0.0
        feats[f"{prefix}_sphericity"] = np.nan
        feats[f"{prefix}_n_components"] = 0
        return feats

    region_bool = region_mask.astype(bool)
    for m in MODALITIES:
        vals = vols[m][region_bool]
        feats[f"{prefix}_{m}_mean"] = float(vals.mean())
        feats[f"{prefix}_{m}_std"] = float(vals.std())
        feats[f"{prefix}_{m}_max"] = float(vals.max())

    try:
        verts, faces, _, _ = measure.marching_cubes(region_mask.astype(np.float32), level=0.5)
        area = measure.mesh_surface_area(verts, faces)
    except Exception:
        area = np.nan
    feats[f"{prefix}_surface_area"] = area

    volume = n_vox * voxel_vol
    if area and area > 0 and not np.isnan(area):
        feats[f"{prefix}_sphericity"] = float((np.pi ** (1 / 3)) * (6 * volume) ** (2 / 3) / area)
    else:
        feats[f"{prefix}_sphericity"] = np.nan

    labeled, n = measure.label(region_mask, return_num=True)
    feats[f"{prefix}_n_components"] = int(n)

    return feats


def map_idh1(val):
    if pd.isna(val):
        return np.nan
    v = str(val).strip().lower()
    if v == "mutated":
        return 1
    if v == "wildtype":
        return 0
    return np.nan  # NOS/NEC / Not Available -> unknown


def main():
    subjects = list_usable_subjects()
    print(f"Usable subjects (all 4 modalities present): {len(subjects)}")

    clinical = pd.read_csv(CLINICAL_CSV)
    clinical["patient_num"] = clinical["ID"].apply(
        lambda x: int(m.group(1)) if (m := re.search(r"UPENN-GBM-(\d+)", str(x))) else None
    )
    clinical = clinical.drop_duplicates(subset="patient_num", keep="first").set_index("patient_num")

    has_gt = {s for s in subjects if (NIFTI_DIR / f"{s}_seg.nii.gz").exists()}
    needs_inference = [s for s in subjects if s not in has_gt]
    print(f"  with ground-truth mask : {len(has_gt)}")
    print(f"  needs model inference  : {len(needs_inference)}")

    model = None
    if needs_inference:
        print(f"Loading segmentor: {CHECKPOINT}")
        model = WaveletUNetPlusPlus(in_channels=4, n_classes=3).to(DEVICE)
        _ck = torch.load(str(CHECKPOINT), map_location=DEVICE, weights_only=False)
        model.load_state_dict(
            _ck.get("model_state_dict", _ck) if isinstance(_ck, dict) else _ck,
            strict=True)
        model.eval()

    rows = []
    for i, sub_id in enumerate(subjects, 1):
        try:
            vols = load_modalities(sub_id)
            img = normalize(vols)

            seg_p = NIFTI_DIR / f"{sub_id}_seg.nii.gz"
            if seg_p.exists():
                seg = nib.load(str(seg_p)).get_fdata().astype(np.float32)
                et = np.isin(seg, ET_LABELS).astype(np.float32)
                tc = (np.isin(seg, ET_LABELS) | (seg == 1)).astype(np.float32)
                wt = (np.isin(seg, ET_LABELS) | (seg == 1) | (seg == 2)).astype(np.float32)
                mask_source = "ground_truth"
            else:
                prob = sliding_window_predict(
                    model, img, DEVICE, SPATIAL_SIZE, SLIDE_STEP, use_tta=False
                )
                pred = postprocess(
                    prob, ET_THR, TC_THR, WT_THR,
                    et_policy=ET_POLICY, et_min_volume=ET_MIN_VOLUME,
                ).astype(np.float32)
                et, tc, wt = pred[0], pred[1], pred[2]
                mask_source = "model_predicted"

            affine = nib.load(str(NIFTI_DIR / f"{sub_id}_FLAIR.nii.gz")).affine
            voxel_vol = float(np.abs(np.linalg.det(affine[:3, :3])))

            vols_by_key = vols  # already keyed flair/t1/t1ce/t2

            feat = {"patient_id": sub_id, "mask_source": mask_source}
            feat.update(region_features(et, vols_by_key, "ET", voxel_vol))
            feat.update(region_features(tc, vols_by_key, "TC", voxel_vol))
            feat.update(region_features(wt, vols_by_key, "WT", voxel_vol))

            n = sub_to_num(sub_id)
            if n is not None and n in clinical.index:
                crow = clinical.loc[n]
                feat["age"] = crow.get("Age_at_scan_years", np.nan)
                gender = str(crow.get("Gender", "")).strip().upper()
                feat["gender_m"] = 1 if gender == "M" else (0 if gender == "F" else np.nan)
                feat["idh1_label"] = map_idh1(crow.get("IDH1", np.nan))
                feat["idh1_raw"] = crow.get("IDH1", np.nan)
            else:
                feat["age"] = np.nan
                feat["gender_m"] = np.nan
                feat["idh1_label"] = np.nan
                feat["idh1_raw"] = np.nan

            rows.append(feat)

        except Exception as e:
            print(f"  [{i}/{len(subjects)}] {sub_id} FAILED: {e}")
            continue

        if i % 25 == 0 or i == len(subjects):
            print(f"  [{i}/{len(subjects)}] {sub_id} done (mask_source={mask_source})")

        if model is not None and i % 50 == 0:
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(df)} rows -> {OUT_CSV}")
    print(df["mask_source"].value_counts())
    print(df["idh1_label"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
