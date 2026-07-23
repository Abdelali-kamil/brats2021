"""UPenn-GBM volume loader.

Loads the four modalities at full resolution, z-scores each within its non-zero
region, and builds the three nested BraTS target regions.

Label convention differs between the two cohorts: enhancing tumour is 4 in
BraTS2021 and 3 in UPenn-GBM. Neither cohort uses the other's value, so both
are accepted and a single loader serves both.
"""
from __future__ import annotations

import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch

ET_LABELS = (3, 4)

# Modality file suffixes in UPenn-GBM, in the order the fine-tuned checkpoints
# were trained with.
MODALITY_SUFFIXES = ("FLAIR", "T1w", "ce-gd_T1w", "T2w")

# Input preprocessing is a property of the checkpoint, not of the dataset.
#
# The BraTS2021 loader (brats_gbm/data/brats.py) stacks channels as
# [t1, t1ce, t2, flair] and normalises with percentile-clipped min-max to
# [0, 1]. The UPenn loader stacks [flair, t1, t1ce, t2] and z-scores. Those are
# a channel permutation *and* a different intensity distribution, so feeding a
# BraTS-trained checkpoint the UPenn convention hands it scrambled input.
#
# Measured on 5 UPenn validation subjects with segmentor_epoch_650, mean Dice:
#   UPenn convention (what was previously used) : 0.203
#   correct order, wrong normalisation          : 0.356
#   wrong order, correct normalisation          : 0.160
#   BraTS convention (what it was trained on)   : 0.766
#
# Almost the whole apparent BraTS-to-UPenn "domain gap" was this mismatch. Each
# checkpoint must be fed the convention it was trained under.
PREPROCESSING = {
    # suffix order, normaliser name
    "upenn": (("FLAIR", "T1w", "ce-gd_T1w", "T2w"), "zscore"),
    "brats": (("T1w", "ce-gd_T1w", "T2w", "FLAIR"), "minmax"),
}


def subject_number(sub_id: str) -> int | None:
    m = re.search(r"sub-(\d+)", str(sub_id))
    return int(m.group(1)) if m else None


def clinical_patient_number(id_str: str) -> int | None:
    m = re.search(r"UPENN-GBM-(\d+)", str(id_str))
    return int(m.group(1)) if m else None


def regions_from_seg(seg: np.ndarray) -> np.ndarray:
    """Stack the nested [ET, TC, WT] regions from a label volume."""
    et = np.isin(seg, ET_LABELS)
    tc = et | (seg == 1)
    wt = tc | (seg == 2)
    return np.stack([et, tc, wt]).astype(np.float32)


def minmax_percentile(img: np.ndarray, low_perc: int = 1, high_perc: int = 99) -> np.ndarray:
    """Percentile-clipped min-max to [0, 1] — the BraTS2021 training convention."""
    out = np.empty_like(img)
    for c in range(img.shape[0]):
        x = img[c]
        nz = x > 0
        if not nz.any():
            out[c] = x
            continue
        low, high = np.percentile(x[nz], [low_perc, high_perc])
        x = np.clip(x, low, high)
        rng = x.max() - x.min()
        out[c] = (x - x.min()) / rng if rng else x
    return out


def normalise(img: np.ndarray, kind: str) -> np.ndarray:
    if kind == "zscore":
        return znorm_nonzero(img)
    if kind == "minmax":
        return minmax_percentile(img)
    raise ValueError(f"unknown normaliser {kind!r}")


def znorm_nonzero(img: np.ndarray) -> np.ndarray:
    """Z-score each channel over its non-zero voxels; background stays 0."""
    out = img.copy()
    for c in range(out.shape[0]):
        mask = out[c] > 0
        if mask.any():
            out[c] = (out[c] - out[c][mask].mean()) / (out[c][mask].std() + 1e-8)
            out[c][~mask] = 0.0
    return out


class UPennDataset(torch.utils.data.Dataset):
    """Full-resolution UPenn-GBM cases, optionally carrying a clinical label."""

    def __init__(
        self,
        data_dir: str,
        subject_ids: list[str],
        clinical_csv: str | None = None,
        target_label: str = "IDH1",
        preprocessing: str = "upenn",
    ):
        if preprocessing not in PREPROCESSING:
            raise ValueError(
                f"unknown preprocessing {preprocessing!r}; "
                f"expected one of {tuple(PREPROCESSING)}")
        self.data_dir = Path(data_dir)
        self.subject_ids = list(subject_ids)
        self.target_label = target_label
        self.preprocessing = preprocessing
        self.suffixes, self.normaliser = PREPROCESSING[preprocessing]
        self.clinical = None

        if clinical_csv and Path(clinical_csv).exists():
            self.clinical = pd.read_csv(clinical_csv)
            if "ID" in self.clinical.columns:
                self.clinical["patient_num"] = self.clinical["ID"].apply(
                    clinical_patient_number
                )

    def __len__(self) -> int:
        return len(self.subject_ids)

    def _map_label(self, val) -> int:
        """Map a clinical string to {0, 1}; -1 marks unusable (NOS/NEC/blank)."""
        if pd.isna(val):
            return -1
        v = str(val).strip().lower()
        if self.target_label == "IDH1":
            return 1 if v == "mutant" else (0 if v == "wildtype" else -1)
        if self.target_label == "MGMT":
            return 1 if v == "methylated" else (0 if v == "unmethylated" else -1)
        return -1

    def _get_label(self, sub_id: str) -> int:
        if self.clinical is None or self.target_label not in self.clinical.columns:
            return -1
        n = subject_number(sub_id)
        if n is None:
            return -1
        rows = self.clinical[self.clinical["patient_num"] == n]
        for _, row in rows.iterrows():
            y = self._map_label(row[self.target_label])
            if y != -1:
                return y
        return -1

    def labels_only(self, idx: int) -> np.ndarray:
        """Boolean [3,D,H,W] target without touching the four modality volumes.

        `__getitem__` loads and z-scores roughly 3.5 GB of imaging per subject.
        Evaluation needs the targets alone, so reading just the segmentation
        turns a multi-minute setup into a few seconds.
        """
        sub_id = self.subject_ids[idx]
        seg_p = self.data_dir / f"{sub_id}_seg.nii.gz"
        if not seg_p.exists():
            raise FileNotFoundError(f"Missing segmentation: {seg_p}")
        seg = nib.load(str(seg_p)).get_fdata().astype(np.float32)
        return regions_from_seg(seg).astype(bool)

    def __getitem__(self, idx: int) -> dict:
        sub_id = self.subject_ids[idx]

        def load(suffix: str) -> np.ndarray:
            p = self.data_dir / f"{sub_id}_{suffix}.nii.gz"
            if not p.exists():
                raise FileNotFoundError(f"Missing: {p}")
            return nib.load(str(p)).get_fdata().astype(np.float32)

        img = np.stack([load(s) for s in self.suffixes], axis=0)
        img = normalise(img, self.normaliser)

        seg_p = self.data_dir / f"{sub_id}_seg.nii.gz"
        seg = (
            nib.load(str(seg_p)).get_fdata().astype(np.float32)
            if seg_p.exists()
            else np.zeros(img.shape[1:], dtype=np.float32)
        )

        return {
            "image": torch.from_numpy(img).float(),
            "label": torch.from_numpy(regions_from_seg(seg)).float(),
            "patient_id": sub_id,
            "cls_label": torch.tensor(self._get_label(sub_id), dtype=torch.long),
        }
