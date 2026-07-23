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
MODALITY_SUFFIXES = ("FLAIR", "T1w", "ce-gd_T1w", "T2w")


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
    ):
        self.data_dir = Path(data_dir)
        self.subject_ids = list(subject_ids)
        self.target_label = target_label
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

    def __getitem__(self, idx: int) -> dict:
        sub_id = self.subject_ids[idx]

        def load(suffix: str) -> np.ndarray:
            p = self.data_dir / f"{sub_id}_{suffix}.nii.gz"
            if not p.exists():
                raise FileNotFoundError(f"Missing: {p}")
            return nib.load(str(p)).get_fdata().astype(np.float32)

        img = np.stack([load(s) for s in MODALITY_SUFFIXES], axis=0)
        img = znorm_nonzero(img)

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
