"""Single source of truth for every train/val/test split in the project.

Both the fine-tuning script and the evaluation script previously carried their
own copy of the UPenn split function. They agreed, but nothing enforced that.
Every split now comes from here, and `freeze_upenn_split` writes the resulting
assignment to CSV so a reviewer can audit exactly which subject sat in which
partition.

Splitting rules, unchanged from the runs that produced the released
checkpoints, so frozen splits reproduce historical results bit for bit:

  UPenn : sorted `*_seg.nii.gz` basenames, permuted with
          numpy default_rng(seed), sliced test / val / train.
  BraTS : sorted `BraTS2021_*` directory names, partitioned with
          torch.utils.data.random_split under a manual-seed generator.
"""
from __future__ import annotations

import glob
import os
import pathlib

import numpy as np
import pandas as pd

UPENN_TEST_RATIO = 0.20
UPENN_VAL_RATIO = 0.10
BRATS_TRAIN_FRAC = 0.8
SEED = 42


def upenn_subjects(nifti_dir: str) -> list[str]:
    """Subjects that carry an expert segmentation, in deterministic order."""
    seg_files = sorted(glob.glob(os.path.join(nifti_dir, "*_seg.nii.gz")))
    return [os.path.basename(f).replace("_seg.nii.gz", "") for f in seg_files]


def upenn_split(
    nifti_dir: str,
    test_ratio: float = UPENN_TEST_RATIO,
    val_ratio: float = UPENN_VAL_RATIO,
    seed: int = SEED,
) -> tuple[list[str], list[str], list[str]]:
    """Return (train, val, test) subject ids."""
    subjects = upenn_subjects(nifti_dir)
    if not subjects:
        raise FileNotFoundError(f"No *_seg.nii.gz found in {nifti_dir}")

    idx = np.random.default_rng(seed).permutation(len(subjects))
    n_test = int(len(subjects) * test_ratio)
    n_val = int(len(subjects) * val_ratio)

    test = [subjects[i] for i in idx[:n_test]]
    val = [subjects[i] for i in idx[n_test:n_test + n_val]]
    train = [subjects[i] for i in idx[n_test + n_val:]]
    return train, val, test


def upenn_cv_folds(
    nifti_dir: str,
    n_folds: int = 5,
    seed: int = SEED,
) -> list[dict[str, list[str]]]:
    """Nested cross-validation folds over every expert-segmented subject.

    Each subject is tested exactly once. Within a fold the non-test subjects
    split again into train and an inner validation set; that inner set is the
    only data allowed to pick thresholds, post-processing parameters and the
    checkpoint. The fold's test subjects are untouched until final scoring.
    """
    subjects = upenn_subjects(nifti_dir)
    if not subjects:
        raise FileNotFoundError(f"No *_seg.nii.gz found in {nifti_dir}")

    order = np.random.default_rng(seed).permutation(len(subjects))
    shuffled = [subjects[i] for i in order]
    chunks = [list(c) for c in np.array_split(np.array(shuffled, dtype=object), n_folds)]

    folds = []
    for k in range(n_folds):
        test = chunks[k]
        rest = [s for j, c in enumerate(chunks) if j != k for s in c]
        # Inner validation carved from `rest` only — never from `test`.
        n_val = max(1, int(round(len(rest) * UPENN_VAL_RATIO)))
        inner = np.random.default_rng(seed + 1000 + k).permutation(len(rest))
        val = [rest[i] for i in inner[:n_val]]
        train = [rest[i] for i in inner[n_val:]]
        folds.append({"fold": k, "train": train, "val": val, "test": test})
    return folds


def freeze_upenn_split(nifti_dir: str, out_csv: str) -> pd.DataFrame:
    """Write the subject->partition assignment so the split is auditable."""
    train, val, test = upenn_split(nifti_dir)
    rows = (
        [{"subject_id": s, "partition": "train"} for s in sorted(train)]
        + [{"subject_id": s, "partition": "val"} for s in sorted(val)]
        + [{"subject_id": s, "partition": "test"} for s in sorted(test)]
    )
    df = pd.DataFrame(rows)
    pathlib.Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def freeze_upenn_cv(nifti_dir: str, out_csv: str, n_folds: int = 5) -> pd.DataFrame:
    """Write the cross-validation fold assignment."""
    rows = []
    for f in upenn_cv_folds(nifti_dir, n_folds=n_folds):
        for part in ("train", "val", "test"):
            rows.extend(
                {"fold": f["fold"], "subject_id": s, "partition": part}
                for s in sorted(f[part])
            )
    df = pd.DataFrame(rows)
    pathlib.Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def brats_split(
    data_root: str,
    train_frac: float = BRATS_TRAIN_FRAC,
    seed: int = SEED,
) -> tuple[set[str], set[str]]:
    """Reproduce train.py's BraTS partition exactly.

    Returns (train_ids, heldout_ids). The held-out portion doubled as the
    model-selection set during training, so it is internal validation rather
    than a clean test set; see docs/METHODOLOGY.md.
    """
    import torch
    from torch.utils.data import random_split

    root = pathlib.Path(data_root)
    pats = sorted(d.name for d in root.glob("BraTS2021_*") if d.is_dir())
    n = len(pats)
    train_n = int(train_frac * n)
    tr, va = random_split(
        list(range(n)), [train_n, n - train_n],
        generator=torch.Generator().manual_seed(seed),
    )
    return {pats[i] for i in tr.indices}, {pats[i] for i in va.indices}


def assert_disjoint(**partitions: list[str] | set[str]) -> None:
    """Raise if any two named partitions share a subject."""
    names = list(partitions)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = set(partitions[a]) & set(partitions[b])
            if overlap:
                raise AssertionError(
                    f"Leakage: {len(overlap)} subject(s) in both '{a}' and "
                    f"'{b}': {sorted(overlap)[:5]}"
                )
