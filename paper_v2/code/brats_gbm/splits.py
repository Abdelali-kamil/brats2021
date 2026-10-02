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
# Fraction of the 251 held-out BraTS cases used for model selection; the
# remainder is the clean test partition. See `brats_split_3way`.
BRATS_VAL_FRAC_OF_HELDOUT = 0.5
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


def brats_split_3way(
    data_root: str,
    train_frac: float = BRATS_TRAIN_FRAC,
    seed: int = SEED,
    val_frac_of_heldout: float = BRATS_VAL_FRAC_OF_HELDOUT,
) -> tuple[set[str], set[str], set[str]]:
    """Three-way BraTS partition: train / val / test.

    `brats_split` reproduces the historical two-way partition, whose 251
    held-out cases doubled as the model-selection set. Any score reported on
    them is therefore optimistic — the defect `docs/METHODOLOGY.md` records as
    threshold tuning on the evaluation data, in a milder form.

    This function keeps the 1000 training cases bit-identical to that partition,
    so a model trained under it differs from the released checkpoint only in
    training procedure, and splits the held-out 251 in two:

      val  : drives LR scheduling and checkpoint selection, as the 251 did.
      test : never consulted during training or post-processing choice, and
             scored exactly once.

    A number from `test` is what the paper can defend. Note that the released
    checkpoint selected on the *whole* 251, so its score on this `test` subset
    is still optimistic and the two are not equally clean — see the caveat in
    `docs/METHODOLOGY.md`.
    """
    train_ids, heldout = brats_split(data_root, train_frac=train_frac, seed=seed)

    ordered = sorted(heldout)
    idx = np.random.default_rng(seed + 7).permutation(len(ordered))
    n_val = int(round(len(ordered) * val_frac_of_heldout))

    val = {ordered[i] for i in idx[:n_val]}
    test = {ordered[i] for i in idx[n_val:]}
    return train_ids, val, test


def freeze_brats_split(data_root: str, out_csv: str) -> pd.DataFrame:
    """Write the BraTS case->partition assignment so the split is auditable."""
    train, val, test = brats_split_3way(data_root)
    rows = (
        [{"case_id": c, "partition": "train"} for c in sorted(train)]
        + [{"case_id": c, "partition": "val"} for c in sorted(val)]
        + [{"case_id": c, "partition": "test"} for c in sorted(test)]
    )
    df = pd.DataFrame(rows)
    pathlib.Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


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
