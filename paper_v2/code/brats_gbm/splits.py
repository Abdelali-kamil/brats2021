"""Single source of truth for every BraTS 2021 train/val/test split in the project.

Every split comes from here, and `freeze_brats_split` writes the resulting
assignment to CSV so a reviewer can audit exactly which case sat in which
partition.

Splitting rule, unchanged from the runs that produced the released
checkpoints, so frozen splits reproduce historical results bit for bit:

  BraTS : sorted `BraTS2021_*` directory names, partitioned with
          torch.utils.data.random_split under a manual-seed generator.
"""
from __future__ import annotations

import glob
import os
import pathlib

import numpy as np
import pandas as pd

BRATS_TRAIN_FRAC = 0.8
# Fraction of the 251 held-out BraTS cases used for model selection; the
# remainder is the clean test partition. See `brats_split_3way`.
BRATS_VAL_FRAC_OF_HELDOUT = 0.5
SEED = 42


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
