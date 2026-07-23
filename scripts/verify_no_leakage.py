#!/usr/bin/env python3
"""Assert every data-integrity property the reported results depend on.

Run this before trusting any number in results/. It exits non-zero on the first
violation and prints what failed. Checks:

  1. UPenn train / validation / test partitions are mutually disjoint.
  2. The frozen split on disk still matches what splits.upenn_split produces.
  3. Cross-validation folds are disjoint and test each subject exactly once.
  4. BraTS train and internal-validation partitions are disjoint, and the
     reported per-case file covers the internal-validation partition.
  5. The IDH1 primary cohort shares no subject with segmentor fine-tuning.
  6. Every subject in the IDH1 primary cohort has the same mask provenance.
  7. Reported UPenn test results cover exactly the test partition.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brats_gbm.splits import (  # noqa: E402
    assert_disjoint,
    brats_split,
    upenn_cv_folds,
    upenn_split,
)

NIFTI_DIR = ROOT / "upenn_nifti"
DATA_ROOT = ROOT / "data"

checks: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    checks.append((name, bool(condition), detail))
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}"
          + (f"  — {detail}" if detail else ""))


def main() -> None:
    print("Data-integrity verification\n" + "=" * 72)

    print("\nUPenn-GBM segmentation split")
    train, val, test = upenn_split(str(NIFTI_DIR))
    try:
        assert_disjoint(train=train, val=val, test=test)
        check("train/val/test mutually disjoint", True,
              f"{len(train)}/{len(val)}/{len(test)}")
    except AssertionError as e:
        check("train/val/test mutually disjoint", False, str(e))

    total = len(train) + len(val) + len(test)
    check("partitions cover every expert-segmented subject",
          total == 147, f"{total} subjects")

    frozen = ROOT / "results" / "upenn" / "split_assignment.csv"
    if frozen.exists():
        f = pd.read_csv(frozen)
        same = (
            set(f[f.partition == "test"].subject_id) == set(test)
            and set(f[f.partition == "val"].subject_id) == set(val)
        )
        check("frozen split on disk matches splits.upenn_split", same)
    else:
        check("frozen split on disk exists", False, f"{frozen} missing")

    print("\nUPenn-GBM cross-validation folds")
    folds = upenn_cv_folds(str(NIFTI_DIR))
    ok = True
    seen: list[str] = []
    for f in folds:
        try:
            assert_disjoint(**{k: f[k] for k in ("train", "val", "test")})
        except AssertionError:
            ok = False
        seen.extend(f["test"])
    check("every fold's train/val/test disjoint", ok)
    check("each subject tested exactly once",
          len(seen) == len(set(seen)) == 147, f"{len(seen)} test slots")

    print("\nBraTS2021 split")
    if DATA_ROOT.exists():
        btrain, bval = brats_split(str(DATA_ROOT))
        check("train and internal-validation disjoint", not (btrain & bval),
              f"{len(btrain)}/{len(bval)}")
        per_case = ROOT / "results" / "brats" / "per_case_internal_validation.csv"
        if per_case.exists():
            ids = set(pd.read_csv(per_case).Patient_ID)
            check("reported per-case file covers exactly internal validation",
                  ids == bval, f"{len(ids)} cases")
        else:
            check("internal-validation per-case file exists", False,
                  "run scripts/summarize_brats.py")
    else:
        check("BraTS data directory present", False, f"{DATA_ROOT} missing")

    print("\nIDH1 classification cohort")
    feat = ROOT / "results" / "classification" / "radiomic_features.csv"
    if feat.exists():
        df = pd.read_csv(feat)
        labelled = df[df.idh1_label.notna()]
        primary = labelled[labelled.mask_source == "model_predicted"]
        overlap = set(primary.patient_id) & (set(train) | set(val))
        check("primary cohort disjoint from segmentor fine-tuning",
              not overlap, f"n={len(primary)}, overlap={len(overlap)}")
        check("primary cohort has uniform mask provenance",
              primary.mask_source.nunique() == 1,
              f"{primary.mask_source.unique().tolist()}")

        prev = labelled.groupby("mask_source").idh1_label.mean()
        if len(prev) > 1:
            ratio = prev.max() / max(prev.min(), 1e-9)
            print(f"       note: prevalence differs {ratio:.1f}x across mask "
                  f"provenance in the mixed cohort — this is why the primary "
                  f"cohort is restricted")
    else:
        check("radiomic feature table exists", False, f"{feat} missing")

    print("\nUPenn reported results")
    summary = ROOT / "results" / "upenn" / "segmentation_summary.csv"
    for csv in sorted((ROOT / "results" / "upenn").glob("per_case_*.csv")):
        ids = set(pd.read_csv(csv).Patient_ID)
        check(f"{csv.name} covers exactly the test partition", ids == set(test),
              f"{len(ids)} cases")
    if not summary.exists():
        check("segmentation summary exists", False,
              "run scripts/evaluate_upenn.py")

    failed = [c for c in checks if not c[1]]
    print("\n" + "=" * 72)
    if failed:
        print(f"{len(failed)} of {len(checks)} checks FAILED:")
        for name, _, detail in failed:
            print(f"  - {name} {detail}")
        sys.exit(1)
    print(f"All {len(checks)} checks passed.")


if __name__ == "__main__":
    main()
