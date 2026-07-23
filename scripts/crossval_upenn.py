#!/usr/bin/env python3
"""5-fold cross-validated fine-tuning and evaluation on UPenn-GBM.

Why this exists
---------------
The headline UPenn result comes from a single 104/14/29 split. Twenty-nine test
cases put a roughly +/-0.05 confidence interval on mean Dice, which is wider
than most of the differences the project wants to claim. Cross-validation
evaluates every one of the 147 expert-segmented subjects exactly once, by a
model that never saw it, so the effective test size rises from 29 to 147 and
the intervals tighten accordingly.

Protocol per fold
-----------------
  1. Partition into test / inner-validation / train (see splits.upenn_cv_folds).
  2. Fine-tune from the BraTS checkpoint on the fold's training subjects.
  3. Select thresholds, ET policy and the checkpoint on inner validation.
  4. Score the fold's test subjects once.

Pooling the five test partitions gives out-of-fold scores for all 147 subjects.
No subject ever contributes to any decision that affects its own score.

Cost
----
Five fine-tuning runs plus five evaluations. On a single RTX 5090 with the
default 80 epochs this is roughly 10-14 hours wall clock. Reduce with
--epochs for a cheaper approximation; the fine-tune is largely converged by
epoch 40 (see logs/finetune_v3.log).

Usage
-----
    python scripts/crossval_upenn.py                 # all 5 folds
    python scripts/crossval_upenn.py --folds 0 1     # resume a subset
    python scripts/crossval_upenn.py --epochs 40     # cheaper approximation
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brats_gbm.eval.stats import print_summary, summarise_segmentation  # noqa: E402
from brats_gbm.splits import assert_disjoint, freeze_upenn_cv, upenn_cv_folds  # noqa: E402

NIFTI_DIR = ROOT / "upenn_nifti"
OUT_DIR = ROOT / "results" / "upenn" / "crossval"
CKPT_DIR = ROOT / "checkpoints" / "crossval"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="*", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--nifti-dir", default=str(NIFTI_DIR))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--dry-run", action="store_true",
                    help="verify fold construction and exit without training")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    folds = upenn_cv_folds(args.nifti_dir)
    freeze_upenn_cv(args.nifti_dir, str(out_dir / "cv_fold_assignment.csv"))

    # Structural guarantees, checked before any GPU time is spent.
    all_test: list[str] = []
    for f in folds:
        assert_disjoint(**{f"fold{f['fold']}_{k}": f[k] for k in ("train", "val", "test")})
        all_test.extend(f["test"])
    if len(all_test) != len(set(all_test)):
        raise AssertionError("a subject appears in more than one test fold")
    print(f"{len(folds)} folds over {len(all_test)} subjects; "
          f"each tested exactly once; all partitions disjoint")
    for f in folds:
        print(f"  fold {f['fold']}: train={len(f['train'])} "
              f"val={len(f['val'])} test={len(f['test'])}")

    if args.dry_run:
        print("\ndry run — fold construction verified, nothing trained")
        return

    selected = args.folds if args.folds is not None else [f["fold"] for f in folds]
    per_fold = []

    for f in folds:
        if f["fold"] not in selected:
            continue
        k = f["fold"]
        print(f"\n{'=' * 72}\nfold {k}\n{'=' * 72}")

        fold_ckpt = CKPT_DIR / f"fold{k}"
        fold_ckpt.mkdir(parents=True, exist_ok=True)
        split_json = out_dir / f"fold{k}_subjects.json"
        split_json.write_text(json.dumps(
            {key: f[key] for key in ("train", "val", "test")}, indent=2))

        train_cmd = [
            sys.executable, "-u", str(ROOT / "scripts" / "train_upenn.py"),
            "--subject-split", str(split_json),
            "--epochs", str(args.epochs),
            "--save-dir", str(fold_ckpt),
        ]
        print("  " + " ".join(train_cmd))
        rc = subprocess.run(train_cmd, cwd=ROOT).returncode
        if rc != 0:
            print(f"  fold {k} training failed (exit {rc}); skipping")
            continue

        eval_cmd = [
            sys.executable, "-u", str(ROOT / "scripts" / "evaluate_upenn.py"),
            "--subject-split", str(split_json),
            "--setups", f"crossval_fold{k}",
            "--out-dir", str(out_dir / f"fold{k}"),
        ]
        print("  " + " ".join(eval_cmd))
        rc = subprocess.run(eval_cmd, cwd=ROOT).returncode
        if rc != 0:
            print(f"  fold {k} evaluation failed (exit {rc}); skipping")
            continue

        case_csv = out_dir / f"fold{k}" / f"per_case_crossval_fold{k}.csv"
        if case_csv.exists():
            df = pd.read_csv(case_csv)
            df["fold"] = k
            per_fold.append(df)

    if not per_fold:
        print("\nNo folds completed; nothing to pool.")
        return

    pooled = pd.concat(per_fold, ignore_index=True)
    pooled.to_csv(out_dir / "per_case_pooled.csv", index=False)
    summary = summarise_segmentation(pooled, label="crossval_pooled")
    summary.to_csv(out_dir / "crossval_summary.csv", index=False)
    print_summary(summary, f"POOLED OUT-OF-FOLD (n={len(pooled)})")
    print(f"\nWrote {out_dir}")


if __name__ == "__main__":
    main()
