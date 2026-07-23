#!/usr/bin/env python3
"""Summarise BraTS2021 segmentation over the internal-validation partition.

Two corrections relative to how these numbers were originally reported.

First, the pooling. Inference was run over all 1251 cases and the per-case
scores were averaged together, which mixes the 1000 training cases into the
headline figure. Only the 251 cases outside training are meaningful, and they
are what this script reports; the training-case average is printed alongside
purely so the generalisation gap is visible.

Second, the naming. `train_brats.py` makes a two-way split, and the 251-case
partition served as the model-selection set during training. Calling it a test
set would overstate it, so it is labelled internal validation throughout. The
project's clean external test is UPenn-GBM, evaluated by scripts/evaluate_upenn.py.

Per-case scores are read from an existing results CSV rather than recomputed,
so this is cheap to rerun.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brats_gbm.eval.stats import (  # noqa: E402
    bootstrap_ci,
    print_summary,
    summarise_segmentation,
)
from brats_gbm.splits import brats_split  # noqa: E402

DEFAULT_CSV = ROOT / "results" / "brats" / "per_case_all_1251.csv"
DATA_ROOT = ROOT / "data"
OUT_DIR = ROOT / "results" / "brats"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-csv", default=str(DEFAULT_CSV))
    ap.add_argument("--data-root", default=str(DATA_ROOT))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    res = pd.read_csv(args.results_csv)
    if "Patient_ID" not in res.columns:
        raise SystemExit(f"{args.results_csv} has no Patient_ID column")

    train_ids, val_ids = brats_split(args.data_root)
    res["split"] = np.where(
        res.Patient_ID.isin(val_ids), "internal_validation",
        np.where(res.Patient_ID.isin(train_ids), "train", "unknown"))

    counts = res.split.value_counts().to_dict()
    print(f"source : {args.results_csv}  ({len(res)} rows)")
    print(f"split  : {counts}")
    if counts.get("unknown"):
        print(f"warning: {counts['unknown']} cases matched neither partition")

    held = res[res.split == "internal_validation"].copy()
    trained = res[res.split == "train"].copy()
    if held.empty:
        raise SystemExit("No internal-validation cases found — check --data-root.")

    val_summary = summarise_segmentation(held, label="internal_validation")
    print_summary(val_summary, f"INTERNAL VALIDATION — report these (n={len(held)})")

    val_summary.to_csv(out_dir / "segmentation_summary.csv", index=False)
    held.to_csv(out_dir / "per_case_internal_validation.csv", index=False)

    if not trained.empty:
        train_summary = summarise_segmentation(trained, label="train")
        print_summary(train_summary, f"TRAINING CASES — memorised, do not report (n={len(trained)})")

        # Unpaired difference: the two partitions contain different patients,
        # so resample each independently rather than pairing them.
        av = trained[["Dice_ET", "Dice_TC", "Dice_WT"]].mean(axis=1).to_numpy()
        bv = held[["Dice_ET", "Dice_TC", "Dice_WT"]].mean(axis=1).to_numpy()
        rng = np.random.default_rng(42)
        boot = [
            rng.choice(av, av.size, replace=True).mean()
            - rng.choice(bv, bv.size, replace=True).mean()
            for _ in range(2000)
        ]
        lo, hi = np.percentile(boot, 2.5), np.percentile(boot, 97.5)
        print(f"\ngeneralisation gap (train - validation): {av.mean() - bv.mean():+.4f}  "
              f"95% CI [{lo:+.4f}, {hi:+.4f}]")
        if lo <= 0 <= hi:
            print("  interval includes zero — no detectable overfitting to the "
                  "training cases")

        pooled = res[["Dice_ET", "Dice_TC", "Dice_WT"]].mean(axis=1).mean()
        print(f"pooled over all {len(res)} cases (the original, inflated figure): {pooled:.4f}")

    print(f"\nWrote {out_dir / 'segmentation_summary.csv'}")


if __name__ == "__main__":
    main()
