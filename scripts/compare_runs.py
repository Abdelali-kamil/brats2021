#!/usr/bin/env python3
"""Compare two segmentation runs on the same cases — the before/after harness.

Given two per-case CSVs (as written by evaluate_brats.py / evaluate_upenn.py,
schema ``Patient_ID, Dice_ET, HD95_ET, ...``), this aligns them on Patient_ID
and reports, per region and for the mean:

  * baseline Dice with a bootstrap 95% CI,
  * candidate Dice with a bootstrap 95% CI,
  * the paired difference (candidate - baseline) with a paired bootstrap CI.

The paired interval resamples patients jointly, so it reflects the
within-patient change and is the correct test for "did this help". A change is
credible only when its paired CI excludes zero.

This does not tune anything and does not touch the test data itself — it only
reads scores that were already computed under a fixed protocol, so it cannot
introduce leakage. Works for both BraTS and UPenn CSVs (identical schema).

Example
-------
    # BraTS: current 650 checkpoint vs an augmentation-trained checkpoint
    python scripts/evaluate_brats.py --partition internal_validation \\
        --checkpoint checkpoints/segmentor_epoch_650.pth --tag baseline
    python scripts/evaluate_brats.py --partition internal_validation \\
        --checkpoint checkpoints/segmentor_aug_best.pth   --tag aug
    python scripts/compare_runs.py \\
        --baseline results/brats/per_case_baseline_recomputed.csv \\
        --candidate results/brats/per_case_aug_recomputed.csv \\
        --name-baseline "epoch650" --name-candidate "augmented"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brats_gbm.eval.stats import bootstrap_ci, fmt_ci, paired_diff_ci  # noqa: E402

REGIONS = ("ET", "TC", "WT")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", required=True, help="baseline per-case CSV")
    ap.add_argument("--candidate", required=True, help="candidate per-case CSV")
    ap.add_argument("--name-baseline", default="baseline")
    ap.add_argument("--name-candidate", default="candidate")
    ap.add_argument("--out-csv", default=None,
                    help="optional path to write the comparison table")
    args = ap.parse_args()

    a = pd.read_csv(args.baseline)
    b = pd.read_csv(args.candidate)
    if "Patient_ID" not in a.columns or "Patient_ID" not in b.columns:
        raise SystemExit("both CSVs must have a Patient_ID column")

    merged = a.merge(b, on="Patient_ID", suffixes=("_base", "_cand"))
    n_only_base = len(set(a["Patient_ID"]) - set(b["Patient_ID"]))
    n_only_cand = len(set(b["Patient_ID"]) - set(a["Patient_ID"]))
    if len(merged) == 0:
        raise SystemExit("no shared Patient_IDs between the two CSVs")
    if n_only_base or n_only_cand:
        print(f"[warn] {n_only_base} cases only in baseline, "
              f"{n_only_cand} only in candidate; comparing the "
              f"{len(merged)} shared cases (paired).")

    print(f"\nPaired comparison on {len(merged)} shared cases")
    print(f"  baseline  = {args.name_baseline}  ({args.baseline})")
    print(f"  candidate = {args.name_candidate}  ({args.candidate})")
    print("=" * 92)
    print(f"{'Region':<6} | {args.name_baseline[:22]:<22} | "
          f"{args.name_candidate[:22]:<22} | {'Δ (cand - base) [95% CI]':<28}")
    print("-" * 92)

    rows = []
    for region in list(REGIONS) + ["Mean"]:
        if region == "Mean":
            base_vals = merged[[f"Dice_{r}_base" for r in REGIONS]].mean(axis=1)
            cand_vals = merged[[f"Dice_{r}_cand" for r in REGIONS]].mean(axis=1)
        else:
            base_vals = merged[f"Dice_{region}_base"]
            cand_vals = merged[f"Dice_{region}_cand"]

        bp, blo, bhi = bootstrap_ci(base_vals)
        cp, clo, chi = bootstrap_ci(cand_vals)
        dp, dlo, dhi = paired_diff_ci(cand_vals, base_vals)
        credible = (dlo > 0) or (dhi < 0)
        flag = "  *" if credible else ""

        print(f"{region:<6} | {fmt_ci(bp, blo, bhi):<22} | "
              f"{fmt_ci(cp, clo, chi):<22} | {fmt_ci(dp, dlo, dhi):<24}{flag}")
        rows.append({
            "region": region,
            "baseline_dice": bp, "baseline_lo": blo, "baseline_hi": bhi,
            "candidate_dice": cp, "candidate_lo": clo, "candidate_hi": chi,
            "delta": dp, "delta_lo": dlo, "delta_hi": dhi,
            "ci_excludes_zero": credible,
        })

    print("=" * 92)
    print("  * paired 95% CI excludes zero — the change is credible on these cases.")
    print("  (A difference whose CI includes zero is not resolvable at this sample size.)")

    if args.out_csv:
        out = Path(args.out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
