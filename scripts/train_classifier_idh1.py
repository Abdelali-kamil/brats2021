#!/usr/bin/env python3
"""IDH1 mutation classification from segmentation-derived radiomic features.

Cohort definition is the important part of this script.

The feature table covers 629 subjects, but the masks behind those features come
from two different sources: expert ground truth for the 147 subjects that have
one, and model predictions for the other 482. That split is not random with
respect to the outcome — IDH1-mutant prevalence is 0.9% among ground-truth
subjects and 4.1% among model-predicted ones, a 4.5-fold difference. Because
mask provenance changes the distribution of every shape feature (sphericity,
component count, surface area), a classifier trained across both groups can
learn to recognise the mask source and use it as a proxy for the label. Even
with `mask_source` dropped as an explicit column the signal survives in the
geometry. Ground-truth subjects are also exactly the subjects the segmentor
was fine-tuned on, so their masks are optimistic in a second way.

The primary analysis therefore uses only the model-predicted subjects: one
uniform mask provenance, and zero overlap with segmentation training by
construction (a subject has a predicted mask precisely because it has no
expert mask, and fine-tuning required an expert mask). The mixed cohort is
reported afterwards as a sensitivity analysis to show what the artefact was
worth.

With 17 positives in the primary cohort this analysis is underpowered no
matter how carefully it is run. Confidence intervals are reported on every
metric and should be read as the main result.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brats_gbm.classification import (  # noqa: E402
    build_models,
    cross_validate,
    print_model_summary,
)
from brats_gbm.splits import upenn_split  # noqa: E402

FEATURES_CSV = ROOT / "results" / "classification" / "radiomic_features.csv"
OUT_DIR = ROOT / "results" / "classification"
NIFTI_DIR = ROOT / "upenn_nifti"

NON_FEATURE_COLS = {"patient_id", "mask_source", "idh1_label", "idh1_raw"}


def run_cohort(df: pd.DataFrame, feature_cols: list[str], name: str) -> dict:
    X = df[feature_cols].to_numpy(float)
    y = df["idh1_label"].to_numpy(int)
    print(f"\n{'=' * 72}\ncohort: {name}")
    print(f"  n={len(y)}  positives={int(y.sum())} ({y.mean() * 100:.1f}%)  "
          f"features={len(feature_cols)}\n{'=' * 72}")

    out = {}
    for model_name, model in build_models().items():
        res = cross_validate(X, y, model)
        out[model_name] = res["summary"]
        print_model_summary(model_name, res["summary"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(FEATURES_CSV))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.features)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    feature_cols = [c for c in feature_cols
                    if pd.api.types.is_numeric_dtype(df[c])]

    labelled = df[df["idh1_label"].notna()].copy()
    labelled["idh1_label"] = labelled["idh1_label"].astype(int)

    print(f"feature table : {args.features}")
    print(f"subjects      : {len(df)} total, {len(labelled)} labelled")
    print(f"mask source   : {labelled.mask_source.value_counts().to_dict()}")

    prevalence = labelled.groupby("mask_source")["idh1_label"].mean()
    print("\nIDH1-mutant prevalence by mask provenance:")
    for src, p in prevalence.items():
        n = int((labelled.mask_source == src).sum())
        print(f"  {src:<16} {p * 100:5.2f}%  (n={n})")
    print("  -> provenance correlates with the outcome; the mixed cohort is "
          "reported only as a sensitivity analysis.")

    # Primary cohort: uniform mask provenance, disjoint from segmentor training.
    primary = labelled[labelled.mask_source == "model_predicted"].reset_index(drop=True)
    train_subs, val_subs, _ = upenn_split(str(NIFTI_DIR))
    overlap = set(primary.patient_id) & (set(train_subs) | set(val_subs))
    if overlap:
        raise AssertionError(
            f"{len(overlap)} primary-cohort subjects were used to fine-tune the "
            f"segmentor: {sorted(overlap)[:5]}")
    print(f"\nverified: primary cohort shares 0 subjects with segmentor "
          f"fine-tuning (train+val = {len(train_subs) + len(val_subs)})")

    results = {
        "primary_model_predicted_only": run_cohort(
            primary, feature_cols, "PRIMARY — model-predicted masks only"),
        "sensitivity_mixed_provenance": run_cohort(
            labelled.reset_index(drop=True), feature_cols,
            "SENSITIVITY — mixed provenance (not for headline reporting)"),
    }
    results["cohort_sizes"] = {
        "primary_n": int(len(primary)),
        "primary_positives": int(primary.idh1_label.sum()),
        "mixed_n": int(len(labelled)),
        "mixed_positives": int(labelled.idh1_label.sum()),
    }
    results["prevalence_by_mask_source"] = {k: float(v) for k, v in prevalence.items()}

    (out_dir / "idh1_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_dir / 'idh1_results.json'}")


if __name__ == "__main__":
    main()
