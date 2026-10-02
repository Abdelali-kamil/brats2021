#!/usr/bin/env python3
"""MGMT promoter methylation classification on BraTS2021.

Cohort: 577 cases with both imaging and an MGMT label, features derived from
expert segmentations. Classes are near balanced (301 methylated / 276 not),
which makes this a far better-powered analysis than the UPenn IDH1 one — no
rare-event problem, no mask-provenance confound, and no segmentation model in
the loop.

Protocol is identical to the IDH1 analysis (`brats_gbm.classification`):
repeated stratified 5-fold cross-validation, 10 repeats, decision thresholds
fitted on cross-fitted training-fold scores, bootstrap confidence intervals
over patients.

A note on expectations. Predicting MGMT methylation from MRI is a task with
contested signal: the RSNA-MICCAI 2021 challenge that used these labels was won
at roughly 0.62 AUC, and several subsequent analyses argue the imaging signal is
weak or absent. A result near chance here would be consistent with that
literature and is worth reporting as such — the useful output of this script may
well be a negative one, and the AUC confidence interval is what decides it.

This project's earlier MGMT attempt reported accuracy 0.62 and AUC 0.65 on 116
cases with specificity 0.455, from a pipeline that could not be regenerated. It
is superseded by this analysis and archived.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brats_gbm.classification import (  # noqa: E402
    build_models,
    cross_validate,
    print_model_summary,
)

FEATURES_CSV = ROOT / "results" / "classification" / "brats_mgmt_features.csv"
OUT_DIR = ROOT / "results" / "classification"
NON_FEATURE_COLS = {"case_id", "mask_source", "mgmt_label"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(FEATURES_CSV))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.features)
    feature_cols = [c for c in df.columns
                    if c not in NON_FEATURE_COLS and pd.api.types.is_numeric_dtype(df[c])]
    labelled = df[df.mgmt_label.notna()].reset_index(drop=True)
    labelled["mgmt_label"] = labelled["mgmt_label"].astype(int)

    print(f"feature table : {args.features}")
    print(f"cohort        : {len(labelled)} cases, {len(feature_cols)} features")
    print(f"labels        : {labelled.mgmt_label.value_counts().to_dict()} "
          f"({labelled.mgmt_label.mean() * 100:.1f}% methylated)")
    print(f"mask source   : {labelled.mask_source.unique().tolist()} "
          f"— uniform, so provenance cannot act as a label proxy")

    X = labelled[feature_cols].to_numpy(float)
    y = labelled["mgmt_label"].to_numpy(int)

    print(f"\n{'=' * 72}\nBraTS2021 MGMT — repeated stratified 5-fold CV, 10 repeats\n{'=' * 72}")
    results = {}
    for name, model in build_models().items():
        res = cross_validate(X, y, model)
        results[name] = res["summary"]
        print_model_summary(name, res["summary"])

    best = max(results, key=lambda k: results[k]["auc_patient_bootstrap"]["mean"])
    pb = results[best]["auc_patient_bootstrap"]
    print(f"\n{'=' * 72}")
    print(f"best model: {best}  AUC {pb['mean']:.3f} "
          f"[{pb['ci_low']:.3f}, {pb['ci_high']:.3f}]")
    if pb["ci_low"] <= 0.5:
        print("The interval includes chance. On this cohort, with these features,")
        print("MGMT status is not predictable from the segmentation-derived")
        print("radiomics. That is a legitimate negative result and consistent with")
        print("the published difficulty of this task — report it as such rather")
        print("than searching for a configuration that clears 0.5.")
    else:
        print("The interval excludes chance, but check the effect size before")
        print("claiming clinical usefulness.")

    results["cohort"] = {
        "n": int(len(y)), "n_methylated": int(y.sum()),
        "n_unmethylated": int(len(y) - y.sum()),
        "mask_source": "ground_truth", "n_features": len(feature_cols),
    }
    (out_dir / "mgmt_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_dir / 'mgmt_results.json'}")


if __name__ == "__main__":
    main()
