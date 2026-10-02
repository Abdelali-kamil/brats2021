#!/usr/bin/env python3
"""Assemble the UPenn-GBM MGMT cohort: imaging radiomics + clinical covariates.

This is the only cohort in the project that can support the multimodal fusion
claim. BraTS2021 ships no clinical metadata, so a KAN fusing "imaging and
clinical tabular data" has nothing to fuse there.

Clinical covariates: age, sex, and whether gross total resection exceeded 90%.

KPS (Karnofsky performance status) is deliberately excluded. The clinical file
records it as the string "Not Available" for 596 of 671 subjects — 89% missing
— which `notna()` counts as present and is easy to mistake for a complete
column. Imputing 89% of a covariate would manufacture a predictor rather than
measure one.

Note the encoding: GTR is stored as "Y"/"N", not "Yes"/"No", with "Not
Applicable" and "Not Available" as separate missing categories.

Cohort restrictions, both carried over from the IDH1 analysis for the same
reasons:

  * MGMT must be Methylated or Unmethylated. "Indeterminate" and "Not
    Available" are dropped rather than imputed — imputing an outcome invents
    the thing being predicted.
  * Mask provenance must be uniform. Features derived from expert masks and
    from model predictions have systematically different shape statistics, and
    in this cohort provenance also correlates with outcome, so mixing them lets
    a classifier read mask source as a label proxy.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brats_gbm.splits import upenn_split  # noqa: E402

FEATURES = ROOT / "results" / "classification" / "radiomic_features.csv"
CLINICAL = ROOT / "metadata" / "UPENN-GBM_clinical_info_v2.1.csv"
OUT = ROOT / "results" / "classification" / "upenn_mgmt_cohort.csv"
NIFTI_DIR = ROOT / "upenn_nifti"

CLINICAL_COLS = ["age", "gender_m", "gtr_over90"]


def clinical_num(id_str: str) -> int | None:
    m = re.search(r"UPENN-GBM-(\d+)", str(id_str))
    return int(m.group(1)) if m else None


def subject_num(sub_id: str) -> int | None:
    m = re.search(r"sub-(\d+)", str(sub_id))
    return int(m.group(1)) if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(FEATURES))
    ap.add_argument("--clinical", default=str(CLINICAL))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    feat = pd.read_csv(args.features)
    clin = pd.read_csv(args.clinical)
    clin["num"] = clin["ID"].apply(clinical_num)
    feat["num"] = feat["patient_id"].apply(subject_num)

    clin["mgmt_label"] = clin["MGMT"].map({"Methylated": 1, "Unmethylated": 0})
    clin["gtr_over90"] = clin["GTR_over90percent"].map({"Y": 1, "N": 0})

    keep = clin[["num", "mgmt_label", "gtr_over90"]]
    df = feat.merge(keep, on="num", how="inner")

    print(f"imaging features : {len(feat)} subjects")
    print(f"clinical records : {len(clin)} subjects")
    print(f"merged           : {len(df)}")

    labelled = df[df.mgmt_label.notna()].copy()
    print(f"MGMT labelled    : {len(labelled)} "
          f"(dropped {len(df) - len(labelled)} Indeterminate/Not-Available)")

    primary = labelled[labelled.mask_source == "model_predicted"].copy()
    print(f"uniform provenance: {len(primary)} (model_predicted only)")

    train, val, _ = upenn_split(str(NIFTI_DIR))
    overlap = set(primary.patient_id) & (set(train) | set(val))
    if overlap:
        raise AssertionError(
            f"{len(overlap)} subjects were used to fine-tune the segmentor")
    print(f"verified          : 0 overlap with segmentor fine-tuning")

    primary["mgmt_label"] = primary["mgmt_label"].astype(int)
    n_pos = int(primary.mgmt_label.sum())
    print(f"\nfinal cohort: n={len(primary)}, {n_pos} methylated "
          f"({n_pos / len(primary) * 100:.1f}%)")

    missing = primary[CLINICAL_COLS].isna().sum().to_dict()
    print(f"clinical missingness: {missing}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    primary.drop(columns=["num"]).to_csv(args.out, index=False)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
