#!/usr/bin/env python3
"""XGBoost classification baseline, comparable to the reported models.

Baseline for the paper's comparison table: Chen & Guestrin, *XGBoost: A Scalable
Tree Boosting System*, KDD 2016 (https://github.com/dmlc/xgboost), used as a
gradient-boosted-trees baseline on the same segmentation-derived radiomic feature
tables the random-forest / logistic-regression models use.

Comparability is the whole point, so this re-implements no evaluation: it reuses
`brats_gbm.classification.cross_validate` — the exact repeated-stratified-5-fold
protocol, cross-fitted decision threshold, and patient-level bootstrap CIs used
by the reported models. Only the estimator changes. The reported metric names map
directly onto the paper's classification columns:

    Recall = sensitivity      Precision = ppv       AUC = auc

Two tasks, matching the two classification cohorts in the repo:

  * ``mgmt`` — BraTS MGMT methylation, 577 cases, ~balanced. This is the cohort
    whose precision (~65-85%) matches the paper's classification table.
  * ``idh1`` — UPenn IDH1 mutation, primary = model-predicted masks only
    (17 positives / 416). Reported for completeness; its ~4% prevalence makes
    precision structurally small, so it is NOT the paper table's cohort.

No hyperparameter search, matching the repo's policy: with a few dozen positives
a grid search overfits the selection itself, so a shallow, regularised booster is
fixed a priori and only the decision threshold is fitted. Class imbalance is
handled with ``scale_pos_weight``, XGBoost's analogue of ``class_weight``.

Runs on CPU in seconds from committed feature tables — no GPU or imaging data.
Writes results/baselines/xgboost_classification.json.
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

from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from brats_gbm.classification import SEED, cross_validate, print_model_summary  # noqa: E402

OUT_DIR = ROOT / "results" / "baselines"
NIFTI_DIR = ROOT / "upenn_nifti"

# Per-task cohort definition, copied from the reported classifiers verbatim.
TASKS = {
    "mgmt": {
        "features": ROOT / "results" / "classification" / "brats_mgmt_features.csv",
        "label": "mgmt_label",
        "non_feature": {"case_id", "mask_source", "mgmt_label"},
        "cohort": "all_labelled",   # uniform ground-truth provenance
        "title": "MGMT methylation (BraTS)",
    },
    "idh1": {
        "features": ROOT / "results" / "classification" / "radiomic_features.csv",
        "label": "idh1_label",
        "non_feature": {"patient_id", "mask_source", "idh1_label", "idh1_raw"},
        "cohort": "model_predicted",   # primary cohort, disjoint from fine-tuning
        "title": "IDH1 mutation (UPenn-GBM)",
    },
}


def build_xgb(scale_pos_weight: float, seed: int = SEED) -> Pipeline:
    """XGBoost fixed a priori; only ``scale_pos_weight`` reflects the cohort."""
    from xgboost import XGBClassifier

    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            min_child_weight=1.0, scale_pos_weight=scale_pos_weight,
            objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", n_jobs=-1, random_state=seed,
        )),
    ])


def evaluate(X: np.ndarray, y: np.ndarray, label: str) -> dict:
    n_pos = int(y.sum())
    spw = ((len(y) - n_pos) / n_pos) if n_pos else 1.0
    print(f"\n{'=' * 72}\n{label}")
    print(f"  n={len(y)}  positives={n_pos} ({y.mean() * 100:.1f}%)  "
          f"features={X.shape[1]}  scale_pos_weight={spw:.2f}\n{'=' * 72}")
    res = cross_validate(X, y, build_xgb(spw))
    print_model_summary("xgboost", res["summary"])
    return res["summary"]


def load_task(task: str) -> dict:
    cfg = TASKS[task]
    df = pd.read_csv(cfg["features"])
    labelled = df[df[cfg["label"]].notna()].copy()
    labelled[cfg["label"]] = labelled[cfg["label"]].astype(int)
    feats = [c for c in df.columns if c not in cfg["non_feature"]
             and pd.api.types.is_numeric_dtype(df[c])]

    print(f"\n### task: {cfg['title']}")
    print(f"feature table : {cfg['features']}")
    print(f"labelled      : {len(labelled)}  |  features: {len(feats)}")

    out = {"task": cfg["title"], "features_csv": str(cfg["features"]),
           "n_features": len(feats)}

    if cfg["cohort"] == "model_predicted":
        primary = labelled[labelled.mask_source == "model_predicted"].reset_index(drop=True)
        # Same disjointness guard as train_classifier_idh1.py, tolerant of a
        # checkout without upenn_nifti/ (the property holds by construction).
        try:
            from brats_gbm.splits import upenn_split
            tr, va, _ = upenn_split(str(NIFTI_DIR))
            overlap = set(primary.patient_id) & (set(tr) | set(va))
            if overlap:
                raise AssertionError(f"{len(overlap)} primary subjects overlap fine-tuning")
            print(f"verified: primary cohort disjoint from segmentor fine-tuning")
        except (FileNotFoundError, OSError):
            print("note: upenn_nifti/ absent — disjointness holds by construction, live check skipped")
        X = primary[feats].to_numpy(float)
        y = primary[cfg["label"]].to_numpy(int)
        out["cohort"] = "primary (model-predicted masks only)"
        out["primary"] = evaluate(X, y, "PRIMARY — model-predicted masks only")
        Xm = labelled[feats].to_numpy(float)
        ym = labelled[cfg["label"]].to_numpy(int)
        out["sensitivity_mixed"] = evaluate(Xm, ym, "SENSITIVITY — mixed provenance")
    else:
        X = labelled[feats].to_numpy(float)
        y = labelled[cfg["label"]].to_numpy(int)
        out["cohort"] = "all labelled (uniform provenance)"
        out["primary"] = evaluate(X, y, cfg["title"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["mgmt", "idh1", "both"], default="both")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    tasks = ["mgmt", "idh1"] if args.task == "both" else [args.task]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "method": "XGBoost",
        "modality": "T (tabular radiomic features)",
        "paper": "Chen & Guestrin, XGBoost: A Scalable Tree Boosting System, KDD 2016",
        "code": "https://github.com/dmlc/xgboost",
        "protocol": "brats_gbm.classification.cross_validate — repeated (10x) "
                    "stratified 5-fold; Recall=sensitivity, Precision=ppv, AUC=auc",
        "tasks": {t: load_task(t) for t in tasks},
    }
    out_path = out_dir / "xgboost_classification.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
