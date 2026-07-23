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
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brats_gbm.eval.stats import bootstrap_ci  # noqa: E402
from brats_gbm.splits import upenn_split  # noqa: E402

FEATURES_CSV = ROOT / "results" / "classification" / "radiomic_features.csv"
OUT_DIR = ROOT / "results" / "classification"
NIFTI_DIR = ROOT / "upenn_nifti"

NON_FEATURE_COLS = {"patient_id", "mask_source", "idh1_label", "idh1_raw"}
N_SPLITS, N_REPEATS, SEED = 5, 10, 42


def build_models() -> dict[str, Pipeline]:
    """Two models with class weighting; no hyperparameter search.

    With 17 positives a grid search would overfit the selection itself, so the
    models are fixed a priori and only the decision threshold is fitted, inside
    each training fold.
    """
    return {
        "logreg": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced", max_iter=5000,
                C=0.1, random_state=SEED)),
        ]),
        "random_forest": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=500, min_samples_leaf=2,
                class_weight="balanced_subsample", random_state=SEED, n_jobs=-1)),
        ]),
    }


def youden_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    """Operating point maximising sensitivity + specificity - 1."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y_true, prob)
    return float(thr[int(np.argmax(tpr - fpr))])


def fit_threshold(model: Pipeline, X_tr: np.ndarray, y_tr: np.ndarray) -> float:
    """Choose the decision threshold using cross-fitted training-fold scores.

    Reading the threshold off the model's own fitted predictions would be
    resubstitution, and a random forest separates its training data almost
    perfectly — the resulting threshold sits near 1.0 and no held-out case ever
    reaches it, which manifests as zero sensitivity that says more about the
    threshold than the model. Cross-fitting inside the training fold gives
    honest scores to pick from, so high- and low-variance models compete
    fairly. The held-out fold is still never touched.
    """
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    try:
        prob_cv = cross_val_predict(
            model, X_tr, y_tr, cv=inner, method="predict_proba", n_jobs=1)[:, 1]
    except ValueError:
        # Too few positives to cross-fit; fall back to the class prior.
        return float(np.clip(y_tr.mean(), 0.01, 0.99))
    return youden_threshold(y_tr, prob_cv)


def point_metrics(y: np.ndarray, prob: np.ndarray, pred: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "auc": roc_auc_score(y, prob) if len(np.unique(y)) > 1 else float("nan"),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "accuracy": (tp + tn) / max(1, tp + tn + fp + fn),
        "sensitivity": tp / max(1, tp + fn),
        "specificity": tn / max(1, tn + fp),
        "ppv": tp / (tp + fp) if (tp + fp) else float("nan"),
        "f1": f1_score(y, pred, zero_division=0),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def cross_validate(X: np.ndarray, y: np.ndarray, model: Pipeline) -> dict:
    """Repeated stratified CV returning per-repeat metrics and pooled OOF scores.

    Repeating the 5-fold split 10 times matters here: with 17 positives a
    single split puts 3-4 positives in each test fold, so one unlucky
    partition moves the AUC by 0.1. The spread across repeats is reported.
    """
    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)

    n_repeat_folds = N_SPLITS
    oof_prob = np.zeros((N_REPEATS, len(y)))
    oof_pred = np.zeros((N_REPEATS, len(y)), dtype=int)

    for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        repeat = i // n_repeat_folds
        model.fit(X[train_idx], y[train_idx])
        prob_test = model.predict_proba(X[test_idx])[:, 1]
        thr = fit_threshold(model, X[train_idx], y[train_idx])
        oof_prob[repeat, test_idx] = prob_test
        oof_pred[repeat, test_idx] = (prob_test >= thr).astype(int)

    per_repeat = [point_metrics(y, oof_prob[r], oof_pred[r]) for r in range(N_REPEATS)]

    # Patient-level bootstrap CI on the repeat-averaged out-of-fold scores.
    mean_prob = oof_prob.mean(axis=0)
    rng = np.random.default_rng(SEED)
    boot_auc = []
    for _ in range(2000):
        s = rng.integers(0, len(y), len(y))
        if len(np.unique(y[s])) > 1:
            boot_auc.append(roc_auc_score(y[s], mean_prob[s]))

    summary = {}
    for key in ("auc", "balanced_accuracy", "accuracy", "sensitivity",
                "specificity", "ppv", "f1"):
        vals = [m[key] for m in per_repeat]
        p, lo, hi = bootstrap_ci(vals)
        summary[key] = {"mean": p, "ci_low": lo, "ci_high": hi,
                        "across_repeat_sd": float(np.nanstd(vals))}

    summary["auc_patient_bootstrap"] = {
        "mean": float(roc_auc_score(y, mean_prob)),
        "ci_low": float(np.percentile(boot_auc, 2.5)),
        "ci_high": float(np.percentile(boot_auc, 97.5)),
    }
    summary["confusion_totals"] = {
        k: int(np.sum([m[k] for m in per_repeat])) for k in ("tn", "fp", "fn", "tp")
    }
    summary["degenerate"] = bool(summary["sensitivity"]["mean"] < 1e-9)
    return {"summary": summary, "oof_prob": mean_prob}


def run_cohort(df: pd.DataFrame, feature_cols: list[str], name: str) -> dict:
    X = df[feature_cols].to_numpy(float)
    y = df["idh1_label"].to_numpy(int)
    print(f"\n{'=' * 72}\ncohort: {name}")
    print(f"  n={len(y)}  positives={int(y.sum())} ({y.mean() * 100:.1f}%)  "
          f"features={len(feature_cols)}\n{'=' * 72}")

    out = {}
    for model_name, model in build_models().items():
        res = cross_validate(X, y, model)
        s = res["summary"]
        flag = "  [DEGENERATE: never predicts positive]" if s["degenerate"] else ""
        print(f"\n  {model_name}{flag}")
        for key in ("auc", "balanced_accuracy", "sensitivity", "specificity", "f1"):
            m = s[key]
            print(f"    {key:<18} {m['mean']:.3f}  95% CI [{m['ci_low']:.3f}, {m['ci_high']:.3f}]")
        pb = s["auc_patient_bootstrap"]
        print(f"    {'auc (patient CI)':<18} {pb['mean']:.3f}  "
              f"95% CI [{pb['ci_low']:.3f}, {pb['ci_high']:.3f}]")
        out[model_name] = s
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
