"""Shared cross-validation machinery for the molecular-marker classifiers.

Both the UPenn IDH1 and BraTS MGMT analyses run through this, so their
protocols are identical by construction and their numbers are comparable.

Protocol
--------
Repeated stratified 5-fold cross-validation (10 repeats). Within each training
fold the decision threshold is fitted on *cross-fitted* scores, never on the
model's own resubstitution predictions — a random forest separates its training
data almost perfectly, so a resubstitution threshold sits near 1.0 and no
held-out case ever reaches it, which shows up as zero sensitivity that says
more about the threshold than the model.

Repeating the split ten times matters whenever the positive count is small: one
unlucky partition can move AUC by 0.1, and the spread across repeats is
reported alongside the mean.

Every metric carries a bootstrap confidence interval with patients as the
resampling unit.
"""
from __future__ import annotations

import numpy as np
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

from brats_gbm.eval.stats import bootstrap_ci

N_SPLITS, N_REPEATS, SEED, N_BOOT = 5, 10, 42, 2000

METRIC_KEYS = ("auc", "balanced_accuracy", "accuracy", "sensitivity",
               "specificity", "ppv", "f1")


def build_models(seed: int = SEED) -> dict[str, Pipeline]:
    """Two models with class weighting and no hyperparameter search.

    No grid search on purpose: with a few dozen positives it would overfit the
    selection itself, so both models are specified a priori and only the
    decision threshold is fitted.
    """
    return {
        "logreg": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=5000,
                                       C=0.1, random_state=seed)),
        ]),
        "random_forest": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=500, min_samples_leaf=2,
                class_weight="balanced_subsample", random_state=seed, n_jobs=-1)),
        ]),
    }


def youden_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y_true, prob)
    return float(thr[int(np.argmax(tpr - fpr))])


def fit_threshold(model: Pipeline, X_tr: np.ndarray, y_tr: np.ndarray,
                  seed: int = SEED) -> float:
    """Threshold from cross-fitted training-fold scores; test fold untouched."""
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    try:
        prob_cv = cross_val_predict(model, X_tr, y_tr, cv=inner,
                                    method="predict_proba", n_jobs=1)[:, 1]
    except ValueError:
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


def cross_validate(X: np.ndarray, y: np.ndarray, model: Pipeline,
                   seed: int = SEED) -> dict:
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                 random_state=seed)
    oof_prob = np.zeros((N_REPEATS, len(y)))
    oof_pred = np.zeros((N_REPEATS, len(y)), dtype=int)

    for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        repeat = i // N_SPLITS
        model.fit(X[train_idx], y[train_idx])
        prob_test = model.predict_proba(X[test_idx])[:, 1]
        thr = fit_threshold(model, X[train_idx], y[train_idx], seed)
        oof_prob[repeat, test_idx] = prob_test
        oof_pred[repeat, test_idx] = (prob_test >= thr).astype(int)

    per_repeat = [point_metrics(y, oof_prob[r], oof_pred[r]) for r in range(N_REPEATS)]
    mean_prob = oof_prob.mean(axis=0)

    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(N_BOOT):
        s = rng.integers(0, len(y), len(y))
        if len(np.unique(y[s])) > 1:
            boot.append(roc_auc_score(y[s], mean_prob[s]))

    summary: dict = {}
    for key in METRIC_KEYS:
        vals = [m[key] for m in per_repeat]
        p, lo, hi = bootstrap_ci(vals)
        summary[key] = {"mean": p, "ci_low": lo, "ci_high": hi,
                        "across_repeat_sd": float(np.nanstd(vals))}

    summary["auc_patient_bootstrap"] = {
        "mean": float(roc_auc_score(y, mean_prob)),
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
    }
    summary["confusion_totals"] = {
        k: int(np.sum([m[k] for m in per_repeat])) for k in ("tn", "fp", "fn", "tp")}
    summary["degenerate"] = bool(summary["sensitivity"]["mean"] < 1e-9)
    summary["n"] = int(len(y))
    summary["n_positive"] = int(y.sum())
    return {"summary": summary, "oof_prob": mean_prob}


def print_model_summary(name: str, s: dict) -> None:
    flag = "  [DEGENERATE: never predicts positive]" if s["degenerate"] else ""
    print(f"\n  {name}{flag}")
    pb = s["auc_patient_bootstrap"]
    print(f"    {'auc (patient CI)':<18} {pb['mean']:.3f}  "
          f"95% CI [{pb['ci_low']:.3f}, {pb['ci_high']:.3f}]")
    for key in ("balanced_accuracy", "accuracy", "sensitivity", "specificity",
                "ppv", "f1"):
        m = s[key]
        print(f"    {key:<18} {m['mean']:.3f}  "
              f"95% CI [{m['ci_low']:.3f}, {m['ci_high']:.3f}]")
    if pb["ci_low"] <= 0.5:
        print("    NOTE: AUC interval includes 0.5 — not distinguishable from chance.")
