import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import roc_auc_score, f1_score, recall_score, confusion_matrix, balanced_accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Optional: XGBoost
USE_XGB = True
if USE_XGB:
    from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE

RANDOM_STATE = 42
TARGET_COL = "TARGET_COL"   # <-- replace
ID_COL = "ID_COL"           # <-- replace

# 1) Load
df = pd.read_csv("features.csv")

# 2) Basic checks
assert TARGET_COL in df.columns, "Target column missing"
assert ID_COL in df.columns, "ID column missing"

y = df[TARGET_COL].astype(int).values
X = df.drop(columns=[TARGET_COL, ID_COL])

# Keep numeric columns only for now
X = X.select_dtypes(include=[np.number]).copy()

# 3) Define pipelines + grids
pipelines = {}

# Baseline Logistic
pipelines["logreg_balanced"] = (
    Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("select", SelectKBest(score_func=mutual_info_classif)),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE))
    ]),
    {
        "select__k": [10, 20, 30, "all"],
        "clf__C": [0.01, 0.1, 1, 10]
    }
)

# SVM RBF + class weight
pipelines["svm_rbf_balanced"] = (
    Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("select", SelectKBest(score_func=mutual_info_classif)),
        ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE))
    ]),
    {
        "select__k": [10, 20, 30, "all"],
        "clf__C": [0.1, 1, 10, 50],
        "clf__gamma": ["scale", 0.1, 0.01, 0.001]
    }
)

# Random Forest
pipelines["rf_balanced"] = (
    Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("select", SelectKBest(score_func=mutual_info_classif)),
        ("clf", RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE))
    ]),
    {
        "select__k": [10, 20, 30, "all"],
        "clf__n_estimators": [200, 500],
        "clf__max_depth": [None, 5, 10],
        "clf__min_samples_split": [2, 5, 10]
    }
)

# SVM + SMOTE variant
pipelines["svm_rbf_smote"] = (
    ImbPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("select", SelectKBest(score_func=mutual_info_classif)),
        ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE))
    ]),
    {
        "select__k": [10, 20, 30, "all"],
        "clf__C": [0.1, 1, 10, 50],
        "clf__gamma": ["scale", 0.1, 0.01, 0.001]
    }
)

if USE_XGB:
    pipelines["xgb"] = (
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("select", SelectKBest(score_func=mutual_info_classif)),
            ("clf", XGBClassifier(
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                n_estimators=300
            ))
        ]),
        {
            "select__k": [10, 20, 30, "all"],
            "clf__max_depth": [3, 5, 7],
            "clf__learning_rate": [0.01, 0.05, 0.1],
            "clf__subsample": [0.8, 1.0],
            "clf__colsample_bytree": [0.8, 1.0]
        }
    )

# 4) Nested CV
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

results = []
oof_preds = {}  # out-of-fold predictions for each model
oof_true = np.zeros(len(y), dtype=int)

for model_name in pipelines:
    oof_preds[model_name] = np.zeros(len(y), dtype=float)

for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    oof_true[test_idx] = y_test

    for model_name, (pipe, grid) in pipelines.items():
        gs = GridSearchCV(
            estimator=pipe,
            param_grid=grid,
            scoring="roc_auc",
            cv=inner_cv,
            n_jobs=-1,
            refit=True
        )
        gs.fit(X_train, y_train)
        proba = gs.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        auc = roc_auc_score(y_test, proba)
        f1 = f1_score(y_test, pred)
        sens = recall_score(y_test, pred)  # recall of positive class
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        bacc = balanced_accuracy_score(y_test, pred)

        results.append({
            "fold": fold,
            "model": model_name,
            "auc": auc,
            "f1": f1,
            "sensitivity": sens,
            "specificity": spec,
            "balanced_acc": bacc
        })

        oof_preds[model_name][test_idx] = proba

res_df = pd.DataFrame(results)

# 5) Summary
summary = res_df.groupby("model")[["auc","f1","sensitivity","specificity","balanced_acc"]].agg(["mean","std"])
print(summary)

# 6) Bootstrap CI for AUC difference (best vs baseline)
def bootstrap_auc_diff(y_true, p_new, p_old, n_boot=2000, random_state=42):
    rng = np.random.default_rng(random_state)
    diffs = []
    n = len(y_true)
    idx = np.arange(n)
    for _ in range(n_boot):
        s = rng.choice(idx, size=n, replace=True)
        if len(np.unique(y_true[s])) < 2:
            continue
        d = roc_auc_score(y_true[s], p_new[s]) - roc_auc_score(y_true[s], p_old[s])
        diffs.append(d)
    diffs = np.array(diffs)
    return diffs.mean(), np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)

# choose best model by mean fold AUC
mean_auc = res_df.groupby("model")["auc"].mean().sort_values(ascending=False)
best_model = mean_auc.index[0]
baseline_model = "logreg_balanced"

mean_diff, ci_low, ci_high = bootstrap_auc_diff(
    y_true=oof_true,
    p_new=oof_preds[best_model],
    p_old=oof_preds[baseline_model],
    n_boot=2000,
    random_state=RANDOM_STATE
)

print(f"Best model: {best_model}")
print(f"AUC diff vs baseline: {mean_diff:.4f} (95% CI {ci_low:.4f}, {ci_high:.4f})")