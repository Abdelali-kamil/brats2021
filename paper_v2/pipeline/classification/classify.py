#!/usr/bin/env python3
"""Amendment 14A: glioma vs other neoplasm, repeated stratified 5-fold CV (10 repeats), the
project's a-priori models; plus a centre+scanner-only reference. Writes results.json + summary."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, "/home/kamilabdelali/brats2021")
from brats_gbm.classification import build_models, cross_validate
from sklearn.pipeline import Pipeline; from sklearn.linear_model import LogisticRegression
D = "/mnt/data1/kamil_research/experiments/classification_africa"
meta = pd.read_csv(f"{D}/africa_meta.csv").set_index("case")
out, lines = {}, []
def report(name, res):
    s = res["summary"]; pb = s["auc_patient_bootstrap"]
    out[name] = s
    lines.append(f"{name:34s} AUC {pb['mean']:.3f} [{pb['ci_low']:.3f}, {pb['ci_high']:.3f}]  bal.acc {s['balanced_accuracy']['mean']:.3f}"
                 f"  sens {s['sensitivity']['mean']:.3f}  spec {s['specificity']['mean']:.3f}  (n={s['n']}, glioma={s['n_positive']})")
    print(lines[-1], flush=True)
for src in sys.argv[1:]:
    f = pd.read_csv(f"{D}/features_{src}.csv").set_index("case").loc[meta.index]
    y = f.y_glioma.values.astype(int); Xf = f.drop(columns=["y_glioma", "mask_source"]).select_dtypes("number")
    Xf = Xf.loc[:, ~Xf.isna().all()]
    for m, model in build_models().items():
        report(f"{src} / {m}", cross_validate(Xf.values.astype(float), y, model))
y = (meta.group == "glioma").astype(int).values
Xc = pd.get_dummies(meta[["center", "scanner"]].astype(str)).values.astype(float)
report("centre+scanner only / logreg", cross_validate(Xc, y, Pipeline([("clf", LogisticRegression(class_weight="balanced", max_iter=5000, C=0.1))])))
json.dump(out, open(f"{D}/results_{'_'.join(sys.argv[1:])}.json", "w"), indent=1)
open(f"{D}/summary_{'_'.join(sys.argv[1:])}.txt", "w").write("\n".join(lines) + "\n")
