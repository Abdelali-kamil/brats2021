"""Peer-review analysis (2026-10-10): re-runs the BraTS-Africa glioma vs other-neoplasm classification (logistic regression,
repeated stratified 5-fold CV, as classify.py) for each mask source to obtain the out-of-fold probabilities, checks that the
published AUCs are reproduced, and reports paired patient-level bootstrap (2,000) intervals for AUC differences.
Run from experiments/classification_africa (needs the features_*.csv files there)."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/kamilabdelali/brats2021")
from brats_gbm.classification import build_models, cross_validate
from sklearn.metrics import roc_auc_score
D="/mnt/data1/kamil_research/experiments/classification_africa"
meta=pd.read_csv(f"{D}/africa_meta.csv").set_index("case")
srcs=["ours_runC","expert","ours_runD_rot","nnunet","swin","segresnet","unetr","unet3d"]
oof={}; y=None
for src in srcs:
    f=pd.read_csv(f"{D}/features_{src}.csv").set_index("case").loc[meta.index]
    yy=f.y_glioma.values.astype(int); X=f.drop(columns=["y_glioma","mask_source"]).select_dtypes("number"); X=X.loc[:,~X.isna().all()]
    y=yy if y is None else y; assert (y==yy).all()
    for m,model in [("logreg",build_models()["logreg"])]:
        r=cross_validate(X.values.astype(float), yy, model); oof[(src,m)]=r["oof_prob"]
        print(f"{src:14s} {m:13s} AUC {roc_auc_score(yy,r['oof_prob']):.3f}", flush=True)
np.save("oof_logreg.npy", {"y":y,"oof":oof}, allow_pickle=True)
rng=np.random.default_rng(0); idx=rng.integers(0,len(y),(2000,len(y)))
def paired(a,b,m):
    pa,pb=oof[(a,m)],oof[(b,m)]; d=roc_auc_score(y,pa)-roc_auc_score(y,pb); bs=[]
    for s in idx:
        if len(np.unique(y[s]))<2: continue
        bs.append(roc_auc_score(y[s],pa[s])-roc_auc_score(y[s],pb[s]))
    lo,hi=np.percentile(bs,[2.5,97.5]); print(f"  {a:13s} - {b:13s} [{m}] {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
print("\npaired patient-level bootstrap (2,000) of AUC differences")
for m in ("logreg",):
    for a,b in [("expert","ours_runC"),("ours_runD_rot","ours_runC"),("nnunet","ours_runC"),("swin","ours_runC"),("segresnet","ours_runC"),("unetr","ours_runC"),("unet3d","ours_runC")]:
        paired(a,b,m)
