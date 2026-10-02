"""Shared: GT regions for BraTS 2023 labels (NCR=1, SNFH=2, ET=3), per-method
prediction conventions, and scoring with the SAME score_case/summarise_segmentation."""
import sys, numpy as np, nibabel as nib, pandas as pd
sys.path.insert(0, "/mnt/data1/kamil_research/experiments/eval_v2/code")
from brats_gbm.eval.metrics import score_case
from brats_gbm.eval.stats import summarise_segmentation, print_summary
X = "/mnt/data1/kamil_research/experiments/external_africa"
def gt_regions(path):
    s = np.asarray(nib.load(path).dataobj).astype(np.int16)
    et = s == 3; tc = et | (s == 1); return np.stack([et, tc, tc | (s == 2)])
def pred_regions(lab, convention):
    if convention == "brats21":        # NCR 1, ED 2, ET 4 (ours after postprocess, Swin)
        et = lab == 4; tc = et | (lab == 1); return np.stack([et, tc, tc | (lab == 2)])
    if convention == "nnunet_regions":  # Dataset137: WT={1,2,3}, TC={2,3}, ET={3}
        return np.stack([lab == 3, (lab == 2) | (lab == 3), lab >= 1])
    raise ValueError(convention)
def finish(rows, tag):
    df = pd.DataFrame(rows); df.to_csv(f"{X}/results/per_case_{tag}.csv", index=False)
    for g, sub in df.groupby("group"):
        s = summarise_segmentation(sub.drop(columns=["group"]), label=f"{tag}_{g}")
        s.to_csv(f"{X}/results/summary_{tag}_{g}.csv", index=False); print_summary(s, f"{tag} [{g}] (n={len(sub)})")
