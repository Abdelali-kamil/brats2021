"""Score Swin UNETR predictions on OUR 125-case BraTS held-out test set, using
the exact metric path that produced the Wavelet U-Net++ and nnU-Net numbers
(brats_gbm.eval.score_case + summarise_segmentation, 95% bootstrap CIs).

Mirror of /mnt/data1/kamil_research/baselines/nnunet/eval/score_nnunet_test.py;
the only difference is the prediction label convention. infer_test125.py writes
upstream test.py's BraTS-raw labels (NCR=1, ED=2, ET=4), same as the GT:
ET={4}, TC={1,4}, WT={1,2,4}.

Usage: python score_swin_test.py PRED_DIR IDS_FILE OUT_PREFIX LABEL
"""
import sys, os
import numpy as np, nibabel as nib, pandas as pd
REPO = "/mnt/data1/kamil_research/experiments/repo_aug_cosine"
sys.path.insert(0, REPO)
from brats_gbm.eval.metrics import score_case
from brats_gbm.eval.stats import summarise_segmentation

PRED_DIR, IDS_FILE, OUT_PREFIX, LABEL = sys.argv[1:5]
GT_ROOT = "/mnt/data1/kamil_research/data/brats2021"


def regions(seg):
    et = seg == 4
    tc = et | (seg == 1)
    wt = tc | (seg == 2)
    return np.stack([et, tc, wt])


ids = [l.strip() for l in open(IDS_FILE) if l.strip()]
rows = []
for i, cid in enumerate(ids, 1):
    pred = nib.load(os.path.join(PRED_DIR, cid + ".nii.gz")).get_fdata().astype(np.int16)
    seg = nib.load(os.path.join(GT_ROOT, cid, cid + "_seg.nii.gz")).get_fdata().astype(np.int16)
    assert pred.shape == seg.shape, f"{cid}: pred {pred.shape} vs gt {seg.shape}"
    s = score_case(regions(pred), regions(seg))
    s["Patient_ID"] = cid
    rows.append(s)
    if i % 25 == 0:
        print(f"  [{i}/{len(ids)}] {cid} ET {s['Dice_ET']:.3f} TC {s['Dice_TC']:.3f} WT {s['Dice_WT']:.3f}")

df = pd.DataFrame(rows)
df.to_csv(OUT_PREFIX + "_per_case.csv", index=False)
summary = summarise_segmentation(df, label=LABEL)
summary.to_csv(OUT_PREFIX + "_summary.csv", index=False)
print("\n=== %s, BraTS held-out TEST (n=%d), our metric ===" % (LABEL, len(df)))
print(summary.to_string(index=False))
print("\nWrote", OUT_PREFIX + "_summary.csv")
