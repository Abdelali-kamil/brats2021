#!/usr/bin/env python3
"""Score a directory of predicted label maps (<case>.nii.gz) on BraTS-Africa.
  score_labelmaps.py TAG PRED_DIR CONVENTION(brats21|nnunet_regions)"""
import sys, os, numpy as np, nibabel as nib, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_ext import *
tag, pdir, conv = sys.argv[1:4]
cases = pd.read_csv(f"{X}/cases_gated.csv"); cases = cases[(~cases.overlap_flag) & (cases.group != "unlisted")]
os.makedirs(f"{X}/results", exist_ok=True); rows = []
for r in cases.itertuples():
    p = nib.load(f"{pdir}/{r.case}.nii.gz"); lab = np.asarray(p.dataobj).astype(np.int16)
    assert lab.shape == nib.load(r.seg).shape, r.case
    s = score_case(pred_regions(lab, conv), gt_regions(r.seg)); s.update(Patient_ID=r.case, group=r.group); rows.append(s)
finish(rows, tag)
