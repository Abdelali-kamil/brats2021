#!/usr/bin/env python3
"""Swin UNETR on BraTS-Africa: same recipe as its BraTS test scoring (upstream
test.py: NormalizeIntensity nonzero channel-wise, roi 96^3, overlap 0.6, sigmoid>0.5,
no TTA, fp32), channel order flair, t1ce, t1, t2; labels NCR1/ED2/ET4."""
import os, numpy as np, nibabel as nib, pandas as pd, torch
from functools import partial
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR
from monai.transforms import NormalizeIntensity
X = "/mnt/data1/kamil_research/experiments/external_africa"; OUT = f"{X}/pred_swin"; os.makedirs(OUT, exist_ok=True)
m = SwinUNETR(in_channels=4, out_channels=3, feature_size=48, drop_rate=0.0, attn_drop_rate=0.0, dropout_path_rate=0.0)
m.load_state_dict(torch.load("/home/kamilabdelali/brats2021/baselines/repos/swin_unetr/SwinUNETR/BRATS21/runs/swin_unetr_frozen/model.pt",
                             map_location="cpu", weights_only=False)["state_dict"]); m = m.cuda().eval()
norm = NormalizeIntensity(nonzero=True, channel_wise=True)
infer = partial(sliding_window_inference, roi_size=[96, 96, 96], sw_batch_size=1, predictor=m, overlap=0.6)
cases = pd.read_csv(f"{X}/cases_gated.csv")
with torch.no_grad():
    for r in cases.itertuples():
        out = f"{OUT}/{r.case}.nii.gz"
        if os.path.exists(out): continue
        raw = np.stack([nib.load(p).get_fdata().astype(np.float32) for p in (r.t2f, r.t1c, r.t1n, r.t2w)])
        seg = (torch.sigmoid(infer(torch.as_tensor(np.asarray(norm(raw)))[None].cuda()))[0] > 0.5).cpu().numpy()
        lab = np.zeros(seg.shape[1:], np.uint8); lab[seg[1]] = 2; lab[seg[0]] = 1; lab[seg[2]] = 4
        g = nib.load(r.seg); nib.save(nib.Nifti1Image(lab, g.affine), out); print(r.case, flush=True)
