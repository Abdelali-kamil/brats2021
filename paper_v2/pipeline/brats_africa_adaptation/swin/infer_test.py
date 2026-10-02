"""Adapted Swin UNETR on TEST 35: same inference recipe as its BraTS test (roi 96, overlap 0.6,
sigmoid>0.5, no TTA, fp32), channels flair,t1ce,t1,t2; labels NCR1/ED2/ET4."""
import os, numpy as np, nibabel as nib, torch
from functools import partial
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR
from monai.transforms import NormalizeIntensity
A = "/mnt/data1/kamil_research/experiments/adapt_africa"; S = "/mnt/data1/kamil_research/brats_africa/brats21fmt"
CK = "/home/kamilabdelali/brats2021/baselines/repos/swin_unetr/SwinUNETR/BRATS21/runs/adapt_africa_swin/model.pt"
OUT = f"{A}/swin/pred_test"; os.makedirs(OUT, exist_ok=True)
ck = torch.load(CK, map_location="cpu", weights_only=False); print("adapted Swin epoch", ck.get("epoch"), "best_acc", ck.get("best_acc"))
m = SwinUNETR(in_channels=4, out_channels=3, feature_size=48); m.load_state_dict(ck["state_dict"], strict=True); m = m.cuda().eval()
norm = NormalizeIntensity(nonzero=True, channel_wise=True)
infer = partial(sliding_window_inference, roi_size=[96, 96, 96], sw_batch_size=1, predictor=m, overlap=0.6)
with torch.no_grad():
    for c in [l.strip() for l in open(f"{A}/test_ids.txt") if l.strip()]:
        raw = np.stack([nib.load(f"{S}/{c}/{c}_{k}.nii.gz").get_fdata().astype(np.float32) for k in ("flair", "t1ce", "t1", "t2")])
        seg = (torch.sigmoid(infer(torch.as_tensor(np.asarray(norm(raw)))[None].cuda()))[0] > 0.5).cpu().numpy()
        lab = np.zeros(seg.shape[1:], np.uint8); lab[seg[1]] = 2; lab[seg[0]] = 1; lab[seg[2]] = 4
        nib.save(nib.Nifti1Image(lab, nib.load(f"{S}/{c}/{c}_seg.nii.gz").affine), f"{OUT}/{c}.nii.gz"); print(c, flush=True)
