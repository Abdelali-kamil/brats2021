"""Swin UNETR: upstream BRATS21 test.py recipe as scored (roi 96^3, overlap 0.6,
sigmoid>0.5, no TTA, fp32), weights = model.pt (ep199)."""
import sys, numpy as np, nibabel as nib, torch
sys.path.insert(0, "/mnt/data1/kamil_research/experiments/efficiency")
from common import *
from functools import partial
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR
from monai.transforms import NormalizeIntensity
m = SwinUNETR(in_channels=4, out_channels=3, feature_size=48, drop_rate=0.0, attn_drop_rate=0.0,
              dropout_path_rate=0.0, use_checkpoint=False)
m.load_state_dict(torch.load("/home/kamilabdelali/brats2021/baselines/repos/swin_unetr/SwinUNETR/BRATS21/runs/swin_unetr_frozen/model.pt",
                             map_location="cpu", weights_only=False)["state_dict"])
m = m.cuda().eval(); cnt = Counter(m); norm = NormalizeIntensity(nonzero=True, channel_wise=True)
infer = partial(sliding_window_inference, roi_size=[96, 96, 96], sw_batch_size=1, predictor=m, overlap=0.6)
def load(c):
    return np.stack([nib.load(f"{DATA}/{c}/{c}_{s}.nii.gz").get_fdata().astype(np.float32)
                     for s in ("flair", "t1ce", "t1", "t2")])
def run(raw):
    x = torch.as_tensor(np.asarray(norm(raw)))[None].cuda()
    seg = (torch.sigmoid(infer(x))[0] > 0.5).cpu().numpy()
    lab = np.zeros(seg.shape[1:], np.uint8); lab[seg[1]] = 2; lab[seg[0]] = 1; lab[seg[2]] = 4
    return lab
with torch.no_grad():
    run(load(WARMUP)); rows = []
    for c in CASES:
        raw = load(c); cnt.n = 0
        _, s, g = timed(lambda: run(raw)); rows.append({"case": c, "sec": s, "peak_gib": g, "n_forward": cnt.n})
save("swin_unetr", sum(p.numel() for p in m.parameters()), flops_per_forward(m, (1, 4, 96, 96, 96)),
     (1, 4, 96, 96, 96), rows, {"protocol": "roi 96^3, overlap 0.6, no TTA, fp32"})
