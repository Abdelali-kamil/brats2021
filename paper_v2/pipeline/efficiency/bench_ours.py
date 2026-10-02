"""Wavelet U-Net++ (ours): 128^3 window, stride 64, Gaussian blending, 8-flip TTA
(the reported protocol) or --no-tta; val-selected post-processing (WT largest)."""
import sys, argparse, numpy as np, nibabel as nib, torch
sys.path.insert(0, "/mnt/data1/kamil_research/experiments/eval_v2/code")
sys.path.insert(0, "/mnt/data1/kamil_research/experiments/efficiency")
from common import *
from brats_gbm.data.image import irm_min_max_preprocess
from brats_gbm.eval.inference import sliding_window_predict
from brats_gbm.eval import postprocess as pp
from brats_gbm.model import WaveletUNetPlusPlus
ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True)
ap.add_argument("--downsample", default="dwt"); ap.add_argument("--ckpt", default=None)
ap.add_argument("--no-tta", action="store_true"); ap.add_argument("--base-filters", type=int, default=16)
ap.add_argument("--axis-both", action="store_true", help="orientation TTA: average nib order and transposed order")
ap.add_argument("--rot-tta", action="store_true", help="Amendment 12: also average over a 90-degree axial rotation")
a = ap.parse_args()
m = WaveletUNetPlusPlus(4, 3, downsample=a.downsample, base_filters=a.base_filters).cuda().eval()
if a.ckpt:
    sd = torch.load(a.ckpt, map_location="cuda", weights_only=False)["model_state"]
    m.load_state_dict({k: v for k, v in sd.items() if not k.startswith("ds_heads.")})
cnt = Counter(m)
def load(c):
    return np.stack([nib.load(f"{DATA}/{c}/{c}_{s}.nii.gz").get_fdata().astype(np.float32)
                     for s in ("t1", "t1ce", "t2", "flair")])
def predict(img):
    prob = sliding_window_predict(m, img, "cuda", use_tta=not a.no_tta)
    if a.axis_both:
        t = np.ascontiguousarray(img.transpose(0, 3, 2, 1))
        prob = (prob + sliding_window_predict(m, t, "cuda", use_tta=not a.no_tta).transpose(0, 3, 2, 1)) / 2
    return prob
def run(raw):
    img = np.stack([irm_min_max_preprocess(ch) for ch in raw])
    prob = predict(img)
    if a.rot_tta:
        prob = (prob + np.rot90(predict(np.ascontiguousarray(np.rot90(img, 1, axes=(1, 2)))), -1, axes=(1, 2))) / 2
    return pp.postprocess(prob, 0.5, 0.5, 0.5, wt_policy="largest")
with torch.no_grad():
    run(load(WARMUP)); rows = []
    for c in CASES:
        raw = load(c); cnt.n = 0
        _, s, g = timed(lambda: run(raw)); rows.append({"case": c, "sec": s, "peak_gib": g, "n_forward": cnt.n})
params = sum(p.numel() for p in m.parameters())
save(a.name, params, flops_per_forward(m, (1, 4, 128, 128, 128)), (1, 4, 128, 128, 128), rows,
     {"protocol": "128^3 stride 64, gaussian, " + ("no TTA" if a.no_tta else "8-flip TTA") + (", orientation TTA (2 axis orders)" if a.axis_both else "") + (", 90-degree rotation TTA" if a.rot_tta else "") + f", F={a.base_filters}, WT largest"})
