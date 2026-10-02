#!/usr/bin/env python3
"""Save sigmoid probabilities (<case>.npz, float16) for a case list, WITHOUT scoring.
Same model construction (architecture/norm/input normalisation from the checkpoint
config), preprocessing and inference (128^3, stride 64, Gaussian, 8-flip TTA) as
evaluate_brats.py. Used so TEST predictions exist before any selection is made.
  infer_probs.py CKPT CASES_FILE OUT_DIR"""
import sys, pathlib, numpy as np, nibabel as nib, torch
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from brats_gbm.data.image import irm_min_max_preprocess, zscore_normalise
from brats_gbm.eval.inference import sliding_window_predict
from brats_gbm.model import WaveletUNetPlusPlus
ck_path, cases_file, out = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
orient = sys.argv[4] if len(sys.argv) > 4 else "nib"   # start model's BraTS-VAL-chosen mode (Amendment 10)
ck = torch.load(ck_path, map_location="cuda", weights_only=False); sd = ck["model_state"]; cfg = ck.get("config", {}) or {}
down = cfg.get("downsample") or {4: "dwt", 8: "dwt3d"}[sd["conv1_0.conv.0.weight"].shape[1] // sd["conv0_0.conv.0.weight"].shape[0]]
bf = cfg.get("base_filters") or int(sd["conv0_0.conv.0.weight"].shape[0])
nm = cfg.get("norm") or ("batch" if any(k.endswith("running_mean") for k in sd) else "instance")
NORM_IN = zscore_normalise if (cfg.get("normalisation") or "minmax") == "zscore" else irm_min_max_preprocess
m = WaveletUNetPlusPlus(4, 3, downsample=down, base_filters=bf, norm=nm,
                        deep_supervision=any(k.startswith("ds_heads.") for k in sd)).cuda()
m.load_state_dict(sd, strict=True); m.eval()
DATA = ROOT / "data"
print(f"infer_probs: {down} w{bf} {nm} input={cfg.get('normalisation') or 'minmax'} axis={orient} -> {out}", flush=True)
with torch.no_grad():
    for c in [l.strip() for l in open(cases_file) if l.strip()]:
        if (out / f"{c}.npz").exists(): continue
        img = np.stack([NORM_IN(nib.load(str(DATA / c / f"{c}_{k}.nii.gz")).get_fdata().astype(np.float32))
                        for k in ("t1", "t1ce", "t2", "flair")])
        prob = sliding_window_predict(m, img, "cuda", use_tta=True)
        if orient == "both":
            ps = sliding_window_predict(m, np.ascontiguousarray(img.transpose(0, 3, 2, 1)), "cuda", use_tta=True)
            prob = (np.asarray(prob) + np.asarray(ps).transpose(0, 3, 2, 1)) / 2.0
        np.savez_compressed(out / f"{c}.npz", prob=np.asarray(prob, dtype=np.float16)); print(c, flush=True)
