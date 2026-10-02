#!/usr/bin/env python3
"""Ours on BraTS-Africa: identical inference to the BraTS test (128^3, stride 64,
Gaussian, 8-flip TTA, thr 0.5) and the checkpoint's BraTS-VAL-selected post-processing.
  eval_ours_external.py TAG CKPT WT_POLICY FLOOR_MULT"""
import sys, os, numpy as np, nibabel as nib, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_ext import *
from brats_gbm.data.image import irm_min_max_preprocess, zscore_normalise
from brats_gbm.eval.inference import sliding_window_predict
from brats_gbm.eval import postprocess as pp
from brats_gbm.model import WaveletUNetPlusPlus
tag, ckpt, policy, mult = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
etmin = int(sys.argv[5]) if len(sys.argv) > 5 else 200   # 200 = historical default; 0 = rule off
orient = sys.argv[6] if len(sys.argv) > 6 else "nib"     # BraTS-VAL-chosen inference mode (Amendment 10)
ROT = orient.endswith("+rot"); orient = orient.replace("+rot", "")   # Amendments 12/15: 90-degree rotation TTA
ck = torch.load(ckpt, map_location="cuda", weights_only=False); sd = ck["model_state"]
cfg = ck.get("config", {}) or {}
down = cfg.get("downsample") or {4: "dwt", 8: "dwt3d"}[sd["conv1_0.conv.0.weight"].shape[1] // sd["conv0_0.conv.0.weight"].shape[0]]
bf = cfg.get("base_filters") or int(sd["conv0_0.conv.0.weight"].shape[0])
nm = cfg.get("norm") or ("batch" if any(k.endswith("running_mean") for k in sd) else "instance")
NORM_IN = zscore_normalise if (cfg.get("normalisation") or "minmax") == "zscore" else irm_min_max_preprocess
m = WaveletUNetPlusPlus(4, 3, downsample=down, deep_supervision=any(k.startswith("ds_heads.") for k in sd), base_filters=bf, norm=nm).cuda()
m.load_state_dict(sd, strict=True); m.eval()
pp.MIN_VOXELS = {k: int(round(v * mult)) for k, v in pp.MIN_VOXELS.items()}
cases = pd.read_csv(f"{X}/cases_gated.csv"); cases = cases[(~cases.overlap_flag) & (cases.group != "unlisted")]
os.makedirs(f"{X}/results", exist_ok=True); rows = []
print(f"{tag}: downsample={down} width={bf} norm={nm} input={cfg.get('normalisation') or 'minmax'} policy={policy} x{mult} ET>={etmin} axis={orient}{'+rot' if ROT else ''} cases={len(cases)}", flush=True)
MASKS = f"{X}/results/masks_{tag}"; os.makedirs(MASKS, exist_ok=True)   # predicted label maps (for classification, Amendment 14)
def to_labels(reg):   # regions (ET, TC, WT) -> BraTS 2021 labels: 4 ET, 1 NCR (TC minus ET), 2 ED (WT minus TC)
    et, tc, wt = (np.asarray(x, bool) for x in reg); lab = np.zeros(et.shape, np.uint8)
    lab[wt] = 2; lab[tc] = 1; lab[et] = 4; return lab
with torch.no_grad():
    for i, r in enumerate(cases.itertuples(), 1):
        mp = f"{MASKS}/{r.case}.nii.gz"
        if os.path.exists(mp):   # resume: re-score the saved mask instead of re-predicting
            lab = np.asarray(nib.load(mp).dataobj)
            pred = np.stack([lab == 4, (lab == 4) | (lab == 1), lab > 0])
            s = score_case(pred, gt_regions(r.seg)); s.update(Patient_ID=r.case, group=r.group); rows.append(s)
            print(f"  [{i}/{len(cases)}] {r.case} {r.group} (saved mask) ET {s['Dice_ET']:.3f} TC {s['Dice_TC']:.3f} WT {s['Dice_WT']:.3f}", flush=True)
            continue
        img = np.stack([NORM_IN(nib.load(p).get_fdata().astype(np.float32))
                        for p in (r.t1n, r.t1c, r.t2w, r.t2f)])        # BraTS order t1, t1ce, t2, flair
        def predict(img):
            prob = sliding_window_predict(m, img, "cuda", use_tta=True)
            if orient == "both":   # orientation TTA: also predict in training (z,y,x) order and average
                ps = sliding_window_predict(m, np.ascontiguousarray(img.transpose(0, 3, 2, 1)), "cuda", use_tta=True)
                prob = (np.asarray(prob) + np.asarray(ps).transpose(0, 3, 2, 1)) / 2.0
            return np.asarray(prob)
        prob = predict(img)
        if ROT:
            prob = (prob + np.rot90(predict(np.ascontiguousarray(np.rot90(img, 1, axes=(1, 2)))), -1, axes=(1, 2))) / 2.0
        pred = pp.postprocess(prob, 0.5, 0.5, 0.5, wt_policy=policy, et_policy="none" if etmin == 0 else "min_volume", et_min_volume=max(etmin, 1))
        pred = np.asarray(pred).astype(bool); ref = nib.load(r.seg)
        nib.save(nib.Nifti1Image(to_labels(pred), ref.affine, ref.header), mp + ".tmp.nii.gz"); os.replace(mp + ".tmp.nii.gz", mp)
        s = score_case(pred, gt_regions(r.seg)); s.update(Patient_ID=r.case, group=r.group); rows.append(s)
        print(f"  [{i}/{len(cases)}] {r.case} {r.group} ET {s['Dice_ET']:.3f} TC {s['Dice_TC']:.3f} WT {s['Dice_WT']:.3f}", flush=True)
finish(rows, tag)
