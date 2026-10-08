#!/usr/bin/env python3
"""Amendment 19: extra-baseline inference on OUR 125-case BraTS held-out test set (copy of the Swin UNETR script;
only the network and its crop size differ).

Follows the pinned upstream recipe (BRATS21/test.py @ 21ed8e57) exactly, with
only the changes needed to run it on this split and this MONAI version:

  * transforms: upstream get_loader(test_mode=True) -- LoadImaged +
    NormalizeIntensityd(nonzero, channel_wise); no resampling, so predictions
    are on the native 240x240x155 grid of the ground truth.
  * roi 96^3 (the size the model was TRAINED at; upstream test.py defaults to
    128, which would not match main.py's 96).
  * sliding window, overlap 0.6, sw_batch 1, sigmoid > 0.5, no TTA (upstream).
  * label map as upstream: WT->2, TC->1, ET->4 (BraTS raw convention).
  * affine is taken from the case's GT seg file rather than
    batch["image_meta_dict"], which MONAI 1.6 MetaTensors no longer provide.
  * iterates the dataset directly (no DataLoader workers), so the SSH nofile
    limit cannot kill it.

Usage:
  python infer_test125.py --ckpt model.pt --out /path/to/pred_dir
"""
import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import torch
from functools import partial

REPO = "/home/kamilabdelali/brats2021/baselines/extra3/BRATS21"
sys.path.insert(0, REPO)
from utils.data_utils import get_loader  # noqa: E402

from monai.inferers import sliding_window_inference  # noqa: E402
from models_extra3 import build_model, ROI  # noqa: E402

DATA_DIR = "/mnt/data1/kamil_research/data/brats2021"
IDS = "/home/kamilabdelali/brats2021/baselines/common/test_ids.txt"
CHANNELS = ("flair", "t1ce", "t1", "t2")  # training order, see RUN_METADATA.txt


def write_datalist(path, ids):
    rows = [{"fold": 0,
             "image": [f"{c}/{c}_{ch}.nii.gz" for ch in CHANNELS],
             "label": f"{c}/{c}_seg.nii.gz"} for c in ids]
    with open(path, "w") as f:
        json.dump({"training": rows}, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--overlap", type=float, default=0.6)
    ap.add_argument("--model_name", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ids = [l.strip() for l in open(IDS) if l.strip()]
    assert len(ids) == 125, len(ids)
    dl_path = os.path.join(args.out, "datalist_test125.json")
    write_datalist(dl_path, ids)

    largs = SimpleNamespace(data_dir=DATA_DIR, json_list=dl_path, fold=0,
                            test_mode=True, distributed=False, workers=0,
                            roi_x=ROI[args.model_name][0], roi_y=ROI[args.model_name][1], roi_z=ROI[args.model_name][2])
    ds = get_loader(largs).dataset
    assert len(ds) == 125

    model = build_model(args.model_name)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["state_dict"])
    model.eval().cuda()
    print(f"checkpoint {args.ckpt}: epoch {ck.get('epoch')} best_acc {ck.get('best_acc')}", flush=True)

    infer = partial(sliding_window_inference, roi_size=list(ROI[args.model_name]),
                    sw_batch_size=1, predictor=model, overlap=args.overlap)

    t0 = time.time()
    with torch.no_grad():
        for i, cid in enumerate(ids):
            out_p = os.path.join(args.out, cid + ".nii.gz")
            if os.path.exists(out_p):
                continue
            item = ds[i]
            src = str(item["image"].meta["filename_or_obj"])
            assert cid in src, (cid, src)
            image = torch.as_tensor(item["image"])[None].cuda()
            seg = (torch.sigmoid(infer(image))[0] > 0.5).cpu().numpy()
            lab = np.zeros(seg.shape[1:], np.uint8)
            lab[seg[1]] = 2
            lab[seg[0]] = 1
            lab[seg[2]] = 4
            gt = nib.load(os.path.join(DATA_DIR, cid, f"{cid}_seg.nii.gz"))
            assert lab.shape == gt.shape, (cid, lab.shape, gt.shape)
            nib.save(nib.Nifti1Image(lab, gt.affine), out_p)
            print(f"[{i + 1}/125] {cid}  {time.time() - t0:.0f}s", flush=True)
    print("Finished inference!", flush=True)


if __name__ == "__main__":
    main()
