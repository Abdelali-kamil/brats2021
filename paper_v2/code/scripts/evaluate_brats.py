#!/usr/bin/env python3
"""Evaluate the segmentor on BraTS2021 with the current inference pipeline.

Why this exists
---------------
The BraTS per-case scores this project previously reported came from an
archived pipeline whose inference settings were not recorded. They cannot be
reproduced by the current code: on BraTS2021_01628, for instance, the stored
file reports Dice 0.065/0.206/0.203 while the current pipeline predicts nothing
at all above threshold 0.5 (peak probabilities 0.005/0.011/0.487). Whatever
produced those numbers used a different operating point, a different crop, or
different preprocessing.

Numbers that cannot be regenerated should not be published, so this script
recomputes them under one fixed protocol: full-volume
sliding-window inference at 128^3 with 64^3 stride, Gaussian blending, 8-flip
test-time augmentation, and the shared post-processing.

Operating point
---------------
Fixed at 0.5, the value `train_brats.py` validates against. In the historical
two-way split there is no separate partition to tune on — the 251-case partition doubled as the
model-selection set during training — so tuning a threshold here would be
fitting on the data being reported. It is used as specified, not selected.

Preprocessing
-------------
The BraTS training convention: [t1, t1ce, t2, flair] channel order with
percentile-clipped min-max normalisation. See docs/METHODOLOGY.md for why this
matters.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brats_gbm.data.image import irm_min_max_preprocess, zscore_normalise  # noqa: E402
from brats_gbm.eval import postprocess as pp  # noqa: E402
from brats_gbm.eval.inference import sliding_window_predict  # noqa: E402
from brats_gbm.eval.metrics import score_case  # noqa: E402
from brats_gbm.eval.stats import print_summary, summarise_segmentation  # noqa: E402
from brats_gbm.model import WaveletUNetPlusPlus  # noqa: E402
from brats_gbm.splits import brats_split  # noqa: E402

DATA = ROOT / "data"
CKPT = ROOT / "checkpoints" / "segmentor_epoch_650.pth"
OUT_DIR = ROOT / "results" / "brats"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BRATS_ORDER = ("t1", "t1ce", "t2", "flair")
THR = 0.5
ET_LABEL, NCR_LABEL, ED_LABEL = 4, 1, 2


# Input normalisation is a property of the checkpoint (config["normalisation"]);
# set in main() before any case is loaded. "minmax" = the published model.
NORMALISER = irm_min_max_preprocess


def load_case(case_id: str) -> tuple[np.ndarray, np.ndarray]:
    d = DATA / case_id
    img = np.stack([
        NORMALISER(
            nib.load(str(d / f"{case_id}_{m}.nii.gz")).get_fdata().astype(np.float32))
        for m in BRATS_ORDER
    ])
    seg = nib.load(str(d / f"{case_id}_seg.nii.gz")).get_fdata().astype(np.float32)
    et = seg == ET_LABEL
    tc = et | (seg == NCR_LABEL)
    wt = tc | (seg == ED_LABEL)
    return img, np.stack([et, tc, wt])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="internal_validation",
                    choices=["internal_validation", "train", "all"])
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate only the first N cases (for a quick check)")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--tag", default=None)
    ap.add_argument("--wt-policy", default="components", choices=("components", "largest"),
                    help="whole-tumour component policy (see brats_gbm/eval/postprocess.py)")
    ap.add_argument("--floor-mult", type=float, default=1.0,
                    help="scale the per-region component floors. The shipped "
                         "values (ET 5 / TC 20 / WT 50 voxels) were not tuned "
                         "for BraTS.")
    ap.add_argument("--cases", default=None, metavar="FILE_OR_LIST",
                    help="restrict to these case ids: a comma-separated list or "
                         "a file with one id per line. --limit takes the first N, "
                         "which is no use when you want specific cases.")
    ap.add_argument("--save-probs", default=None, metavar="DIR",
                    help="also write the raw sigmoid probabilities per case as "
                         ".npz. Post-processing can then be re-tuned without "
                         "re-running inference, which is what blocked the HD95 "
                         "work on 2026-09-17.")
    ap.add_argument("--et-policy", default="min_volume", choices=("min_volume", "none"),
                    help="BraTS ET small-volume rule (default = historical behaviour)")
    ap.add_argument("--et-min-volume", type=int, default=200)
    ap.add_argument("--axis-order", default="nib", choices=("nib", "sitk", "both"),
                    help="nib = historical (nibabel x,y,z order); sitk = the (z,y,x) order the "
                         "training loader (SimpleITK) feeds the network. Probabilities and scores are "
                         "always returned/computed in nibabel order.")
    ap.add_argument("--rot-tta", action="store_true",
                    help="Amendment 12: also average over a 90-degree in-plane (axial) rotation; "
                         "with the 8 flips this covers all 16 axial symmetries x slice flip.")
    ap.add_argument("--step", type=int, default=64,
                    help="sliding-window stride per axis (128^3 window). 64 = the "
                         "reported protocol; 32 = 75%% overlap.")
    ap.add_argument("--checkpoint", default=str(CKPT),
                    help="checkpoint to evaluate; the historical default no longer exists")
    args = ap.parse_args()

    train_ids, val_ids = brats_split(str(DATA))
    if args.partition == "internal_validation":
        cases = sorted(val_ids)
    elif args.partition == "train":
        cases = sorted(train_ids)
    else:
        cases = sorted(train_ids | val_ids)
    if args.cases:
        src = pathlib.Path(args.cases)
        wanted = [c.strip() for c in
                  (src.read_text().split() if src.exists() else args.cases.split(","))
                  if c.strip()]
        missing = [c for c in wanted if c not in set(cases)]
        if missing:
            raise SystemExit(f"not in partition '{args.partition}': {missing[:5]}")
        cases = [c for c in cases if c in set(wanted)]
    if args.limit:
        cases = cases[:args.limit]

    tag = args.tag or args.partition
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = pathlib.Path(getattr(args, "checkpoint", CKPT))
    if not ckpt_path.exists():
        raise SystemExit(f"checkpoint not found: {ckpt_path}")
    ck = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=False)
    _sd = ck.get("model_state", ck) if isinstance(ck, dict) else ck
    # Deep-supervision checkpoints carry auxiliary heads; in eval mode the model
    # returns the main head only, so inference is identical either way.
    _ds = any(k.startswith("ds_heads.") for k in _sd)
    # Architecture comes from the checkpoint: its recorded config if present,
    # else the first encoder block's input width (4C = 2D DWT, 8C = 3D DWT).
    _cfg = ck.get("config", {}) if isinstance(ck, dict) else {}
    _down = _cfg.get("downsample") if isinstance(_cfg, dict) else None
    if not _down:
        _w = _sd["conv1_0.conv.0.weight"].shape[1] // _sd["conv0_0.conv.0.weight"].shape[0]
        _down = {4: "dwt", 8: "dwt3d"}[_w]
    _bf = (_cfg.get("base_filters") if isinstance(_cfg, dict) else None) or int(_sd["conv0_0.conv.0.weight"].shape[0])
    _norm = (_cfg.get("norm") if isinstance(_cfg, dict) else None) or \
        ("batch" if any(k.endswith("running_mean") for k in _sd) else "instance")
    _inorm = (_cfg.get("normalisation") if isinstance(_cfg, dict) else None) or "minmax"
    global NORMALISER
    NORMALISER = zscore_normalise if _inorm == "zscore" else irm_min_max_preprocess
    print(f"architecture  : downsample={_down} base_filters={_bf} deep_supervision={_ds} norm={_norm} input={_inorm}")
    model = WaveletUNetPlusPlus(in_channels=4, n_classes=3, downsample=_down,
                                deep_supervision=_ds, base_filters=_bf, norm=_norm).to(DEVICE)
    # Checkpoints come in two shapes: a bare state_dict, or the training
    # script's wrapper with the weights under "model_state".
    if isinstance(ck, dict):
        sd = next((ck[k] for k in ("model_state", "model_state_dict", "state_dict")
                   if k in ck), ck)
    else:
        sd = ck
    model.load_state_dict(sd, strict=True)
    if isinstance(ck, dict) and "epoch" in ck:
        print(f"checkpoint epoch: {ck['epoch']}  val_metric={ck.get('val_metric')}")
    model.eval()

    print(f"checkpoint    : {ckpt_path.name}")
    print(f"partition     : {args.partition}  ({len(cases)} cases)")
    print(f"preprocessing : {BRATS_ORDER} + percentile min-max")
    print(f"threshold     : {THR} (fixed, not tuned)")
    print(f"TTA           : {not args.no_tta}")
    print(f"window stride : {args.step}")
    print(f"axis order    : {args.axis_order}" + ("  + 90-degree rotation TTA" if args.rot_tta else ""))
    print()

    if args.floor_mult != 1.0:
        pp.MIN_VOXELS = {k: int(round(v * args.floor_mult))
                         for k, v in pp.MIN_VOXELS.items()}
        print(f"component floors: {pp.MIN_VOXELS}  (x{args.floor_mult})")
    if args.wt_policy != "components":
        print(f"WT policy       : {args.wt_policy}")

    rows = []
    for i, case_id in enumerate(cases, 1):
        try:
            img, gt = load_case(case_id)
        except FileNotFoundError as e:
            print(f"  [{i}/{len(cases)}] {case_id}  SKIP ({e})")
            continue

        def predict_oriented(img):
            if args.axis_order in ("sitk", "both"):
                # network sees (C, z, y, x) as in training; map the probabilities back to (x, y, z)
                prob_s = sliding_window_predict(model, np.ascontiguousarray(img.transpose(0, 3, 2, 1)), DEVICE,
                                                use_tta=not args.no_tta, step_size=(args.step,) * 3)
                prob_s = np.ascontiguousarray(np.asarray(prob_s).transpose(0, 3, 2, 1))
                if args.axis_order == "both":   # orientation TTA: average of both array orders
                    prob_n = sliding_window_predict(model, img, DEVICE, use_tta=not args.no_tta,
                                                    step_size=(args.step,) * 3)
                    prob = (np.asarray(prob_n) + prob_s) / 2.0
                else:
                    prob = prob_s
            else:
                prob = sliding_window_predict(model, img, DEVICE, use_tta=not args.no_tta,
                                              step_size=(args.step,) * 3)
            return prob

        prob = predict_oriented(img)
        if args.rot_tta:   # Amendment 12: rotate the axial (x, y) plane by 90 degrees and back
            prob_r = predict_oriented(np.ascontiguousarray(np.rot90(img, k=1, axes=(1, 2))))
            prob = (np.asarray(prob) + np.rot90(np.asarray(prob_r), k=-1, axes=(1, 2))) / 2.0
        if args.save_probs:
            # float16 halves the footprint and costs nothing at threshold 0.5.
            pdir = pathlib.Path(args.save_probs)
            pdir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(pdir / f"{case_id}.npz",
                                prob=np.asarray(prob, dtype=np.float16))
        pred = pp.postprocess(prob, THR, THR, THR, wt_policy=args.wt_policy,
                             et_policy=args.et_policy, et_min_volume=args.et_min_volume)
        row = {"Patient_ID": case_id}
        row.update(score_case(pred, gt))
        rows.append(row)
        print(f"  [{i}/{len(cases)}] {case_id}  "
              f"ET {row['Dice_ET']:.3f}  TC {row['Dice_TC']:.3f}  "
              f"WT {row['Dice_WT']:.3f}", flush=True)
        del prob, pred
        if i % 25 == 0:
            torch.cuda.empty_cache()

    if not rows:
        raise SystemExit("No cases evaluated.")

    df = pd.DataFrame(rows)
    per_case = out_dir / f"per_case_{tag}_recomputed.csv"
    df.to_csv(per_case, index=False)

    summary = summarise_segmentation(df, label=tag)
    summary["threshold"] = THR
    summary["tta"] = not args.no_tta
    summary.to_csv(out_dir / f"summary_{tag}_recomputed.csv", index=False)
    print_summary(summary, f"BraTS2021 {args.partition} (n={len(df)}) — recomputed")
    print(f"\nWrote {per_case}")


if __name__ == "__main__":
    main()
