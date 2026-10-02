"""Tune BraTS post-processing on the val partition, then apply once to test.

Motivation (see analysis/ET_WT_error_analysis.md): the two historical
evaluations each won a different region and neither was tuned. The old pipeline
held WT at 0.9247 Dice / 4.43 HD95 but left ET at 0.8498 because stray voxels on
empty-ET cases score 0.0. The recomputed pipeline fixed three of those four
cases (ET 0.8564) but regressed WT to 0.9006 Dice / 10.17 HD95 -- the signature
of distant false-positive components that `wt_policy="largest"` exists to remove.

The old evaluator refused to tune because the 251-case partition doubled as the
model-selection set. `brats_split_3way` removes that objection: val (126) is
already the selection set, test (125) is untouched. So we sweep on val only and
spend test exactly once.

Inference is the expensive part and is independent of every knob being swept, so
probabilities are cached to disk on the first pass and every later sweep is free.

Usage:
  # stage 1 -- cache val probabilities and sweep (GPU once, then CPU)
  python scripts/tune_postprocess_brats.py --checkpoint <ckpt> --stage sweep

  # stage 2 -- apply the winning config to test, exactly once
  python scripts/tune_postprocess_brats.py --checkpoint <ckpt> --stage apply \
      --et-policy min_volume --et-min-volume 200 --wt-policy largest --thr 0.5
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brats_gbm.data.image import irm_min_max_preprocess  # noqa: E402
from brats_gbm.eval import postprocess as pp  # noqa: E402
from brats_gbm.eval.inference import sliding_window_predict  # noqa: E402
from brats_gbm.eval.metrics import score_case  # noqa: E402
from brats_gbm.eval.stats import summarise_segmentation  # noqa: E402
from brats_gbm.model import WaveletUNetPlusPlus  # noqa: E402
from brats_gbm.splits import brats_split_3way  # noqa: E402

DATA = ROOT / "data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BRATS_ORDER = ("t1", "t1ce", "t2", "flair")
ET_LABEL, NCR_LABEL, ED_LABEL = 4, 1, 2

# Swept on val only. Grids come from postprocess.py -- no new method invented.
THR_GRID = [0.4, 0.5, 0.6]


def load_case(case_id: str) -> tuple[np.ndarray, np.ndarray]:
    d = DATA / case_id
    img = np.stack([
        irm_min_max_preprocess(
            nib.load(str(d / f"{case_id}_{m}.nii.gz")).get_fdata().astype(np.float32))
        for m in BRATS_ORDER
    ])
    seg = nib.load(str(d / f"{case_id}_seg.nii.gz")).get_fdata().astype(np.float32)
    et = seg == ET_LABEL
    tc = et | (seg == NCR_LABEL)
    wt = tc | (seg == ED_LABEL)
    return img, np.stack([et, tc, wt])


def build_model(ckpt_path: Path, downsample: str | None):
    ck = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=False)
    arm = downsample
    if arm is None:
        arm = (ck.get("config", {}) or {}).get("downsample", "dwt") \
            if isinstance(ck, dict) else "dwt"
    model = WaveletUNetPlusPlus(in_channels=4, n_classes=3, downsample=arm).to(DEVICE)
    state = ck.get("model_state", ck.get("model_state_dict", ck)) \
        if isinstance(ck, dict) else ck
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, arm


def cache_probabilities(model, cases, cache_dir: Path, use_tta: bool) -> None:
    """Run inference once per case and persist the probability map (fp16)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    for i, cid in enumerate(cases, 1):
        out = cache_dir / f"{cid}.npy"
        gt_out = cache_dir / f"{cid}_gt.npy"
        if out.exists() and gt_out.exists():
            continue
        img, gt = load_case(cid)
        prob = sliding_window_predict(model, img, DEVICE, use_tta=use_tta)
        np.save(out, np.asarray(prob, dtype=np.float16))
        np.save(gt_out, gt.astype(bool))
        if i % 10 == 0 or i == len(cases):
            print(f"  cached {i}/{len(cases)}", flush=True)


def score_config(cases, cache_dir: Path, thr, et_policy, et_min_volume, wt_policy):
    rows = []
    for cid in cases:
        prob = np.load(cache_dir / f"{cid}.npy").astype(np.float32)
        gt = np.load(cache_dir / f"{cid}_gt.npy")
        pred = pp.postprocess(prob, thr, thr, thr,
                              et_policy=et_policy,
                              et_min_volume=et_min_volume,
                              wt_policy=wt_policy)
        s = score_case(pred, gt)
        s["Patient_ID"] = cid
        rows.append(s)
    return pd.DataFrame(rows)


def mean_dice(df: pd.DataFrame) -> float:
    cols = [c for c in df.columns if c.lower().startswith("dice")]
    per_region = [c for c in cols if c.split("_")[-1] in ("ET", "TC", "WT")]
    return float(df[per_region].mean(axis=1).mean()) if per_region else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--downsample", default=None)
    ap.add_argument("--stage", choices=["sweep", "apply"], default="sweep")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--cache-dir", default="/mnt/data1/kamil_research/experiments/_probcache")
    ap.add_argument("--out-dir", default="/mnt/data1/kamil_research/experiments/analysis")
    ap.add_argument("--tag", default="repro1")
    # only for --stage apply
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--et-policy", default="min_volume")
    ap.add_argument("--et-min-volume", type=int, default=200)
    ap.add_argument("--wt-policy", default="components")
    args = ap.parse_args()

    _, val_ids, test_ids = brats_split_3way(str(DATA))
    val_cases, test_cases = sorted(val_ids), sorted(test_ids)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    model, arm = build_model(Path(args.checkpoint), args.downsample)
    use_tta = not args.no_tta
    print(f"checkpoint: {args.checkpoint}\ndownsample: {arm}\nTTA: {use_tta}")

    if args.stage == "sweep":
        cache = Path(args.cache_dir) / f"{args.tag}_val_tta{int(use_tta)}"
        print(f"\n[1/2] caching val probabilities ({len(val_cases)} cases) -> {cache}")
        cache_probabilities(model, val_cases, cache, use_tta)

        print(f"\n[2/2] sweeping post-processing on VAL ({len(val_cases)} cases)")
        et_cands = [("none", 0)] + [("min_volume", v) for v in pp.ET_MIN_VOLUME_GRID if v > 0]
        results = []
        for thr, (et_pol, et_vol), wt_pol in itertools.product(
                THR_GRID, et_cands, pp.WT_POLICIES):
            df = score_config(val_cases, cache, thr, et_pol, et_vol, wt_pol)
            md = mean_dice(df)
            rec = {"thr": thr, "et_policy": et_pol, "et_min_volume": et_vol,
                   "wt_policy": wt_pol, "val_mean_dice": md}
            for r in ("ET", "TC", "WT"):
                c = f"Dice_{r}"
                if c in df: rec[f"val_dice_{r}"] = float(df[c].mean())
                h = f"HD95_{r}"
                if h in df: rec[f"val_hd95_{r}"] = float(np.nanmean(df[h]))
            results.append(rec)
            print(f"  thr={thr} et={et_pol}/{et_vol:<4} wt={wt_pol:<11} "
                  f"val_mean_dice={md:.4f}", flush=True)

        res = pd.DataFrame(results).sort_values("val_mean_dice", ascending=False)
        p = out_dir / f"postproc_sweep_val_{args.tag}.csv"
        res.to_csv(p, index=False)
        best = res.iloc[0].to_dict()
        (out_dir / f"postproc_best_{args.tag}.json").write_text(json.dumps(best, indent=2))
        print(f"\nsaved sweep -> {p}")
        print("\n=== BEST ON VAL (selection made here, test untouched) ===")
        print(json.dumps(best, indent=2))
        print("\nNow rerun with --stage apply using these values to score test once.")
    else:
        cache = Path(args.cache_dir) / f"{args.tag}_test_tta{int(use_tta)}"
        print(f"\ncaching TEST probabilities ({len(test_cases)} cases)")
        cache_probabilities(model, test_cases, cache, use_tta)
        print("\napplying the val-selected configuration to TEST, once:")
        print(f"  thr={args.thr} et_policy={args.et_policy} "
              f"et_min_volume={args.et_min_volume} wt_policy={args.wt_policy}")
        df = score_config(test_cases, cache, args.thr, args.et_policy,
                          args.et_min_volume, args.wt_policy)
        per = out_dir / f"per_case_test_{args.tag}.csv"
        df.to_csv(per, index=False)
        summary = summarise_segmentation(df, label="test")
        s = out_dir / f"summary_test_{args.tag}.csv"
        pd.DataFrame(summary).to_csv(s, index=False) if not isinstance(summary, pd.DataFrame) \
            else summary.to_csv(s, index=False)
        print(f"\nsaved -> {per}\nsaved -> {s}")
        print(f"\nTEST mean Dice: {mean_dice(df):.4f}  (n={len(df)})")


if __name__ == "__main__":
    main()
