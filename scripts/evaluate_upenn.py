#!/usr/bin/env python3
"""Evaluate segmentation checkpoints on the UPenn-GBM held-out test set.

Protocol, in order, with the test set touched exactly once at the end:

  1. Run inference on the validation and test subjects. Probability maps are
     cached to disk, so every later sweep is free and no tuning decision ever
     needs a second forward pass.
  2. Select the three region thresholds on validation.
  3. Select the enhancing-tumour policy and its volume cutoff on validation,
     holding the thresholds from step 2 fixed.
  4. Score the test set once with the selected configuration and report
     bootstrap confidence intervals.

Nothing in steps 2 or 3 sees a test subject. The selected configuration is
written alongside the results so a reviewer can confirm what was chosen and on
what basis.

Usage
-----
    python scripts/evaluate_upenn.py                      # all default setups
    python scripts/evaluate_upenn.py --setups upenn_v3_best
    python scripts/evaluate_upenn.py --no-cache           # force recompute
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brats_gbm.data.upenn import UPennDataset  # noqa: E402
from brats_gbm.eval import postprocess as pp  # noqa: E402
from brats_gbm.eval.inference import (  # noqa: E402
    average_probability_maps,
    sliding_window_predict,
)
from brats_gbm.eval.metrics import score_case  # noqa: E402
from brats_gbm.eval.stats import (  # noqa: E402
    paired_diff_ci,
    print_summary,
    summarise_segmentation,
)
from brats_gbm.model import WaveletUNetPlusPlus  # noqa: E402
from brats_gbm.splits import assert_disjoint, freeze_upenn_split, upenn_split  # noqa: E402

NIFTI_DIR = ROOT / "upenn_nifti"
CLINICAL_CSV = ROOT / "upenn_data" / "UPENN-GBM_clinical_info_v2.1.csv"
RESULTS_DIR = ROOT / "results" / "upenn"
CACHE_DIR = ROOT / "cache" / "upenn_probs"

ET_GRID = [0.30, 0.40, 0.50]
TC_GRID = [0.30, 0.40, 0.50]
WT_GRID = [0.30, 0.40, 0.50]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Named setups. "baseline_brats" is the BraTS-trained segmentor applied to
# UPenn with no adaptation; it quantifies the domain gap that fine-tuning
# closes. "ensemble_top5" averages the top-5 validation checkpoints saved
# during fine-tuning.
SETUPS: dict[str, dict] = {
    "baseline_brats": {
        "checkpoints": [ROOT / "checkpoints" / "segmentor_epoch_650.pth"],
        "description": "BraTS2021-trained, zero-shot on UPenn-GBM",
    },
    "upenn_v3_best": {
        "checkpoints": [ROOT / "checkpoints" / "upenn_v3_best.pth"],
        "description": "Fine-tuned on UPenn-GBM, best EMA validation checkpoint",
    },
    "upenn_v3_last": {
        "checkpoints": [ROOT / "checkpoints" / "upenn_v3_last.pth"],
        "description": "Fine-tuned on UPenn-GBM, final epoch",
    },
    "ensemble_top5": {
        "checkpoints": sorted((ROOT / "checkpoints" / "upenn_v3_topk").glob("*.pth")),
        "description": "Probability-average of the top-5 validation checkpoints",
    },
}


# --------------------------------------------------------------------------
# inference + caching
# --------------------------------------------------------------------------
def load_model(path: Path) -> torch.nn.Module:
    model = WaveletUNetPlusPlus(in_channels=4, n_classes=3).to(DEVICE)
    ckpt = torch.load(str(path), map_location=DEVICE, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def probs_for_checkpoint(
    ckpt: Path,
    dataset: UPennDataset,
    split_name: str,
    use_cache: bool = True,
) -> dict[str, np.ndarray]:
    """Probability maps for every subject, cached on disk as float16."""
    cache_dir = CACHE_DIR / ckpt.stem / split_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    out: dict[str, np.ndarray] = {}
    model = None

    for i in range(len(dataset)):
        sub_id = dataset.subject_ids[i]
        cache_file = cache_dir / f"{sub_id}.npy"

        if use_cache and cache_file.exists():
            out[sub_id] = np.load(cache_file)
            continue

        if model is None:
            model = load_model(ckpt)
            print(f"    loaded {ckpt.name}")

        sample = dataset[i]
        prob = sliding_window_predict(model, sample["image"].numpy(), DEVICE)
        prob16 = prob.astype(np.float16)
        np.save(cache_file, prob16)
        out[sub_id] = prob16
        print(f"    {split_name}/{sub_id}  ET_max={prob[0].max():.3f}", flush=True)

    if model is not None:
        del model
        torch.cuda.empty_cache()
    return out


def ground_truth(dataset: UPennDataset) -> dict[str, np.ndarray]:
    """Boolean [3,D,H,W] targets keyed by subject id."""
    return {
        dataset.subject_ids[i]: dataset[i]["label"].numpy().astype(bool)
        for i in range(len(dataset))
    }


# --------------------------------------------------------------------------
# validation-only selection
# --------------------------------------------------------------------------
def mean_dice(preds: dict[str, np.ndarray], gts: dict[str, np.ndarray]) -> float:
    from brats_gbm.eval.metrics import dice_score

    per_case = [
        np.mean([dice_score(preds[p][c], gts[p][c]) for c in range(3)])
        for p in preds
    ]
    return float(np.mean(per_case))


def select_thresholds(
    val_probs: dict[str, np.ndarray],
    val_gt: dict[str, np.ndarray],
) -> tuple[float, float, float]:
    """Grid-search region thresholds on validation, no ET policy applied yet."""
    best, best_cfg = -1.0, (0.40, 0.40, 0.40)

    for et_t in ET_GRID:
        for tc_t in TC_GRID:
            for wt_t in WT_GRID:
                preds = {}
                for pid, prob in val_probs.items():
                    p = pp.apply_thresholds(prob.astype(np.float32), et_t, tc_t, wt_t)
                    preds[pid] = pp.enforce_hierarchy(p)
                score = mean_dice(preds, val_gt)
                if score > best:
                    best, best_cfg = score, (et_t, tc_t, wt_t)

    print(f"    thresholds ET={best_cfg[0]:.2f} TC={best_cfg[1]:.2f} "
          f"WT={best_cfg[2]:.2f}  (val mean Dice {best:.4f})")
    return best_cfg


def select_et_policy(
    val_probs: dict[str, np.ndarray],
    val_gt: dict[str, np.ndarray],
    thresholds: tuple[float, float, float],
) -> tuple[str, int, pd.DataFrame]:
    """Choose the ET policy on validation with thresholds held fixed."""
    et_t, tc_t, wt_t = thresholds
    candidates = [("min_volume", v) for v in pp.ET_MIN_VOLUME_GRID]
    candidates.append(("rescue", 0))

    rows = []
    for policy, min_vol in candidates:
        preds = {
            pid: pp.postprocess(
                prob.astype(np.float32), et_t, tc_t, wt_t,
                et_policy=policy, et_min_volume=min_vol,
            )
            for pid, prob in val_probs.items()
        }
        from brats_gbm.eval.metrics import dice_score

        et_dice = float(np.mean([dice_score(preds[p][0], val_gt[p][0]) for p in preds]))
        rows.append({
            "et_policy": policy,
            "et_min_volume": min_vol,
            "val_mean_dice": mean_dice(preds, val_gt),
            "val_et_dice": et_dice,
        })

    table = pd.DataFrame(rows).sort_values("val_mean_dice", ascending=False)
    best = table.iloc[0]
    print(f"    ET policy  {best['et_policy']} (min_volume={int(best['et_min_volume'])})"
          f"  val mean Dice {best['val_mean_dice']:.4f}")
    return str(best["et_policy"]), int(best["et_min_volume"]), table


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def score_split(
    probs: dict[str, np.ndarray],
    gts: dict[str, np.ndarray],
    thresholds: tuple[float, float, float],
    et_policy: str,
    et_min_volume: int,
) -> pd.DataFrame:
    et_t, tc_t, wt_t = thresholds
    rows = []
    for pid, prob in probs.items():
        pred = pp.postprocess(
            prob.astype(np.float32), et_t, tc_t, wt_t,
            et_policy=et_policy, et_min_volume=et_min_volume,
        )
        row = {"Patient_ID": pid}
        row.update(score_case(pred, gts[pid]))
        row["Pred_Empty_ET"] = int(pred[0].sum() == 0)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("Patient_ID").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--setups", nargs="*", default=list(SETUPS))
    ap.add_argument("--nifti-dir", default=str(NIFTI_DIR))
    ap.add_argument("--out-dir", default=str(RESULTS_DIR))
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--subject-split", default=None,
                    help="JSON with explicit {train,val,test} lists, as written "
                         "by crossval_upenn.py. Defaults to the frozen split.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    use_cache = not args.no_cache

    if args.subject_split:
        split = json.loads(Path(args.subject_split).read_text())
        train_subs, val_subs, test_subs = split["train"], split["val"], split["test"]
        # A cross-validation fold trains its own checkpoint; register it so
        # --setups crossval_foldK resolves.
        fold_dir = Path(args.subject_split).stem.replace("_subjects", "")
        ckpt = ROOT / "checkpoints" / "crossval" / fold_dir / "upenn_v3_best.pth"
        SETUPS[f"crossval_{fold_dir}"] = {
            "checkpoints": [ckpt],
            "description": f"Cross-validation {fold_dir}, best inner-validation checkpoint",
        }
    else:
        train_subs, val_subs, test_subs = upenn_split(args.nifti_dir)
    assert_disjoint(train=train_subs, val=val_subs, test=test_subs)
    freeze_upenn_split(args.nifti_dir, str(out_dir / "split_assignment.csv"))

    print(f"device {DEVICE}")
    print(f"subjects  train={len(train_subs)}  val={len(val_subs)}  test={len(test_subs)}")
    print("split verified disjoint and frozen -> results/upenn/split_assignment.csv")

    val_ds = UPennDataset(args.nifti_dir, val_subs, str(CLINICAL_CSV))
    test_ds = UPennDataset(args.nifti_dir, test_subs, str(CLINICAL_CSV))
    val_gt = ground_truth(val_ds)
    test_gt = ground_truth(test_ds)

    summaries, selections, per_case = [], [], {}

    for name in args.setups:
        spec = SETUPS[name]
        ckpts = [Path(c) for c in spec["checkpoints"]]
        missing = [c for c in ckpts if not c.exists()]
        if not ckpts or missing:
            print(f"\n[skip] {name}: missing {[str(m) for m in missing] or 'checkpoints'}")
            continue

        print(f"\n{'=' * 72}\n{name}  —  {spec['description']}\n{'=' * 72}")

        print("  validation inference")
        val_maps = [probs_for_checkpoint(c, val_ds, "val", use_cache) for c in ckpts]
        val_probs = (
            val_maps[0] if len(val_maps) == 1
            else {p: average_probability_maps([m[p] for m in val_maps]) for p in val_maps[0]}
        )

        thresholds = select_thresholds(val_probs, val_gt)
        et_policy, et_min_vol, policy_table = select_et_policy(val_probs, val_gt, thresholds)
        policy_table.insert(0, "setup", name)
        selections.append(policy_table)
        del val_probs, val_maps

        print("  test inference")
        test_maps = [probs_for_checkpoint(c, test_ds, "test", use_cache) for c in ckpts]
        test_probs = (
            test_maps[0] if len(test_maps) == 1
            else {p: average_probability_maps([m[p] for m in test_maps]) for p in test_maps[0]}
        )

        df = score_split(test_probs, test_gt, thresholds, et_policy, et_min_vol)
        df.to_csv(out_dir / f"per_case_{name}.csv", index=False)
        per_case[name] = df
        del test_probs, test_maps

        summary = summarise_segmentation(df, label=name)
        summary["et_threshold"] = thresholds[0]
        summary["tc_threshold"] = thresholds[1]
        summary["wt_threshold"] = thresholds[2]
        summary["et_policy"] = et_policy
        summary["et_min_volume"] = et_min_vol
        summaries.append(summary)
        print_summary(summary, f"TEST — {name} (n={len(df)})")

    if not summaries:
        print("\nNo setups evaluated.")
        return

    all_summary = pd.concat(summaries, ignore_index=True)
    all_summary.to_csv(out_dir / "segmentation_summary.csv", index=False)
    pd.concat(selections, ignore_index=True).to_csv(
        out_dir / "validation_selection.csv", index=False)

    # Domain-adaptation effect, paired over the same test patients.
    if "baseline_brats" in per_case and "upenn_v3_best" in per_case:
        a = per_case["upenn_v3_best"].set_index("Patient_ID")["Dice_Mean"]
        b = per_case["baseline_brats"].set_index("Patient_ID")["Dice_Mean"]
        common = a.index.intersection(b.index)
        d, lo, hi = paired_diff_ci(a.loc[common], b.loc[common])
        print(f"\nFine-tuning effect (paired, n={len(common)}): "
              f"mean Dice +{d:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
        (out_dir / "domain_adaptation.json").write_text(json.dumps(
            {"n_paired": int(len(common)), "delta_mean_dice": d,
             "ci_low": lo, "ci_high": hi}, indent=2))

    print(f"\nWrote results to {out_dir}")


if __name__ == "__main__":
    main()
