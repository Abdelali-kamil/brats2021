#!/usr/bin/env python3
"""Compare validation-selected post-processing against a fixed default.

The problem
-----------
`evaluate_upenn.py` selects six things on validation: three region thresholds,
the ET policy, its volume cutoff, and the WT component policy. The validation
split has 14 subjects. That is far too few to support a six-way selection, and
the consequence is measurable: widening the search space raised validation mean
Dice for upenn_v3_best from 0.838 to 0.851 while its test score *fell* from
0.816 to 0.807. Validation improved because the search found configurations
that fit those 14 subjects; test degraded because those configurations did not
generalise. The setup that moved furthest, upenn_v3_best, was the only one to
pick the unusual combination (WT=largest, ET cutoff 300) — a fingerprint of
fitting noise.

The response
------------
Report a configuration that involves no selection at all: threshold 0.5 in every
region, standard component cleanup, no ET volume rule, no WT largest-component
rule. It is fixed in advance, identical for every setup, and cannot overfit
anything because nothing is fitted.

This choice is made on statistical grounds — 14 subjects cannot support six
selected parameters — and not because of how the test numbers came out. The
distinction matters: choosing a selection strategy by its test performance is
test-set tuning one level up. To keep that honest, this script reports both
configurations side by side for every setup, and the decision to lead with the
fixed one is argued from the sample size, not the outcome.

The validation-selected configuration remains available as a sensitivity
analysis, and cross-validation over all 147 subjects
(`scripts/crossval_upenn.py`) is what would actually make selection reliable.

Runs entirely off cached probability maps; no GPU.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brats_gbm.data.upenn import UPennDataset  # noqa: E402
from brats_gbm.eval import postprocess as pp  # noqa: E402
from brats_gbm.eval.inference import average_probability_maps  # noqa: E402
from brats_gbm.eval.metrics import score_case  # noqa: E402
from brats_gbm.eval.stats import print_summary, summarise_segmentation  # noqa: E402
from brats_gbm.splits import upenn_split  # noqa: E402

CACHE = ROOT / "cache" / "upenn_probs"
OUT_DIR = ROOT / "results" / "upenn"

# The fixed configuration: no selection, identical for every setup.
FIXED = dict(et_thr=0.5, tc_thr=0.5, wt_thr=0.5,
             et_policy="min_volume", et_min_volume=0, wt_policy="components")

# Which cached checkpoint directories make up each setup.
SETUP_MEMBERS = {
    "baseline_brats": ["segmentor_epoch_650__brats"],
    "upenn_v3_best": ["upenn_v3_best__upenn"],
    "upenn_v3_last": ["upenn_v3_last__upenn"],
    "ensemble_top5": [f"ep{e}__upenn" for e in
                      ("0054_val0.8576", "0062_val0.8569", "0077_val0.8585",
                       "0078_val0.8577", "0080_val0.8573")],
    "upenn_v4_best": ["upenn_v3_best__brats"],
    "ensemble_v3_v4": ["upenn_v3_best__upenn", "upenn_v3_best__brats"],
}


def load_maps(dirs: list[str], split: str) -> dict[str, np.ndarray] | None:
    per_member = []
    for d in dirs:
        p = CACHE / d / split
        if not p.exists():
            return None
        m = {f.stem: np.load(f) for f in sorted(p.glob("*.npy"))}
        if not m:
            return None
        per_member.append(m)
    if len(per_member) == 1:
        return per_member[0]
    common = set.intersection(*(set(m) for m in per_member))
    return {pid: average_probability_maps([m[pid] for m in per_member])
            for pid in sorted(common)}


def score(maps: dict[str, np.ndarray], gt: dict[str, np.ndarray], cfg: dict) -> pd.DataFrame:
    rows = []
    for pid, prob in maps.items():
        pred = pp.postprocess(
            prob.astype(np.float32), cfg["et_thr"], cfg["tc_thr"], cfg["wt_thr"],
            et_policy=cfg["et_policy"], et_min_volume=cfg["et_min_volume"],
            wt_policy=cfg["wt_policy"])
        row = {"Patient_ID": pid}
        row.update(score_case(pred, gt[pid]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("Patient_ID").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nifti-dir", default=str(ROOT / "upenn_nifti"))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _, _, test_subs = upenn_split(args.nifti_dir)
    ds = UPennDataset(args.nifti_dir, test_subs)
    gt = {ds.subject_ids[i]: ds.labels_only(i) for i in range(len(ds))}

    selected = pd.read_csv(out_dir / "segmentation_summary.csv")
    sel_mean = selected[selected.region == "MEAN"].set_index("label")

    print("Fixed configuration (no selection): "
          f"thresholds {FIXED['et_thr']}/{FIXED['tc_thr']}/{FIXED['wt_thr']}, "
          f"ET {FIXED['et_policy']}(cutoff {FIXED['et_min_volume']}), "
          f"WT {FIXED['wt_policy']}\n")

    rows, frames = [], []
    for name, dirs in SETUP_MEMBERS.items():
        maps = load_maps(dirs, "test")
        if maps is None:
            print(f"  [skip] {name}: cached maps missing")
            continue
        df = score(maps, gt, FIXED)
        df.to_csv(out_dir / f"per_case_{name}_fixedcfg.csv", index=False)
        summary = summarise_segmentation(df, label=name)
        summary["config"] = "fixed"
        frames.append(summary)

        fixed_mean = float(summary[summary.region == "MEAN"].iloc[0]["dice"])
        sel = float(sel_mean.loc[name, "dice"]) if name in sel_mean.index else np.nan
        rows.append({"setup": name, "fixed_dice": fixed_mean,
                     "validation_selected_dice": sel,
                     "delta_fixed_minus_selected": fixed_mean - sel})
        print(f"  {name:<18} fixed {fixed_mean:.4f}   "
              f"val-selected {sel:.4f}   delta {fixed_mean - sel:+.4f}")

    comp = pd.DataFrame(rows)
    comp.to_csv(out_dir / "selection_sensitivity.csv", index=False)
    all_fixed = pd.concat(frames, ignore_index=True)
    all_fixed.to_csv(out_dir / "segmentation_summary_fixedcfg.csv", index=False)

    n_better = int((comp.delta_fixed_minus_selected > 0).sum())
    print(f"\nfixed beats validation-selected on {n_better}/{len(comp)} setups; "
          f"mean delta {comp.delta_fixed_minus_selected.mean():+.4f}")
    if n_better > len(comp) / 2:
        print("Selecting six parameters on 14 subjects is not paying for itself.")

    best = comp.loc[comp.fixed_dice.idxmax()]
    print(f"\nbest under the fixed configuration: {best.setup} "
          f"({best.fixed_dice:.4f})")

    for _, r in comp.iterrows():
        sub = all_fixed[all_fixed.label == r.setup]
        print_summary(sub, f"{r.setup} — fixed configuration (n=29)")


if __name__ == "__main__":
    main()
