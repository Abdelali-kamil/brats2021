#!/usr/bin/env python3
"""
Report BraTS segmentation metrics on the held-out split only.

get_datasets() returns all 1251 cases with no split; train.py then takes a
random 80% (seed 42) for training. Per-patient result files therefore cover
cases the model was trained on, and pooling them reports partly-memorised
performance.

The per-patient numbers themselves are fine — only the pooling was wrong — so
this recomputes the summary over the 251 held-out cases without re-running
inference. Use these figures when reporting.
"""
import argparse
import pathlib

import numpy as np
import pandas as pd
import torch
from torch.utils.data import random_split

DATA_ROOT = pathlib.Path("/home/kamilabdelali/brats2021/data")
TRAIN_FRAC = 0.8
SEED = 42


def split_ids(seed: int = SEED, train_frac: float = TRAIN_FRAC):
    """Reproduce train.py's split exactly: sorted case dirs, torch random_split."""
    pats = sorted(d.name for d in DATA_ROOT.glob("BraTS2021_*") if d.is_dir())
    n = len(pats)
    train_n = int(train_frac * n)
    tr, va = random_split(
        list(range(n)), [train_n, n - train_n],
        generator=torch.Generator().manual_seed(seed),
    )
    return {pats[i] for i in tr.indices}, {pats[i] for i in va.indices}


def summarise(df: pd.DataFrame, label: str):
    print(f"\n{label}  (n={len(df)})")
    print("-" * 58)
    print(f"{'Region':<8} | {'Dice':<18} | {'HD95 (mm)':<18}")
    for reg in ("ET", "TC", "WT"):
        d = df[f"Dice_{reg}"]
        h = pd.to_numeric(df[f"HD95_{reg}"], errors="coerce").dropna()
        hd = f"{h.mean():.4f} ± {h.std():.4f}" if len(h) else "n/a"
        print(f"{reg:<8} | {d.mean():.4f} ± {d.std():.4f} | {hd}")
    mean = df[["Dice_ET", "Dice_TC", "Dice_WT"]].mean(axis=1)
    print("-" * 58)
    print(f"{'MEAN':<8} | {mean.mean():.4f} ± {mean.std():.4f}")
    return float(mean.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_csv", nargs="?",
                    default="final_results_segmentor_epoch_650.csv")
    ap.add_argument("--out", default="heldout_results.csv")
    args = ap.parse_args()

    res = pd.read_csv(args.results_csv)
    if "Patient_ID" not in res.columns:
        raise SystemExit(f"{args.results_csv} has no Patient_ID column")

    train_ids, val_ids = split_ids()
    res["split"] = np.where(res.Patient_ID.isin(val_ids), "held-out",
                     np.where(res.Patient_ID.isin(train_ids), "train", "unknown"))

    counts = res.split.value_counts().to_dict()
    print(f"source : {args.results_csv}  ({len(res)} rows)")
    print(f"split  : {counts}")

    held = res[res.split == "held-out"]
    trained = res[res.split == "train"]
    if held.empty:
        raise SystemExit("No held-out cases found - check the split settings.")

    m_held = summarise(held, "HELD-OUT — report these")
    if not trained.empty:
        m_train = summarise(trained, "Training cases — do not report")
        print(f"\ngeneralisation gap: {m_train - m_held:+.4f}")
    print(f"pooled (as previously reported): "
          f"{res[['Dice_ET','Dice_TC','Dice_WT']].mean(axis=1).mean():.4f}")

    held.drop(columns=["split"]).to_csv(args.out, index=False)
    print(f"\nheld-out per-patient results -> {args.out}")


if __name__ == "__main__":
    main()
