#!/usr/bin/env python3
"""Build a per-case radiomic feature table for BraTS2021 MGMT classification.

Uses the **expert segmentation** for every case, not a model prediction.

That choice removes two problems at once. Mask provenance is uniform across the
whole cohort, so nothing analogous to the UPenn leakage (where mask source
correlated 4.7x with the outcome) can arise. And the segmentation model never
touches this analysis, so the 1000 cases it trained on carry no advantage over
the 251 it did not — there is simply no segmentation model in the loop.

The cost is that these features describe an idealised mask. A deployed system
would segment first and inherit that error, so these numbers describe the
ceiling available to a radiomic MGMT classifier given perfect segmentation,
not what a full pipeline would achieve. That is the right question to ask
first: if the signal is absent with perfect masks, no amount of segmentation
quality will produce it.

Labels come from the RSNA-MICCAI BraTS 2021 challenge MGMT annotations.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brats_gbm.features import all_region_features, znorm  # noqa: E402

DATA = ROOT / "data"
LABELS = ROOT / "metadata" / "brats_mgmt_labels.csv"
OUT_CSV = ROOT / "results" / "classification" / "brats_mgmt_features.csv"

# Same modality keys as the UPenn table so feature names line up.
MODALITIES = {"flair": "flair", "t1": "t1", "t1ce": "t1ce", "t2": "t2"}
ET_LABEL, NCR_LABEL, ED_LABEL = 4, 1, 2


def load_case(case_id: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], float]:
    d = DATA / case_id
    vols = {}
    for key, suffix in MODALITIES.items():
        p = d / f"{case_id}_{suffix}.nii.gz"
        if not p.exists():
            raise FileNotFoundError(p)
        vols[key] = nib.load(str(p)).get_fdata().astype(np.float32)

    seg_p = d / f"{case_id}_seg.nii.gz"
    if not seg_p.exists():
        raise FileNotFoundError(seg_p)
    seg_img = nib.load(str(seg_p))
    seg = seg_img.get_fdata().astype(np.float32)

    et = seg == ET_LABEL
    tc = et | (seg == NCR_LABEL)
    wt = tc | (seg == ED_LABEL)
    regions = {"ET": et, "TC": tc, "WT": wt}

    voxel_vol = float(abs(np.linalg.det(seg_img.affine[:3, :3])))
    return znorm(vols), regions, voxel_vol


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(LABELS))
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--out", default=str(OUT_CSV))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    lab = pd.read_csv(args.labels)
    lab["case_id"] = lab["BraTS21ID"].apply(lambda x: f"BraTS2021_{int(x):05d}")
    lab = lab.rename(columns={"MGMT_value": "mgmt_label"})

    available = {p.name for p in Path(args.data).glob("BraTS2021_*") if p.is_dir()}
    usable = lab[lab.case_id.isin(available)].reset_index(drop=True)
    missing = len(lab) - len(usable)
    print(f"labels        : {len(lab)}")
    print(f"with imaging  : {len(usable)}   (dropped {missing} without imaging)")
    print(f"label balance : {usable.mgmt_label.value_counts().to_dict()}")
    print(f"masks         : expert segmentation (uniform provenance)\n")

    if args.limit:
        usable = usable.head(args.limit)

    rows = []
    for i, r in usable.iterrows():
        try:
            vols, regions, voxel_vol = load_case(r.case_id)
        except FileNotFoundError as e:
            print(f"  [{i+1}/{len(usable)}] {r.case_id} SKIP ({e.args[0]})")
            continue
        feats = {"case_id": r.case_id, "mgmt_label": int(r.mgmt_label),
                 "mask_source": "ground_truth"}
        feats.update(all_region_features(regions, vols, voxel_vol))
        rows.append(feats)
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(usable)}] {r.case_id} done", flush=True)

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\n{len(df)} cases, {df.shape[1]} columns -> {args.out}")


if __name__ == "__main__":
    main()
