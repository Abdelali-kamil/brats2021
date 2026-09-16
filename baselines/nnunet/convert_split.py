"""Convert BraTS2021 to nnU-Net format, honouring this project's frozen split.

nnU-Net ships `Dataset137_BraTS21.py`, but it hardcodes its author's data path
and copies *every* case into imagesTr. Used as-is here it would put our 251
held-out cases into nnU-Net's training set, and the comparison would be void.

This wrapper keeps the parts that must stay nnU-Net's:

  * `copy_BraTS_segmentation_and_convert_labels_to_nnUNet` -- their label
    remap (BraTS 0/1/2/4 -> nnU-Net 0/1/2/3, ET becomes 3)
  * `generate_dataset_json` -- their dataset descriptor, same channel order
    (T1, T1ce, T2, FLAIR) and same region definitions

and changes only which cases go where:

  imagesTr / labelsTr : the 1000 training cases, and nothing else
  imagesTs            : the 251 held-out cases (val + test), images only

nnU-Net then does its own internal cross-validation for checkpoint selection
inside the 1000, never seeing a held-out case, and we predict on imagesTs at
the end. That keeps both scoring options open -- the 251-case internal
validation or the 125-case clean test -- without retraining.

  python baselines/nnunet/convert_split.py
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
SPLIT = ROOT / "baselines" / "common" / "brats_split_frozen.json"

TASK_ID, TASK_NAME = 137, "BraTS2021"
FOLDER = f"Dataset{TASK_ID:03d}_{TASK_NAME}"

# nnU-Net's own channel order, from Dataset137_BraTS21.py.
MODALITIES = [("t1", "0000"), ("t1ce", "0001"), ("t2", "0002"), ("flair", "0003")]


def _one_case(job: tuple[str, str, str, bool]) -> str:
    case, src_root, out_base, is_train = job
    from nnunetv2.dataset_conversion.Dataset137_BraTS21 import (
        copy_BraTS_segmentation_and_convert_labels_to_nnUNet,
    )

    src = pathlib.Path(src_root) / case
    images = pathlib.Path(out_base) / ("imagesTr" if is_train else "imagesTs")
    for mod, idx in MODALITIES:
        shutil.copy(src / f"{case}_{mod}.nii.gz", images / f"{case}_{idx}.nii.gz")
    if is_train:
        # Their label conversion, unmodified.
        copy_BraTS_segmentation_and_convert_labels_to_nnUNet(
            str(src / f"{case}_seg.nii.gz"),
            str(pathlib.Path(out_base) / "labelsTr" / f"{case}.nii.gz"),
        )
    return case


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--workers", type=int, default=4,
                    help="kept low by default: a training run shares this machine")
    args = ap.parse_args()

    from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
    from nnunetv2.paths import nnUNet_raw

    if not nnUNet_raw:
        sys.exit("nnUNet_raw is unset. Export the three nnUNet_* variables first.")

    split = json.loads(SPLIT.read_text())
    train = split["train"]
    heldout = sorted(split["val"] + split["test"])
    print(f"train {len(train)}  held-out {len(heldout)} "
          f"(val {len(split['val'])} + test {len(split['test'])})")
    assert not (set(train) & set(heldout)), "train/held-out overlap"

    out_base = pathlib.Path(nnUNet_raw) / FOLDER
    for sub in ("imagesTr", "labelsTr", "imagesTs"):
        (out_base / sub).mkdir(parents=True, exist_ok=True)

    jobs = ([(c, args.data, str(out_base), True) for c in train]
            + [(c, args.data, str(out_base), False) for c in heldout])
    print(f"converting {len(jobs)} cases with {args.workers} workers -> {out_base}")

    with multiprocessing.Pool(args.workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(_one_case, jobs), 1):
            if i % 100 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}", flush=True)

    # Their descriptor, their channel names, their region definitions.
    generate_dataset_json(
        str(out_base),
        channel_names={0: "T1", 1: "T1ce", 2: "T2", 3: "Flair"},
        labels={"background": 0, "whole tumor": (1, 2, 3),
                "tumor core": (2, 3), "enhancing tumor": (3,)},
        num_training_cases=len(train),
        file_ending=".nii.gz",
        regions_class_order=(1, 2, 3),
        license="see https://www.synapse.org/#!Synapse:syn25829067/wiki/610863",
        reference="see https://www.synapse.org/#!Synapse:syn25829067/wiki/610863",
        dataset_release="1.0",
    )

    # Record provenance next to the data, so the split used is auditable.
    (out_base / "SPLIT_PROVENANCE.json").write_text(json.dumps({
        "split_source": str(SPLIT.relative_to(ROOT)),
        "split_rule": split["source"],
        "imagesTr": len(train),
        "imagesTs": len(heldout),
        "note": ("imagesTr holds the 1000 training cases only. The 251 held-out "
                 "cases are images-only in imagesTs and carry no labels here, so "
                 "nnU-Net cannot train or select on them."),
    }, indent=2))
    print(f"done. {out_base}")


if __name__ == "__main__":
    main()
