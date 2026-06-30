#!/usr/bin/env python3
"""check_determinism.py — Load same patient 5 times, compare GT voxels."""
from brats import get_datasets

ds = get_datasets()

# Find index of BraTS2021_00109
target = "BraTS2021_00109"
idx = None
for i in range(len(ds)):
    pid = str(ds[i].get("patient_id", ds[i].get("id", i)))
    if pid == target:
        idx = i
        break

if idx is None:
    print(f"❌ Patient {target} not found in dataset!")
else:
    print(f"Testing determinism for {target} at index {idx}\n")
    for run in range(5):
        item = ds[idx]
        lbl  = item["label"]
        img  = item["image"]
        print(f"  Run {run+1}: ET={lbl[0].sum():.0f}  TC={lbl[1].sum():.0f}  WT={lbl[2].sum():.0f}  "
              f"| image_mean={img.mean():.6f}")

    print()
    print("If numbers differ across runs → random crop/augmentation is active during evaluation!")
    print("If numbers are identical     → the bug is elsewhere in the eval loop.")