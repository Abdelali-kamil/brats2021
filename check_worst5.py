# check_worst5.py
import nibabel as nib
import numpy as np
import os

DATA_DIR = "/home/kamilabdelali/brats2021/data"
worst = ["BraTS2021_01628","BraTS2021_01530","BraTS2021_01616",
         "BraTS2021_01433","BraTS2021_01480"]

for pid in worst:
    seg_path = os.path.join(DATA_DIR, pid, f"{pid}_seg.nii.gz")
    seg = nib.load(seg_path).get_fdata()
    et = np.sum(seg == 4)
    tc = np.sum((seg == 1) | (seg == 4))
    wt = np.sum(seg > 0)
    print(f"{pid}: ET={et:6d}  TC={tc:6d}  WT={wt:6d}  "
          f"{'⚠️ TINY ET' if et < 50 else ''}"
          f"{'⚠️ COMPLEX' if wt > 100000 else ''}")