import os
import re
import argparse
import numpy as np
import pandas as pd
import torch
import nibabel as nib
import scipy.ndimage as ndimage
from pathlib import Path

from model import WaveletUNetPlusPlus

# ============================================================
# CONFIG
# ============================================================
DEFAULT_IMAGES_DIR = "/home/kamilabdelali/anotherdata/images"
DEFAULT_LABELS_DIR = "/home/kamilabdelali/anotherdata/labels"
DEFAULT_CHECKPOINT = "checkpoints_pontine_binary_full_continue_t02/pontine_binary_full_best.pth"
DEFAULT_OUTPUT_CSV = "pontine_eval_fixed.csv"

# ============================================================
# PREPROCESSING (Exact style that gave you 0.56 before)
# ============================================================
def normalize_brain_zscore(x):
    """Z-score on brain voxels only — matches your original working script."""
    x = x.astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    mask = x > 0
    if np.any(mask):
        mean = x[mask].mean()
        std = x[mask].std()
        if std > 1e-8:
            x[mask] = (x[mask] - mean) / std
            x[~mask] = 0.0
        else:
            x = np.zeros_like(x)
    return np.clip(x, -5.0, 5.0).astype(np.float32)

def resize_3d(arr, target_shape, order):
    factors = [t / a for t, a in zip(target_shape, arr.shape)]
    return ndimage.zoom(arr, zoom=factors, order=order)

# ============================================================
# HELPERS
# ============================================================
def match_pairs(images_dir, labels_dir):
    img_files = sorted(list(Path(images_dir).glob("*.nii*")), key=lambda p: p.name)
    lbl_files = sorted(list(Path(labels_dir).glob("*.nii*")), key=lambda p: p.name)
    img_map = {p.name.replace(".nii.gz", "").replace(".nii", ""): p for p in img_files}
    lbl_map = {p.name.replace(".nii.gz", "").replace(".nii", ""): p for p in lbl_files}
    common = sorted(set(img_map.keys()) & set(lbl_map.keys()))
    return [{"Patient_ID": c, "Image": img_map[c], "Label": lbl_map[c]} for c in common]

def dice_score(pred, gt):
    p, g = pred.astype(bool), gt.astype(bool)
    if not p.any() and not g.any():
        return 1.0
    if not p.any() or not g.any():
        return 0.0
    return 2.0 * (p & g).sum() / (p.sum() + g.sum())

def keep_largest_component(mask):
    lbl, n = ndimage.label(mask)
    
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, lbl, range(1, n + 1))
    largest = np.argmax(sizes) + 1
    return (lbl == largest).astype(np.uint8)

# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.35, help="Try 0.3, 0.35, or 0.5")
    parser.add_argument("--keep_largest", action="store_true", default=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Pontine Binary Model (n_classes=1)
    print("Loading pontine binary model...")
    model = WaveletUNetPlusPlus(in_channels=4, n_classes=1).to(device)
    ckpt = torch.load(DEFAULT_CHECKPOINT, map_location=device)
    state_dict = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    pairs = match_pairs(DEFAULT_IMAGES_DIR, DEFAULT_LABELS_DIR)
    print(f"Evaluating {len(pairs)} patients...")

    results = []
    for i, p in enumerate(pairs):
        try:
            # 1. Load data
            img = nib.load(str(p["Image"])).get_fdata()
            lbl = nib.load(str(p["Label"])).get_fdata()

            # 2. Normalize (brain-only z-score)
            img_norm = normalize_brain_zscore(img)

            # 3. Stack 4 channels (model expects 4 inputs)
            img_4ch = np.stack([img_norm] * 4, axis=0)  # [4, D, H, W]

            # 4. Resize FULL volume to 128³ (preserves lesion location)
            x = np.stack([resize_3d(img_4ch[c], (128, 128, 128), order=1) for c in range(4)], axis=0)
            y = (resize_3d((lbl > 0).astype(np.float32), (128, 128, 128), order=0) > 0.5).astype(np.uint8)

            # 5. Predict
            x_t = torch.from_numpy(x).float().unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(x_t)
                if isinstance(out, (list, tuple)):
                    out = out[0]
                prob = torch.sigmoid(out).cpu().numpy()[0, 0]

            # 6. Threshold
            mask = (prob >= args.threshold).astype(np.uint8)

            # 7. Post-process
            if args.keep_largest:
                mask = keep_largest_component(mask)

            # 8. Dice
            d = dice_score(mask, y)
            results.append({"Patient_ID": p["Patient_ID"], "Dice": d})

            # Debug first 10 patients
            if i < 10:
                print(f"{p['Patient_ID']:>10s} | MaxProb: {prob.max():.4f} | "
                      f"PredVox: {mask.sum():>6d} | GTVox: {y.sum():>6d} | Dice: {d:.4f}")
            elif i == 10:
                print("... (hiding remaining lines, will print final average)")

        except Exception as e:
            print(f"Error on {p['Patient_ID']}: {e}")

    df = pd.DataFrame(results)
    df.to_csv(DEFAULT_OUTPUT_CSV, index=False)

    if len(df) > 0:
        mean_dice = df["Dice"].mean()
        non_zero = (df["Dice"] > 0).sum()
        print(f"\n{'='*50}")
        print(f"Average Dice        : {mean_dice:.4f}")
        print(f"Non-zero Dice cases : {non_zero} / {len(df)}")
        print(f"{'='*50}")
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()