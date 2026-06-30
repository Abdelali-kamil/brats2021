#!/usr/bin/env python3
"""
Debug overlay generator for Wavelet U-Net++ external binary dataset evaluation.

This script:
1. Reads FINAL_REPORT_ALIGNED.csv
2. Selects worst / best / random valid cases
3. Loads image and label NIfTI files
4. Runs the trained Wavelet U-Net++ model
5. Saves overlay PNGs showing:

    grayscale = image
    green     = ground-truth label
    red       = prediction
    yellow    = overlap

Recommended usage:

python debug_overlay_predictions.py \
  --csv FINAL_REPORT_ALIGNED.csv \
  --checkpoint checkpoints/segmentor_epoch_650.pth \
  --out_dir debug_overlays_valid_worst \
  --target_shape 128 128 128 \
  --preprocess_mode resize \
  --pad_multiple 16 \
  --num_cases 20 \
  --mode worst
"""

import os
import re
import argparse
import warnings

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

from model import WaveletUNetPlusPlus

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ============================================================
# MODEL / BraTS CONFIG
# ============================================================
CHANNEL_ET = 0
CHANNEL_TC = 1
CHANNEL_WT = 2

BINARY_CHANNEL = CHANNEL_WT

THRESH_ET = 0.46
THRESH_TC = 0.50
THRESH_WT = 0.50

LOW_ET = 0.15
LOW_TC = 0.25
LOW_WT = 0.25

MIN_VOXELS_ET = 1
MIN_VOXELS_TC = 5
MIN_VOXELS_WT = 20

USE_TTA = True
USE_POSTPROCESS = True
APPLY_SIGMOID = True


# ============================================================
# GENERAL HELPERS
# ============================================================
def patient_id_from_path(path):
    name = os.path.basename(path)
    name = re.sub(r"\.nii(?:\.gz)?$", "", name)
    return name


def minmax_norm(x):
    x = x.astype(np.float32)

    mn = float(np.min(x))
    mx = float(np.max(x))

    if mx - mn < 1e-8:
        return np.zeros_like(x, dtype=np.float32)

    return ((x - mn) / (mx - mn + 1e-8)).astype(np.float32)


def load_nifti_canonical(path):
    """
    Load NIfTI and convert to closest canonical orientation.

    Important:
    The image and label are both passed through this function, so orientation
    should remain matched if they originally share the same affine/orientation.
    """
    obj = nib.as_closest_canonical(nib.load(path))
    return obj.get_fdata()


def resize_3d(vol, target_shape=(128, 128, 128), order=1):
    zoom = (
        target_shape[0] / vol.shape[0],
        target_shape[1] / vol.shape[1],
        target_shape[2] / vol.shape[2],
    )

    return ndimage.zoom(vol, zoom=zoom, order=order)


def center_pad_or_crop_3d(vol, target_shape=(128, 128, 128), pad_value=0):
    out = vol

    for axis, target_size in enumerate(target_shape):
        current_size = out.shape[axis]

        if current_size > target_size:
            start = (current_size - target_size) // 2
            end = start + target_size

            slicer = [slice(None)] * 3
            slicer[axis] = slice(start, end)

            out = out[tuple(slicer)]

    pad_width = []

    for axis, target_size in enumerate(target_shape):
        current_size = out.shape[axis]

        if current_size < target_size:
            total_pad = target_size - current_size
            before = total_pad // 2
            after = total_pad - before
            pad_width.append((before, after))
        else:
            pad_width.append((0, 0))

    if any(p != (0, 0) for p in pad_width):
        out = np.pad(out, pad_width, mode="constant", constant_values=pad_value)

    return out


def make_same_size_3d(vol, target_shape, order=1, mode="resize"):
    if mode == "resize":
        return resize_3d(
            vol,
            target_shape=target_shape,
            order=order,
        )

    if mode == "padcrop":
        return center_pad_or_crop_3d(
            vol,
            target_shape=target_shape,
            pad_value=0,
        )

    raise ValueError(f"Unknown preprocess mode: {mode}")


def pad_to_multiple_3d(vol, multiple=16, pad_value=0):
    if multiple <= 1:
        return vol

    pad_width = []

    for axis in range(3):
        size = vol.shape[axis]
        rem = size % multiple
        add = 0 if rem == 0 else multiple - rem

        before = add // 2
        after = add - before

        pad_width.append((before, after))

    if any(p != (0, 0) for p in pad_width):
        vol = np.pad(
            vol,
            pad_width,
            mode="constant",
            constant_values=pad_value,
        )

    return vol


# ============================================================
# IMAGE / LABEL LOADING
# ============================================================
def load_image_as_4ch(img_path, target_shape, preprocess_mode, pad_multiple):
    """
    Load external image as 4-channel input.

    If the image is 3D:
        duplicate it into 4 channels.

    If the image is 4D:
        use available channels, then truncate/pad to 4 channels.

    Output shape:
        [4, D, H, W] or [4, X, Y, Z] depending on original convention.
    """
    data = load_nifti_canonical(img_path)

    channels = []

    if data.ndim == 3:
        x = minmax_norm(data)

        x = make_same_size_3d(
            x,
            target_shape=target_shape,
            order=1,
            mode=preprocess_mode,
        )

        x = pad_to_multiple_3d(
            x,
            multiple=pad_multiple,
            pad_value=0,
        )

        channels = [x, x, x, x]

    elif data.ndim == 4:
        # Case: [X, Y, Z, C]
        if data.shape[-1] <= 10:
            for c in range(data.shape[-1]):
                ch = minmax_norm(data[..., c])

                ch = make_same_size_3d(
                    ch,
                    target_shape=target_shape,
                    order=1,
                    mode=preprocess_mode,
                )

                ch = pad_to_multiple_3d(
                    ch,
                    multiple=pad_multiple,
                    pad_value=0,
                )

                channels.append(ch)

        # Case: [C, X, Y, Z]
        elif data.shape[0] <= 10:
            for c in range(data.shape[0]):
                ch = minmax_norm(data[c])

                ch = make_same_size_3d(
                    ch,
                    target_shape=target_shape,
                    order=1,
                    mode=preprocess_mode,
                )

                ch = pad_to_multiple_3d(
                    ch,
                    multiple=pad_multiple,
                    pad_value=0,
                )

                channels.append(ch)

        else:
            raise ValueError(f"Unsupported 4D image shape: {data.shape}")

    else:
        raise ValueError(f"Unsupported image ndim: {data.ndim}, shape={data.shape}")

    if len(channels) >= 4:
        channels = channels[:4]
    else:
        while len(channels) < 4:
            channels.append(channels[-1].copy())

    return np.stack(channels, axis=0).astype(np.float32)


def load_binary_label(label_path, target_shape, preprocess_mode, pad_multiple):
    """
    Load binary label.

    Any value > 0 is treated as lesion.
    """
    data = load_nifti_canonical(label_path)

    if data.ndim == 4:
        if data.shape[-1] == 1:
            data = data[..., 0]
        elif data.shape[0] == 1:
            data = data[0]
        else:
            data = np.max(data, axis=-1)

    if data.ndim != 3:
        raise ValueError(f"Unsupported label shape: {data.shape}")

    label = (data > 0).astype(np.float32)

    label = make_same_size_3d(
        label,
        target_shape=target_shape,
        order=0,
        mode=preprocess_mode,
    )

    label = (label > 0.5).astype(np.float32)

    label = pad_to_multiple_3d(
        label,
        multiple=pad_multiple,
        pad_value=0,
    )

    label = (label > 0.5).astype(np.float32)

    return label


# ============================================================
# POSTPROCESSING
# ============================================================
def remove_small_noise(tensor_mask, min_voxels):
    if min_voxels <= 0:
        return tensor_mask

    np_mask = tensor_mask.detach().cpu().numpy().astype(np.uint8)

    labels, n = ndimage.label(np_mask)

    if n == 0:
        return tensor_mask

    sizes = np.bincount(labels.ravel())

    valid = np.where(sizes >= min_voxels)[0]
    valid = valid[valid != 0]

    if len(valid) == 0:
        sizes[0] = 0
        largest = sizes.argmax()

        if largest == 0:
            return tensor_mask

        valid = np.array([largest])

    clean = np.isin(labels, valid).astype(np.float32)

    return torch.from_numpy(clean).to(tensor_mask.device)


def fill_holes_3d(tensor_mask):
    np_mask = tensor_mask.detach().cpu().numpy().astype(np.uint8)
    filled = ndimage.binary_fill_holes(np_mask)

    return torch.from_numpy(filled.astype(np.float32)).to(tensor_mask.device)


def postprocess_brats_masks(pred_mask):
    """
    BraTS hierarchy:
        ET inside TC inside WT.
    """
    pred_mask = pred_mask.contiguous()

    for b in range(pred_mask.shape[0]):
        pred_mask[b, CHANNEL_ET] = remove_small_noise(
            pred_mask[b, CHANNEL_ET],
            MIN_VOXELS_ET,
        )

        pred_mask[b, CHANNEL_TC] = remove_small_noise(
            pred_mask[b, CHANNEL_TC],
            MIN_VOXELS_TC,
        )
        pred_mask[b, CHANNEL_TC] = fill_holes_3d(pred_mask[b, CHANNEL_TC])

        pred_mask[b, CHANNEL_WT] = remove_small_noise(
            pred_mask[b, CHANNEL_WT],
            MIN_VOXELS_WT,
        )
        pred_mask[b, CHANNEL_WT] = fill_holes_3d(pred_mask[b, CHANNEL_WT])

        pred_mask[b, CHANNEL_TC] = torch.maximum(
            pred_mask[b, CHANNEL_TC],
            pred_mask[b, CHANNEL_ET],
        )

        pred_mask[b, CHANNEL_WT] = torch.maximum(
            pred_mask[b, CHANNEL_WT],
            pred_mask[b, CHANNEL_TC],
        )

    return pred_mask


# ============================================================
# MODEL LOADING / PREDICTION
# ============================================================
def load_model(checkpoint_path, device):
    model = WaveletUNetPlusPlus(
        in_channels=4,
        n_classes=3,
    ).to(device)

    ckpt = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    clean_state = {}

    for k, v in state_dict.items():
        clean_state[k.replace("module.", "")] = v

    model.load_state_dict(clean_state, strict=True)
    model.eval()

    print(f"Loaded checkpoint: {checkpoint_path}")

    return model


def predict(model, image_t):
    """
    Input:
        image_t: [B, 4, X, Y, Z]

    Output:
        raw_out:   raw logits
        probs:     sigmoid probabilities
        pred_mask: thresholded/postprocessed masks
    """
    raw_out = model(image_t)
    probs = torch.sigmoid(raw_out) if APPLY_SIGMOID else raw_out

    if USE_TTA:
        flip_configs = [
            [-1],
            [-2],
            [-3],
            [-1, -2],
            [-1, -3],
            [-2, -3],
            [-1, -2, -3],
        ]

        for dims in flip_configs:
            raw_f = model(torch.flip(image_t, dims=dims))
            prob_f = torch.sigmoid(raw_f) if APPLY_SIGMOID else raw_f
            probs += torch.flip(prob_f, dims=dims)

        probs = probs / (1 + len(flip_configs))

    pred_mask = torch.zeros_like(probs)

    pred_mask[:, CHANNEL_ET] = (probs[:, CHANNEL_ET] > THRESH_ET).float()
    pred_mask[:, CHANNEL_TC] = (probs[:, CHANNEL_TC] > THRESH_TC).float()
    pred_mask[:, CHANNEL_WT] = (probs[:, CHANNEL_WT] > THRESH_WT).float()

    for b in range(pred_mask.shape[0]):
        if pred_mask[b, CHANNEL_ET].sum() == 0 and probs[b, CHANNEL_ET].max() > LOW_ET:
            pred_mask[b, CHANNEL_ET] = (probs[b, CHANNEL_ET] > LOW_ET).float()

        if pred_mask[b, CHANNEL_TC].sum() == 0 and probs[b, CHANNEL_TC].max() > LOW_TC:
            pred_mask[b, CHANNEL_TC] = (probs[b, CHANNEL_TC] > LOW_TC).float()

        if pred_mask[b, CHANNEL_WT].sum() == 0 and probs[b, CHANNEL_WT].max() > LOW_WT:
            pred_mask[b, CHANNEL_WT] = (probs[b, CHANNEL_WT] > LOW_WT).float()

    if USE_POSTPROCESS:
        pred_mask = postprocess_brats_masks(pred_mask)

    return raw_out, probs, pred_mask


# ============================================================
# METRICS / DEBUG INFO
# ============================================================
def dice_np(pred, gt, smooth=1e-6):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    pred_sum = pred.sum()
    gt_sum = gt.sum()

    if pred_sum + gt_sum == 0:
        return np.nan

    inter = np.logical_and(pred, gt).sum()

    return float((2.0 * inter + smooth) / (pred_sum + gt_sum + smooth))


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


# ============================================================
# OVERLAY HELPERS
# ============================================================
def make_overlay_slice(image_slice, gt_slice, pred_slice):
    image_slice = minmax_norm(image_slice)

    rgb = np.stack(
        [image_slice, image_slice, image_slice],
        axis=-1,
    )

    gt = gt_slice > 0
    pred = pred_slice > 0
    overlap = gt & pred

    # Prediction = red
    rgb[pred, 0] = 1.0
    rgb[pred, 1] = 0.0
    rgb[pred, 2] = 0.0

    # Ground truth = green
    rgb[gt, 0] = 0.0
    rgb[gt, 1] = 1.0
    rgb[gt, 2] = 0.0

    # Overlap = yellow
    rgb[overlap, 0] = 1.0
    rgb[overlap, 1] = 1.0
    rgb[overlap, 2] = 0.0

    return rgb


def get_best_axial_slice(mask):
    sums = mask.sum(axis=(0, 1))

    if sums.max() == 0:
        return mask.shape[2] // 2

    return int(np.argmax(sums))


def get_best_coronal_slice(mask):
    sums = mask.sum(axis=(0, 2))

    if sums.max() == 0:
        return mask.shape[1] // 2

    return int(np.argmax(sums))


def get_best_sagittal_slice(mask):
    sums = mask.sum(axis=(1, 2))

    if sums.max() == 0:
        return mask.shape[0] // 2

    return int(np.argmax(sums))


def save_case_overlay(pid, image_4ch, gt, pred, probs_wt, out_dir, csv_dice=None):
    os.makedirs(out_dir, exist_ok=True)

    image = image_4ch[0]

    z = get_best_axial_slice(gt)
    y = get_best_coronal_slice(gt)
    x = get_best_sagittal_slice(gt)

    axial = make_overlay_slice(
        image[:, :, z],
        gt[:, :, z],
        pred[:, :, z],
    )

    coronal = make_overlay_slice(
        image[:, y, :],
        gt[:, y, :],
        pred[:, y, :],
    )

    sagittal = make_overlay_slice(
        image[x, :, :],
        gt[x, :, :],
        pred[x, :, :],
    )

    dice_here = dice_np(pred, gt)

    gt_vox = int(gt.sum())
    pred_vox = int(pred.sum())
    overlap_vox = int(np.logical_and(pred > 0, gt > 0).sum())
    prob_max = float(np.max(probs_wt))
    prob_mean = float(np.mean(probs_wt))
    vox_gt_low = int((probs_wt > LOW_WT).sum())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].imshow(np.rot90(axial))
    axes[0].set_title(f"Axial z={z}")

    axes[1].imshow(np.rot90(coronal))
    axes[1].set_title(f"Coronal y={y}")

    axes[2].imshow(np.rot90(sagittal))
    axes[2].set_title(f"Sagittal x={x}")

    for ax in axes:
        ax.axis("off")

    if csv_dice is None or np.isnan(csv_dice):
        csv_dice_text = "CSV Dice=NA"
    else:
        csv_dice_text = f"CSV Dice={csv_dice:.6f}"

    if np.isnan(dice_here):
        dice_text = "Overlay Dice=NaN"
    else:
        dice_text = f"Overlay Dice={dice_here:.6f}"

    title = (
        f"{pid} | green=GT, red=Pred, yellow=Overlap\n"
        f"{csv_dice_text} | {dice_text} | "
        f"GT={gt_vox} | Pred={pred_vox} | Overlap={overlap_vox} | "
        f"WT max={prob_max:.4f} | WT mean={prob_mean:.4f} | vox WT>{LOW_WT}={vox_gt_low}"
    )

    fig.suptitle(title, fontsize=10)

    out_path = os.path.join(out_dir, f"{pid}_overlay.png")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

    print(
        f"Saved: {out_path} | "
        f"Dice={dice_here} | GT={gt_vox} | Pred={pred_vox} | "
        f"Overlap={overlap_vox} | WTmax={prob_max:.4f}"
    )


# ============================================================
# CSV SELECTION
# ============================================================
def prepare_dataframe(csv_path):
    df = pd.read_csv(csv_path)

    required_cols = [
        "Patient_ID",
        "Dice",
        "GT_Voxels",
        "Pred_Voxels",
        "Image_Path",
        "Label_Path",
    ]

    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    df["Patient_ID"] = df["Patient_ID"].astype(str)
    df["Dice"] = pd.to_numeric(df["Dice"], errors="coerce")
    df["GT_Voxels"] = pd.to_numeric(df["GT_Voxels"], errors="coerce")
    df["Pred_Voxels"] = pd.to_numeric(df["Pred_Voxels"], errors="coerce")

    return df


def select_cases(df, mode, num_cases, allow_empty_gt=False):
    if allow_empty_gt:
        df_valid = df[df["Dice"].notna()].copy()
    else:
        df_valid = df[
            df["Dice"].notna() &
            (df["GT_Voxels"] > 0)
        ].copy()

    if len(df_valid) == 0:
        raise ValueError("No valid rows found after filtering.")

    if mode == "worst":
        sub = df_valid.nsmallest(num_cases, "Dice")
    elif mode == "best":
        sub = df_valid.nlargest(num_cases, "Dice")
    elif mode == "random":
        sub = df_valid.sample(
            min(num_cases, len(df_valid)),
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return sub


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        type=str,
        default="FINAL_REPORT_ALIGNED.csv",
        help="CSV generated by test_another_data.py",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/segmentor_epoch_650.pth",
        help="Model checkpoint path",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="debug_overlays",
        help="Directory to save overlay PNGs",
    )

    parser.add_argument(
        "--target_shape",
        type=int,
        nargs=3,
        default=[128, 128, 128],
        help="Target shape used during evaluation",
    )

    parser.add_argument(
        "--preprocess_mode",
        type=str,
        default="resize",
        choices=["resize", "padcrop"],
        help="Preprocessing mode used during evaluation",
    )

    parser.add_argument(
        "--pad_multiple",
        type=int,
        default=16,
        help="Pad dimensions to multiple of this value",
    )

    parser.add_argument(
        "--num_cases",
        type=int,
        default=20,
        help="Number of cases to visualize",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="worst",
        choices=["worst", "best", "random"],
        help="Select worst, best, or random cases",
    )

    parser.add_argument(
        "--allow_empty_gt",
        action="store_true",
        help="Allow empty GT cases in selection",
    )

    args = parser.parse_args()

    target_shape = tuple(args.target_shape)

    print("=" * 72)
    print("Debug Overlay Generator")
    print("=" * 72)
    print(f"CSV              : {args.csv}")
    print(f"Checkpoint       : {args.checkpoint}")
    print(f"Output dir       : {args.out_dir}")
    print(f"Target shape     : {target_shape}")
    print(f"Preprocess mode  : {args.preprocess_mode}")
    print(f"Pad multiple     : {args.pad_multiple}")
    print(f"Mode             : {args.mode}")
    print(f"Num cases        : {args.num_cases}")
    print(f"Allow empty GT   : {args.allow_empty_gt}")
    print("=" * 72)

    df = prepare_dataframe(args.csv)

    print("\nCSV summary:")
    print(f"Total rows        : {len(df)}")
    print(f"Valid Dice rows   : {df['Dice'].notna().sum()}")
    print(f"Non-empty GT rows : {(df['GT_Voxels'] > 0).sum()}")
    print(f"Empty GT rows     : {(df['GT_Voxels'] == 0).sum()}")
    print(f"Empty pred rows   : {(df['Pred_Voxels'] == 0).sum()}")

    sub = select_cases(
        df=df,
        mode=args.mode,
        num_cases=args.num_cases,
        allow_empty_gt=args.allow_empty_gt,
    )

    print("\nSelected cases:")
    print(
        sub[
            [
                "Patient_ID",
                "Dice",
                "GT_Voxels",
                "Pred_Voxels",
            ]
        ].to_string(index=False)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nDevice: {device}")

    model = load_model(
        checkpoint_path=args.checkpoint,
        device=device,
    )

    print(f"\nCreating overlays for {len(sub)} cases...")

    for _, row in sub.iterrows():
        pid = str(row["Patient_ID"])
        img_path = str(row["Image_Path"])
        lbl_path = str(row["Label_Path"])
        csv_dice = safe_float(row["Dice"])

        if not os.path.exists(img_path):
            print(f"WARNING: Image path does not exist: {img_path}")
            continue

        if not os.path.exists(lbl_path):
            print(f"WARNING: Label path does not exist: {lbl_path}")
            continue

        try:
            image_4ch = load_image_as_4ch(
                img_path=img_path,
                target_shape=target_shape,
                preprocess_mode=args.preprocess_mode,
                pad_multiple=args.pad_multiple,
            )

            gt = load_binary_label(
                label_path=lbl_path,
                target_shape=target_shape,
                preprocess_mode=args.preprocess_mode,
                pad_multiple=args.pad_multiple,
            )

            image_t = torch.from_numpy(image_4ch).unsqueeze(0).float().to(device)

            with torch.no_grad():
                raw_out, probs, pred_mask = predict(
                    model=model,
                    image_t=image_t,
                )

            pred = pred_mask[0, BINARY_CHANNEL].detach().cpu().numpy()
            probs_wt = probs[0, BINARY_CHANNEL].detach().cpu().numpy()

            save_case_overlay(
                pid=pid,
                image_4ch=image_4ch,
                gt=gt,
                pred=pred,
                probs_wt=probs_wt,
                out_dir=args.out_dir,
                csv_dice=csv_dice,
            )

        except Exception as e:
            print(f"ERROR processing patient {pid}: {e}")

    print("\nDone.")
    print(f"Overlays saved to: {args.out_dir}")


if __name__ == "__main__":
    main()