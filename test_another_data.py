#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OOM-safe evaluation for fine-tuned pontine binary Wavelet U-Net++ model.

Model:
    WaveletUNetPlusPlus(in_channels=4, n_classes=1)

Latest checkpoint:
    checkpoints_pontine_binary_full_continue_t02/pontine_binary_full_best.pth

Dataset:
    images: /home/kamilabdelali/anotherdata/images
    labels: /home/kamilabdelali/anotherdata/labels

This script:
- Evaluates binary NIfTI segmentation.
- Uses your fine-tuned binary checkpoint.
- Uses output channel 0.
- Converts single-channel NIfTI images to 4-channel input by repeating the image.
- Uses no TTA by default to avoid CUDA OOM.
- Uses mixed precision inference on CUDA.
- Saves per-patient Dice, HD95, sensitivity, precision to CSV.
"""

import os
import gc
import re
import argparse
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
import scipy.ndimage as ndimage
import torch

from medpy import metric
from model import WaveletUNetPlusPlus


warnings.filterwarnings("ignore", category=RuntimeWarning)


# ============================================================
# DEFAULT CONFIG
# ============================================================

DEFAULT_IMAGES_DIR = "/home/kamilabdelali/anotherdata/images"
DEFAULT_LABELS_DIR = "/home/kamilabdelali/anotherdata/labels"

# Latest best checkpoint after continuation training
DEFAULT_CHECKPOINT = "checkpoints_pontine_binary_full_continue_t02/pontine_binary_full_best.pth"

DEFAULT_OUTPUT_CSV = "pontine_eval_continue_t02.csv"

# Use same shape as training for best consistency.
# If CUDA OOM happens, run with --target_shape 96 96 96 or 80 80 80.
DEFAULT_TARGET_SHAPE = (128, 128, 128)

MODEL_IN_CHANNELS = 4
MODEL_N_CLASSES = 1

APPLY_SIGMOID = True

DEFAULT_BINARY_CHANNEL = 0
DEFAULT_THRESHOLD = 0.20
DEFAULT_MIN_VOXELS = 0


# ============================================================
# MEMORY HELPERS
# ============================================================

def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def is_cuda_oom_error(e):
    msg = str(e).lower()
    return "out of memory" in msg or "cuda error: out of memory" in msg


# ============================================================
# FILE HELPERS
# ============================================================

def natural_sort_key(s):
    s = str(s)
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", s)
    ]


def list_nifti_files(folder):
    folder = Path(folder)
    files = list(folder.glob("*.nii")) + list(folder.glob("*.nii.gz"))
    return sorted(files, key=lambda p: natural_sort_key(p.name))


def case_id_from_path(path):
    name = Path(path).name

    if name.endswith(".nii.gz"):
        return name[:-7]

    if name.endswith(".nii"):
        return name[:-4]

    return Path(path).stem


def match_image_label_pairs(images_dir, labels_dir):
    image_files = list_nifti_files(images_dir)
    label_files = list_nifti_files(labels_dir)

    image_map = {case_id_from_path(p): p for p in image_files}
    label_map = {case_id_from_path(p): p for p in label_files}

    common_ids = sorted(
        set(image_map.keys()) & set(label_map.keys()),
        key=natural_sort_key,
    )

    pairs = []

    for cid in common_ids:
        pairs.append(
            {
                "Patient_ID": cid,
                "Image_Path": str(image_map[cid]),
                "Label_Path": str(label_map[cid]),
            }
        )

    unmatched_images = sorted(
        set(image_map.keys()) - set(label_map.keys()),
        key=natural_sort_key,
    )

    unmatched_labels = sorted(
        set(label_map.keys()) - set(image_map.keys()),
        key=natural_sort_key,
    )

    return pairs, image_files, label_files, unmatched_images, unmatched_labels


# ============================================================
# NIFTI LOADING / PREPROCESSING
# ============================================================

def load_nifti(path):
    nii = nib.load(str(path))
    arr = nii.get_fdata()
    return arr


def normalize_one_volume(x):
    x = x.astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    nonzero = x[x != 0]

    if nonzero.size > 10:
        mean = float(nonzero.mean())
        std = float(nonzero.std())
    else:
        mean = float(x.mean())
        std = float(x.std())

    if std < 1e-8:
        return np.zeros_like(x, dtype=np.float32)

    x = (x - mean) / std
    x = np.clip(x, -5.0, 5.0)

    return x.astype(np.float32)


def resize_3d(arr, target_shape, order):
    factors = (
        target_shape[0] / arr.shape[0],
        target_shape[1] / arr.shape[1],
        target_shape[2] / arr.shape[2],
    )

    out = ndimage.zoom(arr, zoom=factors, order=order)
    return out


def convert_image_to_4_channels(img):
    """
    Convert loaded image to [4, D, H, W].

    Supported:
    - 3D image [D, H, W] -> repeated 4 times
    - 4D image [D, H, W, 4] -> converted to [4, D, H, W]
    - 4D image [4, D, H, W] -> kept as [4, D, H, W]

    If 4D but not 4 channels, first channel is repeated.
    """

    img = np.asarray(img)

    if img.ndim == 3:
        img = normalize_one_volume(img)
        img_4ch = np.stack([img, img, img, img], axis=0)
        return img_4ch.astype(np.float32)

    if img.ndim == 4:
        # Channels first: [4, D, H, W]
        if img.shape[0] == 4:
            channels = []
            for c in range(4):
                channels.append(normalize_one_volume(img[c]))
            return np.stack(channels, axis=0).astype(np.float32)

        # Channels last: [D, H, W, 4]
        if img.shape[-1] == 4:
            channels = []
            for c in range(4):
                channels.append(normalize_one_volume(img[..., c]))
            return np.stack(channels, axis=0).astype(np.float32)

        # 4D but not 4 channels: use first channel and repeat
        one = normalize_one_volume(img[..., 0])
        img_4ch = np.stack([one, one, one, one], axis=0)
        return img_4ch.astype(np.float32)

    raise ValueError(f"Unsupported image shape: {img.shape}")


def resize_4ch_image(img_4ch, target_shape):
    """
    img_4ch shape: [4, D, H, W]
    return shape: [4, target_D, target_H, target_W]
    """

    resized = []

    for c in range(img_4ch.shape[0]):
        resized_c = resize_3d(img_4ch[c], target_shape, order=1)
        resized.append(resized_c.astype(np.float32))

    return np.stack(resized, axis=0).astype(np.float32)


def prepare_label_binary(lbl, target_shape):
    lbl = np.asarray(lbl)

    if lbl.ndim == 4:
        if lbl.shape[-1] == 1:
            lbl = lbl[..., 0]
        elif lbl.shape[0] == 1:
            lbl = lbl[0]
        else:
            # If multi-channel label, union all positive channels
            if lbl.shape[-1] <= 10:
                lbl = np.any(lbl > 0, axis=-1).astype(np.uint8)
            else:
                lbl = np.any(lbl > 0, axis=0).astype(np.uint8)

    if lbl.ndim != 3:
        raise ValueError(f"Unsupported label shape: {lbl.shape}")

    lbl = (lbl > 0).astype(np.uint8)
    lbl = resize_3d(lbl, target_shape, order=0)
    lbl = (lbl > 0).astype(np.uint8)

    return lbl


def preprocess_case(image_path, label_path, target_shape):
    img = load_nifti(image_path)
    lbl = load_nifti(label_path)

    img_4ch = convert_image_to_4_channels(img)
    img_4ch = resize_4ch_image(img_4ch, target_shape)

    lbl_bin = prepare_label_binary(lbl, target_shape)

    image_t = torch.from_numpy(img_4ch).float().unsqueeze(0)
    # shape: [1, 4, D, H, W]

    return image_t, lbl_bin


# ============================================================
# METRICS
# ============================================================

def dice_binary_np(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    pred_sum = int(pred.sum())
    gt_sum = int(gt.sum())

    if pred_sum == 0 and gt_sum == 0:
        return np.nan

    if pred_sum == 0 or gt_sum == 0:
        return 0.0

    inter = np.logical_and(pred, gt).sum()
    dice = (2.0 * inter) / (pred_sum + gt_sum)

    return float(dice)


def hd95_binary_np(pred, gt):
    pred = pred.astype(np.uint8)
    gt = gt.astype(np.uint8)

    pred_nz = np.count_nonzero(pred) > 0
    gt_nz = np.count_nonzero(gt) > 0

    if not pred_nz and not gt_nz:
        return np.nan

    if pred_nz and gt_nz:
        try:
            return float(metric.binary.hd95(pred, gt))
        except Exception:
            return np.nan

    return np.nan


def sensitivity_binary_np(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    tp = np.logical_and(pred, gt).sum()
    fn = np.logical_and(~pred, gt).sum()

    denom = tp + fn

    if denom == 0:
        return np.nan

    return float(tp / denom)


def precision_binary_np(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()

    denom = tp + fp

    if denom == 0:
        return np.nan

    return float(tp / denom)


# ============================================================
# POSTPROCESSING
# ============================================================

def remove_small_components_np(mask, min_voxels):
    mask = mask.astype(np.uint8)

    if min_voxels <= 0:
        return mask

    labels, n = ndimage.label(mask)

    if n == 0:
        return mask

    sizes = np.bincount(labels.ravel())
    keep = np.zeros_like(mask, dtype=bool)

    for comp_id in range(1, n + 1):
        if sizes[comp_id] >= min_voxels:
            keep |= labels == comp_id

    return keep.astype(np.uint8)


def keep_largest_component_np(mask):
    mask = mask.astype(np.uint8)

    labels, n = ndimage.label(mask)

    if n == 0:
        return mask

    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    largest = sizes.argmax()

    return (labels == largest).astype(np.uint8)


def fill_holes_np(mask):
    return ndimage.binary_fill_holes(mask.astype(bool)).astype(np.uint8)


def postprocess_binary_mask(mask, min_voxels=0, fill_holes=False, keep_largest=False):
    mask = mask.astype(np.uint8)

    if min_voxels > 0:
        mask = remove_small_components_np(mask, min_voxels=min_voxels)

    if keep_largest:
        mask = keep_largest_component_np(mask)

    if fill_holes:
        mask = fill_holes_np(mask)

    return mask.astype(np.uint8)


# ============================================================
# MODEL
# ============================================================

def load_model(device, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print("Building model:")
    print(f"  WaveletUNetPlusPlus(in_channels={MODEL_IN_CHANNELS}, n_classes={MODEL_N_CLASSES})")

    model = WaveletUNetPlusPlus(
        in_channels=MODEL_IN_CHANNELS,
        n_classes=MODEL_N_CLASSES,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    clean_state_dict = {}

    for k, v in state_dict.items():
        if k.startswith("module."):
            clean_state_dict[k[7:]] = v
        else:
            clean_state_dict[k] = v

    model.load_state_dict(clean_state_dict, strict=True)
    model.eval()

    print(f"✅ Weights loaded successfully from: {checkpoint_path}")

    return model


# ============================================================
# INFERENCE
# ============================================================

def forward_model(model, x, apply_sigmoid=True):
    out = model(x)

    if isinstance(out, (tuple, list)):
        out = out[0]

    if apply_sigmoid:
        out = torch.sigmoid(out)

    return out


def predict_probs_oom_safe(model, image_t, device, use_tta=False):
    """
    Returns probs on CPU.
    Shape for binary model: [1, 1, D, H, W]
    """

    image_t = image_t.to(device, non_blocking=True)

    with torch.inference_mode():
        if device.type == "cuda":
            try:
                autocast_ctx = torch.amp.autocast("cuda", enabled=True)
            except Exception:
                autocast_ctx = torch.cuda.amp.autocast(enabled=True)
        else:
            autocast_ctx = torch.amp.autocast("cpu", enabled=False)

        with autocast_ctx:
            probs = forward_model(model, image_t, apply_sigmoid=APPLY_SIGMOID)

            if use_tta:
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
                    x_flip = torch.flip(image_t, dims=dims)
                    p_flip = forward_model(model, x_flip, apply_sigmoid=APPLY_SIGMOID)
                    p_flip = torch.flip(p_flip, dims=dims)

                    probs = probs + p_flip

                    del x_flip
                    del p_flip
                    cleanup_memory()

                probs = probs / float(1 + len(flip_configs))

    probs_cpu = probs.detach().float().cpu()

    del probs
    del image_t

    cleanup_memory()

    return probs_cpu


# ============================================================
# EVALUATE SINGLE CASE
# ============================================================

def evaluate_case(
    model,
    pair,
    device,
    target_shape,
    binary_channel,
    threshold,
    min_voxels,
    use_tta,
    fill_holes,
    keep_largest,
    debug=False,
):
    pid = pair["Patient_ID"]
    image_path = pair["Image_Path"]
    label_path = pair["Label_Path"]

    image_t, label_bin = preprocess_case(
        image_path=image_path,
        label_path=label_path,
        target_shape=target_shape,
    )

    probs = predict_probs_oom_safe(
        model=model,
        image_t=image_t,
        device=device,
        use_tta=use_tta,
    )

    if probs.ndim != 5:
        raise ValueError(f"Expected output [B, C, D, H, W], got {tuple(probs.shape)}")

    if probs.shape[1] <= binary_channel:
        raise ValueError(
            f"Model output has {probs.shape[1]} channels, but binary_channel={binary_channel}"
        )

    prob = probs[0, binary_channel].numpy()

    pred_mask_raw = (prob >= threshold).astype(np.uint8)

    pred_mask = postprocess_binary_mask(
        pred_mask_raw,
        min_voxels=min_voxels,
        fill_holes=fill_holes,
        keep_largest=keep_largest,
    )

    dice = dice_binary_np(pred_mask, label_bin)
    hd95 = hd95_binary_np(pred_mask, label_bin)
    sens = sensitivity_binary_np(pred_mask, label_bin)
    prec = precision_binary_np(pred_mask, label_bin)

    gt_voxels = int(label_bin.sum())
    pred_voxels = int(pred_mask.sum())
    pred_raw_voxels = int(pred_mask_raw.sum())
    overlap_voxels = int(np.logical_and(pred_mask > 0, label_bin > 0).sum())

    result = {
        "Patient_ID": pid,
        "Dice": dice,
        "HD95": hd95,
        "Sensitivity": sens,
        "Precision": prec,
        "GT_Voxels": gt_voxels,
        "Pred_Raw_Voxels": pred_raw_voxels,
        "Pred_Voxels": pred_voxels,
        "Overlap_Voxels": overlap_voxels,
        "Prob_Min": float(prob.min()),
        "Prob_Max": float(prob.max()),
        "Prob_Mean": float(prob.mean()),
        "Prob_Std": float(prob.std()),
        "Voxels_Above_Threshold": int((prob >= threshold).sum()),
        "Binary_Channel": int(binary_channel),
        "Threshold": float(threshold),
        "Target_Shape": str(tuple(target_shape)),
        "Image_Path": image_path,
        "Label_Path": label_path,
        "Error": "",
    }

    if debug:
        print(f"\n[DEBUG] {pid}")
        print(f"  image tensor shape : {tuple(image_t.shape)}")
        print(f"  probs shape        : {tuple(probs.shape)}")
        print(f"  binary channel     : {binary_channel}")
        print(f"  threshold          : {threshold}")
        print(f"  prob min/max/mean  : {prob.min():.6f} / {prob.max():.6f} / {prob.mean():.6f}")
        print(f"  GT voxels          : {gt_voxels}")
        print(f"  Pred raw voxels    : {pred_raw_voxels}")
        print(f"  Pred final voxels  : {pred_voxels}")
        print(f"  Overlap voxels     : {overlap_voxels}")
        print(f"  Dice               : {dice}")
        print(f"  HD95               : {hd95}")
        print(f"  Sensitivity        : {sens}")
        print(f"  Precision          : {prec}")

    del image_t
    del probs
    del prob
    del pred_mask_raw
    del pred_mask
    del label_bin

    cleanup_memory()

    return result


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--images_dir", type=str, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--labels_dir", type=str, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output_csv", type=str, default=DEFAULT_OUTPUT_CSV)

    parser.add_argument(
        "--target_shape",
        type=int,
        nargs=3,
        default=DEFAULT_TARGET_SHAPE,
        help="Example: --target_shape 128 128 128",
    )

    parser.add_argument(
        "--binary_channel",
        type=int,
        default=DEFAULT_BINARY_CHANNEL,
        help="For binary model, use 0.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
    )

    parser.add_argument(
        "--min_voxels",
        type=int,
        default=DEFAULT_MIN_VOXELS,
    )

    parser.add_argument(
        "--fill_holes",
        action="store_true",
        help="Fill holes in predicted masks.",
    )

    parser.add_argument(
        "--keep_largest",
        action="store_true",
        help="Keep only largest connected component.",
    )

    parser.add_argument(
        "--no_tta",
        action="store_true",
        help="Disable TTA. Recommended.",
    )

    parser.add_argument(
        "--use_tta",
        action="store_true",
        help="Enable TTA. Not recommended if CUDA OOM happens.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
    )

    parser.add_argument(
        "--max_cases",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--debug_first",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--save_every",
        type=int,
        default=25,
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    target_shape = tuple(args.target_shape)

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.no_tta:
        use_tta = False
    elif args.use_tta:
        use_tta = True
    else:
        use_tta = False

    print("=" * 80)
    print("Fine-tuned Binary WaveletUNetPlusPlus Evaluation")
    print("=" * 80)
    print(f"Images dir        : {args.images_dir}")
    print(f"Labels dir        : {args.labels_dir}")
    print(f"Checkpoint        : {args.checkpoint}")
    print(f"Output CSV        : {args.output_csv}")
    print(f"Device            : {device}")
    print(f"Target shape      : {target_shape}")
    print(f"Use TTA           : {use_tta}")
    print(f"Model in_channels : {MODEL_IN_CHANNELS}")
    print(f"Model n_classes   : {MODEL_N_CLASSES}")
    print(f"Binary channel    : {args.binary_channel}")
    print(f"Threshold         : {args.threshold}")
    print(f"Min voxels        : {args.min_voxels}")
    print(f"Fill holes        : {args.fill_holes}")
    print(f"Keep largest      : {args.keep_largest}")
    print("=" * 80)

    cleanup_memory()

    pairs, image_files, label_files, unmatched_images, unmatched_labels = match_image_label_pairs(
        args.images_dir,
        args.labels_dir,
    )

    print(f"Found image files : {len(image_files)}")
    print(f"Found label files : {len(label_files)}")
    print(f"Found pairs       : {len(pairs)}")
    print(f"Unmatched images  : {len(unmatched_images)}")
    print(f"Unmatched labels  : {len(unmatched_labels)}")

    if len(pairs) == 0:
        raise RuntimeError("No matched image-label pairs found.")

    print("\nSample pairs:")
    for p in pairs[:5]:
        print(f"  IMG: {Path(p['Image_Path']).name}")
        print(f"  LBL: {Path(p['Label_Path']).name}")

    try:
        first_label = load_nifti(pairs[0]["Label_Path"])
        unique_vals = np.unique(first_label)
        if unique_vals.size > 20:
            unique_vals = unique_vals[:20]
        print(f"\nLabel unique preview: {unique_vals.tolist()}")
    except Exception as e:
        print(f"\nCould not preview labels: {e}")

    if args.max_cases is not None:
        pairs = pairs[:args.max_cases]
        print(f"\nUsing max_cases    : {args.max_cases}")

    print("\nLoading model...")
    model = load_model(device, args.checkpoint)

    cleanup_memory()

    results = []

    print(f"\nStarting evaluation for {len(pairs)} patients...")

    for idx, pair in enumerate(pairs):
        try:
            debug = args.debug_first > 0 and idx < args.debug_first

            result = evaluate_case(
                model=model,
                pair=pair,
                device=device,
                target_shape=target_shape,
                binary_channel=args.binary_channel,
                threshold=args.threshold,
                min_voxels=args.min_voxels,
                use_tta=use_tta,
                fill_holes=args.fill_holes,
                keep_largest=args.keep_largest,
                debug=debug,
            )

            results.append(result)

            del result
            cleanup_memory()

        except RuntimeError as e:
            if is_cuda_oom_error(e):
                print("\n" + "=" * 80)
                print(f"❌ CUDA OOM at patient: {pair['Patient_ID']}")
                print("=" * 80)
                print("GPU memory is not enough for this input/model.")
                print("The script will stop now instead of repeating OOM errors.")
                print("\nTry:")
                print("  python test_another_data.py --target_shape 96 96 96 --no_tta")
                print("or:")
                print("  python test_another_data.py --target_shape 80 80 80 --no_tta")
                print("or CPU:")
                print("  python test_another_data.py --device cpu --target_shape 128 128 128 --no_tta")
                print("=" * 80)

                cleanup_memory()

                if len(results) > 0:
                    partial_csv = args.output_csv.replace(".csv", "_PARTIAL.csv")
                    pd.DataFrame(results).to_csv(partial_csv, index=False)
                    print(f"Partial results saved to: {partial_csv}")

                raise

            print(f"❌ Runtime error processing {pair['Patient_ID']}: {e}")
            traceback.print_exc()

            results.append(
                {
                    "Patient_ID": pair["Patient_ID"],
                    "Dice": np.nan,
                    "HD95": np.nan,
                    "Sensitivity": np.nan,
                    "Precision": np.nan,
                    "GT_Voxels": np.nan,
                    "Pred_Raw_Voxels": np.nan,
                    "Pred_Voxels": np.nan,
                    "Overlap_Voxels": np.nan,
                    "Prob_Min": np.nan,
                    "Prob_Max": np.nan,
                    "Prob_Mean": np.nan,
                    "Prob_Std": np.nan,
                    "Voxels_Above_Threshold": np.nan,
                    "Binary_Channel": args.binary_channel,
                    "Threshold": args.threshold,
                    "Target_Shape": str(tuple(target_shape)),
                    "Image_Path": pair["Image_Path"],
                    "Label_Path": pair["Label_Path"],
                    "Error": str(e),
                }
            )

            cleanup_memory()

        except Exception as e:
            print(f"❌ Error processing {pair['Patient_ID']}: {e}")
            traceback.print_exc()

            results.append(
                {
                    "Patient_ID": pair["Patient_ID"],
                    "Dice": np.nan,
                    "HD95": np.nan,
                    "Sensitivity": np.nan,
                    "Precision": np.nan,
                    "GT_Voxels": np.nan,
                    "Pred_Raw_Voxels": np.nan,
                    "Pred_Voxels": np.nan,
                    "Overlap_Voxels": np.nan,
                    "Prob_Min": np.nan,
                    "Prob_Max": np.nan,
                    "Prob_Mean": np.nan,
                    "Prob_Std": np.nan,
                    "Voxels_Above_Threshold": np.nan,
                    "Binary_Channel": args.binary_channel,
                    "Threshold": args.threshold,
                    "Target_Shape": str(tuple(target_shape)),
                    "Image_Path": pair["Image_Path"],
                    "Label_Path": pair["Label_Path"],
                    "Error": str(e),
                }
            )

            cleanup_memory()

        done = idx + 1

        if done % args.save_every == 0 or done == len(pairs):
            print(f"✅ Processed {done}/{len(pairs)}")

            temp_csv = args.output_csv.replace(".csv", "_TEMP.csv")
            pd.DataFrame(results).to_csv(temp_csv, index=False)

    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)

    dice_values = pd.to_numeric(df["Dice"], errors="coerce")
    hd95_values = pd.to_numeric(df["HD95"], errors="coerce")
    sens_values = pd.to_numeric(df["Sensitivity"], errors="coerce")
    prec_values = pd.to_numeric(df["Precision"], errors="coerce")

    pred_empty = (pd.to_numeric(df["Pred_Voxels"], errors="coerce") == 0).sum()
    gt_empty = (pd.to_numeric(df["GT_Voxels"], errors="coerce") == 0).sum()

    print("\n" + "=" * 80)
    print("Evaluation complete")
    print("=" * 80)
    print(f"Saved CSV          : {args.output_csv}")
    print(f"Total cases        : {len(df)}")
    print(f"Valid Dice cases   : {dice_values.notna().sum()} / {len(df)}")
    print(f"Mean Dice          : {dice_values.mean():.4f}")
    print(f"Median Dice        : {dice_values.median():.4f}")
    print(f"STD Dice           : {dice_values.std():.4f}")
    print(f"Mean HD95          : {hd95_values.mean():.4f}")
    print(f"Median HD95        : {hd95_values.median():.4f}")
    print(f"Mean Sensitivity   : {sens_values.mean():.4f}")
    print(f"Mean Precision     : {prec_values.mean():.4f}")
    print(f"Empty predictions  : {pred_empty}")
    print(f"Empty GT masks     : {gt_empty}")
    print("=" * 80)

    print("\nWorst 10 cases by Dice:")
    try:
        worst = df.sort_values("Dice", ascending=True).head(10)
        cols = [
            "Patient_ID",
            "Dice",
            "HD95",
            "Sensitivity",
            "Precision",
            "GT_Voxels",
            "Pred_Voxels",
            "Overlap_Voxels",
            "Prob_Max",
        ]
        print(worst[cols].to_string(index=False))
    except Exception:
        pass

    print("\nBest 10 cases by Dice:")
    try:
        best = df.sort_values("Dice", ascending=False).head(10)
        cols = [
            "Patient_ID",
            "Dice",
            "HD95",
            "Sensitivity",
            "Precision",
            "GT_Voxels",
            "Pred_Voxels",
            "Overlap_Voxels",
            "Prob_Max",
        ]
        print(best[cols].to_string(index=False))
    except Exception:
        pass


if __name__ == "__main__":
    main()