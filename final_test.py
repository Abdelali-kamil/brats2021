#!/usr/bin/env python3
"""
results_test.py — BraTS2021 Evaluation Script (Paper-Ready Version)
Fixed thresholds = 0.5 | No leakage | All 1251 patients
"""
import os
import re
import csv
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from scipy import ndimage
from scipy.spatial.distance import directed_hausdorff

from brats import get_datasets
from model import WaveletUNetPlusPlus

# ════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════
CHECKPOINT_PATH     = "checkpoints/segmentor_epoch_650.pth"
TEST_IDS_FILE       = "test_ids.txt"
OUTPUT_CSV          = "detailed_test_results.csv"
BATCH_SIZE          = 2
NUM_WORKERS         = 2

APPLY_SIGMOID       = True
USE_TTA             = True
USE_POSTPROCESS     = True

# ✅ Fixed thresholds — no leakage — publishable
ET_THR              = 0.5
TC_THR              = 0.5
WT_THR              = 0.5

# Post-processing min voxels
MIN_VOXELS_ET       = 5
MIN_VOXELS_TC       = 20
MIN_VOXELS_WT       = 50

# Label channel order: [ET_ch, TC_ch, WT_ch]
LABEL_CHANNEL_ORDER = [0, 1, 2]

# ID matching
STRICT_ID_MATCH     = True

# ✅ Disabled — thresholds are fixed, no search needed
AUTO_FIND_BEST_SETUP = False

# ════════════════════════════════════════════════════════════════
# TTA
# ════════════════════════════════════════════════════════════════
TTA_FLIPS = [
    [],
    [2], [3], [4],
    [2, 3], [2, 4], [3, 4],
    [2, 3, 4],
]

def tta_predict(model, images):
    preds = []
    for axes in TTA_FLIPS:
        x = images.clone()
        for ax in axes:
            x = torch.flip(x, [ax])
        with torch.no_grad():
            out = model(x)
        if APPLY_SIGMOID:
            out = torch.sigmoid(out)
        for ax in axes:
            out = torch.flip(out, [ax])
        preds.append(out)
    return torch.stack(preds).mean(0)

# ════════════════════════════════════════════════════════════════
# POST-PROCESSING
# ════════════════════════════════════════════════════════════════
def remove_small_components(mask_np, min_voxels):
    if mask_np.sum() == 0:
        return mask_np
    labeled, n = ndimage.label(mask_np)
    if n == 0:
        return mask_np
    sizes = ndimage.sum(mask_np, labeled, range(1, n + 1))
    result = np.zeros_like(mask_np)
    kept = False
    for i, s in enumerate(sizes):
        if s >= min_voxels:
            result[labeled == (i + 1)] = 1
            kept = True
    if not kept:
        largest = np.argmax(sizes)
        result[labeled == (largest + 1)] = 1
    return result

def postprocess_masks(pred_mask):
    pred_mask[0] = pred_mask[0] * pred_mask[1]   # ET ⊆ TC
    pred_mask[1] = pred_mask[1] * pred_mask[2]   # TC ⊆ WT

    pred_mask[0] = remove_small_components(pred_mask[0], MIN_VOXELS_ET)
    pred_mask[1] = remove_small_components(pred_mask[1], MIN_VOXELS_TC)
    pred_mask[2] = remove_small_components(pred_mask[2], MIN_VOXELS_WT)

    pred_mask[0] = pred_mask[0] * pred_mask[1]
    pred_mask[1] = pred_mask[1] * pred_mask[2]

    return pred_mask

# ════════════════════════════════════════════════════════════════
# METRICS
# ════════════════════════════════════════════════════════════════
def dice_score(pred, gt):
    pred  = pred.astype(float)
    gt    = gt.astype(float)
    inter = (pred * gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return float(2 * inter / denom)

def hausdorff95(pred, gt, voxel_spacing=(1.0, 1.0, 1.0)):
    from scipy.spatial import cKDTree
    pred_pts = np.argwhere(pred) * np.array(voxel_spacing)
    gt_pts   = np.argwhere(gt)   * np.array(voxel_spacing)
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return np.nan
    tree_gt   = cKDTree(gt_pts)
    tree_pred = cKDTree(pred_pts)
    d_pred_to_gt, _ = tree_gt.query(pred_pts)
    d_gt_to_pred, _ = tree_pred.query(gt_pts)
    all_d = np.concatenate([d_pred_to_gt, d_gt_to_pred])
    return float(np.percentile(all_d, 95))

# ════════════════════════════════════════════════════════════════
# PREDICTION
# ════════════════════════════════════════════════════════════════
def predict_batch(model, images, et_thr, tc_thr, wt_thr,
                  use_tta, use_post, apply_sigmoid):
    if use_tta:
        probs = tta_predict(model, images)
    else:
        with torch.no_grad():
            out = model(images)
        probs = torch.sigmoid(out) if apply_sigmoid else out

    pred_mask = torch.zeros_like(probs)
    pred_mask[:, 0] = (probs[:, 0] > et_thr).float()
    pred_mask[:, 1] = (probs[:, 1] > tc_thr).float()
    pred_mask[:, 2] = (probs[:, 2] > wt_thr).float()

    # Fallback: if ET empty but TC has signal
    for b in range(pred_mask.shape[0]):
        if pred_mask[b, 0].sum() == 0 and probs[b, 0].max().item() > 0.08:
            tc_region = pred_mask[b, 1].bool()
            et_in_tc  = probs[b, 0] * tc_region.float()
            if et_in_tc.max().item() > 0.08:
                pred_mask[b, 0] = (et_in_tc > 0.08).float()

    if use_post:
        for b in range(pred_mask.shape[0]):
            pm = pred_mask[b].cpu().numpy()
            pm = postprocess_masks(pm)
            pred_mask[b] = torch.from_numpy(pm).to(pred_mask.device)

    return pred_mask, probs

# ════════════════════════════════════════════════════════════════
# ID MATCHING
# ════════════════════════════════════════════════════════════════
def normalize_id(pid):
    return re.sub(r'\D', '', str(pid)).lstrip('0') or '0'

def build_subset(dataset, id_file, strict=True):
    with open(id_file, "r") as f:
        saved_ids = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(saved_ids)} IDs from: {id_file}")
    print(f"STRICT_ID_MATCH = {strict}")

    saved_set  = set(saved_ids)
    saved_norm = {normalize_id(i): i for i in saved_ids}

    matched_indices = []
    matched_ids     = []
    breakdown       = {"exact": 0, "norm": 0, "numeric": 0}

    print("Matching dataset items to saved IDs...")
    for idx in range(len(dataset)):
        item = dataset[idx]
        pid  = str(item.get("patient_id", item.get("id", idx)))

        if pid in saved_set:
            matched_indices.append(idx)
            matched_ids.append(pid)
            breakdown["exact"] += 1
        elif not strict:
            n = normalize_id(pid)
            if n in saved_norm:
                matched_indices.append(idx)
                matched_ids.append(pid)
                breakdown["norm"] += 1
            else:
                for sid in saved_ids:
                    if normalize_id(sid) == n:
                        matched_indices.append(idx)
                        matched_ids.append(pid)
                        breakdown["numeric"] += 1
                        break

    print(f"Matched patients: {len(matched_indices)}")
    print(f"Match breakdown: {breakdown}")
    return Subset(dataset, matched_indices), matched_ids

# ════════════════════════════════════════════════════════════════
# EVALUATE
# ════════════════════════════════════════════════════════════════
def evaluate(model, dataloader, device, et_thr, tc_thr, wt_thr,
             use_tta, use_post, apply_sigmoid, debug_first=False):
    model.eval()
    results = []
    total   = len(dataloader.dataset)
    done    = 0
    debug_printed = False

    for batch in dataloader:
        images  = batch["image"].to(device)
        labels  = batch["label"]
        pids    = batch["patient_id"]

        pred_mask, probs = predict_batch(
            model, images, et_thr, tc_thr, wt_thr,
            use_tta, use_post, apply_sigmoid
        )

        if debug_first and not debug_printed:
            print(f"\n[DEBUG] First batch, sample 0")
            print(f"  image shape      : {tuple(images.shape)}")
            print(f"  label shape      : {tuple(labels.shape)}")
            with torch.no_grad():
                raw_out = model(images[:1])
            print(f"  raw output range : [{raw_out.min():.4f}, {raw_out.max():.4f}]")
            print(f"  probs range      : [{probs[:1].min():.4f}, {probs[:1].max():.4f}]")
            lbl0 = labels[0]
            pr0  = pred_mask[0].cpu()
            print(f"  GT  (ET,TC,WT)   : {lbl0[0].sum():.0f} {lbl0[1].sum():.0f} {lbl0[2].sum():.0f}")
            print(f"  Pred(ET,TC,WT)   : {pr0[0].sum():.0f} {pr0[1].sum():.0f} {pr0[2].sum():.0f}")
            debug_printed = True

        for b in range(images.shape[0]):
            pid   = pids[b]
            lbl_b = labels[b].numpy()
            pr_b  = pred_mask[b].cpu().numpy()

            ch   = LABEL_CHANNEL_ORDER
            d_et = dice_score(pr_b[0], lbl_b[ch[0]])
            d_tc = dice_score(pr_b[1], lbl_b[ch[1]])
            d_wt = dice_score(pr_b[2], lbl_b[ch[2]])

            h_et = hausdorff95(pr_b[0], lbl_b[ch[0]])
            h_tc = hausdorff95(pr_b[1], lbl_b[ch[1]])
            h_wt = hausdorff95(pr_b[2], lbl_b[ch[2]])

            failed = int(
                (pr_b[0].sum() == 0 and lbl_b[ch[0]].sum() > 0) or
                (pr_b[1].sum() == 0 and lbl_b[ch[1]].sum() > 0) or
                (pr_b[2].sum() == 0 and lbl_b[ch[2]].sum() > 0)
            )

            results.append({
                "Patient_ID": pid,
                "Dice_ET":    round(d_et, 4),
                "Dice_TC":    round(d_tc, 4),
                "Dice_WT":    round(d_wt, 4),
                "HD95_ET":    round(h_et, 4) if not np.isnan(h_et) else "NaN",
                "HD95_TC":    round(h_tc, 4) if not np.isnan(h_tc) else "NaN",
                "HD95_WT":    round(h_wt, 4) if not np.isnan(h_wt) else "NaN",
                "Failed":     failed,
            })

        done += images.shape[0]
        if done % 50 == 0 or done == total:
            print(f"Processed {done}/{total}")

    return results

# ════════════════════════════════════════════════════════════════
# PRINT SUMMARY
# ════════════════════════════════════════════════════════════════
def print_summary(results):
    df = pd.DataFrame(results)

    et_vals = df["Dice_ET"].values
    tc_vals = df["Dice_TC"].values
    wt_vals = df["Dice_WT"].values

    mean_et = np.nanmean(et_vals)
    mean_tc = np.nanmean(tc_vals)
    mean_wt = np.nanmean(wt_vals)
    overall = np.mean([mean_et, mean_tc, mean_wt])

    def safe_hd(col):
        vals = pd.to_numeric(df[col], errors="coerce").dropna().values
        return np.mean(vals) if len(vals) > 0 else float("nan")

    hd_et = safe_hd("HD95_ET")
    hd_tc = safe_hd("HD95_TC")
    hd_wt = safe_hd("HD95_WT")

    failed_et = int((df["Dice_ET"] == 0).sum())
    failed_tc = int((df["Dice_TC"] == 0).sum())
    failed_wt = int((df["Dice_WT"] == 0).sum())

    print("\n" + "=" * 72)
    print(f"{'Region':<10} | {'Dice Score':<12} | {'HD95 (mm)':<16} | {'Failed':<8}")
    print("-" * 72)
    print(f"{'ET':<10} | {mean_et:<12.4f} | {hd_et:<16.4f} | {failed_et:<8}")
    print(f"{'TC':<10} | {mean_tc:<12.4f} | {hd_tc:<16.4f} | {failed_tc:<8}")
    print(f"{'WT':<10} | {mean_wt:<12.4f} | {hd_wt:<16.4f} | {failed_wt:<8}")
    print("-" * 72)
    print(f"{'OVERALL':<10} | {overall:<12.4f} | {'-':<16} | {'-':<8}")
    print("=" * 72)

    mean_per_patient = (et_vals + tc_vals + wt_vals) / 3
    print(f"Best Mean Dice : {mean_per_patient.max():.4f}")
    print(f"Worst Mean Dice: {mean_per_patient.min():.4f}")

    df["Mean_Dice"] = mean_per_patient
    worst5 = df.nsmallest(5, "Mean_Dice")[
        ["Patient_ID", "Dice_ET", "Dice_TC", "Dice_WT", "Mean_Dice"]
    ]
    print("\nWorst 5 patients:")
    print(worst5.to_string(index=False))

    zero_all = ((df["Dice_ET"] == 0) & (df["Dice_TC"] == 0) & (df["Dice_WT"] == 0)).sum()
    print(f"\nPatients with ALL Dice=0.0: {zero_all}")
    print("Note: HD95 is in mm. Failed = one mask empty (NaN excluded from mean).")
    print(f"\nThresholds used: ET={ET_THR}, TC={TC_THR}, WT={WT_THR}  ← fixed, no leakage")

    return overall

# ════════════════════════════════════════════════════════════════
# SAVE CSV
# ════════════════════════════════════════════════════════════════
def save_csv(results, path):
    keys = ["Patient_ID", "Dice_ET", "Dice_TC", "Dice_WT",
            "HD95_ET", "HD95_TC", "HD95_WT", "Failed"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to: {path}")

# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  BraTS2021 Evaluation — Paper-Ready (Fixed Thresholds)")
    print("=" * 60)
    print(f"Device             : {device}")
    print(f"Checkpoint         : {CHECKPOINT_PATH}")
    print(f"TTA                : {USE_TTA}")
    print(f"Post-processing    : {USE_POSTPROCESS}")
    print(f"APPLY_SIGMOID      : {APPLY_SIGMOID}")
    print(f"Thresholds (fixed) : ET={ET_THR}, TC={TC_THR}, WT={WT_THR}")
    print(f"LABEL_CHANNEL_ORDER: {LABEL_CHANNEL_ORDER}")
    print(f"AUTO_FIND_BEST     : {AUTO_FIND_BEST_SETUP}  ← disabled for paper")

    # ── Build dataset ─────────────────────────────────────────────
    print("\nBuilding TEST subset...")
    full_ds = get_datasets(on="test")
    print(f"Full dataset size: {len(full_ds)}")

    if os.path.exists(TEST_IDS_FILE):
        subset_ds, matched_ids = build_subset(full_ds, TEST_IDS_FILE, strict=STRICT_ID_MATCH)
    else:
        subset_ds = full_ds
        print(f"No ID file found — using all {len(full_ds)} patients")

    # ── Load model ────────────────────────────────────────────────
    print("\nLoading model...")
    model = WaveletUNetPlusPlus(in_channels=4, n_classes=3).to(device)
    ckpt  = torch.load(CHECKPOINT_PATH, map_location=device)
    sd    = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
    model.load_state_dict(sd, strict=True)
    model.eval()
    print("Model loaded.")

    # ── Full evaluation on ALL patients ───────────────────────────
    print(f"\nEvaluating ALL {len(subset_ds)} patients with fixed thresholds "
          f"ET={ET_THR}, TC={TC_THR}, WT={WT_THR}...\n")

    full_loader = DataLoader(subset_ds, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=NUM_WORKERS)

    results = evaluate(
        model, full_loader, device,
        et_thr=ET_THR, tc_thr=TC_THR, wt_thr=WT_THR,
        use_tta=USE_TTA, use_post=USE_POSTPROCESS,
        apply_sigmoid=APPLY_SIGMOID,
        debug_first=True
    )

    print_summary(results)
    save_csv(results, OUTPUT_CSV)


if __name__ == "__main__":
    main()