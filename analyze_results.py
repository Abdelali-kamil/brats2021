import os
import re
import random
import warnings
import numpy as np
import pandas as pd
import torch
import scipy.ndimage as ndimage

from torch.utils.data import DataLoader, Subset
from medpy import metric

from brats import get_datasets
from model import WaveletUNetPlusPlus

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# CONFIG
# ============================================================
CHECKPOINTS = [
    "checkpoints/segmentor_epoch_660.pth",
    "checkpoints/segmentor_epoch_630.pth",
    "checkpoints/segmentor_epoch_640.pth",
    "checkpoints/segmentor_epoch_600.pth",
    "checkpoints/segmentor_epoch_650.pth",
]

TEST_IDS_FILE = "test_ids.txt"

BATCH_SIZE = 2
NUM_WORKERS = 4
SEED = 42

APPLY_SIGMOID = True
USE_TTA = True
USE_POSTPROCESS = True

THRESH_ET = 0.46
THRESH_TC = 0.50
THRESH_WT = 0.50

LOW_ET = 0.15
LOW_TC = 0.25
LOW_WT = 0.25

MIN_VOXELS_ET = 1
MIN_VOXELS_TC = 5
MIN_VOXELS_WT = 20

LABEL_CHANNEL_ORDER = [0, 1, 2]  # ET=0, TC=1, WT=2

# Output files
RUNS_SUMMARY_CSV = "multi_run_summary.csv"
FINAL_MEAN_STD_TXT = "final_mean_std_report.txt"


# ============================================================
# REPRODUCIBILITY
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# ID HELPERS
# ============================================================
def norm_id_str(s):
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"\.nii(?:\.gz)?$", "", s)
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def load_patient_ids(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing split file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        ids = [line.strip() for line in f if line.strip()]
    if not ids:
        raise ValueError(f"No patient IDs found in {path}")
    return ids


def normalize_patient_id(pid):
    if isinstance(pid, (list, tuple)):
        pid = pid[0]
    if torch.is_tensor(pid):
        try:
            pid = pid.item()
        except Exception:
            pass
    return str(pid)


# ============================================================
# DATASET
# ============================================================
def build_subset(ids_file):
    # IMPORTANT: on="test" => no random crop/augmentation
    full_dataset = get_datasets(on="test")
    ids_raw = load_patient_ids(ids_file)

    ids_raw_set = set(ids_raw)
    ids_norm_set = set(norm_id_str(x) for x in ids_raw)

    matched_indices = []
    for idx in range(len(full_dataset)):
        item = full_dataset[idx]
        pid_str = str(item.get("patient_id", item.get("id", str(idx))))
        pid_nrm = norm_id_str(pid_str)

        if pid_str in ids_raw_set or pid_nrm in ids_norm_set:
            matched_indices.append(idx)

    if not matched_indices:
        raise RuntimeError(f"No dataset items matched IDs in {ids_file}")

    print(f"Full dataset size: {len(full_dataset)}")
    print(f"Loaded IDs       : {len(ids_raw)}")
    print(f"Matched patients : {len(matched_indices)}")

    return Subset(full_dataset, matched_indices)


# ============================================================
# METRICS
# ============================================================
def dice_coefficient_safe(y_pred, y_true, smooth=1e-5):
    y_pred = y_pred.contiguous().view(-1).float()
    y_true = y_true.contiguous().view(-1).float()
    inter = (y_pred * y_true).sum()
    denom = y_pred.sum() + y_true.sum()
    if denom.item() == 0:
        return torch.tensor(1.0, device=y_pred.device)
    return (2.0 * inter + smooth) / (denom + smooth)


def get_hd95(pred, gt):
    pred = pred.astype(np.uint8)
    gt = gt.astype(np.uint8)
    pred_nz = np.count_nonzero(pred) > 0
    gt_nz = np.count_nonzero(gt) > 0

    if not pred_nz and not gt_nz:
        return 0.0
    if pred_nz and gt_nz:
        try:
            return float(metric.binary.hd95(pred, gt))
        except Exception:
            return np.nan
    return np.nan


# ============================================================
# POST-PROCESS
# ============================================================
def remove_small_noise(tensor_mask, min_voxels):
    np_mask = tensor_mask.detach().cpu().numpy().astype(np.uint8)
    labels, n = ndimage.label(np_mask)
    if n == 0:
        return tensor_mask

    sizes = np.bincount(labels.ravel())
    valid = np.where(sizes >= min_voxels)[0]
    valid = valid[valid != 0]

    if len(valid) == 0:
        sizes[0] = 0
        valid = np.array([sizes.argmax()])

    clean = np.isin(labels, valid).astype(np.float32)
    return torch.from_numpy(clean).to(tensor_mask.device)


def fill_holes_3d(tensor_mask):
    np_mask = tensor_mask.detach().cpu().numpy().astype(np.uint8)
    filled = ndimage.binary_fill_holes(np_mask)
    return torch.from_numpy(filled.astype(np.float32)).to(tensor_mask.device)


def postprocess_masks(pred_mask):
    pred_mask = pred_mask.contiguous()
    for b in range(pred_mask.shape[0]):
        # ET
        pred_mask[b, 0] = remove_small_noise(pred_mask[b, 0], MIN_VOXELS_ET)

        # TC
        pred_mask[b, 1] = remove_small_noise(pred_mask[b, 1], MIN_VOXELS_TC)
        pred_mask[b, 1] = fill_holes_3d(pred_mask[b, 1])

        # WT
        pred_mask[b, 2] = remove_small_noise(pred_mask[b, 2], MIN_VOXELS_WT)
        pred_mask[b, 2] = fill_holes_3d(pred_mask[b, 2])

        # BraTS hierarchy: ET ⊆ TC ⊆ WT
        pred_mask[b, 1] = torch.maximum(pred_mask[b, 1], pred_mask[b, 0])
        pred_mask[b, 2] = torch.maximum(pred_mask[b, 2], pred_mask[b, 1])

    return pred_mask


# ============================================================
# INFERENCE
# ============================================================
def predict_mask(model, images):
    raw_out = model(images)
    probs = torch.sigmoid(raw_out) if APPLY_SIGMOID else raw_out

    if USE_TTA:
        flip_configs = [
            [-1], [-2], [-3],
            [-1, -2], [-1, -3], [-2, -3],
            [-1, -2, -3],
        ]
        for dims in flip_configs:
            raw_f = model(torch.flip(images, dims=dims))
            prob_f = torch.sigmoid(raw_f) if APPLY_SIGMOID else raw_f
            probs += torch.flip(prob_f, dims=dims)
        probs = probs / (1 + len(flip_configs))

    pred_mask = torch.zeros_like(probs)
    pred_mask[:, 0] = (probs[:, 0] > THRESH_ET).float()
    pred_mask[:, 1] = (probs[:, 1] > THRESH_TC).float()
    pred_mask[:, 2] = (probs[:, 2] > THRESH_WT).float()

    # fallback for empty predictions
    for b in range(pred_mask.shape[0]):
        if pred_mask[b, 0].sum() == 0 and probs[b, 0].max() > LOW_ET:
            pred_mask[b, 0] = (probs[b, 0] > LOW_ET).float()
        if pred_mask[b, 1].sum() == 0 and probs[b, 1].max() > LOW_TC:
            pred_mask[b, 1] = (probs[b, 1] > LOW_TC).float()
        if pred_mask[b, 2].sum() == 0 and probs[b, 2].max() > LOW_WT:
            pred_mask[b, 2] = (probs[b, 2] > LOW_WT).float()

    if USE_POSTPROCESS:
        pred_mask = postprocess_masks(pred_mask)

    return pred_mask


# ============================================================
# MODEL
# ============================================================
def load_model(device, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = WaveletUNetPlusPlus(in_channels=4, n_classes=3).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


# ============================================================
# EVALUATE ONE RUN
# ============================================================
def evaluate_one_checkpoint(model, loader, total_count):
    rows = []
    device = next(model.parameters()).device

    with torch.no_grad():
        for i, batch in enumerate(loader):
            images = batch["image"].float().to(device)
            labels = batch["label"].float().to(device)
            labels = labels[:, LABEL_CHANNEL_ORDER, ...]

            if labels.shape[1] != 3:
                raise ValueError(f"Expected 3 label channels, got {labels.shape}")

            pids_raw = batch.get("patient_id", None)
            B = images.shape[0]

            if pids_raw is None:
                patient_ids = [f"idx_{i}_{b}" for b in range(B)]
            elif isinstance(pids_raw, torch.Tensor):
                patient_ids = [str(p.item()) for p in pids_raw]
            else:
                patient_ids = [str(p) for p in pids_raw]

            pred_mask = predict_mask(model, images)

            for b in range(B):
                pid = normalize_patient_id(patient_ids[b])

                d_et = dice_coefficient_safe(pred_mask[b, 0], labels[b, 0]).item()
                d_tc = dice_coefficient_safe(pred_mask[b, 1], labels[b, 1]).item()
                d_wt = dice_coefficient_safe(pred_mask[b, 2], labels[b, 2]).item()

                pm_np = pred_mask[b].cpu().numpy()
                lb_np = labels[b].cpu().numpy()

                rows.append({
                    "Patient_ID": pid,
                    "Dice_ET": d_et,
                    "Dice_TC": d_tc,
                    "Dice_WT": d_wt,
                    "Dice_Mean": (d_et + d_tc + d_wt) / 3.0,
                    "HD95_ET": get_hd95(pm_np[0], lb_np[0]),
                    "HD95_TC": get_hd95(pm_np[1], lb_np[1]),
                    "HD95_WT": get_hd95(pm_np[2], lb_np[2]),
                    "Pred_Empty": int(pred_mask[b].sum().item() == 0),
                })

            done = len(rows)
            if done % 50 == 0 or done == total_count:
                print(f"Processed {done}/{total_count}")

    return pd.DataFrame(rows)


# ============================================================
# RUN SUMMARY
# ============================================================
def summarize_run(df, checkpoint_name):
    out = {
        "checkpoint": checkpoint_name,
        "dice_et": df["Dice_ET"].mean(),
        "dice_tc": df["Dice_TC"].mean(),
        "dice_wt": df["Dice_WT"].mean(),
        "dice_mean": df["Dice_Mean"].mean(),
        "hd95_et": pd.to_numeric(df["HD95_ET"], errors="coerce").dropna().mean(),
        "hd95_tc": pd.to_numeric(df["HD95_TC"], errors="coerce").dropna().mean(),
        "hd95_wt": pd.to_numeric(df["HD95_WT"], errors="coerce").dropna().mean(),
        "pred_empty_count": int(df["Pred_Empty"].sum()),
        "best_patient": df["Dice_Mean"].max(),
        "worst_patient": df["Dice_Mean"].min(),
    }
    return out

### `analyze_results.py`
"""
BraTS2021 Multi-Run Evaluation
- Evaluate multiple checkpoints (independent trainings)
- Report per-run metrics
- Report Mean ± STD across runs
"""

import os
import re
import random
import warnings
import numpy as np
import pandas as pd
import torch
import scipy.ndimage as ndimage

from torch.utils.data import DataLoader, Subset
from medpy import metric

from brats import get_datasets
from model import WaveletUNetPlusPlus

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# CONFIG
# ============================================================
CHECKPOINTS = [
   "checkpoints/segmentor_epoch_660.pth",
    "checkpoints/segmentor_epoch_630.pth",
    "checkpoints/segmentor_epoch_640.pth",
    "checkpoints/segmentor_epoch_600.pth",
    "checkpoints/segmentor_epoch_650.pth",
]

TEST_IDS_FILE = "test_ids.txt"

# Output files
RUN_SUMMARY_CSV = "run_summary.csv"         # one row per checkpoint
AGG_SUMMARY_CSV = "aggregate_summary.csv"   # mean±std final table

BATCH_SIZE = 2
NUM_WORKERS = 4
SEED = 42  # for dataloader/eval reproducibility

APPLY_SIGMOID = True
USE_TTA = True
USE_POSTPROCESS = True

# Thresholds
THRESH_ET = 0.46
THRESH_TC = 0.50
THRESH_WT = 0.50

# Fallback thresholds
LOW_ET = 0.15
LOW_TC = 0.25
LOW_WT = 0.25

# Min connected component sizes
MIN_VOXELS_ET = 1
MIN_VOXELS_TC = 5
MIN_VOXELS_WT = 20

# Label order in your dataset
LABEL_CHANNEL_ORDER = [0, 1, 2]  # ET=0, TC=1, WT=2
DEBUG_FIRST_BATCH = True


# ============================================================
# SEED
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# ID UTILS
# ============================================================
def norm_id_str(s):
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"\.nii(?:\.gz)?$", "", s)
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def load_patient_ids(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing split file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        ids = [line.strip() for line in f if line.strip()]
    if not ids:
        raise ValueError(f"No patient IDs found in {path}")
    return ids


def normalize_patient_id(pid):
    if isinstance(pid, (list, tuple)):
        pid = pid[0]
    if torch.is_tensor(pid):
        try:
            pid = pid.item()
        except Exception:
            pass
    return str(pid)


# ============================================================
# DATASET
# ============================================================
def build_subset(ids_file):
    full_dataset = get_datasets(on="test")
    ids_raw = load_patient_ids(ids_file)

    print(f"Full dataset size: {len(full_dataset)}")
    print(f"Loaded {len(ids_raw)} IDs from: {ids_file}")

    ids_raw_set = set(ids_raw)
    ids_norm_set = set(norm_id_str(t) for t in ids_raw)

    matched_indices = []
    matched_pids = []
    match_mode = {"exact": 0, "norm": 0}

    for idx in range(len(full_dataset)):
        item = full_dataset[idx]
        pid_str = str(item.get("patient_id", item.get("id", str(idx))))
        pid_nrm = norm_id_str(pid_str)

        if pid_str in ids_raw_set:
            matched_indices.append(idx)
            matched_pids.append(pid_str)
            match_mode["exact"] += 1
        elif pid_nrm in ids_norm_set:
            matched_indices.append(idx)
            matched_pids.append(pid_str)
            match_mode["norm"] += 1

    if not matched_indices:
        raise RuntimeError(f"No dataset items matched IDs in {ids_file}")

    matched_norms = set(norm_id_str(x) for x in matched_pids)
    missing = [t for t in ids_raw if norm_id_str(t) not in matched_norms]
    if missing:
        print(f"⚠️ Missing IDs: {len(missing)}")
        for m in missing[:10]:
            print(f"  {m}")

    print(f"Matched patients: {len(matched_indices)}")
    print(f"Match breakdown: {match_mode}")

    return Subset(full_dataset, matched_indices)


# ============================================================
# METRICS
# ============================================================
def dice_coefficient_safe(y_pred, y_true, smooth=1e-5):
    y_pred = y_pred.contiguous().view(-1).float()
    y_true = y_true.contiguous().view(-1).float()
    intersection = (y_pred * y_true).sum()
    denom = y_pred.sum() + y_true.sum()
    if denom.item() == 0:
        return torch.tensor(1.0, device=y_pred.device)
    return (2.0 * intersection + smooth) / (denom + smooth)


def get_hd95(pred, gt):
    pred = pred.astype(np.uint8)
    gt = gt.astype(np.uint8)

    pred_nz = np.count_nonzero(pred) > 0
    gt_nz = np.count_nonzero(gt) > 0

    if not pred_nz and not gt_nz:
        return 0.0
    if pred_nz and gt_nz:
        try:
            return float(metric.binary.hd95(pred, gt))
        except Exception:
            return np.nan
    return np.nan


# ============================================================
# POST-PROCESSING
# ============================================================
def remove_small_noise(tensor_mask, min_voxels):
    np_mask = tensor_mask.detach().cpu().numpy().astype(np.uint8)
    labels, n = ndimage.label(np_mask)
    if n == 0:
        return tensor_mask

    sizes = np.bincount(labels.ravel())
    valid = np.where(sizes >= min_voxels)[0]
    valid = valid[valid != 0]

    if len(valid) == 0:
        sizes[0] = 0
        valid = np.array([sizes.argmax()])

    clean = np.isin(labels, valid).astype(np.float32)
    return torch.from_numpy(clean).to(tensor_mask.device)


def fill_holes_3d(tensor_mask):
    np_mask = tensor_mask.detach().cpu().numpy().astype(np.uint8)
    filled = ndimage.binary_fill_holes(np_mask)
    return torch.from_numpy(filled.astype(np.float32)).to(tensor_mask.device)


def postprocess_masks(pred_mask):
    pred_mask = pred_mask.contiguous()
    for b in range(pred_mask.shape[0]):
        # ET
        pred_mask[b, 0] = remove_small_noise(pred_mask[b, 0], MIN_VOXELS_ET)

        # TC
        pred_mask[b, 1] = remove_small_noise(pred_mask[b, 1], MIN_VOXELS_TC)
        pred_mask[b, 1] = fill_holes_3d(pred_mask[b, 1])

        # WT
        pred_mask[b, 2] = remove_small_noise(pred_mask[b, 2], MIN_VOXELS_WT)
        pred_mask[b, 2] = fill_holes_3d(pred_mask[b, 2])

        # enforce ET ⊆ TC ⊆ WT
        pred_mask[b, 1] = torch.maximum(pred_mask[b, 1], pred_mask[b, 0])
        pred_mask[b, 2] = torch.maximum(pred_mask[b, 2], pred_mask[b, 1])

    return pred_mask


# ============================================================
# INFERENCE
# ============================================================
def predict_mask(model, images):
    raw_out = model(images)
    probs = torch.sigmoid(raw_out) if APPLY_SIGMOID else raw_out

    if USE_TTA:
        flip_configs = [
            [-1], [-2], [-3],
            [-1, -2], [-1, -3], [-2, -3],
            [-1, -2, -3]
        ]
        for dims in flip_configs:
            raw_f = model(torch.flip(images, dims=dims))
            prob_f = torch.sigmoid(raw_f) if APPLY_SIGMOID else raw_f
            probs += torch.flip(prob_f, dims=dims)
        probs = probs / (1 + len(flip_configs))

    pred_mask = torch.zeros_like(probs)
    pred_mask[:, 0] = (probs[:, 0] > THRESH_ET).float()
    pred_mask[:, 1] = (probs[:, 1] > THRESH_TC).float()
    pred_mask[:, 2] = (probs[:, 2] > THRESH_WT).float()

    for b in range(pred_mask.shape[0]):
        if pred_mask[b, 0].sum() == 0 and probs[b, 0].max() > LOW_ET:
            pred_mask[b, 0] = (probs[b, 0] > LOW_ET).float()
        if pred_mask[b, 1].sum() == 0 and probs[b, 1].max() > LOW_TC:
            pred_mask[b, 1] = (probs[b, 1] > LOW_TC).float()
        if pred_mask[b, 2].sum() == 0 and probs[b, 2].max() > LOW_WT:
            pred_mask[b, 2] = (probs[b, 2] > LOW_WT).float()

    if USE_POSTPROCESS:
        pred_mask = postprocess_masks(pred_mask)

    return pred_mask, raw_out, probs


# ============================================================
# MODEL
# ============================================================
def load_model(device, checkpoint_path):
    model = WaveletUNetPlusPlus(in_channels=4, n_classes=3).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


# ============================================================
# EVALUATE ONE RUN
# ============================================================
def evaluate(model, loader, total_count):
    results_list = []
    device = next(model.parameters()).device
    first_batch_done = False

    with torch.no_grad():
        for i, batch in enumerate(loader):
            images = batch["image"].float().to(device)
            labels = batch["label"].float().to(device)
            labels = labels[:, LABEL_CHANNEL_ORDER, ...]

            if labels.shape[1] != 3:
                raise ValueError(f"Expected 3 label channels, got {labels.shape}")

            B = images.shape[0]
            pids_raw = batch.get("patient_id", None)
            if pids_raw is None:
                patient_ids = [f"idx_{i}_{b}" for b in range(B)]
            elif isinstance(pids_raw, torch.Tensor):
                patient_ids = [str(p.item()) for p in pids_raw]
            else:
                patient_ids = [str(p) for p in pids_raw]

            pred_mask, raw_out, probs = predict_mask(model, images)

            if not first_batch_done and DEBUG_FIRST_BATCH:
                first_batch_done = True
                b0 = 0
                et_gt = labels[b0, 0].sum().item()
                tc_gt = labels[b0, 1].sum().item()
                wt_gt = labels[b0, 2].sum().item()
                print("\n[DEBUG] First batch, sample 0")
                print(f"  image shape      : {tuple(images.shape)}")
                print(f"  label shape      : {tuple(labels.shape)}")
                print(f"  raw output range : [{raw_out.min():.4f}, {raw_out.max():.4f}]")
                print(f"  probs range      : [{probs.min():.4f}, {probs.max():.4f}]")
                print(f"  GT  (ET,TC,WT)   : {et_gt:.0f} {tc_gt:.0f} {wt_gt:.0f}")
                print(
                    f"  Pred(ET,TC,WT)   : "
                    f"{pred_mask[b0,0].sum():.0f} "
                    f"{pred_mask[b0,1].sum():.0f} "
                    f"{pred_mask[b0,2].sum():.0f}"
                )

            for b in range(B):
                pid = normalize_patient_id(patient_ids[b])

                d_et = dice_coefficient_safe(pred_mask[b, 0], labels[b, 0]).item()
                d_tc = dice_coefficient_safe(pred_mask[b, 1], labels[b, 1]).item()
                d_wt = dice_coefficient_safe(pred_mask[b, 2], labels[b, 2]).item()

                pm_np = pred_mask[b].cpu().numpy()
                lb_np = labels[b].cpu().numpy()

                results_list.append({
                    "Patient_ID": pid,
                    "Dice_ET": d_et,
                    "Dice_TC": d_tc,
                    "Dice_WT": d_wt,
                    "Dice_Mean": (d_et + d_tc + d_wt) / 3.0,
                    "HD95_ET": get_hd95(pm_np[0], lb_np[0]),
                    "HD95_TC": get_hd95(pm_np[1], lb_np[1]),
                    "HD95_WT": get_hd95(pm_np[2], lb_np[2]),
                    "Pred_Empty": int(pred_mask[b].sum().item() == 0),
                })

            done = len(results_list)
            if done % 50 == 0 or done == total_count:
                print(f"Processed {done}/{total_count}")

    return pd.DataFrame(results_list)


# ============================================================
# SUMMARY HELPERS
# ============================================================
def summarize_run(df, ckpt_name):
    row = {
        "checkpoint": ckpt_name,
        "Dice_ET": df["Dice_ET"].mean(),
        "Dice_TC": df["Dice_TC"].mean(),
        "Dice_WT": df["Dice_WT"].mean(),
        "Dice_Mean": df["Dice_Mean"].mean(),
        "HD95_ET": pd.to_numeric(df["HD95_ET"], errors="coerce").dropna().mean(),
        "HD95_TC": pd.to_numeric(df["HD95_TC"], errors="coerce").dropna().mean(),
        "HD95_WT": pd.to_numeric(df["HD95_WT"], errors="coerce").dropna().mean(),
        "Empty_Pred_Count": int(df["Pred_Empty"].sum()),
        "Best_Patient_DiceMean": df["Dice_Mean"].max(),
        "Worst_Patient_DiceMean": df["Dice_Mean"].min(),
    }
    return row


def print_run_table(run_df):
    print("\n" + "=" * 90)
    print("Per-run summary")
    print("=" * 90)
    cols = ["checkpoint", "Dice_ET", "Dice_TC", "Dice_WT", "Dice_Mean", "HD95_ET", "HD95_TC", "HD95_WT"]
    print(run_df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("=" * 90)


def print_aggregate(run_df):
    print("\nFinal Mean ± STD across runs:")
    for m in ["Dice_ET", "Dice_TC", "Dice_WT", "Dice_Mean", "HD95_ET", "HD95_TC", "HD95_WT"]:
        mean_val = run_df[m].mean()
        std_val = run_df[m].std(ddof=1) if len(run_df) > 1 else 0.0
        print(f"{m:<10}: {mean_val:.4f} ± {std_val:.4f}")


# ============================================================
# MAIN
# ============================================================
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("BraTS2021 Multi-Run Evaluation")
    print("=" * 60)
    print(f"Device             : {device}")
    print(f"TTA                : {USE_TTA} (7 flip combos)")
    print(f"Post-processing    : {USE_POSTPROCESS}")
    print(f"APPLY_SIGMOID      : {APPLY_SIGMOID}")
    print(f"Thresholds         : ET={THRESH_ET}, TC={THRESH_TC}, WT={THRESH_WT}")
    print(f"Fallback Thresholds: ET={LOW_ET}, TC={LOW_TC}, WT={LOW_WT}")
    print(f"Min Voxels         : ET={MIN_VOXELS_ET}, TC={MIN_VOXELS_TC}, WT={MIN_VOXELS_WT}")
    print(f"Label order        : {LABEL_CHANNEL_ORDER}")

    # checkpoint check
    missing_ckpts = [p for p in CHECKPOINTS if not os.path.exists(p)]
    if missing_ckpts:
        print("\n❌ Missing checkpoints:")
        for p in missing_ckpts:
            print(f" - {p}")
        raise FileNotFoundError("Some checkpoints are missing. Train/save them first.")

    print("\nBuilding TEST subset...")
    test_dataset = build_subset(TEST_IDS_FILE)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    all_run_rows = []

    for run_idx, ckpt_path in enumerate(CHECKPOINTS, start=1):
        print("\n" + "#" * 90)
        print(f"Run {run_idx}/{len(CHECKPOINTS)} | Checkpoint: {ckpt_path}")
        print("#" * 90)

        model = load_model(device, ckpt_path)
        df = evaluate(model, test_loader, len(test_dataset))

        # Save per-patient results for this run
        safe_name = os.path.basename(ckpt_path).replace(".pth", "")
        per_patient_csv = f"final_results_{safe_name}.csv"
        df.to_csv(per_patient_csv, index=False)
        print(f"Saved per-patient results: {per_patient_csv}")

        row = summarize_run(df, ckpt_name=ckpt_path)
        all_run_rows.append(row)

        # quick console stats
        print(
            f"Run Dice_Mean={row['Dice_Mean']:.4f}, "
            f"Dice_ET={row['Dice_ET']:.4f}, Dice_TC={row['Dice_TC']:.4f}, Dice_WT={row['Dice_WT']:.4f}"
        )

    run_df = pd.DataFrame(all_run_rows)
    run_df.to_csv(RUN_SUMMARY_CSV, index=False)
    print(f"\nSaved run summary: {RUN_SUMMARY_CSV}")

    print_run_table(run_df)
    print_aggregate(run_df)

    # Save aggregate table
    agg_rows = []
    for m in ["Dice_ET", "Dice_TC", "Dice_WT", "Dice_Mean", "HD95_ET", "HD95_TC", "HD95_WT"]:
        agg_rows.append({
            "Metric": m,
            "Mean": run_df[m].mean(),
            "Std": run_df[m].std(ddof=1) if len(run_df) > 1 else 0.0
        })
    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(AGG_SUMMARY_CSV, index=False)
    print(f"Saved aggregate summary: {AGG_SUMMARY_CSV}")

    # Worst 5 from the LAST run (optional print)
    print("\nWorst 5 patients from the last run:")
    last_ckpt = CHECKPOINTS[-1]
    last_csv = f"final_results_{os.path.basename(last_ckpt).replace('.pth', '')}.csv"
    last_df = pd.read_csv(last_csv)
    worst5 = last_df.nsmallest(5, "Dice_Mean")[["Patient_ID", "Dice_ET", "Dice_TC", "Dice_WT", "Dice_Mean"]]
    print(worst5.to_string(index=False))


if __name__ == "__main__":
    main()