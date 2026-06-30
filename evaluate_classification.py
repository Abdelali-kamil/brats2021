#!/usr/bin/env python3
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import re
import sys
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split

from brats import get_datasets
from model import WaveletUNetPlusPlus
from batch_utlis import custom_collate, determinist_collate

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ============================================================
# CONFIG
# ============================================================
EPOCHS              = 120
VAL_EVERY           = 1
PATIENCE            = 8
BATCH_SIZE          = 2          # real batch size
ACCUM_STEPS         = 4          # effective batch size = 8
LEARNING_RATE_CLASS = 3e-4
WEIGHT_DECAY        = 1e-4

MGMT_LABELS_CSV     = "train_labels.csv"
SEG_CKPT            = "checkpoints/segmentor_epoch_650.pth"
OUT_DIR             = "checkpoints"
OUTPUT_CSV          = "classification_results.csv"
os.makedirs(OUT_DIR, exist_ok=True)

FORCE_SEG_CPU = False
MIN_FREE_GPU_GB = 8.0

has_cuda = torch.cuda.is_available()

# segmentor device
if FORCE_SEG_CPU:
    seg_device = torch.device("cpu")
    print("⚠️ FORCE_SEG_CPU=True -> Segmentor on CPU")
else:
    if has_cuda:
        free_b, total_b = torch.cuda.mem_get_info()
        free_gb = free_b / (1024 ** 3)
        total_gb = total_b / (1024 ** 3)
        seg_device = torch.device("cuda") if free_gb >= MIN_FREE_GPU_GB else torch.device("cpu")
        print(f"GPU memory: free={free_gb:.2f} GB / total={total_gb:.2f} GB")
        if seg_device.type == "cpu":
            print(f"⚠️ Free VRAM < {MIN_FREE_GPU_GB:.1f} GB, using CPU for segmentor to avoid OOM.")
    else:
        seg_device = torch.device("cpu")

# classifier/backbone device
cls_device = torch.device("cuda" if has_cuda else "cpu")
seg_on_cuda = seg_device.type == "cuda"
cls_on_cuda = cls_device.type == "cuda"
pin_mem = seg_on_cuda or cls_on_cuda

print(f"Segmentor device: {seg_device}  |  Backbone/Classifier: {cls_device}\n")

def autocast_if_seg_cuda():
    return torch.amp.autocast(device_type="cuda") if seg_on_cuda else nullcontext()

# ============================================================
# LABEL LOADER
# ============================================================
def load_mgmt_labels(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Labels CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, dtype=str)
    cols_lower = {c.lower(): c for c in df.columns}

    id_col = None
    for cand in ["brats21id", "patient_id", "case_id", "id", "pid"]:
        if cand in cols_lower:
            id_col = cols_lower[cand]
            break
    if id_col is None:
        raise ValueError(f"No ID column found. Columns: {list(df.columns)}")

    label_col = None
    for cand in ["mgmt_value", "mgmt", "mgmtvalue", "label", "grade"]:
        if cand in cols_lower:
            label_col = cols_lower[cand]
            break
    if label_col is None:
        for c in df.columns:
            vals = set(df[c].dropna().astype(str).str.strip().unique())
            if vals.issubset({"0", "1", "0.0", "1.0"}):
                label_col = c
                break
    if label_col is None:
        raise ValueError(f"No binary label column found. Columns: {list(df.columns)}")

    df[id_col] = df[id_col].astype(str).str.strip()
    df[id_col] = df[id_col].apply(lambda x: x.zfill(5) if x.isdigit() and len(x) <= 5 else x)
    df[label_col] = df[label_col].astype(float).astype(int)

    label_map = dict(zip(df[id_col], df[label_col]))
    n0 = sum(v == 0 for v in label_map.values())
    n1 = sum(v == 1 for v in label_map.values())

    print(f"✅ Loaded {len(label_map)} MGMT labels | MGMT0={n0} MGMT1={n1}")
    print(f"   ID col='{id_col}', Label col='{label_col}'")
    return label_map, n0, n1

grade_labels, n_mgmt0, n_mgmt1 = load_mgmt_labels(MGMT_LABELS_CSV)

# ============================================================
# ID HELPERS
# ============================================================
def extract_pid(text):
    text = str(text)
    m = re.search(r"BraTS2021[_\-](\d{5})", text)
    if m:
        return m.group(1)
    m = re.search(r"(?<!\d)(\d{5})(?!\d)", text)
    if m:
        return m.group(1)
    return None

def normalize_pid(pid):
    if pid is None:
        return None
    pid = str(pid).strip()
    if pid.isdigit() and len(pid) <= 5:
        return pid.zfill(5)
    return pid

def get_patient_id(dataset, idx):
    try:
        sample = dataset[idx]
        if isinstance(sample, dict):
            for key in ["patient_id", "id", "pid", "name", "case_id", "path"]:
                if key in sample:
                    pid = extract_pid(sample[key]) or str(sample[key])
                    pid = normalize_pid(pid)
                    if pid:
                        return pid
    except Exception:
        pass

    for attr in ["patients", "patient_ids", "ids", "names", "data_list"]:
        if hasattr(dataset, attr):
            try:
                raw = getattr(dataset, attr)[idx]
                pid = extract_pid(raw) or str(raw)
                pid = normalize_pid(pid)
                if pid:
                    return pid
            except Exception:
                pass
    return None

def build_pid_to_index(dataset):
    pid2idx = {}
    duplicates = {}
    for i in range(len(dataset)):
        pid = get_patient_id(dataset, i)
        if pid is None:
            continue
        if pid not in pid2idx:
            pid2idx[pid] = i
        else:
            duplicates.setdefault(pid, [pid2idx[pid]]).append(i)
    return pid2idx, duplicates

# ============================================================
# DATASET LOAD
# ============================================================
print("Loading dataset...")
train_full = get_datasets(seed=42, on="test")  # deterministic
val_full   = get_datasets(seed=42, on="test")  # deterministic
print(f"Train dataset: {len(train_full)} | Val dataset: {len(val_full)}")

train_pid2idx, dup_train = build_pid_to_index(train_full)
val_pid2idx, dup_val     = build_pid_to_index(val_full)

if dup_train:
    print(f"⚠️ Duplicate PIDs in train_full: {len(dup_train)} (using first occurrence)")
if dup_val:
    print(f"⚠️ Duplicate PIDs in val_full: {len(dup_val)} (using first occurrence)")

label_pid_set = set(normalize_pid(k) for k in grade_labels.keys())
common_pids = sorted(set(train_pid2idx.keys()) & set(val_pid2idx.keys()) & label_pid_set)

if len(common_pids) == 0:
    raise RuntimeError("No common PIDs across train_full, val_full, and label CSV.")

all_grades = [grade_labels[pid] for pid in common_pids]

print(f"\nMatched common labeled PIDs: {len(common_pids)}")
print("First 5 matches:")
for pid in common_pids[:5]:
    print(f"  pid={pid} label={grade_labels[pid]} train_idx={train_pid2idx[pid]} val_idx={val_pid2idx[pid]}")

train_pids, val_pids = train_test_split(
    common_pids,
    test_size=0.20,
    random_state=42,
    stratify=all_grades,
)

train_global_idx = [train_pid2idx[pid] for pid in train_pids]
val_global_idx   = [val_pid2idx[pid] for pid in val_pids]
train_grades     = [grade_labels[pid] for pid in train_pids]
val_grades       = [grade_labels[pid] for pid in val_pids]

train_dataset = Subset(train_full, train_global_idx)
val_dataset   = Subset(val_full,   val_global_idx)

print(f"\nTrain: {len(train_dataset)} | Val: {len(val_dataset)}")
print(f"Train MGMT0/MGMT1: {train_grades.count(0)}/{train_grades.count(1)}")
print(f"Val   MGMT0/MGMT1: {val_grades.count(0)}/{val_grades.count(1)}")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=pin_mem,
    collate_fn=custom_collate,
    drop_last=False,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=pin_mem,
    collate_fn=determinist_collate,
    drop_last=False,
)

# ============================================================
# SEGMENTOR
# ============================================================
print("\nLoading segmentation model...")
segmentor = WaveletUNetPlusPlus(in_channels=4, n_classes=3)

if not os.path.exists(SEG_CKPT):
    print(f"Checkpoint not found: {SEG_CKPT}")
    sys.exit(1)

ckpt = torch.load(SEG_CKPT, map_location="cpu")
state_dict = ckpt.get("model_state", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
segmentor.load_state_dict(state_dict, strict=True)
print(f"✅ Loaded checkpoint: {SEG_CKPT}")

for p in segmentor.parameters():
    p.requires_grad = False
segmentor.eval().to(seg_device)
print(f"Segmentor frozen on {seg_device}. ✅")

# ============================================================
# MULTI-SCALE HOOKS
# ============================================================
enc_features = {}

def make_hook(name):
    def hook(module, inputs, output):
        enc_features[name] = output.detach().to(device=cls_device, dtype=torch.float32)
    return hook

hook_layers = ["conv0_0", "conv1_0", "conv2_0", "conv3_0", "conv4_0"]
registered = []
for lname in hook_layers:
    if hasattr(segmentor, lname):
        getattr(segmentor, lname).register_forward_hook(make_hook(lname))
        registered.append(lname)

print(f"✅ Hooks registered on {cls_device}: {registered}")

layer_dims = {
    "conv0_0": 16,
    "conv1_0": 32,
    "conv2_0": 64,
    "conv3_0": 128,
    "conv4_0": 256,
}
print(f"Feature channels: {layer_dims}")

# ============================================================
# CLASSIFIER BACKBONE
# ============================================================
class LightClassBackbone(nn.Module):
    def __init__(self, in_ch=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, 8, 3, stride=2, padding=1),
            nn.InstanceNorm3d(8), nn.LeakyReLU(0.1, inplace=True),

            nn.Conv3d(8, 16, 3, stride=2, padding=1),
            nn.InstanceNorm3d(16), nn.LeakyReLU(0.1, inplace=True),

            nn.Conv3d(16, 32, 3, stride=2, padding=1),
            nn.InstanceNorm3d(32), nn.LeakyReLU(0.1, inplace=True),

            nn.Conv3d(32, 64, 3, stride=2, padding=1),
            nn.InstanceNorm3d(64), nn.LeakyReLU(0.1, inplace=True),

            nn.Conv3d(64, 64, 3, stride=2, padding=1),
            nn.InstanceNorm3d(64), nn.LeakyReLU(0.1, inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.out_dim = 64

    def forward(self, x):
        return self.pool(self.net(x)).flatten(1)

class ScalePool(nn.Module):
    def __init__(self, in_ch, out_ch=32):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.reduce = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch, out_ch),
            nn.LayerNorm(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.reduce(self.gap(x))

class ClassificationHead(nn.Module):
    def __init__(self, layer_dims, backbone_dim=64, num_classes=2):
        super().__init__()
        self.scale_pools = nn.ModuleDict({
            name: ScalePool(ch, out_ch=32) for name, ch in layer_dims.items()
        })
        seg_feat_dim = 32 * len(layer_dims)
        total_dim = seg_feat_dim + backbone_dim

        self.head = nn.Sequential(
            nn.LayerNorm(total_dim),
            nn.Linear(total_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, enc_feats_dict, backbone_feat):
        scale_vecs = [
            self.scale_pools[name](enc_feats_dict[name].float())
            for name in self.scale_pools if name in enc_feats_dict
        ]
        combined = torch.cat(scale_vecs + [backbone_feat.float()], dim=1)
        return self.head(combined)

backbone = LightClassBackbone(in_ch=4).to(cls_device)
classifier = ClassificationHead(layer_dims, backbone_dim=backbone.out_dim).to(cls_device)

n_params = sum(p.numel() for p in backbone.parameters()) + sum(p.numel() for p in classifier.parameters())
print(f"✅ Backbone + Classifier on {cls_device} | Trainable params: {n_params/1e6:.2f}M")

# ============================================================
# LOSS / OPT
# ============================================================
w0 = np.sqrt(n_mgmt1 / (n_mgmt0 + n_mgmt1))
w1 = np.sqrt(n_mgmt0 / (n_mgmt0 + n_mgmt1))
ws = w0 + w1
w0, w1 = w0 / ws, w1 / ws

class_weights = torch.tensor([w0, w1], dtype=torch.float32, device=cls_device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

train_params = list(backbone.parameters()) + list(classifier.parameters())
optimizer = torch.optim.AdamW(train_params, lr=LEARNING_RATE_CLASS, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=30, T_mult=2, eta_min=1e-6
)

print(f"\nClass weights: MGMT0={w0:.3f} MGMT1={w1:.3f}")
print(f"Batch={BATCH_SIZE}, Accum={ACCUM_STEPS}, Effective batch={BATCH_SIZE * ACCUM_STEPS}")

# ============================================================
# HELPERS
# ============================================================
def get_label_tensor_from_global_indices(base_dataset, global_indices):
    labels = []
    for gi in global_indices:
        pid = normalize_pid(get_patient_id(base_dataset, gi))
        y = grade_labels.get(pid, -1)
        if y == -1:
            raise RuntimeError(f"Missing label for pid={pid}, idx={gi}")
        labels.append(y)
    return torch.tensor(labels, dtype=torch.long)

def dice_coeff(pred, gt, smooth=1e-5):
    pred = pred.contiguous().view(-1)
    gt = gt.contiguous().view(-1)
    inter = (pred * gt).sum()
    return ((2.0 * inter + smooth) / (pred.sum() + gt.sum() + smooth)).item()

# ============================================================
# TRAIN
# ============================================================
print(f"\nTraining for max {EPOCHS} epochs...\n")

best_val_auc = 0.0
best_val_acc = 0.0
epochs_no_imp = 0

BEST_CLF_CKPT = os.path.join(OUT_DIR, "classifier_best.pth")
BEST_BB_CKPT  = os.path.join(OUT_DIR, "backbone_best.pth")

for epoch in range(1, EPOCHS + 1):
    backbone.train()
    classifier.train()
    segmentor.eval()

    running_loss = 0.0
    cursor = 0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(train_loader):
        images_seg = batch["image"].float().to(seg_device, non_blocking=seg_on_cuda)
        images_cls = batch["image"].float().to(cls_device, non_blocking=cls_on_cuda)
        B = images_seg.size(0)

        local_positions = list(range(cursor, cursor + B))
        cursor += B
        global_idxs = [train_dataset.indices[pos] for pos in local_positions]
        labels_clf = get_label_tensor_from_global_indices(train_full, global_idxs).to(cls_device)

        enc_features.clear()
        with torch.no_grad():
            with autocast_if_seg_cuda():
                _ = segmentor(images_seg)

        feats = {k: enc_features[k] for k in registered if k in enc_features}
        enc_features.clear()

        bb_feat = backbone(images_cls)
        logits = classifier(feats, bb_feat)
        loss = criterion(logits, labels_clf) / ACCUM_STEPS
        loss.backward()

        if (step + 1) % ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(train_params, max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item() * ACCUM_STEPS

        del images_seg, images_cls, feats, bb_feat, logits, loss, labels_clf

    # flush remainder
    if len(train_loader) % ACCUM_STEPS != 0:
        torch.nn.utils.clip_grad_norm_(train_params, max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    scheduler.step()
    avg_loss = running_loss / max(len(train_loader), 1)

    # ---------------- VALIDATE ----------------
    if epoch % VAL_EVERY == 0 or epoch == EPOCHS:
        backbone.eval()
        classifier.eval()
        segmentor.eval()

        val_preds, val_probs, val_true = [], [], []
        dice_scores = []
        vcursor = 0

        with torch.no_grad():
            for batch in val_loader:
                images_seg = batch["image"].float().to(seg_device, non_blocking=seg_on_cuda)
                images_cls = batch["image"].float().to(cls_device, non_blocking=cls_on_cuda)
                labels_seg = batch["label"].float()
                B = images_seg.size(0)

                local_positions = list(range(vcursor, vcursor + B))
                vcursor += B
                global_idxs = [val_dataset.indices[pos] for pos in local_positions]
                labels_clf = get_label_tensor_from_global_indices(val_full, global_idxs).to(cls_device)

                enc_features.clear()
                with autocast_if_seg_cuda():
                    raw_seg_dev = segmentor(images_seg)
                raw_seg_cpu = raw_seg_dev.detach().cpu()

                feats = {k: enc_features[k] for k in registered if k in enc_features}
                enc_features.clear()

                bb_feat = backbone(images_cls)
                logits = classifier(feats, bb_feat)

                probs = torch.softmax(logits, dim=1)[:, 1]
                preds = torch.argmax(logits, dim=1)

                val_preds.extend(preds.detach().cpu().numpy().tolist())
                val_probs.extend(probs.detach().cpu().numpy().tolist())
                val_true.extend(labels_clf.detach().cpu().numpy().tolist())

                pred_mask = (torch.sigmoid(raw_seg_cpu) > 0.5).float()
                for b in range(B):
                    d = np.mean([
                        dice_coeff(pred_mask[b, 0], labels_seg[b, 0]),
                        dice_coeff(pred_mask[b, 1], labels_seg[b, 1]),
                        dice_coeff(pred_mask[b, 2], labels_seg[b, 2]),
                    ])
                    dice_scores.append(d)

                del images_seg, images_cls, labels_seg, raw_seg_dev, raw_seg_cpu
                del pred_mask, feats, bb_feat, logits, labels_clf, probs, preds

        val_acc = accuracy_score(val_true, val_preds) * 100.0
        val_rec = recall_score(val_true, val_preds, zero_division=0) * 100.0
        val_pre = precision_score(val_true, val_preds, zero_division=0) * 100.0
        val_auc = roc_auc_score(val_true, val_probs) if len(np.unique(val_true)) > 1 else 0.5
        avg_dice = float(np.mean(dice_scores)) if len(dice_scores) else float("nan")
        lr_now = optimizer.param_groups[0]["lr"]

        val_arr = np.array(val_true)
        pred_arr = np.array(val_preds)
        c0_acc = (pred_arr[val_arr == 0] == 0).mean() * 100.0 if (val_arr == 0).any() else 0.0
        c1_acc = (pred_arr[val_arr == 1] == 1).mean() * 100.0 if (val_arr == 1).any() else 0.0

        print(
            f"Epoch [{epoch:3d}/{EPOCHS}] Loss={avg_loss:.4f} | "
            f"Acc={val_acc:.1f}% (MGMT0={c0_acc:.0f}% MGMT1={c1_acc:.0f}%) "
            f"Rec={val_rec:.1f}% Pre={val_pre:.1f}% "
            f"AUC={val_auc:.4f} Dice={avg_dice:.4f} LR={lr_now:.1e}"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_acc = val_acc
            epochs_no_imp = 0
            torch.save(classifier.state_dict(), BEST_CLF_CKPT)
            torch.save(backbone.state_dict(), BEST_BB_CKPT)
            print(f"  ✅ Best saved (AUC={best_val_auc:.4f}, Acc={best_val_acc:.1f}%)")
        else:
            epochs_no_imp += 1
            print(f"  No improvement ({epochs_no_imp}/{PATIENCE})")
            if epochs_no_imp >= PATIENCE:
                print(f"Early stopping at epoch {epoch}.")
                break

# ============================================================
# FINAL EVALUATION
# ============================================================
print(f"\n{'='*60}")
print("FINAL EVALUATION ON VALIDATION SET")
print(f"{'='*60}")

classifier.load_state_dict(torch.load(BEST_CLF_CKPT, map_location=cls_device))
backbone.load_state_dict(torch.load(BEST_BB_CKPT, map_location=cls_device))
classifier.eval()
backbone.eval()
segmentor.eval()

all_preds, all_probs, all_labels_final = [], [], []
all_pids_final, dice_scores_final = [], []
vcursor = 0

with torch.no_grad():
    for batch in val_loader:
        images_seg = batch["image"].float().to(seg_device, non_blocking=seg_on_cuda)
        images_cls = batch["image"].float().to(cls_device, non_blocking=cls_on_cuda)
        labels_seg = batch["label"].float()
        B = images_seg.size(0)

        local_positions = list(range(vcursor, vcursor + B))
        vcursor += B
        global_idxs = [val_dataset.indices[pos] for pos in local_positions]
        labels_clf = get_label_tensor_from_global_indices(val_full, global_idxs).to(cls_device)
        pids = [normalize_pid(get_patient_id(val_full, gi)) for gi in global_idxs]

        enc_features.clear()
        with autocast_if_seg_cuda():
            raw_seg_dev = segmentor(images_seg)
        raw_seg_cpu = raw_seg_dev.detach().cpu()

        feats = {k: enc_features[k] for k in registered if k in enc_features}
        enc_features.clear()

        bb_feat = backbone(images_cls)
        logits = classifier(feats, bb_feat)

        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)

        pred_mask = (torch.sigmoid(raw_seg_cpu) > 0.5).float()
        for b in range(B):
            d = np.mean([
                dice_coeff(pred_mask[b, 0], labels_seg[b, 0]),
                dice_coeff(pred_mask[b, 1], labels_seg[b, 1]),
                dice_coeff(pred_mask[b, 2], labels_seg[b, 2]),
            ])
            dice_scores_final.append(d)

        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_probs.extend(probs.detach().cpu().numpy().tolist())
        all_labels_final.extend(labels_clf.detach().cpu().numpy().tolist())
        all_pids_final.extend(pids)

        del images_seg, images_cls, labels_seg, raw_seg_dev, raw_seg_cpu
        del pred_mask, feats, bb_feat, logits, labels_clf, probs, preds

all_preds = np.array(all_preds)
all_probs = np.array(all_probs)
all_labels_final = np.array(all_labels_final)
avg_dice = float(np.mean(dice_scores_final)) if len(dice_scores_final) else float("nan")

acc = accuracy_score(all_labels_final, all_preds) * 100.0
rec = recall_score(all_labels_final, all_preds, zero_division=0) * 100.0
pre = precision_score(all_labels_final, all_preds, zero_division=0) * 100.0
auc = roc_auc_score(all_labels_final, all_probs) if len(np.unique(all_labels_final)) > 1 else 0.5
cm = confusion_matrix(all_labels_final, all_preds)

c0_acc = cm[0, 0] / cm[0].sum() * 100.0 if cm.shape == (2, 2) and cm[0].sum() > 0 else 0.0
c1_acc = cm[1, 1] / cm[1].sum() * 100.0 if cm.shape == (2, 2) and cm[1].sum() > 0 else 0.0

print(f"\n  Mean Dice  : {avg_dice:.4f} ({avg_dice*100:.2f}%)")
print(f"  Accuracy   : {acc:.2f}%")
print(f"  Recall     : {rec:.2f}%")
print(f"  Precision  : {pre:.2f}%")
print(f"  AUC Score  : {auc:.4f}")

print("\nPer-class accuracy:")
if cm.shape == (2, 2):
    print(f"  MGMT0 : {c0_acc:.1f}%  ({cm[0,0]}/{cm[0].sum()} correct)")
    print(f"  MGMT1 : {c1_acc:.1f}%  ({cm[1,1]}/{cm[1].sum()} correct)")

print("\nConfusion Matrix:")
print("                 Pred_0  Pred_1")
if cm.shape == (2, 2):
    print(f"  True MGMT0  :  {cm[0,0]:6d}  {cm[0,1]:6d}")
    print(f"  True MGMT1  :  {cm[1,0]:6d}  {cm[1,1]:6d}")
else:
    print(cm)

print("\nClassification Report:")
print(classification_report(
    all_labels_final,
    all_preds,
    target_names=["MGMT0 (0)", "MGMT1 (1)"],
    digits=4,
    zero_division=0
))

results_df = pd.DataFrame({
    "BraTS21ID": [str(p).zfill(5) if str(p).isdigit() else str(p) for p in all_pids_final],
    "true_label": all_labels_final,
    "pred_label": all_preds,
    "prob_class1": np.round(all_probs, 4),
    "dice_mean": np.round(dice_scores_final, 4),
})

seg_csv = "final_test_results.csv"
if os.path.exists(seg_csv):
    seg_df = pd.read_csv(seg_csv, dtype=str)
    for col in ["Patient_ID", "patient_id", "BraTS21ID", "id"]:
        if col in seg_df.columns:
            seg_df = seg_df.rename(columns={col: "BraTS21ID"})
            seg_df["BraTS21ID"] = seg_df["BraTS21ID"].astype(str).str.strip()
            break
    merged = results_df.merge(seg_df, on="BraTS21ID", how="left")
    merged.to_csv(OUTPUT_CSV, index=False)
else:
    results_df.to_csv(OUTPUT_CSV, index=False)

print(f"\n✅ Saved results → {OUTPUT_CSV}")