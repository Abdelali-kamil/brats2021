import argparse
import json
import os, sys, glob, math, random, heapq
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from brats_gbm.model import WaveletUNetPlusPlus
from brats_gbm.splits import upenn_split, assert_disjoint
from brats_gbm.data.upenn import PREPROCESSING, normalise

# ============================================================
# CONFIG
# ============================================================
UPENN_NIFTI_DIR = str(ROOT / "upenn_nifti")

# Start from the BraTS-trained segmentor, NOT a previous UPenn checkpoint:
# every earlier UPenn checkpoint was trained against an empty ET target and has
# unlearned enhancing-tumour prediction. Resuming from one would inherit that.
RESUME_FROM     = str(ROOT / "checkpoints" / "segmentor_epoch_650.pth")

# Enhancing-tumour label: 4 in BraTS, 3 in UPENN-GBM.
ET_LABELS = (3, 4)

SPATIAL_SIZE    = (128, 128, 128)

# OOM-safe defaults
BATCH_SIZE      = 1
GRAD_ACCUM_STEPS = 2           # effective batch = BATCH_SIZE * GRAD_ACCUM_STEPS
USE_AMP         = True         # mixed precision

NUM_WORKERS     = 2
NUM_EPOCHS      = 80
PEAK_LR         = 3e-5
WARMUP_EPOCHS   = 5
MIN_LR          = 1e-6
WEIGHT_DECAY    = 1e-5

CHANNEL_WEIGHTS = [1.0, 1.5, 1.5]
FOCAL_GAMMA     = 2.0
FOCAL_ALPHA     = 0.75

TOP_K           = 5
SAVE_TOPK_DIR   = str(ROOT / "checkpoints" / "upenn_v3_topk")

EMA_ALPHA       = 0.3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VAL_ET_THR = 0.30
VAL_TC_THR = 0.40
VAL_WT_THR = 0.35


# ============================================================
# AUGMENTATION
# ============================================================
def augment_volume(img: np.ndarray, label: np.ndarray):
    # random flips
    for ax in [1, 2, 3]:  # D,H,W in [C,D,H,W]
        if random.random() < 0.5:
            img = np.flip(img, axis=ax)
            label = np.flip(label, axis=ax)

    # intensity scale + shift
    for c in range(img.shape[0]):
        scale = random.uniform(0.85, 1.15)
        shift = random.uniform(-0.10, 0.10)
        img[c] = img[c] * scale + shift

    # gaussian noise
    if random.random() < 0.30:
        sigma = random.uniform(0.00, 0.08)
        img = img + np.random.randn(*img.shape).astype(np.float32) * sigma

    return img.copy(), label.copy()


# ============================================================
# CROP / PAD  (FIXED)
# ============================================================
def crop_or_pad(arr: np.ndarray, target: tuple) -> np.ndarray:
    """
    Center-crop or zero-pad array to `target` over last len(target) dims.
    Works for [D,H,W] and [C,D,H,W].
    """
    if arr.ndim not in (3, 4):
        raise ValueError(f"Expected 3D or 4D array, got shape {arr.shape}")

    spatial_in = arr.shape[-len(target):]
    out_shape = arr.shape[:-len(target)] + target
    out = np.zeros(out_shape, dtype=arr.dtype)

    src_slices = [slice(None)] * (arr.ndim - len(target))
    dst_slices = [slice(None)] * (arr.ndim - len(target))

    for s, t in zip(spatial_in, target):
        src_start = max(0, (s - t) // 2)
        dst_start = max(0, (t - s) // 2)
        length = min(s, t)
        src_slices.append(slice(src_start, src_start + length))
        dst_slices.append(slice(dst_start, dst_start + length))

    out[tuple(dst_slices)] = arr[tuple(src_slices)]
    return out


# ============================================================
# DATASET
# ============================================================
class UPENNDataset(Dataset):
    """UPenn training crops.

    `preprocessing` must match the checkpoint being fine-tuned from. The BraTS
    segmentor was trained on [t1, t1ce, t2, flair] with percentile min-max;
    feeding it z-scored [flair, t1, t1ce, t2] hands it a channel permutation
    and a different intensity scale, so fine-tuning has to spend capacity
    undoing that before it can learn anything about the new cohort. Matching
    the source convention is the scientifically correct default here.
    """

    def __init__(self, data_dir: str, subject_ids: list, do_augment: bool = False,
                 preprocessing: str = "brats"):
        self.data_dir = Path(data_dir)
        self.subject_ids = list(subject_ids)
        self.do_augment = do_augment
        if preprocessing not in PREPROCESSING:
            raise ValueError(f"unknown preprocessing {preprocessing!r}")
        self.preprocessing = preprocessing
        self.suffixes, self.normaliser = PREPROCESSING[preprocessing]

    def __len__(self):
        return len(self.subject_ids)

    def _load_mod(self, sub_id: str, suffix: str) -> np.ndarray:
        p = self.data_dir / f"{sub_id}_{suffix}.nii.gz"
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")
        return nib.load(str(p)).get_fdata().astype(np.float32)

    def __getitem__(self, idx: int):
        sub_id = self.subject_ids[idx]

        seg = self._load_mod(sub_id, "seg")

        # Channel order and normaliser both come from the declared convention.
        img = np.stack([self._load_mod(sub_id, s) for s in self.suffixes], axis=0)
        img = normalise(img, self.normaliser)
        img = crop_or_pad(img, SPATIAL_SIZE)

        # BraTS regions
        # Enhancing tumour is label 4 in BraTS but label 3 in UPENN-GBM.
        # Reading only label 4 here made the ET target permanently empty and
        # trained the model to stop predicting enhancing tumour.
        et = np.isin(seg, ET_LABELS).astype(np.float32)
        tc = (np.isin(seg, ET_LABELS) | (seg == 1)).astype(np.float32)
        wt = (np.isin(seg, ET_LABELS) | (seg == 1) | (seg == 2)).astype(np.float32)
        label = np.stack([et, tc, wt], axis=0)
        label = crop_or_pad(label, SPATIAL_SIZE)

        if self.do_augment:
            img, label = augment_volume(img, label)

        # fail-fast shape checks
        if img.shape != (4, *SPATIAL_SIZE):
            raise ValueError(f"Bad image shape for {sub_id}: {img.shape}")
        if label.shape != (3, *SPATIAL_SIZE):
            raise ValueError(f"Bad label shape for {sub_id}: {label.shape}")

        return {
            "image": torch.from_numpy(img).float(),
            "label": torch.from_numpy(label).float(),
        }




# ============================================================
# LOSSES
# ============================================================
def dice_loss_per_channel(probs: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-5):
    inter = (probs * targets).sum(dim=(0, 2, 3, 4))
    union = probs.sum(dim=(0, 2, 3, 4)) + targets.sum(dim=(0, 2, 3, 4))
    return 1.0 - (2.0 * inter + smooth) / (union + smooth)


def focal_loss_per_channel(logits: torch.Tensor, targets: torch.Tensor,
                           gamma: float = 2.0, alpha: float = 0.75):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = torch.exp(-bce)
    weight = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    focal = weight * (1.0 - pt) ** gamma * bce
    return focal.mean(dim=(0, 2, 3, 4))


def combined_loss_fn(logits: torch.Tensor, targets: torch.Tensor,
                     channel_weights: list, gamma: float = 2.0, alpha: float = 0.75):
    probs = torch.sigmoid(logits)
    w = torch.tensor(channel_weights, dtype=torch.float32, device=logits.device)
    dl = dice_loss_per_channel(probs, targets)
    fl = focal_loss_per_channel(logits, targets, gamma, alpha)
    per_c = 0.5 * dl + 0.5 * fl
    return (per_c * w).mean()


# ============================================================
# SCHEDULER
# ============================================================
def make_lr_lambda(warmup_epochs: int, total_epochs: int, peak_lr: float, min_lr: float):
    base_ratio = 1e-8 / peak_lr
    min_ratio = min_lr / peak_lr

    def _fn(epoch: int) -> float:
        if epoch < warmup_epochs:
            return base_ratio + (1.0 - base_ratio) * (epoch / max(1, warmup_epochs))
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return _fn


# ============================================================
# VALIDATION
# ============================================================
@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, et_thr: float, tc_thr: float, wt_thr: float):
    model.eval()
    et_d, tc_d, wt_d = [], [], []

    for batch in loader:
        images = batch["image"].to(DEVICE, non_blocking=True)
        labels = batch["label"].numpy()
        probs = torch.sigmoid(model(images)).detach().cpu().numpy()

        for b in range(images.shape[0]):
            def _dice(pr, gt):
                pr, gt = pr.astype(np.float32), gt.astype(np.float32)
                inter = (pr * gt).sum()
                denom = pr.sum() + gt.sum()
                return 1.0 if denom == 0 else float(2.0 * inter / denom)

            et_d.append(_dice((probs[b, 0] > et_thr).astype(float), labels[b, 0]))
            tc_d.append(_dice((probs[b, 1] > tc_thr).astype(float), labels[b, 1]))
            wt_d.append(_dice((probs[b, 2] > wt_thr).astype(float), labels[b, 2]))

    model.train()
    et = float(np.mean(et_d))
    tc = float(np.mean(tc_d))
    wt = float(np.mean(wt_d))
    return (et + tc + wt) / 3.0, et, tc, wt


# ============================================================
# TOP-K CHECKPOINTS
# ============================================================
class TopKCheckpoints:
    def __init__(self, k: int, save_dir: str):
        self.k = k
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.heap = []  # (score, path_str)

    def update(self, model: nn.Module, val_mean: float, epoch: int):
        fname = self.save_dir / f"ep{epoch:04d}_val{val_mean:.4f}.pth"
        torch.save(
            {"epoch": epoch, "val_mean": val_mean, "model_state_dict": model.state_dict()},
            str(fname)
        )
        heapq.heappush(self.heap, (val_mean, str(fname)))
        if len(self.heap) > self.k:
            _, worst_path = heapq.heappop(self.heap)
            try:
                Path(worst_path).unlink()
            except FileNotFoundError:
                pass

    def all_paths(self):
        return [p for _, p in sorted(self.heap, key=lambda x: -x[0])]


# ============================================================
# MAIN
# ============================================================
def main():
    global NUM_EPOCHS
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subject-split", default=None,
                    help="JSON with explicit {train,val,test} subject lists "
                         "(used by crossval_upenn.py). Defaults to the frozen "
                         "project split from brats_gbm.splits.")
    ap.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    ap.add_argument("--save-dir", default=str(ROOT / "checkpoints"))
    ap.add_argument("--nifti-dir", default=UPENN_NIFTI_DIR)
    ap.add_argument("--preprocessing", default="brats", choices=list(PREPROCESSING),
                    help="input convention; must match the resumed checkpoint "
                         "(default: brats, matching RESUME_FROM)")
    args = ap.parse_args()

    NUM_EPOCHS = args.epochs
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    topk_dir = save_dir / "topk"

    if args.subject_split:
        split = json.loads(Path(args.subject_split).read_text())
        train_subs, val_subs = split["train"], split["val"]
        held = split.get("test", [])
    else:
        train_subs, val_subs, held = upenn_split(args.nifti_dir)

    # Training must never see a subject reserved for validation or test.
    assert_disjoint(train=train_subs, val=val_subs, test=held)

    print(f"Device : {DEVICE}")
    print(f"Resume : {RESUME_FROM}")
    print(f"Epochs : {NUM_EPOCHS}  Peak LR : {PEAK_LR:.1e}")
    print(f"Loss weights ET/TC/WT : {CHANNEL_WEIGHTS}")
    print(f"Batch={BATCH_SIZE}, GradAccum={GRAD_ACCUM_STEPS}, AMP={USE_AMP}")
    print(f"Save dir : {save_dir}")
    print(f"Preproc  : {args.preprocessing} "
          f"(order {PREPROCESSING[args.preprocessing][0]}, "
          f"{PREPROCESSING[args.preprocessing][1]})")
    print(f"Subjects -> Train:{len(train_subs)} Val:{len(val_subs)} "
          f"Held-out:{len(held)} (never loaded here)")

    train_ds = UPENNDataset(args.nifti_dir, train_subs, do_augment=True,
                            preprocessing=args.preprocessing)
    val_ds   = UPENNDataset(args.nifti_dir, val_subs, do_augment=False,
                            preprocessing=args.preprocessing)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    model = WaveletUNetPlusPlus(in_channels=4, n_classes=3).to(DEVICE)

    ckpt = torch.load(RESUME_FROM, map_location=DEVICE)
    if isinstance(ckpt, dict):
        sd = (
            ckpt.get("model_state_dict")
            or ckpt.get("model_state")
            or ckpt.get("state_dict")
            or ckpt
        )
    else:
        sd = ckpt
    model.load_state_dict(sd, strict=True)
    print(f"✅ Loaded weights from: {RESUME_FROM}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=PEAK_LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=make_lr_lambda(WARMUP_EPOCHS, NUM_EPOCHS, PEAK_LR, MIN_LR)
    )

    amp_enabled = (DEVICE.type == "cuda" and USE_AMP)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    topk = TopKCheckpoints(TOP_K, str(topk_dir))
    best_ema = 0.0
    ema_val = None

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader, start=1):
            images = batch["image"].to(DEVICE, non_blocking=True)
            labels = batch["label"].to(DEVICE, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(images)
                loss = combined_loss_fn(logits, labels, CHANNEL_WEIGHTS, FOCAL_GAMMA, FOCAL_ALPHA)
                loss = loss / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()

            if step % GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * GRAD_ACCUM_STEPS

        # flush remainder if needed
        if len(train_loader) % GRAD_ACCUM_STEPS != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        avg_loss = total_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        val_mean, et, tc, wt = validate(model, val_loader, VAL_ET_THR, VAL_TC_THR, VAL_WT_THR)

        ema_val = val_mean if ema_val is None else EMA_ALPHA * val_mean + (1.0 - EMA_ALPHA) * ema_val

        print(
            f"Epoch {epoch:03d}/{NUM_EPOCHS} | lr={current_lr:.2e} | "
            f"loss={avg_loss:.4f} | val={val_mean:.4f}(ema={ema_val:.4f}) "
            f"ET={et:.4f} TC={tc:.4f} WT={wt:.4f}"
        )

        topk.update(model, val_mean, epoch)

        if ema_val > best_ema:
            best_ema = ema_val
            torch.save(
                {
                    "epoch": epoch,
                    "val_mean": val_mean,
                    "ema_val": ema_val,
                    "model_state_dict": model.state_dict(),
                },
                str(save_dir / "upenn_v3_best.pth"),
            )
            print(f"  ✅ New EMA-best: upenn_v3_best.pth (ema={ema_val:.4f}, raw={val_mean:.4f})")

    torch.save(
        {"epoch": NUM_EPOCHS, "val_mean": val_mean, "model_state_dict": model.state_dict()},
        str(save_dir / "upenn_v3_last.pth"),
    )

    print("\n" + "=" * 60)
    print("Training complete.")
    print(f"Best EMA checkpoint : {save_dir / 'upenn_v3_best.pth'}  (ema={best_ema:.4f})")
    print(f"Last checkpoint     : {save_dir / 'upenn_v3_last.pth'}")
    print(f"Top-{TOP_K} pool        : {topk_dir}/")
    print("  " + "\n  ".join(topk.all_paths()))
    print("=" * 60)


if __name__ == "__main__":
    main()