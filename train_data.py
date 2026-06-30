#!/usr/bin/env python3

import os
import gc
import random
import argparse
import warnings

import torch
import torch.optim as optim
import torch.nn as nn

import nibabel as nib
import numpy as np

from skimage.transform import resize
from torch.utils.data import Dataset, DataLoader

from model import WaveletUNetPlusPlus

warnings.filterwarnings("ignore")


# ============================================================
# Reproducibility
# ============================================================

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Dataset
# ============================================================

class PontineDataset(Dataset):
    def __init__(
        self,
        images_dir,
        labels_dir,
        file_list,
        target_shape=(128, 128, 128),
        augment=True,
    ):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.file_list = file_list
        self.target_shape = target_shape
        self.augment = augment

    def __len__(self):
        return len(self.file_list)

    def load_nifti(self, path):
        obj = nib.as_closest_canonical(nib.load(path))
        return obj.get_fdata().astype(np.float32)

    def normalize_image(self, img):
        p1, p99 = np.percentile(img, [1, 99])
        img = np.clip(img, p1, p99)

        mean = np.mean(img)
        std = np.std(img)

        img = (img - mean) / (std + 1e-8)

        return img.astype(np.float32)

    def prepare_image_4ch(self, img_data):
        channels = []

        if img_data.ndim == 3:
            img = self.normalize_image(img_data)

            img = resize(
                img,
                self.target_shape,
                order=1,
                preserve_range=True,
                anti_aliasing=True,
            ).astype(np.float32)

            channels = [img, img, img, img]

        elif img_data.ndim == 4:
            # H, W, D, C
            if img_data.shape[-1] <= 10:
                for c in range(img_data.shape[-1]):
                    ch = self.normalize_image(img_data[..., c])

                    ch = resize(
                        ch,
                        self.target_shape,
                        order=1,
                        preserve_range=True,
                        anti_aliasing=True,
                    ).astype(np.float32)

                    channels.append(ch)

            # C, H, W, D
            elif img_data.shape[0] <= 10:
                for c in range(img_data.shape[0]):
                    ch = self.normalize_image(img_data[c])

                    ch = resize(
                        ch,
                        self.target_shape,
                        order=1,
                        preserve_range=True,
                        anti_aliasing=True,
                    ).astype(np.float32)

                    channels.append(ch)

            else:
                raise ValueError(f"Unsupported 4D image shape: {img_data.shape}")

            if len(channels) >= 4:
                channels = channels[:4]
            else:
                while len(channels) < 4:
                    channels.append(channels[-1].copy())

        else:
            raise ValueError(f"Unsupported image shape: {img_data.shape}")

        img_input = np.stack(channels, axis=0).astype(np.float32)

        return img_input

    def prepare_label_1ch(self, label_data):
        if label_data.ndim == 4:
            if label_data.shape[-1] == 1:
                label_data = label_data[..., 0]
            elif label_data.shape[0] == 1:
                label_data = label_data[0]
            else:
                label_data = np.max(label_data, axis=-1)

        label = (label_data > 0).astype(np.float32)

        label = resize(
            label,
            self.target_shape,
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        ).astype(np.float32)

        label = (label > 0.5).astype(np.float32)

        label = np.expand_dims(label, axis=0)

        return label.astype(np.float32)

    def random_augment(self, img, label):
        # img shape   : 4, H, W, D
        # label shape : 1, H, W, D

        # Random flips
        for axis in [1, 2, 3]:
            if random.random() < 0.5:
                img = np.flip(img, axis=axis).copy()
                label = np.flip(label, axis=axis).copy()

        # Random intensity scaling
        if random.random() < 0.5:
            scale = random.uniform(0.9, 1.1)
            shift = random.uniform(-0.1, 0.1)
            img = img * scale + shift

        # Gaussian noise
        if random.random() < 0.3:
            noise = np.random.normal(0, 0.03, size=img.shape).astype(np.float32)
            img = img + noise

        return img.astype(np.float32), label.astype(np.float32)

    def __getitem__(self, idx):
        img_name = self.file_list[idx]

        img_path = os.path.join(self.images_dir, img_name)
        label_path = os.path.join(self.labels_dir, img_name)

        img_data = self.load_nifti(img_path)
        label_data = self.load_nifti(label_path)

        img_input = self.prepare_image_4ch(img_data)
        label_input = self.prepare_label_1ch(label_data)

        if self.augment:
            img_input, label_input = self.random_augment(img_input, label_input)

        return (
            torch.from_numpy(img_input).float(),
            torch.from_numpy(label_input).float(),
            img_name,
        )


# ============================================================
# Loss and Metrics
# ============================================================

class DiceBCELoss(nn.Module):
    def __init__(self, bce_weight=0.3, dice_weight=0.7, smooth=1e-5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)

        dims = tuple(range(1, probs.ndim))

        intersection = torch.sum(probs * targets, dim=dims)
        denominator = torch.sum(probs, dim=dims) + torch.sum(targets, dim=dims)

        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        dice_loss = 1.0 - dice.mean()

        loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss

        return loss


def dice_score_torch(logits, targets, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    dims = tuple(range(1, preds.ndim))

    intersection = torch.sum(preds * targets, dim=dims)
    pred_sum = torch.sum(preds, dim=dims)
    target_sum = torch.sum(targets, dim=dims)

    dice = torch.zeros_like(intersection)

    both_empty = (pred_sum == 0) & (target_sum == 0)
    normal = ~both_empty

    dice[both_empty] = float("nan")

    dice[normal] = (2.0 * intersection[normal]) / (
        pred_sum[normal] + target_sum[normal] + 1e-8
    )

    return dice


# ============================================================
# Checkpoint loading
# ============================================================

def load_checkpoint_flexible(path, device):
    ckpt = torch.load(path, map_location=device)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    clean_state = {}

    for k, v in state.items():
        clean_state[k.replace("module.", "")] = v

    return clean_state


def load_partial_weights(model, checkpoint_path, device):
    print(f"Loading checkpoint: {checkpoint_path}")

    pretrained = load_checkpoint_flexible(checkpoint_path, device)
    model_state = model.state_dict()

    compatible = {}
    loaded = 0
    skipped = 0

    for k, v in pretrained.items():
        if k in model_state and model_state[k].shape == v.shape:
            compatible[k] = v
            loaded += 1
        else:
            skipped += 1

    model_state.update(compatible)
    model.load_state_dict(model_state, strict=True)

    print(f"Compatible layers loaded: {loaded}")
    print(f"Skipped layers          : {skipped}")

    return model


# ============================================================
# Training on full data
# ============================================================

def train_full_data(args):
    seed_everything(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 72)
    print("Pontine Binary FULL-DATA Training")
    print("=" * 72)
    print(f"Images dir      : {args.images_dir}")
    print(f"Labels dir      : {args.labels_dir}")
    print(f"Checkpoint      : {args.checkpoint}")
    print(f"Save dir        : {args.save_dir}")
    print(f"Target shape    : {tuple(args.target_shape)}")
    print(f"Epochs          : {args.epochs}")
    print(f"Batch size      : {args.batch_size}")
    print(f"LR              : {args.lr}")
    print(f"Device          : {device}")
    print("=" * 72)

    all_files = sorted(
        [
            f
            for f in os.listdir(args.images_dir)
            if f.endswith(".nii.gz") or f.endswith(".nii")
        ]
    )

    all_files = [
        f for f in all_files if os.path.exists(os.path.join(args.labels_dir, f))
    ]

    random.shuffle(all_files)

    print(f"Total training cases: {len(all_files)}")

    dataset = PontineDataset(
        args.images_dir,
        args.labels_dir,
        all_files,
        target_shape=tuple(args.target_shape),
        augment=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = WaveletUNetPlusPlus(
        in_channels=4,
        n_classes=1,
    ).to(device)

    if args.checkpoint is not None and os.path.exists(args.checkpoint):
        model = load_partial_weights(model, args.checkpoint, device)
    else:
        print("No checkpoint loaded. Training from scratch.")

    criterion = DiceBCELoss(
        bce_weight=args.bce_weight,
        dice_weight=args.dice_weight,
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_train_dice = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()

        train_loss = 0.0
        train_dices = []

        for step, batch in enumerate(loader, start=1):
            images, labels, names = batch

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()

            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

            with torch.no_grad():
                dice = dice_score_torch(
                    logits.detach(),
                    labels.detach(),
                    threshold=args.threshold,
                )

                dice_np = dice.detach().cpu().numpy()

                for d in dice_np:
                    if not np.isnan(d):
                        train_dices.append(float(d))

            del images, labels, logits, loss

        avg_train_loss = train_loss / max(len(loader), 1)
        mean_train_dice = float(np.mean(train_dices)) if len(train_dices) > 0 else 0.0

        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Loss: {avg_train_loss:.6f} | "
            f"Train Dice: {mean_train_dice:.4f} | "
            f"LR: {current_lr:.8f}"
        )

        # Save best training checkpoint
        if mean_train_dice > best_train_dice:
            best_train_dice = mean_train_dice

            best_path = os.path.join(args.save_dir, "pontine_binary_full_best.pth")

            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_train_dice": best_train_dice,
                    "args": vars(args),
                },
                best_path,
            )

            print(
                f"💾 Saved best full-data model: {best_path} | "
                f"Best Train Dice: {best_train_dice:.4f}"
            )

        # Save periodic checkpoint
        if epoch % args.save_every == 0:
            save_path = os.path.join(
                args.save_dir,
                f"pontine_binary_full_epoch_{epoch}.pth",
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_train_dice": best_train_dice,
                    "args": vars(args),
                },
                save_path,
            )

            print(f"💾 Saved checkpoint: {save_path}")

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    final_path = os.path.join(args.save_dir, "pontine_binary_full_final.pth")

    torch.save(
        {
            "epoch": args.epochs,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_train_dice": best_train_dice,
            "args": vars(args),
        },
        final_path,
    )

    print("=" * 72)
    print("FULL-DATA Training finished.")
    print(f"Best Train Dice : {best_train_dice:.4f}")
    print(f"Final model     : {final_path}")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--images_dir",
        type=str,
        default="/home/kamilabdelali/anotherdata/images",
    )

    parser.add_argument(
        "--labels_dir",
        type=str,
        default="/home/kamilabdelali/anotherdata/labels",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/segmentor_epoch_650.pth",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints_pontine_binary_full",
    )

    parser.add_argument(
        "--target_shape",
        type=int,
        nargs=3,
        default=[128, 128, 128],
    )

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--bce_weight", type=float, default=0.3)
    parser.add_argument("--dice_weight", type=float, default=0.7)

    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--mode", type=str, default="full", choices=["full", "val"])

    parser.add_argument("--save_every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--amp", action="store_true")

    args = parser.parse_args()

    train_full_data(args)


if __name__ == "__main__":
    main()