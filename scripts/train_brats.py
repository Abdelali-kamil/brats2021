#!/usr/bin/env python3
"""Train the segmentor on BraTS2021.

The 80/20 partition this script creates is a two-way split: the 20% portion is
used both to watch validation Dice during training and, historically, to report
performance. It is therefore internal validation, not a clean test set. The
clean external test in this project is UPenn-GBM. See docs/METHODOLOGY.md.
"""
import os
import sys
import glob
import json
import argparse
import datetime
import random
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brats_gbm.data.brats import get_brats_train_val
from brats_gbm.data.collate import custom_collate
from brats_gbm.model import WaveletUNetPlusPlus


# -------------------------
# Helpers
# -------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def auto_num_workers(requested=None, prefer=4):
    if requested is not None:
        return max(0, requested)
    cpu = os.cpu_count() or 4
    return min(prefer, max(0, cpu - 1))


def dice_numpy(pred, gt, eps=1e-6):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return float((2.0 * inter) / (denom + eps))


def pick_device(device_arg: str):
    """
    device_arg: auto | cpu | cuda | cuda:0 ...
    """
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_arg)


# -------------------------
# Loss
# -------------------------
class WeightedFocalDiceLoss(nn.Module):
    def __init__(self, alpha_bce=1.0, alpha_dice=1.0, gamma=2.0, eps=1e-6, weight=None):
        super().__init__()
        self.alpha_bce = alpha_bce
        self.alpha_dice = alpha_dice
        self.gamma = gamma
        self.eps = eps
        if weight is not None:
            self.register_buffer("weight", torch.tensor(weight, dtype=torch.float32))
        else:
            self.weight = None

    def focal_bce(self, logits, targets):
        prob = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = prob * targets + (1 - prob) * (1 - targets)
        focal_term = (1 - p_t) ** self.gamma
        loss = focal_term * bce
        if self.weight is not None:
            w = self.weight.view(1, -1, *([1] * (loss.dim() - 2))).to(loss.device)
            loss = loss * w
        return loss.mean()

    def dice_loss(self, logits, targets):
        probs = torch.sigmoid(logits)
        dims = tuple(range(2, probs.dim()))
        inter = (probs * targets).sum(dims)
        denom = probs.sum(dims) + targets.sum(dims)
        dice = (2.0 * inter + self.eps) / (denom + self.eps)
        return 1.0 - dice.mean()

    def forward(self, logits, targets):
        logits = logits.float()
        targets = targets.float()
        if self.gamma > 0:
            loss_bce = self.focal_bce(logits, targets)
        else:
            loss_bce = F.binary_cross_entropy_with_logits(logits, targets)
        loss_dice = self.dice_loss(logits, targets)
        return self.alpha_bce * loss_bce + self.alpha_dice * loss_dice


# -------------------------
# Eval
# -------------------------
@torch.no_grad()
def evaluate(model, val_loader, device, thresholds=(0.5, 0.5, 0.5)):
    model.eval()
    dices = [[] for _ in range(3)]

    for batch in val_loader:
        images = batch["image"].to(device, non_blocking=(device.type == "cuda"))
        labels = batch["label"]

        if isinstance(labels, torch.Tensor):
            labels = labels.to(device, non_blocking=(device.type == "cuda")).float()
        else:
            labels = torch.tensor(np.asarray(labels), dtype=torch.float32, device=device)

        logits = model(images)
        probs = torch.sigmoid(logits)

        # Expect [B,3,...]
        probs_np = probs.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()

        for b in range(probs_np.shape[0]):
            for ch in range(3):
                pred = (probs_np[b, ch] > thresholds[ch]).astype(np.uint8)
                gt = (labels_np[b, ch] > 0.5).astype(np.uint8)
                dices[ch].append(dice_numpy(pred, gt))

    mean_et = float(np.mean(dices[0])) if len(dices[0]) else float("nan")
    mean_tc = float(np.mean(dices[1])) if len(dices[1]) else float("nan")
    mean_wt = float(np.mean(dices[2])) if len(dices[2]) else float("nan")
    overall = float(np.mean([mean_et, mean_tc, mean_wt]))

    print(f"[VAL] Dice ET/TC/WT: {mean_et:.4f}/{mean_tc:.4f}/{mean_wt:.4f} | Overall: {overall:.4f}")
    return overall


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda | cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accum-steps", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=750) # TARGET EPOCH
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--clip-norm", type=float, default=2.0)
    parser.add_argument("--weight-channels", nargs=3, type=float, default=[1.0, 1.0, 1.0])
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--downsample", type=str, default="dwt",
                        choices=["dwt", "maxpool_matched"],
                        help="encoder downsampling operator; 'maxpool_matched' is "
                             "the Table IX ablation baseline (variant A)")
    parser.add_argument("--tag", type=str, default="segmentor",
                        help="checkpoint basename; <tag>_best.pth / <tag>_last.pth")
    parser.add_argument("--data-aug", dest="data_aug", action="store_true", default=True,
                        help="spatial and intensity augmentation on the training "
                             "split (default: on)")
    parser.add_argument("--no-data-aug", dest="data_aug", action="store_false",
                        help="disable augmentation; reproduces the pre-2026-09 "
                             "training procedure")
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "plateau"],
                        help="cosine anneals over --epochs; 'plateau' is the "
                             "previous ReduceLROnPlateau behaviour")

    # Empty by default: this must be an explicit choice. It previously defaulted
    # to checkpoints/segmentor_epoch_650.pth, so a run intended to start from
    # scratch silently continued that checkpoint instead, and its "from-scratch"
    # result would have been nothing of the kind.
    parser.add_argument("--resume", type=str, default="",
                        help="checkpoint to resume from; empty means train from "
                             "scratch")


    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every", type=int, default=1)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    device = pick_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_device(device)

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Torch: {torch.__version__}")
    if device.type == "cuda":
        print(f"[INFO] CUDA: {torch.version.cuda}")
        print(f"[INFO] GPU: {torch.cuda.get_device_name(device)}")

    # Dataset. The split comes from brats_gbm.splits, the single source of truth,
    # rather than a second random_split here that happened to agree with it.
    train_dataset, val_dataset, test_ids = get_brats_train_val(
        seed=args.seed, data_aug=args.data_aug)
    print(f"[INFO] Split: train={len(train_dataset)}, val={len(val_dataset)}, "
          f"test={len(test_ids)} (test is held out, never scored here)")
    print(f"[INFO] Augmentation: {'on' if args.data_aug else 'OFF'} | "
          f"train crop: random | val crop: centre (deterministic)")

    num_workers = auto_num_workers(args.num_workers)
    pin_memory = (device.type == "cuda")

    train_loader = DataLoader(
        train_dataset,
        batch_size=max(1, args.batch_size),
        shuffle=True,
        collate_fn=custom_collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=custom_collate,
        num_workers=max(0, num_workers // 2),
        pin_memory=pin_memory,
        persistent_workers=(max(0, num_workers // 2) > 0),
    )

    # Model/optim
    model = WaveletUNetPlusPlus(downsample=args.downsample).to(device)
    criterion = WeightedFocalDiceLoss(weight=args.weight_channels).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    # ReduceLROnPlateau drove the learning rate to its 1e-7 floor by epoch 76 in
    # the Table IX ablation and the remaining ~120 epochs changed nothing. Cosine
    # anneals over the whole budget instead, so late epochs still train.
    if args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, args.epochs - args.start_epoch), eta_min=1e-7
        )
        scheduler_desc = (f"CosineAnnealingLR(T_max={max(1, args.epochs - args.start_epoch)}, "
                          f"eta_min=1e-7)")
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-7
        )
        scheduler_desc = "ReduceLROnPlateau(mode=max, factor=0.5, patience=3, min_lr=1e-7)"

    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    start_epoch = args.start_epoch
    best_metric = -float("inf")

    # Resume - UNIVERSAL ADAPTER
    if args.resume and os.path.exists(args.resume):
        print(f"[INFO] Resuming from: {args.resume}")
        # weights_only=False: these checkpoints carry the run configuration
        # alongside the tensors, and torch >= 2.6 defaults weights_only=True,
        # which refuses to unpickle anything but tensors. The file is one this
        # script wrote, so the trust condition the flag guards is satisfied.
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        
        # Scenario 1: Standard format (has model_state, optim_state, etc.)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
            if "optim_state" in ckpt and ckpt["optim_state"] is not None:
                optimizer.load_state_dict(ckpt["optim_state"])
            if use_amp and scaler is not None and "scaler_state" in ckpt and ckpt["scaler_state"] is not None:
                scaler.load_state_dict(ckpt["scaler_state"])
            if "epoch" in ckpt:
                start_epoch = int(ckpt["epoch"])
            if "val_metric" in ckpt and ckpt["val_metric"] is not None:
                best_metric = float(ckpt["val_metric"])

        # Scenario 2: Alternative common dictionary format
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])
            print("[WARN] Loaded weights from 'state_dict'. Optimizer memory starting fresh.")
            start_epoch = 650
            
        # Scenario 3: The file is JUST the raw model weights
        else:
            model.load_state_dict(ckpt)
            print("[WARN] Loaded raw model weights directly. Optimizer memory starting fresh.")
            start_epoch = 650
            
        # A resume from <tag>_last.pth carries that epoch's score, which is not
        # the best score seen. Left as the floor, the first epoch to beat *last*
        # would overwrite <tag>_best.pth with a model worse than the one already
        # in it, silently destroying the best result of the run. Raise the floor
        # to whatever the existing best checkpoint actually holds.
        best_path = os.path.join(args.save_dir, f"{args.tag}_best.pth")
        if os.path.exists(best_path) and os.path.abspath(best_path) != os.path.abspath(args.resume):
            try:
                prev = torch.load(best_path, map_location="cpu", weights_only=False)
                prev_metric = prev.get("val_metric")
                if prev_metric is not None and float(prev_metric) > best_metric:
                    print(f"[INFO] {args.tag}_best.pth holds {float(prev_metric):.4f} "
                          f"(epoch {prev.get('epoch')}), better than the resumed "
                          f"{best_metric:.4f}; keeping it as the floor.")
                    best_metric = float(prev_metric)
            except Exception as e:  # a corrupt best must not abort a resume
                print(f"[WARN] could not read {best_path}: {e}")

        print(f"[INFO] start_epoch={start_epoch}, best_metric={best_metric:.4f}")

    # Provenance. The released segmentor_epoch_650.pth is a bare state_dict with
    # no record of how it was produced, which left its training protocol
    # unverifiable after the fact. Every checkpoint written from here carries the
    # configuration that produced it so that can never recur.
    run_config = {
        "downsample": args.downsample,
        "tag": args.tag,
        "seed": args.seed,
        "optimizer": "AdamW",
        "lr": args.lr,
        "scheduler": scheduler_desc,
        "loss": "WeightedFocalDiceLoss(gamma=2.0, alpha_bce=1.0, alpha_dice=1.0)",
        "weight_channels": list(args.weight_channels),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "accum_steps": args.accum_steps,
        "effective_batch": max(1, args.batch_size) * max(1, args.accum_steps),
        "clip_norm": args.clip_norm,
        "amp": bool(use_amp),
        "patch": [128, 128, 128],
        "normalisation": "minmax",
        "augmentation": (
            "flips(z,y,x p=0.5) + rot90(in-plane) + per-channel intensity "
            "scale/shift(p=0.5) + gamma(p=0.3) + gaussian noise(p=0.2)"
            if args.data_aug else "none"
        ),
        "train_crop": "random",
        "val_crop": "centre (deterministic)",
        "split": "brats_split_3way(seed=%d): train/val/test, test held out" % args.seed,
        "resume_from": args.resume or "scratch",
        "n_train": len(train_dataset),
        "n_val": len(val_dataset),
        "resumed_from": args.resume or None,
        # str(): torch.__version__ is a TorchVersion, not a plain str, and
        # pickling it puts a non-tensor global in the checkpoint that a
        # weights_only load then rejects.
        "torch": str(torch.__version__),
        "started": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    print("[INFO] run config:\n" + json.dumps(run_config, indent=2))

    # Train
    accum_steps = max(1, args.accum_steps)
    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0

        for bidx, batch in enumerate(train_loader):
            try:
                images = batch["image"].to(device, non_blocking=pin_memory)
                labels = batch["label"].to(device, non_blocking=pin_memory).float()
            except Exception as e:
                print(f"[WARN] Batch move failed at bidx={bidx}: {e}")
                continue

            try:
                if use_amp:
                    with torch.amp.autocast("cuda"):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                loss = loss / accum_steps

                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                running_loss += float(loss.item()) * accum_steps

                if (bidx + 1) % accum_steps == 0:
                    if use_amp:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_norm)

                    if use_amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                if bidx % 10 == 0:
                    avg_loss = running_loss / max(1, bidx + 1)
                    lr = optimizer.param_groups[0]["lr"]
                    print(f"[Epoch {epoch}/{args.epochs}] Batch {bidx}/{len(train_loader)}  loss={avg_loss:.4f} lr={lr:.2e}")

            except Exception as e:
                print(f"[ERROR] Train step failed at epoch={epoch}, batch={bidx}: {e}")
                traceback.print_exc()
                optimizer.zero_grad(set_to_none=True)
                continue

        # final optimizer step if remainder exists
        if len(train_loader) % accum_steps != 0:
            try:
                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_norm)

                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            except Exception as e:
                print(f"[WARN] Final remainder step failed: {e}")
                optimizer.zero_grad(set_to_none=True)

        # Eval
        if epoch % max(1, args.eval_every) == 0:
            val_metric = evaluate(model, val_loader, device)
        else:
            val_metric = None

        # ReduceLROnPlateau steps on the validation metric; cosine steps on the
        # epoch and must be called every epoch, including ones without an eval.
        if args.scheduler == "cosine":
            scheduler.step()
        elif val_metric is not None:
            scheduler.step(val_metric)

        # Persist last, and best-so-far. Writing one checkpoint per epoch costs
        # ~125 MB each (weights + AdamW state), i.e. ~94 GB over a 750-epoch run,
        # which does not fit on this machine alongside a second training arm.
        # Resume needs `last`; evaluation needs `best`; nothing needs the rest.
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "val_metric": val_metric,
            "best_metric": best_metric,
            "config": run_config,
        }

        last_path = os.path.join(args.save_dir, f"{args.tag}_last.pth")
        torch.save(ckpt, last_path)

        if val_metric is not None and val_metric > best_metric:
            best_metric = val_metric
            ckpt["best_metric"] = best_metric
            best_path = os.path.join(args.save_dir, f"{args.tag}_best.pth")
            torch.save(ckpt, best_path)
            print(f"[INFO] New best mean Dice {best_metric:.4f} -> {best_path}")

    print("[INFO] Training complete.")


if __name__ == "__main__":
    main()