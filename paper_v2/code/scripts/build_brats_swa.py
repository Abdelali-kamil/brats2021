#!/usr/bin/env python3
"""Average the weights of the final BraTS checkpoints (SWA).

Averaging the weights of several checkpoints from the end of a training run is
a standard, essentially free way to gain a fraction of a Dice point. Late in
training with a decayed learning rate the iterates wander around one basin, and
their average sits nearer its centre than any individual iterate does, which
generalises slightly better.

Why this is not test-set tuning
-------------------------------
The rule is fixed in advance and uses no data at all: take the last K
checkpoints by epoch number and average them uniformly. No held-out set is
consulted to pick K, to pick which epochs, or to decide whether to use the
result. That matters here because BraTS has no untouched selection partition —
the 251-case set doubled as the model-selection set during training — so any
data-driven choice would compound an existing weakness. A data-independent rule
does not.

Both the averaged model and the original epoch-650 checkpoint are evaluated and
reported side by side. Neither is selected over the other on the reported set.

Status: NOT USED in the reported results
----------------------------------------
This script is retained as a guarded tool, but SWA is deliberately not part of
the final protocol, for two reasons found by running it.

1. WaveletUNetPlusPlus uses `nn.BatchNorm3d`, which carries `running_mean` and
   `running_var`. Averaging weights across checkpoints leaves those statistics
   inconsistent with the averaged weights, so plain averaging is invalid; a
   correct SWA would need a recalibration pass over training data
   (`torch.optim.swa_utils.update_bn`). The guard below refuses to write a
   checkpoint rather than produce a silently wrong one.

2. The a-priori window does not land in a good region. The recorded validation
   metric across the 100 archived checkpoints that carry one peaks at epoch 700
   (0.8670), while the last five epochs sit at 0.834-0.861. Moving the window
   to a better region would mean choosing epochs by their score on the same
   251-case partition that is being reported, which is precisely the selection
   bias this project is documenting rather than adding to.

Averaging a window chosen for being late is defensible; averaging one chosen
for scoring well on the reported set is not. Since the honest window is not the
useful one, no SWA result is reported.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brats_gbm.model import WaveletUNetPlusPlus  # noqa: E402

ARCHIVE = Path("/home/kamilabdelali/brats2021_archive_20260723/checkpoints_all_epochs")
OUT = ROOT / "checkpoints" / "segmentor_swa_last5.pth"
LAST_K = 5

# Buffers that carry dataset statistics and would be invalid after averaging.
STATEFUL_BUFFERS = ("running_mean", "running_var", "num_batches_tracked")


# Checkpoints from different eras of this project use different key names.
STATE_KEYS = ("model_state_dict", "model_state", "state_dict")


def state_dict_of(ckpt) -> dict:
    if not isinstance(ckpt, dict):
        return ckpt
    for k in STATE_KEYS:
        if k in ckpt and isinstance(ckpt[k], dict):
            return ckpt[k]
    return ckpt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=str(ARCHIVE))
    ap.add_argument("--last-k", type=int, default=LAST_K)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    archive = Path(args.archive)
    if not archive.exists():
        raise SystemExit(f"checkpoint archive not found: {archive}")

    epochs = []
    for f in archive.glob("segmentor_epoch_*.pth"):
        m = re.search(r"segmentor_epoch_(\d+)\.pth", f.name)
        if m:
            epochs.append((int(m.group(1)), f))
    if len(epochs) < args.last_k:
        raise SystemExit(f"need >= {args.last_k} checkpoints, found {len(epochs)}")

    epochs.sort()
    chosen = epochs[-args.last_k:]
    print(f"archive     : {archive}")
    print(f"available   : {len(epochs)} checkpoints, epochs "
          f"{epochs[0][0]}..{epochs[-1][0]}")
    print(f"rule        : uniform average of the last {args.last_k} by epoch number "
          f"(fixed a priori, no data consulted)")
    print(f"selected    : {[e for e, _ in chosen]}\n")

    model = WaveletUNetPlusPlus(in_channels=4, n_classes=3)
    reference = set(model.state_dict())

    avg: dict[str, torch.Tensor] = {}
    for i, (epoch, path) in enumerate(chosen, 1):
        sd = state_dict_of(torch.load(str(path), map_location="cpu", weights_only=False))
        stateful = [k for k in sd if any(b in k for b in STATEFUL_BUFFERS)]
        if stateful:
            raise SystemExit(
                f"{path.name} carries running statistics ({stateful[:3]}); weight "
                f"averaging would leave them inconsistent. Recalibrate before use.")
        if set(sd) != reference:
            raise SystemExit(f"{path.name} does not match the model's parameter set")

        for k, v in sd.items():
            t = v.detach().to(torch.float64)
            avg[k] = t if i == 1 else avg[k] + t
        print(f"  [{i}/{len(chosen)}] epoch {epoch}")

    for k in avg:
        avg[k] = (avg[k] / len(chosen)).to(torch.float32)

    model.load_state_dict(avg, strict=True)  # fails loudly if shapes drifted
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": avg,
        "swa_epochs": [e for e, _ in chosen],
        "swa_rule": f"uniform average of last {args.last_k} checkpoints by epoch",
    }, out)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
