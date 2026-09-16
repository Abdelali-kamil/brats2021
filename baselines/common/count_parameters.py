"""Parameter counts for every comparison method, measured from its own code.

Like the segmentation scorer, this measures each method the same way: the model
is instantiated from the method's official repository at the pinned commit, with
that repo's own default configuration for the BraTS task (4 input modalities,
3 output regions), and its parameters are counted by the same function.

Numbers are therefore comparable across rows, and none is copied from a paper.

  python baselines/common/count_parameters.py
  -> results/baselines/parameter_counts.json

nnU-Net is absent by necessity: it self-configures its architecture during
`nnUNetv2_plan_and_preprocess`, so it has no parameter count until planning has
run on this dataset. Count it afterwards from the generated plan.
"""
from __future__ import annotations

import json
import pathlib
import sys
import traceback

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
REPOS = ROOT / "baselines" / "repos"
OUT = ROOT / "results" / "baselines" / "parameter_counts.json"

# BraTS task shape, identical for every segmentation model below.
IN_CH, OUT_CH = 4, 3


def count(model: torch.nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"params_total": total, "params_trainable": trainable,
            "params_total_M": round(total / 1e6, 2)}


def _shim_timm_layers() -> None:
    """Map the pre-0.9 `timm.models.layers.*` paths onto modern timm.

    SelfMedMAE is pinned to a 2024 commit that imports `timm.models.layers.
    helpers`, a path timm 1.x moved to `timm.layers`. Aliasing the module in
    sys.modules lets their code import unmodified -- we adapt the environment
    around their source rather than editing it, so the model that gets counted
    is still theirs.
    """
    import importlib
    import types
    for old_path, new_path in (
        ("timm.models.layers", "timm.layers"),
        ("timm.models.layers.helpers", "timm.layers.helpers"),
    ):
        if old_path in sys.modules:
            continue
        try:
            sys.modules[old_path] = importlib.import_module(new_path)
        except ModuleNotFoundError:
            sys.modules[old_path] = types.ModuleType(old_path)


# --- one builder per method, each using that repo's own defaults -------------

def build_ours():
    sys.path.insert(0, str(ROOT))
    from brats_gbm.model import WaveletUNetPlusPlus
    return WaveletUNetPlusPlus(in_channels=IN_CH, n_classes=OUT_CH), {
        "config": "WaveletUNetPlusPlus(in_channels=4, n_classes=3)",
        "source": "this repo",
    }


def build_swin_unetr():
    # The BRATS21 recipe's own defaults: feature_size 48, 96^3 ROI.
    from monai.networks.nets import SwinUNETR
    try:
        m = SwinUNETR(img_size=(96, 96, 96), in_channels=IN_CH,
                      out_channels=OUT_CH, feature_size=48)
    except TypeError:
        # MONAI >= 1.5 dropped img_size from the signature.
        m = SwinUNETR(in_channels=IN_CH, out_channels=OUT_CH, feature_size=48)
    return m, {
        "config": "SwinUNETR(feature_size=48, roi=96^3) — BRATS21/main.py defaults",
        "source": "monai.networks.nets (the BRATS21 recipe's model)",
    }


def build_selfmedmae():
    """Built exactly as lib/trainers/seg_trainer.py builds it, from their own
    BraTS config `configs/unetr_msdbrats_1gpu.yaml` (roi 128^3, patch 16,
    in_chans 4, ViT-base encoder + UNETR decoder).

    `num_classes` is the one value their config leaves to the dataset; it is set
    to 3 here so the output matches this task's three tumour regions.
    """
    import types

    import yaml

    repo = REPOS / "selfmedmae"
    # Repo root resolves `lib.models.*`; `lib/` itself resolves the bare
    # `import networks` inside lib/models/mae.py. Both are needed.
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "lib"))
    _shim_timm_layers()

    cfg = yaml.safe_load((repo / "configs" / "unetr_msdbrats_1gpu.yaml").read_text())
    cfg["num_classes"] = OUT_CH
    args = types.SimpleNamespace(**cfg)

    import timm
    if int(timm.__version__.split(".")[0]) >= 1:
        raise RuntimeError(
            f"needs timm <0.9 (its pinned 2024 code calls "
            f"vision_transformer.Block(drop=...), renamed to proj_drop in timm 1.x); "
            f"this environment has timm {timm.__version__}. Build the method's own "
            f"venv per baselines/selfmedmae/SETUP.md and count it there."
        )

    import networks
    from lib.models.unetr3d import UNETR3D
    model = UNETR3D(encoder=getattr(networks, args.enc_arch),
                    decoder=getattr(networks, args.dec_arch), args=args)
    return model, {
        "config": (f"UNETR3D({args.enc_arch}+{args.dec_arch}, embed {args.encoder_embed_dim}, "
                   f"depth {args.encoder_depth}, patch {args.patch_size}, roi {args.roi_x}^3)"),
        "source": "selfmedmae configs/unetr_msdbrats_1gpu.yaml — their own BraTS config",
    }


def build_mtanet():
    """MTANet cannot be instantiated for this task. Documented, not worked around.

    Two independent blockers, both verified in the pinned clone:

    1. It is a **2D** network. MATNet.py and pvtv2.py contain 39 and 8 `Conv2d`
       layers respectively and **zero** `Conv3d`. BraTS is 3D volumetric data,
       so the published model does not accept this task's input at all.
    2. `MTANet.__init__` unconditionally runs
       `torch.load('lib/pvt_v2_b2.pth')` -- a pretrained PVT-v2 backbone the
       repository does not ship and does not provide a download for.

    Forcing a number here would mean building a different, 3D model and calling
    it MTANet. The honest output is this explanation.
    """
    raise RuntimeError(
        "not applicable to 3D BraTS: 2D-only architecture (39 Conv2d / 0 Conv3d) "
        "and __init__ requires the unshipped pretrained backbone lib/pvt_v2_b2.pth"
    )


BUILDERS = {
    "ours": build_ours,
    "swin_unetr": build_swin_unetr,
    "selfmedmae": build_selfmedmae,
    "mtanet": build_mtanet,
}


def main() -> None:
    results, failures = {}, {}
    for key, builder in BUILDERS.items():
        try:
            model, meta = builder()
            model.eval()
            results[key] = {**count(model), **meta}
            print(f"  {key:<12} {results[key]['params_total_M']:>8.2f} M   "
                  f"{meta['config']}")
        except Exception as exc:  # noqa: BLE001 - report, never abort the sweep
            failures[key] = f"{type(exc).__name__}: {exc}"
            print(f"  {key:<12} {'FAILED':>8}   {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=1)

    results["nnunet"] = {
        "params_total": None,
        "note": ("Self-configuring: architecture is chosen by "
                 "nnUNetv2_plan_and_preprocess for this dataset. Count it from "
                 "the generated plan once preprocessing has run."),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(
        {"task": {"in_channels": IN_CH, "out_channels": OUT_CH},
         "results": results, "failures": failures}, indent=2))
    print(f"\nwrote {OUT}")
    if failures:
        print(f"{len(failures)} method(s) could not be instantiated; see the JSON.")


if __name__ == "__main__":
    main()
