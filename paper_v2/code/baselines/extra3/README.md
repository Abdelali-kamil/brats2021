# SegResNet, UNETR and 3D U-Net on the identical split (Amendments 19/19b)

`BRATS21/` is the official MONAI BraTS 2021 training pipeline used for the Swin UNETR baseline
(github.com/Project-MONAI/research-contributions, `SwinUNETR/BRATS21` at commit 21ed8e5;
Apache License 2.0, copyright MONAI Consortium — the original headers are kept). Changes:

- `models_extra3.py` (new): builds `segresnet` (MONAI SegResNet, BraTS tutorial configuration),
  `unetr` (MONAI UNETR, official configuration), `unet3d` (pytorch-3dunet UNet3D) or `swinunetr`.
- `main.py` (modified): `--model_name` selects the network; `AsDiscrete(threshold=0.5)` replaces the
  `logit_thresh` argument that MONAI 1.6 ignores (same fix as for Swin UNETR).
- `trainer.py` (modified): label masks are cast to float before the loss (MONAI 1.6 squared Dice).

Everything else (data loading, augmentation, AdamW 1e-4, warm-up cosine, 300 epochs, validation every
25 epochs, best-by-validation checkpoint) is unchanged. SegResNet uses its tutorial's 224x224x144 crops
and squared Dice loss; 3D U-Net and UNETR use 96^3 crops as Swin UNETR.

Run order: `queue_extra3.sh` (first run; the 3D U-Net run collapsed under mixed precision at epoch 24),
`queue_extra3_v2.sh` (adds the Amendment-19b collapse rule: a run whose validation Dice is < 0.05 for all
three regions is retrained once in fp32) and `queue_extra3_v3.sh` (the same, one instance per network so
the three trained in parallel; the UNETR mixed-precision run also collapsed and was retrained in fp32).
`infer_test125_extra3.py` scores the 125 test cases once (overlap 0.6, sigmoid > 0.5, no TTA).
