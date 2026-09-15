# Swin UNETR (segmentation baseline — replaces ResGANet)

Hatamizadeh et al., *Swin UNETR: Swin Transformers for Semantic Segmentation of
Brain Tumors in MRI Images*, MICCAI BrainLes 2021 ·
https://github.com/Project-MONAI/research-contributions (SwinUNETR/BRATS21).
Modality **I**. A transformer segmentation network designed and validated on
BraTS. Needs a CUDA GPU.

## 1. Environment (isolated)

```bash
python -m venv baselines/swin_unetr/.venv && source baselines/swin_unetr/.venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install "monai[all]" nibabel einops tensorboard
```

The official training code is `baselines/repos/swin_unetr/SwinUNETR/BRATS21/`
(from the pinned clone). Swin UNETR is also in MONAI core
(`monai.networks.nets.SwinUNETR`), so you can either use the repo's `main.py` or
MONAI's BraTS tutorial harness.

## 2. Data

Point the repo's dataloader at your `data/BraTS2021_XXXXX/` root and its JSON
datalist. **Fairness:** edit the datalist so the validation fold is exactly this
project's 251-case internal-validation split (from `brats_gbm/splits.py`), not
the repo's default random fold.

## 3. Train + predict

```bash
cd baselines/repos/swin_unetr/SwinUNETR/BRATS21
python main.py --json_list <your_datalist.json> --data_dir /path/to/data \
       --feature_size 48 --use_checkpoint --save_checkpoint --logdir ours
# then run its test/inference script to write predicted segmentations
```

Swin UNETR predicts the three overlapping regions (ET/TC/WT) directly. Save each
case's prediction as a **label map** (necrotic=1, edema=2, ET=4) so the shared
scorer reads it; if the repo writes region-channel masks, collapse them to a
label map first (WT→2, TC→1, ET→4, applied inward).

## 4. Score identically

```bash
python baselines/common/score_segmentation.py --key swin_unetr \
       --pred <pred_dir> --gt <gt_seg_dir>
#   -> results/baselines/swin_unetr_segmentation.json
```
