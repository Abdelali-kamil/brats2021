# nnU-Net (segmentation baseline)

Isensee et al., *nnU-Net*, Nature Methods 2021 · https://github.com/MIC-DKFZ/nnUNet
Modality **I**. The standard strong segmentation baseline. Needs a CUDA GPU.

## 1. Environment (isolated)

```bash
python -m venv baselines/nnunet/.venv && source baselines/nnunet/.venv/bin/activate
# NOTE: cu121 does NOT support this machine's RTX 5090 (Blackwell, sm_120).
# Let pip resolve a CUDA 13 build, which is what the driver here provides.
pip install torch
pip install -e baselines/repos/nnunet          # the pinned clone from fetch.sh
source baselines/nnunet/env.sh   # paths on /mnt/data1, which has the headroom
```

## 2. Convert BraTS2021 to nnU-Net format

nnU-Net ships a BraTS converter. From the pinned clone:

nnU-Net's own `Dataset137_BraTS21.py` takes **no arguments** (it hardcodes its
author's data path) and copies **every** case into `imagesTr` — which would put
our 251 held-out cases into its training set. Use the wrapper instead, which
calls nnU-Net's own label conversion and dataset-json generation but respects
`baselines/common/brats_split_frozen.json`:

```bash
source baselines/nnunet/env.sh          # sets the three nnUNet_* paths
python baselines/nnunet/convert_split.py --workers 4
```

It writes the 1000 training cases to `imagesTr`/`labelsTr` and the 251 held-out
cases, images only, to `imagesTs`. nnU-Net therefore cannot train or select on
a held-out case, and both scoring options (251-case internal validation or the
125-case clean test) stay open without retraining.

This writes `Dataset137_BraTS2021` with 4 channels (`_0000..0003` =
FLAIR,T1,T1ce,T2) and **relabels ET from 4 to 3** (necrotic=1, edema=2, ET=3).
Remember the `--et-label 3` flag at scoring time.

**Fairness:** train on the same cases this project trains on and hold out the
same internal-validation cases. Build the fold list from `brats_gbm/splits.py`
(`brats_split(...)`) and give nnU-Net a custom split so its held-out set equals
this project's 251-case internal-validation partition — otherwise the comparison
is not like-for-like.

## 3. Train + predict

```bash
nnUNetv2_plan_and_preprocess -d 137 --verify_dataset_integrity -np 4
nnUNetv2_train 137 3d_fullres 0        # (repeat folds 1-4 for the ensemble, optional)
nnUNetv2_predict -i <held_out_images_dir> -o baselines/nnunet/pred \
                 -d 137 -c 3d_fullres -f 0
```

## 4. Score identically and emit the result file

```bash
python baselines/common/score_segmentation.py --key nnunet \
       --pred baselines/nnunet/pred --gt <gt_seg_dir> --et-label 3
#   -> results/baselines/nnunet_segmentation.json
```

`--gt` is the ground-truth `*_seg.nii.gz` for the held-out cases (BraTS raw
labels 1/2/4, so keep `--gt-et-label 4` if your GT is un-relabelled; pass it
explicitly if in doubt).
