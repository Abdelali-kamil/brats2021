# nnU-Net (segmentation baseline)

Isensee et al., *nnU-Net*, Nature Methods 2021 · https://github.com/MIC-DKFZ/nnUNet
Modality **I**. The standard strong segmentation baseline. Needs a CUDA GPU.

## 1. Environment (isolated)

```bash
python -m venv baselines/nnunet/.venv && source baselines/nnunet/.venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -e baselines/repos/nnunet          # the pinned clone from fetch.sh
export nnUNet_raw=$PWD/baselines/nnunet/nnUNet_raw
export nnUNet_preprocessed=$PWD/baselines/nnunet/nnUNet_preprocessed
export nnUNet_results=$PWD/baselines/nnunet/nnUNet_results
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"
```

## 2. Convert BraTS2021 to nnU-Net format

nnU-Net ships a BraTS converter. From the pinned clone:

```bash
python baselines/repos/nnunet/nnunetv2/dataset_conversion/Dataset137_BraTS21.py \
       -i /path/to/data                      # your data/BraTS2021_XXXXX/ root
```

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
nnUNetv2_plan_and_preprocess -d 137 --verify_dataset_integrity
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
