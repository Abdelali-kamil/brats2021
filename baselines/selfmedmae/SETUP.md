# SelfMedMAE (segmentation baseline, self-supervised pre-training)

*Self Pre-training with Masked Autoencoders for Medical Image Classification and
Segmentation*, ISBI 2023 · https://github.com/cvlab-stonybrook/SelfMedMAE
Modality **I+T** in the paper table (here evaluated on segmentation; the
classification head is optional). Needs a CUDA GPU.

## 1. Environment (isolated)

```bash
python -m venv baselines/selfmedmae/.venv && source baselines/selfmedmae/.venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r baselines/repos/selfmedmae/requirements.txt   # or the repo's listed deps (monai, timm, ...)
```

## 2. Two stages

SelfMedMAE is (a) MAE self-pre-training on your unlabelled BraTS volumes, then
(b) fine-tuning a UNETR-style decoder for segmentation.

```bash
cd baselines/repos/selfmedmae
# (a) pre-train the MAE on BraTS (see the repo's pretrain config/script)
python main_pretrain.py  --data_dir /path/to/data  ...        # exact flags per the pinned repo
# (b) fine-tune for segmentation, initialised from the MAE encoder
python main_finetune.py  --data_dir /path/to/data  --pretrained <mae_ckpt>  ...
```

**Fairness:** the fine-tune/validation split must equal this project's 251-case
internal-validation split (`brats_gbm/splits.py`); pre-training may use the full
training pool but must not see held-out cases.

## 3. Predict + score identically

Run the repo's inference to write per-case predicted label maps, then:

```bash
python baselines/common/score_segmentation.py --key selfmedmae \
       --pred <pred_dir> --gt <gt_seg_dir>
#   -> results/baselines/selfmedmae_segmentation.json
```

If you also run its classification head, write
`results/baselines/selfmedmae_classification.json` in the flat schema
(`recall_mean/… /auc_mean`, see `baselines/README.md`) using the MGMT cohort.
