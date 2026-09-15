# MTANet (segmentation + classification, multi-task)

*MTANet: Multi-Task Attention Network for Automatic Medical Image Segmentation
and Classification*, IEEE TMI 2024 · https://github.com/yatingling/MTANet
Modality **I+T**. The closest published analogue to a joint seg+cls network.
Needs a CUDA GPU.

## 1. Environment (isolated)

```bash
python -m venv baselines/mtanet/.venv && source baselines/mtanet/.venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
# install the repo's own requirements (check baselines/repos/mtanet for a
# requirements.txt / environment.yml at the pinned commit)
pip install -r baselines/repos/mtanet/requirements.txt 2>/dev/null || true
```

> The official MTANet code is less standardised than nnU-Net/MONAI and may be
> demo-oriented (2D or a specific dataset). Read `baselines/repos/mtanet` first;
> adapting its dataloader to 3D multi-modal BraTS and to this project's split is
> the main effort. If it cannot be made to run on BraTS honestly, record that and
> fall back to citing the paper (not reproduced) — do not fabricate a row.

## 2. Data + split

Feed the four BraTS modalities as the image input and (for the classification
head) the MGMT label. **Fairness:** validation split = this project's 251-case
internal-validation split (`brats_gbm/splits.py`).

## 3. Predict + score identically

Segmentation → predicted label maps → shared scorer:

```bash
python baselines/common/score_segmentation.py --key mtanet \
       --pred <pred_dir> --gt <gt_seg_dir>
#   -> results/baselines/mtanet_segmentation.json
```

Classification head → write `results/baselines/mtanet_classification.json`
(flat schema, MGMT cohort). To keep it identical to the other classifiers, feed
MTANet's held-out predicted probabilities through the same metric definitions
(Recall=sensitivity, Precision=PPV, AUC), or better, its features through
`brats_gbm.classification.cross_validate`.
