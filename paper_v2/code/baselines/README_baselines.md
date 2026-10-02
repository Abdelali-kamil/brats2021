# Baselines — reproduce the comparison table on your server

Everything needed to download, run, and **fairly** score the comparison methods
against this project's Wavelet U-Net++, and collect the numbers into one table
for the paper.

> **This must run where the GPU and the real data are.** The methods below train
> deep networks on BraTS2021 / UPenn-GBM. They cannot run in a checkout without
> the imaging data and a CUDA GPU. Do all of this on your server. The only piece
> that runs anywhere is XGBoost (CPU, from the committed feature tables), which is
> already done — see `results/baselines/xgboost_classification.json`.

## The fairness principle (why this is publishable)

Two rules make the table a fair comparison rather than a table of numbers copied
from other papers:

1. **Same data, same split.** Every method trains and is evaluated on the *same*
   BraTS2021 cases and the *same* held-out split this project uses
   (`brats_gbm/splits.py`). Numbers from the baselines' own papers were produced
   on their own datasets and are **not** comparable — do not paste them in.
2. **Same scorer for everyone.** Every segmentation method's raw predictions —
   baselines *and* Wavelet U-Net++ — are scored by the identical code,
   `baselines/common/score_segmentation.py`, which calls this project's own
   `brats_gbm.eval.metrics.score_case` (ET/TC/WT Dice + HD95). No method is
   scored by its own repo's metric. Likewise every classifier goes through
   `brats_gbm.classification.cross_validate` (same folds, threshold, CIs).

This is also how your own method is held to the same standard: you score its
predictions with the same `score_segmentation.py --key ours`.

## Workflow

```bash
# 1. Download all official code at pinned commits (into baselines/repos/, gitignored)
bash baselines/fetch.sh

# 2. For each method, follow its recipe to train + predict on your split
#    (each writes predictions your server can then score)
less baselines/nnunet/SETUP.md        # and swin_unetr, selfmedmae, mtanet, mmgl, daft

# 3. Score every method identically, writing results/baselines/<key>_segmentation.json
python baselines/common/score_segmentation.py --key nnunet \
       --pred <nnunet_pred_dir> --gt <gt_seg_dir>
#    (nnU-Net relabels ET to 3: add  --et-label 3  for its predictions)

# 4. Also score YOUR method through the same scorer, for a like-for-like row
python baselines/common/score_segmentation.py --key ours \
       --pred <your_pred_dir> --gt <gt_seg_dir>

# 5. XGBoost classification is already produced; run any classifier baselines
#    so they write results/baselines/<key>_classification.json (schema below)

# 6. Assemble the single comparison table
python baselines/common/make_comparison_table.py
#    -> results/baselines/comparison.md  and  comparison.csv
```

## Method roster

| key | method | modality | task(s) | official code |
|---|---|:--:|---|---|
| `xgboost` | XGBoost | T | classification | github.com/dmlc/xgboost (pip) |
| `mmgl` | MMGL | I+T | classification | github.com/SsGood/MMGL |
| `daft` | DAFT | I+T | classification | github.com/ai-med/DAFT |
| `nnunet` | nnU-Net | I | segmentation | github.com/MIC-DKFZ/nnUNet |
| `selfmedmae` | SelfMedMAE | I+T | segmentation (+cls) | github.com/cvlab-stonybrook/SelfMedMAE |
| `swin_unetr` | Swin UNETR | I | segmentation | github.com/Project-MONAI/research-contributions |
| `mtanet` | MTANet | I+T | segmentation + classification | github.com/yatingling/MTANet |
| `ours` | Wavelet U-Net++ | I | segmentation | this repo |

`daft` replaces **MMCL** and `swin_unetr` replaces **ResGANet** — both originals
lack official code, so they cannot be reproduced under rule 1. See
`docs/RELATED_WORK.md`. If you must keep MMCL/ResGANet in the paper, they can
only be cited from their own papers, clearly marked as *not* reproduced on this
cohort.

## Result-file schemas (what the table reads)

Each method writes one small JSON per task under `results/baselines/`.

**Segmentation** — produced by `score_segmentation.py` (don't hand-write):
```json
{"key":"nnunet","n_cases":251,"dice_mean":0.91,"dice_sd":0.02,
 "hd95_mean":3.1,"hd95_sd":1.4,"per_region":{...}}
```

**Classification** — flat schema (means are fractions in [0,1]):
```json
{"recall_mean":0.62,"recall_sd":0.16,"precision_mean":0.80,
 "precision_sd":0.17,"auc_mean":0.61,"auc_sd":0.06}
```
Recall = sensitivity, Precision = PPV. Classification cohort is BraTS **MGMT**
(the balanced task; see the caveat below). `xgboost` is read from its own richer
`xgboost_classification.json` automatically.

## Honesty caveat you should carry into the paper

On this project's committed radiomic features, under its repeated-CV protocol,
XGBoost reaches **AUC ≈ 0.57 on MGMT — near chance**, consistent with
`docs/RESUME.md` (MGMT fusion ~0.63, imaging-only ~0.54) and the
`train_classifier_mgmt.py` docstring. The 0.80–0.89 classification AUCs some
baseline papers report were obtained on their own data. When you run the
classifier baselines here, expect honest numbers in the same near-chance range,
not the papers' figures. Report what the runs give.

## Effort expectation

This is real GPU work, not a push-button run: each repo has its own data format,
its own torch/CUDA version (use a **separate environment per method**), and needs
adapting to this project's split and BraTS label convention. Budget days of GPU
time and per-method debugging. The payoff is a table every number of which came
from a real, identically-scored run.
