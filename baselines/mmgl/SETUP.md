# MMGL (classification baseline, multi-modal graph learning)

*Multi-Modal Graph Learning for Disease Prediction*, IEEE TMI 2022 ·
https://github.com/SsGood/MMGL
Modality **I+T**. A population-graph disease-prediction model. Runs on a modest
GPU (or CPU for the small graph). See `brats_gbm/gnn.py` for why this is a
*transductive* baseline distinct from this project's inductive classifier.

## 1. Environment (isolated)

```bash
python -m venv baselines/mmgl/.venv && source baselines/mmgl/.venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r baselines/repos/mmgl/requirements.txt   # torch-geometric etc. per the pinned repo
```

## 2. Data

MMGL expects a feature matrix (nodes = patients) plus modality groups. Use the
committed feature tables so it sees the same inputs as the other classifiers:

- imaging features: `results/classification/brats_mgmt_features.csv`
- (optional) clinical/tabular: `metadata/` CSVs, keyed by case id
- labels: `mgmt_label`

Format these into MMGL's expected input (see its `data/` loader at the pinned
commit). **Fairness:** use the same repeated stratified 5-fold CV as the other
classifiers; do not let graph construction peek across the fold boundary.

## 3. Run + emit the result file

```bash
cd baselines/repos/mmgl
python train.py --dataset <your_mgmt_config> ...     # exact flags per the repo
```

Collect per-fold held-out probabilities and write
`results/baselines/mmgl_classification.json` in the flat schema
(`recall_mean/recall_sd/precision_mean/precision_sd/auc_mean/auc_sd`, fractions
in [0,1]; Recall=sensitivity, Precision=PPV). Expect a near-chance MGMT AUC on
this cohort (see the caveat in `baselines/README.md`).
