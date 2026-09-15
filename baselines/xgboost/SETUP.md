# XGBoost (classification baseline) — DONE, runs anywhere

Chen & Guestrin, *XGBoost: A Scalable Tree Boosting System*, KDD 2016 ·
https://github.com/dmlc/xgboost

Modality **T** (tabular radiomic features). No GPU, no imaging data — runs from
the committed feature tables, so it is already produced.

```bash
pip install "xgboost==3.2.0"
python scripts/baseline_xgboost_classification.py --task both
#   -> results/baselines/xgboost_classification.json
```

- Uses the **same** protocol as the reported RF/logreg
  (`brats_gbm.classification.cross_validate`): repeated (10×) stratified 5-fold,
  cross-fitted threshold, patient-level bootstrap CIs. No grid search.
- `mgmt` is the table's cohort (577 balanced cases). `idh1` is reported for
  completeness (17 positives / 416; precision is structurally small there).
- Metric mapping: **Recall = sensitivity, Precision = PPV, AUC = auc**.

Reproduced result (this data): MGMT **AUC 0.574 [0.524, 0.620]** — near chance,
matching the repo's own MGMT findings. IDH1 primary AUC 0.908 [0.830, 0.963].
`make_comparison_table.py` reads this file into the classification row.
