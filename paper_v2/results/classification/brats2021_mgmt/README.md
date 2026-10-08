# BraTS 2021 MGMT promoter methylation (577 cases with labels)

Quoted in the paper (Section 5.5, last sentences): MGMT status is close to chance from region features.

- `mgmt_radiomic_results.json` — radiomic features, logistic regression and random forest, repeated
  stratified CV; logistic regression AUC 0.583 [0.538, 0.628] (patient bootstrap). Script:
  `code/scripts/train_classifier_mgmt.py`.
- `kan_transformer_gnn_results.json` — the KAN / region-Transformer / graph classifier ladder under nested
  CV (`code/brats_gbm/kan.py`, `gnn.py`); full model AUC 0.624 [0.575, 0.670], plain MLP 0.617.
  Only the BraTS 2021 parts of the original result file are kept (the paper does not use UPenn-GBM).
