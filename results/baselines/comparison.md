# Baseline comparison — segmentation and classification

One table, real reproduced numbers only. `pending` = no result file yet;
produce it by running that baseline on your server (see `baselines/<method>/SETUP.md`), which writes the result file this script reads.

Metrics: Dice ↑ and Recall/Precision/AUC ↑ as mean±SD in %, HD95 ↓ in mm.
Recall = sensitivity, Precision = PPV. Classification cohort is BraTS **MGMT**.

| Method | Modal | Dice ↑ | HD95 ↓ | Recall ↑ | Precision ↑ | AUC ↑ | Status |
|---|:--:|:--:|:--:|:--:|:--:|:--:|---|
| **_Single task_** | | | | | | | |
| XGBoost | T | – | – | 35.9±3.3 | 59.5±3.3 | 57.4±1.9 | reproduced (MGMT, this data) |
| MMGL | I+T | – | – | pending | pending | pending | pending (run on server) |
| DAFT ¹ | I+T | – | – | pending | pending | pending | pending (run on server) |
| nnU-Net | I | pending | pending | – | – | – | pending (run on server) |
| **_Multitasking_** | | | | | | | |
| SelfMedMAE | I+T | pending | pending | pending | pending | pending | pending (run on server) |
| Swin UNETR ¹ | I | pending | pending | – | – | – | pending (run on server) |
| MTANet | I+T | pending | pending | pending | pending | pending | pending (run on server) |
| **_Ours_** | | | | | | | |
| Wavelet U-Net++ (Ours) | I | 88.3±1.9 | 6.11±2.96 | – | – | – | reproduced (this project) |

¹ DAFT: replaces MMCL (no official code)
¹ Swin UNETR: replaces ResGANet (no official code)

> **Reproducibility caveat (MGMT).** On this project's committed radiomic
> features, under its own repeated-CV protocol, XGBoost reaches AUC ≈ 0.57 on
> MGMT — near chance, consistent with `docs/RESUME.md` and the
> `train_classifier_mgmt.py` docstring. Classification AUCs in the 0.80–0.89
> range reported by the source papers were obtained on *their own* datasets and
> are not reproduced here; do not present them as run on this cohort.
