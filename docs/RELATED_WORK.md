# Comparison methods and related work

External methods this project cites or benchmarks against, with their source
paper and official code where one exists. Grouped by the role each plays in
this repository: a **tabular/clinical baseline** for the IDH1/MGMT
classification, **multimodal / graph disease-prediction** methods for the
fusion experiments, and **segmentation (or joint segmentation+classification)**
networks for the BraTS2021 / UPenn-GBM segmentor.

> **Verify before citing.** The table reproduces references collected for the
> thesis. Venues and code URLs below are believed correct, but author lists,
> years, page numbers and exact titles should be checked against the original
> publication before they go into a bibliography.

> **Two substitutions for reproducibility.** The original list included **MMCL**
> (an ambiguous acronym with no locatable code) and **ResGANet** (no official
> repository). Because the paper's comparison must be *reproduced on this
> project's own data* (see `baselines/README.md`), both are replaced by
> methods that have official code and are at least as comparable:
> **MMCL → DAFT** (3D image + tabular clinical fusion — exactly this project's
> setting) and **ResGANet → Swin UNETR** (a transformer segmentation network
> designed and validated on BraTS). If MMCL/ResGANet must remain in the paper,
> they can only be cited from their own papers and clearly marked as *not*
> reproduced on this cohort.

## Summary

| Method | Role here | Paper | Official code |
|---|---|---|---|
| **XGBoost** | Tabular/clinical baseline | Chen & Guestrin, *XGBoost: A Scalable Tree Boosting System*, KDD 2016 | [github.com/dmlc/xgboost](https://github.com/dmlc/xgboost) |
| **MMGL** | Multimodal graph disease prediction | *Multi-Modal Graph Learning for Disease Prediction*, IEEE TMI 2022 | [github.com/SsGood/MMGL](https://github.com/SsGood/MMGL) |
| **DAFT** *(replaces MMCL)* | 3D image + tabular clinical fusion | Pölsterl et al., *Combining 3D Image and Tabular Data via the Dynamic Affine Feature Map Transform*, MICCAI 2021 | [github.com/ai-med/DAFT](https://github.com/ai-med/DAFT) |
| **nnU-Net** | Segmentation baseline | Isensee et al., *nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation*, Nature Methods 2021 | [github.com/MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet) |
| **SelfMedMAE** | Self-supervised pre-training (MAE) | *Self Pre-training with Masked Autoencoders for Medical Image Classification and Segmentation*, ISBI 2023 | [github.com/cvlab-stonybrook/SelfMedMAE](https://github.com/cvlab-stonybrook/SelfMedMAE) |
| **Swin UNETR** *(replaces ResGANet)* | Transformer segmentation (BraTS) | Hatamizadeh et al., *Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images*, MICCAI BrainLes 2021 | [github.com/Project-MONAI/research-contributions](https://github.com/Project-MONAI/research-contributions) |
| **MTANet** | Joint segmentation + classification | *MTANet: Multi-Task Attention Network for Automatic Medical Image Segmentation and Classification*, IEEE TMI 2024 | [github.com/yatingling/MTANet](https://github.com/yatingling/MTANet) |

## Notes per method

### XGBoost — tabular/clinical baseline
Tianqi Chen & Carlos Guestrin, *XGBoost: A Scalable Tree Boosting System*,
KDD 2016. Used here not as an imaging method but as a strong gradient-boosted-
trees baseline on the clinical/radiomic feature table, i.e. the comparison
point for the fusion classifier. Related in spirit to the random forest already
reported for IDH1 in `README.md`. Code: <https://github.com/dmlc/xgboost>.

### MMGL — Multi-Modal Graph Learning for Disease Prediction
IEEE Transactions on Medical Imaging (TMI), 2022. Learns a population graph
over patients and predicts a diagnosis by message passing across it. Already
cited in `brats_gbm/gnn.py`, where the docstring notes that MMGL is
**transductive** (it needs the whole cohort at inference) and therefore solves
a different problem from this project's *inductive*, retrieval-augmented
classifier — a patient arriving alone gets the same prediction as inside a
batch. Keep that distinction if MMGL appears in the related-work section.
Code: <https://github.com/SsGood/MMGL>.

### DAFT — 3D image + tabular clinical fusion (replaces MMCL)
Pölsterl, Wolf & Wachinger, *Combining 3D Image and Tabular Data via the Dynamic
Affine Feature Map Transform*, MICCAI 2021. DAFT modulates a 3D CNN's feature
maps with a patient's tabular clinical data — precisely the imaging + clinical
fusion this project performs for MGMT/IDH prediction, which makes it a more
directly comparable baseline than the ambiguous MMCL it replaces. Official
PyTorch code, runnable. Code: <https://github.com/ai-med/DAFT>. Setup/run recipe:
`baselines/daft/SETUP.md`.

(The original **MMCL** entry was dropped: the acronym is ambiguous — most likely
"Multi-Modal Contrastive Learning", used by several distinct papers — and no
single official repository could be identified to reproduce it on this cohort.)

### nnU-Net — segmentation baseline
Fabian Isensee, Paul F. Jaeger, Simon A. A. Kohl, Jens Petersen,
Klaus H. Maier-Hein, *nnU-Net: a self-configuring method for deep learning-
based biomedical image segmentation*, Nature Methods 2021. The standard
self-configuring segmentation baseline. Note `docs/RESUME.md`: **no nnU-Net
comparison has yet been run on this project's own 251-case split**, so any
number quoted for it must come from that split, not from the nnU-Net paper's
leaderboard figures. Code: <https://github.com/MIC-DKFZ/nnUNet>.

### SelfMedMAE — self-supervised pre-training
*Self Pre-training with Masked Autoencoders for Medical Image Classification
and Segmentation*, ISBI 2023 (the user's table spells it "SelfmedMAE"; the
repository is capitalised **SelfMedMAE**). Masked-autoencoder self-pre-training
on the target medical images before fine-tuning for classification or
segmentation — the comparison point for pre-training strategy against this
project's transfer-from-BraTS approach.
Code: <https://github.com/cvlab-stonybrook/SelfMedMAE>.

### Swin UNETR — transformer segmentation on BraTS (replaces ResGANet)
Hatamizadeh, Nath, Tang, Yang, Roth & Xu, *Swin UNETR: Swin Transformers for
Semantic Segmentation of Brain Tumors in MRI Images*, MICCAI BrainLes 2021. A
Swin-transformer encoder with a U-Net-style decoder, designed and validated on
BraTS brain-tumour segmentation, with official MONAI code (and available in MONAI
core as `monai.networks.nets.SwinUNETR`). Chosen to replace ResGANet, which has
no official repository and so cannot be reproduced on this project's data. Code:
<https://github.com/Project-MONAI/research-contributions> (SwinUNETR/BRATS21).
Setup/run recipe: `baselines/swin_unetr/SETUP.md`.

(The original **ResGANet** entry — *Residual Group Attention Network for medical
image classification and segmentation*, Medical Image Analysis 2022 — was
dropped for lack of any official code to reproduce.)

### MTANet — joint segmentation + classification
*MTANet: Multi-Task Attention Network for Automatic Medical Image Segmentation
and Classification*, IEEE TMI 2024. A multi-task attention network that does
segmentation and classification jointly — the closest published analogue to
this project's Innovation 4 (joint segmentation–classification optimisation,
noted in `docs/RESUME.md`). The user's table truncated the URL to
"MTANe"; the repository is **github.com/yatingling/MTANet**.
Code: <https://github.com/yatingling/MTANet>.

## Grouping for a related-work section

- **Segmentation:** nnU-Net (baseline), SelfMedMAE (pre-training), Swin UNETR,
  MTANet.
- **Classification / fusion:** XGBoost (tabular baseline), MMGL, DAFT
  (image+tabular fusion), and MTANet again for the joint-task angle.
- **Most directly comparable to this project's contributions:** MMGL vs. the
  inductive retrieval-augmented classifier (`brats_gbm/gnn.py`); MTANet vs. the
  joint segmentation–classification proposal (`docs/RESUME.md`).

## BibTeX (skeletons — check fields before use)

```bibtex
@inproceedings{chen2016xgboost,
  title     = {{XGBoost}: A Scalable Tree Boosting System},
  author    = {Chen, Tianqi and Guestrin, Carlos},
  booktitle = {Proceedings of the 22nd ACM SIGKDD International Conference on
               Knowledge Discovery and Data Mining (KDD)},
  year      = {2016}
}

@article{mmgl2022,
  title   = {Multi-Modal Graph Learning for Disease Prediction},
  journal = {IEEE Transactions on Medical Imaging},
  year    = {2022},
  note    = {Verify authors/volume/pages; code: github.com/SsGood/MMGL}
}

@article{isensee2021nnunet,
  title   = {{nnU-Net}: a self-configuring method for deep learning-based
             biomedical image segmentation},
  author  = {Isensee, Fabian and Jaeger, Paul F. and Kohl, Simon A. A. and
             Petersen, Jens and Maier-Hein, Klaus H.},
  journal = {Nature Methods},
  year    = {2021}
}

@inproceedings{selfmedmae2023,
  title     = {Self Pre-training with Masked Autoencoders for Medical Image
               Classification and Segmentation},
  booktitle = {IEEE International Symposium on Biomedical Imaging (ISBI)},
  year      = {2023},
  note      = {Verify authors; code: github.com/cvlab-stonybrook/SelfMedMAE}
}

@inproceedings{hatamizadeh2021swinunetr,
  title     = {{Swin UNETR}: Swin Transformers for Semantic Segmentation of
               Brain Tumors in {MRI} Images},
  author    = {Hatamizadeh, Ali and Nath, Vishwesh and Tang, Yucheng and
               Yang, Dong and Roth, Holger R. and Xu, Daguang},
  booktitle = {MICCAI Brain Lesion (BrainLes) Workshop},
  year      = {2021},
  note      = {replaces ResGANet; code: github.com/Project-MONAI/research-contributions}
}

@inproceedings{polsterl2021daft,
  title     = {Combining {3D} Image and Tabular Data via the Dynamic Affine
               Feature Map Transform},
  author    = {P{\"o}lsterl, Sebastian and Wolf, Tom Nuno and Wachinger, Christian},
  booktitle = {MICCAI},
  year      = {2021},
  note      = {replaces MMCL; code: github.com/ai-med/DAFT}
}

@article{mtanet2024,
  title   = {{MTANet}: Multi-Task Attention Network for Automatic Medical Image
             Segmentation and Classification},
  journal = {IEEE Transactions on Medical Imaging},
  year    = {2024},
  note    = {Verify authors/volume/pages; code: github.com/yatingling/MTANet}
}

% MMCL and ResGANet dropped (no reproducible official code) — see substitutions above.
```
