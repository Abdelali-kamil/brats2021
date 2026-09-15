# DAFT (classification baseline — replaces MMCL)

Pölsterl, Wolf, Wachinger, *Combining 3D Image and Tabular Data via the Dynamic
Affine Feature Map Transform*, MICCAI 2021 · https://github.com/ai-med/DAFT
Modality **I+T**. Fuses a 3D image with tabular clinical data — the closest
runnable analogue to this project's imaging+clinical fusion, which is why it
replaces the code-less MMCL. Needs a CUDA GPU.

## 1. Environment (isolated)

```bash
python -m venv baselines/daft/.venv && source baselines/daft/.venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -e baselines/repos/daft          # or its requirements.txt at the pinned commit
```

## 2. Data

DAFT takes a 3D volume plus a tabular vector per subject.

- image: the BraTS/UPenn volume (or an ROI crop around the tumour)
- tabular: the same radiomic/clinical features the other classifiers use
  (`results/classification/brats_mgmt_features.csv` + `metadata/` clinical CSVs)
- label: `mgmt_label`

DAFT's repo expects an HDF5 built by its own preprocessing script — convert your
cohort into that layout (see `baselines/repos/daft/daft/data_utils/` at the
pinned commit). **Fairness:** same repeated stratified 5-fold split as the other
classifiers.

## 3. Train + emit the result file

```bash
cd baselines/repos/daft
python train.py --task clf --net daft --dataset <your_mgmt.h5> ...   # flags per the repo
```

Collect per-fold held-out probabilities and write
`results/baselines/daft_classification.json` in the flat schema (see
`baselines/README.md`; Recall=sensitivity, Precision=PPV, AUC). As with the
other classifiers, expect a near-chance MGMT AUC on this cohort — report the real
number.
