# Brain tumour segmentation and molecular-marker classification

Segmentation of glioma sub-regions on **BraTS2021**, domain adaptation to
**UPenn-GBM**, and IDH1 mutation classification from segmentation-derived
radiomic features.

The central result is the domain-adaptation one: a segmentor trained on
BraTS2021 collapses when applied unchanged to UPenn-GBM, and fine-tuning
recovers almost all of the loss.

## Layout

```
brats_gbm/                 importable package — all shared logic
  splits.py                every train/val/test partition, one definition
  model.py                 WaveletUNet++ (DWT-based encoder)
  data/
    brats.py               BraTS2021 loader
    upenn.py               UPenn-GBM loader
    collate.py             variable-shape batch collation
  eval/
    inference.py           sliding window + Gaussian blending + flip TTA
    postprocess.py         hierarchy, components, ET policy
    metrics.py             Dice, surface HD95
    stats.py               bootstrap confidence intervals

scripts/                   entry points, all argparse-driven
  train_brats.py           train on BraTS2021
  train_upenn.py           fine-tune on UPenn-GBM
  evaluate_upenn.py        evaluate on the UPenn test set
  crossval_upenn.py        5-fold cross-validated fine-tune + evaluate
  summarize_brats.py       BraTS internal-validation summary with CIs
  extract_radiomic_features.py
  train_classifier_idh1.py IDH1 classification
  visualize_upenn.py       qualitative overlays
  verify_no_leakage.py     assert every data-integrity property
  run_pipeline.sh          full pipeline, end to end

results/                   committed outputs (CSV/JSON, no binaries)
docs/METHODOLOGY.md        protocol, corrections, limitations
docs/ARCHIVE.md            what was removed and why
```

Data (`data/`, `upenn_nifti/`, `upenn_data/`), model weights (`checkpoints/`)
and cached probability maps (`cache/`) are gitignored.

## Setup

```bash
pip install -r requirements.txt
```

Expected data layout, obtained separately:

```
data/BraTS2021_XXXXX/BraTS2021_XXXXX_{flair,t1,t1ce,t2,seg}.nii.gz
upenn_nifti/sub-XXX_{FLAIR,T1w,ce-gd_T1w,T2w,seg}.nii.gz
upenn_data/UPENN-GBM_clinical_info_v2.1.csv
```

## Reproducing the results

Run the integrity checks first. They are cheap and catch a misplaced data
directory before it costs GPU hours.

```bash
python scripts/verify_no_leakage.py
```

### Segmentation

```bash
# BraTS2021 — internal-validation summary with bootstrap CIs (CPU, seconds)
python scripts/summarize_brats.py

# UPenn-GBM — all four setups (GPU, ~2.5 h; probability maps are cached,
# so reruns that only change post-processing are near-instant)
python scripts/evaluate_upenn.py
```

`evaluate_upenn.py` selects region thresholds and the ET post-processing policy
on the 14 validation subjects, then scores the 29 test subjects **once**. The
selected configuration and the validation score that justified it are written
to `results/upenn/validation_selection.csv`.

### Classification

```bash
python scripts/extract_radiomic_features.py    # GPU, ~1.5 h for 629 subjects
python scripts/train_classifier_idh1.py        # CPU, ~3 min
```

### Training from scratch

```bash
python scripts/train_brats.py --epochs 700 --save-dir checkpoints
python scripts/train_upenn.py --epochs 80 --save-dir checkpoints
```

`train_upenn.py` fine-tunes from the BraTS checkpoint. It does not resume from
an earlier UPenn checkpoint by design — see `docs/METHODOLOGY.md`.

### Cross-validation (implemented, not run)

```bash
python scripts/crossval_upenn.py --dry-run   # verify folds, no GPU
python scripts/crossval_upenn.py             # ~10-14 GPU-hours
```

This evaluates all 147 expert-segmented subjects instead of 29 and is the
single biggest available improvement to the tightness of the reported
intervals. **No cross-validated result is reported anywhere in `results/`,**
because the folds have not been trained.

## Results

Full numbers in `results/`; protocol and caveats in `docs/METHODOLOGY.md`.

**BraTS2021**, 251 internal-validation cases — mean Dice **0.890**
[0.870, 0.907]; ET 0.850, TC 0.894, WT 0.925. The train-minus-validation gap
is +0.009 with a CI containing zero, so no overfitting is detectable. This
partition was used for model selection during training, so it is internal
validation rather than a clean test set.

**UPenn-GBM**, 29 held-out test subjects — the BraTS model applied zero-shot
scores far below the fine-tuned model; the paired improvement and its
confidence interval are in `results/upenn/domain_adaptation.json`.

**IDH1 classification**, 416 subjects with 17 positives — random forest,
AUC **0.893** [0.827, 0.950]. Underpowered; read the intervals, not the point
estimates.

## Interpreting these numbers

Three things a reader should know before quoting anything here.

The BraTS figure is internal validation. The clean external evaluation in this
project is UPenn-GBM.

The UPenn test set has 29 subjects, so differences smaller than about 0.05
mean Dice are not resolvable. The fine-tuned-versus-zero-shot gap is far larger
than that and is the claim worth making; the ordering among fine-tuned variants
is not.

IDH1 classification rests on 17 positive cases. It is a feasibility result.
Positive predictive value in particular will not transfer to a general glioma
population, whose IDH1-mutant prevalence is much higher than this
glioblastoma-enriched cohort's 4%.

## History

This repository was reorganised and its evaluation protocol corrected. Earlier
reported figures — in particular a segmentation result with ET Dice above WT
Dice — came from thresholds fitted on the test set and should not be cited.
`docs/METHODOLOGY.md` documents each correction; `docs/ARCHIVE.md` records what
was removed.
