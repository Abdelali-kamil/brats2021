# Where to pick up

Last session: 2026-07-27.

## What changed this session

The deep classification module was **redesigned to be BraTS-2021-primary**, in
line with the project's scope (BraTS is the main dataset; UPenn-GBM is external
validation only). Previously the KAN+GNN ablation ran only on UPenn MGMT, which
inverted that scope. The current state:

- `brats_gbm/gnn.py` — `ClinicalImagingKANGNN` now includes a region-token
  **Transformer branch** (paper §V-F) and treats the clinical branch as
  optional, so it runs imaging-only on BraTS and multimodal on UPenn.
- `scripts/ablation_kan_gnn.py` — rewritten. Primary task is **BraTS 2021 MGMT**
  (n=577, imaging-only, 51 features shared with UPenn), nested 5×3 CV. Adds a
  **UPenn external-validation** leg and a UPenn-only multimodal reference.
- Results in `results/classification/kan_gnn_brats.json` (the old
  `kan_gnn_ablation.json` is the superseded UPenn-only run).
- Figures regenerated: `results/figures/fig1_ablation_auc_forest.*`,
  `fig2_attribution.*`, `fig_architecture_implemented.*`.

## Results (final)

BraTS 2021 MGMT, nested 5×3 CV, 3 repeats:

| Rung | AUC [95% CI] |
|---|---|
| MLP (baseline) | 0.617 [0.568, 0.663] |
| KAN | 0.602 [0.553, 0.648] |
| KAN + Transformer | 0.645 [0.599, 0.692] |
| KAN + GNN | 0.615 [0.568, 0.662] |
| KAN + Transformer + GNN (full) | 0.624 [0.575, 0.670] |

- External BraTS→UPenn (full model): **0.537 [0.460, 0.612]** — chance.
- UPenn multimodal (imaging+clinical+gate): **0.599 [0.523, 0.671]**.

**Reading:** all BraTS intervals overlap (mean 95% CI half-width ±0.047); the
full module does not beat the plain MLP; KAN ≤ MLP; the memory bank contributes
~25% of the representation but no AUC; external transfer is at chance. This is a
negative result and is reported as one, consistent with MGMT-from-MRI being a
weak-signal task (radiomic RF 0.583; RSNA-MICCAI 2021 winner ~0.62).

## In flight (started 2026-07-29)

**Downsampling ablation — DWT vs width-matched max-pooling** (paper Table IX).
`bash scripts/run_ablation.sh` trains both arms from scratch on the seed-42
1000/251 split at a matched 200-epoch budget, then scores each through the
reported evaluation pipeline. Progress:
`tail -F logs/ablation_ablation_{maxpool,dwt}.log`; outputs land in
`results/brats/ablation/`. Re-running skips arms that already have a
`*_best.pth` (set `FORCE=1` to redo). Protocol in `docs/METHODOLOGY.md`.

- **Arm A (max-pooling): done.** Best mean Dice **0.8199 at epoch 43**
  (`checkpoints/ablation/ablation_maxpool_best.pth`). Trained 199 epochs before
  being stopped — see below.
- **Arm B (DWT): running**, started 2026-07-30 12:05, ~19 h for 200 epochs, then
  ~4–8 h of TTA evaluation for both arms. Expect results 2026-07-31.

The epoch budget was cut from 750 to 200 mid-experiment. `--epochs 750` in
`train_brats.py` is a *resume* target (the original run continued an epoch-650
checkpoint by 100 epochs); as a from-scratch budget it is ~10x too large here.
Arm A peaked at epoch 43 and hit its 1e-7 LR floor at epoch 76, then sat in
noise for 150 epochs. Continuing would have burned ~53 h per arm to no effect.
Both arms now get 200 epochs and the best-validation checkpoint is reported.

When it lands, fill in Table IX of the paper — the max-pooling row is currently
a red `TBD` placeholder — and remove the red note in the Ablation section. Do
not pre-write the conclusion: a result where max-pooling matches or beats the
DWT is a real possibility and should be reported as found.

**Two defects fixed this session, both pre-existing:**

- `brats_gbm/data/brats.py` used a bare `from image import ...` that never
  resolved under `train_brats.py`, behind an `except ImportError` that swallowed
  it. BraTS training could not run from the repository at all. Fixed to the
  package-qualified import every other module uses.
- `train_brats.py` wrote a full ~125 MB checkpoint every epoch (~94 GB per run,
  more than the free space allows for two runs) and never saved a distinct
  best. Now writes `<tag>_best.pth` / `<tag>_last.pth` only, each carrying the
  full run configuration.

The released `segmentor_epoch_650.pth` therefore cannot have been produced by
the code as it stands, so its training protocol is unverifiable. Reported
results are unaffected — the checkpoint is untouched and its evaluation
reproduces to 1.1e-16 — but the paper's Methods section describes current code,
not a confirmed history.

## Outstanding

- **BraTS k-fold cross-validation** for a fold-level mean ± std in paper
  Table VI — deferred by decision until the ablation lands. ~10 days at 5 folds
  × 750 epochs. Requires an inner validation split per fold for LR scheduling
  and checkpoint choice, so fold scores stay genuinely held out. Table VI
  currently carries a caption note explaining why our row has no ±.
- **nnU-Net segmentation baseline** on this project's 251-case split — still no
  comparison numbers computed on the project's own split.
- **UPenn 5-fold cross-validation** (`scripts/crossval_upenn.py`) — implemented,
  verified (`--dry-run`), not trained (~10–14 GPU-hours). Would tighten the
  segmentation intervals from n=29 to n=147.
- The learned lesion/slice/global encoders of the original proposal remain
  unbuilt; the module uses radiomic features, not segmentation-encoder
  bottleneck features. Documented in `scripts/make_architecture_figure.py`.

## Repository state

`python -m pytest tests/ -q` → 46 pass. Segmentation results unchanged and
final. Classification redesigned and rerun this session.
