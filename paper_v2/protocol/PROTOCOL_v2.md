# Evaluation protocol v2 — model-improvement campaign

> Public copy (2026-10-03): passages about a dataset that is not used in the paper
> were removed and are marked [removed]. Nothing else was changed; the authors keep the original
> dated file.

Written 2026-09-24 ~21:00 (+08:00), BEFORE Run A finished and before Run C
started. Supersedes nothing; extends improve_v1/PROTOCOL.txt.

## Data partitions (fixed, never changed)
- BraTS 2021, `brats_split_3way(seed=42)`: train 1000 / val 126 / test 125.
  Frozen lists: `~/brats2021/baselines/common/brats_split_frozen.csv`.
- Training uses train only. Checkpoint selection and ALL post-processing /
  inference choices use val only. Test is never used to choose anything.
- External cohort: BraTS-Africa (BraTS-SSA 2023). Used for evaluation only —
  no training, no tuning, no selection. Before use it must pass the same
  patient-overlap check (tumour-mask Dice + image
  correlation against all 1251 BraTS 2021 cases).
- [removed]

## Candidate models (ours)
| id | description | params |
|---|---|---|
| ep253 | aug_cosine_repro1 epoch 253 (current reported model) | 10.40 M |
| RunA | same architecture, deep supervision, 400-ep cosine, from scratch | 10.40 M |
| RunC | 3D Haar DWT (all axes), deep supervision, 400-ep cosine, from scratch | 15.10 M |
All: seed 42, batch 2 x accum 4, AdamW 2e-4, same augmentation and loss.

## Per-model evaluation (identical for every candidate)
1. Checkpoint = the run's best epoch by the training script's val metric
   (centre-crop, 126 val cases). No other checkpoint is considered.
2. Full val evaluation: sliding window 128^3, stride 64, Gaussian blending,
   8-flip TTA, threshold 0.5 (`evaluate_brats.py`).
3. Post-processing sweep on val (`sweep_postproc_hd95.py`); setting chosen by
   the fixed rule: lowest mean HD95 among settings within 0.002 Dice of the
   as-shipped setting.
4. Test (125) scored ONCE with that setting. A guard refuses a second scoring.

## Headline model selection (decided on VAL only)
The headline "ours" = the candidate with the highest step-3 val mean Dice.
Tie (<0.001) -> fewer parameters. Every candidate's test result is still
reported, in an ablation table, so nothing is hidden; but the headline is the
val winner even if another candidate happens to score higher on test.

## Comparison with baselines (fixed)
- nnU-Net (3d_fullres, fold all, checkpoint_final, native TTA): test 0.9101.
- Swin UNETR (best-by-val ep199, upstream recipe): test 0.9028.
- Same 125 test cases, same scorer (`score_case` + `summarise_segmentation`).
- Statistics: paired bootstrap 95% CI of the per-case mean-Dice difference
  (10,000 resamples, seed 0) and two-sided Wilcoxon signed-rank; per region too.
  Reported whichever way they come out. No claim of superiority without a
  paired CI excluding 0; "comparable" only if the paired CI includes 0.

## Efficiency (measured on an otherwise idle GPU)
Params; forward FLOPs per patch at each model's own window; per-case
wall-clock and peak GPU memory under each method's OWN reported inference
protocol, plus ours without TTA. Same 10 test cases, same GPU, 1 warm-up case.

## Not allowed
Re-scoring test for a non-selected setting; changing thresholds after seeing
test; dropping a baseline or a candidate from the ablation table; changing this
file after results exist (amendments go below with a timestamp and reason).

## Amendments
(none)

### Amendment 1 — 2026-09-24 ~22:20 (timestamp corrected from "~22:45", which was a mistake), before any BraTS-Africa data was opened
External cohort source: TCIA "PKG - BraTS-Africa.zip" (146 subjects, CC BY 4.0),
obtained via Hugging Face re-uploads (krohn/brats-africa, Astride1/brats-africa;
identical LFS sha256 eb2c34fa0f1a2380...) because TCIA's Aspera node is
unreachable from this network. Integrity: sha256 must match, and cases must match
voxel-wise the independent BraTS 2023 SSA challenge mirror (MedOtter/brats2023-ssa)
where they overlap. (Zenodo 14510932 was checked and is a trained model, not data.)
External evaluation rules:
- Gate: zero patient overlap with BraTS 2021 by the patient-overlap check; any matched
  subject is excluded and reported.
- Every labelled subject that passes the gate is evaluated. No subject-level
  exclusions after seeing predictions.
- No training, tuning, threshold or post-processing choice on this cohort. Each
  method runs exactly as for the BraTS test: ours with its BraTS-val-selected
  post-processing; nnU-Net native; Swin UNETR upstream recipe (model.pt).
- Each method gets its own channel order/normalisation.
  Labels: BraTS 2023 convention NCR=1, SNFH/ED=2, ET=3 -> ET={3}, TC={1,3}, WT={1,2,3}.
- All our candidates (ep253, RunA, RunC) are scored; the headline is still the
  BraTS-val winner, never picked by Africa results.
- Same scorer and paired statistics as the BraTS test.

### Amendment 2 — 2026-09-24 ~22:30, before any BraTS-Africa image or label was opened
TCIA's metadata sheet (BraTS-Africa_TCIA_datainfo_v2.xlsx, downloaded from TCIA)
lists two groups: sheet 1 = diffuse gliomas (~96 subjects), sheet 2 = 51 other
neoplasms (meningioma, metastases, ventricular/choroid plexus masses, a few
paediatric-type gliomas). All three methods were trained on adult glioma only.
- PRIMARY external analysis = the sheet-1 glioma subjects (the population the
  models are for). This replaces "every labelled subject" in Amendment 1 for the
  primary claim; the gate and all other rules are unchanged.
- SECONDARY = sheet-2 other neoplasms, reported separately as an
  out-of-distribution stress test, never pooled with the primary result and
  never used for any generalisation claim about glioma.
- Subject-to-group assignment comes from the TCIA sheet only, fixed before any
  prediction exists; subjects absent from both sheets are listed and excluded.

### Amendment 3 — 2026-09-24 ~22:55, after integrity/format checks, BEFORE any model was run on BraTS-Africa
Integrity: package sha256 OK; 5 random official challenge cases (MedOtter mirror)
are voxel- and affine-identical to the package. Groups: TCIA sheet == folder
names exactly (95 glioma, 51 other neoplasm). Shapes 240x240x155, labels {0,1,2,3}.
Format finding: BraTS 2021 arrays are stored LPS (affine diag -1,-1,1); 144/146
BraTS-Africa arrays are stored RAS (diag 1,1,1). Brain-mask agreement with the
BraTS 2021 grid rises from ~0.68-0.75 to ~0.83-0.92 Dice when the A-P axis is
flipped, confirming the arrays (not just headers) differ. All three methods
consume raw arrays, so un-harmonised input would test a storage-convention
mismatch, not generalisation.
Rule: every BraTS-Africa volume AND label map is reoriented, from its own
header only (nibabel io_orientation -> LPS, i.e. axis flips; no resampling,
no interpolation), onto the BraTS 2021 array layout, identically for all
methods, before any inference. Cases already LPS are unchanged. This is a
format harmonisation fixed by the headers, not a tuned choice.

### Amendment 4 — 2026-09-25 ~00:30: BraTS-Africa ADAPTATION experiment (pre-registered before any fine-tuning)
Motivation (stated honestly): zero-shot BraTS-Africa glioma results were seen first
(ours ep253 0.739, Swin UNETR 0.836). The zero-shot results are reported
unchanged; this is an ADDITIONAL experiment, reported alongside, never instead.
Data: the 95 glioma subjects only, reoriented (Amendment 3), labels relabelled to
the BraTS 2021 convention (ET 3->4) so every method's own loader reads them.
Split (patient-level, fixed, sha256 c57ada6e...): adapt_africa/split.json,
seed 42 permutation: TEST 35 / VAL 12 / TRAIN 48. Other neoplasms not used.
Fairness rules (chosen to leave no per-method tuning):
- Every method starts from its BraTS 2021 weights used for the BraTS test
  (ours: ep253 — the current headline; nnU-Net: fold-all checkpoint_final;
  Swin UNETR: model.pt ep199) and uses ITS OWN DEFAULT training recipe and
  default learning rate (ours: AdamW 2e-4 cosine, bs2/accum4, same aug/loss;
  nnU-Net: official -pretrained_weights fine-tuning, SGD 0.01 poly; Swin:
  AdamW 1e-4 warmup-cosine).
- Equal training-sample budget ~14,400 (= 300 passes over the 48 cases):
  ours 300 epochs, Swin 300 epochs, nnU-Net 29 epochs x 250 iters x batch 2.
- Checkpoint = each method's own standard (ours best-by-val, Swin best-by-val,
  nnU-Net checkpoint_final); val = the 12 VAL cases only. Ours: post-processing
  from the VAL sweep by the fixed rule. No other choices.
- TEST 35 scored once per method, same scorer; paired stats vs baselines AND
  paired before/after (zero-shot vs adapted on the same 35 cases).
- If Run A/C later becomes the headline, it is adapted with the identical
  recipe and reported in addition; ep253's adapted result is not removed.

### Amendment 5 — 2026-09-25 ~12:00, BEFORE Run A (or any later candidate) is evaluated
User's objective changed (25 Sep): maximise BraTS 2021 Dice; model size/compute no
longer a constraint. Changes apply ONLY to candidates not yet evaluated (Run A,
Run C, ensembles); ep253 keeps its already-scored result, test is not re-scored.
1. Post-processing selection for new candidates = highest VAL mean Dice (ties
   within 0.001 -> lower mean HD95), over an extended VAL grid:
   WT policy {components, largest} x component floors {x1, x5, x20} x ET
   min-volume rule {off, 100, 200, 500, 1000 voxels} (BraTS-standard ET removal;
   200 was the untuned default). Threshold stays 0.5. TEST scored once with the winner.
2. Run C redefined (not yet started): 3D Haar DWT, base width 24 (34.0 M params,
   ~nnU-Net's 31.2 M), deep supervision, 400-epoch cosine, seed 42, bs2/accum4.
   Early check (VAL only): at epoch 60, if Run C's best val (training-script metric)
   is > 0.010 below Run A's best over its first 60 epochs (same schedule), it is
   flagged for replacement by a 2D-DWT width-24 run. The flag is logged; stopping is
   a manual decision recorded here.
3. Final ensemble: after Run C, probability-averaged combinations of {ep253, Run A,
   Run C} are compared on VAL (same grid/rule); the best VAL combination is scored
   on TEST once. Reported as an ensemble, separately from single-model results.
   Note for the paper: nnU-Net's result is its single fold-all model; its usual
   BraTS setup is a 5-fold ensemble, so ensemble-vs-single must be disclosed.
4. Headline = best VAL candidate among single models and the ensemble; every
   candidate's TEST result is reported (Amendment-free rules above unchanged).

### Amendment 6 — 2026-09-25 ~17:45: BraTS-Africa 5-fold cross-validated ADAPTATION (before any CV training)
Adds to (does not replace) Amendment 4. The 95 glioma subjects are split into 5
folds of 19 (adapt_africa/cv/cv_split.json, seed 2026, sha256 58d22f94d14b7578...). Per fold: TEST = the
fold (19), from the other 76: VAL 12 (seed 100+k), TRAIN 64. Every subject is tested once.
Rules identical to Amendment 4 (each method's own default recipe from its BraTS
weights; ours/Swin best-by-val, nnU-Net final; same scorer; paired stats), except:
- Budget = 150 passes over the 64 TRAIN cases = 9,600 samples per fold (fixed now,
  for time; equal for all): ours 150 epochs, Swin 150 epochs (val every 5),
  nnU-Net 19 epochs x 250 x 2 = 9,500.
- Ours starts from the best SINGLE model by BraTS VAL at the time ours' CV runs
  (Run C if it wins, else Run A/ep253) — never chosen by Africa results. Ours'
  post-processing per fold from its 12-case VAL with the Amendment-5 Dice grid;
  baselines use their native outputs as everywhere else.
- Results pooled over the 5 folds (n=95), reported with zero-shot on the same 95,
  paired before/after and ours vs baselines. Amendment-4 single-split results stay reported.

### Amendment 7 — 2026-09-25 ~23:15, before Run C finished and before Run D exists
Run D (new candidate): identical to the BraTS-VAL winner of {Run A, Run C}
(architecture, deep supervision, 400-ep schedule, seed, batch) EXCEPT
InstanceNorm3d(affine) instead of BatchNorm3d and per-channel z-score input
normalisation over brain voxels (the nnU-Net conventions). Motivation stated:
BatchNorm at batch 2 and min-max inputs are the most likely causes of the scanner-
shift gap seen on BraTS-Africa and of part of the BraTS gap. Evaluated exactly like
every candidate (eval_run.sh: VAL Dice sweep -> TEST once; then BraTS-Africa
zero-shot). The evaluators read norm/normalisation from the checkpoint config.
Order on the single GPU: Run D trains right after Run C's evaluation; ours' 5-fold
Africa CV (Amendment 6) runs after Run D is evaluated, starting from the best
SINGLE model by BraTS VAL among {ep253, Run A, Run C, Run D}.
Ensemble (Amendment 5 #3, extended): all combinations of >=2 of {ep253, Run A,
Run C, Run D}, probability-averaged, each with the Amendment-5 VAL Dice sweep; the
best (combination, setting) by VAL Dice is scored on TEST once.

### Amendment 8 — 2026-09-25 ~23:10, before any of ours' CV folds has trained
Ours' Africa CV (Amendment 6) is extended with an adapted-ensemble option:
- M1 = best SINGLE model by BraTS VAL (Amendment 6/7 rule); M2 = second best by
  BraTS VAL among {ep253, runA, runC, runD}. Both fine-tuned per fold with the
  identical Amendment-6 recipe (own architecture flags, 150 epochs, warm start).
- Per fold, on the fold's 12 VAL cases only: candidates {M1, M2, M1+M2 averaged},
  each with the Amendment-5 Dice grid; winner = highest VAL Dice (ties within
  0.001 -> M1 alone). The fold's 19 TEST cases are scored once with the winner.
- Reported: (a) the Amendment-6 primary = M1 alone with its own VAL-selected
  setting (unchanged), and (b) this val-selected variant, pooled over 95. Both
  are stated; (b) is disclosed as possibly an ensemble, compared with single
  baselines. Test predictions are generated as probabilities without scoring,
  then scored once per reported analysis.

### Amendment 9 — 2026-09-25 ~23:35, before any of ours' CV folds has trained
User's deadline (~29–30 Sep, "Balanced" plan): the Amendment-8 adapted ensemble is
DROPPED. Ours' Africa CV = M1 only (best single model by BraTS VAL), exactly the
Amendment-6 procedure; test predictions still saved as probabilities first and
scored once. Run D and the BraTS-2021 ensemble (Amendment 7) are unchanged.

### Amendment 10 — 2026-09-26 ~01:00, before Run C/Run D are evaluated and before any CV fold trains
User approved the "all feasible options" list. For candidates not yet evaluated:
1. Orientation TTA: VAL is evaluated twice (historical nibabel array order, and
   the average of nibabel order + training (SimpleITK) order); each gets the
   Amendment-5 Dice sweep; the higher VAL Dice decides the inference mode; TEST
   is scored once with it. (Pilot on 20 VAL cases of ep253: +0.0012.)
2. Snapshots: _last.pth of Run C and Run D is copied at epochs 360,370,...,400.
   For Run D (InstanceNorm, so no BatchNorm statistics to recompute) a weight
   average (SWA) of these five snapshots is an additional candidate "runD_swa",
   evaluated exactly like every candidate.
3. The Amendment-7 ensemble search includes runD_swa and uses each member's
   VAL-chosen inference mode.
4. Amendment 9 is reversed: ours' Africa CV again fine-tunes M1 and M2 and chooses
   among {M1, M2, M1+M2} on each fold's VAL (Amendment 8), if the timeline holds.
Skipped by judgement (disclosed): per-region threshold tuning (did not transfer
before; triples sweep cost); rotation TTA deferred (4x inference) unless GPU time remains.

### Amendment 11 — 2026-09-28 ~14:50, after a 55-hour GPU outage; before Run C has finished or been evaluated, before any of ours' CV folds has trained
GPU fault #4 (26 Sep 07:49, Run C at epoch 166) left the GPU unusable until the
server was rebooted on 28 Sep 14:29 (~55 h lost). With the deadline before 1 Oct:
1. Run D (and runD_swa) is CANCELLED: ~33 h of training plus evaluation cannot fit
   next to Run C and the CV. It is disclosed as planned but not run.
2. The BraTS-2021 ensemble search (Amendments 7/10) runs over the available
   candidates {ep253, runA, runC}, otherwise unchanged, as soon as Run C is evaluated.
3. Ours' Africa CV starts NOW, in parallel with Run C training (the GPU has room for
   both). M1/M2 = the top two by BraTS VAL among the models already evaluated at this
   time: runA (0.8928) and ep253 (0.8911), each with its evaluated inference mode
   (nibabel order). Run C is not a CV start model because it is not evaluated; this
   is fixed now, before Run C's validation result is known.
4. Order: M1 on all five folds first (train + probabilities), then M2 on all five
   folds, then the per-fold selection (Amendment 8). The Amendment-8 variant is
   reported only if M2 completes on all five folds; otherwise the per-fold selection
   runs with M1 only and only the Amendment-6 M1 result is reported.
Everything else (split, recipe, grid, test-once rules) is unchanged.

### Amendment 12 — 2026-09-29 ~16:15, after Run C's TEST was scored (0.8973), before the ensemble search has started
Amendment 10 deferred rotation TTA "unless GPU time remains". With Run D cancelled
(Amendment 11), ~30 h remain before the deadline, so:
1. New candidate "runC_rot": the SAME frozen Run C weights and Run C's VAL-chosen axis
   mode ("both"), plus in-plane 90-degree rotation TTA: the prediction is averaged over
   axial rotations by 0 and 90 degrees, each with the existing 8-flip TTA (with the flips
   this covers all 16 axial-plane symmetries x slice flip). Evaluated exactly like every
   candidate: VAL (126) with the Amendment-5 Dice sweep; TEST (125) scored once with its
   VAL-selected setting; reported whatever it shows.
2. The ensemble search (Amendments 7/10/11) runs over {ep253, runA, runC, runC_rot} once
   runC_rot is evaluated; if runC_rot is not evaluated by 30 Sep 08:00, it runs without it.
   The headline remains the best-VAL candidate (single model or ensemble).
3. Disclosure: this was decided after Run C's test score was known. It is the condition
   pre-specified in Amendment 10 (GPU time remaining); choice among candidates uses VAL only,
   and every candidate's TEST result is reported.

### Amendment 13 — 2026-09-29 ~17:00, SECONDARY analysis decided after ours' CV result (0.848) was known (user approved)
Amendment 11 fixed the CV start models {runA, ep253} only because Run C was not yet
evaluated. Run C is now the best single model by BraTS VAL (0.8994), i.e. the model the
original Amendment 6/7 rule would have chosen. As a clearly labelled SECONDARY analysis:
- Ours' 5-fold CV is repeated from Run C (same cv_split.json, same Amendment-6 recipe:
  150 epochs, warm start, own architecture flags, code_ours_v2), inference in Run C's
  VAL-chosen mode ("both"), per-fold post-processing chosen on the fold's 12 VAL cases
  (Amendment-5 grid), each fold's 19 TEST cases scored once -> per_case_ours_runC.csv.
- Reported NEXT TO the Amendment-6/11 result (0.848), never instead of it, with this
  timing disclosed. Existing files are not modified (separate ckpt_MC / probs_MC_* dirs).

### Amendment 14 — 2026-09-29 ~18:00, classification added (user: "segmentation + classification", BraTS 2021 + BraTS-Africa); written before any classification result below
A. BraTS-Africa tumour-type classification (new): glioma (95, positive class) vs other
   neoplasm (51), all 146 subjects.
   - Features: the project's existing segmentation-derived region features
     (brats_gbm.features.all_region_features on z-normalised T1/T1ce/T2/FLAIR; regions ET/TC/WT).
   - Masks: PRIMARY = Run C's zero-shot predicted masks (the BraTS-trained segmenter never saw
     African data; same inference and post-processing as the reported Run C zero-shot result),
     i.e. the full segment-then-classify pipeline. REFERENCE = expert masks (upper bound).
   - Classifiers: the existing a-priori models (class-weighted logistic regression C=0.1;
     random forest 500 trees, min leaf 2), no hyperparameter search.
   - Evaluation: repeated stratified 5-fold CV (10 repeats, seed 42), decision threshold
     from inner CV on training folds only; AUC with patient-level bootstrap CI (2,000),
     balanced accuracy, sensitivity, specificity. Both models reported; no selection.
   - Confound reference: a centre+scanner-only logistic model (one-hot), same CV, to show how
     much of any signal could be site-driven.
B. BraTS 2021 MGMT (existing, re-reported): radiomic baselines and the KAN/Transformer/GNN
   module under nested CV on 577 cases (results/classification/*.json, 2026-09-15). [removed]

### Amendment 13b — 2026-09-29 ~18:00, before any fold of the Run C CV (Amendment 13) has trained
Mirroring Amendment 8 for the Run C CV: per fold, on the fold's 12 VAL cases only, choose
among {MC, MC+M1, MC+M2, MC+M1+M2} (probability averaging of the already-saved per-fold
probabilities; M1 = from Run A, M2 = from ep253) x the Amendment-5 grid; highest VAL Dice,
ties within 0.001 -> fewer members, then lower HD95. The fold's TEST cases are scored once
-> per_case_ours_runC_sel.csv. Reported next to MC alone (per_case_ours_runC.csv) and next to
the Amendment-6/8 results; disclosed as a secondary, possibly-ensemble analysis.

### Amendment 15 — 2026-09-29 ~21:30, Run D reinstated with stronger augmentation (user: "forget time, improve results, 1-2 days is fine"); written before Run D is trained
Motivation (from results already reported): our deficit on BraTS-Africa is concentrated in the
whole tumour; the baselines z-score their inputs and train with much stronger augmentation.
1. Run D = Run C's architecture (3D DWT, F=24, deep supervision) + InstanceNorm3d (affine) +
   per-channel z-score inputs over non-zero voxels + STRONG augmentation, from scratch,
   400 epochs, otherwise Run C's recipe (AdamW 2e-4, cosine, batch 2 x accumulation 4, seed 42).
   Strong augmentation (nnU-Net-inspired; on each 128^3 training patch): existing flips + in-plane
   rot90; with p=0.25 a joint random rotation (+-30 deg about each axis) and isotropic scaling
   (0.7-1.4) of image (linear) and labels (nearest); per channel: Gaussian noise (p=0.15, var
   U(0,0.1)), Gaussian blur (p=0.2, sigma U(0.5,1)), multiplicative brightness (p=0.15, U(0.75,1.25)),
   contrast (p=0.15, U(0.75,1.25), range-preserving), simulated low resolution (p=0.25, zoom
   U(0.5,1) nearest down / cubic up), gamma (p=0.3, U(0.7,1.5), range-preserving; inverted with p=0.1).
2. Run D is evaluated exactly like every candidate (eval_run.sh: VAL nib/both + sweep -> TEST once),
   plus its rotation-TTA variant runD_rot (Amendment-12 procedure) -> BraTS-Africa zero-shot for the
   VAL-selected one of {runD, runD_rot} -> 5-fold Africa CV from Run D (Amendment-13 procedure,
   label MD) with the Amendment-13b per-fold ensemble option extended to {MD, MD+MC, MD+M1, MD+MC+M1}.
3. A second ensemble search ("ensemble2") over {ep253, runA, runC, runC_rot, runD, runD_rot}
   on VAL, TEST once. The BraTS headline remains the best-VAL candidate; every TEST result is reported.
4. Disclosure: decided after the Run C / ensemble-v1 results were known; motivated by the reported
   BraTS-Africa error pattern, not by any test score of the new candidates.
Amendment 15 addendum (same time, before any Run D result): in ensemble2 a model and its rotation
variant (same weights) are never combined; 29 combinations of >=2 members.
Amendment 15b — 2026-09-30 ~00:00, user decision before Run D has trained: Run D trains 200 epochs
(cosine schedule over 200 epochs, T_max=200) instead of 400, to see results sooner; its results are then
evaluated exactly as specified in Amendment 15 (VAL -> TEST once; BraTS-Africa zero-shot). The Africa CV
from Run D and ensemble2 run only if the user decides to continue after seeing those results. Disclosed:
Run D has half of Run C's training epochs.
Amendment 15b decision — 2026-10-01 23:05: after seeing Run D's BraTS and BraTS-Africa zero-shot results, the user chose to continue: Africa CV from Run D (+15md option) and ensemble2 are released.

### Amendment 15c — 2026-10-01 ~23:13, secondary classification with Run D masks (post-hoc, disclosed)
Run D (runD_rot) zero-shot Africa masks exist (146). The Amendment 14A pipeline is re-run unchanged (same 54 features, same LR C=0.1 / RF 500, same repeated stratified 5-fold ×10, same bootstrap) with mask source `ours_runD_rot`. This is a post-hoc secondary analysis: Run D itself was added after the primary results were known. The primary classification result stays the Run C masks (AUC 0.762); the Run D result is reported alongside it, marked †, whichever way it comes out.

### Amendment 16 — 2026-10-01 ~23:28, classification: tumour-location features (post-hoc extension, disclosed; written before any location feature is computed)
User: "do best and make for me good results". Motivation is a-priori radiology, not a feature search: meningiomas are extra-axial and dural-based (touch the brain surface), 4th-ventricular/ventricular masses sit midline and (for the 4th ventricle) low and posterior, while diffuse gliomas are intra-axial in the white matter. The 54 Amendment-14A features contain no location information.
- **Exactly these 8 features are added** (one set, fixed now; no other set will be tried): brain mask = any of the 4 sequences ≠ 0, holes filled; depth = Euclidean distance to the brain-mask boundary (1 mm voxels); R = max depth in the brain. (1) fraction of WT voxels with depth ≤ 5 mm; (2) same for TC; (3) min depth of TC / R; (4) mean depth of WT / R; (5) |WT-centroid x − brain-centroid x| / brain half-width (L–R); (6) WT-centroid A–P position within the brain bounding box (0 = posterior); (7) WT-centroid S–I position (0 = inferior); (8) distance WT centroid → brain centroid / R. Empty region → NaN (median-imputed inside the CV, as before).
- Same classifiers, same repeated stratified 5-fold ×10, same bootstrap, same seeds. Mask sources: ours_runC (primary of this extension), expert (reference), ours_runD_rot (secondary).
- Reported **whatever the result**, next to the 54-feature results (which stay in the paper), marked post-hoc.

### Amendment 17 — 2026-10-01 ~23:32, Run D2: Run D trained to Run C's epoch budget (post-hoc, disclosed; written before Run D2 exists)
User: "do best and make for me good results". Run D was cut to 200 epochs (Amendment 15b) while Run C had 400, so epochs are confounded with normalisation/augmentation in the Run C vs Run D comparison, and Run D may be under-trained.
- **Run D2** = Run D's final weights (`runD_in_z_last.pth`, epoch 200) + one more 200-epoch cosine cycle (warm restart: fresh AdamW at lr 2e-4, T_max 200, eta_min 1e-7), identical flags otherwise (3D DWT, F 24, deep supervision, InstanceNorm, z-score, strong aug, seed 42, batch 2 × accum 4) → 400 epochs in total. Checkpoint = best-by-validation epoch of the new cycle. Starts only after the CV from Run D has finished (one GPU).
- Evaluation exactly as Run D: `eval_run.sh runD2` (VAL nib/both → Dice sweep → TEST once), `eval_run_rot2.sh runD2_rot runD2`; BraTS-Africa zero-shot of the VAL-better of the two. All reported whatever the result.
- **Gate (validation only)**: if the best VAL Dice of {runD2, runD2_rot} exceeds that of {runD, runD_rot} (0.8940), then (a) **ensemble3** = the ensemble2 procedure with runD/runD_rot replaced by runD2/runD2_rot (29 combinations, VAL → TEST once), and (b) the BraTS-Africa CV is repeated from Run D2 (label MD2, same split, recipe, selection; per-fold option {MD2, MD2+MC, MD2+M1, MD2+MC+M1}). If the gate fails, neither runs and that is reported.
- The BraTS 2021 headline remains the candidate with the highest VAL Dice among all candidates; every candidate's TEST result is reported.
- **Amendment 16 result (2026-10-02 00:1x)**: no gain. LR/RF AUC with location features: Run C 0.758/0.747 (was 0.762/0.758), expert 0.814/0.782 (was 0.826/0.778), Run D 0.714/0.751. Reported in the paper as a negative result; the 54-feature analysis stays primary.

### Amendment 18 — 2026-10-04, classification with the baselines' masks (post-hoc, disclosed; written before any of these results exist)
For a comparison table the user asked for, the Amendment-14A classification is repeated, unchanged (same 54 features, same LR C=0.1 / RF 500, same repeated stratified 5-fold ×10, same bootstrap, same seeds), with the zero-shot BraTS-Africa masks of nnU-Net and Swin UNETR (their saved predictions, label conventions as in their scoring). Reported whatever the result, marked post-hoc; the primary classification result stays ours (Run C masks).
- **Amendment 18 result (2026-10-04)**: AUC (LR / RF) with nnU-Net masks AUC 0.770 [0.688, 0.845] / AUC 0.749; with Swin UNETR masks AUC 0.767 [0.686, 0.843] / AUC 0.760. Mask alignment verified (WT Dice identical to their scoring).

### Amendment 19 — 2026-10-04, three more baselines on our split (written before any of them is trained)
User: "compare with 5 papers, not just 2" -> train 3D U-Net, SegResNet and UNETR on the frozen split, so that five
published methods (with nnU-Net and Swin UNETR) are compared head-to-head. Added after our results were known; disclosed.
- Code: authors' official implementations (repos and versions in ~/code_for_papers/README.txt): 3D U-Net = pytorch-3dunet
  UNet3D default config (original is Caffe); SegResNet = MONAI SegResNet with the official MONAI BraTS tutorial config
  (blocks_down 1-2-2-4, init_filters 16, dropout 0.2); UNETR = MONAI UNETR with the official UNETR config (feature 16,
  hidden 768, mlp 3072, 12 heads, perceptron patch embedding, instance norm, residual blocks).
- Training: the SAME official BRATS21 pipeline as the Swin UNETR baseline (copy in ~/brats2021/baselines/extra3/BRATS21, only
  the network switch added): datalist_frozen_train1000_val126.json, channel order flair/t1ce/t1/t2, AdamW 1e-4, warmup-cosine,
  300 epochs, batch 1, validation every 25 epochs, AMP; crops 96^3 (3D U-Net, UNETR, as Swin) and 224x224x144 with squared
  Dice loss (smooth_nr 0, smooth_dr 1e-5) for SegResNet (its tutorial's crop and loss). Checkpoint = best-by-validation (model.pt).
- Inference/scoring exactly as Swin UNETR: sliding window at the training crop, overlap 0.6, sigmoid > 0.5, no TTA, no
  post-processing; BraTS 2021 test (125) scored ONCE with score_swin_test.py; BraTS-Africa zero-shot with score_labelmaps.py;
  classification with the Amendment-14A pipeline on their zero-shot masks. No BraTS-Africa fine-tuning CV for these three.
- All three reported whatever the result, including if they beat Wavelet U-Net++.
- Compatibility fix (2026-10-04, before training): trainer.py casts the bool label masks to float before the loss (MONAI 1.6 squared Dice cannot take bool); identical values for the plain Dice loss.

### Amendment 20 — 2026-10-04 ~15:00, same-conditions comparison of all six methods (user: "the comparison must be conducted under the same conditions, so that the results are fair and directly comparable"); written before any result below exists. Post-hoc with respect to earlier results; disclosed (†).
Why: so far the methods were compared at their own default inference — nnU-Net with mirroring TTA; Swin UNETR / 3D U-Net /
SegResNet / UNETR with no TTA (overlap 0.6, constant blending); ours as Ensemble 3 (3 models, 8 flips x 2 axis orders x 2
rotations, validation-chosen post-processing). Those results are kept and reported unchanged (headline rule unchanged:
Ensemble 3). This amendment adds ONE identical evaluation setting for all six methods.
- Methods: nnU-Net, Swin UNETR, 3D U-Net, SegResNet, UNETR, Wavelet U-Net++.
- Training: as already done/under way (same 1,000 cases, each method's official recipe). Identical training is not possible
  without changing the methods (nnU-Net configures its own training); stated as a limitation.
- Inference "M" (identical for all): ONE model (no ensembles); NO test-time augmentation of any kind (no flips, rotations or
  axis-order averaging; ours uses the single "nib" axis order, its historical single-pass mode); sliding window at the
  network's own training patch size with 50% overlap and Gaussian blending (ours 128^3 stride 64; nnU-Net tile_step_size 0.5,
  use_gaussian; MONAI models overlap 0.5, mode "gaussian"); per-region sigmoid > 0.5; label map with the same nesting for all
  (WT->2, TC->1, ET->4, i.e. ET within TC within WT); NO post-processing of any kind. Each model keeps its own input
  preprocessing (part of the trained model).
- Checkpoint, chosen on the same 126 validation cases for all: MONAI models = their best-by-validation model.pt (unchanged);
  nnU-Net = checkpoint_final or checkpoint_best, whichever has the higher mean validation Dice under M; ours = the single
  model with the highest mean validation Dice under M among base (ep253), Run A, Run C, Run D, Run D2 (each at its
  best-by-validation checkpoint). Validation Dice under M is computed for all six and reported.
- BraTS 2021 test (125): each method under M scored ONCE, same scorer (brats_gbm.eval.score_case via score_swin_test.py);
  ours vs each baseline: paired bootstrap (10,000, seed 0) + Wilcoxon, mean and per region, Dice and HD95.
- BraTS-Africa zero-shot: all 146 subjects under M, score_labelmaps.py; glioma (95) primary.
- Classification: Amendment-14A pipeline (54 features; LR C=0.1, RF 500 trees; stratified 5-fold x10) on each method's
  zero-shot masks under M.
- BraTS-Africa fine-tuning CV: 3D U-Net, SegResNet and UNETR are fine-tuned on the same folds (cv/cv_split.json) with the
  Swin UNETR fine-tuning recipe (150 epochs, validation every 5, start = their BraTS model). Then the adapted fold models of
  all six (nnU-Net, Swin UNETR, ours from the start model chosen above, the three new) are inferred under M on their
  fold-test gliomas and pooled over the 95. If our chosen model has no existing CV, its CV is run with our existing recipe.
- Reported whatever the result, including where baselines beat Wavelet U-Net++. The six-method table under M becomes the
  main comparison table; the default-setting results stay in their tables.
- Correction found while preparing this: the nnU-Net predictions (test and BraTS-Africa) were made with nnUNetv2_predict,
  which does NOT apply post-processing (nnUNetv2_apply_postprocessing was never run; nnU-Net derives it only from 5-fold CV,
  which a "fold all" model lacks). The paper's phrase "its native post-processing" is wrong and will be corrected.
- Addendum (2026-10-04 ~15:00, user decision before any setting-M result existed): the paper stays as it was — headline
  Ensemble 3 (0.899) and the comparison at each method's default settings (the three Amendment-19 baselines are added there
  with their default settings). The setting-M runs are completed and published in the code repository only, as a
  supplementary same-conditions analysis; they do not enter the paper. The nnU-Net post-processing correction stays.
- Addendum 2 (2026-10-04 ~15:00): user "just complete the other 3 papers" -> setting-M runs STOPPED before any TEST or
  BraTS-Africa result under M existed (only VAL for the base model, 0.8897, and inference of some adapted CV folds had run;
  kept on disk, unused). Only Amendment 19 continues: 3D U-Net, SegResNet, UNETR at their default settings.

### Amendment 19b — 2026-10-04 ~15:05, training-collapse rule for the three Amendment-19 baselines (written before any of them is evaluated)
Observed: 3D U-Net (AMP, the pipeline default) trained normally to epoch 23 (loss 0.31-0.35), then collapsed at ~iteration 330
of epoch 24 to all-background output (loss ~0.8-1.0; first validation, epoch 24: Dice 0.0/0.0/0.0) and did not recover.
Sudden death under fp16 mixed precision; pytorch-3dunet's own trainer is fp32. Rule, identical for all three networks:
- A training run is "collapsed" if any validation reports Dice < 0.05 for all three regions, or if it ends with best
  validation Dice < 0.5. A collapsed run is stopped and retrained ONCE from scratch in fp32 (--noamp), all else unchanged.
- 3D U-Net is therefore retrained in fp32 now. SegResNet and UNETR start with AMP (pipeline default) under the same rule.
- Collapsed runs are kept (moved to ~/delete/unused_checkpoints), not used, and reported in the repository.
Unchanged: data, split, optimiser, schedule, epochs, crops, losses, checkpoint rule, inference and scoring (Amendment 19).
- **Amendment 19/19b result (2026-10-06)**: all three trained (3D U-Net and UNETR in fp32 after AMP collapses; SegResNet with AMP).
  BraTS 2021 test (125, scored once): SegResNet 0.905 [0.884, 0.923], UNETR 0.881 [0.860, 0.899], 3D U-Net 0.862 [0.835, 0.887].
  Ensemble 3 vs each: SegResNet -0.006 [-0.015, +0.002] (n.s.), UNETR +0.018 [+0.002, +0.033], 3D U-Net +0.037 [+0.022, +0.053].
  BraTS-Africa zero-shot gliomas: SegResNet 0.846, UNETR 0.799, 3D U-Net 0.805. Classification AUC (LR): 0.772 / 0.748 / 0.771.

### Amendment 21 — 2026-10-10 ~11:04 (+08:00), non-wavelet control for Run C (peer review; written before the control is trained)
Reason: an external review noted that no U-Net++ with conventional downsampling was trained, so the contribution of the
wavelet transform itself is not isolated. One control is added (user decision 2026-10-10: run the control; no extra seeds,
no no-TTA test scoring).
- Candidate `ctrl_maxpool3d`: Run C's frozen code snapshot (experiments/runC_dwt3d/code, copied) with ONE change, the
  encoder downsampling: `downsample="maxpool3d_matched"` = 2x2x2 max-pooling followed by a 1x1x1 convolution C -> 8C, so every
  encoder block receives the same tensor shape as with the 3D Haar DWT (8C channels at half resolution); decoder upsampling
  (2,2,2). Everything else as Run C: width F=24, deep supervision, BatchNorm, min-max inputs, standard augmentation,
  focal-BCE + Dice loss, AdamW 2e-4, 400-epoch cosine, batch 2 x accumulation 4, AMP, seed 42, same split, centre-crop
  validation, checkpoint = best validation epoch. Parameters 34,355,451 vs 33,960,891 for Run C (+394,560, +1.2%: the
  projections); GFLOPs and runtime are measured and reported, not assumed equal.
- Evaluation identical to Run C: eval_run.sh (VAL in both axis orders, Amendment-5 30-setting post-processing grid, highest
  VAL Dice selects order and setting; TEST scored ONCE); no rotation TTA, so the comparator is Run C without rotation TTA.
  BraTS-Africa zero-shot (95 gliomas primary, 51 other neoplasms separately) with the BraTS-VAL-selected settings.
- Primary comparison (fixed now): Run C minus control, mean Dice on the 125 BraTS 2021 test cases, paired bootstrap (10,000,
  fresh seed-0 generator) 95% CI + two-sided Wilcoxon. Interpretation: the wavelet transform is said to improve BraTS 2021
  Dice only if the CI excludes zero in Run C's favour; if the CI includes zero, "no difference detected" (equivalence is not
  tested); if it favours the control, that is reported as such. Secondary: per-region Dice, HD95 (defined-only and with
  undefined values = 373.13 mm), BraTS-Africa zero-shot glioma mean Dice (same tests). One seed only.
- Reported in the paper and the repository whatever it shows. It does not change the headline (Ensemble 3) or any existing
  result, and the control is not added to any ensemble search.
- Collapse rule as Amendment 19b (validation Dice < 0.05 for all regions, or best < 0.5 -> retrain once from scratch in
  fp32). Crashes: resume from _last.pth, as for Run C.
- Code check before launch: all five evaluated checkpoints (base, Runs A, C, D, D2) load strictly and give bit-identical
  outputs with the extended model code; 48/48 tests pass; control peak memory 12.6 GiB at batch 2, 128^3, AMP.

### Amendment 22 — 2026-10-10 ~12:57 (+08:00), comparison baselines changed by the authors (written before any new baseline run)
Author decision (user, 2026-10-10): nnU-Net and Swin UNETR are removed from the paper's comparison tables and text and
replaced by SegResNet and 3D U-Net (already trained and scored once under Amendment 19) and by the deeply supervised 3D U-Net
of Henry et al. (BrainLes 2020, LNCS pp. 327-339, doi 10.1007/978-3-030-72084-1_30; official code linked in the paper,
github.com/lescientifik/open_brats2020). UNETR stays. nnU-Net and Swin UNETR results are NOT deleted: they stay in this
repository and the paper states in one sentence that both were also evaluated and where their results are. Our model,
checkpoints and all existing results are unchanged. Candidate check (2026-10-10): Scale Attention Network (Yuan, BrainLes
2020/2021) was proposed but excluded because no official public code could be found; MedNeXt, UNETR++, nnFormer and TransBTS
were not used (author choice).
- Henry et al., reproduced on our split: official code at e5f98961 (repository has no licence file: used, not
  redistributed) with two changes only: (i) the dataset uses our fixed split (train 1000 / val 126; test never loaded)
  instead of the repository's random 5-fold split; (ii) the Ranger optimiser (lessw2020, c82a3046) has its deprecated
  positional-scalar tensor calls rewritten in keyword form for current PyTorch (identical arithmetic). Configuration = the
  paper's pipeline A, "U-Net like" variant (its ablation showed no clear benefit of the attention block or patient removal):
  EquiUnet, width 48, group normalisation, deep supervision, Dice loss, Ranger lr 1e-4, 200 epochs with cosine decay in the
  second half, batch 1, fp16, then stochastic weight averaging (5 x 30 epochs, lr 5e-5), i.e. `--arch EquiUnet --width 48
  --deep_sup --swa`. The final model is the SWA model (the repository's rule); validation is used only by its own
  monitoring. One model, not the authors' challenge ensemble. Inference: the repository's inference.py with its TTA
  (`--tta`), sigmoid > 0.5, its label assembly, no further post-processing. TEST (125) scored ONCE with our scorer.
- BraTS-Africa for Henry et al.: zero-shot with the same inference; classification with its zero-shot masks (Amendment 14A
  pipeline).
- BraTS-Africa 5-fold CV fine-tuning (Amendment 6 folds and rules: each method's own recipe from its BraTS weights, budget
  ~9,600 samples = 150 epochs over the 64 TRAIN cases, same scorer) for SegResNet and 3D U-Net (MONAI pipeline exactly as for
  Swin UNETR: val every 5 epochs, model.pt = best by val, their BraTS inference; 3D U-Net in fp32 as trained) and for
  Henry et al. (its code: start from the SWA weights, 150 epochs, Ranger lr 1e-4, no SWA, so the repository keeps the final
  model; its inference with TTA).
- Efficiency for SegResNet, 3D U-Net and Henry et al. with the existing benchmark rules (each under its own inference).
- Paired statistics of our configurations against each baseline as before (bootstrap 10,000 + Wilcoxon, unadjusted).
- Collapse rule as Amendment 19b. All results are reported whatever they show.

### Amendment 23 — 2026-10-13 ~17:15 (+08:00), BraTS-Africa zero-shot of the headline Ensemble 3 (user: "you can try A"); written before any Ensemble 3 BraTS-Africa prediction exists
- Model: the headline Ensemble 3 exactly as frozen by Amendment 17 (selection.json: ep253 + runC_rot + runD2_rot,
  probability average, each member with its own BraTS-VAL-chosen inference mode — ep253 nib; runC_rot both+rot;
  runD2_rot both+rot — then WT components, floor x1, ET min-volume 100). Nothing is re-tuned on BraTS-Africa.
- Data, preprocessing and scorer as Amendments 1-2/14 (146 gated subjects; gliomas (95) primary, other neoplasms (51) as a
  stress test). Each subject predicted once; masks saved for the classification pipeline (Amendment 14A), run afterwards.
- Reported: mean Dice with bootstrap CI; paired (bootstrap 10,000 + Wilcoxon, unadjusted) Ensemble 3 minus Run D2 + rot,
  Run D + rot, SegResNet and Henry et al. on the 95 gliomas. Reported whatever it shows; it does not replace Run D2 as the
  model fine-tuned in the CV (Amendment 6/18), which stays as run.
- Disclosed deviations in Amendment 22's Henry et al. evaluation (found 2026-10-12, before its TEST scoring): (i) upstream
  src/inference.py with --tta appended both the padded crop and the uncropped volume (torch.stack size mismatch on every
  case); the extra append was removed, TTA unchanged; (ii) evaluate.sh read RUN_PATH while train.sh wrote RUN_DIR (path file
  created; final SWA run moved to henry_dsunet/final_run/ so the CV step could not move the initial weights); (iii) the VAL
  sanity gate selected 0 cases because the split CSV has CRLF line ends (fixed; TEST untouched until the gate passed:
  VAL5 ET/TC/WT .871/.934/.944). Details in henry_dsunet/SOURCES.txt.

### Amendment 24 — 2026-10-14 ~00:30 (+08:00), TransBTS added as a comparison baseline (user: "download code ... and run"); written before any TransBTS training
- Method: TransBTS (Wang et al., MICCAI 2021, LNCS 12901:109-119), official code github.com/Wenxuan-1119/TransBTS @ 7dcfc31
  (Apache-2.0; pristine copy in ~/code_for_papers/6_TransBTS/official_code; working copy experiments/transbts/code).
- Recipe = the repository's train.py defaults: TransBTS (downsample 8x, skip connections, learned PE), z-score over the brain
  mask (preprocess.py process_f32b0), pad 155->160, random 128^3 crop + flips + intensity shift, softmax Dice loss, Adam lr 2e-4
  (poly 0.9 per epoch), weight decay 1e-5, amsgrad, fp32, 1000 epochs, effective batch 8, no validation, final (last-epoch)
  model. Trained on our 1,000 TRAIN cases only (fixed split; VAL and TEST never loaded in training).
- Single-GPU adaptations (not changing the method): batch 2 x 4 gradient-accumulation steps (= the authors' 4 GPUs x 2, BatchNorm
  per 2 samples as in their DDP without SyncBN); preprocessing computed on load from the original NIfTI instead of cached
  pickles (verified bit-identical on BraTS2021_00002); resume enabled (upstream hard-coded resume=''), rolling resume checkpoint.
- Inference = the repository's predict.py: 8 overlapping 128^3 crops, 8-flip TTA, softmax average, argmax, labels 1/2/4,
  no post-processing (its post_process flag is a no-op). Fixes: masks saved with the reference image geometry; ZeroDivision
  after the last case with TTA. Scored with our common scorer.
- Before the one-time TEST: sanity gate on the first 5 VAL cases (mean WT >= 0.70 and mean Dice >= 0.60), as for Henry et al.
  Then: BraTS 2021 TEST (125) ONCE; BraTS-Africa zero-shot (146; gliomas primary); classification with its masks (Amendment 14A);
  BraTS-Africa 5-fold CV fine-tuning under Amendment 6 rules (from its BraTS weights, 150 epochs over the 64 TRAIN cases, its
  recipe, final model, its inference); efficiency benchmark. Paired statistics vs our configurations as before.
- Failure rule: a crash or NaN loss resumes from the last rolling checkpoint (at most 3 times); if it still fails, it is
  reported as not trained. All results reported whatever they show.

Amendment 24 addendum 2 — 2026-10-14 ~12:52 (+08:00), TransBTS restarted with the repository's background-inclusive Dice loss
(user decision, "Restart, authors' 2nd loss"); written before the restarted run exists and before any TransBTS VAL/TEST prediction.
- Observed in the first run (default criterion softmax_dice = Dice of classes 1, 2, 3 only, no background term), from TRAIN-batch
  logs only: class-1 (NCR) Dice 0.002 at epoch 5 -> 0.008 at epoch 137 while classes 2/3 rose to 0.84/0.85. A check of the epoch-99
  checkpoint on tumour-centred 128^3 crops of 2 VAL cases (BraTS2021_00049, _00062; CPU, no score kept, nothing tuned on it):
  argmax = class 1 on ~1.98 M of 2.10 M voxels (class-1 softmax ~0.47 in background), ED/ET Dice ~0.96, NCR ~0. I.e. the default
  loss gives background no gradient and the run collapsed to "background = NCR" (WT/TC would be ~0; the VAL5 gate would fail).
  Labels/cache verified correct (cache = NIfTI exactly; label 1 present in 38/40 sampled TRAIN cases, ~12k voxels).
- Change: the full 1000-epoch run restarts FROM SCRATCH with --criterion softmax_dice2, the alternative loss in the authors' own
  models/criterions.py (same Dice of classes 1-3 + Dice of class 0). Everything else unchanged (recipe, seed 1000, batch 2 x 4,
  1000 epochs, final model, inference). The same criterion is used for the BraTS-Africa CV fine-tuning. Early check (TRAIN logs
  only): class-1 Dice must clearly rise within the first ~10 epochs; if it does not, report to the user before continuing.
- The first run is kept, not used: code/checkpoint/FAILED_softmax_dice_TransBTS_fixedsplit_20261014 (+ its train_*.log files).
  Reported as a deviation from the repository default in the paper/repository if TransBTS is included.

### Amendment 25 — 2026-10-14 ~01:00 (+08:00), one standard procedure for every comparison method (user: "all the papers I'm comparing against follow exactly the same conditions and evaluation protocol ... apply the same procedure to all of them"; training rule chosen by the user: each method's own official recipe)
Applies to every baseline already run and to any added later; the user decides afterwards which methods the paper reports.
1. Code: the authors' official code at a recorded commit, kept pristine in ~/code_for_papers/<n>_<method>/official_code;
   our launchers and every patch (as a diff) in our_run_scripts/. Patches only for running on our hardware/data or for bugs,
   never changing the method; each is disclosed.
2. Training: the authors' own published recipe (architecture, preprocessing, augmentation, loss, optimiser, schedule, epochs,
   model-selection rule). If the authors published no BraTS recipe, the official MONAI BraTS 2021 pipeline (as used for
   SegResNet, UNETR, 3D U-Net). Data: our 1,000 TRAIN cases only; VAL (126) only where the method's own rule selects with
   validation; TEST (125) never seen before step 4. One run, no seed search.
3. Sanity gate (methods added from now on): the method's own inference on the first 5 VAL cases, common scorer; the test is
   scored only if mean WT >= 0.70 and mean Dice >= 0.60.
4. BraTS 2021 TEST (125) scored ONCE with the method's own official inference and the common scorer (Dice ET/TC/WT, mean;
   HD95 and HD95* = undefined set to 373.13 mm); bootstrap CI; paired bootstrap (10,000) + Wilcoxon vs our configurations.
5. BraTS-Africa zero-shot: same 146 gated subjects (95 gliomas primary, 51 other neoplasms as stress test), same staging and scorer.
6. Classification (glioma vs other neoplasm): features from its zero-shot masks, Amendment 14A pipeline.
7. BraTS-Africa 5-fold CV fine-tuning under Amendment 6 rules (same folds; from its BraTS weights; ~9,600 samples = 150 epochs
   over the 64 TRAIN cases; its own recipe and selection rule; its own inference); pooled out-of-fold scoring.
8. Efficiency: params, GFLOPs per forward pass, passes per case, s/case (mean of 10 test volumes after one warm-up, CUDA-
   synchronised, its own inference incl. TTA, and without TTA) and peak memory, on the otherwise idle RTX 5090.
Audit at this date: missing = UNETR step 7; step 8 for SegResNet, UNETR, 3D U-Net, Henry et al. Both are now run under this
amendment (efficiency before TransBTS training, on the idle GPU); TransBTS follows all steps (Amendment 24).
- Amendment 24 addendum (2026-10-14 ~01:12, before any TransBTS training): further fixes in the TransBTS code, disclosed:
  predict.py wrote 160-slice label maps (padded depth) for 155-slice images -> saved at the prediction depth; test.py silently
  continued with random weights when the checkpoint was missing -> now stops; train.py gets --init_weights (weights only) for
  the BraTS-Africa CV start. Evaluation script transbts/evaluate.sh follows Amendment 25 steps 3-8.

### Amendment 26 — 2026-10-14 ~01:14 (+08:00), nnFormer added under the standard procedure (Amendment 25) (user: "download in my project same process like other"); written before any nnFormer preprocessing or training
- Method: nnFormer (Zhou et al., IEEE TIP 32:4036-4045, 2023), official code github.com/282857341/nnFormer @ d48aaeb (MIT);
  pristine copy ~/code_for_papers/8_nnFormer/official_code; working copy experiments/nnformer.
- Recipe = the authors' brain-tumour configuration: nnU-Net-v1 pipeline (plan_and_preprocess: crop, z-score over non-zero,
  target spacing), trainer nnFormerTrainerV2_nnformer_tumor (embedding 96, depths 2-2-2-2, heads 3-6-12-24, window 4-4-8-4,
  patch 128^3, batch 2, no deep supervision, SGD lr 1e-2 poly 0.9, 1000 epochs x 250 iterations, nnU-Net augmentation and
  CE+Dice loss, mixed precision as in the code); model selection = its own rule, model_best (EMA validation Dice) on VAL,
  which is the checkpoint its inference uses (-chk model_best).
- Data: task Task003_tumor in the authors' layout; imagesTr = our TRAIN 1,000 + VAL 126 (the trainer trains on TRAIN and
  validates on VAL via our split file: the authors' hard-coded MSD case lists are replaced, disclosed patch); imagesTs = TEST
  125, never preprocessed with or seen by training. Channels in the authors' order (FLAIR, T1w, T1gd, T2w); labels converted
  to the authors' convention (1 edema, 2 non-enhancing/necrotic core, 3 enhancing) and predictions converted back to BraTS
  1/2/4 for the common scorer. Note: nnU-Net's planning fingerprint uses all imagesTr cases (TRAIN + VAL), as the framework does.
- Inference: nnFormer_predict 3d_fullres, its defaults (sliding window step 0.5, Gaussian, mirroring TTA, mixed precision),
  no post-processing (its tumour evaluation uses raw predictions).
- BraTS-Africa CV (Amendment 6 budget for nnU-Net-type trainers): 19 epochs x 250 iterations x batch 2 = 9,500 samples per
  fold, from its BraTS weights, validation on the fold's 12 VAL cases, its own model_best rule.
- Then Amendment 25 steps 3-8 exactly as for every method. Runs after TransBTS (fixed GPU order); preprocessing is CPU-only.
- Amendment 26 addendum (2026-10-14 ~01:30, before any nnFormer training): the authors' tumour trainer loads
  /home/xychen/jsguo/weight/tumor_pretrain.model by default (not released; origin unknown, possibly trained on MSD/BraTS 2016-17
  cases that are part of BraTS 2021). Disabled (load_pretrain_weight=False): nnFormer trains from scratch on our 1,000 TRAIN
  cases like every other method. Environment: batchgenerators 0.21 (the nnU-Net-v1-era version the code imports).
- Amendment 26 addendum 2 (2026-10-17 ~12:15, after training ended 11:48, before ANY nnFormer prediction or score existed):
  the first VAL5 sanity run crashed while loading model_best.model (PyTorch >= 2.6 torch.load default weights_only=True rejects
  the numpy scalars nnFormer saves in its checkpoints) -> SANITY_FAILED with no prediction made. Fix (loading only, as for
  TransBTS on 10-14; user OK "Ok go"): weights_only=False in training/model_restore.py:168, run/load_pretrained_weights.py:21,
  training/network_training/network_trainer.py:344 (originals *.orig; own checkpoints only). Model, inference settings and
  procedure unchanged. SANITY_FAILED marker moved to ~/delete; evaluation restarted 12:17 from step 0 (VAL5 gate).

### Amendment 27 — 2026-10-14 ~16:30 (+08:00), Run E: Run D's recipe trained for 400 epochs in one cosine cycle (user: "prepare for task 1 and 3 after 17 October"; options chosen: "Lessons 1+3 only", "400 epochs"); written before any Run E file or weight exists
- Motivation (from VAL/TEST/log evidence already reported, no new look at TEST): lesson 1 = the 3D-DWT width-24 deep-supervision
  network (Run C/D architecture) gave the largest gain; lesson 3 = more epochs helped only with strong augmentation (Run D 200 ep
  -> D2 200+200 warm restart: VAL 0.8890 -> 0.8942), not without it (ep253 300 ep -> Run A 400 ep: VAL 0.8881 -> 0.8885).
- **Run E** (tag runE_long400) = Run D exactly (frozen code experiments/runD_in_z/code, arch_flags.txt = --downsample dwt3d
  --base-filters 24 --deep-supervision; --norm instance --normalisation zscore --aug strong; AdamW 2e-4, batch 2 x accumulation 4,
  focal-BCE + Dice, seed 42, same split 1000/126/125) with ONE change: 400 epochs from scratch, one cosine cycle (T_max 400,
  eta_min 1e-7) — i.e. Run D as first pre-registered in Amendment 15 before 15b cut it to 200. Checkpoint = best epoch by the
  training script's VAL metric (as every run). Nothing else changes (no new loss, sampling, logging or selection rule).
- Start: after the TransBTS and nnFormer evaluations have finished (EVALUATED, FAILED or SANITY_FAILED for both) and not before
  2026-10-17, so that it does not slow them or disturb their efficiency timing. Fault-tolerant resume as Run D.
- Evaluation = exactly Run D2's: eval_v2/eval_run.sh runE (VAL 126 -> post-processing rule -> TEST 125 once), then
  eval_v2/eval_run_rot2.sh runE_rot runE (rotation TTA; VAL -> TEST once). Every candidate's TEST is scored once and reported
  whatever it shows. Judged on VAL against the existing models (Run D2 rot VAL 0.8950, Run C rot 0.9006, Ensemble 3 0.9079).
  The paper's headline (Ensemble 3) is NOT changed by this run; any new ensemble, BraTS-Africa evaluation or paper change needs
  a further amendment and the user's decision.

### Amendment 28 — 2026-10-17 ~12:35 (+08:00), VAL-only investigation of Ensemble 3 (user: "investigate and improve Ensemble 3 using only the training and validation data ... keep the current TEST results unchanged"); written before any of the analyses below was run
- Data: the saved VAL probability maps (126 cases, float16 [ET,TC,WT]) of the three Ensemble 3 members (eval_ep253/probs_val,
  eval_v2/work_runC_rot/probs_val, eval_v2/work_runD2_rot/probs_val) and the VAL ground truth. No inference, no retraining.
  The TEST probability maps and TEST labels are NOT read by any script of this amendment; the scored TEST results stay as they are.
- Caveat stated in advance: VAL already chose every checkpoint, the members and the post-processing of Ensemble 3 (870
  configurations, Amendment 17), so VAL is reused and gains on it are optimistic; any revised configuration needs a fresh
  independent test set for confirmation (none is scored here).
- Reference E0 = Ensemble 3 exactly (equal mean of the 3 members, thresholds 0.5, wt_policy components, floors x1,
  ET min-volume 100). Its VAL mean Dice must reproduce 0.90787.
- Step 1, error analysis on E0's VAL predictions (descriptive only): per case and region, FP voxels in predicted components
  that touch no GT voxel of that region ("FP blob"), remaining FP ("boundary FP"), FN voxels in GT components that no
  prediction touches ("missed region"), remaining FN ("boundary FN"); for TC also core<->edema confusion (TC FN predicted as
  WT-only, TC FP that is GT edema); empty-GT cases. Worst 10 VAL cases by TC and by WT Dice are listed with their dominant error.
- Candidates (fixed now; 17 + E0):
  LOO (3): each 2-member ensemble with E0's post-processing.
  Weights (6): (w_ep253, w_runC_rot, w_runD2_rot) in {(2,1,1),(1,2,1),(1,1,2),(1,2,2),(2,1,2),(2,2,1)}, normalised, E0 post-proc.
  Component rules on E0 (8), applied after E0's post-processing, ET/TC then re-nested by the hierarchy:
    WTrel f: drop WT components with volume < f x the largest WT component, f in {0.05, 0.10, 0.20};
    TCrel f: drop TC components with volume < f x the largest TC component, f in {0.05, 0.10, 0.20};
    WTconf t: drop WT components whose maximum ensemble WT probability < t, t in {0.8, 0.9} (never the largest component).
  Second stage only if >= 1 weight AND >= 1 component rule each pass the rule below: their combination (best of each) is evaluated.
- Reported for every candidate vs E0 on the same 126 VAL cases: mean and per-region Dice, HD95 (mean over defined values, n
  undefined), paired difference with bootstrap 95% CI (10,000, fresh seed-0 generator per comparison) + two-sided Wilcoxon,
  per-fold differences, and the change on E0's worst 10 cases by TC and by WT. For component rules: GT voxels removed
  (removed components that overlap the GT of that region), number of cases affected.
- Folds: the 126 VAL cases split once into 5 folds by np.random.default_rng(0).permutation (sorted case ids), fixed now.
  No BraTS cross-validation folds exist for our models (single 1000/126/125 split), so these VAL folds are the consistency check.
- A candidate is "reliably better" only if ALL hold: (a) VAL mean-Dice gain > 0 with bootstrap CI lower bound > 0;
  (b) gain >= 0 in >= 4 of the 5 VAL folds; (c) component rules: GT voxels removed <= 0.1 % of all VAL GT voxels of that region
  and no case loses > 0.01 Dice in any region from the removal; (d) nested check: selecting the best of {E0 + all candidates}
  on 4 folds and scoring it on the held-out fold gives a mean held-out gain > 0 over E0. Unadjusted for the 17 comparisons
  (stated in the report). If no candidate satisfies (a)-(d), the report says that nothing reliably improves VAL and E0 stays.
- Code/output: eval_v2/amendment28/ (analysis script, config, per-case CSVs, comparison table, report).
- Amendment 28 addendum (2026-10-17 ~12:50, user clarifications, before any candidate result was seen): (1) E0 must reproduce
  Ensemble 3 exactly (same saved member VAL probabilities, unchanged since the 10-03 selection; equal mean; same post-processing),
  checked per region against ensemble3/ensemble_val_search.csv and reported before any comparison. (2) This is an EXPLORATORY
  validation analysis: folds inside the 126 previously used VAL cases cannot undo the earlier 870-configuration selection.
  (3) In the nested check each outer held-out fold is excluded from every new weight / post-processing choice; choices inherited
  from the earlier full-VAL search (checkpoints, member inference modes, member set, thresholds, wt_policy, floors, ET min-volume)
  are listed explicitly and not re-chosen. (4) Full-VAL scores and pooled outer-fold results are reported separately.
  (5) TEST unchanged; the 8 CPU workers are reduced if they measurably slow the running TransBTS training / nnFormer evaluation.
- Amendment 28 RESULT (2026-10-17 ~13:45, exploratory, VAL only, TEST not read): E0 reproduces Ensemble 3 exactly (ET 0.884133,
  TC 0.907550, WT 0.931933, mean 0.907872; differences 0). No candidate meets rules (a)-(c); nested outer-fold selection gives a
  pooled held-out change of -0.0055 [-0.0168, +0.0006] (rule d not met). E0 (Ensemble 3) is retained unchanged. Errors are
  dominated by boundary / core<->edema confusion (86 % of TC FN are predicted as edema), not FP blobs (TC 3.6 %, WT 2.8 % of
  error voxels). Full output: eval_v2/amendment28/report.txt, comparison_val.csv, worst_cases_val.csv.
- Amendment 28 clarifications (2026-10-17, user): (1) BraTS2021_01624 verified from its label file: GT ET = TC = 0 voxels (edema
  only, 33,560); E0 predicts 0 ET/TC (Dice 1.0, HD95 0); LOO-ep253 / LOO-runC_rot / W112 / W122 predict 457-2,589 ET/TC voxels
  (Dice 0, HD95 undefined, excluded from HD95 means); case kept in all primary results. (2) The nested outer-fold analysis
  selected by mean Dice only (rules (a)-(c) not applied inside the selection folds) and is labelled "mean-Dice-only nested
  selection (exploratory)", not an evaluation of the full prespecified rule. (3) Conclusion wording: none of the tested
  post-processing or re-weighting candidates demonstrated a reliable improvement in this exploratory analysis. E0 (Ensemble 3)
  retained. Details: eval_v2/amendment28/clarifications.txt.

- Amendment 24 addendum 3 (2026-10-18, user-approved): the TransBTS efficiency benchmark failed (bench_transbts.py default
  --ckpt pointed to the collapsed first run). It is re-run with the identical script and settings and --ckpt set to
  TransBTS_dice2_20261014/model_epoch_last.pth (the checkpoint every TransBTS score used), after Run E has finished and with no
  job of ours on the GPU (transbts/bench_rerun_after_runE.sh). No retraining, no code edit.
- Amendment 27 addendum (2026-10-18, user): Run E runs and is evaluated unchanged (VAL -> TEST once -> rotation-TTA variant).
  Additionally, once Run E's VAL selection exists, Run E is compared with Run D on VAL only under the same evaluation variant
  (runE vs runD, runE_rot vs runD_rot; eval_v2/runE_vs_runD/compare_val.py, each run with its own VAL-selected post-processing,
  verified to reproduce Run D's recorded VAL 0.8937 / 0.8940). Run E's TEST result is not used to select or modify Run F
  (eval_v2/proposal_20261017, pending).

### Amendment 29 — 2026-10-19 ~22:25 (+08:00), Run F: phenotype-balanced case sampling on the Run D recipe (user: "ok go", option A, after the 10-19 review); written before any Run F file, weight or result exists
- Motivation (VAL / TRAIN only; TEST not used): eval_v2/next_step_20261019/val_decomposition.txt shows that the 8 VAL cases with a
  predominantly non-enhancing core (GT ET/TC < 0.25) have TC Dice 0.42-0.45 in all nine of our models (DWT / max-pool, Run C / D
  recipes, 200 / 200+200 / 400 epochs, TTA, Ensemble 3) and carry 32-40 % of the TC shortfall; trainfit/ shows Run C under-fits this
  phenotype on its own TRAIN cases (no-TTA TC 0.713 on the 65 TRAIN cases with ET/TC < 0.25 vs 0.931 on 65 random other TRAIN cases;
  VAL other cases 0.926). Run E (400 epochs, uniform sampling) left the subgroup unchanged (0.421 vs 0.421). Caveat: the 0.25 cutoff
  was derived from the VAL error analysis (VAL results for this change are optimistic); annotation ambiguity (NCR vs edema) is an
  alternative explanation that this experiment cannot exclude.
- **Run F** (tag runF_phenoD; dir experiments/runF_phenoD) = Run D exactly (code = a COPY of the frozen experiments/runD_in_z/code;
  arch_flags.txt = --downsample dwt3d --base-filters 24 --deep-supervision; --norm instance --normalisation zscore --aug strong;
  AdamW 2e-4, cosine T_max 200, batch 2 x accumulation 4, focal-BCE + Dice, AMP, seed 42, 200 epochs, same split, random 128^3
  crops, checkpoint = best training-time centre-crop VAL metric) with ONE change: training cases are drawn by
  torch WeightedRandomSampler(replacement=True, num_samples=1000 per epoch, generator seeded 42) instead of DataLoader shuffle.
  Target group = the 65 TRAIN cases with GT TC > 0 and ET/TC < 0.25, computed from TRAIN labels only (list + sha256 frozen in the run
  dir before training); weight ratio r = 0.20 x 935 / (0.80 x 65) = 3.596, i.e. expected target share 20 % of draws (6.5 % now),
  other cases equal weight. New flags --case-sampling {uniform,phenotype} (default uniform = unchanged code path),
  --target-list, --target-share; target draws per epoch are logged. Pre-launch checks: diff limited to the sampler; sampler dry run
  (share 20 +- 1 % over 200 epochs); smoke run on the copy; tests.
- **Control** = Run D as already trained (same frozen code, same seed, uninterrupted run; saved VAL probabilities reused). One run per
  arm: seed-to-seed variation is not estimated. Run F tests the whole sampling strategy (weighting + with-replacement draws); it does
  not isolate the weighting alone.
- **Evaluation = VAL ONLY, no TEST, no BraTS-Africa, no new ensemble.** eval_v2/eval_run_valonly.sh = eval_run.sh steps 1-3 (VAL in
  nib and both orders, Amendment-5 30-setting Dice sweep, higher VAL Dice selects) with the TEST step removed; then the rotation-TTA
  variant runF_rot on VAL only (eval_run_rot2.sh steps 1-3). Comparison: eval_v2/runE_vs_runD/compare_val.py runF runD and
  runF_rot runD_rot. TRAIN-fit check (mechanism): evaluate_brats.py --no-tta --axis-order nib with each run's VAL-selected
  post-processing on trainfit/train_target65.txt and train_other65.txt, for Run D (control, run before Run F results exist) and Run F.
- **Success criteria (fixed now), Run F vs Run D and Run F rot vs Run D rot, VAL 126, each with its own VAL-selected post-processing:**
  T1 targeted (EXPLORATORY, n = 8, VAL-derived cutoff): mean Dice_TC on the 8 VAL cases with ET/TC < 0.25 improves by >= +0.05 and
     improves in >= 5 of 8;
  T2 mechanism (TRAIN): no-TTA Dice_TC on the 65 TRAIN target cases improves by >= +0.05, and the 65 other TRAIN cases' mean Dice
     does not fall by more than 0.01;
  T3 reported only: Dice_TC on the 18 VAL cases with ET/TC < 0.5;
  O1 overall: paired mean-Dice difference >= 0 (point estimate) and bootstrap 95 % CI lower bound > -0.005;
  O2 per region: no region (ET, TC, WT) lower by more than 0.005;
  R1 serious regressions: at most 2 cases losing > 0.10 Dice in any region; no empty GT ET/TC region that Run D scores 1.0 becomes 0;
  R2 HD95: per-region mean (defined values) not worse by more than 0.5 mm; undefined counts reported;
  F  exploratory: Amendment-28 VAL folds, >= 3 of 5 with non-negative mean difference (reported).
  Outcome (i) candidate improvement = T1, T2, O1, O2, R1, R2 all hold for both variants; (ii) subgroup-only = T1/T2 hold but O1 or
  O2 fails (reported, not adopted); (iii) no effect / harm (reported). Reported whatever it shows.
- Even outcome (i) changes neither the headline (Ensemble 3) nor the paper: any new ensemble, TEST scoring or paper change needs a
  further amendment and the user's decision, and a confirmatory claim needs an independent, previously unused labelled cohort
  (BraTS 2021 TEST has been inspected repeatedly and cannot confirm a revised method).
- Compute: ~11-13 h training (Run D: 10.75 h), ~3 h VAL evaluation, ~3 h rotation variant. Fault-tolerant resume (up to 10 attempts)
  as Run D; boot-hook entry added. Never stopped unless the user asks.
- Amendment 29 addendum (2026-10-20 ~13:50, user: "ok you go best", after an external audit; written while the Run F VAL evaluation
  was at step 1/4, before any criteria output or per-case Run F VAL score existed): outcome (ii) "subgroup-only" means T1 AND T2 hold
  (both) while O1 or O2 fails. criteria.py line 95 had implemented T1 OR T2; corrected to T1 AND T2 (backup
  criteria.py.bak_20261020_1350_pre_T1andT2). Outcome (i) is unchanged (all six criteria, both variants). Separately noted:
  queue_runF.sh writes DONE even if a train-fit, the rotation evaluation or a criteria step fails; the running script is not edited
  (bash reads it incrementally), so every step's output is checked by hand before DONE is trusted.
