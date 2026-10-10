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
