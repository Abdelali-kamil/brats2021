# What was removed, and why

Everything listed here was moved out of the repository during the July 2026
cleanup, to:

```
/home/kamilabdelali/brats2021_archive_20260723/
```

Nothing was deleted outright except one item, noted at the end. The commit
immediately before the cleanup (`fb23107`, "Snapshot before repository cleanup
and methodology fixes") contains every text file as it stood beforehand, so
the pre-cleanup state is also recoverable from git history alone.

## Removed because the results are invalid

These produced or contained figures derived from thresholds fitted on the test
set. See "The test-set tuning that was removed" in `METHODOLOGY.md`.

| File | Note |
|---|---|
| `FINAL_ONE_RESULT_FOR_PROF.txt` | Reported ET 0.931 / TC 0.726 / WT 0.793. The ET figure came from `et_thr = 0.015` fitted on the test set. Do not cite. |
| `final_results_report.txt` | Ranked seven setups on the test set and reported the winner. |
| `final_results_all.py`, `final_test.py`, `results_test.py`, `final_results.py` | The scripts implementing that sweep. |
| `final_summary_table.csv`, `final_dice_bar.png`, `model_ranking.png` | Figures and tables built from the above. |
| `FINAL_REPORT_ALIGNED.csv`, `FINAL_REPORT_ALIGNED_CLEAN.csv`, `final_results.csv`, `delta_vs_ens_all_three.csv` | Per-case exports from the same runs. |
| `DICE_FILTERED_SUBSET_ANALYSIS.csv`, `DICE_THRESHOLD_COUNTS.csv`, `BEST_DICE_CASES.csv`, `WORST_DICE_CASES.csv`, `TEST_OOM_SAFE_64.csv` | Subset analyses selected on test-set Dice. |

Superseded by `scripts/evaluate_upenn.py` and `results/upenn/`.

## Removed because superseded

| File | Replaced by |
|---|---|
| `upenn/evaluate_upenn.py` | `scripts/evaluate_upenn.py` — same inference, but selection moved to validation and CIs added |
| `upenn/train_classifier_idh1.py` | `scripts/train_classifier_idh1.py` — cohort restricted, thresholds cross-fitted |
| `upenn/build_comparison_summary.py` | `summarise_segmentation` in `brats_gbm/eval/stats.py` |
| `code/heldout_summary.py` | `scripts/summarize_brats.py` — adds bootstrap CIs, renames the partition honestly |
| `code/brats.py`, `code/model.py`, `code/batch_utlis.py` | `brats_gbm/` package copies |
| `code/classifier.py` | Never ran: `TARGET_COL` was still a placeholder and the `features.csv` it read did not exist |
| `code/evaluate_classification.py`, `code/evaluate.py` | Folded into the current evaluation scripts |
| `upenn/upenn_idh1_cv_metrics.json`, `upenn/upenn_idh1_predictions.csv` | `results/classification/idh1_results.json` |
| `outputs/` | `results/` |

## Removed because one-off or debug

`analyze_dice_distribution.py`, `analyze_results.py`, `check_determinism.py`,
`check_mismatch.py`, `check_worst5.py`, `compare_ids.py`, `diagnose_ids.py`,
`debug_overlay_predictions.py`, `fix_download.py`, `fix_worst_cases.py`,
`image.py`, `class.py`, `plot_results.py`, `show_metrics.py`,
`stress_test.py`, `server_test.py`, `test_and_visualize.py`,
`test_another_data.py`, `test_data.py`, `visualize_results.py`,
`train_data.py`, `Grad-cam.py`, plus the `debug_overlays/`,
`debug_overlays_valid_best/`, `gradcam_results/`, `results/` and
`final_results/` output directories.

`config.py`, `checkpoints.py` and `encoders.py` were imported by nothing.

## Removed because they were filename accidents

Four files created by mistyped shell redirects: `=`, `0`, `0],`, and
`__init__.pu` (a typo for `__init__.py`, and empty).

## Checkpoints

| Item | Size | Note |
|---|---|---|
| `code/checkpoints/` | 15 GB | 151 per-epoch segmentor checkpoints plus classifier checkpoints. Only `segmentor_epoch_650.pth` is referenced by any reported result; it was moved to `checkpoints/`. |
| `contaminated_et4/` | 240 MB | Four `.pth` files whose directory name records a contamination whose nature was never documented. Nothing in the repository referenced them. Retained in the archive rather than deleted, precisely because what was wrong with them is unknown. |
| `checkpoints_pontine_binary_full*/` | — | Three directories from the separate pontine-infarction line of work, unrelated to the BraTS/UPenn pipelines. |

## Pontine-infarction files

`pontine_eval_continue_t02.csv` and `pontine_eval_fixed.csv` belong to a
different project (435 cases, mean Dice 0.58) that shares no code path with
this repository. Archived rather than integrated.

## Deleted outright

One item was deleted rather than archived: a second, abandoned git repository
at the project root, holding **30 GB** in `objects/`.

It had no commits, no refs, no packed-refs and an empty index. `git fsck`
reported 12,337 dangling blobs and zero dangling commits or trees, and git's
own `gc.log` recorded "There are too many unreachable loose objects". It was
the residue of a `git add` that staged the imaging data and was never
committed — with no commit or tree objects there are no filenames or structure
to recover, and the underlying NIfTI files remain in `data/` and `upenn_data/`
regardless. Deleting it took the filesystem from 84% to 81% full.

The repository's real history was in `code/.git` and was promoted to the
project root, preserving all three original commits.
