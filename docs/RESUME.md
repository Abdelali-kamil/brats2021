# Where to pick up

Last session: 2026-09-04.

## In flight

**BraTS retraining with augmentation and random crops** (`aug_cosine_v1`).
Running since 2026-09-03, resumed 2026-09-04 after a GPU fault. Follow with
`tail -f logs/aug_cosine_v1.log`.

- 300 epochs, cosine annealing, batch 1 x accum 8 (effective batch 8, unchanged
  from the historical configuration). Batch size is 1 rather than 2 only because
  the GPU is shared with another user's jobs; throughput per sample is flat
  across batch sizes on this hardware, so nothing is lost but BatchNorm sample
  count.
- Best so far **0.8627** mean Dice at epoch 42, on the 126-case *validation*
  split under the training-time metric (a single centre-crop forward pass). That
  is not comparable to the 0.8828 of Table IV, which uses sliding-window
  inference with flip TTA and is measured on all 251 held-out cases. Expect the
  full protocol to score higher than the training-time metric.
- A copy of the epoch-42 checkpoint is kept at
  `checkpoints/aug_cosine_v1_best_ep42_0.8627.backup.pth`, so this run cannot
  leave the project worse off than it started.

**Caveat on the resume.** The cosine schedule restarts its cycle at epoch 47
rather than continuing the original 300-epoch curve, so the learning rate
returned to 1.88e-04 and anneals over the remaining epochs. This is a warm
restart, not an uninterrupted schedule, and the paper's methods must say so.
Validation dipped to 0.8494/0.8433 immediately after the restart, which is
expected. **If it has not exceeded 0.8627 by roughly epoch 100, the restart cost
more than it gained** and the epoch-42 checkpoint is the one to keep.

## When it finishes

Evaluate on the partition that has never been touched:

```bash
python scripts/evaluate_brats.py --partition test \
    --checkpoint checkpoints/aug_cosine_v1_best.pth --tag aug_cosine_v1_test
```

`--partition test` is the 125 cases held back by `brats_split_3way`; `val` is
the 126 that drove selection. Their union is exactly the historical
`internal_validation` partition, so a `test` number is directly comparable to
the old one case for case, with the difference that nothing selected on it.

For an apples-to-apples comparison, score the *released* checkpoint on the same
partition too. Note that it selected on all 251, so its `test` score is still
optimistic; the two are not equally clean and the paper should say which is
which.

## Completed since the last note

- **Downsampling ablation** (paper Table IX) — done 2026-07-31. Max-pooling
  0.8145, DWT 0.8144, paired difference -0.0000 [-0.0127, +0.0118], and DWT
  worse on HD95 in all three regions. The wavelet claim is withdrawn in the
  paper rather than qualified.
- **UPenn 5-fold cross-validation** — done 2026-08-11. Pooled out-of-fold mean
  Dice 0.8474 [0.8282, 0.8646], fold sigma 0.021, no failures among 147.
- **Failure analysis** (paper Section XI) — `scripts/make_failure_figures.py`.
  ET detection collapses below ~300 voxels; the tumour-core channel depends on
  enhancement (median Dice_TC 0.008 on 8 barely-enhancing cases against 0.953 on
  243). Cannot be checked externally: UPenn-GBM is GBM-only and has one eligible
  subject.
- **Training defects fixed** (2805ff2, b5e5b1e) — training ran on a fixed centre
  crop with no augmentation; `--resume` defaulted to a checkpoint, so a
  from-scratch run silently continued it; `torch.load` rejected the config block
  under torch>=2.6; and resuming from `_last.pth` could overwrite a better
  `_best.pth`.
- **Repository history rewritten** — a 21.9 GB `data_backup.tar.gz` and 18
  unrelated 119 MB checkpoints made the repo unpushable. 24 GB to 13 MB, all 35
  commits preserved, tip trees verified identical, now on GitHub.
- **Manuscript under version control** in `paper/`, previously an uncommitted
  file in a home directory.

## Outstanding

- **nnU-Net segmentation baseline** on this project's split — still no
  comparison numbers computed on the project's own partition.
- **BraTS k-fold cross-validation** for a fold-level mean +/- std in Table VI.
  Table VI still carries the caption note explaining why our row has no +/-.
- The learned lesion/slice/global encoders of the original proposal remain
  unbuilt; the classification module uses radiomic features, not
  segmentation-encoder bottleneck features.
- No LaTeX toolchain on this machine, so `paper/paper.tex` is syntax-checked but
  not compiled here. Build it on Overleaf after `bash paper/collect_figures.sh`.

## Repository state

`python -m pytest tests/ -q` -> 53 pass. Segmentation and classification results
in `results/` are unchanged and final; the retraining above will add a new
result rather than replace them.
