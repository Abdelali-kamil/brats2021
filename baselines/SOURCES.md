# Source code for every comparison method

Every number in the paper's comparison table must be produced by the code listed
here, run on our own server, on our own split. No number is copied from a source
paper — those were produced on other datasets and are not comparable. See
`baselines/README.md` for the fairness rules and `docs/RELATED_WORK.md` for why
each method is in the table.

All repositories below are cloned at a **pinned commit**, so a rerun a year from
now compiles the same table. Clones live in `baselines/repos/` and are gitignored
(code is fetched, never vendored).

```bash
bash baselines/fetch.sh          # clone/pin all
bash baselines/fetch.sh nnunet   # just one
```

## The roster

| Key | Method | Paper | Official code | Pinned commit | Commit date | License |
|---|---|---|---|---|---|---|
| `xgboost` | XGBoost | Chen & Guestrin, KDD 2016 | [dmlc/xgboost](https://github.com/dmlc/xgboost) | PyPI `xgboost==3.2.0` | — | Apache-2.0 |
| `mmgl` | MMGL | *Multi-Modal Graph Learning for Disease Prediction*, IEEE TMI 2022 | [SsGood/MMGL](https://github.com/SsGood/MMGL) | `c13b05c190f85e614ccb3ace60bb6b152a1f4a1b` | 2024-03-08 | MIT |
| `daft` | DAFT *(replaces MMCL)* | Pölsterl et al., MICCAI 2021 | [ai-med/DAFT](https://github.com/ai-med/DAFT) | `b36974b1f8d9f46fd410e946b6ead28078f0622e` | 2024-06-14 | **GPL-3.0** |
| `nnunet` | nnU-Net | Isensee et al., Nature Methods 2021 | [MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet) | `ded2aa3a4c81a9caae37054224d8ebac2d19a061` | 2026-09-14 | Apache-2.0 |
| `selfmedmae` | SelfMedMAE | *Self Pre-training with Masked Autoencoders…*, ISBI 2023 | [cvlab-stonybrook/SelfMedMAE](https://github.com/cvlab-stonybrook/SelfMedMAE) | `21f38b1ff8fb9ce3948651c1e356f3cc3fddd45a` | 2024-01-09 | Apache-2.0 |
| `swin_unetr` | Swin UNETR *(replaces ResGANet)* | Hatamizadeh et al., MICCAI BrainLes 2021 | [Project-MONAI/research-contributions](https://github.com/Project-MONAI/research-contributions) | `21ed8e57c7256834d4fbaf19579ca25ad3d135ee` | 2025-05-09 | Apache-2.0 |
| `mtanet` | MTANet | Ling et al., IEEE TMI 43(2):674–685, 2024 | [yatingling/MTANet](https://github.com/yatingling/MTANet) | `7a5f5d163382bce8856c1552a8b9ed9c0a6980e6` | 2024-04-11 | **none stated** |
| `ours` | Wavelet U-Net++ | this project | this repo | — | — | — |

Two substitutions, both because the original has no official code and so cannot
be reproduced under our fairness rules: **MMCL → DAFT** and **ResGANet → Swin
UNETR**. If MMCL or ResGANet must stay in the paper, they can only be cited from
their own papers and marked explicitly as *not reproduced on this cohort*.

## What is actually in each clone

Verified by inspection after `fetch.sh`, 2026-09-15.

### `nnunet` — complete, runnable
Full package with CLI entry points, a BraTS2021 converter
(`nnunetv2/dataset_conversion/Dataset137_BraTS21.py`), planning, training and
inference. Relabels ET from 4 to 3, so scoring needs `--et-label 3`.
Recipe: `baselines/nnunet/SETUP.md`.

### `swin_unetr` — complete, runnable
The monorepo holds many projects; ours is `swin_unetr/SwinUNETR/BRATS21`, which
is the BraTS21 recipe with its own training and inference scripts.
Recipe: `baselines/swin_unetr/SETUP.md`.

### `selfmedmae` — complete, runnable
Pre-training plus fine-tuning code. Expects its own data layout and config files;
budget adaptation time. Recipe: `baselines/selfmedmae/SETUP.md`.

### `mmgl` — complete, runnable
Classification only. **Transductive**: needs the whole cohort at inference, which
is a different problem from our inductive classifier. Keep that distinction in
the related-work text. Recipe: `baselines/mmgl/SETUP.md`.

### `daft` — complete, runnable
3D image + tabular fusion, the closest match to our MGMT/IDH setting.
**GPL-3.0**: fine to run for benchmarking, but do not copy its source into this
repository — that would impose GPL on our code. We only clone and run it.
Recipe: `baselines/daft/SETUP.md`.

### `mtanet` — INCOMPLETE, cannot run as published
The repository contains **five files only**: `MATNet.py`, `pvt.py`, `pvt_v2.py`,
`pvtv2.py`, `README.md`. That is model-definition code. There is **no training
script, no data loader, no config, and no LICENSE file**, and the README requires
a pretrained backbone `lib/pvt_v2_b2.pth` that the repo does not ship.

Consequence: MTANet cannot be reproduced by running "their code" the way the
others can. Producing an MTANet row would mean writing our own training and data
pipeline around their model definition — which is our code, not theirs, and must
be disclosed as such in the paper. Decide before spending GPU time:

  1. Write the missing pipeline, report the row, and state plainly that only the
     architecture is the authors' and the training pipeline is ours; or
  2. Drop MTANet from the reproduced table and cite it as not reproducible, the
     same treatment given to MMCL and ResGANet.

The absent license is a second reason for caution — no stated terms means no
granted permission to redistribute; running it locally for evaluation is the
conservative limit.

## Re-verifying the pins

```bash
cd baselines/repos
for r in nnunet swin_unetr selfmedmae mtanet mmgl daft; do
  printf '%-12s %s\n' "$r" "$(git -C $r rev-parse HEAD)"
done
```

Compare against the table above. A mismatch means someone moved a clone off its
pin, and any numbers produced from it are not the pinned-commit numbers.

## Citation caveat

The paper/venue strings above were carried over from `docs/RELATED_WORK.md`,
which warns that author lists, years, page numbers and exact titles should be
checked against the original publications before they enter a bibliography. The
commit SHAs, dates, licenses and file inventories on this page were read directly
from the clones and are verified. Only the MTANet citation was taken verbatim
from its own README, where the authors state it.
