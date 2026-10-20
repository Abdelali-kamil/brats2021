# TransBTS

- Paper: Wang et al., MICCAI 2021, LNCS 12901:109-119
- Code: official_code/ (Wenxuan-1119/TransBTS @ 7dcfc31, Apache-2.0)
- How we ran it: our_run_scripts/ (train.sh, patches_vs_7dcfc31.diff); working copy /mnt/data1/kamil_research/experiments/transbts (Amendment 24)
- Loss: --criterion softmax_dice2 (the authors' own background-inclusive Dice in models/criterions.py), not the repository
  default softmax_dice, which collapsed in our first run (background predicted as NCR; Amendment 24 addendum 2, 2026-10-14)
- Status: see ../PROGRESS.md

## Result files
- BraTS 2021 test (per case): /home/kamilabdelali/brats2021/results/brats/transbts_test_per_case.csv (when done)
- BraTS-Africa zero-shot (per case): /mnt/data1/kamil_research/experiments/external_africa/results/per_case_transbts.csv (when done)
- Classification summary: /mnt/data1/kamil_research/experiments/classification_africa/ (rows 'transbts / logreg', 'transbts / random_forest')
- BraTS-Africa CV (per case, pooled): /mnt/data1/kamil_research/experiments/adapt_africa/cv/per_case_cv_transbts.csv (when done)
- Efficiency: /mnt/data1/kamil_research/experiments/efficiency/results/transbts.json (when done)
