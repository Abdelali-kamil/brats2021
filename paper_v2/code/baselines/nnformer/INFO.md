# nnFormer

- Paper: Zhou et al., IEEE TIP 32:4036-4045, 2023 (doi 10.1109/TIP.2023.3293771)
- Code: official_code/ (282857341/nnFormer @ d48aaeb, MIT)
- How we ran it: our_run_scripts/ (convert_to_nnformer.py, split_fixed.json, patches_vs_d48aaeb.diff); working copy /mnt/data1/kamil_research/experiments/nnformer
- Status: see ../PROGRESS.md

## Result files
- BraTS 2021 test (per case): /home/kamilabdelali/brats2021/results/brats/nnformer_test_per_case.csv (when done)
- BraTS-Africa zero-shot (per case): /mnt/data1/kamil_research/experiments/external_africa/results/per_case_nnformer.csv (when done)
- Classification summary: /mnt/data1/kamil_research/experiments/classification_africa/ (rows 'nnformer / logreg', 'nnformer / random_forest')
- BraTS-Africa CV (per case, pooled): /mnt/data1/kamil_research/experiments/adapt_africa/cv/per_case_cv_nnformer.csv (when done)
- Efficiency: /mnt/data1/kamil_research/experiments/efficiency/results/nnformer.json (when done)
