# Henry et al. deeply supervised 3D U-Net

- Paper: Henry et al., BrainLes 2020, LNCS 12658 (doi 10.1007/978-3-030-72084-1_30)
- Code: official_code/ (lescientifik/open_brats2020 @ e5f9896) + dependency_ranger_optimizer/
- How we ran it: our_run_scripts/ (train.sh, evaluate.sh, bench_henry.py, patches_vs_e5f9896.diff); working copy /mnt/data1/kamil_research/experiments/henry_dsunet (Amendment 22)
- Status: see ../PROGRESS.md

## Result files
- BraTS 2021 test (per case): /home/kamilabdelali/brats2021/results/brats/henry_dsunet_test_per_case.csv
- BraTS-Africa zero-shot (per case): /mnt/data1/kamil_research/experiments/external_africa/results/per_case_henry_dsunet.csv
- Classification summary: /mnt/data1/kamil_research/experiments/classification_africa/ (rows 'henry_dsunet / logreg', 'henry_dsunet / random_forest')
- BraTS-Africa CV (per case, pooled): /mnt/data1/kamil_research/experiments/adapt_africa/cv/per_case_cv_henry_dsunet.csv
- Efficiency: /mnt/data1/kamil_research/experiments/efficiency/results/henry_dsunet.json
