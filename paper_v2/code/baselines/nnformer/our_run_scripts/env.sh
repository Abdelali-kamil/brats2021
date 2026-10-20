# nnFormer paths (Amendment 26); source before any nnFormer_* command
W=/mnt/data1/kamil_research/experiments/nnformer
export nnFormer_raw_data_base=$W/DATASET/nnFormer_raw nnFormer_preprocessed=$W/DATASET/nnFormer_preprocessed RESULTS_FOLDER=$W/DATASET/nnFormer_trained_models
export NNFORMER_FIXED_SPLIT=$W/split_fixed.json PATH=$W/.venv/bin:$PATH PYTHONUNBUFFERED=1
