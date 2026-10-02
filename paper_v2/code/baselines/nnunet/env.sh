# Source before any nnU-Net command. Data lives on /mnt/data1 (1.6T free);
# the repo volume has far less headroom.
export nnUNet_raw=/mnt/data1/kamil_research/baselines/nnunet/nnUNet_raw
export nnUNet_preprocessed=/mnt/data1/kamil_research/baselines/nnunet/nnUNet_preprocessed
export nnUNet_results=/mnt/data1/kamil_research/baselines/nnunet/nnUNet_results
export nnUNet_n_proc_DA=4   # a training run shares this machine
