"""nnU-Net v2 3d_fullres, fold all, checkpoint_final, as scored: tile step 0.5,
Gaussian, mirroring TTA on, nnU-Net's own preprocessing + export inside the timing."""
import sys, torch
sys.path.insert(0, "/mnt/data1/kamil_research/experiments/efficiency")
from common import *
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
RAW = "/mnt/data1/kamil_research/baselines/nnunet/nnUNet_raw/Dataset137_BraTS2021/imagesTs"
pr = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                     perform_everything_on_device=True, device=torch.device("cuda"), allow_tqdm=False)
pr.initialize_from_trained_model_folder(
    "/mnt/data1/kamil_research/baselines/nnunet/nnUNet_results/Dataset137_BraTS2021/nnUNetTrainer__nnUNetPlans__3d_fullres",
    use_folds=("all",), checkpoint_name="checkpoint_final.pth")
net = pr.network; cnt = Counter(net); rw = pr.plans_manager.image_reader_writer_class()
def load(c): return rw.read_images([f"{RAW}/{c}_{i:04d}.nii.gz" for i in range(4)])
with torch.no_grad():
    img, props = load(WARMUP); pr.predict_single_npy_array(img, props); rows = []
    for c in CASES:
        img, props = load(c); cnt.n = 0
        _, s, g = timed(lambda: pr.predict_single_npy_array(img, props))
        rows.append({"case": c, "sec": s, "peak_gib": g, "n_forward": cnt.n})
save("nnunet", sum(p.numel() for p in net.parameters()), flops_per_forward(net, (1, 4, 128, 128, 128)),
     (1, 4, 128, 128, 128), rows, {"protocol": "tile 128^3 step 0.5, gaussian, mirroring TTA, nnU-Net pre/post-processing"})
