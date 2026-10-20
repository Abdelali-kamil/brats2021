"""Amendment 26 step 8: nnFormer under the inference it is scored with (nnFormer_predict 3d_fullres -chk model_best):
nnU-Net-v1 preprocessing of the case (crop, z-score), sliding window 128^3 step 0.5, Gaussian, mirroring TTA (--no-tta: off),
mixed precision, argmax + export to the original geometry. Timed per case from the 4 NIfTI files (reading included, unlike
the other benchmarks, because the nnU-Net-v1 preprocessing API starts from file paths) to the written label map."""
import sys, os, argparse, tempfile, torch
sys.path.insert(0, "/mnt/data1/kamil_research/experiments/efficiency")
from common import *
from nnformer.training.model_restore import load_model_and_checkpoint_files
from nnformer.inference.segmentation_export import save_segmentation_nifti_from_softmax
ap = argparse.ArgumentParser(); ap.add_argument("--name", default="nnformer"); ap.add_argument("--no-tta", action="store_true"); a = ap.parse_args()
RUN = open("/mnt/data1/kamil_research/experiments/nnformer/RUN_PATH").read().strip()
trainer, params = load_model_and_checkpoint_files(os.path.dirname(RUN), folds=[0], mixed_precision=True, checkpoint_name="model_best")
trainer.load_checkpoint_ram(params[0], False); net = trainer.network; cnt = Counter(net)
RAW = "/mnt/data1/kamil_research/experiments/nnformer/DATASET/nnFormer_raw/nnFormer_raw_data/Task003_tumor/imagesTs"
TMP = tempfile.mkdtemp(dir="/tmp/claude-1006") if os.path.isdir("/tmp/claude-1006") else tempfile.mkdtemp()
def run(c):
    d, _, props = trainer.preprocess_patient([f"{RAW}/{c}_{i:04d}.nii.gz" for i in range(4)])
    sm = trainer.predict_preprocessed_data_return_seg_and_softmax(d, do_mirroring=not a.no_tta, mirror_axes=trainer.data_aug_params['mirror_axes'],
                                                                   use_sliding_window=True, step_size=0.5, use_gaussian=True,
                                                                   all_in_gpu=False, verbose=False, mixed_precision=True)[1]
    save_segmentation_nifti_from_softmax(sm, f"{TMP}/{c}.nii.gz", props, 1, None, None, None, None, None, None, 0)
with torch.no_grad():
    run(WARMUP); rows = []
    for c in CASES:
        cnt.n = 0
        _, s, g = timed(lambda: run(c)); rows.append({"case": c, "sec": s, "peak_gib": g, "n_forward": cnt.n})
save(a.name, sum(p.numel() for p in net.parameters()), flops_per_forward(net, (1, 4, 128, 128, 128)), (1, 4, 128, 128, 128), rows,
     {"protocol": "nnU-Net-v1 sliding window 128^3 step 0.5, gaussian, " + ("no TTA" if a.no_tta else "mirroring TTA") + ", mixed precision, incl. NIfTI read/write"})
