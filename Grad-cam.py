import os
import re
import random
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

from brats import get_datasets
from model import WaveletUNetPlusPlus

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# CONFIG
# ============================================================
CHECKPOINT_PATH = "checkpoints/segmentor_epoch_650.pth"
TEST_IDS_FILE   = "test_ids.txt"
SAVE_DIR        = "./gradcam_results"

# Which patient & slice to visualise
# (PATIENT_INDEX = 0-based index inside the test subset)
PATIENT_INDEX   = 0
SLICE_INDEX     = 77

SEED        = 42
BATCH_SIZE  = 1
NUM_WORKERS = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# REPRODUCIBILITY
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# ID UTILS  (same as analyze_results.py)
# ============================================================
def norm_id_str(s):
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"\.nii(?:\.gz)?$", "", s)
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def load_patient_ids(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing split file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        ids = [line.strip() for line in f if line.strip()]
    if not ids:
        raise ValueError(f"No patient IDs found in {path}")
    return ids


# ============================================================
# DATASET  (same as analyze_results.py)
# ============================================================
def build_subset(ids_file):
    full_dataset = get_datasets(on="test")
    ids_raw = load_patient_ids(ids_file)

    ids_raw_set = set(ids_raw)
    ids_norm_set = set(norm_id_str(t) for t in ids_raw)

    matched_indices = []
    for idx in range(len(full_dataset)):
        item = full_dataset[idx]
        pid_str = str(item.get("patient_id", item.get("id", str(idx))))
        pid_nrm = norm_id_str(pid_str)

        if pid_str in ids_raw_set or pid_nrm in ids_norm_set:
            matched_indices.append(idx)

    if not matched_indices:
        raise RuntimeError(f"No dataset items matched IDs in {ids_file}")

    print(f"Full dataset size: {len(full_dataset)}")
    print(f"Loaded IDs       : {len(ids_raw)}")
    print(f"Matched patients : {len(matched_indices)}")

    return Subset(full_dataset, matched_indices)


# ============================================================
# MODEL  (same as analyze_results.py)
# ============================================================
def load_model(device, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = WaveletUNetPlusPlus(in_channels=4, n_classes=3).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print(f"Model loaded from: {checkpoint_path}")
    return model


# ============================================================
# GRAD-CAM  (3D U-Net)
# ============================================================
class GradCAM3D:
    """
    Grad-CAM for a 3D segmentation model.
    Hooks on the target Conv3d layer, backprops through the
    total predicted volume, and upsamples CAM to input shape.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None

        self.fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self.bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        self.fwd_hook.remove()
        self.bwd_hook.remove()

    @torch.enable_grad()
    def generate(self, x):
        """
        x: [1, 4, D, H, W] on device
        returns: np array [D, H, W] normalised [0, 1]
        """
        self.model.zero_grad()

        out = self.model(x)  # [1, 3, D, H, W]

        # Backprop through sum of all tumour predictions
        score = out.sum()
        score.backward()

        # GAP over spatial dims (D, H, W)
        # activations/gradients: [1, C, d, h, w]
        weights = self.gradients.mean(dim=(2, 3, 4), keepdim=True)  # [1, C, 1, 1, 1]
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [1, 1, d, h, w]
        cam = torch.relu(cam)

        # Upsample to input resolution
        cam = nn.functional.interpolate(
            cam, size=x.shape[2:], mode='trilinear', align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()  # [D, H, W]

        # Normalise
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def find_target_conv3d(model):
    """
    Finds the deepest Conv3d layer.
    Run once — it prints all Conv3d names so you can override manually.
    """
    print("\n── Conv3d layers in WaveletUNetPlusPlus ──")
    last_conv = None
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv3d):
            print(f"  {name:60s}  out_ch={m.out_channels}")
            last_conv = m
    print("──────────────────────────────────────────\n")

    if last_conv is None:
        raise RuntimeError("No Conv3d layer found.")
    return last_conv


# ============================================================
# VISUALISATION
# ============================================================
def save_gradcam_figure(flair_slice, cam_slice, label_slice,
                        patient_id, slice_idx, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    H, W = flair_slice.shape

    # Resize CAM to match image
    cam_resized = cv2.resize(cam_slice, (W, H), interpolation=cv2.INTER_LINEAR)
    cam_resized = cv2.GaussianBlur(cam_resized, (11, 11), 4)

    # Heatmap
    cam_uint8   = np.uint8(255 * cam_resized)
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Original FLAIR as RGB
    orig_uint8 = np.uint8(255 * flair_slice)
    orig_rgb   = cv2.cvtColor(orig_uint8, cv2.COLOR_GRAY2RGB)

    # Overlay
    overlay = cv2.addWeighted(heatmap_rgb, 0.45, orig_rgb, 0.55, 0)

    # CAM masked to tumour region only
    cam_on_tumor = cam_resized * label_slice

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle(
        f"Grad-CAM  |  Patient: {patient_id}  |  Slice: {slice_idx}",
        fontsize=14, fontweight='bold'
    )

    axes[0].imshow(flair_slice, cmap='gray')
    axes[0].set_title("FLAIR Input")
    axes[0].axis('off')

    axes[1].imshow(heatmap_rgb)
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis('off')

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis('off')

    axes[3].imshow(label_slice, cmap='hot')
    axes[3].set_title("Ground Truth")
    axes[3].axis('off')

    axes[4].imshow(cam_on_tumor, cmap='jet')
    axes[4].set_title("CAM on Tumour")
    axes[4].axis('off')

    plt.tight_layout()
    fname = f"{patient_id}_slice{slice_idx:03d}_gradcam.png"
    fpath = os.path.join(save_dir, fname)
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {fpath}")


# ============================================================
# BATCH MODE  (optional — multiple patients)
# ============================================================
def run_batch(model, target_layer, test_subset, patient_indices, slice_idx, save_dir):
    """
    Generate Grad-CAM for a list of patient indices.
    patient_indices : list of ints, e.g. [0, 1, 2, 5, 10]
    """
    gradcam = GradCAM3D(model, target_layer)
    loader  = DataLoader(test_subset, batch_size=1, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=True)

    all_batches = list(loader)

    for idx in patient_indices:
        if idx >= len(all_batches):
            print(f"  [SKIP] Index {idx} out of range ({len(all_batches)} patients)")
            continue

        batch = all_batches[idx]

        images = batch["image"].float().to(device)   # [1, 4, D, H, W]
        labels = batch["label"].float().to(device)   # [1, 3, D, H, W]

        pid_raw = batch.get("patient_id", ["unknown"])
        if isinstance(pid_raw, torch.Tensor):
            patient_id = str(pid_raw[0].item())
        else:
            patient_id = str(pid_raw[0])

        _, _, D, H, W = images.shape
        print(f"\nPatient {patient_id} | Shape {tuple(images.shape)}")

        if slice_idx >= D:
            print(f"  [SKIP] Slice {slice_idx} >= depth {D}")
            continue

        # Generate 3D CAM
        cam_3d = gradcam.generate(images)  # [D, H, W]

        # Extract 2D slices
        flair_slice = images[0, 3, slice_idx, :, :].cpu().numpy()   # FLAIR = channel 3
        flair_norm  = (flair_slice - flair_slice.min()) / (flair_slice.max() - flair_slice.min() + 1e-8)

        cam_slice   = cam_3d[slice_idx, :, :]                        # [H, W]

        label_slice = labels[0, :, slice_idx, :, :].sum(dim=0).cpu().numpy()  # sum all tumour regions
        label_slice = (label_slice > 0).astype(np.float32)

        save_gradcam_figure(flair_norm, cam_slice, label_slice,
                            patient_id, slice_idx, save_dir)

    gradcam.remove_hooks()
    print("\nBatch Grad-CAM complete.")


# ============================================================
# MAIN
# ============================================================
def main():
    set_seed(SEED)

    # Build test subset
    test_subset = build_subset(TEST_IDS_FILE)
    loader = DataLoader(
        test_subset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # Load model
    model = load_model(device, CHECKPOINT_PATH)

    # Print all Conv3d layers — helps you pick the best target manually
    target_layer = find_target_conv3d(model)

    # ── SINGLE PATIENT MODE ────────────────────────────────────
    all_batches = list(loader)
    if PATIENT_INDEX >= len(all_batches):
        raise IndexError(f"PATIENT_INDEX={PATIENT_INDEX} but only {len(all_batches)} patients available.")

    batch = all_batches[PATIENT_INDEX]

    images = batch["image"].float().to(device)   # [1, 4, D, H, W]
    labels = batch["label"].float().to(device)   # [1, 3, D, H, W]

    pid_raw = batch.get("patient_id", ["unknown"])
    if isinstance(pid_raw, torch.Tensor):
        patient_id = str(pid_raw[0].item())
    else:
        patient_id = str(pid_raw[0])

    _, _, D, H, W = images.shape
    print(f"\nPatient: {patient_id}")
    print(f"Volume shape: {tuple(images.shape)}  (Batch, Channels, Depth, Height, Width)")
    print(f"Visualising axial slice {SLICE_INDEX} / {D}")

    if SLICE_INDEX >= D:
        raise IndexError(f"SLICE_INDEX={SLICE_INDEX} >= depth D={D}")

    # Generate 3D CAM
    gradcam = GradCAM3D(model, target_layer)
    cam_3d = gradcam.generate(images)
    gradcam.remove_hooks()

    # Extract 2D slices
    flair_slice = images[0, 3, SLICE_INDEX, :, :].cpu().numpy()   # modality 3 = FLAIR
    flair_norm  = (flair_slice - flair_slice.min()) / (flair_slice.max() - flair_slice.min() + 1e-8)

    cam_slice   = cam_3d[SLICE_INDEX, :, :]                        # [H, W]

    label_slice = labels[0, :, SLICE_INDEX, :, :].sum(dim=0).cpu().numpy()
    label_slice = (label_slice > 0).astype(np.float32)

    save_gradcam_figure(flair_norm, cam_slice, label_slice,
                        patient_id, SLICE_INDEX, SAVE_DIR)

    # ── OPTIONAL BATCH MODE ────────────────────────────────────
    # Uncomment to run on multiple patients:
    # run_batch(
    #     model=model,
    #     target_layer=target_layer,
    #     test_subset=test_subset,
    #     patient_indices=[0, 1, 2, 5, 10, 15],
    #     slice_idx=77,
    #     save_dir=SAVE_DIR
    # )

    print("\nDone.")


if __name__ == "__main__":
    main()