import os
import torch
import numpy as np
import nibabel as nib
import scipy.ndimage as ndimage
from pathlib import Path
import matplotlib.pyplot as plt
from model import WaveletUNetPlusPlus

# ============================================================
# CONFIG - UPDATE THESE PATHS
# ============================================================
IMG_PATH = "/home/kamilabdelali/anotherdata/images"
LBL_PATH = "/home/kamilabdelali/anotherdata/labels"
CHECKPOINT = "checkpoints/segmentor_epoch_650.pth"
# ============================================================

def normalize_brain(x):
    x = x.astype(np.float32)
    mask = x > 0
    if np.any(mask):
        x[mask] = (x[mask] - x[mask].mean()) / (x[mask].std() + 1e-8)
        x[~mask] = 0
    return np.clip(x, -5, 5)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. LOAD CHECKPOINT & DETECT ARCHITECTURE
    ckpt = torch.load(CHECKPOINT, map_location=device)
    state_dict = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
    
    # Auto-detect n_classes from checkpoint weights
    # Look for the last layer (usually 'out.weight' or 'final.weight')
    out_key = [k for k in state_dict.keys() if 'out' in k or 'final' in k and 'weight' in k][-1]
    n_classes = state_dict[out_key].shape[0]
    print(f"Detected Model Classes from Checkpoint: {n_classes}")

    model = WaveletUNetPlusPlus(in_channels=4, n_classes=n_classes).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # 2. PICK ONE PATIENT
    img_files = sorted(list(Path(IMG_PATH).glob("*.nii*")))
    lbl_files = sorted(list(Path(LBL_PATH).glob("*.nii*")))
    
    if not img_files:
        print("No images found!")
        return

    test_img = img_files[0]
    test_lbl = [l for l in lbl_files if test_img.stem.split('.')[0] in l.name][0]

    print(f"\nAnalyzing Patient: {test_img.name}")

    # Load and Preprocess
    img = nib.load(str(test_img)).get_fdata()
    lbl = nib.load(str(test_lbl)).get_fdata()
    
    # Prep 4-channel input
    img_norm = normalize_brain(img)
    x = np.stack([img_norm] * 4, axis=0)
    
    # Resize to model shape
    factors = [128/a for a in img.shape]
    x_input = np.stack([ndimage.zoom(x[c], factors, order=1) for c in range(4)], axis=0)
    x_tensor = torch.from_numpy(x_input).float().unsqueeze(0).to(device)

    # 3. PREDICT
    with torch.no_grad():
        out = model(x_tensor)
        if isinstance(out, (list, tuple)): out = out[0]
        prob = torch.sigmoid(out).cpu().numpy()[0] # [n_classes, 128, 128, 128]

    # 4. DIAGNOSIS PRINTING
    print("-" * 30)
    print(f"Max Probability in Prediction: {prob.max():.4f}")
    
    for c in range(n_classes):
        pred_mask = (prob[c] > 0.35).astype(np.uint8)
        print(f"Channel {c} | Predicted Voxels: {pred_mask.sum()} | GT Voxels (scaled): {int((lbl>0).sum() * np.prod(factors))}")

    # 5. VISUALIZE (Save to PNG)
    slice_idx = 64
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("MRI Input")
    plt.imshow(x_input[0, :, :, slice_idx], cmap='gray')
    
    plt.subplot(1, 3, 2)
    plt.title("Ground Truth Label")
    plt.imshow(ndimage.zoom((lbl>0).astype(float), factors, order=0)[:, :, slice_idx], cmap='jet')
    
    plt.subplot(1, 3, 3)
    plt.title("AI Prediction (Channel 0)")
    plt.imshow(prob[0, :, :, slice_idx], cmap='jet')
    
    plt.savefig("diagnostic_view.png")
    print("-" * 30)
    print("Result saved to 'diagnostic_view.png'. Download and look at it!")

if __name__ == "__main__":
    main()