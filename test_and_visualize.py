import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

from brats import get_datasets
from model import WaveletUNetPlusPlus

def visualize_prediction(patient_id, image, true_mask, pred_mask, save_path):
    """
    Function to plot 3 images side-by-side: Original MRI, Ground Truth, and Model Prediction.
    """
    # We take Slice index 70 because it is usually in the center of the brain 
    # and shows the tumor clearly.
    slice_idx = 70 
    
    # Convert tensor to standard numpy array for plotting
    # Image channel 2 is usually T2 or FLAIR which shows tumor well
    img_slice = image[2, slice_idx, :, :].cpu().numpy()
    
    # Channel 1 is usually the tumor core in the label/mask
    true_slice = true_mask[1, slice_idx, :, :].cpu().numpy() 
    pred_slice = pred_mask[1, slice_idx, :, :].cpu().numpy()

    plt.figure(figsize=(15, 5))
    
    # 1. Original Image (MRI)
    plt.subplot(1, 3, 1)
    plt.title(f"Patient {patient_id} - MRI (FLAIR)")
    plt.imshow(img_slice, cmap='gray')
    plt.axis('off')

    # 2. Ground Truth (Doctor's Annotation)
    plt.subplot(1, 3, 2)
    plt.title("Ground Truth (Doctor)")
    plt.imshow(true_slice, cmap='gray')
    plt.axis('off')

    # 3. AI Prediction (Wavelet U-Net)
    plt.subplot(1, 3, 3)
    plt.title("AI Prediction (Wavelet U-Net)")
    plt.imshow(pred_slice, cmap='gray')
    plt.axis('off')

    # Save the figure
    plt.savefig(save_path)
    plt.close()
    print(f"🖼️ Saved visualization to: {save_path}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Testing on Device: {device}")

    # 1. Load Data
    # Note: We rely on the robustness check in brats.py to skip corrupted files
    dataset = get_datasets(seed=42, debug=False, on="train") 
    
    # We create a dataloader with shuffle=False to check specific patients sequentially
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 2. Load the Trained Model
    print("📂 Loading Trained Model...")
    model = WaveletUNetPlusPlus(in_channels=4, n_classes=3).to(device)
    
    # Path to the model checkpoint
    # IMPORTANT: Change '50' to the actual epoch number you want to test (e.g., 50, 100, etc.)
    # Since you resumed training, you likely have 'segmentor_epoch_50.pth' or higher.
    checkpoint_path = "checkpoints/segmentor_epoch_300.pth" 
    
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"✅ Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"❌ Checkpoint not found at {checkpoint_path}. Please check the filename in 'checkpoints' folder.")
        # Attempt to look for epoch 1 just in case, for testing purposes
        fallback_path = "checkpoints/segmentor_epoch_1.pth"
        if os.path.exists(fallback_path):
             print(f"⚠️ Falling back to Epoch 1 for testing: {fallback_path}")
             model.load_state_dict(torch.load(fallback_path, map_location=device))
        else:
             return

    model.eval() # Set model to evaluation mode (stops training updates)

    # 3. Create Results Directory
    os.makedirs("results", exist_ok=True)

    print("⏳ Generating predictions...")
    
    with torch.no_grad(): # No gradient calculation needed for testing
        for i, batch in enumerate(dataloader):
            # Test on the first 5 patients only
            if i >= 5: break 
            
            images = batch['image'].float().to(device)
            labels = batch['label'].float().to(device)
            p_id = batch['patient_id'][0]

            # Make Prediction
            logits = model(images)
            probs = torch.sigmoid(logits) # Convert logits to 0-1 probabilities
            
            # Convert probabilities to binary mask (0 or 1) using 0.5 threshold
            pred_mask = (probs > 0.5).float()

            # Visualize and Save
            visualize_prediction(
                p_id, 
                images[0], 
                labels[0], 
                pred_mask[0], 
                save_path=f"results/result_{p_id}.png"
            )

    print("\n✅ Done! Check the 'results' folder to see the images.")

if __name__ == "__main__":
    main()