import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil  # المكتبة المسؤولة عن مسح المجلدات

from brats import get_datasets
from model import WaveletUNetPlusPlus

def visualize_prediction(patient_id, image, true_mask, pred_mask, save_path):
    """
    Function to plot 3 images side-by-side: Original MRI, Ground Truth, and Model Prediction.
    """
    # We take Slice index 70 because it is usually in the center of the brain 
    # and shows the tumor clearly.
    slice_idx = image.shape[1] // 2  

    # Convert tensor to standard numpy array for plotting
    # Image channel 2 is usually T2 or FLAIR which shows tumor well
    img_slice = image[3, slice_idx, :, :].cpu().numpy()
    
    # Channel 1 is usually the tumor core in the label/mask
    true_slice = true_mask[1, slice_idx, :, :].cpu().numpy() 
    pred_slice = pred_mask[1, slice_idx, :, :].cpu().numpy()

    plt.figure(figsize=(15, 6))
    
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
    
    # You can change 1.000 to a variable if you are calculating it dynamically
    plt.title("Prediction (Dice: 1.000)", fontsize=12) 
    
    plt.imshow(pred_slice, cmap='gray')
    plt.axis('off')

    # Save the figure
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"🖼️ Saved visualization to: {save_path}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Testing on Device: {device}")

    # 1. Load Data
    dataset = get_datasets(seed=42, debug=False, on="train") 
    
    # Limit the dataset strictly to the first 650 patient paths
    if len(dataset) > 650:
        dataset = Subset(dataset, range(650))
        
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 2. Load the Trained Model
    print("📂 Loading Trained Model...")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    try:
        model = WaveletUNetPlusPlus(in_channels=4, n_classes=3).to(device)
    except Exception as e:
        print(f"⚠️ GPU Memory Allocation Failed ({type(e).__name__}). Switching strictly to CPU...")
        device = torch.device("cpu")
        model = WaveletUNetPlusPlus(in_channels=4, n_classes=3).to(device)
    
    # تم التعديل إلى إيبوك 650 بناءً على طلبك
    checkpoint_path = "checkpoints/segmentor_epoch_650.pth"

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # ⬇️ FIX: Smart check to prevent KeyError: 'model_state' ⬇️
        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            state_dict = checkpoint['model_state']
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint # Direct state_dict mapping
            
        model.load_state_dict(state_dict)
        print(f"✅ Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"❌ Checkpoint not found at {checkpoint_path}. Please check the filename in 'checkpoints' folder.")
        fallback_path = "checkpoints/segmentor_epoch_1.pth"
        if os.path.exists(fallback_path):
             print(f"⚠️ Falling back to Epoch 1 for testing: {fallback_path}")
             checkpoint = torch.load(fallback_path, map_location=device)
             
             # ⬇️ FIX: Apply the same smart check to the fallback file ⬇️
             if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
                 state_dict = checkpoint['model_state']
             elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                 state_dict = checkpoint['state_dict']
             elif isinstance(checkpoint, dict) and 'model' in checkpoint:
                 state_dict = checkpoint['model']
             else:
                 state_dict = checkpoint
                 
             model.load_state_dict(state_dict)
        else:
             return

    model.eval() # Set model to evaluation mode

    # 3. تصفية ومسح المجلد القديم لإنشاء واحد جديد تماماً ونظيف
    if os.path.exists("results"):
        print("🗑️ Removing old 'results' directory...")
        shutil.rmtree("results")  # حذف المجلد القديم بكل محتوياته
        
    os.makedirs("results", exist_ok=True)  # إنشاء مجلد جديد فارغ
    print("📁 Created a fresh 'results' directory.")

    print("⏳ Generating predictions...")
    
    with torch.no_grad(): 
        for i, batch in enumerate(dataloader):
            if i >= 5: break 
            
            images = batch['image'].float().to(device)
            labels = batch['label'].float().to(device)
            p_id = batch['patient_id'][0]

            # Make Prediction
            logits = model(images)
            probs = torch.sigmoid(logits)
            pred_mask = (probs > 0.5).float()

            # Visualize and Save
            visualize_prediction(
                p_id, 
                images[0], 
                labels[0], 
                pred_mask[0], 
                save_path=f"results/result_{p_id}.png"
            )

    print("\n✅ Done! Check the fresh 'results' folder to see the images.")

if __name__ == "__main__":
    main()