import torch
import numpy as np
from torch.utils.data import DataLoader
from brats import get_datasets
from model import WaveletUNetPlusPlus
import os

def dice_coefficient(y_pred, y_true):
    """
    Calculates the Dice Score:
    - 1.0 means perfect overlap (Excellent)
    - 0.0 means no overlap (Poor)
    """
    smooth = 1.0
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    intersection = (y_pred * y_true).sum()
    return (2. * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Evaluating on Device: {device}")

    # 1. Load Data
    # We use the dataset to calculate accuracy across patients
    dataset = get_datasets(seed=42, debug=False, on="train") 
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # 2. Load Model
    print("📂 Loading Model...")
    model = WaveletUNetPlusPlus(in_channels=4, n_classes=3).to(device)
    
    # Path to your best checkpoint (e.g., epoch 50)
    checkpoint_path = "checkpoints/segmentor_epoch_50.pth"
    
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"✅ Loaded weights from {checkpoint_path}")
    else:
        print(f"❌ Model file not found at {checkpoint_path}")
        print("   Please check the filename in your 'checkpoints' folder.")
        return

    model.eval()
    
    dice_scores = []
    
    print("⏳ Calculating Dice Scores for all patients...")
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            images = batch['image'].float().to(device)
            labels = batch['label'].float().to(device) # [Batch, 3, D, H, W]
            
            # Predict
            logits = model(images)
            probs = torch.sigmoid(logits)
            pred_mask = (probs > 0.5).float()
            
            # Calculate Dice for Channel 1 (Tumor Core) as it is the most critical
            # Labels Channel 1 = Tumor Core
            score = dice_coefficient(pred_mask[:, 1, ...], labels[:, 1, ...])
            dice_scores.append(score.item())
            
            # Print progress every 10 patients
            if (i+1) % 10 == 0:
                print(f"   Patient {i+1}/{len(dataset)} -> Dice Score: {score.item():.4f}")

    # 3. Final Results
    average_dice = np.mean(dice_scores)
    print("-" * 30)
    print(f"🏆 Final Evaluation Results:")
    print(f"   Total Patients Evaluated: {len(dice_scores)}")
    print(f"   Average Dice Score: {average_dice:.4f} ({(average_dice*100):.2f}%)")
    print("-" * 30)

    # Interpretation
    if average_dice > 0.80:
        print("🌟 Excellent Result! Your model is performing at a research level.")
    elif average_dice > 0.60:
        print("✅ Good Result. Can be improved with more training epochs.")
    else:
        print("⚠️ Result needs improvement. Try training for 100 epochs.")

if __name__ == "__main__":
    main()