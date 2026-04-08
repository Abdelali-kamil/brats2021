import pandas as pd
import matplotlib.pyplot as plt
import torch
from brats import get_datasets
from model import WaveletUNetPlusPlus

def visualize_worst_case():
    device = torch.device("cuda")
    df = pd.read_csv("detailed_results.csv")
    worst_id = df.iloc[0]['Patient_ID'] # أول مريض هو الأقل سكور
    
    dataset = get_datasets()
    # البحث عن بيانات المريض صاحب السكور الأقل
    patient_data = next(item for item in dataset if item["patient_id"] == worst_id)
    image = patient_data['image'].unsqueeze(0).float().to(device)
    label = patient_data['label']

    model = WaveletUNetPlusPlus(in_channels=4, n_classes=3).to(device)
    model.load_state_dict(torch.load("checkpoints/segmentor_epoch_300.pth"))
    model.eval()

    with torch.no_grad():
        probs = torch.sigmoid(model(image))
        pred = (probs > 0.4).float().cpu().squeeze(0)

    # عرض شريحة تحتوي على الورم (مثلاً الشريحة 80)
    slice_idx = 80 
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1); plt.imshow(image[0, 0, slice_idx].cpu(), cmap='gray'); plt.title("T1ce Image")
    plt.subplot(1, 3, 2); plt.imshow(label[2, slice_idx], cmap='Reds', alpha=0.5); plt.title("Ground Truth (WT)")
    plt.subplot(1, 3, 3); plt.imshow(pred[2, slice_idx], cmap='Greens', alpha=0.5); plt.title(f"Prediction (Dice: {df.iloc[0]['Mean_Dice']})")
    plt.savefig("worst_case_visualized.png")
    print(f"✅ Visualization saved for patient {worst_id}")

if __name__ == "__main__":
    visualize_worst_case()