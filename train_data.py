import torch
import torch.optim as optim
import torch.nn as nn
import nibabel as nib
import numpy as np
import os
import gc
from model import WaveletUNetPlusPlus
from skimage.transform import resize
from torch.utils.data import Dataset, DataLoader

# 1. كلاس تحميل البيانات مع المعالجة الكاملة
class PontineDataset(Dataset):
    def __init__(self, images_dir, labels_dir, file_list):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        img_path = os.path.join(self.images_dir, img_name)
        label_path = os.path.join(self.labels_dir, img_name)

        # تحميل البيانات
        img_data = nib.load(img_path).get_fdata()
        label_data = nib.load(label_path).get_fdata()

        # Z-score Normalization
        p1, p99 = np.percentile(img_data, [1, 99])
        img_data = np.clip(img_data, p1, p99)
        img_data = (img_data - np.mean(img_data)) / (np.std(img_data) + 1e-8)
        img_data = resize(img_data, (128, 128, 128), order=1, preserve_range=True)

        # Binary Label Resize
        label_data = (label_data > 0).astype(np.float32)
        label_data = resize(label_data, (128, 128, 128), order=0, preserve_range=True)

        img_input = np.stack([img_data]*4, axis=0)
        label_input = np.stack([label_data]*3, axis=0)

        return torch.from_numpy(img_input).float(), torch.from_numpy(label_input).float()

def train_full_dataset():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    IMAGES_DIR = "/home/kamilabdelali/anotherdata/images"
    LABELS_DIR = "/home/kamilabdelali/anotherdata/labels"
    # سنبدأ من آخر نجاح وصلت له (الـ 0.50 Dice) لنبني عليه
    LAST_CHECKPOINT = "checkpoints/fine_tuned_100_epoch_50.pth" 
    
    all_files = sorted([f for f in os.listdir(IMAGES_DIR) if f.endswith('.nii.gz')])
    
    # تقسيم البيانات (90% تدريب، 10% تحقق)
    split_idx = int(len(all_files) * 0.9)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]

    print(f"🚀 Training on ALL data: {len(train_files)} cases | Val: {len(val_files)} cases")

    train_loader = DataLoader(PontineDataset(IMAGES_DIR, LABELS_DIR, train_files), batch_size=1, shuffle=True)
    val_loader = DataLoader(PontineDataset(IMAGES_DIR, LABELS_DIR, val_files), batch_size=1, shuffle=False)

    model = WaveletUNetPlusPlus(in_channels=4, n_classes=3).to(device)
    model.load_state_dict(torch.load(LAST_CHECKPOINT, map_location=device))
    
    optimizer = optim.Adam(model.parameters(), lr=5e-6) # Learning Rate أصغر للحفاظ على الاستقرار
    criterion = nn.BCEWithLogitsLoss() 

    num_epochs = 100 # بما أن البيانات كثيرة، سنعطي الموديل 100 دورة ليتشبع بالمعرفة
    
    print("🔥 Starting the ULTIMATE Training Phase...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            del images, labels, outputs
            torch.cuda.empty_cache()

        # التحقق (Validation)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                del images, labels, outputs

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss/len(train_loader):.6f} | Val Loss: {val_loss/len(val_loader):.6f}")
        
        # حفظ الأوزان كل 20 دورة
        if (epoch + 1) % 20 == 0:
            save_path = f"checkpoints/ULTIMATE_model_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"💾 Saved: {save_path}")

    print("✅ MISSION ACCOMPLISHED! Your model is now trained on the full dataset.")

if __name__ == "__main__":
    train_full_dataset()