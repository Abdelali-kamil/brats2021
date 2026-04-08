import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import pandas as pd
from brats import get_datasets
from model import WaveletUNetPlusPlus
import os

def fix_low_dice():
    device = torch.device("cuda")
    
    # 1. قراءة المرضى الفاشلين من التقرير الأخير
    if not os.path.exists("detailed_results.csv"):
        print("❌ Error: Run analyze_results.py first!")
        return
    
    df = pd.read_csv("detailed_results.csv")
    failed_ids = df[df['Mean_Dice'] < 0.1]['Patient_ID'].tolist()
    
    if len(failed_ids) == 0:
        print("✅ No patients below 0.1 found!")
        return

    print(f"🛠️ Found {len(failed_ids)} hard cases. Starting Targeted Fine-tuning...")

    # 2. تجهيز البيانات لهذه الحالات فقط
    full_dataset = get_datasets()
    indices = [i for i, s in enumerate(full_dataset) if s['patient_id'] in failed_ids]
    subset_dataset = Subset(full_dataset, indices)
    train_loader = DataLoader(subset_dataset, batch_size=1, shuffle=True)

    # 3. تحميل الموديل الحالي (أوزان 300)
    model = WaveletUNetPlusPlus(in_channels=4, n_classes=3).to(device)
    model.load_state_dict(torch.load("checkpoints/segmentor_epoch_300.pth"))
    
    # 4. استخدام Learning Rate منخفض جداً (للحفاظ على ذكاء الموديل العام)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-6)
    
    # دالة خسارة تركز على البكسلات الصغيرة (BCE + Dice)
    criterion_bce = nn.BCEWithLogitsLoss()

    # 5. تدريب مكثف لـ 15 دورة فقط على الحالات الصعبة
    model.train()
    for epoch in range(15):
        epoch_loss = 0
        for batch in train_loader:
            img = batch['image'].float().to(device)
            mask = batch['label'].float().to(device)
            
            optimizer.zero_grad()
            pred = model(img)
            loss = criterion_bce(pred, mask)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        print(f"🔄 Epoch [{epoch+1}/15] - Loss: {epoch_loss/len(train_loader):.6f}")

    # 6. حفظ الأوزان المنقذة
    if not os.path.exists("checkpoints"): os.makedirs("checkpoints")
    torch.save(model.state_dict(), "checkpoints/segmentor_epoch_300_FIXED.pth")
    print("✅ Done! Saved as: checkpoints/segmentor_epoch_300_FIXED.pth")

if __name__ == "__main__":
    fix_low_dice()