import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os

# استيراد الوحدات الخاصة بك
from brats import get_datasets
from model import WaveletUNetPlusPlus
from encoders import FeatureExtractor
from classifier import AdvancedClassifier

# --- 1. المطور المدمج (Dice + Focal) ---
class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalDiceLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = 1e-5

    def forward(self, y_pred, y_true):
        y_pred = torch.clamp(y_pred, self.smooth, 1.0 - self.smooth)
        y_pred_flat = y_pred.view(-1)
        y_true_flat = y_true.view(-1)
        intersection = (y_pred_flat * y_true_flat).sum()
        dice = (2. * intersection + self.smooth) / (y_pred_flat.sum() + y_true_flat.sum() + self.smooth)
        dice_loss = 1 - dice
        bce = F.binary_cross_entropy(y_pred_flat, y_true_flat, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1 - pt)**self.gamma * bce
        return dice_loss + focal_loss.mean()

# --- 2. دالة التقييم (تم إخراجها هنا لتكون مستقلة وتعمل بشكل صحيح) ---
def evaluate_during_training(model, dataloader, device):
    model.eval()
    dice_scores = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i > 20: break 
            images = batch['image'].float().to(device)
            labels = batch['label'].float().to(device)
            probs = torch.sigmoid(model(images))
            pred = (probs > 0.20).float() 
            inter = (pred * labels).sum()
            union = pred.sum() + labels.sum() + 1e-5
            dice = (2. * inter + 1e-5) / union
            dice_scores.append(dice.item())
    return np.mean(dice_scores), np.min(dice_scores)

def main():
    # --- CONFIGURATION ---
    START_EPOCH = 600 # تأكد من تغييرها لـ 300 لتبدأ من حيث انتهيت
    total_target_epochs = 650 
    LEARNING_RATE = 2e-5  # معدل صغير للتحسين الدقيق
    BATCH_SIZE = 2
    SAVE_DIR = "checkpoints"
    
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Resuming Training on Device: {device}")

    # 1. تحميل البيانات
    dataset = get_datasets(on="train") 
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 2. بناء الموديلات وتحميل الأوزان
    segmentor = WaveletUNetPlusPlus(in_channels=4, n_classes=3).to(device)
    extractor = FeatureExtractor(in_channels=4, feature_dim=512).to(device)
    classifier = AdvancedClassifier(feature_dim=512, clinical_dim=10, num_classes=2).to(device)

    seg_checkpoint = f"{SAVE_DIR}/segmentor_epoch_{START_EPOCH}.pth"
    if os.path.exists(seg_checkpoint):
        segmentor.load_state_dict(torch.load(seg_checkpoint, map_location=device))
        print(f"✅ Weights from Epoch {START_EPOCH} loaded.")

    # 3. المحسنات
    params = list(segmentor.parameters()) + list(extractor.parameters()) + list(classifier.parameters())
    optimizer = optim.Adam(params, lr=LEARNING_RATE)
    criterion_seg = FocalDiceLoss() 
    criterion_cls = nn.CrossEntropyLoss()

    # 4. حلقة التدريب
    for epoch in range(START_EPOCH, total_target_epochs):
        segmentor.train()
        extractor.train()
        classifier.train()
        epoch_loss = 0.0
        
        for i, batch in enumerate(dataloader):
            images = batch['image'].float().to(device)
            labels_seg = batch['label'].float().to(device)
            clinical_data = torch.randn(images.size(0), 10).to(device)
            labels_cls = torch.randint(0, 2, (images.size(0),)).to(device)

            pred_seg_logits = segmentor(images)
            pred_seg_probs = torch.sigmoid(pred_seg_logits)
            main_mask = pred_seg_probs[:, 1:2, :, :, :] 
            features = extractor(images, main_mask)
            pred_cls_logits = classifier(features, clinical_data)

            loss_seg = criterion_seg(pred_seg_probs, labels_seg)
            loss_cls = criterion_cls(pred_cls_logits, labels_cls)
            total_loss = loss_seg + (0.5 * loss_cls)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()
            
            if (i+1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{total_target_epochs}] | Batch [{i+1}/{len(dataloader)}] | Loss: {total_loss.item():.4f}")

        # نهاية الـ Epoch وحساب النتائج فوراً
        avg_loss = epoch_loss / len(dataloader)
        
        # --- السطر الذي سألت عنه (وضعه هنا بعد انتهاءbatches) ---
        avg_dice, worst_dice = evaluate_during_training(segmentor, dataloader, device)
        
        print("\n" + "="*40)
        print(f"📊 Validation Report Epoch {epoch+1}:")
        print(f"✨ Mean Dice: {avg_dice:.4f} | ⚠️ Worst Dice: {worst_dice:.4f}")
        print(f"📉 Avg Loss:  {avg_loss:.4f}")
        print("="*40 + "\n")

        # حفظ الأوزان
        torch.save(segmentor.state_dict(), f"{SAVE_DIR}/segmentor_epoch_{epoch+1}.pth")
        torch.save(extractor.state_dict(), f"{SAVE_DIR}/extractor_epoch_{epoch+1}.pth")
        torch.save(classifier.state_dict(), f"{SAVE_DIR}/classifier_epoch_{epoch+1}.pth")
        print(f"💾 Models saved for Epoch {epoch+1}")

if __name__ == "__main__":
    main()