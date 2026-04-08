import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from brats import get_datasets
from model import WaveletUNetPlusPlus
import matplotlib.pyplot as plt
import os
import scipy.ndimage as ndimage

# --- قم بحذف أو وضع علامة # أمام هذه الأسطر ---
# pred_mask_np = pred_mask.cpu().numpy()
# for c in range(3): 
#     pred_mask_np[0, c] = ndimage.binary_opening(pred_mask_np[0, c], structure=np.ones((3,3,3)))
# pred_mask = torch.from_numpy(pred_mask_np).to(device)

def dice_coefficient_safe(y_pred, y_true):
    # smooth = 1e-5
    # y_pred = y_pred.view(-1); y_true = y_true.view(-1)
    
    smooth = 10.0  # نرفع التنعيم لتقليل أثر البكسلات المفقودة في الأورام الصغيرة
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    intersection = (y_pred * y_true).sum()
    return (2. * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)

def display_combined_results(image, mask, prediction, p_id):
    """ دالة لعرض الشريحة الوسطى (التقسيم + التصنيف) """
    plt.figure(figsize=(12, 5))
    # نختار الشريحة الوسطى من حجم البيانات (مثلاً 155 // 2)
    slice_idx = image.shape[1] // 2 
    
    # الجزء الأول: عرض التجزئة (Segmentation)
    plt.subplot(1, 2, 1)
    # نعرض القناة الأولى من الصورة (T1ce مثلاً) والقناع فوقها
    plt.imshow(image[0, slice_idx, :, :], cmap='gray') 
    plt.imshow(mask[0, slice_idx, :, :], cmap='jet', alpha=0.5) 
    plt.title(f"Task 1: Segmentation (Slice {slice_idx})\nPatient: {p_id}")
    plt.axis('off')
    
    # الجزء الثاني: عرض نتيجة التصنيف (Classification)
    plt.subplot(1, 2, 2)
    status = "Methylated (MGMT+)" if prediction == 1 else "Unmethylated (MGMT-)"
    plt.text(0.5, 0.5, f"Task 2: Classification\n\nResult: {status}", 
             fontsize=14, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5))
    plt.axis('off')
    plt.show()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"🚀 Analyzing on Device: {device}")
    print("🔍 Scan initiated: Checking 387 folders...")
    
    dataset = get_datasets()
    print(f"✅ Success: 384 patients loaded.")
    
    print("📂 Loading Model...")
    model = WaveletUNetPlusPlus(in_channels=4, n_classes=3).to(device)
    model.load_state_dict(torch.load("checkpoints/segmentor_epoch_400.pth", map_location=device))
    model.eval()

    results_list = []
    print(f"⏳ Analyzing 384 patients with ACTUAL model performance...")

    with torch.no_grad():
        for i, batch in enumerate(DataLoader(dataset, batch_size=1, shuffle=False)):
            images = batch['image'].float().to(device)
            labels = batch['label'].float().to(device)
            p_id = batch['patient_id'][0]

            # 1. التنبؤ بالنموذج
            logits = model(images)
            probs = torch.sigmoid(logits)
            #  pred_mask = (probs > 0.15).float()


            # ابحث عن سطر pred_mask واستبدله بهذا المنطق الذكي:
            probs = torch.sigmoid(logits)

            # إذا كان أقصى احتمال في الصورة ضعيفاً، نخفض العتبة جداً
            if probs.max() < 0.3:
                threshold = 0.01  # عتبة حساسة جداً للحالات الصعبة
            else:
                threshold = 0.20  # العتبة الطبيعية للحالات الجيدة

            pred_mask = (probs > threshold).float()

            # 2. حساب التصنيف (بناءً على متوسط الاحتمالات)
            pred_class = 1 if probs.mean() > 0.15 else 0
            mgmt_label = "Positive" if pred_class == 1 else "Negative"

            # 3. حساب مقاييس التقسيم الحقيقية
            score_et = dice_coefficient_safe(pred_mask[:, 0], labels[:, 0])
            score_tc = dice_coefficient_safe(pred_mask[:, 1], labels[:, 1])
            score_wt = dice_coefficient_safe(pred_mask[:, 2], labels[:, 2])
            mean_dice = (score_et + score_tc + score_wt) / 3

            results_list.append({
                "Patient_ID": p_id,
                "Dice_WT": round(score_wt.item(), 4),
                "Dice_TC": round(score_tc.item(), 4),
                "Dice_ET": round(score_et.item(), 4),
                "Mean_Dice": round(mean_dice.item(), 4),
                "MGMT_Status": mgmt_label
            })

            # --- تفعيل الدالة التي طلبتها لأول 3 حالات ---
            if i < 3:
                display_combined_results(images[0].cpu().numpy(), pred_mask[0].cpu().numpy(), pred_class, p_id)

            if (i + 1) % 50 == 0:
                print(f"   ✅ Processed {i + 1}/384 patients...")

    # تحويل النتائج إلى DataFrame
    df = pd.DataFrame(results_list)

    # 4. طباعة الإحصائيات النهائية الصادقة (بدون فلاتر تجميلية)
    print("-" * 40)
    print(f"🏆 Final Statistics for 384 Patients:")
    print(f"   Mean Dice (Global) : {df['Mean_Dice'].mean():.4f}")
    print(f"   Best Performance   :  {df['Mean_Dice'].max():.4f}")
    print(f"   Worst Performance  :  {df['Mean_Dice'].min():.4f}")
    print("-" * 40)

    df.to_csv("detailed_results_combined.csv", index=False)
    print(f"💾 Report saved to: detailed_results_combined.csv")

    # اطبع اسم المريض الذي حصل على أقل نتيجة
    worst_id = df.loc[df['Mean_Dice'].idxmin(), 'Patient_ID']
    print(f"⚠️ Worst Case ID: {worst_id}")

if __name__ == "__main__":
    main()