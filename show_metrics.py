import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from brats import get_datasets
from model import WaveletUNetPlusPlus

def load_checkpoint_safely(model, checkpoint_path, device):
    """ دالة ذكية لضمان تحميل الأوزان بنجاح دون أخطاء """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        if 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)

def visualize_worst_case():
    # 1. إجبار الكود على الـ CPU لمنع الـ Out of Memory نهائياً
    device = torch.device("cpu")
    print(f"🚀 Analyzing Worst Case on Device: {device} (Safe Memory Mode)")
    
    csv_path = "detailed_brats_results.csv"
    if not os.path.exists(csv_path):
        print(f"❌ Error: '{csv_path}' not found! Please check the file path.")
        return
        
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip() # تنظيف مسافات أسماء الأعمدة
    
    # 🔍 طباعة الأعمدة المتاحة في ملفك لحل أي لغز في التسمية
    print(f"📊 Available columns in your CSV: {list(df.columns)}")
    
    # 🌟 كاشف الأعمدة الذكي للبحث عن عمود السكور تلقائياً
    target_col = None
    possible_cols = ['Mean_Dice', 'Mean', 'Dice', 'WT', 'Whole_Tumor', 'Whole Tumor', 'Dice_WT', 'WT_Dice']
    
    for col in possible_cols:
        if col in df.columns:
            target_col = col
            break
            
    # حل احتياطي إذا كانت التسمية غريبة: البحث عن أي عمود يحتوي كلمات دليلة
    if target_col is None:
        for col in df.columns:
            if 'dice' in col.lower() or 'wt' in col.lower() or 'mean' in col.lower():
                target_col = col
                break
        # إذا فشل تماماً، نأخذ العمود الثاني افتراضياً كبديل
        if target_col is None:
            target_col = df.columns[1]
            
    print(f"🎯 Selected Scoring Column: '{target_col}'")
    
    # محاولة معرفة اسم عمود معرف المريض (Patient ID) بنفس الطريقة الذكية
    id_col = 'Patient_ID' if 'Patient_ID' in df.columns else ('Patient' if 'Patient' in df.columns else df.columns[0])
    
    # 2. ترتيب البيانات تصاعدياً بناءً على العمود المكتشف لضمان جلب الأسوأ حقيقياً
    df_sorted = df.sort_values(by=target_col, ascending=True)
    worst_id = df_sorted.iloc[0][id_col]  
    worst_dice = df_sorted.iloc[0][target_col]
    print(f"🔍 True Worst Performing Patient: {worst_id} (Score: {worst_dice:.4f})")
    
    print("⏳ Loading Dataset Structures...")
    dataset = get_datasets(seed=42, debug=False, on="train")
    
    # 3. القفزة الفورية السحرية لمنع تحميل الـ 1251 ملفاً وتجنب تجمّد الكود
    patient_data = None
    if worst_id == "BraTS2021_01477":
        try:
            print(f"🎯 Instant Shortcut Active! Fetching index [1059] directly for {worst_id}...")
            patient_data = dataset[1059]
        except Exception:
            print("⚠️ Shortcut index missed, switching to sequential fallback...")

    # حل احتياطي آمن في حال اختلف الترتيب الداخلي للمجلدات لأي سبب
    if patient_data is None:
        print("🔍 Searching for patient via safe fallback scan...")
        for item in dataset:
            if item["patient_id"] == worst_id:
                patient_data = item
                break

    if patient_data is None:
        print(f"❌ Error: Patient {worst_id} not found in dataset.")
        return

    # نقل البيانات بأمان إلى الـ CPU
    image = patient_data['image'].unsqueeze(0).float().to(device)
    label = patient_data['label']

    # 4. بناء النموذج وتحميل الأوزان للإيبوك 650 حصراً على الـ CPU
    model = WaveletUNetPlusPlus(in_channels=4, n_classes=3).to(device)
    checkpoint_path = "checkpoints/segmentor_epoch_650.pth"
    
    if os.path.exists(checkpoint_path):
        load_checkpoint_safely(model, checkpoint_path, device)
        print(f"✅ Loaded checkpoint successfully on CPU: {checkpoint_path}")
    else:
        print(f"❌ Checkpoint not found at {checkpoint_path}")
        return

    # تشغيل التوقع الآمن
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(image))
        pred = (probs > 0.4).float().cpu().squeeze(0)

    # 5. اختيار الشريحة الذكية التي تحتوي على أكبر كتلة ورمية تلقائياً
    label_np = label.cpu().numpy() if torch.is_tensor(label) else label
    tumor_channel = label_np[2] # قناة الورم بالكامل WT
    
    slice_sums = tumor_channel.sum(axis=(1, 2))
    if slice_sums.max() > 0:
        slice_idx = int(np.argmax(slice_sums))
        print(f"🎯 Automatically selected slice index {slice_idx} (contains largest tumor area)")
    else:
        slice_idx = 75 
        print(f"⚠️ Falling back to slice {slice_idx}")

    img_slice = image[0, 1, slice_idx].cpu().numpy()  
    true_slice = label_np[2, slice_idx]
    pred_slice = pred[2, slice_idx].numpy()

    # 6. الرسم البياني والمقارنة البصرية
    plt.figure(figsize=(15, 5))
    
    # الصورة الأصلية
    plt.subplot(1, 3, 1)
    plt.imshow(img_slice, cmap='gray')
    plt.title(f"Patient {worst_id}\nMRI Slice {slice_idx}")
    plt.axis('off')

    # تراكب قناع الطبيب (Ground Truth) باللون الأحمر الشفاف
    plt.subplot(1, 3, 2)
    plt.imshow(img_slice, cmap='gray') 
    masked_true = np.ma.masked_where(true_slice == 0, true_slice)
    plt.imshow(masked_true, cmap='Reds', alpha=0.6) 
    plt.title("Ground Truth (WT)")
    plt.axis('off')

    # تراكب توقع النموذج (Prediction) باللون الأخضر الشفاف
    plt.subplot(1, 3, 3)
    plt.imshow(img_slice, cmap='gray') 
    masked_pred = np.ma.masked_where(pred_slice == 0, pred_slice)
    plt.imshow(masked_pred, cmap='Greens', alpha=0.6) 
    plt.title(f"Prediction (Dice: {worst_dice:.4f})")
    plt.axis('off')

    # حفظ الصورة النهائية تلقائياً وبدقة ممتازة
    output_filename = "worst_case_visualized.png"
    plt.savefig(output_filename, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"🖼️ Finished! Visualization saved as '{output_filename}' using CPU & Epoch 650.\n")

if __name__ == "__main__":
    visualize_worst_case()