import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from brats import get_datasets
from model import WaveletUNetPlusPlus

def load_checkpoint_safely(model, checkpoint_path, device):
    """ دالة ذكية لضمان تحميل أوزان النموذج بنجاح دون أي أخطاء في الـ State Dict """
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
    # 🌟 إجبار الكود على العمل على الـ CPU لتجنب مشكلة الـ CUDA Out of Memory تماماً
    device = torch.device("cpu")
    print(f"\n🚀 Analyzing Worst Case on Device: {device} (Safe Memory Mode)")
    
    # 1. قراءة وترتيب ملف النتائج الرقمية
    csv_path = "detailed_results.csv"
    if not os.path.exists(csv_path):
        print(f"❌ Error: '{csv_path}' not found! Please run evaluation first.")
        return
        
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip() # تنظيف الفراغات من أسماء الأعمدة
    
    # ترتيب البيانات تصاعدياً ليكون المريض صاحب أقل سكور (الأسوأ) هو الأول دائماً
    df_sorted = df.sort_values(by='Mean_Dice', ascending=True)
    
    worst_id = df_sorted.iloc[0]['Patient_ID']  
    worst_dice = df_sorted.iloc[0]['Mean_Dice']
    print(f"🔍 True Worst Performing Patient: {worst_id} (Whole 3D Dice: {worst_dice:.4f})")
    
    # 2. تحميل بنية البيانات 
    print("⏳ Loading Dataset Structures...")
    dataset = get_datasets(seed=42, debug=False, on="train")
    
    # 3. النفاذ الفوري الفائق لمنع تجمّد الكود أو الانتظار طويلأً
    patient_data = None
    
    # قفزة سحرية: بما أننا علمنا برمجياً أن مريض الـ 10% يقع عند المؤشر 1059، سنطلبه مباشرة!
    if worst_id == "BraTS2021_01477":
        try:
            print(f"🎯 Instant Shortcut Active! Fetching index [1059] directly for {worst_id}...")
            patient_data = dataset[1059]
        except Exception:
            print("⚠️ Shortcut index missed, switching to meta-scan...")

    # حل احتياطي سريع جداً إذا لم ينجح الاختصار المباشر
    if patient_data is None:
        print("🔍 Searching for patient index using fast meta-scan...")
        patient_idx = None
        if hasattr(dataset, 'data') and isinstance(dataset.data, list):
            for idx, item in enumerate(dataset.data):
                if isinstance(item, dict):
                    if item.get('patient_id') == worst_id or ('image' in item and worst_id in str(item['image'])):
                        patient_idx = idx
                        break
        
        if patient_idx is not None:
            print(f"🎯 Found via meta-scan at index [{patient_idx}]. Loading single brain scan...")
            patient_data = dataset[patient_idx]

    # حل احتياطي نهائي بمؤشر تقدم مرئي إذا فشلت كل الطرق السابقة
    if patient_data is None:
        print("⚠️ Direct paths missed. Searching sequentially with progress tracker...")
        for idx, item in enumerate(dataset):
            if idx % 50 == 0:
                print(f"🔄 Progress: Checked {idx}/1251 patients... Please wait.")
            if item["patient_id"] == worst_id:
                patient_data = item
                break

    if patient_data is None:
        print(f"❌ Error: Patient {worst_id} not found in dataset mapping.")
        return

    # استخراج وتجهيز أشعة المريض على الـ CPU
    image = patient_data['image'].unsqueeze(0).float().to(device)
    label = patient_data['label']

    # 4. بناء النموذج وتغذية أوزان الإيبوك 650 حصراً على الـ CPU
    model = WaveletUNetPlusPlus(in_channels=4, n_classes=3).to(device)
    checkpoint_path = "checkpoints/segmentor_epoch_650.pth"
    
    if os.path.exists(checkpoint_path):
        load_checkpoint_safely(model, checkpoint_path, device)
        print(f"✅ Loaded checkpoint successfully on CPU: {checkpoint_path}")
    else:
        print(f"❌ Checkpoint not found at {checkpoint_path}")
        return

    # تشغيل وضع التقييم والتوقع بدون استهلاك الذاكرة (No Gradient)
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(image))
        pred = (probs > 0.4).float().cpu().squeeze(0)

    # 5. اختيار الشريحة (Slice) التي تحتوي على أكبر كتلة ورمية تلقائياً
    label_np = label.cpu().numpy() if torch.is_tensor(label) else label
    tumor_channel = label_np[2] # قناة الورم بالكامل (Whole Tumor)
    
    slice_sums = tumor_channel.sum(axis=(1, 2))
    if slice_sums.max() > 0:
        slice_idx = int(np.argmax(slice_sums))
        print(f"🎯 Automatically selected slice index {slice_idx} (contains largest tumor area)")
    else:
        slice_idx = 75 
        print(f"⚠️ Tumor mask is empty or sparse. Falling back to slice {slice_idx}")

    # استخراج الشرائح ثنائية الأبعاد للرسم
    img_slice = image[0, 1, slice_idx].cpu().numpy() # قناة الأشعة الأصلية
    true_slice = label_np[2, slice_idx]
    pred_slice = pred[2, slice_idx].numpy()

    # 6. رسم ومقارنة النتائج بصرياً باستخدام Matplotlib
    plt.figure(figsize=(15, 5))
    
    # الجزء الأول: صورة الأشعة الأصلية للمريض
    plt.subplot(1, 3, 1)
    plt.imshow(img_slice, cmap='gray')
    plt.title(f"Patient: {worst_id}\nMRI Slice: {slice_idx}")
    plt.axis('off')

    # الجزء الثاني: قناع الحقيقة الطبي (Ground Truth) باللون الأحمر الشفاف
    plt.subplot(1, 3, 2)
    plt.imshow(img_slice, cmap='gray') 
    masked_true = np.ma.masked_where(true_slice == 0, true_slice)
    plt.imshow(masked_true, cmap='Reds', alpha=0.6) 
    plt.title("Ground Truth (WT)")
    plt.axis('off')

    # الجزء الثالث: قناع توقع الذكاء الاصطناعي (Prediction) باللون الأخضر الشفاف
    plt.subplot(1, 3, 3)
    plt.imshow(img_slice, cmap='gray') 
    masked_pred = np.ma.masked_where(pred_slice == 0, pred_slice)
    plt.imshow(masked_pred, cmap='Greens', alpha=0.6) 
    plt.title(f"Prediction (Dice: {worst_dice:.4f})")
    plt.axis('off')

    # حفظ الصورة النهائية تلقائياً وبجودة عالية
    output_filename = "worst_case_visualized.png"
    plt.savefig(output_filename, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"🖼️ Success! Visualization saved as '{output_filename}' using CPU & Epoch 650.\n")

if __name__ == "__main__":
    visualize_worst_case()