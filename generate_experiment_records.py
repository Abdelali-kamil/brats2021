import torch
import torch.nn.functional as F
import nibabel as nib
import numpy as np
import pandas as pd
import os
import gc
from model import WaveletUNetPlusPlus
from skimage.transform import resize

def dice_coefficient(y_pred, y_true):
    smooth = 1.0
    y_pred = (y_pred > 0.5).float()
    y_true = (y_true > 0.5).float()
    intersection = (y_pred * y_true).sum()
    return (2. * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- المسارات الصحيحة بناءً على جهازك ---
    DATA_DIR = "/home/kamilabdelali/brats2021/data"
    CHECKPOINT = "checkpoints/ULTIMATE_model_epoch_400.pth"
    OUTPUT_CSV = "detailed_results_combined.csv"
    
    # التأكد من وجود ملف الأوزان
    if not os.path.exists(CHECKPOINT):
        print(f"❌ Error: Checkpoint not found at {CHECKPOINT}")
        return

    print(f"🚀 Loading weights from: {CHECKPOINT}")
    model = WaveletUNetPlusPlus(in_channels=4, n_classes=3).to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.eval()
    
    results = []
    # الحصول على أسماء مجلدات المرضى (BraTS2021_00000, إلخ)
    patient_folders = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])

    print(f"📊 Processing {len(patient_folders)} patients...")

    with torch.no_grad():
        for patient_id in patient_folders:
            try:
                # تحديد المسارات داخل مجلد كل مريض
                patient_path = os.path.join(DATA_DIR, patient_id)
                
                # في BraTS2021 الأسماء تتبع نمط: PatientID_flair.nii.gz و PatientID_seg.nii.gz
                img_path = os.path.join(patient_path, f"{patient_id}_flair.nii.gz")
                label_path = os.path.join(patient_path, f"{patient_id}_seg.nii.gz")
                
                if not os.path.exists(img_path):
                    print(f"⚠️ Skipping {patient_id}: Image not found")
                    continue

                # تحميل ومعالجة الصورة
                img_data = nib.load(img_path).get_fdata()
                p1, p99 = np.percentile(img_data, [1, 99])
                img_data = np.clip(img_data, p1, p99)
                img_data = (img_data - np.mean(img_data)) / (np.std(img_data) + 1e-8)
                img_resized = resize(img_data, (128, 128, 128), order=1, preserve_range=True)
                
                # إعداد المدخلات (4 قنوات)
                img_input = np.stack([img_resized]*4, axis=0)
                img_input = torch.from_numpy(img_input).float().unsqueeze(0).to(device)

                # التوقع
                logits = model(img_input)
                probs = torch.sigmoid(logits)

                # حساب الـ Dice إذا كان الـ Label موجوداً
                if os.path.exists(label_path):
                    label_data = (nib.load(label_path).get_fdata() > 0).astype(np.float32)
                    label_resized = resize(label_data, (128, 128, 128), order=0, preserve_range=True)
                    label_tensor = torch.from_numpy(label_resized).to(device)

                    # القنوات: 0=WT, 1=TC, 2=ET
                    dice_wt = dice_coefficient(probs[0, 0], label_tensor).item()
                    dice_tc = dice_coefficient(probs[0, 1], label_tensor).item()
                    dice_et = dice_coefficient(probs[0, 2], label_tensor).item()
                    mean_dice = (dice_wt + dice_tc + dice_et) / 3

                    results.append({
                        "Patient_ID": patient_id,
                        "Dice_WT": round(dice_wt, 4),
                        "Dice_TC": round(dice_tc, 4),
                        "Dice_ET": round(dice_et, 4),
                        "Mean_Dice": round(mean_dice, 4)
                    })
                    print(f"✅ {patient_id} | Mean Dice: {mean_dice:.4f}")

                # تنظيف الذاكرة
                del img_input, logits, probs, img_data, img_resized
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"❌ Error in {patient_id}: {e}")

    # حفظ السجلات النهائية
    if results:
        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n✨ Done! Experiment records saved to: {OUTPUT_CSV}")
        print(f"📈 Final Mean Dice: {df['Mean_Dice'].mean():.4f}")
    else:
        print("❌ No results generated. Check your data paths.")

if __name__ == "__main__":
    main()