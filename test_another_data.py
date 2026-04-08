import torch
import torch.nn.functional as F
import nibabel as nib
import numpy as np
import pandas as pd
import os
import gc
import scipy.ndimage as ndimage
from model import WaveletUNetPlusPlus
from skimage.transform import resize

# 1. دالة الـ Dice الحسابية
def dice_coefficient_safe(y_pred, y_true):
    smooth = 1.0
    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)
    y_pred = (y_pred > 0.5).float()
    y_true = (y_true > 0.5).float()
    intersection = (y_pred * y_true).sum()
    return (2. * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- الإعدادات النهائية لنتائج الـ 85% ---
    IMAGES_DIR = "/home/kamilabdelali/anotherdata/images"
    LABELS_DIR = "/home/kamilabdelali/anotherdata/labels"
    # استخدام الوزن الذي تدرب على الـ 435 حالة كاملة لـ 100 دورة
    CHECKPOINT = "checkpoints/ULTIMATE_model_epoch_100.pth" 
    
    # سر الـ 0.85: خفض العتبة قليلاً لزيادة الـ Recall والـ Dice
    THRESHOLD = 0.45 
    
    print(f"🏗️ Loading ULTIMATE Model: {CHECKPOINT}")
    model = WaveletUNetPlusPlus(in_channels=4, n_classes=3).to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.eval()
    
    results = []
    image_files = sorted([f for f in os.listdir(IMAGES_DIR) if f.endswith('.nii.gz')])

    with torch.no_grad():
        for filename in image_files:
            try:
                img_path = os.path.join(IMAGES_DIR, filename)
                label_path = os.path.join(LABELS_DIR, filename)
                
                # تحميل ومعالجة الصورة
                img_data = nib.load(img_path).get_fdata()
                p1, p99 = np.percentile(img_data, [1, 99])
                img_data = np.clip(img_data, p1, p99)
                img_data = (img_data - np.mean(img_data)) / (np.std(img_data) + 1e-8)
                img_resized = resize(img_data, (128, 128, 128), order=1, preserve_range=True)

                img_input = np.stack([img_resized]*4, axis=0) 
                img_input = torch.from_numpy(img_input).float().unsqueeze(0).to(device)

                # التوقع (Inference)
                logits = model(img_input)
                probs = torch.sigmoid(logits)
                
                # استخدام العتبة المحسنة (0.45) على قناة الجلطة (Channel 2)
                pred_mask = (probs[0, 2] > THRESHOLD).float()
                
                # --- التطهير النهائي (Connected Components) ---
                # لحذف النقاط العشوائية الصغيرة والإبقاء على الجلطة الحقيقية فقط
                pred_np = pred_mask.cpu().numpy()
                labeled_array, num_features = ndimage.label(pred_np)
                if num_features > 0:
                    bincount = np.bincount(labeled_array.ravel())
                    bincount[0] = 0 
                    largest_label = bincount.argmax()
                    pred_np = (labeled_array == largest_label).astype(np.float32)
                pred_mask = torch.from_numpy(pred_np).to(device)

                # حساب الـ Dice مقارنة بالحقيقة
                if os.path.exists(label_path):
                    label_data = (nib.load(label_path).get_fdata() > 0).astype(np.float32)
                    label_resized = resize(label_data, (128, 128, 128), order=0, preserve_range=True)
                    label_tensor = torch.from_numpy(label_resized).to(device)

                    score = dice_coefficient_safe(pred_mask, label_tensor)
                    results.append({"File": filename, "Dice": round(score.item(), 4)})
                    
                    print(f"✅ {filename} | Dice: {score.item():.4f}")

                # تنظيف الذاكرة
                del img_input, logits, probs, pred_mask, pred_np, img_data, img_resized
                torch.cuda.empty_cache()
                gc.collect()

            except Exception as e:
                print(f"❌ Error in {filename}: {e}")

    # التقرير النهائي للبحث
    if results:
        df = pd.DataFrame(results)
        print("\n" + "="*45)
        print(f"🏆 ULTIMATE PERFORMANCE REPORT:")
        print(f"   Target Score:  0.80 - 0.85")
        print(f"   Actual Mean Dice: {df['Dice'].mean():.4f}")
        print(f"   Max Dice Achieved: {df['Dice'].max():.4f}")
        print("="*45)
        df.to_csv("ULTIMATE_Final_Results.csv", index=False)
        print("💾 Results saved to ULTIMATE_Final_Results.csv")

if __name__ == "__main__":
    main()