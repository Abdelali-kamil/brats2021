import os
import glob
import nibabel as nib

# Path to your folder
path = '/home/kamilabdelali/brats2021'

print("--- STARTING TEST ---")

# 1. Search for image files (.nii.gz)
files = glob.glob(os.path.join(path, '**', '*.nii.gz'), recursive=True)

if len(files) == 0:
    print("❌ ERROR: No images found.")
    print("   Please extract the .tar files first.")
else:
    print(f"✅ FOUND: {len(files)} images.")

    # 2. Try to open the first image
    try:
        first_file = files[0]
        img = nib.load(first_file)
        print(f"✅ SUCCESS: Can open file -> {os.path.basename(first_file)}")
        print(f"   Shape: {img.shape}")
        print("--- DATA IS WORKING ---")
    except:
        print("❌ ERROR: File exists but cannot be opened (Corrupt).")