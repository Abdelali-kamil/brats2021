import pathlib
import SimpleITK as sitk
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import random 

try:
    from image import pad_or_crop_image, irm_min_max_preprocess, zscore_normalise
except ImportError:
    print("⚠️ Warning: image.py helpers not found. Ensure they are in the same directory.")

class Brats(Dataset):
    def __init__(self, patients_dir, training=True, data_aug=False, no_seg=False, normalisation="minmax"):
        super(Brats, self).__init__()
        self.normalisation = normalisation
        self.data_aug = data_aug  
        self.training = training
        self.datas = []
        self.no_seg = no_seg
        
        self.patterns = ["_t1", "_t1ce", "_t2", "_flair"]
        if not no_seg:
            self.patterns += ["_seg"]
            
        print(f"🔍 Scan initiated: Checking {len(patients_dir)} folders...")
        
        valid_count = 0
        for patient_dir in patients_dir:
            patient_id = patient_dir.name
            is_valid = True
            current_paths = []
            for pattern in self.patterns:
                found = list(patient_dir.glob(f"*{pattern}.nii*"))
                if len(found) > 0:
                    current_paths.append(found[0])
                else:
                    is_valid = False
                    break
            
            if not is_valid:
                continue 

            patient_dict = {
                "id": patient_id,
                "t1": current_paths[0],
                "t1ce": current_paths[1],
                "t2": current_paths[2],
                "flair": current_paths[3],
                "seg": current_paths[4] if not no_seg else None
            }
            self.datas.append(patient_dict)
            valid_count += 1

        print(f"✅ Success: {valid_count} patients loaded from {len(patients_dir)} folders.")

    def __getitem__(self, idx):
        _p = self.datas[idx]
        
        img_data = {k: self.load_nii(_p[k]) for k in ["t1", "t1ce", "t2", "flair"]}
        
        if self.normalisation == "minmax":
            img_data = {k: irm_min_max_preprocess(v) for k, v in img_data.items()}
        else:
            img_data = {k: zscore_normalise(v) for k, v in img_data.items()}
            
        image = np.stack([img_data[k] for k in ["t1", "t1ce", "t2", "flair"]])
        
        if not self.no_seg and _p["seg"] is not None:
            mask = self.load_nii(_p["seg"])
            et = (mask == 4)
            tc = np.logical_or(mask == 4, mask == 1)
            wt = np.logical_or(tc, mask == 2)
            label = np.stack([et, tc, wt])
        else:
            label = np.zeros((3, *image.shape[1:]))
            
        # ✅ KEY FIX: pass training=self.training for deterministic center crop during eval
        image, label = pad_or_crop_image(image, label, target_size=(128, 128, 128), training=self.training)
            
        if self.training and self.data_aug:
            image, label = self.augment(image, label)
            
        return {
            "patient_id": _p["id"],
            "image": torch.from_numpy(image.astype("float32")),
            "label": torch.from_numpy(label.astype("float32"))
        }

    def augment(self, img, lbl):
        if random.random() > 0.5:
            img, lbl = np.flip(img, axis=2).copy(), np.flip(lbl, axis=2).copy()
        if random.random() > 0.5:
            img, lbl = np.flip(img, axis=3).copy(), np.flip(lbl, axis=3).copy()
        return img, lbl

    @staticmethod
    def load_nii(path):
        return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))

    def __len__(self):
        return len(self.datas)

# ✅ KEY FIX: default changed from "train" to "test"
def get_datasets(seed=42, debug=False, on="test"):
    root_path = pathlib.Path("/home/kamilabdelali/brats2021/data").resolve()
    patients_dir = sorted([d for d in root_path.glob("BraTS2021_*") if d.is_dir()])
    
    if debug:
        patients_dir = patients_dir[:10]
        
    if len(patients_dir) == 0:
        print("⚠️ No folders in /data, checking main directory...")
        root_path = pathlib.Path("/home/kamilabdelali/brats2021").resolve()
        patients_dir = sorted([d for d in root_path.glob("BraTS2021_*") if d.is_dir()])

    print(f"📂 Found {len(patients_dir)} candidate patient folders.")
    
    return Brats(patients_dir, training=(on == "train"), normalisation="minmax", data_aug=False)


if __name__ == "__main__":
    print("🚀 Running brats.py Test Sequence...")
    try:
        dataset = get_datasets()
        print(f"📊 Total Patients: {len(dataset)}")
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"✅ Image Shape: {sample['image'].shape}")
            print(f"✅ Label Shape: {sample['label'].shape}")
        else:
            print("⚠️ Warning: No valid patients found.")
    except Exception as e:
        print(f"❌ An error occurred: {e}")