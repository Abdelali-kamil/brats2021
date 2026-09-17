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
        """On-the-fly augmentation for the training split.

        `img` is [C, D, H, W] in [0, 1] (percentile min-max), `lbl` is
        [3, D, H, W] binary. Spatial transforms are applied identically to
        image and label; intensity transforms touch the image only, inside the
        brain (non-zero) region so the zero background is preserved.
        """
        # --- Spatial: independent flips on all three spatial axes (D, H, W) ---
        for ax in (1, 2, 3):
            if random.random() < 0.5:
                img = np.flip(img, axis=ax).copy()
                lbl = np.flip(lbl, axis=ax).copy()

        # --- Intensity: per-channel scale / shift / gamma inside the brain ---
        img = np.ascontiguousarray(img)
        for c in range(img.shape[0]):
            brain = img[c] > 0
            if not brain.any():
                continue
            vals = img[c][brain]
            if random.random() < 0.5:                       # multiplicative scale
                vals = vals * random.uniform(0.9, 1.1)
            if random.random() < 0.5:                       # additive shift
                vals = vals + random.uniform(-0.1, 0.1)
            if random.random() < 0.5:                       # random gamma
                vals = np.clip(vals, 0.0, None)
                mn, mx = float(vals.min()), float(vals.max())
                if mx > mn:
                    g = random.uniform(0.7, 1.5)
                    vals = ((vals - mn) / (mx - mn)) ** g * (mx - mn) + mn
            img[c][brain] = vals
        np.clip(img, 0.0, 1.0, out=img)
        return img, lbl

    @staticmethod
    def load_nii(path):
        return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))

    def __len__(self):
        return len(self.datas)

def _find_patient_dirs():
    """Locate BraTS2021_* case folders, trying the known data roots in order."""
    candidates = [
        pathlib.Path("data").resolve(),
        pathlib.Path("/home/kamilabdelali/brats2021/data").resolve(),
        pathlib.Path("/home/kamilabdelali/brats2021").resolve(),
    ]
    for root_path in candidates:
        dirs = sorted([d for d in root_path.glob("BraTS2021_*") if d.is_dir()])
        if dirs:
            print(f"📂 Found {len(dirs)} candidate patient folders under {root_path}")
            return dirs
    print("⚠️ No BraTS2021_* folders found in any known data root.")
    return []


def get_datasets(seed=42, debug=False, data_aug=True, train_frac=0.8):
    """Return (train_dataset, val_dataset).

    The split is the same seeded 80/20 partition used everywhere else in the
    project (see brats_gbm.splits.brats_split and scripts/evaluate_brats.py), so
    the 20% internal-validation set stays identical across training and
    evaluation. The training split gets random crops + augmentation; the
    validation split gets a deterministic centre crop and no augmentation.
    """
    import torch
    from torch.utils.data import random_split

    patients_dir = _find_patient_dirs()
    if debug:
        patients_dir = patients_dir[:10]

    n = len(patients_dir)
    train_n = int(train_frac * n)
    tr, va = random_split(
        list(range(n)), [train_n, n - train_n],
        generator=torch.Generator().manual_seed(seed),
    )
    train_dirs = [patients_dir[i] for i in tr.indices]
    val_dirs = [patients_dir[i] for i in va.indices]

    print(f"[INFO] BraTS split: train={len(train_dirs)}, val={len(val_dirs)} "
          f"(data_aug={data_aug})")

    train_ds = Brats(train_dirs, training=True, data_aug=data_aug, normalisation="minmax")
    val_ds = Brats(val_dirs, training=False, data_aug=False, normalisation="minmax")
    return train_ds, val_ds


if __name__ == "__main__":
    print("🚀 Running brats.py Test Sequence...")
    try:
        train_dataset, val_dataset = get_datasets()
        print(f"📊 Train Patients: {len(train_dataset)} | Val Patients: {len(val_dataset)}")
        if len(train_dataset) > 0:
            sample = train_dataset[0]
            print(f"✅ Image Shape: {sample['image'].shape}")
            print(f"✅ Label Shape: {sample['label'].shape}")
        else:
            print("⚠️ Warning: No valid patients found.")
    except Exception as e:
        print(f"❌ An error occurred: {e}")