import pathlib
import SimpleITK as sitk
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import random 

try:
    # Package-qualified, matching how every other module imports these helpers.
    from brats_gbm.data.image import (
        pad_or_crop_image, irm_min_max_preprocess, zscore_normalise)
except ImportError:
    # Fallback for running this file directly, with brats_gbm/data/ on sys.path.
    # Left to raise if it also fails: swallowing the error here previously turned
    # a missing import into a NameError inside a DataLoader worker at train time.
    from image import (
        pad_or_crop_image, irm_min_max_preprocess, zscore_normalise)

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
        """Spatial and intensity augmentation for training patches.

        Spatial transforms are axis-aligned — flips on all three axes and
        90-degree rotations in plane — so they need no interpolation, cannot
        blur a label boundary, and cost nothing measurable in the loader. The
        crop is cubic (128^3), so a rotation in any plane keeps the shape.
        Image and label are transformed together; a transform applied to one
        and not the other would silently destroy the correspondence, which
        `tests/test_augmentation.py` asserts against.

        Intensity transforms act on the image only, per channel, because the
        four modalities are normalised independently.
        """
        # Spatial — image and label together.
        for axis in (1, 2, 3):  # z, y, x; axis 0 is channel/region
            if random.random() < 0.5:
                img = np.flip(img, axis=axis)
                lbl = np.flip(lbl, axis=axis)
        k = random.randint(0, 3)
        if k:
            img = np.rot90(img, k=k, axes=(2, 3))
            lbl = np.rot90(lbl, k=k, axes=(2, 3))

        # flip/rot90 return views over the original buffer; the intensity
        # transforms below write in place, so a contiguous copy is required.
        img = np.ascontiguousarray(img, dtype=np.float32)
        lbl = np.ascontiguousarray(lbl)

        # Intensity — image only, independently per modality.
        for c in range(img.shape[0]):
            if random.random() < 0.5:
                img[c] = img[c] * random.uniform(0.9, 1.1) + random.uniform(-0.1, 0.1)
            if random.random() < 0.3:
                lo, hi = float(img[c].min()), float(img[c].max())
                if hi > lo:
                    scaled = (img[c] - lo) / (hi - lo)
                    img[c] = scaled ** random.uniform(0.7, 1.5) * (hi - lo) + lo
            if random.random() < 0.2:
                img[c] = img[c] + np.random.normal(0.0, 0.02, img[c].shape)
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


def get_brats_train_val(data_root=None, seed=42, data_aug=True):
    """Training and validation datasets under the three-way split.

    `get_datasets` returns a single dataset built with ``training=False``, which
    `scripts/train_brats.py` then partitioned with `random_split`. Both halves
    therefore inherited ``training=False``, and two consequences followed that
    were never intended:

      * `pad_or_crop_image` took a *deterministic centre crop*, so every epoch
        saw the identical 128^3 window of each volume. On 155x240x240 data that
        is 28% of the axial plane; the periphery was never trained on, while
        sliding-window inference is applied to all of it.
      * ``data_aug=False`` was hardcoded, so `Brats.augment` never ran.

    This function builds the two partitions separately so each gets the flags it
    should have: random cropping and augmentation for training, and the
    deterministic centre crop with no augmentation for validation, so model
    selection is not measured through a moving target.

    Returns ``(train_dataset, val_dataset, test_ids)``. The test ids are
    returned rather than a dataset: nothing in training may touch them.
    """
    from brats_gbm.splits import assert_disjoint, brats_split_3way

    root = pathlib.Path(data_root or "/home/kamilabdelali/brats2021/data").resolve()
    train_ids, val_ids, test_ids = brats_split_3way(str(root), seed=seed)
    assert_disjoint(train=train_ids, val=val_ids, test=test_ids)

    def dirs(ids):
        return sorted((root / i for i in ids), key=lambda p: p.name)

    train_dataset = Brats(
        dirs(train_ids), training=True, data_aug=data_aug, normalisation="minmax")
    val_dataset = Brats(
        dirs(val_ids), training=False, data_aug=False, normalisation="minmax")
    return train_dataset, val_dataset, sorted(test_ids)


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