"""Amendment 26: our BraTS 2021 split -> nnFormer/nnU-Net-v1 raw layout (Task003_tumor, the authors' tumour task name).
imagesTr = TRAIN (1000) + VAL (126) (VAL used only by the trainer's own model selection, via NNFORMER_FIXED_SPLIT);
imagesTs = TEST (125), never preprocessed or seen in training. Channels in the authors' order: 0 FLAIR, 1 T1w, 2 T1gd, 3 T2w
(symlinks to the original files). Labels in the authors' convention: 1 edema, 2 non-enhancing/necrotic core, 3 enhancing
(BraTS 2021: 2 ED -> 1, 1 NCR -> 2, 4 ET -> 3)."""
import os, json, numpy as np, nibabel as nib
D = "/mnt/data1/kamil_research/data/brats2021"; X = "/mnt/data1/kamil_research/experiments/nnformer"
R = f"{X}/DATASET/nnFormer_raw/nnFormer_raw_data/Task003_tumor"
sp = json.load(open(f"{X}/split_fixed.json")); test = [l.strip() for l in open("/mnt/data1/kamil_research/experiments/transbts/test.txt")]
for sub in ("imagesTr", "labelsTr", "imagesTs"): os.makedirs(f"{R}/{sub}", exist_ok=True)
CH = ("flair", "t1", "t1ce", "t2")
def link(c, sub):
    for i, m in enumerate(CH):
        dst = f"{R}/{sub}/{c}_{i:04d}.nii.gz"
        if not os.path.lexists(dst): os.symlink(f"{D}/{c}/{c}_{m}.nii.gz", dst)
tr = sp["train"] + sp["val"]
for n, c in enumerate(tr, 1):
    link(c, "imagesTr"); out = f"{R}/labelsTr/{c}.nii.gz"
    if not os.path.exists(out):
        g = nib.load(f"{D}/{c}/{c}_seg.nii.gz"); s = np.asarray(g.dataobj).astype(np.uint8); o = np.zeros_like(s)
        o[s == 2] = 1; o[s == 1] = 2; o[s == 4] = 3
        assert set(np.unique(s)) <= {0, 1, 2, 4}, (c, np.unique(s))
        nib.save(nib.Nifti1Image(o, g.affine, g.header), out)
    if n % 200 == 0: print(n, flush=True)
for c in test: link(c, "imagesTs")
assert not set(test) & set(tr)
json.dump({"name": "Brain_Tumor", "description": "BraTS 2021, our fixed split (Amendment 26)", "tensorImageSize": "4D",
           "modality": {"0": "FLAIR", "1": "T1w", "2": "t1gd", "3": "T2w"},
           "labels": {"0": "background", "1": "edema", "2": "non-enhancing tumor", "3": "enhancing tumour"},
           "numTraining": len(tr), "numTest": len(test),
           "training": [{"image": f"./imagesTr/{c}.nii.gz", "label": f"./labelsTr/{c}.nii.gz"} for c in tr],
           "test": [f"./imagesTs/{c}.nii.gz" for c in test]}, open(f"{R}/dataset.json", "w"), indent=1)
print("done", len(tr), "train+val,", len(test), "test")
