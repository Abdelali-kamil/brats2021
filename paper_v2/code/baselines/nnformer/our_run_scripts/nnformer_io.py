"""Amendment 26 helpers.
  link_inputs <out_dir> <ids file> <case root> [flair_t1_t1ce_t2 names]  -> <case>_0000..0003 symlinks (authors' channel order)
  to_brats <pred_dir> <out_dir>            -> authors' labels (1 ED, 2 NCR, 3 ET) to BraTS 1 NCR / 2 ED / 4 ET, same geometry
  cv_fold <k>                              -> BraTS-Africa CV fold k as its own Task003_tumor (imagesTr = fold TRAIN 64 + VAL 12)"""
import os, sys, json, numpy as np, nibabel as nib
W = "/mnt/data1/kamil_research/experiments/nnformer"; A = "/mnt/data1/kamil_research/experiments/adapt_africa/cv"
AF = "/mnt/data1/kamil_research/brats_africa/brats21fmt"; CH = ("flair", "t1", "t1ce", "t2")
def link_inputs(out, ids, root):
    os.makedirs(out, exist_ok=True)
    for c in [l.strip() for l in open(ids) if l.strip()]:
        for i, m in enumerate(CH):
            d = f"{out}/{c}_{i:04d}.nii.gz"
            if not os.path.lexists(d): os.symlink(f"{root}/{c}/{c}_{m}.nii.gz", d)
def to_brats(src, dst):
    os.makedirs(dst, exist_ok=True)
    for f in sorted(os.listdir(src)):
        if not f.endswith(".nii.gz"): continue
        g = nib.load(f"{src}/{f}"); s = np.asarray(g.dataobj).astype(np.uint8); o = np.zeros_like(s)
        o[s == 1] = 2; o[s == 2] = 1; o[s == 3] = 4
        nib.save(nib.Nifti1Image(o, g.affine, g.header), f"{dst}/{f}")
def cv_fold(k):
    B = f"{W}/DATASET_cv/fold{k}"; R = f"{B}/nnFormer_raw/nnFormer_raw_data/Task003_tumor"
    tr = [l.strip() for l in open(f"{A}/fold{k}/train_ids.txt") if l.strip()]; va = [l.strip() for l in open(f"{A}/fold{k}/val_ids.txt") if l.strip()]
    for sub in ("imagesTr", "labelsTr"): os.makedirs(f"{R}/{sub}", exist_ok=True)
    for c in tr + va:
        for i, m in enumerate(CH):
            d = f"{R}/imagesTr/{c}_{i:04d}.nii.gz"
            if not os.path.lexists(d): os.symlink(f"{AF}/{c}/{c}_{m}.nii.gz", d)
        out = f"{R}/labelsTr/{c}.nii.gz"
        if not os.path.exists(out):
            g = nib.load(f"{AF}/{c}/{c}_seg.nii.gz"); s = np.asarray(g.dataobj).astype(np.uint8); o = np.zeros_like(s)
            o[s == 2] = 1; o[s == 1] = 2; o[s == 4] = 3; nib.save(nib.Nifti1Image(o, g.affine, g.header), out)
    meta = json.load(open(f"{W}/DATASET/nnFormer_raw/nnFormer_raw_data/Task003_tumor/dataset.json"))
    meta.update(description=f"BraTS-Africa CV fold {k} (Amendment 6/26)", numTraining=len(tr + va), numTest=0, test=[],
                training=[{"image": f"./imagesTr/{c}.nii.gz", "label": f"./labelsTr/{c}.nii.gz"} for c in tr + va])
    json.dump(meta, open(f"{R}/dataset.json", "w"), indent=1)
    json.dump({"train": tr, "val": va}, open(f"{B}/split.json", "w"))
    link_inputs(f"{B}/test_in", f"{A}/fold{k}/test_ids.txt", AF)
    print(f"fold {k}: {len(tr)} train, {len(va)} val")
if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "link_inputs": link_inputs(*sys.argv[2:5])
    elif cmd == "to_brats": to_brats(*sys.argv[2:4])
    elif cmd == "cv_fold": cv_fold(int(sys.argv[2]))
