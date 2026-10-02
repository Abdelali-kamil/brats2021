#!/usr/bin/env python3
"""Discover BraTS-Africa cases, check integrity/orientation, assign PROTOCOL_v2
Amendment-2 groups from TCIA's sheet, stage nnU-Net inputs. Writes cases.csv."""
import glob, os, re, sys, zipfile, collections, xml.etree.ElementTree as ET
import numpy as np, nibabel as nib, pandas as pd
ROOT = "/mnt/data1/kamil_research/brats_africa"; X = os.path.dirname(os.path.abspath(__file__))
REF = "/mnt/data1/kamil_research/data/brats2021/BraTS2021_00000/BraTS2021_00000_flair.nii.gz"

def sheet_ids():
    z = zipfile.ZipFile(f"{ROOT}/BraTS-Africa_TCIA_datainfo_v2.xlsx")
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    ss = ["".join(t.itertext()) for t in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", ns)]
    out = {}
    for k, grp in (("xl/worksheets/sheet1.xml", "glioma"), ("xl/worksheets/sheet2.xml", "other_neoplasm")):
        rows = list(ET.fromstring(z.read(k)).iter("{%s}row" % ns["m"]))[1:]
        for r in rows:
            c = r.find("m:c", ns); v = c.find("m:v", ns) if c is not None else None
            if v is None: continue
            sid = ss[int(v.text)] if c.get("t") == "s" else v.text
            if sid and sid.strip().isdigit(): out[int(sid)] = grp
    return out

segs = sorted(glob.glob(f"{ROOT}/pkg/**/*-seg.nii.gz", recursive=True))
print(len(segs), "seg files found")
groups = sheet_ids(); print("sheet ids:", collections.Counter(groups.values()))
ref = nib.load(REF); rows, problems = [], []
for s in segs:
    case = os.path.basename(s).replace("-seg.nii.gz", ""); d = os.path.dirname(s)
    m = re.search(r"BraTS-SSA-(\d{5})-(\d{3})", case)
    sid = int(m.group(1)) if m else None
    files = {k: f"{d}/{case}-{k}.nii.gz" for k in ("t1n", "t1c", "t2w", "t2f")}
    miss = [k for k, f in files.items() if not os.path.exists(f)]
    if miss: problems.append((case, "missing " + ",".join(miss))); continue
    g = nib.load(s); lab = np.unique(np.asarray(g.dataobj)).astype(int).tolist()
    ok_shape = g.shape == ref.shape; ok_aff = np.allclose(g.affine, ref.affine, atol=1e-3)
    if not set(lab) <= {0, 1, 2, 3}: problems.append((case, f"labels {lab}"))
    rows.append(dict(case=case, subject=sid, timepoint=m.group(2) if m else "", group=groups.get(sid, "unlisted"),
                     shape_ok=ok_shape, affine_eq_brats21=ok_aff, labels=" ".join(map(str, lab)),
                     wt_voxels=int((np.asarray(g.dataobj) > 0).sum()), seg=s, **files))
df = pd.DataFrame(rows); df.to_csv(f"{X}/cases.csv", index=False)
print(df.group.value_counts().to_string()); print("shape ok:", df.shape_ok.all(), "| affine == BraTS2021:", df.affine_eq_brats21.value_counts().to_dict())
print("timepoints per subject >1:", int((df.groupby("subject").size() > 1).sum()))
print("empty GT (WT=0):", int((df.wt_voxels == 0).sum()))
print("problems:", problems[:10], len(problems))
# nnU-Net staging (channel order of Dataset137: T1, T1ce, T2, Flair)
nn = f"{X}/nnunet_in"; os.makedirs(nn, exist_ok=True)
for r in rows:
    for i, k in enumerate(("t1n", "t1c", "t2w", "t2f")):
        dst = f"{nn}/{r['case']}_{i:04d}.nii.gz"
        if not os.path.lexists(dst): os.symlink(r[k], dst)
print("staged", len(rows), "cases for nnU-Net")
