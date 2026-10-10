"""Peer-review analysis (2026-10-10): counts undefined HD95 values (one empty mask) per model and region on the BraTS 2021
test set, and recomputes mean HD95 with undefined values set to 373.13 mm (diagonal of the 240x240x155 mm image), plus paired
comparisons of penalised HD95. Run from results/brats2021/per_case."""
import pandas as pd, numpy as np
from scipy.stats import wilcoxon
P=float(np.sqrt(240**2+240**2+155**2))   # image diagonal in mm (1 mm voxels)
F={"nnU-Net":"nnunet_test_per_case","Swin UNETR":"swin_unetr_test_per_case","SegResNet":"segresnet_test_per_case","UNETR":"unetr_test_per_case","3D U-Net":"unet3d_test_per_case",
"base":"per_case_final_ep253_test_tuned_recomputed","Run A":"per_case_runA_test_tuned_recomputed","Run C":"per_case_runC_test_tuned_recomputed","Run C rot":"per_case_runC_rot_test_tuned_recomputed",
"Ens 1":"per_case_ensemble_test_tuned_recomputed","Run D":"per_case_runD_test_tuned_recomputed","Run D rot":"per_case_runD_rot_test_tuned_recomputed","Run D2":"per_case_runD2_test_tuned_recomputed",
"Run D2 rot":"per_case_runD2_rot_test_tuned_recomputed","Ens 2":"per_case_ensemble2_test_tuned_recomputed","Ens 3":"per_case_ensemble3_test_tuned_recomputed"}
R=("ET","TC","WT"); d={k:pd.read_csv(v+".csv").set_index("Patient_ID") for k,v in F.items()}
def hd_cur(x): return x[[f"HD95_{r}" for r in R]].apply(pd.to_numeric,errors="coerce").mean(axis=1,skipna=True)
def hd_pen(x): return x[[f"HD95_{r}" for r in R]].apply(pd.to_numeric,errors="coerce").fillna(P).mean(axis=1)
rows=[]
for k,x in d.items():
    h=x[[f"HD95_{r}" for r in R]].apply(pd.to_numeric,errors="coerce")
    und={r:int(h[f"HD95_{r}"].isna().sum()) for r in R}
    rows.append([k,und["ET"],und["TC"],und["WT"],int(h.isna().all(axis=1).sum()),int(hd_cur(x).notna().sum()),hd_cur(x).mean(),hd_pen(x).mean()])
t=pd.DataFrame(rows,columns=["model","undef ET","undef TC","undef WT","cases all undef","cases in mean","HD95 reported","HD95 penalised"])
print(f"penalty = image diagonal {P:.2f} mm"); print(t.round(2).to_string(index=False))
print("\npaired, Ensemble 3 / Run C rot minus baseline, PENALISED HD95 (all 125 cases)")
for o in ("Ens 3","Run C rot","Run C"):
    for b in ("nnU-Net","Swin UNETR","SegResNet","UNETR","3D U-Net"):
        x=(hd_pen(d[o])-hd_pen(d[b])).values; bs=x[np.random.default_rng(0).integers(0,len(x),(10000,len(x)))].mean(1); lo,hi=np.percentile(bs,[2.5,97.5])
        print(f"  {o:9s} - {b:10s} {x.mean():+7.2f} [{lo:+7.2f}, {hi:+7.2f}] mm  Wilcoxon p={wilcoxon(x).pvalue:.3g}")
