"""Peer-review checks (2026-10-10), run on existing per-case results only:
Run A - base without the two collapsed cases; Run D2 vs Run D zero-shot; Run C - base with penalised HD95."""
import pandas as pd, numpy as np
from scipy.stats import wilcoxon
B = "/home/kamilabdelali/release/brats2021/paper_v2/results"
md = lambda d: d[["Dice_ET", "Dice_TC", "Dice_WT"]].mean(axis=1)
def paired(name, a, b):
    ids = a.index.intersection(b.index); x = (a.loc[ids] - b.loc[ids]).dropna().values
    bs = x[np.random.default_rng(0).integers(0, len(x), (10000, len(x)))].mean(1); lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"{name:55s} n={len(x)} {x.mean():+.4f} [{lo:+.4f}, {hi:+.4f}] Wilcoxon p={wilcoxon(x).pvalue:.3g}")
L = lambda p: pd.read_csv(p).set_index("Patient_ID")
b = L(f"{B}/brats2021/per_case/per_case_final_ep253_test_tuned_recomputed.csv"); a = L(f"{B}/brats2021/per_case/per_case_runA_test_tuned_recomputed.csv")
paired("Run A - base, all 125", md(a), md(b))
keep = [c for c in a.index if c not in ("BraTS2021_00348", "BraTS2021_00149")]
paired("Run A - base, without the two collapsed cases", md(a).loc[keep], md(b).loc[keep])
z = lambda k: (lambda d: d[d.group == "glioma"])(L(f"{B}/brats_africa_zeroshot/per_case_{k}.csv"))
paired("Africa zero-shot: Run D2 rot - Run D rot (95 gliomas)", md(z("ours_runD2_rot")), md(z("ours_runD_rot")))
P = float(np.sqrt(240**2 + 240**2 + 155**2)); hp = lambda d: d[["HD95_ET", "HD95_TC", "HD95_WT"]].apply(pd.to_numeric, errors="coerce").fillna(P).mean(axis=1)
c = L(f"{B}/brats2021/per_case/per_case_runC_test_tuned_recomputed.csv")
paired("Run C - base, HD95 penalised (mm)", hp(c), hp(b))
