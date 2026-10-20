#!/usr/bin/env python3
"""VAL-only cross-model error decomposition (2026-10-19, input to the next-experiment recommendation). TEST is not read.
Per-case VAL scores come from each run's saved VAL probabilities with its own VAL-selected post-processing (the scoring
function of runE_vs_runD/compare_val.py, which reproduces each run's recorded VAL Dice).
For each model: mean Dice; region shortfall (sum over cases of 1 - Dice, i.e. 126 x (1 - mean)) split into
  no-core cases (GT TC = 0: 01179, 01624), the low-ET/TC group (ET/TC < 0.25, n = 8; cutoff derived from VAL, EXPLORATORY),
  small cores (TC < 5,000 voxels, outside the previous groups), and the rest; plus TC Dice without the 2 no-core cases."""
import sys, pathlib
import numpy as np, pandas as pd
V = pathlib.Path("/mnt/data1/kamil_research/experiments/eval_v2"); OUT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(V / "runE_vs_runD"))
import compare_val as cv  # noqa: E402

cv.OUT = OUT   # write freshly scored per-case files here, not into runE_vs_runD/
MODELS = ["runC", "runC_rot", "ctrl_maxpool3d", "runD", "runD_rot", "runD2", "runD2_rot", "runE"] + \
         (["runE_rot"] if (V / "work_runE_rot" / "selection.txt").exists() else [])
diag = pd.read_csv(V / "proposal_20261017" / "diag_val_tc_errors.csv").set_index("Patient_ID")
cases = sorted(f.stem for f in (V / "work_runD" / "probs_val").glob("*.npz")); assert len(cases) == 126
nocore = [c for c in cases if c not in diag.index]; assert nocore == cv.NOCORE, nocore   # diag lists the 124 cases with a core
low = [c for c in diag.index if diag.loc[c, "et_frac_tc"] < 0.25]
small = [c for c in diag.index if diag.loc[c, "tc"] < 5000 and c not in low]
rest = [c for c in cases if c not in set(nocore) | set(low) | set(small)]
print(f"groups: no-core {len(nocore)} {nocore}; low-ET/TC {len(low)}; small core (<5000 vox, not low) {len(small)}; rest {len(rest)}")

D = {}
for m in MODELS:
    f = OUT / f"per_case_val_{m}.csv"
    if f.exists():
        D[m] = pd.read_csv(f).set_index("Patient_ID")
    else:
        s, d = cv.run(m, cases); D[m] = d
    assert abs(D[m].Dice_Mean.mean() - cv.selection(m)["val_dice"]) < 6e-5, m
E0 = pd.read_csv(V / "amendment28" / "per_case_val.csv"); E0 = E0[E0.candidate == "E0"].set_index("Patient_ID"); D["ensemble3 (E0)"] = E0

rows = []
for m, d in D.items():
    r = dict(model=m, mean=d.Dice_Mean.mean(), ET=d.Dice_ET.mean(), TC=d.Dice_TC.mean(), WT=d.Dice_WT.mean(),
             TC_without_nocore=d.loc[[c for c in cases if c not in nocore], "Dice_TC"].mean(),
             TC_low8=d.loc[low, "Dice_TC"].mean(), mean_low8=d.loc[low, "Dice_Mean"].mean(),
             nocore_TC_dice=" ".join(f"{d.loc[c, 'Dice_TC']:.0f}" for c in nocore))
    for reg in ("ET", "TC", "WT"):
        sf = 1 - d[f"Dice_{reg}"]; tot = sf.sum()
        r[f"{reg}_shortfall"] = tot
        for g, ids in (("nocore", nocore), ("low8", low), ("small", small), ("rest", rest)):
            r[f"{reg}_%{g}"] = 100 * sf.loc[ids].sum() / tot
    rows.append(r)
T = pd.DataFrame(rows).set_index("model")
T.to_csv(OUT / "val_decomposition.csv")
pd.set_option("display.width", 250)
print("\nVAL mean Dice per region; TC without the 2 no-core cases; low-ET/TC group (n=8)")
print(T[["mean", "ET", "TC", "WT", "TC_without_nocore", "TC_low8", "mean_low8", "nocore_TC_dice"]].round(4).to_string())
for reg in ("TC", "WT", "ET"):
    print(f"\n{reg} shortfall (sum of 1-Dice over 126 cases) and % from each group")
    print(T[[f"{reg}_shortfall", f"{reg}_%nocore", f"{reg}_%low8", f"{reg}_%small", f"{reg}_%rest"]].round(1).to_string())
# per-case TC in the low group across models (consistency of the weakness)
print("\nDice_TC on the 8 low-ET/TC VAL cases, per model")
print(pd.DataFrame({m: d.loc[low, "Dice_TC"] for m, d in D.items()}).round(3).to_string())
