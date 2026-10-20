#!/usr/bin/env python3
"""Amendment 29 success criteria, implemented BEFORE any Run F result exists (2026-10-19). VAL + TRAIN only; TEST never read.

  python criteria.py runF runD              [--trainfit-a DIR --trainfit-b DIR]
  python criteria.py runF_rot runD_rot      [--trainfit-a DIR --trainfit-b DIR]   (T2 uses the base-variant TRAIN-fit runs)

Per-case VAL scores: each run's saved VAL probabilities + its own VAL-selected post-processing (compare_val.py scoring, which
asserts that the recomputed VAL Dice equals selection.txt). Paired bootstrap: 10,000, fresh seed-0 generator, as Amendments 21/28.
"""
import sys, argparse, pathlib
import numpy as np, pandas as pd

E = pathlib.Path("/mnt/data1/kamil_research/experiments"); V = E / "eval_v2"
sys.path.insert(0, str(V / "runE_vs_runD"))
import compare_val as cv  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("a"); ap.add_argument("b")
ap.add_argument("--trainfit-a", default=None, help="dir with summary_*train_target65* / *train_other65* CSVs for run a")
ap.add_argument("--trainfit-b", default=None)
ap.add_argument("--out", default=str(pathlib.Path(__file__).resolve().parent / "results"))
args = ap.parse_args()
OUT = pathlib.Path(args.out); OUT.mkdir(exist_ok=True); cv.OUT = OUT

diag = pd.read_csv(V / "proposal_20261017" / "diag_val_tc_errors.csv").set_index("Patient_ID")   # 124 VAL cases with a core
cases = sorted(f.stem for f in cv.selection(args.b)["probs"].glob("*.npz")); assert len(cases) == 126
low8 = sorted(diag.index[diag.et_frac_tc < 0.25]); low18 = sorted(diag.index[diag.et_frac_tc < 0.5])
assert len(low8) == 8 and len(low18) == 18
_, A = cv.run(args.a, cases); _, B = cv.run(args.b, cases)
L = []
def P(t=""): print(t); L.append(t)

def boot(x):
    x = np.asarray(x, float); bs = x[np.random.default_rng(0).integers(0, len(x), (10000, len(x)))].mean(1)
    return x.mean(), *np.percentile(bs, [2.5, 97.5])

res = {}
P(f"Amendment 29 criteria: {args.a} vs {args.b} (VAL 126; TEST not read)")
# T1
d = A.loc[low8, "Dice_TC"] - B.loc[low8, "Dice_TC"]
res["T1"] = d.mean() >= 0.05 and int((d > 1e-9).sum()) >= 5
P(f"T1 targeted (8 VAL cases ET/TC<0.25, EXPLORATORY): TC {A.loc[low8,'Dice_TC'].mean():.4f} vs {B.loc[low8,'Dice_TC'].mean():.4f}, "
  f"diff {d.mean():+.4f}, improved in {int((d > 1e-9).sum())}/8  (need >= +0.05 and >= 5/8) -> {'PASS' if res['T1'] else 'FAIL'}")
# T2
if args.trainfit_a and args.trainfit_b:
    def tf(dirp, grp):
        f = list(pathlib.Path(dirp).glob(f"per_case_*_{grp}_recomputed.csv")); assert len(f) == 1, (dirp, grp, f)
        return pd.read_csv(f[0]).set_index("Patient_ID")
    ta, tb, oa, ob = tf(args.trainfit_a, "train_target65"), tf(args.trainfit_b, "train_target65"), \
                     tf(args.trainfit_a, "train_other65"), tf(args.trainfit_b, "train_other65")
    dt = ta.Dice_TC.mean() - tb.Dice_TC.mean(); do = oa.Dice_Mean.mean() - ob.Dice_Mean.mean()
    res["T2"] = dt >= 0.05 and do >= -0.01
    P(f"T2 mechanism (TRAIN, no TTA): target65 TC {ta.Dice_TC.mean():.4f} vs {tb.Dice_TC.mean():.4f} ({dt:+.4f}, need >= +0.05); "
      f"other65 mean {oa.Dice_Mean.mean():.4f} vs {ob.Dice_Mean.mean():.4f} ({do:+.4f}, need >= -0.01) -> {'PASS' if res['T2'] else 'FAIL'}")
else:
    res["T2"] = None; P("T2 mechanism: not evaluated (TRAIN-fit directories not given)")
# T3 (reported)
d3 = A.loc[low18, "Dice_TC"] - B.loc[low18, "Dice_TC"]
P(f"T3 reported (18 VAL cases ET/TC<0.5): TC diff {d3.mean():+.4f}, improved {int((d3 > 1e-9).sum())}/18")
# O1
m, lo, hi = boot(A.Dice_Mean - B.Dice_Mean)
res["O1"] = m >= 0 and lo > -0.005
P(f"O1 overall: mean Dice {A.Dice_Mean.mean():.4f} vs {B.Dice_Mean.mean():.4f}, diff {m:+.4f} [{lo:+.4f}, {hi:+.4f}] "
  f"(need point >= 0 and lower > -0.005) -> {'PASS' if res['O1'] else 'FAIL'}")
# O2
o2 = {r: A[f"Dice_{r}"].mean() - B[f"Dice_{r}"].mean() for r in ("ET", "TC", "WT")}
res["O2"] = all(v >= -0.005 for v in o2.values())
P("O2 per region: " + "  ".join(f"{r} {v:+.4f}" for r, v in o2.items()) + f"  (none < -0.005) -> {'PASS' if res['O2'] else 'FAIL'}")
# R1
drops = {c: max(B.loc[c, f"Dice_{r}"] - A.loc[c, f"Dice_{r}"] for r in ("ET", "TC", "WT")) for c in cases}
serious = sorted(c for c, v in drops.items() if v > 0.10)
flips = [(c, r) for c in cases for r in ("ET", "TC")
         if B.loc[c, f"pred_vox_{r}"] == 0 and B.loc[c, f"Dice_{r}"] == 1.0 and A.loc[c, f"Dice_{r}"] == 0.0]
res["R1"] = len(serious) <= 2 and not flips
P(f"R1 serious regressions: {len(serious)} case(s) losing > 0.10 in a region {serious} (need <= 2); empty-region 1->0 flips {flips} "
  f"-> {'PASS' if res['R1'] else 'FAIL'}")
# R2
r2 = {}
for r in ("ET", "TC", "WT"):
    h = f"HD95_{r}"; x = (A[h] - B[h]).dropna(); r2[r] = (x.mean(), len(x), int(A[h].isna().sum()), int(B[h].isna().sum()))
res["R2"] = all(v[0] <= 0.5 for v in r2.values())
P("R2 HD95 (defined pairs): " + "  ".join(f"{r} {v[0]:+.3f} mm (n={v[1]}, undefined {v[2]}/{v[3]})" for r, v in r2.items())
  + f"  (none > +0.5 mm) -> {'PASS' if res['R2'] else 'FAIL'}")
# F (reported)
perm = np.random.default_rng(0).permutation(len(cases))           # identical to compare_val.py / Amendment 28
fold = pd.Series({cases[j]: k % 5 for k, j in enumerate(perm)})
fd = list((A.Dice_Mean - B.Dice_Mean).groupby(fold).mean())
P("F folds (exploratory; Amendment-28 folds): " + " ".join(f"{v:+.4f}" for v in fd) + f"  ({sum(v >= 0 for v in fd)}/5 >= 0)")
# no-core cases reported
for c in cv.NOCORE:
    P(f"  no-core {c}: TC {A.loc[c,'Dice_TC']:.0f} vs {B.loc[c,'Dice_TC']:.0f} (pred TC vox {int(A.loc[c,'pred_vox_TC'])} vs {int(B.loc[c,'pred_vox_TC'])})")
need = ["T1", "T2", "O1", "O2", "R1", "R2"]
if all(res[k] for k in need):
    verdict = "(i) all criteria hold for this variant"
elif res["T1"] and res["T2"] and not (res["O1"] and res["O2"]):   # Amendment 29 addendum (10-20): T1 AND T2
    verdict = "(ii) subgroup effect without overall non-inferiority (not adopted)"
else:
    verdict = "criteria not met for this variant (see lines above); T2 missing" if res["T2"] is None else "criteria not met for this variant"
P(f"VERDICT for {args.a} vs {args.b}: {verdict}. Outcome (i) requires BOTH the base and the rotation variant to pass.")
(OUT / f"criteria_{args.a}_vs_{args.b}.txt").write_text("\n".join(L) + "\n")
