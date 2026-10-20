#!/usr/bin/env python3
"""VAL-only comparison of two of our evaluated runs under the same evaluation variant (2026-10-18, user request; Run F
decision input). Each run is scored from its saved full-VAL probabilities (eval_run.sh / eval_run_rot2.sh, probs_val) with
its own VAL-selected post-processing (work_<tag>/selection.txt), exactly as eval_v2 selected it. TEST is never read.
Reports: overall and per-region Dice / HD95 with paired bootstrap 95 % CI (10,000, fresh seed-0 generator per comparison)
+ two-sided Wilcoxon; the 8 targeted VAL cases (ET/TC < 0.25; cutoff derived from VAL, EXPLORATORY); the 2 no-core VAL
cases; case-level changes > 0.10 Dice in any region; per-fold differences on the Amendment-28 VAL folds (EXPLORATORY).

  python compare_val.py runE runD          # base variants
  python compare_val.py runE_rot runD_rot  # rotation-TTA variants
"""
import json, pathlib, sys
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import wilcoxon

E = pathlib.Path("/mnt/data1/kamil_research/experiments"); V = E / "eval_v2"; CODE = V / "code"
sys.path.insert(0, str(CODE)); sys.path.insert(0, str(CODE / "baselines" / "common"))
from brats_gbm.eval import postprocess as pp           # noqa: E402
from brats_gbm.eval.metrics import score_case          # noqa: E402
from sweep_postproc_hd95 import load_gt                # noqa: E402
OUT = pathlib.Path(__file__).resolve().parent; REG = ("ET", "TC", "WT"); ORIG = dict(pp.MIN_VOXELS)
NOCORE = ["BraTS2021_01179", "BraTS2021_01624"]


def selection(tag):
    f = (V / f"work_{tag}" / "selection.txt").read_text().split()
    return dict(tag=f[0], val_dice=float(f[1]), wt=f[2], mult=int(f[3]), et_min=int(f[5]), orient=f[6],
                probs=V / f"work_{tag}" / "probs_val")


def post(prob, s):   # identical to ensemble*_search.py post()
    pp.MIN_VOXELS = {k: v * s["mult"] for k, v in ORIG.items()}
    out = pp.postprocess(prob, 0.5, 0.5, 0.5, wt_policy=s["wt"], et_policy="none" if s["et_min"] == 0 else "min_volume",
                         et_min_volume=max(s["et_min"], 1))
    pp.MIN_VOXELS = dict(ORIG)
    return out


def score(args):
    s, case = args
    pred = post(np.load(s["probs"] / f"{case}.npz")["prob"].astype(np.float32), s)
    r = score_case(pred, load_gt(case))
    r.update(Patient_ID=case, **{f"pred_vox_{g}": int(pred[i].sum()) for i, g in enumerate(REG)})
    return r


def run(tag, cases):
    s = selection(tag); assert len(list(s["probs"].glob("*.npz"))) == 126, tag
    with ProcessPoolExecutor(4) as ex:
        d = pd.DataFrame(list(ex.map(score, [(s, c) for c in cases]))).set_index("Patient_ID")
    d.to_csv(OUT / f"per_case_val_{tag}.csv")
    return s, d


def paired(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    bs = x[np.random.default_rng(0).integers(0, len(x), (10000, len(x)))].mean(1); lo, hi = np.percentile(bs, [2.5, 97.5])
    return x.mean(), lo, hi, (wilcoxon(x).pvalue if np.any(x != 0) else 1.0), len(x)


def main(a, b):
    cases = sorted(f.stem for f in selection(b)["probs"].glob("*.npz")); assert len(cases) == 126
    sa, A = run(a, cases); sb, B = run(b, cases)
    L = []; P = lambda t="": (print(t), L.append(t))
    P(f"VAL-only comparison {a} vs {b} (126 cases, each with its own VAL-selected post-processing; TEST not read)")
    for s, d in ((sa, A), (sb, B)):
        P(f"  {s['tag']}: selection {s['wt']} floors x{s['mult']} ET-min {s['et_min']} axis {s['orient']} | recorded VAL "
          f"{s['val_dice']:.4f}, recomputed {d.Dice_Mean.mean():.4f}")
        assert abs(d.Dice_Mean.mean() - s["val_dice"]) < 6e-5, "recomputed VAL does not match selection.txt"
    P(f"\n{'':10s} {a:>10s} {b:>10s}   diff ({a} - {b}) [95% CI]   Wilcoxon p   {a} better/worse")
    for m in ["Dice_Mean"] + [f"Dice_{r}" for r in REG]:
        mu, lo, hi, p, n = paired(A[m] - B[m]); dd = A[m] - B[m]
        P(f"{m:10s} {A[m].mean():10.4f} {B[m].mean():10.4f}   {mu:+.4f} [{lo:+.4f}, {hi:+.4f}]   {p:.3g}   "
          f"{int((dd > 1e-9).sum())}/{int((dd < -1e-9).sum())}")
    for r in REG:
        h = f"HD95_{r}"; mu, lo, hi, p, n = paired(A[h] - B[h])
        P(f"{h:10s} {A[h].mean():10.2f} {B[h].mean():10.2f}   {mu:+.3f} [{lo:+.3f}, {hi:+.3f}] mm (n={n} defined pairs)  p={p:.3g}"
          f"   undefined {int(A[h].isna().sum())} / {int(B[h].isna().sum())}")
    diag = pd.read_csv(V / "proposal_20261017" / "diag_val_tc_errors.csv").set_index("Patient_ID")
    tgt = sorted(diag.index[diag.et_frac_tc < 0.25])
    assert len(tgt) == 8, len(tgt)
    P(f"\nTargeted VAL cases (ET/TC < 0.25, n=8; EXPLORATORY, cutoff derived from VAL): Dice_TC {a} vs {b}")
    for c in tgt:
        P(f"  {c} ET/TC {diag.loc[c, 'et_frac_tc']:.2f}  TC {A.loc[c, 'Dice_TC']:.3f} vs {B.loc[c, 'Dice_TC']:.3f} "
          f"({A.loc[c, 'Dice_TC'] - B.loc[c, 'Dice_TC']:+.3f})   mean {A.loc[c, 'Dice_Mean']:.3f} vs {B.loc[c, 'Dice_Mean']:.3f}")
    dt = A.loc[tgt, "Dice_TC"] - B.loc[tgt, "Dice_TC"]
    P(f"  mean Dice_TC {A.loc[tgt, 'Dice_TC'].mean():.3f} vs {B.loc[tgt, 'Dice_TC'].mean():.3f} (diff {dt.mean():+.3f}); "
      f"{a} better in {int((dt > 1e-9).sum())}/8, worse in {int((dt < -1e-9).sum())}/8")
    P("\nNo-core VAL cases (GT ET = TC = 0; Dice 1 / HD95 0 when both empty, Dice 0 / HD95 undefined when only one empty):")
    for c in NOCORE:
        P(f"  {c}: ET {A.loc[c, 'Dice_ET']:.3f} vs {B.loc[c, 'Dice_ET']:.3f}  TC {A.loc[c, 'Dice_TC']:.3f} vs {B.loc[c, 'Dice_TC']:.3f}  "
          f"WT {A.loc[c, 'Dice_WT']:.3f} vs {B.loc[c, 'Dice_WT']:.3f} | predicted voxels ET/TC/WT {a} "
          f"{A.loc[c, 'pred_vox_ET']}/{A.loc[c, 'pred_vox_TC']}/{A.loc[c, 'pred_vox_WT']} vs {b} "
          f"{B.loc[c, 'pred_vox_ET']}/{B.loc[c, 'pred_vox_TC']}/{B.loc[c, 'pred_vox_WT']} (GT WT {int(load_gt(c)[2].sum())})")
    D = A[[f"Dice_{r}" for r in REG]] - B[[f"Dice_{r}" for r in REG]]
    P(f"\nCase-level changes > 0.10 Dice in any region ({a} - {b}):")
    worse = D[(D < -0.10).any(axis=1)]; better = D[(D > 0.10).any(axis=1)]
    P(f"  regressions ({len(worse)}): " + ("; ".join(f"{c} " + " ".join(f"{r} {D.loc[c, f'Dice_{r}']:+.3f}" for r in REG) for c in worse.index) or "none"))
    P(f"  improvements ({len(better)}): " + ("; ".join(f"{c} " + " ".join(f"{r} {D.loc[c, f'Dice_{r}']:+.3f}" for r in REG) for c in better.index) or "none"))
    perm = np.random.default_rng(0).permutation(len(cases)); fold = pd.Series({cases[j]: k % 5 for k, j in enumerate(perm)})
    fg = (A.Dice_Mean - B.Dice_Mean).groupby(fold).mean()
    P("\nPer-fold mean-Dice difference on the Amendment-28 VAL folds (EXPLORATORY): " + " ".join(f"{v:+.4f}" for v in fg)
      + f"  ({int((fg >= 0).sum())}/5 >= 0)")
    (OUT / f"report_{a}_vs_{b}.txt").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
