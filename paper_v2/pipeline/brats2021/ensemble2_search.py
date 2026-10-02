#!/usr/bin/env python3
"""PROTOCOL_v2 Amendment 15: ensemble2 search (same procedure as Amendments 5/7/10/11) (Run D cancelled by Amendment 11; each member uses
its own VAL-chosen inference mode, i.e. whatever its probs_val/probs_test hold).

Every combination of >= 2 available candidates among {ep253, runA, runC, runC_rot} (Amendments 11/12) is
probability-averaged per case and evaluated on VAL (126) over the Amendment-5 grid
(WT policy x floors x ET min-volume). Winner = highest VAL mean Dice; ties within
0.001 -> fewer members, then lower HD95. The winner is scored on TEST (125) ONCE.

  python ensemble_search.py            # search on VAL, then test the winner once
"""
import itertools, json, pathlib, sys
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor

E = pathlib.Path("/mnt/data1/kamil_research/experiments")
CODE = E / "eval_v2" / "code"
sys.path.insert(0, str(CODE)); sys.path.insert(0, str(CODE / "baselines" / "common"))
from brats_gbm.eval import postprocess as pp           # noqa: E402
from brats_gbm.eval.metrics import score_case          # noqa: E402
from brats_gbm.eval.stats import summarise_segmentation, print_summary  # noqa: E402
from sweep_postproc_hd95 import load_gt                # noqa: E402

RES = pathlib.Path("/home/kamilabdelali/brats2021/results/brats")
OUT = E / "eval_v2" / "ensemble2"; OUT.mkdir(exist_ok=True)
SRC = {"ep253": (E / "eval_ep253/probs_val", E / "eval_ep253/probs_test")}
for t in ("runA", "runC", "runC_rot", "runD", "runD_rot"):   # Amendment 15
    SRC[t] = (E / f"eval_v2/work_{t}/probs_val", E / f"eval_v2/work_{t}/probs_test")
avail = [k for k, (v, t) in SRC.items() if len(list(v.glob("*.npz"))) == 126 and len(list(t.glob("*.npz"))) == 125]
SAME = lambda a, b: a.replace("_rot", "") == b.replace("_rot", "")   # a model and its rotation variant share weights
COMBOS = [c for r in range(2, len(avail) + 1) for c in itertools.combinations(avail, r)
          if not any(SAME(a, b) for a, b in itertools.combinations(c, 2))]
GRID = list(itertools.product(("components", "largest"), (1, 5, 20), (0, 100, 200, 500, 1000)))
ORIG = dict(pp.MIN_VOXELS)


def post(prob, wt, mult, et):
    pp.MIN_VOXELS = {k: v * mult for k, v in ORIG.items()}
    out = pp.postprocess(prob, 0.5, 0.5, 0.5, wt_policy=wt, et_policy="none" if et == 0 else "min_volume",
                         et_min_volume=max(et, 1))
    pp.MIN_VOXELS = dict(ORIG)
    return out


def val_case(case):
    probs = {k: np.load(SRC[k][0] / f"{case}.npz")["prob"].astype(np.float32) for k in avail}
    gt = load_gt(case); res = []
    for combo in COMBOS:
        p = sum(probs[k] for k in combo) / len(combo)
        res.append([score_case(post(p, *g), gt) for g in GRID])
    return res


def main():
    guard = RES / "summary_ensemble2_test_tuned_recomputed.csv"
    if guard.exists():
        sys.exit("ensemble TEST already scored; refusing a second scoring")
    print("candidates:", avail, "| combinations:", len(COMBOS), "| settings each:", len(GRID), flush=True)
    cases = sorted(f.stem for f in SRC[avail[0]][0].glob("*.npz"))
    with ProcessPoolExecutor(16) as ex:
        per_case = list(ex.map(val_case, cases))
    rows = []
    for ci, combo in enumerate(COMBOS):
        for gi, (wt, mult, et) in enumerate(GRID):
            d = pd.DataFrame([pc[ci][gi] for pc in per_case])
            rows.append(dict(combo="+".join(combo), n_members=len(combo), wt_policy=wt, floor_mult=mult, et_min_volume=et,
                             Dice_ET=d.Dice_ET.mean(), Dice_TC=d.Dice_TC.mean(), Dice_WT=d.Dice_WT.mean(),
                             Dice_Mean=d.Dice_Mean.mean(), HD95_mean=d[["HD95_ET", "HD95_TC", "HD95_WT"]].mean(skipna=True).mean()))
    df = pd.DataFrame(rows); df.to_csv(OUT / "ensemble_val_search.csv", index=False)
    best_by_combo = df.loc[df.groupby("combo").Dice_Mean.idxmax()].sort_values("Dice_Mean", ascending=False)
    print(best_by_combo[["combo", "Dice_Mean", "wt_policy", "floor_mult", "et_min_volume"]].round(4).to_string(index=False))
    top = df.Dice_Mean.max()
    w = df[df.Dice_Mean >= top - 0.001].sort_values(["n_members", "HD95_mean"]).iloc[0]
    sel = dict(combo=w.combo, wt_policy=w.wt_policy, floor_mult=int(w.floor_mult), et_min_volume=int(w.et_min_volume),
               val_dice=float(w.Dice_Mean))
    (OUT / "selection.json").write_text(json.dumps(sel, indent=1))
    print("VAL-selected ensemble:", sel, flush=True)
    members = sel["combo"].split("+"); rows = []
    for case in sorted(f.stem for f in SRC[members[0]][1].glob("*.npz")):
        p = sum(np.load(SRC[k][1] / f"{case}.npz")["prob"].astype(np.float32) for k in members) / len(members)
        s = score_case(post(p, sel["wt_policy"], sel["floor_mult"], sel["et_min_volume"]), load_gt(case))
        s["Patient_ID"] = case; rows.append(s)
    d = pd.DataFrame(rows)
    d.to_csv(RES / "per_case_ensemble2_test_tuned_recomputed.csv", index=False)
    s = summarise_segmentation(d, label="ensemble2_test_tuned"); s.to_csv(guard, index=False)
    print_summary(s, f"ENSEMBLE2 {sel['combo']} — BraTS2021 TEST (n={len(d)}), scored once")


if __name__ == "__main__":
    main()
