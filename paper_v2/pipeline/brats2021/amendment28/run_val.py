#!/usr/bin/env python3
"""PROTOCOL_v2 Amendment 28, stage 1: score every pre-registered candidate (config.json) on the 126 VAL cases from the saved
VAL probability maps of the Ensemble 3 members, plus the error decomposition of E0 (= Ensemble 3). VAL only: no TEST
probability map or TEST label is read. Output: per_case_val.csv (candidate x case), errors_E0_val.csv, cc_removed_val.csv.

  python run_val.py [--workers 4] [--only E0,WTrel0.1,...]   # finished cases cached in cache/<case>.pkl (resume-safe)
"""
import argparse, json, os, pathlib, pickle, sys
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from scipy import ndimage

E = pathlib.Path("/mnt/data1/kamil_research/experiments")
CODE = E / "eval_v2" / "code"
sys.path.insert(0, str(CODE)); sys.path.insert(0, str(CODE / "baselines" / "common"))
from brats_gbm.eval import postprocess as pp           # noqa: E402
from brats_gbm.eval.metrics import score_case          # noqa: E402
from sweep_postproc_hd95 import load_gt                # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent
CFG = json.loads((OUT / "config.json").read_text())
VALDIR = {"ep253": E / "eval_ep253/probs_val", "runC_rot": E / "eval_v2/work_runC_rot/probs_val",
          "runD2_rot": E / "eval_v2/work_runD2_rot/probs_val"}
REG = ("ET", "TC", "WT")
assert pp.MIN_VOXELS == {"ET": 5, "TC": 20, "WT": 50}   # floors x1 = E0


def post_e0(p):
    e = CFG["E0"]
    return pp.postprocess(p, *e["thresholds"], wt_policy=e["wt_policy"], et_policy="min_volume", et_min_volume=e["et_min_volume"])


def weighted(probs, w):
    return sum(wi * probs[m] for wi, m in zip(w, CFG["members"]) if wi) / float(sum(w))


def cc_rule(pred, prob, kind, v):
    """Apply one component rule to a post-processed E0 prediction. Returns (new pred, removed mask, region index)."""
    pred = pred.copy()
    i = 1 if kind == "TCrel" else 2
    lab, n = ndimage.label(pred[i])
    if n <= 1:
        return pred, np.zeros_like(pred[i]), i
    sizes = ndimage.sum(pred[i], lab, range(1, n + 1)); big = int(np.argmax(sizes)) + 1
    if kind in ("WTrel", "TCrel"):
        keep = np.flatnonzero(sizes >= v * sizes.max()) + 1
    else:  # WTconf: max ensemble WT probability inside the component, never the largest component
        mx = ndimage.maximum(prob[2], lab, range(1, n + 1))
        keep = np.union1d(np.flatnonzero(mx >= v) + 1, [big])
    new = np.isin(lab, keep); removed = pred[i] & ~new; pred[i] = new
    return pp.enforce_hierarchy(pred), removed, i


def candidates():
    c = {"E0": ("w", CFG["E0"]["weights"])}
    for pair in CFG["loo"]:
        c["LOO-" + [m for m in CFG["members"] if m not in pair][0]] = ("w", [1 if m in pair else 0 for m in CFG["members"]])
    for w in CFG["weights"]:
        c["W" + "".join(map(str, w))] = ("w", w)
    for kind, v in CFG["cc_rules"]:
        c[f"{kind}{v:g}"] = ("cc", (kind, v))
    return c


def errors(pred, gt, case):
    rows = []
    for i, r in enumerate(REG):
        P, G = pred[i], gt[i]; fp = P & ~G; fn = G & ~P
        lp, npc = ndimage.label(P); lg, ngc = ndimage.label(G)
        blob_ids = [k for k in range(1, npc + 1) if not G[lp == k].any()] if npc else []
        miss_ids = [k for k in range(1, ngc + 1) if not P[lg == k].any()] if ngc else []
        blob = int(np.isin(lp, blob_ids).sum()) if blob_ids else 0
        miss = int(np.isin(lg, miss_ids).sum()) if miss_ids else 0
        row = dict(Patient_ID=case, region=r, gt_vox=int(G.sum()), pred_vox=int(P.sum()), tp=int((P & G).sum()),
                   fp=int(fp.sum()), fn=int(fn.sum()), fp_blob=blob, n_fp_blobs=len(blob_ids), fp_boundary=int(fp.sum()) - blob,
                   fn_missed=miss, n_missed=len(miss_ids), fn_boundary=int(fn.sum()) - miss, n_gt_comp=ngc, n_pred_comp=npc)
        if r == "TC":   # core <-> edema confusion
            row["fn_pred_as_edema"] = int((fn & pred[2]).sum()); row["fp_gt_edema"] = int((fp & gt[2]).sum())
        rows.append(row)
    return rows


def val_case(args):
    case, only = args
    cache = OUT / "cache" / f"{case}.pkl"   # one finished case = one file (atomic rename): resume skips it
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    probs = {m: np.load(VALDIR[m] / f"{case}.npz")["prob"].astype(np.float32) for m in CFG["members"]}
    gt = load_gt(case); out, rem = [], []
    p0 = weighted(probs, CFG["E0"]["weights"]); e0 = post_e0(p0)
    err = errors(e0, gt, case)
    for name, (kind, a) in candidates().items():
        if only and name not in only:
            continue
        if kind == "w":
            pred = e0 if name == "E0" else post_e0(weighted(probs, a))
        else:
            pred, removed, i = cc_rule(e0, p0, *a)
            rem.append(dict(candidate=name, Patient_ID=case, region=REG[i], removed_vox=int(removed.sum()),
                            removed_gt_vox=int((removed & gt[i]).sum()), gt_vox=int(gt[i].sum())))
        s = score_case(pred, gt); s.update(candidate=name, Patient_ID=case); out.append(s)
    if not only:
        tmp = cache.with_suffix(f".tmp{os.getpid()}"); tmp.write_bytes(pickle.dumps((out, err, rem))); os.replace(tmp, cache)
    return out, err, rem


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only", default=""); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(); (OUT / "cache").mkdir(exist_ok=True)
    only = set(filter(None, a.only.split(",")))
    cases = sorted(f.stem for f in VALDIR["ep253"].glob("*.npz"))
    assert len(cases) == 126 and all(len(list(d.glob("*.npz"))) == 126 for d in VALDIR.values())
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(val_case, [(c, only) for c in cases]))
    pd.DataFrame([r for o, _, _ in res for r in o]).to_csv(OUT / "per_case_val.csv", index=False)
    pd.DataFrame([r for _, e, _ in res for r in e]).to_csv(OUT / "errors_E0_val.csv", index=False)
    pd.DataFrame([r for _, _, m in res for r in m]).to_csv(OUT / "cc_removed_val.csv", index=False)
    d = pd.read_csv(OUT / "per_case_val.csv")
    e0 = d[d.candidate == "E0"].Dice_Mean.mean()
    print(f"E0 VAL mean Dice {e0:.6f} (expected {CFG['E0']['expected_val_dice']:.6f})")
    assert abs(e0 - CFG["E0"]["expected_val_dice"]) < 1e-6, "E0 does not reproduce Ensemble 3's VAL Dice"


if __name__ == "__main__":
    main()
