#!/usr/bin/env python3
"""PROTOCOL_v2 Amendment 28, stage 2: comparison of every candidate with E0 (= Ensemble 3) on the same 126 VAL cases,
from run_val.py's outputs. Writes comparison_val.csv, worst_cases_val.csv, error_summary_val.csv and report.txt."""
import json, pathlib
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

OUT = pathlib.Path(__file__).resolve().parent
CFG = json.loads((OUT / "config.json").read_text()); RULE = CFG["rule"]
d = pd.read_csv(OUT / "per_case_val.csv"); err = pd.read_csv(OUT / "errors_E0_val.csv"); rem = pd.read_csv(OUT / "cc_removed_val.csv")
REG = ("ET", "TC", "WT"); DC = ["Dice_Mean"] + [f"Dice_{r}" for r in REG]
W = {c: g.set_index("Patient_ID").sort_index() for c, g in d.groupby("candidate", sort=False)}
cases = sorted(W["E0"].index); e0 = W["E0"]
perm = np.random.default_rng(CFG["folds"]["seed"]).permutation(len(cases))
fold = pd.Series({cases[j]: k % CFG["folds"]["k"] for k, j in enumerate(perm)})
hd = lambda x: x[[f"HD95_{r}" for r in REG]].mean(axis=1, skipna=True)   # per-case mean over defined regions
lines = []
P = lambda s="": (print(s), lines.append(s))


P("Amendment 28 — EXPLORATORY validation analysis of Ensemble 3 (126 VAL cases). These VAL cases already chose every member")
P("checkpoint, the members and the post-processing of Ensemble 3 (870 configurations, Amendment 17); fold splits inside them")
P("cannot undo that selection, so nothing here is an unbiased estimate. TEST was not read. Choices INHERITED from the earlier")
P("full-VAL selection (not re-chosen per fold): member checkpoints (best-by-VAL epochs), each member's inference mode (rotation")
P("TTA for runC_rot/runD2_rot, chosen on VAL), the member set, thresholds 0.5, wt_policy components, floors x1, ET min-volume 100.")
P("New choices (the 17 fixed candidates in config.json) are selected in the nested check with the outer fold held out.")
P("Motivation note: the component rules were suggested partly by earlier observations of FP-blob collapses (Run A, TEST);")
P("no rule parameter was derived from TEST data.\n")
ref = pd.read_csv(OUT.parent / "ensemble3" / "ensemble_val_search.csv")
ref = ref[(ref.combo == "ep253+runC_rot+runD2_rot") & (ref.wt_policy == "components") & (ref.floor_mult == 1) & (ref.et_min_volume == 100)].iloc[0]
P("E0 reproduction (full VAL, 126): " + "  ".join(f"{m} {e0[m].mean():.6f} vs original {ref[m]:.6f}" for m in DC))
assert all(abs(e0[m].mean() - ref[m]) < 1e-9 for m in DC), "E0 does not reproduce Ensemble 3"
P("-> E0 reproduces Ensemble 3 exactly (members' saved VAL probabilities unchanged since the 2026-10-03 selection).\n")


def paired(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    bs = x[np.random.default_rng(0).integers(0, len(x), (10000, len(x)))].mean(1); lo, hi = np.percentile(bs, [2.5, 97.5])
    p = wilcoxon(x).pvalue if np.any(x != 0) else 1.0
    return x.mean(), lo, hi, p


# ---------- step 1: error analysis of E0 ----------
P("== Step 1: E0 (Ensemble 3) VAL error decomposition (126 cases), voxels summed over cases")
es = err.groupby("region")[["gt_vox", "fp", "fp_blob", "fp_boundary", "fn", "fn_missed", "fn_boundary"]].sum()
es["fp_blob_%err"] = 100 * es.fp_blob / (es.fp + es.fn); es["fp_bound_%err"] = 100 * es.fp_boundary / (es.fp + es.fn)
es["fn_missed_%err"] = 100 * es.fn_missed / (es.fp + es.fn); es["fn_bound_%err"] = 100 * es.fn_boundary / (es.fp + es.fn)
tc = err[err.region == "TC"]
P(es.round(1).to_string())
P(f"TC errors that are core<->edema confusion: FN predicted as edema {tc.fn_pred_as_edema.sum()} vox "
  f"({100 * tc.fn_pred_as_edema.sum() / tc.fn.sum():.1f} % of TC FN), FP that is GT edema {tc.fp_gt_edema.sum()} vox "
  f"({100 * tc.fp_gt_edema.sum() / tc.fp.sum():.1f} % of TC FP)")
es.to_csv(OUT / "error_summary_val.csv")
P(f"cases with empty GT: " + ", ".join(f"{r} {int((err[err.region == r].gt_vox == 0).sum())}" for r in REG))


def dominant(row):
    parts = {"FP blob": row.fp_blob, "boundary FP": row.fp_boundary, "missed region": row.fn_missed, "boundary FN": row.fn_boundary}
    if row.region == "TC":
        parts = {"FP blob": row.fp_blob, "missed region": row.fn_missed,
                 "core->edema (FN)": row.fn_pred_as_edema, "edema->core (FP)": row.fp_gt_edema,
                 "other boundary FP": row.fp_boundary - row.fp_gt_edema, "other boundary FN": row.fn_boundary - row.fn_pred_as_edema}
    if row.gt_vox == 0:
        return "empty GT, predicted %d vox" % row.pred_vox
    k = max(parts, key=parts.get); tot = row.fp + row.fn
    return f"{k} ({100 * parts[k] / tot:.0f} % of errors)" if tot else "none"


worst = {}
for r in ("TC", "WT"):
    ids = e0[f"Dice_{r}"].sort_values().index[:10]; worst[r] = list(ids)
    P(f"\nWorst 10 VAL cases by Dice_{r} (E0):")
    er = err[err.region == r].set_index("Patient_ID")
    for c in ids:
        x = er.loc[c]
        P(f"  {c} Dice_{r} {e0.loc[c, f'Dice_{r}']:.3f} (ET {e0.loc[c, 'Dice_ET']:.3f} TC {e0.loc[c, 'Dice_TC']:.3f} WT {e0.loc[c, 'Dice_WT']:.3f}) "
          f"GT {x.gt_vox} pred {x.pred_vox} FP {x.fp} (blob {x.fp_blob} in {x.n_fp_blobs}) FN {x.fn} (missed {x.fn_missed} in "
          f"{x.n_missed}) -> {dominant(x)}")
gap = {r: 1 - e0[f"Dice_{r}"].mean() for r in REG}
for r in ("TC", "WT"):
    P(f"share of the total VAL Dice_{r} shortfall (1 - mean = {gap[r]:.4f}) from the worst 10 cases: "
      f"{100 * (1 - e0.loc[worst[r], f'Dice_{r}']).sum() / (1 - e0[f'Dice_{r}']).sum():.1f} %; worst 2: "
      f"{100 * (1 - e0.loc[worst[r][:2], f'Dice_{r}']).sum() / (1 - e0[f'Dice_{r}']).sum():.1f} %")

# ---------- steps 2-4: candidates vs E0 ----------
P("\n== FULL-VAL results (all 126 cases; optimistic, VAL reused): candidates vs E0 (paired bootstrap 10,000 seed 0 + two-sided Wilcoxon; unadjusted for 17 comparisons)")
rows = []
for c, x in W.items():
    x = x.loc[cases]; r = dict(candidate=c)
    for m in DC:
        r[m] = x[m].mean()
    for g in REG:
        h = x[f"HD95_{g}"]; r[f"HD95_{g}"] = h.mean(skipna=True); r[f"HD95_{g}_undef"] = int(h.isna().sum())
    if c != "E0":
        for m in DC:
            mu, lo, hi, p = paired(x[m] - e0[m]); r[f"d_{m}"] = mu; r[f"d_{m}_lo"] = lo; r[f"d_{m}_hi"] = hi; r[f"p_{m}"] = p
        mu, lo, hi, p = paired(hd(x) - hd(e0)); r.update(d_HD95=mu, d_HD95_lo=lo, d_HD95_hi=hi, p_HD95=p)
        dm = x.Dice_Mean - e0.Dice_Mean
        fg = dm.groupby(fold).mean(); r["fold_gains"] = " ".join(f"{v:+.4f}" for v in fg); r["folds_nonneg"] = int((fg >= -1e-12).sum())
        r["cases_better"] = int((dm > 1e-9).sum()); r["cases_worse"] = int((dm < -1e-9).sum())
        for g in ("TC", "WT"):
            dw = x.loc[worst[g], f"Dice_{g}"] - e0.loc[worst[g], f"Dice_{g}"]
            r[f"worst10_{g}_d"] = dw.mean(); r[f"worst10_{g}_up"] = int((dw > 0.01).sum()); r[f"worst10_{g}_down"] = int((dw < -0.01).sum())
        r["cases_any_region_loss_gt_0.05"] = int(((x[[f"Dice_{g}" for g in REG]] - e0[[f"Dice_{g}" for g in REG]]) < -0.05).any(axis=1).sum())
        r["max_case_loss"] = float(-(x[[f"Dice_{g}" for g in REG]] - e0[[f"Dice_{g}" for g in REG]]).min().min())
        rr = rem[rem.candidate == c]
        if len(rr):
            totgt = err[err.region == rr.region.iloc[0]].gt_vox.sum()
            r.update(removed_vox=int(rr.removed_vox.sum()), removed_gt_vox=int(rr.removed_gt_vox.sum()),
                     removed_gt_frac=rr.removed_gt_vox.sum() / totgt, cases_changed=int((rr.removed_vox > 0).sum()),
                     cases_gt_removed=int((rr.removed_gt_vox > 0).sum()))
        ok = [r["d_Dice_Mean_lo"] > 0, r["folds_nonneg"] >= RULE["min_folds_nonneg"]]
        if len(rr):
            ok.append(r["removed_gt_frac"] <= RULE["cc_max_gt_removed_frac"] and r["max_case_loss"] <= RULE["cc_max_case_dice_loss"])
        r["passes_a_b_c"] = all(ok)
    rows.append(r)
T = pd.DataFrame(rows); T.to_csv(OUT / "comparison_val.csv", index=False)
fmt = lambda r, m: f"{r[f'd_{m}']:+.4f} [{r[f'd_{m}_lo']:+.4f},{r[f'd_{m}_hi']:+.4f}]"
P(f"{'candidate':12s} {'Mean':>6s} {'ET':>6s} {'TC':>6s} {'WT':>6s} {'HD95 ET/TC/WT (undef)':>26s} | d Mean [CI]              p      "
  f"d TC [CI]                d WT [CI]                d HD95 mm [CI]           folds>=0 better/worse worst10 TC/WT  pass")
for _, r in T.iterrows():
    base = (f"{r.candidate:12s} {r.Dice_Mean:.4f} {r.Dice_ET:.4f} {r.Dice_TC:.4f} {r.Dice_WT:.4f} "
            f"{r.HD95_ET:5.2f}/{r.HD95_TC:5.2f}/{r.HD95_WT:5.2f} ({r.HD95_ET_undef:.0f},{r.HD95_TC_undef:.0f},{r.HD95_WT_undef:.0f})")
    if r.candidate == "E0":
        P(base + " | reference"); continue
    P(base + f" | {fmt(r, 'Dice_Mean')} {r.p_Dice_Mean:.3g}  {fmt(r, 'Dice_TC')} {fmt(r, 'Dice_WT')} "
      f"{r.d_HD95:+.3f} [{r.d_HD95_lo:+.3f},{r.d_HD95_hi:+.3f}]  {r.folds_nonneg}/5  {r.cases_better}/{r.cases_worse}  "
      f"{r.worst10_TC_d:+.4f}/{r.worst10_WT_d:+.4f}  {'YES' if r.passes_a_b_c else 'no'}")
P("\nworst-10 cases (E0's 10 lowest by TC / by WT): mean change, # improved > 0.01 / # worse > 0.01; cases losing > 0.05 in any region")
for _, r in T[T.candidate != "E0"].iterrows():
    P(f"  {r.candidate:12s} TC {r.worst10_TC_d:+.4f} ({r.worst10_TC_up:.0f} up / {r.worst10_TC_down:.0f} down)  "
      f"WT {r.worst10_WT_d:+.4f} ({r.worst10_WT_up:.0f} up / {r.worst10_WT_down:.0f} down)  any-region loss > 0.05: {r['cases_any_region_loss_gt_0.05']:.0f}")
P("\nfold consistency (FULL-VAL candidates, no selection): per-fold mean-Dice gain vs E0, folds 0-4:")
for _, r in T[T.candidate != "E0"].iterrows():
    P(f"  {r.candidate:12s} {r.fold_gains}")
cc = T[T.candidate.str.contains("rel|conf")]
P("\ncomponent rules: voxels removed / of which GT tumour (same region) / cases changed / cases losing GT voxels / max per-case Dice loss")
for _, r in cc.iterrows():
    P(f"  {r.candidate:12s} {r.removed_vox:8.0f} / {r.removed_gt_vox:6.0f} ({100 * r.removed_gt_frac:.3f} % of GT) / "
      f"{r.cases_changed:.0f} / {r.cases_gt_removed:.0f} / {r.max_case_loss:.4f}")

# ---------- step 5: nested selection across the 5 VAL folds ----------
P("\n== OUTER-FOLD results, MEAN-DICE-ONLY nested selection (exploratory): for each outer fold k, the candidate with the highest")
P("   mean Dice on the other 4 folds is chosen from {E0 + 17 candidates} (ties -> E0). Rules (a)-(c) of Amendment 28 were NOT")
P("   applied inside the selection folds, so this is NOT an evaluation of the full prespecified selection rule.")
P("   (the held-out fold takes no part in any new weight / post-processing choice) and scored on fold k; the 126 held-out")
P("   predictions are then pooled. Inherited full-VAL choices (see header) are NOT re-chosen per fold.")
held, pooled = [], []
for k in range(CFG["folds"]["k"]):
    tr = fold.index[fold != k]; te = fold.index[fold == k]
    means = {c: x.loc[tr, "Dice_Mean"].mean() for c, x in W.items()}
    best = max(means, key=lambda c: (round(means[c], 12), c == "E0"))
    g = W[best].loc[te, "Dice_Mean"].mean() - e0.loc[te, "Dice_Mean"].mean(); held.append(g)
    pooled.append(W[best].loc[te])
    P(f"  outer fold {k} (n={len(te)}): chosen {best:12s} (inner 4-fold gain {means[best] - means['E0']:+.4f}) "
      f"held-out E0 {e0.loc[te, 'Dice_Mean'].mean():.4f} -> chosen {W[best].loc[te, 'Dice_Mean'].mean():.4f} gain {g:+.4f}")
pooled = pd.concat(pooled).loc[cases]
P("  pooled outer-fold (126): " + "  ".join(f"{m} {pooled[m].mean():.4f} (E0 {e0[m].mean():.4f})" for m in DC))
for m in DC:
    mu, lo, hi, p = paired(pooled[m] - e0[m])
    P(f"    pooled outer-fold minus E0 {m:9s} {mu:+.4f} [{lo:+.4f}, {hi:+.4f}] Wilcoxon p={p:.3g}")
P(f"  mean held-out gain over folds {np.mean(held):+.4f} (mean-Dice-only selection) -> rule (d) as written {'met' if np.mean(held) > 0 else 'NOT met'}")
passing = T[(T.candidate != "E0") & T.passes_a_b_c.astype('boolean').fillna(False).astype(bool)].candidate.tolist()
P(f"\nCandidates passing (a)-(c): {passing or 'none'}; rule (d): {'met' if np.mean(held) > 0 else 'not met'}")
(OUT / "report.txt").write_text("\n".join(lines) + "\n")
pd.DataFrame([dict(region=g, rank=i + 1, Patient_ID=c, **{f"{cand}": W[cand].loc[c, f'Dice_{g}'] for cand in W})
              for g in ("TC", "WT") for i, c in enumerate(worst[g])]).to_csv(OUT / "worst_cases_val.csv", index=False)
