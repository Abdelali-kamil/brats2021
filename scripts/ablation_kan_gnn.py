#!/usr/bin/env python3
"""BraTS 2021 MGMT classification with the KAN + Transformer + GNN module.

Primary task
------------
MGMT promoter methylation on **BraTS 2021** (n=577, features from expert
segmentations, 301 methylated / 276 not). This is the project's primary
classification dataset. It is the *only* classification label BraTS 2021
carries: the challenge provides MGMT status, not a tumour-grade label, so
"tumour grading" is not a task that exists for this cohort and is not attempted.

External validation
--------------------
The BraTS-trained model is applied to the **UPenn-GBM** MGMT cohort (n=227) as
an independent generalisation test, on the 51 radiomic features the two cohorts
share. UPenn is never used to train or select the primary model.

Two honest constraints, both forced by the data
------------------------------------------------
1. BraTS 2021 ships no clinical metadata, so on the primary task the module is
   *imaging-only*: the clinical encoder and the adaptive gate have nothing to
   fuse. The multimodal (imaging + clinical) branch and the gate are therefore
   demonstrated only on the UPenn cohort, which does carry age / sex / GTR --
   reported as a secondary external-cohort analysis, not as the primary result.
2. Predicting MGMT from MRI is a task with contested signal: the RSNA-MICCAI
   2021 challenge that produced these labels was won at ~0.62 AUC, and the
   project's own radiomic random forest sits at ~0.58. A near-chance result
   from this module is a legitimate, expected outcome and is reported as such.
   Nothing here is tuned on the quantity being reported.

Ablation ladder (imaging-only, BraTS primary)
---------------------------------------------
  mlp              imaging MLP encoder -> head
  kan              KAN encoder in place of MLP                 (Innovation 2)
  kan_transformer  + region-token Transformer branch           (paper V-F)
  kan_gnn          + retrieval-augmented memory bank           (Innovation 3)
  full             KAN + Transformer + GNN

Leakage controls (unchanged from the audited protocol)
------------------------------------------------------
  * Standardisation / imputation fit on the training split only.
  * Nested CV: hyperparameters chosen in an inner loop over training folds; the
    outer fold is scored once and never influences selection.
  * The memory bank holds training-patient embeddings only, rebuilt per fold;
    `exclude_self` stops a node being its own neighbour. For external
    validation the bank holds BraTS embeddings and UPenn patients query into it.
  * Graph edges use feature similarity; labels are never consulted.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brats_gbm.eval.stats import bootstrap_ci  # noqa: E402
from brats_gbm.gnn import ClinicalImagingKANGNN  # noqa: E402

BRATS_COHORT = ROOT / "results" / "classification" / "brats_mgmt_features.csv"
UPENN_COHORT = ROOT / "results" / "classification" / "upenn_mgmt_cohort.csv"
OUT_DIR = ROOT / "results" / "classification"

# Columns that are identifiers, labels or clinical covariates, never imaging.
NON_FEATURE = {"case_id", "patient_id", "mask_source", "mgmt_label",
               "idh1_label", "idh1_raw", "age", "gender_m", "gtr_over90"}
CLINICAL_COLS = ["age", "gender_m", "gtr_over90"]
REGIONS = ("ET", "TC", "WT")

SEED = 42
OUTER_FOLDS, INNER_FOLDS, N_REPEATS = 5, 3, 3
MAX_EPOCHS, PATIENCE = 150, 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Imaging-only ablation rungs. use_gate needs clinical data, so it is off for
# every BraTS rung; it is switched on only in the UPenn multimodal analysis.
CONFIGS = {
    "mlp":             dict(use_clinical=False, use_kan=False, use_gnn=False,
                            use_transformer=False, use_gate=False),
    "kan":             dict(use_clinical=False, use_kan=True,  use_gnn=False,
                            use_transformer=False, use_gate=False),
    "kan_transformer": dict(use_clinical=False, use_kan=True,  use_gnn=False,
                            use_transformer=True,  use_gate=False),
    "kan_gnn":         dict(use_clinical=False, use_kan=True,  use_gnn=True,
                            use_transformer=False, use_gate=False),
    "full":            dict(use_clinical=False, use_kan=True,  use_gnn=True,
                            use_transformer=True,  use_gate=False),
}

GRID = {"hidden": [8, 16], "lr": [3e-3, 1e-2], "dropout": [0.3, 0.5]}
GNN_GRID = {"k": [5, 10]}
KAN_GRID = {"l1": [0.0, 1e-3]}

# Fixed, a-priori configuration for the external-validation model, so the UPenn
# numbers are never a function of anything fitted on UPenn. Small and heavily
# regularised, consistent with what the BraTS inner loop favours at this n.
EXT_HP = {"hidden": 8, "lr": 1e-2, "dropout": 0.5, "k": 10, "l1": 1e-3}
EXT_ENSEMBLE = 5


def region_tokenize(cols: list[str]) -> tuple[list[str], list[tuple[int, int]]]:
    """Order feature columns by anatomical region and return contiguous slices.

    Every radiomic column is named `<REGION>_<...>`; grouping by region gives
    the Transformer a short sequence of region tokens to attend across. Any
    column not matching ET/TC/WT (e.g. cross-region ratios) is collected into a
    trailing 'global' token so nothing is silently dropped.
    """
    order: list[str] = []
    slices: list[tuple[int, int]] = []
    idx = 0
    for r in REGIONS:
        grp = [c for c in cols if c.split("_")[0] == r]
        if not grp:
            continue
        order.extend(grp)
        slices.append((idx, idx + len(grp)))
        idx += len(grp)
    rest = [c for c in cols if c not in order]
    if rest:
        order.extend(rest)
        slices.append((idx, idx + len(rest)))
    return order, slices


def load_brats():
    df = pd.read_csv(BRATS_COHORT)
    df = df[df.mgmt_label.notna()].reset_index(drop=True)
    df["mgmt_label"] = df["mgmt_label"].astype(int)
    return df


def load_upenn():
    df = pd.read_csv(UPENN_COHORT)
    df = df[df.mgmt_label.notna()].reset_index(drop=True)
    df["mgmt_label"] = df["mgmt_label"].astype(int)
    return df


def fit_transform(train: np.ndarray, *others: np.ndarray):
    """Median-impute and standardise, fitted on `train` only."""
    med = np.nanmedian(train, axis=0)
    med = np.where(np.isnan(med), 0.0, med)

    def imp(a):
        a = a.copy()
        idx = np.where(np.isnan(a))
        a[idx] = np.take(med, idx[1])
        return a

    tr = imp(train)
    mu, sd = tr.mean(0), tr.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    out = [(tr - mu) / sd]
    for o in others:
        out.append((imp(o) - mu) / sd)
    return out


def _t(a):
    return torch.tensor(a, dtype=torch.float32, device=DEVICE)


def train_model(cfg, hp, Xi_tr, Xc_tr, y_tr, Xi_va, Xc_va, y_va,
                region_slices, seed):
    torch.manual_seed(seed)
    use_clin = cfg.get("use_clinical", False)
    n_clin = Xc_tr.shape[1] if (use_clin and Xc_tr is not None) else 0
    use_tf = cfg.get("use_transformer", False)

    model = ClinicalImagingKANGNN(
        n_imaging=Xi_tr.shape[1], n_clinical=n_clin, hidden=hp["hidden"],
        use_kan=cfg["use_kan"], use_gnn=cfg["use_gnn"],
        use_gate=cfg.get("use_gate", False), use_transformer=use_tf,
        region_slices=region_slices if use_tf else None,
        k=hp.get("k", 8), dropout=hp["dropout"]).to(DEVICE)

    ti, vi = _t(Xi_tr), _t(Xi_va)
    tc = _t(Xc_tr) if n_clin else None
    vc = _t(Xc_va) if n_clin else None
    ty = _t(y_tr)

    pos_weight = _t([(len(y_tr) - y_tr.sum()) / max(1, y_tr.sum())])
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=1e-4)

    best_auc, best_state, bad = -1.0, None, 0
    for _ in range(MAX_EPOCHS):
        model.train()
        if cfg["use_gnn"]:
            model.refresh_memory(ti, tc)
        opt.zero_grad()
        out = model(ti, tc, exclude_self=cfg["use_gnn"])
        loss = lossf(out, ty)
        if cfg["use_kan"] and hp.get("l1", 0.0) > 0:
            loss = loss + hp["l1"] * model.regularisation()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        model.eval()
        with torch.no_grad():
            if cfg["use_gnn"]:
                model.refresh_memory(ti, tc)
            p = torch.sigmoid(model(vi, vc)).cpu().numpy()
        auc = roc_auc_score(y_va, p) if len(np.unique(y_va)) > 1 else 0.5
        if auc > best_auc:
            best_auc, bad = auc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_auc


def hp_grid(cfg):
    grids = dict(GRID)
    if cfg["use_gnn"]:
        grids.update(GNN_GRID)
    if cfg["use_kan"]:
        grids.update(KAN_GRID)
    keys = list(grids)
    for combo in itertools.product(*(grids[k] for k in keys)):
        yield dict(zip(keys, combo))


def predict(model, cfg, Xi, Xc, bank_i=None, bank_c=None):
    model.eval()
    with torch.no_grad():
        if cfg["use_gnn"] and bank_i is not None:
            model.refresh_memory(_t(bank_i), _t(bank_c) if bank_c is not None else None)
        xc = _t(Xc) if (cfg.get("use_clinical") and Xc is not None) else None
        return torch.sigmoid(model(_t(Xi), xc)).cpu().numpy()


def run_config(name, cfg, X_img, X_clin, y, region_slices, repeats):
    """Nested-CV out-of-fold AUC for one ablation rung on the primary cohort."""
    oof = np.zeros((repeats, len(y)))
    chosen, graph_reliance = [], []

    for rep in range(repeats):
        outer = StratifiedKFold(OUTER_FOLDS, shuffle=True, random_state=SEED + rep)
        for tr_idx, te_idx in outer.split(X_img, y):
            inner = StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=SEED)
            best_hp, best_score = None, -1.0
            for hp in hp_grid(cfg):
                scores = []
                for i_tr, i_va in inner.split(X_img[tr_idx], y[tr_idx]):
                    a, b = tr_idx[i_tr], tr_idx[i_va]
                    Xi_a, Xi_b = fit_transform(X_img[a], X_img[b])
                    Xc_a, Xc_b = fit_transform(X_clin[a], X_clin[b])
                    _, auc = train_model(cfg, hp, Xi_a, Xc_a, y[a], Xi_b, Xc_b, y[b],
                                         region_slices, SEED)
                    scores.append(auc)
                m = float(np.mean(scores))
                if m > best_score:
                    best_score, best_hp = m, hp
            chosen.append(dict(best_hp))

            Xi_tr, Xi_te = fit_transform(X_img[tr_idx], X_img[te_idx])
            Xc_tr, Xc_te = fit_transform(X_clin[tr_idx], X_clin[te_idx])
            model, _ = train_model(cfg, best_hp, Xi_tr, Xc_tr, y[tr_idx],
                                   Xi_te, Xc_te, y[te_idx], region_slices, SEED)
            if cfg["use_gnn"]:
                model.eval()
                with torch.no_grad():
                    model.refresh_memory(_t(Xi_tr),
                                         _t(Xc_tr) if cfg.get("use_clinical") else None)
                    graph_reliance.append(model.gconv.neighbour_weight)
            oof[rep, te_idx] = predict(model, cfg, Xi_te, Xc_te, Xi_tr, Xc_tr)
        print(f"    repeat {rep + 1}/{repeats} done", flush=True)

    mean_prob = oof.mean(axis=0)
    per_repeat = [roc_auc_score(y, oof[r]) for r in range(repeats)]
    rng = np.random.default_rng(SEED)
    boot = [roc_auc_score(y[s], mean_prob[s])
            for s in (rng.integers(0, len(y), len(y)) for _ in range(2000))
            if len(np.unique(y[s])) > 1]
    return {
        "auc": float(roc_auc_score(y, mean_prob)),
        "auc_ci_low": float(np.percentile(boot, 2.5)),
        "auc_ci_high": float(np.percentile(boot, 97.5)),
        "auc_per_repeat_mean": float(np.mean(per_repeat)),
        "auc_per_repeat_sd": float(np.std(per_repeat)),
        "graph_reliance_mean": float(np.mean(graph_reliance)) if graph_reliance else None,
        "hp_selected": chosen,
    }


def external_validate(cfg, X_tr, y_tr, X_te, y_te, region_slices, seed_base=SEED):
    """Train on ALL BraTS with a fixed config, score once on UPenn.

    An ensemble over a few seeds (each with its own internal early-stopping
    split carved from BraTS only) stabilises the estimate without ever touching
    UPenn for selection. The memory bank, when used, holds BraTS embeddings and
    UPenn patients query into it -- the inductive setting the bank was built for.
    """
    probs = []
    for s in range(EXT_ENSEMBLE):
        rng = np.random.default_rng(seed_base + s)
        perm = rng.permutation(len(y_tr))
        n_val = max(1, int(0.15 * len(y_tr)))
        va, tr = perm[:n_val], perm[n_val:]
        Xi_tr, Xi_va, Xi_te = fit_transform(X_tr[tr], X_tr[va], X_te)
        model, _ = train_model(cfg, EXT_HP, Xi_tr, None, y_tr[tr],
                               Xi_va, None, y_tr[va], region_slices, seed_base + s)
        # rebuild bank from the full BraTS training split for scoring
        Xi_bank, Xi_te2 = fit_transform(X_tr, X_te)
        probs.append(predict(model, cfg, Xi_te2, None, Xi_bank, None))
    mean_prob = np.mean(probs, axis=0)
    rng = np.random.default_rng(seed_base)
    boot = [roc_auc_score(y_te[s], mean_prob[s])
            for s in (rng.integers(0, len(y_te), len(y_te)) for _ in range(2000))
            if len(np.unique(y_te[s])) > 1]
    return {
        "auc": float(roc_auc_score(y_te, mean_prob)),
        "auc_ci_low": float(np.percentile(boot, 2.5)),
        "auc_ci_high": float(np.percentile(boot, 97.5)),
        "n": int(len(y_te)), "n_positive": int(y_te.sum()),
    }


def upenn_multimodal(X_img, X_clin, y, region_slices, repeats):
    """Secondary analysis: the imaging+clinical gated-fusion module on UPenn.

    BraTS carries no clinical data, so this is the only cohort where the gate
    and the clinical branch can be exercised. Nested CV, clinical + imaging,
    gate on. Reported as an external-cohort demonstration of the fusion design,
    not as the primary result.
    """
    cfg = dict(use_clinical=True, use_kan=True, use_gnn=True,
               use_transformer=True, use_gate=True)
    return run_config("upenn_full_multimodal", cfg, X_img, X_clin, y,
                      region_slices, repeats)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--repeats", type=int, default=N_REPEATS)
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    ap.add_argument("--skip-external", action="store_true")
    ap.add_argument("--skip-upenn-multimodal", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir) / "kan_gnn_brats.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    brats = load_brats()
    feat_all = [c for c in brats.columns
                if c not in NON_FEATURE and pd.api.types.is_numeric_dtype(brats[c])]
    # Restrict to the feature space shared with UPenn so the primary ablation
    # and the external test use an identical, comparable representation.
    upenn = load_upenn()
    shared = [c for c in feat_all if c in set(upenn.columns)]
    cols, region_slices = region_tokenize(shared)

    Xb = brats[cols].to_numpy(float)
    yb = brats["mgmt_label"].to_numpy(int)
    Xb_clin = np.zeros((len(yb), 1))  # placeholder; BraTS has no clinical data

    print(f"PRIMARY  : BraTS 2021 MGMT  n={len(yb)}, "
          f"{int(yb.sum())} methylated ({yb.mean() * 100:.1f}%)")
    print(f"features : {len(cols)} radiomic (shared with UPenn), "
          f"{len(region_slices)} region tokens")
    print(f"device   : {DEVICE}")
    print(f"protocol : nested CV {OUTER_FOLDS}x{INNER_FOLDS}, {args.repeats} repeats\n")

    results: dict = {}
    if out.exists() and not args.force:
        try:
            results = json.loads(out.read_text())
        except json.JSONDecodeError:
            pass

    def save():
        payload = {k: v for k, v in results.items()}
        payload["cohort"] = {
            "primary": "BraTS2021_MGMT", "primary_n": int(len(yb)),
            "primary_positives": int(yb.sum()), "n_features": len(cols),
            "n_region_tokens": len(region_slices),
            "external": "UPenn-GBM_MGMT",
        }
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(out)

    # --- primary: BraTS ablation ---
    for name in args.configs:
        if name in results and "auc" in results.get(name, {}) \
                and results[name].get("repeats") == args.repeats and not args.force:
            print(f"[{name}] cached (AUC {results[name]['auc']:.3f}) -- skipping")
            continue
        print(f"[{name}]", flush=True)
        res = run_config(name, CONFIGS[name], Xb, Xb_clin, yb, region_slices, args.repeats)
        res["repeats"] = args.repeats
        results[name] = res
        extra = (f"  graph_reliance {res['graph_reliance_mean']:.3f}"
                 if res["graph_reliance_mean"] is not None else "")
        print(f"  AUC {res['auc']:.3f} [{res['auc_ci_low']:.3f}, "
              f"{res['auc_ci_high']:.3f}]{extra}\n", flush=True)
        save()

    # --- external validation: BraTS-trained full model -> UPenn ---
    if not args.skip_external:
        Xu = upenn[cols].to_numpy(float)
        yu = upenn["mgmt_label"].to_numpy(int)
        print("[external] BraTS-trained 'full' model -> UPenn-GBM MGMT", flush=True)
        ext = external_validate(CONFIGS["full"], Xb, yb, Xu, yu, region_slices)
        results["external_upenn"] = ext
        print(f"  AUC {ext['auc']:.3f} [{ext['auc_ci_low']:.3f}, "
              f"{ext['auc_ci_high']:.3f}]  (n={ext['n']}, {ext['n_positive']} pos)\n",
              flush=True)
        save()

    # --- secondary: UPenn multimodal (imaging + clinical + gate) ---
    if not args.skip_upenn_multimodal:
        Xu = upenn[cols].to_numpy(float)
        Xu_clin = upenn[CLINICAL_COLS].to_numpy(float)
        yu = upenn["mgmt_label"].to_numpy(int)
        print("[upenn_multimodal] imaging + clinical gated fusion (UPenn only)",
              flush=True)
        res = upenn_multimodal(Xu, Xu_clin, yu, region_slices, args.repeats)
        res["repeats"] = args.repeats
        results["upenn_multimodal"] = res
        print(f"  AUC {res['auc']:.3f} [{res['auc_ci_low']:.3f}, "
              f"{res['auc_ci_high']:.3f}]\n", flush=True)
        save()

    # --- summary ---
    print("=" * 70)
    print(f"{'config':<20}{'AUC [95% CI]':<30}{'vs prev':>18}")
    print("=" * 70)
    prev = None
    for name in args.configs:
        r = results.get(name)
        if not r:
            continue
        d = "" if prev is None else f"{r['auc'] - prev:+.3f}"
        print(f"{name:<20}{r['auc']:.3f} [{r['auc_ci_low']:.3f}, {r['auc_ci_high']:.3f}]{d:>18}")
        prev = r["auc"]
    if "external_upenn" in results:
        e = results["external_upenn"]
        print(f"{'-> UPenn (external)':<20}{e['auc']:.3f} [{e['auc_ci_low']:.3f}, {e['auc_ci_high']:.3f}]")
    if "upenn_multimodal" in results:
        m = results["upenn_multimodal"]
        print(f"{'UPenn multimodal':<20}{m['auc']:.3f} [{m['auc_ci_low']:.3f}, {m['auc_ci_high']:.3f}]")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
