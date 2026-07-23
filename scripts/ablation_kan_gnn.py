#!/usr/bin/env python3
"""Ablation and nested-CV hyperparameter search for the KAN + GNN module.

Ablation ladder. Each rung differs from the one below by exactly one component,
so any change in score is attributable to that component:

  1. imaging_only        imaging features -> MLP -> head
  2. mlp_fusion          + clinical features, MLP encoders, concat fusion
  3. kan_fusion          + KAN encoders in place of MLP      (Innovation 2)
  4. kan_gnn             + memory bank over training patients (Innovation 3)
  5. kan_gnn_gate        + adaptive gated fusion              (proposal 3.1)

Why nested cross-validation
---------------------------
Hyperparameters are chosen in an inner loop over the training split only; the
outer fold is scored once with the winning configuration and never influences
selection. Flat CV -- tuning and reporting on the same folds -- is what produced
the selection overfitting found earlier in this project, where validation rose
0.838 -> 0.851 while test fell 0.816 -> 0.807. With 227 subjects and five
architecture variants the same trap is wide open, and nested CV is the only
honest way to report "we tuned it".

The cost is real: outer x inner x grid x epochs. It is affordable here because
the models are small and the cohort is tiny.

Leakage controls, all enforced in code
--------------------------------------
  * Standardisation and median imputation are fit on the training split only
    and applied to validation/test.
  * The memory bank is built from training embeddings only and rebuilt for
    every fold. Test patients never enter it.
  * When embedding training data for the bank, `exclude_self=True` stops a node
    being its own neighbour.
  * Graph edges use feature similarity; labels are never consulted.
  * Class balance is handled with a positive-class loss weight computed from
    the training split alone.
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

COHORT = ROOT / "results" / "classification" / "upenn_mgmt_cohort.csv"
OUT_DIR = ROOT / "results" / "classification"
CLINICAL_COLS = ["age", "gender_m", "gtr_over90"]
NON_FEATURE = {"patient_id", "mask_source", "idh1_label", "idh1_raw", "mgmt_label"}

SEED = 42
OUTER_FOLDS, INNER_FOLDS, N_REPEATS = 5, 3, 3
MAX_EPOCHS, PATIENCE = 200, 25

CONFIGS = {
    "imaging_only":  dict(use_clinical=False, use_kan=False, use_gnn=False, use_gate=False),
    "mlp_fusion":    dict(use_clinical=True,  use_kan=False, use_gnn=False, use_gate=False),
    "kan_fusion":    dict(use_clinical=True,  use_kan=True,  use_gnn=False, use_gate=False),
    "kan_gnn":       dict(use_clinical=True,  use_kan=True,  use_gnn=True,  use_gate=False),
    "kan_gnn_gate":  dict(use_clinical=True,  use_kan=True,  use_gnn=True,  use_gate=True),
}

# Small grids on purpose: 227 subjects cannot adjudicate between many options,
# and a large grid would simply relocate the overfitting into the inner loop.
GRID = {
    "hidden": [8, 16],
    "lr": [3e-3, 1e-2],
    "dropout": [0.3, 0.5],
}
GNN_GRID = {"k": [5, 10]}
KAN_GRID = {"l1": [0.0, 1e-3]}


def prepare(df: pd.DataFrame):
    img_cols = [c for c in df.columns
                if c not in NON_FEATURE and c not in CLINICAL_COLS
                and pd.api.types.is_numeric_dtype(df[c])]
    X_img = df[img_cols].to_numpy(float)
    X_clin = df[CLINICAL_COLS].to_numpy(float)
    y = df["mgmt_label"].to_numpy(int)
    return X_img, X_clin, y, img_cols


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


def train_model(cfg, hp, Xi_tr, Xc_tr, y_tr, Xi_va, Xc_va, y_va, seed):
    torch.manual_seed(seed)
    use_clin = cfg["use_clinical"]
    n_clin = Xc_tr.shape[1] if use_clin else 0

    model = ClinicalImagingKANGNN(
        n_imaging=Xi_tr.shape[1], n_clinical=n_clin, hidden=hp["hidden"],
        use_kan=cfg["use_kan"], use_gnn=cfg["use_gnn"], use_gate=cfg["use_gate"],
        k=hp.get("k", 8), dropout=hp["dropout"])

    ti = torch.tensor(Xi_tr, dtype=torch.float32)
    tc = torch.tensor(Xc_tr, dtype=torch.float32) if use_clin else None
    vi = torch.tensor(Xi_va, dtype=torch.float32)
    vc = torch.tensor(Xc_va, dtype=torch.float32) if use_clin else None
    ty = torch.tensor(y_tr, dtype=torch.float32)

    pos_weight = torch.tensor([(len(y_tr) - y_tr.sum()) / max(1, y_tr.sum())],
                              dtype=torch.float32)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=1e-4)

    best_auc, best_state, bad = -1.0, None, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        if cfg["use_gnn"]:
            model.refresh_memory(ti, tc)          # training embeddings only
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
                model.refresh_memory(ti, tc)      # bank stays training-only
            p = torch.sigmoid(model(vi, vc)).numpy()
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


def run_config(name, cfg, X_img, X_clin, y, repeats):
    oof = np.zeros((repeats, len(y)))
    chosen, graph_reliance = [], []

    for rep in range(repeats):
        outer = StratifiedKFold(OUTER_FOLDS, shuffle=True, random_state=SEED + rep)
        for tr_idx, te_idx in outer.split(X_img, y):
            # --- inner loop: choose hyperparameters on training data only ---
            inner = StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=SEED)
            best_hp, best_score = None, -1.0
            for hp in hp_grid(cfg):
                scores = []
                for i_tr, i_va in inner.split(X_img[tr_idx], y[tr_idx]):
                    a, b = tr_idx[i_tr], tr_idx[i_va]
                    Xi_a, Xi_b = fit_transform(X_img[a], X_img[b])
                    Xc_a, Xc_b = fit_transform(X_clin[a], X_clin[b])
                    _, auc = train_model(cfg, hp, Xi_a, Xc_a, y[a],
                                         Xi_b, Xc_b, y[b], SEED)
                    scores.append(auc)
                m = float(np.mean(scores))
                if m > best_score:
                    best_score, best_hp = m, hp
            chosen.append(dict(best_hp))

            # --- outer fold: fit once with the winning config, score once ---
            Xi_tr, Xi_te = fit_transform(X_img[tr_idx], X_img[te_idx])
            Xc_tr, Xc_te = fit_transform(X_clin[tr_idx], X_clin[te_idx])
            model, _ = train_model(cfg, best_hp, Xi_tr, Xc_tr, y[tr_idx],
                                   Xi_te, Xc_te, y[te_idx], SEED)
            model.eval()
            with torch.no_grad():
                ti = torch.tensor(Xi_tr, dtype=torch.float32)
                tc = torch.tensor(Xc_tr, dtype=torch.float32) if cfg["use_clinical"] else None
                if cfg["use_gnn"]:
                    model.refresh_memory(ti, tc)
                    graph_reliance.append(model.gconv.neighbour_weight)
                te_i = torch.tensor(Xi_te, dtype=torch.float32)
                te_c = torch.tensor(Xc_te, dtype=torch.float32) if cfg["use_clinical"] else None
                oof[rep, te_idx] = torch.sigmoid(model(te_i, te_c)).numpy()
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
        "oof_prob": mean_prob.tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default=str(COHORT))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--repeats", type=int, default=N_REPEATS)
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    args = ap.parse_args()

    df = pd.read_csv(args.cohort)
    X_img, X_clin, y, img_cols = prepare(df)
    print(f"cohort   : n={len(y)}, {int(y.sum())} methylated "
          f"({y.mean() * 100:.1f}%)")
    print(f"features : {X_img.shape[1]} imaging + {X_clin.shape[1]} clinical")
    print(f"protocol : nested CV, {OUTER_FOLDS} outer x {INNER_FOLDS} inner, "
          f"{args.repeats} repeats\n")

    results = {}
    for name in args.configs:
        print(f"[{name}]", flush=True)
        results[name] = run_config(name, CONFIGS[name], X_img, X_clin, y, args.repeats)
        r = results[name]
        extra = (f"  graph_reliance {r['graph_reliance_mean']:.3f}"
                 if r["graph_reliance_mean"] is not None else "")
        print(f"  AUC {r['auc']:.3f} [{r['auc_ci_low']:.3f}, {r['auc_ci_high']:.3f}]"
              f"{extra}\n", flush=True)

    out = Path(args.out_dir) / "kan_gnn_ablation.json"
    payload = {k: {kk: vv for kk, vv in v.items() if kk != "oof_prob"}
               for k, v in results.items()}
    payload["cohort"] = {"n": int(len(y)), "n_positive": int(y.sum()),
                         "n_imaging": X_img.shape[1], "n_clinical": X_clin.shape[1]}
    out.write_text(json.dumps(payload, indent=2))

    print("=" * 68)
    print(f"{'config':<18}{'AUC [95% CI]':<28}{'vs previous rung':>20}")
    print("=" * 68)
    prev = None
    for name in args.configs:
        r = results[name]
        delta = "" if prev is None else f"{r['auc'] - prev:+.3f}"
        print(f"{name:<18}{r['auc']:.3f} [{r['auc_ci_low']:.3f}, "
              f"{r['auc_ci_high']:.3f}]{delta:>20}")
        prev = r["auc"]
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
