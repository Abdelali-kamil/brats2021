#!/usr/bin/env python3
"""Assemble ONE comparison table (segmentation + classification) for the paper.

Reads whatever real result files exist under results/baselines/ (plus this
project's own committed segmentation summary for the "Ours" row) and writes a
single table in the paper's Table-II layout:

    Method | Modal | Dice ↑ | HD95 ↓ | Recall ↑ | Precision ↑ | AUC ↑ | Status

Cells with no result file yet are printed as "pending" — they are produced by
running each baseline on your server (see baselines/<method>/SETUP.md). Nothing
here invents a number: a cell is filled only from a result file on disk.

Expected result files (each baseline writes one after it runs on your data):

  results/baselines/<key>_segmentation.json
      {"dice_mean": 0.87, "dice_sd": 0.018, "hd95_mean": 6.8, "hd95_sd": 0.4}
      (dice in [0,1]; hd95 in mm; means over the same test cases as "Ours")

  results/baselines/<key>_classification.json     # flat schema
      {"recall_mean":0.62,"recall_sd":0.16,"precision_mean":0.80,
       "precision_sd":0.17,"auc_mean":0.61,"auc_sd":0.06}

XGBoost is read directly from results/baselines/xgboost_classification.json,
which scripts/baseline_xgboost_classification.py already produced here.

Usage:
    python baselines/common/make_comparison_table.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results" / "baselines"
OURS_SEG_CSV = ROOT / "results" / "brats" / "summary_internal_validation_recomputed.csv"

# Roster in the paper's order. ResGANet/MMCL (no official code) are replaced by
# runnable methods: Swin UNETR (segmentation) and DAFT (image+tabular fusion).
ROSTER = [
    # key, display, modality, group, does_seg, does_cls, note
    ("xgboost",    "XGBoost",                "T",   "single", False, True,  ""),
    ("mmgl",       "MMGL",                   "I+T", "single", False, True,  ""),
    ("daft",       "DAFT",                   "I+T", "single", False, True,  "replaces MMCL (no official code)"),
    ("nnunet",     "nnU-Net",                "I",   "single", True,  False, ""),
    ("selfmedmae", "SelfMedMAE",             "I+T", "multi",  True,  True,  ""),
    ("swin_unetr", "Swin UNETR",             "I",   "multi",  True,  False, "replaces ResGANet (no official code)"),
    ("mtanet",     "MTANet",                 "I+T", "multi",  True,  True,  ""),
    ("ours",       "Wavelet U-Net++ (Ours)", "I",   "ours",   True,  False, ""),
]

GROUP_TITLES = {
    "single": "Single task",
    "multi":  "Multitasking",
    "ours":   "Ours",
}


def pct(mean, sd=None):
    if mean is None:
        return "pending"
    return f"{mean * 100:.1f}±{sd * 100:.1f}" if sd is not None else f"{mean * 100:.1f}"


def mm(mean, sd=None):
    if mean is None:
        return "pending"
    return f"{mean:.2f}±{sd:.2f}" if sd is not None else f"{mean:.2f}"


def load_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def ours_seg():
    """Mean Dice/HD95 over ET/TC/WT from the committed BraTS summary."""
    if not OURS_SEG_CSV.exists():
        return {}
    d = pd.read_csv(OURS_SEG_CSV)
    return {
        "dice_mean": float(d["dice"].mean()), "dice_sd": float(d["dice"].std()),
        "hd95_mean": float(d["hd95"].mean()), "hd95_sd": float(d["hd95"].std()),
        "status": "reproduced (this project)",
    }


def xgboost_cls():
    """Recall/Precision/AUC for XGBoost from its result file (MGMT = the table's cohort)."""
    j = load_json(BASE / "xgboost_classification.json")
    if not j:
        return {}
    prim = j.get("tasks", {}).get("mgmt", {}).get("primary")
    if not prim:
        return {}
    return {
        "recall_mean": prim["sensitivity"]["mean"], "recall_sd": prim["sensitivity"]["across_repeat_sd"],
        "precision_mean": prim["ppv"]["mean"], "precision_sd": prim["ppv"]["across_repeat_sd"],
        "auc_mean": prim["auc_patient_bootstrap"]["mean"], "auc_sd": prim["auc"]["across_repeat_sd"],
        "status": "reproduced (MGMT, this data)",
    }


def seg_cell(key):
    if key == "ours":
        return ours_seg()
    return load_json(BASE / f"{key}_segmentation.json") or {}


def cls_cell(key):
    if key == "xgboost":
        return xgboost_cls()
    return load_json(BASE / f"{key}_classification.json") or {}


def build_rows():
    rows = []
    for key, name, modal, group, does_seg, does_cls, note in ROSTER:
        seg = seg_cell(key) if does_seg else {}
        cls = cls_cell(key) if does_cls else {}
        statuses = [d.get("status") for d in (seg, cls) if d.get("status")]
        if not does_seg and not does_cls:
            status = "—"
        elif not statuses:
            status = "pending (run on server)"
        else:
            status = "; ".join(dict.fromkeys(statuses))
        rows.append({
            "key": key, "method": name, "modal": modal, "group": group, "note": note,
            "dice": pct(seg.get("dice_mean"), seg.get("dice_sd")) if does_seg else "–",
            "hd95": mm(seg.get("hd95_mean"), seg.get("hd95_sd")) if does_seg else "–",
            "recall": pct(cls.get("recall_mean"), cls.get("recall_sd")) if does_cls else "–",
            "precision": pct(cls.get("precision_mean"), cls.get("precision_sd")) if does_cls else "–",
            "auc": pct(cls.get("auc_mean"), cls.get("auc_sd")) if does_cls else "–",
            "status": status,
        })
    return rows


def write_markdown(rows):
    out = BASE / "comparison.md"
    L = ["# Baseline comparison — segmentation and classification",
         "",
         "One table, real reproduced numbers only. `pending` = no result file yet;",
         "produce it by running that baseline on your server (see "
         "`baselines/<method>/SETUP.md`), which writes the result file this script reads.",
         "",
         "Metrics: Dice ↑ and Recall/Precision/AUC ↑ as mean±SD in %, HD95 ↓ in mm.",
         "Recall = sensitivity, Precision = PPV. Classification cohort is BraTS **MGMT**.",
         "",
         "| Method | Modal | Dice ↑ | HD95 ↓ | Recall ↑ | Precision ↑ | AUC ↑ | Status |",
         "|---|:--:|:--:|:--:|:--:|:--:|:--:|---|"]
    last_group = None
    for r in rows:
        if r["group"] != last_group:
            L.append(f"| **_{GROUP_TITLES[r['group']]}_** | | | | | | | |")
            last_group = r["group"]
        name = r["method"] + (f" ¹" if r["note"] else "")
        L.append(f"| {name} | {r['modal']} | {r['dice']} | {r['hd95']} | "
                 f"{r['recall']} | {r['precision']} | {r['auc']} | {r['status']} |")
    notes = [f"¹ {r['method']}: {r['note']}" for r in rows if r["note"]]
    if notes:
        L += ["", *notes]
    L += ["",
          "> **Reproducibility caveat (MGMT).** On this project's committed radiomic",
          "> features, under its own repeated-CV protocol, XGBoost reaches AUC ≈ 0.57 on",
          "> MGMT — near chance, consistent with `docs/RESUME.md` and the",
          "> `train_classifier_mgmt.py` docstring. Classification AUCs in the 0.80–0.89",
          "> range reported by the source papers were obtained on *their own* datasets and",
          "> are not reproduced here; do not present them as run on this cohort."]
    out.write_text("\n".join(L) + "\n")
    return out


def write_csv(rows):
    out = BASE / "comparison.csv"
    cols = ["group", "method", "modal", "dice", "hd95", "recall", "precision", "auc", "status", "note"]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return out


def main():
    rows = build_rows()
    md = write_markdown(rows)
    cv = write_csv(rows)
    print("wrote:")
    print(" ", md)
    print(" ", cv)
    print()
    print(Path(md).read_text())


if __name__ == "__main__":
    main()
