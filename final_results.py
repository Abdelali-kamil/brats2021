# final_results_only.py
# One script, no output CSV files, prints final UPENN classification + segmentation results.
# Run: python final_results_only.py

import glob
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)

def safe_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])  # always 2x2
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else np.nan

def safe_auc(y_true, y_prob):
    return roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan

def print_line():
    print("-" * 90)

# Auto-discover files (no hardcoded missing filenames)
cls_files = sorted(glob.glob("upenn_classification_results*.csv"))
seg_files = sorted(glob.glob("upenn_segmentation_results*.csv"))
thr_files = sorted(glob.glob("upenn_test_results*.csv"))

print("\nUPENN FINAL RESULTS (NO FILE SAVING)")
print_line()
print("Classification files found:", cls_files if cls_files else "None")
print("Segmentation files found  :", seg_files if seg_files else "None")
print("Threshold summary files   :", thr_files if thr_files else "None")
print_line()

# =========================
# CLASSIFICATION
# =========================
if cls_files:
    cls_rows = []
    for f in cls_files:
        df = pd.read_csv(f)

        # Required columns check
        req = {"cls_label", "pred_label", "pred_prob"}
        if not req.issubset(df.columns):
            print(f"[SKIP] {f} missing columns. Found: {list(df.columns)}")
            continue

        y = df["cls_label"].astype(int).values
        yp = df["pred_label"].astype(int).values
        pr = df["pred_prob"].astype(float).values

        row = {
            "file": f,
            "n": len(df),
            "n_class0": int((y == 0).sum()),
            "n_class1": int((y == 1).sum()),
            "accuracy": accuracy_score(y, yp),
            "precision": precision_score(y, yp, zero_division=0),
            "recall_sensitivity": recall_score(y, yp, zero_division=0),
            "specificity": safe_specificity(y, yp),
            "f1": f1_score(y, yp, zero_division=0),
            "auc": safe_auc(y, pr),
        }
        cls_rows.append(row)

    if cls_rows:
        cls_df = pd.DataFrame(cls_rows).sort_values(
            by=["auc", "recall_sensitivity", "accuracy"], ascending=False, na_position="last"
        )

        print("\nCLASSIFICATION (UPENN)")
        print_line()
        print(cls_df.to_string(index=False))

        # Best row selection
        best_cls = cls_df.iloc[0]
        print_line()
        print("BEST CLASSIFICATION FILE:", best_cls["file"])
        print(
            f"AUC={best_cls['auc']:.6f} | ACC={best_cls['accuracy']:.6f} | "
            f"PRE={best_cls['precision']:.6f} | REC={best_cls['recall_sensitivity']:.6f} | "
            f"SPEC={best_cls['specificity']:.6f} | F1={best_cls['f1']:.6f}"
        )
        if np.isnan(best_cls["auc"]):
            print("NOTE: AUC is NaN because only one class exists in y_true for this file.")
    else:
        print("\nNo valid classification files after column check.")
else:
    print("\nNo UPENN classification files found.")

# =========================
# SEGMENTATION
# =========================
if seg_files:
    seg_rows = []
    for f in seg_files:
        df = pd.read_csv(f)

        req = {"Dice_ET", "Dice_TC", "Dice_WT", "HD95_ET", "HD95_TC", "HD95_WT"}
        if not req.issubset(df.columns):
            print(f"[SKIP] {f} missing columns. Found: {list(df.columns)}")
            continue

        row = {
            "file": f,
            "n": len(df),
            "failed_rate": df["Failed"].mean() if "Failed" in df.columns else np.nan,

            "Dice_ET_mean": df["Dice_ET"].mean(skipna=True),
            "Dice_TC_mean": df["Dice_TC"].mean(skipna=True),
            "Dice_WT_mean": df["Dice_WT"].mean(skipna=True),
            "Dice_macro_mean": df[["Dice_ET", "Dice_TC", "Dice_WT"]].mean(axis=1, skipna=True).mean(),

            "HD95_ET_mean": df["HD95_ET"].mean(skipna=True),
            "HD95_TC_mean": df["HD95_TC"].mean(skipna=True),
            "HD95_WT_mean": df["HD95_WT"].mean(skipna=True),
            "HD95_macro_mean": df[["HD95_ET", "HD95_TC", "HD95_WT"]].mean(axis=1, skipna=True).mean(),
        }
        seg_rows.append(row)

    if seg_rows:
        # Sort by Dice_macro high, then HD95_macro low
        seg_df = pd.DataFrame(seg_rows).sort_values(
            by=["Dice_macro_mean", "HD95_macro_mean"], ascending=[False, True], na_position="last"
        )

        print("\nSEGMENTATION (UPENN)")
        print_line()
        print(seg_df.to_string(index=False))

        best_seg = seg_df.iloc[0]
        print_line()
        print("BEST SEGMENTATION FILE:", best_seg["file"])
        print(
            f"Dice_ET={best_seg['Dice_ET_mean']:.6f} | Dice_TC={best_seg['Dice_TC_mean']:.6f} | "
            f"Dice_WT={best_seg['Dice_WT_mean']:.6f} | Dice_mean={best_seg['Dice_macro_mean']:.6f}"
        )
        print(
            f"HD95_ET={best_seg['HD95_ET_mean']:.6f} | HD95_TC={best_seg['HD95_TC_mean']:.6f} | "
            f"HD95_WT={best_seg['HD95_WT_mean']:.6f} | HD95_mean={best_seg['HD95_macro_mean']:.6f}"
        )
        if not np.isnan(best_seg["failed_rate"]):
            print(f"Failed rate={best_seg['failed_rate']:.6f}")
    else:
        print("\nNo valid segmentation files after column check.")
else:
    print("\nNo UPENN segmentation files found.")

# =========================
# OPTIONAL: threshold summary
# =========================
if thr_files:
    print("\nUPENN TEST THRESHOLD SUMMARY")
    print_line()
    for f in thr_files:
        df = pd.read_csv(f)
        print(f"\n{f}")
        print(df.to_string(index=False))