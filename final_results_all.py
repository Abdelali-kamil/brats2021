#!/usr/bin/env python3

import os
import argparse
import numpy as np
import pandas as pd


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def mean_safe(series):
    x = to_num(series).dropna().values
    if len(x) == 0:
        return np.nan
    return float(np.mean(x))


def overall_from_dice(df):
    et = mean_safe(df["Dice_ET"])
    tc = mean_safe(df["Dice_TC"])
    wt = mean_safe(df["Dice_WT"])
    return float(np.nanmean([et, tc, wt]))


def fmt(v, nd=6):
    if pd.isna(v):
        return "NaN"
    return f"{float(v):.{nd}f}"


def require_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")


def add_rank_column(df):
    if "overall_dice_mean" not in df.columns:
        raise ValueError("comparison CSV must contain column: overall_dice_mean")
    out = df.copy()
    out["overall_dice_mean"] = to_num(out["overall_dice_mean"])
    out = out.sort_values("overall_dice_mean", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def choose_best_setup(comparison_df):
    ranked = add_rank_column(comparison_df)
    best_setup = str(ranked.loc[0, "setup"])
    return best_setup, ranked


def best_seg_csv_from_setup(setup_name):
    # expected names in your project:
    # upenn_v2_last.pth -> v2_segmentation_upenn_v2_last.csv
    # ens_all_three      -> v2_segmentation_ens_all_three.csv
    base = setup_name.replace(".pth", "")
    return f"v2_segmentation_{base}.csv"


def summarize_best(seg_df):
    summary = {}
    summary["Dice_ET"] = mean_safe(seg_df["Dice_ET"]) if "Dice_ET" in seg_df.columns else np.nan
    summary["Dice_TC"] = mean_safe(seg_df["Dice_TC"]) if "Dice_TC" in seg_df.columns else np.nan
    summary["Dice_WT"] = mean_safe(seg_df["Dice_WT"]) if "Dice_WT" in seg_df.columns else np.nan
    summary["Overall_Dice"] = float(np.nanmean([summary["Dice_ET"], summary["Dice_TC"], summary["Dice_WT"]]))

    summary["HD95_ET"] = mean_safe(seg_df["HD95_ET"]) if "HD95_ET" in seg_df.columns else np.nan
    summary["HD95_TC"] = mean_safe(seg_df["HD95_TC"]) if "HD95_TC" in seg_df.columns else np.nan
    summary["HD95_WT"] = mean_safe(seg_df["HD95_WT"]) if "HD95_WT" in seg_df.columns else np.nan

    summary["ZeroDice_ET"] = int((to_num(seg_df["Dice_ET"]) == 0).sum()) if "Dice_ET" in seg_df.columns else np.nan
    summary["ZeroDice_TC"] = int((to_num(seg_df["Dice_TC"]) == 0).sum()) if "Dice_TC" in seg_df.columns else np.nan
    summary["ZeroDice_WT"] = int((to_num(seg_df["Dice_WT"]) == 0).sum()) if "Dice_WT" in seg_df.columns else np.nan

    if "Failed" in seg_df.columns:
        summary["Failed_Cases"] = int(to_num(seg_df["Failed"]).fillna(0).sum())
    else:
        summary["Failed_Cases"] = 0

    return summary


def build_text_report(ranked_df, best_setup, best_seg_csv, best_summary):
    lines = []
    lines.append("FINAL RESULTS REPORT")
    lines.append("")

    lines.append("A) ALL SETUPS RANKED")
    cols = ["rank", "setup", "et_thr", "tc_thr", "wt_thr", "overall_dice_mean"]
    show_cols = [c for c in cols if c in ranked_df.columns]
    table = ranked_df[show_cols].copy()
    if "overall_dice_mean" in table.columns:
        table["overall_dice_mean"] = to_num(table["overall_dice_mean"]).map(lambda x: np.nan if pd.isna(x) else round(float(x), 6))
    lines.append(table.to_string(index=False))
    lines.append("")

    lines.append("B) BEST SETUP")
    lines.append(f"Best setup: {best_setup}")
    lines.append(f"Best per-case CSV: {best_seg_csv}")
    lines.append("")

    lines.append("C) BEST SETUP METRICS")
    key_order = [
        "Dice_ET", "Dice_TC", "Dice_WT", "Overall_Dice",
        "HD95_ET", "HD95_TC", "HD95_WT",
        "ZeroDice_ET", "ZeroDice_TC", "ZeroDice_WT", "Failed_Cases"
    ]
    for k in key_order:
        v = best_summary.get(k, np.nan)
        if isinstance(v, (int, np.integer)):
            lines.append(f"{k}: {v}")
        else:
            lines.append(f"{k}: {fmt(v)}")
    lines.append("")

    # explicit NaN proof line for ET HD95
    lines.append("D) HD95_ET INTERPRETATION")
    if pd.isna(best_summary.get("HD95_ET", np.nan)):
        lines.append("HD95_ET is NaN because no valid numeric HD95_ET values were available in the best per-case CSV.")
    else:
        lines.append("HD95_ET is numeric and computed as mean(skipna=True).")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Create final results report for all setups.")
    parser.add_argument("--comparison_csv", type=str, default="v2_checkpoint_comparison.csv")
    parser.add_argument("--best_seg_csv", type=str, default="", help="Optional: force best seg CSV path")
    parser.add_argument("--out_csv", type=str, default="final_all_results_ranked.csv")
    parser.add_argument("--out_txt", type=str, default="final_results_report.txt")
    args = parser.parse_args()

    require_file(args.comparison_csv)
    comparison_df = pd.read_csv(args.comparison_csv)

    best_setup, ranked_df = choose_best_setup(comparison_df)

    if args.best_seg_csv.strip():
        best_seg_csv = args.best_seg_csv.strip()
    else:
        best_seg_csv = best_seg_csv_from_setup(best_setup)

    require_file(best_seg_csv)
    best_seg_df = pd.read_csv(best_seg_csv)

    # If comparison "overall_dice_mean" differs from recomputed best overall, keep both visible
    recomputed_overall = overall_from_dice(best_seg_df)

    best_summary = summarize_best(best_seg_df)

    # Save ranked csv
    ranked_df.to_csv(args.out_csv, index=False)

    # Build report
    report_txt = build_text_report(ranked_df, best_setup, best_seg_csv, best_summary)
    report_txt += f"\nRecomputed best overall dice from per-case CSV: {fmt(recomputed_overall)}\n"

    with open(args.out_txt, "w") as f:
        f.write(report_txt)

    # Print to terminal
    print(report_txt)
    print(f"Saved: {args.out_csv}")
    print(f"Saved: {args.out_txt}")
    print("Done ✅")


if __name__ == "__main__":
    main()