#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze Dice distribution from final segmentation evaluation CSV.

Default input:
    FINAL_BINARY_FINE_TUNED_128_T02.csv

This script prints and saves:
- Full-dataset Dice statistics
- Number of cases above Dice thresholds
- Filtered/subset analysis
- Best cases
- Worst cases
- Safe report wording

Important:
- Overall Mean Dice is the real full-dataset result.
- High Dice values such as >0.90 can only be reported as best-case or subset performance.
- Filtered statistics must be clearly labeled and not reported as full test performance.

This version does NOT require openpyxl.
It saves CSV and TXT files only.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT_CSV = "FINAL_BINARY_FINE_TUNED_128_T02.csv"
DEFAULT_OUTPUT_SUMMARY_CSV = "DICE_DISTRIBUTION_SUMMARY.csv"
DEFAULT_OUTPUT_COUNTS_CSV = "DICE_THRESHOLD_COUNTS.csv"
DEFAULT_OUTPUT_FILTERED_CSV = "DICE_FILTERED_SUBSET_ANALYSIS.csv"
DEFAULT_OUTPUT_BEST_CASES_CSV = "BEST_DICE_CASES.csv"
DEFAULT_OUTPUT_WORST_CASES_CSV = "WORST_DICE_CASES.csv"
DEFAULT_OUTPUT_REPORT_TXT = "DICE_REPORT_WORDING.txt"


def safe_numeric(df, col):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def print_header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def load_results(csv_path):
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = ["Patient_ID", "Dice"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column missing from CSV: {col}")

    numeric_cols = [
        "Dice",
        "HD95",
        "Sensitivity",
        "Precision",
        "GT_Voxels",
        "Pred_Voxels",
        "Overlap_Voxels",
        "Prob_Max",
        "Prob_Mean",
        "Prob_Min",
        "Prob_Std",
    ]

    for col in numeric_cols:
        df = safe_numeric(df, col)

    return df


def summarize_basic(df):
    dice = df["Dice"]

    summary = {
        "Total cases": len(df),
        "Valid Dice cases": int(dice.notna().sum()),
        "Mean Dice": float(dice.mean()),
        "Median Dice": float(dice.median()),
        "STD Dice": float(dice.std()),
        "Min Dice": float(dice.min()),
        "Max Dice": float(dice.max()),
        "Q10 Dice": float(dice.quantile(0.10)),
        "Q25 Dice": float(dice.quantile(0.25)),
        "Q50 Dice": float(dice.quantile(0.50)),
        "Q75 Dice": float(dice.quantile(0.75)),
        "Q90 Dice": float(dice.quantile(0.90)),
        "Q95 Dice": float(dice.quantile(0.95)),
    }

    if "HD95" in df.columns:
        summary["Mean HD95"] = float(df["HD95"].mean())
        summary["Median HD95"] = float(df["HD95"].median())

    if "Sensitivity" in df.columns:
        summary["Mean Sensitivity"] = float(df["Sensitivity"].mean())

    if "Precision" in df.columns:
        summary["Mean Precision"] = float(df["Precision"].mean())

    if "Pred_Voxels" in df.columns:
        summary["Empty predictions"] = int((df["Pred_Voxels"] == 0).sum())

    if "GT_Voxels" in df.columns:
        summary["Empty GT masks"] = int((df["GT_Voxels"] == 0).sum())

    return summary


def count_thresholds(df):
    dice = df["Dice"]
    total = len(df)

    thresholds = [
        0.90,
        0.85,
        0.80,
        0.75,
        0.70,
        0.65,
        0.60,
        0.50,
        0.40,
        0.30,
        0.20,
        0.10,
    ]

    rows = []

    for t in thresholds:
        count = int((dice >= t).sum())
        percent = 100.0 * count / total if total > 0 else np.nan

        rows.append(
            {
                "Condition": f"Dice >= {t:.2f}",
                "Cases": count,
                "Percent": percent,
            }
        )

    zero_count = int((dice == 0).sum())
    zero_percent = 100.0 * zero_count / total if total > 0 else np.nan

    rows.append(
        {
            "Condition": "Dice == 0.00",
            "Cases": zero_count,
            "Percent": zero_percent,
        }
    )

    low_count = int((dice < 0.10).sum())
    low_percent = 100.0 * low_count / total if total > 0 else np.nan

    rows.append(
        {
            "Condition": "Dice < 0.10",
            "Cases": low_count,
            "Percent": low_percent,
        }
    )

    return pd.DataFrame(rows)


def summarize_filtered(df):
    """
    These are NOT full-dataset results.
    They are only for understanding performance distribution.
    """

    filters = {
        "All cases": df,
        "Dice > 0": df[df["Dice"] > 0],
        "Dice >= 0.10": df[df["Dice"] >= 0.10],
        "Dice >= 0.20": df[df["Dice"] >= 0.20],
        "Dice >= 0.30": df[df["Dice"] >= 0.30],
        "Dice >= 0.50": df[df["Dice"] >= 0.50],
        "Dice >= 0.70": df[df["Dice"] >= 0.70],
        "Dice >= 0.80": df[df["Dice"] >= 0.80],
        "Dice >= 0.90": df[df["Dice"] >= 0.90],
    }

    rows = []

    for name, sub in filters.items():
        if len(sub) == 0:
            rows.append(
                {
                    "Subset": name,
                    "Cases": 0,
                    "Mean Dice": np.nan,
                    "Median Dice": np.nan,
                    "Mean HD95": np.nan,
                    "Median HD95": np.nan,
                    "Mean Sensitivity": np.nan,
                    "Mean Precision": np.nan,
                }
            )
            continue

        row = {
            "Subset": name,
            "Cases": len(sub),
            "Mean Dice": sub["Dice"].mean(),
            "Median Dice": sub["Dice"].median(),
        }

        if "HD95" in sub.columns:
            row["Mean HD95"] = sub["HD95"].mean()
            row["Median HD95"] = sub["HD95"].median()
        else:
            row["Mean HD95"] = np.nan
            row["Median HD95"] = np.nan

        if "Sensitivity" in sub.columns:
            row["Mean Sensitivity"] = sub["Sensitivity"].mean()
        else:
            row["Mean Sensitivity"] = np.nan

        if "Precision" in sub.columns:
            row["Mean Precision"] = sub["Precision"].mean()
        else:
            row["Mean Precision"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def get_display_columns(df):
    preferred_cols = [
        "Patient_ID",
        "Dice",
        "HD95",
        "Sensitivity",
        "Precision",
        "GT_Voxels",
        "Pred_Voxels",
        "Overlap_Voxels",
        "Prob_Max",
    ]

    return [c for c in preferred_cols if c in df.columns]


def print_basic_summary(summary):
    print_header("FULL-DATASET RESULT")

    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key:<24}: {value:.4f}")
        else:
            print(f"{key:<24}: {value}")

    print("\nImportant:")
    print("  This full-dataset result is the main result you should report.")
    print("  Do not report filtered subset Dice as the full evaluation Dice.")


def print_threshold_counts(counts_df):
    print_header("DICE THRESHOLD COUNTS")

    for _, row in counts_df.iterrows():
        print(
            f"{row['Condition']:<16}: "
            f"{int(row['Cases']):>4} cases "
            f"({row['Percent']:>6.2f}%)"
        )


def print_filtered_summary(filtered_df):
    print_header("FILTERED / SUBSET ANALYSIS")

    print(
        filtered_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}" if pd.notna(x) else "nan",
        )
    )

    print("\nWarning:")
    print("  These filtered results are only for analysis.")
    print("  They must not be presented as the full test performance.")


def print_best_and_worst(df, top_k):
    display_cols = get_display_columns(df)

    print_header(f"BEST {top_k} CASES BY DICE")
    best = df.sort_values("Dice", ascending=False).head(top_k)

    print(
        best[display_cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}" if pd.notna(x) else "nan",
        )
    )

    print_header(f"WORST {top_k} CASES BY DICE")
    worst = df.sort_values("Dice", ascending=True).head(top_k)

    print(
        worst[display_cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}" if pd.notna(x) else "nan",
        )
    )

    return best, worst


def generate_report_text(summary, counts_df):
    max_dice = summary["Max Dice"]
    mean_dice = summary["Mean Dice"]
    median_dice = summary["Median Dice"]

    cases_09 = counts_df[counts_df["Condition"] == "Dice >= 0.90"]["Cases"].iloc[0]
    percent_09 = counts_df[counts_df["Condition"] == "Dice >= 0.90"]["Percent"].iloc[0]

    cases_08 = counts_df[counts_df["Condition"] == "Dice >= 0.80"]["Cases"].iloc[0]
    percent_08 = counts_df[counts_df["Condition"] == "Dice >= 0.80"]["Percent"].iloc[0]

    cases_07 = counts_df[counts_df["Condition"] == "Dice >= 0.70"]["Cases"].iloc[0]
    percent_07 = counts_df[counts_df["Condition"] == "Dice >= 0.70"]["Percent"].iloc[0]

    report_text = (
        f"The fine-tuned binary Wavelet U-Net++ model was evaluated on "
        f"{summary['Total cases']} pontine infarction cases. "
        f"The full-dataset mean Dice score was {mean_dice:.4f}, "
        f"with a median Dice score of {median_dice:.4f}. "
        f"The maximum Dice score was {max_dice:.4f}. "
        f"{int(cases_09)} cases ({percent_09:.2f}%) achieved Dice scores of at least 0.90, "
        f"{int(cases_08)} cases ({percent_08:.2f}%) achieved Dice scores of at least 0.80, "
        f"and {int(cases_07)} cases ({percent_07:.2f}%) achieved Dice scores of at least 0.70. "
        f"These results indicate that the model achieved high segmentation accuracy in a subset "
        f"of cases, although the overall mean performance was reduced by low-overlap or failed cases."
    )

    print_header("SAFE REPORT WORDING")

    print("You can write this:")
    print()
    print(report_text)

    print()
    print("Do NOT write this:")
    print()
    print("  The model achieved a mean Dice of 0.90.")
    print()
    print("Because your full-dataset mean Dice is not 0.90.")

    return report_text


def save_outputs(
    summary,
    counts_df,
    filtered_df,
    best,
    worst,
    report_text,
    summary_csv,
    counts_csv,
    filtered_csv,
    best_csv,
    worst_csv,
    report_txt,
):
    summary_rows = []

    for key, value in summary.items():
        summary_rows.append(
            {
                "Metric": key,
                "Value": value,
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    summary_df.to_csv(summary_csv, index=False)
    counts_df.to_csv(counts_csv, index=False)
    filtered_df.to_csv(filtered_csv, index=False)
    best.to_csv(best_csv, index=False)
    worst.to_csv(worst_csv, index=False)

    with open(report_txt, "w", encoding="utf-8") as f:
        f.write("SAFE REPORT WORDING\n")
        f.write("=" * 80 + "\n\n")
        f.write(report_text + "\n\n")
        f.write("IMPORTANT:\n")
        f.write("Do not report filtered subset Dice as full-dataset performance.\n")
        f.write("The full-dataset mean Dice is the main evaluation result.\n")

    print_header("FILES SAVED")
    print(f"Summary CSV        : {summary_csv}")
    print(f"Threshold CSV      : {counts_csv}")
    print(f"Filtered CSV       : {filtered_csv}")
    print(f"Best cases CSV     : {best_csv}")
    print(f"Worst cases CSV    : {worst_csv}")
    print(f"Report wording TXT : {report_txt}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_csv",
        type=str,
        default=DEFAULT_INPUT_CSV,
        help="Input evaluation CSV file.",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Number of best/worst cases to print.",
    )

    parser.add_argument(
        "--summary_csv",
        type=str,
        default=DEFAULT_OUTPUT_SUMMARY_CSV,
        help="Output summary CSV.",
    )

    parser.add_argument(
        "--counts_csv",
        type=str,
        default=DEFAULT_OUTPUT_COUNTS_CSV,
        help="Output Dice threshold counts CSV.",
    )

    parser.add_argument(
        "--filtered_csv",
        type=str,
        default=DEFAULT_OUTPUT_FILTERED_CSV,
        help="Output filtered subset analysis CSV.",
    )

    parser.add_argument(
        "--best_csv",
        type=str,
        default=DEFAULT_OUTPUT_BEST_CASES_CSV,
        help="Output best cases CSV.",
    )

    parser.add_argument(
        "--worst_csv",
        type=str,
        default=DEFAULT_OUTPUT_WORST_CASES_CSV,
        help="Output worst cases CSV.",
    )

    parser.add_argument(
        "--report_txt",
        type=str,
        default=DEFAULT_OUTPUT_REPORT_TXT,
        help="Output report wording TXT file.",
    )

    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Do not save output CSV/TXT files.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print_header("DICE DISTRIBUTION ANALYSIS")
    print(f"Input CSV: {args.input_csv}")

    df = load_results(args.input_csv)

    summary = summarize_basic(df)
    counts_df = count_thresholds(df)
    filtered_df = summarize_filtered(df)

    print_basic_summary(summary)
    print_threshold_counts(counts_df)
    print_filtered_summary(filtered_df)

    best, worst = print_best_and_worst(df, args.top_k)

    report_text = generate_report_text(summary, counts_df)

    if not args.no_save:
        save_outputs(
            summary=summary,
            counts_df=counts_df,
            filtered_df=filtered_df,
            best=best,
            worst=worst,
            report_text=report_text,
            summary_csv=args.summary_csv,
            counts_csv=args.counts_csv,
            filtered_csv=args.filtered_csv,
            best_csv=args.best_csv,
            worst_csv=args.worst_csv,
            report_txt=args.report_txt,
        )


if __name__ == "__main__":
    main()