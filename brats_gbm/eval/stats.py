"""Bootstrap confidence intervals over cases.

Every reported metric resamples patients (not voxels), because the patient is
the independent unit. Percentile intervals from 2000 resamples with a fixed
seed, so a rerun reproduces the published numbers exactly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

N_BOOT = 2000
SEED = 42


def bootstrap_ci(
    values,
    n_boot: int = N_BOOT,
    seed: int = SEED,
    statistic=np.mean,
) -> tuple[float, float, float]:
    """Return (point_estimate, ci_low, ci_high) at the 95% level.

    NaNs are dropped first, which matters for HD95 where undefined cases are
    legitimately NaN rather than zero.
    """
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    if x.size == 1:
        return float(x[0]), float(x[0]), float(x[0])

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    boot = statistic(x[idx], axis=1)
    return (
        float(statistic(x)),
        float(np.percentile(boot, 2.5)),
        float(np.percentile(boot, 97.5)),
    )


def paired_diff_ci(
    a,
    b,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> tuple[float, float, float]:
    """CI on mean(a) - mean(b) with cases resampled jointly.

    Use this for before/after fine-tuning on the same patients: the paired
    resample keeps each patient's two scores together, so the interval
    reflects the within-patient improvement rather than two independent means.
    """
    df = pd.DataFrame({"a": pd.to_numeric(pd.Series(a), errors="coerce"),
                       "b": pd.to_numeric(pd.Series(b), errors="coerce")}).dropna()
    if df.empty:
        return float("nan"), float("nan"), float("nan")

    x = df["a"].to_numpy(float)
    y = df["b"].to_numpy(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    boot = x[idx].mean(axis=1) - y[idx].mean(axis=1)
    return (
        float(x.mean() - y.mean()),
        float(np.percentile(boot, 2.5)),
        float(np.percentile(boot, 97.5)),
    )


def fmt_ci(point: float, lo: float, hi: float, nd: int = 4) -> str:
    if np.isnan(point):
        return "n/a"
    return f"{point:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def summarise_segmentation(df: pd.DataFrame, label: str = "") -> pd.DataFrame:
    """Per-region Dice/HD95 with bootstrap CIs for a per-case results frame."""
    rows = []
    for region in ("ET", "TC", "WT"):
        d = bootstrap_ci(df[f"Dice_{region}"])
        h = bootstrap_ci(df[f"HD95_{region}"])
        rows.append({
            "label": label,
            "region": region,
            "n": int(pd.to_numeric(df[f"Dice_{region}"], errors="coerce").notna().sum()),
            "dice": d[0], "dice_lo": d[1], "dice_hi": d[2],
            "hd95": h[0], "hd95_lo": h[1], "hd95_hi": h[2],
            "n_zero_dice": int((pd.to_numeric(df[f"Dice_{region}"], errors="coerce") == 0).sum()),
        })

    mean_dice = df[["Dice_ET", "Dice_TC", "Dice_WT"]].mean(axis=1)
    md = bootstrap_ci(mean_dice)
    # Mean HD95 averages the three regions per case, skipping undefined ones.
    mean_hd95 = df[["HD95_ET", "HD95_TC", "HD95_WT"]].apply(
        pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
    mh = bootstrap_ci(mean_hd95)
    rows.append({
        "label": label, "region": "MEAN", "n": len(df),
        "dice": md[0], "dice_lo": md[1], "dice_hi": md[2],
        "hd95": mh[0], "hd95_lo": mh[1], "hd95_hi": mh[2],
        "n_zero_dice": "",
    })
    return pd.DataFrame(rows)


def print_summary(summary: pd.DataFrame, title: str) -> None:
    print(f"\n{title}")
    print("-" * 76)
    print(f"{'Region':<7} | {'n':>4} | {'Dice [95% CI]':<28} | {'HD95 mm [95% CI]':<28}")
    print("-" * 76)
    for _, r in summary.iterrows():
        print(
            f"{r['region']:<7} | {r['n']:>4} | "
            f"{fmt_ci(r['dice'], r['dice_lo'], r['dice_hi']):<28} | "
            f"{fmt_ci(r['hd95'], r['hd95_lo'], r['hd95_hi']):<28}"
        )
