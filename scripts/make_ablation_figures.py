#!/usr/bin/env python
"""Thesis figures for the KAN+GNN MGMT-methylation ablation.

Reads results/classification/kan_gnn_ablation.json and writes two figures to
results/figures/ as vector PDF (for the thesis) + 300-dpi PNG (for preview):

  fig1_ablation_auc_forest   AUC point estimate + 95% CI per configuration
  fig2_attribution           (a) controlled ΔAUC component effects vs a noise band
                             (b) graph reliance of the memory-bank (GNN) variants

Colors are the data-viz reference palette (colorblind-validated): first three
categorical slots for Fig 1 families, the blue<->red diverging pair for Fig 2.
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results", "classification", "kan_gnn_ablation.json")
FIGDIR = os.path.join(ROOT, "results", "figures")
os.makedirs(FIGDIR, exist_ok=True)

# ---- reference palette (validated; do not edit values) ----------------------
SURFACE   = "#fcfcfb"
INK       = "#0b0b0b"   # primary
INK2      = "#52514e"   # secondary
MUTED     = "#898781"   # axis / labels
GRID      = "#e1e0d9"
BASELINE  = "#c3c2b7"
BLUE      = "#2a78d6"   # cat slot 1 / diverging cool / positive
ORANGE    = "#eb6834"   # cat slot 2
AQUA      = "#1baf7a"   # cat slot 3
RED       = "#e34948"   # diverging warm / negative
BAND      = "#f0efec"   # diverging neutral midpoint (noise band)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 10,
    "text.color": INK,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK2,
    "xtick.color": MUTED,
    "ytick.color": INK,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.linewidth": 0.8,
    "svg.fonttype": "none",
})

with open(RESULTS) as f:
    D = json.load(f)
cohort = D["cohort"]

PRETTY = {
    "imaging_only":  "Imaging only",
    "clinical_only": "Clinical only",
    "mlp_fusion":    "MLP fusion",
    "kan_fusion":    "KAN fusion",
    "mlp_gnn":       "MLP + GNN",
    "kan_gnn":       "KAN + GNN",
    "kan_gnn_gate":  "KAN + GNN + gate",
}
FAMILY = {  # config -> (family label, color)
    "imaging_only":  ("Baseline / control", BLUE),
    "clinical_only": ("Baseline / control", BLUE),
    "mlp_fusion":    ("Fusion (no graph)",  ORANGE),
    "kan_fusion":    ("Fusion (no graph)",  ORANGE),
    "mlp_gnn":       ("Fusion + memory-bank GNN", AQUA),
    "kan_gnn":       ("Fusion + memory-bank GNN", AQUA),
    "kan_gnn_gate":  ("Fusion + memory-bank GNN", AQUA),
}

configs = [k for k in D if k != "cohort"]
# typical 95% CI half-width -> "noise band" reference used in Fig 2
half_widths = [(D[k]["auc_ci_high"] - D[k]["auc_ci_low"]) / 2 for k in configs]
NOISE = sum(half_widths) / len(half_widths)


def _spines(ax, keep=("bottom",)):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)


def _footer(fig, text):
    fig.text(0.01, 0.008, text, ha="left", va="bottom", fontsize=7.3,
             color=MUTED, style="italic")


# =============================================================================
# Figure 1 — AUC forest plot (ranked, colored by architecture family)
# =============================================================================
def figure1():
    order = sorted(configs, key=lambda k: D[k]["auc"], reverse=True)
    order = order[::-1]  # matplotlib y grows upward -> best ends up on top
    y = list(range(len(order)))

    fig, ax = plt.subplots(figsize=(7.4, 5.1))

    # chance reference (label sits on the line with a surface halo)
    ax.axvline(0.5, color=MUTED, ls=(0, (4, 3)), lw=1.0, zorder=1)
    ax.text(0.5, 3.0, "chance", rotation=90, color=MUTED, fontsize=8,
            ha="center", va="center", zorder=5,
            bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))

    best = max(configs, key=lambda k: D[k]["auc"])

    for yi, k in zip(y, order):
        m = D[k]["auc"]
        lo, hi = D[k]["auc_ci_low"], D[k]["auc_ci_high"]
        color = FAMILY[k][1]
        ax.plot([lo, hi], [yi, yi], color=color, lw=2.2, solid_capstyle="round",
                zorder=3, alpha=0.9)
        # CI end caps
        for x in (lo, hi):
            ax.plot([x, x], [yi - 0.12, yi + 0.12], color=color, lw=1.6, zorder=3)
        # point estimate with surface ring so it reads over the line
        ax.plot(m, yi, "o", ms=8.5, color=color, mec=SURFACE, mew=1.6, zorder=4)
        # direct value label (identity never color-alone)
        ax.text(hi + 0.006, yi, f"{m:.3f}", va="center", ha="left",
                fontsize=9, color=INK,
                fontweight="bold" if k == best else "normal")

    ax.set_yticks(y)
    ax.set_yticklabels([PRETTY[k] for k in order], fontsize=10)
    # bold the best row label
    for lbl, k in zip(ax.get_yticklabels(), order):
        if k == best:
            lbl.set_fontweight("bold")
    ax.set_xlabel("Held-out AUC  (nested 5×3 CV, 3 repeats)", color=INK2)
    ax.set_xlim(0.44, 0.73)
    ax.set_ylim(-0.6, len(order) - 0.2)
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.05))
    ax.grid(axis="x", color=GRID, lw=0.7, zorder=0)
    ax.tick_params(length=0)
    _spines(ax, keep=("bottom",))

    ax.set_title("MGMT-methylation ablation: discrimination by configuration",
                 fontsize=12.5, fontweight="bold", loc="left", color=INK, pad=18)
    ax.text(0, 1.02,
            f"UPENN-GBM cohort, n={cohort['n']} ({cohort['n_positive']} methylated, "
            f"{100*cohort['n_positive']/cohort['n']:.1f}%) · "
            f"{cohort['n_imaging']} imaging + {cohort['n_clinical']} clinical features · "
            "whiskers = 95% CI",
            transform=ax.transAxes, fontsize=8.7, color=INK2, va="bottom")

    # family legend — horizontal, below the plot (out of the data area)
    seen, handles = set(), []
    for k in ["imaging_only", "mlp_fusion", "mlp_gnn"]:
        fam, col = FAMILY[k]
        if fam in seen:
            continue
        seen.add(fam)
        handles.append(Line2D([0], [0], marker="o", ls="none", ms=8, color=col,
                              mec=SURFACE, mew=1.2, label=fam))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.075),
               ncol=3, frameon=False, fontsize=8.8, handletextpad=0.4,
               columnspacing=1.6)

    _footer(fig, "All 95% CIs overlap and every interval includes or nears chance — "
                 "differences are within confidence-interval noise at this sample size.")
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    _save(fig, "fig1_ablation_auc_forest")


# =============================================================================
# Figure 2 — attribution: (a) controlled ΔAUC effects, (b) graph reliance
# =============================================================================
def figure2():
    auc = {k: D[k]["auc"] for k in configs}
    # controlled, one-factor-at-a-time contrasts, grouped into sections
    groups = [
        ("Feature contribution", [
            ("+ Clinical  (vs imaging-only)", auc["mlp_fusion"] - auc["imaging_only"]),
            ("+ Imaging  (vs clinical-only)", auc["mlp_fusion"] - auc["clinical_only"]),
        ]),
        ("Memory-bank GNN effect", [
            ("on MLP encoder", auc["mlp_gnn"] - auc["mlp_fusion"]),
            ("on KAN encoder", auc["kan_gnn"] - auc["kan_fusion"]),
        ]),
        ("KAN vs MLP encoder", [
            ("without GNN", auc["kan_fusion"] - auc["mlp_fusion"]),
            ("with GNN",    auc["kan_gnn"]   - auc["mlp_gnn"]),
        ]),
        ("Gated fusion", [
            ("KAN+GNN  →  +gate", auc["kan_gnn_gate"] - auc["kan_gnn"]),
        ]),
    ]

    # lay out top-to-bottom: a header row then its bars, with a gap between groups
    XL, XR = -0.115, 0.12
    bar_y, bar_lab, bar_val, headers = [], [], [], []
    level, GAP, HEAD = 0.0, 0.7, 0.95
    for gi, (gname, items) in enumerate(groups):
        if gi:
            level += GAP
        headers.append((-level, gname))
        level += HEAD
        for lab, dv in items:
            bar_y.append(-level); bar_lab.append(lab); bar_val.append(dv)
            level += 1.0
    ytop, ybot = headers[0][0] + 0.7, bar_y[-1] - 0.7

    fig = plt.figure(figsize=(7.6, 7.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.5, 1.0], hspace=0.34)
    axA = fig.add_subplot(gs[0])
    axB = fig.add_subplot(gs[1])

    # ---- Panel A: diverging ΔAUC -------------------------------------------
    axA.axvspan(-NOISE, NOISE, color=BAND, zorder=0)            # noise band
    for xb in (-NOISE, NOISE):
        axA.axvline(xb, color=MUTED, ls=(0, (3, 3)), lw=0.8, zorder=1)
    axA.axvline(0, color=INK, lw=1.1, zorder=2)

    for yi, lab, dv in zip(bar_y, bar_lab, bar_val):
        col = BLUE if dv >= 0 else RED
        axA.barh(yi, dv, height=0.66, color=col, zorder=3,
                 edgecolor=SURFACE, linewidth=0.8)
        off = 0.0018 if dv >= 0 else -0.0018
        axA.text(dv + off, yi, f"{dv:+.3f}", va="center",
                 ha="left" if dv >= 0 else "right", fontsize=8.8, color=INK)

    axA.set_yticks(bar_y)
    axA.set_yticklabels(bar_lab, fontsize=9.4)
    axA.set_xlim(XL, XR)
    axA.set_ylim(ybot, ytop)
    axA.xaxis.set_major_locator(plt.MultipleLocator(0.05))
    axA.grid(axis="x", color=GRID, lw=0.7, zorder=0)
    axA.tick_params(length=0)
    _spines(axA, keep=())
    axA.set_xlabel("Δ AUC  (contrast − reference)", color=INK2)

    # section headers with an underline rule
    for yh, name in headers:
        axA.text(XL + 0.002, yh, name, ha="left", va="center", fontsize=8.9,
                 fontweight="bold", color=INK2)
        axA.plot([XL + 0.002, XR], [yh - 0.5, yh - 0.5], color=GRID, lw=0.8,
                 zorder=1)
    # band annotation, tucked at the bottom-right of the band
    axA.text(NOISE, ybot + 0.25, "±0.076\n95% CI", fontsize=7.4, color=MUTED,
             ha="center", va="bottom", linespacing=0.95)

    axA.set_title("Component attribution: controlled one-factor contrasts",
                  fontsize=12.5, fontweight="bold", loc="left", color=INK, pad=16)
    axA.text(0, 1.025,
             "Each bar isolates one change, others held fixed. "
             "Blue = helps, red = hurts; grey band = within-noise (±0.076).",
             transform=axA.transAxes, fontsize=8.6, color=INK2, va="bottom")

    # ---- Panel B: graph reliance -------------------------------------------
    gnn = [k for k in ["mlp_gnn", "kan_gnn", "kan_gnn_gate"]]
    yb = list(range(len(gnn)))[::-1]
    for yi, k in zip(yb, gnn):
        r = D[k]["graph_reliance_mean"]
        axB.barh(yi, r, height=0.6, color=BLUE, edgecolor=SURFACE, linewidth=0.8,
                 zorder=3)
        axB.text(r + 0.008, yi, f"{r:.2f}", va="center", ha="left",
                 fontsize=8.7, color=INK)
    axB.set_yticks(yb)
    axB.set_yticklabels([PRETTY[k] for k in gnn], fontsize=9.3)
    axB.set_xlim(0, 1.0)
    axB.xaxis.set_major_locator(plt.MultipleLocator(0.25))
    axB.grid(axis="x", color=GRID, lw=0.7, zorder=0)
    axB.tick_params(length=0)
    _spines(axB, keep=("bottom",))
    axB.set_xlabel("Fraction of representation drawn from graph neighbours "
                   "(memory bank)", color=INK2)
    axB.set_title("Graph reliance of the GNN variants", fontsize=11,
                  fontweight="bold", loc="left", color=INK, pad=8)

    _footer(fig, "The memory bank does contribute (~25% of the fused representation), "
                 "but that contribution does not translate into an AUC gain (Panel A).")
    fig.subplots_adjust(left=0.24, right=0.97, top=0.90, bottom=0.09)
    _save(fig, "fig2_attribution")


def _save(fig, stem):
    pdf = os.path.join(FIGDIR, stem + ".pdf")
    png = os.path.join(FIGDIR, stem + ".png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.relpath(pdf, ROOT)}  and  {os.path.relpath(png, ROOT)}")


if __name__ == "__main__":
    print(f"noise band (mean 95% CI half-width) = ±{NOISE:.4f}")
    figure1()
    figure2()
    print("done.")
