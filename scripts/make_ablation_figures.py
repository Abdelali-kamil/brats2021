#!/usr/bin/env python
"""Thesis figures for the BraTS 2021 MGMT-methylation ablation (KAN+Transformer+GNN).

Reads results/classification/kan_gnn_brats.json and writes two figures to
results/figures/ as vector PDF (for the thesis) + 300-dpi PNG (for preview):

  fig1_ablation_auc_forest   AUC point estimate + 95% CI per configuration,
                             with the BraTS->UPenn external-validation point.
  fig2_attribution           (a) controlled ΔAUC component effects vs a noise band
                             (b) graph reliance of the memory-bank (GNN) variants

Primary cohort is BraTS 2021 MGMT (imaging only). UPenn-GBM appears only as the
external-validation point and the imaging+clinical multimodal reference.

Colors are the data-viz reference palette (colorblind-validated).
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results", "classification", "kan_gnn_brats.json")
FIGDIR = os.path.join(ROOT, "results", "figures")
os.makedirs(FIGDIR, exist_ok=True)

# ---- reference palette (validated; do not edit values) ----------------------
SURFACE   = "#fcfcfb"
INK       = "#0b0b0b"
INK2      = "#52514e"
MUTED     = "#898781"
GRID      = "#e1e0d9"
BASELINE  = "#c3c2b7"
BLUE      = "#2a78d6"
ORANGE    = "#eb6834"
AQUA      = "#1baf7a"
RED       = "#e34948"
BAND      = "#f0efec"

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

# Primary BraTS ablation rungs, in ladder order.
RUNGS = ["mlp", "kan", "kan_transformer", "kan_gnn", "full"]
PRETTY = {
    "mlp":             "MLP (baseline)",
    "kan":             "KAN",
    "kan_transformer": "KAN + Transformer",
    "kan_gnn":         "KAN + GNN",
    "full":            "KAN + Transformer + GNN",
}
FAMILY = {  # config -> (family label, color)
    "mlp":             ("Baseline",              BLUE),
    "kan":             ("Single component",      ORANGE),
    "kan_transformer": ("Single component",      ORANGE),
    "kan_gnn":         ("Single component",      ORANGE),
    "full":            ("Full module",           AQUA),
}

configs = [k for k in RUNGS if k in D]
half_widths = [(D[k]["auc_ci_high"] - D[k]["auc_ci_low"]) / 2 for k in configs]
NOISE = sum(half_widths) / len(half_widths)
N = cohort.get("primary_n")
NPOS = cohort.get("primary_positives")
NFEAT = cohort.get("n_features")


def _spines(ax, keep=("bottom",)):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)


def _footer(fig, text):
    fig.text(0.01, 0.008, text, ha="left", va="bottom", fontsize=7.3,
             color=MUTED, style="italic")


# =============================================================================
# Figure 1 — AUC forest plot
# =============================================================================
def figure1():
    order = sorted(configs, key=lambda k: D[k]["auc"], reverse=True)[::-1]
    rows = [("rung", k) for k in order]

    # external-validation point (BraTS-trained full model -> UPenn), shown below
    # the primary rungs behind a divider so cohorts are never conflated.
    ext = D.get("external_upenn")
    mm = D.get("upenn_multimodal")
    extra = []
    if mm:
        extra.append(("upenn_mm", "UPenn multimodal (img+clin)"))
    if ext:
        extra.append(("external", "BraTS→UPenn (external)"))
    rows = extra + rows  # extras at the bottom (y grows upward)

    y = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(7.6, 5.2))

    ax.axvline(0.5, color=MUTED, ls=(0, (4, 3)), lw=1.0, zorder=1)
    ax.text(0.5, len(rows) - 1.0, "chance", rotation=90, color=MUTED, fontsize=8,
            ha="center", va="center", zorder=5,
            bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))

    best = max(configs, key=lambda k: D[k]["auc"])
    labels = []
    for yi, (kind, key) in zip(y, rows):
        if kind == "rung":
            d = D[key]; color = FAMILY[key][1]; label = PRETTY[key]
        elif kind == "external":
            d = ext; color = MUTED; label = key
        else:
            d = mm; color = MUTED; label = key
        m, lo, hi = d["auc"], d["auc_ci_low"], d["auc_ci_high"]
        ax.plot([lo, hi], [yi, yi], color=color, lw=2.2, solid_capstyle="round",
                zorder=3, alpha=0.9)
        for x in (lo, hi):
            ax.plot([x, x], [yi - 0.12, yi + 0.12], color=color, lw=1.6, zorder=3)
        ax.plot(m, yi, "o", ms=8.5, color=color, mec=SURFACE, mew=1.6, zorder=4)
        ax.text(hi + 0.006, yi, f"{m:.3f}", va="center", ha="left", fontsize=9,
                color=INK, fontweight="bold" if key == best else "normal")
        labels.append(label)

    # divider between external/multimodal reference rows and the primary rungs
    if extra:
        ax.axhline(len(extra) - 0.5, color=GRID, lw=1.0, ls=(0, (2, 2)), zorder=1)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    for lbl, (kind, key) in zip(ax.get_yticklabels(), rows):
        if key == best:
            lbl.set_fontweight("bold")
        if kind != "rung":
            lbl.set_color(INK2); lbl.set_style("italic")
    ax.set_xlabel("AUC  (BraTS: nested 5×3 CV, 3 repeats · external: single held-out test)",
                  color=INK2)
    ax.set_xlim(0.44, 0.75)
    ax.set_ylim(-0.6, len(rows) - 0.2)
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.05))
    ax.grid(axis="x", color=GRID, lw=0.7, zorder=0)
    ax.tick_params(length=0)
    _spines(ax, keep=("bottom",))

    ax.set_title("BraTS 2021 MGMT ablation: discrimination by configuration",
                 fontsize=12.5, fontweight="bold", loc="left", color=INK, pad=18)
    ax.text(0, 1.02,
            f"BraTS 2021 (primary), n={N} ({NPOS} methylated, "
            f"{100*NPOS/N:.1f}%) · {NFEAT} radiomic features (imaging only) · "
            "whiskers = 95% CI",
            transform=ax.transAxes, fontsize=8.7, color=INK2, va="bottom")

    handles = [
        Line2D([0], [0], marker="o", ls="none", ms=8, color=BLUE, mec=SURFACE,
               mew=1.2, label="Baseline (MLP)"),
        Line2D([0], [0], marker="o", ls="none", ms=8, color=ORANGE, mec=SURFACE,
               mew=1.2, label="Single component"),
        Line2D([0], [0], marker="o", ls="none", ms=8, color=AQUA, mec=SURFACE,
               mew=1.2, label="Full module"),
        Line2D([0], [0], marker="o", ls="none", ms=8, color=MUTED, mec=SURFACE,
               mew=1.2, label="UPenn (external)"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.03),
               ncol=4, frameon=False, fontsize=8.6, handletextpad=0.4,
               columnspacing=1.4)

    _footer(fig, "All primary-cohort intervals overlap; the added components do not "
                 "separate from the MLP baseline, and external transfer is at chance.")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    _save(fig, "fig1_ablation_auc_forest")


# =============================================================================
# Figure 2 — attribution
# =============================================================================
def figure2():
    auc = {k: D[k]["auc"] for k in configs}
    groups = [
        ("Encoder: KAN vs MLP", [
            ("KAN − MLP", auc["kan"] - auc["mlp"]),
        ]),
        ("Added components (vs KAN)", [
            ("+ Transformer", auc["kan_transformer"] - auc["kan"]),
            ("+ Memory-bank GNN", auc["kan_gnn"] - auc["kan"]),
        ]),
        ("Full module", [
            ("Full − MLP baseline", auc["full"] - auc["mlp"]),
        ]),
    ]
    if "external_upenn" in D:
        groups.append(("External generalization", [
            ("BraTS→UPenn − BraTS (full)", D["external_upenn"]["auc"] - auc["full"]),
        ]))

    XL, XR = -0.12, 0.12
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
    gs = fig.add_gridspec(2, 1, height_ratios=[3.4, 1.0], hspace=0.36)
    axA = fig.add_subplot(gs[0])
    axB = fig.add_subplot(gs[1])

    axA.axvspan(-NOISE, NOISE, color=BAND, zorder=0)
    for xb in (-NOISE, NOISE):
        axA.axvline(xb, color=MUTED, ls=(0, (3, 3)), lw=0.8, zorder=1)
    axA.axvline(0, color=INK, lw=1.1, zorder=2)

    for yi, lab, dv in zip(bar_y, bar_lab, bar_val):
        col = BLUE if dv >= 0 else RED
        axA.barh(yi, dv, height=0.62, color=col, zorder=3, edgecolor=SURFACE,
                 linewidth=0.8)
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

    for yh, name in headers:
        axA.text(XL + 0.002, yh, name, ha="left", va="center", fontsize=8.9,
                 fontweight="bold", color=INK2)
        axA.plot([XL + 0.002, XR], [yh - 0.5, yh - 0.5], color=GRID, lw=0.8, zorder=1)
    axA.text(NOISE, ybot + 0.25, f"±{NOISE:.3f}\n95% CI", fontsize=7.4, color=MUTED,
             ha="center", va="bottom", linespacing=0.95)

    axA.set_title("Component attribution: controlled one-factor contrasts",
                  fontsize=12.5, fontweight="bold", loc="left", color=INK, pad=16)
    axA.text(0, 1.025,
             "Each bar isolates one change, others held fixed. "
             f"Blue = helps, red = hurts; grey band = within-noise (±{NOISE:.3f}).",
             transform=axA.transAxes, fontsize=8.6, color=INK2, va="bottom")

    # ---- Panel B: graph reliance -------------------------------------------
    gnn = [k for k in ["kan_gnn", "full"] if D.get(k, {}).get("graph_reliance_mean") is not None]
    yb = list(range(len(gnn)))[::-1]
    for yi, k in zip(yb, gnn):
        r = D[k]["graph_reliance_mean"]
        axB.barh(yi, r, height=0.55, color=BLUE, edgecolor=SURFACE, linewidth=0.8,
                 zorder=3)
        axB.text(r + 0.008, yi, f"{r:.2f}", va="center", ha="left", fontsize=8.7,
                 color=INK)
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

    _footer(fig, "The memory bank contributes ~25% of the fused representation, but that "
                 "contribution does not translate into an AUC gain (Panel A).")
    fig.subplots_adjust(left=0.26, right=0.97, top=0.90, bottom=0.09)
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
