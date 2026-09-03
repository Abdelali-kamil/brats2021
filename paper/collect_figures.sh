#!/usr/bin/env bash
# Assemble the figures paper.tex needs into paper/figures/.
#
# The figures are not stored here. Each one is an output of the analysis that
# produced it and already lives under results/, so a second copy in this
# directory would be a copy that silently goes stale the next time a figure is
# regenerated. This script gathers them instead, and paper/figures/ is
# gitignored.
#
# Usage:  bash paper/collect_figures.sh   (from the repository root)
#
# Every name below must match an \includegraphics in paper.tex. The script
# fails if any is missing rather than leaving LaTeX to report it as a missing
# file at compile time.
set -euo pipefail

cd "$(dirname "$0")/.."
out="paper/figures"
mkdir -p "$out"

figures=(
    brats_BraTS2021_00413
    brats_BraTS2021_00773
    brats_BraTS2021_01628
    crossval_boxplot
    fig1_ablation_auc_forest
    fig2_attribution
    fig_architecture_implemented
    fig_modalities_brats
    fig_failure_modes
    upenn_sub-251
    upenn_sub-330
    upenn_sub-380
)

missing=0
for name in "${figures[@]}"; do
    # Prefer the vector PDF; fall back to the raster PNG where only that exists.
    src=$(find results -name "$name.pdf" | head -1)
    [ -z "$src" ] && src=$(find results -name "$name.png" | head -1)
    if [ -z "$src" ]; then
        echo "MISSING: $name (no .pdf or .png under results/)" >&2
        missing=1
        continue
    fi
    cp "$src" "$out/$(basename "$src")"
    echo "  $name <- $src"
done

if [ "$missing" -ne 0 ]; then
    echo "" >&2
    echo "Some figures are absent. Regenerate them before compiling:" >&2
    echo "  python scripts/make_ablation_figures.py" >&2
    echo "  python scripts/make_failure_figures.py" >&2
    echo "  python scripts/make_architecture_figure.py" >&2
    exit 1
fi

echo ""
echo "${#figures[@]} figures collected into $out/"
