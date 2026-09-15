#!/usr/bin/env bash
# Download every baseline's OFFICIAL source at a pinned commit, into
# baselines/repos/<name>/. Idempotent: re-running skips repos already cloned.
#
# Pinned commits (recorded 2026-09-15) make the baselines reproducible for the
# paper — the same code every time, regardless of upstream changes. To move to
# a newer upstream, update the SHA next to the repo and re-run.
#
#   bash baselines/fetch.sh            # clone/pin all
#   bash baselines/fetch.sh nnunet     # clone/pin just one (by name)
#
# XGBoost is NOT cloned here — it installs from PyPI (see baselines/xgboost/).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p repos

clone_pin() {
  local name="$1" url="$2" sha="$3" dst="repos/$1"
  [[ $# -gt 3 && -n "${ONLY:-}" && "$ONLY" != "$name" ]] && return 0
  if [[ -d "$dst/.git" ]]; then
    echo "[skip]  $name already present"
  else
    echo "[clone] $name  <-  $url"
    git clone --filter=blob:none "$url" "$dst"
  fi
  git -C "$dst" fetch --quiet origin "$sha" 2>/dev/null || true
  if git -C "$dst" checkout --quiet "$sha" 2>/dev/null; then
    echo "[pin]   $name @ ${sha:0:12}"
  else
    echo "[WARN]  $name: pinned commit ${sha:0:12} not fetchable; left on default branch"
  fi
}

ONLY="${1:-}"
[[ -n "$ONLY" ]] && echo "fetching only: $ONLY"

clone_pin nnunet     https://github.com/MIC-DKFZ/nnUNet.git                      ded2aa3a4c81a9caae37054224d8ebac2d19a061 x
clone_pin swin_unetr https://github.com/Project-MONAI/research-contributions.git 21ed8e57c7256834d4fbaf19579ca25ad3d135ee x
clone_pin selfmedmae https://github.com/cvlab-stonybrook/SelfMedMAE.git          21f38b1ff8fb9ce3948651c1e356f3cc3fddd45a x
clone_pin mtanet     https://github.com/yatingling/MTANet.git                    7a5f5d163382bce8856c1552a8b9ed9c0a6980e6 x
clone_pin mmgl       https://github.com/SsGood/MMGL.git                          c13b05c190f85e614ccb3ace60bb6b152a1f4a1b x
clone_pin daft       https://github.com/ai-med/DAFT.git                          b36974b1f8d9f46fd410e946b6ead28078f0622e x

echo
echo "done. Cloned repos are under baselines/repos/ (gitignored)."
echo "XGBoost: pip install xgboost==3.2.0  (no clone; see baselines/xgboost/SETUP.md)"
echo "Next: follow baselines/<method>/SETUP.md for each method you want to run."
