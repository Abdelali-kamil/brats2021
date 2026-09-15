# Running this pipeline on a server

How to stand up this project on your own server. Read the tiers first — what
you can run depends on what you have besides the code.

## What the repo ships vs. what you must supply

The repository contains **code** and the **committed result CSVs/figures** in
`results/`. It does **not** contain (both are gitignored):

- **Imaging data** — BraTS2021 (~34 GB) and UPenn-GBM NIfTI (~59 GB), obtained
  separately (see `README.md` "Setup"). Not redistributable here.
- **Trained checkpoints** — `checkpoints/*.pth`. The evaluation scripts load
  named checkpoints (`segmentor_epoch_650.pth`, `upenn_v3_best.pth`, …). These
  are **not** in the repo, so the GPU evaluation cannot run until you either
  obtain them or train from scratch.

So there are three tiers:

| Tier | Needs | GPU? | Time | Command |
|---|---|---|---|---|
| **0 — smoke test** | nothing extra (uses committed CSVs) | no | minutes | `pytest`, `summarize_brats.py`, `make_report.py` |
| **1 — full evaluation** | data **+** checkpoints | yes | ~2–3 h | `scripts/run_pipeline.sh` |
| **2 — train from scratch** | data | yes | days | `train_brats.py` → `train_upenn.py` |

Start with Tier 0 to confirm the environment is healthy on your server before
you move any data.

## Recommended hardware ("best" for this repo)

The segmentor is a 3D Wavelet U-Net++ trained at **batch size 1** with **8-step
gradient accumulation** and **AMP (fp16 autocast)** — memory-hungry per sample,
so a single strong GPU is the right shape, not many small ones.

- **GPU:** one modern NVIDIA card with **≥ 24 GB VRAM**. RTX 4090 / A5000 /
  A6000 are comfortable; an **A100 (40 or 80 GB)** is ideal for fastest
  training and lets you raise `--batch-size`. 24 GB is the practical floor for
  training; inference-only (Tier 1) fits in less.
- **CUDA:** 12.x driver. `torch >= 2.4` with a matching CUDA build (the results
  were made with a CUDA 12.8 build; any recent CUDA build reproduces the
  numbers, bit-exactness aside — see `requirements.txt`).
- **CPU/RAM:** 8+ cores, 32 GB+ system RAM (data loading + sliding-window
  inference). More cores help `--num-workers`.
- **Disk:** ~**200 GB** free — 93 GB raw data + ~59 GB working NIfTI + ~7 GB
  probability-map cache + checkpoints. An SSD noticeably speeds data loading.
- **OS / Python:** Linux, **Python 3.10–3.12**.

No GPU? Only Tier 0 is practical. Training and sliding-window inference need
CUDA; on CPU they run but are impractically slow.

## 1. Environment

```bash
git clone <your-remote>/brats2021.git
cd brats2021

python3 -m venv .venv           # or: conda create -n brats python=3.11
source .venv/bin/activate

# Install a CUDA build of torch that matches your driver FIRST, e.g. CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121
# then the rest (pure-Python scientific stack; no pyradiomics needed —
# radiomics are computed in-repo, and the DWT is implemented in model.py):
pip install -r requirements.txt

python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

## 2. Tier 0 — smoke test (do this first)

Runs from the committed CSVs; no data, no GPU:

```bash
python -m pytest tests/ -q          # 46 tests, ~2 s
python scripts/summarize_brats.py   # BraTS internal-validation summary from results/
python scripts/make_report.py       # regenerate docs/RESULTS.md from results/
```

If those pass, the code and your Python environment are good on the server.

## 3. Tier 1 — full evaluation (data + checkpoints + GPU)

Place data exactly as `README.md` specifies:

```
data/BraTS2021_XXXXX/BraTS2021_XXXXX_{flair,t1,t1ce,t2,seg}.nii.gz
upenn_nifti/sub-XXX_{FLAIR,T1w,ce-gd_T1w,T2w,seg}.nii.gz
```

Put the trained checkpoints under `checkpoints/` with the names
`scripts/evaluate_upenn.py` expects (`segmentor_epoch_650.pth`,
`upenn_v3_best.pth`, `upenn_v3_last.pth`, `upenn_v3_topk/*.pth`,
`v4/upenn_v3_best.pth`, …). Then:

```bash
python scripts/verify_no_leakage.py   # cheap integrity checks — run before GPU hours
./scripts/run_pipeline.sh             # integrity → BraTS summary → UPenn eval → IDH1
./scripts/run_pipeline.sh --with-features   # also re-extract radiomics (~1.5 h)
```

`run_pipeline.sh` sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, runs
sequentially (assumes a shared GPU), and tees every stage into `logs/`.

## 4. Tier 2 — train from scratch (data + GPU)

```bash
python scripts/train_brats.py --epochs 750 --save-dir checkpoints
python scripts/train_upenn.py --epochs 80  --save-dir checkpoints   # fine-tunes from the BraTS checkpoint
```

Useful `train_brats.py` flags: `--device auto|cuda:0`, `--batch-size` (raise it
if you have an A100), `--accum-steps` (default 8; effective batch = batch ×
accum), `--num-workers`, `--resume checkpoints/....pth`. AMP turns on
automatically on CUDA.

## 5. Long runs: detach and monitor

Training is long — never rely on your SSH session staying up.

```bash
# tmux (simplest)
tmux new -s brats
./scripts/run_pipeline.sh; # ... detach with Ctrl-b d, reattach: tmux attach -t brats

# or nohup
mkdir -p logs
nohup ./scripts/run_pipeline.sh > logs/pipeline.out 2>&1 &

./scripts/watch.sh status        # follow a running stage (repo helper)
tail -f logs/pipeline.out
nvidia-smi -l 5                  # watch GPU memory/util
```

**SLURM cluster:** wrap the same commands in an `sbatch` script requesting one
GPU (`#SBATCH --gres=gpu:1`), enough RAM and time, and `source .venv/bin/activate`
before the command. Ask and I'll write the job script for your scheduler.

## Troubleshooting

- **CUDA OOM** during training → keep `--batch-size 1`, raise `--accum-steps`;
  the pipeline already sets `expandable_segments:True`.
- **`torch.cuda.is_available()` is False** → the installed torch is CPU-only or
  mismatched to your driver. Reinstall torch from the correct `cuXXX` index.
- **Scripts can't find data/checkpoints** → paths are relative to the repo
  root and must match the layout above exactly (channel names included).
- **`verify_no_leakage.py` fails on a clean checkout** → expected before
  results exist; `run_pipeline.sh` tolerates the first run and re-checks at the
  end.
