# Where to pick up

Last session: 2026-07-24.

## Check the ablation first

It was left running detached and should have finished overnight.

```bash
cd ~/brats2021
tail -30 logs/ablation_kan_gnn.log        # did it finish?
cat results/classification/kan_gnn_ablation.json
```

The final table prints at the end of the log. If the log stops mid-way without
a summary table, the job died — rerun with `python scripts/ablation_kan_gnn.py`.
There is no checkpointing, so a rerun starts from scratch (~2 h).

## Results as of the last update

Ablation ladder, UPenn MGMT (n=227, 98 methylated, nested CV):

| Rung | AUC [95% CI] | Δ |
|---|---|---|
| `imaging_only` | 0.536 [0.463, 0.614] | — |
| `mlp_fusion` | **0.629** [0.552, 0.701] | +0.093 |
| `kan_fusion` | 0.562 [0.487, 0.635] | −0.067 |
| `kan_gnn` | pending | |
| `kan_gnn_gate` | pending | |

Two findings so far. Clinical variables (age, sex, GTR) carry the signal —
imaging alone sits at chance. And **KAN underperformed the plain MLP**, which
the parameter count predicted: KAN is 8–10× larger at 54 input features, and
n=227 cannot support that.

## Next steps, in priority order

1. **`clinical_only` control** — already added to `CONFIGS`. Minutes to run:
   ```bash
   python scripts/ablation_kan_gnn.py --configs clinical_only
   ```
   This is the decisive control. If it matches `mlp_fusion` (~0.63), then the
   imaging branch contributes nothing and "multimodal fusion" is really
   clinical prediction. Every downstream claim depends on this number.

2. **`mlp_gnn` rung** — not yet added. The ladder confounds two changes at
   rung 4: it adds the GNN on top of an encoder that is already underperforming.
   Adding MLP+GNN isolates the memory bank against the *best* encoder instead of
   the worst. Add to `CONFIGS` in `scripts/ablation_kan_gnn.py`:
   ```python
   "mlp_gnn": dict(use_clinical=True, use_kan=False, use_gnn=True, use_gate=False),
   ```

3. **Lesion Encoder** (proposal §2.2(2), the cheap half). The segmentor's
   `conv4_0` bottleneck is 256×D×H×W; mask-pooling it gives a learned 256-d
   per-patient embedding without training anything new. Slots into the same
   harness as an extra imaging-feature source and directly tests whether learned
   features beat the 54 radiomics at the `imaging_only` rung. Hours, not weeks.

4. **ViT Global Encoder**, then **joint segmentation-classification
   optimisation** (Innovation 4). Both substantially larger.

## Outstanding from earlier (paused at your request)

- **Proposal §5.2 correction.** The reported 0.8974 mean DSC is pooled over the
  1000 training cases. Held-out is **0.8828** [0.8638, 0.8992]. HD95 values also
  need the surface-based definition: 3.24 / 4.82 / 10.17 mm, not 2.04 / 1.56 /
  1.97. See `docs/METHODOLOGY.md`.
- **nnU-Net baseline** on the same 251-case split. No comparison numbers
  currently exist that were computed on this project's own split.

## Repository state

Clean. All segmentation and classification results final and committed.
`python -m pytest tests/ -q` → 46 pass. `python scripts/verify_no_leakage.py`
→ 21 checks pass. `./scripts/watch.sh status` shows any running job.
