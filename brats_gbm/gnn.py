"""Population-graph GNN with an inductive memory bank.

Implements Innovation 3 of the proposal: a graph of historical patients that a
new patient is attached to at inference time, so their prediction is refined by
similar prior cases.

Why this design is leakage-free
-------------------------------
The obvious way to build a patient graph is to run k-NN over the whole cohort
and message-pass. That is *transductive*: a test patient's embedding then
depends on other test patients, the graph structure was chosen with knowledge
of the test set, and the resulting score does not estimate what happens when a
single new patient arrives alone. It is the same class of error as fitting a
threshold on the test set.

The memory bank avoids it by construction:

  * The graph contains **training patients only**. It is built inside each
    cross-validation fold from that fold's training split.
  * A query patient attaches to its k nearest **training** nodes. Query-to-query
    edges never exist, so test patients cannot see each other.
  * Edges come from feature similarity alone. Labels are never consulted, at
    any stage.
  * Message passing is one-directional: memory -> query. The bank is frozen
    during inference, so processing patients in a different order, or one at a
    time, gives identical predictions.

That last property is what makes this genuinely inductive and deployable, and
it is worth stating in the thesis: a patient arriving alone at 3am gets exactly
the prediction they would have got inside a batch.

What the mechanism actually is
------------------------------
Retrieval-augmented classification, not population-graph learning. The cited
population-GNN literature (Kipf & Welling; MMGL) is transductive and solves a
different problem. Describing this as retrieval-augmented is both more accurate
and a cleaner novelty claim, and it sidesteps the leakage critique entirely.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_knn(query: torch.Tensor, bank: torch.Tensor, k: int,
               exclude_self: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k most cosine-similar bank rows for each query row.

    Cosine rather than Euclidean because the feature vector mixes volumes,
    intensities and ratios whose scales differ by orders of magnitude even
    after standardisation; direction is the more stable notion of similarity.
    """
    q = F.normalize(query, dim=1)
    b = F.normalize(bank, dim=1)
    sim = q @ b.t()  # [n_query, n_bank]
    if exclude_self:
        # Building the training graph: a node must not be its own neighbour.
        n = min(sim.shape)
        sim[torch.arange(n), torch.arange(n)] = -2.0
    k = min(k, sim.shape[1])
    vals, idx = sim.topk(k, dim=1)
    return vals, idx


class GraphConv(nn.Module):
    """Similarity-weighted neighbourhood aggregation with a self term.

    Deliberately simple. With a few hundred patients and ~50 features, an
    expressive graph operator would overfit long before its expressiveness
    helped; this is a learned interpolation between a node's own
    representation and its neighbours' mean.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_neigh = nn.Linear(in_dim, out_dim)
        # Learned, sigmoid-bounded mixing weight. Initialised negative so the
        # model starts by trusting its own features and must earn the use of
        # neighbours -- if the graph carries no signal, training can drive this
        # to zero and the layer degrades gracefully to an MLP.
        self.alpha = nn.Parameter(torch.tensor(-1.0))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, neigh_feat: torch.Tensor,
                neigh_w: torch.Tensor) -> torch.Tensor:
        """x: [n, d]; neigh_feat: [n, k, d]; neigh_w: [n, k] similarities."""
        w = torch.softmax(neigh_w, dim=1).unsqueeze(-1)
        agg = (neigh_feat * w).sum(dim=1)
        a = torch.sigmoid(self.alpha)
        return self.dropout(F.relu((1 - a) * self.lin_self(x) + a * self.lin_neigh(agg)))

    @property
    def neighbour_weight(self) -> float:
        """How much the layer relies on neighbours; 0 means the graph is unused."""
        return float(torch.sigmoid(self.alpha).detach())


class MemoryBank(nn.Module):
    """Frozen store of training-patient embeddings, queried by similarity."""

    def __init__(self, k: int = 8):
        super().__init__()
        self.k = k
        self.register_buffer("bank_feat", torch.empty(0), persistent=False)

    @torch.no_grad()
    def build(self, train_embeddings: torch.Tensor) -> None:
        """Populate from training embeddings only. Call once per fold."""
        self.bank_feat = train_embeddings.detach().clone()

    def lookup(self, query: torch.Tensor, exclude_self: bool = False):
        if self.bank_feat.numel() == 0:
            raise RuntimeError("MemoryBank.build() must be called before lookup()")
        sims, idx = cosine_knn(query, self.bank_feat, self.k, exclude_self)
        return self.bank_feat[idx], sims  # [n, k, d], [n, k]


class ClinicalImagingKANGNN(nn.Module):
    """Imaging + clinical fusion, optional KAN encoders, optional memory bank.

    Every component is switchable so the ablation can isolate one change at a
    time: `use_kan` swaps the MLP encoders for KAN, `use_gnn` enables the
    memory bank, `use_gate` enables adaptive gated fusion.
    """

    def __init__(
        self,
        n_imaging: int,
        n_clinical: int,
        hidden: int = 16,
        use_kan: bool = False,
        use_gnn: bool = False,
        use_gate: bool = False,
        use_transformer: bool = False,
        region_slices: list[tuple[int, int]] | None = None,
        k: int = 8,
        dropout: float = 0.3,
        n_heads: int = 2,
        grid_size: int = 3,
        spline_order: int = 3,
    ):
        super().__init__()
        self.use_kan = use_kan
        self.use_gnn = use_gnn
        self.use_gate = use_gate
        self.n_clinical = n_clinical

        def encoder(in_dim: int, out_dim: int) -> nn.Module:
            if use_kan:
                from brats_gbm.kan import KAN
                return KAN([in_dim, out_dim], grid_size=grid_size,
                           spline_order=spline_order)
            return nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())

        self.img_enc = encoder(n_imaging, hidden)
        self.clin_enc = encoder(n_clinical, hidden) if n_clinical > 0 else None

        # Transformer branch (paper V-F). The imaging features carry an
        # anatomical grouping -- every radiomic value belongs to ET, TC or WT
        # (plus a small block of cross-region ratios). `region_slices` names
        # those contiguous column groups, so self-attention runs over a short
        # sequence of *region tokens* rather than over one flat vector, which is
        # the only construction that gives a Transformer something to attend
        # across on tabular input. Its pooled output is added residually to the
        # imaging embedding, matching the proposal's f_comb = f_trans + f_kan.
        # Disabled unless at least two regions are supplied.
        self.use_transformer = bool(
            use_transformer and region_slices and len(region_slices) >= 2)
        if self.use_transformer:
            self.region_slices = list(region_slices)
            self.tok_proj = nn.ModuleList(
                [nn.Linear(e - s, hidden) for (s, e) in self.region_slices])
            layer = nn.TransformerEncoderLayer(
                d_model=hidden, nhead=n_heads, dim_feedforward=hidden * 2,
                dropout=dropout, batch_first=True, activation="gelu")
            self.transformer = nn.TransformerEncoder(layer, num_layers=1)
        else:
            self.transformer = None

        fused_dim = hidden * (2 if n_clinical > 0 else 1)
        if use_gate and n_clinical > 0:
            # Adaptive gated fusion (proposal 3.1): a per-sample weight over the
            # two streams, so a case with uninformative clinical data can lean
            # on imaging and vice versa.
            self.gate = nn.Sequential(nn.Linear(fused_dim, 2), nn.Softmax(dim=1))
            fused_dim = hidden
        else:
            self.gate = None

        self.norm = nn.LayerNorm(fused_dim)
        self.gconv = GraphConv(fused_dim, fused_dim, dropout) if use_gnn else None
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(fused_dim, 1)
        self.memory = MemoryBank(k=k) if use_gnn else None

    def embed(self, x_img: torch.Tensor, x_clin: torch.Tensor | None) -> torch.Tensor:
        h = self.img_enc(x_img)
        if self.transformer is not None:
            tokens = torch.stack(
                [proj(x_img[:, s:e])
                 for proj, (s, e) in zip(self.tok_proj, self.region_slices)],
                dim=1)  # [n, n_regions, hidden]
            h = h + self.transformer(tokens).mean(dim=1)  # residual: f_comb = f_trans + f_kan
        if self.clin_enc is not None and x_clin is not None:
            c = self.clin_enc(x_clin)
            if self.gate is not None:
                w = self.gate(torch.cat([h, c], dim=1))
                h = w[:, :1] * h + w[:, 1:] * c
            else:
                h = torch.cat([h, c], dim=1)
        return self.norm(h)

    def forward(self, x_img: torch.Tensor, x_clin: torch.Tensor | None = None,
                exclude_self: bool = False) -> torch.Tensor:
        h = self.embed(x_img, x_clin)
        if self.gconv is not None:
            neigh, sims = self.memory.lookup(h, exclude_self=exclude_self)
            h = self.gconv(h, neigh, sims)
        return self.head(self.dropout(h)).squeeze(-1)

    @torch.no_grad()
    def refresh_memory(self, x_img: torch.Tensor,
                       x_clin: torch.Tensor | None = None) -> None:
        """Rebuild the bank from training embeddings. Training data only."""
        if self.memory is None:
            return
        was_training = self.training
        self.eval()
        self.memory.build(self.embed(x_img, x_clin))
        self.train(was_training)

    def regularisation(self, l1: float = 1.0, entropy: float = 1.0) -> torch.Tensor:
        if not self.use_kan:
            return torch.zeros((), device=self.head.weight.device)
        total = self.img_enc.regularisation(l1, entropy)
        if self.clin_enc is not None:
            total = total + self.clin_enc.regularisation(l1, entropy)
        return total
