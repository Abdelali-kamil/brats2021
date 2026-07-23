"""Kolmogorov-Arnold Network layers (B-spline edge activations).

Implemented from scratch rather than depending on `pykan` or `efficient-kan`:
the layer is short, the dependency risk is real for a thesis that must still
run in a year, and writing it means the spline basis is inspectable — which
matters because interpretability is the whole reason KAN is here.

Background
----------
An MLP puts fixed non-linearities on nodes and learns linear weights on edges.
A KAN inverts that: edges carry learnable univariate functions and nodes merely
sum. The Kolmogorov-Arnold representation theorem motivates the form — any
multivariate continuous function decomposes into compositions of univariate
functions and addition.

Each edge function is

    phi(x) = w_base * silu(x) + w_spline * sum_i c_i * B_i(x)

where B_i are B-spline basis functions on a fixed grid. The SiLU term is a
residual path that keeps optimisation well behaved when the spline
coefficients are still near zero; without it early training is slow and
unstable.

Interpretability, honestly stated
---------------------------------
Every edge function can be plotted (`edge_function` below), which is genuinely
more inspectable than an MLP weight matrix. But with `in_features` x
`out_features` edges, a layer mapping 54 features to 32 units has 1728
functions to look at. KAN buys *local* interpretability — what this network
does to this feature along this edge — not a global explanation. Claiming more
than that would overstate it.

Parameter count is `in*out*(grid_size + spline_order + 2)`, which for typical
clinical-scale inputs is larger than the equivalent MLP, not smaller. The
efficiency claims in the KAN literature come from symbolic-regression tasks
with tiny input dimension; they do not automatically transfer to tabular
clinical data, and this project should not repeat them uncritically.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class KANLayer(nn.Module):
    """One KAN layer: learnable B-spline function on every input-output edge."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: tuple[float, float] = (-2.0, 2.0),
        scale_noise: float = 0.1,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Uniform knot vector, extended by `spline_order` on each side so the
        # basis is well defined at the boundaries. Fixed, not learned: a moving
        # grid makes the learned functions incomparable across training runs,
        # which would defeat the interpretability argument.
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.arange(-spline_order, grid_size + spline_order + 1, dtype=torch.float32)
        grid = grid * h + grid_range[0]
        self.register_buffer("grid", grid.expand(in_features, -1).contiguous())

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order))
        self.spline_scaler = nn.Parameter(torch.empty(out_features, in_features))

        nn.init.kaiming_uniform_(self.base_weight, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.spline_scaler, a=5 ** 0.5)
        with torch.no_grad():
            self.spline_weight.normal_(0.0, scale_noise / (grid_size ** 0.5))

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """Cox-de Boor recursion. Returns [batch, in_features, n_basis]."""
        grid = self.grid  # [in, grid_size + 2*order + 1]
        x = x.unsqueeze(-1)  # [batch, in, 1]

        # Order 0: indicator of the half-open knot interval.
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)

        for k in range(1, self.spline_order + 1):
            left_den = grid[:, k:-1] - grid[:, : -(k + 1)]
            right_den = grid[:, k + 1:] - grid[:, 1:-k]
            left = (x - grid[:, : -(k + 1)]) / torch.where(
                left_den == 0, torch.ones_like(left_den), left_den)
            right = (grid[:, k + 1:] - x) / torch.where(
                right_den == 0, torch.ones_like(right_den), right_den)
            bases = left * bases[..., :-1] + right * bases[..., 1:]
        return bases.contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(F.silu(x), self.base_weight)
        bases = self.b_splines(x)  # [batch, in, n_basis]
        scaled = self.spline_weight * self.spline_scaler.unsqueeze(-1)
        spline = torch.einsum("bik,oik->bo", bases, scaled)
        return base + spline

    @torch.no_grad()
    def edge_function(self, in_idx: int, out_idx: int, n: int = 200,
                      lo: float = -2.0, hi: float = 2.0):
        """Sample one edge's learned univariate function, for plotting.

        This is the interpretability payoff: the curve applied to input
        `in_idx` on its way to unit `out_idx`, which can be shown to a
        clinician or fitted symbolically.
        """
        xs = torch.linspace(lo, hi, n, device=self.base_weight.device)
        probe = torch.zeros(n, self.in_features, device=xs.device)
        probe[:, in_idx] = xs
        bases = self.b_splines(probe)[:, in_idx, :]
        coef = self.spline_weight[out_idx, in_idx] * self.spline_scaler[out_idx, in_idx]
        ys = F.silu(xs) * self.base_weight[out_idx, in_idx] + bases @ coef
        return xs.cpu(), ys.cpu()

    def regularisation(self, l1: float = 1.0, entropy: float = 1.0) -> torch.Tensor:
        """L1 + coefficient-entropy penalty, as proposed in the KAN paper.

        Sparsifying the edge functions is what makes a KAN readable — without
        it every edge stays weakly active and there is nothing to interpret.
        """
        w = self.spline_weight.abs().mean(dim=-1)
        l1_term = w.sum()
        p = w / (w.sum() + 1e-9)
        ent = -(p * (p + 1e-9).log()).sum()
        return l1 * l1_term + entropy * ent


class KAN(nn.Module):
    """Stack of KAN layers with LayerNorm between them.

    LayerNorm is not decoration: the spline grid is fixed to `grid_range`, so
    inputs drifting outside it fall on the flat extrapolation region where all
    basis functions are zero and only the SiLU residual survives. Normalising
    between layers keeps activations inside the modelled range.
    """

    def __init__(self, layer_sizes: list[int], grid_size: int = 5,
                 spline_order: int = 3, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(KANLayer(layer_sizes[i], layer_sizes[i + 1],
                                        grid_size=grid_size, spline_order=spline_order))
            self.norms.append(nn.LayerNorm(layer_sizes[i + 1])
                              if i < len(layer_sizes) - 2 else nn.Identity())
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer, norm in zip(self.layers, self.norms):
            x = self.dropout(norm(layer(x)))
        return x

    def regularisation(self, l1: float = 1.0, entropy: float = 1.0) -> torch.Tensor:
        return sum(layer.regularisation(l1, entropy) for layer in self.layers)
