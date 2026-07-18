"""
model.py - Graph WaveNet for traffic speed forecasting.

Paper: Wu et al., "Graph WaveNet for Deep Spatial-Temporal Graph Modeling",
IJCAI 2019. https://arxiv.org/abs/1906.00121

Adapted to this task:
  Input  : hist        [B, N=1260, T=15, 1]  (per-road speed channel)
           event_feat  [B, N, E]              (per-road event mask, broadcast over T)
           node_feat   [N, F]                 (static per-road features)
           adj_fixed   [N, N]                 (binary from matrix.npy, normalized)
  Output : pred        [B, N, 3]              (h5, h10, h15)

Architecture:
  - Stacked ST-Conv blocks (gated TCN + graph conv).
  - Graph conv uses BOTH a fixed (predefined) adjacency AND a learned adaptive
    adjacency (source/target node embeddings, softmax-normalized).
  - Residual + skip connections like WaveNet.
  - Final linear head projects the per-node, per-time representation to 3
    horizon outputs in one shot.

Memory budget on a 16GB GPU at batch_size=32: ~3-5GB. Comfortably fits.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Graph adjacency normalization
# ---------------------------------------------------------------------------
def normalize_adj(adj: torch.Tensor) -> torch.Tensor:
    """
    Symmetric Laplacian normalization of [N,N] adjacency.
    A_hat = D^{-1/2} (A + I) D^{-1/2}
    """
    N = adj.shape[0]
    eye = torch.eye(N, device=adj.device, dtype=adj.dtype)
    A = adj + eye
    deg = A.sum(dim=1).clamp(min=1.0)
    d_inv_sqrt = deg.pow(-0.5)
    D_inv = torch.diag(d_inv_sqrt)
    return D_inv @ A @ D_inv


# ---------------------------------------------------------------------------
# Graph convolution layer (fixed + adaptive adjacency)
# ---------------------------------------------------------------------------
class GraphConv(nn.Module):
    """
    y = sum_k  ReLU(alpha * A_k_fixed * x * W_k_fixed
                    + (1-alpha) * A_adaptive * x * W_k_adaptive)

    Only the 1-hop case is implemented here (the standard GCN form).
    Stacking layers gives multi-hop receptive field.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.W_fixed = nn.Linear(in_dim, out_dim, bias=True)
        self.W_adaptive = nn.Linear(in_dim, out_dim, bias=True)
        # alpha is learned; sigmoid keeps it in [0,1]
        self.alpha_logit = nn.Parameter(torch.tensor(0.0))   # sigmoid(0)=0.5

    def forward(
        self,
        x: torch.Tensor,            # [B, N, in_dim]  or [N, in_dim] (broadcast)
        adj_fixed_norm: torch.Tensor,   # [N, N]
        adj_adaptive: torch.Tensor,     # [N, N]
    ) -> torch.Tensor:
        # Fixed-graph path
        h_fixed = self.W_fixed(x)
        h_fixed = torch.matmul(adj_fixed_norm, h_fixed)

        # Adaptive-graph path
        h_adapt = self.W_adaptive(x)
        h_adapt = torch.matmul(adj_adaptive, h_adapt)

        alpha = torch.sigmoid(self.alpha_logit)
        return alpha * h_fixed + (1.0 - alpha) * h_adapt


# ---------------------------------------------------------------------------
# Gated temporal convolution (gated TCN, 1D causal-ish dilated conv)
# ---------------------------------------------------------------------------
class GatedTCN(nn.Module):
    """
    1D dilated conv over time with gated activation (tanh * sigmoid).
    Input/output: [B, N, T, C].  Conv kernel slides along T.

    Pad on the left so output time length == input time length.
    """

    def __init__(self, in_dim: int, out_dim: int, kernel_size: int = 2, dilation: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation   # left-pad amount

        # Two parallel convs for the gating
        self.conv_filter = nn.Conv2d(in_dim, out_dim, (1, kernel_size), dilation=(1, dilation))
        self.conv_gate   = nn.Conv2d(in_dim, out_dim, (1, kernel_size), dilation=(1, dilation))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, N, T]  (note: we move time/channel dims outside)
        Returns: [B, out_dim, N, T]
        """
        # left-pad time dimension
        x = F.pad(x, (self.padding, 0))               # pad last dim (time) on the left
        h_filter = torch.tanh(self.conv_filter(x))
        h_gate = torch.sigmoid(self.conv_gate(x))
        return h_filter * h_gate


# ---------------------------------------------------------------------------
# One ST-Conv block: (TCN -> GraphConv -> TCN) with residual + skip
# ---------------------------------------------------------------------------
class STBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dilation: int,
        kernel_size: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.tcn1 = GatedTCN(in_dim, hidden_dim, kernel_size=kernel_size, dilation=dilation)
        self.gc   = GraphConv(hidden_dim, out_dim)
        self.tcn2 = GatedTCN(out_dim, out_dim, kernel_size=kernel_size, dilation=dilation)

        self.dropout = nn.Dropout(dropout)

        # residual: project in_dim -> out_dim if shapes differ
        self.residual = (
            nn.Conv2d(in_dim, out_dim, kernel_size=1)
            if in_dim != out_dim
            else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,             # [B, in_dim, N, T]
        adj_fixed_norm: torch.Tensor,
        adj_adaptive: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (output, skip_connection).
          output: [B, out_dim, N, T]      -- feeds into the next block
          skip:   [B, out_dim, N, T']     -- summed into the final skip path
        """
        residual = self.residual(x)

        # TCN1 -> [B, hidden, N, T]
        h = self.tcn1(x)
        h = self.dropout(h)

        # GraphConv: operate on the full time axis by looping over T.
        # Equivalent: reshape [B, hidden, N, T] -> [B*T, N, hidden] -> matmul adj.
        B, C, N, T = h.shape
        h = h.permute(0, 3, 2, 1).reshape(B * T, N, C)        # [B*T, N, hidden]
        h = self.gc(h, adj_fixed_norm, adj_adaptive)          # [B*T, N, out_dim]
        h = h.reshape(B, T, N, -1).permute(0, 3, 2, 1)        # [B, out_dim, N, T]
        h = self.dropout(h)

        # TCN2 -> [B, out_dim, N, T]
        h = self.tcn2(h)
        h = self.dropout(h)

        # The skip is typically the output at the last time step
        skip = h[:, :, :, -1:]                                # [B, out_dim, N, 1]

        # Residual + activation
        out = F.relu(h + residual)
        return out, skip


# ---------------------------------------------------------------------------
# Adaptive adjacency from learned node embeddings
# ---------------------------------------------------------------------------
class AdaptiveAdj(nn.Module):
    """
    Learned soft adjacency: A = softmax(ReLU(E1 * E2^T))
    where E1, E2 are [N, d_emb] source/target node embeddings.
    """

    def __init__(self, n_nodes: int, emb_dim: int = 10):
        super().__init__()
        self.E1 = nn.Parameter(torch.randn(n_nodes, emb_dim) * 0.1)
        self.E2 = nn.Parameter(torch.randn(n_nodes, emb_dim) * 0.1)

    def forward(self) -> torch.Tensor:
        # [N, N]
        A = torch.matmul(self.E1, self.E2.t())    # raw scores
        A = F.relu(A)
        # row-wise softmax
        A = F.softmax(A, dim=-1)
        return A


# ---------------------------------------------------------------------------
# Full Graph WaveNet model
# ---------------------------------------------------------------------------
class GraphWaveNet(nn.Module):
    """
    Multi-horizon Graph WaveNet for the traffic forecasting task.

    Args:
        n_nodes:       number of road segments (1260)
        in_channels:   input feature channels per (road, time) = 1 (speed only)
        hidden_dim:    internal ST block width
        out_dim:       ST block output width
        n_blocks:      number of stacked ST blocks (dilation doubles each block)
        n_node_feat:   dim of static per-road features (F from node_feat tensor)
        n_event_feat:  dim of per-road event features (E from event_feat tensor)
        n_horizons:    number of output horizons (3: h5, h10, h15)
        emb_dim:       dimension of learned node embeddings for adaptive adj
        kernel_size:   TCN kernel size (2 = WaveNet default)
        dropout:       dropout rate inside ST blocks
    """

    def __init__(
        self,
        n_nodes: int = 1260,
        in_channels: int = 1,
        hidden_dim: int = 32,
        out_dim: int = 32,
        n_blocks: int = 4,
        n_node_feat: int = 15,
        n_event_feat: int = 8,
        n_horizons: int = 3,
        emb_dim: int = 10,
        kernel_size: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_horizons = n_horizons

        # --- Input embedding: project per-(road,time) features to hidden_dim ---
        # Inputs at each time step: [speed, node_feat (broadcast), event_feat (broadcast)]
        in_dim = in_channels + n_node_feat + n_event_feat
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # --- Adaptive adjacency (learned) ---
        self.adaptive_adj = AdaptiveAdj(n_nodes, emb_dim=emb_dim)

        # --- ST blocks (dilation: 1, 2, 4, 8, ...) ---
        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            dilation = 2 ** i
            block_in = hidden_dim if i > 0 else hidden_dim
            block_out = out_dim if i == n_blocks - 1 else hidden_dim
            self.blocks.append(
                STBlock(
                    in_dim=block_in,
                    hidden_dim=hidden_dim,
                    out_dim=block_out,
                    dilation=dilation,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
            )

        # --- Output head: skip connections -> per-horizon predictions ---
        # We collect skip tensors (each [B, out_dim, N, 1]) from every block,
        # sum them, then apply a small MLP that emits n_horizons per node.
        skip_dim = out_dim * n_blocks
        self.head = nn.Sequential(
            nn.Linear(skip_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_horizons),
        )

    def forward(
        self,
        hist: torch.Tensor,             # [B, T=15, N=1260]
        node_feat: torch.Tensor,        # [N, F]
        event_feat: torch.Tensor,       # [B, N, E]
        adj_fixed: torch.Tensor,        # [N, N]  (binary or already normalized)
        return_adaptive: bool = False,
    ) -> torch.Tensor:
        """
        Returns preds [B, N, n_horizons].
        """
        B, T, N = hist.shape
        assert N == self.n_nodes, f"expected {self.n_nodes} nodes, got {N}"

        # Normalize fixed adj once (cache by storing on self on first call)
        if not hasattr(self, "_adj_fixed_norm") or self._adj_fixed_norm.shape[0] != N:
            self._adj_fixed_norm = normalize_adj(adj_fixed).to(hist.device)
        adj_fixed_norm = self._adj_fixed_norm

        # Adaptive adjacency (learned, recomputed every forward)
        adj_adaptive = self.adaptive_adj()

        # --- Build per-(road, time) input embeddings ---
        # hist:           [B, T, N] -> [B, T, N, 1]
        # node_feat:      [N, F]    -> broadcast to [B, T, N, F]
        # event_feat:     [B, N, E] -> broadcast to [B, T, N, E]
        nf = node_feat.to(hist.device).unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
        ef = event_feat.to(hist.device).unsqueeze(1).expand(-1, T, -1, -1)
        x = torch.cat([hist.unsqueeze(-1), nf, ef], dim=-1)   # [B, T, N, in_dim]
        x = self.input_proj(x)                                 # [B, T, N, hidden_dim]

        # --- Re-arrange to [B, hidden, N, T] for TCN ---
        x = x.permute(0, 3, 2, 1)                              # [B, hidden, N, T]

        # --- Pass through ST blocks; collect skips ---
        skips = []
        for block in self.blocks:
            x, skip = block(x, adj_fixed_norm, adj_adaptive)
            skips.append(skip)

        # skips: each [B, out_dim_or_hidden, N, 1]; concat along channel dim
        skip_cat = torch.cat(skips, dim=1)                     # [B, skip_dim, N, 1]
        skip_cat = skip_cat.squeeze(-1)                        # [B, skip_dim, N]

        # Permute to [B, N, skip_dim] then apply head
        skip_cat = skip_cat.permute(0, 2, 1)                   # [B, N, skip_dim]
        preds = self.head(skip_cat)                            # [B, N, n_horizons]

        if return_adaptive:
            return preds, adj_adaptive
        return preds


# ---------------------------------------------------------------------------
# Masked MSE loss
# ---------------------------------------------------------------------------
def masked_mse_loss(
    preds: torch.Tensor,        # [B, N, H]
    targets: torch.Tensor,      # [B, H, N]  (note: dataset returns [N, 3, 1260] order)
    mask: torch.Tensor,         # [B, H, N]
) -> torch.Tensor:
    """
    Returns scalar MSE over masked positions.

    Note on axis order:
      dataset returns targets as [B, n_horizons, n_roads] (horizon, road per sample).
      Model outputs preds as [B, n_roads, n_horizons].
      We align them here.
    """
    preds_aligned = preds.permute(0, 2, 1)    # [B, H, N]
    sq = (preds_aligned - targets) ** 2 * mask
    return sq.sum() / mask.sum().clamp(min=1.0)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np
    from dataset import (
        build_splits,
        build_node_features,
        load_adj,
        N_ROADS,
    )

    print("=== Smoke test: building GraphWaveNet forward pass ===")
    model = GraphWaveNet(
        n_nodes=N_ROADS,
        n_node_feat=build_node_features().shape[1],
        n_event_feat=8,
        n_blocks=4,
        hidden_dim=32,
        out_dim=32,
    )

    # Tiny subset for the smoke test
    train, val = build_splits(val_frac=0.2, train_stride=50, val_stride=50)
    print(f"val: hist={val.hist.shape}")

    nf = torch.from_numpy(build_node_features())                  # [N, F]
    adj = torch.from_numpy(load_adj())                            # [N, N]

    # Mini batch
    B = 4
    hist = torch.from_numpy(val.hist[:B])                         # [B, T, N]
    evt  = torch.from_numpy(val.event_feat[:B])                   # [B, N, E]
    tgt  = torch.from_numpy(val.targets[:B])                      # [B, H, N]
    mask = torch.from_numpy(val.target_mask[:B])                  # [B, H, N]

    print(f"input shapes: hist={tuple(hist.shape)}, evt={tuple(evt.shape)}, tgt={tuple(tgt.shape)}")
    print(f"param count: {sum(p.numel() for p in model.parameters()):,}")

    out = model(hist, nf, evt, adj)
    print(f"output shape: {tuple(out.shape)}  (expected ({B}, {N_ROADS}, 3))")

    loss = masked_mse_loss(out, tgt, mask)
    print(f"masked MSE on random init: {loss.item():.4f}")

    # Backward to verify grad flow
    loss.backward()
    n_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
    n_total = sum(1 for _ in model.parameters())
    print(f"params with grad: {n_with_grad}/{n_total}")
